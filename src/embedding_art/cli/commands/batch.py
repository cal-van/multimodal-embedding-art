import sys
from pathlib import Path
from typing import Any

import click
import yaml
from rich.table import Table

from embedding_art.cli.utils import (
    console, 
    handle_exception
)


def _parse_batch_file(batch_path: Path) -> list[dict[str, Any]]:
    """Parse a batch YAML file and validate its structure."""
    if not batch_path.exists():
        raise click.UsageError(f"Batch file not found: {batch_path}")

    try:
        with open(batch_path) as f:
            content = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise click.UsageError(f"Invalid YAML in batch file: {e}") from e

    if content is None:
        raise click.UsageError("Batch file is empty")

    if "jobs" not in content:
        raise click.UsageError("Batch file must contain a 'jobs' key")

    jobs = content["jobs"]
    if not isinstance(jobs, list):
        raise click.UsageError("'jobs' must be a list")

    return jobs


def _validate_batch_job(job: dict[str, Any], job_index: int) -> None:
    """Validate a single batch job has required fields."""
    has_target = job.get("target_text") or job.get("target_image") or job.get("target_audio")
    if not has_target:
        raise click.UsageError(
            f"Job {job_index + 1}: At least one target (target_text, target_image, or target_audio) is required"
        )


def _print_batch_dry_run(jobs: list[dict[str, Any]], device: str) -> None:
    """Print a summary of batch jobs for dry-run mode."""
    console.print(f"[bold]Batch Dry Run: {len(jobs)} jobs[/bold]")
    console.print()

    for i, job in enumerate(jobs):
        job_table = Table(title=f"Job {i + 1}", show_header=False, box=None)
        job_table.add_column("Setting", style="bold cyan")
        job_table.add_column("Value", style="white")

        # Target concepts
        if job.get("target_text"):
            for text, weight in job["target_text"]:
                job_table.add_row("Text Target", f"{text} (weight: {weight})")

        if job.get("target_image"):
            for path, weight in job["target_image"]:
                job_table.add_row("Image Target", f"{path} (weight: {weight})")

        if job.get("target_audio"):
            for path, weight in job["target_audio"]:
                job_table.add_row("Audio Target", f"{path} (weight: {weight})")

        # Output path
        if job.get("output"):
            job_table.add_row("Output", job["output"])

        # Overrides
        if job.get("steps"):
            job_table.add_row("Steps", str(job["steps"]))

        if job.get("learning_rate"):
            job_table.add_row("Learning Rate", str(job["learning_rate"]))

        if job.get("seed"):
            job_table.add_row("Seed", str(job["seed"]))

        console.print(job_table)
        console.print()

    console.print(f"[dim]Device: {device}[/dim]")
    console.print("[dim]Dry run complete. No models were loaded.[/dim]")


@click.command()
@click.argument("batch_file", type=click.Path(exists=True))
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device to use (default: mps or from config)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would be done without running optimization",
)
@click.pass_context
def batch(ctx, batch_file: str, device: str | None, dry_run: bool) -> None:
    """Process multiple concepts from a YAML batch file."""
    loaded_config = ctx.obj.get("config", {}) if ctx.obj else {}
    debug_mode = ctx.obj.get("debug", False) if ctx.obj else False
    
    batch_path = Path(batch_file)
    jobs = _parse_batch_file(batch_path)

    # Empty jobs list is a no-op
    if not jobs:
        console.print("[dim]No jobs to process.[/dim]")
        return

    # Validate all jobs before starting
    for i, job in enumerate(jobs):
        _validate_batch_job(job, i)

    # Get device from config or default
    device = device if device is not None else loaded_config.get("device", "mps")

    if dry_run:
        _print_batch_dry_run(jobs, device)
        return

    try:
        from embedding_art import Concept, EmbeddingArtEngine, OptimizationConfig
        from embedding_art.encoders import ImageBindEncoder
        from embedding_art.generators import SDXLImageGenerator

        console.print(f"[bold]Processing {len(jobs)} jobs...[/bold]")
        console.print("[bold]Loading models...[/bold]")

        # Load encoder and generator once for all jobs
        encoder = ImageBindEncoder(device=device)
        generator = SDXLImageGenerator(device=device)

        engine = EmbeddingArtEngine(encoder, device=device)
        engine.register_generator("image", generator)

        # Process each job
        for i, job in enumerate(jobs):
            console.print(f"\n[bold cyan]Job {i + 1}/{len(jobs)}[/bold cyan]")

            # Build combined concept
            concepts = []
            weights = []

            for text, weight in job.get("target_text", []):
                concepts.append(Concept.from_text(text, encoder))
                weights.append(weight)

            for path, weight in job.get("target_image", []):
                concepts.append(Concept.from_image(path, encoder))
                weights.append(weight)

            for path, weight in job.get("target_audio", []):
                concepts.append(Concept.from_audio(path, encoder))
                weights.append(weight)

            if len(concepts) == 1:
                target = concepts[0]
            else:
                target = Concept.combine(concepts, weights)

            console.print(f"[bold]Target:[/bold] {target.description}")

            # Build config with job-specific overrides
            opt_config = loaded_config.get("optimization", {})
            steps = job.get("steps", opt_config.get("steps", 2000))
            lr = job.get("learning_rate", opt_config.get("learning_rate", 0.1))
            seed = job.get("seed", opt_config.get("seed"))

            config = OptimizationConfig(
                steps=steps,
                learning_rate=lr,
                seed=seed,
            )

            console.print(f"[dim]Steps: {steps}, LR: {lr}[/dim]")

            # Run optimization
            result = engine.optimize(target, "image", config)

            console.print(f"[green]Similarity: {result.final_similarity:.4f}[/green]")

            # Save output
            output_path = job.get("output")
            if output_path is None:
                desc = target.description.replace('"', "").replace(" ", "_")[:50]
                output_path = f"outputs/{desc}.png"

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            img = result.get_final_image(generator)
            img.save(output_path)

            console.print(f"[green]Saved to {output_path}[/green]")

        console.print(f"\n[bold green]Batch complete! Processed {len(jobs)} jobs.[/bold green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Batch processing cancelled[/yellow]")
        sys.exit(1)
    except Exception as e:
        handle_exception(e, debug_mode)
        sys.exit(1)
