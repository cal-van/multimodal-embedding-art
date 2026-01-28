"""
Command-line interface for embedding-art.

Usage:
    embed-art optimize --target-text "goldfish" 1.0 --output image
    embed-art interpolate -a "goldfish" -b "flamingo" -s 10 -o image
"""

import sys
from pathlib import Path
from typing import Any

import click
import yaml
from rich.console import Console
from rich.table import Table

from embedding_art.core.config import load_config
from embedding_art.exceptions import (
    EmbeddingArtError,
    ImageBindNotInstalledError,
    ModelLoadError,
    OutOfMemoryError,
    UpscalerError,
)

console = Console()

# Global state
_debug_mode = False
_loaded_config: dict[str, Any] = {}


def handle_exception(e: Exception) -> None:
    """Handle an exception with user-friendly output."""
    if _debug_mode:
        console.print_exception()
        return

    if isinstance(e, ImageBindNotInstalledError):
        console.print(f"[bold red]Error:[/bold red] {e}")
    elif isinstance(e, OutOfMemoryError):
        console.print(f"[bold red]Out of Memory:[/bold red] {e}")
    elif isinstance(e, ModelLoadError):
        console.print(f"[bold red]Model Load Failed:[/bold red] {e}")
    elif isinstance(e, UpscalerError):
        console.print(f"[bold red]Upscaling Failed:[/bold red] {e}")
    elif isinstance(e, EmbeddingArtError):
        console.print(f"[bold red]Error:[/bold red] {e}")
    elif isinstance(e, click.UsageError):
        raise e
    else:
        console.print(f"[bold red]Unexpected Error:[/bold red] {type(e).__name__}: {e}")
        console.print("[dim]Use --debug for full traceback[/dim]")


@click.group()
@click.version_option(version="0.1.0")
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Show full tracebacks on errors",
)
@click.option(
    "--config",
    type=click.Path(exists=False),
    default=None,
    help="Path to YAML config file (default: config.yaml in current directory)",
)
def cli(debug: bool, config: str | None) -> None:
    """Generate art by optimizing toward coordinates in multimodal embedding space."""
    global _debug_mode, _loaded_config
    _debug_mode = debug

    # Load config file
    try:
        _loaded_config = load_config(config)
    except yaml.YAMLError as e:
        raise click.UsageError(f"Invalid YAML in config file: {e}") from e


def _print_dry_run_summary(
    target_text: tuple[tuple[str, float], ...],
    target_image: tuple[tuple[str, float], ...],
    target_audio: tuple[tuple[str, float], ...],
    output_modality: str,
    output_path: str | None,
    steps: int,
    lr: float,
    seed: int | None,
    device: str,
) -> None:
    """Print a summary of what optimization would do without running it."""
    table = Table(title="Dry Run Configuration", show_header=False, box=None)
    table.add_column("Setting", style="bold cyan")
    table.add_column("Value", style="white")

    table.add_row("Device", device)
    table.add_row("Steps", str(steps))
    table.add_row("Learning Rate", str(lr))
    table.add_row("Output Modality", output_modality)

    if output_path:
        table.add_row("Output Path", output_path)
    else:
        table.add_row("Output Path", "(auto-generated)")

    if seed is not None:
        table.add_row("Seed", str(seed))
    else:
        table.add_row("Seed", "(random)")

    console.print(table)
    console.print()

    # Print target concepts
    concepts_table = Table(title="Target Concepts", show_header=True)
    concepts_table.add_column("Type", style="bold")
    concepts_table.add_column("Value", style="white")
    concepts_table.add_column("Weight", style="cyan")

    for text, weight in target_text:
        concepts_table.add_row("Text", text, str(weight))

    for path, weight in target_image:
        concepts_table.add_row("Image", str(path), str(weight))

    for path, weight in target_audio:
        concepts_table.add_row("Audio", str(path), str(weight))

    console.print(concepts_table)
    console.print()
    console.print("[dim]Dry run complete. No models were loaded.[/dim]")


@cli.command()
@click.option(
    "-t",
    "--target-text",
    multiple=True,
    nargs=2,
    type=(str, float),
    help="Text concept and weight (e.g., -t 'goldfish' 1.0)",
)
@click.option(
    "-i",
    "--target-image",
    multiple=True,
    nargs=2,
    type=(click.Path(exists=True), float),
    help="Image path and weight",
)
@click.option(
    "-a",
    "--target-audio",
    multiple=True,
    nargs=2,
    type=(click.Path(exists=True), float),
    help="Audio path and weight",
)
@click.option(
    "-o",
    "--output",
    type=click.Choice(["image", "audio", "video"]),
    default="image",
    help="Output modality",
)
@click.option(
    "-p",
    "--output-path",
    type=click.Path(),
    default=None,
    help="Output file path",
)
@click.option(
    "--steps",
    type=int,
    default=None,
    help="Number of optimization steps (default: 2000 or from config)",
)
@click.option(
    "--lr",
    type=float,
    default=None,
    help="Learning rate (default: 0.1 or from config)",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility",
)
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
@click.option(
    "--checkpoint-dir",
    type=click.Path(),
    default=None,
    help="Directory to save checkpoints during optimization",
)
@click.option(
    "--resume",
    type=click.Path(exists=True),
    default=None,
    help="Path to a checkpoint file to resume optimization from",
)
@click.option(
    "--low-memory",
    is_flag=True,
    default=False,
    help="Enable aggressive memory-saving mode (offloading + fp16 + cache clearing)",
)
@click.option(
    "--offload",
    is_flag=True,
    default=False,
    help="Offload models to CPU when not in use to save GPU memory",
)
@click.option(
    "--fp16",
    is_flag=True,
    default=False,
    help="Use mixed precision (float16) for reduced memory usage",
)
def optimize(
    target_text,
    target_image,
    target_audio,
    output,
    output_path,
    steps,
    lr,
    seed,
    device,
    dry_run,
    checkpoint_dir,
    resume,
    low_memory,
    offload,
    fp16,
):
    """Optimize an output toward a target concept."""
    # Validate inputs before heavy imports
    if not target_text and not target_image and not target_audio:
        raise click.UsageError("At least one target concept is required")

    # Merge CLI args with config file values (needed for both dry-run and real run)
    opt_config = _loaded_config.get("optimization", {})
    steps = steps if steps is not None else opt_config.get("steps", 2000)
    lr = lr if lr is not None else opt_config.get("learning_rate", 0.1)
    seed = seed if seed is not None else opt_config.get("seed")
    device = device if device is not None else _loaded_config.get("device", "mps")

    if dry_run:
        _print_dry_run_summary(
            target_text=target_text,
            target_image=target_image,
            target_audio=target_audio,
            output_modality=output,
            output_path=output_path,
            steps=steps,
            lr=lr,
            seed=seed,
            device=device,
        )
        return

    try:
        from embedding_art import Concept, EmbeddingArtEngine, OptimizationConfig
        from embedding_art.core.memory import MemoryConfig
        from embedding_art.encoders import ImageBindEncoder
        from embedding_art.generators import SDXLImageGenerator

        # Build memory configuration
        memory_config: MemoryConfig | None = None
        if low_memory:
            memory_config = MemoryConfig.low_memory()
            console.print("[dim]Low memory mode enabled[/dim]")
        elif offload or fp16:
            memory_config = MemoryConfig(
                offload_to_cpu=offload,
                use_mixed_precision=fp16,
                empty_cache_every=50 if offload else 0,
                track_memory=True,
            )
            if offload:
                console.print("[dim]Model offloading enabled[/dim]")
            if fp16:
                console.print("[dim]Mixed precision (fp16) enabled[/dim]")

        console.print("[bold]Loading models...[/bold]")

        # Load encoder
        encoder = ImageBindEncoder(device=device)

        # Build combined concept
        concepts = []
        weights = []

        for text, weight in target_text:
            concepts.append(Concept.from_text(text, encoder))
            weights.append(weight)

        for path, weight in target_image:
            concepts.append(Concept.from_image(path, encoder))
            weights.append(weight)

        for path, weight in target_audio:
            concepts.append(Concept.from_audio(path, encoder))
            weights.append(weight)

        if len(concepts) == 1:
            target = concepts[0]
        else:
            target = Concept.combine(concepts, weights)

        console.print(f"[bold]Target:[/bold] {target.description}")

        # Load generator
        if output == "image":
            generator = SDXLImageGenerator(device=device)
        else:
            raise click.UsageError(f"Output modality '{output}' not yet implemented")

        # Setup engine
        engine = EmbeddingArtEngine(encoder, device=device)
        engine.register_generator(output, generator)

        # Run optimization
        config = OptimizationConfig(
            steps=steps,
            learning_rate=lr,
            seed=seed,
        )

        if resume:
            console.print(f"[bold]Resuming from checkpoint: {resume}[/bold]")
        console.print(f"[bold]Optimizing for {steps} steps...[/bold]")
        result = engine.optimize(
            target,
            output,
            config,
            checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
            resume_from=Path(resume) if resume else None,
            memory_config=memory_config,
        )

        console.print(f"[green]Final similarity: {result.final_similarity:.4f}[/green]")
        console.print(f"[dim]Elapsed: {result.elapsed_seconds:.1f}s[/dim]")

        # Save output
        if output_path is None:
            # Generate default name
            desc = target.description.replace('"', "").replace(" ", "_")[:50]
            output_path = f"outputs/{desc}.png"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        img = result.get_final_image(generator)
        img.save(output_path)

        console.print(f"[bold green]Saved to {output_path}[/bold green]")

    except click.UsageError:
        raise
    except KeyboardInterrupt:
        console.print("\n[yellow]Optimization cancelled[/yellow]")
        sys.exit(1)
    except Exception as e:
        handle_exception(e)
        sys.exit(1)


@cli.command()
@click.option("-a", "--concept-a", required=True, help="Starting concept (text)")
@click.option("-b", "--concept-b", required=True, help="Ending concept (text)")
@click.option("-s", "--steps", type=int, default=10, help="Number of interpolation points")
@click.option(
    "-o",
    "--output",
    type=click.Choice(["image"]),
    default="image",
    help="Output modality",
)
@click.option(
    "-d",
    "--output-dir",
    type=click.Path(),
    default="outputs/interpolation",
    help="Output directory",
)
@click.option("--opt-steps", type=int, default=500, help="Optimization steps per point")
@click.option("--device", type=str, default="mps", help="Device to use")
def interpolate(concept_a, concept_b, steps, output, output_dir, opt_steps, device):
    """Generate outputs along interpolation between two concepts."""
    try:
        from embedding_art import Concept, EmbeddingArtEngine, OptimizationConfig
        from embedding_art.encoders import ImageBindEncoder
        from embedding_art.generators import SDXLImageGenerator

        console.print("[bold]Loading models...[/bold]")

        encoder = ImageBindEncoder(device=device)
        generator = SDXLImageGenerator(device=device)

        engine = EmbeddingArtEngine(encoder, device=device)
        engine.register_generator("image", generator)

        a = Concept.from_text(concept_a, encoder)
        b = Concept.from_text(concept_b, encoder)

        console.print(f"[bold]Interpolating:[/bold] {a.description} → {b.description}")
        console.print(f"[dim]Steps: {steps}, opt_steps: {opt_steps}[/dim]")

        config = OptimizationConfig(steps=opt_steps)

        results = engine.interpolation_series(a, b, "image", steps=steps, config=config)

        # Save outputs
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, result in enumerate(results):
            t = i / (steps - 1) if steps > 1 else 0.5
            img = result.get_final_image(generator)
            img.save(output_dir / f"{i:03d}_t{t:.2f}.png")

        console.print(f"[bold green]Saved {len(results)} images to {output_dir}[/bold green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interpolation cancelled[/yellow]")
        sys.exit(1)
    except Exception as e:
        handle_exception(e)
        sys.exit(1)


@cli.command()
@click.option("-t", "--text", default=None, help="Text to embed")
@click.option("-i", "--image", type=click.Path(exists=True), default=None, help="Image to embed")
@click.option("-s", "--save", type=click.Path(), default=None, help="Save embedding to .pt file")
@click.option("--device", type=str, default="mps", help="Device to use")
def embed(text, image, save, device):
    """Compute and optionally save an embedding."""
    if not text and not image:
        raise click.UsageError("Either --text or --image is required")

    try:
        from embedding_art import Concept
        from embedding_art.encoders import ImageBindEncoder

        encoder = ImageBindEncoder(device=device)

        if text:
            concept = Concept.from_text(text, encoder)
        else:
            concept = Concept.from_image(image, encoder)

        console.print(f"[bold]Concept:[/bold] {concept.description}")
        console.print(f"[dim]Embedding shape: {concept.embedding.shape}[/dim]")

        if save:
            concept.save(save)
            console.print(f"[green]Saved to {save}[/green]")

    except click.UsageError:
        raise
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled[/yellow]")
        sys.exit(1)
    except Exception as e:
        handle_exception(e)
        sys.exit(1)


@cli.command()
@click.argument("embedding_a", type=click.Path(exists=True))
@click.argument("embedding_b", type=click.Path(exists=True))
def compare(embedding_a, embedding_b):
    """Compare two saved embeddings."""
    try:
        from embedding_art import Concept

        a = Concept.load(embedding_a)
        b = Concept.load(embedding_b)

        similarity = a.similarity(b)

        console.print(f"[bold]A:[/bold] {a.description}")
        console.print(f"[bold]B:[/bold] {b.description}")
        console.print(f"[bold]Cosine similarity:[/bold] {similarity:.4f}")

    except Exception as e:
        handle_exception(e)
        sys.exit(1)


def _parse_batch_file(batch_path: Path) -> list[dict[str, Any]]:
    """
    Parse a batch YAML file and validate its structure.

    Args:
        batch_path: Path to the batch YAML file.

    Returns:
        List of job dictionaries.

    Raises:
        click.UsageError: If the batch file is invalid.
    """
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
    """
    Validate a single batch job has required fields.

    Args:
        job: The job dictionary to validate.
        job_index: Index of the job (for error messages).

    Raises:
        click.UsageError: If the job is invalid.
    """
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


@cli.command()
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
def batch(batch_file: str, device: str | None, dry_run: bool) -> None:
    """Process multiple concepts from a YAML batch file.

    The batch file should have this format:

    \b
    jobs:
      - target_text: [["goldfish", 1.0]]
        output: outputs/goldfish.png
      - target_text: [["flamingo", 1.0]]
        output: outputs/flamingo.png
        steps: 500  # optional override

    Each job can specify:
      - target_text: List of [text, weight] pairs
      - target_image: List of [path, weight] pairs
      - target_audio: List of [path, weight] pairs
      - output: Output file path
      - steps: Override optimization steps
      - learning_rate: Override learning rate
      - seed: Override random seed
    """
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
    device = device if device is not None else _loaded_config.get("device", "mps")

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
            opt_config = _loaded_config.get("optimization", {})
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
        handle_exception(e)
        sys.exit(1)


@cli.command()
@click.argument("input", type=click.Path(exists=True))
@click.option(
    "-s",
    "--scale",
    type=click.Choice(["2", "4"]),
    default="4",
    help="Upscaling factor (default: 4)",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Output file path (default: input_upscaled.png)",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device to use (default: mps or from config)",
)
def upscale(input: str, scale: str, output: str | None, device: str | None) -> None:
    """Upscale an image using Real-ESRGAN.

    Takes an input image and produces a higher resolution version.
    Supports 2x and 4x upscaling.

    Example:
        embed-art upscale image.png --scale 4 --output image_4x.png
    """
    from PIL import Image

    input_path = Path(input)

    # Determine output path
    if output is None:
        output_path = input_path.parent / f"{input_path.stem}_upscaled{input_path.suffix}"
    else:
        output_path = Path(output)

    # Get device from config or default
    device = device if device is not None else _loaded_config.get("device", "mps")
    scale_int = int(scale)

    try:
        from embedding_art.upscalers import RealESRGANUpscaler

        console.print(f"[bold]Loading Real-ESRGAN ({scale_int}x)...[/bold]")

        upscaler = RealESRGANUpscaler(device=device, scale=scale_int)

        console.print(f"[bold]Upscaling:[/bold] {input_path}")

        # Load and upscale
        image = Image.open(input_path).convert("RGB")
        upscaled = upscaler.upscale(image)

        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        upscaled.save(output_path)

        console.print(f"[bold green]Saved to {output_path}[/bold green]")
        console.print(
            f"[dim]Input: {image.size[0]}x{image.size[1]} -> Output: {upscaled.size[0]}x{upscaled.size[1]}[/dim]"
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]Upscaling cancelled[/yellow]")
        sys.exit(1)
    except Exception as e:
        handle_exception(e)
        sys.exit(1)


def _print_grid_dry_run_1d(
    concepts: list[tuple[str, float]],
    cols: int,
    output_path: str | None,
    device: str,
    opt_steps: int,
) -> None:
    """Print dry run summary for 1D grid generation."""
    table = Table(title="1D Grid Dry Run Configuration", show_header=False, box=None)
    table.add_column("Setting", style="bold cyan")
    table.add_column("Value", style="white")

    table.add_row("Mode", "1D Interpolation")
    table.add_row("Columns", str(cols))
    table.add_row("Device", device)
    table.add_row("Optimization Steps", str(opt_steps))

    if output_path:
        table.add_row("Output Path", output_path)
    else:
        table.add_row("Output Path", "(auto-generated)")

    console.print(table)
    console.print()

    concepts_table = Table(title="Concepts to Interpolate", show_header=True)
    concepts_table.add_column("Endpoint", style="bold")
    concepts_table.add_column("Text", style="white")
    concepts_table.add_column("Weight", style="cyan")

    for i, (text, weight) in enumerate(concepts):
        endpoint = "Start" if i == 0 else "End" if i == len(concepts) - 1 else f"Point {i}"
        concepts_table.add_row(endpoint, text, str(weight))

    console.print(concepts_table)
    console.print()
    console.print("[dim]Dry run complete. No models were loaded.[/dim]")


def _print_grid_dry_run_2d(
    corners: tuple[str, str, str, str],
    size: int,
    output_path: str | None,
    device: str,
    opt_steps: int,
) -> None:
    """Print dry run summary for 2D grid generation."""
    table = Table(title="2D Grid Dry Run Configuration", show_header=False, box=None)
    table.add_column("Setting", style="bold cyan")
    table.add_column("Value", style="white")

    table.add_row("Mode", "2D Interpolation")
    table.add_row("Grid Size", f"{size}x{size}")
    table.add_row("Total Images", str(size * size))
    table.add_row("Device", device)
    table.add_row("Optimization Steps", str(opt_steps))

    if output_path:
        table.add_row("Output Path", output_path)
    else:
        table.add_row("Output Path", "(auto-generated)")

    console.print(table)
    console.print()

    corners_table = Table(title="Corner Concepts", show_header=True)
    corners_table.add_column("Position", style="bold")
    corners_table.add_column("Concept", style="white")

    corners_table.add_row("Top-Left", corners[0])
    corners_table.add_row("Top-Right", corners[1])
    corners_table.add_row("Bottom-Left", corners[2])
    corners_table.add_row("Bottom-Right", corners[3])

    console.print(corners_table)
    console.print()
    console.print("[dim]Dry run complete. No models were loaded.[/dim]")


@cli.command()
@click.option(
    "-t",
    "--target-text",
    multiple=True,
    nargs=2,
    type=(str, float),
    help="Text concept and weight for 1D interpolation (use twice for start/end)",
)
@click.option(
    "--corners",
    nargs=4,
    type=str,
    default=None,
    help="Four corner concepts for 2D interpolation (top-left, top-right, bottom-left, bottom-right)",
)
@click.option(
    "--cols",
    type=int,
    default=5,
    help="Number of columns for 1D grid (default: 5)",
)
@click.option(
    "--size",
    type=int,
    default=3,
    help="Grid size for 2D interpolation (creates size x size grid, default: 3)",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Output file path for the grid image",
)
@click.option(
    "--opt-steps",
    type=int,
    default=500,
    help="Optimization steps per grid cell (default: 500)",
)
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
def grid(
    target_text: tuple[tuple[str, float], ...],
    corners: tuple[str, str, str, str] | None,
    cols: int,
    size: int,
    output: str | None,
    opt_steps: int,
    device: str | None,
    dry_run: bool,
) -> None:
    """Generate a grid of images by interpolating between concepts.

    Two modes are available:

    \b
    1D Grid (row interpolation):
        Use -t twice to specify start and end concepts.
        Creates a single row of interpolated images.

    \b
        embed-art grid -t "fire" 1.0 -t "water" 1.0 --cols 5 -o grid.png

    \b
    2D Grid (corner interpolation):
        Use --corners to specify 4 corner concepts.
        Creates an NxN grid with bilinear interpolation.

    \b
        embed-art grid --corners "fire" "water" "earth" "air" --size 3 -o grid.png
    """
    # Validate inputs
    has_targets = len(target_text) > 0
    has_corners = corners is not None

    if not has_targets and not has_corners:
        raise click.UsageError("Either -t/--target-text (for 1D) or --corners (for 2D) is required")

    if has_targets and has_corners:
        raise click.UsageError("Cannot use both -t/--target-text and --corners. Choose one mode.")

    if has_targets and len(target_text) < 2:
        raise click.UsageError("1D grid requires at least 2 concepts (use -t twice)")

    if has_corners and len(corners) != 4:
        raise click.UsageError("--corners requires exactly 4 concepts")

    # Get device from config or default
    device = device if device is not None else _loaded_config.get("device", "mps")

    if dry_run:
        if has_targets:
            _print_grid_dry_run_1d(
                concepts=list(target_text),
                cols=cols,
                output_path=output,
                device=device,
                opt_steps=opt_steps,
            )
        else:
            _print_grid_dry_run_2d(
                corners=corners,
                size=size,
                output_path=output,
                device=device,
                opt_steps=opt_steps,
            )
        return

    try:
        from embedding_art import Concept, EmbeddingArtEngine, OptimizationConfig
        from embedding_art.core.grid import interpolate_1d_grid, interpolate_2d_grid, stitch_grid
        from embedding_art.encoders import ImageBindEncoder
        from embedding_art.generators import SDXLImageGenerator

        console.print("[bold]Loading models...[/bold]")

        encoder = ImageBindEncoder(device=device)
        generator = SDXLImageGenerator(device=device)

        engine = EmbeddingArtEngine(encoder, device=device)
        engine.register_generator("image", generator)

        config = OptimizationConfig(steps=opt_steps)

        if has_targets:
            # 1D grid mode
            console.print(f"[bold]Generating 1D grid with {cols} columns...[/bold]")

            # Build concepts (use first and last for interpolation, ignore weights for now)
            concept_start = Concept.from_text(target_text[0][0], encoder)
            concept_end = Concept.from_text(target_text[-1][0], encoder)

            console.print(
                f"[dim]Interpolating: {concept_start.description} -> {concept_end.description}[/dim]"
            )

            concepts = interpolate_1d_grid(concept_start, concept_end, cols)

            # Optimize each concept
            images = []
            for i, concept in enumerate(concepts):
                console.print(f"[cyan]Optimizing cell {i+1}/{cols}...[/cyan]")
                result = engine.optimize(concept, "image", config, progress=True)
                images.append(result.get_final_image(generator))
                console.print(f"[green]  Similarity: {result.final_similarity:.4f}[/green]")

            # Stitch into single row
            grid_image = stitch_grid([images])

        else:
            # 2D grid mode
            console.print(f"[bold]Generating {size}x{size} grid...[/bold]")

            # Build corner concepts
            corner_concepts = [Concept.from_text(c, encoder) for c in corners]

            console.print(
                f"[dim]Corners: {', '.join(c.description for c in corner_concepts)}[/dim]"
            )

            concepts_grid = interpolate_2d_grid(corner_concepts, size)

            # Optimize each concept in the grid
            images_grid = []
            total = size * size
            count = 0
            for row_idx, row in enumerate(concepts_grid):
                row_images = []
                for col_idx, concept in enumerate(row):
                    count += 1
                    console.print(
                        f"[cyan]Optimizing cell ({row_idx+1},{col_idx+1}) [{count}/{total}]...[/cyan]"
                    )
                    result = engine.optimize(concept, "image", config, progress=True)
                    row_images.append(result.get_final_image(generator))
                    console.print(f"[green]  Similarity: {result.final_similarity:.4f}[/green]")
                images_grid.append(row_images)

            # Stitch into grid
            grid_image = stitch_grid(images_grid)

        # Save output
        if output is None:
            if has_targets:
                output = "outputs/grid_1d.png"
            else:
                output = "outputs/grid_2d.png"

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        grid_image.save(output_path)

        console.print(f"[bold green]Saved grid to {output_path}[/bold green]")
        console.print(f"[dim]Grid size: {grid_image.size[0]}x{grid_image.size[1]}[/dim]")

    except click.UsageError:
        raise
    except KeyboardInterrupt:
        console.print("\n[yellow]Grid generation cancelled[/yellow]")
        sys.exit(1)
    except Exception as e:
        handle_exception(e)
        sys.exit(1)


@cli.group()
def visualize() -> None:
    """Visualize embeddings and optimization progress."""
    pass


@visualize.command()
@click.argument("checkpoint", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Output file path for the plot (default: show interactively)",
)
@click.option(
    "--show-loss",
    is_flag=True,
    default=False,
    help="Include loss history on secondary axis",
)
@click.option(
    "--title",
    type=str,
    default=None,
    help="Title for the plot",
)
def similarity(checkpoint: str, output: str | None, show_loss: bool, title: str | None) -> None:
    """Plot similarity history from an optimization checkpoint.

    CHECKPOINT is a path to a saved optimization checkpoint file (.pt).
    """
    import torch

    from embedding_art.visualization import plot_similarity_history

    try:
        # Load checkpoint
        checkpoint_data = torch.load(checkpoint, weights_only=False)

        # Extract similarity history
        if "similarity_history" not in checkpoint_data:
            raise click.UsageError("Checkpoint does not contain similarity_history")

        similarity_history = checkpoint_data["similarity_history"]
        loss_history = checkpoint_data.get("loss_history") if show_loss else None

        # Create plot
        fig = plot_similarity_history(
            similarity_history,
            loss_history=loss_history,
            title=title,
        )

        # Save or show
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150)
            console.print(f"[bold green]Saved plot to {output_path}[/bold green]")
        else:
            import matplotlib.pyplot as plt

            plt.show()

    except click.UsageError:
        raise
    except Exception as e:
        handle_exception(e)
        sys.exit(1)


@visualize.command()
@click.option(
    "-t",
    "--text",
    multiple=True,
    type=str,
    help="Text concept to include (can be specified multiple times)",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Output file path for the plot (default: show interactively)",
)
@click.option(
    "--method",
    type=click.Choice(["tsne", "pca"]),
    default="tsne",
    help="Dimensionality reduction method (default: tsne)",
)
@click.option(
    "--title",
    type=str,
    default=None,
    help="Title for the plot",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device to use for encoding (default: mps or from config)",
)
def embeddings(
    text: tuple[str, ...],
    output: str | None,
    method: str,
    title: str | None,
    device: str | None,
) -> None:
    """Create a 2D projection of concept embeddings.

    Requires at least 2 text concepts. Uses t-SNE or PCA for dimensionality reduction.

    Example:
        embed-art visualize embeddings -t "fire" -t "water" -t "earth" -t "air"
    """
    if len(text) < 2:
        raise click.UsageError("Need at least 2 text concepts for 2D projection")

    try:
        from embedding_art import Concept
        from embedding_art.encoders import ImageBindEncoder
        from embedding_art.visualization import plot_embeddings_2d

        # Get device from config or default
        device = device if device is not None else _loaded_config.get("device", "mps")

        console.print("[bold]Loading encoder...[/bold]")
        encoder = ImageBindEncoder(device=device)

        # Create concepts
        console.print(f"[bold]Encoding {len(text)} concepts...[/bold]")
        concepts = [Concept.from_text(t, encoder) for t in text]

        # Create plot
        console.print(f"[bold]Creating {method.upper()} projection...[/bold]")
        fig = plot_embeddings_2d(concepts, method=method, title=title)

        # Save or show
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150)
            console.print(f"[bold green]Saved plot to {output_path}[/bold green]")
        else:
            import matplotlib.pyplot as plt

            plt.show()

    except click.UsageError:
        raise
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled[/yellow]")
        sys.exit(1)
    except Exception as e:
        handle_exception(e)
        sys.exit(1)


@visualize.command()
@click.option(
    "-t",
    "--text",
    multiple=True,
    type=str,
    help="Text concept to include (can be specified multiple times)",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Output file path for the plot (default: show interactively)",
)
@click.option(
    "--title",
    type=str,
    default=None,
    help="Title for the plot",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device to use for encoding (default: mps or from config)",
)
def distances(
    text: tuple[str, ...],
    output: str | None,
    title: str | None,
    device: str | None,
) -> None:
    """Create a distance matrix heatmap showing pairwise similarities.

    Requires at least 2 text concepts.

    Example:
        embed-art visualize distances -t "fire" -t "water" -t "earth"
    """
    if len(text) < 2:
        raise click.UsageError("Need at least 2 text concepts for distance matrix")

    try:
        from embedding_art import Concept
        from embedding_art.encoders import ImageBindEncoder
        from embedding_art.visualization import plot_distance_matrix

        # Get device from config or default
        device = device if device is not None else _loaded_config.get("device", "mps")

        console.print("[bold]Loading encoder...[/bold]")
        encoder = ImageBindEncoder(device=device)

        # Create concepts
        console.print(f"[bold]Encoding {len(text)} concepts...[/bold]")
        concepts = [Concept.from_text(t, encoder) for t in text]

        # Create plot
        console.print("[bold]Computing similarity matrix...[/bold]")
        fig = plot_distance_matrix(concepts, title=title)

        # Save or show
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150)
            console.print(f"[bold green]Saved plot to {output_path}[/bold green]")
        else:
            import matplotlib.pyplot as plt

            plt.show()

    except click.UsageError:
        raise
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled[/yellow]")
        sys.exit(1)
    except Exception as e:
        handle_exception(e)
        sys.exit(1)


if __name__ == "__main__":
    cli()
