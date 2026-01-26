"""
Command-line interface for embedding-art.

Usage:
    embed-art optimize --target-text "goldfish" 1.0 --output image
    embed-art interpolate -a "goldfish" -b "flamingo" -s 10 -o image
"""

import sys
import traceback
from pathlib import Path
from typing import Any

import click
import yaml
from rich.console import Console

from embedding_art.core.config import load_config
from embedding_art.exceptions import (
    EmbeddingArtError,
    ImageBindNotInstalledError,
    ModelLoadError,
    OutOfMemoryError,
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
):
    """Optimize an output toward a target concept."""
    # Validate inputs before heavy imports
    if not target_text and not target_image and not target_audio:
        raise click.UsageError("At least one target concept is required")

    try:
        from embedding_art import Concept, EmbeddingArtEngine, OptimizationConfig
        from embedding_art.encoders import ImageBindEncoder
        from embedding_art.generators import SDXLImageGenerator

        # Merge CLI args with config file values
        opt_config = _loaded_config.get("optimization", {})
        steps = steps if steps is not None else opt_config.get("steps", 2000)
        lr = lr if lr is not None else opt_config.get("learning_rate", 0.1)
        seed = seed if seed is not None else opt_config.get("seed")
        device = device if device is not None else _loaded_config.get("device", "mps")

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

        console.print(f"[bold]Optimizing for {steps} steps...[/bold]")
        result = engine.optimize(target, output, config)

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


if __name__ == "__main__":
    cli()
