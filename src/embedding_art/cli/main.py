"""
Command-line interface for embedding-art.

Usage:
    embed-art optimize --target-text "goldfish" 1.0 --output image
    embed-art interpolate -a "goldfish" -b "flamingo" -s 10 -o image
"""

import click
from pathlib import Path

from rich.console import Console

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Generate art by optimizing toward coordinates in multimodal embedding space."""
    pass


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
    default=2000,
    help="Number of optimization steps",
)
@click.option(
    "--lr",
    type=float,
    default=0.1,
    help="Learning rate",
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
    default="mps",
    help="Device to use (mps, cuda, cpu)",
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
    from embedding_art import Concept, EmbeddingArtEngine, OptimizationConfig
    from embedding_art.encoders import ImageBindEncoder
    from embedding_art.generators import SDXLImageGenerator

    # Validate inputs
    if not target_text and not target_image and not target_audio:
        raise click.UsageError("At least one target concept is required")

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


@cli.command()
@click.option("-t", "--text", default=None, help="Text to embed")
@click.option("-i", "--image", type=click.Path(exists=True), default=None, help="Image to embed")
@click.option("-s", "--save", type=click.Path(), default=None, help="Save embedding to .pt file")
@click.option("--device", type=str, default="mps", help="Device to use")
def embed(text, image, save, device):
    """Compute and optionally save an embedding."""
    from embedding_art import Concept
    from embedding_art.encoders import ImageBindEncoder

    if not text and not image:
        raise click.UsageError("Either --text or --image is required")

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


@cli.command()
@click.argument("embedding_a", type=click.Path(exists=True))
@click.argument("embedding_b", type=click.Path(exists=True))
def compare(embedding_a, embedding_b):
    """Compare two saved embeddings."""
    from embedding_art import Concept

    a = Concept.load(embedding_a)
    b = Concept.load(embedding_b)

    similarity = a.similarity(b)

    console.print(f"[bold]A:[/bold] {a.description}")
    console.print(f"[bold]B:[/bold] {b.description}")
    console.print(f"[bold]Cosine similarity:[/bold] {similarity:.4f}")


if __name__ == "__main__":
    cli()
