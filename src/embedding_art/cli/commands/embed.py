import sys

import click

from embedding_art.cli.utils import (
    console, 
    handle_exception
)


@click.command()
@click.option("-t", "--text", default=None, help="Text to embed")
@click.option("-i", "--image", type=click.Path(exists=True), default=None, help="Image to embed")
@click.option("-s", "--save", type=click.Path(), default=None, help="Save embedding to .pt file")
@click.option("--device", type=str, default=None, help="Device to use")
@click.pass_context
def embed(ctx, text, image, save, device):
    """Compute and optionally save an embedding."""
    loaded_config = ctx.obj.get("config", {}) if ctx.obj else {}
    debug_mode = ctx.obj.get("debug", False) if ctx.obj else False
    
    device = device if device is not None else loaded_config.get("device", "mps")

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
        handle_exception(e, debug_mode)
        sys.exit(1)


@click.command()
@click.argument("embedding_a", type=click.Path(exists=True))
@click.argument("embedding_b", type=click.Path(exists=True))
@click.pass_context
def compare(ctx, embedding_a, embedding_b):
    """Compare two saved embeddings."""
    debug_mode = ctx.obj.get("debug", False) if ctx.obj else False

    try:
        from embedding_art import Concept

        a = Concept.load(embedding_a)
        b = Concept.load(embedding_b)

        similarity = a.similarity(b)

        console.print(f"[bold]A:[/bold] {a.description}")
        console.print(f"[bold]B:[/bold] {b.description}")
        console.print(f"[bold]Cosine similarity:[/bold] {similarity:.4f}")

    except Exception as e:
        handle_exception(e, debug_mode)
        sys.exit(1)
