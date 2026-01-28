import sys
from pathlib import Path

import click

from embedding_art.cli.utils import (
    console, 
    handle_exception
)


@click.command()
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
@click.option("--device", type=str, default=None, help="Device to use")
@click.pass_context
def interpolate(ctx, concept_a, concept_b, steps, output, output_dir, opt_steps, device):
    """Generate outputs along interpolation between two concepts."""
    # Get config/debug from context
    loaded_config = ctx.obj.get("config", {}) if ctx.obj else {}
    debug_mode = ctx.obj.get("debug", False) if ctx.obj else False
    
    # Use config device if not provided and not default
    # Note: CLI default for device was argument default "mps" in original code, 
    # but here we use None default to fallback to config.
    # Original main.py: default="mps" in @click.option.
    # If I change default to None, I can check config.
    # But usually best to keep defaults explicit or use None to detect user intent.
    # In my optimize.py I used default=None.
    # Here I used default=None in my new code above.
    
    device = device if device is not None else loaded_config.get("device", "mps")

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
        handle_exception(e, debug_mode)
        sys.exit(1)
