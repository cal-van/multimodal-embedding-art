import sys
from pathlib import Path

import click

from embedding_art.cli.utils import console, handle_exception


@click.command()
@click.option(
    "-t",
    "--target-text",
    multiple=True,
    nargs=2,
    type=(str, float),
    help="Text concept and weight (e.g., -t 'goldfish' 1.0)",
)
@click.option(
    "-o",
    "--output",
    type=click.Choice(["image", "audio", "video"]),
    default="image",
    help="Output modality",
)
@click.option(
    "--encoders",
    type=str,
    default="imagebind",
    show_default=True,
    help="Comma-separated list of encoder names to compare",
)
@click.option(
    "--steps",
    type=int,
    default=None,
    help="Number of optimization steps (default: 2000 or from config)",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="outputs/compare",
    show_default=True,
    help="Directory to save comparison results",
)
@click.pass_context
def compare(
    ctx: click.Context,
    target_text,
    output,
    encoders,
    steps,
    output_dir,
):
    """Render the same concept with multiple encoders for side-by-side comparison."""
    loaded_config = ctx.obj.get("config", {}) if ctx.obj else {}
    debug_mode = ctx.obj.get("debug", False) if ctx.obj else False

    if not target_text:
        raise click.UsageError("At least one target concept is required (-t)")

    # Parse encoder list
    encoder_names = [name.strip() for name in encoders.split(",") if name.strip()]
    if not encoder_names:
        raise click.UsageError("--encoders must contain at least one encoder name")

    # Merge CLI args with config file defaults
    opt_config = loaded_config.get("optimization", {})
    steps = steps if steps is not None else opt_config.get("steps", 2000)

    try:
        from embedding_art.core.concept_spec import ConceptSpec
        from embedding_art.core.config import OptimizationConfig
        from embedding_art.core.engine import EmbeddingArtEngine
        from embedding_art.encoders.defaults import create_default_registry
        from embedding_art.generators import SDXLImageGenerator, AudioLDMGenerator, SVDVideoGenerator

        registry = create_default_registry()

        device = loaded_config.get("device", "mps")

        # Build ConceptSpec(s) from CLI flags — let each encoder resolve them independently
        specs = [ConceptSpec(text=text, weight=weight) for text, weight in target_text]

        if len(specs) == 1:
            target = specs[0]
        else:
            # Use the first spec as the combined target (weighted combine not yet supported
            # at the ConceptSpec level; callers needing true arithmetic should use render)
            target = specs[0]

        console.print(f"[bold]Target:[/bold] {target.describe()}")
        console.print(f"[bold]Encoders:[/bold] {', '.join(encoder_names)}")

        engine = EmbeddingArtEngine.from_registry(registry, device=device)

        if output == "image":
            generator = SDXLImageGenerator(device=device)
        elif output == "audio":
            generator = AudioLDMGenerator(device=device)
        elif output == "video":
            generator = SVDVideoGenerator(device=device)
        else:
            raise click.UsageError(f"Output modality '{output}' not yet implemented")

        engine.register_generator(output, generator)

        config = OptimizationConfig(steps=steps)

        console.print(f"[bold]Comparing {len(encoder_names)} encoder(s) for {steps} steps...[/bold]")
        results = engine.render_compare(
            target,
            encoder_names=encoder_names,
            output_modality=output,
            config=config,
        )

        # Save each result
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for encoder_name, result in results.items():
            console.print(
                f"[green]{encoder_name}[/green] similarity: {result.final_similarity:.4f}"
            )
            if output == "image":
                img = _decode_image(result, generator)
                save_path = out_dir / f"{encoder_name}.png"
                img.save(save_path)
                console.print(f"  Saved to {save_path}")

        console.print(f"[bold green]All results saved to {out_dir}[/bold green]")

    except Exception as e:
        handle_exception(e, debug_mode)
        sys.exit(1)


def _decode_image(result, generator):
    """Decode the output tensor from a RenderResult to a PIL Image."""
    import torch
    from PIL import Image

    # Support both old OptimizationResult (get_final_image) and new RenderResult (.output)
    if hasattr(result, "get_final_image"):
        return result.get_final_image(generator)

    tensor = result.output
    with torch.no_grad():
        img_array = tensor[0].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    return Image.fromarray((img_array * 255).astype("uint8"))
