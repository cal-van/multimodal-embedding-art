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
    "-i",
    "--target-image",
    multiple=True,
    nargs=2,
    type=(click.Path(exists=True), float),
    help="Image path and weight",
)
@click.option(
    "-o",
    "--output",
    type=click.Choice(["image", "audio", "video"]),
    default="image",
    help="Output modality",
)
@click.option(
    "--encoder",
    type=str,
    default="imagebind",
    show_default=True,
    help="Encoder to use",
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
    default="mps",
    show_default=True,
    help="Device to use",
)
@click.option(
    "--output-path",
    type=click.Path(),
    default=None,
    help="Where to save the result",
)
@click.option(
    "--similarity-weight",
    type=float,
    default=1.0,
    show_default=True,
    help="Weight applied to the cosine-similarity loss term",
)
@click.option(
    "--feature-matching-weight",
    type=float,
    default=0.0,
    show_default=True,
    help="Weight applied to the feature-matching loss term",
)
@click.pass_context
def render(
    ctx: click.Context,
    target_text,
    target_image,
    output,
    encoder,
    steps,
    lr,
    seed,
    device,
    output_path,
    similarity_weight,
    feature_matching_weight,
):
    """Render an output toward a target concept using the v2 registry pipeline."""
    loaded_config = ctx.obj.get("config", {}) if ctx.obj else {}
    debug_mode = ctx.obj.get("debug", False) if ctx.obj else False

    if not target_text and not target_image:
        raise click.UsageError("At least one target concept is required (-t or -i)")

    # Merge CLI args with config file defaults
    opt_config = loaded_config.get("optimization", {})
    steps = steps if steps is not None else opt_config.get("steps", 2000)
    lr = lr if lr is not None else opt_config.get("learning_rate", 0.1)
    seed = seed if seed is not None else opt_config.get("seed")

    try:
        from embedding_art.core.concept import Concept
        from embedding_art.core.concept_spec import ConceptSpec
        from embedding_art.core.config import LossConfig, OptimizationConfig
        from embedding_art.core.engine import EmbeddingArtEngine
        from embedding_art.encoders.defaults import create_default_registry
        from embedding_art.generators import (
            SDXLImageGenerator,
            AudioLDMGenerator,
            SVDVideoGenerator,
        )

        # Create registry and load encoder
        console.print("[bold]Loading encoder...[/bold]")
        registry = create_default_registry()

        # Build combined concept from flags
        concepts = []
        weights = []

        encoder_instance = registry.load(encoder, device=device)

        for text, weight in target_text:
            concepts.append(Concept.from_text(text, encoder_instance))
            weights.append(weight)

        for path, weight in target_image:
            concepts.append(Concept.from_image(path, encoder_instance))
            weights.append(weight)

        if len(concepts) == 1:
            target = concepts[0]
        else:
            target = Concept.combine(concepts, weights)

        console.print(f"[bold]Target:[/bold] {target.description}")

        # Create engine and register generator
        engine = EmbeddingArtEngine.from_registry(registry, default_encoder=encoder, device=device)

        if output == "image":
            generator = SDXLImageGenerator(device=device)
        elif output == "audio":
            generator = AudioLDMGenerator(device=device)
        elif output == "video":
            generator = SVDVideoGenerator(device=device)
        else:
            raise click.UsageError(f"Output modality '{output}' not yet implemented")

        engine.register_generator(output, generator)

        # Build config — wire the loss weights from CLI flags through LossConfig
        loss_config = LossConfig(
            similarity_weight=similarity_weight,
            feature_matching_weight=feature_matching_weight,
        )
        config = OptimizationConfig(
            steps=steps,
            learning_rate=lr,
            seed=seed,
            loss=loss_config,
        )

        console.print(f"[bold]Rendering for {steps} steps...[/bold]")
        result = engine.render(
            target,
            encoder_name=encoder,
            output_modality=output,
            config=config,
        )

        console.print(f"[green]Final similarity: {result.final_similarity:.4f}[/green]")

        # Determine output path
        if output_path is None:
            desc = target.description.replace('"', "").replace(" ", "_")[:50]
            ext = ".png" if output == "image" else (".wav" if output == "audio" else ".gif")
            output_path = f"outputs/{desc}{ext}"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output == "image":
            img = (
                result.get_final_image(generator)
                if hasattr(result, "get_final_image")
                else _decode_image(result, generator)
            )
            img.save(output_path)
        else:
            raise click.UsageError(
                f"Saving '{output}' output not yet implemented in render command"
            )

        console.print(f"[bold green]Saved to {output_path}[/bold green]")

    except Exception as e:
        handle_exception(e, debug_mode)
        sys.exit(1)


def _decode_image(result, generator):
    """Decode the output tensor from a RenderResult to a PIL Image."""
    import torch
    from PIL import Image

    tensor = result.output
    with torch.no_grad():
        img_array = tensor[0].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    return Image.fromarray((img_array * 255).astype("uint8"))
