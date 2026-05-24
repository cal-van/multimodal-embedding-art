from __future__ import annotations

import sys
from pathlib import Path

import click

from embedding_art.cli.utils import console, handle_exception


@click.command("direct")
@click.option(
    "-t",
    "--target-text",
    multiple=True,
    nargs=2,
    type=(str, float),
    help="Text target with weight (e.g., -t 'goldfish' 1.0)",
)
@click.option(
    "--renderer",
    type=click.Choice(["raw", "projection", "ip-adapter"]),
    default="raw",
    show_default=True,
    help="Renderer to use",
)
@click.option(
    "--encoder",
    default=None,
    help="Encoder name",
)
@click.option(
    "--embed-dim",
    default=1024,
    type=int,
    show_default=True,
    help="Embedding dimension (for raw renderer)",
)
@click.option(
    "--output-shape",
    default="1,3,256,256",
    show_default=True,
    help="Output shape for raw renderer (comma-separated)",
)
@click.option(
    "-o",
    "--output-dir",
    default="./outputs/direct",
    show_default=True,
    help="Output directory",
)
@click.pass_context
def direct(ctx, target_text, renderer, encoder, embed_dim, output_shape, output_dir):
    """Direct rendering without optimization loop (v3)."""
    loaded_config = ctx.obj.get("config", {}) if ctx.obj else {}
    debug_mode = ctx.obj.get("debug", False) if ctx.obj else False

    if not target_text:
        raise click.UsageError("At least one target text is required (-t)")

    try:
        import torch
        from PIL import Image

        from embedding_art.core.concept import Concept
        from embedding_art.core.concept_spec import ConceptSpec
        from embedding_art.core.engine import EmbeddingArtEngine
        from embedding_art.encoders.defaults import create_default_registry

        device = loaded_config.get("device", "mps")
        encoder_name = encoder or "imagebind"

        console.print("[bold]Loading encoder...[/bold]")
        registry = create_default_registry()
        engine = EmbeddingArtEngine.from_registry(
            registry, default_encoder=encoder_name, device=device
        )

        # Build ConceptSpec from the first target (combine if multiple)
        encoder_instance = registry.load(encoder_name, device=device)
        concepts = []
        weights = []
        for text, weight in target_text:
            concepts.append(Concept.from_text(text, encoder_instance))
            weights.append(weight)

        if len(concepts) == 1:
            spec = ConceptSpec(text=target_text[0][0])
        else:
            spec = ConceptSpec(text=" + ".join(t for t, _ in target_text))

        # Parse output shape
        shape = tuple(int(x) for x in output_shape.split(","))

        # Instantiate renderer
        if renderer == "raw":
            from embedding_art.renderers.raw import RawDecoder

            renderer_instance = RawDecoder(
                embed_dim=embed_dim, output_shape=shape, device=device
            )
        elif renderer == "projection":
            from embedding_art.generators import SDXLImageGenerator
            from embedding_art.renderers.projection import ProjectionDecoder

            generator = SDXLImageGenerator(device=device)
            renderer_instance = ProjectionDecoder(
                embed_dim=embed_dim, generator=generator, device=device
            )
        elif renderer == "ip-adapter":
            from embedding_art.renderers.ip_adapter import IPAdapterRenderer

            renderer_instance = IPAdapterRenderer(device=device)
        else:
            raise click.UsageError(f"Unknown renderer: {renderer}")

        console.print(f"[bold]Rendering with {renderer} renderer...[/bold]")
        result = engine.render_direct(spec, renderer_instance, encoder=encoder_name)

        console.print(f"[green]Render complete (similarity: {result.final_similarity:.4f})[/green]")

        # Save output
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        desc = "_".join(t for t, _ in target_text)[:50]
        save_path = output_path / f"{desc}.png"

        with torch.no_grad():
            img_array = result.output[0].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        img = Image.fromarray((img_array * 255).astype("uint8"))
        img.save(save_path)

        console.print(f"[bold green]Saved to {save_path}[/bold green]")

    except Exception as e:
        handle_exception(e, debug_mode)
        sys.exit(1)
