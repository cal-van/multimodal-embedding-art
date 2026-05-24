from __future__ import annotations

import sys
from pathlib import Path

import click

from embedding_art.cli.utils import console, handle_exception


@click.command("feature-viz")
@click.option(
    "-t",
    "--target-text",
    multiple=True,
    nargs=2,
    type=(str, float),
    help="Text target with weight (e.g., -t 'goldfish' 1.0)",
)
@click.option(
    "--encoder",
    default=None,
    help="Encoder name",
)
@click.option(
    "--renderer",
    type=click.Choice(["raw", "projection", "ip-adapter"]),
    default="raw",
    show_default=True,
    help="Renderer to use",
)
@click.option(
    "--sae",
    "sae_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to SAE artifact directory",
)
@click.option(
    "--max-features",
    default=10,
    type=int,
    show_default=True,
    help="Max features to render",
)
@click.option(
    "-o",
    "--output-dir",
    default="./outputs/features",
    show_default=True,
    help="Output directory",
)
@click.pass_context
def feature_viz(ctx, target_text, encoder, renderer, sae_path, max_features, output_dir):
    """Render individual SAE features of a concept (v3)."""
    loaded_config = ctx.obj.get("config", {}) if ctx.obj else {}
    debug_mode = ctx.obj.get("debug", False) if ctx.obj else False

    if not target_text:
        raise click.UsageError("At least one target text is required (-t)")

    try:
        import torch
        from PIL import Image

        from embedding_art.core.concept_spec import ConceptSpec
        from embedding_art.core.engine import EmbeddingArtEngine
        from embedding_art.encoders.defaults import create_default_registry
        from embedding_art.sae.lens import SAELens

        device = loaded_config.get("device", "mps")
        encoder_name = encoder or "imagebind"

        console.print("[bold]Loading encoder...[/bold]")
        registry = create_default_registry()
        engine = EmbeddingArtEngine.from_registry(
            registry, default_encoder=encoder_name, device=device
        )

        spec = ConceptSpec(text=target_text[0][0])

        # Load SAE
        sae = SAELens(encoder_name=encoder_name, artifact_path=Path(sae_path))

        # Infer embed_dim from encoder
        encoder_instance = registry.load(encoder_name, device=device)
        embed_dim = (
            encoder_instance.card.embedding_dim if hasattr(encoder_instance, "card") else 1024
        )

        # Instantiate renderer
        if renderer == "raw":
            from embedding_art.renderers.raw import RawDecoder

            renderer_instance = RawDecoder(embed_dim=embed_dim, device=device)
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

        console.print("[bold]Decomposing and rendering features...[/bold]")
        results = engine.render_features(
            spec,
            renderer_instance,
            sae,
            encoder=encoder_name,
            max_features=max_features,
        )

        # Save outputs
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for name, result in results.items():
            safe_name = name.replace(" ", "_").replace("/", "_")[:50]
            save_path = output_path / f"feature_{safe_name}.png"
            with torch.no_grad():
                img_array = result.output[0].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
            img = Image.fromarray((img_array * 255).astype("uint8"))
            img.save(save_path)

        console.print(
            f"[bold green]Saved {len(results)} feature renders to {output_path}[/bold green]"
        )

    except Exception as e:
        handle_exception(e, debug_mode)
        sys.exit(1)
