from __future__ import annotations

import sys
from pathlib import Path

import click

from embedding_art.cli.utils import console, handle_exception


@click.command("probe")
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
    type=click.Choice(["raw"]),
    default="raw",
    show_default=True,
    help="Renderer for layer outputs",
)
@click.option(
    "--layers",
    default=None,
    help="Comma-separated layer indices (default: all)",
)
@click.option(
    "-o",
    "--output-dir",
    default="./outputs/probe",
    show_default=True,
    help="Output directory",
)
@click.pass_context
def probe(ctx, target_text, encoder, renderer, layers, output_dir):
    """Render multi-layer encoder states (v3)."""
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

        device = loaded_config.get("device", "mps")
        encoder_name = encoder or "imagebind"

        console.print("[bold]Loading encoder...[/bold]")
        registry = create_default_registry()
        engine = EmbeddingArtEngine.from_registry(
            registry, default_encoder=encoder_name, device=device
        )

        spec = ConceptSpec(text=target_text[0][0])

        # Parse layers
        layer_list = None
        if layers is not None:
            layer_list = [int(x.strip()) for x in layers.split(",")]

        # Instantiate renderer — infer embed_dim from the encoder
        encoder_instance = registry.load(encoder_name, device=device)
        embed_dim = (
            encoder_instance.card.embedding_dim if hasattr(encoder_instance, "card") else 1024
        )
        if renderer == "raw":
            from embedding_art.renderers.raw import RawDecoder

            renderer_instance = RawDecoder(embed_dim=embed_dim, device=device)
        else:
            raise click.UsageError(f"Unknown renderer: {renderer}")

        console.print("[bold]Capturing encoder states...[/bold]")
        results = engine.render_state(
            spec, renderer_instance, encoder=encoder_name, layers=layer_list
        )

        # Save outputs
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        desc = "_".join(t for t, _ in target_text)[:50]
        for i, result in enumerate(results):
            save_path = output_path / f"{desc}_layer_{i:03d}.png"
            with torch.no_grad():
                img_array = result.output[0].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
            img = Image.fromarray((img_array * 255).astype("uint8"))
            img.save(save_path)

        console.print(
            f"[bold green]Saved {len(results)} layer renders to {output_path}[/bold green]"
        )

    except Exception as e:
        handle_exception(e, debug_mode)
        sys.exit(1)
