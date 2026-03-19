import sys
from pathlib import Path

import click
from rich.table import Table

from embedding_art.cli.utils import console, handle_exception


@click.command()
@click.option(
    "-t",
    "--target-text",
    type=str,
    required=True,
    help="Text to decompose into SAE features",
)
@click.option(
    "--encoder",
    type=str,
    default="imagebind",
    show_default=True,
    help="Encoder to use",
)
@click.option(
    "--sae",
    "sae_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to a trained SAE artifact",
)
@click.option(
    "--top",
    type=int,
    default=20,
    show_default=True,
    help="Number of top features to display",
)
@click.pass_context
def decompose(
    ctx: click.Context,
    target_text,
    encoder,
    sae_path,
    top,
):
    """Decompose a text embedding into top SAE features."""
    loaded_config = ctx.obj.get("config", {}) if ctx.obj else {}
    debug_mode = ctx.obj.get("debug", False) if ctx.obj else False

    try:
        from embedding_art.core.concept import Concept
        from embedding_art.encoders.defaults import create_default_registry
        from embedding_art.sae.lens import SAELens

        device = loaded_config.get("device", "mps")

        registry = create_default_registry()
        encoder_instance = registry.load(encoder, device=device)

        sae = SAELens(encoder_name=encoder, artifact_path=Path(sae_path))
        concept = Concept.from_text(target_text, encoder_instance)
        decomp = concept.decompose(sae)

        # Sort active features by activation strength descending, take top N
        sorted_features = sorted(
            decomp.active_features.items(), key=lambda kv: kv[1], reverse=True
        )[:top]

        table = Table(title=f'Top {top} SAE features for "{target_text}"', show_header=True)
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Feature", style="bold cyan")
        table.add_column("Activation", style="green")

        for rank, (name, val) in enumerate(sorted_features, start=1):
            table.add_row(str(rank), name, f"{val:.4f}")

        console.print(table)

    except Exception as e:
        handle_exception(e, debug_mode)
        sys.exit(1)
