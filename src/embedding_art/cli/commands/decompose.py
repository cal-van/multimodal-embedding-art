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
        import torch
        from embedding_art.core.concept import Concept
        from embedding_art.encoders.defaults import create_default_registry

        device = loaded_config.get("device", "mps")

        registry = create_default_registry()
        encoder_instance = registry.load(encoder, device=device)

        concept = Concept.from_text(target_text, encoder_instance)
        embedding = concept.embedding  # shape: [1024]

        # Load SAE and decompose
        sae = torch.load(sae_path, map_location="cpu")

        # SAE inference: compute feature activations
        # Supports dicts with 'encoder_weight'/'encoder_bias' keys or objects with an encode() method.
        if hasattr(sae, "encode"):
            activations = sae.encode(embedding.unsqueeze(0)).squeeze(0)
        elif isinstance(sae, dict):
            weight = sae["encoder_weight"]  # [n_features, dim]
            bias = sae.get("encoder_bias", torch.zeros(weight.shape[0]))
            pre_act = embedding @ weight.T + bias
            activations = torch.relu(pre_act)
        else:
            raise click.UsageError(
                f"Unrecognised SAE format: {type(sae)}. "
                "Expected an object with encode() or a state-dict with 'encoder_weight'."
            )

        # Retrieve feature names if available
        feature_names = getattr(sae, "feature_names", None)
        if isinstance(sae, dict):
            feature_names = sae.get("feature_names", None)

        # Select top-N by activation magnitude
        values, indices = activations.topk(min(top, activations.numel()))

        table = Table(title=f'Top {top} SAE features for "{target_text}"', show_header=True)
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Feature", style="bold cyan")
        table.add_column("Activation", style="green")

        for rank, (idx, val) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
            if feature_names is not None and idx < len(feature_names):
                name = feature_names[idx]
            else:
                name = f"feature_{idx}"
            table.add_row(str(rank), name, f"{val:.4f}")

        console.print(table)

    except Exception as e:
        handle_exception(e, debug_mode)
        sys.exit(1)
