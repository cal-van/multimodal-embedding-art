"""SAE (Sparse Autoencoder) commands for collecting, training, labelling, and inspecting."""

import sys
from pathlib import Path

import click
from rich.table import Table

from embedding_art.cli.utils import console, handle_exception


@click.group()
def sae():
    """Sparse Autoencoder (SAE) workflow: collect → train → label → inspect."""


# ---------------------------------------------------------------------------
# sae collect
# ---------------------------------------------------------------------------


@sae.command("collect")
@click.option(
    "--encoder",
    type=str,
    default="imagebind",
    show_default=True,
    help="Encoder to use for generating embeddings",
)
@click.option(
    "--dataset",
    "dataset_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to input dataset (directory of images or a manifest file)",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(),
    required=True,
    help="Path to save collected embeddings (.pt file)",
)
@click.option(
    "--max-samples",
    type=int,
    default=10_000,
    show_default=True,
    help="Maximum number of samples to collect",
)
@click.pass_context
def collect(ctx: click.Context, encoder, dataset_path, output_path, max_samples):
    """Collect embeddings from a dataset for SAE training."""
    loaded_config = ctx.obj.get("config", {}) if ctx.obj else {}
    debug_mode = ctx.obj.get("debug", False) if ctx.obj else False

    try:
        from embedding_art.encoders.defaults import create_default_registry
        from embedding_art.sae.training import collect_embeddings

        device = loaded_config.get("device", "mps")
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)

        registry = create_default_registry()
        encoder_instance = registry.load(encoder, device=device)

        console.print(f"[bold]Collecting embeddings from {dataset_path}...[/bold]")

        saved_path = collect_embeddings(
            encoder=encoder_instance,
            dataset_path=dataset_path,
            output_path=output_path,
            max_samples=max_samples,
        )
        console.print(f"[bold green]Embeddings saved to {saved_path}[/bold green]")

    except Exception as e:
        handle_exception(e, debug_mode)
        sys.exit(1)


# ---------------------------------------------------------------------------
# sae train
# ---------------------------------------------------------------------------


@sae.command("train")
@click.option(
    "--embeddings",
    "embeddings_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to collected embeddings (.pt file from 'sae collect')",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(),
    required=True,
    help="Path to save the trained SAE model (.pt file)",
)
@click.option(
    "--features",
    type=int,
    default=4096,
    show_default=True,
    help="Number of SAE latent features",
)
@click.option(
    "--sparsity",
    type=int,
    default=32,
    show_default=True,
    help="Target number of active features per sample (TopK sparsity)",
)
@click.option(
    "--lambda",
    "lambda_",
    type=float,
    default=1e-3,
    show_default=True,
    help="L1 sparsity penalty coefficient",
)
@click.pass_context
def train(ctx: click.Context, embeddings_path, output_path, features, sparsity, lambda_):
    """Train a Sparse Autoencoder on collected embeddings."""
    debug_mode = ctx.obj.get("debug", False) if ctx.obj else False

    try:
        import torch

        from embedding_art.sae.training import train_sae

        embeddings_path = Path(embeddings_path)
        output_path = Path(output_path)

        # Peek at the embeddings file to get embed_dim for the training call.
        data = torch.load(embeddings_path, map_location="cpu", weights_only=True)
        if isinstance(data, dict):
            embed_dim = data["embeddings"].shape[1]
            n_samples = data["embeddings"].shape[0]
        else:
            embed_dim = data.shape[1]
            n_samples = data.shape[0]

        console.print(
            f"[bold]Training SAE: {embed_dim}d → {features} features "
            f"(λ={lambda_}, TopK-{sparsity}) on {n_samples} samples[/bold]"
        )

        train_sae(
            embeddings_path=embeddings_path,
            output_path=output_path,
            embed_dim=embed_dim,
            n_features=features,
            k=sparsity,
            lambda_gs=lambda_,
        )
        console.print(f"[bold green]SAE saved to {output_path}[/bold green]")

    except Exception as e:
        handle_exception(e, debug_mode)
        sys.exit(1)


# ---------------------------------------------------------------------------
# sae train-stack — per-modality SAE stack (M3)
# ---------------------------------------------------------------------------


@sae.command("train-stack")
@click.option(
    "--image-embeddings",
    type=click.Path(exists=True),
    default=None,
    help="Path to image-modality embeddings (.pt from 'sae collect').",
)
@click.option(
    "--audio-embeddings",
    type=click.Path(exists=True),
    default=None,
    help="Path to audio-modality embeddings.",
)
@click.option(
    "--video-embeddings",
    type=click.Path(exists=True),
    default=None,
    help="Path to video-modality embeddings.",
)
@click.option(
    "--text-embeddings",
    type=click.Path(exists=True),
    default=None,
    help="Path to text-modality embeddings.",
)
@click.option(
    "--shared-embeddings",
    type=click.Path(exists=True),
    default=None,
    help="Path to mean-pooled-across-modalities embeddings (the shared SAE).",
)
@click.option(
    "--output-root",
    type=click.Path(),
    required=True,
    help="Root directory under which per-modality subdirs will be written.",
)
@click.option(
    "--features",
    type=int,
    default=4096,
    show_default=True,
    help="SAE bottleneck width (same across modalities).",
)
@click.option(
    "--sparsity",
    type=int,
    default=32,
    show_default=True,
    help="TopK sparsity (active features per input).",
)
@click.option(
    "--iterations",
    type=int,
    default=25000,
    show_default=True,
    help="Per-modality gradient steps.",
)
@click.option(
    "--embed-dim",
    type=int,
    default=768,
    show_default=True,
    help="Canonical embedding dimension. 768 for LanguageBind.",
)
@click.pass_context
def train_stack(
    ctx: click.Context,
    image_embeddings,
    audio_embeddings,
    video_embeddings,
    text_embeddings,
    shared_embeddings,
    output_root,
    features,
    sparsity,
    iterations,
    embed_dim,
) -> None:
    """Train the per-modality SAE stack (M3).

    Trains one SAE per supplied modality embedding file. At least one of
    --image-embeddings / --audio-embeddings / --video-embeddings /
    --text-embeddings / --shared-embeddings must be supplied.
    """
    debug_mode = ctx.obj.get("debug", False) if ctx.obj else False

    try:
        from embedding_art.sae.multimodal_stack import train_multimodal_sae_stack

        embeddings_by_modality = {}
        for modality, path in [
            ("image", image_embeddings),
            ("audio", audio_embeddings),
            ("video", video_embeddings),
            ("text", text_embeddings),
            ("shared", shared_embeddings),
        ]:
            if path is not None:
                embeddings_by_modality[modality] = Path(path)

        if not embeddings_by_modality:
            raise click.UsageError(
                "Supply at least one of --image-embeddings, "
                "--audio-embeddings, --video-embeddings, --text-embeddings, "
                "--shared-embeddings."
            )

        console.print(
            f"[bold]Training SAE stack for {sorted(embeddings_by_modality)} → "
            f"{features} features each, TopK-{sparsity}[/bold]"
        )

        train_multimodal_sae_stack(
            embeddings_by_modality=embeddings_by_modality,
            output_root=Path(output_root),
            embed_dim=embed_dim,
            n_features=features,
            k=sparsity,
            n_iterations=iterations,
        )
        console.print(f"[bold green]SAE stack saved under {output_root}[/bold green]")

    except Exception as e:
        handle_exception(e, debug_mode)
        sys.exit(1)


# ---------------------------------------------------------------------------
# sae label
# ---------------------------------------------------------------------------


@sae.command("label")
@click.option(
    "--model",
    "model_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to a trained SAE model (.pt file)",
)
@click.option(
    "--encoder",
    type=str,
    default="imagebind",
    show_default=True,
    help="Encoder to use for generating probe embeddings",
)
@click.pass_context
def label(ctx: click.Context, model_path, encoder):
    """Interactively label SAE features using text probes."""
    loaded_config = ctx.obj.get("config", {}) if ctx.obj else {}
    debug_mode = ctx.obj.get("debug", False) if ctx.obj else False

    try:
        import torch

        from embedding_art.core.concept import Concept
        from embedding_art.encoders.defaults import create_default_registry

        device = loaded_config.get("device", "mps")
        model_path = Path(model_path)

        model = torch.load(model_path, map_location="cpu")

        registry = create_default_registry()
        encoder_instance = registry.load(encoder, device=device)

        n_features = model.encoder.out_features
        feature_names: list[str] = getattr(model, "feature_names", None) or [
            f"feature_{i}" for i in range(n_features)
        ]

        console.print(
            f"[bold]SAE has {n_features} features. Enter a text probe to see top activated features.[/bold]"
        )
        console.print("Type [bold]quit[/bold] or press Ctrl-C to stop.\n")

        while True:
            try:
                probe = click.prompt("Probe text", prompt_suffix=" > ")
            except (EOFError, KeyboardInterrupt):
                break

            if probe.strip().lower() in {"quit", "exit", "q"}:
                break

            concept = Concept.from_text(probe.strip(), encoder_instance)
            acts = model.encode(concept.embedding.unsqueeze(0)).squeeze(0)
            values, indices = acts.topk(min(10, acts.numel()))

            table = Table(title=f'Top features for "{probe}"', show_header=True)
            table.add_column("Rank", style="dim", width=6)
            table.add_column("Feature", style="bold cyan")
            table.add_column("Activation", style="green")

            for rank, (idx, val) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
                table.add_row(str(rank), feature_names[idx], f"{val:.4f}")

            console.print(table)
            console.print()

        console.print(
            "[dim]Label session ended. Model not re-saved (labels are for inspection only).[/dim]"
        )

    except Exception as e:
        handle_exception(e, debug_mode)
        sys.exit(1)


# ---------------------------------------------------------------------------
# sae auto-label
# ---------------------------------------------------------------------------


@sae.command("auto-label")
@click.option(
    "--model",
    "model_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to a trained SAE checkpoint directory or weights file.",
)
@click.option(
    "--encoder",
    type=str,
    default="languagebind",
    show_default=True,
    help="Encoder used to embed the labelling vocabulary.",
)
@click.option(
    "--device",
    type=str,
    default="mps",
    show_default=True,
    help="Torch device for encoding vocabulary words.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(),
    default=None,
    help=(
        "Destination JSON file (defaults to feature_labels.json next to the SAE). "
        "This is the file the showcase command auto-loads."
    ),
)
@click.option(
    "--use-vlm/--no-vlm",
    default=False,
    show_default=True,
    help=(
        "Use the optional VLM labeller (requires --vlm-adapter or the "
        "ANTHROPIC_API_KEY env var). Falls back to cosine labels per-feature "
        "on any failure."
    ),
)
@click.pass_context
def auto_label(
    ctx: click.Context,
    model_path: str,
    encoder: str,
    device: str,
    output_path: str | None,
    use_vlm: bool,
) -> None:
    """Generate feature_labels.json non-interactively.

    Uses :class:`~embedding_art.sae.labelling.CosineLabeller` by default;
    pass --use-vlm to opt into :class:`VLMLabeller` (with a graceful
    fallback to cosine labels per-feature on any LLM failure).
    """
    debug_mode = ctx.obj.get("debug", False) if ctx.obj else False
    try:
        import json

        from embedding_art.encoders.defaults import create_default_registry
        from embedding_art.sae.labelling import CosineLabeller, VLMLabeller

        sae_path = Path(model_path)
        registry = create_default_registry()
        encoder_instance = registry.load(encoder, device=device)

        labeller: object
        if use_vlm:
            llm_callable = _resolve_vlm_callable()
            if llm_callable is None:
                console.print(
                    "[yellow]No VLM adapter resolved (ANTHROPIC_API_KEY missing). "
                    "Falling back to cosine labeller.[/yellow]"
                )
                labeller = CosineLabeller()
            else:
                labeller = VLMLabeller(llm_callable=llm_callable)
        else:
            labeller = CosineLabeller()

        labels = labeller.label_all(sae_path=sae_path, encoder=encoder_instance)

        if output_path is None:
            target = (
                sae_path / "feature_labels.json"
                if sae_path.is_dir()
                else sae_path.parent / "feature_labels.json"
            )
        else:
            target = Path(output_path)

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({i: label for i, label in enumerate(labels)}, indent=2))
        console.print(f"[bold green]Wrote {len(labels)} feature labels to {target}[/bold green]")

    except Exception as e:
        handle_exception(e, debug_mode)
        sys.exit(1)


def _resolve_vlm_callable() -> object | None:
    """Try to construct a default Anthropic-backed VLM callable.

    Returns ``None`` if the ``anthropic`` package isn't installed or the
    ``ANTHROPIC_API_KEY`` env var isn't set — keeping the optional
    dependency truly optional.
    """
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None

    try:
        import anthropic
    except ImportError:
        return None

    client = anthropic.Anthropic()

    def adapter(prompt: str) -> str:
        msg = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=20,
            messages=[{"role": "user", "content": prompt}],
        )
        # Defensive: extract the first text block.
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                return block.text
        return ""

    return adapter


# ---------------------------------------------------------------------------
# sae inspect
# ---------------------------------------------------------------------------


@sae.command("inspect")
@click.option(
    "--model",
    "model_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to a trained SAE model (.pt file)",
)
@click.option(
    "--top",
    type=int,
    default=20,
    show_default=True,
    help="Number of top features to display by decoder norm",
)
@click.pass_context
def inspect(ctx: click.Context, model_path, top):
    """Inspect a trained SAE: show feature norms and names."""
    debug_mode = ctx.obj.get("debug", False) if ctx.obj else False

    try:
        import torch

        model_path = Path(model_path)
        model = torch.load(model_path, map_location="cpu")

        n_features = model.encoder.out_features
        feature_names: list[str] = getattr(model, "feature_names", None) or [
            f"feature_{i}" for i in range(n_features)
        ]

        # Rank by decoder column norm as a proxy for "importance"
        decoder_weight = model.decoder.weight  # [dim, n_features]
        norms = decoder_weight.norm(dim=0)  # [n_features]
        values, indices = norms.topk(min(top, n_features))

        table = Table(
            title=f"Top {top} SAE features by decoder norm ({model_path.name})",
            show_header=True,
        )
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Feature", style="bold cyan")
        table.add_column("Decoder Norm", style="green")

        for rank, (idx, val) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
            table.add_row(str(rank), feature_names[idx], f"{val:.4f}")

        console.print(table)
        console.print(f"\n[dim]Total features: {n_features}[/dim]")

    except Exception as e:
        handle_exception(e, debug_mode)
        sys.exit(1)
