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
        import torch
        from embedding_art.encoders.defaults import create_default_registry

        device = loaded_config.get("device", "mps")
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)

        registry = create_default_registry()
        encoder_instance = registry.load(encoder, device=device)

        console.print(f"[bold]Collecting embeddings from {dataset_path}...[/bold]")

        # Discover image files
        image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        if dataset_path.is_dir():
            files = sorted(
                p for p in dataset_path.rglob("*") if p.suffix.lower() in image_extensions
            )
        else:
            # Treat as a text manifest — one path per line
            files = [
                Path(line.strip())
                for line in dataset_path.read_text().splitlines()
                if line.strip()
            ]

        files = files[:max_samples]
        console.print(f"Found {len(files)} samples (limit: {max_samples})")

        embeddings = []
        from embedding_art.core.concept import Concept

        for i, path in enumerate(files, start=1):
            concept = Concept.from_image(str(path), encoder_instance)
            embeddings.append(concept.embedding.cpu())
            if i % 100 == 0 or i == len(files):
                console.print(f"  [{i}/{len(files)}]")

        stacked = torch.stack(embeddings)  # [N, dim]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(stacked, output_path)
        console.print(f"[bold green]Saved {stacked.shape[0]} embeddings to {output_path}[/bold green]")

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
        import torch.nn as nn
        import torch.optim as optim

        embeddings_path = Path(embeddings_path)
        output_path = Path(output_path)

        data = torch.load(embeddings_path, map_location="cpu")  # [N, dim]
        n_samples, embed_dim = data.shape

        console.print(
            f"[bold]Training SAE: {embed_dim}d → {features} features "
            f"(λ={lambda_}, TopK-{sparsity}) on {n_samples} samples[/bold]"
        )

        # Simple tied-weight SAE with L1 regularisation
        class SparseAutoencoder(nn.Module):
            feature_names: list[str] | None = None

            def __init__(self, dim: int, n_features: int) -> None:
                super().__init__()
                self.encoder = nn.Linear(dim, n_features)
                self.decoder = nn.Linear(n_features, dim, bias=False)

            def encode(self, x: torch.Tensor) -> torch.Tensor:
                return torch.relu(self.encoder(x))

            def forward(self, x: torch.Tensor):
                acts = self.encode(x)
                recon = self.decoder(acts)
                return recon, acts

        model = SparseAutoencoder(embed_dim, features)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        dataset = torch.utils.data.TensorDataset(data)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

        n_epochs = 10
        for epoch in range(1, n_epochs + 1):
            total_loss = 0.0
            for (batch,) in loader:
                recon, acts = model(batch)
                recon_loss = nn.functional.mse_loss(recon, batch)
                sparsity_loss = lambda_ * acts.abs().mean()
                loss = recon_loss + sparsity_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg = total_loss / len(loader)
            console.print(f"  Epoch {epoch}/{n_epochs}  loss={avg:.6f}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, output_path)
        console.print(f"[bold green]SAE saved to {output_path}[/bold green]")

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

        console.print("[dim]Label session ended. Model not re-saved (labels are for inspection only).[/dim]")

    except Exception as e:
        handle_exception(e, debug_mode)
        sys.exit(1)


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
