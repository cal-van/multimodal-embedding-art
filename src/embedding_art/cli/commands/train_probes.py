"""
``embed-art train-probes`` — train a panel of linear probes over frozen
LanguageBind embeddings.

Probes are trained from a JSON manifest of the shape::

    {
      "encoder": "languagebind",
      "device": "mps",
      "out_dir": "outputs/probes/",
      "probes": [
        {
          "name": "is_animal",
          "multilabel": true,
          "examples": [
            {"text": "a goldfish",  "labels": {"is_animal": 1.0}},
            {"text": "a flamingo",  "labels": {"is_animal": 1.0}},
            {"text": "a car",       "labels": {"is_animal": 0.0}}
          ]
        },
        {
          "name": "scariness",
          "multiclass": ["calm", "tense", "scary"],
          "examples": [
            {"text": "a kitten on grass",   "label": "calm"},
            {"text": "a tense argument",    "label": "tense"},
            {"text": "a horror movie still", "label": "scary"}
          ]
        }
      ]
    }

The CLI:
1. Loads the canonical encoder.
2. Encodes every example through ``encode_text`` (multimodal
   probes-from-files are a planned follow-up).
3. Trains one :class:`LinearProbe` per entry.
4. Persists everything to ``out_dir`` via :func:`save_probe_manifest`.
5. Writes a ``training_metrics.json`` alongside.

A trained probe directory is the input
:class:`InterpretationBundle` already accepts via its
``linear_probes`` kwarg.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import click
import torch

from embedding_art.cli.utils import console, handle_exception

logger = logging.getLogger(__name__)


@click.command("train-probes")
@click.option(
    "--manifest",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to a probe-training manifest JSON.",
)
@click.option(
    "--out-dir",
    "out_dir",
    type=click.Path(),
    default=None,
    help="Override the manifest's out_dir. Defaults to the manifest field.",
)
@click.option(
    "--epochs",
    type=int,
    default=200,
    show_default=True,
    help="Adam training epochs per probe.",
)
@click.option(
    "--learning-rate",
    type=float,
    default=1e-2,
    show_default=True,
)
@click.option(
    "--weight-decay",
    type=float,
    default=1e-4,
    show_default=True,
)
@click.option(
    "--val-fraction",
    type=float,
    default=0.2,
    show_default=True,
    help="Fraction of examples reserved for the validation split.",
)
@click.option("--seed", type=int, default=0, show_default=True)
@click.pass_context
def train_probes(
    ctx: click.Context,
    manifest: str,
    out_dir: str | None,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    val_fraction: float,
    seed: int,
) -> None:
    """Train a panel of linear probes from a JSON manifest."""
    debug_mode = ctx.obj.get("debug", False) if ctx.obj else False
    try:
        _train_probes_impl(
            manifest_path=Path(manifest),
            out_dir_override=Path(out_dir) if out_dir else None,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            val_fraction=val_fraction,
            seed=seed,
        )
    except Exception as e:
        handle_exception(e, debug_mode)
        sys.exit(1)


def _train_probes_impl(
    *,
    manifest_path: Path,
    out_dir_override: Path | None,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    val_fraction: float,
    seed: int,
) -> None:
    from embedding_art.encoders.defaults import create_default_registry
    from embedding_art.evaluation.linear_probes import (
        ProbeDataset,
        save_probe_manifest,
        train_linear_probe,
    )

    manifest = json.loads(manifest_path.read_text())
    encoder_name = manifest.get("encoder", "languagebind")
    device = manifest.get("device", "mps")
    out_dir = out_dir_override or Path(manifest.get("out_dir", "outputs/probes/"))
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Loading canonical encoder ({encoder_name})...[/bold]")
    registry = create_default_registry()
    encoder = registry.load(encoder_name, device=device)

    trained: dict[str, Any] = {}
    metrics_summary: dict[str, dict[str, Any]] = {}

    for entry in manifest.get("probes", []):
        name = entry["name"]
        examples = entry["examples"]
        if not examples:
            console.print(f"[yellow]Skipping probe '{name}': no examples.[/yellow]")
            continue

        embeddings: list[torch.Tensor] = []
        for ex in examples:
            text = ex["text"]
            emb = encoder.encode_text(text)
            embeddings.append(emb.squeeze().detach().cpu())
        emb_tensor = torch.stack(embeddings)

        if "multiclass" in entry:
            label_names = list(entry["multiclass"])
            label_idx = {lbl: i for i, lbl in enumerate(label_names)}
            labels = torch.tensor([label_idx[ex["label"]] for ex in examples], dtype=torch.long)
            dataset = ProbeDataset(embeddings=emb_tensor, labels=labels, label_names=label_names)
        else:
            # Multi-label binary. ``labels`` per example is {label: 0.0/1.0}.
            label_names = sorted({lbl for ex in examples for lbl in ex["labels"]})
            label_matrix = torch.zeros(len(examples), len(label_names))
            for i, ex in enumerate(examples):
                for j, lbl in enumerate(label_names):
                    label_matrix[i, j] = float(ex["labels"].get(lbl, 0.0))
            dataset = ProbeDataset(
                embeddings=emb_tensor,
                labels=label_matrix,
                label_names=label_names,
            )

        console.print(
            f"[cyan]Training probe[/cyan] {name} "
            f"(n={dataset.num_examples}, d={dataset.num_features}, "
            f"{'multilabel' if dataset.is_multilabel else 'multiclass'})"
        )
        probe, metrics = train_linear_probe(
            dataset,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            val_fraction=val_fraction,
            seed=seed,
        )
        trained[name] = probe
        metrics_summary[name] = metrics.to_dict()

    if not trained:
        console.print("[red]No probes were trained — manifest was empty.[/red]")
        return

    save_probe_manifest(trained, out_dir)
    (out_dir / "training_metrics.json").write_text(json.dumps(metrics_summary, indent=2))
    console.print(
        f"[green]Saved {len(trained)} probe(s) to[/green] {out_dir}\n"
        f"Training metrics: {out_dir / 'training_metrics.json'}"
    )
