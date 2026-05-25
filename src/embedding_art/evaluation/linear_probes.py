"""Linear-probe training and inference over frozen LanguageBind embeddings.

A *linear probe* is the standard interpretability tool that asks:
"is this property linearly readable from the encoder's embedding?"

* Binary probes: "is this animal?", "is this scary?", "is this loud?"
* Multi-label: a panel of binary probes evaluated jointly.
* Multi-class: "which of {animal, vehicle, instrument, …} is this?"

All variants share the same training surface — :func:`train_linear_probe`
takes a :class:`ProbeDataset` of embeddings + labels and returns a
trained :class:`LinearProbe` module plus metrics (val accuracy, AUC).
LanguageBind's shared space is explicitly designed for linear probing
so a 50-line, 1000-step Adam training run on a few hundred examples
is sufficient and trains in milliseconds on M1 Max.

Module surface
--------------

* :class:`ProbeDataset` — embeddings, labels, label names.
* :class:`LinearProbe` — frozen-input linear classifier.
* :func:`train_linear_probe` — Adam trainer with optional validation split.
* :func:`save_linear_probe` / :func:`load_linear_probe` — disk round-trip.
* :func:`evaluate_probes` — apply a dict of probes to a single embedding,
  returning the {label_name: probability} map :class:`InterpretationBundle`
  consumes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

logger = logging.getLogger(__name__)


@dataclass
class ProbeDataset:
    """Training corpus for a linear probe.

    Args:
        embeddings: Tensor of shape ``[N, D]``. Frozen encoder
            embeddings (typically already L2-normalised).
        labels: Tensor of shape ``[N]`` for multi-class
            (integer class indices) or ``[N, K]`` for multi-label
            binary (float 0/1).
        label_names: For multi-class, list of length ``num_classes``.
            For multi-label, list of length ``K``.
    """

    embeddings: torch.Tensor
    labels: torch.Tensor
    label_names: list[str]

    @property
    def is_multilabel(self) -> bool:
        return self.labels.ndim == 2

    @property
    def num_examples(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.embeddings.shape[-1])

    @property
    def num_classes(self) -> int:
        if self.is_multilabel:
            return int(self.labels.shape[1])
        # Use the user-provided label_names — there may be classes
        # the training split never sees.
        return len(self.label_names)


class LinearProbe(nn.Module):
    """Single ``nn.Linear`` over frozen encoder embeddings.

    Forward returns probabilities — sigmoid for multi-label binary,
    softmax for multi-class. The :func:`evaluate_probes` helper uses
    this contract: probe(embedding) → {label: probability}.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        *,
        label_names: list[str] | None = None,
        multilabel: bool = False,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)
        self.multilabel = multilabel
        self.label_names = (
            list(label_names) if label_names else [f"class_{i}" for i in range(num_classes)]
        )
        if len(self.label_names) != num_classes:
            raise ValueError(
                f"label_names has {len(self.label_names)} entries but " f"num_classes={num_classes}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = self.linear(x)
        if self.multilabel:
            return torch.sigmoid(logits)
        return F.softmax(logits, dim=-1)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@dataclass
class ProbeTrainingMetrics:
    """Summary returned by :func:`train_linear_probe`."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_accuracy: float = 0.0
    val_auroc: float | None = None  # binary / multilabel mean AUROC, when computable
    final_train_loss: float = 0.0
    num_epochs: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
            "val_auroc": self.val_auroc,
            "final_train_loss": self.final_train_loss,
            "num_epochs": self.num_epochs,
        }


def _split_indices(n: int, val_fraction: float, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(round(n * val_fraction))) if n > 1 and val_fraction > 0 else 0
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    if len(train_idx) == 0:
        # Fall back to using all data for training if val_fraction
        # was too aggressive for a tiny corpus.
        train_idx = perm
        val_idx = perm[:0]
    return train_idx, val_idx


def _binary_auroc(probs: torch.Tensor, labels: torch.Tensor) -> float | None:
    """Single-class AUROC via the Mann–Whitney U statistic.

    Returns ``None`` when one of the classes is missing in
    ``labels`` (AUROC is undefined). Mean of per-class AUROCs is
    handled at the caller level.
    """
    labels = labels.long()
    if labels.numel() == 0 or labels.unique().numel() < 2:
        return None
    pos = probs[labels == 1]
    neg = probs[labels == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return None
    # Rank-based formulation: P(score_pos > score_neg).
    diffs = pos.unsqueeze(1) - neg.unsqueeze(0)
    score = (diffs > 0).float().mean() + 0.5 * (diffs == 0).float().mean()
    return float(score.item())


def train_linear_probe(
    dataset: ProbeDataset,
    *,
    learning_rate: float = 1e-2,
    weight_decay: float = 1e-4,
    epochs: int = 200,
    batch_size: int | None = None,
    val_fraction: float = 0.2,
    seed: int = 0,
    device: str | torch.device = "cpu",
    verbose: bool = False,
) -> tuple[LinearProbe, ProbeTrainingMetrics]:
    """Train a linear probe over frozen embeddings.

    Uses Adam with cross-entropy (multi-class) or BCE (multi-label).
    Splits the dataset into train/val using ``val_fraction``; reports
    final val accuracy and (when meaningful) mean AUROC.

    Returns the trained :class:`LinearProbe` and a
    :class:`ProbeTrainingMetrics` record. The probe is moved to
    ``device`` before training so callers can run it inline against
    the canonical encoder's device.
    """
    if dataset.num_examples < 2:
        raise ValueError(f"Need at least 2 training examples, got {dataset.num_examples}")

    probe = LinearProbe(
        in_features=dataset.num_features,
        num_classes=dataset.num_classes,
        label_names=dataset.label_names,
        multilabel=dataset.is_multilabel,
    ).to(device)

    train_idx, val_idx = _split_indices(dataset.num_examples, val_fraction=val_fraction, seed=seed)
    x_train = dataset.embeddings[train_idx].to(device)
    y_train = dataset.labels[train_idx].to(device)
    x_val = dataset.embeddings[val_idx].to(device) if len(val_idx) > 0 else None
    y_val = dataset.labels[val_idx].to(device) if len(val_idx) > 0 else None

    if dataset.is_multilabel:
        y_train = y_train.float()
        if y_val is not None:
            y_val = y_val.float()

    opt = torch.optim.Adam(probe.parameters(), lr=learning_rate, weight_decay=weight_decay)

    metrics = ProbeTrainingMetrics()
    if batch_size is None or batch_size >= x_train.shape[0]:
        # Full-batch training is the standard probe setting; tiny D=768
        # corpora train in milliseconds.
        batches = [(x_train, y_train)]
    else:
        # Deterministic batching using the seed.
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(x_train.shape[0], generator=g)
        batches = []
        for i in range(0, x_train.shape[0], batch_size):
            ids = perm[i : i + batch_size]
            batches.append((x_train[ids], y_train[ids]))

    for epoch in range(epochs):
        probe.train()
        epoch_loss = 0.0
        for xb, yb in batches:
            opt.zero_grad()
            logits = probe.logits(xb)
            if dataset.is_multilabel:
                loss = F.binary_cross_entropy_with_logits(logits, yb)
            else:
                loss = F.cross_entropy(logits, yb.long())
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item()) * xb.shape[0]
        epoch_loss /= x_train.shape[0]
        metrics.train_loss.append(epoch_loss)
        if x_val is not None and y_val is not None:
            with torch.no_grad():
                probe.eval()
                v_logits = probe.logits(x_val)
                if dataset.is_multilabel:
                    v_loss = F.binary_cross_entropy_with_logits(v_logits, y_val)
                else:
                    v_loss = F.cross_entropy(v_logits, y_val.long())
                metrics.val_loss.append(float(v_loss.item()))
        if verbose and epoch % max(1, epochs // 10) == 0:
            logger.info("epoch %d/%d train_loss=%.4f", epoch + 1, epochs, epoch_loss)

    metrics.num_epochs = epochs
    metrics.final_train_loss = metrics.train_loss[-1] if metrics.train_loss else 0.0

    # Final validation accuracy + AUROC.
    if x_val is not None and y_val is not None and y_val.numel() > 0:
        with torch.no_grad():
            probe.eval()
            v_probs = probe(x_val)
            if dataset.is_multilabel:
                pred = (v_probs > 0.5).float()
                # Per-example accuracy: average over label dims, then mean.
                metrics.val_accuracy = float((pred == y_val).float().mean().item())
                aurocs: list[float] = []
                for k in range(v_probs.shape[1]):
                    a = _binary_auroc(v_probs[:, k], y_val[:, k])
                    if a is not None:
                        aurocs.append(a)
                metrics.val_auroc = float(sum(aurocs) / len(aurocs)) if aurocs else None
            else:
                preds = v_probs.argmax(dim=-1)
                metrics.val_accuracy = float((preds == y_val.long()).float().mean().item())
                metrics.val_auroc = None

    return probe, metrics


def save_linear_probe(probe: LinearProbe, path: str | Path) -> None:
    """Persist a probe to disk as a single ``.pt`` file.

    Stores both the state_dict and the structural metadata (
    label names, in_features, multilabel flag) so that
    :func:`load_linear_probe` can reconstruct without ambient
    arguments.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": probe.state_dict(),
        "label_names": probe.label_names,
        "in_features": int(probe.linear.in_features),
        "num_classes": int(probe.linear.out_features),
        "multilabel": bool(probe.multilabel),
    }
    torch.save(payload, path)


def load_linear_probe(path: str | Path, *, device: str | torch.device = "cpu") -> LinearProbe:
    payload = torch.load(path, map_location=device)
    probe = LinearProbe(
        in_features=payload["in_features"],
        num_classes=payload["num_classes"],
        label_names=payload["label_names"],
        multilabel=payload["multilabel"],
    )
    probe.load_state_dict(payload["state_dict"])
    probe.to(device)
    return probe


def evaluate_probes(
    embedding: torch.Tensor, probes: dict[str, LinearProbe]
) -> dict[str, float | dict[str, float]]:
    """Evaluate a dict of probes against one embedding.

    Returns a mapping ``{probe_name: probability_or_distribution}``:
    * Binary probes return a single float (probability that the
      property is present).
    * Multi-class probes return a dict ``{label: probability}``.

    Designed to match the :class:`InterpretationBundle.linear_probes`
    shape it consumes.
    """
    out: dict[str, float | dict[str, float]] = {}
    flat = (
        embedding.reshape(-1)[: probes[next(iter(probes))].linear.in_features] if probes else None
    )
    for name, probe in probes.items():
        probe.eval()
        x = embedding.detach().to(next(probe.parameters()).device).float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            probs = probe(x)
        if probe.multilabel:
            # Per-label probability.
            scores = {lbl: float(probs[0, i].item()) for i, lbl in enumerate(probe.label_names)}
            # Convenience: if there's a single class, expose the bare float too.
            if len(scores) == 1:
                out[name] = next(iter(scores.values()))
            else:
                out[name] = scores
        else:
            scores = {lbl: float(probs[0, i].item()) for i, lbl in enumerate(probe.label_names)}
            out[name] = scores
    # ``flat`` was only used to short-circuit a degenerate case earlier
    # in development; retained as no-op to keep mypy happy.
    del flat
    return out


def save_probe_manifest(probes: dict[str, LinearProbe], path: str | Path) -> None:
    """Persist a multi-probe set to a directory + index json.

    Convention::

        probes/
            index.json        # {name: filename}
            <name>.pt         # individual probe checkpoints
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    index: dict[str, str] = {}
    for name, probe in probes.items():
        fname = f"{name}.pt"
        save_linear_probe(probe, path / fname)
        index[name] = fname
    (path / "index.json").write_text(json.dumps(index, indent=2))


def load_probe_manifest(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> dict[str, LinearProbe]:
    path = Path(path)
    index_path = path / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"no index.json at {index_path}")
    index = json.loads(index_path.read_text())
    probes: dict[str, LinearProbe] = {}
    for name, fname in index.items():
        probes[name] = load_linear_probe(path / fname, device=device)
    return probes
