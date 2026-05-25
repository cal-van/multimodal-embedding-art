"""
CorrSteer — feature-importance via correlation with the optimisation trajectory.

For each SAE feature, compute the Pearson correlation between its
activation across the optimisation trajectory and the cosine-similarity
trajectory toward the target embedding. Features with strong positive
correlation are the ones whose activation tracks the optimiser's success;
strong negative correlation marks features that the optimiser actively
suppresses.

This is a straightforward 2026 read of the technique:

* Run the optimisation, capture a per-step embedding snapshot
  ``[N_steps, D]`` (or run a single batch of snapshots through the SAE).
* Run the SAE forward to get ``[N_steps, N_features]``.
* Pair with the per-step cosine-similarity ``[N_steps]`` (already in the
  ``OptimizationHistory``).
* For each feature column, compute ``pearson(activation_t, similarity_t)``.
* Return the top-K features by ``|correlation|``.

A simple, dependency-free Pearson implementation is used so we don't pull
SciPy in. Constant columns (zero variance) are reported with correlation
0 rather than ``NaN`` so downstream consumers never have to filter NaNs.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def pearson_correlation_matrix(
    activations: torch.Tensor, similarities: torch.Tensor
) -> torch.Tensor:
    """Pearson correlation of each column of ``activations`` against ``similarities``.

    Args:
        activations: ``[N_steps, N_features]`` float tensor.
        similarities: ``[N_steps]`` float tensor.

    Returns:
        ``[N_features]`` tensor. Columns whose variance is zero (e.g.
        a feature that never activates) get ``0.0`` instead of ``NaN``.
    """
    if activations.ndim != 2:
        raise ValueError(f"activations must be 2-D, got shape {tuple(activations.shape)}")
    if similarities.ndim != 1:
        raise ValueError(f"similarities must be 1-D, got shape {tuple(similarities.shape)}")
    if activations.shape[0] != similarities.shape[0]:
        raise ValueError(
            "activations and similarities must share their first dimension; "
            f"got {activations.shape[0]} vs {similarities.shape[0]}"
        )
    if activations.shape[0] < 2:
        # Pearson with one sample is undefined; return zeros.
        return torch.zeros(activations.shape[1], dtype=activations.dtype)

    a = activations.float()
    s = similarities.float()

    a_mean = a.mean(dim=0, keepdim=True)
    s_mean = s.mean()
    a_centered = a - a_mean
    s_centered = s - s_mean

    # numerator: sum_t a_centered[t] * s_centered[t]
    numerator = (a_centered * s_centered.unsqueeze(1)).sum(dim=0)
    # denominator: sqrt(sum_t a_centered[t]^2) * sqrt(sum_t s_centered[t]^2)
    a_var = (a_centered**2).sum(dim=0)
    s_var = (s_centered**2).sum()
    denom = torch.sqrt(a_var * s_var)

    eps = 1e-12
    correlation = torch.where(
        denom > eps,
        numerator / denom.clamp(min=eps),
        torch.zeros_like(numerator),
    )
    # Defensive: any remaining NaN/Inf becomes zero.
    correlation = torch.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    return correlation


def compute_corrsteer(
    *,
    sae: Any,
    embedding_history: list[torch.Tensor] | torch.Tensor,
    similarity_history: list[float] | torch.Tensor,
    top_k: int = 16,
) -> list[tuple[int, float]]:
    """Run CorrSteer over an optimisation's embedding + similarity history.

    Args:
        sae: Trained SAE; called as ``sae(embeddings)`` and expected to
            return a ``[N_steps, N_features]`` activation tensor.
        embedding_history: Per-step embedding snapshots. Accepts a list
            of ``[D]`` or ``[1, D]`` tensors, or a pre-stacked
            ``[N_steps, D]`` tensor.
        similarity_history: Per-step cosine similarity, length-N. Either
            a Python list of floats or a 1-D tensor.
        top_k: Number of features to return, sorted by
            ``|correlation|`` descending. Ties broken by feature index.

    Returns:
        ``[(feature_index, signed_correlation), ...]`` length ≤ ``top_k``.
        Returns an empty list on any failure (logged at WARNING).
    """
    try:
        if isinstance(embedding_history, list):
            if not embedding_history:
                return []
            normalised = [e.squeeze() for e in embedding_history]
            embeddings = torch.stack(normalised, dim=0)
        else:
            embeddings = embedding_history
        if embeddings.ndim != 2:
            raise ValueError(
                f"embedding_history must yield a [N, D] tensor, got {tuple(embeddings.shape)}"
            )

        if isinstance(similarity_history, list):
            similarities = torch.tensor(similarity_history, dtype=torch.float32)
        else:
            similarities = similarity_history.float()

        if similarities.numel() != embeddings.shape[0]:
            raise ValueError(
                "embedding_history and similarity_history must have the same length; "
                f"got {embeddings.shape[0]} vs {similarities.numel()}"
            )

        if embeddings.shape[0] < 2:
            return []

        with torch.no_grad():
            activations = sae(embeddings)
        if not isinstance(activations, torch.Tensor):
            raise TypeError(
                f"SAE forward returned {type(activations).__name__}, expected torch.Tensor"
            )
        if activations.ndim != 2 or activations.shape[0] != embeddings.shape[0]:
            raise ValueError(
                "SAE forward must return [N_steps, N_features]; got "
                f"shape {tuple(activations.shape)}"
            )

        correlations = pearson_correlation_matrix(
            activations.detach().cpu(), similarities.detach().cpu()
        )

        abs_corr = correlations.abs()
        k = min(top_k, correlations.numel())
        # torch.topk on the absolute value preserves the signed correlation
        # in the returned (sorted) view.
        _, sort_idx = abs_corr.topk(k, largest=True)
        out: list[tuple[int, float]] = [
            (int(idx.item()), float(correlations[idx].item())) for idx in sort_idx
        ]
        # Drop entries that are exactly zero (no signal). They sort to the
        # end naturally, but emitting them in the JSON would be noise.
        return [(f, c) for f, c in out if c != 0.0] or out
    except Exception as exc:
        logger.warning("compute_corrsteer failed: %s", exc, exc_info=True)
        return []
