"""
``InterpretationBundle`` — the unified record of what a rendering means.

Composed by ``run_interpretation`` for every showcase output. Includes:

* ``text_anchor`` — top-K vocabulary words nearest the final embedding in
  language space. Always populated when an encoder exposing
  ``encode_text`` is available.
* ``sae_decomposition`` — top-K SAE feature activations decomposing the
  final embedding. Optional; requires a trained SAE.
* ``sae_corrsteer`` — CorrSteer-style ranking: for each SAE feature,
  Pearson correlation between its activation and the optimisation's
  cosine-similarity trajectory. Optional; requires both an SAE and an
  ``OptimizationHistory`` with per-step embedding snapshots.
* ``linear_probes`` — dict of {probe_name: activation_value}. Optional;
  requires a pre-trained probe set.
* ``attribution_shape`` — shape of the gradient attribution map written
  to disk. The map itself isn't stored in JSON (too large); the caller
  saves it as a separate file and the bundle records the path.

The bundle serialises to a JSON-friendly dict via ``to_dict``. All
tensor-valued fields are converted to lists.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch

from embedding_art.interpretation.attribution import gradient_attribution
from embedding_art.interpretation.text_anchor import text_anchor_readout

logger = logging.getLogger(__name__)


@dataclass
class InterpretationBundle:
    """Per-rendering interpretability artefact."""

    text_anchor: list[tuple[str, float]] = field(default_factory=list)
    sae_decomposition: dict[int, float] | None = None
    sae_feature_labels: dict[int, str] | None = None
    sae_corrsteer: list[tuple[int, float]] | None = None
    linear_probes: dict[str, float] | None = None
    attribution_path: str | None = None
    attribution_shape: tuple[int, ...] | None = None
    final_similarity: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Render to a JSON-serialisable dict."""
        return {
            "text_anchor": [{"word": w, "similarity": s} for w, s in self.text_anchor],
            "sae_decomposition": self.sae_decomposition,
            "sae_feature_labels": self.sae_feature_labels,
            "sae_corrsteer": (
                [{"feature": f, "correlation": c} for f, c in self.sae_corrsteer]
                if self.sae_corrsteer is not None
                else None
            ),
            "linear_probes": self.linear_probes,
            "attribution_path": self.attribution_path,
            "attribution_shape": (
                list(self.attribution_shape) if self.attribution_shape is not None else None
            ),
            "final_similarity": self.final_similarity,
        }


def run_interpretation(
    *,
    target: Any,
    output: torch.Tensor,
    encoder: Any,
    sae: Any = None,
    sae_feature_labels: dict[int, str] | None = None,
    linear_probes: dict[str, Any] | None = None,
    top_k_text: int = 20,
    top_k_features: int = 16,
    final_similarity: float | None = None,
    embedding_history: list[torch.Tensor] | torch.Tensor | None = None,
    similarity_history: list[float] | torch.Tensor | None = None,
) -> InterpretationBundle:
    """Assemble an :class:`InterpretationBundle` for a rendered output.

    Args:
        target: The target :class:`~embedding_art.core.concept.Concept`.
            Provides the target embedding for attribution and the
            shared-space reference for the SAE decomposition.
        output: The rendered output tensor (image / audio / video).
        encoder: The canonical encoder (typically LanguageBind). Must
            expose ``encode_text``, ``encode_for_optimization``.
        sae: Optional trained SAE. When provided, the bundle includes
            an SAE decomposition of the current embedding.
        sae_feature_labels: Optional mapping from feature index → human
            label produced by ``sae.training.label_features``.
        linear_probes: Optional mapping from probe name → probe object
            with ``__call__(embedding) -> float``. When provided, each
            probe is evaluated against the current embedding.
        top_k_text: How many text-anchor words to keep.
        top_k_features: How many SAE features to keep in the
            decomposition record.
        final_similarity: The optimisation's final cosine similarity.
            Stored in the bundle for the showcase summary.

    Returns:
        A populated :class:`InterpretationBundle`. Optional fields are
        ``None`` when the corresponding inputs were not supplied.
    """
    bundle = InterpretationBundle(final_similarity=final_similarity)

    current_emb = encoder.encode_for_optimization(output).detach()

    if hasattr(encoder, "encode_text"):
        try:
            bundle.text_anchor = text_anchor_readout(current_emb, encoder, top_k=top_k_text)
        except Exception as exc:
            logger.warning("text_anchor_readout failed: %s", exc, exc_info=True)
            bundle.text_anchor = []

    if sae is not None:
        bundle.sae_decomposition = _run_sae_decomposition(
            current_emb=current_emb, sae=sae, top_k=top_k_features
        )
        if sae_feature_labels is not None and bundle.sae_decomposition is not None:
            bundle.sae_feature_labels = {
                feat_idx: sae_feature_labels.get(feat_idx, f"feature_{feat_idx}")
                for feat_idx in bundle.sae_decomposition
            }

        if embedding_history is not None and similarity_history is not None:
            from embedding_art.interpretation.corrsteer import compute_corrsteer

            corrsteer = compute_corrsteer(
                sae=sae,
                embedding_history=embedding_history,
                similarity_history=similarity_history,
                top_k=top_k_features,
            )
            if corrsteer:
                bundle.sae_corrsteer = corrsteer

    if linear_probes is not None:
        bundle.linear_probes = _run_linear_probes(current_emb=current_emb, probes=linear_probes)

    return bundle


def compute_attribution_to_disk(
    *,
    output: torch.Tensor,
    target_embedding: torch.Tensor,
    encoder: Any,
    path: Any,
) -> tuple[int, ...]:
    """Compute the gradient attribution map and save it to ``path``.

    Returns the saved tensor's shape. Saves as a torch ``.pt`` file so
    downstream visualisers can load and render in any colourmap.
    """
    attribution = gradient_attribution(output, target_embedding, encoder)
    torch.save(attribution, path)
    return tuple(attribution.shape)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_sae_decomposition(
    *, current_emb: torch.Tensor, sae: Any, top_k: int
) -> dict[int, float] | None:
    """Decompose ``current_emb`` into a sparse SAE feature dict."""
    try:
        if current_emb.dim() == 1:
            current_emb = current_emb.unsqueeze(0)
        activations = sae(current_emb)  # [1, n_features]
        activations = activations.detach().cpu().squeeze(0)
        active_indices = (activations > 0).nonzero(as_tuple=True)[0]
        if len(active_indices) == 0:
            return {}

        active_values = activations[active_indices]
        top_k = min(top_k, len(active_values))
        top_values, sort_indices = active_values.topk(top_k)
        top_feature_indices = active_indices[sort_indices]

        return {
            int(feat_idx): float(val)
            for feat_idx, val in zip(top_feature_indices.tolist(), top_values.tolist())
        }
    except Exception as exc:
        logger.warning("SAE decomposition failed: %s", exc, exc_info=True)
        return None


def _run_linear_probes(*, current_emb: torch.Tensor, probes: dict[str, Any]) -> dict[str, float]:
    """Evaluate each probe against the current embedding."""
    results: dict[str, float] = {}
    for name, probe in probes.items():
        try:
            value = probe(current_emb)
            if isinstance(value, torch.Tensor):
                value = value.item()
            results[name] = float(value)
        except Exception as exc:
            logger.warning("Linear probe '%s' failed: %s", name, exc, exc_info=True)
            results[name] = float("nan")
    return results
