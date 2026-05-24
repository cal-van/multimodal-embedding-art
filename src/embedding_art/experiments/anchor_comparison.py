"""
Anchor-comparison experiment (M9).

Given a concept and zero or more modality-specific reference inputs (a
text string, an image file, an audio file, a video file), encode each
through the canonical multimodal encoder and report:

* The 768-d canonical embedding from each modality.
* Pairwise cosine similarities (the cross-modal agreement matrix).
* For each modality, a text-anchor readout — the top-K nearest
  vocabulary words in language space.
* When a trained SAE is supplied, pairwise SAE-decomposition deltas:
  the symmetric difference of each pair's top-K active features.

This is the canonical 'Platonic representation' probe under
LanguageBind: do the same concept's text/image/audio/video encodings
actually land at the same point in the shared space?

The result is a JSON-serialisable :class:`AnchorComparison` record. The
:func:`run_anchor_comparison` orchestrator handles graceful degradation:
modalities for which no reference is supplied are simply absent from the
result; failures in the SAE decomposition are logged and skipped without
killing the rest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

from embedding_art.interpretation.text_anchor import text_anchor_readout

logger = logging.getLogger(__name__)


@dataclass
class AnchorComparison:
    """Per-modality anchor-comparison record."""

    concept_label: str
    encoder_name: str
    modalities: dict[str, dict[str, Any]] = field(default_factory=dict)
    cosine_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    sae_feature_overlap: dict[str, dict[str, dict[str, Any]]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Render to a JSON-friendly dict (drops tensor fields)."""
        modalities_dict = {}
        for mod, record in self.modalities.items():
            clean = {k: v for k, v in record.items() if k != "embedding"}
            # Always keep an 'embedding_dim' field so the manifest reader
            # can tell which modalities actually ran.
            if "embedding" in record:
                clean["embedding_dim"] = int(record["embedding"].shape[-1])
            modalities_dict[mod] = clean

        out = {
            "concept_label": self.concept_label,
            "encoder": self.encoder_name,
            "modalities": modalities_dict,
            "cosine_matrix": self.cosine_matrix,
        }
        if self.sae_feature_overlap is not None:
            out["sae_feature_overlap"] = self.sae_feature_overlap
        return out


def run_anchor_comparison(
    *,
    concept_label: str,
    encoder: Any,
    encoder_name: str,
    text: str | None = None,
    image_path: str | Path | None = None,
    audio_path: str | Path | None = None,
    video_path: str | Path | None = None,
    sae: Any = None,
    top_k_text: int = 20,
    top_k_features: int = 32,
) -> AnchorComparison:
    """Encode the concept through each available modality and compare.

    Args:
        concept_label: Human-readable label of the concept (e.g.
            ``"thunder"``). Stored in the result for the manifest.
        encoder: The canonical multimodal encoder (typically
            LanguageBind). Must expose the per-modality
            ``encode_text`` / ``encode_image`` / ``encode_audio`` /
            ``encode_video`` methods as appropriate.
        encoder_name: Name used in the result for traceability.
        text: Optional text reference.
        image_path: Optional path to an image reference.
        audio_path: Optional path to an audio reference.
        video_path: Optional path to a video reference.
        sae: Optional trained SAE. When provided, also compute feature
            overlap pairwise across modalities.
        top_k_text: Top-K text-anchor words to record per modality.
        top_k_features: Number of SAE features to retain per
            modality for the overlap computation.

    Returns:
        Populated :class:`AnchorComparison`.
    """
    result = AnchorComparison(concept_label=concept_label, encoder_name=encoder_name)

    modality_inputs = [
        ("text", text, "encode_text"),
        ("image", image_path, "encode_image"),
        ("audio", audio_path, "encode_audio"),
        ("video", video_path, "encode_video"),
    ]

    for modality, ref, encode_method_name in modality_inputs:
        if ref is None:
            continue
        if not hasattr(encoder, encode_method_name):
            logger.warning(
                "Encoder %s has no %s — skipping %s modality.",
                encoder_name,
                encode_method_name,
                modality,
            )
            continue

        try:
            encoded = getattr(encoder, encode_method_name)(ref)
            if not isinstance(encoded, torch.Tensor):
                raise TypeError(f"encode_{modality} returned {type(encoded)}")

            embedding = encoded.detach().cpu().float()
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)

            record: dict[str, Any] = {
                "reference": str(ref) if isinstance(ref, (str, Path)) else "<tensor>",
                "embedding": embedding,
                "text_anchor": [],
            }

            try:
                record["text_anchor"] = [
                    {"word": w, "similarity": s}
                    for w, s in text_anchor_readout(embedding, encoder, top_k=top_k_text)
                ]
            except Exception as exc:
                logger.warning(
                    "text-anchor readout failed for %s: %s", modality, exc, exc_info=True
                )

            result.modalities[modality] = record

        except Exception as exc:
            logger.warning("Failed to encode %s: %s", modality, exc, exc_info=True)

    result.cosine_matrix = _compute_cosine_matrix(result.modalities)

    if sae is not None:
        result.sae_feature_overlap = _compute_sae_feature_overlap(
            modalities=result.modalities,
            sae=sae,
            top_k_features=top_k_features,
        )

    return result


def _compute_cosine_matrix(modalities: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Compute the pairwise cosine-similarity matrix across modalities."""
    matrix: dict[str, dict[str, float]] = {}
    keys = sorted(modalities.keys())
    for k1 in keys:
        matrix[k1] = {}
        emb1 = F.normalize(modalities[k1]["embedding"], dim=-1)
        for k2 in keys:
            if k1 == k2:
                matrix[k1][k2] = 1.0
                continue
            emb2 = F.normalize(modalities[k2]["embedding"], dim=-1)
            sim = (emb1 * emb2).sum(dim=-1).mean().item()
            matrix[k1][k2] = float(sim)
    return matrix


def _compute_sae_feature_overlap(
    *, modalities: dict[str, dict[str, Any]], sae: Any, top_k_features: int
) -> dict[str, dict[str, dict[str, Any]]]:
    """Per-pair SAE feature overlap.

    For each pair of modalities (m1, m2), decompose both embeddings
    through the SAE, take the top-K active features of each, and report
    intersection / unique-to-m1 / unique-to-m2 / Jaccard similarity.
    """
    feature_sets: dict[str, set[int]] = {}
    for modality, record in modalities.items():
        try:
            embedding = record["embedding"]
            activations = sae(embedding).detach().cpu().squeeze(0)
            active = (activations > 0).nonzero(as_tuple=True)[0]
            if len(active) == 0:
                feature_sets[modality] = set()
                continue
            values = activations[active]
            k = min(top_k_features, len(values))
            _, top_indices_within = values.topk(k)
            top_features = active[top_indices_within]
            feature_sets[modality] = {int(idx) for idx in top_features.tolist()}
        except Exception as exc:
            logger.warning("SAE decomposition failed for %s: %s", modality, exc, exc_info=True)
            feature_sets[modality] = set()

    overlap: dict[str, dict[str, dict[str, Any]]] = {}
    keys = sorted(feature_sets.keys())
    for k1 in keys:
        overlap[k1] = {}
        for k2 in keys:
            if k1 == k2:
                overlap[k1][k2] = {
                    "intersection": sorted(feature_sets[k1]),
                    "unique_to_a": [],
                    "unique_to_b": [],
                    "jaccard": 1.0 if feature_sets[k1] else 0.0,
                }
                continue
            inter = feature_sets[k1] & feature_sets[k2]
            union = feature_sets[k1] | feature_sets[k2]
            overlap[k1][k2] = {
                "intersection": sorted(inter),
                "unique_to_a": sorted(feature_sets[k1] - feature_sets[k2]),
                "unique_to_b": sorted(feature_sets[k2] - feature_sets[k1]),
                "jaccard": (len(inter) / len(union)) if union else 0.0,
            }
    return overlap
