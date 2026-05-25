"""HTTP surface for the experiments package.

Currently exposes a single synchronous endpoint:

* ``POST /experiments/anchor-compare`` — Encodes one or more text references
  through the canonical multimodal encoder and reports pairwise cosine
  similarities + per-reference top-K text-anchor readouts.

This intentionally does NOT use the JobManager / websocket lifecycle: the
work is fast (sub-second for typical inputs) and the response payload is
small enough to return inline. Multipart upload support (for image / audio /
video references) is a planned follow-up.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/experiments", tags=["experiments"])


class AnchorCompareRequest(BaseModel):
    """Body for ``POST /experiments/anchor-compare``."""

    concept_label: str = Field(..., description="Human-readable concept label.")
    texts: list[str] = Field(..., description="Two or more text references to encode and compare.")
    encoder: str = "languagebind"
    top_k_text: int = 12

    @field_validator("texts")
    @classmethod
    def _at_least_two(cls, v: list[str]) -> list[str]:
        cleaned = [t for t in v if t and t.strip()]
        if len(cleaned) < 2:
            raise ValueError("anchor-compare requires at least two non-empty text references")
        return cleaned


class AnchorCompareTextEntry(BaseModel):
    label: str
    text: str
    embedding_dim: int
    text_anchor: list[dict[str, Any]] = Field(default_factory=list)


class AnchorCompareResponse(BaseModel):
    concept_label: str
    encoder: str
    entries: list[AnchorCompareTextEntry]
    # ``cosine_matrix[label_i][label_j]`` is the cosine similarity between
    # the two text encodings. Symmetric by construction.
    cosine_matrix: dict[str, dict[str, float]]


@router.post("/anchor-compare", response_model=AnchorCompareResponse)
async def anchor_compare(request: AnchorCompareRequest) -> AnchorCompareResponse:
    """Encode all texts in one shared space; return pairwise cosines + anchors.

    This is the Platonic-representation probe restricted to text inputs:
    do semantically similar text references actually land at the same
    point in the canonical encoder's embedding space?
    """
    from embedding_art.encoders.defaults import create_default_registry
    from embedding_art.interpretation.text_anchor import text_anchor_readout

    try:
        registry = create_default_registry()
        encoder = registry.load(request.encoder)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load encoder '{request.encoder}': {e}"
        ) from e

    if not hasattr(encoder, "encode_text"):
        raise HTTPException(
            status_code=500,
            detail=f"Encoder '{request.encoder}' does not expose encode_text",
        )

    embeddings: list[torch.Tensor] = []
    entries: list[AnchorCompareTextEntry] = []
    seen_labels: set[str] = set()

    for raw in request.texts:
        text = raw.strip()
        label = text if text not in seen_labels else f"{text} ({len(seen_labels)})"
        seen_labels.add(label)
        try:
            emb = encoder.encode_text(text)
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"encode_text failed for '{text}': {e}"
            ) from e

        if not isinstance(emb, torch.Tensor):
            raise HTTPException(
                status_code=500,
                detail=f"encode_text returned {type(emb).__name__}, expected torch.Tensor",
            )
        flat = emb.squeeze().detach().cpu()
        if flat.ndim == 0 or flat.numel() == 0:
            raise HTTPException(status_code=500, detail="encode_text returned an empty tensor")

        anchor_records: list[dict[str, Any]] = []
        try:
            raw_anchor = text_anchor_readout(flat, encoder, top_k=request.top_k_text)
            anchor_records = [{"word": word, "similarity": float(sim)} for word, sim in raw_anchor]
        except Exception:
            logger.warning("text_anchor_readout failed for '%s'", text, exc_info=True)

        entries.append(
            AnchorCompareTextEntry(
                label=label,
                text=text,
                embedding_dim=int(flat.shape[-1]),
                text_anchor=anchor_records,
            )
        )
        embeddings.append(flat)

    cosine_matrix = _pairwise_cosine(embeddings, [e.label for e in entries])

    return AnchorCompareResponse(
        concept_label=request.concept_label,
        encoder=request.encoder,
        entries=entries,
        cosine_matrix=cosine_matrix,
    )


def _pairwise_cosine(
    embeddings: list[torch.Tensor], labels: list[str]
) -> dict[str, dict[str, float]]:
    """Symmetric label-keyed cosine matrix."""
    matrix: dict[str, dict[str, float]] = {label: {} for label in labels}
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            if j < i:
                # Mirror the already-computed value (cosine is symmetric).
                matrix[label_i][label_j] = matrix[label_j][label_i]
                continue
            if i == j:
                matrix[label_i][label_j] = 1.0
                continue
            a = embeddings[i].float()
            b = embeddings[j].float()
            sim = float(torch.nn.functional.cosine_similarity(a, b, dim=-1).item())
            matrix[label_i][label_j] = sim
    return matrix
