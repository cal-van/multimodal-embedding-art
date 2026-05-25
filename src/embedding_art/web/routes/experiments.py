"""HTTP surface for the experiments package.

Exposes two synchronous endpoints:

* ``POST /experiments/anchor-compare`` — Encodes one or more text references
  through the canonical multimodal encoder and reports pairwise cosine
  similarities + per-reference top-K text-anchor readouts. JSON body.
* ``POST /experiments/anchor-compare-multimodal`` — Encodes the *same*
  concept through up to four modalities (text + image + audio + video)
  in one shared space and reports the pairwise cosine matrix + per-
  modality text-anchor readouts. Multipart body for file uploads.

These intentionally do NOT use the JobManager / websocket lifecycle:
the work is fast (sub-second for text, a few seconds for multimodal
with image/audio/video uploads) and the response payloads are small
enough to return inline.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
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


# ---------------------------------------------------------------------------
# Multi-modal anchor compare
# ---------------------------------------------------------------------------


class AnchorCompareMultimodalEntry(BaseModel):
    """Per-modality record for the multimodal anchor-compare response."""

    modality: str
    embedding_dim: int
    text_anchor: list[dict[str, Any]] = Field(default_factory=list)
    source: str | None = None


class AnchorCompareMultimodalResponse(BaseModel):
    concept_label: str
    encoder: str
    entries: list[AnchorCompareMultimodalEntry]
    # ``cosine_matrix[modality_i][modality_j]`` — symmetric.
    cosine_matrix: dict[str, dict[str, float]]


async def _spool_upload(upload: UploadFile | None, suffix: str) -> Path | None:
    """Write an UploadFile to a temp file and return the path, or ``None``.

    Caller is responsible for ``Path.unlink()``. We use a tempfile rather
    than streaming so the underlying encoder loaders (LanguageBind's
    image / audio / video sub-encoders) can be given a filesystem path
    without a separate streaming-decoder branch.
    """
    if upload is None:
        return None
    data = await upload.read()
    if not data:
        return None
    fd = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        fd.write(data)
    finally:
        fd.close()
    return Path(fd.name)


@router.post(
    "/anchor-compare-multimodal",
    response_model=AnchorCompareMultimodalResponse,
)
async def anchor_compare_multimodal(
    concept_label: str = Form(...),
    encoder: str = Form("languagebind"),
    top_k_text: int = Form(12),
    text: str | None = Form(None),
    image: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
    video: UploadFile | None = File(None),
) -> AnchorCompareMultimodalResponse:
    """Multimodal Platonic-representation probe.

    Encode the *same* concept across text + image + audio + video in
    one shared LanguageBind embedding space and return the cross-
    modal cosine matrix + per-modality text-anchor readouts. Supply
    only the modalities you have references for — at least one is
    required, but the experiment is most informative with three or
    four.
    """
    from embedding_art.encoders.defaults import create_default_registry
    from embedding_art.experiments import EncoderActivationCache, run_anchor_comparison

    refs_present = sum(x is not None for x in (text and text.strip() or None, image, audio, video))
    if refs_present == 0:
        raise HTTPException(
            status_code=400,
            detail="anchor-compare-multimodal requires at least one of text / image / audio / video",
        )

    try:
        registry = create_default_registry()
        loaded_encoder = registry.load(encoder)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load encoder '{encoder}': {e}"
        ) from e

    image_path = await _spool_upload(image, suffix=".png")
    audio_path = await _spool_upload(audio, suffix=".wav")
    video_path = await _spool_upload(video, suffix=".mp4")

    # Activation cache lives under the process working dir; identical
    # uploads (e.g. the user re-running the same comparison) avoid the
    # repeat encoder forward.
    cache = EncoderActivationCache(
        Path(os.getcwd()) / "outputs" / "anchor_compare_cache",
        encoder_id=encoder,
    )

    try:
        result = run_anchor_comparison(
            concept_label=concept_label,
            encoder=loaded_encoder,
            encoder_name=encoder,
            text=text.strip() if text else None,
            image_path=image_path,
            audio_path=audio_path,
            video_path=video_path,
            top_k_text=top_k_text,
            cache=cache,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"anchor-compare-multimodal failed: {e}") from e
    finally:
        # Clean up spooled uploads. Don't fail if a file disappeared.
        for tmp in (image_path, audio_path, video_path):
            if tmp is not None:
                try:
                    tmp.unlink()
                except FileNotFoundError:
                    pass

    entries: list[AnchorCompareMultimodalEntry] = []
    for modality, record in result.modalities.items():
        anchor_records: list[dict[str, Any]] = []
        for word, sim in record.get("text_anchor", []) or []:
            anchor_records.append({"word": word, "similarity": float(sim)})
        entries.append(
            AnchorCompareMultimodalEntry(
                modality=modality,
                embedding_dim=int(record.get("embedding").shape[-1]),
                text_anchor=anchor_records,
                source=record.get("source"),
            )
        )

    return AnchorCompareMultimodalResponse(
        concept_label=result.concept_label,
        encoder=result.encoder_name,
        entries=entries,
        cosine_matrix=result.cosine_matrix,
    )
