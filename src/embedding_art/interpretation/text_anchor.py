"""
Text-anchor readout — the headline LanguageBind interpretability move.

For any embedding (regardless of which modality produced it), project back
into language space by finding the top-K vocabulary words whose text
embedding has highest cosine similarity to it. Reports both the words and
their similarities so the caller can detect 'flat' readouts where no word
dominates.

This is the canonical interpretability signal for v3 because LanguageBind
*anchors* all modalities on language. The same readout works on an image
embedding, an audio embedding, a video embedding, or a text embedding — and
because they share the space, the readout is comparable across modalities.

Performance note: encoding the vocabulary is the expensive step (one
``encode_text`` call per word). The vocabulary embeddings are cached
per-encoder instance, so subsequent readouts on the same encoder are cheap.
"""

from __future__ import annotations

import logging
from typing import Any
from weakref import WeakKeyDictionary

import torch
import torch.nn.functional as F  # noqa: N812

logger = logging.getLogger(__name__)


# Cache vocabulary embeddings per (encoder, vocab_tuple) so repeated calls
# during a single showcase run don't re-encode 1500 words four times.
_VOCAB_CACHE: WeakKeyDictionary[Any, dict[tuple, torch.Tensor]] = WeakKeyDictionary()


def text_anchor_readout(
    embedding: torch.Tensor,
    encoder: Any,
    vocab: list[str] | None = None,
    top_k: int = 20,
) -> list[tuple[str, float]]:
    """Project ``embedding`` back into language space.

    Encodes every word in ``vocab`` via ``encoder.encode_text`` and returns
    the top-K closest by cosine similarity.

    Args:
        embedding: ``[D]`` or ``[1, D]`` float tensor in the canonical
            embedding space. Will be unit-normalised before comparison.
        encoder: Any encoder exposing ``encode_text(word) -> [1, D]``.
        vocab: Optional list of vocabulary words. Defaults to the
            built-in :func:`embedding_art.sae.training._get_vocab` list
            (≈1500 common English words).
        top_k: How many top-similarity words to return.

    Returns:
        List of (word, cosine_similarity) tuples in descending order. Has
        length ``min(top_k, len(vocab))``.
    """
    if vocab is None:
        from embedding_art.sae.training import _get_vocab

        vocab = _get_vocab(1500)

    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)
    target_norm = F.normalize(embedding, dim=-1)  # [1, D]

    vocab_embeddings = _encode_or_cache_vocab(encoder, vocab)  # [V, D]

    sims = (vocab_embeddings @ target_norm.squeeze(0)).cpu()  # [V]
    top_k = min(top_k, len(vocab))
    top_values, top_indices = sims.topk(top_k)

    return [(vocab[idx], float(val)) for val, idx in zip(top_values.tolist(), top_indices.tolist())]


def _encode_or_cache_vocab(encoder: Any, vocab: list[str]) -> torch.Tensor:
    """Return cached vocabulary embeddings or compute and cache them."""
    cache_key = tuple(vocab)
    try:
        encoder_cache = _VOCAB_CACHE.setdefault(encoder, {})
    except TypeError:
        # Non-hashable / non-weakref-able encoder; skip caching.
        encoder_cache = None

    if encoder_cache is not None and cache_key in encoder_cache:
        return encoder_cache[cache_key]

    embeddings: list[torch.Tensor] = []
    for word in vocab:
        try:
            emb = encoder.encode_text(word)  # [1, D]
        except Exception:
            logger.warning("text_anchor: could not encode '%s' — using zero vector.", word)
            placeholder_dim = embeddings[0].shape[-1] if embeddings else 768
            emb = torch.zeros(1, placeholder_dim)
        embeddings.append(emb.detach().cpu().float())

    matrix = torch.cat(embeddings, dim=0)  # [V, D]
    matrix = F.normalize(matrix, dim=-1)
    if encoder_cache is not None:
        encoder_cache[cache_key] = matrix
    return matrix
