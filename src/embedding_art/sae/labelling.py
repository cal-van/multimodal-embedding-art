"""
SAE feature labelling.

Two strategies are provided:

* :class:`CosineLabeller` — the default. For every SAE feature, scores the
  decoder direction against a fixed vocabulary of common English words and
  returns the highest-cosine word as the label. This is the existing
  :func:`embedding_art.sae.training.label_features` logic wrapped in a
  consistent protocol so it can be swapped for richer alternatives.

* :class:`VLMLabeller` — accepts a free-form ``llm_callable: (prompt: str)
  -> str`` adapter (e.g. an Anthropic / OpenAI Messages API thin wrapper)
  and, for each feature, prompts the LLM with the top-K nearest vocabulary
  words plus the feature's activating-example summary. The LLM is asked
  for a single short label. Falls back to the cosine label on any failure
  so a broken / quota-limited API never breaks the pipeline.

Both labellers implement the :class:`FeatureLabeller` protocol:

.. code-block:: python

    class FeatureLabeller(Protocol):
        def label_all(
            self,
            *,
            sae_path: Path,
            encoder: Any,
        ) -> list[str]: ...

Design note: a feature's "meaning" is fundamentally tied to the data the
SAE was trained on. The vocabulary-cosine method is a coarse proxy —
appropriate for the v3 showcase but eventually deserving of the richer
VLM-over-activating-examples treatment. The protocol shape lets both
coexist.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class FeatureLabeller(Protocol):
    """Anything that maps an SAE checkpoint to a list of feature labels."""

    def label_all(self, *, sae_path: Path, encoder: Any) -> list[str]: ...


class CosineLabeller:
    """The default vocabulary-cosine labeller.

    Wraps :func:`embedding_art.sae.training.label_features` to expose the
    :class:`FeatureLabeller` interface. The vocabulary size is fixed at
    the wrapped function's default (15k words) — the underlying list
    currently caps below that, which is fine.
    """

    def __init__(self, vocab_size: int = 15000) -> None:
        self.vocab_size = vocab_size

    def label_all(self, *, sae_path: Path, encoder: Any) -> list[str]:
        from embedding_art.sae.training import label_features

        return label_features(sae_path=sae_path, encoder=encoder, vocab_size=self.vocab_size)


class VLMLabeller:
    """Label each feature via an LLM prompt over the top-K nearest words.

    For every feature this calls ``llm_callable`` once with a prompt of
    the form::

        Given that an internal feature responds most strongly to the
        following words: storm, thunder, crackling, electric, lightning
        — give a single concise (1-3 words) label naming the underlying
        concept.

    The LLM's reply is stripped and used as the label. On any failure
    (network error, empty response, etc.) we fall back to the cosine
    label so a broken LLM endpoint never breaks the pipeline.

    Args:
        llm_callable: A function taking a single prompt string and
            returning a single response string. The simplest production
            adapter is roughly::

                def adapter(prompt: str) -> str:
                    return anthropic.Anthropic().messages.create(
                        model="claude-3-5-sonnet-latest",
                        max_tokens=20,
                        messages=[{"role": "user", "content": prompt}],
                    ).content[0].text
        top_k_words: How many nearest words to include in the prompt.
        fallback: The cosine labeller used to produce a fallback label
            when the LLM call fails. Pre-loaded with the same vocabulary
            so the labels stay consistent across runs.
    """

    DEFAULT_PROMPT_TEMPLATE = (
        "An internal feature in a multimodal neural network "
        "responds most strongly to these vocabulary words: {words}. "
        "Give a single concise 1-3 word label naming the underlying concept. "
        "Reply with the label only, no punctuation or quotes."
    )

    def __init__(
        self,
        llm_callable: Any,
        *,
        top_k_words: int = 8,
        fallback: FeatureLabeller | None = None,
        prompt_template: str | None = None,
    ) -> None:
        self.llm_callable = llm_callable
        self.top_k_words = top_k_words
        self.fallback = fallback or CosineLabeller()
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE

    def label_all(self, *, sae_path: Path, encoder: Any) -> list[str]:
        # First compute the top-K nearest vocabulary words per feature.
        # The straightforward way is to use the cosine machinery and
        # extract the top-K rather than top-1.
        per_feature_words = self._top_k_words_per_feature(sae_path=sae_path, encoder=encoder)

        # Always have a fallback ready in case any specific LLM call fails.
        fallback_labels = self.fallback.label_all(sae_path=sae_path, encoder=encoder)
        labels: list[str] = []

        for feat_idx, words in enumerate(per_feature_words):
            prompt = self.prompt_template.format(words=", ".join(words))
            try:
                response = self.llm_callable(prompt)
                if not isinstance(response, str):
                    raise TypeError(f"llm_callable returned {type(response).__name__}, want str")
                label = response.strip().strip("\"'")
                if not label:
                    raise ValueError("empty response")
                labels.append(label)
            except Exception as exc:
                logger.warning(
                    "VLM labelling failed for feature %d (falling back to cosine label '%s'): %s",
                    feat_idx,
                    fallback_labels[feat_idx],
                    exc,
                )
                labels.append(fallback_labels[feat_idx])

        return labels

    def _top_k_words_per_feature(self, *, sae_path: Path, encoder: Any) -> list[list[str]]:
        """Build ``[n_features][top_k]`` nearest vocabulary words per feature.

        Re-implements the inner loop of :func:`label_features` keeping the
        top-K rather than just top-1.
        """
        import torch
        import torch.nn.functional as F  # noqa: N812

        from embedding_art.sae.training import _get_vocab  # type: ignore[attr-defined]

        weights_file = sae_path / "sae_weights.pt" if sae_path.is_dir() else sae_path
        state = torch.load(weights_file, weights_only=True)
        w_dec = state["W_dec"]  # [embed_dim, n_features]
        n_features = w_dec.shape[1]

        vocab = _get_vocab(
            self.fallback.vocab_size if hasattr(self.fallback, "vocab_size") else 15000
        )
        word_embeddings: list[torch.Tensor] = []
        for word in vocab:
            try:
                emb = encoder.encode_text(word)
            except Exception:
                logger.warning("Could not encode word '%s'", word, exc_info=True)
                word_embeddings.append(torch.zeros(1, w_dec.shape[0]))
                continue
            word_embeddings.append(emb.detach().cpu().float())

        vocab_emb = torch.cat(word_embeddings, dim=0)
        vocab_emb = F.normalize(vocab_emb, dim=-1)
        decoder_cols = F.normalize(w_dec.T.float(), dim=-1)
        sim = decoder_cols @ vocab_emb.T  # [n_features, V]

        k = min(self.top_k_words, sim.shape[1])
        _, top_indices = sim.topk(k, dim=-1)
        per_feature: list[list[str]] = []
        for feat_idx in range(n_features):
            words = [vocab[i] for i in top_indices[feat_idx].tolist()]
            per_feature.append(words)
        return per_feature


class ActivatingExamplesLabeller:
    """Label each feature via an LLM prompt over the top-K *activating examples*.

    Strictly richer than :class:`VLMLabeller`: instead of asking the LLM
    to guess a concept from semantically-related vocabulary words, this
    finds the actual corpus entries (image / audio / video / text paths
    or strings) whose SAE encoding lights up the feature most strongly.
    The LLM sees the *source names* of those examples and is asked for
    the concept they share. This is the canonical pattern from
    Anthropic's circuits work (Bricken et al. 2023; Templeton et al.
    2024).

    Expects the corpus file saved by :func:`embedding_art.sae.training.
    collect_embeddings` (i.e. a dict with both ``"embeddings"`` and
    ``"sources"`` keys). When the file lacks sources we degrade to the
    cosine fallback rather than fabricating.

    Args:
        corpus_path: Path to the .pt file produced by ``sae collect``.
            Must contain ``"embeddings"`` (``[N, embed_dim]``) and
            ``"sources"`` (``list[str]`` of length ``N``).
        llm_callable: Same shape as :class:`VLMLabeller` —
            ``(prompt: str) -> str``.
        top_k_examples: How many activating examples per feature to
            include in the prompt.
        fallback: Used per-feature when the LLM call fails. Defaults
            to :class:`CosineLabeller`.
    """

    DEFAULT_PROMPT_TEMPLATE = (
        "An internal feature in a multimodal neural network activates "
        "most strongly on these {n} examples: {examples}. "
        "Give a single concise 1-3 word label naming the concept they share. "
        "Reply with the label only, no punctuation or quotes."
    )

    def __init__(
        self,
        *,
        corpus_path: Path,
        llm_callable: Any,
        top_k_examples: int = 8,
        fallback: FeatureLabeller | None = None,
        prompt_template: str | None = None,
    ) -> None:
        self.corpus_path = Path(corpus_path)
        self.llm_callable = llm_callable
        self.top_k_examples = top_k_examples
        self.fallback = fallback or CosineLabeller()
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE

    def label_all(self, *, sae_path: Path, encoder: Any) -> list[str]:
        per_feature_examples = self._top_k_examples_per_feature(sae_path=sae_path)
        if per_feature_examples is None:
            logger.warning(
                "Corpus at %s has no 'sources' key; falling back to cosine labelling.",
                self.corpus_path,
            )
            return self.fallback.label_all(sae_path=sae_path, encoder=encoder)

        fallback_labels = self.fallback.label_all(sae_path=sae_path, encoder=encoder)
        labels: list[str] = []
        for feat_idx, examples in enumerate(per_feature_examples):
            prompt = self.prompt_template.format(
                n=len(examples),
                examples=", ".join(examples),
            )
            try:
                response = self.llm_callable(prompt)
                if not isinstance(response, str):
                    raise TypeError(f"llm_callable returned {type(response).__name__}, want str")
                label = response.strip().strip("\"'")
                if not label:
                    raise ValueError("empty response")
                labels.append(label)
            except Exception as exc:
                logger.warning(
                    "Activating-examples labelling failed for feature %d "
                    "(falling back to cosine label '%s'): %s",
                    feat_idx,
                    fallback_labels[feat_idx],
                    exc,
                )
                labels.append(fallback_labels[feat_idx])
        return labels

    def _top_k_examples_per_feature(self, *, sae_path: Path) -> list[list[str]] | None:
        """Return ``[n_features][top_k]`` source names per feature.

        Returns ``None`` when the corpus has no ``"sources"`` key, so
        the caller can degrade gracefully.
        """
        import torch

        corpus = torch.load(self.corpus_path, weights_only=True)
        if not isinstance(corpus, dict) or "sources" not in corpus:
            return None
        sources: list[str] = list(corpus["sources"])
        embeddings: torch.Tensor = corpus["embeddings"].float()

        weights_file = sae_path / "sae_weights.pt" if sae_path.is_dir() else sae_path
        state = torch.load(weights_file, weights_only=True)
        w_enc = state["W_enc"].float()  # [n_features, embed_dim]
        bias = state.get("bias")
        if bias is None:
            bias = torch.zeros(w_enc.shape[0])
        else:
            bias = bias.float()

        # Activations of every corpus row at every feature:  [N, n_features].
        # GroupSparseSAE has per-modality pre-biases — we omit them here
        # because the corpus is single-modality and the offset is shared.
        acts = (embeddings @ w_enc.T + bias).clamp_min_(0.0)
        n_features = acts.shape[1]
        k = min(self.top_k_examples, acts.shape[0])
        _, top_idx = acts.topk(k, dim=0)  # [k, n_features]

        per_feature: list[list[str]] = []
        for feat_idx in range(n_features):
            indices = top_idx[:, feat_idx].tolist()
            per_feature.append([sources[i] for i in indices])
        return per_feature
