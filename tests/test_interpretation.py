"""
Tests for the interpretation bundle (M7).

Uses small mocked encoders and SAEs so the tests run instantly without
loading real LanguageBind or trained SAE checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import torch

from embedding_art.interpretation import (
    InterpretationBundle,
    gradient_attribution,
    run_interpretation,
    text_anchor_readout,
)

# ---------------------------------------------------------------------------
# Fake encoder
# ---------------------------------------------------------------------------


@dataclass
class _FakeEncoder:
    """Deterministic text encoder for unit tests.

    ``encode_text(word)`` returns a fixed-length embedding derived from a
    hash of the word so the same word always maps to the same point and
    different words to different points.
    """

    embed_dim: int = 32

    def encode_text(self, word: str) -> torch.Tensor:
        seed = abs(hash(word)) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        emb = torch.randn(1, self.embed_dim, generator=gen)
        return emb / emb.norm(dim=-1, keepdim=True)

    def encode_for_optimization(self, x: torch.Tensor) -> torch.Tensor:
        # Reduce x to embed_dim by flattening + projection-like operation.
        if x.dim() == 3:
            x = x.unsqueeze(0)
        flat = x.flatten(start_dim=1)
        if flat.shape[1] < self.embed_dim:
            flat = flat.repeat(1, (self.embed_dim // flat.shape[1]) + 1)
        emb = flat[:, : self.embed_dim]
        return emb / emb.norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# text_anchor_readout
# ---------------------------------------------------------------------------


class TestTextAnchorReadout:
    """The text-anchor readout returns top-K vocab words by cosine similarity."""

    def test_returns_top_k_tuples(self) -> None:
        encoder = _FakeEncoder()
        embedding = torch.randn(1, 32)
        result = text_anchor_readout(
            embedding, encoder, vocab=["dog", "cat", "bird", "fish", "tree"], top_k=3
        )
        assert len(result) == 3
        assert all(isinstance(t, tuple) and len(t) == 2 for t in result)
        assert all(isinstance(t[0], str) for t in result)
        assert all(isinstance(t[1], float) for t in result)

    def test_results_are_sorted_descending(self) -> None:
        encoder = _FakeEncoder()
        embedding = torch.randn(1, 32)
        result = text_anchor_readout(
            embedding, encoder, vocab=["dog", "cat", "bird", "fish", "tree"], top_k=5
        )
        sims = [s for _, s in result]
        assert sims == sorted(sims, reverse=True)

    def test_self_match_is_top(self) -> None:
        """When the embedding IS a vocab word, that word should rank first."""
        encoder = _FakeEncoder()
        vocab = ["dog", "cat", "bird", "fish", "tree"]
        target_word = "fish"
        embedding = encoder.encode_text(target_word)
        result = text_anchor_readout(embedding, encoder, vocab=vocab, top_k=5)
        assert result[0][0] == target_word

    def test_top_k_caps_at_vocab_size(self) -> None:
        encoder = _FakeEncoder()
        result = text_anchor_readout(torch.randn(1, 32), encoder, vocab=["a", "b"], top_k=100)
        assert len(result) == 2

    def test_works_with_1d_embedding(self) -> None:
        encoder = _FakeEncoder()
        embedding = torch.randn(32)
        result = text_anchor_readout(embedding, encoder, vocab=["dog", "cat"], top_k=2)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# gradient_attribution
# ---------------------------------------------------------------------------


class TestGradientAttribution:
    """``gradient_attribution`` produces a same-shape, non-negative tensor."""

    def test_output_shape_matches_input(self) -> None:
        encoder = _FakeEncoder()
        output = torch.rand(1, 3, 32, 32)
        target = torch.randn(1, 32)
        attribution = gradient_attribution(output, target, encoder)
        assert attribution.shape == output.shape

    def test_attribution_is_nonnegative(self) -> None:
        encoder = _FakeEncoder()
        output = torch.rand(1, 3, 32, 32)
        target = torch.randn(1, 32)
        attribution = gradient_attribution(output, target, encoder)
        assert (attribution >= 0).all()

    def test_attribution_is_not_all_zero(self) -> None:
        encoder = _FakeEncoder()
        output = torch.rand(1, 3, 32, 32)
        target = torch.randn(1, 32)
        attribution = gradient_attribution(output, target, encoder)
        assert attribution.sum() > 0


# ---------------------------------------------------------------------------
# InterpretationBundle
# ---------------------------------------------------------------------------


class TestInterpretationBundle:
    """``InterpretationBundle.to_dict`` is JSON-friendly."""

    def test_empty_bundle_to_dict(self) -> None:
        bundle = InterpretationBundle()
        d = bundle.to_dict()
        # All optional fields are None or empty.
        assert d["text_anchor"] == []
        assert d["sae_decomposition"] is None
        assert d["sae_corrsteer"] is None
        assert d["linear_probes"] is None
        assert d["attribution_path"] is None
        assert d["attribution_shape"] is None

    def test_populated_bundle_to_dict(self) -> None:
        bundle = InterpretationBundle(
            text_anchor=[("goldfish", 0.95), ("fish", 0.88)],
            sae_decomposition={5: 1.5, 12: 0.7},
            sae_feature_labels={5: "scales", 12: "orange"},
            sae_corrsteer=[(5, 0.92), (12, 0.45)],
            linear_probes={"is_animal": 0.99},
            attribution_path="attribution.pt",
            attribution_shape=(1, 3, 1024, 1024),
            final_similarity=0.91,
        )
        d = bundle.to_dict()
        assert d["text_anchor"] == [
            {"word": "goldfish", "similarity": 0.95},
            {"word": "fish", "similarity": 0.88},
        ]
        assert d["sae_decomposition"] == {5: 1.5, 12: 0.7}
        assert d["sae_corrsteer"] == [
            {"feature": 5, "correlation": 0.92},
            {"feature": 12, "correlation": 0.45},
        ]
        assert d["linear_probes"] == {"is_animal": 0.99}
        assert d["attribution_shape"] == [1, 3, 1024, 1024]


# ---------------------------------------------------------------------------
# run_interpretation
# ---------------------------------------------------------------------------


class TestRunInterpretation:
    """``run_interpretation`` orchestrates the optional signals correctly."""

    def test_minimal_run_returns_text_anchor(self) -> None:
        encoder = _FakeEncoder()
        target = MagicMock()
        target.embedding = encoder.encode_text("goldfish")
        output = torch.rand(1, 3, 32, 32)
        bundle = run_interpretation(
            target=target, output=output, encoder=encoder, final_similarity=0.9
        )
        assert isinstance(bundle, InterpretationBundle)
        assert len(bundle.text_anchor) > 0
        assert bundle.sae_decomposition is None
        assert bundle.sae_corrsteer is None
        assert bundle.linear_probes is None
        assert bundle.final_similarity == 0.9

    def test_sae_decomposition_when_sae_provided(self) -> None:
        encoder = _FakeEncoder()
        target = MagicMock()
        target.embedding = encoder.encode_text("goldfish")
        output = torch.rand(1, 3, 32, 32)

        # Fake SAE that returns a sparse activation vector.
        fake_sae = MagicMock()
        activations = torch.zeros(1, 64)
        activations[0, 5] = 1.5
        activations[0, 12] = 0.7
        activations[0, 30] = 0.3
        fake_sae.return_value = activations

        bundle = run_interpretation(
            target=target,
            output=output,
            encoder=encoder,
            sae=fake_sae,
            top_k_features=2,
        )
        assert bundle.sae_decomposition is not None
        assert len(bundle.sae_decomposition) == 2
        # Top features by value should be 5 (1.5) and 12 (0.7).
        assert 5 in bundle.sae_decomposition
        assert 12 in bundle.sae_decomposition

    def test_linear_probes_evaluated_when_provided(self) -> None:
        encoder = _FakeEncoder()
        target = MagicMock()
        target.embedding = encoder.encode_text("goldfish")
        output = torch.rand(1, 3, 32, 32)

        probes = {
            "is_animal": MagicMock(return_value=torch.tensor(0.95)),
            "is_warm_coloured": MagicMock(return_value=torch.tensor(0.78)),
        }
        bundle = run_interpretation(
            target=target,
            output=output,
            encoder=encoder,
            linear_probes=probes,
        )
        assert bundle.linear_probes is not None
        assert set(bundle.linear_probes.keys()) == {"is_animal", "is_warm_coloured"}
        assert abs(bundle.linear_probes["is_animal"] - 0.95) < 1e-5
        assert abs(bundle.linear_probes["is_warm_coloured"] - 0.78) < 1e-5
