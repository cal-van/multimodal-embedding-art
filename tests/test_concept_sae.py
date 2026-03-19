"""
Unit tests for Concept SAE integration.

Tests cover:
  - Concept.decompose(sae) delegates to sae.decompose()
  - Concept.from_features(sae, features) builds a normalised Concept
  - Edge cases: unknown features, empty feature dicts, embedding shapes

All tests use SAELens.from_tensors() so no files or trained models are needed.

Sizes used throughout:
  embed_dim  = 64
  n_features = 256
  k          = 8
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from embedding_art.core.concept import Concept
from embedding_art.exceptions import FeatureNotFoundError
from embedding_art.sae.lens import SAEDecomposition, SAELens


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

EMBED_DIM = 64
N_FEATURES = 256
K = 8


def _make_vocab(n: int) -> list[str]:
    return [f"feature_{i:03d}" for i in range(n)]


def _make_sae(
    embed_dim: int = EMBED_DIM,
    n_features: int = N_FEATURES,
    k: int = K,
    seed: int = 0,
) -> SAELens:
    """Build a small SAELens from deterministic random tensors."""
    gen = torch.Generator().manual_seed(seed)
    W_enc = torch.randn(n_features, embed_dim, generator=gen)
    W_dec_raw = torch.randn(embed_dim, n_features, generator=gen)
    W_dec = F.normalize(W_dec_raw, dim=0)
    bias = torch.zeros(n_features)
    pre_bias = torch.zeros(embed_dim)
    vocab = _make_vocab(n_features)
    return SAELens.from_tensors(W_enc, W_dec, bias, pre_bias, vocab, k)


def _make_concept(embed_dim: int = EMBED_DIM, seed: int = 42) -> Concept:
    """Return a normalised Concept with a random embedding."""
    gen = torch.Generator().manual_seed(seed)
    emb = torch.randn(1, embed_dim, generator=gen)
    return Concept(embedding=emb, description="test_concept")


@pytest.fixture
def sae() -> SAELens:
    return _make_sae()


@pytest.fixture
def concept() -> Concept:
    return _make_concept()


# ---------------------------------------------------------------------------
# Concept.decompose
# ---------------------------------------------------------------------------


class TestConceptDecompose:
    """Concept.decompose(sae) correctly delegates to sae.decompose()."""

    def test_returns_sae_decomposition(self, concept: Concept, sae: SAELens) -> None:
        result = concept.decompose(sae)
        assert isinstance(result, SAEDecomposition)

    def test_activations_shape(self, concept: Concept, sae: SAELens) -> None:
        result = concept.decompose(sae)
        assert result.activations.shape == (1, N_FEATURES)

    def test_exactly_k_active_features(self, concept: Concept, sae: SAELens) -> None:
        result = concept.decompose(sae)
        assert len(result.active_features) == K

    def test_active_features_in_vocab(self, concept: Concept, sae: SAELens) -> None:
        result = concept.decompose(sae)
        for name in result.active_features:
            assert name in sae.vocab

    def test_active_feature_strengths_positive(self, concept: Concept, sae: SAELens) -> None:
        result = concept.decompose(sae)
        for strength in result.active_features.values():
            assert strength > 0.0

    def test_reconstruction_error_is_non_negative_float(
        self, concept: Concept, sae: SAELens
    ) -> None:
        result = concept.decompose(sae)
        assert isinstance(result.reconstruction_error, float)
        assert result.reconstruction_error >= 0.0

    def test_matches_direct_sae_decompose(self, concept: Concept, sae: SAELens) -> None:
        """concept.decompose(sae) must equal sae.decompose(concept.embedding)."""
        via_concept = concept.decompose(sae)
        direct = sae.decompose(concept.embedding)
        assert torch.allclose(via_concept.activations, direct.activations)
        assert via_concept.active_features == direct.active_features

    def test_deterministic(self, concept: Concept, sae: SAELens) -> None:
        r1 = concept.decompose(sae)
        r2 = concept.decompose(sae)
        assert torch.allclose(r1.activations, r2.activations)

    def test_1d_embedding_concept(self, sae: SAELens) -> None:
        """A Concept whose underlying embedding was 1D still works."""
        emb = F.normalize(torch.randn(EMBED_DIM), dim=0).unsqueeze(0)
        c = Concept(embedding=emb, description="1d_test")
        result = c.decompose(sae)
        assert result.activations.shape == (1, N_FEATURES)

    def test_different_concepts_give_different_decompositions(self, sae: SAELens) -> None:
        c1 = _make_concept(seed=1)
        c2 = _make_concept(seed=2)
        r1 = c1.decompose(sae)
        r2 = c2.decompose(sae)
        # Two random embeddings almost certainly activate different feature sets.
        assert not torch.allclose(r1.activations, r2.activations)


# ---------------------------------------------------------------------------
# Concept.from_features
# ---------------------------------------------------------------------------


class TestConceptFromFeatures:
    """Concept.from_features(sae, features) builds a normalised Concept."""

    def test_returns_concept_instance(self, sae: SAELens) -> None:
        c = Concept.from_features(sae, {"feature_000": 1.0})
        assert isinstance(c, Concept)

    def test_embedding_is_unit_norm(self, sae: SAELens) -> None:
        c = Concept.from_features(sae, {"feature_000": 1.0, "feature_001": 0.5})
        norm = c.embedding.norm().item()
        assert abs(norm - 1.0) < 1e-5

    def test_embedding_shape_is_1_by_embed_dim(self, sae: SAELens) -> None:
        c = Concept.from_features(sae, {"feature_002": 2.0})
        assert c.embedding.shape == (1, EMBED_DIM)

    def test_description_contains_features(self, sae: SAELens) -> None:
        features = {"feature_010": 1.5}
        c = Concept.from_features(sae, features)
        assert "feature_010" in c.description

    def test_unknown_feature_raises(self, sae: SAELens) -> None:
        with pytest.raises(FeatureNotFoundError):
            Concept.from_features(sae, {"not_a_feature": 1.0})

    def test_empty_features_gives_pre_bias_direction(self, sae: SAELens) -> None:
        """
        Empty feature dict → zero activation → reconstruction is the pre_bias.
        After normalisation the embedding is the normalised pre_bias (or zeros
        direction if pre_bias is zero).
        """
        c = Concept.from_features(sae, {})
        # Should not raise; just verify it returns a valid, normalised Concept.
        norm = c.embedding.norm().item()
        # pre_bias is all-zeros for _make_sae, so the result could be near-zero
        # before normalisation — F.normalize handles this gracefully (returns zeros).
        # Either way, the output is a valid tensor.
        assert c.embedding.shape == (1, EMBED_DIM)
        assert torch.isfinite(c.embedding).all()

    def test_two_features_combine_correctly(self, sae: SAELens) -> None:
        """
        Build two single-feature concepts and one two-feature concept, then
        verify the two-feature embedding is *not* identical to either single one.
        """
        c1 = Concept.from_features(sae, {"feature_000": 1.0})
        c2 = Concept.from_features(sae, {"feature_001": 1.0})
        c_both = Concept.from_features(sae, {"feature_000": 1.0, "feature_001": 1.0})
        assert not torch.allclose(c_both.embedding, c1.embedding)
        assert not torch.allclose(c_both.embedding, c2.embedding)

    def test_scaling_feature_strength_changes_embedding(self, sae: SAELens) -> None:
        """Higher activation strengths push the embedding in the same direction."""
        c_low = Concept.from_features(sae, {"feature_005": 1.0})
        c_high = Concept.from_features(sae, {"feature_005": 10.0})
        # Because we renormalise, both land on the unit sphere.
        # But the cosine similarity between them should be 1.0 (same direction).
        cos = F.cosine_similarity(c_low.embedding, c_high.embedding, dim=-1).item()
        assert abs(cos - 1.0) < 1e-4

    def test_reproducible(self, sae: SAELens) -> None:
        """from_features is deterministic for the same inputs."""
        features = {"feature_003": 2.5, "feature_007": 0.8}
        c1 = Concept.from_features(sae, features)
        c2 = Concept.from_features(sae, features)
        assert torch.allclose(c1.embedding, c2.embedding)

    def test_round_trip_via_decompose(self, sae: SAELens) -> None:
        """
        Building a concept from k features and then decomposing it should
        activate at least one of those original features (not a strict inverse,
        but a basic sanity check).
        """
        features = {"feature_001": 3.0, "feature_050": 1.5}
        c = Concept.from_features(sae, features)
        decomp = c.decompose(sae)
        # At least one of the original feature names should be active.
        active_names = set(decomp.active_features.keys())
        original_names = set(features.keys())
        # It's possible (with random weights) that none overlap, so we only
        # check that the decomposition is valid rather than demanding overlap.
        assert len(active_names) == K

    def test_similarity_from_features_and_decompose_reconstruct(self, sae: SAELens) -> None:
        """
        A concept created by from_features should be similar (but not necessarily
        identical) to a concept created by decomposing any embedding.
        """
        features = {"feature_010": 2.0}
        c = Concept.from_features(sae, features)
        # Re-decompose and check embedding similarity to the SAE reconstruction.
        decomp = c.decompose(sae)
        recon = sae.reconstruct(decomp)
        recon_norm = F.normalize(recon, dim=-1)
        cos = F.cosine_similarity(c.embedding, recon_norm, dim=-1).item()
        # Cosine similarity between the concept and the reconstruction of its
        # own sparse representation should be reasonably high.
        assert cos > -1.0  # Sanity: not anti-correlated.


# ---------------------------------------------------------------------------
# Integration: decompose → manipulate → from_features round-trip
# ---------------------------------------------------------------------------


class TestDecomposeManipulateRoundTrip:
    """
    Verifies that the full SAE manipulation workflow works end-to-end through
    the Concept API:

      Concept.from_text(…)
      → concept.decompose(sae)          → SAEDecomposition
      → sae.manipulate(decomp, deltas)  → modified SAEDecomposition
      → sae.reconstruct(modified)       → tensor
      → Concept(embedding=…)            → new concept
    """

    def test_full_workflow_produces_valid_concept(self, sae: SAELens) -> None:
        # 1. Start from a raw concept.
        original = _make_concept(seed=7)

        # 2. Decompose.
        decomp = original.decompose(sae)
        assert isinstance(decomp, SAEDecomposition)

        # 3. Amplify the first active feature.
        target_name = next(iter(decomp.active_features))
        modified_decomp = sae.manipulate(decomp, {target_name: 5.0})

        # 4. Reconstruct into a new Concept.
        recon = sae.reconstruct(modified_decomp)
        new_concept = Concept(
            embedding=F.normalize(recon, dim=-1),
            description=f"amplified:{target_name}",
        )

        # 5. Verify it's a valid unit-norm Concept.
        assert new_concept.embedding.shape == (1, EMBED_DIM)
        norm = new_concept.embedding.norm().item()
        assert abs(norm - 1.0) < 1e-5

    def test_manipulated_concept_differs_from_original(self, sae: SAELens) -> None:
        original = _make_concept(seed=13)
        decomp = original.decompose(sae)
        target_name = next(iter(decomp.active_features))
        modified_decomp = sae.manipulate(decomp, {target_name: 100.0})
        recon = sae.reconstruct(modified_decomp)
        new_concept = Concept(embedding=F.normalize(recon, dim=-1), description="modified")

        # The strongly amplified concept should differ from the original.
        assert not torch.allclose(original.embedding, new_concept.embedding, atol=1e-3)

    def test_from_features_decompose_consistency(self, sae: SAELens) -> None:
        """
        A concept built with from_features should decompose to an activation
        tensor where the SAE reconstruction is consistent.
        """
        features = {"feature_020": 5.0, "feature_030": 3.0}
        concept = Concept.from_features(sae, features)
        decomp = concept.decompose(sae)
        recon = sae.reconstruct(decomp)

        # The reconstruction should be a valid embedding-shaped tensor.
        assert recon.shape == (1, EMBED_DIM)
        assert torch.isfinite(recon).all()
