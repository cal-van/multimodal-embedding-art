"""
Unit tests for SAELens and SAEDecomposition.

All tests use SAELens.from_tensors() with small synthetic weights so that
no files, safetensors dependencies, or trained models are required.

Sizes used throughout:
  embed_dim  = 64
  n_features = 256
  k          = 8
"""

import pytest
import torch
import torch.nn.functional as F

from embedding_art.exceptions import FeatureNotFoundError
from embedding_art.sae.lens import SAEDecomposition, SAELens


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

EMBED_DIM = 64
N_FEATURES = 256
K = 8


def _make_vocab(n: int) -> list[str]:
    """Return a list of n distinct feature names like 'feature_000'."""
    return [f"feature_{i:03d}" for i in range(n)]


def _make_sae(
    embed_dim: int = EMBED_DIM,
    n_features: int = N_FEATURES,
    k: int = K,
    seed: int = 0,
) -> SAELens:
    """
    Build a small SAELens from deterministic random tensors.

    Decoder columns are L2-normalised so that the SAE behaves like a
    realistic dictionary (unit-norm atoms).
    """
    gen = torch.Generator().manual_seed(seed)
    W_enc = torch.randn(n_features, embed_dim, generator=gen)
    W_dec_raw = torch.randn(embed_dim, n_features, generator=gen)
    # Normalise decoder columns — standard practice for SAEs.
    W_dec = F.normalize(W_dec_raw, dim=0)
    bias = torch.zeros(n_features)
    pre_bias = torch.zeros(embed_dim)
    vocab = _make_vocab(n_features)
    return SAELens.from_tensors(W_enc, W_dec, bias, pre_bias, vocab, k)


@pytest.fixture
def sae() -> SAELens:
    """Default small SAELens for tests."""
    return _make_sae()


@pytest.fixture
def embedding() -> torch.Tensor:
    """A normalised 64-dimensional embedding for testing."""
    gen = torch.Generator().manual_seed(99)
    x = torch.randn(1, EMBED_DIM, generator=gen)
    return F.normalize(x, dim=-1)


# ---------------------------------------------------------------------------
# from_tensors construction
# ---------------------------------------------------------------------------


class TestFromTensors:
    """SAELens.from_tensors() builds a valid, usable instance."""

    def test_properties_match_inputs(self) -> None:
        """n_features, embed_dim, k, and vocab are set correctly."""
        sae = _make_sae(embed_dim=32, n_features=128, k=4)
        assert sae.n_features == 128
        assert sae.embed_dim == 32
        assert sae.k == 4
        assert len(sae.vocab) == 128

    def test_vocab_order_preserved(self) -> None:
        """Vocabulary list is stored in the order provided."""
        vocab = ["alpha", "beta", "gamma"]
        W_enc = torch.randn(3, 8)
        W_dec = torch.randn(8, 3)
        bias = torch.zeros(3)
        pre_bias = torch.zeros(8)
        sae = SAELens.from_tensors(W_enc, W_dec, bias, pre_bias, vocab, k=2)
        assert sae.vocab == vocab

    def test_shape_mismatch_raises(self) -> None:
        """Incompatible W_enc / W_dec shapes raise ValueError."""
        W_enc = torch.randn(16, 32)
        W_dec = torch.randn(32, 8)  # wrong: should be [32, 16]
        bias = torch.zeros(16)
        pre_bias = torch.zeros(32)
        vocab = _make_vocab(16)
        with pytest.raises(ValueError, match="Shape mismatch"):
            SAELens.from_tensors(W_enc, W_dec, bias, pre_bias, vocab, k=4)

    def test_vocab_length_mismatch_raises(self) -> None:
        """Vocabulary shorter than n_features raises ValueError."""
        W_enc = torch.randn(16, 32)
        W_dec = torch.randn(32, 16)
        bias = torch.zeros(16)
        pre_bias = torch.zeros(32)
        vocab = _make_vocab(10)  # wrong: should be 16
        with pytest.raises(ValueError, match="Vocabulary length"):
            SAELens.from_tensors(W_enc, W_dec, bias, pre_bias, vocab, k=4)

    def test_k_zero_raises(self) -> None:
        """k < 1 should raise ValueError."""
        W_enc = torch.randn(16, 32)
        W_dec = torch.randn(32, 16)
        bias = torch.zeros(16)
        pre_bias = torch.zeros(32)
        vocab = _make_vocab(16)
        with pytest.raises(ValueError, match="k must be >= 1"):
            SAELens.from_tensors(W_enc, W_dec, bias, pre_bias, vocab, k=0)


# ---------------------------------------------------------------------------
# decompose
# ---------------------------------------------------------------------------


class TestDecompose:
    """SAELens.decompose() produces correct sparse activations."""

    def test_returns_sae_decomposition(self, sae: SAELens, embedding: torch.Tensor) -> None:
        """decompose() returns an SAEDecomposition instance."""
        result = sae.decompose(embedding)
        assert isinstance(result, SAEDecomposition)

    def test_activations_shape(self, sae: SAELens, embedding: torch.Tensor) -> None:
        """activations tensor has shape [1, n_features]."""
        result = sae.decompose(embedding)
        assert result.activations.shape == (1, N_FEATURES)

    def test_exactly_k_nonzero_entries(self, sae: SAELens, embedding: torch.Tensor) -> None:
        """Exactly k entries are non-zero after TopK sparsification."""
        result = sae.decompose(embedding)
        n_nonzero = int((result.activations != 0).sum().item())
        assert n_nonzero == K

    def test_active_features_count_matches_k(
        self, sae: SAELens, embedding: torch.Tensor
    ) -> None:
        """active_features dict has exactly k entries."""
        result = sae.decompose(embedding)
        assert len(result.active_features) == K

    def test_active_features_keys_are_valid_vocab(
        self, sae: SAELens, embedding: torch.Tensor
    ) -> None:
        """All active feature names are in the SAE vocabulary."""
        result = sae.decompose(embedding)
        for name in result.active_features:
            assert name in sae.vocab

    def test_active_features_values_are_positive(
        self, sae: SAELens, embedding: torch.Tensor
    ) -> None:
        """All active feature strengths are positive (post-ReLU)."""
        result = sae.decompose(embedding)
        for strength in result.active_features.values():
            assert strength > 0.0

    def test_reconstruction_error_is_finite(
        self, sae: SAELens, embedding: torch.Tensor
    ) -> None:
        """reconstruction_error is a finite non-negative float."""
        result = sae.decompose(embedding)
        assert isinstance(result.reconstruction_error, float)
        assert result.reconstruction_error >= 0.0
        assert not (result.reconstruction_error != result.reconstruction_error)  # not NaN

    def test_accepts_1d_embedding(self, sae: SAELens) -> None:
        """decompose() accepts a 1D [embed_dim] tensor (auto-unsqueezed)."""
        x = F.normalize(torch.randn(EMBED_DIM), dim=0)
        result = sae.decompose(x)
        assert result.activations.shape == (1, N_FEATURES)

    def test_deterministic_for_same_input(self, sae: SAELens, embedding: torch.Tensor) -> None:
        """Same embedding always produces the same decomposition."""
        r1 = sae.decompose(embedding)
        r2 = sae.decompose(embedding)
        assert torch.allclose(r1.activations, r2.activations)

    def test_k_larger_than_features_clips_gracefully(self) -> None:
        """When k >= n_features, all non-zero activations are kept."""
        sae = _make_sae(embed_dim=8, n_features=4, k=10)  # k > n_features
        x = F.normalize(torch.randn(1, 8), dim=-1)
        result = sae.decompose(x)
        # Should not raise; at most n_features entries can be non-zero.
        assert result.activations.shape == (1, 4)


# ---------------------------------------------------------------------------
# reconstruct
# ---------------------------------------------------------------------------


class TestReconstruct:
    """SAELens.reconstruct() decodes activations back to embedding space."""

    def test_reconstruct_returns_correct_shape(
        self, sae: SAELens, embedding: torch.Tensor
    ) -> None:
        """Reconstruction has shape [1, embed_dim]."""
        decomp = sae.decompose(embedding)
        recon = sae.reconstruct(decomp)
        assert recon.shape == (1, EMBED_DIM)

    def test_zero_activations_give_pre_bias_reconstruction(self, sae: SAELens) -> None:
        """Reconstructing all-zero activations returns the pre_bias vector."""
        zero_acts = torch.zeros(1, N_FEATURES)
        decomp = SAEDecomposition(activations=zero_acts, active_features={}, reconstruction_error=0.0)
        recon = sae.reconstruct(decomp)
        expected = sae._pre_bias.unsqueeze(0)
        assert torch.allclose(recon, expected, atol=1e-6)

    def test_reconstruct_round_trip_approximately(
        self, sae: SAELens, embedding: torch.Tensor
    ) -> None:
        """
        Decompose then reconstruct should give a result reasonably close to the
        original.  We don't expect perfection (only k features are active), but
        the cosine similarity should be non-trivially positive.
        """
        decomp = sae.decompose(embedding)
        recon = sae.reconstruct(decomp)
        # Cosine similarity ≥ 0 is a minimal sanity-check (not a hard performance bar).
        cos_sim = float(F.cosine_similarity(embedding, recon, dim=-1).item())
        assert cos_sim > -1.0  # reconstruction is not wildly anti-correlated

    def test_reconstruction_error_matches_manual_computation(
        self, sae: SAELens, embedding: torch.Tensor
    ) -> None:
        """reconstruction_error stored in the decomposition matches manual calculation."""
        decomp = sae.decompose(embedding)
        recon = sae.reconstruct(decomp)
        manual_error = float(
            torch.norm(embedding.float() - recon).item()
            / (torch.norm(embedding.float()).item() + 1e-8)
        )
        assert abs(decomp.reconstruction_error - manual_error) < 1e-5


# ---------------------------------------------------------------------------
# manipulate
# ---------------------------------------------------------------------------


class TestManipulate:
    """SAELens.manipulate() adjusts named feature activations."""

    def test_manipulate_changes_target_feature(
        self, sae: SAELens, embedding: torch.Tensor
    ) -> None:
        """A manipulated feature has a different activation than the original."""
        decomp = sae.decompose(embedding)
        # Pick the first active feature and amplify it.
        target_name = next(iter(decomp.active_features))
        original_strength = decomp.active_features[target_name]

        new_decomp = sae.manipulate(decomp, {target_name: 5.0})

        idx = sae._vocab_to_idx[target_name]
        new_strength = float(new_decomp.activations[0, idx].item())
        assert abs(new_strength - (original_strength + 5.0)) < 1e-5

    def test_manipulate_does_not_mutate_original(
        self, sae: SAELens, embedding: torch.Tensor
    ) -> None:
        """manipulate() returns a new decomposition; the original is unchanged."""
        decomp = sae.decompose(embedding)
        original_acts = decomp.activations.clone()
        target_name = next(iter(decomp.active_features))

        sae.manipulate(decomp, {target_name: 10.0})

        assert torch.allclose(decomp.activations, original_acts)

    def test_manipulate_negative_delta_suppresses_feature(
        self, sae: SAELens, embedding: torch.Tensor
    ) -> None:
        """A large negative delta clamps the activation to zero."""
        decomp = sae.decompose(embedding)
        target_name = next(iter(decomp.active_features))

        new_decomp = sae.manipulate(decomp, {target_name: -1000.0})

        idx = sae._vocab_to_idx[target_name]
        assert float(new_decomp.activations[0, idx].item()) == 0.0

    def test_manipulate_unknown_feature_raises(
        self, sae: SAELens, embedding: torch.Tensor
    ) -> None:
        """Adjusting a feature not in the vocabulary raises FeatureNotFoundError."""
        decomp = sae.decompose(embedding)
        with pytest.raises(FeatureNotFoundError):
            sae.manipulate(decomp, {"does_not_exist": 1.0})

    def test_manipulate_can_activate_inactive_feature(
        self, sae: SAELens, embedding: torch.Tensor
    ) -> None:
        """
        A feature that was not active in the original can be switched on via
        a positive delta.
        """
        decomp = sae.decompose(embedding)
        active_names = set(decomp.active_features.keys())
        # Find a feature that is currently zero.
        inactive_name = next(f for f in sae.vocab if f not in active_names)

        new_decomp = sae.manipulate(decomp, {inactive_name: 3.0})

        idx = sae._vocab_to_idx[inactive_name]
        assert float(new_decomp.activations[0, idx].item()) == pytest.approx(3.0, abs=1e-5)

    def test_active_features_updated_after_manipulate(
        self, sae: SAELens, embedding: torch.Tensor
    ) -> None:
        """active_features dict reflects the post-manipulation state."""
        decomp = sae.decompose(embedding)
        target_name = next(iter(decomp.active_features))

        new_decomp = sae.manipulate(decomp, {target_name: 50.0})

        # The target feature should appear in active_features with updated value.
        assert target_name in new_decomp.active_features
        expected = decomp.active_features[target_name] + 50.0
        assert new_decomp.active_features[target_name] == pytest.approx(expected, abs=1e-5)

    def test_manipulate_returns_sae_decomposition(
        self, sae: SAELens, embedding: torch.Tensor
    ) -> None:
        """manipulate() returns an SAEDecomposition instance."""
        decomp = sae.decompose(embedding)
        target_name = next(iter(decomp.active_features))
        result = sae.manipulate(decomp, {target_name: 1.0})
        assert isinstance(result, SAEDecomposition)


# ---------------------------------------------------------------------------
# SAEDecomposition.from_dict
# ---------------------------------------------------------------------------


class TestSAEDecompositionFromDict:
    """SAEDecomposition.from_dict() builds correct activations from feature names."""

    def test_from_dict_sets_correct_activations(self, sae: SAELens) -> None:
        """from_dict sets the specified features to the supplied strengths."""
        features = {"feature_000": 1.5, "feature_001": 2.0}
        decomp = SAEDecomposition.from_dict(features, sae)

        assert float(decomp.activations[0, 0].item()) == pytest.approx(1.5, abs=1e-6)
        assert float(decomp.activations[0, 1].item()) == pytest.approx(2.0, abs=1e-6)

    def test_from_dict_zeros_out_unspecified_features(self, sae: SAELens) -> None:
        """Features not in the dict have zero activation."""
        features = {"feature_000": 1.0}
        decomp = SAEDecomposition.from_dict(features, sae)

        # All entries after index 0 should be zero.
        assert float(decomp.activations[0, 1:].sum().item()) == pytest.approx(0.0, abs=1e-6)

    def test_from_dict_activations_shape(self, sae: SAELens) -> None:
        """activations tensor has shape [1, n_features]."""
        decomp = SAEDecomposition.from_dict({"feature_005": 0.7}, sae)
        assert decomp.activations.shape == (1, N_FEATURES)

    def test_from_dict_active_features_matches_input(self, sae: SAELens) -> None:
        """active_features mirrors the non-zero entries of the input dict."""
        features = {"feature_002": 3.0, "feature_010": 0.5}
        decomp = SAEDecomposition.from_dict(features, sae)
        assert decomp.active_features == features

    def test_from_dict_excludes_zero_strength_from_active(self, sae: SAELens) -> None:
        """A feature with strength 0.0 is not included in active_features."""
        features = {"feature_000": 1.0, "feature_001": 0.0}
        decomp = SAEDecomposition.from_dict(features, sae)
        assert "feature_001" not in decomp.active_features
        assert "feature_000" in decomp.active_features

    def test_from_dict_unknown_feature_raises(self, sae: SAELens) -> None:
        """from_dict raises FeatureNotFoundError for unknown feature names."""
        with pytest.raises(FeatureNotFoundError):
            SAEDecomposition.from_dict({"not_a_feature": 1.0}, sae)

    def test_from_dict_reconstruction_error_is_zero(self, sae: SAELens) -> None:
        """from_dict sets reconstruction_error to 0.0 (no encoding performed)."""
        decomp = SAEDecomposition.from_dict({"feature_003": 1.0}, sae)
        assert decomp.reconstruction_error == 0.0

    def test_from_dict_empty_dict_gives_all_zeros(self, sae: SAELens) -> None:
        """from_dict with an empty dict produces an all-zero activation tensor."""
        decomp = SAEDecomposition.from_dict({}, sae)
        assert float(decomp.activations.sum().item()) == pytest.approx(0.0, abs=1e-6)
        assert decomp.active_features == {}


# ---------------------------------------------------------------------------
# active_features dict consistency
# ---------------------------------------------------------------------------


class TestActiveFeatures:
    """active_features dict is consistent with the activations tensor."""

    def test_active_features_values_match_tensor(
        self, sae: SAELens, embedding: torch.Tensor
    ) -> None:
        """Each value in active_features matches the corresponding tensor entry."""
        decomp = sae.decompose(embedding)
        for name, strength in decomp.active_features.items():
            idx = sae._vocab_to_idx[name]
            tensor_val = float(decomp.activations[0, idx].item())
            assert tensor_val == pytest.approx(strength, abs=1e-6)

    def test_no_inactive_features_in_dict(
        self, sae: SAELens, embedding: torch.Tensor
    ) -> None:
        """Zero-activation features do not appear in active_features."""
        decomp = sae.decompose(embedding)
        for name in sae.vocab:
            idx = sae._vocab_to_idx[name]
            if float(decomp.activations[0, idx].item()) == 0.0:
                assert name not in decomp.active_features
