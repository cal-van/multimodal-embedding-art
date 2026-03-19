"""
Tests for CompositeLoss — composite loss function combining cosine similarity,
multi-layer feature matching, SAE feature-space loss, and regularization.

Follows TDD: all tests are written before the implementation exists.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
import torch.nn.functional as F

from embedding_art.core.concept import Concept
from embedding_art.core.config import LossConfig
from embedding_art.core.render_result import LossBreakdown
from embedding_art.encoders.features import LayerFeatures
from embedding_art.encoders.registry import EncoderCapability, EncoderCard

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_concept(dim: int = 1024, seed: int = 0) -> Concept:
    """Return a normalized concept with a deterministic embedding."""
    gen = torch.Generator().manual_seed(seed)
    emb = torch.randn(1, dim, generator=gen)
    return Concept(embedding=emb, description="test")


def _make_concept_with_source(dim: int = 1024, seed: int = 0) -> Concept:
    """Return a concept that carries a source_input tensor."""
    concept = _make_concept(dim, seed)
    concept.source_input = torch.randn(1, 3, 64, 64)
    return concept


def _encoder_with_multi_layer(mock_encoder) -> MagicMock:
    """
    Wrap mock_encoder in a MagicMock that declares MULTI_LAYER_FEATURES capability
    and supports get_layer_features / encode_for_optimization.
    """
    encoder = MagicMock()
    encoder.card = EncoderCard(
        name="mock-multilayer",
        capabilities=(
            EncoderCapability.TEXT
            | EncoderCapability.IMAGE
            | EncoderCapability.BACKPROP_OPTIMIZABLE
            | EncoderCapability.MULTI_LAYER_FEATURES
        ),
        embedding_dim=1024,
        memory_estimate_mb=200,
        backprop_cost=1.0,
    )

    def _encode_for_opt(tensor):
        gen = torch.Generator().manual_seed(1)
        emb = torch.randn(1, 1024, generator=gen)
        return F.normalize(emb, dim=-1)

    encoder.encode_for_optimization.side_effect = _encode_for_opt

    def _get_layer_features(tensor):
        # Two layers of ViT-style [1, 16, 64] activations
        return {
            0: LayerFeatures(
                tensor=torch.randn(1, 16, 64),
                spatial=False,
                shape_semantic="batch_tokens_dim",
                layer_name="layer_0",
            ),
            6: LayerFeatures(
                tensor=torch.randn(1, 16, 64),
                spatial=False,
                shape_semantic="batch_tokens_dim",
                layer_name="layer_6",
            ),
        }

    encoder.get_layer_features.side_effect = _get_layer_features
    return encoder


# ---------------------------------------------------------------------------
# Test 1: similarity-only loss — default config, feature_matching_weight=0
# ---------------------------------------------------------------------------


def test_similarity_only_loss(mock_encoder):
    """With feature_matching_weight=0 and no SAE, only 'similarity' is computed."""
    from embedding_art.core.loss import CompositeLoss

    config = LossConfig(similarity_weight=1.0, feature_matching_weight=0.0, sae_feature_weight=0.0)
    loss_fn = CompositeLoss(config, mock_encoder)

    target = _make_concept(seed=1)
    current_output = torch.randn(1, 3, 64, 64)

    breakdown = loss_fn(current_output, target, mock_encoder)

    assert "similarity" in breakdown.components
    assert "feature_matching" not in breakdown.components
    assert "sae_features" not in breakdown.components
    assert "regularization" not in breakdown.components


# ---------------------------------------------------------------------------
# Test 2: all component names when fully configured
# ---------------------------------------------------------------------------


def test_all_components_tracked(mock_encoder):
    """When all components are enabled and calibrated, all four keys appear."""
    from embedding_art.core.loss import CompositeLoss

    ml_encoder = _encoder_with_multi_layer(mock_encoder)

    # Build a mock SAE
    sae = MagicMock()
    decomp_mock = MagicMock()
    decomp_mock.activations = torch.zeros(1, 128)
    sae.decompose.return_value = decomp_mock

    config_with_sae = LossConfig(
        similarity_weight=1.0, feature_matching_weight=1.0, sae_feature_weight=1.0
    )
    loss_fn = CompositeLoss(config_with_sae, ml_encoder, sae=sae)

    target = _make_concept_with_source(seed=2)
    loss_fn.calibrate(target, ml_encoder)

    current_output = torch.randn(1, 3, 64, 64)
    breakdown = loss_fn(current_output, target, ml_encoder)

    assert "similarity" in breakdown.components
    assert "feature_matching" in breakdown.components
    assert "sae_features" in breakdown.components


# ---------------------------------------------------------------------------
# Test 3: calibrate skips without MULTI_LAYER_FEATURES capability
# ---------------------------------------------------------------------------


def test_calibrate_skips_without_capability(mock_encoder):
    """Encoder without MULTI_LAYER_FEATURES → reference_stats stays None."""
    from embedding_art.core.loss import CompositeLoss

    # mock_encoder has BACKPROP_OPTIMIZABLE but NOT MULTI_LAYER_FEATURES
    config = LossConfig(feature_matching_weight=1.0)
    loss_fn = CompositeLoss(config, mock_encoder)

    target = _make_concept_with_source(seed=3)
    loss_fn.calibrate(target, mock_encoder)

    assert loss_fn.reference_stats is None


# ---------------------------------------------------------------------------
# Test 4: calibrate skips when source_input is None
# ---------------------------------------------------------------------------


def test_calibrate_skips_without_source_input(mock_encoder):
    """target.source_input is None → reference_stats stays None."""
    from embedding_art.core.loss import CompositeLoss

    ml_encoder = _encoder_with_multi_layer(mock_encoder)
    config = LossConfig(feature_matching_weight=1.0)
    loss_fn = CompositeLoss(config, ml_encoder)

    target = _make_concept(seed=4)  # no source_input
    assert target.source_input is None

    loss_fn.calibrate(target, ml_encoder)

    assert loss_fn.reference_stats is None


# ---------------------------------------------------------------------------
# Test 5: feature_matching absent when uncalibrated (weight > 0 but no stats)
# ---------------------------------------------------------------------------


def test_feature_matching_disabled_when_uncalibrated(mock_encoder):
    """
    feature_matching_weight > 0 but calibrate() was never called
    (or skipped) → no 'feature_matching' key in components.
    """
    from embedding_art.core.loss import CompositeLoss

    ml_encoder = _encoder_with_multi_layer(mock_encoder)
    config = LossConfig(similarity_weight=1.0, feature_matching_weight=1.0)
    loss_fn = CompositeLoss(config, ml_encoder)
    # Deliberately do NOT call calibrate()

    target = _make_concept(seed=5)
    current_output = torch.randn(1, 3, 64, 64)

    breakdown = loss_fn(current_output, target, ml_encoder)

    assert "similarity" in breakdown.components
    assert "feature_matching" not in breakdown.components


# ---------------------------------------------------------------------------
# Test 6: sae_features absent when no SAE provided
# ---------------------------------------------------------------------------


def test_sae_disabled_when_no_sae(mock_encoder):
    """sae_feature_weight > 0 but sae=None → no 'sae_features' component."""
    from embedding_art.core.loss import CompositeLoss

    config = LossConfig(similarity_weight=1.0, sae_feature_weight=1.0)
    loss_fn = CompositeLoss(config, mock_encoder, sae=None)

    target = _make_concept(seed=6)
    current_output = torch.randn(1, 3, 64, 64)

    breakdown = loss_fn(current_output, target, mock_encoder)

    assert "similarity" in breakdown.components
    assert "sae_features" not in breakdown.components


# ---------------------------------------------------------------------------
# Test 7: similarity component equals -cosine_similarity * weight
# ---------------------------------------------------------------------------


def test_similarity_loss_is_negative_cosine(mock_encoder):
    """
    The 'similarity' component should equal
    -cosine_similarity(current_emb, target.embedding) * similarity_weight.
    """
    from embedding_art.core.loss import CompositeLoss

    similarity_weight = 2.0
    config = LossConfig(
        similarity_weight=similarity_weight,
        feature_matching_weight=0.0,
        sae_feature_weight=0.0,
    )
    loss_fn = CompositeLoss(config, mock_encoder)

    target = _make_concept(seed=7)
    current_output = torch.randn(1, 3, 64, 64)

    breakdown = loss_fn(current_output, target, mock_encoder)

    # Reproduce what the loss function does internally
    current_emb = mock_encoder.encode_for_optimization(current_output)
    expected_sim = F.cosine_similarity(current_emb, target.embedding, dim=-1).mean()
    expected_component = -expected_sim * similarity_weight

    torch.testing.assert_close(
        breakdown.components["similarity"], expected_component, rtol=1e-5, atol=1e-6
    )


# ---------------------------------------------------------------------------
# Test 8: __call__ always returns LossBreakdown
# ---------------------------------------------------------------------------


def test_loss_returns_loss_breakdown_type(mock_encoder):
    """CompositeLoss.__call__ must return a LossBreakdown instance."""
    from embedding_art.core.loss import CompositeLoss

    config = LossConfig()
    loss_fn = CompositeLoss(config, mock_encoder)

    target = _make_concept(seed=8)
    current_output = torch.randn(1, 3, 64, 64)

    result = loss_fn(current_output, target, mock_encoder)

    assert isinstance(result, LossBreakdown)
    assert isinstance(result.total, torch.Tensor)
    assert isinstance(result.components, dict)


# ---------------------------------------------------------------------------
# Test 9: total equals sum of all active components
# ---------------------------------------------------------------------------


def test_total_is_sum_of_components(mock_encoder):
    """LossBreakdown.total must equal the sum of all component tensors."""
    from embedding_art.core.loss import CompositeLoss

    ml_encoder = _encoder_with_multi_layer(mock_encoder)
    config = LossConfig(similarity_weight=1.0, feature_matching_weight=1.0, sae_feature_weight=0.0)
    loss_fn = CompositeLoss(config, ml_encoder)

    # Calibrate so feature_matching fires
    target = _make_concept_with_source(seed=9)
    loss_fn.calibrate(target, ml_encoder)

    current_output = torch.randn(1, 3, 64, 64)
    breakdown = loss_fn(current_output, target, ml_encoder)

    expected_total = sum(breakdown.components.values())
    torch.testing.assert_close(breakdown.total, expected_total, rtol=1e-5, atol=1e-6)
