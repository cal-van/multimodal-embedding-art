"""
Unit tests for LayerFeatures, FeatureStatistics, and ViTStatisticsExtractor.

Tests verify:
- LayerFeatures creation and field access
- FeatureStatistics creation and field access
- ViTStatisticsExtractor output shapes for ViT-style [B, N_patches, D] tensors
- ViTStatisticsExtractor determinism
- Edge cases: single patch, 2D fallback path
"""

import pytest
import torch

from embedding_art.encoders.features import (
    CNNStatisticsExtractor,
    FeatureStatistics,
    FeatureStatisticsExtractor,
    LayerFeatures,
    ViTStatisticsExtractor,
)

# =============================================================================
# LayerFeatures Tests
# =============================================================================


class TestLayerFeatures:
    """Tests for LayerFeatures typed container."""

    def test_stores_tensor_field(self) -> None:
        """LayerFeatures should store and return the provided tensor."""
        tensor = torch.randn(1, 197, 768)

        features = LayerFeatures(
            tensor=tensor,
            spatial=False,
            shape_semantic="batch_tokens_dim",
            layer_name="blocks.11",
        )

        assert features.tensor is tensor

    def test_stores_spatial_flag_false(self) -> None:
        """LayerFeatures should store spatial=False for token-based features."""
        tensor = torch.randn(1, 197, 768)

        features = LayerFeatures(
            tensor=tensor,
            spatial=False,
            shape_semantic="batch_tokens_dim",
            layer_name="blocks.11",
        )

        assert features.spatial is False

    def test_stores_spatial_flag_true(self) -> None:
        """LayerFeatures should store spatial=True for spatial feature maps."""
        tensor = torch.randn(1, 512, 14, 14)

        features = LayerFeatures(
            tensor=tensor,
            spatial=True,
            shape_semantic="batch_channels_height_width",
            layer_name="layer4",
        )

        assert features.spatial is True

    def test_stores_shape_semantic_token(self) -> None:
        """LayerFeatures should store the token shape_semantic string."""
        tensor = torch.randn(1, 197, 768)

        features = LayerFeatures(
            tensor=tensor,
            spatial=False,
            shape_semantic="batch_tokens_dim",
            layer_name="blocks.11",
        )

        assert features.shape_semantic == "batch_tokens_dim"

    def test_stores_shape_semantic_spatial(self) -> None:
        """LayerFeatures should store the spatial shape_semantic string."""
        tensor = torch.randn(1, 512, 14, 14)

        features = LayerFeatures(
            tensor=tensor,
            spatial=True,
            shape_semantic="batch_channels_height_width",
            layer_name="layer4",
        )

        assert features.shape_semantic == "batch_channels_height_width"

    def test_stores_layer_name(self) -> None:
        """LayerFeatures should store the layer_name string."""
        tensor = torch.randn(1, 197, 768)

        features = LayerFeatures(
            tensor=tensor,
            spatial=False,
            shape_semantic="batch_tokens_dim",
            layer_name="blocks.11",
        )

        assert features.layer_name == "blocks.11"

    def test_accepts_arbitrary_layer_names(self) -> None:
        """LayerFeatures should accept any string as layer_name."""
        tensor = torch.randn(1, 197, 768)

        features = LayerFeatures(
            tensor=tensor,
            spatial=False,
            shape_semantic="batch_tokens_dim",
            layer_name="encoder.layer.23.output",
        )

        assert features.layer_name == "encoder.layer.23.output"


# =============================================================================
# FeatureStatistics Tests
# =============================================================================


class TestFeatureStatistics:
    """Tests for FeatureStatistics typed container."""

    def test_stores_mean_tensor(self) -> None:
        """FeatureStatistics should store and return the mean tensor."""
        mean = torch.zeros(768)
        std = torch.ones(768)

        stats = FeatureStatistics(mean=mean, std=std)

        assert stats.mean is mean

    def test_stores_std_tensor(self) -> None:
        """FeatureStatistics should store and return the std tensor."""
        mean = torch.zeros(768)
        std = torch.ones(768)

        stats = FeatureStatistics(mean=mean, std=std)

        assert stats.std is std


# =============================================================================
# ViTStatisticsExtractor Tests — standard 3D input [B, N_patches, D]
# =============================================================================


class TestViTStatisticsExtractorStandard:
    """Tests for ViTStatisticsExtractor with standard [1, 197, 768] ViT input."""

    @pytest.fixture
    def vit_features(self) -> LayerFeatures:
        """Standard ViT features: [batch=1, n_patches=197, dim=768]."""
        torch.manual_seed(0)
        tensor = torch.randn(1, 197, 768)
        return LayerFeatures(
            tensor=tensor,
            spatial=False,
            shape_semantic="batch_tokens_dim",
            layer_name="blocks.11",
        )

    def test_extract_returns_feature_statistics(self, vit_features: LayerFeatures) -> None:
        """extract() should return a FeatureStatistics instance."""
        extractor = ViTStatisticsExtractor()

        result = extractor.extract(vit_features)

        assert isinstance(result, FeatureStatistics)

    def test_mean_has_correct_shape(self, vit_features: LayerFeatures) -> None:
        """mean should have shape [D] — one value per feature dimension."""
        extractor = ViTStatisticsExtractor()

        result = extractor.extract(vit_features)

        # [B=1, N=197, D=768] → mean over N → squeeze B → [768]
        assert result.mean.shape == (768,)

    def test_std_has_correct_shape(self, vit_features: LayerFeatures) -> None:
        """std should have shape [D] — one value per feature dimension."""
        extractor = ViTStatisticsExtractor()

        result = extractor.extract(vit_features)

        assert result.std.shape == (768,)

    def test_mean_values_are_finite(self, vit_features: LayerFeatures) -> None:
        """mean values should all be finite (no NaN or Inf)."""
        extractor = ViTStatisticsExtractor()

        result = extractor.extract(vit_features)

        assert torch.isfinite(result.mean).all()

    def test_std_values_are_finite(self, vit_features: LayerFeatures) -> None:
        """std values should all be finite (no NaN or Inf)."""
        extractor = ViTStatisticsExtractor()

        result = extractor.extract(vit_features)

        assert torch.isfinite(result.std).all()

    def test_std_values_are_non_negative(self, vit_features: LayerFeatures) -> None:
        """std values should all be non-negative."""
        extractor = ViTStatisticsExtractor()

        result = extractor.extract(vit_features)

        assert (result.std >= 0).all()

    def test_mean_computed_over_patches_dimension(self, vit_features: LayerFeatures) -> None:
        """mean should equal torch.mean across the patches (dim=1) axis."""
        extractor = ViTStatisticsExtractor()

        result = extractor.extract(vit_features)

        expected_mean = vit_features.tensor.mean(dim=1).squeeze(0)
        assert torch.allclose(result.mean, expected_mean)

    def test_std_computed_over_patches_dimension(self, vit_features: LayerFeatures) -> None:
        """std should equal torch.std across the patches (dim=1) axis."""
        extractor = ViTStatisticsExtractor()

        result = extractor.extract(vit_features)

        expected_std = vit_features.tensor.std(dim=1).squeeze(0)
        assert torch.allclose(result.std, expected_std)


# =============================================================================
# ViTStatisticsExtractor Tests — determinism
# =============================================================================


class TestViTStatisticsExtractorDeterminism:
    """Tests that ViTStatisticsExtractor produces deterministic results."""

    def test_same_input_produces_identical_mean(self) -> None:
        """Two calls with the same tensor must return identical mean."""
        torch.manual_seed(42)
        tensor = torch.randn(1, 197, 768)
        features = LayerFeatures(
            tensor=tensor,
            spatial=False,
            shape_semantic="batch_tokens_dim",
            layer_name="blocks.11",
        )
        extractor = ViTStatisticsExtractor()

        result_a = extractor.extract(features)
        result_b = extractor.extract(features)

        assert torch.equal(result_a.mean, result_b.mean)

    def test_same_input_produces_identical_std(self) -> None:
        """Two calls with the same tensor must return identical std."""
        torch.manual_seed(42)
        tensor = torch.randn(1, 197, 768)
        features = LayerFeatures(
            tensor=tensor,
            spatial=False,
            shape_semantic="batch_tokens_dim",
            layer_name="blocks.11",
        )
        extractor = ViTStatisticsExtractor()

        result_a = extractor.extract(features)
        result_b = extractor.extract(features)

        assert torch.equal(result_a.std, result_b.std)

    def test_different_inputs_produce_different_mean(self) -> None:
        """Different input tensors should produce different mean values."""
        torch.manual_seed(1)
        tensor_a = torch.randn(1, 197, 768)
        torch.manual_seed(2)
        tensor_b = torch.randn(1, 197, 768)

        extractor = ViTStatisticsExtractor()
        features_a = LayerFeatures(
            tensor=tensor_a, spatial=False, shape_semantic="batch_tokens_dim", layer_name="x"
        )
        features_b = LayerFeatures(
            tensor=tensor_b, spatial=False, shape_semantic="batch_tokens_dim", layer_name="x"
        )

        result_a = extractor.extract(features_a)
        result_b = extractor.extract(features_b)

        assert not torch.equal(result_a.mean, result_b.mean)


# =============================================================================
# ViTStatisticsExtractor Tests — edge cases
# =============================================================================


class TestViTStatisticsExtractorEdgeCases:
    """Edge case tests for ViTStatisticsExtractor."""

    def test_single_patch_token_shape(self) -> None:
        """Extractor should handle [1, 1, D] (single patch) without error."""
        torch.manual_seed(7)
        tensor = torch.randn(1, 1, 768)
        features = LayerFeatures(
            tensor=tensor,
            spatial=False,
            shape_semantic="batch_tokens_dim",
            layer_name="blocks.0",
        )
        extractor = ViTStatisticsExtractor()

        result = extractor.extract(features)

        # single patch: std will be NaN (std of one element), mean is well-defined
        assert result.mean.shape == (768,)
        assert result.std.shape == (768,)
        assert torch.isfinite(result.mean).all()

    def test_2d_input_fallback_returns_correct_shape(self) -> None:
        """A 2D input [B, D] should use the flatten fallback and return shape [D]."""
        torch.manual_seed(3)
        # 2D: [1, 768] — not a typical ViT output, exercises the else branch
        tensor = torch.randn(1, 768)
        features = LayerFeatures(
            tensor=tensor,
            spatial=False,
            shape_semantic="batch_tokens_dim",
            layer_name="head",
        )
        extractor = ViTStatisticsExtractor()

        result = extractor.extract(features)

        # flatten(start_dim=1) on [1, 768] → [1, 768]; mean(dim=1) → [1]; squeeze → scalar []
        assert result.mean.shape == torch.Size([])
        assert result.std.shape == torch.Size([])

    def test_large_vit_large_patch_count(self) -> None:
        """Extractor should handle a larger ViT-Large style tensor [1, 256, 1024]."""
        torch.manual_seed(5)
        tensor = torch.randn(1, 256, 1024)
        features = LayerFeatures(
            tensor=tensor,
            spatial=False,
            shape_semantic="batch_tokens_dim",
            layer_name="blocks.23",
        )
        extractor = ViTStatisticsExtractor()

        result = extractor.extract(features)

        assert result.mean.shape == (1024,)
        assert result.std.shape == (1024,)

    def test_4d_spatial_input_fallback(self) -> None:
        """A 4D spatial input [1, C, H, W] should use the flatten fallback."""
        torch.manual_seed(9)
        tensor = torch.randn(1, 512, 7, 7)
        features = LayerFeatures(
            tensor=tensor,
            spatial=True,
            shape_semantic="batch_channels_height_width",
            layer_name="layer4",
        )
        extractor = ViTStatisticsExtractor()

        result = extractor.extract(features)

        # flatten(start_dim=1): [1, 512*7*7=25088]; mean(dim=1) → [1]; squeeze → scalar []
        assert result.mean.shape == torch.Size([])
        assert result.std.shape == torch.Size([])


# =============================================================================
# Protocol Conformance
# =============================================================================


class TestFeatureStatisticsExtractorProtocol:
    """Verify ViTStatisticsExtractor satisfies the FeatureStatisticsExtractor protocol."""

    def test_vit_extractor_is_instance_of_protocol(self) -> None:
        """ViTStatisticsExtractor should satisfy the FeatureStatisticsExtractor protocol."""
        extractor = ViTStatisticsExtractor()
        tensor = torch.randn(1, 197, 768)
        features = LayerFeatures(
            tensor=tensor,
            spatial=False,
            shape_semantic="batch_tokens_dim",
            layer_name="blocks.11",
        )

        # runtime_checkable Protocol supports isinstance — verifies structural conformance
        assert isinstance(extractor, FeatureStatisticsExtractor)

        result = extractor.extract(features)
        assert isinstance(result, FeatureStatistics)


# =============================================================================
# CNNStatisticsExtractor Tests — standard 4D spatial input [B, C, H, W]
# =============================================================================


class TestCNNStatisticsExtractorStandard:
    """Tests for CNNStatisticsExtractor with standard [1, 512, 14, 14] CNN input."""

    @pytest.fixture
    def cnn_features(self) -> LayerFeatures:
        """Standard CNN features: [batch=1, channels=512, height=14, width=14]."""
        torch.manual_seed(0)
        tensor = torch.randn(1, 512, 14, 14)
        return LayerFeatures(
            tensor=tensor,
            spatial=True,
            shape_semantic="batch_channels_height_width",
            layer_name="layer4",
        )

    def test_extract_returns_feature_statistics(self, cnn_features: LayerFeatures) -> None:
        """extract() should return a FeatureStatistics instance."""
        extractor = CNNStatisticsExtractor()

        result = extractor.extract(cnn_features)

        assert isinstance(result, FeatureStatistics)

    def test_mean_has_correct_shape(self, cnn_features: LayerFeatures) -> None:
        """mean should have shape [C] — one value per channel."""
        extractor = CNNStatisticsExtractor()

        result = extractor.extract(cnn_features)

        # [B=1, C=512, H=14, W=14] -> mean over (H, W) -> squeeze B -> [512]
        assert result.mean.shape == (512,)

    def test_std_has_correct_shape(self, cnn_features: LayerFeatures) -> None:
        """std should have shape [C] — one value per channel."""
        extractor = CNNStatisticsExtractor()

        result = extractor.extract(cnn_features)

        assert result.std.shape == (512,)

    def test_mean_values_are_finite(self, cnn_features: LayerFeatures) -> None:
        """mean values should all be finite (no NaN or Inf)."""
        extractor = CNNStatisticsExtractor()

        result = extractor.extract(cnn_features)

        assert torch.isfinite(result.mean).all()

    def test_std_values_are_finite(self, cnn_features: LayerFeatures) -> None:
        """std values should all be finite (no NaN or Inf)."""
        extractor = CNNStatisticsExtractor()

        result = extractor.extract(cnn_features)

        assert torch.isfinite(result.std).all()

    def test_std_values_are_non_negative(self, cnn_features: LayerFeatures) -> None:
        """std values should all be non-negative."""
        extractor = CNNStatisticsExtractor()

        result = extractor.extract(cnn_features)

        assert (result.std >= 0).all()

    def test_mean_computed_over_spatial_dimensions(self, cnn_features: LayerFeatures) -> None:
        """mean should equal torch.mean over (H, W) dimensions."""
        extractor = CNNStatisticsExtractor()

        result = extractor.extract(cnn_features)

        expected_mean = cnn_features.tensor.mean(dim=(2, 3)).squeeze(0)
        assert torch.allclose(result.mean, expected_mean)

    def test_std_computed_over_spatial_dimensions(self, cnn_features: LayerFeatures) -> None:
        """std should equal torch.std over (H, W) dimensions."""
        extractor = CNNStatisticsExtractor()

        result = extractor.extract(cnn_features)

        expected_std = cnn_features.tensor.std(dim=(2, 3)).squeeze(0)
        assert torch.allclose(result.std, expected_std)


# =============================================================================
# CNNStatisticsExtractor Tests — determinism
# =============================================================================


class TestCNNStatisticsExtractorDeterminism:
    """CNNStatisticsExtractor must produce deterministic results."""

    def test_same_input_produces_identical_mean(self) -> None:
        torch.manual_seed(42)
        tensor = torch.randn(1, 256, 7, 7)
        features = LayerFeatures(
            tensor=tensor,
            spatial=True,
            shape_semantic="batch_channels_height_width",
            layer_name="layer3",
        )
        extractor = CNNStatisticsExtractor()

        result_a = extractor.extract(features)
        result_b = extractor.extract(features)

        assert torch.equal(result_a.mean, result_b.mean)

    def test_same_input_produces_identical_std(self) -> None:
        torch.manual_seed(42)
        tensor = torch.randn(1, 256, 7, 7)
        features = LayerFeatures(
            tensor=tensor,
            spatial=True,
            shape_semantic="batch_channels_height_width",
            layer_name="layer3",
        )
        extractor = CNNStatisticsExtractor()

        result_a = extractor.extract(features)
        result_b = extractor.extract(features)

        assert torch.equal(result_a.std, result_b.std)

    def test_different_inputs_produce_different_mean(self) -> None:
        torch.manual_seed(1)
        tensor_a = torch.randn(1, 256, 7, 7)
        torch.manual_seed(2)
        tensor_b = torch.randn(1, 256, 7, 7)

        extractor = CNNStatisticsExtractor()
        features_a = LayerFeatures(
            tensor=tensor_a,
            spatial=True,
            shape_semantic="batch_channels_height_width",
            layer_name="x",
        )
        features_b = LayerFeatures(
            tensor=tensor_b,
            spatial=True,
            shape_semantic="batch_channels_height_width",
            layer_name="x",
        )

        result_a = extractor.extract(features_a)
        result_b = extractor.extract(features_b)

        assert not torch.equal(result_a.mean, result_b.mean)


# =============================================================================
# CNNStatisticsExtractor Tests — edge cases
# =============================================================================


class TestCNNStatisticsExtractorEdgeCases:
    """Edge case tests for CNNStatisticsExtractor."""

    def test_1x1_spatial_map(self) -> None:
        """[1, C, 1, 1] (global pooled) should still produce shape [C] mean/std."""
        torch.manual_seed(7)
        tensor = torch.randn(1, 2048, 1, 1)
        features = LayerFeatures(
            tensor=tensor,
            spatial=True,
            shape_semantic="batch_channels_height_width",
            layer_name="avgpool",
        )
        extractor = CNNStatisticsExtractor()

        result = extractor.extract(features)

        assert result.mean.shape == (2048,)
        assert result.std.shape == (2048,)

    def test_large_spatial_map(self) -> None:
        """[1, 64, 112, 112] (early layer) should produce shape [64]."""
        torch.manual_seed(3)
        tensor = torch.randn(1, 64, 112, 112)
        features = LayerFeatures(
            tensor=tensor,
            spatial=True,
            shape_semantic="batch_channels_height_width",
            layer_name="layer1",
        )
        extractor = CNNStatisticsExtractor()

        result = extractor.extract(features)

        assert result.mean.shape == (64,)
        assert result.std.shape == (64,)
        assert torch.isfinite(result.mean).all()
        assert torch.isfinite(result.std).all()

    def test_3d_input_fallback(self) -> None:
        """A 3D input [B, N, D] should use the flatten fallback path."""
        torch.manual_seed(5)
        tensor = torch.randn(1, 197, 768)
        features = LayerFeatures(
            tensor=tensor,
            spatial=False,
            shape_semantic="batch_tokens_dim",
            layer_name="shared",
        )
        extractor = CNNStatisticsExtractor()

        result = extractor.extract(features)

        # flatten(start_dim=1) on [1, 197*768]; mean/std over dim=1 -> scalar
        assert result.mean.shape == torch.Size([])
        assert result.std.shape == torch.Size([])

    def test_2d_input_fallback(self) -> None:
        """A 2D input [B, D] should use the flatten fallback path."""
        torch.manual_seed(9)
        tensor = torch.randn(1, 1024)
        features = LayerFeatures(
            tensor=tensor,
            spatial=False,
            shape_semantic="batch_tokens_dim",
            layer_name="fc",
        )
        extractor = CNNStatisticsExtractor()

        result = extractor.extract(features)

        assert result.mean.shape == torch.Size([])
        assert result.std.shape == torch.Size([])


# =============================================================================
# CNNStatisticsExtractor — Protocol conformance
# =============================================================================


class TestCNNStatisticsExtractorProtocol:
    """Verify CNNStatisticsExtractor satisfies the FeatureStatisticsExtractor protocol."""

    def test_cnn_extractor_is_instance_of_protocol(self) -> None:
        extractor = CNNStatisticsExtractor()
        tensor = torch.randn(1, 512, 14, 14)
        features = LayerFeatures(
            tensor=tensor,
            spatial=True,
            shape_semantic="batch_channels_height_width",
            layer_name="layer4",
        )

        assert isinstance(extractor, FeatureStatisticsExtractor)

        result = extractor.extract(features)
        assert isinstance(result, FeatureStatistics)
