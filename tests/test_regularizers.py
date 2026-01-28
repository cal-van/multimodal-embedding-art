"""
Unit tests for regularization losses.

Tests verify behavior of each regularizer through their public API,
ensuring they correctly penalize or reward different input patterns.
"""

import pytest
import torch

from embedding_art.regularizers.base import (
    CompositeRegularizer,
    LatentNorm,
    SpectralRegularizer,
    TotalVariation,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def smooth_image() -> torch.Tensor:
    """
    Create a smooth image tensor with low total variation and low spectral energy.

    This is a gradient that transitions smoothly from dark to light,
    with no sharp edges or high-frequency content.
    """
    batch_size, channels, height, width = 1, 3, 64, 64

    y_coords = torch.linspace(0, 1, height).unsqueeze(1).expand(height, width)

    image = y_coords.unsqueeze(0).unsqueeze(0).expand(batch_size, channels, height, width)
    return image.clone()


@pytest.fixture
def noisy_image() -> torch.Tensor:
    """
    Create a noisy image tensor with high total variation and high spectral energy.

    This is a checkerboard pattern that creates maximum pixel-to-pixel variation.
    """
    batch_size, channels, height, width = 1, 3, 64, 64

    row_indices = torch.arange(height).unsqueeze(1).expand(height, width)
    col_indices = torch.arange(width).unsqueeze(0).expand(height, width)
    checkerboard = ((row_indices + col_indices) % 2).float()

    image = checkerboard.unsqueeze(0).unsqueeze(0).expand(batch_size, channels, height, width)
    return image.clone()


@pytest.fixture
def sharp_edge_image() -> torch.Tensor:
    """
    Create an image with sharp edges (alternating stripes).

    Multiple sharp vertical transitions that create high pixel-to-pixel variation.
    """
    batch_size, channels, height, width = 1, 3, 64, 64

    col_indices = torch.arange(width).unsqueeze(0).expand(height, width)
    stripes = ((col_indices // 4) % 2).float()

    image = stripes.unsqueeze(0).unsqueeze(0).expand(batch_size, channels, height, width)
    return image.clone()


@pytest.fixture
def normal_latent() -> torch.Tensor:
    """
    Create a latent tensor with approximately N(0, 1) distribution.
    """
    torch.manual_seed(42)
    return torch.randn(1, 4, 64, 64)


@pytest.fixture
def abnormal_latent_high_mean() -> torch.Tensor:
    """
    Create a latent tensor with mean significantly different from 0.
    """
    torch.manual_seed(42)
    latent = torch.randn(1, 4, 64, 64)
    return latent + 5.0


@pytest.fixture
def abnormal_latent_high_std() -> torch.Tensor:
    """
    Create a latent tensor with std significantly different from 1.
    """
    torch.manual_seed(42)
    latent = torch.randn(1, 4, 64, 64)
    return latent * 5.0


@pytest.fixture
def abnormal_latent_low_std() -> torch.Tensor:
    """
    Create a latent tensor with std significantly less than 1.
    """
    torch.manual_seed(42)
    latent = torch.randn(1, 4, 64, 64)
    return latent * 0.1


# =============================================================================
# TotalVariation Tests
# =============================================================================


class TestTotalVariation:
    """Tests for TotalVariation regularizer."""

    def test_returns_zero_when_decoded_is_none(self, normal_latent: torch.Tensor) -> None:
        """TotalVariation should return 0 when no decoded image is provided."""
        regularizer = TotalVariation(weight=0.01)

        loss = regularizer(normal_latent, decoded=None)

        assert loss.item() == 0.0

    def test_returns_higher_loss_for_sharp_edges_than_smooth(
        self,
        normal_latent: torch.Tensor,
        smooth_image: torch.Tensor,
        sharp_edge_image: torch.Tensor,
    ) -> None:
        """Images with sharp edges should have higher TV loss than smooth gradients."""
        regularizer = TotalVariation(weight=1.0)

        smooth_loss = regularizer(normal_latent, decoded=smooth_image)
        sharp_loss = regularizer(normal_latent, decoded=sharp_edge_image)

        assert sharp_loss.item() > smooth_loss.item()

    def test_returns_higher_loss_for_noisy_than_smooth(
        self, normal_latent: torch.Tensor, smooth_image: torch.Tensor, noisy_image: torch.Tensor
    ) -> None:
        """Noisy checkerboard should have higher TV loss than smooth gradient."""
        regularizer = TotalVariation(weight=1.0)

        smooth_loss = regularizer(normal_latent, decoded=smooth_image)
        noisy_loss = regularizer(normal_latent, decoded=noisy_image)

        assert noisy_loss.item() > smooth_loss.item()

    def test_weight_parameter_scales_loss_correctly(
        self, normal_latent: torch.Tensor, noisy_image: torch.Tensor
    ) -> None:
        """Weight parameter should scale the loss proportionally."""
        regularizer_low = TotalVariation(weight=0.01)
        regularizer_high = TotalVariation(weight=0.1)

        loss_low = regularizer_low(normal_latent, decoded=noisy_image)
        loss_high = regularizer_high(normal_latent, decoded=noisy_image)

        expected_ratio = 0.1 / 0.01
        actual_ratio = loss_high.item() / loss_low.item()

        assert pytest.approx(actual_ratio, rel=1e-5) == expected_ratio

    def test_constant_image_has_zero_tv(self, normal_latent: torch.Tensor) -> None:
        """A constant-value image should have zero total variation."""
        regularizer = TotalVariation(weight=1.0)
        constant_image = torch.ones(1, 3, 64, 64) * 0.5

        loss = regularizer(normal_latent, decoded=constant_image)

        assert loss.item() == 0.0


# =============================================================================
# SpectralRegularizer Tests
# =============================================================================


class TestSpectralRegularizer:
    """Tests for SpectralRegularizer."""

    def test_returns_zero_when_decoded_is_none(self, normal_latent: torch.Tensor) -> None:
        """SpectralRegularizer should return 0 when no decoded image is provided."""
        regularizer = SpectralRegularizer(weight=0.001)

        loss = regularizer(normal_latent, decoded=None)

        assert loss.item() == 0.0

    def test_returns_higher_loss_for_noisy_than_smooth(
        self, normal_latent: torch.Tensor, smooth_image: torch.Tensor, noisy_image: torch.Tensor
    ) -> None:
        """High-frequency noise should have higher spectral loss than smooth images."""
        regularizer = SpectralRegularizer(weight=1.0)

        smooth_loss = regularizer(normal_latent, decoded=smooth_image)
        noisy_loss = regularizer(normal_latent, decoded=noisy_image)

        assert noisy_loss.item() > smooth_loss.item()

    def test_high_freq_threshold_affects_loss(self, normal_latent: torch.Tensor) -> None:
        """Lower threshold should include more frequencies, increasing the loss for mid-freq content."""
        regularizer_high_threshold = SpectralRegularizer(weight=1.0, high_freq_threshold=0.9)
        regularizer_low_threshold = SpectralRegularizer(weight=1.0, high_freq_threshold=0.3)

        batch_size, channels, height, width = 1, 3, 64, 64
        x = torch.linspace(0, 8 * 3.14159, width).unsqueeze(0).expand(height, width)
        mid_freq_pattern = torch.sin(x)
        mid_freq_image = (
            mid_freq_pattern.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, channels, height, width)
            .clone()
        )

        loss_high_threshold = regularizer_high_threshold(normal_latent, decoded=mid_freq_image)
        loss_low_threshold = regularizer_low_threshold(normal_latent, decoded=mid_freq_image)

        assert loss_low_threshold.item() > loss_high_threshold.item()

    def test_weight_parameter_scales_loss_correctly(
        self, normal_latent: torch.Tensor, noisy_image: torch.Tensor
    ) -> None:
        """Weight parameter should scale the loss proportionally."""
        regularizer_low = SpectralRegularizer(weight=0.001)
        regularizer_high = SpectralRegularizer(weight=0.01)

        loss_low = regularizer_low(normal_latent, decoded=noisy_image)
        loss_high = regularizer_high(normal_latent, decoded=noisy_image)

        expected_ratio = 0.01 / 0.001
        actual_ratio = loss_high.item() / loss_low.item()

        assert pytest.approx(actual_ratio, rel=1e-5) == expected_ratio

    def test_handles_grayscale_image(self, normal_latent: torch.Tensor) -> None:
        """SpectralRegularizer should handle single-channel grayscale images."""
        regularizer = SpectralRegularizer(weight=1.0)
        grayscale_noisy = torch.zeros(1, 1, 64, 64)

        row_indices = torch.arange(64).unsqueeze(1).expand(64, 64)
        col_indices = torch.arange(64).unsqueeze(0).expand(64, 64)
        checkerboard = ((row_indices + col_indices) % 2).float()
        grayscale_noisy[0, 0] = checkerboard

        loss = regularizer(normal_latent, decoded=grayscale_noisy)

        assert loss.item() > 0.0


# =============================================================================
# LatentNorm Tests
# =============================================================================


class TestLatentNorm:
    """Tests for LatentNorm regularizer."""

    def test_penalizes_latent_with_high_mean(
        self, normal_latent: torch.Tensor, abnormal_latent_high_mean: torch.Tensor
    ) -> None:
        """Latents with mean far from 0 should have higher loss."""
        regularizer = LatentNorm(weight=1.0)

        normal_loss = regularizer(normal_latent)
        high_mean_loss = regularizer(abnormal_latent_high_mean)

        assert high_mean_loss.item() > normal_loss.item()

    def test_penalizes_latent_with_high_std(
        self, normal_latent: torch.Tensor, abnormal_latent_high_std: torch.Tensor
    ) -> None:
        """Latents with std far above target should have higher loss."""
        regularizer = LatentNorm(weight=1.0, target_std=1.0)

        normal_loss = regularizer(normal_latent)
        high_std_loss = regularizer(abnormal_latent_high_std)

        assert high_std_loss.item() > normal_loss.item()

    def test_penalizes_latent_with_low_std(
        self, normal_latent: torch.Tensor, abnormal_latent_low_std: torch.Tensor
    ) -> None:
        """Latents with std far below target should have higher loss."""
        regularizer = LatentNorm(weight=1.0, target_std=1.0)

        normal_loss = regularizer(normal_latent)
        low_std_loss = regularizer(abnormal_latent_low_std)

        assert low_std_loss.item() > normal_loss.item()

    def test_weight_parameter_scales_loss_correctly(
        self, abnormal_latent_high_mean: torch.Tensor
    ) -> None:
        """Weight parameter should scale the loss proportionally."""
        regularizer_low = LatentNorm(weight=0.1)
        regularizer_high = LatentNorm(weight=1.0)

        loss_low = regularizer_low(abnormal_latent_high_mean)
        loss_high = regularizer_high(abnormal_latent_high_mean)

        expected_ratio = 1.0 / 0.1
        actual_ratio = loss_high.item() / loss_low.item()

        assert pytest.approx(actual_ratio, rel=1e-5) == expected_ratio

    def test_custom_target_std(self, normal_latent: torch.Tensor) -> None:
        """Custom target_std should change what is considered 'normal'."""
        regularizer_std_1 = LatentNorm(weight=1.0, target_std=1.0)
        regularizer_std_2 = LatentNorm(weight=1.0, target_std=2.0)

        loss_std_1 = regularizer_std_1(normal_latent)
        loss_std_2 = regularizer_std_2(normal_latent)

        assert loss_std_2.item() > loss_std_1.item()

    def test_perfect_latent_has_minimal_loss(self) -> None:
        """A latent with exactly mean=0 and std=1 should have near-zero loss."""
        regularizer = LatentNorm(weight=1.0, target_std=1.0)

        perfect_latent = torch.zeros(1, 4, 64, 64)
        perfect_latent[0, 0, :32, :] = 1.0
        perfect_latent[0, 0, 32:, :] = -1.0
        perfect_latent = perfect_latent / perfect_latent.std()

        loss = regularizer(perfect_latent)

        assert loss.item() < 0.1

    def test_ignores_decoded_parameter(
        self, normal_latent: torch.Tensor, noisy_image: torch.Tensor
    ) -> None:
        """LatentNorm should not use the decoded parameter."""
        regularizer = LatentNorm(weight=1.0)

        loss_without_decoded = regularizer(normal_latent, decoded=None)
        loss_with_decoded = regularizer(normal_latent, decoded=noisy_image)

        assert loss_without_decoded.item() == loss_with_decoded.item()


# =============================================================================
# CompositeRegularizer Tests
# =============================================================================


class TestCompositeRegularizer:
    """Tests for CompositeRegularizer."""

    def test_combines_multiple_regularizers(
        self, normal_latent: torch.Tensor, noisy_image: torch.Tensor
    ) -> None:
        """CompositeRegularizer should sum the losses of all component regularizers."""
        tv = TotalVariation(weight=1.0)
        spectral = SpectralRegularizer(weight=1.0)
        latent_norm = LatentNorm(weight=1.0)

        composite = CompositeRegularizer(regularizers=[tv, spectral, latent_norm])

        individual_sum = (
            tv(normal_latent, noisy_image).item()
            + spectral(normal_latent, noisy_image).item()
            + latent_norm(normal_latent, noisy_image).item()
        )
        composite_loss = composite(normal_latent, noisy_image).item()

        assert pytest.approx(composite_loss, rel=1e-5) == individual_sum

    def test_returns_zero_for_empty_regularizer_list(
        self, normal_latent: torch.Tensor, noisy_image: torch.Tensor
    ) -> None:
        """Empty regularizer list should return 0 loss."""
        composite = CompositeRegularizer(regularizers=[])

        loss = composite(normal_latent, decoded=noisy_image)

        assert loss.item() == 0.0

    def test_default_regularizer_list(self, normal_latent: torch.Tensor) -> None:
        """Default CompositeRegularizer should have empty regularizer list."""
        composite = CompositeRegularizer()

        loss = composite(normal_latent)

        assert loss.item() == 0.0

    def test_preset_default_image_returns_nonzero_loss(
        self, normal_latent: torch.Tensor, noisy_image: torch.Tensor
    ) -> None:
        """default_image preset should return non-zero loss for noisy input."""
        composite = CompositeRegularizer.default_image()

        loss = composite(normal_latent, decoded=noisy_image)

        assert loss.item() > 0.0
        assert len(composite.regularizers) == 3

    def test_preset_minimal_returns_nonzero_loss(
        self, abnormal_latent_high_mean: torch.Tensor
    ) -> None:
        """minimal preset should return non-zero loss for abnormal latent."""
        composite = CompositeRegularizer.minimal()

        loss = composite(abnormal_latent_high_mean)

        assert loss.item() > 0.0
        assert len(composite.regularizers) == 1

    def test_preset_heavy_returns_nonzero_loss(
        self, normal_latent: torch.Tensor, noisy_image: torch.Tensor
    ) -> None:
        """heavy preset should return non-zero loss for noisy input."""
        composite = CompositeRegularizer.heavy()

        loss = composite(normal_latent, decoded=noisy_image)

        assert loss.item() > 0.0
        assert len(composite.regularizers) == 3

    def test_presets_return_different_total_losses(
        self, normal_latent: torch.Tensor, noisy_image: torch.Tensor
    ) -> None:
        """Different presets should return different loss values for same input."""
        default_preset = CompositeRegularizer.default_image()
        minimal_preset = CompositeRegularizer.minimal()
        heavy_preset = CompositeRegularizer.heavy()

        default_loss = default_preset(normal_latent, decoded=noisy_image).item()
        minimal_loss = minimal_preset(normal_latent, decoded=noisy_image).item()
        heavy_loss = heavy_preset(normal_latent, decoded=noisy_image).item()

        assert heavy_loss > default_loss
        assert default_loss > minimal_loss

    def test_heavy_preset_has_higher_weights_than_default(self) -> None:
        """heavy preset should have higher weights than default_image preset."""
        default_preset = CompositeRegularizer.default_image()
        heavy_preset = CompositeRegularizer.heavy()

        default_tv_weight = next(
            r.weight for r in default_preset.regularizers if isinstance(r, TotalVariation)
        )
        heavy_tv_weight = next(
            r.weight for r in heavy_preset.regularizers if isinstance(r, TotalVariation)
        )

        assert heavy_tv_weight > default_tv_weight

    def test_minimal_preset_only_has_latent_norm(self) -> None:
        """minimal preset should only contain LatentNorm regularizer."""
        minimal_preset = CompositeRegularizer.minimal()

        assert len(minimal_preset.regularizers) == 1
        assert isinstance(minimal_preset.regularizers[0], LatentNorm)


# =============================================================================
# Edge Cases and Device Handling
# =============================================================================


class TestDeviceHandling:
    """Tests for proper device handling in regularizers."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_total_variation_respects_device(self) -> None:
        """TotalVariation should return loss on same device as input."""
        regularizer = TotalVariation(weight=0.01)
        latent = torch.randn(1, 4, 64, 64).cuda()
        decoded = torch.randn(1, 3, 64, 64).cuda()

        loss = regularizer(latent, decoded=decoded)

        assert loss.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_spectral_regularizer_respects_device(self) -> None:
        """SpectralRegularizer should return loss on same device as input."""
        regularizer = SpectralRegularizer(weight=0.001)
        latent = torch.randn(1, 4, 64, 64).cuda()
        decoded = torch.randn(1, 3, 64, 64).cuda()

        loss = regularizer(latent, decoded=decoded)

        assert loss.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_latent_norm_respects_device(self) -> None:
        """LatentNorm should compute on same device as input."""
        regularizer = LatentNorm(weight=0.1)
        latent = torch.randn(1, 4, 64, 64).cuda()

        loss = regularizer(latent)

        assert loss.device.type == "cuda"

    def test_total_variation_none_decoded_uses_latent_device(self) -> None:
        """When decoded is None, return tensor should be on latent's device."""
        regularizer = TotalVariation(weight=0.01)
        latent = torch.randn(1, 4, 64, 64)

        loss = regularizer(latent, decoded=None)

        assert loss.device == latent.device

    def test_spectral_regularizer_none_decoded_uses_latent_device(self) -> None:
        """When decoded is None, return tensor should be on latent's device."""
        regularizer = SpectralRegularizer(weight=0.001)
        latent = torch.randn(1, 4, 64, 64)

        loss = regularizer(latent, decoded=None)

        assert loss.device == latent.device


class TestBatchHandling:
    """Tests for proper batch dimension handling."""

    def test_total_variation_handles_batch_size_greater_than_one(self) -> None:
        """TotalVariation should handle batched inputs."""
        regularizer = TotalVariation(weight=1.0)
        latent = torch.randn(4, 4, 64, 64)
        decoded = torch.randn(4, 3, 64, 64)

        loss = regularizer(latent, decoded=decoded)

        assert loss.dim() == 0

    def test_spectral_regularizer_handles_batch_size_greater_than_one(self) -> None:
        """SpectralRegularizer should handle batched inputs."""
        regularizer = SpectralRegularizer(weight=1.0)
        latent = torch.randn(4, 4, 64, 64)
        decoded = torch.randn(4, 3, 64, 64)

        loss = regularizer(latent, decoded=decoded)

        assert loss.dim() == 0

    def test_latent_norm_handles_batch_size_greater_than_one(self) -> None:
        """LatentNorm should handle batched inputs."""
        regularizer = LatentNorm(weight=1.0)
        latent = torch.randn(4, 4, 64, 64)

        loss = regularizer(latent)

        assert loss.dim() == 0
