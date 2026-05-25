"""
Tests for ``SD35ImageGenerator`` — the canonical 2026-crispy image backbone.

Fast tests verify the latent geometry, output modality, and helpful error
classification without loading the actual SD3.5 VAE checkpoint (~250MB).
Integration tests are gated on ``@pytest.mark.slow``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from embedding_art.generators.sd35 import (
    LATENT_CHANNELS,
    SD35_SCALING_FACTOR,
    SD35_SHIFT_FACTOR,
    SD35ImageGenerator,
)

# ---------------------------------------------------------------------------
# Static-shape tests using a mocked VAE
# ---------------------------------------------------------------------------


def _build_with_mock_vae(output_size: int = 1024, device: str = "cpu") -> SD35ImageGenerator:
    """Construct an SD35ImageGenerator with a mocked VAE for fast unit tests.

    The mock VAE returns a zero tensor of the correct decoded shape so that
    downstream arithmetic (clamp, scale) doesn't touch real model weights.
    """
    with patch("embedding_art.generators.sd35.AutoencoderKL") as mock_kl:
        mock_vae = MagicMock()
        mock_vae.to.return_value = mock_vae
        mock_vae.eval.return_value = None
        mock_vae.parameters.return_value = iter([])

        def decode(latent):
            b = latent.shape[0]
            result = MagicMock()
            result.sample = torch.zeros(b, 3, output_size, output_size)
            return result

        def encode(image):
            b = image.shape[0]
            sample = torch.zeros(b, LATENT_CHANNELS, output_size // 8, output_size // 8)
            latent_dist = MagicMock()
            latent_dist.sample.return_value = sample
            result = MagicMock()
            result.latent_dist = latent_dist
            return result

        mock_vae.decode.side_effect = decode
        mock_vae.encode.side_effect = encode
        mock_kl.from_pretrained.return_value = mock_vae

        return SD35ImageGenerator(device=device, output_size=output_size)


class TestSD35Constants:
    """SD3.5 model constants must match the published config."""

    def test_latent_channels_is_sixteen(self) -> None:
        """SD3.5 uses a 16-channel latent (vs SDXL's 4)."""
        assert LATENT_CHANNELS == 16

    def test_scaling_factor_matches_published(self) -> None:
        """SD3.5 VAE scaling factor from the model's vae/config.json."""
        assert SD35_SCALING_FACTOR == pytest.approx(1.5305)

    def test_shift_factor_matches_published(self) -> None:
        """SD3.5 VAE shift factor from the model's vae/config.json."""
        assert SD35_SHIFT_FACTOR == pytest.approx(0.0609)


class TestSD35LatentShape:
    """Latent shape matches the SD3.5 spec at multiple output resolutions."""

    def test_latent_shape_at_1024(self) -> None:
        gen = _build_with_mock_vae(output_size=1024)
        assert gen.latent_shape == (1, 16, 128, 128)

    def test_latent_shape_at_512(self) -> None:
        gen = _build_with_mock_vae(output_size=512)
        assert gen.latent_shape == (1, 16, 64, 64)

    def test_non_multiple_of_8_output_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="multiple of 8"):
            _build_with_mock_vae(output_size=1023)


class TestSD35Properties:
    """Basic protocol properties."""

    def test_output_modality_is_image(self) -> None:
        gen = _build_with_mock_vae()
        assert gen.output_modality == "image"

    def test_device_property(self) -> None:
        gen = _build_with_mock_vae(device="cpu")
        assert gen.device == torch.device("cpu")

    def test_output_size_property(self) -> None:
        gen = _build_with_mock_vae(output_size=512)
        assert gen.output_size == 512


class TestSD35InitLatent:
    """init_latent creates an optimizable tensor of the right shape."""

    def test_init_latent_returns_correct_shape(self) -> None:
        gen = _build_with_mock_vae(output_size=1024)
        latent = gen.init_latent(seed=42)
        assert latent.shape == (1, 16, 128, 128)

    def test_init_latent_requires_grad(self) -> None:
        gen = _build_with_mock_vae()
        latent = gen.init_latent(seed=42)
        assert latent.requires_grad is True

    def test_init_latent_deterministic_with_seed(self) -> None:
        gen = _build_with_mock_vae()
        a = gen.init_latent(seed=42)
        b = gen.init_latent(seed=42)
        assert torch.allclose(a, b)

    def test_init_latent_differs_with_different_seeds(self) -> None:
        gen = _build_with_mock_vae()
        a = gen.init_latent(seed=42)
        b = gen.init_latent(seed=43)
        assert not torch.allclose(a, b)


class TestSD35Decode:
    """decode produces the right output shape and value range."""

    def test_decode_returns_correct_shape(self) -> None:
        gen = _build_with_mock_vae(output_size=1024)
        latent = gen.init_latent(seed=42)
        image = gen.decode(latent)
        assert image.shape == (1, 3, 1024, 1024)

    def test_decode_output_in_zero_one_range(self) -> None:
        gen = _build_with_mock_vae()
        latent = gen.init_latent(seed=42)
        image = gen.decode(latent)
        assert image.min() >= 0.0
        assert image.max() <= 1.0


# ---------------------------------------------------------------------------
# Integration tests — require the real SD3.5 VAE checkpoint (~250MB)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSD35IntegrationLoad:
    """End-to-end test loading the real SD3.5 VAE."""

    def test_real_vae_loads(self) -> None:
        gen = SD35ImageGenerator(device="cpu", output_size=512)
        assert gen.vae is not None


@pytest.mark.slow
class TestSD35IntegrationRoundtrip:
    """Encode an image, decode it back, verify approximate fidelity."""

    def test_encode_decode_roundtrip(self) -> None:
        gen = SD35ImageGenerator(device="cpu", output_size=512)
        # Create a synthetic test image.
        image = torch.rand(1, 3, 512, 512)
        latent = gen.encode(image)
        decoded = gen.decode(latent)
        # VAE compression is lossy; expect MSE < 0.1 not perfect identity.
        mse = (image - decoded).pow(2).mean().item()
        assert mse < 0.1
