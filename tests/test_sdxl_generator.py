"""
Tests for SDXLImageGenerator.

These tests verify that the SDXL VAE generator correctly loads,
initializes latents, decodes to images, and encodes images back to latents.

Tests marked with @pytest.mark.slow will download ~335MB of model weights
on first run and require significant memory to run.
"""

import pytest
import torch

from embedding_art.generators import SDXLImageGenerator


@pytest.fixture(scope="module")
def sdxl_generator() -> SDXLImageGenerator:
    """
    Module-scoped fixture for SDXLImageGenerator.

    Using module scope avoids re-loading the ~335MB model weights
    for each test, significantly improving test run time.

    Returns:
        SDXLImageGenerator instance configured for MPS device
    """
    return SDXLImageGenerator(
        model_id="stabilityai/sdxl-vae",
        device="mps",
    )


@pytest.mark.slow
class TestSDXLImageGeneratorInit:
    """Tests for SDXLImageGenerator initialization and properties."""

    def test_should_load_vae_successfully(
        self,
        sdxl_generator: SDXLImageGenerator,
    ) -> None:
        """The generator should load the SDXL VAE without errors."""
        assert sdxl_generator.vae is not None

    def test_should_have_correct_latent_shape(
        self,
        sdxl_generator: SDXLImageGenerator,
    ) -> None:
        """The latent shape should be [1, 4, 128, 128] for SDXL."""
        expected_shape = (1, 4, 128, 128)
        assert sdxl_generator.latent_shape == expected_shape

    def test_should_report_image_output_modality(
        self,
        sdxl_generator: SDXLImageGenerator,
    ) -> None:
        """The output modality should be 'image'."""
        assert sdxl_generator.output_modality == "image"

    def test_should_use_configured_device(
        self,
        sdxl_generator: SDXLImageGenerator,
    ) -> None:
        """The generator should use the configured device."""
        assert sdxl_generator.device == torch.device("mps")


@pytest.mark.slow
class TestSDXLInitLatent:
    """Tests for the init_latent method."""

    def test_should_return_correct_shape(
        self,
        sdxl_generator: SDXLImageGenerator,
    ) -> None:
        """init_latent should return a tensor with shape [1, 4, 128, 128]."""
        latent = sdxl_generator.init_latent()
        expected_shape = (1, 4, 128, 128)
        assert latent.shape == expected_shape

    def test_should_have_requires_grad_enabled(
        self,
        sdxl_generator: SDXLImageGenerator,
    ) -> None:
        """The returned latent should have requires_grad=True for optimization."""
        latent = sdxl_generator.init_latent()
        assert latent.requires_grad is True

    def test_should_be_on_correct_device(
        self,
        sdxl_generator: SDXLImageGenerator,
    ) -> None:
        """The latent should be on the generator's device."""
        latent = sdxl_generator.init_latent()
        # Compare device types because PyTorch may add index (mps vs mps:0)
        assert latent.device.type == sdxl_generator.device.type

    def test_should_produce_reproducible_output_with_seed(
        self,
        sdxl_generator: SDXLImageGenerator,
    ) -> None:
        """Using the same seed should produce identical latents."""
        seed = 42
        latent_1 = sdxl_generator.init_latent(seed=seed)
        latent_2 = sdxl_generator.init_latent(seed=seed)
        assert torch.allclose(latent_1, latent_2)

    def test_should_produce_different_output_with_different_seeds(
        self,
        sdxl_generator: SDXLImageGenerator,
    ) -> None:
        """Different seeds should produce different latents."""
        latent_1 = sdxl_generator.init_latent(seed=42)
        latent_2 = sdxl_generator.init_latent(seed=123)
        assert not torch.allclose(latent_1, latent_2)


@pytest.mark.slow
class TestSDXLDecode:
    """Tests for the decode method."""

    def test_should_produce_correct_output_shape(
        self,
        sdxl_generator: SDXLImageGenerator,
    ) -> None:
        """decode should produce output shape [1, 3, 1024, 1024]."""
        latent = sdxl_generator.init_latent(seed=42)
        output = sdxl_generator.decode(latent)
        expected_shape = (1, 3, 1024, 1024)
        assert output.shape == expected_shape

    def test_should_produce_values_in_valid_range(
        self,
        sdxl_generator: SDXLImageGenerator,
    ) -> None:
        """decoded image values should be clamped to [0, 1]."""
        latent = sdxl_generator.init_latent(seed=42)
        output = sdxl_generator.decode(latent)
        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_should_be_differentiable(
        self,
        sdxl_generator: SDXLImageGenerator,
    ) -> None:
        """decode should allow gradients to flow back for optimization."""
        latent = sdxl_generator.init_latent(seed=42)
        output = sdxl_generator.decode(latent)

        loss = output.mean()
        loss.backward()

        assert latent.grad is not None
        assert latent.grad.shape == latent.shape


@pytest.mark.slow
class TestSDXLRoundTrip:
    """Tests for encode/decode round-trip."""

    def test_encode_should_produce_correct_shape(
        self,
        sdxl_generator: SDXLImageGenerator,
    ) -> None:
        """encode should produce latent shape [1, 4, 128, 128] from 1024x1024 image."""
        image = torch.rand(1, 3, 1024, 1024, device=sdxl_generator.device)
        latent = sdxl_generator.encode(image)
        expected_shape = (1, 4, 128, 128)
        assert latent.shape == expected_shape

    def test_round_trip_should_preserve_general_structure(
        self,
        sdxl_generator: SDXLImageGenerator,
    ) -> None:
        """
        Encoding and decoding should roughly preserve image structure.

        VAE round-trips are lossy, so we don't expect exact reconstruction,
        but the general pixel distribution should be similar.
        """
        original = torch.rand(1, 3, 1024, 1024, device=sdxl_generator.device)
        latent = sdxl_generator.encode(original)
        reconstructed = sdxl_generator.decode(latent.detach())

        # Check that means are roughly similar (within 20%)
        original_mean = original.mean().item()
        reconstructed_mean = reconstructed.mean().item()
        relative_diff = abs(original_mean - reconstructed_mean) / max(original_mean, 1e-6)
        assert relative_diff < 0.2, f"Mean difference too large: {relative_diff:.2%}"

    def test_decode_encode_should_produce_similar_latent(
        self,
        sdxl_generator: SDXLImageGenerator,
    ) -> None:
        """
        Decoding then encoding should produce a latent similar to the original.

        This tests that the VAE's encode/decode operations are roughly inverse.
        """
        original_latent = sdxl_generator.init_latent(seed=42)
        image = sdxl_generator.decode(original_latent.detach())
        reconstructed_latent = sdxl_generator.encode(image)

        # Compute cosine similarity between flattened latents
        flat_original = original_latent.flatten()
        flat_reconstructed = reconstructed_latent.flatten()
        cosine_sim = torch.nn.functional.cosine_similarity(
            flat_original.unsqueeze(0),
            flat_reconstructed.unsqueeze(0),
        )

        # VAE round-trips are lossy, but should maintain reasonable similarity
        assert cosine_sim.item() > 0.5, f"Latent similarity too low: {cosine_sim.item():.3f}"
