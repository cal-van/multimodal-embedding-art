"""
End-to-end tests with real models.

These tests download and run real ImageBind and SDXL VAE models to verify
the full optimization pipeline works correctly on actual hardware.

Marked with @pytest.mark.slow as they require model downloads and GPU/MPS compute.

Requirements:
    - ImageBind must be installed: git clone https://github.com/facebookresearch/ImageBind && pip install -e .
    - SDXL VAE will be downloaded automatically from HuggingFace
    - MPS (Apple Silicon) or CUDA device recommended
"""

import pytest
import torch

from embedding_art.core.concept import Concept
from embedding_art.core.config import AugmentationConfig, OptimizationConfig
from embedding_art.core.engine import EmbeddingArtEngine, OptimizationResult
from embedding_art.generators.image import SDXLImageGenerator
from embedding_art.regularizers import CompositeRegularizer

try:
    from embedding_art.encoders.imagebind import IMAGEBIND_AVAILABLE, ImageBindEncoder
except ImportError:
    IMAGEBIND_AVAILABLE = False
    ImageBindEncoder = None


def get_device() -> str:
    """Get the best available device for testing."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


@pytest.fixture(scope="module")
def real_encoder():
    """
    Load real ImageBind encoder.

    Module-scoped to avoid reloading the model for each test.
    Skipped if ImageBind is not installed.
    """
    if not IMAGEBIND_AVAILABLE:
        pytest.skip(
            "ImageBind not installed. Install with: "
            "git clone https://github.com/facebookresearch/ImageBind && cd ImageBind && pip install -e ."
        )
    device = get_device()
    return ImageBindEncoder(device=device)


@pytest.fixture(scope="module")
def real_generator():
    """
    Load real SDXL VAE generator.

    Module-scoped to avoid reloading the model for each test.
    """
    device = get_device()
    return SDXLImageGenerator(device=device)


@pytest.fixture(scope="module")
def real_engine(real_encoder, real_generator):
    """
    Create engine with real encoder and generator.

    Module-scoped to reuse across tests.
    """
    device = get_device()
    engine = EmbeddingArtEngine(encoder=real_encoder, device=device)
    engine.register_generator("image", real_generator)
    return engine


class TestEndToEndOptimization:
    """End-to-end tests verifying the full optimization pipeline with real models."""

    @pytest.mark.slow
    def test_optimization_reaches_target_similarity_for_simple_concept(
        self, real_engine, real_encoder
    ):
        """
        Should reach similarity >0.7 within 100 steps for a simple text concept.

        This test verifies that the full pipeline (ImageBind + SDXL VAE) can
        successfully optimize a latent to produce an image that embeds close
        to the target text embedding.
        """
        target = Concept.from_text("goldfish", real_encoder)

        config = OptimizationConfig(
            steps=100,
            learning_rate=0.1,
            optimizer="adam",
            scheduler="cosine",
            checkpoint_every=0,
            seed=42,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        result = real_engine.optimize(
            target=target,
            output_modality="image",
            config=config,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
        )

        assert isinstance(result, OptimizationResult)
        assert result.final_similarity > 0.7, (
            f"Expected similarity >0.7, got {result.final_similarity:.4f}. "
            f"Similarity history: {result.similarity_history[:5]}...{result.similarity_history[-5:]}"
        )

    @pytest.mark.slow
    def test_optimization_improves_similarity_over_time(self, real_engine, real_encoder):
        """
        Should show consistent improvement in similarity from start to finish.

        Verifies that optimization is actually working by checking that final
        similarity is better than initial similarity.
        """
        target = Concept.from_text("sunset over ocean", real_encoder)

        config = OptimizationConfig(
            steps=50,
            learning_rate=0.1,
            optimizer="adam",
            scheduler="constant",
            checkpoint_every=0,
            seed=123,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        result = real_engine.optimize(
            target=target,
            output_modality="image",
            config=config,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
        )

        initial_similarity = result.similarity_history[0]
        final_similarity = result.final_similarity

        assert final_similarity > initial_similarity, (
            f"Similarity should improve: {initial_similarity:.4f} -> {final_similarity:.4f}"
        )

    @pytest.mark.slow
    def test_different_concepts_produce_different_results(
        self, real_engine, real_encoder, real_generator
    ):
        """
        Should produce visually different results for different concepts.

        Verifies that the optimization is actually targeting the concept
        by checking that different concepts produce different latents.
        """
        config = OptimizationConfig(
            steps=30,
            learning_rate=0.1,
            optimizer="adam",
            checkpoint_every=0,
            seed=42,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        concept_a = Concept.from_text("fire", real_encoder)
        result_a = real_engine.optimize(
            target=concept_a,
            output_modality="image",
            config=config,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
        )

        concept_b = Concept.from_text("water", real_encoder)
        result_b = real_engine.optimize(
            target=concept_b,
            output_modality="image",
            config=config,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
        )

        latent_diff = (result_a.final_latent - result_b.final_latent).abs().mean().item()
        assert latent_diff > 0.1, (
            f"Different concepts should produce different latents, but diff={latent_diff:.4f}"
        )

    @pytest.mark.slow
    def test_result_can_be_decoded_to_valid_image(self, real_engine, real_encoder, real_generator):
        """
        Should produce a valid PIL Image from the optimization result.

        Verifies the full pipeline including decoding the final latent.
        """
        target = Concept.from_text("mountain landscape", real_encoder)

        config = OptimizationConfig(
            steps=20,
            learning_rate=0.1,
            checkpoint_every=0,
            seed=42,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        result = real_engine.optimize(
            target=target,
            output_modality="image",
            config=config,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
        )

        image = result.get_final_image(real_generator)

        from PIL import Image as PILImage

        assert isinstance(image, PILImage.Image)
        assert image.size == (1024, 1024)
        assert image.mode == "RGB"


class TestImageBindEncoderProperties:
    """Tests verifying ImageBind encoder properties."""

    @pytest.mark.slow
    def test_imagebind_encoder_produces_correct_embedding_dim(self, real_encoder):
        """ImageBind should produce 1024-dimensional embeddings."""
        embedding = real_encoder.encode_text("test")

        assert embedding.shape == (1, 1024)
        assert real_encoder.embedding_dim == 1024

    @pytest.mark.slow
    def test_encoder_device_matches_requested_device(self, real_encoder):
        """Encoder should be on the requested device."""
        expected_device = get_device()
        assert str(real_encoder.device) == expected_device


class TestSDXLGeneratorProperties:
    """Tests verifying SDXL VAE generator properties (no ImageBind required)."""

    @pytest.mark.slow
    def test_sdxl_generator_produces_correct_output_shape(self, real_generator):
        """SDXL VAE should produce 1024x1024 images from 128x128 latents."""
        latent = real_generator.init_latent(seed=42)

        assert latent.shape == (1, 4, 128, 128)

        with torch.no_grad():
            decoded = real_generator.decode(latent)

        assert decoded.shape == (1, 3, 1024, 1024)
        assert decoded.min() >= 0.0
        assert decoded.max() <= 1.0

    @pytest.mark.slow
    def test_generator_device_matches_requested_device(self, real_generator):
        """Generator should be on the requested device."""
        expected_device = get_device()
        assert str(real_generator.device) == expected_device
