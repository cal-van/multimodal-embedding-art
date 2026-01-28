"""
Tests for ImageBindEncoder.

These tests verify the behavior of the ImageBind encoder wrapper.
Since ImageBind requires separate installation from https://github.com/facebookresearch/ImageBind,
tests are skipped if ImageBind is not available.

Installation instructions:
    git clone https://github.com/facebookresearch/ImageBind
    cd ImageBind && pip install -e .
"""

import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from embedding_art.encoders.imagebind import IMAGEBIND_AVAILABLE, ImageBindEncoder
from embedding_art.exceptions import ImageBindNotInstalledError

SKIP_IF_IMAGEBIND_NOT_INSTALLED = pytest.mark.skipif(
    not IMAGEBIND_AVAILABLE,
    reason="ImageBind not installed. Install from https://github.com/facebookresearch/ImageBind",
)


class TestImageBindEncoderImport:
    """Tests for ImageBindEncoder import and availability detection."""

    @pytest.mark.skipif(
        IMAGEBIND_AVAILABLE,
        reason="Only run when ImageBind is NOT installed",
    )
    def test_instantiation_raises_error_without_imagebind(self) -> None:
        """Should raise ImageBindNotInstalledError with installation instructions when missing."""
        with pytest.raises(ImageBindNotInstalledError) as exc_info:
            ImageBindEncoder()

        error_message = str(exc_info.value)
        assert "ImageBind is not installed" in error_message
        assert "git clone" in error_message
        assert "pip install -e" in error_message


@SKIP_IF_IMAGEBIND_NOT_INSTALLED
@pytest.mark.slow
@pytest.mark.integration
class TestImageBindEncoderLoading:
    """Tests for ImageBindEncoder model loading."""

    def test_encoder_instantiation(self) -> None:
        """Should successfully instantiate ImageBindEncoder when ImageBind is installed."""
        encoder = ImageBindEncoder(device="cpu", pretrained=True)

        assert encoder is not None
        assert encoder.embedding_dim == 1024
        assert encoder.device.type == "cpu"

    def test_model_is_frozen(self) -> None:
        """All model parameters should be frozen (requires_grad=False)."""
        encoder = ImageBindEncoder(device="cpu", pretrained=True)

        for param in encoder.model.parameters():
            assert not param.requires_grad, "All parameters should be frozen"


@SKIP_IF_IMAGEBIND_NOT_INSTALLED
@pytest.mark.slow
@pytest.mark.integration
class TestImageBindEncoderEncodeText:
    """Tests for text encoding."""

    @pytest.fixture
    def encoder(self) -> ImageBindEncoder:
        """Fixture providing an ImageBindEncoder instance."""
        return ImageBindEncoder(device="cpu", pretrained=True)

    def test_encode_text_returns_correct_shape(self, encoder: ImageBindEncoder) -> None:
        """encode_text should return tensor of shape [1, 1024]."""
        embedding = encoder.encode_text("a beautiful sunset")

        assert embedding.shape == (1, 1024)

    def test_encode_text_returns_normalized_embedding(self, encoder: ImageBindEncoder) -> None:
        """encode_text should return L2-normalized embedding."""
        embedding = encoder.encode_text("a beautiful sunset")

        norm = torch.norm(embedding, dim=-1)
        assert torch.allclose(norm, torch.tensor([1.0]), atol=1e-5)

    def test_encode_text_different_texts_produce_different_embeddings(
        self, encoder: ImageBindEncoder
    ) -> None:
        """Different texts should produce different embeddings."""
        embedding1 = encoder.encode_text("a beautiful sunset")
        embedding2 = encoder.encode_text("a dark stormy night")

        similarity = torch.cosine_similarity(embedding1, embedding2)
        assert similarity < 0.99, "Different texts should produce different embeddings"

    def test_encode_text_same_text_produces_identical_embedding(
        self, encoder: ImageBindEncoder
    ) -> None:
        """Same text should produce identical embedding."""
        text = "a beautiful sunset"
        embedding1 = encoder.encode_text(text)
        embedding2 = encoder.encode_text(text)

        assert torch.allclose(embedding1, embedding2)


@SKIP_IF_IMAGEBIND_NOT_INSTALLED
@pytest.mark.slow
@pytest.mark.integration
class TestImageBindEncoderEncodeImage:
    """Tests for image encoding."""

    @pytest.fixture
    def encoder(self) -> ImageBindEncoder:
        """Fixture providing an ImageBindEncoder instance."""
        return ImageBindEncoder(device="cpu", pretrained=True)

    @pytest.fixture
    def test_image(self) -> Image.Image:
        """Fixture providing a simple test image."""
        return Image.new("RGB", (224, 224), color=(255, 128, 64))

    def test_encode_image_pil_returns_correct_shape(
        self, encoder: ImageBindEncoder, test_image: Image.Image
    ) -> None:
        """encode_image with PIL Image should return tensor of shape [1, 1024]."""
        embedding = encoder.encode_image(test_image)

        assert embedding.shape == (1, 1024)

    def test_encode_image_pil_returns_normalized_embedding(
        self, encoder: ImageBindEncoder, test_image: Image.Image
    ) -> None:
        """encode_image with PIL Image should return L2-normalized embedding."""
        embedding = encoder.encode_image(test_image)

        norm = torch.norm(embedding, dim=-1)
        assert torch.allclose(norm, torch.tensor([1.0]), atol=1e-5)

    def test_encode_image_path_returns_correct_shape(self, encoder: ImageBindEncoder) -> None:
        """encode_image with file path should return tensor of shape [1, 1024]."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            test_image = Image.new("RGB", (224, 224), color=(128, 64, 255))
            test_image.save(f.name)
            image_path = Path(f.name)

        try:
            embedding = encoder.encode_image(image_path)
            assert embedding.shape == (1, 1024)
        finally:
            image_path.unlink()

    def test_encode_image_different_images_produce_different_embeddings(
        self, encoder: ImageBindEncoder
    ) -> None:
        """Different images should produce different embeddings."""
        image1 = Image.new("RGB", (224, 224), color=(255, 0, 0))
        image2 = Image.new("RGB", (224, 224), color=(0, 0, 255))

        embedding1 = encoder.encode_image(image1)
        embedding2 = encoder.encode_image(image2)

        similarity = torch.cosine_similarity(embedding1, embedding2)
        assert similarity < 0.99, "Different images should produce different embeddings"


@SKIP_IF_IMAGEBIND_NOT_INSTALLED
@pytest.mark.slow
@pytest.mark.integration
class TestImageBindEncoderEncodeForOptimization:
    """Tests for optimization-compatible image encoding."""

    @pytest.fixture
    def encoder(self) -> ImageBindEncoder:
        """Fixture providing an ImageBindEncoder instance."""
        return ImageBindEncoder(device="cpu", pretrained=True)

    def test_encode_for_optimization_returns_correct_shape(self, encoder: ImageBindEncoder) -> None:
        """encode_for_optimization should return tensor of shape [1, 1024]."""
        image_tensor = torch.rand(1, 3, 512, 512)

        embedding = encoder.encode_for_optimization(image_tensor)

        assert embedding.shape == (1, 1024)

    def test_encode_for_optimization_returns_normalized_embedding(
        self, encoder: ImageBindEncoder
    ) -> None:
        """encode_for_optimization should return L2-normalized embedding."""
        image_tensor = torch.rand(1, 3, 512, 512)

        embedding = encoder.encode_for_optimization(image_tensor)

        norm = torch.norm(embedding, dim=-1)
        assert torch.allclose(norm, torch.tensor([1.0]), atol=1e-5)

    def test_encode_for_optimization_handles_224x224_input(self, encoder: ImageBindEncoder) -> None:
        """encode_for_optimization should handle ImageBind's native 224x224 size."""
        image_tensor = torch.rand(1, 3, 224, 224)

        embedding = encoder.encode_for_optimization(image_tensor)

        assert embedding.shape == (1, 1024)

    def test_encode_for_optimization_handles_non_square_input(
        self, encoder: ImageBindEncoder
    ) -> None:
        """encode_for_optimization should handle non-square images by resizing."""
        image_tensor = torch.rand(1, 3, 256, 512)

        embedding = encoder.encode_for_optimization(image_tensor)

        assert embedding.shape == (1, 1024)


@SKIP_IF_IMAGEBIND_NOT_INSTALLED
@pytest.mark.slow
@pytest.mark.integration
class TestImageBindEncoderCrossModalSimilarity:
    """Tests for cross-modal similarity between text and image embeddings."""

    @pytest.fixture
    def encoder(self) -> ImageBindEncoder:
        """Fixture providing an ImageBindEncoder instance."""
        return ImageBindEncoder(device="cpu", pretrained=True)

    def test_text_and_image_embeddings_in_same_space(self, encoder: ImageBindEncoder) -> None:
        """Text and image embeddings should be in the same 1024-dim space."""
        text_embedding = encoder.encode_text("a red square")
        image = Image.new("RGB", (224, 224), color=(255, 0, 0))
        image_embedding = encoder.encode_image(image)

        assert text_embedding.shape == image_embedding.shape == (1, 1024)

        similarity = torch.cosine_similarity(text_embedding, image_embedding)
        assert -1.0 <= similarity.item() <= 1.0
