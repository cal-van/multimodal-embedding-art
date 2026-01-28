"""
Shared pytest fixtures for embedding_art tests.

Provides mock implementations of encoders and generators for testing,
along with common test utilities like sample embeddings and temporary directories.
"""

from collections.abc import Generator as TypingGenerator
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from PIL import Image


class MockEncoder:
    """
    Mock encoder that returns deterministic embeddings for testing.

    Embeddings are generated deterministically based on input content,
    making tests reproducible without requiring actual model weights.
    """

    def __init__(self, embedding_dim: int = 1024, device: str = "cpu") -> None:
        self._embedding_dim = embedding_dim
        self._device = torch.device(device)

    @property
    def embedding_dim(self) -> int:
        """Dimension of the embedding space."""
        return self._embedding_dim

    @property
    def device(self) -> torch.device:
        """Device the encoder is on."""
        return self._device

    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text to a deterministic embedding based on the text hash.

        Args:
            text: Input text string

        Returns:
            Normalized embedding tensor of shape [1, embedding_dim]
        """
        seed = hash(text) % (2**32)
        generator = torch.Generator().manual_seed(seed)
        embedding = torch.randn(1, self._embedding_dim, generator=generator)
        return F.normalize(embedding, dim=-1).to(self._device)

    def encode_image(self, image: Path | Image.Image | torch.Tensor) -> torch.Tensor:
        """
        Encode image to a deterministic embedding.

        For Path inputs, uses the path string as seed.
        For Image inputs, uses a hash of the image size.
        For Tensor inputs, uses a hash of the tensor shape and sum.

        Args:
            image: Input image as Path, PIL Image, or tensor

        Returns:
            Normalized embedding tensor of shape [1, embedding_dim]
        """
        if isinstance(image, Path):
            seed = hash(str(image)) % (2**32)
        elif isinstance(image, Image.Image):
            seed = hash((image.size, image.mode)) % (2**32)
        else:
            seed = hash((image.shape, float(image.sum()))) % (2**32)

        generator = torch.Generator().manual_seed(seed)
        embedding = torch.randn(1, self._embedding_dim, generator=generator)
        return F.normalize(embedding, dim=-1).to(self._device)

    def encode_audio(
        self,
        audio: Path | torch.Tensor,
        start: float = 0.0,
        duration: float = 2.0,
    ) -> torch.Tensor:
        """
        Encode audio to a deterministic embedding.

        Args:
            audio: Input audio as Path or tensor
            start: Start time in seconds (used in seed generation)
            duration: Duration in seconds (used in seed generation)

        Returns:
            Normalized embedding tensor of shape [1, embedding_dim]
        """
        if isinstance(audio, Path):
            seed = hash((str(audio), start, duration)) % (2**32)
        else:
            seed = hash((audio.shape, start, duration)) % (2**32)

        generator = torch.Generator().manual_seed(seed)
        embedding = torch.randn(1, self._embedding_dim, generator=generator)
        return F.normalize(embedding, dim=-1).to(self._device)

    def encode_video(
        self,
        video: Path | torch.Tensor,
        timestamp: float = 0.0,
    ) -> torch.Tensor:
        """
        Encode video frame to a deterministic embedding.

        Args:
            video: Input video as Path or tensor
            timestamp: Timestamp in seconds (used in seed generation)

        Returns:
            Normalized embedding tensor of shape [1, embedding_dim]
        """
        if isinstance(video, Path):
            seed = hash((str(video), timestamp)) % (2**32)
        else:
            seed = hash((video.shape, timestamp)) % (2**32)

        generator = torch.Generator().manual_seed(seed)
        embedding = torch.randn(1, self._embedding_dim, generator=generator)
        return F.normalize(embedding, dim=-1).to(self._device)


class MockGenerator:
    """
    Mock generator that returns simple decoded tensors for testing.

    Generates deterministic outputs based on latent values,
    useful for testing optimization pipelines without real models.
    """

    def __init__(
        self,
        latent_shape: tuple[int, ...] = (1, 4, 64, 64),
        output_modality: str = "image",
        output_shape: tuple[int, ...] = (1, 3, 512, 512),
        device: str = "cpu",
    ) -> None:
        self._latent_shape = latent_shape
        self._output_modality = output_modality
        self._output_shape = output_shape
        self._device = torch.device(device)

    @property
    def latent_shape(self) -> tuple[int, ...]:
        """Shape of the latent tensor this generator accepts."""
        return self._latent_shape

    @property
    def output_modality(self) -> str:
        """Output modality: 'image', 'audio', or 'video'."""
        return self._output_modality

    @property
    def device(self) -> torch.device:
        """Device the generator is on."""
        return self._device

    def init_latent(self, seed: int | None = None) -> torch.Tensor:
        """
        Initialize a random latent for optimization.

        Args:
            seed: Optional random seed for reproducibility

        Returns:
            Latent tensor with requires_grad=True
        """
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
            latent = torch.randn(self._latent_shape, generator=generator)
        else:
            latent = torch.randn(self._latent_shape)

        return latent.to(self._device).requires_grad_(True)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to a simple output tensor.

        The output is deterministically derived from the latent
        via interpolation to the output shape, making gradients flow through.

        Args:
            latent: Latent tensor of shape self.latent_shape

        Returns:
            Decoded output tensor of shape self.output_shape
        """
        output = F.interpolate(
            latent,
            size=self._output_shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        if output.shape[1] != self._output_shape[1]:
            output = output[:, : self._output_shape[1], :, :]
            if output.shape[1] < self._output_shape[1]:
                padding = torch.zeros(
                    output.shape[0],
                    self._output_shape[1] - output.shape[1],
                    *self._output_shape[2:],
                    device=output.device,
                    dtype=output.dtype,
                )
                output = torch.cat([output, padding], dim=1)

        return output.sigmoid()


@pytest.fixture
def mock_encoder() -> MockEncoder:
    """
    Fixture providing a mock encoder with deterministic embeddings.

    Returns a MockEncoder instance configured with 1024-dimensional
    embeddings on CPU device.

    Returns:
        MockEncoder instance
    """
    return MockEncoder(embedding_dim=1024, device="cpu")


@pytest.fixture
def mock_generator() -> MockGenerator:
    """
    Fixture providing a mock generator that returns simple decoded tensors.

    Returns a MockGenerator configured for image generation with
    standard latent and output shapes.

    Returns:
        MockGenerator instance
    """
    return MockGenerator(
        latent_shape=(1, 4, 64, 64),
        output_modality="image",
        output_shape=(1, 3, 512, 512),
        device="cpu",
    )


@pytest.fixture
def sample_embedding() -> torch.Tensor:
    """
    Fixture providing a normalized 1024-dimensional sample embedding.

    The embedding is deterministic (seeded) for reproducibility.

    Returns:
        Normalized tensor of shape [1, 1024]
    """
    generator = torch.Generator().manual_seed(42)
    embedding = torch.randn(1, 1024, generator=generator)
    return F.normalize(embedding, dim=-1)


@pytest.fixture
def device() -> torch.device:
    """
    Fixture providing the CPU device for testing.

    Tests should use CPU to ensure reproducibility and avoid
    GPU-specific issues during CI.

    Returns:
        torch.device for CPU
    """
    return torch.device("cpu")


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> TypingGenerator[Path, None, None]:
    """
    Fixture providing a temporary directory for test outputs.

    The directory is automatically cleaned up after the test.

    Args:
        tmp_path: pytest's built-in temporary path fixture

    Yields:
        Path to the temporary output directory
    """
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    yield output_dir


@pytest.fixture
def isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Fixture providing an isolated working directory for tests.

    Changes the current working directory to a fresh temp directory
    for the duration of the test. This ensures tests that depend on
    CWD (like config file loading) don't interfere with each other
    or pick up the project's actual config files.

    Args:
        tmp_path: pytest's built-in temporary path fixture
        monkeypatch: pytest's monkeypatch fixture for safe patching

    Returns:
        Path to the isolated working directory
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path
