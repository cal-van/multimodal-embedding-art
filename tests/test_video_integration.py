from unittest.mock import patch

import torch
import torch.nn.functional as F

from embedding_art.core.concept import Concept
from embedding_art.core.config import AugmentationConfig, OptimizationConfig
from embedding_art.core.engine import EmbeddingArtEngine, OptimizationResult
from embedding_art.regularizers import CompositeRegularizer


class MockVideoEncoder:
    def __init__(self, embedding_dim: int = 32, device: str = "cpu") -> None:
        self._embedding_dim = embedding_dim
        self._device = torch.device(device)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def device(self) -> torch.device:
        return self._device

    def encode_video(self, video: torch.Tensor, timestamp: float = 0.0) -> torch.Tensor:
        if not isinstance(video, torch.Tensor):
            raise TypeError("MockVideoEncoder expects a tensor input")

        if video.ndim != 5:
            raise ValueError("Video tensor must have shape [B, F, C, H, W]")

        mean_val = video.mean(dim=(1, 2, 3, 4))
        embedding = mean_val[:, None].repeat(1, self._embedding_dim)
        return F.normalize(embedding, dim=-1)


class MockVideoGenerator:
    def __init__(
        self,
        latent_dim: int = 8,
        frames: int = 4,
        size: int = 8,
        device: str = "cpu",
    ) -> None:
        self._latent_shape = (1, latent_dim)
        self._frames = frames
        self._size = size
        self._device = torch.device(device)
        self._output_modality = "video"

        torch.manual_seed(42)
        self._weights = (
            torch.randn(latent_dim, frames * 3 * size * size, device=self._device) * 0.01
        )

    @property
    def latent_shape(self) -> tuple[int, ...]:
        return self._latent_shape

    @property
    def output_modality(self) -> str:
        return self._output_modality

    @property
    def device(self) -> torch.device:
        return self._device

    def init_latent(self, seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
        return torch.randn(*self._latent_shape, device=self._device, requires_grad=True)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        batch = latent.shape[0]
        flat = torch.matmul(latent, self._weights)
        output = flat.view(batch, self._frames, 3, self._size, self._size)
        return torch.sigmoid(output)


def _create_video_engine(device: str = "cpu") -> tuple[EmbeddingArtEngine, MockVideoGenerator]:
    encoder = MockVideoEncoder(device=device)
    generator = MockVideoGenerator(device=device)

    engine = EmbeddingArtEngine(encoder=encoder, device=device)
    engine.register_generator("video", generator)

    return engine, generator


def test_video_optimization_uses_default_video_regularizer() -> None:
    engine, _ = _create_video_engine()
    target = Concept(embedding=torch.randn(1, 32), description="test")

    config = OptimizationConfig(
        steps=1,
        learning_rate=0.1,
        checkpoint_every=0,
        augmentation=AugmentationConfig(random_crop=False, random_flip=False),
    )

    with (
        patch("embedding_art.core.engine.CompositeRegularizer.default_video") as mock_default_video,
        patch("embedding_art.core.engine.CompositeRegularizer.default_image") as mock_default_image,
    ):
        mock_default_video.return_value = CompositeRegularizer.minimal()
        mock_default_image.return_value = CompositeRegularizer.minimal()

        engine.optimize(
            target=target,
            output_modality="video",
            config=config,
            regularizers=None,
            progress=False,
        )

        mock_default_video.assert_called_once()
        mock_default_image.assert_not_called()


def test_video_optimization_loop_runs() -> None:
    engine, generator = _create_video_engine()
    target = Concept(embedding=torch.randn(1, 32), description="test")

    config = OptimizationConfig(
        steps=2,
        learning_rate=0.1,
        checkpoint_every=0,
        augmentation=AugmentationConfig(random_crop=False, random_flip=False),
    )

    result = engine.optimize(
        target=target,
        output_modality="video",
        config=config,
        regularizers=CompositeRegularizer.minimal(),
        progress=False,
    )

    assert isinstance(result, OptimizationResult)
    assert len(result.loss_history) == 2
    assert result.final_latent.shape == generator.latent_shape
    assert -1.0 <= result.final_similarity <= 1.0


def test_video_optimization_loop_runs_with_default_augmentation() -> None:
    engine, generator = _create_video_engine()
    target = Concept(embedding=torch.randn(1, 32), description="test")

    config = OptimizationConfig(
        steps=1,
        learning_rate=0.1,
        checkpoint_every=0,
    )

    result = engine.optimize(
        target=target,
        output_modality="video",
        config=config,
        regularizers=CompositeRegularizer.minimal(),
        progress=False,
    )

    assert result.final_latent.shape == generator.latent_shape
