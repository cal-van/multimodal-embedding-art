"""
Integration tests for the optimization loop.

These tests verify that the optimization engine works end-to-end
using mock encoder and generator implementations that return
deterministic outputs for testing purposes.
"""

from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from embedding_art.core.concept import Concept
from embedding_art.core.config import AugmentationConfig, OptimizationConfig
from embedding_art.core.engine import EmbeddingArtEngine, OptimizationResult
from embedding_art.regularizers import CompositeRegularizer, LatentNorm


class MockEncoder:
    """
    Mock encoder that returns deterministic embeddings.

    For images, the embedding is computed as the mean RGB values
    normalized to a fixed-size vector. This allows the optimization
    to meaningfully improve similarity by changing the generated image.
    """

    def __init__(self, embedding_dim: int = 128, device: str = "cpu"):
        self._embedding_dim = embedding_dim
        self._device = torch.device(device)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def device(self) -> torch.device:
        return self._device

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to a deterministic embedding based on character values."""
        embedding = torch.zeros(1, self._embedding_dim, device=self._device)
        for i, char in enumerate(text):
            idx = i % self._embedding_dim
            embedding[0, idx] += float(ord(char)) / 256.0
        return F.normalize(embedding, dim=-1)

    def encode_image(self, image: Path | Image.Image | torch.Tensor) -> torch.Tensor:
        """Encode image tensor to embedding based on spatial means."""
        if isinstance(image, torch.Tensor):
            return self._encode_image_tensor(image)
        else:
            raise NotImplementedError("Mock encoder only supports tensor images")

    def _encode_image_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image tensor [B, C, H, W] to embedding.

        Uses a grid-based approach: divides image into grid cells and
        computes mean values to create an embedding that changes
        meaningfully as the image content changes.
        """
        b, c, h, w = image.shape

        grid_size = int(self._embedding_dim**0.5)
        if grid_size * grid_size > self._embedding_dim:
            grid_size = grid_size - 1

        embedding = torch.zeros(b, self._embedding_dim, device=image.device)

        cell_h = h // grid_size
        cell_w = w // grid_size

        idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if idx >= self._embedding_dim:
                    break
                cell = image[:, :, i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w]
                embedding[:, idx] = cell.mean(dim=(1, 2, 3))
                idx += 1

        for channel in range(c):
            if idx >= self._embedding_dim:
                break
            embedding[:, idx] = image[:, channel].mean(dim=(1, 2))
            idx += 1

        return F.normalize(embedding, dim=-1)

    def encode_for_optimization(self, image: torch.Tensor) -> torch.Tensor:
        """Alias for encode_image for optimization loop compatibility."""
        return self._encode_image_tensor(image)

    def encode_audio(
        self,
        audio: Path | torch.Tensor,
        start: float = 0.0,
        duration: float = 2.0,
    ) -> torch.Tensor:
        """Encode audio - not implemented for integration tests."""
        raise NotImplementedError("Audio encoding not needed for integration tests")

    def encode_video(
        self,
        video: Path | torch.Tensor,
        timestamp: float = 0.0,
    ) -> torch.Tensor:
        """Encode video - not implemented for integration tests."""
        raise NotImplementedError("Video encoding not needed for integration tests")


class MockGenerator:
    """
    Mock generator that returns simple decoded tensors.

    Uses a learned linear transformation from latent space to image space,
    allowing gradients to flow through for optimization.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        output_size: int = 64,
        device: str = "cpu",
    ):
        self._latent_shape = (1, latent_dim)
        self._output_size = output_size
        self._device = torch.device(device)
        self._output_modality = "image"

        torch.manual_seed(42)
        self._weights = (
            torch.randn(latent_dim, 3 * output_size * output_size, device=self._device) * 0.01
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
        """Initialize a random latent for optimization."""
        if seed is not None:
            torch.manual_seed(seed)

        latent = torch.randn(*self._latent_shape, device=self._device, requires_grad=True)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image tensor [B, C, H, W].

        Uses a simple linear transformation followed by sigmoid
        to produce valid RGB values in [0, 1].
        """
        b = latent.shape[0]
        flat_output = torch.matmul(latent, self._weights)
        output = flat_output.view(b, 3, self._output_size, self._output_size)
        output = torch.sigmoid(output)
        return output


def create_test_engine(
    embedding_dim: int = 128,
    latent_dim: int = 64,
    output_size: int = 64,
    device: str = "cpu",
) -> tuple[EmbeddingArtEngine, MockEncoder, MockGenerator]:
    """Create a test engine with mock encoder and generator."""
    encoder = MockEncoder(embedding_dim=embedding_dim, device=device)
    generator = MockGenerator(latent_dim=latent_dim, output_size=output_size, device=device)

    engine = EmbeddingArtEngine(encoder=encoder, device=device)
    engine.register_generator("image", generator)

    return engine, encoder, generator


def create_test_concept(encoder: MockEncoder, text: str = "test concept") -> Concept:
    """Create a test concept from text."""
    return Concept.from_text(text, encoder)


class TestOptimizationLoopRunsAndImproves:
    """Tests verifying optimization loop runs and improves similarity."""

    @pytest.mark.slow
    def test_optimization_loop_runs_successfully(self):
        """Should complete optimization loop without errors."""
        engine, encoder, generator = create_test_engine()
        target = create_test_concept(encoder)

        config = OptimizationConfig(
            steps=50,
            learning_rate=0.1,
            checkpoint_every=10,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        result = engine.optimize(
            target=target,
            output_modality="image",
            config=config,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
        )

        assert isinstance(result, OptimizationResult)
        assert result.final_latent is not None
        assert result.final_embedding is not None
        assert result.target_embedding is not None

    @pytest.mark.slow
    def test_similarity_improves_during_optimization(self):
        """Should improve similarity from start to finish."""
        engine, encoder, generator = create_test_engine()
        target = create_test_concept(encoder, "distinct concept")

        config = OptimizationConfig(
            steps=100,
            learning_rate=0.5,
            checkpoint_every=0,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        result = engine.optimize(
            target=target,
            output_modality="image",
            config=config,
            regularizers=CompositeRegularizer([LatentNorm(weight=0.01)]),
            progress=False,
        )

        initial_similarity = result.similarity_history[0]
        final_similarity = result.final_similarity

        assert final_similarity > initial_similarity, (
            f"Similarity should improve: {initial_similarity:.4f} -> {final_similarity:.4f}"
        )

    @pytest.mark.slow
    def test_loss_decreases_during_optimization(self):
        """Should decrease loss during optimization."""
        engine, encoder, generator = create_test_engine()
        target = create_test_concept(encoder)

        config = OptimizationConfig(
            steps=100,
            learning_rate=0.5,
            checkpoint_every=0,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        result = engine.optimize(
            target=target,
            output_modality="image",
            config=config,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
        )

        initial_loss = result.loss_history[0]
        final_loss = result.loss_history[-1]

        assert final_loss < initial_loss, (
            f"Loss should decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    def test_callback_is_invoked_each_step(self):
        """Should invoke callback function at each optimization step."""
        engine, encoder, generator = create_test_engine()
        target = create_test_concept(encoder)

        callback_calls: list[tuple[int, float, float, Any]] = []

        def track_callback(step: int, loss: float, similarity: float, latent: torch.Tensor) -> None:
            callback_calls.append((step, loss, similarity, latent.clone()))

        config = OptimizationConfig(
            steps=10,
            learning_rate=0.1,
            checkpoint_every=0,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        engine.optimize(
            target=target,
            output_modality="image",
            config=config,
            callback=track_callback,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
        )

        assert len(callback_calls) == 10
        for i, (step, _, _, _) in enumerate(callback_calls):
            assert step == i


class TestCheckpointsSaved:
    """Tests verifying checkpoints are saved correctly."""

    @pytest.mark.slow
    def test_checkpoints_are_saved_at_configured_intervals(self):
        """Should save checkpoints at configured step intervals."""
        engine, encoder, generator = create_test_engine()
        target = create_test_concept(encoder)

        config = OptimizationConfig(
            steps=100,
            learning_rate=0.1,
            checkpoint_every=25,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        result = engine.optimize(
            target=target,
            output_modality="image",
            config=config,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
        )

        checkpoint_steps = [step for step, _ in result.checkpoints]
        assert checkpoint_steps == [0, 25, 50, 75]

    def test_checkpoints_contain_valid_latent_tensors(self):
        """Should store valid latent tensors in checkpoints."""
        engine, encoder, generator = create_test_engine()
        target = create_test_concept(encoder)

        config = OptimizationConfig(
            steps=30,
            learning_rate=0.1,
            checkpoint_every=10,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        result = engine.optimize(
            target=target,
            output_modality="image",
            config=config,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
        )

        assert len(result.checkpoints) == 3  # steps 0, 10, 20

        for step, latent in result.checkpoints:
            assert isinstance(latent, torch.Tensor)
            assert latent.shape == generator.latent_shape
            assert not latent.requires_grad

    def test_no_checkpoints_when_checkpoint_every_is_zero(self):
        """Should not save checkpoints when checkpoint_every is 0."""
        engine, encoder, generator = create_test_engine()
        target = create_test_concept(encoder)

        config = OptimizationConfig(
            steps=50,
            learning_rate=0.1,
            checkpoint_every=0,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        result = engine.optimize(
            target=target,
            output_modality="image",
            config=config,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
        )

        assert len(result.checkpoints) == 0


class TestInterpolationSeries:
    """Tests for interpolation_series functionality."""

    @pytest.mark.slow
    def test_interpolation_series_generates_correct_number_of_results(self):
        """Should generate the specified number of interpolation steps."""
        engine, encoder, _ = create_test_engine()

        concept_a = create_test_concept(encoder, "cat")
        concept_b = create_test_concept(encoder, "dog")

        config = OptimizationConfig(
            steps=20,
            learning_rate=0.1,
            checkpoint_every=0,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        results = engine.interpolation_series(
            concept_a=concept_a,
            concept_b=concept_b,
            output_modality="image",
            steps=5,
            config=config,
            progress=False,
        )

        assert len(results) == 5
        assert all(isinstance(r, OptimizationResult) for r in results)

    @pytest.mark.slow
    def test_interpolation_endpoints_match_original_concepts(self):
        """Should optimize toward original concepts at endpoints."""
        engine, encoder, _ = create_test_engine()

        concept_a = create_test_concept(encoder, "alpha")
        concept_b = create_test_concept(encoder, "omega")

        config = OptimizationConfig(
            steps=30,
            learning_rate=0.1,
            checkpoint_every=0,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        results = engine.interpolation_series(
            concept_a=concept_a,
            concept_b=concept_b,
            output_modality="image",
            steps=3,
            config=config,
            progress=False,
        )

        first_target = results[0].target_embedding
        last_target = results[-1].target_embedding

        sim_first_to_a = F.cosine_similarity(first_target, concept_a.embedding, dim=-1).item()
        sim_last_to_b = F.cosine_similarity(last_target, concept_b.embedding, dim=-1).item()

        assert sim_first_to_a > 0.99, f"First result should target concept_a: {sim_first_to_a}"
        assert sim_last_to_b > 0.99, f"Last result should target concept_b: {sim_last_to_b}"

    def test_interpolation_with_single_step_returns_midpoint(self):
        """Should return midpoint when steps=1."""
        engine, encoder, _ = create_test_engine()

        concept_a = create_test_concept(encoder, "start")
        concept_b = create_test_concept(encoder, "end")

        config = OptimizationConfig(
            steps=10,
            learning_rate=0.1,
            checkpoint_every=0,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        results = engine.interpolation_series(
            concept_a=concept_a,
            concept_b=concept_b,
            output_modality="image",
            steps=1,
            config=config,
            progress=False,
        )

        assert len(results) == 1


class TestOptimizationResultMethods:
    """Tests for OptimizationResult helper methods."""

    def test_get_final_image_returns_pil_image(self):
        """Should decode final latent to PIL Image."""
        engine, encoder, generator = create_test_engine(output_size=32)
        target = create_test_concept(encoder)

        config = OptimizationConfig(
            steps=5,
            learning_rate=0.1,
            checkpoint_every=0,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        result = engine.optimize(
            target=target,
            output_modality="image",
            config=config,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
        )

        image = result.get_final_image(generator)

        assert isinstance(image, Image.Image)
        assert image.size == (32, 32)
        assert image.mode == "RGB"


class TestDifferentOptimizers:
    """Tests for different optimizer configurations."""

    @pytest.mark.parametrize("optimizer_name", ["adam", "adamw", "sgd"])
    def test_optimizer_types_work(self, optimizer_name: str):
        """Should work with different optimizer types."""
        engine, encoder, _ = create_test_engine()
        target = create_test_concept(encoder)

        config = OptimizationConfig(
            steps=10,
            learning_rate=0.01,
            optimizer=optimizer_name,
            checkpoint_every=0,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        result = engine.optimize(
            target=target,
            output_modality="image",
            config=config,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
        )

        assert result.final_latent is not None


class TestDifferentSchedulers:
    """Tests for different scheduler configurations."""

    @pytest.mark.parametrize("scheduler_name", ["constant", "cosine", "linear"])
    def test_scheduler_types_work(self, scheduler_name: str):
        """Should work with different scheduler types."""
        engine, encoder, _ = create_test_engine()
        target = create_test_concept(encoder)

        config = OptimizationConfig(
            steps=10,
            learning_rate=0.1,
            scheduler=scheduler_name,
            checkpoint_every=0,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        result = engine.optimize(
            target=target,
            output_modality="image",
            config=config,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
        )

        assert result.final_latent is not None


class TestReproducibility:
    """Tests for reproducibility with seeds."""

    def test_same_seed_produces_same_results(self):
        """Should produce identical results with same seed."""
        engine, encoder, _ = create_test_engine()
        target = create_test_concept(encoder)

        config = OptimizationConfig(
            steps=20,
            learning_rate=0.1,
            seed=12345,
            checkpoint_every=0,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        result1 = engine.optimize(
            target=target,
            output_modality="image",
            config=config,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
        )

        engine2, encoder2, _ = create_test_engine()
        target2 = create_test_concept(encoder2)

        result2 = engine2.optimize(
            target=target2,
            output_modality="image",
            config=config,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
        )

        assert torch.allclose(result1.final_latent, result2.final_latent, atol=1e-5)

    def test_different_seeds_produce_different_results(self):
        """Should produce different results with different seeds."""
        engine, encoder, _ = create_test_engine()
        target = create_test_concept(encoder)

        config1 = OptimizationConfig(
            steps=20,
            learning_rate=0.1,
            seed=11111,
            checkpoint_every=0,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        config2 = OptimizationConfig(
            steps=20,
            learning_rate=0.1,
            seed=22222,
            checkpoint_every=0,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        result1 = engine.optimize(
            target=target,
            output_modality="image",
            config=config1,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
        )

        result2 = engine.optimize(
            target=target,
            output_modality="image",
            config=config2,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
        )

        assert not torch.allclose(result1.final_latent, result2.final_latent)
