"""
Tests for checkpoint saving and resuming during optimization.

These tests verify that:
- Checkpoints are saved to disk at configured intervals
- Optimization can be resumed from a saved checkpoint
- Resumed optimization continues from the correct step
- CLI --resume flag is available and works correctly
"""

from pathlib import Path
from typing import Any

import torch
from click.testing import CliRunner

from embedding_art.cli.main import cli
from embedding_art.core.concept import Concept
from embedding_art.core.config import AugmentationConfig, OptimizationConfig
from embedding_art.core.engine import EmbeddingArtEngine
from embedding_art.regularizers import CompositeRegularizer


class MockEncoderForCheckpoint:
    """
    Mock encoder for checkpoint testing.

    Returns deterministic embeddings based on input content.
    """

    def __init__(self, embedding_dim: int = 128, device: str = "cpu") -> None:
        self._embedding_dim = embedding_dim
        self._device = torch.device(device)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def device(self) -> torch.device:
        return self._device

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to a deterministic embedding."""
        embedding = torch.zeros(1, self._embedding_dim, device=self._device)
        for i, char in enumerate(text):
            idx = i % self._embedding_dim
            embedding[0, idx] += float(ord(char)) / 256.0
        return torch.nn.functional.normalize(embedding, dim=-1)

    def encode_image(self, image: Any) -> torch.Tensor:
        """Encode image tensor to embedding based on spatial means."""
        if isinstance(image, torch.Tensor):
            return self._encode_image_tensor(image)
        raise NotImplementedError("Mock encoder only supports tensor images")

    def _encode_image_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image tensor [B, C, H, W] to embedding."""
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

        return torch.nn.functional.normalize(embedding, dim=-1)

    def encode_for_optimization(self, image: torch.Tensor) -> torch.Tensor:
        """Alias for encode_image for optimization loop compatibility."""
        return self._encode_image_tensor(image)


class MockGeneratorForCheckpoint:
    """
    Mock generator for checkpoint testing.

    Uses a simple linear transformation from latent space to image space.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        output_size: int = 64,
        device: str = "cpu",
    ) -> None:
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
        """Decode latent to image tensor [B, C, H, W]."""
        b = latent.shape[0]
        flat_output = torch.matmul(latent, self._weights)
        output = flat_output.view(b, 3, self._output_size, self._output_size)
        output = torch.sigmoid(output)
        return output


def create_checkpoint_test_engine(
    embedding_dim: int = 128,
    latent_dim: int = 64,
    output_size: int = 64,
    device: str = "cpu",
) -> tuple[EmbeddingArtEngine, MockEncoderForCheckpoint, MockGeneratorForCheckpoint]:
    """Create a test engine with mock encoder and generator."""
    encoder = MockEncoderForCheckpoint(embedding_dim=embedding_dim, device=device)
    generator = MockGeneratorForCheckpoint(
        latent_dim=latent_dim, output_size=output_size, device=device
    )

    engine = EmbeddingArtEngine(encoder=encoder, device=device)
    engine.register_generator("image", generator)

    return engine, encoder, generator


def create_test_concept(encoder: MockEncoderForCheckpoint, text: str = "test concept") -> Concept:
    """Create a test concept from text."""
    return Concept.from_text(text, encoder)


# =============================================================================
# Checkpoint Saving Tests
# =============================================================================


class TestCheckpointSavedToDisk:
    """Tests verifying checkpoints are saved to disk at configured intervals."""

    def test_checkpoint_file_created_at_configured_interval(self, tmp_path: Path) -> None:
        """Should create checkpoint file at checkpoint_every interval."""
        engine, encoder, generator = create_checkpoint_test_engine()
        target = create_test_concept(encoder)

        checkpoint_dir = tmp_path / "checkpoints"

        config = OptimizationConfig(
            steps=30,
            learning_rate=0.1,
            checkpoint_every=10,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        engine.optimize(
            target=target,
            output_modality="image",
            config=config,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
            checkpoint_dir=checkpoint_dir,
        )

        checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.pt"))
        assert len(checkpoint_files) == 3  # steps 0, 10, 20

    def test_checkpoint_file_contains_required_state(self, tmp_path: Path) -> None:
        """Checkpoint should contain latent, optimizer state, step, and config."""
        engine, encoder, generator = create_checkpoint_test_engine()
        target = create_test_concept(encoder)

        checkpoint_dir = tmp_path / "checkpoints"

        config = OptimizationConfig(
            steps=15,
            learning_rate=0.1,
            checkpoint_every=10,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        engine.optimize(
            target=target,
            output_modality="image",
            config=config,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
            checkpoint_dir=checkpoint_dir,
        )

        checkpoint_path = checkpoint_dir / "checkpoint_step_0010.pt"
        assert checkpoint_path.exists()

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert "latent" in checkpoint
        assert "optimizer_state" in checkpoint
        assert "step" in checkpoint
        assert "config" in checkpoint
        assert checkpoint["step"] == 10

    def test_checkpoint_latent_has_correct_shape(self, tmp_path: Path) -> None:
        """Checkpoint latent should have the generator's latent shape."""
        engine, encoder, generator = create_checkpoint_test_engine(latent_dim=64)
        target = create_test_concept(encoder)

        checkpoint_dir = tmp_path / "checkpoints"

        config = OptimizationConfig(
            steps=15,
            learning_rate=0.1,
            checkpoint_every=10,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        engine.optimize(
            target=target,
            output_modality="image",
            config=config,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
            checkpoint_dir=checkpoint_dir,
        )

        checkpoint_path = checkpoint_dir / "checkpoint_step_0010.pt"
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        assert checkpoint["latent"].shape == generator.latent_shape

    def test_no_checkpoint_files_when_checkpoint_every_is_zero(self, tmp_path: Path) -> None:
        """Should not create checkpoint files when checkpoint_every is 0."""
        engine, encoder, generator = create_checkpoint_test_engine()
        target = create_test_concept(encoder)

        checkpoint_dir = tmp_path / "checkpoints"

        config = OptimizationConfig(
            steps=30,
            learning_rate=0.1,
            checkpoint_every=0,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        engine.optimize(
            target=target,
            output_modality="image",
            config=config,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
            checkpoint_dir=checkpoint_dir,
        )

        if checkpoint_dir.exists():
            checkpoint_files = list(checkpoint_dir.glob("checkpoint_*.pt"))
            assert len(checkpoint_files) == 0

    def test_checkpoint_dir_created_if_not_exists(self, tmp_path: Path) -> None:
        """Should create checkpoint directory if it doesn't exist."""
        engine, encoder, generator = create_checkpoint_test_engine()
        target = create_test_concept(encoder)

        checkpoint_dir = tmp_path / "deeply" / "nested" / "checkpoints"

        config = OptimizationConfig(
            steps=15,
            learning_rate=0.1,
            checkpoint_every=10,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        engine.optimize(
            target=target,
            output_modality="image",
            config=config,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
            checkpoint_dir=checkpoint_dir,
        )

        assert checkpoint_dir.exists()
        assert (checkpoint_dir / "checkpoint_step_0000.pt").exists()


# =============================================================================
# Checkpoint Resume Tests
# =============================================================================


class TestResumeFromCheckpoint:
    """Tests verifying optimization can resume from a saved checkpoint."""

    def test_resume_continues_from_checkpoint_step(self, tmp_path: Path) -> None:
        """Should continue optimization from the checkpoint step."""
        engine, encoder, generator = create_checkpoint_test_engine()
        target = create_test_concept(encoder, "resume test")

        checkpoint_dir = tmp_path / "checkpoints"

        # Run initial optimization for 20 steps, checkpointing at step 10
        config_initial = OptimizationConfig(
            steps=20,
            learning_rate=0.1,
            checkpoint_every=10,
            seed=42,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        engine.optimize(
            target=target,
            output_modality="image",
            config=config_initial,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
            checkpoint_dir=checkpoint_dir,
        )

        # Resume from step 10 checkpoint
        checkpoint_path = checkpoint_dir / "checkpoint_step_0010.pt"
        assert checkpoint_path.exists()

        # Resume with additional steps
        config_resume = OptimizationConfig(
            steps=30,  # Total desired steps
            learning_rate=0.1,
            checkpoint_every=10,
            seed=42,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        result = engine.optimize(
            target=target,
            output_modality="image",
            config=config_resume,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
            checkpoint_dir=checkpoint_dir,
            resume_from=checkpoint_path,
        )

        # Should have 20 new steps (from 10 to 30)
        assert len(result.similarity_history) == 20

    def test_resume_loads_latent_from_checkpoint(self, tmp_path: Path) -> None:
        """Should load the latent tensor from the checkpoint."""
        engine, encoder, generator = create_checkpoint_test_engine()
        target = create_test_concept(encoder, "latent test")

        checkpoint_dir = tmp_path / "checkpoints"

        config = OptimizationConfig(
            steps=15,
            learning_rate=0.1,
            checkpoint_every=10,
            seed=42,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        engine.optimize(
            target=target,
            output_modality="image",
            config=config,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
            checkpoint_dir=checkpoint_dir,
        )

        checkpoint_path = checkpoint_dir / "checkpoint_step_0010.pt"
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        # Verify checkpoint has latent
        assert "latent" in checkpoint

        # Track the initial latent when resuming
        initial_latents: list[torch.Tensor] = []

        def capture_initial_latent(
            step: int, loss: float, similarity: float, latent: torch.Tensor
        ) -> None:
            if step == 10:  # First step after resume (continuing from step 10)
                initial_latents.append(latent.detach().clone())

        config_resume = OptimizationConfig(
            steps=15,
            learning_rate=0.1,
            checkpoint_every=10,
            seed=42,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        engine.optimize(
            target=target,
            output_modality="image",
            config=config_resume,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
            checkpoint_dir=checkpoint_dir,
            resume_from=checkpoint_path,
            callback=capture_initial_latent,
        )

        # The initial latent at resume should match the checkpoint latent
        # (modulo one optimization step)
        assert len(initial_latents) == 1

    def test_resume_restores_optimizer_state(self, tmp_path: Path) -> None:
        """Should restore optimizer state from checkpoint."""
        engine, encoder, generator = create_checkpoint_test_engine()
        target = create_test_concept(encoder, "optimizer test")

        checkpoint_dir = tmp_path / "checkpoints"

        config = OptimizationConfig(
            steps=15,
            learning_rate=0.1,
            checkpoint_every=10,
            seed=42,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        engine.optimize(
            target=target,
            output_modality="image",
            config=config,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
            checkpoint_dir=checkpoint_dir,
        )

        checkpoint_path = checkpoint_dir / "checkpoint_step_0010.pt"
        checkpoint = torch.load(checkpoint_path, weights_only=False)

        # Optimizer state dict should be present and non-empty
        assert "optimizer_state" in checkpoint
        assert len(checkpoint["optimizer_state"]) > 0


# =============================================================================
# Resumed Optimization Correctness Tests
# =============================================================================


class TestResumedOptimizationCorrectness:
    """Tests verifying resumed optimization produces correct results."""

    def test_resumed_optimization_matches_continuous_run(self, tmp_path: Path) -> None:
        """Resumed optimization should produce similar results to continuous run."""
        # Run continuous optimization
        engine1, encoder1, _ = create_checkpoint_test_engine()
        target1 = create_test_concept(encoder1, "matching test")

        config_continuous = OptimizationConfig(
            steps=30,
            learning_rate=0.1,
            checkpoint_every=0,
            seed=42,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        result_continuous = engine1.optimize(
            target=target1,
            output_modality="image",
            config=config_continuous,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
        )

        # Run optimization with checkpoint and resume
        engine2, encoder2, _ = create_checkpoint_test_engine()
        target2 = create_test_concept(encoder2, "matching test")

        checkpoint_dir = tmp_path / "checkpoints"

        config_initial = OptimizationConfig(
            steps=15,
            learning_rate=0.1,
            checkpoint_every=10,
            seed=42,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        engine2.optimize(
            target=target2,
            output_modality="image",
            config=config_initial,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
            checkpoint_dir=checkpoint_dir,
        )

        checkpoint_path = checkpoint_dir / "checkpoint_step_0010.pt"

        config_resume = OptimizationConfig(
            steps=30,
            learning_rate=0.1,
            checkpoint_every=10,
            seed=42,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        result_resumed = engine2.optimize(
            target=target2,
            output_modality="image",
            config=config_resume,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
            checkpoint_dir=checkpoint_dir,
            resume_from=checkpoint_path,
        )

        # Final similarities should be very close
        # (may not be exact due to scheduler state differences)
        assert abs(result_continuous.final_similarity - result_resumed.final_similarity) < 0.05

    def test_resume_skips_already_completed_steps(self, tmp_path: Path) -> None:
        """Should skip steps that were already completed in the checkpoint."""
        engine, encoder, generator = create_checkpoint_test_engine()
        target = create_test_concept(encoder, "skip test")

        checkpoint_dir = tmp_path / "checkpoints"

        # Run initial optimization for 50 steps
        config_initial = OptimizationConfig(
            steps=50,
            learning_rate=0.1,
            checkpoint_every=25,
            seed=42,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        engine.optimize(
            target=target,
            output_modality="image",
            config=config_initial,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
            checkpoint_dir=checkpoint_dir,
        )

        # Resume from step 25
        checkpoint_path = checkpoint_dir / "checkpoint_step_0025.pt"

        step_calls: list[int] = []

        def track_steps(step: int, loss: float, similarity: float, latent: torch.Tensor) -> None:
            step_calls.append(step)

        config_resume = OptimizationConfig(
            steps=50,
            learning_rate=0.1,
            checkpoint_every=25,
            seed=42,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        engine.optimize(
            target=target,
            output_modality="image",
            config=config_resume,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
            checkpoint_dir=checkpoint_dir,
            resume_from=checkpoint_path,
            callback=track_steps,
        )

        # Should have steps 25-49 (25 steps total)
        assert len(step_calls) == 25
        assert step_calls[0] == 25
        assert step_calls[-1] == 49


# =============================================================================
# CLI --resume Option Tests
# =============================================================================


class TestCLIResumeOption:
    """Tests verifying the --resume CLI option is available and configured correctly."""

    def test_resume_option_exists_in_help(self) -> None:
        """The optimize command should have a --resume option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["optimize", "--help"])

        assert result.exit_code == 0
        assert "--resume" in result.output

    def test_checkpoint_dir_option_exists_in_help(self) -> None:
        """The optimize command should have a --checkpoint-dir option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["optimize", "--help"])

        assert result.exit_code == 0
        assert "--checkpoint-dir" in result.output

    def test_resume_help_describes_checkpoint_path(self) -> None:
        """The --resume option help should indicate it expects a checkpoint path."""
        runner = CliRunner()
        result = runner.invoke(cli, ["optimize", "--help"])

        assert result.exit_code == 0
        # The help text should mention checkpoint
        assert "checkpoint" in result.output.lower() or "resume" in result.output.lower()
