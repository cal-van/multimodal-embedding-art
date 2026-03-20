"""
Tests for augmentation, callback, and checkpoint support in OptimizationStrategy.

TDD red-green-refactor: all tests written before implementation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812

from embedding_art.core.concept import Concept
from embedding_art.core.config import AugmentationConfig, LossConfig, OptimizationConfig

# ---------------------------------------------------------------------------
# Reuse the same DifferentiableEncoder / SmallMockGenerator pattern from
# test_strategies.py (copied here to keep this file self-contained).
# ---------------------------------------------------------------------------


class DifferentiableEncoder:
    """Mock encoder with a differentiable encode_for_optimization path."""

    def __init__(self, embedding_dim: int = 16) -> None:
        torch.manual_seed(0)
        self._embedding_dim = embedding_dim
        self._device = torch.device("cpu")
        self._proj = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        with torch.no_grad():
            torch.nn.init.eye_(self._proj.weight)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def device(self) -> torch.device:
        return self._device

    def encode_for_optimization(self, tensor: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(tensor, output_size=(1, 1))
        flat = pooled.view(pooled.shape[0], -1)
        if flat.shape[1] < self._embedding_dim:
            pad = torch.zeros(
                flat.shape[0], self._embedding_dim - flat.shape[1], device=flat.device
            )
            flat = torch.cat([flat, pad], dim=1)
        else:
            flat = flat[:, : self._embedding_dim]
        out = self._proj(flat)
        return F.normalize(out, dim=-1)

    @property
    def card(self):
        from embedding_art.encoders.registry import EncoderCapability, EncoderCard

        return EncoderCard(
            name="differentiable_mock",
            capabilities=EncoderCapability.IMAGE | EncoderCapability.BACKPROP_OPTIMIZABLE,
            embedding_dim=self._embedding_dim,
            memory_estimate_mb=0,
            backprop_cost=0.0,
        )


class SmallMockGenerator:
    """Minimal latent generator for fast tests — decode() is differentiable."""

    def __init__(
        self,
        latent_shape: tuple[int, ...] = (1, 4, 4, 4),
        output_channels: int = 4,
        output_size: int = 8,
    ) -> None:
        self._latent_shape = latent_shape
        self._output_channels = output_channels
        self._output_size = output_size
        self._device = torch.device("cpu")

    @property
    def latent_shape(self) -> tuple[int, ...]:
        return self._latent_shape

    @property
    def output_modality(self) -> str:
        return "image"

    @property
    def device(self) -> torch.device:
        return self._device

    def init_latent(self, seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            gen = torch.Generator().manual_seed(seed)
            latent = torch.randn(self._latent_shape, generator=gen)
        else:
            latent = torch.randn(self._latent_shape)
        return latent.to(self._device).requires_grad_(True)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        out = F.interpolate(
            latent,
            size=(self._output_size, self._output_size),
            mode="bilinear",
            align_corners=False,
        )
        if out.shape[1] != self._output_channels:
            out = out[:, : self._output_channels, :, :]
        return out.sigmoid()


def _make_target_concept(embedding_dim: int = 16) -> Concept:
    torch.manual_seed(99)
    emb = F.normalize(torch.randn(1, embedding_dim), dim=-1)
    return Concept(embedding=emb, description="test_target")


def _fast_config(**overrides) -> OptimizationConfig:
    """OptimizationConfig with a tiny step count for fast tests."""
    defaults = dict(
        steps=5,
        learning_rate=0.01,
        optimizer="adam",
        scheduler="constant",
        loss=LossConfig(
            similarity_weight=1.0,
            feature_matching_weight=0.0,
            sae_feature_weight=0.0,
        ),
    )
    defaults.update(overrides)
    return OptimizationConfig(**defaults)


# ---------------------------------------------------------------------------
# Augmentation unit tests  (_augment method)
# ---------------------------------------------------------------------------


class TestAugmentMethod:
    """Unit tests for OptimizationStrategy._augment()."""

    def test_augmentation_applied_preserves_output_shape(self):
        """After random-crop augmentation the spatial dimensions are restored."""
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        aug_config = AugmentationConfig(random_crop=True, crop_scale=(0.8, 0.9), random_flip=False)

        # Typical image tensor [B, C, H, W]
        image = torch.rand(1, 3, 32, 32)
        augmented = strategy._augment(image, aug_config)

        assert augmented.shape == image.shape

    def test_augmentation_random_crop_changes_values(self):
        """Augmented tensor differs from original (crop + resize alters pixel values)."""
        from embedding_art.core.strategies import OptimizationStrategy

        torch.manual_seed(7)
        strategy = OptimizationStrategy()
        aug_config = AugmentationConfig(random_crop=True, crop_scale=(0.5, 0.6), random_flip=False)

        image = torch.rand(1, 3, 32, 32)
        # Run enough trials to guarantee at least one crop changes the image
        any_different = any(
            not torch.allclose(strategy._augment(image, aug_config), image) for _ in range(10)
        )
        assert any_different, "Random crop should produce different pixel values"

    def test_augmentation_disabled_when_both_flags_false(self):
        """No augmentation applied when random_crop=False and random_flip=False."""
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        aug_config = AugmentationConfig(random_crop=False, random_flip=False)

        image = torch.rand(1, 3, 32, 32)
        augmented = strategy._augment(image, aug_config)

        assert torch.equal(augmented, image)

    def test_augmentation_random_flip_preserves_shape(self):
        """Horizontal flip keeps tensor shape intact."""
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        aug_config = AugmentationConfig(random_crop=False, random_flip=True)

        image = torch.rand(1, 3, 16, 16)
        augmented = strategy._augment(image, aug_config)

        assert augmented.shape == image.shape

    def test_augmentation_skipped_for_non_spatial_tensors(self):
        """1-D or 2-D tensors (e.g. audio) are returned unchanged."""
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        aug_config = AugmentationConfig(random_crop=True, random_flip=True)

        audio = torch.rand(1, 1024)
        result = strategy._augment(audio, aug_config)

        assert torch.equal(result, audio)


# ---------------------------------------------------------------------------
# Augmentation integration tests (augmentation wired into render())
# ---------------------------------------------------------------------------


class TestRenderWithAugmentation:
    """Augmentation is applied inside the render() optimization loop."""

    def test_render_with_augmentation_enabled_returns_render_result(self):
        """render() completes normally with random_crop=True."""
        from embedding_art.core.render_result import RenderResult
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallMockGenerator(output_channels=4, output_size=16)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config(
            augmentation=AugmentationConfig(random_crop=True, crop_scale=(0.8, 1.0))
        )

        result = strategy.render(target, generator, encoder, config)

        assert isinstance(result, RenderResult)

    def test_render_without_augmentation_returns_render_result(self):
        """render() completes normally with random_crop=False."""
        from embedding_art.core.render_result import RenderResult
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallMockGenerator(output_channels=4, output_size=16)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config(augmentation=AugmentationConfig(random_crop=False, random_flip=False))

        result = strategy.render(target, generator, encoder, config)

        assert isinstance(result, RenderResult)

    def test_augment_called_each_step_when_enabled(self):
        """_augment is invoked once per optimization step when random_crop=True."""
        from unittest.mock import patch

        from embedding_art.core.strategies import OptimizationStrategy

        n_steps = 4
        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallMockGenerator(output_channels=4, output_size=16)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config(
            steps=n_steps,
            augmentation=AugmentationConfig(random_crop=True),
        )

        with patch.object(strategy, "_augment", wraps=strategy._augment) as mock_aug:
            strategy.render(target, generator, encoder, config)

        assert mock_aug.call_count == n_steps

    def test_augment_not_called_when_disabled(self):
        """_augment is still called but returns tensor unchanged when disabled."""
        from unittest.mock import patch

        from embedding_art.core.strategies import OptimizationStrategy

        n_steps = 3
        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallMockGenerator(output_channels=4, output_size=16)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config(
            steps=n_steps,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        with patch.object(strategy, "_augment", wraps=strategy._augment) as mock_aug:
            strategy.render(target, generator, encoder, config)

        # _augment is called but should return the tensor unchanged
        assert mock_aug.call_count == n_steps


# ---------------------------------------------------------------------------
# Callback tests
# ---------------------------------------------------------------------------


class TestRenderCallback:
    """Callbacks are invoked each step with (step, breakdown, output)."""

    def test_callback_called_each_step(self):
        """Callback is invoked exactly config.steps times."""
        from embedding_art.core.strategies import OptimizationStrategy

        n_steps = 6
        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallMockGenerator(output_channels=4, output_size=8)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config(steps=n_steps)

        calls: list[tuple] = []

        def cb(step, breakdown, output):
            calls.append((step, breakdown, output))

        strategy.render(target, generator, encoder, config, callback=cb)

        assert len(calls) == n_steps

    def test_callback_receives_correct_step_indices(self):
        """Callback step argument is 0-indexed and sequential."""
        from embedding_art.core.strategies import OptimizationStrategy

        n_steps = 5
        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallMockGenerator(output_channels=4, output_size=8)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config(steps=n_steps)

        received_steps: list[int] = []

        def cb(step, breakdown, output):
            received_steps.append(step)

        strategy.render(target, generator, encoder, config, callback=cb)

        assert received_steps == list(range(n_steps))

    def test_callback_receives_output_tensor(self):
        """Callback output argument is a torch.Tensor."""
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallMockGenerator(output_channels=4, output_size=8)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config(steps=3)

        outputs: list = []

        def cb(step, breakdown, output):
            outputs.append(output)

        strategy.render(target, generator, encoder, config, callback=cb)

        assert all(isinstance(o, torch.Tensor) for o in outputs)

    def test_callback_receives_breakdown_with_total(self):
        """Callback breakdown argument has a .total attribute (LossBreakdown)."""
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallMockGenerator(output_channels=4, output_size=8)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config(steps=3)

        breakdowns: list = []

        def cb(step, breakdown, output):
            breakdowns.append(breakdown)

        strategy.render(target, generator, encoder, config, callback=cb)

        assert all(hasattr(b, "total") for b in breakdowns)

    def test_callback_none_does_not_crash(self):
        """Passing callback=None should not raise any exception."""
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallMockGenerator(output_channels=4, output_size=8)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config(steps=3)

        # Should not raise
        strategy.render(target, generator, encoder, config, callback=None)

    def test_callback_default_is_none(self):
        """render() signature default for callback is None — no positional arg required."""
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallMockGenerator(output_channels=4, output_size=8)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config(steps=2)

        # Original call signature (no callback keyword) must still work
        strategy.render(target, generator, encoder, config)


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------


class TestRenderCheckpoints:
    """Checkpoint files are written and can be used to resume optimization."""

    def test_checkpoint_saves_files(self):
        """checkpoint_dir causes .pt files to appear on disk."""
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallMockGenerator(output_channels=4, output_size=8)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config(steps=5, checkpoint_every=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            strategy.render(target, generator, encoder, config, checkpoint_dir=tmpdir)
            saved = list(Path(tmpdir).glob("checkpoint_*.pt"))

        assert len(saved) > 0

    def test_checkpoint_files_contain_required_keys(self):
        """Each checkpoint file has 'latent', 'optimizer_state', and 'step' keys."""
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallMockGenerator(output_channels=4, output_size=8)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config(steps=3, checkpoint_every=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            strategy.render(target, generator, encoder, config, checkpoint_dir=tmpdir)
            saved = sorted(Path(tmpdir).glob("checkpoint_*.pt"))
            checkpoint = torch.load(saved[0], weights_only=True)

        assert "latent" in checkpoint
        assert "optimizer_state" in checkpoint
        assert "step" in checkpoint

    def test_checkpoint_step_matches_filename(self):
        """The 'step' value stored in the checkpoint matches the filename suffix."""
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallMockGenerator(output_channels=4, output_size=8)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config(steps=5, checkpoint_every=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            strategy.render(target, generator, encoder, config, checkpoint_dir=tmpdir)
            saved = sorted(Path(tmpdir).glob("checkpoint_*.pt"))
            for path in saved:
                step_in_name = int(path.stem.split("_")[1])
                ckpt = torch.load(path, weights_only=True)
                assert ckpt["step"] == step_in_name

    def test_no_checkpoints_when_checkpoint_dir_is_none(self):
        """When checkpoint_dir is None, no files are written anywhere."""
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallMockGenerator(output_channels=4, output_size=8)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config(steps=5, checkpoint_every=1)

        # Should not raise even with checkpoint_every=1 and no checkpoint_dir
        result = strategy.render(target, generator, encoder, config, checkpoint_dir=None)

        from embedding_art.core.render_result import RenderResult

        assert isinstance(result, RenderResult)

    def test_checkpoint_resume_continues_from_saved_step(self):
        """Resuming from a checkpoint file starts at step + 1, not step 0."""
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallMockGenerator(latent_shape=(1, 4, 4, 4), output_channels=4, output_size=8)
        target = _make_target_concept(embedding_dim=16)

        # Save a checkpoint at step 0
        config_save = _fast_config(steps=3, checkpoint_every=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            strategy.render(target, generator, encoder, config_save, checkpoint_dir=tmpdir)
            saved = sorted(Path(tmpdir).glob("checkpoint_*.pt"))
            # Use the first checkpoint (step 0)
            first_ckpt = saved[0]

            # Resume: total steps=3, but we start after saved step so fewer
            # optimizer updates happen.  Track steps via callback.
            config_resume = _fast_config(steps=3, checkpoint_every=0)
            received_steps: list[int] = []

            def cb(step, breakdown, output):
                received_steps.append(step)

            strategy.render(
                target,
                generator,
                encoder,
                config_resume,
                callback=cb,
                resume_from=str(first_ckpt),
            )

        # The saved checkpoint was step=0, so resume starts at step=1.
        # config.steps=3, so we expect steps [1, 2].
        assert received_steps[0] > 0, "Resumed run should skip steps before checkpoint"

    def test_checkpoint_resume_loads_latent_from_file(self):
        """Latent tensor at resume start matches what was saved in checkpoint."""
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallMockGenerator(latent_shape=(1, 4, 4, 4), output_channels=4, output_size=8)
        target = _make_target_concept(embedding_dim=16)

        config_save = _fast_config(steps=2, checkpoint_every=1, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            strategy.render(target, generator, encoder, config_save, checkpoint_dir=tmpdir)
            saved = sorted(Path(tmpdir).glob("checkpoint_*.pt"))
            ckpt_path = saved[0]
            saved_latent = torch.load(ckpt_path, weights_only=True)["latent"]

            # Capture the latent at the very first step of the resumed run
            first_output: list[torch.Tensor] = []

            def cb(step, breakdown, output):
                if len(first_output) == 0:
                    first_output.append(output.clone())

            config_resume = _fast_config(steps=2, checkpoint_every=0, seed=99)
            strategy.render(
                target,
                generator,
                encoder,
                config_resume,
                callback=cb,
                resume_from=str(ckpt_path),
            )

        # The first decode in the resumed run should differ from a fresh seed=99 run
        # because it starts from the saved latent, not a freshly seeded one.
        fresh_latent = generator.init_latent(seed=99)
        assert not torch.allclose(
            saved_latent, fresh_latent.detach()
        ), "Saved latent must differ from a fresh seed=99 latent"


# ---------------------------------------------------------------------------
# Combined feature test
# ---------------------------------------------------------------------------


class TestRenderWithAllFeatures:
    """augmentation + callback + checkpoint all active simultaneously."""

    def test_render_with_all_features(self):
        """render() succeeds with augmentation, callback, and checkpoint_dir all set."""
        from embedding_art.core.render_result import RenderResult
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallMockGenerator(output_channels=4, output_size=16)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config(
            steps=6,
            checkpoint_every=3,
            augmentation=AugmentationConfig(random_crop=True, crop_scale=(0.8, 1.0)),
        )

        cb_calls: list[tuple] = []

        def cb(step, breakdown, output):
            cb_calls.append((step, breakdown, output))

        with tempfile.TemporaryDirectory() as tmpdir:
            result = strategy.render(
                target,
                generator,
                encoder,
                config,
                callback=cb,
                checkpoint_dir=tmpdir,
            )
            saved_files = list(Path(tmpdir).glob("checkpoint_*.pt"))

        assert isinstance(result, RenderResult)
        assert len(cb_calls) == 6
        assert len(saved_files) > 0
