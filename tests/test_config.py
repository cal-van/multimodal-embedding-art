"""
Unit tests for configuration dataclasses.

Tests verify behavior of AugmentationConfig and OptimizationConfig
through their public API, ensuring defaults are correct and
configuration can be customized as expected.
"""

import pytest

from embedding_art.core.config import AugmentationConfig, OptimizationConfig


# =============================================================================
# AugmentationConfig Tests
# =============================================================================


class TestAugmentationConfigDefaults:
    """Tests for AugmentationConfig default values."""

    def test_random_crop_defaults_to_true(self) -> None:
        """AugmentationConfig should enable random cropping by default."""
        config = AugmentationConfig()

        assert config.random_crop is True

    def test_crop_scale_defaults_to_0_8_to_1_0(self) -> None:
        """AugmentationConfig should use crop scale range (0.8, 1.0) by default."""
        config = AugmentationConfig()

        assert config.crop_scale == (0.8, 1.0)

    def test_random_flip_defaults_to_false(self) -> None:
        """AugmentationConfig should disable random flipping by default."""
        config = AugmentationConfig()

        assert config.random_flip is False

    def test_color_jitter_defaults_to_false(self) -> None:
        """AugmentationConfig should disable color jittering by default."""
        config = AugmentationConfig()

        assert config.color_jitter is False


class TestAugmentationConfigOverrides:
    """Tests for AugmentationConfig field overrides."""

    def test_can_disable_random_crop(self) -> None:
        """Should be able to disable random cropping."""
        config = AugmentationConfig(random_crop=False)

        assert config.random_crop is False

    def test_can_set_custom_crop_scale(self) -> None:
        """Should be able to set custom crop scale range."""
        config = AugmentationConfig(crop_scale=(0.5, 0.9))

        assert config.crop_scale == (0.5, 0.9)

    def test_can_enable_random_flip(self) -> None:
        """Should be able to enable random flipping."""
        config = AugmentationConfig(random_flip=True)

        assert config.random_flip is True

    def test_can_enable_color_jitter(self) -> None:
        """Should be able to enable color jittering."""
        config = AugmentationConfig(color_jitter=True)

        assert config.color_jitter is True

    def test_can_override_multiple_fields(self) -> None:
        """Should be able to override multiple fields at once."""
        config = AugmentationConfig(
            random_crop=False,
            crop_scale=(0.6, 0.95),
            random_flip=True,
            color_jitter=True,
        )

        assert config.random_crop is False
        assert config.crop_scale == (0.6, 0.95)
        assert config.random_flip is True
        assert config.color_jitter is True


# =============================================================================
# OptimizationConfig Tests
# =============================================================================


class TestOptimizationConfigDefaults:
    """Tests for OptimizationConfig default values."""

    def test_steps_defaults_to_2000(self) -> None:
        """OptimizationConfig should use 2000 steps by default."""
        config = OptimizationConfig()

        assert config.steps == 2000

    def test_learning_rate_defaults_to_0_1(self) -> None:
        """OptimizationConfig should use learning rate 0.1 by default."""
        config = OptimizationConfig()

        assert config.learning_rate == 0.1

    def test_optimizer_defaults_to_adam(self) -> None:
        """OptimizationConfig should use adam optimizer by default."""
        config = OptimizationConfig()

        assert config.optimizer == "adam"

    def test_scheduler_defaults_to_cosine(self) -> None:
        """OptimizationConfig should use cosine scheduler by default."""
        config = OptimizationConfig()

        assert config.scheduler == "cosine"

    def test_checkpoint_every_defaults_to_100(self) -> None:
        """OptimizationConfig should checkpoint every 100 steps by default."""
        config = OptimizationConfig()

        assert config.checkpoint_every == 100

    def test_preview_every_defaults_to_50(self) -> None:
        """OptimizationConfig should preview every 50 steps by default."""
        config = OptimizationConfig()

        assert config.preview_every == 50

    def test_seed_defaults_to_none(self) -> None:
        """OptimizationConfig should have no seed by default."""
        config = OptimizationConfig()

        assert config.seed is None

    def test_augmentation_defaults_to_augmentation_config_instance(self) -> None:
        """OptimizationConfig should create AugmentationConfig with defaults."""
        config = OptimizationConfig()

        assert isinstance(config.augmentation, AugmentationConfig)
        assert config.augmentation.random_crop is True
        assert config.augmentation.crop_scale == (0.8, 1.0)


class TestOptimizationConfigOverrides:
    """Tests for OptimizationConfig field overrides."""

    def test_can_set_custom_steps(self) -> None:
        """Should be able to set custom number of steps."""
        config = OptimizationConfig(steps=500)

        assert config.steps == 500

    def test_can_set_custom_learning_rate(self) -> None:
        """Should be able to set custom learning rate."""
        config = OptimizationConfig(learning_rate=0.01)

        assert config.learning_rate == 0.01

    def test_can_set_optimizer_to_adamw(self) -> None:
        """Should be able to use adamw optimizer."""
        config = OptimizationConfig(optimizer="adamw")

        assert config.optimizer == "adamw"

    def test_can_set_optimizer_to_sgd(self) -> None:
        """Should be able to use sgd optimizer."""
        config = OptimizationConfig(optimizer="sgd")

        assert config.optimizer == "sgd"

    def test_can_set_scheduler_to_constant(self) -> None:
        """Should be able to use constant scheduler."""
        config = OptimizationConfig(scheduler="constant")

        assert config.scheduler == "constant"

    def test_can_set_scheduler_to_linear(self) -> None:
        """Should be able to use linear scheduler."""
        config = OptimizationConfig(scheduler="linear")

        assert config.scheduler == "linear"

    def test_can_set_custom_checkpoint_every(self) -> None:
        """Should be able to set custom checkpoint frequency."""
        config = OptimizationConfig(checkpoint_every=200)

        assert config.checkpoint_every == 200

    def test_can_set_custom_preview_every(self) -> None:
        """Should be able to set custom preview frequency."""
        config = OptimizationConfig(preview_every=25)

        assert config.preview_every == 25

    def test_can_set_seed(self) -> None:
        """Should be able to set reproducibility seed."""
        config = OptimizationConfig(seed=42)

        assert config.seed == 42


class TestOptimizationConfigAugmentation:
    """Tests for OptimizationConfig augmentation handling."""

    def test_accepts_augmentation_config_instance(self) -> None:
        """Should accept AugmentationConfig instance directly."""
        augmentation = AugmentationConfig(
            random_crop=False,
            random_flip=True,
        )
        config = OptimizationConfig(augmentation=augmentation)

        assert config.augmentation is augmentation
        assert config.augmentation.random_crop is False
        assert config.augmentation.random_flip is True

    def test_converts_dict_to_augmentation_config(self) -> None:
        """__post_init__ should convert dict to AugmentationConfig."""
        config = OptimizationConfig(
            augmentation={
                "random_crop": False,
                "crop_scale": (0.7, 0.9),
            }
        )

        assert isinstance(config.augmentation, AugmentationConfig)
        assert config.augmentation.random_crop is False
        assert config.augmentation.crop_scale == (0.7, 0.9)

    def test_dict_to_augmentation_preserves_non_overridden_defaults(self) -> None:
        """Dict conversion should keep default values for non-specified fields."""
        config = OptimizationConfig(
            augmentation={
                "random_flip": True,
            }
        )

        assert config.augmentation.random_crop is True
        assert config.augmentation.crop_scale == (0.8, 1.0)
        assert config.augmentation.random_flip is True
        assert config.augmentation.color_jitter is False

    def test_empty_dict_creates_default_augmentation_config(self) -> None:
        """Empty dict should create AugmentationConfig with all defaults."""
        config = OptimizationConfig(augmentation={})

        assert isinstance(config.augmentation, AugmentationConfig)
        assert config.augmentation.random_crop is True
        assert config.augmentation.crop_scale == (0.8, 1.0)
        assert config.augmentation.random_flip is False
        assert config.augmentation.color_jitter is False


class TestOptimizationConfigNested:
    """Tests for nested configuration behavior."""

    def test_nested_augmentation_config_is_independent(self) -> None:
        """Nested AugmentationConfig should be independent from source."""
        augmentation = AugmentationConfig(random_crop=False)
        config = OptimizationConfig(augmentation=augmentation)

        augmentation.random_crop = True

        assert config.augmentation.random_crop is True

    def test_multiple_configs_have_independent_augmentation(self) -> None:
        """Each OptimizationConfig should have its own AugmentationConfig."""
        config1 = OptimizationConfig()
        config2 = OptimizationConfig()

        config1.augmentation.random_crop = False

        assert config2.augmentation.random_crop is True
