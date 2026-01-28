"""
Unit tests for configuration dataclasses and YAML config loading.

Tests verify behavior of AugmentationConfig, OptimizationConfig,
and the load_config function through their public API.
"""

import os
import tempfile
from pathlib import Path

from embedding_art.core.config import (
    AugmentationConfig,
    OptimizationConfig,
    load_config,
)

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


# =============================================================================
# load_config Tests
# =============================================================================


class TestLoadConfigFromFile:
    """Tests for loading configuration from YAML files."""

    def test_returns_dict_with_all_sections(self) -> None:
        """load_config should return dict with optimization, regularization, output, device."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("""
optimization:
  steps: 1000
  learning_rate: 0.05
regularization:
  total_variation: 0.02
output:
  checkpoint_every: 50
device: cuda
""")
            f.flush()
            config = load_config(f.name)

        assert "optimization" in config
        assert "regularization" in config
        assert "output" in config
        assert "device" in config

    def test_optimization_section_values(self) -> None:
        """Optimization section should contain expected values from YAML."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("""
optimization:
  steps: 1500
  learning_rate: 0.2
  optimizer: adamw
  scheduler: linear
""")
            f.flush()
            config = load_config(f.name)

        assert config["optimization"]["steps"] == 1500
        assert config["optimization"]["learning_rate"] == 0.2
        assert config["optimization"]["optimizer"] == "adamw"
        assert config["optimization"]["scheduler"] == "linear"

    def test_regularization_section_values(self) -> None:
        """Regularization section should contain expected values from YAML."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("""
regularization:
  total_variation: 0.05
  spectral: 0.002
  latent_norm: 0.15
""")
            f.flush()
            config = load_config(f.name)

        assert config["regularization"]["total_variation"] == 0.05
        assert config["regularization"]["spectral"] == 0.002
        assert config["regularization"]["latent_norm"] == 0.15

    def test_output_section_values(self) -> None:
        """Output section should contain expected values from YAML."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("""
output:
  checkpoint_every: 200
  preview_every: 100
""")
            f.flush()
            config = load_config(f.name)

        assert config["output"]["checkpoint_every"] == 200
        assert config["output"]["preview_every"] == 100

    def test_device_value(self) -> None:
        """Device should be read correctly from YAML."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("""
device: mps
""")
            f.flush()
            config = load_config(f.name)

        assert config["device"] == "mps"

    def test_nonexistent_file_returns_empty_dict(self) -> None:
        """Loading from nonexistent file should return empty dict."""
        config = load_config("/nonexistent/path/config.yaml")

        assert config == {}

    def test_none_path_with_no_config_file_returns_empty_dict(self, isolated_cwd: Path) -> None:
        """Passing None when no config.yaml exists in CWD returns empty dict."""
        config = load_config(None)

        assert config == {}

    def test_empty_file_returns_empty_dict(self) -> None:
        """Loading an empty YAML file should return empty dict."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("")
            f.flush()
            config = load_config(f.name)

        assert config == {}

    def test_partial_config_missing_sections_are_not_present(self) -> None:
        """Missing sections should not be present in the returned dict."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("""
optimization:
  steps: 500
""")
            f.flush()
            config = load_config(f.name)

        assert config["optimization"]["steps"] == 500
        assert "regularization" not in config
        assert "output" not in config

    def test_accepts_path_object(self) -> None:
        """load_config should accept Path objects, not just strings."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("""
device: cpu
""")
            f.flush()
            config = load_config(Path(f.name))

        assert config["device"] == "cpu"

    def test_augmentation_section_values(self) -> None:
        """Augmentation section within optimization should be parsed."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("""
optimization:
  augmentation:
    random_crop: false
    crop_scale: [0.7, 0.9]
    random_flip: true
    color_jitter: true
""")
            f.flush()
            config = load_config(f.name)

        aug = config["optimization"]["augmentation"]
        assert aug["random_crop"] is False
        assert aug["crop_scale"] == [0.7, 0.9]
        assert aug["random_flip"] is True
        assert aug["color_jitter"] is True


class TestConfigToOptimizationConfig:
    """Tests for converting loaded config dict to OptimizationConfig."""

    def test_create_optimization_config_from_dict(self) -> None:
        """Should be able to create OptimizationConfig from loaded config dict."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("""
optimization:
  steps: 1000
  learning_rate: 0.05
  optimizer: adam
  scheduler: cosine
  checkpoint_every: 50
  preview_every: 25
""")
            f.flush()
            config = load_config(f.name)

        opt_config = OptimizationConfig(**config["optimization"])

        assert opt_config.steps == 1000
        assert opt_config.learning_rate == 0.05
        assert opt_config.optimizer == "adam"
        assert opt_config.scheduler == "cosine"
        assert opt_config.checkpoint_every == 50
        assert opt_config.preview_every == 25

    def test_partial_config_uses_defaults(self) -> None:
        """Partial config should use OptimizationConfig defaults for missing fields."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("""
optimization:
  steps: 500
""")
            f.flush()
            config = load_config(f.name)

        opt_config = OptimizationConfig(**config["optimization"])

        assert opt_config.steps == 500
        # Defaults from OptimizationConfig
        assert opt_config.learning_rate == 0.1
        assert opt_config.optimizer == "adam"
        assert opt_config.scheduler == "cosine"

    def test_augmentation_config_from_nested_dict(self) -> None:
        """AugmentationConfig nested in optimization should be created correctly."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("""
optimization:
  augmentation:
    random_crop: false
    random_flip: true
""")
            f.flush()
            config = load_config(f.name)

        opt_config = OptimizationConfig(**config["optimization"])

        assert isinstance(opt_config.augmentation, AugmentationConfig)
        assert opt_config.augmentation.random_crop is False
        assert opt_config.augmentation.random_flip is True


class TestDefaultConfigPath:
    """Tests for default config file discovery."""

    def test_load_config_finds_default_config_yaml_in_cwd(self) -> None:
        """When no path specified, should look for config.yaml in current directory."""
        original_cwd = os.getcwd()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("""
device: test_device
""")
            os.chdir(tmpdir)

            try:
                config = load_config()
                assert config.get("device") == "test_device"
            finally:
                os.chdir(original_cwd)

    def test_load_config_no_default_file_returns_empty(self) -> None:
        """When no default config exists and no path specified, return empty dict."""
        original_cwd = os.getcwd()

        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)

            try:
                config = load_config()
                assert config == {}
            finally:
                os.chdir(original_cwd)
