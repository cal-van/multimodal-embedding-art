"""
Unit tests for LossConfig and its integration with OptimizationConfig.

LossConfig defines the composite loss weights used during optimization.
It lives on OptimizationConfig.loss and should not disturb any existing
default values on OptimizationConfig.
"""

from embedding_art.core.config import LossConfig, OptimizationConfig

# =============================================================================
# LossConfig defaults
# =============================================================================


class TestLossConfigDefaults:
    """LossConfig should have the correct default values."""

    def test_similarity_weight_defaults_to_1_0(self) -> None:
        """similarity_weight should default to 1.0."""
        config = LossConfig()

        assert config.similarity_weight == 1.0

    def test_feature_matching_weight_defaults_to_0_5(self) -> None:
        """feature_matching_weight should default to 0.5."""
        config = LossConfig()

        assert config.feature_matching_weight == 0.5

    def test_feature_matching_layers_defaults_to_every_4th(self) -> None:
        """feature_matching_layers should default to the string 'every_4th'."""
        config = LossConfig()

        assert config.feature_matching_layers == "every_4th"

    def test_sae_feature_weight_defaults_to_0_0(self) -> None:
        """sae_feature_weight should default to 0.0."""
        config = LossConfig()

        assert config.sae_feature_weight == 0.0

    def test_sae_target_features_defaults_to_none(self) -> None:
        """sae_target_features should default to None."""
        config = LossConfig()

        assert config.sae_target_features is None


# =============================================================================
# LossConfig overrides
# =============================================================================


class TestLossConfigOverrides:
    """LossConfig fields should be overridable at construction time."""

    def test_can_set_similarity_weight(self) -> None:
        """Should be able to set a custom similarity_weight."""
        config = LossConfig(similarity_weight=2.0)

        assert config.similarity_weight == 2.0

    def test_can_set_feature_matching_weight(self) -> None:
        """Should be able to set a custom feature_matching_weight."""
        config = LossConfig(feature_matching_weight=0.0)

        assert config.feature_matching_weight == 0.0

    def test_can_set_feature_matching_layers_as_list(self) -> None:
        """feature_matching_layers should accept a list of ints."""
        config = LossConfig(feature_matching_layers=[4, 8, 12])

        assert config.feature_matching_layers == [4, 8, 12]

    def test_can_set_feature_matching_layers_as_string(self) -> None:
        """feature_matching_layers should accept a string preset."""
        config = LossConfig(feature_matching_layers="all")

        assert config.feature_matching_layers == "all"

    def test_can_set_sae_feature_weight(self) -> None:
        """Should be able to set a non-zero sae_feature_weight."""
        config = LossConfig(sae_feature_weight=0.1)

        assert config.sae_feature_weight == 0.1

    def test_can_set_sae_target_features(self) -> None:
        """sae_target_features should accept a dict of feature id to weight."""
        features = {"feature_42": 0.8, "feature_7": 0.3}
        config = LossConfig(sae_target_features=features)

        assert config.sae_target_features == features

    def test_multiple_configs_have_independent_sae_target_features(self) -> None:
        """Two LossConfig instances should not share sae_target_features state."""
        LossConfig(sae_target_features={"f": 1.0})
        config2 = LossConfig()

        assert config2.sae_target_features is None


# =============================================================================
# OptimizationConfig.loss field
# =============================================================================


class TestOptimizationConfigLossField:
    """OptimizationConfig should expose a .loss field backed by LossConfig."""

    def test_optimization_config_has_loss_field(self) -> None:
        """OptimizationConfig() should have a .loss attribute."""
        config = OptimizationConfig()

        assert hasattr(config, "loss")

    def test_loss_field_is_loss_config_instance(self) -> None:
        """OptimizationConfig.loss should be an instance of LossConfig."""
        config = OptimizationConfig()

        assert isinstance(config.loss, LossConfig)

    def test_loss_field_has_correct_defaults(self) -> None:
        """OptimizationConfig.loss should expose LossConfig defaults."""
        config = OptimizationConfig()

        assert config.loss.similarity_weight == 1.0
        assert config.loss.feature_matching_weight == 0.5
        assert config.loss.feature_matching_layers == "every_4th"
        assert config.loss.sae_feature_weight == 0.0
        assert config.loss.sae_target_features is None

    def test_multiple_optimization_configs_have_independent_loss(self) -> None:
        """Each OptimizationConfig instance should have its own LossConfig."""
        config1 = OptimizationConfig()
        config2 = OptimizationConfig()

        config1.loss.similarity_weight = 99.0

        assert config2.loss.similarity_weight == 1.0

    def test_can_pass_custom_loss_config(self) -> None:
        """Should be able to supply a custom LossConfig at construction time."""
        loss = LossConfig(similarity_weight=0.5, sae_feature_weight=0.2)
        config = OptimizationConfig(loss=loss)

        assert config.loss.similarity_weight == 0.5
        assert config.loss.sae_feature_weight == 0.2


# =============================================================================
# Existing OptimizationConfig defaults are unchanged
# =============================================================================


class TestOptimizationConfigExistingDefaultsUnchanged:
    """Adding LossConfig must not alter any pre-existing OptimizationConfig defaults."""

    def test_steps_still_defaults_to_2000(self) -> None:
        config = OptimizationConfig()
        assert config.steps == 2000

    def test_learning_rate_still_defaults_to_0_1(self) -> None:
        config = OptimizationConfig()
        assert config.learning_rate == 0.1

    def test_optimizer_still_defaults_to_adam(self) -> None:
        config = OptimizationConfig()
        assert config.optimizer == "adam"

    def test_scheduler_still_defaults_to_cosine(self) -> None:
        config = OptimizationConfig()
        assert config.scheduler == "cosine"

    def test_checkpoint_every_still_defaults_to_100(self) -> None:
        config = OptimizationConfig()
        assert config.checkpoint_every == 100

    def test_preview_every_still_defaults_to_50(self) -> None:
        config = OptimizationConfig()
        assert config.preview_every == 50

    def test_seed_still_defaults_to_none(self) -> None:
        config = OptimizationConfig()
        assert config.seed is None

    def test_old_construction_without_loss_still_works(self) -> None:
        """Constructing OptimizationConfig without providing loss should succeed."""
        config = OptimizationConfig(steps=500, learning_rate=0.05)

        assert config.steps == 500
        assert config.learning_rate == 0.05
        assert isinstance(config.loss, LossConfig)
