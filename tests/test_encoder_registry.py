"""
Tests for EncoderCapability, EncoderCard, and EncoderRegistry.

Following Red-Green-Refactor TDD — tests are written before the implementation.
Covers: capability flags, card creation, registry CRUD, lazy loading, caching,
        modality lookup, memory budget check, and loaded-encoder introspection.
"""

import pytest

from embedding_art.encoders.registry import EncoderCapability, EncoderCard, EncoderRegistry
from embedding_art.exceptions import EncoderNotFoundError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _MinimalEncoder:
    """Trivial encoder used as a stand-in for registry tests."""

    card = EncoderCard(
        name="minimal",
        capabilities=EncoderCapability.TEXT | EncoderCapability.IMAGE,
        embedding_dim=512,
        memory_estimate_mb=256,
        backprop_cost=1.0,
    )

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _AudioEncoder:
    """Encoder that supports audio and video."""

    card = EncoderCard(
        name="audio_enc",
        capabilities=EncoderCapability.AUDIO | EncoderCapability.VIDEO,
        embedding_dim=1024,
        memory_estimate_mb=1024,
        backprop_cost=2.0,
    )

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _HeavyEncoder:
    """Very large encoder for memory-budget tests."""

    card = EncoderCard(
        name="heavy",
        capabilities=EncoderCapability.TEXT,
        embedding_dim=4096,
        memory_estimate_mb=8192,
        backprop_cost=5.0,
    )

    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def registry() -> EncoderRegistry:
    """Fresh, empty registry for each test."""
    return EncoderRegistry()


@pytest.fixture
def populated_registry() -> EncoderRegistry:
    """Registry pre-loaded with minimal and audio_enc registrations."""
    reg = EncoderRegistry()
    reg.register("minimal", _MinimalEncoder)
    reg.register("audio_enc", _AudioEncoder)
    return reg


# ---------------------------------------------------------------------------
# EncoderCapability — flag enum
# ---------------------------------------------------------------------------


class TestEncoderCapability:
    def test_text_flag_exists(self):
        assert EncoderCapability.TEXT

    def test_image_flag_exists(self):
        assert EncoderCapability.IMAGE

    def test_audio_flag_exists(self):
        assert EncoderCapability.AUDIO

    def test_video_flag_exists(self):
        assert EncoderCapability.VIDEO

    def test_depth_flag_exists(self):
        assert EncoderCapability.DEPTH

    def test_backprop_optimizable_flag_exists(self):
        assert EncoderCapability.BACKPROP_OPTIMIZABLE

    def test_multi_layer_features_flag_exists(self):
        assert EncoderCapability.MULTI_LAYER_FEATURES

    def test_flags_are_composable_with_or(self):
        combined = EncoderCapability.TEXT | EncoderCapability.IMAGE
        assert EncoderCapability.TEXT in combined
        assert EncoderCapability.IMAGE in combined

    def test_combined_flag_does_not_include_unset_member(self):
        combined = EncoderCapability.TEXT | EncoderCapability.IMAGE
        assert EncoderCapability.AUDIO not in combined

    def test_flag_membership_with_in_operator(self):
        caps = EncoderCapability.AUDIO | EncoderCapability.VIDEO
        assert EncoderCapability.AUDIO in caps
        assert EncoderCapability.VIDEO in caps
        assert EncoderCapability.TEXT not in caps

    def test_flags_have_unique_values(self):
        members = list(EncoderCapability)
        values = [m.value for m in members]
        assert len(values) == len(set(values))

    def test_empty_combination_is_falsy(self):
        empty = EncoderCapability(0)
        assert not empty


# ---------------------------------------------------------------------------
# EncoderCard — frozen dataclass
# ---------------------------------------------------------------------------


class TestEncoderCard:
    def test_stores_name(self):
        card = EncoderCard(
            name="test",
            capabilities=EncoderCapability.TEXT,
            embedding_dim=512,
            memory_estimate_mb=128,
            backprop_cost=1.0,
        )
        assert card.name == "test"

    def test_stores_capabilities(self):
        caps = EncoderCapability.TEXT | EncoderCapability.IMAGE
        card = EncoderCard(
            name="test",
            capabilities=caps,
            embedding_dim=512,
            memory_estimate_mb=128,
            backprop_cost=1.0,
        )
        assert card.capabilities == caps

    def test_stores_embedding_dim(self):
        card = EncoderCard(
            name="test",
            capabilities=EncoderCapability.TEXT,
            embedding_dim=1024,
            memory_estimate_mb=128,
            backprop_cost=1.0,
        )
        assert card.embedding_dim == 1024

    def test_stores_memory_estimate_mb(self):
        card = EncoderCard(
            name="test",
            capabilities=EncoderCapability.TEXT,
            embedding_dim=512,
            memory_estimate_mb=2048,
            backprop_cost=1.0,
        )
        assert card.memory_estimate_mb == 2048

    def test_stores_backprop_cost(self):
        card = EncoderCard(
            name="test",
            capabilities=EncoderCapability.TEXT,
            embedding_dim=512,
            memory_estimate_mb=128,
            backprop_cost=3.5,
        )
        assert card.backprop_cost == 3.5

    def test_card_is_frozen(self):
        card = EncoderCard(
            name="test",
            capabilities=EncoderCapability.TEXT,
            embedding_dim=512,
            memory_estimate_mb=128,
            backprop_cost=1.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            card.name = "changed"  # type: ignore[misc]

    def test_card_equality(self):
        caps = EncoderCapability.TEXT
        card_a = EncoderCard("x", caps, 512, 128, 1.0)
        card_b = EncoderCard("x", caps, 512, 128, 1.0)
        assert card_a == card_b

    def test_card_inequality_on_different_name(self):
        caps = EncoderCapability.TEXT
        card_a = EncoderCard("x", caps, 512, 128, 1.0)
        card_b = EncoderCard("y", caps, 512, 128, 1.0)
        assert card_a != card_b


# ---------------------------------------------------------------------------
# EncoderRegistry — register & list_available
# ---------------------------------------------------------------------------


class TestEncoderRegistryRegister:
    def test_register_adds_encoder(self, registry):
        registry.register("minimal", _MinimalEncoder)
        names = [c.name for c in registry.list_available()]
        assert "minimal" in names

    def test_list_available_returns_encoder_cards(self, registry):
        registry.register("minimal", _MinimalEncoder)
        cards = registry.list_available()
        assert all(isinstance(c, EncoderCard) for c in cards)

    def test_list_available_empty_when_nothing_registered(self, registry):
        assert registry.list_available() == []

    def test_registering_two_encoders_lists_both(self, registry):
        registry.register("minimal", _MinimalEncoder)
        registry.register("audio_enc", _AudioEncoder)
        names = {c.name for c in registry.list_available()}
        assert names == {"minimal", "audio_enc"}

    def test_get_card_returns_correct_card(self, registry):
        registry.register("minimal", _MinimalEncoder)
        card = registry.get_card("minimal")
        assert card.name == "minimal"
        assert card.embedding_dim == 512

    def test_get_card_unknown_raises_encoder_not_found_error(self, registry):
        with pytest.raises(EncoderNotFoundError):
            registry.get_card("nonexistent")

    def test_get_card_error_message_lists_available(self, registry):
        registry.register("minimal", _MinimalEncoder)
        with pytest.raises(EncoderNotFoundError) as exc_info:
            registry.get_card("nonexistent")
        assert "minimal" in str(exc_info.value)


# ---------------------------------------------------------------------------
# EncoderRegistry — load
# ---------------------------------------------------------------------------


class TestEncoderRegistryLoad:
    def test_load_returns_encoder_instance(self, registry):
        registry.register("minimal", _MinimalEncoder)
        instance = registry.load("minimal")
        assert isinstance(instance, _MinimalEncoder)

    def test_load_unknown_raises_encoder_not_found_error(self, registry):
        with pytest.raises(EncoderNotFoundError):
            registry.load("ghost")

    def test_load_error_lists_registered_names(self, registry):
        registry.register("minimal", _MinimalEncoder)
        with pytest.raises(EncoderNotFoundError) as exc_info:
            registry.load("ghost")
        assert "minimal" in str(exc_info.value)

    def test_load_is_cached_returns_same_instance(self, registry):
        registry.register("minimal", _MinimalEncoder)
        first = registry.load("minimal")
        second = registry.load("minimal")
        assert first is second

    def test_load_passes_kwargs_to_constructor(self, registry):
        registry.register("minimal", _MinimalEncoder)
        instance = registry.load("minimal", device="cpu", precision="fp16")
        assert instance.kwargs == {"device": "cpu", "precision": "fp16"}

    def test_load_kwargs_only_used_on_first_call(self, registry):
        """Second call with different kwargs still returns the cached instance."""
        registry.register("minimal", _MinimalEncoder)
        first = registry.load("minimal", device="cpu")
        second = registry.load("minimal", device="mps")
        assert first is second


# ---------------------------------------------------------------------------
# EncoderRegistry — unload
# ---------------------------------------------------------------------------


class TestEncoderRegistryUnload:
    def test_unload_removes_cached_instance(self, registry):
        registry.register("minimal", _MinimalEncoder)
        first = registry.load("minimal")
        registry.unload("minimal")
        second = registry.load("minimal")
        assert first is not second

    def test_unload_removes_from_loaded_encoders(self, registry):
        registry.register("minimal", _MinimalEncoder)
        registry.load("minimal")
        registry.unload("minimal")
        assert "minimal" not in registry.loaded_encoders()

    def test_unload_unknown_does_not_raise(self, registry):
        """Unloading a name that was never loaded is a no-op."""
        registry.unload("ghost")  # Should not raise

    def test_unload_does_not_remove_registration(self, registry):
        """After unload, the encoder is still registered and can be loaded again."""
        registry.register("minimal", _MinimalEncoder)
        registry.load("minimal")
        registry.unload("minimal")
        # Should be re-loadable without re-registering
        instance = registry.load("minimal")
        assert isinstance(instance, _MinimalEncoder)


# ---------------------------------------------------------------------------
# EncoderRegistry — get_for_modality
# ---------------------------------------------------------------------------


class TestEncoderRegistryGetForModality:
    def test_get_for_modality_text_returns_text_encoders(self, populated_registry):
        names = populated_registry.get_for_modality("text")
        assert "minimal" in names
        assert "audio_enc" not in names

    def test_get_for_modality_image_returns_image_encoders(self, populated_registry):
        names = populated_registry.get_for_modality("image")
        assert "minimal" in names
        assert "audio_enc" not in names

    def test_get_for_modality_audio_returns_audio_encoders(self, populated_registry):
        names = populated_registry.get_for_modality("audio")
        assert "audio_enc" in names
        assert "minimal" not in names

    def test_get_for_modality_video_returns_video_encoders(self, populated_registry):
        names = populated_registry.get_for_modality("video")
        assert "audio_enc" in names
        assert "minimal" not in names

    def test_get_for_modality_unknown_returns_empty(self, populated_registry):
        names = populated_registry.get_for_modality("lidar")
        assert names == []

    def test_get_for_modality_case_insensitive(self, populated_registry):
        """Modality names should be case-insensitive."""
        upper = populated_registry.get_for_modality("TEXT")
        lower = populated_registry.get_for_modality("text")
        assert upper == lower


# ---------------------------------------------------------------------------
# EncoderRegistry — can_fit
# ---------------------------------------------------------------------------


class TestEncoderRegistryCanFit:
    def test_can_fit_returns_true_when_within_budget(self, registry):
        registry.register("minimal", _MinimalEncoder)  # 256 MB
        # Budget of 400 MB, safety margin = 80% → effective = 320 MB ≥ 256 MB
        assert registry.can_fit(["minimal"], memory_budget_mb=400) is True

    def test_can_fit_returns_false_when_over_budget_after_margin(self, registry):
        registry.register("minimal", _MinimalEncoder)  # 256 MB
        # Budget of 300 MB, effective = 240 MB < 256 MB
        assert registry.can_fit(["minimal"], memory_budget_mb=300) is False

    def test_can_fit_with_multiple_encoders(self, registry):
        registry.register("minimal", _MinimalEncoder)  # 256 MB
        registry.register("audio_enc", _AudioEncoder)  # 1024 MB
        # Total = 1280 MB, budget 2000 MB, effective = 1600 MB ≥ 1280 MB
        assert registry.can_fit(["minimal", "audio_enc"], memory_budget_mb=2000) is True

    def test_can_fit_multiple_encoders_over_budget(self, registry):
        registry.register("minimal", _MinimalEncoder)  # 256 MB
        registry.register("audio_enc", _AudioEncoder)  # 1024 MB
        # Total = 1280 MB, budget 1400 MB, effective = 1120 MB < 1280 MB
        assert registry.can_fit(["minimal", "audio_enc"], memory_budget_mb=1400) is False

    def test_can_fit_empty_list_always_true(self, registry):
        assert registry.can_fit([], memory_budget_mb=100) is True

    def test_can_fit_unknown_encoder_raises_encoder_not_found_error(self, registry):
        with pytest.raises(EncoderNotFoundError):
            registry.can_fit(["nonexistent"], memory_budget_mb=1000)

    def test_can_fit_exact_80_percent_boundary(self, registry):
        registry.register("minimal", _MinimalEncoder)  # 256 MB
        # Budget = 320 MB, effective = 256 MB — exactly equals required → True
        assert registry.can_fit(["minimal"], memory_budget_mb=320) is True

    def test_can_fit_one_mb_under_boundary(self, registry):
        registry.register("minimal", _MinimalEncoder)  # 256 MB
        # Budget = 319 MB, effective = 255.2 MB < 256 MB → False
        assert registry.can_fit(["minimal"], memory_budget_mb=319) is False


# ---------------------------------------------------------------------------
# EncoderRegistry — loaded_encoders
# ---------------------------------------------------------------------------


class TestEncoderRegistryLoadedEncoders:
    def test_loaded_encoders_empty_initially(self, registry):
        assert registry.loaded_encoders() == []

    def test_loaded_encoders_includes_loaded_name(self, registry):
        registry.register("minimal", _MinimalEncoder)
        registry.load("minimal")
        assert "minimal" in registry.loaded_encoders()

    def test_loaded_encoders_does_not_include_unloaded_name(self, registry):
        registry.register("minimal", _MinimalEncoder)
        registry.load("minimal")
        registry.unload("minimal")
        assert "minimal" not in registry.loaded_encoders()

    def test_loaded_encoders_tracks_multiple(self, registry):
        registry.register("minimal", _MinimalEncoder)
        registry.register("audio_enc", _AudioEncoder)
        registry.load("minimal")
        registry.load("audio_enc")
        loaded = registry.loaded_encoders()
        assert "minimal" in loaded
        assert "audio_enc" in loaded

    def test_loaded_encoders_returns_list(self, registry):
        assert isinstance(registry.loaded_encoders(), list)
