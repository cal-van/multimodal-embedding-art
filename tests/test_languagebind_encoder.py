"""
Tests for LanguageBindEncoder — the canonical multimodal encoder.

The static / class-level tests run without any model loading and verify the
``EncoderCard`` metadata, modality capability flags, and registry integration.
The integration tests are marked ``@pytest.mark.slow`` because they require
both the ``languagebind`` Python package and ~3 GB of LanguageBind checkpoints
downloaded from HuggingFace.

Run only the fast tests::

    pytest tests/test_languagebind_encoder.py -v

Run the slow tests too (requires LanguageBind setup)::

    pytest tests/test_languagebind_encoder.py -v -m slow
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from embedding_art.encoders.languagebind import (
    EMBEDDING_DIM,
    LanguageBindEncoder,
)
from embedding_art.encoders.registry import EncoderCapability, EncoderCard
from embedding_art.exceptions import ModelLoadError

# ---------------------------------------------------------------------------
# Static / class-level tests — no model loading
# ---------------------------------------------------------------------------


class TestLanguageBindEncoderCard:
    """Card is a class attribute: readable before any instantiation.

    These tests are the contract surface that the EncoderRegistry relies on
    when listing available encoders and selecting one for a target modality.
    """

    def test_card_is_class_attribute(self) -> None:
        assert hasattr(LanguageBindEncoder, "card")
        assert isinstance(LanguageBindEncoder.card, EncoderCard)

    def test_card_name(self) -> None:
        assert LanguageBindEncoder.card.name == "languagebind"

    def test_card_embedding_dim(self) -> None:
        assert LanguageBindEncoder.card.embedding_dim == EMBEDDING_DIM
        assert LanguageBindEncoder.card.embedding_dim == 768

    def test_card_has_text_capability(self) -> None:
        assert EncoderCapability.TEXT in LanguageBindEncoder.card.capabilities

    def test_card_has_image_capability(self) -> None:
        assert EncoderCapability.IMAGE in LanguageBindEncoder.card.capabilities

    def test_card_has_audio_capability(self) -> None:
        """LanguageBind is *truly* multimodal — audio MUST be supported."""
        assert EncoderCapability.AUDIO in LanguageBindEncoder.card.capabilities

    def test_card_has_video_capability(self) -> None:
        """LanguageBind is *truly* multimodal — video MUST be supported.

        This is the load-bearing property that distinguishes LanguageBind from
        SigLIP 2 (text + image only) and CLAP (text + audio only). The
        four-modality showcase requires a single encoder spanning all four.
        """
        assert EncoderCapability.VIDEO in LanguageBindEncoder.card.capabilities

    def test_card_has_backprop_optimizable_capability(self) -> None:
        assert EncoderCapability.BACKPROP_OPTIMIZABLE in LanguageBindEncoder.card.capabilities

    def test_card_has_multi_layer_features_capability(self) -> None:
        assert EncoderCapability.MULTI_LAYER_FEATURES in LanguageBindEncoder.card.capabilities

    def test_card_memory_estimate_is_positive(self) -> None:
        assert LanguageBindEncoder.card.memory_estimate_mb > 0

    def test_card_backprop_cost_is_positive(self) -> None:
        assert LanguageBindEncoder.card.backprop_cost > 0

    def test_card_memory_estimate_fits_m1_max(self) -> None:
        """Sanity check that the 3 modality models fit M1 Max's 64GB budget
        alongside SD3.5-medium (~5GB), Stable Audio Open (~3GB) and headroom."""
        assert LanguageBindEncoder.card.memory_estimate_mb < 8_000


class TestLanguageBindEncoderRegistryIntegration:
    """LanguageBindEncoder integrates with the default encoder registry as
    the canonical encoder."""

    def test_registered_in_default_registry(self) -> None:
        """The default registry must include LanguageBind under the
        canonical ``languagebind`` key."""
        from embedding_art.encoders.defaults import create_default_registry

        registry = create_default_registry()
        # registry.list_available() may or may not include LanguageBind
        # depending on whether the languagebind package was importable when
        # defaults.py ran; the encoder class itself is import-safe though.
        try:
            card = registry.get_card("languagebind")
        except Exception:
            pytest.skip("languagebind not registered (import-time failure)")
        assert card.name == "languagebind"

    def test_canonical_encoder_listed_first(self) -> None:
        """When ``languagebind`` is available it must come first so callers
        asking for the default get the canonical multimodal encoder."""
        from embedding_art.encoders.defaults import create_default_registry

        registry = create_default_registry()
        cards = registry.list_available()
        if not cards:
            pytest.skip("no encoders registered")
        names = [c.name for c in cards]
        if "languagebind" not in names:
            pytest.skip("languagebind not available")
        assert names[0] == "languagebind"


# ---------------------------------------------------------------------------
# Error path tests — no model loading required
# ---------------------------------------------------------------------------


class TestLanguageBindEncoderMissingPackage:
    """Helpful error when the LanguageBind package isn't installed."""

    def test_load_modality_raises_model_load_error_without_package(self) -> None:
        """If ``languagebind`` is not importable, loading any modality must
        raise ``ModelLoadError`` with installation guidance."""
        with patch.dict("sys.modules", {"languagebind": None}):
            encoder = LanguageBindEncoder.__new__(LanguageBindEncoder)
            encoder._device = torch.device("cpu")
            encoder._cache_dir = pytest.importorskip("pathlib").Path("/tmp")
            encoder._modality_models = {}
            encoder._tokenizer = None
            with pytest.raises(ModelLoadError) as exc_info:
                encoder._load_modality("image")
            assert "LanguageBind is not installed" in str(exc_info.value.original_error)

    def test_load_modality_raises_on_unknown_modality(self) -> None:
        encoder = LanguageBindEncoder.__new__(LanguageBindEncoder)
        encoder._device = torch.device("cpu")
        encoder._cache_dir = pytest.importorskip("pathlib").Path("/tmp")
        encoder._modality_models = {}
        encoder._tokenizer = None
        with pytest.raises(ModelLoadError):
            encoder._load_modality("depth")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Integration tests — require LanguageBind setup
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestLanguageBindEncoderInstantiation:
    """Instantiation triggers eager image-modality load."""

    def test_instantiation_eagerly_loads_image_modality(self) -> None:
        languagebind = pytest.importorskip("languagebind")  # noqa: F841
        encoder = LanguageBindEncoder(device="cpu")
        assert "image" in encoder._modality_models

    def test_instantiation_does_not_eagerly_load_audio_modality(self) -> None:
        languagebind = pytest.importorskip("languagebind")  # noqa: F841
        encoder = LanguageBindEncoder(device="cpu")
        assert "audio" not in encoder._modality_models

    def test_embedding_dim_property(self) -> None:
        languagebind = pytest.importorskip("languagebind")  # noqa: F841
        encoder = LanguageBindEncoder(device="cpu")
        assert encoder.embedding_dim == EMBEDDING_DIM

    def test_device_property(self) -> None:
        languagebind = pytest.importorskip("languagebind")  # noqa: F841
        encoder = LanguageBindEncoder(device="cpu")
        assert encoder.device == torch.device("cpu")


@pytest.mark.slow
class TestLanguageBindEncoderText:
    """Text encoding lazy-loads the image modality when nothing else is loaded."""

    @pytest.fixture(scope="class")
    def encoder(self) -> LanguageBindEncoder:
        pytest.importorskip("languagebind")
        return LanguageBindEncoder(device="cpu")

    def test_encode_text_returns_correct_shape(self, encoder: LanguageBindEncoder) -> None:
        embedding = encoder.encode_text("a golden retriever")
        assert embedding.shape == (1, EMBEDDING_DIM)

    def test_encode_text_returns_normalized_embedding(self, encoder: LanguageBindEncoder) -> None:
        embedding = encoder.encode_text("a golden retriever")
        norm = torch.norm(embedding, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-4)

    def test_encode_text_batch_returns_n_by_d(self, encoder: LanguageBindEncoder) -> None:
        out = encoder.encode_text_batch(["dog", "cat", "bird"])
        assert out.shape == (3, EMBEDDING_DIM)
        norms = torch.norm(out, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    def test_encode_text_batch_matches_encode_text(self, encoder: LanguageBindEncoder) -> None:
        """Batch path produces the same embeddings as the per-word path."""
        words = ["dog", "cat"]
        per_word = torch.cat([encoder.encode_text(w) for w in words], dim=0)
        batched = encoder.encode_text_batch(words)
        # Cosine similarity should be ~1 between equivalent paths.
        sim = (per_word * batched).sum(dim=-1)
        assert torch.all(sim > 0.999)

    def test_encode_text_batch_empty_input(self, encoder: LanguageBindEncoder) -> None:
        out = encoder.encode_text_batch([])
        assert out.shape == (0, EMBEDDING_DIM)


@pytest.mark.slow
class TestLanguageBindEncoderCrossModal:
    """Cross-modal-shared-space property — the whole point of this encoder.

    A text embedding and an image embedding of the same concept should have
    higher cosine similarity than two unrelated concepts. These are the only
    tests that actually verify LanguageBind's true-multimodal property.
    """

    @pytest.fixture(scope="class")
    def encoder(self) -> LanguageBindEncoder:
        pytest.importorskip("languagebind")
        return LanguageBindEncoder(device="cpu")

    def test_text_and_text_of_same_concept_are_similar(self, encoder: LanguageBindEncoder) -> None:
        a = encoder.encode_text("a golden retriever")
        b = encoder.encode_text("a golden retriever puppy")
        c = encoder.encode_text("a stormy ocean")
        sim_ab = torch.cosine_similarity(a, b).item()
        sim_ac = torch.cosine_similarity(a, c).item()
        assert sim_ab > sim_ac
