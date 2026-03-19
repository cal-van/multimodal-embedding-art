"""
Tests for new v2 exception types.

Covers: EncoderNotFoundError, EncoderCapabilityError, SAENotTrainedError,
MemoryBudgetExceededError, FeatureNotFoundError.

Following Red-Green-Refactor TDD — these are written BEFORE the implementation.
"""

import pytest

from embedding_art.exceptions import (
    EmbeddingArtError,
    EncoderCapabilityError,
    EncoderNotFoundError,
    FeatureNotFoundError,
    MemoryBudgetExceededError,
    SAENotTrainedError,
)

# ---------------------------------------------------------------------------
# EncoderNotFoundError
# ---------------------------------------------------------------------------


class TestEncoderNotFoundError:
    def test_inherits_from_embedding_art_error(self):
        err = EncoderNotFoundError("clip", ["imagebind", "wav2vec"])
        assert isinstance(err, EmbeddingArtError)

    def test_stores_name(self):
        err = EncoderNotFoundError("clip", ["imagebind", "wav2vec"])
        assert err.name == "clip"

    def test_stores_available(self):
        err = EncoderNotFoundError("clip", ["imagebind", "wav2vec"])
        assert err.available == ["imagebind", "wav2vec"]

    def test_message_contains_encoder_name(self):
        err = EncoderNotFoundError("clip", ["imagebind", "wav2vec"])
        assert "clip" in str(err)

    def test_message_lists_available_encoders(self):
        err = EncoderNotFoundError("clip", ["imagebind", "wav2vec"])
        message = str(err)
        assert "imagebind" in message
        assert "wav2vec" in message

    def test_message_with_empty_available(self):
        err = EncoderNotFoundError("clip", [])
        # Should not raise and should still mention the name
        assert "clip" in str(err)

    def test_is_exception(self):
        err = EncoderNotFoundError("clip", ["imagebind"])
        with pytest.raises(EncoderNotFoundError):
            raise err


# ---------------------------------------------------------------------------
# EncoderCapabilityError
# ---------------------------------------------------------------------------


class TestEncoderCapabilityError:
    def test_inherits_from_embedding_art_error(self):
        err = EncoderCapabilityError("imagebind", "encode_audio")
        assert isinstance(err, EmbeddingArtError)

    def test_stores_encoder_name(self):
        err = EncoderCapabilityError("imagebind", "encode_audio")
        assert err.encoder_name == "imagebind"

    def test_stores_capability(self):
        err = EncoderCapabilityError("imagebind", "encode_audio")
        assert err.capability == "encode_audio"

    def test_message_contains_encoder_name(self):
        err = EncoderCapabilityError("imagebind", "encode_audio")
        assert "imagebind" in str(err)

    def test_message_contains_capability(self):
        err = EncoderCapabilityError("imagebind", "encode_audio")
        assert "encode_audio" in str(err)

    def test_is_exception(self):
        err = EncoderCapabilityError("imagebind", "encode_audio")
        with pytest.raises(EncoderCapabilityError):
            raise err


# ---------------------------------------------------------------------------
# SAENotTrainedError
# ---------------------------------------------------------------------------


class TestSAENotTrainedError:
    def test_inherits_from_embedding_art_error(self):
        err = SAENotTrainedError("imagebind")
        assert isinstance(err, EmbeddingArtError)

    def test_stores_encoder_name(self):
        err = SAENotTrainedError("imagebind")
        assert err.encoder_name == "imagebind"

    def test_message_contains_encoder_name(self):
        err = SAENotTrainedError("imagebind")
        assert "imagebind" in str(err)

    def test_message_mentions_sae_or_artifact(self):
        err = SAENotTrainedError("imagebind")
        message = str(err).lower()
        # Should mention SAE or artifact or training
        assert any(term in message for term in ["sae", "artifact", "train"])

    def test_is_exception(self):
        err = SAENotTrainedError("imagebind")
        with pytest.raises(SAENotTrainedError):
            raise err


# ---------------------------------------------------------------------------
# MemoryBudgetExceededError
# ---------------------------------------------------------------------------


class TestMemoryBudgetExceededError:
    def test_inherits_from_embedding_art_error(self):
        err = MemoryBudgetExceededError(8192, 4096, ["imagebind", "sdxl"])
        assert isinstance(err, EmbeddingArtError)

    def test_stores_requested_mb(self):
        err = MemoryBudgetExceededError(8192, 4096, ["imagebind", "sdxl"])
        assert err.requested_mb == 8192

    def test_stores_available_mb(self):
        err = MemoryBudgetExceededError(8192, 4096, ["imagebind", "sdxl"])
        assert err.available_mb == 4096

    def test_stores_encoders(self):
        err = MemoryBudgetExceededError(8192, 4096, ["imagebind", "sdxl"])
        assert err.encoders == ["imagebind", "sdxl"]

    def test_message_contains_requested_mb(self):
        err = MemoryBudgetExceededError(8192, 4096, ["imagebind"])
        assert "8192" in str(err)

    def test_message_contains_available_mb(self):
        err = MemoryBudgetExceededError(8192, 4096, ["imagebind"])
        assert "4096" in str(err)

    def test_message_lists_encoders(self):
        err = MemoryBudgetExceededError(8192, 4096, ["imagebind", "sdxl"])
        message = str(err)
        assert "imagebind" in message
        assert "sdxl" in message

    def test_message_with_empty_encoders(self):
        err = MemoryBudgetExceededError(8192, 4096, [])
        assert "8192" in str(err)
        assert "4096" in str(err)

    def test_is_exception(self):
        err = MemoryBudgetExceededError(8192, 4096, ["imagebind"])
        with pytest.raises(MemoryBudgetExceededError):
            raise err


# ---------------------------------------------------------------------------
# FeatureNotFoundError
# ---------------------------------------------------------------------------


class TestFeatureNotFoundError:
    def test_inherits_from_embedding_art_error(self):
        err = FeatureNotFoundError("dog_bark", ["cat_meow", "bird_chirp"])
        assert isinstance(err, EmbeddingArtError)

    def test_stores_feature_name(self):
        err = FeatureNotFoundError("dog_bark", ["cat_meow", "bird_chirp"])
        assert err.feature_name == "dog_bark"

    def test_stores_available(self):
        err = FeatureNotFoundError("dog_bark", ["cat_meow", "bird_chirp"])
        assert err.available == ["cat_meow", "bird_chirp"]

    def test_message_contains_feature_name(self):
        err = FeatureNotFoundError("dog_bark", ["cat_meow", "bird_chirp"])
        assert "dog_bark" in str(err)

    def test_message_lists_available_features(self):
        err = FeatureNotFoundError("dog_bark", ["cat_meow", "bird_chirp"])
        message = str(err)
        assert "cat_meow" in message
        assert "bird_chirp" in message

    def test_message_with_many_available_truncates(self):
        # With a large available list, the message should not be absurdly long.
        # It may truncate or summarise — we just check it doesn't blow up and
        # still mentions the missing feature name.
        many = [f"feature_{i}" for i in range(100)]
        err = FeatureNotFoundError("dog_bark", many)
        assert "dog_bark" in str(err)

    def test_message_with_empty_available(self):
        err = FeatureNotFoundError("dog_bark", [])
        assert "dog_bark" in str(err)

    def test_is_exception(self):
        err = FeatureNotFoundError("dog_bark", ["cat_meow"])
        with pytest.raises(FeatureNotFoundError):
            raise err
