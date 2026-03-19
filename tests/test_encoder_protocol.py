"""
Tests for the v2 encoder protocol extensions.

Covers:
- MockEncoder.card returns an EncoderCard with the expected capabilities
- MockEncoder.encode(ConceptSpec) dispatches correctly and returns a Concept
- MockEncoder.encode_for_optimization returns a tensor of the right shape
- MockEncoder.unload() is a harmless no-op
- Backward-compat: old encode_text / encode_image still work unchanged
"""

import pytest
import torch

from embedding_art.core.concept import Concept
from embedding_art.core.concept_spec import ConceptSpec
from embedding_art.encoders.registry import EncoderCapability, EncoderCard

# ---------------------------------------------------------------------------
# Helpers — import MockEncoder from conftest
# ---------------------------------------------------------------------------
# conftest.py defines MockEncoder as a plain class (not a fixture), so we can
# import it directly.
from tests.conftest import MockEncoder

# ---------------------------------------------------------------------------
# card property
# ---------------------------------------------------------------------------


class TestMockEncoderCard:
    def test_card_returns_encoder_card(self):
        enc = MockEncoder()
        assert isinstance(enc.card, EncoderCard)

    def test_card_name(self):
        enc = MockEncoder()
        assert enc.card.name == "mock"

    def test_card_has_text_capability(self):
        enc = MockEncoder()
        assert EncoderCapability.TEXT in enc.card.capabilities

    def test_card_has_image_capability(self):
        enc = MockEncoder()
        assert EncoderCapability.IMAGE in enc.card.capabilities

    def test_card_has_audio_capability(self):
        enc = MockEncoder()
        assert EncoderCapability.AUDIO in enc.card.capabilities

    def test_card_has_video_capability(self):
        enc = MockEncoder()
        assert EncoderCapability.VIDEO in enc.card.capabilities

    def test_card_has_backprop_optimizable_capability(self):
        enc = MockEncoder()
        assert EncoderCapability.BACKPROP_OPTIMIZABLE in enc.card.capabilities

    def test_card_embedding_dim_matches_encoder(self):
        enc = MockEncoder(embedding_dim=1024)
        assert enc.card.embedding_dim == 1024

    def test_card_memory_estimate_mb(self):
        enc = MockEncoder()
        assert enc.card.memory_estimate_mb == 100


# ---------------------------------------------------------------------------
# encode(ConceptSpec)
# ---------------------------------------------------------------------------


class TestMockEncoderEncode:
    def test_encode_text_spec_returns_concept(self):
        enc = MockEncoder()
        spec = ConceptSpec(text="goldfish")
        result = enc.encode(spec)
        assert isinstance(result, Concept)

    def test_encode_text_spec_embedding_shape(self):
        enc = MockEncoder(embedding_dim=1024)
        spec = ConceptSpec(text="goldfish")
        result = enc.encode(spec)
        assert result.embedding.shape == (1, 1024)

    def test_encode_text_spec_embedding_is_normalized(self):
        enc = MockEncoder()
        spec = ConceptSpec(text="goldfish")
        result = enc.encode(spec)
        norm = torch.linalg.norm(result.embedding, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)

    def test_encode_text_spec_is_deterministic(self):
        enc = MockEncoder()
        spec = ConceptSpec(text="goldfish")
        r1 = enc.encode(spec)
        r2 = enc.encode(spec)
        assert torch.allclose(r1.embedding, r2.embedding)

    def test_encode_text_spec_different_texts_differ(self):
        enc = MockEncoder()
        r1 = enc.encode(ConceptSpec(text="goldfish"))
        r2 = enc.encode(ConceptSpec(text="flamingo"))
        assert not torch.allclose(r1.embedding, r2.embedding)

    def test_encode_image_spec_returns_concept(self, tmp_path):
        enc = MockEncoder()
        # A real Path isn't required for the mock — it hashes the path string.
        spec = ConceptSpec(image=tmp_path / "test.png")
        result = enc.encode(spec)
        assert isinstance(result, Concept)

    def test_encode_image_spec_embedding_shape(self, tmp_path):
        enc = MockEncoder(embedding_dim=1024)
        spec = ConceptSpec(image=tmp_path / "test.png")
        result = enc.encode(spec)
        assert result.embedding.shape == (1, 1024)

    def test_encode_empty_spec_raises_value_error(self):
        enc = MockEncoder()
        with pytest.raises(ValueError, match="no modality"):
            enc.encode(ConceptSpec())

    def test_encode_text_spec_concept_description_contains_text(self):
        enc = MockEncoder()
        result = enc.encode(ConceptSpec(text="ocean waves"))
        assert "ocean waves" in result.description


# ---------------------------------------------------------------------------
# encode_for_optimization
# ---------------------------------------------------------------------------


class TestMockEncoderEncodeForOptimization:
    def test_returns_tensor(self):
        enc = MockEncoder(embedding_dim=1024)
        tensor = torch.randn(1, 3, 64, 64)
        result = enc.encode_for_optimization(tensor)
        assert isinstance(result, torch.Tensor)

    def test_output_shape_matches_embedding_dim(self):
        enc = MockEncoder(embedding_dim=1024)
        tensor = torch.randn(1, 3, 64, 64)
        result = enc.encode_for_optimization(tensor)
        assert result.shape == (1, 1024)

    def test_output_is_normalized(self):
        enc = MockEncoder(embedding_dim=1024)
        tensor = torch.randn(1, 3, 64, 64)
        result = enc.encode_for_optimization(tensor)
        norm = torch.linalg.norm(result, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)

    def test_deterministic_for_same_input(self):
        enc = MockEncoder(embedding_dim=1024)
        tensor = torch.ones(1, 3, 32, 32)
        r1 = enc.encode_for_optimization(tensor)
        r2 = enc.encode_for_optimization(tensor)
        assert torch.allclose(r1, r2)


# ---------------------------------------------------------------------------
# unload
# ---------------------------------------------------------------------------


class TestMockEncoderUnload:
    def test_unload_does_not_raise(self):
        enc = MockEncoder()
        enc.unload()  # should be a harmless no-op

    def test_encoder_still_usable_after_unload(self):
        enc = MockEncoder()
        enc.unload()
        result = enc.encode_text("test")
        assert result.shape == (1, 1024)


# ---------------------------------------------------------------------------
# Backward compatibility — old encode_text / encode_image interface
# ---------------------------------------------------------------------------


class TestMockEncoderBackwardCompat:
    def test_encode_text_still_works(self):
        enc = MockEncoder()
        result = enc.encode_text("goldfish")
        assert result.shape == (1, 1024)

    def test_encode_image_path_still_works(self, tmp_path):
        from pathlib import Path

        enc = MockEncoder()
        result = enc.encode_image(Path(tmp_path / "test.png"))
        assert result.shape == (1, 1024)

    def test_encode_audio_still_works(self, tmp_path):
        from pathlib import Path

        enc = MockEncoder()
        result = enc.encode_audio(Path(tmp_path / "test.wav"))
        assert result.shape == (1, 1024)

    def test_encode_video_still_works(self, tmp_path):
        from pathlib import Path

        enc = MockEncoder()
        result = enc.encode_video(Path(tmp_path / "test.mp4"))
        assert result.shape == (1, 1024)

    def test_encode_text_result_is_normalized(self):
        enc = MockEncoder()
        result = enc.encode_text("goldfish")
        norm = torch.linalg.norm(result, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)

    def test_encode_text_deterministic(self):
        enc = MockEncoder()
        r1 = enc.encode_text("goldfish")
        r2 = enc.encode_text("goldfish")
        assert torch.allclose(r1, r2)
