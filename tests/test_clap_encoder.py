"""
Tests for CLAPEncoder.

All tests require downloading laion/larger_clap_general from HuggingFace
(~1.2 GB) and are therefore marked @pytest.mark.slow.  Run with:

    pytest tests/test_clap_encoder.py -v -m slow
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from embedding_art.encoders.clap import EMBEDDING_DIM, CLAPEncoder
from embedding_art.encoders.registry import EncoderCapability, EncoderCard
from embedding_art.exceptions import EncoderError

# ---------------------------------------------------------------------------
# Static / class-level tests — no model loading required
# ---------------------------------------------------------------------------


class TestCLAPEncoderCard:
    """Card is a class attribute: readable before any instantiation."""

    def test_card_is_class_attribute(self) -> None:
        """card must be accessible on the class itself, not just on instances."""
        assert hasattr(CLAPEncoder, "card")
        assert isinstance(CLAPEncoder.card, EncoderCard)

    def test_card_name(self) -> None:
        assert CLAPEncoder.card.name == "clap-general"

    def test_card_embedding_dim(self) -> None:
        assert CLAPEncoder.card.embedding_dim == EMBEDDING_DIM
        assert CLAPEncoder.card.embedding_dim == 512

    def test_card_has_text_capability(self) -> None:
        assert EncoderCapability.TEXT in CLAPEncoder.card.capabilities

    def test_card_has_audio_capability(self) -> None:
        assert EncoderCapability.AUDIO in CLAPEncoder.card.capabilities

    def test_card_has_backprop_optimizable_capability(self) -> None:
        assert EncoderCapability.BACKPROP_OPTIMIZABLE in CLAPEncoder.card.capabilities

    def test_card_does_not_have_image_capability(self) -> None:
        assert EncoderCapability.IMAGE not in CLAPEncoder.card.capabilities

    def test_card_does_not_have_video_capability(self) -> None:
        assert EncoderCapability.VIDEO not in CLAPEncoder.card.capabilities

    def test_card_memory_estimate_is_positive(self) -> None:
        assert CLAPEncoder.card.memory_estimate_mb > 0

    def test_card_backprop_cost_is_positive(self) -> None:
        assert CLAPEncoder.card.backprop_cost > 0


# ---------------------------------------------------------------------------
# Integration tests — require model download
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestCLAPEncoderLoading:
    """Model loading and basic properties."""

    def test_instantiation_sets_device(self) -> None:
        encoder = CLAPEncoder(device="cpu")
        assert encoder.device == torch.device("cpu")

    def test_embedding_dim_property(self) -> None:
        encoder = CLAPEncoder(device="cpu")
        assert encoder.embedding_dim == EMBEDDING_DIM

    def test_model_loaded_after_init(self) -> None:
        encoder = CLAPEncoder(device="cpu")
        assert encoder._model is not None
        assert encoder._processor is not None


@pytest.mark.slow
class TestCLAPEncoderText:
    """Text encoding."""

    @pytest.fixture(scope="class")
    def encoder(self) -> CLAPEncoder:
        return CLAPEncoder(device="cpu")

    def test_encode_text_returns_correct_shape(self, encoder: CLAPEncoder) -> None:
        """encode_text should return [1, 512] tensor."""
        embedding = encoder.encode_text("a dog barking")
        assert embedding.shape == (1, EMBEDDING_DIM)

    def test_encode_text_returns_normalized_embedding(self, encoder: CLAPEncoder) -> None:
        """Output must be L2-normalised (unit norm)."""
        embedding = encoder.encode_text("a dog barking")
        norm = torch.norm(embedding, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)

    def test_encode_text_different_texts_differ(self, encoder: CLAPEncoder) -> None:
        """Semantically different texts should not produce identical embeddings."""
        emb1 = encoder.encode_text("a dog barking")
        emb2 = encoder.encode_text("ocean waves crashing")
        sim = torch.cosine_similarity(emb1, emb2).item()
        assert sim < 0.99

    def test_encode_text_deterministic(self, encoder: CLAPEncoder) -> None:
        """Same text should always produce the same embedding."""
        text = "a dog barking"
        assert torch.allclose(encoder.encode_text(text), encoder.encode_text(text))


@pytest.mark.slow
class TestCLAPEncoderAudio:
    """Audio encoding via file path and tensor."""

    @pytest.fixture(scope="class")
    def encoder(self) -> CLAPEncoder:
        return CLAPEncoder(device="cpu")

    @pytest.fixture
    def audio_file(self) -> Path:
        """Write a short synthetic WAV file for testing."""
        import wave

        # 0.5 s of 440 Hz sine wave at 48 kHz, mono, 16-bit PCM.
        sample_rate = 48_000
        duration_s = 0.5
        n_samples = int(sample_rate * duration_s)
        t = np.linspace(0, duration_s, n_samples, endpoint=False)
        samples = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = Path(f.name)

        with wave.open(str(path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(samples.tobytes())

        yield path
        path.unlink(missing_ok=True)

    def test_encode_audio_file_returns_correct_shape(
        self, encoder: CLAPEncoder, audio_file: Path
    ) -> None:
        """encode_audio(Path) should return [1, 512]."""
        embedding = encoder.encode_audio(audio_file)
        assert embedding.shape == (1, EMBEDDING_DIM)

    def test_encode_audio_file_returns_normalized_embedding(
        self, encoder: CLAPEncoder, audio_file: Path
    ) -> None:
        """Output must be unit-normalised."""
        embedding = encoder.encode_audio(audio_file)
        norm = torch.norm(embedding, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)

    def test_encode_audio_tensor_returns_correct_shape(self, encoder: CLAPEncoder) -> None:
        """encode_audio(Tensor) should return [1, 512]."""
        # 1-D mono waveform at 48 kHz for 0.5 s.
        waveform = torch.zeros(24_000)
        embedding = encoder.encode_audio(waveform)
        assert embedding.shape == (1, EMBEDDING_DIM)

    def test_encode_audio_tensor_returns_normalized_embedding(self, encoder: CLAPEncoder) -> None:
        """Tensor audio output must be unit-normalised."""
        waveform = torch.zeros(24_000)
        embedding = encoder.encode_audio(waveform)
        norm = torch.norm(embedding, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)


@pytest.mark.slow
class TestCLAPEncoderEncode:
    """encode(ConceptSpec) — unified dispatch method."""

    @pytest.fixture(scope="class")
    def encoder(self) -> CLAPEncoder:
        return CLAPEncoder(device="cpu")

    @pytest.fixture
    def audio_file(self) -> Path:
        """Write a short synthetic WAV file for testing."""
        import wave

        sample_rate = 48_000
        n_samples = int(sample_rate * 0.5)
        t = np.linspace(0, 0.5, n_samples, endpoint=False)
        samples = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = Path(f.name)

        with wave.open(str(path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(samples.tobytes())

        yield path
        path.unlink(missing_ok=True)

    def test_encode_text_spec_returns_concept(self, encoder: CLAPEncoder) -> None:
        """encode(ConceptSpec(text=...)) should return a Concept with correct shape."""
        from embedding_art.core.concept import Concept
        from embedding_art.core.concept_spec import ConceptSpec

        concept = encoder.encode(ConceptSpec(text="ocean waves"))
        assert isinstance(concept, Concept)
        assert concept.embedding.shape == (1, EMBEDDING_DIM)

    def test_encode_text_spec_embedding_is_normalized(self, encoder: CLAPEncoder) -> None:
        from embedding_art.core.concept_spec import ConceptSpec

        concept = encoder.encode(ConceptSpec(text="ocean waves"))
        norm = torch.norm(concept.embedding, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)

    def test_encode_text_spec_description_contains_text(self, encoder: CLAPEncoder) -> None:
        from embedding_art.core.concept_spec import ConceptSpec

        concept = encoder.encode(ConceptSpec(text="ocean waves"))
        assert "ocean waves" in concept.description

    def test_encode_audio_spec_returns_concept(
        self, encoder: CLAPEncoder, audio_file: Path
    ) -> None:
        """encode(ConceptSpec(audio=...)) should return a Concept with correct shape."""
        from embedding_art.core.concept import Concept
        from embedding_art.core.concept_spec import ConceptSpec

        concept = encoder.encode(ConceptSpec(audio=audio_file))
        assert isinstance(concept, Concept)
        assert concept.embedding.shape == (1, EMBEDDING_DIM)

    def test_encode_audio_spec_description_contains_filename(
        self, encoder: CLAPEncoder, audio_file: Path
    ) -> None:
        from embedding_art.core.concept_spec import ConceptSpec

        concept = encoder.encode(ConceptSpec(audio=audio_file))
        assert audio_file.name in concept.description

    def test_encode_combined_text_and_audio_returns_normalized_concept(
        self, encoder: CLAPEncoder, audio_file: Path
    ) -> None:
        """encode(ConceptSpec(text=..., audio=...)) should combine and re-normalise."""
        from embedding_art.core.concept_spec import ConceptSpec

        concept = encoder.encode(ConceptSpec(text="dog barking", audio=audio_file))
        norm = torch.norm(concept.embedding, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)

    def test_encode_empty_spec_raises(self, encoder: CLAPEncoder) -> None:
        """A ConceptSpec with no supported modality should raise ValueError."""
        from embedding_art.core.concept_spec import ConceptSpec

        with pytest.raises(ValueError):
            encoder.encode(ConceptSpec())

    def test_encode_image_only_spec_raises(self, encoder: CLAPEncoder) -> None:
        """A ConceptSpec with only image set should raise ValueError."""
        from embedding_art.core.concept_spec import ConceptSpec

        with pytest.raises(ValueError):
            encoder.encode(ConceptSpec(image=Path("fake.png")))


@pytest.mark.slow
class TestCLAPEncoderUnsupportedModalities:
    """Unsupported modalities must raise EncoderError (not NotImplementedError)."""

    @pytest.fixture(scope="class")
    def encoder(self) -> CLAPEncoder:
        return CLAPEncoder(device="cpu")

    def test_encode_image_raises_encoder_error(self, encoder: CLAPEncoder) -> None:
        with pytest.raises(EncoderError) as exc_info:
            encoder.encode_image(Path("fake_image.png"))
        assert exc_info.value.modality == "image"

    def test_encode_video_raises_encoder_error(self, encoder: CLAPEncoder) -> None:
        with pytest.raises(EncoderError) as exc_info:
            encoder.encode_video(Path("fake_video.mp4"))
        assert exc_info.value.modality == "video"


@pytest.mark.slow
class TestCLAPEncoderUnload:
    """unload() releases model resources."""

    def test_unload_clears_model(self) -> None:
        """After unload(), _model should be None."""
        encoder = CLAPEncoder(device="cpu")
        assert encoder._model is not None
        encoder.unload()
        assert encoder._model is None

    def test_unload_clears_processor(self) -> None:
        """After unload(), _processor should be None."""
        encoder = CLAPEncoder(device="cpu")
        encoder.unload()
        assert encoder._processor is None
