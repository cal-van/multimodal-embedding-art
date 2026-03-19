"""
Tests for SigLIP2Encoder.

All tests require downloading google/siglip2-so400m-patch14-384 from HuggingFace
(~1.6 GB) and are therefore marked @pytest.mark.slow.  Run with:

    pytest tests/test_siglip2_encoder.py -v -m slow
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

from embedding_art.encoders.registry import EncoderCapability, EncoderCard
from embedding_art.encoders.siglip2 import EMBEDDING_DIM, SigLIP2Encoder
from embedding_art.exceptions import EncoderError


# ---------------------------------------------------------------------------
# Static / class-level tests — no model loading required
# ---------------------------------------------------------------------------


class TestSigLIP2EncoderCard:
    """Card is a class attribute: readable before any instantiation."""

    def test_card_is_class_attribute(self) -> None:
        """card must be accessible on the class itself, not just on instances."""
        assert hasattr(SigLIP2Encoder, "card")
        assert isinstance(SigLIP2Encoder.card, EncoderCard)

    def test_card_name(self) -> None:
        assert SigLIP2Encoder.card.name == "siglip2-so400m"

    def test_card_embedding_dim(self) -> None:
        assert SigLIP2Encoder.card.embedding_dim == EMBEDDING_DIM
        assert SigLIP2Encoder.card.embedding_dim == 1152

    def test_card_has_text_capability(self) -> None:
        assert EncoderCapability.TEXT in SigLIP2Encoder.card.capabilities

    def test_card_has_image_capability(self) -> None:
        assert EncoderCapability.IMAGE in SigLIP2Encoder.card.capabilities

    def test_card_has_backprop_optimizable_capability(self) -> None:
        assert EncoderCapability.BACKPROP_OPTIMIZABLE in SigLIP2Encoder.card.capabilities

    def test_card_has_multi_layer_features_capability(self) -> None:
        assert EncoderCapability.MULTI_LAYER_FEATURES in SigLIP2Encoder.card.capabilities

    def test_card_does_not_have_audio_capability(self) -> None:
        assert EncoderCapability.AUDIO not in SigLIP2Encoder.card.capabilities

    def test_card_does_not_have_video_capability(self) -> None:
        assert EncoderCapability.VIDEO not in SigLIP2Encoder.card.capabilities

    def test_card_memory_estimate_is_positive(self) -> None:
        assert SigLIP2Encoder.card.memory_estimate_mb > 0

    def test_card_backprop_cost_is_positive(self) -> None:
        assert SigLIP2Encoder.card.backprop_cost > 0


# ---------------------------------------------------------------------------
# Integration tests — require model download
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSigLIP2EncoderLoading:
    """Model loading and basic properties."""

    def test_instantiation_sets_device(self) -> None:
        encoder = SigLIP2Encoder(device="cpu")
        assert encoder.device == torch.device("cpu")

    def test_embedding_dim_property(self) -> None:
        encoder = SigLIP2Encoder(device="cpu")
        assert encoder.embedding_dim == EMBEDDING_DIM

    def test_model_loaded_after_init(self) -> None:
        encoder = SigLIP2Encoder(device="cpu")
        assert encoder._model is not None
        assert encoder._processor is not None
        assert encoder._tokenizer is not None


@pytest.mark.slow
class TestSigLIP2EncoderText:
    """Text encoding."""

    @pytest.fixture(scope="class")
    def encoder(self) -> SigLIP2Encoder:
        return SigLIP2Encoder(device="cpu")

    def test_encode_text_returns_correct_shape(self, encoder: SigLIP2Encoder) -> None:
        """encode_text should return [1, 1152] tensor."""
        embedding = encoder.encode_text("a golden retriever")
        assert embedding.shape == (1, EMBEDDING_DIM)

    def test_encode_text_returns_normalized_embedding(self, encoder: SigLIP2Encoder) -> None:
        """Output must be L2-normalised (unit norm)."""
        embedding = encoder.encode_text("a golden retriever")
        norm = torch.norm(embedding, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)

    def test_encode_text_different_texts_differ(self, encoder: SigLIP2Encoder) -> None:
        """Semantically different texts should not produce identical embeddings."""
        emb1 = encoder.encode_text("a golden retriever")
        emb2 = encoder.encode_text("a stormy ocean")
        sim = torch.cosine_similarity(emb1, emb2).item()
        assert sim < 0.99

    def test_encode_text_deterministic(self, encoder: SigLIP2Encoder) -> None:
        """Same text should always produce the same embedding."""
        text = "a golden retriever"
        assert torch.allclose(encoder.encode_text(text), encoder.encode_text(text))


@pytest.mark.slow
class TestSigLIP2EncoderImage:
    """Image encoding via PIL and file path."""

    @pytest.fixture(scope="class")
    def encoder(self) -> SigLIP2Encoder:
        return SigLIP2Encoder(device="cpu")

    @pytest.fixture
    def pil_image(self) -> Image.Image:
        return Image.new("RGB", (384, 384), color=(200, 100, 50))

    def test_encode_image_pil_returns_correct_shape(
        self, encoder: SigLIP2Encoder, pil_image: Image.Image
    ) -> None:
        """encode_image(PIL) should return [1, 1152]."""
        embedding = encoder.encode_image(pil_image)
        assert embedding.shape == (1, EMBEDDING_DIM)

    def test_encode_image_pil_returns_normalized_embedding(
        self, encoder: SigLIP2Encoder, pil_image: Image.Image
    ) -> None:
        """Output must be unit-normalised."""
        embedding = encoder.encode_image(pil_image)
        norm = torch.norm(embedding, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)

    def test_encode_image_path_returns_correct_shape(self, encoder: SigLIP2Encoder) -> None:
        """encode_image(Path) should return [1, 1152]."""
        img = Image.new("RGB", (384, 384), color=(50, 100, 200))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f.name)
            image_path = Path(f.name)
        try:
            embedding = encoder.encode_image(image_path)
            assert embedding.shape == (1, EMBEDDING_DIM)
        finally:
            image_path.unlink()


@pytest.mark.slow
class TestSigLIP2EncoderForOptimization:
    """encode_for_optimization — differentiable path."""

    @pytest.fixture(scope="class")
    def encoder(self) -> SigLIP2Encoder:
        return SigLIP2Encoder(device="cpu")

    def test_returns_correct_shape(self, encoder: SigLIP2Encoder) -> None:
        """Should return [B, 1152] for a [B, 3, H, W] input."""
        tensor = torch.rand(1, 3, 512, 512)
        embedding = encoder.encode_for_optimization(tensor)
        assert embedding.shape == (1, EMBEDDING_DIM)

    def test_returns_normalized_embedding(self, encoder: SigLIP2Encoder) -> None:
        tensor = torch.rand(1, 3, 512, 512)
        embedding = encoder.encode_for_optimization(tensor)
        norm = torch.norm(embedding, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)

    def test_handles_native_resolution(self, encoder: SigLIP2Encoder) -> None:
        """Should handle inputs already at 384×384 without error."""
        tensor = torch.rand(1, 3, 384, 384)
        embedding = encoder.encode_for_optimization(tensor)
        assert embedding.shape == (1, EMBEDDING_DIM)

    def test_gradients_flow(self, encoder: SigLIP2Encoder) -> None:
        """Gradients must propagate through encode_for_optimization for optimization loops."""
        tensor = torch.rand(1, 3, 384, 384, requires_grad=True)
        embedding = encoder.encode_for_optimization(tensor)
        # Scalar loss: cosine sim with a random target
        target = torch.randn(1, EMBEDDING_DIM)
        target = target / target.norm(dim=-1, keepdim=True)
        loss = -(embedding * target).sum()
        loss.backward()
        assert tensor.grad is not None, "Gradient must flow back to input tensor"
        assert tensor.grad.abs().sum() > 0, "Gradient must be non-zero"


@pytest.mark.slow
class TestSigLIP2EncoderLayerFeatures:
    """get_layer_features — multi-layer feature extraction."""

    @pytest.fixture(scope="class")
    def encoder(self) -> SigLIP2Encoder:
        return SigLIP2Encoder(device="cpu")

    @pytest.fixture
    def image_tensor(self) -> torch.Tensor:
        return torch.rand(1, 3, 384, 384)

    def test_returns_dict(self, encoder: SigLIP2Encoder, image_tensor: torch.Tensor) -> None:
        """get_layer_features should return a dict."""
        result = encoder.get_layer_features(image_tensor)
        assert isinstance(result, dict)

    def test_returns_layer_features_instances(
        self, encoder: SigLIP2Encoder, image_tensor: torch.Tensor
    ) -> None:
        """Every value must be a LayerFeatures instance."""
        from embedding_art.encoders.features import LayerFeatures

        result = encoder.get_layer_features(image_tensor)
        assert len(result) > 0
        for v in result.values():
            assert isinstance(v, LayerFeatures)

    def test_layer_indices_are_every_fourth(
        self, encoder: SigLIP2Encoder, image_tensor: torch.Tensor
    ) -> None:
        """Returned layer indices must be a subset of range(0, n, 4) ∪ {last}."""
        result = encoder.get_layer_features(image_tensor)
        indices = sorted(result.keys())
        # First index should be 0
        assert indices[0] == 0
        # All consecutive gaps should be exactly 4 (except possibly the last one)
        for i in range(len(indices) - 1):
            gap = indices[i + 1] - indices[i]
            assert gap == 4 or i == len(indices) - 2, (
                f"Expected gap of 4 between indices, got {gap} between "
                f"{indices[i]} and {indices[i+1]}"
            )

    def test_last_layer_always_included(
        self, encoder: SigLIP2Encoder, image_tensor: torch.Tensor
    ) -> None:
        """The final transformer layer must always be present in the result."""
        result = encoder.get_layer_features(image_tensor)
        indices = sorted(result.keys())
        max_idx = max(indices)
        # The last index returned should equal n_layers - 1; we verify it by
        # checking that there is no gap from max_idx to the theoretical end.
        # We can't know n_layers without running forward pass — so just assert
        # the last index in results matches what the model reports.
        # The simplest check: re-run with output_hidden_states=True and compare.
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        normalized = (image_tensor - mean) / std
        import torch.nn.functional as F

        if normalized.shape[-2:] != (384, 384):
            normalized = F.interpolate(normalized, size=(384, 384), mode="bilinear", align_corners=False)
        with torch.no_grad():
            vision_out = encoder._model.vision_model(
                pixel_values=normalized, output_hidden_states=True
            )
        n_layers = len(vision_out.hidden_states)
        assert max_idx == n_layers - 1

    def test_layer_features_have_correct_batch_dim(
        self, encoder: SigLIP2Encoder, image_tensor: torch.Tensor
    ) -> None:
        """Each LayerFeatures tensor should have batch dim 1."""
        result = encoder.get_layer_features(image_tensor)
        for lf in result.values():
            assert lf.tensor.shape[0] == 1

    def test_accepts_3d_input(self, encoder: SigLIP2Encoder) -> None:
        """get_layer_features should accept a [3, H, W] tensor (no batch dim)."""
        tensor_3d = torch.rand(3, 384, 384)
        result = encoder.get_layer_features(tensor_3d)
        assert isinstance(result, dict)
        assert len(result) > 0


@pytest.mark.slow
class TestSigLIP2EncoderEncode:
    """encode(ConceptSpec) — unified dispatch method."""

    @pytest.fixture(scope="class")
    def encoder(self) -> SigLIP2Encoder:
        return SigLIP2Encoder(device="cpu")

    def test_encode_text_spec_returns_concept(self, encoder: SigLIP2Encoder) -> None:
        """encode(ConceptSpec(text=...)) should return a Concept with correct shape."""
        from embedding_art.core.concept import Concept
        from embedding_art.core.concept_spec import ConceptSpec

        concept = encoder.encode(ConceptSpec(text="ocean waves"))
        assert isinstance(concept, Concept)
        assert concept.embedding.shape == (1, EMBEDDING_DIM)

    def test_encode_text_spec_embedding_is_normalized(self, encoder: SigLIP2Encoder) -> None:
        from embedding_art.core.concept_spec import ConceptSpec

        concept = encoder.encode(ConceptSpec(text="ocean waves"))
        norm = torch.norm(concept.embedding, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)

    def test_encode_text_spec_description_contains_text(self, encoder: SigLIP2Encoder) -> None:
        from embedding_art.core.concept_spec import ConceptSpec

        concept = encoder.encode(ConceptSpec(text="ocean waves"))
        assert "ocean waves" in concept.description

    def test_encode_image_spec_sets_source_input(self, encoder: SigLIP2Encoder) -> None:
        """encode(ConceptSpec(image=...)) should populate concept.source_input."""
        from embedding_art.core.concept_spec import ConceptSpec

        img = Image.new("RGB", (384, 384), color=(100, 150, 200))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f.name)
            image_path = Path(f.name)
        try:
            concept = encoder.encode(ConceptSpec(image=image_path))
            assert concept.source_input is not None
            # source_input should be a [1, 3, H, W] tensor
            assert concept.source_input.dim() == 4
            assert concept.source_input.shape[1] == 3
        finally:
            image_path.unlink()

    def test_encode_image_spec_returns_correct_shape(self, encoder: SigLIP2Encoder) -> None:
        from embedding_art.core.concept_spec import ConceptSpec

        img = Image.new("RGB", (384, 384), color=(100, 150, 200))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f.name)
            image_path = Path(f.name)
        try:
            concept = encoder.encode(ConceptSpec(image=image_path))
            assert concept.embedding.shape == (1, EMBEDDING_DIM)
        finally:
            image_path.unlink()

    def test_encode_empty_spec_raises(self, encoder: SigLIP2Encoder) -> None:
        """A ConceptSpec with no modality should raise ValueError."""
        from embedding_art.core.concept_spec import ConceptSpec

        with pytest.raises(ValueError):
            encoder.encode(ConceptSpec())


@pytest.mark.slow
class TestSigLIP2EncoderUnsupportedModalities:
    """Unsupported modalities must raise EncoderError (not NotImplementedError)."""

    @pytest.fixture(scope="class")
    def encoder(self) -> SigLIP2Encoder:
        return SigLIP2Encoder(device="cpu")

    def test_encode_audio_raises_encoder_error(self, encoder: SigLIP2Encoder) -> None:
        with pytest.raises(EncoderError) as exc_info:
            encoder.encode_audio(Path("fake_audio.wav"))
        assert exc_info.value.modality == "audio"

    def test_encode_video_raises_encoder_error(self, encoder: SigLIP2Encoder) -> None:
        with pytest.raises(EncoderError) as exc_info:
            encoder.encode_video(Path("fake_video.mp4"))
        assert exc_info.value.modality == "video"


@pytest.mark.slow
class TestSigLIP2EncoderUnload:
    """unload() releases model resources."""

    def test_unload_clears_model(self) -> None:
        """After unload(), _model should be None."""
        encoder = SigLIP2Encoder(device="cpu")
        assert encoder._model is not None
        encoder.unload()
        assert encoder._model is None

    def test_unload_clears_processor(self) -> None:
        """After unload(), _processor should be None."""
        encoder = SigLIP2Encoder(device="cpu")
        encoder.unload()
        assert encoder._processor is None

    def test_unload_clears_tokenizer(self) -> None:
        """After unload(), _tokenizer should be None."""
        encoder = SigLIP2Encoder(device="cpu")
        encoder.unload()
        assert encoder._tokenizer is None
