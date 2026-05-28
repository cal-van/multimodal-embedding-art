"""
Unit tests for the renderers module.

Tests verify direct rendering from embeddings to outputs via
RawDecoder, ProjectionDecoder, IPAdapterRenderer, and TextRenderer.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F

from embedding_art.core.render_result import RenderResult
from embedding_art.renderers.base import DirectRenderer
from embedding_art.renderers.ip_adapter import IPAdapterRenderer
from embedding_art.renderers.projection import ProjectionDecoder
from embedding_art.renderers.raw import RawDecoder
from embedding_art.renderers.text import DEFAULT_VOCABULARY, TextRenderer
from tests.conftest import MockEncoder, MockGenerator

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def embedding_1024() -> torch.Tensor:
    gen = torch.Generator().manual_seed(42)
    emb = torch.randn(1, 1024, generator=gen)
    return F.normalize(emb, dim=-1)


@pytest.fixture
def embedding_512() -> torch.Tensor:
    gen = torch.Generator().manual_seed(99)
    emb = torch.randn(1, 512, generator=gen)
    return F.normalize(emb, dim=-1)


@pytest.fixture
def embedding_1152() -> torch.Tensor:
    gen = torch.Generator().manual_seed(77)
    emb = torch.randn(1, 1152, generator=gen)
    return F.normalize(emb, dim=-1)


# =============================================================================
# RawDecoder Tests
# =============================================================================


class TestRawDecoderOutputShape:
    def test_default_shape(self, embedding_1024: torch.Tensor) -> None:
        decoder = RawDecoder(embed_dim=1024)
        result = decoder.render(embedding_1024)
        assert result.output.shape == (1, 3, 256, 256)

    def test_custom_shape(self, embedding_1024: torch.Tensor) -> None:
        decoder = RawDecoder(embed_dim=1024, output_shape=(1, 3, 64, 64))
        result = decoder.render(embedding_1024)
        assert result.output.shape == (1, 3, 64, 64)


class TestRawDecoderDeterministic:
    def test_same_input_same_output(self, embedding_1024: torch.Tensor) -> None:
        decoder = RawDecoder(embed_dim=1024, output_shape=(1, 3, 32, 32))
        r1 = decoder.render(embedding_1024)
        r2 = decoder.render(embedding_1024)
        assert torch.allclose(r1.output, r2.output)


class TestRawDecoderOutputRange:
    def test_output_in_zero_one(self, embedding_1024: torch.Tensor) -> None:
        decoder = RawDecoder(embed_dim=1024, output_shape=(1, 3, 32, 32))
        result = decoder.render(embedding_1024)
        assert result.output.min() >= 0.0
        assert result.output.max() <= 1.0


class TestRawDecoderGradientFlow:
    def test_gradients_flow_through(self) -> None:
        decoder = RawDecoder(embed_dim=1024, output_shape=(1, 3, 16, 16))
        emb = torch.randn(1, 1024, requires_grad=True)
        result = decoder.render(emb)
        loss = result.output.sum()
        loss.backward()
        assert emb.grad is not None
        assert emb.grad.abs().sum() > 0


class TestRawDecoderDifferentEmbedDims:
    def test_512(self, embedding_512: torch.Tensor) -> None:
        decoder = RawDecoder(embed_dim=512, output_shape=(1, 3, 16, 16))
        result = decoder.render(embedding_512)
        assert result.output.shape == (1, 3, 16, 16)

    def test_1024(self, embedding_1024: torch.Tensor) -> None:
        decoder = RawDecoder(embed_dim=1024, output_shape=(1, 3, 16, 16))
        result = decoder.render(embedding_1024)
        assert result.output.shape == (1, 3, 16, 16)

    def test_1152(self, embedding_1152: torch.Tensor) -> None:
        decoder = RawDecoder(embed_dim=1152, output_shape=(1, 3, 16, 16))
        result = decoder.render(embedding_1152)
        assert result.output.shape == (1, 3, 16, 16)


class TestRawDecoderReturnsRenderResult:
    def test_returns_render_result(self, embedding_1024: torch.Tensor) -> None:
        decoder = RawDecoder(embed_dim=1024, output_shape=(1, 3, 16, 16))
        result = decoder.render(embedding_1024)
        assert isinstance(result, RenderResult)


# =============================================================================
# ProjectionDecoder Tests
# =============================================================================


class TestProjectionDecoderOutputShape:
    def test_matches_generator_output(self, embedding_1024: torch.Tensor) -> None:
        gen = MockGenerator(
            latent_shape=(1, 4, 64, 64),
            output_modality="image",
            output_shape=(1, 3, 512, 512),
        )
        decoder = ProjectionDecoder(embed_dim=1024, generator=gen)
        result = decoder.render(embedding_1024)
        assert result.output.shape == (1, 3, 512, 512)


class TestProjectionDecoderUsesGenerator:
    def test_decode_called_on_generator(self, embedding_1024: torch.Tensor) -> None:
        gen = MagicMock()
        gen.latent_shape = (1, 4, 64, 64)
        gen.output_modality = "image"
        gen.decode.return_value = torch.rand(1, 3, 512, 512)
        decoder = ProjectionDecoder(embed_dim=1024, generator=gen)
        decoder.render(embedding_1024)
        gen.decode.assert_called_once()
        latent_arg = gen.decode.call_args[0][0]
        assert latent_arg.shape == (1, 4, 64, 64)


class TestProjectionDecoderDeterministic:
    def test_same_input_same_output(self, embedding_1024: torch.Tensor) -> None:
        gen = MockGenerator(
            latent_shape=(1, 4, 16, 16),
            output_modality="image",
            output_shape=(1, 3, 64, 64),
        )
        decoder = ProjectionDecoder(embed_dim=1024, generator=gen)
        r1 = decoder.render(embedding_1024)
        r2 = decoder.render(embedding_1024)
        assert torch.allclose(r1.output, r2.output)


class TestProjectionDecoderCustomHiddenDims:
    def test_respects_hidden_dims(self, embedding_1024: torch.Tensor) -> None:
        gen = MockGenerator(
            latent_shape=(1, 4, 8, 8),
            output_modality="image",
            output_shape=(1, 3, 32, 32),
        )
        decoder = ProjectionDecoder(embed_dim=1024, generator=gen, hidden_dims=[512, 256])
        result = decoder.render(embedding_1024)
        assert result.output.shape == (1, 3, 32, 32)

    def test_single_hidden_layer(self, embedding_1024: torch.Tensor) -> None:
        gen = MockGenerator(
            latent_shape=(1, 4, 8, 8),
            output_modality="image",
            output_shape=(1, 3, 32, 32),
        )
        decoder = ProjectionDecoder(embed_dim=1024, generator=gen, hidden_dims=[1024])
        result = decoder.render(embedding_1024)
        assert result.output.shape == (1, 3, 32, 32)


class TestProjectionDecoderSaveLoad:
    def test_round_trip_preserves_weights(
        self, embedding_1024: torch.Tensor, tmp_path: Path
    ) -> None:
        gen = MockGenerator(
            latent_shape=(1, 4, 8, 8),
            output_modality="image",
            output_shape=(1, 3, 32, 32),
        )
        decoder = ProjectionDecoder(embed_dim=1024, generator=gen, hidden_dims=[256])
        before = decoder.render(embedding_1024)

        save_path = tmp_path / "proj.pt"
        decoder.save(save_path)

        decoder2 = ProjectionDecoder(embed_dim=1024, generator=gen, hidden_dims=[256])
        decoder2.load(save_path)
        after = decoder2.render(embedding_1024)

        assert torch.allclose(before.output, after.output, atol=1e-6)


class TestProjectionDecoderOutputModality:
    def test_inherits_generator_modality(self) -> None:
        gen = MockGenerator(output_modality="audio")
        decoder = ProjectionDecoder(embed_dim=1024, generator=gen)
        assert decoder.output_modality == "audio"


# =============================================================================
# IPAdapterRenderer Tests
# =============================================================================


class TestIPAdapterLazyLoad:
    def test_pipeline_not_loaded_on_init(self) -> None:
        renderer = IPAdapterRenderer()
        assert renderer._pipeline is None


class TestIPAdapterOutputModality:
    def test_returns_image(self) -> None:
        renderer = IPAdapterRenderer()
        assert renderer.output_modality == "image"


class TestIPAdapterMissingDeps:
    def test_graceful_error_when_diffusers_missing(self, embedding_1024: torch.Tensor) -> None:
        renderer = IPAdapterRenderer()
        with patch.dict(sys.modules, {"diffusers": None}):
            with pytest.raises(ImportError, match="diffusers"):
                renderer.render(embedding_1024)


class TestIPAdapterUnload:
    def test_unload_clears_pipeline(self) -> None:
        renderer = IPAdapterRenderer()
        renderer._pipeline = MagicMock()
        renderer.unload()
        assert renderer._pipeline is None


# =============================================================================
# TextRenderer Tests
# =============================================================================


class TestTextRendererOutputModality:
    def test_returns_text(self) -> None:
        encoder = MockEncoder()
        renderer = TextRenderer(encoder=encoder)
        assert renderer.output_modality == "text"


class TestTextRendererFindsSimilar:
    def test_finds_cat_for_cat_embedding(self) -> None:
        encoder = MockEncoder()
        vocab = ["cat", "dog", "bird", "fish", "tree"]
        renderer = TextRenderer(encoder=encoder, vocabulary=vocab)
        cat_embedding = encoder.encode_text("cat")
        result = renderer.render(cat_embedding, k=5)
        assert result.output.shape[1] == 5
        assert result.final_similarity > 0.9


class TestTextRendererKResults:
    def test_returns_exactly_k_results(self) -> None:
        encoder = MockEncoder()
        vocab = ["cat", "dog", "bird", "fish", "tree", "car", "moon"]
        renderer = TextRenderer(encoder=encoder, vocabulary=vocab)
        embedding = encoder.encode_text("cat")
        result = renderer.render(embedding, k=3)
        assert result.output.shape[1] == 3

    def test_k_larger_than_vocab(self) -> None:
        encoder = MockEncoder()
        vocab = ["cat", "dog"]
        renderer = TextRenderer(encoder=encoder, vocabulary=vocab)
        embedding = encoder.encode_text("cat")
        result = renderer.render(embedding, k=10)
        assert result.output.shape[1] == 2


class TestTextRendererCustomVocabulary:
    def test_uses_provided_vocabulary(self) -> None:
        encoder = MockEncoder()
        custom_vocab = ["alpha", "beta", "gamma"]
        renderer = TextRenderer(encoder=encoder, vocabulary=custom_vocab)
        assert renderer._vocab == custom_vocab


class TestTextRendererDefaultVocabulary:
    def test_default_vocabulary_is_populated(self) -> None:
        encoder = MockEncoder()
        renderer = TextRenderer(encoder=encoder)
        assert len(renderer._vocab) > 100
        assert renderer._vocab is DEFAULT_VOCABULARY


class TestTextRendererReturnsRenderResult:
    def test_returns_render_result(self) -> None:
        encoder = MockEncoder()
        renderer = TextRenderer(encoder=encoder, vocabulary=["cat", "dog"])
        embedding = encoder.encode_text("cat")
        result = renderer.render(embedding)
        assert isinstance(result, RenderResult)


class TestTextRenderResultFields:
    def test_descriptions_populated(self) -> None:
        encoder = MockEncoder()
        vocab = ["cat", "dog", "bird"]
        renderer = TextRenderer(encoder=encoder, vocabulary=vocab)
        embedding = encoder.encode_text("cat")
        result = renderer.render(embedding, k=3)
        assert hasattr(result, "descriptions")
        assert len(result.descriptions) == 3
        assert all(isinstance(d, str) for d in result.descriptions)
        assert all(d in vocab for d in result.descriptions)

    def test_scores_populated(self) -> None:
        encoder = MockEncoder()
        vocab = ["cat", "dog", "bird"]
        renderer = TextRenderer(encoder=encoder, vocabulary=vocab)
        embedding = encoder.encode_text("cat")
        result = renderer.render(embedding, k=3)
        assert hasattr(result, "scores")
        assert len(result.scores) == 3
        assert all(isinstance(s, float) for s in result.scores)

    def test_scores_sorted_descending(self) -> None:
        encoder = MockEncoder()
        vocab = ["cat", "dog", "bird", "fish", "tree"]
        renderer = TextRenderer(encoder=encoder, vocabulary=vocab)
        embedding = encoder.encode_text("cat")
        result = renderer.render(embedding, k=5)
        for i in range(len(result.scores) - 1):
            assert result.scores[i] >= result.scores[i + 1]


# =============================================================================
# DirectRenderer Protocol Compliance
# =============================================================================


class TestRawDecoderIsDirectRenderer:
    def test_isinstance_check(self) -> None:
        decoder = RawDecoder(embed_dim=1024)
        assert isinstance(decoder, DirectRenderer)


class TestProjectionDecoderIsDirectRenderer:
    def test_isinstance_check(self) -> None:
        gen = MockGenerator()
        decoder = ProjectionDecoder(embed_dim=1024, generator=gen)
        assert isinstance(decoder, DirectRenderer)
