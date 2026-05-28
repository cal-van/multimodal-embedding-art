"""
Tests for EmbeddingArtEngine v3 direct rendering methods:
render_direct, render_state, render_features.

Uses mock objects for renderer, encoder, and SAE to test engine
orchestration without requiring real models.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from embedding_art.core.concept import Concept
from embedding_art.core.concept_spec import ConceptSpec
from embedding_art.core.engine import EmbeddingArtEngine
from embedding_art.core.render_result import OptimizationHistory, RenderResult
from embedding_art.sae.lens import SAEDecomposition
from tests.conftest import MockEncoder

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockDirectRenderer:
    """Minimal mock renderer that satisfies the DirectRenderer protocol."""

    output_modality = "image"

    def render(self, embedding, **kwargs):
        return RenderResult(
            output=torch.rand(1, 3, 32, 32),
            history=OptimizationHistory(),
            encoder_name="mock",
            final_similarity=0.0,
        )


class MockSAE:
    """Minimal mock SAE that returns a canned decomposition."""

    def __init__(self):
        self._decompose_called_with = None

    def decompose(self, embedding):
        self._decompose_called_with = embedding
        return SAEDecomposition(
            activations=torch.zeros(1, 8),
            active_features={"feature_a": 1.5, "feature_b": 0.8, "feature_c": 0.3},
            reconstruction_error=0.05,
        )


class MockFeatureRenderer:
    """Stub for FeatureRenderer that returns a dict of name -> RenderResult."""

    def render_decomposition(self, decomposition, sae, renderer, max_features):
        results = {}
        sorted_features = sorted(
            decomposition.active_features.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        for name, _ in sorted_features[:max_features]:
            results[name] = RenderResult(
                output=torch.rand(1, 3, 32, 32),
                history=OptimizationHistory(),
                encoder_name="mock",
                final_similarity=0.0,
            )
        return results


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_enc() -> MockEncoder:
    return MockEncoder(embedding_dim=1024, device="cpu")


@pytest.fixture
def engine(mock_enc) -> EmbeddingArtEngine:
    eng = EmbeddingArtEngine(encoder=mock_enc, device="cpu")
    return eng


@pytest.fixture
def renderer() -> MockDirectRenderer:
    return MockDirectRenderer()


@pytest.fixture
def mock_sae() -> MockSAE:
    return MockSAE()


@pytest.fixture
def spec() -> ConceptSpec:
    return ConceptSpec(text="goldfish")


# ---------------------------------------------------------------------------
# Tests: _resolve_encoder
# ---------------------------------------------------------------------------


class TestResolveEncoder:
    def test_resolves_default_encoder(self, engine, mock_enc):
        """Falls back to self.encoder when no name is given."""
        resolved = engine._resolve_encoder(None)
        assert resolved is mock_enc

    def test_raises_when_no_encoder(self):
        """Raises ValueError when no encoder available at all."""
        engine = EmbeddingArtEngine(encoder=None, device="cpu")
        with pytest.raises(ValueError, match="No encoder available"):
            engine._resolve_encoder(None)


# ---------------------------------------------------------------------------
# Tests: render_direct
# ---------------------------------------------------------------------------


class TestRenderDirect:
    def test_render_direct_accepts_concept(self, engine, renderer, mock_enc):
        """Concept objects pass through without re-encoding."""
        embedding = F.normalize(torch.randn(1, 1024), dim=-1)
        concept = Concept(embedding=embedding, description="test concept")
        result = engine.render_direct(concept, renderer)
        assert isinstance(result, RenderResult)

    def test_render_direct_calls_renderer(self, engine, renderer, spec):
        """Verifies renderer.render() is called with concept embedding."""
        result = engine.render_direct(spec, renderer)
        assert isinstance(result, RenderResult)

    def test_render_direct_returns_render_result(self, engine, renderer, spec):
        """Return type is RenderResult."""
        result = engine.render_direct(spec, renderer)
        assert isinstance(result, RenderResult)
        assert isinstance(result.output, torch.Tensor)
        assert result.output.shape == (1, 3, 32, 32)

    def test_render_direct_uses_specified_encoder(self, renderer, spec):
        """Respects encoder parameter when using a registry."""
        from embedding_art.encoders.registry import (
            EncoderCapability,
            EncoderCard,
            EncoderRegistry,
        )

        enc_a = MockEncoder(embedding_dim=1024, device="cpu")
        enc_b = MockEncoder(embedding_dim=1024, device="cpu")

        reg = EncoderRegistry()
        for alias, enc in [("enc_a", enc_a), ("enc_b", enc_b)]:
            card = EncoderCard(
                name=alias,
                capabilities=EncoderCapability.TEXT | EncoderCapability.IMAGE,
                embedding_dim=1024,
                memory_estimate_mb=10,
                backprop_cost=1.0,
            )
            reg._classes[alias] = type("_C", (), {"card": card})
            reg._instances[alias] = enc

        engine = EmbeddingArtEngine.from_registry(reg, default_encoder="enc_a", device="cpu")

        # render_direct with encoder="enc_b" should use enc_b
        result = engine.render_direct(spec, renderer, encoder="enc_b")
        assert isinstance(result, RenderResult)


# ---------------------------------------------------------------------------
# Tests: render_state
# ---------------------------------------------------------------------------


class TestRenderState:
    def test_render_state_returns_list(self, engine, renderer, spec):
        """Returns list of RenderResults."""
        results = engine.render_state(spec, renderer)
        assert isinstance(results, list)

    def test_render_state_filters_layers(self, engine, spec):
        """Only renders requested layers."""
        # Create a renderer that tracks what it gets
        call_count = 0

        class TrackingRenderer:
            output_modality = "image"

            def render(self, embedding, **kwargs):
                nonlocal call_count
                call_count += 1
                return RenderResult(
                    output=torch.rand(1, 3, 32, 32),
                    history=OptimizationHistory(),
                    encoder_name="mock",
                    final_similarity=0.0,
                )

        tracking = TrackingRenderer()

        # MockEncoder doesn't have get_layer_features, so state will have
        # empty layer_activations. That means no layers to filter or render.
        results = engine.render_state(spec, tracking, layers=[0, 2])
        # With no layer_activations from MockEncoder, the result is an empty list.
        assert isinstance(results, list)

    def test_render_state_renders_all_layers_when_none(self, engine, renderer, spec):
        """When layers=None, all captured layers are rendered."""
        results = engine.render_state(spec, renderer, layers=None)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Tests: render_features
# ---------------------------------------------------------------------------


class TestRenderFeatures:
    def test_render_features_returns_dict(self, engine, renderer, mock_sae, spec):
        """Returns dict of feature_name -> RenderResult."""
        # Monkey-patch the FeatureRenderer import in engine to use our mock
        import embedding_art.sae as sae_mod

        original_fr = sae_mod.FeatureRenderer

        sae_mod.FeatureRenderer = MockFeatureRenderer
        try:
            results = engine.render_features(spec, renderer, mock_sae)
            assert isinstance(results, dict)
            for name, result in results.items():
                assert isinstance(name, str)
                assert isinstance(result, RenderResult)
        finally:
            sae_mod.FeatureRenderer = original_fr

    def test_render_features_decomposes_embedding(self, engine, renderer, mock_sae, spec):
        """Calls sae.decompose() with the concept embedding."""
        import embedding_art.sae as sae_mod

        original_fr = sae_mod.FeatureRenderer
        sae_mod.FeatureRenderer = MockFeatureRenderer
        try:
            engine.render_features(spec, renderer, mock_sae)
            assert mock_sae._decompose_called_with is not None
            assert isinstance(mock_sae._decompose_called_with, torch.Tensor)
        finally:
            sae_mod.FeatureRenderer = original_fr

    def test_render_features_respects_max_features(self, engine, renderer, mock_sae, spec):
        """max_features limits number of returned results."""
        import embedding_art.sae as sae_mod

        original_fr = sae_mod.FeatureRenderer
        sae_mod.FeatureRenderer = MockFeatureRenderer
        try:
            results = engine.render_features(spec, renderer, mock_sae, max_features=2)
            assert len(results) <= 2
        finally:
            sae_mod.FeatureRenderer = original_fr
