from __future__ import annotations

import pytest
import torch

from embedding_art.core.concept import Concept
from embedding_art.core.concept_spec import ConceptSpec
from embedding_art.core.render_result import OptimizationHistory, RenderResult
from embedding_art.encoders.features import LayerFeatures
from embedding_art.encoders.registry import EncoderCapability, EncoderCard
from embedding_art.probes.activation_probe import ActivationProbe, ModelState
from embedding_art.probes.state_renderer import StateRenderer
from tests.conftest import MockEncoder


class MockMultiLayerEncoder(MockEncoder):
    """Mock encoder that supports multi-layer feature extraction."""

    def __init__(self, embedding_dim=1024, n_layers=6, device="cpu"):
        super().__init__(embedding_dim=embedding_dim, device=device)
        self._n_layers = n_layers

    @property
    def card(self) -> EncoderCard:
        """Return an EncoderCard with MULTI_LAYER_FEATURES capability."""
        return EncoderCard(
            name="mock_multilayer",
            capabilities=(
                EncoderCapability.TEXT
                | EncoderCapability.IMAGE
                | EncoderCapability.AUDIO
                | EncoderCapability.VIDEO
                | EncoderCapability.BACKPROP_OPTIMIZABLE
                | EncoderCapability.MULTI_LAYER_FEATURES
            ),
            embedding_dim=self._embedding_dim,
            memory_estimate_mb=100,
            backprop_cost=1.0,
        )

    def get_layer_features(self, tensor: torch.Tensor) -> dict[int, LayerFeatures]:
        """Return deterministic LayerFeatures for each layer."""
        result = {}
        for i in range(self._n_layers):
            gen = torch.Generator().manual_seed(i + 1)
            feat_tensor = torch.randn(1, 16, 8, 8, generator=gen)
            result[i] = LayerFeatures(
                tensor=feat_tensor,
                spatial=True,
                shape_semantic="batch_channels_height_width",
                layer_name=f"layer_{i}",
            )
        return result


class MockRenderer:
    """Mock renderer that satisfies the DirectRenderer duck-type."""

    def __init__(self, output_shape=(1, 3, 64, 64)):
        self._output_shape = output_shape

    @property
    def output_modality(self) -> str:
        return "image"

    def render(self, embedding: torch.Tensor, **kwargs) -> RenderResult:
        """Render an embedding to a deterministic output."""
        output = torch.zeros(self._output_shape)
        output[:, 0, :, :] = embedding.mean().item()
        return RenderResult(
            output=output,
            history=OptimizationHistory(),
            encoder_name="mock_renderer",
            final_similarity=0.0,
        )


# ---------------------------------------------------------------------------
# ModelState tests
# ---------------------------------------------------------------------------


class TestModelState:
    def test_model_state_creation(self):
        """Basic dataclass construction."""
        embedding = torch.nn.functional.normalize(torch.randn(1, 1024), dim=-1)
        activations = {0: torch.randn(1, 16), 2: torch.randn(1, 32)}
        state = ModelState(
            layer_activations=activations,
            final_embedding=embedding,
            encoder_name="test_encoder",
            input_description="test input",
        )
        assert state.encoder_name == "test_encoder"
        assert state.input_description == "test input"
        assert state.layer_names is None

    def test_model_state_layer_access(self):
        """Access layer activations by index."""
        act_0 = torch.randn(1, 16)
        act_3 = torch.randn(1, 32)
        state = ModelState(
            layer_activations={0: act_0, 3: act_3},
            final_embedding=torch.randn(1, 1024),
            encoder_name="test",
            input_description="",
        )
        assert torch.equal(state.layer_activations[0], act_0)
        assert torch.equal(state.layer_activations[3], act_3)
        assert 1 not in state.layer_activations


# ---------------------------------------------------------------------------
# ActivationProbe tests
# ---------------------------------------------------------------------------


class TestActivationProbe:
    def test_capture_from_concept_with_source_input(self):
        """Captures multi-layer when source_input exists."""
        encoder = MockMultiLayerEncoder(n_layers=4)
        source = torch.randn(1, 3, 224, 224)
        concept = Concept(
            embedding=torch.randn(1, 1024),
            description="test concept",
            source_input=source,
        )
        probe = ActivationProbe()
        state = probe.capture_from_concept(encoder, concept)
        assert len(state.layer_activations) == 4
        assert state.layer_names is not None
        assert len(state.layer_names) == 4

    def test_capture_from_concept_without_source_input(self):
        """Captures only final embedding when no source_input."""
        encoder = MockMultiLayerEncoder(n_layers=4)
        concept = Concept(
            embedding=torch.randn(1, 1024),
            description="no source",
        )
        probe = ActivationProbe()
        state = probe.capture_from_concept(encoder, concept)
        assert len(state.layer_activations) == 0
        assert state.final_embedding is not None

    def test_capture_from_spec(self):
        """Integrates with ConceptSpec."""
        encoder = MockMultiLayerEncoder(n_layers=3)
        spec = ConceptSpec(text="hello world")
        probe = ActivationProbe()
        state = probe.capture_from_spec(encoder, spec)
        assert state.encoder_name == "mock_multilayer"
        assert state.final_embedding is not None

    def test_capture_from_tensor(self):
        """Direct tensor capture."""
        encoder = MockMultiLayerEncoder(n_layers=6)
        tensor = torch.randn(1, 3, 224, 224)
        probe = ActivationProbe()
        state = probe.capture_from_tensor(encoder, tensor, description="raw tensor")
        assert len(state.layer_activations) == 6
        assert state.input_description == "raw tensor"

    def test_capture_graceful_degradation(self):
        """Works when encoder lacks get_layer_features."""
        encoder = MockEncoder()
        tensor = torch.randn(1, 3, 224, 224)
        probe = ActivationProbe()
        state = probe.capture_from_tensor(encoder, tensor, description="fallback")
        assert len(state.layer_activations) == 0
        assert state.final_embedding is not None

    def test_capture_preserves_encoder_name(self):
        """encoder_name matches encoder.card.name."""
        encoder = MockMultiLayerEncoder(n_layers=2)
        tensor = torch.randn(1, 3, 224, 224)
        probe = ActivationProbe()
        state = probe.capture_from_tensor(encoder, tensor)
        assert state.encoder_name == encoder.card.name

    def test_layer_activations_are_tensors(self):
        """All values in layer_activations are Tensors."""
        encoder = MockMultiLayerEncoder(n_layers=4)
        tensor = torch.randn(1, 3, 224, 224)
        probe = ActivationProbe()
        state = probe.capture_from_tensor(encoder, tensor)
        for idx, act in state.layer_activations.items():
            assert isinstance(act, torch.Tensor)


# ---------------------------------------------------------------------------
# StateRenderer tests
# ---------------------------------------------------------------------------


class TestStateRenderer:
    def _make_state(self, n_layers=4) -> ModelState:
        """Helper to create a ModelState with n_layers activations."""
        activations = {}
        names = {}
        for i in range(n_layers):
            activations[i] = torch.randn(1, 16)
            names[i] = f"layer_{i}"
        return ModelState(
            layer_activations=activations,
            final_embedding=torch.randn(1, 1024),
            encoder_name="test",
            input_description="test",
            layer_names=names,
        )

    def test_render_layer(self):
        """Renders a specific layer."""
        state = self._make_state(n_layers=4)
        renderer = MockRenderer()
        sr = StateRenderer()
        result = sr.render_layer(state, 2, renderer)
        assert isinstance(result, RenderResult)
        assert result.output.shape == (1, 3, 64, 64)

    def test_render_progression(self):
        """Renders all layers in order."""
        state = self._make_state(n_layers=5)
        renderer = MockRenderer()
        sr = StateRenderer()
        results = sr.render_progression(state, renderer)
        assert len(results) == 5
        for r in results:
            assert isinstance(r, RenderResult)

    def test_render_progression_order(self):
        """Layers are sorted by index."""
        activations = {5: torch.randn(1, 16), 1: torch.randn(1, 16), 3: torch.randn(1, 16)}
        state = ModelState(
            layer_activations=activations,
            final_embedding=torch.randn(1, 1024),
            encoder_name="test",
            input_description="test",
        )
        renderer = MockRenderer()
        sr = StateRenderer()
        results = sr.render_progression(state, renderer)
        assert len(results) == 3
        vals = [activations[k].mean().item() for k in sorted(activations.keys())]
        for i, r in enumerate(results):
            assert r.output[:, 0, 0, 0].item() == pytest.approx(vals[i], abs=1e-5)

    def test_compose_grid_shape(self):
        """Output grid has correct dimensions."""
        state = self._make_state(n_layers=6)
        renderer = MockRenderer(output_shape=(1, 3, 32, 32))
        sr = StateRenderer()
        results = sr.render_progression(state, renderer)
        grid = sr.compose_grid(results, grid_cols=3)
        assert grid.shape == (1, 3, 64, 96)

    def test_compose_grid_single_result(self):
        """Works with a single result."""
        state = self._make_state(n_layers=1)
        renderer = MockRenderer(output_shape=(1, 3, 32, 32))
        sr = StateRenderer()
        results = sr.render_progression(state, renderer)
        grid = sr.compose_grid(results, grid_cols=4)
        assert grid.shape == (1, 3, 32, 32)
