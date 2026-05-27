"""
Tests for EmbeddingArtEngine.render_interpolation.

Verifies:
- Returns a list of RenderResult of length == steps
- Endpoints correspond to t=0 (spec_a) and t=1 (spec_b)
- Single-step case returns one result at t=0.5
- Encoder resolution: registry path and self.encoder path
- Raises ValueError when no encoder is available
- Accepts already-resolved Concept objects (not just ConceptSpec)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

from embedding_art.core.concept import Concept
from embedding_art.core.config import LossConfig, OptimizationConfig
from embedding_art.core.render_result import RenderResult

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 16


def _make_concept(seed: int) -> Concept:
    torch.manual_seed(seed)
    emb = F.normalize(torch.randn(1, EMBEDDING_DIM), dim=-1)
    return Concept(embedding=emb, description=f"concept_{seed}")


def _fast_config() -> OptimizationConfig:
    return OptimizationConfig(
        steps=2,
        learning_rate=0.01,
        optimizer="adam",
        scheduler="constant",
        loss=LossConfig(
            similarity_weight=1.0,
            feature_matching_weight=0.0,
            sae_feature_weight=0.0,
        ),
    )


class DifferentiableEncoder:
    """Minimal differentiable encoder for interpolation tests."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM) -> None:
        torch.manual_seed(0)
        self._embedding_dim = embedding_dim
        self._proj = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        with torch.no_grad():
            torch.nn.init.eye_(self._proj.weight)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def encode_for_optimization(self, tensor: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(tensor, output_size=(1, 1))
        flat = pooled.view(pooled.shape[0], -1)
        if flat.shape[1] < self._embedding_dim:
            pad = torch.zeros(flat.shape[0], self._embedding_dim - flat.shape[1])
            flat = torch.cat([flat, pad], dim=1)
        else:
            flat = flat[:, : self._embedding_dim]
        return F.normalize(self._proj(flat), dim=-1)

    @property
    def card(self) -> Any:
        from embedding_art.encoders.registry import EncoderCapability, EncoderCard

        return EncoderCard(
            name="interp_test_encoder",
            capabilities=EncoderCapability.IMAGE | EncoderCapability.BACKPROP_OPTIMIZABLE,
            embedding_dim=self._embedding_dim,
            memory_estimate_mb=0,
            backprop_cost=0.0,
        )


class SmallMockGenerator:
    """Minimal latent generator with differentiable decode."""

    def __init__(
        self,
        latent_shape: tuple[int, ...] = (1, 4, 4, 4),
        output_channels: int = 4,
        output_size: int = 8,
    ) -> None:
        self._latent_shape = latent_shape
        self._output_channels = output_channels
        self._output_size = output_size

    @property
    def latent_shape(self) -> tuple[int, ...]:
        return self._latent_shape

    def init_latent(self, seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            gen = torch.Generator().manual_seed(seed)
            latent = torch.randn(self._latent_shape, generator=gen)
        else:
            latent = torch.randn(self._latent_shape)
        return latent.requires_grad_(True)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        out = F.interpolate(
            latent,
            size=(self._output_size, self._output_size),
            mode="bilinear",
            align_corners=False,
        )
        if out.shape[1] != self._output_channels:
            out = out[:, : self._output_channels, :, :]
        return out.sigmoid()


def _make_engine_with_encoder_and_generator():
    """Return an engine wired with the small mock encoder and generator."""
    from embedding_art.core.engine import EmbeddingArtEngine

    encoder = DifferentiableEncoder()
    engine = EmbeddingArtEngine(encoder=encoder, device="cpu")
    engine.register_generator("image", SmallMockGenerator())
    return engine, encoder


# ---------------------------------------------------------------------------
# Return type and length
# ---------------------------------------------------------------------------


class TestRenderInterpolationReturnType:
    """render_interpolation returns a list of RenderResult."""

    def test_returns_a_list(self):
        engine, _ = _make_engine_with_encoder_and_generator()
        a, b = _make_concept(1), _make_concept(2)
        config = _fast_config()

        result = engine.render_interpolation(a, b, steps=3, config=config)

        assert isinstance(result, list)

    def test_each_element_is_render_result(self):
        engine, _ = _make_engine_with_encoder_and_generator()
        a, b = _make_concept(1), _make_concept(2)
        config = _fast_config()

        results = engine.render_interpolation(a, b, steps=3, config=config)

        for r in results:
            assert isinstance(r, RenderResult)

    def test_list_length_equals_steps(self):
        engine, _ = _make_engine_with_encoder_and_generator()
        a, b = _make_concept(1), _make_concept(2)
        config = _fast_config()

        for steps in (1, 3, 5):
            results = engine.render_interpolation(a, b, steps=steps, config=config)
            assert len(results) == steps


# ---------------------------------------------------------------------------
# Interpolation endpoints
# ---------------------------------------------------------------------------


class TestRenderInterpolationEndpoints:
    """t=0 step optimizes toward concept_a; t=1 step optimizes toward concept_b."""

    def test_first_step_uses_concept_a_embedding(self):
        """At t=0 the interpolated concept should equal concept_a (slerp identity)."""
        from unittest.mock import patch

        from embedding_art.core.strategies import OptimizationStrategy

        engine, encoder = _make_engine_with_encoder_and_generator()
        a, b = _make_concept(10), _make_concept(20)
        config = _fast_config()

        recorded_targets: list[Concept] = []

        original_render = OptimizationStrategy.render

        def capturing_render(self_inner, target, generator, enc, cfg):
            recorded_targets.append(target)
            return original_render(self_inner, target, generator, enc, cfg)

        with patch.object(OptimizationStrategy, "render", capturing_render):
            engine.render_interpolation(a, b, steps=3, config=config)

        # First call corresponds to t=0 -> should equal concept_a
        first_target = recorded_targets[0]
        assert torch.allclose(first_target.embedding, a.embedding, atol=1e-5)

    def test_last_step_uses_concept_b_embedding(self):
        """At t=1 the interpolated concept should equal concept_b (slerp identity)."""
        from unittest.mock import patch

        from embedding_art.core.strategies import OptimizationStrategy

        engine, encoder = _make_engine_with_encoder_and_generator()
        a, b = _make_concept(10), _make_concept(20)
        config = _fast_config()

        recorded_targets: list[Concept] = []

        original_render = OptimizationStrategy.render

        def capturing_render(self_inner, target, generator, enc, cfg):
            recorded_targets.append(target)
            return original_render(self_inner, target, generator, enc, cfg)

        with patch.object(OptimizationStrategy, "render", capturing_render):
            engine.render_interpolation(a, b, steps=3, config=config)

        # Last call corresponds to t=1 -> should equal concept_b
        last_target = recorded_targets[-1]
        assert torch.allclose(last_target.embedding, b.embedding, atol=1e-5)


# ---------------------------------------------------------------------------
# Single-step case
# ---------------------------------------------------------------------------


class TestRenderInterpolationSingleStep:
    """steps=1 should produce exactly one result."""

    def test_single_step_returns_one_result(self):
        engine, _ = _make_engine_with_encoder_and_generator()
        a, b = _make_concept(3), _make_concept(4)
        config = _fast_config()

        results = engine.render_interpolation(a, b, steps=1, config=config)

        assert len(results) == 1

    def test_single_step_is_render_result(self):
        engine, _ = _make_engine_with_encoder_and_generator()
        a, b = _make_concept(3), _make_concept(4)
        config = _fast_config()

        results = engine.render_interpolation(a, b, steps=1, config=config)

        assert isinstance(results[0], RenderResult)


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------


class TestRenderInterpolationDefaultConfig:
    """When config=None, a default OptimizationConfig is used without crashing."""

    def test_accepts_none_config(self):
        """render_interpolation should not raise when config is None."""
        from embedding_art.core.engine import EmbeddingArtEngine

        encoder = DifferentiableEncoder()
        engine = EmbeddingArtEngine(encoder=encoder, device="cpu")
        # Use a tiny custom generator so the default config's steps don't take long
        tiny_gen = SmallMockGenerator(latent_shape=(1, 4, 2, 2), output_size=4)
        engine.register_generator("image", tiny_gen)

        a, b = _make_concept(5), _make_concept(6)

        # Should not raise; we just verify it returns a list
        results = engine.render_interpolation(a, b, steps=2, config=None)
        assert isinstance(results, list)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Encoder resolution
# ---------------------------------------------------------------------------


class TestRenderInterpolationEncoderResolution:
    """Encoder is resolved via registry when encoder_name is provided."""

    def test_uses_self_encoder_when_no_registry(self):
        """Without a registry, self.encoder must be used and not raise."""
        engine, _ = _make_engine_with_encoder_and_generator()
        a, b = _make_concept(7), _make_concept(8)
        config = _fast_config()

        results = engine.render_interpolation(a, b, steps=2, config=config)

        assert len(results) == 2

    def test_raises_when_no_encoder_available(self):
        from embedding_art.core.engine import EmbeddingArtEngine

        engine = EmbeddingArtEngine(encoder=None, device="cpu")
        engine.register_generator("image", SmallMockGenerator())

        a, b = _make_concept(9), _make_concept(11)
        config = _fast_config()

        with pytest.raises(ValueError, match="No encoder available"):
            engine.render_interpolation(a, b, steps=2, config=config)

    def test_uses_registry_encoder_when_encoder_name_provided(self):
        """When encoder_name is given and a registry is set, that encoder is used."""
        from embedding_art.core.engine import EmbeddingArtEngine

        encoder = DifferentiableEncoder()
        mock_registry = MagicMock()
        mock_registry.load.return_value = encoder

        # Build engine via from_registry so _registry is set
        engine = EmbeddingArtEngine.from_registry(mock_registry, device="cpu")
        engine.register_generator("image", SmallMockGenerator())

        a, b = _make_concept(12), _make_concept(13)
        config = _fast_config()

        engine.render_interpolation(a, b, steps=2, encoder_name="some_encoder", config=config)

        mock_registry.load.assert_called_with("some_encoder")


# ---------------------------------------------------------------------------
# Output tensor
# ---------------------------------------------------------------------------


class TestRenderInterpolationOutputTensor:
    """Each RenderResult must carry a valid output tensor."""

    def test_output_is_tensor(self):
        engine, _ = _make_engine_with_encoder_and_generator()
        a, b = _make_concept(14), _make_concept(15)
        config = _fast_config()

        results = engine.render_interpolation(a, b, steps=3, config=config)

        for r in results:
            assert isinstance(r.output, torch.Tensor)

    def test_output_is_detached(self):
        engine, _ = _make_engine_with_encoder_and_generator()
        a, b = _make_concept(16), _make_concept(17)
        config = _fast_config()

        results = engine.render_interpolation(a, b, steps=3, config=config)

        for r in results:
            assert not r.output.requires_grad
