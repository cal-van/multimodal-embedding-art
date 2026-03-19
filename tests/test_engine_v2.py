"""
Tests for EmbeddingArtEngine v2 methods: from_registry, render, render_compare.

Follows the Red-Green-Refactor TDD cycle. All v2 surface area is tested here;
the old optimize() API is covered by a smoke check to ensure backward compat.

The MockEncoder.encode_for_optimization uses .item() which breaks the autograd
graph. We therefore define a DifferentiableEncoder below that keeps gradients
flowing — identical in spirit to what Task 9 did for strategy tests.
"""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F

from embedding_art.core.concept import Concept
from embedding_art.core.concept_spec import ConceptSpec
from embedding_art.core.config import LossConfig, OptimizationConfig
from embedding_art.core.engine import EmbeddingArtEngine
from embedding_art.core.render_result import RenderResult
from embedding_art.encoders.registry import EncoderCapability, EncoderCard, EncoderRegistry

# ---------------------------------------------------------------------------
# DifferentiableEncoder — gradients flow through encode_for_optimization
# ---------------------------------------------------------------------------


class DifferentiableEncoder:
    """Minimal encoder whose encode_for_optimization keeps autograd alive.

    Uses a fixed linear projection so the output is always differentiable with
    respect to the input tensor. The projection is not trained; it just maps
    pixel space to embedding space in a way that .backward() can traverse.
    """

    EMBEDDING_DIM = 64  # small so tests are fast

    def __init__(self) -> None:
        torch.manual_seed(0)
        # Fixed weight matrix: maps flattened input to embedding space.
        # We won't optimize this — it's just a stable projection.
        self._weight = torch.randn(self.EMBEDDING_DIM, 3 * 32 * 32) * 0.01

    @property
    def card(self) -> EncoderCard:
        return EncoderCard(
            name="differentiable",
            capabilities=(
                EncoderCapability.TEXT
                | EncoderCapability.IMAGE
                | EncoderCapability.BACKPROP_OPTIMIZABLE
            ),
            embedding_dim=self.EMBEDDING_DIM,
            memory_estimate_mb=10,
            backprop_cost=0.1,
        )

    def encode(self, spec: ConceptSpec) -> Concept:
        """Encode a ConceptSpec to a Concept with a deterministic embedding."""
        seed = hash(spec.text or "default") % (2**32)
        gen = torch.Generator().manual_seed(seed)
        emb = torch.randn(1, self.EMBEDDING_DIM, generator=gen)
        return Concept(embedding=F.normalize(emb, dim=-1), description=repr(spec))

    def encode_for_optimization(self, tensor: torch.Tensor) -> torch.Tensor:
        """Differentiable projection: gradients flow back to *tensor*."""
        # Resize to expected input size
        resized = F.interpolate(tensor, size=(32, 32), mode="bilinear", align_corners=False)
        flat = resized.reshape(resized.shape[0], -1)  # [B, 3*32*32]
        # Linear projection — grad flows through this matmul
        out = flat @ self._weight.T  # [B, EMBEDDING_DIM]
        return F.normalize(out, dim=-1)

    def unload(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Minimal generator with a small latent for fast tests
# ---------------------------------------------------------------------------


class SmallMockGenerator:
    """Generator with tiny latent/output shapes so tests run in milliseconds."""

    latent_shape = (1, 1, 8, 8)  # tiny
    output_shape = (1, 3, 32, 32)

    def init_latent(self, seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            gen = torch.Generator().manual_seed(seed)
            latent = torch.randn(self.latent_shape, generator=gen)
        else:
            latent = torch.randn(self.latent_shape)
        return latent.requires_grad_(True)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        out = F.interpolate(latent, size=(32, 32), mode="bilinear", align_corners=False)
        # Pad channels: 1 → 3
        out = out.expand(-1, 3, -1, -1)
        return out.sigmoid()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def diff_encoder() -> DifferentiableEncoder:
    return DifferentiableEncoder()


@pytest.fixture
def small_generator() -> SmallMockGenerator:
    return SmallMockGenerator()


@pytest.fixture
def fast_config() -> OptimizationConfig:
    """2-step config with no feature matching so tests run quickly."""
    return OptimizationConfig(
        steps=2,
        learning_rate=0.01,
        optimizer="adam",
        scheduler="constant",
        loss=LossConfig(
            similarity_weight=1.0,
            feature_matching_weight=0.0,  # off — MockEncoder has no layers
        ),
    )


@pytest.fixture
def registry(diff_encoder) -> EncoderRegistry:
    """Registry pre-populated with one differentiable encoder."""

    class _EncoderClass:
        card = diff_encoder.card

        def __init__(self):
            # Return the shared instance, not a new one
            pass

    # Patch __init__ to return our pre-built instance by storing it externally
    # and using a factory fixture pattern via registry.register + manual load.
    reg = EncoderRegistry()

    # We register a dummy class (for card introspection) then inject the
    # pre-built instance directly so the fixture's diff_encoder is reused.
    reg._classes["differentiable"] = type(
        "_DiffEncoderClass",
        (),
        {"card": diff_encoder.card},
    )
    reg._instances["differentiable"] = diff_encoder
    return reg


# ---------------------------------------------------------------------------
# Tests: from_registry
# ---------------------------------------------------------------------------


class TestFromRegistry:
    def test_from_registry_creates_engine(self, registry):
        """from_registry returns an EmbeddingArtEngine instance."""
        engine = EmbeddingArtEngine.from_registry(registry)
        assert isinstance(engine, EmbeddingArtEngine)

    def test_from_registry_stores_registry(self, registry):
        """Engine created via from_registry stores the registry as _registry."""
        engine = EmbeddingArtEngine.from_registry(registry)
        assert engine._registry is registry

    def test_from_registry_no_default_encoder(self, registry):
        """When default_encoder is None, self.encoder is None."""
        engine = EmbeddingArtEngine.from_registry(registry, default_encoder=None, device="cpu")
        assert engine.encoder is None

    def test_from_registry_loads_default_encoder(self, registry, diff_encoder):
        """Specifying default_encoder loads that encoder into self.encoder."""
        engine = EmbeddingArtEngine.from_registry(
            registry, default_encoder="differentiable", device="cpu"
        )
        assert engine.encoder is diff_encoder


# ---------------------------------------------------------------------------
# Tests: render — spec routing
# ---------------------------------------------------------------------------


class TestRender:
    def test_render_with_concept_spec(self, registry, diff_encoder, small_generator, fast_config):
        """Passing a ConceptSpec materializes a Concept and returns RenderResult."""
        engine = EmbeddingArtEngine.from_registry(
            registry, default_encoder="differentiable", device="cpu"
        )
        engine.register_generator("image", small_generator)

        spec = ConceptSpec(text="ocean waves")
        result = engine.render(
            spec, encoder_name="differentiable", output_modality="image", config=fast_config
        )

        assert isinstance(result, RenderResult)

    def test_render_with_concept(self, registry, diff_encoder, small_generator, fast_config):
        """Passing an existing Concept (not a ConceptSpec) routes it directly."""
        engine = EmbeddingArtEngine.from_registry(
            registry, default_encoder="differentiable", device="cpu"
        )
        engine.register_generator("image", small_generator)

        emb = F.normalize(torch.randn(1, DifferentiableEncoder.EMBEDDING_DIM), dim=-1)
        concept = Concept(embedding=emb, description="test concept")

        result = engine.render(
            concept, encoder_name="differentiable", output_modality="image", config=fast_config
        )

        assert isinstance(result, RenderResult)

    def test_render_result_has_output_tensor(
        self, registry, diff_encoder, small_generator, fast_config
    ):
        """RenderResult.output is a non-empty tensor."""
        engine = EmbeddingArtEngine.from_registry(
            registry, default_encoder="differentiable", device="cpu"
        )
        engine.register_generator("image", small_generator)

        spec = ConceptSpec(text="goldfish")
        result = engine.render(spec, encoder_name="differentiable", config=fast_config)

        assert isinstance(result.output, torch.Tensor)
        assert result.output.numel() > 0

    def test_render_uses_registry_encoder(
        self, registry, diff_encoder, small_generator, fast_config
    ):
        """When encoder_name is given, the engine loads that encoder from the registry."""
        # Engine has no default encoder — relies entirely on encoder_name resolution.
        engine = EmbeddingArtEngine.from_registry(registry, default_encoder=None, device="cpu")
        engine.register_generator("image", small_generator)

        spec = ConceptSpec(text="coral reef")
        # Should not raise even though self.encoder is None
        result = engine.render(spec, encoder_name="differentiable", config=fast_config)
        assert isinstance(result, RenderResult)

    def test_render_falls_back_to_default_encoder(
        self, registry, diff_encoder, small_generator, fast_config
    ):
        """When no encoder_name is given, render falls back to self.encoder."""
        engine = EmbeddingArtEngine.from_registry(
            registry, default_encoder="differentiable", device="cpu"
        )
        engine.register_generator("image", small_generator)

        spec = ConceptSpec(text="flamingo")
        # No encoder_name — must use self.encoder
        result = engine.render(spec, output_modality="image", config=fast_config)
        assert isinstance(result, RenderResult)

    def test_render_raises_without_encoder(self, registry, small_generator, fast_config):
        """render() raises ValueError when no encoder is available anywhere."""
        engine = EmbeddingArtEngine.from_registry(registry, default_encoder=None, device="cpu")
        engine.register_generator("image", small_generator)

        spec = ConceptSpec(text="flamingo")
        with pytest.raises(ValueError, match="No encoder available"):
            engine.render(spec, output_modality="image", config=fast_config)


# ---------------------------------------------------------------------------
# Tests: render_compare
# ---------------------------------------------------------------------------


class TestRenderCompare:
    def _make_two_encoder_registry(self, diff_encoder) -> EncoderRegistry:
        """Registry with two entries pointing to the same encoder class."""
        reg = EncoderRegistry()
        for alias in ("enc_a", "enc_b"):
            card = EncoderCard(
                name=alias,
                capabilities=EncoderCapability.TEXT
                | EncoderCapability.IMAGE
                | EncoderCapability.BACKPROP_OPTIMIZABLE,
                embedding_dim=DifferentiableEncoder.EMBEDDING_DIM,
                memory_estimate_mb=10,
                backprop_cost=0.1,
            )
            reg._classes[alias] = type("_C", (), {"card": card})
            reg._instances[alias] = diff_encoder
        return reg

    def test_render_compare_returns_dict(self, diff_encoder, small_generator, fast_config):
        """render_compare returns a dict keyed by encoder name."""
        reg = self._make_two_encoder_registry(diff_encoder)
        engine = EmbeddingArtEngine.from_registry(reg, device="cpu")
        engine.register_generator("image", small_generator)

        spec = ConceptSpec(text="goldfish")
        results = engine.render_compare(
            spec,
            encoder_names=["enc_a", "enc_b"],
            output_modality="image",
            config=fast_config,
        )

        assert set(results.keys()) == {"enc_a", "enc_b"}

    def test_render_compare_all_values_are_render_results(
        self, diff_encoder, small_generator, fast_config
    ):
        """Every value in the render_compare dict is a RenderResult."""
        reg = self._make_two_encoder_registry(diff_encoder)
        engine = EmbeddingArtEngine.from_registry(reg, device="cpu")
        engine.register_generator("image", small_generator)

        spec = ConceptSpec(text="flamingo")
        results = engine.render_compare(spec, encoder_names=["enc_a", "enc_b"], config=fast_config)

        for name, result in results.items():
            assert isinstance(result, RenderResult), f"{name} did not return RenderResult"

    def test_render_compare_single_encoder(self, diff_encoder, small_generator, fast_config):
        """render_compare with a single encoder still returns a single-key dict."""
        reg = self._make_two_encoder_registry(diff_encoder)
        engine = EmbeddingArtEngine.from_registry(reg, device="cpu")
        engine.register_generator("image", small_generator)

        spec = ConceptSpec(text="ocean")
        results = engine.render_compare(spec, encoder_names=["enc_a"], config=fast_config)
        assert list(results.keys()) == ["enc_a"]
        assert isinstance(results["enc_a"], RenderResult)


# ---------------------------------------------------------------------------
# Tests: backward compatibility — optimize() still works
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_old_optimize_method_exists(self):
        """EmbeddingArtEngine still has an optimize() method."""
        assert hasattr(EmbeddingArtEngine, "optimize")
        assert callable(EmbeddingArtEngine.optimize)

    def test_old_optimize_signature_unchanged(self):
        """optimize() signature still accepts target, output_modality, config, etc."""
        sig = inspect.signature(EmbeddingArtEngine.optimize)
        param_names = list(sig.parameters.keys())
        # Must contain the original positional args
        assert "target" in param_names
        assert "output_modality" in param_names
        assert "config" in param_names

    def test_from_registry_does_not_break_existing_api(self, registry, diff_encoder):
        """An engine from from_registry still exposes register_generator and get_generator."""
        engine = EmbeddingArtEngine.from_registry(
            registry, default_encoder="differentiable", device="cpu"
        )
        # These are part of the v1 API and must still work
        assert hasattr(engine, "register_generator")
        assert hasattr(engine, "get_generator")
        assert hasattr(engine, "interpolation_series")
