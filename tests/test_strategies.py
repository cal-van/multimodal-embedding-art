"""
Tests for RenderingStrategy protocol and OptimizationStrategy implementation.

Follows TDD red-green-refactor: tests written first, implementation follows.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn.functional as F

from embedding_art.core.concept import Concept
from embedding_art.core.config import LossConfig, OptimizationConfig
from embedding_art.core.render_result import RenderResult

# ---------------------------------------------------------------------------
# Differentiable mock helpers (gradients flow through the encoder)
# ---------------------------------------------------------------------------


class DifferentiableEncoder:
    """
    Mock encoder whose encode_for_optimization is fully differentiable.

    Instead of hashing tensor values, we apply a fixed linear projection so
    that gradients flow from the embedding loss back into the latent / output
    parameters.  This lets us verify that the optimisation loop actually
    updates parameters.
    """

    def __init__(self, embedding_dim: int = 16) -> None:
        torch.manual_seed(0)
        self._embedding_dim = embedding_dim
        self._device = torch.device("cpu")
        # Fixed weight matrix: (output_channels * H * W) -> embedding_dim
        # We don't know H/W in advance, so we use adaptive pooling first.
        self._proj = torch.nn.Linear(embedding_dim, embedding_dim, bias=False)
        with torch.no_grad():
            torch.nn.init.eye_(self._proj.weight)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def device(self) -> torch.device:
        return self._device

    def encode_for_optimization(self, tensor: torch.Tensor) -> torch.Tensor:
        """Differentiable path: adaptive-pool -> flatten -> linear -> L2-norm."""
        pooled = F.adaptive_avg_pool2d(tensor, output_size=(1, 1))  # [B, C, 1, 1]
        flat = pooled.view(pooled.shape[0], -1)  # [B, C]
        # Pad or truncate to embedding_dim
        if flat.shape[1] < self._embedding_dim:
            pad = torch.zeros(
                flat.shape[0], self._embedding_dim - flat.shape[1], device=flat.device
            )
            flat = torch.cat([flat, pad], dim=1)
        else:
            flat = flat[:, : self._embedding_dim]
        out = self._proj(flat)
        return F.normalize(out, dim=-1)

    # Minimal card duck-typing so CompositeLoss doesn't crash
    @property
    def card(self):
        from embedding_art.encoders.registry import EncoderCapability, EncoderCard

        return EncoderCard(
            name="differentiable_mock",
            capabilities=EncoderCapability.IMAGE | EncoderCapability.BACKPROP_OPTIMIZABLE,
            embedding_dim=self._embedding_dim,
            memory_estimate_mb=0,
            backprop_cost=0.0,
        )


class SmallMockGenerator:
    """
    Minimal latent generator for fast tests (small shapes, no real models).

    decode() is differentiable: bilinear upsample + sigmoid.
    """

    def __init__(
        self,
        latent_shape: tuple[int, ...] = (1, 4, 4, 4),
        output_channels: int = 4,
        output_size: int = 8,
    ) -> None:
        self._latent_shape = latent_shape
        self._output_channels = output_channels
        self._output_size = output_size
        self._device = torch.device("cpu")

    @property
    def latent_shape(self) -> tuple[int, ...]:
        return self._latent_shape

    @property
    def output_modality(self) -> str:
        return "image"

    @property
    def device(self) -> torch.device:
        return self._device

    def init_latent(self, seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            gen = torch.Generator().manual_seed(seed)
            latent = torch.randn(self._latent_shape, generator=gen)
        else:
            latent = torch.randn(self._latent_shape)
        return latent.to(self._device).requires_grad_(True)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        out = F.interpolate(
            latent,
            size=(self._output_size, self._output_size),
            mode="bilinear",
            align_corners=False,
        )
        # Adjust channels to output_channels
        if out.shape[1] != self._output_channels:
            out = out[:, : self._output_channels, :, :]
        return out.sigmoid()


class SmallDirectGenerator:
    """
    Mock DirectGenerator: holds optimizable parameters directly.

    render() just returns the parameters passed through sigmoid — fully
    differentiable so the optimizer can update them.
    """

    def __init__(self, output_channels: int = 4, output_size: int = 8) -> None:
        self._output_modality = "image"
        self._device = torch.device("cpu")
        self.params = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(1, output_channels, output_size, output_size))]
        )

    @property
    def output_modality(self) -> str:
        return self._output_modality

    @property
    def device(self) -> torch.device:
        return self._device

    def get_optimizable_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.params)

    def render(self) -> torch.Tensor:
        return self.params[0].sigmoid()


def _make_target_concept(embedding_dim: int = 16) -> Concept:
    torch.manual_seed(99)
    emb = F.normalize(torch.randn(1, embedding_dim), dim=-1)
    return Concept(embedding=emb, description="test_target")


def _fast_config(**overrides) -> OptimizationConfig:
    """OptimizationConfig with tiny step count for fast tests."""
    defaults = dict(
        steps=5,
        learning_rate=0.01,
        optimizer="adam",
        scheduler="constant",
        loss=LossConfig(
            similarity_weight=1.0,
            feature_matching_weight=0.0,
            sae_feature_weight=0.0,
        ),
    )
    defaults.update(overrides)
    return OptimizationConfig(**defaults)


# ---------------------------------------------------------------------------
# Red-phase tests (all failing until implementation exists)
# ---------------------------------------------------------------------------


class TestOptimizationStrategyRuns:
    """test_optimization_strategy_runs: the loop executes and returns RenderResult."""

    def test_returns_render_result(self):
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallMockGenerator(latent_shape=(1, 4, 4, 4), output_channels=4, output_size=8)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config()

        result = strategy.render(target, generator, encoder, config)

        assert isinstance(result, RenderResult)

    def test_result_has_output_tensor(self):
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallMockGenerator(latent_shape=(1, 4, 4, 4), output_channels=4, output_size=8)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config()

        result = strategy.render(target, generator, encoder, config)

        assert isinstance(result.output, torch.Tensor)
        assert result.output.shape[0] == 1  # batch dim

    def test_result_output_is_detached(self):
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallMockGenerator(latent_shape=(1, 4, 4, 4), output_channels=4, output_size=8)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config()

        result = strategy.render(target, generator, encoder, config)

        assert not result.output.requires_grad


class TestOptimizationStrategyHistory:
    """test_optimization_strategy_loss_decreases: history is populated across steps."""

    def test_history_has_correct_number_of_entries(self):
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallMockGenerator(latent_shape=(1, 4, 4, 4), output_channels=4, output_size=8)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config(steps=10)

        result = strategy.render(target, generator, encoder, config)

        assert len(result.history) == 10

    def test_history_records_similarity_values(self):
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallMockGenerator(latent_shape=(1, 4, 4, 4), output_channels=4, output_size=8)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config(steps=10)

        result = strategy.render(target, generator, encoder, config)

        assert len(result.history.similarity_values) == 10

    def test_history_step_indices_are_sequential(self):
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallMockGenerator(latent_shape=(1, 4, 4, 4), output_channels=4, output_size=8)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config(steps=5)

        result = strategy.render(target, generator, encoder, config)

        assert result.history.steps == list(range(5))


class TestOptimizationStrategyWithLatentGenerator:
    """test_optimization_strategy_with_latent_generator: calls init_latent and decode."""

    def test_calls_init_latent_once(self):
        from unittest.mock import patch

        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallMockGenerator(latent_shape=(1, 4, 4, 4), output_channels=4, output_size=8)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config(steps=3)

        with patch.object(generator, "init_latent", wraps=generator.init_latent) as mock_init:
            strategy.render(target, generator, encoder, config)

        mock_init.assert_called_once()

    def test_calls_decode_every_step(self):
        from unittest.mock import patch

        from embedding_art.core.strategies import OptimizationStrategy

        n_steps = 4
        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallMockGenerator(latent_shape=(1, 4, 4, 4), output_channels=4, output_size=8)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config(steps=n_steps)

        with patch.object(generator, "decode", wraps=generator.decode) as mock_decode:
            strategy.render(target, generator, encoder, config)

        assert mock_decode.call_count == n_steps

    def test_passes_seed_to_init_latent(self):
        from unittest.mock import patch

        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallMockGenerator(latent_shape=(1, 4, 4, 4), output_channels=4, output_size=8)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config(steps=2, seed=42)

        with patch.object(generator, "init_latent", wraps=generator.init_latent) as mock_init:
            strategy.render(target, generator, encoder, config)

        mock_init.assert_called_once_with(42)


class TestOptimizationStrategyOptimizerBuilding:
    """test_optimization_strategy_builds_adam: correct optimizer type from config."""

    @pytest.mark.parametrize(
        "optimizer_name,expected_class",
        [
            ("adam", torch.optim.Adam),
            ("adamw", torch.optim.AdamW),
            ("sgd", torch.optim.SGD),
        ],
    )
    def test_builds_correct_optimizer_type(self, optimizer_name, expected_class):
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        param = torch.nn.Parameter(torch.randn(4))
        config = _fast_config(optimizer=optimizer_name)

        optimizer = strategy._build_optimizer([param], config)

        assert isinstance(optimizer, expected_class)

    def test_optimizer_uses_configured_learning_rate(self):
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        param = torch.nn.Parameter(torch.randn(4))
        config = _fast_config(learning_rate=0.123)

        optimizer = strategy._build_optimizer([param], config)

        assert abs(optimizer.param_groups[0]["lr"] - 0.123) < 1e-9

    def test_unknown_optimizer_falls_back_to_adam(self):
        """Passing an unexpected string returns Adam (the safe default)."""
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        param = torch.nn.Parameter(torch.randn(4))
        # Bypass Literal type-check at runtime
        config = _fast_config()
        config.optimizer = "nonexistent"  # type: ignore[assignment]

        optimizer = strategy._build_optimizer([param], config)

        assert isinstance(optimizer, torch.optim.Adam)


class TestOptimizationStrategySchedulerBuilding:
    """test_optimization_strategy_builds_cosine_scheduler."""

    def test_builds_cosine_scheduler(self):
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        param = torch.nn.Parameter(torch.randn(4))
        config = _fast_config(scheduler="cosine", steps=100)

        optimizer = strategy._build_optimizer([param], config)
        scheduler = strategy._build_scheduler(optimizer, config)

        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_builds_linear_scheduler(self):
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        param = torch.nn.Parameter(torch.randn(4))
        config = _fast_config(scheduler="linear", steps=100)

        optimizer = strategy._build_optimizer([param], config)
        scheduler = strategy._build_scheduler(optimizer, config)

        assert isinstance(scheduler, torch.optim.lr_scheduler.LinearLR)

    def test_constant_scheduler_returns_none(self):
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        param = torch.nn.Parameter(torch.randn(4))
        config = _fast_config(scheduler="constant")

        optimizer = strategy._build_optimizer([param], config)
        scheduler = strategy._build_scheduler(optimizer, config)

        assert scheduler is None


class TestOptimizationStrategyWithDirectGenerator:
    """DirectGenerator path: uses get_optimizable_parameters + render()."""

    def test_calls_render_not_decode(self):
        from unittest.mock import patch

        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallDirectGenerator(output_channels=4, output_size=8)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config(steps=3)

        with patch.object(generator, "render", wraps=generator.render) as mock_render:
            strategy.render(target, generator, encoder, config)

        assert mock_render.call_count == 3

    def test_does_not_call_init_latent_for_direct_generator(self):
        """DirectGenerators have no init_latent — strategy must not call it."""
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallDirectGenerator(output_channels=4, output_size=8)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config(steps=3)

        # Should not raise AttributeError even though generator has no init_latent
        result = strategy.render(target, generator, encoder, config)
        assert isinstance(result, RenderResult)

    def test_returns_render_result_for_direct_generator(self):
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallDirectGenerator(output_channels=4, output_size=8)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config(steps=5)

        result = strategy.render(target, generator, encoder, config)

        assert isinstance(result, RenderResult)
        assert isinstance(result.output, torch.Tensor)


class TestRenderingStrategyProtocol:
    """test_rendering_strategy_protocol: OptimizationStrategy satisfies the protocol."""

    def test_optimization_strategy_has_render_method(self):
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        assert hasattr(strategy, "render")
        assert callable(strategy.render)

    def test_optimization_strategy_is_instance_of_protocol(self):
        """Runtime isinstance check with Protocol (requires runtime_checkable)."""
        from embedding_art.core.strategies import OptimizationStrategy, RenderingStrategy

        strategy = OptimizationStrategy()
        # Protocol must be decorated with @runtime_checkable for this to work
        assert isinstance(strategy, RenderingStrategy)

    def test_rendering_strategy_is_runtime_checkable(self):
        """Confirm the Protocol itself is inspectable at runtime."""
        from embedding_art.core.strategies import RenderingStrategy

        # Should not raise TypeError — protocol must be @runtime_checkable
        try:
            isinstance(object(), RenderingStrategy)
        except TypeError as exc:
            pytest.fail(f"RenderingStrategy is not @runtime_checkable: {exc}")


class TestEncoderNameInResult:
    """encoder_name field in RenderResult is populated correctly."""

    def test_encoder_name_is_string(self):
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallMockGenerator(latent_shape=(1, 4, 4, 4), output_channels=4, output_size=8)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config()

        result = strategy.render(target, generator, encoder, config)

        assert isinstance(result.encoder_name, str)

    def test_encoder_name_uses_card_name(self):
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallMockGenerator(latent_shape=(1, 4, 4, 4), output_channels=4, output_size=8)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config()

        result = strategy.render(target, generator, encoder, config)

        assert result.encoder_name == "differentiable_mock"


# ---------------------------------------------------------------------------
# DiffusionGuidanceStrategy helpers
# ---------------------------------------------------------------------------


class GuidedMockGenerator:
    """Mock generator that implements generate_guided() and returns a RenderResult."""

    def __init__(
        self,
        output_channels: int = 4,
        output_size: int = 8,
        return_raw_tensor: bool = False,
    ) -> None:
        self._output_channels = output_channels
        self._output_size = output_size
        self._return_raw_tensor = return_raw_tensor
        self.generate_guided_calls: list[dict] = []

    def generate_guided(
        self,
        target_embedding: torch.Tensor,
        encoder: Any,
        loss_fn: Any,
        config: Any,
    ) -> Any:
        """Record call arguments and return a synthetic result."""
        self.generate_guided_calls.append(
            {
                "target_embedding": target_embedding,
                "encoder": encoder,
                "loss_fn": loss_fn,
                "config": config,
            }
        )

        output = torch.rand(1, self._output_channels, self._output_size, self._output_size)

        if self._return_raw_tensor:
            return output

        from embedding_art.core.render_result import OptimizationHistory, RenderResult

        history = OptimizationHistory()
        return RenderResult(
            output=output.detach(),
            history=history,
            encoder_name="guided_mock",
            final_similarity=0.75,
            config=config,
        )


# ---------------------------------------------------------------------------
# DiffusionGuidanceStrategy tests
# ---------------------------------------------------------------------------


class TestDiffusionGuidanceStrategyInit:
    """DiffusionGuidanceStrategy can be constructed with and without SAE."""

    def test_default_init_has_no_sae(self):
        from embedding_art.core.strategies import DiffusionGuidanceStrategy

        strategy = DiffusionGuidanceStrategy()

        assert strategy.sae is None

    def test_init_with_sae_stores_it(self):
        from embedding_art.core.strategies import DiffusionGuidanceStrategy

        sentinel = object()
        strategy = DiffusionGuidanceStrategy(sae=sentinel)

        assert strategy.sae is sentinel


class TestDiffusionGuidanceStrategySatisfiesProtocol:
    """DiffusionGuidanceStrategy must satisfy the RenderingStrategy protocol."""

    def test_has_render_method(self):
        from embedding_art.core.strategies import DiffusionGuidanceStrategy

        strategy = DiffusionGuidanceStrategy()

        assert hasattr(strategy, "render")
        assert callable(strategy.render)

    def test_is_instance_of_rendering_strategy_protocol(self):
        from embedding_art.core.strategies import DiffusionGuidanceStrategy, RenderingStrategy

        strategy = DiffusionGuidanceStrategy()

        assert isinstance(strategy, RenderingStrategy)


class TestDiffusionGuidanceStrategyDelegation:
    """render() delegates to generator.generate_guided() exactly once."""

    def test_calls_generate_guided_once(self):
        from embedding_art.core.strategies import DiffusionGuidanceStrategy

        strategy = DiffusionGuidanceStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = GuidedMockGenerator()
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config()

        strategy.render(target, generator, encoder, config)

        assert len(generator.generate_guided_calls) == 1

    def test_passes_target_embedding_to_generate_guided(self):
        from embedding_art.core.strategies import DiffusionGuidanceStrategy

        strategy = DiffusionGuidanceStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = GuidedMockGenerator()
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config()

        strategy.render(target, generator, encoder, config)

        call_kwargs = generator.generate_guided_calls[0]
        assert torch.equal(call_kwargs["target_embedding"], target.embedding)

    def test_passes_encoder_to_generate_guided(self):
        from embedding_art.core.strategies import DiffusionGuidanceStrategy

        strategy = DiffusionGuidanceStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = GuidedMockGenerator()
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config()

        strategy.render(target, generator, encoder, config)

        call_kwargs = generator.generate_guided_calls[0]
        assert call_kwargs["encoder"] is encoder

    def test_passes_config_to_generate_guided(self):
        from embedding_art.core.strategies import DiffusionGuidanceStrategy

        strategy = DiffusionGuidanceStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = GuidedMockGenerator()
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config()

        strategy.render(target, generator, encoder, config)

        call_kwargs = generator.generate_guided_calls[0]
        assert call_kwargs["config"] is config


class TestDiffusionGuidanceStrategyReturnValue:
    """render() returns a RenderResult regardless of generator output type."""

    def test_returns_render_result_when_generator_returns_render_result(self):
        from embedding_art.core.strategies import DiffusionGuidanceStrategy

        strategy = DiffusionGuidanceStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = GuidedMockGenerator(return_raw_tensor=False)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config()

        result = strategy.render(target, generator, encoder, config)

        assert isinstance(result, RenderResult)

    def test_returns_render_result_when_generator_returns_raw_tensor(self):
        """When generate_guided() returns a raw tensor, it must be wrapped."""
        from embedding_art.core.strategies import DiffusionGuidanceStrategy

        strategy = DiffusionGuidanceStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = GuidedMockGenerator(return_raw_tensor=True)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config()

        result = strategy.render(target, generator, encoder, config)

        assert isinstance(result, RenderResult)

    def test_wraps_raw_tensor_into_output_field(self):
        from embedding_art.core.strategies import DiffusionGuidanceStrategy

        strategy = DiffusionGuidanceStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = GuidedMockGenerator(return_raw_tensor=True)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config()

        result = strategy.render(target, generator, encoder, config)

        assert isinstance(result.output, torch.Tensor)

    def test_passthrough_render_result_from_generator(self):
        """When generator already returns a RenderResult, return it unchanged."""
        from embedding_art.core.strategies import DiffusionGuidanceStrategy

        strategy = DiffusionGuidanceStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = GuidedMockGenerator(return_raw_tensor=False)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config()

        result = strategy.render(target, generator, encoder, config)

        # The mock sets final_similarity=0.75 on the RenderResult it returns
        assert result.final_similarity == 0.75

    def test_encoder_name_populated_for_wrapped_tensor(self):
        """Wrapped raw-tensor result should populate encoder_name from card."""
        from embedding_art.core.strategies import DiffusionGuidanceStrategy

        strategy = DiffusionGuidanceStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = GuidedMockGenerator(return_raw_tensor=True)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config()

        result = strategy.render(target, generator, encoder, config)

        assert result.encoder_name == "differentiable_mock"

    def test_empty_history_for_wrapped_tensor(self):
        """Wrapped raw-tensor result should have an empty OptimizationHistory."""
        from embedding_art.core.render_result import OptimizationHistory
        from embedding_art.core.strategies import DiffusionGuidanceStrategy

        strategy = DiffusionGuidanceStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = GuidedMockGenerator(return_raw_tensor=True)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config()

        result = strategy.render(target, generator, encoder, config)

        assert isinstance(result.history, OptimizationHistory)
        assert len(result.history) == 0


# ---------------------------------------------------------------------------
# Autocast wiring
# ---------------------------------------------------------------------------


class TestAutocastWiring:
    """``OptimizationStrategy`` resolves ``autocast_dtype`` correctly and
    only wraps the forward+loss section in ``torch.autocast`` when
    non-fp32."""

    def test_fp32_resolves_to_nullcontext(self):
        from contextlib import nullcontext

        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        ctx, dtype = strategy._resolve_autocast(encoder, "fp32")
        assert dtype is None
        assert isinstance(ctx, type(nullcontext()))

    def test_bf16_resolves_to_torch_autocast_context(self):
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        ctx, dtype = strategy._resolve_autocast(encoder, "bf16")
        assert dtype == torch.bfloat16
        # torch.autocast is exposed as torch.amp.autocast_mode.autocast
        assert isinstance(ctx, torch.amp.autocast_mode.autocast)

    def test_fp16_resolves_to_torch_autocast_context(self):
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        ctx, dtype = strategy._resolve_autocast(encoder, "fp16")
        assert dtype == torch.float16
        assert isinstance(ctx, torch.amp.autocast_mode.autocast)

    def test_unknown_dtype_raises(self):
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        with pytest.raises(ValueError, match="Unknown autocast_dtype"):
            strategy._resolve_autocast(encoder, "fp8")

    def test_strategy_runs_under_bf16_on_cpu(self):
        """End-to-end smoke test: full loop runs under bf16 autocast."""
        from embedding_art.core.strategies import OptimizationStrategy

        strategy = OptimizationStrategy()
        encoder = DifferentiableEncoder(embedding_dim=16)
        generator = SmallMockGenerator(latent_shape=(1, 4, 4, 4), output_channels=4, output_size=8)
        target = _make_target_concept(embedding_dim=16)
        config = _fast_config(autocast_dtype="bf16")

        result = strategy.render(target, generator, encoder, config)
        assert isinstance(result, RenderResult)
        assert result.output.dtype == torch.float32  # final output is detached and clean


# ---------------------------------------------------------------------------
# Video shape handling (channels-first decode -> frame-first pipeline)
# ---------------------------------------------------------------------------


class _ChannelsFirstVideoGenerator:
    """Decodes to channels-first [B, C, F, H, W] like LTX-Video."""

    def __init__(self, frames: int = 5, size: int = 8) -> None:
        self._frames = frames
        self._size = size
        self._device = torch.device("cpu")

    @property
    def latent_shape(self) -> tuple[int, ...]:
        return (1, 4, self._frames, self._size, self._size)

    @property
    def output_modality(self) -> str:
        return "video"

    @property
    def device(self) -> torch.device:
        return self._device

    def init_latent(self, seed: int | None = None) -> torch.Tensor:
        return torch.randn(self.latent_shape).requires_grad_(True)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        # [B, 4, F, H, W] -> [B, 3, F, H, W] in [0, 1] (channels-first)
        return latent[:, :3].sigmoid()


class _RecordingVideoEncoder:
    """Records the shape passed to encode_video_for_optimization."""

    def __init__(self, embedding_dim: int = 16) -> None:
        self._embedding_dim = embedding_dim
        self._device = torch.device("cpu")
        self.seen_video_shape: tuple[int, ...] | None = None

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def device(self) -> torch.device:
        return self._device

    def encode_video_for_optimization(self, tensor: torch.Tensor) -> torch.Tensor:
        self.seen_video_shape = tuple(tensor.shape)
        flat = tensor.reshape(tensor.shape[0], -1)[:, : self._embedding_dim]
        if flat.shape[1] < self._embedding_dim:
            flat = F.pad(flat, (0, self._embedding_dim - flat.shape[1]))
        return F.normalize(flat, dim=-1)

    @property
    def card(self):
        from embedding_art.encoders.registry import EncoderCapability, EncoderCard

        return EncoderCard(
            name="recording_video_mock",
            capabilities=EncoderCapability.VIDEO | EncoderCapability.BACKPROP_OPTIMIZABLE,
            embedding_dim=self._embedding_dim,
            memory_estimate_mb=0,
            backprop_cost=0.0,
        )


def test_to_canonical_video_permutes_channels_first():
    from embedding_art.core.strategies import OptimizationStrategy

    chans_first = torch.rand(1, 3, 5, 8, 8)  # [B, C, F, H, W]
    out = OptimizationStrategy._to_canonical_video(chans_first)
    assert out.shape == (1, 5, 3, 8, 8)  # -> [B, F, C, H, W]


def test_to_canonical_video_leaves_frame_first_untouched():
    from embedding_art.core.strategies import OptimizationStrategy

    frame_first = torch.rand(1, 5, 3, 8, 8)  # already [B, F, C, H, W]
    out = OptimizationStrategy._to_canonical_video(frame_first)
    assert out.shape == (1, 5, 3, 8, 8)


def test_video_render_feeds_frame_first_to_encoder():
    """A channels-first video generator must reach the video encoder as
    frame-first [B, F, C, H, W] (channels at axis 2), via output_modality."""
    from embedding_art.core.strategies import OptimizationStrategy

    gen = _ChannelsFirstVideoGenerator(frames=5, size=8)
    enc = _RecordingVideoEncoder()
    target = Concept(embedding=F.normalize(torch.randn(1, 16), dim=-1), description="v")
    config = OptimizationConfig(steps=1, loss=LossConfig(feature_matching_weight=0.0))

    OptimizationStrategy().render(target, gen, enc, config, output_modality="video")

    assert enc.seen_video_shape is not None
    # axis 2 must be the channel dim (==3), i.e. frame-first layout.
    assert enc.seen_video_shape[2] == 3, enc.seen_video_shape
    assert enc.seen_video_shape[1] == 5  # frames preserved at axis 1
