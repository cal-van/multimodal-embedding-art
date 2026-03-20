"""
Tests for the INR (Implicit Neural Representation) generator.

TDD red phase: these tests are written before any implementation exists.
They define the full contract for FourierFeatureNetwork and INRGenerator.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

# ---------------------------------------------------------------------------
# FourierFeatureNetwork tests
# ---------------------------------------------------------------------------


class TestFourierFeatureNetwork:
    """Tests for the FourierFeatureNetwork MLP."""

    def test_output_shape(self):
        """forward() returns [1, 3, H, W] matching the configured dimensions."""
        from embedding_art.generators.inr import FourierFeatureNetwork

        net = FourierFeatureNetwork(height=32, width=32, hidden_dim=16, n_layers=2, n_frequencies=8)
        output = net()
        assert output.shape == (1, 3, 32, 32), f"Expected (1, 3, 32, 32), got {output.shape}"

    def test_non_square_output_shape(self):
        """forward() respects non-square height/width."""
        from embedding_art.generators.inr import FourierFeatureNetwork

        net = FourierFeatureNetwork(height=16, width=48, hidden_dim=8, n_layers=2, n_frequencies=4)
        output = net()
        assert output.shape == (1, 3, 16, 48)

    def test_output_range(self):
        """All pixel values lie in [0, 1] — sigmoid is the final activation."""
        from embedding_art.generators.inr import FourierFeatureNetwork

        net = FourierFeatureNetwork(height=16, width=16, hidden_dim=8, n_layers=2, n_frequencies=4)
        output = net()
        assert output.min().item() >= 0.0
        assert output.max().item() <= 1.0

    def test_output_is_deterministic_with_same_weights(self):
        """Two forward passes with no weight updates produce identical output."""
        from embedding_art.generators.inr import FourierFeatureNetwork

        net = FourierFeatureNetwork(height=8, width=8, hidden_dim=8, n_layers=2, n_frequencies=4)
        out1 = net()
        out2 = net()
        assert torch.allclose(out1, out2), "Same weights should produce identical output"

    def test_different_network_init_produces_different_output(self):
        """Two independently initialised networks (no shared seed) differ."""
        from embedding_art.generators.inr import FourierFeatureNetwork

        torch.manual_seed(0)
        net1 = FourierFeatureNetwork(height=16, width=16, hidden_dim=8, n_layers=2, n_frequencies=4)
        torch.manual_seed(1)
        net2 = FourierFeatureNetwork(height=16, width=16, hidden_dim=8, n_layers=2, n_frequencies=4)
        out1 = net1()
        out2 = net2()
        # Outputs should differ (astronomically unlikely to be equal)
        assert not torch.allclose(out1, out2), "Different init should produce different output"

    def test_gradients_flow_through_forward(self):
        """Gradients can backpropagate through forward() to the network parameters."""
        from embedding_art.generators.inr import FourierFeatureNetwork

        net = FourierFeatureNetwork(height=8, width=8, hidden_dim=8, n_layers=2, n_frequencies=4)
        output = net()
        loss = output.mean()
        loss.backward()

        # At least one parameter should have a non-None gradient
        has_grad = any(p.grad is not None for p in net.parameters())
        assert has_grad, "Backprop did not produce any parameter gradients"

    def test_coord_buffer_shape(self):
        """Pre-computed coordinate buffer has shape [H*W, 2]."""
        from embedding_art.generators.inr import FourierFeatureNetwork

        h, w = 12, 20
        net = FourierFeatureNetwork(height=h, width=w, hidden_dim=8, n_layers=2, n_frequencies=4)
        assert net.coords.shape == (h * w, 2)


# ---------------------------------------------------------------------------
# INRGenerator protocol conformance
# ---------------------------------------------------------------------------


class TestINRGeneratorProtocol:
    """INRGenerator must satisfy the DirectGenerator protocol."""

    def _make_generator(self, height: int = 16, width: int = 16) -> "INRGenerator":
        from embedding_art.generators.inr import INRGenerator

        return INRGenerator(
            height=height, width=width, hidden_dim=8, n_layers=2, n_frequencies=4, device="cpu"
        )

    def test_output_modality_is_image(self):
        gen = self._make_generator()
        assert gen.output_modality == "image"

    def test_device_property_returns_torch_device(self):
        gen = self._make_generator()
        assert isinstance(gen.device, torch.device)

    def test_device_is_cpu(self):
        gen = self._make_generator()
        assert gen.device == torch.device("cpu")

    def test_get_optimizable_parameters_returns_list(self):
        gen = self._make_generator()
        params = gen.get_optimizable_parameters()
        assert isinstance(params, list)

    def test_get_optimizable_parameters_non_empty(self):
        """There must be at least one optimizable parameter."""
        gen = self._make_generator()
        params = gen.get_optimizable_parameters()
        assert len(params) > 0

    def test_get_optimizable_parameters_are_nn_parameters(self):
        """Every entry in the list is a torch.nn.Parameter."""
        from torch.nn import Parameter

        gen = self._make_generator()
        params = gen.get_optimizable_parameters()
        for p in params:
            assert isinstance(p, Parameter), f"Expected nn.Parameter, got {type(p)}"

    def test_render_returns_tensor(self):
        gen = self._make_generator()
        out = gen.render()
        assert isinstance(out, torch.Tensor)

    def test_render_output_shape(self):
        """render() returns [1, 3, H, W]."""
        gen = self._make_generator(height=16, width=24)
        out = gen.render()
        assert out.shape == (1, 3, 16, 24)

    def test_render_output_range(self):
        """render() pixels are in [0, 1]."""
        gen = self._make_generator()
        out = gen.render()
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_render_is_differentiable(self):
        """Gradients flow from render() output back through all parameters."""
        gen = self._make_generator()
        out = gen.render()
        loss = out.mean()
        loss.backward()

        params = gen.get_optimizable_parameters()
        has_grad = any(p.grad is not None for p in params)
        assert has_grad, "No parameter gradients after backward through render()"

    def test_satisfies_direct_generator_protocol(self):
        """Runtime check: INRGenerator satisfies the DirectGenerator Protocol."""
        from embedding_art.generators.base import DirectGenerator
        from embedding_art.generators.inr import INRGenerator

        gen = INRGenerator(height=8, width=8, hidden_dim=4, n_layers=2, n_frequencies=2)
        # Protocol is @runtime_checkable only if Protocol defines runtime check —
        # DirectGenerator uses Protocol but is NOT @runtime_checkable, so we use
        # duck-type attribute checks instead.
        assert hasattr(gen, "output_modality")
        assert hasattr(gen, "device")
        assert hasattr(gen, "get_optimizable_parameters")
        assert hasattr(gen, "render")


# ---------------------------------------------------------------------------
# Integration: INRGenerator with OptimizationStrategy
# ---------------------------------------------------------------------------


class DifferentiableEncoder:
    """
    Differentiable encoder for integration tests (gradients flow through it).

    Uses a fixed linear projection so cosine-similarity loss has real gradients.
    """

    def __init__(self, embedding_dim: int = 8) -> None:
        torch.manual_seed(42)
        self._embedding_dim = embedding_dim
        self._device = torch.device("cpu")
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
        """Pool spatial dims → flatten → linear → L2-norm."""
        pooled = F.adaptive_avg_pool2d(tensor, output_size=(1, 1))
        flat = pooled.view(pooled.shape[0], -1)
        # Pad or truncate to embedding_dim
        if flat.shape[1] < self._embedding_dim:
            pad = torch.zeros(
                flat.shape[0],
                self._embedding_dim - flat.shape[1],
                device=flat.device,
            )
            flat = torch.cat([flat, pad], dim=1)
        else:
            flat = flat[:, : self._embedding_dim]
        return F.normalize(self._proj(flat), dim=-1)

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


class TestINRWithOptimizationStrategy:
    """INRGenerator integrates with OptimizationStrategy (3 steps, no real models)."""

    def test_three_step_optimization_runs_without_error(self):
        """OptimizationStrategy.render() completes 3 steps with INRGenerator."""
        from embedding_art.generators.inr import INRGenerator
        from embedding_art.core.strategies import OptimizationStrategy
        from embedding_art.core.concept import Concept
        from embedding_art.core.config import LossConfig, OptimizationConfig

        embedding_dim = 8
        encoder = DifferentiableEncoder(embedding_dim=embedding_dim)

        # Target concept with a fixed normalized embedding
        torch.manual_seed(7)
        target_emb = F.normalize(torch.randn(1, embedding_dim), dim=-1)
        target = Concept(embedding=target_emb, description="test_target")

        gen = INRGenerator(
            height=8, width=8, hidden_dim=8, n_layers=2, n_frequencies=4, device="cpu"
        )

        config = OptimizationConfig(
            steps=3,
            learning_rate=0.01,
            optimizer="adam",
            scheduler="constant",
            loss=LossConfig(similarity_weight=1.0),
        )

        strategy = OptimizationStrategy()
        result = strategy.render(target, gen, encoder, config)

        from embedding_art.core.render_result import RenderResult

        assert isinstance(result, RenderResult)
        assert result.output.shape == (1, 3, 8, 8)

    def test_optimization_records_history(self):
        """OptimizationHistory has entries after the loop runs."""
        from embedding_art.generators.inr import INRGenerator
        from embedding_art.core.strategies import OptimizationStrategy
        from embedding_art.core.concept import Concept
        from embedding_art.core.config import LossConfig, OptimizationConfig

        embedding_dim = 8
        encoder = DifferentiableEncoder(embedding_dim=embedding_dim)

        torch.manual_seed(7)
        target_emb = F.normalize(torch.randn(1, embedding_dim), dim=-1)
        target = Concept(embedding=target_emb, description="test_target")

        gen = INRGenerator(
            height=8, width=8, hidden_dim=8, n_layers=2, n_frequencies=4, device="cpu"
        )
        config = OptimizationConfig(
            steps=3,
            learning_rate=0.01,
            optimizer="adam",
            scheduler="constant",
            loss=LossConfig(similarity_weight=1.0),
        )

        strategy = OptimizationStrategy()
        result = strategy.render(target, gen, encoder, config)

        # History should have exactly 3 entries
        assert len(result.history) == 3

    def test_parameters_change_after_optimization(self):
        """Network weights are updated after 3 gradient steps."""
        from embedding_art.generators.inr import INRGenerator
        from embedding_art.core.strategies import OptimizationStrategy
        from embedding_art.core.concept import Concept
        from embedding_art.core.config import LossConfig, OptimizationConfig

        embedding_dim = 8
        encoder = DifferentiableEncoder(embedding_dim=embedding_dim)

        torch.manual_seed(7)
        target_emb = F.normalize(torch.randn(1, embedding_dim), dim=-1)
        target = Concept(embedding=target_emb, description="test_target")

        gen = INRGenerator(
            height=8, width=8, hidden_dim=8, n_layers=2, n_frequencies=4, device="cpu"
        )

        # Snapshot parameters before optimization
        params_before = [p.clone().detach() for p in gen.get_optimizable_parameters()]

        config = OptimizationConfig(
            steps=3,
            learning_rate=0.1,
            optimizer="adam",
            scheduler="constant",
            loss=LossConfig(similarity_weight=1.0),
        )

        strategy = OptimizationStrategy()
        strategy.render(target, gen, encoder, config)

        params_after = list(gen.get_optimizable_parameters())

        # At least one parameter tensor should have changed
        any_changed = any(
            not torch.allclose(before, after.detach())
            for before, after in zip(params_before, params_after)
        )
        assert any_changed, "Parameters were not updated during optimization"
