"""Tests for ``embedding_art.diffusion_priors.sd35_adapter``.

Real SD3.5 weights are 5 GB and not available in CI. These tests
substitute a stub pipeline via the adapter's ``loader`` kwarg to
verify the integration contract (lazy loading, schedule shapes,
score_fn velocity → noise conversion, conditioning unpacking, VSD
phi factory) without ever touching diffusers.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from embedding_art.diffusion_priors.sd35_adapter import (
    SD35DiffusionAdapter,
    build_vsd_phi_adapter,
)


def _make_stub_pipeline(velocity: torch.Tensor) -> MagicMock:
    """Return a stub SD3.5 pipeline whose transformer returns ``velocity``.

    The transformer is exposed as ``pipeline.transformer`` and returns
    a tuple ``(v,)`` so the adapter's ``[0]`` indexing works.
    """
    pipeline = MagicMock()
    transformer = MagicMock()
    transformer.return_value = (velocity,)
    pipeline.transformer = transformer
    fake_prompt_embeds = torch.randn(1, 154, 4096)
    fake_pooled = torch.randn(1, 2048)
    pipeline.encode_prompt.return_value = (
        fake_prompt_embeds,  # prompt_embeds
        torch.zeros_like(fake_prompt_embeds),  # negative_prompt_embeds
        fake_pooled,  # pooled
        torch.zeros_like(fake_pooled),  # negative_pooled
    )
    return pipeline


class TestSchedules:
    def test_alpha_schedule_is_one_minus_t(self) -> None:
        t = torch.tensor([0.0, 0.25, 0.5, 1.0])
        assert torch.allclose(
            SD35DiffusionAdapter.alpha_schedule(t),
            torch.tensor([1.0, 0.75, 0.5, 0.0]),
        )

    def test_sigma_schedule_is_t(self) -> None:
        t = torch.tensor([0.0, 0.25, 0.5, 1.0])
        assert torch.allclose(SD35DiffusionAdapter.sigma_schedule(t), t)


class TestLazyLoading:
    def test_pipeline_not_loaded_on_construct(self) -> None:
        loader = MagicMock()
        SD35DiffusionAdapter(loader=loader)
        assert loader.call_count == 0

    def test_pipeline_loaded_on_first_score_fn(self) -> None:
        velocity = torch.zeros(1, 16, 8, 8)
        pipeline = _make_stub_pipeline(velocity)
        loader = MagicMock(return_value=pipeline)
        adapter = SD35DiffusionAdapter(loader=loader)
        assert loader.call_count == 0

        x_t = torch.randn(1, 16, 8, 8, requires_grad=False)
        t = torch.tensor([0.5])
        # Pass an explicit 2-tuple conditioning that bypasses encode_prompt
        adapter.score_fn(x_t, t, conditioning=(torch.zeros(1, 1, 4), torch.zeros(1, 4)))
        assert loader.call_count == 1

        # Second call uses the cached pipeline; loader is NOT re-invoked.
        adapter.score_fn(x_t, t, conditioning=(torch.zeros(1, 1, 4), torch.zeros(1, 4)))
        assert loader.call_count == 1


class TestScoreFunctionVelocityToNoise:
    def test_eps_pred_equals_x_t_plus_one_minus_t_times_v(self) -> None:
        """Verifies the rectified-flow conversion eps = x_t + (1 - t) v."""
        x_t = torch.randn(2, 16, 4, 4)
        v_pred = torch.randn(2, 16, 4, 4)
        t = torch.tensor([0.2, 0.7])

        pipeline = _make_stub_pipeline(v_pred)
        adapter = SD35DiffusionAdapter(loader=MagicMock(return_value=pipeline))
        eps_pred = adapter.score_fn(x_t, t, conditioning=(torch.zeros(1, 1, 4), torch.zeros(1, 4)))

        coeff = (1.0 - t).view(-1, 1, 1, 1)
        expected = x_t + coeff * v_pred
        assert torch.allclose(eps_pred, expected)

    def test_at_t_zero_eps_equals_x_t_plus_v(self) -> None:
        """At t=0, alpha=1, sigma=0 — adapter should recover x_t + v."""
        x_t = torch.randn(1, 16, 4, 4)
        v_pred = torch.randn(1, 16, 4, 4)
        t = torch.tensor([0.0])

        pipeline = _make_stub_pipeline(v_pred)
        adapter = SD35DiffusionAdapter(loader=MagicMock(return_value=pipeline))
        eps_pred = adapter.score_fn(x_t, t, conditioning=(torch.zeros(1, 1, 4), torch.zeros(1, 4)))
        assert torch.allclose(eps_pred, x_t + v_pred)

    def test_at_t_one_eps_equals_x_t(self) -> None:
        """At t=1, alpha=0, sigma=1 — adapter should recover x_t (1-t=0)."""
        x_t = torch.randn(1, 16, 4, 4)
        v_pred = torch.randn(1, 16, 4, 4)
        t = torch.tensor([1.0])

        pipeline = _make_stub_pipeline(v_pred)
        adapter = SD35DiffusionAdapter(loader=MagicMock(return_value=pipeline))
        eps_pred = adapter.score_fn(x_t, t, conditioning=(torch.zeros(1, 1, 4), torch.zeros(1, 4)))
        assert torch.allclose(eps_pred, x_t)


class TestEncodePrompt:
    def test_delegates_to_pipeline(self) -> None:
        pipeline = _make_stub_pipeline(torch.zeros(1, 16, 4, 4))
        adapter = SD35DiffusionAdapter(loader=MagicMock(return_value=pipeline))
        out = adapter.encode_prompt("storm")
        assert isinstance(out, tuple) and len(out) == 4
        pipeline.encode_prompt.assert_called_once_with(prompt="storm")

    def test_raises_when_pipeline_has_no_encode_prompt(self) -> None:
        broken = MagicMock(spec=["transformer"])
        adapter = SD35DiffusionAdapter(loader=MagicMock(return_value=broken))
        with pytest.raises(NotImplementedError, match="encode_prompt"):
            adapter.encode_prompt("storm")


class TestConditioningUnpacking:
    def test_4_tuple_uses_index_0_and_2(self) -> None:
        prompt_embeds = torch.zeros(1, 1, 4)
        pooled = torch.zeros(1, 4)
        a, b = SD35DiffusionAdapter._unpack_conditioning(
            (prompt_embeds, torch.zeros(1, 1, 4), pooled, torch.zeros(1, 4))
        )
        assert a is prompt_embeds
        assert b is pooled

    def test_2_tuple_used_as_prompt_pooled(self) -> None:
        prompt_embeds = torch.zeros(1, 1, 4)
        pooled = torch.zeros(1, 4)
        a, b = SD35DiffusionAdapter._unpack_conditioning((prompt_embeds, pooled))
        assert a is prompt_embeds
        assert b is pooled

    def test_single_tensor_pooled_none(self) -> None:
        emb = torch.zeros(1, 1, 4)
        a, b = SD35DiffusionAdapter._unpack_conditioning(emb)
        assert a is emb
        assert b is None


class TestVSDPhiFactory:
    def test_returns_adapter_with_same_config(self) -> None:
        loader = MagicMock()
        base = SD35DiffusionAdapter(
            model_id="x/y", device="cpu", dtype=torch.float32, loader=loader
        )
        phi = build_vsd_phi_adapter(base)
        assert isinstance(phi, SD35DiffusionAdapter)
        assert phi.model_id == "x/y"
        assert phi.device == "cpu"
        assert phi.loader is loader


class TestSDSIntegration:
    """End-to-end smoke test: SDSLoss should run against the adapter."""

    def test_sds_loss_runs_with_adapter(self) -> None:
        from embedding_art.diffusion_priors.sds import SDSLoss

        x_t_shape = (1, 16, 4, 4)
        pipeline = _make_stub_pipeline(torch.randn(*x_t_shape))
        adapter = SD35DiffusionAdapter(loader=MagicMock(return_value=pipeline))

        sds = SDSLoss(
            score_fn=adapter.score_fn,
            sigma_schedule=adapter.sigma_schedule,
            alpha_schedule=adapter.alpha_schedule,
        )
        sds.seed(0)

        x = torch.randn(*x_t_shape, requires_grad=True)
        conditioning = (torch.zeros(1, 1, 4), torch.zeros(1, 4))
        loss = sds(x, conditioning, timestep=0.5)
        assert loss.requires_grad
        # Backprop should produce gradients on x.
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
