"""Tests for ``embedding_art.diffusion_priors.lora`` and the
``build_vsd_phi_adapter`` LoRA integration."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from embedding_art.diffusion_priors.lora import (
    LoRALinear,
    inject_lora_into_transformer,
    lora_disabled,
    lora_parameters,
    set_lora_enabled,
)


def _make_stub_attention_block() -> nn.Module:
    block = nn.Module()
    block.q_proj = nn.Linear(16, 16)
    block.k_proj = nn.Linear(16, 16)
    block.v_proj = nn.Linear(16, 16)
    block.out_proj = nn.Linear(16, 16)
    return block


def _make_stub_diffusers_attention_block() -> nn.Module:
    block = nn.Module()
    block.to_q = nn.Linear(16, 16)
    block.to_k = nn.Linear(16, 16)
    block.to_v = nn.Linear(16, 16)
    # diffusers ``to_out`` is Sequential(Linear, Dropout)
    block.to_out = nn.Sequential(nn.Linear(16, 16), nn.Dropout(0.0))
    return block


class TestLoRALinear:
    def test_zero_initial_delta(self) -> None:
        """LoRA delta is zero at init (lora_B is zero-initialised),
        so wrapped output equals base output."""
        base = nn.Linear(16, 16)
        lora = LoRALinear(base, rank=4)
        x = torch.randn(2, 16)
        out_wrapped = lora(x)
        out_base = base(x)
        torch.testing.assert_close(out_wrapped, out_base, atol=1e-6, rtol=1e-6)

    def test_base_params_are_frozen(self) -> None:
        base = nn.Linear(16, 16)
        lora = LoRALinear(base, rank=4)
        for p in lora.base.parameters():
            assert p.requires_grad is False

    def test_lora_params_are_trainable(self) -> None:
        base = nn.Linear(16, 16)
        lora = LoRALinear(base, rank=4)
        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad

    def test_disabled_is_identity_to_base(self) -> None:
        base = nn.Linear(16, 16)
        lora = LoRALinear(base, rank=4)
        lora.enabled = False
        # Force non-zero LoRA matrices to ensure 'disabled' actually
        # skips the delta.
        with torch.no_grad():
            lora.lora_A.fill_(0.1)
            lora.lora_B.fill_(0.1)
        x = torch.randn(2, 16)
        torch.testing.assert_close(lora(x), base(x), atol=1e-6, rtol=1e-6)

    def test_enabled_with_nonzero_lora_changes_output(self) -> None:
        base = nn.Linear(16, 16)
        lora = LoRALinear(base, rank=4)
        with torch.no_grad():
            lora.lora_A.fill_(0.1)
            lora.lora_B.fill_(0.1)
        x = torch.randn(2, 16)
        out_wrapped = lora(x)
        out_base = base(x)
        assert not torch.allclose(out_wrapped, out_base)

    def test_rejects_invalid_rank(self) -> None:
        base = nn.Linear(16, 16)
        try:
            LoRALinear(base, rank=0)
            raise AssertionError("rank=0 should raise")
        except ValueError:
            pass

    def test_lora_state_returns_clones(self) -> None:
        base = nn.Linear(16, 16)
        lora = LoRALinear(base, rank=4)
        state = lora.lora_state()
        assert "lora_A" in state
        assert "lora_B" in state
        # Mutating the snapshot must not affect the module.
        state["lora_A"].fill_(99.0)
        assert not torch.allclose(lora.lora_A, state["lora_A"])


class TestInjectLora:
    def test_replaces_clip_shape_projections(self) -> None:
        block = _make_stub_attention_block()
        injected = inject_lora_into_transformer(block, rank=4)
        assert len(injected) == 4
        for name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            assert isinstance(getattr(block, name), LoRALinear)

    def test_replaces_diffusers_shape_projections(self) -> None:
        block = _make_stub_diffusers_attention_block()
        injected = inject_lora_into_transformer(block, rank=4)
        assert len(injected) == 4
        for name in ["to_q", "to_k", "to_v"]:
            assert isinstance(getattr(block, name), LoRALinear)
        # to_out is a Sequential; index 0 should be wrapped.
        assert isinstance(block.to_out[0], LoRALinear)

    def test_skips_modules_without_attention_projections(self) -> None:
        tower = nn.Sequential(nn.Linear(16, 16), nn.ReLU())
        injected = inject_lora_into_transformer(tower)
        assert injected == []

    def test_initial_forward_unchanged_after_injection(self) -> None:
        """Injecting LoRA with the default zero-init must not change
        the forward output."""
        block = _make_stub_attention_block()
        x = torch.randn(2, 16)
        before = (
            block.q_proj(x).clone(),
            block.k_proj(x).clone(),
            block.v_proj(x).clone(),
            block.out_proj(x).clone(),
        )
        inject_lora_into_transformer(block)
        after = (
            block.q_proj(x),
            block.k_proj(x),
            block.v_proj(x),
            block.out_proj(x),
        )
        for a, b in zip(before, after):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


class TestSetLoraEnabled:
    def test_toggles_every_wrapper(self) -> None:
        block = _make_stub_attention_block()
        inject_lora_into_transformer(block)
        n = set_lora_enabled(block, enabled=False)
        assert n == 4
        for name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            assert getattr(block, name).enabled is False
        set_lora_enabled(block, enabled=True)
        for name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            assert getattr(block, name).enabled is True

    def test_no_op_on_unmodified_tree(self) -> None:
        tower = nn.Sequential(nn.Linear(16, 16), nn.ReLU())
        assert set_lora_enabled(tower, enabled=False) == 0


class TestLoraDisabledContext:
    def test_disables_then_restores(self) -> None:
        block = _make_stub_attention_block()
        inject_lora_into_transformer(block)
        # All wrappers start enabled.
        assert all(getattr(block, n).enabled for n in ["q_proj", "k_proj"])
        with lora_disabled(block):
            assert all(not getattr(block, n).enabled for n in ["q_proj", "k_proj"])
        assert all(getattr(block, n).enabled for n in ["q_proj", "k_proj"])

    def test_restores_after_exception(self) -> None:
        block = _make_stub_attention_block()
        inject_lora_into_transformer(block)
        try:
            with lora_disabled(block):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert all(getattr(block, n).enabled for n in ["q_proj", "k_proj"])


class TestLoraParameters:
    def test_yields_only_lora_params(self) -> None:
        block = _make_stub_attention_block()
        inject_lora_into_transformer(block, rank=4)
        params = list(lora_parameters(block))
        # 2 params per wrapper (lora_A, lora_B) × 4 wrappers = 8.
        assert len(params) == 8
        # All must be trainable.
        for p in params:
            assert p.requires_grad

    def test_empty_when_no_injection(self) -> None:
        tower = nn.Sequential(nn.Linear(16, 16))
        assert list(lora_parameters(tower)) == []


class _MinimalStubPipeline:
    """A pipeline-shape stub with a transformer attribute holding two
    attention blocks. Used to exercise the SD3.5 adapter's LoRA path
    without loading real diffusion weights."""

    def __init__(self) -> None:
        self.transformer = nn.Module()
        # Build the simplest possible tree with two attention blocks
        # inside a ModuleList.
        self.transformer.blocks = nn.ModuleList(
            [_make_stub_diffusers_attention_block() for _ in range(2)]
        )


def _stub_loader(model_id: str, device: Any, dtype: Any) -> _MinimalStubPipeline:
    return _MinimalStubPipeline()


class TestBuildVsdPhiAdapter:
    def test_injection_returns_phi_adapter_sharing_pipeline(self) -> None:
        from embedding_art.diffusion_priors.sd35_adapter import (
            SD35DiffusionAdapter,
            build_vsd_phi_adapter,
        )

        base = SD35DiffusionAdapter(loader=_stub_loader)
        phi = build_vsd_phi_adapter(base, rank=4, alpha=4.0)

        assert phi._pipeline is base._pipeline
        # 4 projections per block × 2 blocks = 8 wrappers.
        assert len(phi._lora_modules) == 8
        # Each wrapper contributes 2 trainable params.
        params = phi.lora_parameters()
        assert len(params) == 16
        for p in params:
            assert p.requires_grad

    def test_base_adapter_has_no_lora_parameters(self) -> None:
        from embedding_art.diffusion_priors.sd35_adapter import SD35DiffusionAdapter

        base = SD35DiffusionAdapter(loader=_stub_loader)
        assert base.lora_parameters() == []
