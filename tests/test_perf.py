"""Tests for ``embedding_art.perf``.

We can't execute MPS ops in CI (no Apple GPU), so these tests focus
on the *correctness of the wrappers*: that they degrade silently on
non-MPS backends, that the patch functions hit the right attributes,
and that the diffusers knob reporter is honest about what it applied.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
import torch.nn as nn

from embedding_art.perf import (
    PerfProfile,
    apply_diffusers_perf_knobs,
    compile_module,
    default_compile_callback,
    empty_mps_cache,
    mps_available,
    patch_attention_to_sdpa,
)


class TestMpsAvailable:
    def test_returns_bool(self) -> None:
        assert isinstance(mps_available(), bool)

    def test_empty_cache_no_op_on_non_mps(self) -> None:
        # On CPU/CUDA we still call this — must not raise.
        empty_mps_cache()


class TestCompileModule:
    def test_returns_module_unchanged_when_compile_disabled(self, monkeypatch) -> None:
        # Force the absence of torch.compile by deleting the attribute.
        monkeypatch.delattr(torch, "compile", raising=False)
        module = nn.Linear(8, 4)
        assert compile_module(module) is module

    def test_silently_falls_back_on_compile_failure(self, monkeypatch) -> None:
        def _boom(*args, **kwargs):
            raise RuntimeError("inductor down")

        monkeypatch.setattr(torch, "compile", _boom)
        module = nn.Linear(8, 4)
        assert compile_module(module) is module

    def test_returns_compile_result_on_success(self, monkeypatch) -> None:
        sentinel = object()
        monkeypatch.setattr(torch, "compile", lambda *a, **k: sentinel)
        assert compile_module(nn.Linear(8, 4)) is sentinel


class TestPatchAttentionToSdpa:
    def _make_block(self, embed_dim: int = 16, num_heads: int = 2) -> nn.Module:
        block = nn.Module()
        block.q = nn.Linear(embed_dim, embed_dim)
        block.k = nn.Linear(embed_dim, embed_dim)
        block.v = nn.Linear(embed_dim, embed_dim)
        block.out_proj = nn.Linear(embed_dim, embed_dim)
        block.num_heads = num_heads
        return block

    def test_patches_when_attributes_present(self) -> None:
        block = self._make_block()
        assert patch_attention_to_sdpa(block) is True

        x = torch.randn(1, 4, 16)
        out = block.forward(x)
        assert out.shape == x.shape

    def test_returns_false_when_qkv_missing(self) -> None:
        block = nn.Module()
        block.num_heads = 2
        block.out_proj = nn.Linear(8, 8)
        assert patch_attention_to_sdpa(block) is False

    def test_returns_false_when_out_proj_missing(self) -> None:
        block = nn.Module()
        block.q = nn.Linear(8, 8)
        block.k = nn.Linear(8, 8)
        block.v = nn.Linear(8, 8)
        block.num_heads = 2
        assert patch_attention_to_sdpa(block) is False


class TestApplyDiffusersPerfKnobs:
    def test_all_knobs_applied_when_supported(self) -> None:
        pipeline = MagicMock()
        report = apply_diffusers_perf_knobs(pipeline)
        assert report == {
            "fuse_qkv": True,
            "enable_attention_slicing": True,
            "enable_vae_tiling": True,
        }
        pipeline.fuse_qkv_projections.assert_called_once()
        pipeline.enable_attention_slicing.assert_called_once_with("max")
        pipeline.vae.enable_tiling.assert_called_once()

    def test_missing_methods_are_silently_skipped(self) -> None:
        pipeline = MagicMock(spec=[])  # no methods at all
        report = apply_diffusers_perf_knobs(pipeline)
        assert report == {
            "fuse_qkv": False,
            "enable_attention_slicing": False,
            "enable_vae_tiling": False,
        }

    def test_individual_knobs_can_be_disabled(self) -> None:
        pipeline = MagicMock()
        report = apply_diffusers_perf_knobs(
            pipeline,
            fuse_qkv=False,
            enable_attention_slicing=False,
            enable_vae_tiling=False,
        )
        assert report == {
            "fuse_qkv": False,
            "enable_attention_slicing": False,
            "enable_vae_tiling": False,
        }
        pipeline.fuse_qkv_projections.assert_not_called()


class TestPerfProfile:
    def test_can_be_entered_and_exited_without_op(self, tmp_path) -> None:
        # Without torch.profiler activities, this should at least not
        # raise. The trace export may or may not happen depending on
        # the active PyTorch.
        with PerfProfile(output_dir=tmp_path) as prof:
            x = torch.randn(8, 8)
            (x @ x.T).sum()
        # summary() should be a string (possibly empty on platforms
        # where the profiler refused to initialise).
        assert isinstance(prof.summary(), str)


class TestDefaultCompileCallback:
    def test_none_returns_identity(self) -> None:
        m = nn.Linear(8, 4)
        cb = default_compile_callback(m, None)
        assert cb(m) is m

    def test_unknown_mode_returns_identity(self) -> None:
        m = nn.Linear(8, 4)
        cb = default_compile_callback(m, "lightspeed")
        assert cb(m) is m

    def test_known_mode_invokes_compile(self, monkeypatch) -> None:
        sentinel = object()
        monkeypatch.setattr(torch, "compile", lambda *a, **k: sentinel)
        m = nn.Linear(8, 4)
        cb = default_compile_callback(m, "reduce-overhead")
        assert cb(m) is sentinel
