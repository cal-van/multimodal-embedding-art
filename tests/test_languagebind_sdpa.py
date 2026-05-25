"""Tests for ``LanguageBindEncoder.enable_sdpa_attention`` / ``_patch_clip_attention``.

The patching logic is purely structural — it walks ``nn.Module.modules()``
and rebinds the ``forward`` of every block matching the CLIP attention
shape. We exercise it against tiny in-tree CLIP-shaped blocks rather
than the real ``LanguageBindImage`` checkpoint (which is ~700 MB).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from embedding_art.encoders.languagebind import LanguageBindEncoder


def _make_clip_attention_block(embed_dim: int = 16, num_heads: int = 2) -> nn.Module:
    """Returns a minimal ``nn.Module`` with CLIP attention's structural
    shape: q_proj, k_proj, v_proj, out_proj, num_heads.
    """
    block = nn.Module()
    block.q_proj = nn.Linear(embed_dim, embed_dim)
    block.k_proj = nn.Linear(embed_dim, embed_dim)
    block.v_proj = nn.Linear(embed_dim, embed_dim)
    block.out_proj = nn.Linear(embed_dim, embed_dim)
    block.num_heads = num_heads
    return block


class _StubVisionTower(nn.Module):
    """A wrapper that contains N attention blocks."""

    def __init__(self, n_blocks: int = 3) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_make_clip_attention_block() for _ in range(n_blocks)])
        # Add a non-attention module to ensure the patcher skips it.
        self.linear = nn.Linear(16, 16)


class TestPatchCLIPAttention:
    def test_patches_every_clip_shaped_block(self) -> None:
        tower = _StubVisionTower(n_blocks=4)
        n = LanguageBindEncoder._patch_clip_attention(tower)
        assert n == 4

    def test_skips_modules_without_qkv(self) -> None:
        tower = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16))
        n = LanguageBindEncoder._patch_clip_attention(tower)
        assert n == 0

    def test_patched_block_forward_is_callable(self) -> None:
        block = _make_clip_attention_block()
        LanguageBindEncoder._patch_clip_attention(nn.ModuleList([block]))
        x = torch.randn(1, 8, 16)
        out = block.forward(x)
        # The patched forward returns a (output, attn_weights) tuple to
        # match the HF CLIPAttention contract.
        assert isinstance(out, tuple)
        out_tensor, attn_weights = out
        assert out_tensor.shape == x.shape
        # SDPA doesn't return weights — must be None.
        assert attn_weights is None

    def test_patched_block_accepts_optional_kwargs(self) -> None:
        """Patched forward must accept and tolerate optional kwargs."""
        block = _make_clip_attention_block()
        LanguageBindEncoder._patch_clip_attention(nn.ModuleList([block]))
        x = torch.randn(1, 4, 16)
        # CLIPAttention is called with positional + kwargs of varied
        # shapes; the patched forward must tolerate them.
        out, _ = block.forward(
            x,
            attention_mask=None,
            causal_attention_mask=None,
            output_attentions=False,
        )
        assert out.shape == x.shape

    def test_output_attentions_returns_none_weights(self) -> None:
        block = _make_clip_attention_block()
        LanguageBindEncoder._patch_clip_attention(nn.ModuleList([block]))
        x = torch.randn(1, 4, 16)
        _, w = block.forward(x, output_attentions=True)
        assert w is None


class TestEnableSDPAAttention:
    def test_no_models_loaded_returns_empty_report(self) -> None:
        encoder = LanguageBindEncoder.__new__(LanguageBindEncoder)
        # Skip __init__ — directly assign empty modality_models for
        # the test (we're only exercising the patcher, not the loader).
        encoder._modality_models = {}
        report = encoder.enable_sdpa_attention()
        assert report == {}

    def test_patches_every_loaded_modality(self) -> None:
        encoder = LanguageBindEncoder.__new__(LanguageBindEncoder)
        encoder._modality_models = {
            "image": {"model": _StubVisionTower(n_blocks=2)},
            "audio": {"model": _StubVisionTower(n_blocks=3)},
        }
        report = encoder.enable_sdpa_attention()
        assert report == {"image": 2, "audio": 3}

    def test_missing_model_entries_are_skipped(self) -> None:
        encoder = LanguageBindEncoder.__new__(LanguageBindEncoder)
        encoder._modality_models = {
            "image": {"model": _StubVisionTower(n_blocks=2)},
            "audio": {},  # no 'model' key
        }
        report = encoder.enable_sdpa_attention()
        assert report == {"image": 2}
