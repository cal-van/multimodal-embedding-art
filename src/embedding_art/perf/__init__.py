"""Apple-Silicon-aware performance utilities.

This module is the single home for the perf-oriented levers we apply
on M1/M2 Max:

* :func:`mps_available` — is MPS the right device this run.
* :func:`empty_mps_cache` — release the MPS allocator's working set
  between modality boundaries.
* :func:`compile_module` — opt-in ``torch.compile`` wrapper that
  degrades to identity on platforms / PyTorch versions where the MPS
  Inductor backend isn't usable.
* :func:`patch_attention_to_sdpa` — best-effort monkey-patch that
  routes a ViT attention block's forward to
  :func:`torch.nn.functional.scaled_dot_product_attention`, which on
  PyTorch 2.4+ MPS dispatches to a flash-attention-style kernel.
* :class:`PerfProfile` — thin wrapper around ``torch.profiler`` that
  emits a Chrome-trace and a per-op summary table to a directory.

Everything in here is *off by default*. Callers opt in via
:class:`OptimizationConfig` / showcase CLI flags. The module never
hard-imports anything beyond ``torch`` itself so import cost stays
negligible.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch

logger = logging.getLogger(__name__)


def mps_available() -> bool:
    """True iff MPS is available *and* built into the active PyTorch."""
    return torch.backends.mps.is_available() and torch.backends.mps.is_built()


def empty_mps_cache() -> None:
    """Release the MPS allocator's cached memory.

    Apple's unified memory keeps cached tensors around until pressure
    forces eviction; explicitly emptying between modality boundaries
    prevents the 4-modality showcase from accumulating peak working
    sets. No-op on non-MPS backends.
    """
    if not mps_available():
        return
    if hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def compile_module(
    module: Any,
    *,
    mode: str = "reduce-overhead",
    dynamic: bool = True,
    fullgraph: bool = False,
) -> Any:
    """Wrap ``module`` with :func:`torch.compile` if safe.

    Args:
        module: An ``nn.Module``, a callable, or anything else
            ``torch.compile`` accepts.
        mode: Compile mode (``"reduce-overhead"`` is the default and
            gives the best wins on the optimisation hot path).
        dynamic: Use dynamic shapes — required for the text-anchor
            batching path where vocab size can vary.
        fullgraph: ``True`` insists on a single graph; we default to
            ``False`` so any unsupported op falls back cleanly.

    Returns:
        The compiled module on success; the *original* module on any
        failure (e.g. unsupported PyTorch version, MPS backend lacks
        a required op). The fallback is silent — callers can't tell
        whether the compile succeeded, only that the call site is
        stable.
    """
    if not hasattr(torch, "compile"):
        logger.info("torch.compile unavailable; returning module as-is")
        return module
    try:
        return torch.compile(module, mode=mode, dynamic=dynamic, fullgraph=fullgraph)
    except Exception as exc:  # pragma: no cover - depends on PyTorch internals
        logger.warning("torch.compile failed (%s); returning module as-is", exc)
        return module


def patch_attention_to_sdpa(
    attention_block: Any,
    forward_attr: str = "forward",
) -> None:
    """Re-route an attention block's forward to ``scaled_dot_product_attention``.

    This is intentionally narrow — it operates on a single attention
    block at a time, leaving the surrounding model intact. The block
    must expose ``q``, ``k``, ``v`` projection submodules and an
    ``out_proj`` (or ``proj``) submodule. Any other shape is treated
    as 'not patchable' and the call returns ``False``.

    Returns:
        ``True`` when the patch was applied, ``False`` otherwise.
    """
    if not all(hasattr(attention_block, name) for name in ("q", "k", "v")):
        return False
    out = getattr(attention_block, "out_proj", None) or getattr(attention_block, "proj", None)
    if out is None:
        return False

    num_heads = getattr(attention_block, "num_heads", None)
    if num_heads is None:
        return False

    def _new_forward(x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        b, n, d = x.shape
        head_dim = d // num_heads
        q = attention_block.q(x).view(b, n, num_heads, head_dim).transpose(1, 2)
        k = attention_block.k(x).view(b, n, num_heads, head_dim).transpose(1, 2)
        v = attention_block.v(x).view(b, n, num_heads, head_dim).transpose(1, 2)
        # PyTorch 2.4+ MPS routes this to flash-attention when shapes allow.
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(b, n, d)
        return out(attn)

    setattr(attention_block, forward_attr, _new_forward)  # type: ignore[arg-type]
    return True


def apply_diffusers_perf_knobs(
    pipeline: Any,
    *,
    fuse_qkv: bool = True,
    enable_attention_slicing: bool = True,
    enable_vae_tiling: bool = True,
) -> dict[str, bool]:
    """Apply the diffusers MPS perf knobs that aren't on by default.

    Each knob is attempted independently; an unsupported one is
    silently skipped. Returns a dict reporting which knobs actually
    applied so callers can log it.
    """
    report: dict[str, bool] = {
        "fuse_qkv": False,
        "enable_attention_slicing": False,
        "enable_vae_tiling": False,
    }
    if fuse_qkv and hasattr(pipeline, "fuse_qkv_projections"):
        try:
            pipeline.fuse_qkv_projections()
            report["fuse_qkv"] = True
        except Exception as exc:  # pragma: no cover
            logger.warning("fuse_qkv_projections failed: %s", exc)
    if enable_attention_slicing and hasattr(pipeline, "enable_attention_slicing"):
        try:
            pipeline.enable_attention_slicing("max")
            report["enable_attention_slicing"] = True
        except Exception as exc:  # pragma: no cover
            logger.warning("enable_attention_slicing failed: %s", exc)
    if enable_vae_tiling:
        vae = getattr(pipeline, "vae", None)
        if vae is not None and hasattr(vae, "enable_tiling"):
            try:
                vae.enable_tiling()
                report["enable_vae_tiling"] = True
            except Exception as exc:  # pragma: no cover
                logger.warning("vae.enable_tiling failed: %s", exc)
    return report


class PerfProfile:
    """Context manager wrapping ``torch.profiler.profile``.

    Usage::

        with PerfProfile(output_dir=Path("outputs/profile")) as prof:
            run_optimisation(...)
        prof.summary()  # prints top ops

    Skips entirely when the profiler is unavailable.
    """

    def __init__(
        self,
        output_dir: Any = None,
        *,
        record_shapes: bool = True,
        with_stack: bool = False,
    ) -> None:
        self._output_dir = output_dir
        self._record_shapes = record_shapes
        self._with_stack = with_stack
        self._prof: Any = None

    def __enter__(self) -> PerfProfile:
        try:
            from torch.profiler import ProfilerActivity, profile

            activities = [ProfilerActivity.CPU]
            if mps_available():
                # ProfilerActivity.MPS only exists in newer PyTorch
                # (2.5+). Fall back gracefully if absent.
                if hasattr(ProfilerActivity, "MPS"):
                    activities.append(ProfilerActivity.MPS)
            self._prof = profile(
                activities=activities,
                record_shapes=self._record_shapes,
                with_stack=self._with_stack,
            )
            self._prof.__enter__()
        except Exception as exc:  # pragma: no cover
            logger.warning("torch.profiler unavailable: %s", exc)
            self._prof = None
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._prof is None:
            return
        self._prof.__exit__(exc_type, exc, tb)
        if self._output_dir is not None:
            try:
                from pathlib import Path

                out = Path(self._output_dir)
                out.mkdir(parents=True, exist_ok=True)
                self._prof.export_chrome_trace(str(out / "trace.json"))
            except Exception as exc:  # pragma: no cover
                logger.warning("profile export failed: %s", exc)

    def summary(
        self,
        sort_by: str = "self_cpu_time_total",
        row_limit: int = 20,
    ) -> str:
        """Return a per-op summary table as a string.

        Useful for printing or attaching to a report. Returns the
        empty string when the profiler didn't run.
        """
        if self._prof is None:
            return ""
        return self._prof.key_averages().table(sort_by=sort_by, row_limit=row_limit)


def default_compile_callback(
    module: Any,
    config_mode: str | None,
) -> Callable[[Any], Any]:
    """Return a compile-or-identity function based on a config string.

    ``config_mode`` accepts ``None``, ``"none"``, ``"default"``,
    ``"reduce-overhead"``, or ``"max-autotune"``. Anything else
    is treated as ``"none"``.
    """
    if not config_mode or config_mode == "none":
        return lambda m: m
    valid = {"default", "reduce-overhead", "max-autotune"}
    if config_mode not in valid:
        logger.warning(
            "unknown compile mode %r; valid modes %s — disabling",
            config_mode,
            sorted(valid),
        )
        return lambda m: m
    return lambda m: compile_module(m, mode=config_mode)
