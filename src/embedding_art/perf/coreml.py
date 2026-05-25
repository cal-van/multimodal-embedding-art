"""CoreML / ANE infrastructure for inference-only probe encoders.

The cross-encoder probe path (re-encode rendered outputs with SigLIP2 /
CLAP / DINOv3 and compare to a text projection) is non-differentiable.
On Apple Silicon those forwards can run on the Apple Neural Engine
(ANE) via CoreML for a 4-6x speedup, freeing GPU time for the actual
optimisation loop.

This module is **infrastructure only** in the current cut: the public
surface is stable, but the conversion path requires running on the
target M1/M2 Max machine (coremltools 8.x doesn't cross-compile from
Linux). Callers can:

1. Stage a probe encoder via :func:`compile_probe_to_coreml`. The
   call returns a :class:`CompileResult` reporting whether the
   conversion succeeded; on Linux / non-mac platforms it returns a
   "skipped" result with a clear reason.
2. Load a previously-compiled probe via :func:`load_compiled_probe`.

The compile step is idempotent: if a ``.mlpackage`` already exists at
the target path, ``compile_probe_to_coreml`` returns it without
re-running conversion.

When ANE-resident probes aren't available, the evaluation path falls
back to running the standard PyTorch probe encoder. The fallback is
silent and zero-cost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompileResult:
    """Outcome of a probe -> CoreML conversion attempt.

    Attributes:
        ok: True iff the conversion succeeded or a cached
            ``.mlpackage`` was reused.
        path: Output ``.mlpackage`` path (may not exist if ``ok`` is
            False).
        reason: Human-readable reason when ``ok`` is False.
    """

    ok: bool
    path: Path
    reason: str = ""


def coreml_available() -> bool:
    """True iff ``coremltools`` is importable.

    On Linux this is typically False (coremltools is macOS-first and
    older releases are Linux-broken). On macOS with coremltools 8+
    this is True.
    """
    try:
        import coremltools  # noqa: F401

        return True
    except ImportError:
        return False


def compile_probe_to_coreml(
    encoder_or_module: Any,
    output_path: Path,
    *,
    example_input: Any = None,
    compute_units: str = "ALL",
    force: bool = False,
) -> CompileResult:
    """Convert a probe encoder forward to an ``.mlpackage`` for ANE.

    Args:
        encoder_or_module: A ``torch.nn.Module`` or duck-typed encoder
            exposing a callable ``forward`` / ``__call__`` that takes
            the same input shape as ``example_input``.
        output_path: Destination ``.mlpackage`` directory.
        example_input: A ``torch.Tensor`` matching the encoder's
            expected input shape. Required when ``output_path``
            doesn't already exist (used to trace the model).
        compute_units: CoreML compute unit hint. ``"ALL"`` lets CoreML
            choose between CPU / GPU / ANE; ``"CPU_AND_NE"`` forces
            ANE-or-CPU.
        force: When True, re-run conversion even if ``output_path``
            already exists.

    Returns:
        :class:`CompileResult` reporting success / failure and the
        output path.
    """
    output_path = Path(output_path)
    if output_path.exists() and not force:
        return CompileResult(ok=True, path=output_path, reason="cached")

    if not coreml_available():
        return CompileResult(
            ok=False,
            path=output_path,
            reason="coremltools not installed (Linux-incompatible). "
            "Run `pip install coremltools` on macOS to enable.",
        )

    if example_input is None:
        return CompileResult(
            ok=False,
            path=output_path,
            reason="example_input is required for tracing; pass a "
            "tensor matching the encoder's expected input shape.",
        )

    try:
        import coremltools as ct
        import torch
    except ImportError as exc:  # pragma: no cover - guarded above
        return CompileResult(ok=False, path=output_path, reason=str(exc))

    try:
        module = (
            encoder_or_module
            if isinstance(encoder_or_module, torch.nn.Module)
            else _wrap_callable(encoder_or_module)
        )
        module = module.eval()
        traced = torch.jit.trace(module, example_input)
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(shape=tuple(example_input.shape))],
            compute_units=getattr(ct.ComputeUnit, compute_units, ct.ComputeUnit.ALL),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mlmodel.save(str(output_path))
        return CompileResult(ok=True, path=output_path)
    except Exception as exc:
        logger.warning("CoreML conversion failed: %s", exc)
        return CompileResult(ok=False, path=output_path, reason=str(exc))


def load_compiled_probe(path: Path) -> Any | None:
    """Load a previously-compiled ``.mlpackage`` for inference.

    Returns the loaded CoreML model, or ``None`` when coremltools
    isn't available or the path doesn't exist. Callers should fall
    back to the standard PyTorch probe in the None case.
    """
    if not Path(path).exists():
        return None
    if not coreml_available():
        return None
    try:
        import coremltools as ct

        return ct.models.MLModel(str(path))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("CoreML load failed: %s", exc)
        return None


def _wrap_callable(fn: Any) -> Any:
    """Wrap a callable in an ``nn.Module`` so torch.jit.trace can
    consume it.

    Most probe encoders are already nn.Modules, but a few are duck-
    typed wrappers with a ``__call__`` that delegates to an inner
    module. This wrapper makes both shapes acceptable to ``ct.convert``.
    """
    import torch

    class _Wrapper(torch.nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.inner(x)

    return _Wrapper(fn)
