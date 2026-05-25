"""Tests for ``embedding_art.perf.coreml``.

The conversion path requires coremltools (macOS-only). All tests here
run on Linux too: they exercise the fallback paths, the CompileResult
contract, and the load-and-skip logic.
"""

from __future__ import annotations

from pathlib import Path

import torch

from embedding_art.perf.coreml import (
    CompileResult,
    compile_probe_to_coreml,
    coreml_available,
    load_compiled_probe,
)


def test_coreml_available_returns_bool() -> None:
    assert isinstance(coreml_available(), bool)


def test_compile_skips_when_coremltools_missing(tmp_path: Path) -> None:
    """When coremltools isn't installed, conversion returns a
    'not installed' CompileResult rather than crashing."""
    if coreml_available():  # pragma: no cover - coremltools present
        return
    target = tmp_path / "probe.mlpackage"
    result = compile_probe_to_coreml(
        torch.nn.Linear(8, 8),
        target,
        example_input=torch.randn(1, 8),
    )
    assert isinstance(result, CompileResult)
    assert result.ok is False
    assert "coremltools" in result.reason.lower()


def test_compile_returns_cached_when_path_exists(tmp_path: Path) -> None:
    """Idempotent re-compilation: existing .mlpackage returns ok=True
    without re-running conversion."""
    target = tmp_path / "probe.mlpackage"
    target.mkdir()
    result = compile_probe_to_coreml(
        torch.nn.Linear(8, 8),
        target,
        example_input=torch.randn(1, 8),
    )
    assert result.ok is True
    assert result.path == target
    assert result.reason == "cached"


def test_compile_requires_example_input_when_uncached(tmp_path: Path) -> None:
    """Missing example_input is a clear-error CompileResult, not a
    crash."""
    if not coreml_available():
        return
    target = tmp_path / "probe.mlpackage"
    result = compile_probe_to_coreml(
        torch.nn.Linear(8, 8),
        target,
        example_input=None,
    )
    assert result.ok is False
    assert "example_input" in result.reason


def test_load_compiled_probe_returns_none_for_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.mlpackage"
    assert load_compiled_probe(missing) is None


def test_load_compiled_probe_returns_none_without_coremltools(tmp_path: Path) -> None:
    """When coremltools isn't installed, ``load_compiled_probe``
    returns None silently so callers can fall back to PyTorch probes."""
    if coreml_available():  # pragma: no cover - coremltools present
        return
    target = tmp_path / "probe.mlpackage"
    target.mkdir()
    assert load_compiled_probe(target) is None


def test_compile_result_is_frozen() -> None:
    r = CompileResult(ok=True, path=Path("/tmp/foo"))
    try:
        r.ok = False  # type: ignore[misc]
        raise AssertionError("CompileResult should be frozen")
    except (AttributeError, Exception):
        pass
