"""
Tests for the ``embed-art compile-probes`` / ``embed-art bench-probes`` CLIs
(issue 75o).

These tests are designed to pass on Linux: the conversion path returns
a graceful "skipped" CompileResult when coremltools is not available,
and the benchmark command runs the PyTorch baseline regardless. The
actual ANE speedup claim can only be validated on M1/M2 hardware and
is covered by the validation runbook in docs/.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from click.testing import CliRunner

from embedding_art.cli.main import cli
from embedding_art.perf.coreml import compile_probe_to_coreml, coreml_available
from embedding_art.perf.probe_bench import speedup, time_callable
from embedding_art.perf.probe_registry import (
    REGISTRY,
    ProbeSpec,
    available_probes,
    get_probe,
)


class TestProbeRegistry:
    """The registry of CoreML-conversion probes."""

    def test_known_probes(self) -> None:
        names = available_probes()
        assert "siglip2-image" in names
        assert "clap-audio" in names

    def test_get_probe_known(self) -> None:
        spec = get_probe("siglip2-image")
        assert isinstance(spec, ProbeSpec)
        assert spec.modality == "image"
        # SO400M is a 384x384 input.
        assert spec.example_input_shape == (1, 3, 384, 384)

    def test_get_probe_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            get_probe("does-not-exist")


class TestProbeBench:
    """The dependency-light benchmark helper."""

    def test_time_callable_runs_warmup_and_iters(self) -> None:
        calls = {"n": 0}

        def fn() -> None:
            calls["n"] += 1

        stats = time_callable(fn, name="noop", n_warmup=3, n_iters=5)
        assert calls["n"] == 8
        assert stats.n_iters == 5
        assert stats.n_warmup == 3
        assert stats.median_ms >= 0.0

    def test_time_callable_orders_percentiles(self) -> None:
        # Use a varying-cost callable so the percentiles spread.
        i = {"k": 0}

        def fn() -> None:
            i["k"] += 1
            time.sleep(0.001 * (i["k"] % 5))

        stats = time_callable(fn, name="vary", n_warmup=1, n_iters=10)
        assert stats.p10_ms <= stats.median_ms <= stats.p90_ms

    def test_speedup_quotient(self) -> None:
        baseline = time_callable(lambda: time.sleep(0.005), name="b", n_warmup=1, n_iters=3)
        candidate = time_callable(lambda: None, name="c", n_warmup=1, n_iters=3)
        assert speedup(baseline, candidate) > 1.0


class TestCompileProbeFallback:
    """On Linux compile_probe_to_coreml returns a graceful skipped result."""

    def test_skipped_when_coremltools_missing(self, tmp_path: Path) -> None:
        # In our Linux test env coremltools is not installed; ensure the
        # function reports a clear reason and does not raise.
        result = compile_probe_to_coreml(
            torch.nn.Linear(4, 2),
            tmp_path / "no.mlpackage",
            example_input=torch.randn(1, 4),
        )
        if not coreml_available():
            assert result.ok is False
            assert "coremltools" in result.reason.lower()
        else:  # pragma: no cover - macOS-only
            assert result.ok is True

    def test_returns_cached_when_existing(self, tmp_path: Path) -> None:
        path = tmp_path / "existing.mlpackage"
        path.mkdir()
        result = compile_probe_to_coreml(
            torch.nn.Linear(4, 2),
            path,
            example_input=torch.randn(1, 4),
        )
        assert result.ok is True
        assert result.reason == "cached"


class TestCompileProbesCli:
    """The ``embed-art compile-probes`` CLI surface."""

    def test_compile_writes_report_on_linux(self, tmp_path: Path) -> None:
        # Patch the loader so we don't pull SigLIP 2 weights in CI.
        def fake_loader() -> tuple[torch.nn.Module, torch.Tensor]:
            return torch.nn.Linear(4, 2).eval(), torch.randn(1, 4)

        with patch.dict(
            REGISTRY,
            {
                "fake-probe": ProbeSpec(
                    name="fake-probe",
                    modality="image",
                    example_input_shape=(1, 4),
                    loader=fake_loader,
                )
            },
            clear=False,
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "compile-probes",
                    "-o",
                    str(tmp_path),
                    "--probes",
                    "fake-probe",
                ],
            )
            assert result.exit_code == 0, result.output
            report = tmp_path / "compile_report.json"
            assert report.exists()
            entries = json.loads(report.read_text())
            assert len(entries) == 1
            entry = entries[0]
            assert entry["name"] == "fake-probe"
            # On Linux this will not produce a .mlpackage but should
            # still record a clear, structured result.
            if not coreml_available():
                assert entry["ok"] is False
                assert "coremltools" in entry["reason"].lower()

    def test_unknown_probe_is_rejected(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "compile-probes",
                "-o",
                str(tmp_path),
                "--probes",
                "non-existent-probe",
            ],
        )
        assert result.exit_code != 0


class TestBenchProbesCli:
    """The ``embed-art bench-probes`` CLI surface."""

    def test_bench_runs_pytorch_on_linux(self, tmp_path: Path) -> None:
        def fake_loader() -> tuple[torch.nn.Module, torch.Tensor]:
            return torch.nn.Linear(4, 2).eval(), torch.randn(1, 4)

        with patch.dict(
            REGISTRY,
            {
                "fake-probe": ProbeSpec(
                    name="fake-probe",
                    modality="image",
                    example_input_shape=(1, 4),
                    loader=fake_loader,
                )
            },
            clear=False,
        ):
            runner = CliRunner()
            output = tmp_path / "bench.json"
            result = runner.invoke(
                cli,
                [
                    "bench-probes",
                    "--probes-dir",
                    str(tmp_path),
                    "--probes",
                    "fake-probe",
                    "--output",
                    str(output),
                    "--n-iters",
                    "3",
                    "--n-warmup",
                    "1",
                ],
            )
            assert result.exit_code == 0, result.output
            assert output.exists()
            data = json.loads(output.read_text())
            assert data[0]["name"] == "fake-probe"
            assert "pytorch" in data[0]
            # On Linux the coreml path is skipped.
            if not coreml_available():
                assert "coreml" not in data[0]
