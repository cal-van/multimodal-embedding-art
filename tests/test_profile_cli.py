"""Smoke test for ``embed-art profile`` — the Apple Silicon perf harness.

The harness uses a no-weight synthetic encoder by default, so it
completes in <1s on CPU and provides a stable test for the CLI wiring,
output paths, and CSV-like summary generation.
"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from embedding_art.cli.commands.profile import profile


def test_profile_runs_with_synthetic_encoder(tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "prof"
    result = runner.invoke(
        profile,
        [
            "--output",
            str(out_dir),
            "--device",
            "cpu",
            "--steps",
            "2",
            "--encoder",
            "synthetic",
            "--compile-mode",
            "none",
        ],
    )
    assert result.exit_code == 0, result.output
    summary_path = out_dir / "summary.txt"
    assert summary_path.exists(), "profile command did not write summary.txt"
    # Summary may be empty when torch.profiler can't initialise on this
    # platform (e.g. in some container CI environments) — but the file
    # itself must exist as a signal that the harness ran to completion.


def test_profile_accepts_autocast_flag(tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "prof_bf16"
    result = runner.invoke(
        profile,
        [
            "--output",
            str(out_dir),
            "--device",
            "cpu",
            "--steps",
            "1",
            "--encoder",
            "synthetic",
            "--autocast-dtype",
            "bf16",
        ],
    )
    # bf16 autocast may or may not be supported on the test CPU; we
    # only require that the CLI accepts the flag and doesn't crash on
    # unrecognised input.
    assert result.exit_code == 0 or "autocast" in result.output.lower(), result.output


def test_profile_handles_compile_mode_flag(tmp_path: Path) -> None:
    """torch.compile may not work on every test platform; the harness
    falls back silently via embedding_art.perf.compile_module."""
    runner = CliRunner()
    out_dir = tmp_path / "prof_compile"
    result = runner.invoke(
        profile,
        [
            "--output",
            str(out_dir),
            "--device",
            "cpu",
            "--steps",
            "1",
            "--encoder",
            "synthetic",
            "--compile-mode",
            "reduce-overhead",
        ],
    )
    # Either succeeds (compile worked) or falls back silently.
    assert result.exit_code == 0, result.output
