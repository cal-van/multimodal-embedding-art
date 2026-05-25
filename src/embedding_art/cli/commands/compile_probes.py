"""
``embed-art compile-probes`` and ``embed-art bench-probes`` (issue 75o).

* ``compile-probes`` converts the registered probe encoders (SigLIP 2
  image, CLAP audio) to CoreML ``.mlpackage`` artefacts. Idempotent:
  existing artefacts are reused unless ``--force`` is passed.

* ``bench-probes`` runs both the original PyTorch path and the
  CoreML-compiled path on the same dummy input and reports median
  forward-pass latency. On Linux (or when coremltools is missing) only
  the PyTorch path runs and the report says so.

Both commands are safe to run on Linux: conversion falls back to a
clear-error CompileResult without raising. The CoreML-vs-PyTorch
speedup claim itself can only be validated on M1/M2 Mac hardware.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import click

from embedding_art.cli.utils import console, handle_exception

logger = logging.getLogger(__name__)


def _parse_probes(spec: str) -> list[str]:
    return [p.strip() for p in spec.split(",") if p.strip()]


@click.command("compile-probes")
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True),
    default=".cache/coreml-probes",
    show_default=True,
    help="Destination directory for the compiled .mlpackage artefacts.",
)
@click.option(
    "--probes",
    "probes_spec",
    type=str,
    default="siglip2-image",
    show_default=True,
    help="Comma-separated probe names. Use 'all' to compile every " "registered probe.",
)
@click.option(
    "--compute-units",
    type=click.Choice(["ALL", "CPU_AND_NE", "CPU_ONLY", "CPU_AND_GPU"]),
    default="ALL",
    show_default=True,
    help="CoreML compute units hint. CPU_AND_NE forces ANE-or-CPU only.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-run conversion even when a cached .mlpackage exists.",
)
@click.pass_context
def compile_probes(
    ctx: click.Context,
    output_dir: str,
    probes_spec: str,
    compute_units: str,
    force: bool,
) -> None:
    """Compile registered probe encoders to CoreML."""
    debug_mode = ctx.obj.get("debug", False) if ctx.obj else False
    try:
        _compile_impl(
            output_dir=Path(output_dir),
            probes_spec=probes_spec,
            compute_units=compute_units,
            force=force,
        )
    except Exception as e:
        handle_exception(e, debug_mode)
        sys.exit(1)


def _compile_impl(
    *,
    output_dir: Path,
    probes_spec: str,
    compute_units: str,
    force: bool,
) -> None:
    from embedding_art.perf.coreml import compile_probe_to_coreml, coreml_available
    from embedding_art.perf.probe_registry import available_probes, get_probe

    names = available_probes() if probes_spec.strip() == "all" else _parse_probes(probes_spec)
    if not names:
        raise click.UsageError("at least one probe must be requested")

    output_dir.mkdir(parents=True, exist_ok=True)

    if not coreml_available():
        console.print(
            "[yellow]coremltools is not available on this platform. "
            "Compilation will return skipped results. To actually produce "
            ".mlpackage files run this command on macOS with `pip install "
            "coremltools`.[/yellow]"
        )

    results: list[dict[str, Any]] = []
    for name in names:
        spec = get_probe(name)
        output_path = output_dir / f"{spec.name}.mlpackage"
        console.print(f"[bold]Compiling probe '{name}' → {output_path}[/bold]")
        try:
            module, example = spec.loader()
        except Exception as exc:
            console.print(f"  [red]Failed to load probe '{name}': {exc}[/red]")
            results.append({"name": name, "ok": False, "reason": f"loader: {exc}"})
            continue

        result = compile_probe_to_coreml(
            module,
            output_path,
            example_input=example,
            compute_units=compute_units,
            force=force,
        )
        if result.ok:
            console.print(f"  [green]ok[/green]: {result.path} ({result.reason or 'compiled'})")
        else:
            console.print(f"  [yellow]skipped[/yellow]: {result.reason}")
        results.append(
            {
                "name": name,
                "ok": result.ok,
                "path": str(result.path),
                "reason": result.reason,
                "modality": spec.modality,
                "example_input_shape": list(spec.example_input_shape),
            }
        )

    report_path = output_dir / "compile_report.json"
    report_path.write_text(json.dumps(results, indent=2))
    console.print(f"[green]Wrote {report_path}[/green]")


@click.command("bench-probes")
@click.option(
    "--probes-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default=".cache/coreml-probes",
    show_default=True,
    help="Directory containing the compiled .mlpackage artefacts.",
)
@click.option(
    "--probes",
    "probes_spec",
    type=str,
    default="siglip2-image",
    show_default=True,
    help="Comma-separated probe names. Use 'all' to benchmark every registered probe.",
)
@click.option(
    "--n-iters",
    type=int,
    default=20,
    show_default=True,
    help="Forward passes to time per backend.",
)
@click.option(
    "--n-warmup",
    type=int,
    default=3,
    show_default=True,
    help="Forward passes to discard before timing.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, file_okay=True),
    default=None,
    help="Optional path to write the benchmark report as JSON.",
)
@click.option(
    "--device",
    type=str,
    default="cpu",
    show_default=True,
    help="Torch device for the PyTorch baseline.",
)
@click.pass_context
def bench_probes(
    ctx: click.Context,
    probes_dir: str,
    probes_spec: str,
    n_iters: int,
    n_warmup: int,
    output: str | None,
    device: str,
) -> None:
    """Benchmark PyTorch vs CoreML probe forwards."""
    debug_mode = ctx.obj.get("debug", False) if ctx.obj else False
    try:
        _bench_impl(
            probes_dir=Path(probes_dir),
            probes_spec=probes_spec,
            n_iters=n_iters,
            n_warmup=n_warmup,
            output=Path(output) if output else None,
            device=device,
        )
    except Exception as e:
        handle_exception(e, debug_mode)
        sys.exit(1)


def _bench_impl(
    *,
    probes_dir: Path,
    probes_spec: str,
    n_iters: int,
    n_warmup: int,
    output: Path | None,
    device: str,
) -> None:
    import torch

    from embedding_art.perf.coreml import coreml_available, load_compiled_probe
    from embedding_art.perf.probe_bench import speedup, time_callable
    from embedding_art.perf.probe_registry import available_probes, get_probe

    names = available_probes() if probes_spec.strip() == "all" else _parse_probes(probes_spec)
    if not names:
        raise click.UsageError("at least one probe must be requested")

    report: list[dict[str, Any]] = []
    for name in names:
        console.print(f"[bold]Benchmark: {name}[/bold]")
        spec = get_probe(name)
        try:
            module, example = spec.loader()
        except Exception as exc:
            console.print(f"  [red]Failed to load probe '{name}': {exc}[/red]")
            report.append({"name": name, "error": f"loader: {exc}"})
            continue

        torch_device = torch.device(device)
        module = module.to(torch_device).eval()
        example_t = example.to(torch_device)

        with torch.inference_mode():
            torch_stats = time_callable(
                lambda: module(example_t),
                name=f"{name}:torch",
                n_warmup=n_warmup,
                n_iters=n_iters,
            )
        console.print(
            f"  pytorch: median {torch_stats.median_ms:.2f} ms "
            f"(p10 {torch_stats.p10_ms:.2f} / p90 {torch_stats.p90_ms:.2f})"
        )

        coreml_path = probes_dir / f"{spec.name}.mlpackage"
        coreml_stats = None
        speedup_x = None
        if coreml_path.exists() and coreml_available():
            ml_model = load_compiled_probe(coreml_path)
            if ml_model is not None:
                ml_input = {ml_model.input_description._fd_spec[0].name: example.numpy()}  # type: ignore[attr-defined]
                try:
                    coreml_stats = time_callable(
                        lambda: ml_model.predict(ml_input),
                        name=f"{name}:coreml",
                        n_warmup=n_warmup,
                        n_iters=n_iters,
                    )
                    speedup_x = speedup(torch_stats, coreml_stats)
                    console.print(
                        f"  coreml:  median {coreml_stats.median_ms:.2f} ms "
                        f"({speedup_x:.2f}x vs pytorch)"
                    )
                except Exception as exc:  # pragma: no cover - mac-only path
                    console.print(f"  [yellow]coreml backend failed: {exc}[/yellow]")
        else:
            console.print(
                "  [yellow]coreml: no .mlpackage at "
                f"{coreml_path} (run `embed-art compile-probes` first, "
                "and ensure you're on macOS with coremltools installed)[/yellow]"
            )

        entry: dict[str, Any] = {"name": name, "pytorch": torch_stats.to_dict()}
        if coreml_stats is not None:
            entry["coreml"] = coreml_stats.to_dict()
        if speedup_x is not None:
            entry["coreml_speedup_vs_pytorch"] = round(speedup_x, 3)
        report.append(entry)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2))
        console.print(f"[green]Wrote benchmark report: {output}[/green]")
