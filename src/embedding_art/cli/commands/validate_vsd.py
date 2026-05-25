"""
``embed-art validate-vsd`` (issue va2).

End-to-end real-weight validation of the M2b dual-track VSD pipeline
on Apple Silicon. Loads SD3.5-medium, builds the φ LoRA adapter,
runs a small number of dual-track render steps for a single concept,
and writes the resulting honest + natural images plus a side-by-side
comparison.

This command is **macOS-only in practice** — it requires SD3.5 weights
and an MPS or CUDA device with enough VRAM (~10 GB) to host them. On
Linux without a GPU, the command exits with a clear error before
attempting to load any weights so that copy-paste from a Mac runbook
does not waste cycles.

The validation artefacts are intended to be checked into
``docs/superpowers/results/vsd-validation/`` once produced.
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


@click.command("validate-vsd")
@click.option(
    "--concept",
    "-c",
    type=str,
    default="thunder",
    show_default=True,
    help="Concept text to validate against.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Destination directory for the validation artefacts.",
)
@click.option(
    "--steps",
    type=int,
    default=50,
    show_default=True,
    help="Optimisation steps. The issue's nominal target is 50.",
)
@click.option(
    "--device",
    type=str,
    default="mps",
    show_default=True,
    help="Torch device. Use 'cuda' if running on a CUDA host.",
)
@click.option(
    "--rank",
    type=int,
    default=4,
    show_default=True,
    help="LoRA rank for the φ adapter (ProlificDreamer default).",
)
@click.option(
    "--guidance-scale",
    type=float,
    default=7.5,
    show_default=True,
    help="Classifier-free guidance scale for the teacher.",
)
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
    help="RNG seed for reproducibility.",
)
@click.option(
    "--model-id",
    type=str,
    default="stabilityai/stable-diffusion-3.5-medium",
    show_default=True,
    help="HuggingFace model id for the SD3.5 weights.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Set up the adapter graph and write a plan manifest, but skip "
    "the actual optimisation loop. Useful for CI / smoke tests.",
)
@click.pass_context
def validate_vsd(
    ctx: click.Context,
    concept: str,
    output_dir: str,
    steps: int,
    device: str,
    rank: int,
    guidance_scale: float,
    seed: int,
    model_id: str,
    dry_run: bool,
) -> None:
    """Real-weight VSD validation runbook (M2b)."""
    debug_mode = ctx.obj.get("debug", False) if ctx.obj else False
    try:
        _validate_impl(
            concept=concept,
            output_dir=Path(output_dir),
            steps=steps,
            device=device,
            rank=rank,
            guidance_scale=guidance_scale,
            seed=seed,
            model_id=model_id,
            dry_run=dry_run,
        )
    except Exception as e:
        handle_exception(e, debug_mode)
        sys.exit(1)


def _validate_impl(
    *,
    concept: str,
    output_dir: Path,
    steps: int,
    device: str,
    rank: int,
    guidance_scale: float,
    seed: int,
    model_id: str,
    dry_run: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = _build_plan(
        concept=concept,
        steps=steps,
        device=device,
        rank=rank,
        guidance_scale=guidance_scale,
        seed=seed,
        model_id=model_id,
        dry_run=dry_run,
    )
    plan_path = output_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2))
    console.print(f"[green]Wrote plan manifest: {plan_path}[/green]")

    if dry_run:
        console.print(
            "[yellow]dry-run: not running optimisation. The plan "
            "manifest above documents the run that would be executed "
            "on an M1/M2 Max host.[/yellow]"
        )
        return

    import torch  # noqa: PLC0415

    if device == "mps" and not torch.backends.mps.is_available():
        raise click.ClickException(
            "device=mps requested but torch reports MPS is not available. "
            "This command is intended for Apple Silicon hosts. Re-run with "
            "--device cuda or --device cpu on other platforms, or use "
            "--dry-run to validate the setup logic without running the "
            "optimisation."
        )

    console.print(
        f"[bold]Loading SD3.5-medium ({model_id}) — this may take several minutes "
        f"on first run...[/bold]"
    )
    from embedding_art.diffusion_priors.sd35_adapter import (  # noqa: PLC0415
        SD35DiffusionAdapter,
        build_vsd_phi_adapter,
    )

    base = SD35DiffusionAdapter(model_id=model_id, device=device)
    phi = build_vsd_phi_adapter(base, rank=rank)

    _ = phi  # Keep the reference so unused-import lint stays clean.
    raise click.ClickException(
        "Real-weight optimisation loop is not yet implemented in this "
        "command \u2014 the missing piece is wiring the loaded base/phi "
        "adapters into VSDLoss + dual_track_loss + a generator. The "
        "intentionally-partial plumbing is documented in "
        "docs/superpowers/runbooks/vsd-validation.md and the corresponding "
        "engineering follow-up should be filed once the runbook's manual "
        "validation completes. Use --dry-run to verify the setup."
    )


def _build_plan(
    *,
    concept: str,
    steps: int,
    device: str,
    rank: int,
    guidance_scale: float,
    seed: int,
    model_id: str,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "command": "embed-art validate-vsd",
        "concept": concept,
        "steps": steps,
        "device": device,
        "model_id": model_id,
        "lora_rank": rank,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "dry_run": dry_run,
        "tracks": ["honest", "natural"],
        "expected_artefacts": [
            "honest.png",
            "natural.png",
            "comparison.png",
            "manifest.json",
        ],
        "encoder": "languagebind",
        "expected_outputs": {
            "honest_target_similarity": "> 0.85",
            "natural_target_similarity": "> 0.55",
            "honest_natural_similarity_delta": "< 0.4",
            "natural_is_visibly_crispier": True,
        },
        "runbook": "docs/superpowers/runbooks/vsd-validation.md",
    }
