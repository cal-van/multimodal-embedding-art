"""
``embed-art text-anchor-sweep`` — M4 empirical sweep CLI (issue iik).

Runs a synthetic gradient-descent sweep over the ``(similarity_weight,
text_anchor_weight)`` grid using a decoder-free harness. Useful both as
a smoke test for the loss math and as a way to characterise per-modality
recommended defaults without paying the full optimisation cost.

The default invocation is encoder-free: target and anchor embeddings
are sampled from a Gaussian with a controllable angular separation, so
the harness runs in seconds on CPU. With ``--encoder`` the harness
loads the canonical multimodal encoder and uses ``encode_text`` to
produce both target and anchor embeddings from real text labels.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import torch
import torch.nn.functional as F  # noqa: N812

from embedding_art.cli.utils import console, handle_exception
from embedding_art.experiments.text_anchor_sweep import (
    run_text_anchor_sweep,
    write_report,
)

logger = logging.getLogger(__name__)


def _parse_floats(spec: str) -> list[float]:
    """Parse a comma-separated list of floats."""
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def _synthetic_pair(
    *, dim: int, separation_deg: float, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate (target, anchor) at the requested angular separation."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    target = F.normalize(torch.randn(dim, generator=gen), dim=-1)
    # Build a unit vector orthogonal to target.
    raw = torch.randn(dim, generator=gen)
    raw = raw - (raw @ target) * target
    orth = F.normalize(raw, dim=-1)
    theta = torch.deg2rad(torch.tensor(separation_deg))
    anchor = (torch.cos(theta) * target + torch.sin(theta) * orth).float()
    return target, F.normalize(anchor, dim=-1)


@click.command("text-anchor-sweep")
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Directory for the sweep report (JSON + markdown).",
)
@click.option(
    "--modality",
    type=click.Choice(["image", "audio", "video", "text"]),
    default="image",
    show_default=True,
    help="Modality label written into the result.",
)
@click.option(
    "--similarity-weights",
    "sim_weights_spec",
    type=str,
    default="1.0",
    show_default=True,
    help="Comma-separated similarity-weight grid.",
)
@click.option(
    "--anchor-weights",
    "anchor_weights_spec",
    type=str,
    default="0.0,0.05,0.1,0.25,0.5,1.0,2.0",
    show_default=True,
    help="Comma-separated text_anchor_weight grid.",
)
@click.option(
    "--steps",
    type=int,
    default=400,
    show_default=True,
    help="Adam steps per grid point.",
)
@click.option(
    "--learning-rate",
    type=float,
    default=0.05,
    show_default=True,
    help="Adam learning rate for the free-embedding optimisation.",
)
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
    help="Initialisation seed.",
)
@click.option(
    "--dim",
    type=int,
    default=768,
    show_default=True,
    help="Embedding dimension (used only when --encoder is not set).",
)
@click.option(
    "--angular-separation",
    type=float,
    default=45.0,
    show_default=True,
    help=(
        "Angular separation between synthetic target and anchor "
        "(degrees). Larger → less alignment by default. Used only when "
        "--encoder is not set."
    ),
)
@click.option(
    "--encoder",
    type=str,
    default=None,
    help=(
        "Optional canonical multimodal encoder. When supplied with "
        "--target-text and --anchor-text, both embeddings come from "
        "encode_text() and the sweep characterises the real loss "
        "surface on this encoder."
    ),
)
@click.option(
    "--device",
    type=str,
    default="cpu",
    show_default=True,
    help="Torch device for the sweep.",
)
@click.option(
    "--target-text",
    type=str,
    default=None,
    help="Target text (used only when --encoder is set).",
)
@click.option(
    "--anchor-text",
    type=str,
    default=None,
    help="Anchor text (used only when --encoder is set).",
)
@click.pass_context
def text_anchor_sweep(
    ctx: click.Context,
    output_dir: str,
    modality: str,
    sim_weights_spec: str,
    anchor_weights_spec: str,
    steps: int,
    learning_rate: float,
    seed: int,
    dim: int,
    angular_separation: float,
    encoder: str | None,
    device: str,
    target_text: str | None,
    anchor_text: str | None,
) -> None:
    """Run the M4 text-anchor empirical sweep."""
    debug_mode = ctx.obj.get("debug", False) if ctx.obj else False
    try:
        _impl(
            output_dir=Path(output_dir),
            modality=modality,
            similarity_weights=_parse_floats(sim_weights_spec),
            anchor_weights=_parse_floats(anchor_weights_spec),
            steps=steps,
            learning_rate=learning_rate,
            seed=seed,
            dim=dim,
            angular_separation=angular_separation,
            encoder_name=encoder,
            device=device,
            target_text=target_text,
            anchor_text=anchor_text,
        )
    except Exception as e:
        handle_exception(e, debug_mode)
        sys.exit(1)


def _impl(
    *,
    output_dir: Path,
    modality: str,
    similarity_weights: list[float],
    anchor_weights: list[float],
    steps: int,
    learning_rate: float,
    seed: int,
    dim: int,
    angular_separation: float,
    encoder_name: str | None,
    device: str,
    target_text: str | None,
    anchor_text: str | None,
) -> None:
    """Build the (target, anchor) pair and run the sweep."""
    if encoder_name is not None:
        if not (target_text and anchor_text):
            raise click.UsageError("--encoder requires --target-text and --anchor-text")
        from embedding_art.encoders.defaults import create_default_registry

        console.print(f"[bold]Loading canonical encoder ({encoder_name})...[/bold]")
        registry = create_default_registry()
        enc = registry.load(encoder_name, device=device)
        target = enc.encode_text(target_text)
        anchor = enc.encode_text(anchor_text)
    else:
        console.print(
            f"[bold]Synthetic sweep: dim={dim}, separation={angular_separation:.1f}°[/bold]"
        )
        target, anchor = _synthetic_pair(dim=dim, separation_deg=angular_separation, seed=seed)

    console.print(
        f"[bold]Running sweep — {len(similarity_weights)}×{len(anchor_weights)} grid, "
        f"{steps} steps each...[/bold]"
    )
    result = run_text_anchor_sweep(
        target_embedding=target,
        anchor_embedding=anchor,
        modality=modality,
        similarity_weights=similarity_weights,
        anchor_weights=anchor_weights,
        steps=steps,
        learning_rate=learning_rate,
        seed=seed,
        device=device,
    )

    json_path, md_path = write_report(result, output_dir)
    console.print(f"[green]Wrote {json_path}[/green]")
    console.print(f"[green]Wrote {md_path}[/green]")

    if result.recommended:
        r = result.recommended
        console.print(
            f"\n[bold]Recommended default for {modality}:[/bold] "
            f"text_anchor_weight = {r.anchor_weight:.3f} "
            f"(target_sim={r.target_similarity:.4f}, anchor_sim={r.anchor_similarity:.4f})"
        )
    # Also dump a one-line summary as JSON to stdout for scripted callers.
    summary = {
        "modality": result.modality,
        "recommended_anchor_weight": (
            result.recommended.anchor_weight if result.recommended else None
        ),
        "recommended_similarity_weight": (
            result.recommended.similarity_weight if result.recommended else None
        ),
        "n_points": len(result.points),
        "n_pareto": len(result.pareto_front),
    }
    console.print(f"[dim]{json.dumps(summary)}[/dim]")
