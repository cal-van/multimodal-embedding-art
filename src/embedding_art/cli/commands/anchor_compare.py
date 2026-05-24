"""
``embed-art anchor-compare`` — the M9 anchor-comparison experiment CLI.

Encodes the same concept through every supplied modality reference,
reports pairwise cosine similarities, top-K text-anchor readouts per
modality, and (when an SAE is supplied) feature-overlap statistics.
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


@click.command("anchor-compare")
@click.option(
    "--label",
    "concept_label",
    type=str,
    required=True,
    help="Concept label written into the result (e.g. 'thunder').",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    required=True,
    help="Output directory for anchor_comparison.json + per-modality summaries.",
)
@click.option(
    "--text",
    type=str,
    default=None,
    help="Optional text reference for the concept.",
)
@click.option(
    "--image",
    "image_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Optional image reference path.",
)
@click.option(
    "--audio",
    "audio_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Optional audio reference path.",
)
@click.option(
    "--video",
    "video_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Optional video reference path.",
)
@click.option(
    "--encoder",
    type=str,
    default="languagebind",
    show_default=True,
    help="Canonical multimodal encoder.",
)
@click.option(
    "--device",
    type=str,
    default="mps",
    show_default=True,
    help="Torch device.",
)
@click.option(
    "--sae-path",
    type=click.Path(exists=True, dir_okay=True, file_okay=True),
    default=None,
    help="Optional trained SAE checkpoint for feature-overlap analysis.",
)
@click.option(
    "--top-k-text",
    type=int,
    default=20,
    show_default=True,
    help="Top-K text-anchor words per modality.",
)
@click.option(
    "--top-k-features",
    type=int,
    default=32,
    show_default=True,
    help="Top-K SAE features per modality for the overlap computation.",
)
@click.pass_context
def anchor_compare(
    ctx: click.Context,
    concept_label: str,
    output_dir: str,
    text: str | None,
    image_path: str | None,
    audio_path: str | None,
    video_path: str | None,
    encoder: str,
    device: str,
    sae_path: str | None,
    top_k_text: int,
    top_k_features: int,
) -> None:
    """Compare a single concept encoded through multiple modality references."""
    debug_mode = ctx.obj.get("debug", False) if ctx.obj else False

    try:
        _anchor_compare_impl(
            concept_label=concept_label,
            output_dir=Path(output_dir),
            text=text,
            image_path=Path(image_path) if image_path else None,
            audio_path=Path(audio_path) if audio_path else None,
            video_path=Path(video_path) if video_path else None,
            encoder_name=encoder,
            device=device,
            sae_path=Path(sae_path) if sae_path else None,
            top_k_text=top_k_text,
            top_k_features=top_k_features,
        )
    except Exception as e:
        handle_exception(e, debug_mode)
        sys.exit(1)


def _anchor_compare_impl(
    *,
    concept_label: str,
    output_dir: Path,
    text: str | None,
    image_path: Path | None,
    audio_path: Path | None,
    video_path: Path | None,
    encoder_name: str,
    device: str,
    sae_path: Path | None,
    top_k_text: int,
    top_k_features: int,
) -> None:
    """Orchestrate the anchor-comparison experiment and write the result."""
    from embedding_art.cli.commands.showcase import _load_sae
    from embedding_art.encoders.defaults import create_default_registry
    from embedding_art.experiments import run_anchor_comparison

    if all(ref is None for ref in (text, image_path, audio_path, video_path)):
        raise click.UsageError(
            "anchor-compare requires at least one of --text, --image, --audio, --video."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Loading canonical encoder ({encoder_name})...[/bold]")
    registry = create_default_registry()
    encoder = registry.load(encoder_name, device=device)

    sae = _load_sae(sae_path) if sae_path else None

    console.print(f"[bold]Comparing concept '{concept_label}' across supplied modalities...[/bold]")

    result = run_anchor_comparison(
        concept_label=concept_label,
        encoder=encoder,
        encoder_name=encoder_name,
        text=text,
        image_path=image_path,
        audio_path=audio_path,
        video_path=video_path,
        sae=sae,
        top_k_text=top_k_text,
        top_k_features=top_k_features,
    )

    output_path = output_dir / "anchor_comparison.json"
    output_path.write_text(json.dumps(result.to_dict(), indent=2))
    console.print(f"[green]Wrote {output_path}[/green]")

    _write_summary_card(result, output_dir / "anchor_comparison.md")
    console.print(f"[green]Wrote {output_dir / 'anchor_comparison.md'}[/green]")


def _write_summary_card(result: Any, path: Path) -> None:
    """Render a markdown summary of the anchor-comparison result."""
    lines: list[str] = []
    lines.append(f"# anchor-comparison: {result.concept_label}\n")
    lines.append(f"_Canonical encoder: `{result.encoder_name}`_\n")

    lines.append("## Per-modality references\n")
    for mod, record in result.modalities.items():
        lines.append(f"### {mod}\n")
        lines.append(f"- Reference: `{record.get('reference', '<unknown>')}`\n")
        if record.get("text_anchor"):
            top = record["text_anchor"][:10]
            words = ", ".join(f"{w['word']} ({w['similarity']:.3f})" for w in top)
            lines.append(f"- Top-10 text-anchor: {words}\n")
        if "embedding" in record:
            lines.append(f"- Embedding dim: {record['embedding'].shape[-1]}\n")
        lines.append("")

    lines.append("## Pairwise cosine similarity\n")
    mods = sorted(result.modalities.keys())
    if mods:
        header = "| | " + " | ".join(mods) + " |"
        sep = "| --- |" + " --- |" * len(mods)
        lines.append(header)
        lines.append(sep)
        for m1 in mods:
            row = [m1] + [f"{result.cosine_matrix.get(m1, {}).get(m2, 0.0):.3f}" for m2 in mods]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    if result.sae_feature_overlap is not None:
        lines.append("## SAE feature overlap (Jaccard)\n")
        if mods:
            header = "| | " + " | ".join(mods) + " |"
            sep = "| --- |" + " --- |" * len(mods)
            lines.append(header)
            lines.append(sep)
            for m1 in mods:
                row = [m1] + [
                    f"{result.sae_feature_overlap.get(m1, {}).get(m2, {}).get('jaccard', 0.0):.3f}"
                    for m2 in mods
                ]
                lines.append("| " + " | ".join(row) + " |")

    path.write_text("\n".join(lines))
