"""
``embed-art showcase`` — render one concept across all four modalities (M10a).

The minimum viable showcase: encode the concept once using the canonical
LanguageBind multimodal encoder, then drive image / audio / video generators
toward that same target embedding and save all artefacts to a single output
directory with a JSON manifest and a markdown card.

Why this works structurally
---------------------------
LanguageBind produces a 768-d embedding that is *the same point* whether you
encoded text, an image, audio, or video. So when the showcase optimises an
image latent (via the image VAE) and an audio latent (via AudioLDM2's VAE)
and a video latent (via SVD) toward the same LanguageBind target, all three
optimisations are pulling toward semantically equivalent geometry in a single
shared space. This is the structural property the rest of the system needs
for cross-modal showcase to even make sense — it's the M1 LanguageBind
foundation paying off.

Per-modality generator choices
------------------------------
* **Image** — SD3.5-medium VAE (M2a). 16-channel flow-matching latent;
  materially sharper than SDXL's 4-channel DDPM latent. SDXL is still
  selectable via ``--image-backbone sdxl`` for ablation.
* **Audio** — Stable Audio Open 1.0 (M6 default). 64-channel DAC latent,
  44.1 kHz, up to 47-sec clips. AudioLDM2 selectable via
  ``--audio-backbone audioldm2`` for ablation.
* **Video** — LTX-Video 0.9.5 (M6 default). 128-channel CausalVideoAutoencoder
  latent, 768×512 @ 24 fps. SVD selectable via ``--video-backbone svd``
  for ablation.
* **Text** — A literal text card describing the target concept and the per-
  modality cosine similarities. Required for the side-by-side showcase even
  though there is no 'text rendering' problem to solve.

Dual-track rendering
--------------------
The ``--tracks`` flag selects one or both of:

* **honest** — strong embedding/feature alignment, minimal regularisation.
  Renders 'what the model thinks' the concept looks like. The interpretability
  artefact.
* **natural** — reduced embedding alignment, heavy regularisation. Renders
  a more conventionally-natural-looking sample of the concept. The aesthetic
  artefact.

When both are requested (``--tracks honest,natural``), each is rendered
into its own subdirectory and the top-level ``manifest.json`` references
both. The contrast between the two is the showcase artefact.

Note: the 'natural' track currently uses heavy regularisation as a
proxy for the M2b VSD prior. When M2b real-weight integration lands,
the natural track will switch to VSD with the existing ``--tracks``
flag unchanged.

Output layout
-------------
::

    <output_dir>/
      image.png
      audio.wav
      video.mp4         (or .gif if mp4 export unavailable)
      text-card.md
      manifest.json     (concept description, per-modality similarities,
                         encoder name, optimisation steps, seed, paths)

Interpretation + evaluation bundles
-----------------------------------
Stubs for the M7 (interpretation) and M8 (evaluation) bundles are NOT
included in this commit; ``manifest.json`` records the per-modality final
cosine similarity which is the minimum diagnostic, and the bundles can be
added in follow-up commits once the trained MSAEs (M3) and cross-encoder
probes are available.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click
import torch
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from embedding_art.cli.utils import console, handle_exception


def _emit(callback: Callable[[dict[str, Any]], None] | None, event: dict[str, Any]) -> None:
    """Invoke ``callback`` with ``event`` if a callback is provided.

    Errors in the callback are swallowed so that broadcasting issues
    (e.g. a closed websocket) never break the rendering loop.
    """
    if callback is None:
        return
    try:
        callback(event)
    except Exception:  # pragma: no cover - defensive
        logging.getLogger(__name__).exception("progress_callback raised; ignoring")


logger = logging.getLogger(__name__)


DEFAULT_STEPS = 1000
DEFAULT_ENCODER = "languagebind"
VALID_TRACKS = ("honest", "natural")


@click.command()
@click.option(
    "-t",
    "--target-text",
    type=str,
    required=True,
    help="Text description of the target concept (e.g. -t 'goldfish').",
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(),
    required=True,
    help="Output directory. Will contain image.png, audio.wav, video.mp4, "
    "text-card.md, manifest.json.",
)
@click.option(
    "--encoder",
    type=str,
    default=DEFAULT_ENCODER,
    show_default=True,
    help="Canonical multimodal encoder to use as the shared target space.",
)
@click.option(
    "--modalities",
    type=str,
    default="image,audio,video,text",
    show_default=True,
    help="Comma-separated subset of modalities to render. Use to skip slow "
    "renderings during iteration.",
)
@click.option(
    "--steps",
    type=int,
    default=DEFAULT_STEPS,
    show_default=True,
    help="Optimisation steps per modality.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility (applied to all modalities).",
)
@click.option(
    "--device",
    type=str,
    default="mps",
    show_default=True,
    help="Torch device.",
)
@click.option(
    "--image-backbone",
    type=click.Choice(["sd35", "sdxl"]),
    default="sd35",
    show_default=True,
    help="Image generator backbone. SD3.5 is the M2a canonical default.",
)
@click.option(
    "--audio-backbone",
    type=click.Choice(["stable-audio-open", "audioldm2"]),
    default="stable-audio-open",
    show_default=True,
    help="Audio generator backbone. Stable Audio Open is the M6 canonical default "
    "(44.1 kHz, 64-ch DAC). audioldm2 selectable for ablation.",
)
@click.option(
    "--video-backbone",
    type=click.Choice(["ltx-video", "svd"]),
    default="ltx-video",
    show_default=True,
    help="Video generator backbone. LTX-Video 0.9.5 is the M6 canonical default "
    "(768×512 @ 24 fps). svd selectable for ablation.",
)
@click.option(
    "--tracks",
    type=str,
    default="honest",
    show_default=True,
    help="Comma-separated subset of {honest,natural}. honest = high-alignment, "
    "low-regularisation (the interpretability artefact). natural = reduced "
    "alignment + heavy regularisation (the aesthetic artefact). When both are "
    "selected each is rendered into a subdirectory of the output.",
)
@click.option(
    "--realism",
    type=float,
    default=None,
    help="Continuous honest<->natural dial in [0.0, 1.0]. 0.0 = honest "
    "(max alignment, minimal regularisation — 'what the model sees', the "
    "AI-interpretable artefact); 1.0 = natural (heavy regularisation — the "
    "human-legible, photographic artefact). When set, renders a single "
    "bundle at that dial position and overrides --tracks. Omit to use the "
    "discrete named tracks instead.",
)
@click.option(
    "--autocast-dtype",
    type=click.Choice(["fp32", "fp16", "bf16"]),
    default="fp32",
    show_default=True,
    help="Mixed-precision autocast for the forward+loss section of each "
    "optimisation step. 'bf16' is recommended on M1 Max and ~2x faster; "
    "'fp32' is the safe default.",
)
@click.option(
    "--compile-mode",
    type=click.Choice(["none", "default", "reduce-overhead", "max-autotune"]),
    default="none",
    show_default=True,
    help="Apple Silicon: wrap the loss callable in torch.compile. "
    "'reduce-overhead' is the recommended setting on PyTorch 2.5+ MPS for a "
    "1.5-2.5x speedup on the optimisation hot loop. 'none' disables.",
)
@click.option(
    "--interpret/--no-interpret",
    default=True,
    show_default=True,
    help="Compute the interpretation bundle (text-anchor readout, gradient "
    "attribution, SAE decomposition if available) for every modality.",
)
@click.option(
    "--sae-path",
    type=click.Path(exists=True, dir_okay=True, file_okay=True),
    default=None,
    help="Path to trained SAE weights for the interpretation bundle's SAE "
    "decomposition + CorrSteer fields. When omitted those fields are skipped.",
)
@click.option(
    "--evaluate/--no-evaluate",
    default=True,
    show_default=True,
    help="Compute the evaluation card (cross-encoder probes + cross-modal "
    "agreement matrix) for the showcase bundle.",
)
@click.option(
    "--linear-probes-dir",
    "linear_probes_dir",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    default=None,
    help="Directory containing a trained linear-probe manifest "
    "(see `embed-art train-probes`). When provided, each modality's "
    "interpretation bundle includes per-probe activations.",
)
@click.option(
    "--text-anchor-weight",
    type=float,
    default=0.0,
    show_default=True,
    help="Auxiliary loss weight for the M4 text-anchor signal. When > 0 the "
    "composite loss adds a cosine term pulling the optimisation embedding "
    "toward the encoder's text projection of the concept label. See the "
    "empirical sweep at docs/superpowers/results/text-anchor-sweep/ for "
    "recommended values; 0.25 is a conservative starting point for "
    "image/audio/video tracks.",
)
@click.option(
    "--probes",
    type=str,
    default="",
    show_default=True,
    help="Comma-separated cross-encoder probes to run during the M8 "
    "evaluation card (e.g. 'siglip2-so400m,clap-general'). Each probe "
    "re-encodes the rendered outputs with an independent encoder and "
    "reports cosine similarity to that encoder's projection of the "
    "target text — the 'does another encoder agree?' check. Probes are "
    "resolved through the same registry as --encoder; unknown names are "
    "skipped with a warning. Default empty (no probes).",
)
@click.option(
    "--seed-stability",
    type=int,
    default=0,
    show_default=True,
    help="Re-run the canonical rendering this many additional times with "
    "distinct seeds and report the variance in per-modality final cosine "
    "similarity. 0 disables (the default — single-seed run). Values >= 1 "
    "add a `seed_stability` block to the evaluation card with mean/std "
    "per modality. Heavy: each extra run is a full optimisation pass.",
)
@click.pass_context
def showcase(
    ctx: click.Context,
    target_text: str,
    output_dir: str,
    encoder: str,
    modalities: str,
    steps: int,
    seed: int | None,
    device: str,
    image_backbone: str,
    audio_backbone: str,
    video_backbone: str,
    tracks: str,
    realism: float | None,
    autocast_dtype: str,
    compile_mode: str,
    interpret: bool,
    sae_path: str | None,
    evaluate: bool,
    linear_probes_dir: str | None,
    text_anchor_weight: float,
    probes: str,
    seed_stability: int,
) -> None:
    """Render one concept across all four modalities into a single showcase bundle."""
    debug_mode = ctx.obj.get("debug", False) if ctx.obj else False

    track_list = [t.strip() for t in tracks.split(",") if t.strip()]
    for t in track_list:
        if t not in VALID_TRACKS:
            click.echo(
                f"Invalid track '{t}'. Valid tracks: {', '.join(VALID_TRACKS)}",
                err=True,
            )
            sys.exit(2)

    if realism is not None and (not math.isfinite(realism) or not 0.0 <= realism <= 1.0):
        click.echo(f"Invalid --realism {realism}. Must be a finite value in [0.0, 1.0].", err=True)
        sys.exit(2)

    probe_encoders = _load_probe_encoders(
        [p.strip() for p in probes.split(",") if p.strip()],
        device=device,
    )

    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir)
    out_path = out_path.parent / f"{out_path.name}_{timestamp}"

    try:
        _showcase_impl(
            target_text=target_text,
            output_dir=out_path,
            encoder_name=encoder,
            modalities=[m.strip() for m in modalities.split(",") if m.strip()],
            steps=steps,
            seed=seed,
            device=device,
            image_backbone=image_backbone,
            audio_backbone=audio_backbone,
            video_backbone=video_backbone,
            tracks=track_list,
            realism=realism,
            autocast_dtype=autocast_dtype,
            compile_mode=compile_mode,
            probe_encoders=probe_encoders,
            seed_stability=seed_stability,
            interpret=interpret,
            sae_path=Path(sae_path) if sae_path else None,
            evaluate=evaluate,
            linear_probes_dir=Path(linear_probes_dir) if linear_probes_dir else None,
            text_anchor_weight=text_anchor_weight,
        )
    except Exception as e:
        handle_exception(e, debug_mode)
        sys.exit(1)


def _showcase_impl(
    *,
    target_text: str,
    output_dir: Path,
    encoder_name: str,
    modalities: list[str],
    steps: int,
    seed: int | None,
    device: str,
    image_backbone: str,
    audio_backbone: str = "stable-audio-open",
    video_backbone: str = "ltx-video",
    tracks: list[str] | None = None,
    autocast_dtype: str = "fp32",
    compile_mode: str = "none",
    interpret: bool = True,
    sae_path: Path | None = None,
    evaluate: bool = True,
    probe_encoders: dict[str, Any] | None = None,
    seed_stability: int = 0,
    linear_probes_dir: Path | None = None,
    text_anchor_weight: float = 0.0,
    realism: float | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Orchestrate the per-modality renderings and assemble the bundle.

    When ``tracks`` is a single track, the bundle is written flat into
    ``output_dir``. When ``tracks`` requests both honest and natural, each
    track gets its own subdirectory and the top-level ``manifest.json``
    summarises both.

    ``realism`` (``[0.0, 1.0]``), when provided, overrides ``tracks`` and
    renders a single flat bundle at that point on the continuous
    honest<->natural dial (see :func:`interpolate_loss_config`). The track is
    labelled ``realism-<value>`` so the manifest records the dial position.
    """
    from embedding_art.core.concept import Concept
    from embedding_art.core.engine import EmbeddingArtEngine
    from embedding_art.encoders.defaults import create_default_registry

    if realism is not None:
        # The continuous dial is a single custom render; it supersedes the
        # discrete named tracks.
        tracks = [_realism_label(realism)]
    elif tracks is None:
        tracks = ["honest"]

    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Loading canonical encoder ({encoder_name})...[/bold]")
    registry = create_default_registry()
    encoder = registry.load(encoder_name, device=device)

    console.print(f"[bold]Encoding target: '{target_text}'[/bold]")
    target = Concept.from_text(target_text, encoder)
    console.print(f"  embedding dim: {target.embedding.shape[-1]}")

    # The engine + SAE are shared across tracks; only LossConfig differs.
    engine = EmbeddingArtEngine.from_registry(registry, default_encoder=encoder_name, device=device)
    sae = _load_sae(sae_path) if sae_path else None
    sae_feature_labels = _load_sae_labels(sae_path) if sae_path else None
    linear_probes = _load_linear_probes(linear_probes_dir) if linear_probes_dir else None

    multi_track = len(tracks) > 1
    per_track_manifests: dict[str, dict[str, Any]] = {}

    _emit(
        progress_callback,
        {
            "type": "showcase_start",
            "target_text": target_text,
            "encoder": encoder_name,
            "modalities": modalities,
            "tracks": tracks,
            "multi_track": multi_track,
        },
    )

    for track in tracks:
        track_output_dir = (output_dir / track) if multi_track else output_dir
        track_output_dir.mkdir(parents=True, exist_ok=True)
        _emit(progress_callback, {"type": "track_start", "track": track})
        per_track_manifests[track] = _render_track(
            track=track,
            target=target,
            target_text=target_text,
            text_anchor_weight=text_anchor_weight,
            realism=realism,
            encoder=encoder,
            encoder_name=encoder_name,
            engine=engine,
            modalities=modalities,
            steps=steps,
            seed=seed,
            device=device,
            output_dir=track_output_dir,
            image_backbone=image_backbone,
            audio_backbone=audio_backbone,
            video_backbone=video_backbone,
            autocast_dtype=autocast_dtype,
            compile_mode=compile_mode,
            sae=sae,
            sae_feature_labels=sae_feature_labels,
            linear_probes=linear_probes,
            interpret=interpret,
            evaluate=evaluate,
            probe_encoders=probe_encoders,
            progress_callback=progress_callback,
        )
        _emit(
            progress_callback,
            {
                "type": "track_complete",
                "track": track,
                "summary": _summarise_track(per_track_manifests[track]),
            },
        )

    if seed_stability > 0:
        # Re-run only the honest (or single) track N extra times with
        # distinct seeds; aggregate per-modality final similarity.
        stability_track = tracks[0]
        stability_report = _compute_seed_stability(
            n_extra_runs=seed_stability,
            base_seed=seed,
            track=stability_track,
            target=target,
            target_text=target_text,
            text_anchor_weight=text_anchor_weight,
            encoder=encoder,
            encoder_name=encoder_name,
            engine=engine,
            modalities=modalities,
            steps=steps,
            device=device,
            output_dir=output_dir / ".stability",
            image_backbone=image_backbone,
            audio_backbone=audio_backbone,
            video_backbone=video_backbone,
            autocast_dtype=autocast_dtype,
            compile_mode=compile_mode,
            sae=sae,
            sae_feature_labels=sae_feature_labels,
            linear_probes=linear_probes,
        )
        # Attach to the relevant track's manifest evaluation card.
        canonical_manifest = per_track_manifests[stability_track]
        canonical_manifest.setdefault("evaluation", {})["seed_stability"] = stability_report
        canonical_manifest_path = (
            output_dir / stability_track / "manifest.json"
            if multi_track
            else output_dir / "manifest.json"
        )
        canonical_manifest_path.write_text(json.dumps(canonical_manifest, indent=2))
        console.print(
            f"[bold green]Seed-stability report ({seed_stability} extra runs) "
            f"merged into {canonical_manifest_path}[/bold green]"
        )

    if multi_track:
        top_manifest: dict[str, Any] = {
            "concept": {
                "text": target_text,
                "description": target.description,
                "embedding_dim": int(target.embedding.shape[-1]),
            },
            "encoder": encoder_name,
            "device": device,
            "steps": steps,
            "seed": seed,
            "tracks": tracks,
            "image_backbone": image_backbone,
            "audio_backbone": audio_backbone,
            "video_backbone": video_backbone,
            "per_track": {
                t: {
                    "path": t,
                    "manifest": f"{t}/manifest.json",
                    "summary": _summarise_track(per_track_manifests[t]),
                }
                for t in tracks
            },
        }
        top_manifest_path = output_dir / "manifest.json"
        top_manifest_path.write_text(json.dumps(top_manifest, indent=2))
        console.print(f"[bold green]Multi-track manifest written: {top_manifest_path}[/bold green]")
        console.print(f"[bold green]Showcase complete: {output_dir}[/bold green]")


def _render_track(
    *,
    track: str,
    target: Any,
    target_text: str,
    text_anchor_weight: float = 0.0,
    realism: float | None = None,
    encoder: Any,
    encoder_name: str,
    engine: Any,
    modalities: list[str],
    steps: int,
    seed: int | None,
    device: str,
    output_dir: Path,
    image_backbone: str,
    audio_backbone: str,
    video_backbone: str,
    autocast_dtype: str,
    compile_mode: str = "none",
    sae: Any = None,
    sae_feature_labels: dict[int, str] | None = None,
    linear_probes: dict[str, Any] | None = None,
    interpret: bool = True,
    evaluate: bool = True,
    probe_encoders: dict[str, Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Render a single track into ``output_dir``.

    When ``realism`` is provided the loss is taken from the continuous dial
    (:func:`interpolate_loss_config`); otherwise ``track`` selects one of the
    named endpoints. Returns the per-track manifest dict and writes it to
    ``output_dir/manifest.json``.
    """
    from embedding_art.core.config import OptimizationConfig

    if realism is not None:
        loss_config = interpolate_loss_config(realism, text_anchor_weight=text_anchor_weight)
    else:
        loss_config = _build_track_loss_config(track, text_anchor_weight=text_anchor_weight)

    config = OptimizationConfig(
        steps=steps,
        learning_rate=0.1,
        seed=seed,
        loss=loss_config,
        autocast_dtype=autocast_dtype,  # type: ignore[arg-type]
        compile_mode=compile_mode,  # type: ignore[arg-type]
    )

    console.print(f"[bold magenta]Rendering track: {track}[/bold magenta]")

    manifest: dict[str, Any] = {
        "concept": {
            "text": target_text,
            "description": target.description,
            "embedding_dim": int(target.embedding.shape[-1]),
        },
        "encoder": encoder_name,
        "device": device,
        "steps": steps,
        "seed": seed,
        "track": track,
        "realism": realism,
        "image_backbone": image_backbone,
        "audio_backbone": audio_backbone,
        "video_backbone": video_backbone,
        "modalities": {},
    }

    def _emit_modality_complete(modality: str, record: dict[str, Any]) -> None:
        _emit(
            progress_callback,
            {
                "type": "modality_complete",
                "track": track,
                "modality": modality,
                "similarity": record.get("final_similarity"),
                "text_anchor": (record.get("interpretation") or {}).get("text_anchor"),
            },
        )
        # Release the just-completed modality's generator BEFORE emptying the
        # cache — otherwise all three VAEs (+ LanguageBind) stay resident and
        # empty_mps_cache frees nothing (the Python refs are still live). Drop
        # the engine's ref, force a GC pass, then empty the allocator.
        if getattr(config, "empty_mps_cache_between_modalities", True):
            import gc

            from embedding_art.perf import empty_mps_cache

            generators = getattr(engine, "_generators", None)
            if isinstance(generators, dict):
                generators.pop(modality, None)
            gc.collect()
            empty_mps_cache()

    if "image" in modalities:
        _emit(progress_callback, {"type": "modality_start", "track": track, "modality": "image"})
        manifest["modalities"]["image"] = _render_image(
            engine=engine,
            target=target,
            encoder=encoder,
            encoder_name=encoder_name,
            config=config,
            output_dir=output_dir,
            device=device,
            backbone=image_backbone,
            interpret=interpret,
            sae=sae,
            sae_feature_labels=sae_feature_labels,
            linear_probes=linear_probes,
            progress_callback=progress_callback,
            track=track,
        )
        _emit_modality_complete("image", manifest["modalities"]["image"])

    if "audio" in modalities:
        _emit(progress_callback, {"type": "modality_start", "track": track, "modality": "audio"})
        manifest["modalities"]["audio"] = _render_audio(
            engine=engine,
            target=target,
            encoder=encoder,
            encoder_name=encoder_name,
            config=config,
            output_dir=output_dir,
            device=device,
            backbone=audio_backbone,
            interpret=interpret,
            sae=sae,
            sae_feature_labels=sae_feature_labels,
            linear_probes=linear_probes,
            progress_callback=progress_callback,
            track=track,
        )
        _emit_modality_complete("audio", manifest["modalities"]["audio"])

    if "video" in modalities:
        _emit(progress_callback, {"type": "modality_start", "track": track, "modality": "video"})
        manifest["modalities"]["video"] = _render_video(
            engine=engine,
            target=target,
            encoder=encoder,
            encoder_name=encoder_name,
            config=config,
            output_dir=output_dir,
            device=device,
            backbone=video_backbone,
            interpret=interpret,
            sae=sae,
            sae_feature_labels=sae_feature_labels,
            linear_probes=linear_probes,
            progress_callback=progress_callback,
            track=track,
        )
        _emit_modality_complete("video", manifest["modalities"]["video"])

    if "text" in modalities:
        _emit(progress_callback, {"type": "modality_start", "track": track, "modality": "text"})
        manifest["modalities"]["text"] = _render_text_card(
            target=target,
            target_text=target_text,
            output_dir=output_dir,
            manifest=manifest,
        )
        _emit_modality_complete("text", manifest["modalities"]["text"])

    if evaluate:
        manifest["evaluation"] = _compute_evaluation_card(
            target=target,
            encoder_name=encoder_name,
            modalities_data=manifest["modalities"],
            output_dir=output_dir,
            device=device,
            target_text=target_text,
            probe_encoders=probe_encoders,
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    console.print(f"[bold green]Manifest written: {manifest_path}[/bold green]")
    return manifest


# Endpoints of the realism<->honesty dial. ``honest`` (realism=0.0) maximises
# embedding/feature alignment with minimal regularisation — the generator's raw
# attempt to *occupy the target coordinate*, the AI-interpretable artefact.
# ``natural`` (realism=1.0) trades alignment for heavy regularisation, producing
# the conventionally-photographic, human-legible artefact. These two numbers
# reproduce the historical honest/natural presets exactly.
_HONEST_SIMILARITY = 1.0
_HONEST_FEATURE_MATCHING = 0.5
_NATURAL_SIMILARITY = 0.4
_NATURAL_FEATURE_MATCHING = 0.15
# Per-regulariser weight endpoints (honest -> natural). A weight of 0 at the
# honest end means that regulariser is simply absent there, so realism=0.0
# reproduces ``CompositeRegularizer.minimal()`` (LatentNorm only) and
# realism=1.0 reproduces ``CompositeRegularizer.heavy()`` (all three).
_TV_WEIGHTS = (0.0, 0.1)
_SPECTRAL_WEIGHTS = (0.0, 0.01)
_LATENT_NORM_WEIGHTS = (0.01, 0.5)


def interpolate_loss_config(realism: float, *, text_anchor_weight: float = 0.0) -> Any:
    """Build a :class:`LossConfig` at a point on the honest<->natural dial.

    ``realism`` is a continuous slider in ``[0.0, 1.0]``:

    * **0.0 — honest / AI-interpretable.** Maximum embedding + feature
      alignment, minimal regularisation. The generator's raw attempt to
      occupy the target coordinate; renders 'what the model sees'.
    * **1.0 — natural / human-legible.** Reduced alignment, heavy
      regularisation. A conventionally-photographic sample of the concept.

    Every loss weight interpolates linearly between the two endpoints, so the
    dial is continuous and monotonic. The endpoints are byte-identical to the
    legacy ``honest`` / ``natural`` track presets.

    ``text_anchor_weight`` is forwarded unchanged (the M4 auxiliary loss is
    encoder-agnostic; its default of 0.0 is a no-op).
    """
    from embedding_art.core.config import LossConfig
    from embedding_art.regularizers.base import (
        CompositeRegularizer,
        LatentNorm,
        SpectralRegularizer,
        TotalVariation,
    )

    if not 0.0 <= realism <= 1.0:
        raise ValueError(f"realism must be in [0.0, 1.0], got {realism}")

    def _lerp(endpoints: tuple[float, float]) -> float:
        lo, hi = endpoints
        return lo + (hi - lo) * realism

    tv_w = _lerp(_TV_WEIGHTS)
    spec_w = _lerp(_SPECTRAL_WEIGHTS)
    norm_w = _lerp(_LATENT_NORM_WEIGHTS)

    # Gate zero-weight terms out so the honest endpoint stays cheap (no FFT /
    # TV passes) and the regulariser shapes match minimal()/heavy() exactly.
    regularizers: list[Any] = []
    if tv_w > 0.0:
        regularizers.append(TotalVariation(weight=tv_w))
    if spec_w > 0.0:
        regularizers.append(SpectralRegularizer(weight=spec_w))
    regularizers.append(LatentNorm(weight=norm_w))

    return LossConfig(
        similarity_weight=_lerp((_HONEST_SIMILARITY, _NATURAL_SIMILARITY)),
        feature_matching_weight=_lerp((_HONEST_FEATURE_MATCHING, _NATURAL_FEATURE_MATCHING)),
        text_anchor_weight=text_anchor_weight,
        regularization=CompositeRegularizer(regularizers=regularizers),
    )


def _realism_label(realism: float) -> str:
    """Stable, filesystem-safe label for a point on the realism dial."""
    return f"realism-{realism:.2f}"


def _build_track_loss_config(track: str, *, text_anchor_weight: float = 0.0) -> Any:
    """Return the LossConfig for a named track — the dial's endpoints.

    The named tracks are the two ends of :func:`interpolate_loss_config`:
    ``honest`` == realism 0.0, ``natural`` == realism 1.0. Kept as a thin
    alias so the discrete dual-track ('render both ends and contrast them')
    path and its tests stay stable while the continuous dial drives single
    custom renders.
    """
    if track == "honest":
        return interpolate_loss_config(0.0, text_anchor_weight=text_anchor_weight)
    if track == "natural":
        return interpolate_loss_config(1.0, text_anchor_weight=text_anchor_weight)
    raise ValueError(f"Unknown track '{track}'. Valid tracks: {VALID_TRACKS}")


def _summarise_track(track_manifest: dict[str, Any]) -> dict[str, Any]:
    """Reduce a per-track manifest to a small summary block for the top-level."""
    summary: dict[str, Any] = {}
    for mod, data in track_manifest.get("modalities", {}).items():
        if mod == "text":
            continue
        summary[mod] = {
            "path": data.get("path"),
            "final_similarity": data.get("final_similarity"),
            "backbone": data.get("backbone"),
        }
    if "evaluation" in track_manifest:
        summary["_evaluation"] = track_manifest["evaluation"]
    return summary


# ---------------------------------------------------------------------------
# Per-modality renderers
# ---------------------------------------------------------------------------


def _render_image(
    *,
    engine: Any,
    target: Any,
    encoder: Any,
    encoder_name: str,
    config: Any,
    output_dir: Path,
    device: str,
    backbone: str,
    interpret: bool = True,
    sae: Any = None,
    sae_feature_labels: dict[int, str] | None = None,
    linear_probes: dict[str, Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    track: str = "honest",
) -> dict[str, Any]:
    console.print("[bold]Rendering image...[/bold]")
    if backbone == "sd35":
        from embedding_art.generators.sd35 import SD35ImageGenerator

        generator = SD35ImageGenerator(device=device)
    else:
        from embedding_art.generators import SDXLImageGenerator

        generator = SDXLImageGenerator(device=device)

    engine.register_generator("image", generator)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress_bar:
        task_id = progress_bar.add_task("Optimising image...", total=config.steps)

        def step_callback(step: int, breakdown: Any, output: torch.Tensor) -> None:
            sim = float(breakdown.similarity) if hasattr(breakdown, "similarity") else 0.0
            loss = float(breakdown.total) if hasattr(breakdown, "total") else 0.0
            progress_bar.update(
                task_id,
                completed=step + 1,
                description=f"Optimising image (loss={loss:.4f}, sim={sim:.4f})",
            )
            if progress_callback:
                _emit(
                    progress_callback,
                    {
                        "type": "step",
                        "track": track,
                        "modality": "image",
                        "step": step + 1,
                        "total_steps": config.steps,
                        "loss": loss,
                        "similarity": sim,
                    },
                )

        result = engine.render(
            target,
            encoder_name=encoder_name,
            output_modality="image",
            config=config,
            callback=step_callback,
        )

    image_path = output_dir / "image.png"
    _save_image_output(result, generator, image_path)

    modality_record: dict[str, Any] = {
        "path": str(image_path.relative_to(output_dir)),
        "final_similarity": float(result.final_similarity),
        "backbone": backbone,
    }

    if interpret:
        modality_record["interpretation"] = _run_modality_interpretation(
            target=target,
            output=result.output,
            encoder=encoder,
            sae=sae,
            sae_feature_labels=sae_feature_labels,
            linear_probes=linear_probes,
            final_similarity=result.final_similarity,
            output_dir=output_dir,
            modality="image",
        )

    return modality_record


def _render_audio(
    *,
    engine: Any,
    target: Any,
    encoder: Any,
    encoder_name: str,
    config: Any,
    output_dir: Path,
    device: str,
    backbone: str = "stable-audio-open",
    interpret: bool = True,
    sae: Any = None,
    sae_feature_labels: dict[int, str] | None = None,
    linear_probes: dict[str, Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    track: str = "honest",
) -> dict[str, Any]:
    console.print(f"[bold]Rendering audio ({backbone})...[/bold]")
    if backbone == "stable-audio-open":
        from embedding_art.generators.stable_audio_open import StableAudioOpenGenerator

        generator = StableAudioOpenGenerator(device=device)
    elif backbone == "audioldm2":
        from embedding_art.generators import AudioLDMGenerator

        generator = AudioLDMGenerator(device=device)
    else:
        raise ValueError(
            f"Unknown audio backbone '{backbone}'. Valid: stable-audio-open, audioldm2."
        )

    engine.register_generator("audio", generator)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress_bar:
        task_id = progress_bar.add_task(f"Optimising audio ({backbone})...", total=config.steps)

        def step_callback(step: int, breakdown: Any, output: torch.Tensor) -> None:
            sim = float(breakdown.similarity) if hasattr(breakdown, "similarity") else 0.0
            loss = float(breakdown.total) if hasattr(breakdown, "total") else 0.0
            progress_bar.update(
                task_id,
                completed=step + 1,
                description=f"Optimising audio (loss={loss:.4f}, sim={sim:.4f})",
            )
            if progress_callback:
                _emit(
                    progress_callback,
                    {
                        "type": "step",
                        "track": track,
                        "modality": "audio",
                        "step": step + 1,
                        "total_steps": config.steps,
                        "loss": loss,
                        "similarity": sim,
                    },
                )

        result = engine.render(
            target,
            encoder_name=encoder_name,
            output_modality="audio",
            config=config,
            callback=step_callback,
        )

    audio_path = output_dir / "audio.wav"
    _save_audio_output(result, generator, audio_path)

    modality_record: dict[str, Any] = {
        "path": str(audio_path.relative_to(output_dir)),
        "final_similarity": float(result.final_similarity),
        "backbone": backbone,
    }

    if interpret:
        modality_record["interpretation"] = _run_modality_interpretation(
            target=target,
            output=result.output,
            encoder=encoder,
            sae=sae,
            sae_feature_labels=sae_feature_labels,
            linear_probes=linear_probes,
            final_similarity=result.final_similarity,
            output_dir=output_dir,
            modality="audio",
        )

    return modality_record


def _render_video(
    *,
    engine: Any,
    target: Any,
    encoder: Any,
    encoder_name: str,
    config: Any,
    output_dir: Path,
    device: str,
    backbone: str = "ltx-video",
    interpret: bool = True,
    sae: Any = None,
    sae_feature_labels: dict[int, str] | None = None,
    linear_probes: dict[str, Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    track: str = "honest",
) -> dict[str, Any]:
    console.print(f"[bold]Rendering video ({backbone})...[/bold]")
    if backbone == "ltx-video":
        from embedding_art.generators.ltx_video import LTXVideoGenerator

        generator = LTXVideoGenerator(device=device)
    elif backbone == "svd":
        from embedding_art.generators import SVDVideoGenerator

        generator = SVDVideoGenerator(device=device)
    else:
        raise ValueError(f"Unknown video backbone '{backbone}'. Valid: ltx-video, svd.")

    engine.register_generator("video", generator)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress_bar:
        task_id = progress_bar.add_task(f"Optimising video ({backbone})...", total=config.steps)

        def step_callback(step: int, breakdown: Any, output: torch.Tensor) -> None:
            sim = float(breakdown.similarity) if hasattr(breakdown, "similarity") else 0.0
            loss = float(breakdown.total) if hasattr(breakdown, "total") else 0.0
            progress_bar.update(
                task_id,
                completed=step + 1,
                description=f"Optimising video (loss={loss:.4f}, sim={sim:.4f})",
            )
            if progress_callback:
                _emit(
                    progress_callback,
                    {
                        "type": "step",
                        "track": track,
                        "modality": "video",
                        "step": step + 1,
                        "total_steps": config.steps,
                        "loss": loss,
                        "similarity": sim,
                    },
                )

        result = engine.render(
            target,
            encoder_name=encoder_name,
            output_modality="video",
            config=config,
            callback=step_callback,
        )

    video_path = output_dir / "video.mp4"
    fallback_path = output_dir / "video.gif"
    written = _save_video_output(result, generator, video_path, fallback_path)

    modality_record: dict[str, Any] = {
        "path": str(written.relative_to(output_dir)),
        "final_similarity": float(result.final_similarity),
        "backbone": backbone,
    }

    if interpret:
        modality_record["interpretation"] = _run_modality_interpretation(
            target=target,
            output=result.output,
            encoder=encoder,
            sae=sae,
            sae_feature_labels=sae_feature_labels,
            linear_probes=linear_probes,
            final_similarity=result.final_similarity,
            output_dir=output_dir,
            modality="video",
        )

    return modality_record


def _render_text_card(
    *,
    target: Any,
    target_text: str,
    output_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Produce a markdown text card summarising the concept."""
    card_path = output_dir / "text-card.md"

    lines: list[str] = []
    lines.append(f"# {target_text}\n")
    lines.append(f"**Concept description:** `{target.description}`\n")
    lines.append(f"**Encoder:** `{manifest['encoder']}`\n")
    lines.append(f"**Embedding dim:** {manifest['concept']['embedding_dim']}\n")
    lines.append(f"**Optimisation steps:** {manifest['steps']}\n")
    if manifest.get("seed") is not None:
        lines.append(f"**Seed:** {manifest['seed']}\n")
    lines.append("\n## Per-modality renderings\n")
    for modality_name, modality_meta in manifest["modalities"].items():
        if modality_name == "text":
            continue
        sim = modality_meta.get("final_similarity")
        path = modality_meta.get("path", "(unrendered)")
        sim_str = f"{sim:.4f}" if sim is not None else "—"
        backbone = modality_meta.get("backbone", "—")
        lines.append(
            f"- **{modality_name}** ({backbone}): `{path}` — "
            f"final cosine similarity to target = {sim_str}\n"
        )
    card_path.write_text("".join(lines))
    return {"path": str(card_path.relative_to(output_dir))}


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _run_modality_interpretation(
    *,
    target: Any,
    output: Any,
    encoder: Any,
    sae: Any,
    sae_feature_labels: dict[int, str] | None,
    final_similarity: float,
    output_dir: Path,
    modality: str,
    embedding_history: Any = None,
    similarity_history: Any = None,
    linear_probes: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Compute the interpretation bundle for one modality and save its
    attribution map to disk. Returns the JSON-friendly dict to inline in the
    manifest, or ``None`` if the bundle could not be computed.

    When ``embedding_history`` and ``similarity_history`` are provided, the
    bundle additionally includes a CorrSteer ranking — Pearson correlation
    between each SAE feature's per-step activation and the cosine-similarity
    trajectory.
    """
    from embedding_art.interpretation import run_interpretation
    from embedding_art.interpretation.bundle import compute_attribution_to_disk

    try:
        bundle = run_interpretation(
            target=target,
            output=output,
            encoder=encoder,
            sae=sae,
            sae_feature_labels=sae_feature_labels,
            final_similarity=float(final_similarity),
            embedding_history=embedding_history,
            similarity_history=similarity_history,
            linear_probes=linear_probes,
        )

        attribution_path = output_dir / f"{modality}_attribution.pt"
        try:
            shape = compute_attribution_to_disk(
                output=output,
                target_embedding=target.embedding,
                encoder=encoder,
                path=attribution_path,
            )
            bundle.attribution_path = str(attribution_path.relative_to(output_dir))
            bundle.attribution_shape = shape
        except Exception as exc:
            logger.warning(
                "Skipping %s attribution map (encoder may not be differentiable): %s",
                modality,
                exc,
            )

        return bundle.to_dict()
    except Exception as exc:
        logger.warning("Interpretation bundle for %s failed: %s", modality, exc, exc_info=True)
        return None


def _load_linear_probes(probes_dir: Path) -> dict[str, Any] | None:
    """Load a directory of trained linear probes as a name → probe dict.

    The returned dict is the shape :class:`InterpretationBundle` expects
    for its ``linear_probes`` argument: callable probes that take an
    embedding tensor and return either a single float or a label →
    probability mapping.
    """
    try:
        from embedding_art.evaluation.linear_probes import (
            evaluate_probes,
            load_probe_manifest,
        )

        probes = load_probe_manifest(probes_dir)
    except Exception as exc:
        logger.warning("Could not load linear probes from %s: %s", probes_dir, exc)
        return None

    # ``InterpretationBundle._run_linear_probes`` invokes each value
    # like ``probe(current_emb)``. Wrap so the call returns the
    # appropriately-shaped scalar/dict.
    wrapped: dict[str, Any] = {}
    for name, probe in probes.items():

        def _wrap(emb: Any, _probe: Any = probe, _name: str = name) -> Any:
            return evaluate_probes(emb, {_name: _probe})[_name]

        wrapped[name] = _wrap
    return wrapped


def _load_sae(sae_path: Path) -> Any:
    """Load a trained SAE checkpoint. Returns ``None`` if loading fails."""
    try:
        import torch

        from embedding_art.sae.training import GroupSparseSAE

        weights_file = sae_path / "sae_weights.pt" if sae_path.is_dir() else sae_path
        state = torch.load(weights_file, weights_only=True)
        w_enc = state["W_enc"]
        n_features, embed_dim = w_enc.shape
        # Default TopK to a reasonable value; the saved checkpoint should also
        # ship a config but this is the lazy fallback.
        sae = GroupSparseSAE(embed_dim=embed_dim, n_features=n_features, k=32)
        sae.load_state_dict(state)
        sae.eval()
        return sae
    except Exception as exc:
        logger.warning("Could not load SAE from %s: %s", sae_path, exc, exc_info=True)
        return None


def _load_probe_encoders(probe_names: list[str], *, device: str) -> dict[str, Any] | None:
    """Resolve a list of probe-encoder names through the default registry.

    Each name is looked up in ``create_default_registry()`` and loaded
    onto ``device``. Names that can't be resolved (missing extras,
    download failures) are logged and skipped, never raised. Returns
    ``None`` when the input list is empty so callers can short-circuit
    the M8 probe path.

    This is the production-grade entry point for ``embed-art showcase
    --probes siglip2-so400m,clap-general``. The bundle's evaluation
    card surfaces every probe's cosine similarity to its own projection
    of the target text.
    """
    if not probe_names:
        return None

    from embedding_art.encoders.defaults import create_default_registry

    registry = create_default_registry()
    probes: dict[str, Any] = {}
    for name in probe_names:
        try:
            probes[name] = registry.load(name, device=device)
            console.print(f"[cyan]Loaded probe '{name}' for cross-encoder evaluation.[/cyan]")
        except Exception as exc:
            logger.warning("Could not load probe '%s' (skipping): %s", name, exc, exc_info=True)
            console.print(
                f"[yellow]Probe '{name}' unavailable — skipping. "
                "Install its extras or check the encoder registry.[/yellow]"
            )
    return probes or None


def _compute_seed_stability(
    *,
    n_extra_runs: int,
    base_seed: int | None,
    track: str,
    target: Any,
    target_text: str,
    text_anchor_weight: float,
    encoder: Any,
    encoder_name: str,
    engine: Any,
    modalities: list[str],
    steps: int,
    device: str,
    output_dir: Path,
    image_backbone: str,
    audio_backbone: str,
    video_backbone: str,
    autocast_dtype: str,
    compile_mode: str,
    sae: Any = None,
    sae_feature_labels: dict[int, str] | None = None,
    linear_probes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-render the target N times with distinct seeds and report variance.

    The canonical first run lives outside this helper (in the main
    showcase loop). This helper does the *extra* runs and aggregates
    their final per-modality cosine similarities into
    ``{modality: {mean, std, n, values}}``. Each extra run writes its
    artefacts into ``output_dir / seed_<n>`` so they can be inspected
    individually, but the manifest only carries summary statistics.

    Returns:
        A dict shaped:

        .. code-block:: python

            {
                "n_extra_runs": int,
                "base_seed": int | None,
                "per_modality": {
                    "image": {  # SeedStabilityReport.to_dict()
                        "n_seeds": int,
                        "mean_similarity": float,
                        "std_similarity": float,
                        "min_similarity": float,
                        "max_similarity": float,
                        "feature_overlap_jaccard": float | None,
                    },
                    ...
                },
            }

    The per-modality dict shape matches
    :class:`~embedding_art.evaluation.stability.SeedStabilityReport.to_dict`
    so downstream consumers (frontend, manifest readers) can treat
    each modality block uniformly.
    """
    from embedding_art.evaluation.stability import compute_seed_stability

    output_dir.mkdir(parents=True, exist_ok=True)

    base = 0 if base_seed is None else int(base_seed)
    per_modality_sims: dict[str, list[float]] = {}
    per_modality_feature_sets: dict[str, list[set[int]]] = {}
    for i in range(n_extra_runs):
        run_seed = base + 1 + i
        run_dir = output_dir / f"seed_{run_seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            manifest = _render_track(
                track=track,
                target=target,
                target_text=target_text,
                text_anchor_weight=text_anchor_weight,
                encoder=encoder,
                encoder_name=encoder_name,
                engine=engine,
                modalities=modalities,
                steps=steps,
                seed=run_seed,
                device=device,
                output_dir=run_dir,
                image_backbone=image_backbone,
                audio_backbone=audio_backbone,
                video_backbone=video_backbone,
                autocast_dtype=autocast_dtype,
                compile_mode=compile_mode,
                sae=sae,
                sae_feature_labels=sae_feature_labels,
                linear_probes=linear_probes,
                interpret=sae is not None,  # SAE features needed for feature-set overlap
                evaluate=False,
                probe_encoders=None,
            )
        except Exception as exc:
            logger.warning("Seed-stability run %d failed: %s", run_seed, exc, exc_info=True)
            continue

        for mod, data in manifest.get("modalities", {}).items():
            if mod == "text":
                continue
            sim = data.get("final_similarity")
            if sim is None or not isinstance(sim, (int, float)):
                continue
            per_modality_sims.setdefault(mod, []).append(float(sim))

            # Optional: feature-set overlap if interpretation bundle has SAE features.
            interp = data.get("interpretation") or {}
            sae_block = interp.get("sae_features") or []
            if sae_block:
                feat_ids = {int(f["index"]) for f in sae_block if "index" in f}
                per_modality_feature_sets.setdefault(mod, []).append(feat_ids)

    per_modality: dict[str, dict[str, Any]] = {}
    for mod, sims in per_modality_sims.items():
        feature_sets = per_modality_feature_sets.get(mod)
        # Only pass feature_sets when its length matches similarities.
        if feature_sets is not None and len(feature_sets) != len(sims):
            feature_sets = None
        report = compute_seed_stability(
            similarities=sims,
            feature_sets=feature_sets,
        )
        per_modality[mod] = report.to_dict()

    return {
        "n_extra_runs": n_extra_runs,
        "base_seed": base_seed,
        "per_modality": per_modality,
    }


def _load_sae_labels(sae_path: Path) -> dict[int, str] | None:
    """Load feature labels saved alongside the SAE weights, if present."""
    try:
        labels_file = (
            sae_path / "feature_labels.json"
            if sae_path.is_dir()
            else sae_path.parent / "feature_labels.json"
        )
        if not labels_file.exists():
            return None
        return {int(k): v for k, v in json.loads(labels_file.read_text()).items()}
    except Exception as exc:
        logger.warning("Could not load SAE labels: %s", exc, exc_info=True)
        return None


def _compute_evaluation_card(
    *,
    target: Any,
    encoder_name: str,
    modalities_data: dict[str, dict[str, Any]],
    output_dir: Path,
    device: str,
    target_text: str | None = None,
    probe_encoders: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute the evaluation card (M8): cross-modal agreement matrix +
    per-modality final cosine similarity + optional cross-encoder probes.

    When ``probe_encoders`` is supplied (a mapping ``{name: encoder}``),
    each rendered output is re-encoded with each probe and the cosine
    similarity to the probe's projection of ``target_text`` is reported.
    Probes that lack the required ``encode_<modality>`` method are
    silently skipped (e.g. DINOv3 has no audio).
    """
    similarities = {
        mod: float(data.get("final_similarity", float("nan")))
        for mod, data in modalities_data.items()
        if mod != "text"
    }

    # Cross-modal agreement: read each modality's interpretation bundle's
    # text-anchor readout (if available) and report the Jaccard overlap of
    # the top-K text words. This is a coarse cross-modal alignment proxy
    # that does NOT require running the cross-encoder probes; the real
    # cross-encoder evaluation is left for an M8 follow-up.
    text_anchors: dict[str, set[str]] = {}
    for mod, data in modalities_data.items():
        if mod == "text":
            continue
        interp = data.get("interpretation") or {}
        anchors = interp.get("text_anchor") or []
        if anchors:
            text_anchors[mod] = {a["word"] for a in anchors[:10]}

    agreement: dict[str, dict[str, float]] = {}
    mods = sorted(text_anchors.keys())
    for m1 in mods:
        agreement[m1] = {}
        for m2 in mods:
            if m1 == m2:
                agreement[m1][m2] = 1.0
                continue
            inter = text_anchors[m1] & text_anchors[m2]
            union = text_anchors[m1] | text_anchors[m2]
            agreement[m1][m2] = (len(inter) / len(union)) if union else 0.0

    card: dict[str, Any] = {
        "per_modality_similarity": similarities,
        "cross_modal_text_anchor_agreement_jaccard": agreement,
        "encoder": encoder_name,
    }

    if probe_encoders:
        from embedding_art.evaluation.probes import (
            compute_cross_encoder_probes,
            reports_to_table,
        )

        modality_outputs: dict[str, Any] = {}
        for mod, data in modalities_data.items():
            if mod == "text":
                continue
            saved = data.get("output_path") or data.get("path") or data.get("file")
            if saved:
                modality_outputs[mod] = Path(saved)
        reports = compute_cross_encoder_probes(
            target_text=target_text,
            modality_outputs=modality_outputs,
            probes=probe_encoders,
        )
        card["cross_encoder_probes"] = {
            "table": reports_to_table(reports),
            "reports": [r.to_dict() for r in reports],
        }

    return card


def _save_image_output(result: Any, generator: Any, path: Path) -> None:
    """Save a RenderResult's image output as PNG."""
    import torch
    from PIL import Image

    tensor = result.output
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = (tensor.detach().cpu().clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).numpy()
    Image.fromarray(arr).save(path)
    console.print(f"  saved {path}")


def _save_audio_output(result: Any, generator: Any, path: Path) -> None:
    """Save a RenderResult's audio output as WAV."""
    import torch

    tensor = result.output
    if tensor.dim() > 2:
        tensor = tensor.squeeze()
    waveform = tensor.detach().cpu().to(torch.float32).clamp(-1.0, 1.0)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    sample_rate = getattr(generator, "SAMPLE_RATE", 16000)
    try:
        import torchaudio

        torchaudio.save(str(path), waveform, sample_rate)
    except ImportError:
        import wave

        pcm = (waveform[0].numpy() * 32767).astype("int16")
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
    console.print(f"  saved {path}")


def _save_video_output(result: Any, generator: Any, mp4_path: Path, gif_path: Path) -> Path:
    """Save a RenderResult's video output as MP4 (preferred) or GIF (fallback).

    Returns the path actually written.
    """
    import torch

    tensor = result.output  # Expect [B, C, F, H, W] or [F, C, H, W] or [B, F, C, H, W]
    if tensor.dim() == 5 and tensor.shape[0] == 1:
        tensor = tensor[0]  # [C, F, H, W] or [F, C, H, W]
    if tensor.dim() == 4 and tensor.shape[0] == 3:
        # [C, F, H, W] -> [F, C, H, W]
        tensor = tensor.permute(1, 0, 2, 3)

    frames = (tensor.detach().cpu().clamp(0, 1) * 255).to(torch.uint8)
    if frames.dim() != 4:
        raise RuntimeError(
            f"Unexpected video tensor shape after normalisation: {tuple(frames.shape)}"
        )
    # Frames now [F, C, H, W]; permute to [F, H, W, C] for PIL.
    frames_np = frames.permute(0, 2, 3, 1).numpy()

    from PIL import Image as _PILImage

    pil_frames = [_PILImage.fromarray(f) for f in frames_np]

    try:
        import imageio.v3 as iio

        iio.imwrite(mp4_path, frames_np, fps=8)
        console.print(f"  saved {mp4_path}")
        return mp4_path
    except (ImportError, Exception) as exc:  # noqa: BLE001
        logger.info("Falling back to GIF for video (mp4 export failed: %s)", exc)
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=125,  # ms per frame = 8 fps
            loop=0,
        )
        console.print(f"  saved {gif_path}")
        return gif_path
