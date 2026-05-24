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

Per-modality generator choices (M10a status)
--------------------------------------------
* **Image** — SD3.5-medium VAE (M2a). 16-channel flow-matching latent;
  materially sharper than SDXL's 4-channel DDPM latent.
* **Audio** — AudioLDM2. Stable Audio Open upgrade is the M6 work item.
* **Video** — SVD (Stable Video Diffusion). LTX-Video upgrade is the M6
  work item.
* **Text** — A literal text card describing the target concept and the per-
  modality cosine similarities. Required for the side-by-side showcase even
  though there is no 'text rendering' problem to solve.

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
import sys
from pathlib import Path
from typing import Any

import click

from embedding_art.cli.utils import console, handle_exception

logger = logging.getLogger(__name__)


DEFAULT_STEPS = 1000
DEFAULT_ENCODER = "languagebind"


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
    interpret: bool,
    sae_path: str | None,
    evaluate: bool,
) -> None:
    """Render one concept across all four modalities into a single showcase bundle."""
    debug_mode = ctx.obj.get("debug", False) if ctx.obj else False

    try:
        _showcase_impl(
            target_text=target_text,
            output_dir=Path(output_dir),
            encoder_name=encoder,
            modalities=[m.strip() for m in modalities.split(",") if m.strip()],
            steps=steps,
            seed=seed,
            device=device,
            image_backbone=image_backbone,
            interpret=interpret,
            sae_path=Path(sae_path) if sae_path else None,
            evaluate=evaluate,
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
    interpret: bool = True,
    sae_path: Path | None = None,
    evaluate: bool = True,
) -> None:
    """Orchestrate the per-modality renderings and assemble the bundle."""
    from embedding_art.core.concept import Concept
    from embedding_art.core.config import LossConfig, OptimizationConfig
    from embedding_art.core.engine import EmbeddingArtEngine
    from embedding_art.encoders.defaults import create_default_registry

    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Loading canonical encoder ({encoder_name})...[/bold]")
    registry = create_default_registry()
    encoder = registry.load(encoder_name, device=device)

    console.print(f"[bold]Encoding target: '{target_text}'[/bold]")
    target = Concept.from_text(target_text, encoder)
    console.print(f"  embedding dim: {target.embedding.shape[-1]}")

    config = OptimizationConfig(
        steps=steps,
        learning_rate=0.1,
        seed=seed,
        loss=LossConfig(similarity_weight=1.0, feature_matching_weight=0.0),
    )

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
        "image_backbone": image_backbone,
        "modalities": {},
    }

    engine = EmbeddingArtEngine.from_registry(registry, default_encoder=encoder_name, device=device)

    sae = _load_sae(sae_path) if sae_path else None
    sae_feature_labels = _load_sae_labels(sae_path) if sae_path else None

    if "image" in modalities:
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
        )

    if "audio" in modalities:
        manifest["modalities"]["audio"] = _render_audio(
            engine=engine,
            target=target,
            encoder=encoder,
            encoder_name=encoder_name,
            config=config,
            output_dir=output_dir,
            device=device,
            interpret=interpret,
            sae=sae,
            sae_feature_labels=sae_feature_labels,
        )

    if "video" in modalities:
        manifest["modalities"]["video"] = _render_video(
            engine=engine,
            target=target,
            encoder=encoder,
            encoder_name=encoder_name,
            config=config,
            output_dir=output_dir,
            device=device,
            interpret=interpret,
            sae=sae,
            sae_feature_labels=sae_feature_labels,
        )

    if "text" in modalities:
        manifest["modalities"]["text"] = _render_text_card(
            target=target,
            target_text=target_text,
            output_dir=output_dir,
            manifest=manifest,
        )

    if evaluate:
        manifest["evaluation"] = _compute_evaluation_card(
            target=target,
            encoder_name=encoder_name,
            modalities_data=manifest["modalities"],
            output_dir=output_dir,
            device=device,
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    console.print(f"[bold green]Manifest written: {manifest_path}[/bold green]")
    console.print(f"[bold green]Showcase complete: {output_dir}[/bold green]")


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
) -> dict[str, Any]:
    console.print("[bold]Rendering image...[/bold]")
    if backbone == "sd35":
        from embedding_art.generators.sd35 import SD35ImageGenerator

        generator = SD35ImageGenerator(device=device)
    else:
        from embedding_art.generators import SDXLImageGenerator

        generator = SDXLImageGenerator(device=device)

    engine.register_generator("image", generator)
    result = engine.render(
        target,
        encoder_name=encoder_name,
        output_modality="image",
        config=config,
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
    interpret: bool = True,
    sae: Any = None,
    sae_feature_labels: dict[int, str] | None = None,
) -> dict[str, Any]:
    console.print("[bold]Rendering audio...[/bold]")
    from embedding_art.generators import AudioLDMGenerator

    generator = AudioLDMGenerator(device=device)
    engine.register_generator("audio", generator)
    result = engine.render(
        target,
        encoder_name=encoder_name,
        output_modality="audio",
        config=config,
    )

    audio_path = output_dir / "audio.wav"
    _save_audio_output(result, generator, audio_path)

    modality_record: dict[str, Any] = {
        "path": str(audio_path.relative_to(output_dir)),
        "final_similarity": float(result.final_similarity),
        "backbone": "audioldm2",
    }

    if interpret:
        modality_record["interpretation"] = _run_modality_interpretation(
            target=target,
            output=result.output,
            encoder=encoder,
            sae=sae,
            sae_feature_labels=sae_feature_labels,
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
    interpret: bool = True,
    sae: Any = None,
    sae_feature_labels: dict[int, str] | None = None,
) -> dict[str, Any]:
    console.print("[bold]Rendering video...[/bold]")
    from embedding_art.generators import SVDVideoGenerator

    generator = SVDVideoGenerator(device=device)
    engine.register_generator("video", generator)
    result = engine.render(
        target,
        encoder_name=encoder_name,
        output_modality="video",
        config=config,
    )

    video_path = output_dir / "video.mp4"
    fallback_path = output_dir / "video.gif"
    written = _save_video_output(result, generator, video_path, fallback_path)

    modality_record: dict[str, Any] = {
        "path": str(written.relative_to(output_dir)),
        "final_similarity": float(result.final_similarity),
        "backbone": "svd",
    }

    if interpret:
        modality_record["interpretation"] = _run_modality_interpretation(
            target=target,
            output=result.output,
            encoder=encoder,
            sae=sae,
            sae_feature_labels=sae_feature_labels,
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
) -> dict[str, Any] | None:
    """Compute the interpretation bundle for one modality and save its
    attribution map to disk. Returns the JSON-friendly dict to inline in the
    manifest, or ``None`` if the bundle could not be computed."""
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
) -> dict[str, Any]:
    """Compute the evaluation card (M8): cross-modal agreement matrix +
    per-modality final cosine similarity summary.

    This is a minimum-viable evaluation: it does NOT yet include the
    SigLIP2 / DINOv3 / CLAP cross-encoder probes (those are the M8 full
    deliverable). What it produces:
      * Per-modality final similarity to the canonical target.
      * Cross-modal agreement matrix: pairwise cosine similarities between
        the per-modality output embeddings in the canonical space (so
        a perfectly cross-modally-aligned showcase has all-ones off-diagonal).
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

    return {
        "per_modality_similarity": similarities,
        "cross_modal_text_anchor_agreement_jaccard": agreement,
        "encoder": encoder_name,
        "notes": (
            "Minimum-viable evaluation: per-modality final cosine + Jaccard "
            "overlap of top-10 text-anchor words. Cross-encoder probes "
            "(SigLIP2/CLAP/DINOv3) and seed-stability are the M8 follow-up."
        ),
    }


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
