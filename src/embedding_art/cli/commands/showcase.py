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

    if "image" in modalities:
        manifest["modalities"]["image"] = _render_image(
            engine=engine,
            target=target,
            encoder_name=encoder_name,
            config=config,
            output_dir=output_dir,
            device=device,
            backbone=image_backbone,
        )

    if "audio" in modalities:
        manifest["modalities"]["audio"] = _render_audio(
            engine=engine,
            target=target,
            encoder_name=encoder_name,
            config=config,
            output_dir=output_dir,
            device=device,
        )

    if "video" in modalities:
        manifest["modalities"]["video"] = _render_video(
            engine=engine,
            target=target,
            encoder_name=encoder_name,
            config=config,
            output_dir=output_dir,
            device=device,
        )

    if "text" in modalities:
        manifest["modalities"]["text"] = _render_text_card(
            target=target,
            target_text=target_text,
            output_dir=output_dir,
            manifest=manifest,
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
    encoder_name: str,
    config: Any,
    output_dir: Path,
    device: str,
    backbone: str,
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

    return {
        "path": str(image_path.relative_to(output_dir)),
        "final_similarity": float(result.final_similarity),
        "backbone": backbone,
    }


def _render_audio(
    *,
    engine: Any,
    target: Any,
    encoder_name: str,
    config: Any,
    output_dir: Path,
    device: str,
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

    return {
        "path": str(audio_path.relative_to(output_dir)),
        "final_similarity": float(result.final_similarity),
        "backbone": "audioldm2",
    }


def _render_video(
    *,
    engine: Any,
    target: Any,
    encoder_name: str,
    config: Any,
    output_dir: Path,
    device: str,
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

    return {
        "path": str(written.relative_to(output_dir)),
        "final_similarity": float(result.final_similarity),
        "backbone": "svd",
    }


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
