import asyncio
import logging
import os
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
import soundfile as sf
import torch
from diffusers.utils import export_to_video
from torchvision.transforms.functional import to_pil_image

from embedding_art import EmbeddingArtEngine, OptimizationConfig
from embedding_art.core.concept_spec import ConceptSpec
from embedding_art.core.config import LossConfig
from embedding_art.core.render_result import LossBreakdown
from embedding_art.encoders.defaults import create_default_registry
from embedding_art.generators.audio import AudioLDMGenerator
from embedding_art.generators.video import SVDVideoGenerator
from embedding_art.web.sockets import manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_STEPS = 30
DEFAULT_LEARNING_RATE = 0.1

# Global engine instance (lazy loaded)
_engine: EmbeddingArtEngine | None = None
_executor = ThreadPoolExecutor(max_workers=1)


def get_engine() -> EmbeddingArtEngine:
    global _engine
    if _engine is None:
        logger.info("Initializing Engine (this may take a while)...")
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        registry = create_default_registry()
        _engine = EmbeddingArtEngine.from_registry(
            registry, default_encoder="imagebind", device=device
        )

        # Register generators
        from embedding_art.generators.diffusion import SDXLDiffusionGenerator

        _engine.register_generator("image", SDXLDiffusionGenerator(device=device))

        try:
            _engine.register_generator("audio", AudioLDMGenerator(device=device))
        except Exception as e:
            logger.warning(f"Audio generator failed to load: {e}")

        try:
            _engine.register_generator("video", SVDVideoGenerator(device=device))
        except Exception as e:
            logger.warning(f"Video generator failed to load: {e}")

    return _engine


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Represents an optimization job.

    Three ``kind`` values are supported:

    * ``"single"`` (default) — a single-modality optimisation job.
    * ``"compare"`` — same concept across multiple encoders.
    * ``"showcase"`` — v3 four-modality bundle in a single shared
      encoder space; writes a directory of artefacts and a manifest.
    """

    id: str
    target_text: list[str] = field(default_factory=list)
    output_modality: str = "image"
    encoder_name: str = "languagebind"
    similarity_weight: float = 1.0
    feature_matching_weight: float = 0.5
    # For compare jobs: list of encoder names to compare
    compare_encoder_names: list[str] = field(default_factory=list)
    # For showcase jobs.
    kind: str = "single"
    modalities: list[str] = field(default_factory=lambda: ["image", "audio", "video", "text"])
    image_backbone: str = "sd35"
    audio_backbone: str = "stable-audio-open"
    video_backbone: str = "ltx-video"
    tracks: list[str] = field(default_factory=lambda: ["honest"])
    autocast_dtype: str = "fp32"
    interpret: bool = True
    evaluate: bool = True
    sae_path: str | None = None
    showcase_steps: int = 200
    seed: int | None = None
    manifest_path: str | None = None
    output_dir: str | None = None
    status: JobStatus = JobStatus.QUEUED
    created_at: datetime = field(default_factory=datetime.now)
    progress: float = 0.0
    error: str | None = None
    result_path: str | None = None
    logs: list[str] = field(default_factory=list)


class JobManager:
    """Manages background optimization jobs."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()

    def create_job(
        self,
        target_text: list[str],
        output_modality: str = "image",
        encoder_name: str = "imagebind",
        similarity_weight: float = 1.0,
        feature_matching_weight: float = 0.5,
    ) -> Job:
        """Create a new optimization job."""
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            target_text=target_text,
            output_modality=output_modality,
            encoder_name=encoder_name,
            similarity_weight=similarity_weight,
            feature_matching_weight=feature_matching_weight,
        )
        self._jobs[job_id] = job
        return job

    def create_compare_job(
        self,
        target_text: str,
        output_modality: str = "image",
        encoder_names: list[str] | None = None,
        steps: int = DEFAULT_STEPS,
    ) -> Job:
        """Create a comparison job that runs render_compare across multiple encoders."""
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            target_text=[target_text],
            output_modality=output_modality,
            encoder_name=encoder_names[0] if encoder_names else "languagebind",
            compare_encoder_names=encoder_names or [],
            kind="compare",
        )
        self._jobs[job_id] = job
        return job

    def create_showcase_job(
        self,
        target_text: str,
        modalities: list[str] | None = None,
        encoder_name: str = "languagebind",
        image_backbone: str = "sd35",
        audio_backbone: str = "stable-audio-open",
        video_backbone: str = "ltx-video",
        tracks: list[str] | None = None,
        autocast_dtype: str = "fp32",
        interpret: bool = True,
        evaluate: bool = True,
        sae_path: str | None = None,
        steps: int = 200,
        seed: int | None = None,
    ) -> Job:
        """Create a v3 four-modality showcase job.

        Writes its output bundle to ``outputs/showcase/<job_id>/`` and
        records the manifest URL on the job.
        """
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            target_text=[target_text],
            kind="showcase",
            modalities=list(modalities) if modalities else ["image", "audio", "video", "text"],
            encoder_name=encoder_name,
            image_backbone=image_backbone,
            audio_backbone=audio_backbone,
            video_backbone=video_backbone,
            tracks=list(tracks) if tracks else ["honest"],
            autocast_dtype=autocast_dtype,
            interpret=interpret,
            evaluate=evaluate,
            sae_path=sae_path,
            showcase_steps=steps,
            seed=seed,
        )
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Job | None:
        """Get job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        """List all jobs."""
        return list(self._jobs.values())

    async def start_job(self, job_id: str) -> None:
        """Start a job in a background thread."""
        job = self.get_job(job_id)
        if not job:
            return

        loop = asyncio.get_running_loop()
        loop.run_in_executor(_executor, self._run_job_sync, job, loop)

    def _run_job_sync(self, job: Job, loop: asyncio.AbstractEventLoop) -> None:
        """Synchronous execution of the job."""

        def broadcast(msg: dict):
            try:
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(manager.broadcast_to_job(job.id, msg), loop)
            except RuntimeError:
                pass

        _run_job_sync_direct(job, loop, broadcast)


def _run_job_sync_direct(
    job: Job,
    loop: asyncio.AbstractEventLoop,
    broadcast: Callable[[dict], None],
) -> None:
    """
    Execute a job synchronously, using *broadcast* to send WebSocket messages.

    Extracted from JobManager so tests can call it without needing a live loop.
    """
    try:
        job.status = JobStatus.RUNNING
        broadcast({"type": "status", "status": "running"})

        if job.kind == "showcase":
            _run_showcase_job(job, broadcast)
            return

        job.logs.append("Initializing engine...")
        broadcast({"type": "log", "message": "Initializing engine..."})

        engine = get_engine()

        job.logs.append(f"Optimizing for targets: {job.target_text}")
        broadcast({"type": "log", "message": f"Optimizing for targets: {job.target_text}"})

        # Build ConceptSpec from the first target text
        spec = ConceptSpec(text=job.target_text[0])

        # Build LossConfig from job loss weights
        loss_config = LossConfig(
            similarity_weight=job.similarity_weight,
            feature_matching_weight=job.feature_matching_weight,
        )

        config = OptimizationConfig(
            steps=DEFAULT_STEPS,
            learning_rate=DEFAULT_LEARNING_RATE,
            loss=loss_config,
        )

        # Build the strategy with a progress callback so we can stream updates
        from embedding_art.core.strategies import OptimizationStrategy

        def progress_callback(step: int, breakdown: LossBreakdown, output: torch.Tensor) -> None:
            _broadcast_progress(
                broadcast=broadcast,
                job_id=job.id,
                step=step,
                total_steps=config.steps,
                breakdown=breakdown,
            )
            job.progress = step / config.steps if config.steps > 0 else 0.0

        strategy = OptimizationStrategy()

        # Wrap the strategy so the callback is injected automatically
        class _StrategyWithCallback:
            def render(self, target, generator, encoder, cfg):
                return strategy.render(
                    target,
                    generator,
                    encoder,
                    cfg,
                    callback=progress_callback,
                )

        # Compare job vs. single-encoder job
        if job.compare_encoder_names:
            results = engine.render_compare(
                spec,
                encoder_names=job.compare_encoder_names,
                output_modality=job.output_modality,
                config=config,
            )
            # For compare jobs, save the first result as the primary output
            first_result = next(iter(results.values()))
            render_result = first_result
        else:
            render_result = engine.render(
                spec,
                encoder_name=job.encoder_name,
                output_modality=job.output_modality,
                strategy=_StrategyWithCallback(),
                config=config,
            )

        # Save result
        outputs_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(outputs_dir, exist_ok=True)

        ext = "png"
        if job.output_modality == "audio":
            ext = "wav"
        elif job.output_modality == "video":
            ext = "mp4"

        filename = f"{job.id}.{ext}"
        filepath = os.path.join(outputs_dir, filename)

        if job.output_modality == "image":
            # result.output is already a decoded tensor [B, C, H, W]
            image = to_pil_image(render_result.output.squeeze(0).clamp(0, 1))
            image.save(filepath)

        elif job.output_modality == "audio":
            # Decode audio tensor — render_result.output shape [B, Samples]
            audio_np = render_result.output.squeeze(0).cpu().numpy()
            generator_instance = engine.get_generator("audio")
            sample_rate = getattr(generator_instance, "SAMPLE_RATE", 16000)
            sf.write(filepath, audio_np, sample_rate)

        elif job.output_modality == "video":
            # render_result.output shape [B, F, C, H, W]
            frames_tensor = render_result.output.squeeze(0)  # [F, C, H, W]
            frames = []
            for frame in frames_tensor:
                frames.append(to_pil_image(frame.clamp(0, 1)))
            frames_np = [np.array(f) for f in frames]
            export_to_video(frames_np, filepath, fps=8)

        # Set relative URL for frontend
        job.result_path = f"/outputs/{filename}"
        job.logs.append(f"Saved result to {job.result_path}")

        final_sim = render_result.final_similarity
        job.status = JobStatus.COMPLETED
        job.progress = 1.0
        job.logs.append(f"Finished! Final similarity: {final_sim:.4f}")
        broadcast({"type": "status", "status": "completed"})
        broadcast({"type": "progress", "progress": 1.0})
        broadcast({"type": "log", "message": f"Finished! Final similarity: {final_sim:.4f}"})
        broadcast({"type": "result", "url": job.result_path})
        broadcast({"type": "log", "message": f"Saved result to {job.result_path}"})

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Job failed: {e}")
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.logs.append(f"Error: {e}")
        job.logs.append(f"Traceback:\n{tb}")
        broadcast({"type": "status", "status": "failed"})
        broadcast({"type": "error", "message": str(e)})
        broadcast({"type": "log", "message": f"Error: {e}"})
        broadcast({"type": "log", "message": "Check server logs for full traceback"})


def _broadcast_progress(
    *,
    broadcast: Callable[[dict], None],
    job_id: str,
    step: int,
    total_steps: int,
    breakdown: LossBreakdown,
) -> None:
    """Build and send a structured progress message with loss breakdown."""
    progress = step / total_steps if total_steps > 0 else 0.0
    components = {k: v.item() for k, v in breakdown.components.items()}
    msg = f"Step {step}: loss={breakdown.total.item():.4f}"
    broadcast(
        {
            "type": "progress",
            "progress": progress,
            "step": step,
            "loss": breakdown.total.item(),
            "components": components,
            "log": msg,
        }
    )


def _run_showcase_job(job: Job, broadcast: Callable[[dict], None]) -> None:
    """Execute a v3 four-modality showcase job in-process.

    The work is delegated to ``embedding_art.cli.commands.showcase._showcase_impl``
    so the HTTP and CLI surfaces stay in lockstep. The manifest URL is
    written back to ``job.manifest_path`` / ``job.result_path`` once the
    bundle is on disk.
    """
    try:
        from pathlib import Path

        from embedding_art.cli.commands.showcase import _showcase_impl

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        outputs_dir = os.path.join(os.getcwd(), "outputs", "showcase", job.id)
        Path(outputs_dir).mkdir(parents=True, exist_ok=True)

        target = job.target_text[0]
        job.logs.append(f"Rendering showcase for '{target}' to {outputs_dir}")
        broadcast({"type": "log", "message": f"Rendering showcase for '{target}'"})

        def _showcase_progress(event: dict) -> None:
            """Forward structured showcase events to the websocket broadcaster.

            The CLI _showcase_impl emits a handful of event types:
            ``showcase_start`` / ``track_start`` / ``modality_start`` /
            ``modality_complete`` / ``track_complete``. Each becomes both a
            human-readable log line (so the Logs panel stays useful) and a
            structured ``showcase_event`` message (so the UI can render
            live text-anchor readouts, current modality, etc.).
            """
            kind = event.get("type", "")
            if kind == "showcase_start":
                msg = (
                    f"Showcase start: target='{event.get('target_text')}' "
                    f"modalities={event.get('modalities')} "
                    f"tracks={event.get('tracks')}"
                )
            elif kind == "track_start":
                msg = f"Track start: {event.get('track')}"
            elif kind == "modality_start":
                msg = (
                    f"Rendering modality '{event.get('modality')}' " f"(track={event.get('track')})"
                )
            elif kind == "modality_complete":
                sim = event.get("similarity")
                sim_str = f"{sim:.3f}" if isinstance(sim, (int, float)) else "?"
                msg = (
                    f"Completed '{event.get('modality')}' "
                    f"(track={event.get('track')}, sim={sim_str})"
                )
                anchor = event.get("text_anchor")
                if isinstance(anchor, list) and anchor:
                    words = [a.get("word", "?") for a in anchor[:5] if isinstance(a, dict)]
                    if words:
                        msg += f" anchor=[{', '.join(words)}]"
            elif kind == "track_complete":
                msg = f"Track complete: {event.get('track')}"
            else:
                msg = f"event: {kind}"
            job.logs.append(msg)
            broadcast({"type": "log", "message": msg})
            broadcast({"type": "showcase_event", "event": event})

        _showcase_impl(
            target_text=target,
            output_dir=Path(outputs_dir),
            encoder_name=job.encoder_name,
            modalities=job.modalities,
            steps=job.showcase_steps,
            seed=job.seed,
            device=device,
            image_backbone=job.image_backbone,
            audio_backbone=job.audio_backbone,
            video_backbone=job.video_backbone,
            tracks=job.tracks,
            autocast_dtype=job.autocast_dtype,
            interpret=job.interpret,
            sae_path=Path(job.sae_path) if job.sae_path else None,
            evaluate=job.evaluate,
            progress_callback=_showcase_progress,
        )

        manifest_url = f"/outputs/showcase/{job.id}/manifest.json"
        job.output_dir = outputs_dir
        job.manifest_path = manifest_url
        job.result_path = manifest_url
        job.status = JobStatus.COMPLETED
        job.progress = 1.0
        job.logs.append(f"Showcase manifest: {manifest_url}")
        broadcast({"type": "status", "status": "completed"})
        broadcast({"type": "progress", "progress": 1.0})
        broadcast({"type": "result", "url": manifest_url, "kind": "showcase"})
        broadcast({"type": "log", "message": f"Showcase manifest: {manifest_url}"})
    except Exception as e:  # pragma: no cover - error path mirrors single-job path
        tb = traceback.format_exc()
        logger.error(f"Showcase job failed: {e}")
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.logs.append(f"Error: {e}")
        job.logs.append(f"Traceback:\n{tb}")
        broadcast({"type": "status", "status": "failed"})
        broadcast({"type": "error", "message": str(e)})
        broadcast({"type": "log", "message": f"Error: {e}"})


# Global instance
job_manager = JobManager()
