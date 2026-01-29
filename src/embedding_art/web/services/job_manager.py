import logging
import asyncio
import uuid
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import soundfile as sf
import numpy as np
import torch
import traceback
from diffusers.utils import export_to_video

from embedding_art import EmbeddingArtEngine, OptimizationConfig
from embedding_art.core.concept import Concept
from embedding_art.encoders.imagebind import ImageBindEncoder
from embedding_art.generators.image import SDXLImageGenerator
from embedding_art.generators.audio import AudioLDMGenerator
from embedding_art.generators.video import SVDVideoGenerator
from embedding_art.regularizers import CompositeRegularizer, TotalVariation, SpectralRegularizer, LatentNorm
from embedding_art.web.sockets import manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_STEPS = 30
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_GUIDANCE_SCALE = 250.0 # Clipped Raw Gradients
IMAGEBIND_SCALE_NORMALIZED = 0.2

# Global engine instance (lazy loaded)
_engine: EmbeddingArtEngine | None = None
_executor = ThreadPoolExecutor(max_workers=1)

def get_engine() -> EmbeddingArtEngine:
    global _engine
    if _engine is None:
        logger.info("Initializing Engine (this may take a while)...")
        # Initialize components
        # Note: Using 'cpu' for safety in dev env if no cuda/mps. 
        # Ideally check torch.device or allow config.
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        encoder = ImageBindEncoder(device=device)
        _engine = EmbeddingArtEngine(encoder=encoder, device=device)
        
        # Register generators
        from embedding_art.generators.diffusion import SDXLDiffusionGenerator
        _engine.register_generator("image", SDXLDiffusionGenerator(device=device))

        # Lazy load these if possible to save memory, but for now register all
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
    """Represents an optimization job."""

    id: str
    target_text: list[str] = field(default_factory=list)
    output_modality: str = "image"
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

    def create_job(self, target_text: list[str], output_modality: str = "image") -> Job:
        """Create a new job."""
        job_id = str(uuid.uuid4())
        job = Job(id=job_id, target_text=target_text, output_modality=output_modality)
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
        # Run in executor to avoid blocking main thread
        loop.run_in_executor(_executor, self._run_job_sync, job, loop)

    def _run_job_sync(self, job: Job, loop: asyncio.AbstractEventLoop) -> None:
        """Synchronous execution of the job."""

        def broadcast(msg: dict):
            # Fire and forget broadcast
            try:
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(manager.broadcast_to_job(job.id, msg), loop)
            except RuntimeError:
                pass # Loop might be closed

        try:
            job.status = JobStatus.RUNNING
            job.logs.append("Initializing engine...")
            broadcast({"type": "status", "status": "running"})
            broadcast({"type": "log", "message": "Initializing engine..."})
            
            engine = get_engine()
            
            job.logs.append(f"Optimizing for targets: {job.target_text}")
            broadcast({"type": "log", "message": f"Optimizing for targets: {job.target_text}"})
            
            # Create target concept
            # Assuming first text for now. Support multiple later.
            target_concept = Concept.from_text(job.target_text[0], engine.encoder)
            
            # Run optimization / generation
            
            # Activate ImageBind Guidance! 
            # Scale of 250.0 is used with Raw Gradients (Clipped).
            # Restored dynamics but clipped for safety.
            config = OptimizationConfig(
                steps=DEFAULT_STEPS, 
                learning_rate=DEFAULT_LEARNING_RATE, 
                guidance_scale=DEFAULT_GUIDANCE_SCALE
            )

            # Callback to update progress
            def step_callback(step: int, loss: float, sim: float, latent: torch.Tensor) -> None:
                job.progress = (step + 1) / DEFAULT_STEPS
                if step % 5 == 0:
                    msg = f"Step {step}"
                    if loss != 0.0:
                        msg += f": loss={loss:.4f}, sim={sim:.4f}"
                        
                    job.logs.append(msg)
                    broadcast({
                        "type": "progress", 
                        "progress": job.progress, 
                        "step": step, 
                        "loss": loss, 
                        "similarity": sim,
                        "log": msg
                    })

            # Select regularizers based on modality
            # Manual configuration for stronger regularization on images
            regs = CompositeRegularizer.default_image()
            if job.output_modality == "image":
                regs = CompositeRegularizer(
                    regularizers=[
                        TotalVariation(weight=0.25),      # Light smoothing (was 2.0 which killed everything)
                        SpectralRegularizer(weight=0.01), # Standard anti-noise (was 0.1)
                        LatentNorm(weight=0.5),           # Keep latents valid
                    ]
                )
            elif job.output_modality == "audio":
                regs = CompositeRegularizer.default_audio()
            elif job.output_modality == "video":
                regs = CompositeRegularizer.default_video()

            result = engine.optimize(
                target=target_concept,
                output_modality=job.output_modality,
                config=config,
                regularizers=regs,
                callback=step_callback,
                progress=False # Disable tqdm
            )
            
            # Save result
            OUTPUTS_DIR = os.path.join(os.getcwd(), "outputs")
            
            # Determine extension based on modality
            ext = "png"
            if job.output_modality == "audio":
                ext = "wav"
            elif job.output_modality == "video":
                ext = "mp4"
                
            filename = f"{job.id}.{ext}"
            filepath = os.path.join(OUTPUTS_DIR, filename)
            
            # Get specific generator and save
            if job.output_modality == "image":
                image = result.get_final_image(engine.get_generator("image"))
                image.save(filepath)
            
            elif job.output_modality == "audio":
                sample_rate, audio_data = result.get_final_audio(engine.get_generator("audio"))
                # audio_data is numpy array [samples]
                sf.write(filepath, audio_data, sample_rate)
                
            elif job.output_modality == "video":
                frames = result.get_final_video(engine.get_generator("video"))
                # frames is list[PIL.Image]
                # Convert to list of numpy arrays for export_to_video
                frames_np = [np.array(f) for f in frames]
                export_to_video(frames_np, filepath, fps=8)
            
            # Set relative URL for frontend
            job.result_path = f"/outputs/{filename}"
            job.logs.append(f"Saved result to {job.result_path}")
            
            # Final status update with result
            job.status = JobStatus.COMPLETED
            job.progress = 1.0
            job.logs.append(f"Finished! Final similarity: {result.final_similarity:.4f}")
            broadcast({"type": "status", "status": "completed"})
            broadcast({"type": "progress", "progress": 1.0})
            broadcast({"type": "log", "message": f"Finished! Final similarity: {result.final_similarity:.4f}"})
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

# Global instance
job_manager = JobManager()
