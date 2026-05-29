"""
LTX-Video 0.9.5 generator (M6 upgrade from Stable Video Diffusion).

LTX-Video is Lightricks' open video diffusion model. v0.9.5 is the 2B
DiT-based model that produces 5-second clips at 768×512 resolution and
24 fps. Substantially sharper than Stable Video Diffusion (the v1/v2
default) and fits comfortably on M1 Max with the VAE-only decode path.

Architecture
------------
* CausalVideoAutoencoder (CVA) with 128 latent channels.
* DiT video transformer (we skip — embedding-art conditions on the shared
  LanguageBind embedding via latent optimisation).
* T5 text conditioner (also unused).

Memory profile on M1 Max
------------------------
* Model: ~5 GB.
* Latent for 5 s @ 24 fps × 768×512: ``[1, 128, T_lat, 96, 64]`` where
  ``T_lat ≈ frames / 8 = 15``.
* Decode runtime: ~10-30 s on MPS for 5 s clips.

Notes
-----
* Open RAIL-M licensed.
* The VAE-only decode path here is sufficient for embedding-art; the
  diffusion sampler is only needed for the 'natural' VSD track.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from embedding_art.exceptions import GeneratorError, ModelLoadError, OutOfMemoryError

logger = logging.getLogger(__name__)


class LTXVideoGenerator:
    """LTX-Video 0.9.5 VAE-only video generator.

    Decodes a 128-channel video latent into an RGB frame stack.

    Attributes
    ----------
    LATENT_CHANNELS : int
        CVA latent channel count (128).
    DEFAULT_HEIGHT : int
        Default frame height in pixels (512).
    DEFAULT_WIDTH : int
        Default frame width in pixels (768).
    DEFAULT_FPS : int
        Default frame rate (24).
    SPATIAL_DOWNSAMPLE : int
        Spatial downsampling factor from frame to latent (8).
    TEMPORAL_DOWNSAMPLE : int
        Temporal downsampling factor from frames to latent (8).
    SCALING_FACTOR : float
        Latent scaling factor for CVA (1.0; CVA is identity-scaled by default).
    """

    LATENT_CHANNELS = 128
    DEFAULT_HEIGHT = 512
    DEFAULT_WIDTH = 768
    DEFAULT_FPS = 24
    SPATIAL_DOWNSAMPLE = 8
    TEMPORAL_DOWNSAMPLE = 8
    SCALING_FACTOR = 1.0

    def __init__(
        self,
        model_id: str = "Lightricks/LTX-Video",
        device: str = "mps",
        num_frames: int = 121,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
        fps: int = DEFAULT_FPS,
    ) -> None:
        if num_frames <= 0 or height <= 0 or width <= 0:
            raise ValueError("num_frames, height, width must all be positive")

        self._device = torch.device(device)
        self._model_id = model_id
        self._num_frames = int(num_frames)
        self._height = int(height)
        self._width = int(width)
        self._fps = int(fps)

        self._latent_t = max(1, num_frames // self.TEMPORAL_DOWNSAMPLE)
        self._latent_h = max(1, height // self.SPATIAL_DOWNSAMPLE)
        self._latent_w = max(1, width // self.SPATIAL_DOWNSAMPLE)

        # Load ONLY the VAE. The full LTXPipeline pulls the multi-GB video DiT +
        # T5 that latent optimisation never uses; loading it and discarding all
        # but .vae blew the memory budget. Fall back to the full pipeline only
        # if the installed diffusers can't load the VAE subfolder.
        pipeline = None
        try:
            from diffusers import AutoencoderKLLTXVideo

            self.vae = AutoencoderKLLTXVideo.from_pretrained(
                model_id, subfolder="vae", torch_dtype=torch.float32
            )
        except (ImportError, OSError, ValueError):
            try:
                from diffusers import LTXPipeline

                pipeline = LTXPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
                self.vae = pipeline.vae
            except ImportError as exc:
                raise GeneratorError(
                    "LTX-Video requires diffusers>=0.32 with LTXPipeline support"
                ) from exc
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    raise OutOfMemoryError(
                        operation="loading LTX-Video pipeline",
                        device=device,
                        original_error=e,
                    ) from e
                raise ModelLoadError(model_name=model_id, original_error=e) from e
            except OSError as e:
                raise ModelLoadError(model_name=model_id, original_error=e) from e

        try:
            self.vae.to(self._device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise OutOfMemoryError(
                    operation="moving LTX-Video VAE to device",
                    device=device,
                    original_error=e,
                ) from e
            raise

        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        del pipeline

        # Apple Silicon perf: tile the video VAE decode so 5-second
        # 720p clips don't peak above the M1 Max unified-memory budget.
        from embedding_art.perf import apply_diffusers_perf_knobs

        apply_diffusers_perf_knobs(
            self,
            fuse_qkv=False,
            enable_attention_slicing=False,
            enable_vae_tiling=True,
        )

    @property
    def latent_shape(self) -> tuple[int, ...]:
        """``[1, 128, T_lat, H_lat, W_lat]`` — CVA latent shape."""
        return (1, self.LATENT_CHANNELS, self._latent_t, self._latent_h, self._latent_w)

    @property
    def output_modality(self) -> str:
        return "video"

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def fps(self) -> int:
        return self._fps

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    def init_latent(self, seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
        return torch.randn(self.latent_shape, device=self._device, requires_grad=True)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent → video frames ``[1, 3, T, H, W]`` in [0, 1]."""
        scaled = latent * self.SCALING_FACTOR
        try:
            frames = self.vae.decode(scaled).sample
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise OutOfMemoryError(
                    operation="LTX-Video VAE decode",
                    device=str(self._device),
                    original_error=e,
                ) from e
            raise GeneratorError(f"LTX-Video decode failed: {e}") from e
        return (frames.clamp(-1, 1) + 1) / 2

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode video frames → latent. Reverses :meth:`decode`."""
        try:
            posterior: Any = self.vae.encode(frames * 2 - 1).latent_dist
            latent = posterior.mode() if hasattr(posterior, "mode") else posterior.sample()
        except RuntimeError as e:
            raise GeneratorError(f"LTX-Video encode failed: {e}") from e
        return latent / self.SCALING_FACTOR
