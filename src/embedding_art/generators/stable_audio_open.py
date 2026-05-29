"""
Stable Audio Open 1.0 generator (M6 upgrade from AudioLDM2).

Stable Audio Open is a 1.21B-parameter latent diffusion model trained on
audio-text pairs. It produces 47-second clips at 44.1 kHz from a 64-channel
VAE latent — substantially higher quality than AudioLDM2's 10.24-second
16 kHz output, making it the v3-default audio backbone.

Architecture
------------
* DAC (Descript Audio Codec) VAE: 64 latent channels, ~2048× temporal
  downsampling so the latent is short even for long clips.
* DiT (diffusion transformer) score model.
* T5 text conditioner (we ignore this — embedding-art conditions on the
  shared LanguageBind embedding instead).

Memory profile on M1 Max
------------------------
* Model: ~5 GB (DAC ~1 GB, DiT ~3 GB, T5 if used ~1 GB).
* Latent for 10 s @ 44.1 kHz at 1024× temporal compression:
  ``[1, 64, 1, T_latent]`` where ``T_latent = (10 * 44100) / 2048 ≈ 215``.
* Decode runtime: <1 s on MPS for 10 s clips.

Notes
-----
* MIT licensed (permissive).
* Stable Audio Open is the canonical Stability audio backbone for 2026.
* The VAE-only decode path (this file) is sufficient for embedding-art
  because we optimize the latent directly. The diffusion sampler is only
  needed for the 'natural' VSD track (M5 follow-up).
* Lazy import of ``diffusers`` so the rest of the codebase isn't forced
  to load the model when only the type is referenced.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from embedding_art.exceptions import GeneratorError, ModelLoadError, OutOfMemoryError

logger = logging.getLogger(__name__)


class StableAudioOpenGenerator:
    """Stable Audio Open 1.0 VAE-only audio generator.

    Decodes a 64-channel latent (the DAC autoencoder's latent space)
    into a 44.1 kHz audio waveform. Forward pass:

        latent [1, 64, 1, T_latent]
            → DAC decode (scaling applied)
            → waveform [1, 1, T_audio]

    The latent dimensions are calculated from the configured ``audio_length_in_s``.

    Attributes
    ----------
    LATENT_CHANNELS : int
        DAC latent channel count (64).
    SAMPLE_RATE : int
        Output audio sample rate in Hz (44100).
    COMPRESSION_RATIO : int
        Total temporal compression from waveform to latent (≈2048).
    SCALING_FACTOR : float
        Latent scaling factor used by the DAC decode pipeline.
    """

    LATENT_CHANNELS = 64
    SAMPLE_RATE = 44100
    COMPRESSION_RATIO = 2048
    SCALING_FACTOR = 1.0  # Stable Audio Open VAE is identity-scaled.
    DEFAULT_AUDIO_LENGTH = 10.0
    MAX_AUDIO_LENGTH = 47.0

    def __init__(
        self,
        model_id: str = "stabilityai/stable-audio-open-1.0",
        device: str = "mps",
        audio_length_in_s: float = DEFAULT_AUDIO_LENGTH,
    ) -> None:
        if audio_length_in_s <= 0 or audio_length_in_s > self.MAX_AUDIO_LENGTH:
            raise ValueError(
                f"audio_length_in_s must be in (0, {self.MAX_AUDIO_LENGTH}]; got {audio_length_in_s}"
            )

        self._device = torch.device(device)
        self._model_id = model_id
        self._audio_length = float(audio_length_in_s)

        time_frames = int(audio_length_in_s * self.SAMPLE_RATE)
        self._latent_length = max(1, time_frames // self.COMPRESSION_RATIO)

        # Load ONLY the VAE. The full StableAudioPipeline pulls the multi-GB DiT
        # + T5 text encoder that latent optimisation never uses; loading it and
        # discarding all but .vae blew the memory budget. Fall back to the full
        # pipeline only if the installed diffusers can't load the VAE subfolder.
        pipeline = None
        try:
            from diffusers import AutoencoderOobleck

            self.vae = AutoencoderOobleck.from_pretrained(
                model_id, subfolder="vae", torch_dtype=torch.float32
            )
        except (ImportError, OSError, ValueError):
            try:
                from diffusers import StableAudioPipeline

                pipeline = StableAudioPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
                self.vae = pipeline.vae
            except ImportError as exc:
                raise GeneratorError(
                    "Stable Audio Open requires diffusers>=0.27 with StableAudioPipeline support"
                ) from exc
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    raise OutOfMemoryError(
                        operation="loading Stable Audio Open pipeline",
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
                    operation="moving Stable Audio Open VAE to device",
                    device=device,
                    original_error=e,
                ) from e
            raise

        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        del pipeline

        # Apple Silicon perf: enable VAE tiling for the audio decoder so
        # 47-second clips don't peak above the M1 Max unified-memory
        # budget. Other knobs (QKV fusion, attention slicing) are full-
        # pipeline only.
        from embedding_art.perf import apply_diffusers_perf_knobs

        apply_diffusers_perf_knobs(
            self,
            fuse_qkv=False,
            enable_attention_slicing=False,
            enable_vae_tiling=True,
        )

    @property
    def latent_shape(self) -> tuple[int, ...]:
        """``[1, 64, 1, T_latent]`` — DAC latent shape."""
        return (1, self.LATENT_CHANNELS, 1, self._latent_length)

    @property
    def output_modality(self) -> str:
        return "audio"

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def audio_length(self) -> float:
        return self._audio_length

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    def init_latent(self, seed: int | None = None) -> torch.Tensor:
        """Initialise a random latent for optimisation."""
        if seed is not None:
            torch.manual_seed(seed)
        latent = torch.randn(self.latent_shape, device=self._device, requires_grad=True)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent → audio waveform [1, 1, T_audio]."""
        scaled = latent * self.SCALING_FACTOR
        try:
            audio = self.vae.decode(scaled).sample
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise OutOfMemoryError(
                    operation="Stable Audio Open VAE decode",
                    device=str(self._device),
                    original_error=e,
                ) from e
            raise GeneratorError(f"Stable Audio Open decode failed: {e}") from e
        return audio

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode waveform → latent. Reverses :meth:`decode` for round-trip tests."""
        try:
            posterior: Any = self.vae.encode(audio).latent_dist
            latent = posterior.mode() if hasattr(posterior, "mode") else posterior.sample()
        except RuntimeError as e:
            raise GeneratorError(f"Stable Audio Open encode failed: {e}") from e
        return latent / self.SCALING_FACTOR
