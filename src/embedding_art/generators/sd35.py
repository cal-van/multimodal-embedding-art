"""
SD3.5-medium image generator — canonical 2026-crispy image backbone (M2 / v3).

Replaces the v1 SDXL-VAE-only generator (``image.SDXLImageGenerator``) for new
work.  SD3.5-medium ships a 16-channel flow-matching latent that decodes to
materially sharper 1024×1024 RGB than SDXL's 4-channel DDPM latent, and uses a
Multimodal Diffusion Transformer (MMDiT) which makes Variational Score
Distillation (VSD) — the M2 prior loss — straightforward to plumb on top.

This module exposes a *VAE-only* generator that is sufficient for the latent-
space optimisation loop used in v1/v2:

* ``init_latent`` initialises a random 16-channel latent.
* ``decode`` runs the SD3.5 VAE decoder to produce a [B, 3, 1024, 1024] image
  in [0, 1].
* ``encode`` runs the reverse for round-trip tests.

Full VSD prior support (a LoRA-modified MMDiT supplying the variational
"phi" model, plus the flow-matching scheduler) is the M2b commit; this
module's decode interface is shaped so the VSD module can layer cleanly on
top without changing the generator's external contract.

Model card: https://huggingface.co/stabilityai/stable-diffusion-3.5-medium

Memory on M1 Max
----------------
* SD3.5-medium VAE (fp32): ~250 MB
* Full SD3.5-medium with transformer + 3× text encoders: ~5 GB (loaded by
  ``sd35_diffusion.py`` in M2b only — VAE alone is the slim path for the
  optimisation loop).
"""

from __future__ import annotations

import logging

import torch
from diffusers import AutoencoderKL

from embedding_art.exceptions import (
    GeneratorError,
    ModelLoadError,
    OutOfMemoryError,
)

logger = logging.getLogger(__name__)

# SD3.5 VAE constants — pulled from the model's vae/config.json on HuggingFace.
# Decode pipeline: x = (latent / scaling_factor) + shift_factor → vae.decode(x).
# (Confirm with stabilityai/stable-diffusion-3.5-medium/vae/config.json.)
SD35_SCALING_FACTOR = 1.5305
SD35_SHIFT_FACTOR = 0.0609

# Native SD3.5 latent geometry: 16 channels, 1024/8 = 128 spatial, decodes to
# 1024×1024 RGB. Smaller outputs use latent_size = output_size // 8.
LATENT_CHANNELS = 16
DEFAULT_OUTPUT_SIZE = 1024


class SD35ImageGenerator:
    """SD3.5-medium VAE image generator.

    Decodes 16-channel SD3.5 latents into RGB images in [0, 1].

    Parameters
    ----------
    model_id:
        HuggingFace model identifier. Defaults to ``stabilityai/stable-diffusion-3.5-medium``.
        Only the ``vae`` subfolder of this checkpoint is loaded by this class.
    device:
        PyTorch device string (e.g. ``"mps"``, ``"cuda"``, ``"cpu"``).
    output_size:
        Native output resolution (must be a multiple of 8). Defaults to 1024.
    dtype:
        VAE compute dtype. Defaults to ``torch.float32`` for MPS numerical
        stability; switch to ``torch.float16`` on CUDA to halve VAE memory.
    """

    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-3.5-medium",
        device: str = "mps",
        output_size: int = DEFAULT_OUTPUT_SIZE,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if output_size % 8 != 0:
            raise ValueError(
                f"output_size must be a multiple of 8 (SD3.5 VAE compression "
                f"ratio); got {output_size}"
            )

        self._device = torch.device(device)
        self._model_id = model_id
        self._output_size = output_size
        self._latent_size = output_size // 8
        self._dtype = dtype

        try:
            logger.info("Loading SD3.5 VAE from %s", model_id)
            self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise OutOfMemoryError(
                    operation="loading SD3.5 VAE",
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
                    operation="moving SD3.5 VAE to device",
                    device=device,
                    original_error=e,
                ) from e
            raise

        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------
    # Generator protocol
    # ------------------------------------------------------------------

    @property
    def latent_shape(self) -> tuple[int, ...]:
        """Latent tensor shape: ``(1, 16, latent_size, latent_size)``."""
        return (1, LATENT_CHANNELS, self._latent_size, self._latent_size)

    @property
    def output_modality(self) -> str:
        return "image"

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def output_size(self) -> int:
        return self._output_size

    def init_latent(
        self,
        seed: int | None = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Initialise a random latent for optimisation.

        SD3.5's latent is approximately unit-variance Gaussian (after the
        scaling+shift correction in ``decode``), so ``scale=1.0`` is the
        natural initialisation. Lowering ``scale`` to ~0.1 mimics the v1 SDXL
        path's behaviour if needed for legacy comparisons.
        """
        if seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(seed)
        else:
            generator = None
        latent = (
            torch.randn(
                self.latent_shape,
                device=self._device,
                dtype=self._dtype,
                generator=generator,
            )
            * scale
        )
        latent.requires_grad_(True)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a latent tensor to an RGB image in [0, 1].

        Args:
            latent: ``[B, 16, latent_size, latent_size]`` float tensor.

        Returns:
            ``[B, 3, output_size, output_size]`` float tensor in [0, 1].
        """
        try:
            scaled_latent = (latent / SD35_SCALING_FACTOR) + SD35_SHIFT_FACTOR
            decoded = self.vae.decode(scaled_latent).sample
            decoded = (decoded + 1) / 2  # VAE outputs [-1, 1]
            decoded = decoded.clamp(0, 1)
            return decoded
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise OutOfMemoryError(
                    operation="decoding SD3.5 latent to image",
                    device=str(self._device),
                    original_error=e,
                ) from e
            raise GeneratorError(operation="SD3.5 VAE decode", original_error=e) from e

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Encode an RGB image (in [0, 1]) into the SD3.5 latent space.

        Used for round-trip tests and for seed-from-image flows.

        Args:
            image: ``[B, 3, H, W]`` float tensor in [0, 1].

        Returns:
            ``[B, 16, H/8, W/8]`` float tensor.
        """
        try:
            image = image * 2 - 1
            with torch.no_grad():
                latent_dist = self.vae.encode(image).latent_dist
                latent = latent_dist.sample()
            latent = (latent - SD35_SHIFT_FACTOR) * SD35_SCALING_FACTOR
            return latent
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise OutOfMemoryError(
                    operation="encoding image to SD3.5 latent",
                    device=str(self._device),
                    original_error=e,
                ) from e
            raise GeneratorError(operation="SD3.5 VAE encode", original_error=e) from e
