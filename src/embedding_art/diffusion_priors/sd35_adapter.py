"""SD3.5-medium adapter for SDS / VSD priors (M2b).

This module bridges :mod:`embedding_art.diffusion_priors.sds` and
:mod:`embedding_art.diffusion_priors.vsd` to real Stable Diffusion 3.5
weights. SD3.5 is a *rectified-flow* MMDiT model, so the noise-prediction
contract of SDS/VSD has to be derived from velocity predictions.

Rectified-flow conventions used here
------------------------------------

Forward process is the linear interpolation::

    x_t = (1 - t) * x_0 + t * ε,   ε ~ N(0, I),   t ∈ [0, 1]

so by direct algebra::

    α_t = 1 - t
    σ_t = t

The transformer predicts the velocity ``v(x_t, t, y) = ε - x_0``; from the
identity ``x_t = (1 - t)·x_0 + t·ε`` and ``v = ε - x_0`` we recover
``ε_pred = x_t + (1 - t) * v_pred``. The :meth:`score_fn` method does this
conversion so callers see the DreamFusion ``ε`` interface that
:class:`SDSLoss` and :class:`VSDLoss` expect.

The pipeline is loaded *lazily* on the first :meth:`score_fn` /
:meth:`encode_prompt` call so importing this module never costs the 5 GB
of Stable Diffusion weights. A factory ``loader`` callable can be
substituted for testing — see :class:`SD35DiffusionAdapter`.

This adapter does NOT modify the pipeline weights. For the LoRA-tuned
``φ`` model that VSD requires, see :func:`build_vsd_phi_adapter` —
it returns a *second* adapter wrapping a LoRA-augmented copy of the
same pipeline, so the optimiser only updates LoRA parameters.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _default_loader(model_id: str, device: torch.device, dtype: torch.dtype) -> Any:
    """Default SD3.5 pipeline loader.

    Isolated as a private function so tests can substitute a stub via
    :class:`SD35DiffusionAdapter`'s ``loader`` argument without touching
    diffusers.
    """
    from diffusers import StableDiffusion3Pipeline  # local import: lazy

    pipeline = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipeline.to(device)
    if hasattr(pipeline, "set_progress_bar_config"):
        pipeline.set_progress_bar_config(disable=True)
    return pipeline


@dataclass
class SD35DiffusionAdapter:
    """Adapter that exposes SD3.5 as a noise-prediction prior.

    Args:
        model_id: HuggingFace model id. Defaults to SD3.5-medium.
        device: Torch device (``"mps"``, ``"cuda"``, ``"cpu"``).
        dtype: Compute dtype for the pipeline (``torch.float32`` is safe
            on MPS).
        loader: Callable ``(model_id, device, dtype) -> pipeline``. Defaults
            to ``_default_loader``. Override in tests to bypass real
            weight loading.

    The pipeline is *not* loaded by ``__init__``. The first call to
    :meth:`score_fn` or :meth:`encode_prompt` triggers loading.
    """

    model_id: str = "stabilityai/stable-diffusion-3.5-medium"
    device: torch.device | str = "mps"
    dtype: torch.dtype = torch.float32
    loader: Callable[[str, torch.device, torch.dtype], Any] = field(default=_default_loader)
    _pipeline: Any = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> Any:
        if self._pipeline is None:
            dev = torch.device(self.device) if isinstance(self.device, str) else self.device
            logger.info("Loading SD3.5 pipeline %s on %s", self.model_id, dev)
            self._pipeline = self.loader(self.model_id, dev, self.dtype)
        return self._pipeline

    @property
    def pipeline(self) -> Any:
        return self._ensure_loaded()

    # ------------------------------------------------------------------
    # Schedules (rectified flow)
    # ------------------------------------------------------------------

    @staticmethod
    def alpha_schedule(t: torch.Tensor) -> torch.Tensor:
        """``α_t = 1 - t`` for rectified flow."""
        return 1.0 - t

    @staticmethod
    def sigma_schedule(t: torch.Tensor) -> torch.Tensor:
        """``σ_t = t`` for rectified flow."""
        return t

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------

    def encode_prompt(self, text: str) -> Any:
        """Encode a text prompt to SD3.5 conditioning tensors.

        Returns whatever the pipeline's ``encode_prompt`` returns —
        SD3.5 uses three text encoders so this is typically a tuple of
        ``(prompt_embeds, negative_prompt_embeds, pooled, negative_pooled)``.
        """
        pipeline = self._ensure_loaded()
        if not hasattr(pipeline, "encode_prompt"):
            raise NotImplementedError(
                f"pipeline {type(pipeline).__name__} has no encode_prompt method"
            )
        return pipeline.encode_prompt(prompt=text)

    # ------------------------------------------------------------------
    # Score function
    # ------------------------------------------------------------------

    def score_fn(self, x_t: torch.Tensor, t: torch.Tensor, conditioning: Any) -> torch.Tensor:
        """Predict the additive-noise ``ε`` at time ``t``.

        Args:
            x_t: Noised latent of shape ``[B, C, H, W]``.
            t: Timesteps in ``[0, 1]`` of shape ``[B]``.
            conditioning: Output of :meth:`encode_prompt` (typically a
                tuple of conditioning tensors).

        Returns:
            ``ε_pred`` of the same shape as ``x_t``. SD3.5's transformer
            predicts the velocity ``v``; this method converts it via
            ``ε = x_t + (1 - t) * v`` (see module docstring).
        """
        pipeline = self._ensure_loaded()
        transformer = self._get_transformer(pipeline)
        prompt_embeds, pooled = self._unpack_conditioning(conditioning)

        # SD3.5's MMDiT consumes timesteps in either [0, 1000] or as a
        # raw normalised float — adapt to either. The standard
        # diffusers contract is *integer* timesteps scaled to 1000; we
        # follow that.
        t_scaled = t * 1000.0
        with torch.no_grad():
            v_pred = transformer(
                hidden_states=x_t,
                timestep=t_scaled,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled,
                return_dict=False,
            )[0]

        # Broadcast (1 - t) over spatial dims.
        coeff = (1.0 - t).view(-1, *([1] * (x_t.dim() - 1)))
        return x_t + coeff * v_pred

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_transformer(pipeline: Any) -> Any:
        """Return the MMDiT transformer attribute (across diffusers versions).

        diffusers 0.30+ exposes it as ``pipeline.transformer``. Stubs
        used in tests may attach the callable differently; we fall
        back to the pipeline itself if no transformer attribute is
        present (useful for tiny stub pipelines).
        """
        if hasattr(pipeline, "transformer"):
            return pipeline.transformer
        return pipeline

    @staticmethod
    def _unpack_conditioning(conditioning: Any) -> tuple[Any, Any]:
        """Extract ``(prompt_embeds, pooled)`` from various return shapes.

        SD3.5's ``encode_prompt`` returns a 4-tuple
        ``(prompt_embeds, negative_prompt_embeds, pooled, negative_pooled)``.
        Tests sometimes pass a 2-tuple ``(prompt_embeds, pooled)`` for
        brevity. Single-tensor conditioning is treated as
        ``(emb, None)``.
        """
        if isinstance(conditioning, tuple):
            if len(conditioning) >= 3:
                return conditioning[0], conditioning[2]
            if len(conditioning) == 2:
                return conditioning[0], conditioning[1]
            if len(conditioning) == 1:
                return conditioning[0], None
        return conditioning, None


def build_vsd_phi_adapter(base: SD35DiffusionAdapter) -> SD35DiffusionAdapter:
    """Return a LoRA-augmented adapter sharing weights with ``base``.

    The intended use: ``base`` is the frozen prior; the returned adapter
    is the *trainable φ model* that VSD updates. In the current
    skeleton implementation the LoRA injection is a TODO — the adapter
    is structurally identical, so VSD can still be exercised against
    a clean SD3.5 model. Wiring the real LoRA layers is the next
    follow-up; see comments inline.
    """
    # TODO: insert LoRA adapters into base.pipeline.transformer.
    # The intended shape is:
    #   1. Deep-copy or share weights with base.
    #   2. Inject diffusers.LoraConfig over the transformer's
    #      Q/K/V/Out projections at a small rank (e.g. r=4).
    #   3. Mark only the LoRA parameters as trainable.
    # For now we return a clean copy of base; VSDLoss will still run,
    # but the variational refinement is the identity until LoRA is
    # plumbed in.
    return SD35DiffusionAdapter(
        model_id=base.model_id,
        device=base.device,
        dtype=base.dtype,
        loader=base.loader,
    )
