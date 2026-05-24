"""IP-Adapter renderer for embedding-conditioned SDXL diffusion."""

from __future__ import annotations

from typing import Any

import torch

from embedding_art.core.render_result import OptimizationHistory, RenderResult


class IPAdapterRenderer:
    """Uses IP-Adapter to condition SDXL diffusion on an embedding.

    The diffusers pipeline is lazy-loaded on first render to avoid
    importing heavy dependencies at module level.
    """

    def __init__(
        self,
        model_id: str = "h94/IP-Adapter",
        sdxl_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str = "cpu",
    ) -> None:
        self._model_id = model_id
        self._sdxl_id = sdxl_id
        self._device = torch.device(device)
        self._pipeline = None

    @property
    def output_modality(self) -> str:
        return "image"

    def _load_pipeline(self) -> None:
        try:
            from diffusers import StableDiffusionXLPipeline
        except ImportError:
            raise ImportError(
                "IPAdapterRenderer requires the 'diffusers' package. "
                "Install it with: pip install diffusers transformers accelerate"
            )
        self._pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._sdxl_id, torch_dtype=torch.float16
        ).to(self._device)
        self._pipeline.load_ip_adapter(self._model_id, subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")

    def render(
        self,
        embedding: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        **kwargs: Any,
    ) -> RenderResult:
        if self._pipeline is None:
            self._load_pipeline()
        embedding = embedding.to(self._device)
        result = self._pipeline(
            prompt="",
            ip_adapter_image_embeds=[embedding],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        import torchvision.transforms.functional as TF

        output = TF.to_tensor(result.images[0]).unsqueeze(0)
        history = OptimizationHistory()
        return RenderResult(
            output=output,
            history=history,
            encoder_name="ip_adapter",
            final_similarity=0.0,
        )

    def unload(self) -> None:
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
