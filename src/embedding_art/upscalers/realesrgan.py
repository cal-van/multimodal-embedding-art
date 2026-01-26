"""
Real-ESRGAN image upscaler.

Uses the Real-ESRGAN model to upscale images while preserving detail
and reducing artifacts. Supports 2x and 4x upscaling.
"""

import torch
from PIL import Image

from embedding_art.exceptions import ModelLoadError, OutOfMemoryError, UpscalerError

# Supported scale factors and their corresponding model weights
SCALE_WEIGHTS = {
    2: "weights/RealESRGAN_x2.pth",
    4: "weights/RealESRGAN_x4.pth",
}


class RealESRGANUpscaler:
    """
    Image upscaler using Real-ESRGAN.

    Upscales images by 2x or 4x while preserving detail and reducing artifacts.
    Uses the py-real-esrgan package for the underlying model.

    Example:
        upscaler = RealESRGANUpscaler(device='mps', scale=4)
        upscaled = upscaler.upscale(image)  # PIL Image in, PIL Image out
    """

    def __init__(
        self,
        device: str = "mps",
        scale: int = 4,
    ) -> None:
        """
        Initialize the Real-ESRGAN upscaler.

        Args:
            device: Device to run the model on ('cpu', 'mps', or 'cuda')
            scale: Default upscaling factor (2 or 4)
        """
        if scale not in SCALE_WEIGHTS:
            raise ValueError(f"Scale must be one of {list(SCALE_WEIGHTS.keys())}, got {scale}")

        self._device = torch.device(device)
        self._scale = scale
        self._models: dict[int, object] = {}

        # Lazily load models - don't load until first use
        self._loaded = False

    @property
    def device(self) -> torch.device:
        """Device the upscaler is running on."""
        return self._device

    @property
    def scale(self) -> int:
        """Default upscaling factor."""
        return self._scale

    def _get_model(self, scale: int) -> object:
        """
        Get or lazily load the model for the given scale.

        Args:
            scale: Scale factor (2 or 4)

        Returns:
            RealESRGAN model instance
        """
        if scale not in self._models:
            try:
                from RealESRGAN import RealESRGAN

                model = RealESRGAN(self._device, scale=scale)
                model.load_weights(SCALE_WEIGHTS[scale], download=True)
                self._models[scale] = model
            except ImportError as e:
                raise ModelLoadError(
                    model_name="Real-ESRGAN",
                    original_error=ImportError(
                        "py-real-esrgan is not installed. "
                        "Install it with: pip install py-real-esrgan"
                    ),
                ) from e
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    raise OutOfMemoryError(
                        operation="loading Real-ESRGAN model",
                        device=str(self._device),
                        original_error=e,
                    ) from e
                raise ModelLoadError(
                    model_name="Real-ESRGAN",
                    original_error=e,
                ) from e

        return self._models[scale]

    def upscale(
        self,
        image: Image.Image,
        scale: int | None = None,
    ) -> Image.Image:
        """
        Upscale an image using Real-ESRGAN.

        Args:
            image: Input PIL Image (will be converted to RGB if necessary)
            scale: Optional scale override (2 or 4). Uses default if not specified.

        Returns:
            Upscaled PIL Image

        Raises:
            UpscalerError: If upscaling fails
            OutOfMemoryError: If GPU runs out of memory
        """
        if scale is None:
            scale = self._scale

        if scale not in SCALE_WEIGHTS:
            raise ValueError(f"Scale must be one of {list(SCALE_WEIGHTS.keys())}, got {scale}")

        # Ensure RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")

        try:
            model = self._get_model(scale)
            result = model.predict(image)
            return result
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise OutOfMemoryError(
                    operation="upscaling image",
                    device=str(self._device),
                    original_error=e,
                ) from e
            raise UpscalerError(
                operation="Real-ESRGAN upscaling",
                original_error=e,
            ) from e
        except Exception as e:
            raise UpscalerError(
                operation="Real-ESRGAN upscaling",
                original_error=e,
            ) from e
