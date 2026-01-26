"""
ImageBind encoder wrapper.

ImageBind provides a unified embedding space for text, image, audio, video, depth, thermal, and IMU.
We use it as our primary encoder for multimodal concept representation.

Requires: pip install -e . from https://github.com/facebookresearch/ImageBind
"""

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

# ImageBind imports - will fail if not installed
try:
    from imagebind import data as imagebind_data
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType

    IMAGEBIND_AVAILABLE = True
except ImportError:
    IMAGEBIND_AVAILABLE = False
    imagebind_data = None
    imagebind_model = None
    ModalityType = None


class ImageBindEncoder:
    """
    Wrapper around Meta's ImageBind model.

    ImageBind encodes multiple modalities into a shared 1024-dimensional space,
    allowing cross-modal operations like text→image optimization.
    """

    EMBEDDING_DIM = 1024
    IMAGE_SIZE = 224
    AUDIO_DURATION = 2.0  # ImageBind expects 2-second clips
    AUDIO_SAMPLE_RATE = 16000

    def __init__(
        self,
        device: str = "mps",
        pretrained: bool = True,
    ):
        if not IMAGEBIND_AVAILABLE:
            raise ImportError(
                "ImageBind is not installed. Please install it:\n"
                "  git clone https://github.com/facebookresearch/ImageBind\n"
                "  cd ImageBind && pip install -e ."
            )

        self._device = torch.device(device)
        self.model = imagebind_model.imagebind_huge(pretrained=pretrained)
        self.model.eval()
        self.model.to(self._device)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    @property
    def embedding_dim(self) -> int:
        return self.EMBEDDING_DIM

    @property
    def device(self) -> torch.device:
        return self._device

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to embedding."""
        inputs = {
            ModalityType.TEXT: imagebind_data.load_and_transform_text([text], self._device)
        }

        with torch.no_grad():
            embeddings = self.model(inputs)

        return F.normalize(embeddings[ModalityType.TEXT], dim=-1)

    def encode_image(self, image: Path | Image.Image | torch.Tensor) -> torch.Tensor:
        """Encode image to embedding."""
        if isinstance(image, torch.Tensor):
            # Already a tensor - use encode_for_optimization
            return self.encode_for_optimization(image)

        if isinstance(image, Image.Image):
            # Save to temp file for ImageBind's loader
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                image.save(f.name)
                image_path = f.name
        else:
            image_path = str(image)

        inputs = {
            ModalityType.VISION: imagebind_data.load_and_transform_vision_data(
                [image_path], self._device
            )
        }

        with torch.no_grad():
            embeddings = self.model(inputs)

        return F.normalize(embeddings[ModalityType.VISION], dim=-1)

    def encode_for_optimization(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode image tensor during optimization.

        This method is differentiable and works with tensors that have gradients.
        The input should be [B, C, H, W] with values in [0, 1].
        """
        # Resize to ImageBind's expected size
        if image_tensor.shape[-2:] != (self.IMAGE_SIZE, self.IMAGE_SIZE):
            image_tensor = F.interpolate(
                image_tensor,
                size=(self.IMAGE_SIZE, self.IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
            )

        # Normalize with ImageBind's normalization
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=image_tensor.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=image_tensor.device)
        image_tensor = (image_tensor - mean[None, :, None, None]) / std[None, :, None, None]

        # Get embedding through the vision encoder
        embeddings = self.model({ModalityType.VISION: image_tensor})

        return F.normalize(embeddings[ModalityType.VISION], dim=-1)

    def encode_audio(
        self,
        audio: Path | torch.Tensor,
        start: float = 0.0,
        duration: float = 2.0,
    ) -> torch.Tensor:
        """Encode audio to embedding."""
        if isinstance(audio, torch.Tensor):
            # TODO: Handle tensor audio during optimization
            raise NotImplementedError("Tensor audio encoding not yet implemented")

        audio_path = str(audio)

        inputs = {
            ModalityType.AUDIO: imagebind_data.load_and_transform_audio_data(
                [audio_path], self._device
            )
        }

        with torch.no_grad():
            embeddings = self.model(inputs)

        return F.normalize(embeddings[ModalityType.AUDIO], dim=-1)

    def encode_video(
        self,
        video: Path | torch.Tensor,
        timestamp: float = 0.0,
    ) -> torch.Tensor:
        """Encode video frame to embedding."""
        if isinstance(video, torch.Tensor):
            # TODO: Handle tensor video during optimization
            raise NotImplementedError("Tensor video encoding not yet implemented")

        video_path = str(video)

        # ImageBind's video loader samples 2 frames by default
        inputs = {
            ModalityType.VISION: imagebind_data.load_and_transform_video_data(
                [video_path], self._device
            )
        }

        with torch.no_grad():
            embeddings = self.model(inputs)

        return F.normalize(embeddings[ModalityType.VISION], dim=-1)
