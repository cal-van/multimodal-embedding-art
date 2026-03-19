"""
ImageBind encoder wrapper.

ImageBind provides a unified embedding space for text, image, audio, video, depth, thermal, and IMU.
We use it as our primary encoder for multimodal concept representation.

Requires: pip install -e . from https://github.com/facebookresearch/ImageBind
"""

# ImageBind imports - will fail if not installed
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from embedding_art.core.concept import Concept
from embedding_art.core.concept_spec import ConceptSpec
from embedding_art.encoders.registry import EncoderCapability, EncoderCard
from embedding_art.exceptions import (
    EncoderError,
    ImageBindNotInstalledError,
    ModelLoadError,
    OutOfMemoryError,
)

logger = logging.getLogger(__name__)

try:
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType

    from imagebind import data as imagebind_data

    IMAGEBIND_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ImageBind import failed: {e}")
    # Don't print traceback unless debug logging is enabled
    logger.debug("ImageBind import failure traceback:", exc_info=True)

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
            raise ImageBindNotInstalledError()

        self._device = torch.device(device)

        try:
            self.model = imagebind_model.imagebind_huge(pretrained=pretrained)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise OutOfMemoryError(
                    operation="loading ImageBind model",
                    device=device,
                    original_error=e,
                ) from e
            raise ModelLoadError(model_name="ImageBind", original_error=e) from e
        except OSError as e:
            raise ModelLoadError(model_name="ImageBind", original_error=e) from e

        self.model.eval()

        try:
            self.model.to(self._device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise OutOfMemoryError(
                    operation="moving ImageBind model to device",
                    device=device,
                    original_error=e,
                ) from e
            raise

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
        try:
            inputs = {
                ModalityType.TEXT: imagebind_data.load_and_transform_text([text], self._device)
            }

            with torch.no_grad():
                embeddings = self.model(inputs)

            return F.normalize(embeddings[ModalityType.TEXT], dim=-1)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise OutOfMemoryError(
                    operation="encoding text",
                    device=str(self._device),
                    original_error=e,
                ) from e
            raise EncoderError(modality="text", original_error=e) from e

    def encode_image(self, image: Path | Image.Image | torch.Tensor) -> torch.Tensor:
        """Encode image to embedding."""
        try:
            if isinstance(image, torch.Tensor):
                # Already a tensor - use encode_for_optimization
                return self.encode_for_optimization(image)

            if isinstance(image, Image.Image):
                # Save to temp file for ImageBind's loader
                import os
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    image.save(f.name)
                    image_path = f.name
                # Schedule cleanup after we're done with the file
                try:
                    inputs = {
                        ModalityType.VISION: imagebind_data.load_and_transform_vision_data(
                            [image_path], self._device
                        )
                    }
                    with torch.no_grad():
                        embeddings = self.model(inputs)
                    return F.normalize(embeddings[ModalityType.VISION], dim=-1)
                finally:
                    os.unlink(image_path)
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
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise OutOfMemoryError(
                    operation="encoding image",
                    device=str(self._device),
                    original_error=e,
                ) from e
            raise EncoderError(modality="image", original_error=e) from e

    def encode_for_optimization(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode image tensor during optimization.

        This method is differentiable and works with tensors that have gradients.
        The input should be [B, C, H, W] with values in [0, 1].
        """
        try:
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
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise OutOfMemoryError(
                    operation="encoding image tensor",
                    device=str(image_tensor.device),
                    original_error=e,
                ) from e
            raise EncoderError(modality="image", original_error=e) from e

    def encode_audio(
        self,
        audio: Path | torch.Tensor,
        start: float = 0.0,
        duration: float = 2.0,
    ) -> torch.Tensor:
        """Encode audio to embedding."""
        try:
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
        except NotImplementedError:
            raise
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise OutOfMemoryError(
                    operation="encoding audio",
                    device=str(self._device),
                    original_error=e,
                ) from e
            raise EncoderError(modality="audio", original_error=e) from e

    def encode_video(
        self,
        video: Path | torch.Tensor,
        timestamp: float = 0.0,
    ) -> torch.Tensor:
        """Encode video frame to embedding."""
        try:
            if isinstance(video, torch.Tensor):
                if video.ndim == 4:
                    return self.encode_for_optimization(video)
                if video.ndim != 5:
                    raise ValueError("Video tensor must have shape [B, F, C, H, W]")

                batch_size, num_frames, channels, height, width = video.shape
                video_flat = video.view(batch_size * num_frames, channels, height, width)
                embeddings = self.encode_for_optimization(video_flat)
                embeddings = embeddings.view(batch_size, num_frames, -1).mean(dim=1)
                return F.normalize(embeddings, dim=-1)

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
        except NotImplementedError:
            raise
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise OutOfMemoryError(
                    operation="encoding video",
                    device=str(self._device),
                    original_error=e,
                ) from e
            raise EncoderError(modality="video", original_error=e) from e

    # ------------------------------------------------------------------
    # v2 duck-typed interface
    # ------------------------------------------------------------------

    @property
    def card(self) -> EncoderCard:
        """Return an EncoderCard describing ImageBind's capabilities."""
        return EncoderCard(
            name="imagebind",
            capabilities=(
                EncoderCapability.TEXT
                | EncoderCapability.IMAGE
                | EncoderCapability.AUDIO
                | EncoderCapability.VIDEO
                | EncoderCapability.DEPTH
                | EncoderCapability.BACKPROP_OPTIMIZABLE
            ),
            embedding_dim=self.embedding_dim,
            memory_estimate_mb=3000,
            backprop_cost=1.0,
        )

    def encode(self, spec: ConceptSpec) -> Concept:
        """Dispatch a ConceptSpec to the appropriate encode_* method.

        Priority order: text > image > audio > video.

        Args:
            spec: Concept specification.

        Returns:
            A Concept whose embedding is derived from the chosen modality.

        Raises:
            ValueError: If no modality field is set on *spec*.
        """
        if spec.text is not None:
            embedding = self.encode_text(spec.text)
            return Concept(embedding=embedding, description=f'text:"{spec.text}"')
        if spec.image is not None:
            embedding = self.encode_image(spec.image)
            return Concept(embedding=embedding, description=f"image:{spec.image.name}")
        if spec.audio is not None:
            embedding = self.encode_audio(spec.audio)
            return Concept(embedding=embedding, description=f"audio:{spec.audio.name}")
        if spec.video is not None:
            embedding = self.encode_video(spec.video)
            return Concept(embedding=embedding, description=f"video:{spec.video.name}")
        raise ValueError(
            "ConceptSpec has no modality field set — "
            "at least one of text/image/audio/video must be provided"
        )

    def unload(self) -> None:
        """Delete the model and clear the device cache to free memory."""
        import gc

        del self.model
        gc.collect()
        if self._device.type == "mps":
            torch.mps.empty_cache()
        elif self._device.type == "cuda":
            torch.cuda.empty_cache()
