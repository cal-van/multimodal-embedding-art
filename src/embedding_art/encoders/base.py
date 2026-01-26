"""
Base encoder protocol.

Encoders map various modalities into a shared embedding space.
"""

from pathlib import Path
from typing import Protocol

import torch
from PIL import Image


class Encoder(Protocol):
    """Protocol for multimodal encoders."""

    @property
    def embedding_dim(self) -> int:
        """Dimension of the embedding space."""
        ...

    @property
    def device(self) -> torch.device:
        """Device the encoder is on."""
        ...

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to embedding. Returns [1, embed_dim] tensor."""
        ...

    def encode_image(self, image: Path | Image.Image | torch.Tensor) -> torch.Tensor:
        """Encode image to embedding. Returns [1, embed_dim] tensor."""
        ...

    def encode_audio(
        self,
        audio: Path | torch.Tensor,
        start: float = 0.0,
        duration: float = 2.0,
    ) -> torch.Tensor:
        """Encode audio to embedding. Returns [1, embed_dim] tensor."""
        ...

    def encode_video(
        self,
        video: Path | torch.Tensor,
        timestamp: float = 0.0,
    ) -> torch.Tensor:
        """Encode video frame to embedding. Returns [1, embed_dim] tensor."""
        ...
