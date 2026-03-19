"""
Base encoder protocol.

Encoders map various modalities into a shared embedding space.

v2 duck-typed extensions
------------------------
The following methods are part of the v2 encoder interface but are intentionally
*not* declared in the ``Encoder`` Protocol.  This preserves backward compatibility
— existing encoders that implement only the v1 methods still satisfy the protocol.
Callers that need v2 behaviour check for the methods with ``hasattr``.

* ``card -> EncoderCard`` — metadata describing capabilities and memory footprint.
* ``encode(spec: ConceptSpec) -> Concept`` — single-method dispatch from a spec.
* ``encode_for_optimization(tensor: Tensor) -> Tensor`` — differentiable encoding.
* ``unload() -> None`` — release model weights from memory.
"""

from pathlib import Path
from typing import Protocol

import torch
from PIL import Image


class Encoder(Protocol):
    """Protocol for multimodal encoders.

    All four modality methods (encode_text, encode_image, encode_audio,
    encode_video) are required for v1 compliance.  See the module docstring for
    the v2 duck-typed extensions.
    """

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
