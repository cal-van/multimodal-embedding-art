"""
Base generator protocol.

Generators decode latent vectors into outputs (images, audio, video).
"""

from typing import Protocol

import torch


class Generator(Protocol):
    """Protocol for output generators."""

    @property
    def latent_shape(self) -> tuple[int, ...]:
        """Shape of the latent tensor this generator accepts."""
        ...

    @property
    def output_modality(self) -> str:
        """Output modality: 'image', 'audio', or 'video'."""
        ...

    @property
    def device(self) -> torch.device:
        """Device the generator is on."""
        ...

    def init_latent(self, seed: int | None = None) -> torch.Tensor:
        """
        Initialize a random latent for optimization.

        Args:
            seed: Optional random seed for reproducibility

        Returns:
            Latent tensor with requires_grad=True
        """
        ...

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to output tensor.

        Args:
            latent: Latent tensor of shape self.latent_shape

        Returns:
            Decoded output tensor (e.g., [B, C, H, W] for images)
        """
        ...
