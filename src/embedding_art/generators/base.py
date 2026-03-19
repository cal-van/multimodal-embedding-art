"""
Base generator protocol.

Generators decode latent vectors into outputs (images, audio, video).

Three protocol variants are provided:

* ``Generator`` — the original v1 protocol (kept for backward compatibility).
* ``LatentGenerator`` — v2 superset of Generator; same interface, new name.
* ``DirectGenerator`` — directly optimized parameters (INR, pixel-space, etc.).
* ``DiffusionGenerator`` — latent generator extended with embedding-guided diffusion.
"""

from typing import Any, Protocol

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


class LatentGenerator(Protocol):
    """Generator that decodes from a latent space.

    Superset of ``Generator`` for v2.  The interface is identical; the new
    name makes intent explicit in v2 code that explicitly distinguishes
    latent-space generators from direct-parameter and diffusion generators.
    """

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
        """Initialize a random latent for optimization."""
        ...

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to output tensor."""
        ...


class DirectGenerator(Protocol):
    """Generator where the output parameters are directly optimized (INR, pixel).

    Instead of decoding a latent, the generator holds optimizable
    ``torch.nn.Parameter`` tensors that are updated directly by the optimizer.
    ``render()`` produces the output from those parameters.
    """

    @property
    def output_modality(self) -> str:
        """Output modality: 'image', 'audio', or 'video'."""
        ...

    @property
    def device(self) -> torch.device:
        """Device the generator is on."""
        ...

    def get_optimizable_parameters(self) -> list[torch.nn.Parameter]:
        """Return all parameters to be passed to the optimizer."""
        ...

    def render(self) -> torch.Tensor:
        """Produce the current output from internal parameters."""
        ...


class DiffusionGenerator(Protocol):
    """Generator with embedding-guided diffusion denoising.

    Extends the latent-generator interface with ``generate_guided``, which
    runs a full diffusion denoising pass steered by a target embedding.
    """

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
        """Initialize a random latent for optimization."""
        ...

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to output tensor."""
        ...

    def generate_guided(
        self,
        target_embedding: torch.Tensor,
        encoder: Any,
        loss_fn: Any,
        config: Any,
    ) -> Any:
        """Run embedding-guided diffusion denoising toward *target_embedding*.

        Args:
            target_embedding: The embedding to optimize toward.
            encoder: Encoder used to re-encode intermediate outputs.
            loss_fn: Loss function for guidance signal.
            config: Optimization / diffusion configuration.

        Returns:
            Implementation-defined result (typically a ``RenderResult``).
        """
        ...
