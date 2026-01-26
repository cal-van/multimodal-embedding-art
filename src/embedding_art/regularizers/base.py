"""
Regularization losses for optimization stability.

Without regularization, optimization produces adversarial noise that maximizes
similarity but looks nothing like the concept. Regularizers encourage coherent outputs.
"""

from dataclasses import dataclass, field
from typing import Protocol

import torch
import torch.nn.functional as F


class Regularizer(Protocol):
    """Protocol for regularization losses."""

    weight: float

    def __call__(self, latent: torch.Tensor, decoded: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute regularization loss.

        Args:
            latent: The latent being optimized
            decoded: The decoded output (if needed by regularizer)

        Returns:
            Scalar loss tensor
        """
        ...


@dataclass
class TotalVariation:
    """
    Total variation loss for spatial smoothness.

    Penalizes high-frequency changes in adjacent pixels, encouraging
    smoother outputs without excessive noise.
    """

    weight: float = 0.01

    def __call__(self, latent: torch.Tensor, decoded: torch.Tensor | None = None) -> torch.Tensor:
        if decoded is None:
            return torch.tensor(0.0, device=latent.device)

        # Compute differences along height and width
        diff_h = decoded[:, :, 1:, :] - decoded[:, :, :-1, :]
        diff_w = decoded[:, :, :, 1:] - decoded[:, :, :, :-1]

        # L1 total variation
        tv_loss = diff_h.abs().mean() + diff_w.abs().mean()

        return self.weight * tv_loss


@dataclass
class SpectralRegularizer:
    """
    Spectral regularizer to penalize high-frequency noise.

    Uses FFT to penalize energy in high-frequency components,
    which typically correspond to adversarial noise patterns.
    """

    weight: float = 0.001
    high_freq_threshold: float = 0.5  # Fraction of spectrum considered "high frequency"

    def __call__(self, latent: torch.Tensor, decoded: torch.Tensor | None = None) -> torch.Tensor:
        if decoded is None:
            return torch.tensor(0.0, device=latent.device)

        # Convert to grayscale if needed
        if decoded.shape[1] == 3:
            gray = 0.299 * decoded[:, 0] + 0.587 * decoded[:, 1] + 0.114 * decoded[:, 2]
        else:
            gray = decoded[:, 0]

        # Compute 2D FFT
        fft = torch.fft.fft2(gray)
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.abs(fft_shift)

        # Create high-frequency mask
        h, w = magnitude.shape[-2:]
        cy, cx = h // 2, w // 2
        y = torch.arange(h, device=decoded.device) - cy
        x = torch.arange(w, device=decoded.device) - cx
        Y, X = torch.meshgrid(y, x, indexing="ij")
        dist = torch.sqrt(X.float() ** 2 + Y.float() ** 2)
        max_dist = min(cy, cx)
        high_freq_mask = dist > (self.high_freq_threshold * max_dist)

        # Penalize high-frequency energy
        high_freq_energy = (magnitude * high_freq_mask).mean()

        return self.weight * high_freq_energy


@dataclass
class LatentNorm:
    """
    Latent norm regularizer to keep latent in typical distribution.

    VAE latents are typically Gaussian N(0, 1). This regularizer
    penalizes latents that drift too far from this distribution.
    """

    weight: float = 0.1
    target_std: float = 1.0

    def __call__(self, latent: torch.Tensor, decoded: torch.Tensor | None = None) -> torch.Tensor:
        # Penalize deviation from target standard deviation
        latent_std = latent.std()
        std_loss = (latent_std - self.target_std).abs()

        # Penalize mean deviation from 0
        mean_loss = latent.mean().abs()

        return self.weight * (std_loss + mean_loss)


@dataclass
class CompositeRegularizer:
    """
    Combines multiple regularizers into one.

    Provides preset configurations for common use cases.
    """

    regularizers: list[Regularizer] = field(default_factory=list)

    def __call__(self, latent: torch.Tensor, decoded: torch.Tensor | None = None) -> torch.Tensor:
        if not self.regularizers:
            return torch.tensor(0.0, device=latent.device)

        total = sum(reg(latent, decoded) for reg in self.regularizers)
        return total

    @classmethod
    def default_image(cls) -> "CompositeRegularizer":
        """Default regularization for image optimization."""
        return cls(
            regularizers=[
                TotalVariation(weight=0.01),
                SpectralRegularizer(weight=0.001),
                LatentNorm(weight=0.1),
            ]
        )

    @classmethod
    def minimal(cls) -> "CompositeRegularizer":
        """Minimal regularization - more raw/adversarial aesthetics."""
        return cls(
            regularizers=[
                LatentNorm(weight=0.01),
            ]
        )

    @classmethod
    def heavy(cls) -> "CompositeRegularizer":
        """Heavy regularization - more coherent/smooth outputs."""
        return cls(
            regularizers=[
                TotalVariation(weight=0.1),
                SpectralRegularizer(weight=0.01),
                LatentNorm(weight=0.5),
            ]
        )
