"""
Audio-specific regularizers.

These regularizers are designed for spectrogram-based audio generation models
like AudioLDM. They operate primarily on the latent space which preserves
time-frequency structure.
"""

from dataclasses import dataclass

import torch


@dataclass
class AudioTotalVariation:
    """
    Anisotropic Total Variation for audio latents.

    Allows controlling smoothness separately for frequency and time dimensions.
    Audio often requires different regularization strength for:
    - Frequency (height): Smoothness avoids "scratchy" artifacts
    - Time (width): Smoothness affects transient preservation

    Operates on the latent tensor [B, C, H, W] where H is frequency-like
    and W is time-like.
    """

    freq_weight: float = 1.0
    time_weight: float = 1.0
    weight: float = 1.0  # Global scaling

    def __call__(self, latent: torch.Tensor, decoded: torch.Tensor | None = None) -> torch.Tensor:
        """
        Compute TV loss on the latent.

        Args:
            latent: Latent tensor [B, C, H, W]
            decoded: Ignored (optimization target is latent smoothness)
        """
        # Compute differences
        # Height is frequency (dim 2)
        diff_freq = latent[:, :, 1:, :] - latent[:, :, :-1, :]

        # Width is time (dim 3)
        diff_time = latent[:, :, :, 1:] - latent[:, :, :, :-1]

        # L1 norm of differences
        loss_freq = diff_freq.abs().mean()
        loss_time = diff_time.abs().mean()

        total_loss = (self.freq_weight * loss_freq) + (self.time_weight * loss_time)
        return self.weight * total_loss
