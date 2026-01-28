"""
Video-specific regularizers.

These regularizers encourage temporal consistency in video outputs.
"""

from dataclasses import dataclass

import torch


@dataclass
class TemporalCoherence:
    """
    Penalize flicker by discouraging large frame-to-frame changes.

    Expects decoded video of shape [B, F, C, H, W].
    """

    weight: float = 0.05

    def __call__(self, latent: torch.Tensor, decoded: torch.Tensor | None = None) -> torch.Tensor:
        if decoded is None:
            return torch.tensor(0.0, device=latent.device)

        if decoded.ndim != 5:
            raise ValueError("TemporalCoherence expects decoded video with shape [B, F, C, H, W]")

        frame_diffs = decoded[:, 1:] - decoded[:, :-1]
        loss = frame_diffs.abs().mean()

        return self.weight * loss
