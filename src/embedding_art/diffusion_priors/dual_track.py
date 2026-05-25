"""
Dual-track rendering (M5).

Combines the 'honest' track (embedding/feature alignment to the target)
with the 'natural' track (diffusion-prior loss via SDS or VSD) into a
single composite loss whose weighting is controlled by a configurable
balance parameter.

The honest track renders 'what the model thinks the concept looks like'
— it pushes the output toward the target embedding in canonical
LanguageBind space and trusts the encoder's geometry. The natural track
pushes toward 'a sample that looks like a natural image' — it uses the
diffusion model's score as a regulariser.

A balance ``alpha ∈ [0, 1]`` gives the canonical bracket:
* ``alpha = 1.0`` — pure honest (the v3 default for the interpretability
  artefact). Rendering reflects the model's representation; SD3.5's
  flow-matching latent keeps it sharp without forcing photorealism.
* ``alpha = 0.0`` — pure natural (recognisable, photorealistic-style
  output).
* ``alpha = 0.5`` — equal blend; produces a sample that's both
  semantically faithful and natural-looking.

The contrast between ``alpha = 1.0`` and ``alpha = 0.0`` for the same
target *is* the artefact: 'model-honest' next to 'natural' for the same
concept.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class DualTrackConfig:
    """Configuration for blending honest + natural losses."""

    alpha: float = 1.0
    """Blend weight. 1.0 = pure honest, 0.0 = pure natural."""

    honest_weight: float = 1.0
    """Scalar multiplier on the honest loss before blending."""

    natural_weight: float = 1.0
    """Scalar multiplier on the natural loss before blending."""

    def validate(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"DualTrackConfig.alpha must be in [0, 1]; got {self.alpha}")
        if self.honest_weight < 0:
            raise ValueError("honest_weight must be >= 0")
        if self.natural_weight < 0:
            raise ValueError("natural_weight must be >= 0")


def dual_track_loss(
    *,
    honest_loss_fn: Callable[[], torch.Tensor],
    natural_loss_fn: Callable[[], torch.Tensor] | None,
    config: DualTrackConfig,
) -> torch.Tensor:
    """Compute the blended dual-track loss.

    Args:
        honest_loss_fn: Zero-arg callable returning the scalar honest
            (embedding-alignment) loss. Called every step.
        natural_loss_fn: Zero-arg callable returning the scalar natural
            (diffusion-prior) loss. May be ``None`` when no diffusion
            prior is configured (e.g. ``alpha == 1.0`` and no SDS/VSD
            backbone loaded); ignored in that case.
        config: Dual-track configuration.

    Returns:
        Scalar blended loss::

            alpha * honest_weight * honest + (1 - alpha) * natural_weight * natural
    """
    config.validate()

    honest = config.honest_weight * honest_loss_fn()

    if config.alpha >= 1.0 or natural_loss_fn is None:
        return config.alpha * honest

    natural = config.natural_weight * natural_loss_fn()
    return config.alpha * honest + (1 - config.alpha) * natural


def make_dual_track_tracks(alpha_honest: float, alpha_natural: float) -> list[DualTrackConfig]:
    """Convenience factory: return the canonical (honest, natural) pair.

    Most showcase invocations want both tracks rendered for the same
    concept (so the user can compare honest-aliens to natural-images
    side-by-side). This helper returns the two configs to iterate over.

    Args:
        alpha_honest: Blend weight for the honest track. Typically 1.0.
        alpha_natural: Blend weight for the natural track. Typically 0.0
            for pure-VSD output or 0.3 for a mostly-natural-but-aware
            blend.

    Returns:
        ``[DualTrackConfig(alpha=alpha_honest),
           DualTrackConfig(alpha=alpha_natural)]``.
    """
    return [DualTrackConfig(alpha=alpha_honest), DualTrackConfig(alpha=alpha_natural)]


def _make_dummy_natural_loss() -> Any:
    """Internal helper used in tests where we want a no-op natural loss."""
    return lambda: torch.zeros(())
