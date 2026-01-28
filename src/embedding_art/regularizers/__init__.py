"""Regularization losses for optimization stability."""

from embedding_art.regularizers.base import (
    CompositeRegularizer,
    LatentNorm,
    Regularizer,
    SpectralRegularizer,
    TotalVariation,
)
from embedding_art.regularizers.video import TemporalCoherence

__all__ = [
    "Regularizer",
    "TotalVariation",
    "SpectralRegularizer",
    "LatentNorm",
    "CompositeRegularizer",
    "TemporalCoherence",
]
