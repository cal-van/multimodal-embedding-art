"""Regularization losses for optimization stability."""

from embedding_art.regularizers.base import (
    CompositeRegularizer,
    LatentNorm,
    Regularizer,
    SpectralRegularizer,
    TotalVariation,
)

__all__ = [
    "Regularizer",
    "TotalVariation",
    "SpectralRegularizer",
    "LatentNorm",
    "CompositeRegularizer",
]
