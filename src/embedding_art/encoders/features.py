"""
Layer feature extraction and statistics for multi-layer feature matching.

Different encoder architectures produce different intermediate representations.
FeatureStatisticsExtractor handles the per-architecture differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@dataclass
class LayerFeatures:
    """Typed container for intermediate encoder layer activations."""

    tensor: torch.Tensor
    spatial: bool
    shape_semantic: str  # "batch_tokens_dim" | "batch_channels_height_width"
    layer_name: str


@dataclass
class FeatureStatistics:
    """Summary statistics for a layer's features."""

    mean: torch.Tensor
    std: torch.Tensor


@runtime_checkable
class FeatureStatisticsExtractor(Protocol):
    """Protocol for extracting statistics from layer features."""

    def extract(self, features: LayerFeatures) -> FeatureStatistics: ...


class ViTStatisticsExtractor:
    """
    Statistics extractor for Vision Transformer encoders.

    ViT intermediate layers produce patch token sequences [B, N_patches, D].
    Computes per-feature-dimension statistics across patches.

    For 3D inputs [B, N_patches, D]:
        - mean and std are computed over the N_patches dimension, yielding [D].

    For other dimensionalities (2D, 4D spatial maps, etc.):
        - The tensor is flattened to [B, -1] and statistics are computed over
          that single merged dimension, yielding a scalar [] after squeeze.
    """

    def extract(self, features: LayerFeatures) -> FeatureStatistics:
        tensor = features.tensor
        if tensor.dim() == 3:
            # [B, N_patches, D] -> mean/std over patches dimension
            mean = tensor.mean(dim=1).squeeze(0)  # [D]
            std = tensor.std(dim=1).squeeze(0)  # [D]
        else:
            mean = tensor.flatten(start_dim=1).mean(dim=1).squeeze(0)
            std = tensor.flatten(start_dim=1).std(dim=1).squeeze(0)
        return FeatureStatistics(mean=mean, std=std)
