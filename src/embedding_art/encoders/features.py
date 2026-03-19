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


class CNNStatisticsExtractor:
    """Statistics extractor for CNN-based encoders.

    CNN intermediate layers produce feature maps ``[B, C, H, W]``.
    Computes per-channel mean and std by reducing over the spatial dimensions
    (height and width), yielding one scalar per channel.

    For non-4D inputs the extractor falls back to a flat mean/std over the
    merged non-batch dimensions, matching the behaviour of
    ``ViTStatisticsExtractor`` for non-standard shapes.

    Typical CNN layer output shapes and the resulting statistic shapes:

    * ``[1, 512, 14, 14]`` → mean/std shape ``[512]``
    * ``[1, 2048, 1, 1]`` → mean/std shape ``[2048]``  (after global pooling)
    """

    def extract(self, features: LayerFeatures) -> FeatureStatistics:
        """Compute per-channel mean and std over spatial dimensions.

        Args:
            features: Layer activations.  The ``tensor`` field is expected to
                have shape ``[B, C, H, W]`` for the standard spatial path.

        Returns:
            ``FeatureStatistics`` with ``mean`` and ``std`` of shape ``[C]``
            for 4-D inputs, or a scalar for other shapes.
        """
        tensor = features.tensor
        if tensor.dim() == 4:
            # [B, C, H, W] -> mean/std per channel over spatial dims
            mean = tensor.mean(dim=(2, 3)).squeeze(0)  # [C]
            std = tensor.std(dim=(2, 3)).squeeze(0)  # [C]
        else:
            mean = tensor.flatten(start_dim=1).mean(dim=1).squeeze(0)
            std = tensor.flatten(start_dim=1).std(dim=1).squeeze(0)
        return FeatureStatistics(mean=mean, std=std)


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
