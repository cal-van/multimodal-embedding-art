import pytest
import torch

from embedding_art.regularizers.base import CompositeRegularizer
from embedding_art.regularizers.video import TemporalCoherence


def test_temporal_coherence_returns_zero_without_decoded() -> None:
    latent = torch.randn(1, 4, 4, 8, 8)
    regularizer = TemporalCoherence(weight=1.0)

    loss = regularizer(latent, decoded=None)

    assert loss.item() == 0.0


def test_temporal_coherence_zero_for_constant_frames() -> None:
    latent = torch.randn(1, 4, 4, 8, 8)
    decoded = torch.zeros(1, 4, 3, 8, 8)
    regularizer = TemporalCoherence(weight=1.0)

    loss = regularizer(latent, decoded=decoded)

    assert loss.item() == 0.0


def test_temporal_coherence_penalizes_flicker() -> None:
    latent = torch.randn(1, 4, 4, 8, 8)
    smooth = torch.zeros(1, 4, 3, 8, 8)
    flicker = smooth.clone()
    flicker[:, 1::2] = 1.0
    regularizer = TemporalCoherence(weight=1.0)

    smooth_loss = regularizer(latent, decoded=smooth)
    flicker_loss = regularizer(latent, decoded=flicker)

    assert flicker_loss.item() > smooth_loss.item()


def test_temporal_coherence_weight_scales_loss() -> None:
    latent = torch.randn(1, 4, 4, 8, 8)
    decoded = torch.zeros(1, 4, 3, 8, 8)
    decoded[:, 1::2] = 1.0

    low_weight = TemporalCoherence(weight=0.5)
    high_weight = TemporalCoherence(weight=2.0)

    loss_low = low_weight(latent, decoded=decoded)
    loss_high = high_weight(latent, decoded=decoded)

    expected_ratio = 2.0 / 0.5
    actual_ratio = loss_high.item() / loss_low.item()

    assert pytest.approx(actual_ratio, rel=1e-6) == expected_ratio


def test_default_video_factory_contains_temporal_coherence() -> None:
    reg = CompositeRegularizer.default_video()
    has_temporal = any(isinstance(r, TemporalCoherence) for r in reg.regularizers)

    assert has_temporal
