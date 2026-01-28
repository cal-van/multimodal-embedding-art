import torch

from embedding_art.regularizers.audio import AudioTotalVariation
from embedding_art.regularizers.base import CompositeRegularizer


def test_audio_total_variation_shapes():
    """Test that AudioTV handles shapes correctly."""
    # latent shape: [B, C, H, W]
    # AudioLDM latent: [1, 8, 16, 64]
    latent = torch.randn(1, 8, 16, 64)
    # Decoded mel: [1, 1, 64, 256] (1 channel, 64 freq bins, 256 time steps)
    decoded = torch.randn(1, 1, 64, 256)

    tv = AudioTotalVariation(freq_weight=1.0, time_weight=1.0)
    loss = tv(latent, decoded)

    assert loss.ndim == 0
    assert loss > 0


def test_audio_total_variation_anisotropy():
    """Test that frequency and time weights are applied separately."""
    # Create a signal that changes only in frequency (vertical stripes)
    # Shape: [1, 1, 4, 4]
    # Time is constant (dim 3), Freq changes (dim 2)
    # Create a signal that changes only in frequency (vertical stripes)
    # Shape: [1, 1, 4, 4] -> Use as latent [1, 1, 4, 4] (channels=1 is fine for logic)
    latent_signal = torch.tensor(
        [[[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]]]
    )

    # CASE 1: Only penalize frequency changes
    tv_freq = AudioTotalVariation(freq_weight=1.0, time_weight=0.0)
    # Pass signal as LATENT (1st arg)
    loss_freq = tv_freq(latent_signal)
    assert loss_freq > 0

    # CASE 2: Only penalize time changes
    tv_time = AudioTotalVariation(freq_weight=0.0, time_weight=1.0)
    loss_time = tv_time(latent_signal)
    # Should be 0 because the signal is constant across time dimensions
    assert loss_time == 0


def test_default_audio_factory():
    """Test that the default_audio factory returns a CompositeRegularizer."""
    reg = CompositeRegularizer.default_audio()
    assert isinstance(reg, CompositeRegularizer)
    # Check if it contains AudioTotalVariation
    has_audio_tv = any(isinstance(r, AudioTotalVariation) for r in reg.regularizers)
    assert has_audio_tv
