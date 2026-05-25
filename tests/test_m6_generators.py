"""
Tests for the M6 generator upgrades: Stable Audio Open + LTX-Video.

Mocks the diffusers pipelines so the tests run without downloading the
real ~5 GB checkpoints.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch


def _make_mock_vae(latent_channels: int, sample_shape: tuple[int, ...]) -> MagicMock:
    """Build a fake VAE whose decode/encode return tensors of the right shape."""
    vae = MagicMock()

    def fake_decode(latent: torch.Tensor):
        result = MagicMock()
        result.sample = torch.rand(sample_shape)
        return result

    def fake_encode(audio: torch.Tensor):
        result = MagicMock()
        latent_dist = MagicMock()
        latent_dist.mode.return_value = torch.randn(
            (audio.shape[0], latent_channels, *audio.shape[2:])
        )
        result.latent_dist = latent_dist
        return result

    vae.decode = fake_decode
    vae.encode = fake_encode
    vae.parameters.return_value = []
    vae.to.return_value = vae
    vae.eval.return_value = vae
    return vae


class TestStableAudioOpenGenerator:
    """Stable Audio Open VAE-only generator basics."""

    def test_latent_shape_for_default_audio_length(self) -> None:
        from embedding_art.generators.stable_audio_open import (
            StableAudioOpenGenerator,
        )

        with patch("diffusers.StableAudioPipeline") as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.vae = _make_mock_vae(latent_channels=64, sample_shape=(1, 1, 441000))
            mock_pipeline_cls.from_pretrained.return_value = mock_pipeline

            gen = StableAudioOpenGenerator(device="cpu", audio_length_in_s=10.0)
            shape = gen.latent_shape
            assert shape[0] == 1
            assert shape[1] == 64  # LATENT_CHANNELS
            assert shape[2] == 1
            # T_latent ≈ (10 * 44100) / 2048 ≈ 215.
            assert 200 <= shape[3] <= 230

    def test_init_latent_returns_correct_shape(self) -> None:
        from embedding_art.generators.stable_audio_open import (
            StableAudioOpenGenerator,
        )

        with patch("diffusers.StableAudioPipeline") as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.vae = _make_mock_vae(64, (1, 1, 441000))
            mock_pipeline_cls.from_pretrained.return_value = mock_pipeline

            gen = StableAudioOpenGenerator(device="cpu", audio_length_in_s=5.0)
            latent = gen.init_latent(seed=42)
            assert latent.shape == gen.latent_shape
            assert latent.requires_grad

    def test_decode_returns_waveform(self) -> None:
        from embedding_art.generators.stable_audio_open import (
            StableAudioOpenGenerator,
        )

        with patch("diffusers.StableAudioPipeline") as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.vae = _make_mock_vae(64, (1, 1, 220500))
            mock_pipeline_cls.from_pretrained.return_value = mock_pipeline

            gen = StableAudioOpenGenerator(device="cpu", audio_length_in_s=5.0)
            latent = gen.init_latent(seed=42)
            audio = gen.decode(latent)
            assert audio.shape[0] == 1
            assert audio.shape[1] == 1  # mono

    def test_invalid_audio_length_rejected(self) -> None:
        from embedding_art.generators.stable_audio_open import (
            StableAudioOpenGenerator,
        )

        with pytest.raises(ValueError):
            StableAudioOpenGenerator(device="cpu", audio_length_in_s=0.0)

        with pytest.raises(ValueError):
            StableAudioOpenGenerator(device="cpu", audio_length_in_s=100.0)

    def test_output_modality(self) -> None:
        from embedding_art.generators.stable_audio_open import (
            StableAudioOpenGenerator,
        )

        with patch("diffusers.StableAudioPipeline") as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.vae = _make_mock_vae(64, (1, 1, 441000))
            mock_pipeline_cls.from_pretrained.return_value = mock_pipeline

            gen = StableAudioOpenGenerator(device="cpu")
            assert gen.output_modality == "audio"
            assert gen.sample_rate == 44100


class TestLTXVideoGenerator:
    """LTX-Video VAE-only generator basics."""

    def test_latent_shape_for_default_dimensions(self) -> None:
        from embedding_art.generators.ltx_video import LTXVideoGenerator

        with patch("diffusers.LTXPipeline") as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.vae = _make_mock_vae(128, (1, 3, 121, 512, 768))
            mock_pipeline_cls.from_pretrained.return_value = mock_pipeline

            gen = LTXVideoGenerator(device="cpu", num_frames=121, height=512, width=768)
            shape = gen.latent_shape
            assert shape[0] == 1
            assert shape[1] == 128  # LATENT_CHANNELS
            # Spatial downsample is 8; 512/8 = 64, 768/8 = 96.
            assert shape[3] == 64
            assert shape[4] == 96

    def test_init_latent_returns_correct_shape(self) -> None:
        from embedding_art.generators.ltx_video import LTXVideoGenerator

        with patch("diffusers.LTXPipeline") as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.vae = _make_mock_vae(128, (1, 3, 33, 256, 384))
            mock_pipeline_cls.from_pretrained.return_value = mock_pipeline

            gen = LTXVideoGenerator(device="cpu", num_frames=33, height=256, width=384)
            latent = gen.init_latent(seed=42)
            assert latent.shape == gen.latent_shape
            assert latent.requires_grad

    def test_decode_normalises_to_unit_range(self) -> None:
        """Decode result should be in [0, 1] (we normalise from [-1, 1])."""
        from embedding_art.generators.ltx_video import LTXVideoGenerator

        # Make the VAE return a [-1, 1] tensor; decoded output should be [0, 1].
        vae = MagicMock()

        def fake_decode(latent):
            result = MagicMock()
            result.sample = torch.full((1, 3, 33, 256, 384), -1.0)
            return result

        vae.decode = fake_decode
        vae.parameters.return_value = []
        vae.to.return_value = vae
        vae.eval.return_value = vae

        with patch("diffusers.LTXPipeline") as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.vae = vae
            mock_pipeline_cls.from_pretrained.return_value = mock_pipeline

            gen = LTXVideoGenerator(device="cpu", num_frames=33, height=256, width=384)
            latent = gen.init_latent(seed=42)
            video = gen.decode(latent)
            # All values should be exactly 0 (normalised from -1).
            assert torch.allclose(video, torch.zeros_like(video))

    def test_invalid_dimensions_rejected(self) -> None:
        from embedding_art.generators.ltx_video import LTXVideoGenerator

        with pytest.raises(ValueError):
            LTXVideoGenerator(device="cpu", num_frames=0)
        with pytest.raises(ValueError):
            LTXVideoGenerator(device="cpu", height=-1)

    def test_output_modality(self) -> None:
        from embedding_art.generators.ltx_video import LTXVideoGenerator

        with patch("diffusers.LTXPipeline") as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.vae = _make_mock_vae(128, (1, 3, 121, 512, 768))
            mock_pipeline_cls.from_pretrained.return_value = mock_pipeline

            gen = LTXVideoGenerator(device="cpu")
            assert gen.output_modality == "video"
            assert gen.fps == 24
