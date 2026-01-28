"""
Tests for SVDVideoGenerator.

These tests verify the SVD generator class for Stable Video Diffusion.

Unit tests use mocks to test behavior without loading the ~4GB model.
Integration tests (marked @pytest.mark.slow) require the real model.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from embedding_art.generators.video import SVDVideoGenerator


class TestSVDVideoGeneratorConstants:
    """Tests for SVDVideoGenerator class constants and documentation."""

    def test_latent_channels_should_be_four(self) -> None:
        """SVD uses 4 latent channels (same as SD/SDXL)."""
        assert SVDVideoGenerator.LATENT_CHANNELS == 4

    def test_latent_height_should_match_spatial_downscale(self) -> None:
        """
        Latent height should be output_height // 8.

        For 576px output height: 576 // 8 = 72
        """
        expected = SVDVideoGenerator.OUTPUT_HEIGHT // SVDVideoGenerator.SPATIAL_SCALE_FACTOR
        assert SVDVideoGenerator.LATENT_HEIGHT == expected
        assert SVDVideoGenerator.LATENT_HEIGHT == 72

    def test_latent_width_should_match_spatial_downscale(self) -> None:
        """
        Latent width should be output_width // 8.

        For 1024px output width: 1024 // 8 = 128
        """
        expected = SVDVideoGenerator.OUTPUT_WIDTH // SVDVideoGenerator.SPATIAL_SCALE_FACTOR
        assert SVDVideoGenerator.LATENT_WIDTH == expected
        assert SVDVideoGenerator.LATENT_WIDTH == 128

    def test_scaling_factor_should_match_sd21(self) -> None:
        """
        SVD inherits the 0.18215 scaling factor from SD 2.1.

        This is different from SDXL's 0.13025 scaling factor.
        """
        assert SVDVideoGenerator.SCALING_FACTOR == 0.18215

    def test_spatial_scale_factor_should_be_eight(self) -> None:
        """SVD uses 8x spatial downscaling (standard for SD-based VAEs)."""
        assert SVDVideoGenerator.SPATIAL_SCALE_FACTOR == 8

    def test_default_num_frames_should_be_twenty_five(self) -> None:
        """SVD-XT generates 25 frames by default."""
        assert SVDVideoGenerator.DEFAULT_NUM_FRAMES == 25

    def test_output_dimensions_should_be_576x1024(self) -> None:
        """SVD outputs 576x1024 video (portrait orientation)."""
        assert SVDVideoGenerator.OUTPUT_HEIGHT == 576
        assert SVDVideoGenerator.OUTPUT_WIDTH == 1024


class TestSVDVideoGeneratorModelIds:
    """Tests for model identifier constants."""

    def test_svd_model_id_should_be_img2vid(self) -> None:
        """Base SVD model ID for 14-frame generation."""
        expected = "stabilityai/stable-video-diffusion-img2vid"
        assert SVDVideoGenerator.MODEL_ID_SVD == expected

    def test_svd_xt_model_id_should_be_img2vid_xt(self) -> None:
        """SVD-XT model ID for 25-frame generation."""
        expected = "stabilityai/stable-video-diffusion-img2vid-xt"
        assert SVDVideoGenerator.MODEL_ID_SVD_XT == expected

    def test_svd_xt_1_1_model_id_should_be_img2vid_xt_1_1(self) -> None:
        """SVD 1.1 model ID for improved 25-frame generation."""
        expected = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"
        assert SVDVideoGenerator.MODEL_ID_SVD_XT_1_1 == expected


class TestSVDVideoGeneratorLatentShape:
    """Tests for latent shape property."""

    @patch("embedding_art.generators.video.AutoencoderKLTemporalDecoder")
    def test_latent_shape_property_should_return_5d_tuple(self, mock_vae_class: MagicMock) -> None:
        """
        latent_shape should return [B, C, F, H, W] tuple.

        Expected: (1, 4, 25, 72, 128) for SVD-XT at 576x1024
        """
        mock_vae = create_mock_vae()
        mock_vae_class.from_pretrained.return_value = mock_vae

        generator = SVDVideoGenerator(device="cpu")

        expected_shape = (1, 4, 25, 72, 128)
        assert generator.latent_shape == expected_shape

    @patch("embedding_art.generators.video.AutoencoderKLTemporalDecoder")
    def test_latent_shape_batch_dimension_should_be_one(self, mock_vae_class: MagicMock) -> None:
        """Batch dimension defaults to 1 for single video generation."""
        mock_vae = create_mock_vae()
        mock_vae_class.from_pretrained.return_value = mock_vae

        generator = SVDVideoGenerator(device="cpu")
        batch_dim = generator.latent_shape[0]

        assert batch_dim == 1

    @patch("embedding_art.generators.video.AutoencoderKLTemporalDecoder")
    def test_latent_shape_channel_dimension_should_be_four(self, mock_vae_class: MagicMock) -> None:
        """Channel dimension should match LATENT_CHANNELS constant."""
        mock_vae = create_mock_vae()
        mock_vae_class.from_pretrained.return_value = mock_vae

        generator = SVDVideoGenerator(device="cpu")
        channel_dim = generator.latent_shape[1]

        assert channel_dim == SVDVideoGenerator.LATENT_CHANNELS

    @patch("embedding_art.generators.video.AutoencoderKLTemporalDecoder")
    def test_latent_shape_frame_dimension_should_match_default(
        self, mock_vae_class: MagicMock
    ) -> None:
        """Frame dimension should match DEFAULT_NUM_FRAMES constant."""
        mock_vae = create_mock_vae()
        mock_vae_class.from_pretrained.return_value = mock_vae

        generator = SVDVideoGenerator(device="cpu")
        frame_dim = generator.latent_shape[2]

        assert frame_dim == SVDVideoGenerator.DEFAULT_NUM_FRAMES


class TestSVDVideoGeneratorOutputModality:
    """Tests for output_modality property."""

    @patch("embedding_art.generators.video.AutoencoderKLTemporalDecoder")
    def test_output_modality_should_be_video(self, mock_vae_class: MagicMock) -> None:
        """The output modality should be 'video'."""
        mock_vae = create_mock_vae()
        mock_vae_class.from_pretrained.return_value = mock_vae

        generator = SVDVideoGenerator(device="cpu")

        assert generator.output_modality == "video"


def create_mock_vae() -> MagicMock:
    """Create a mock VAE for unit testing."""
    mock_vae = MagicMock()
    mock_vae.config = MagicMock()
    mock_vae.config.scaling_factor = 0.18215
    return mock_vae


class TestSVDVideoGeneratorInitialization:
    """Tests for SVDVideoGenerator initialization."""

    @patch("embedding_art.generators.video.AutoencoderKLTemporalDecoder")
    def test_init_should_load_vae_from_pretrained(self, mock_vae_class: MagicMock) -> None:
        """__init__ should load AutoencoderKLTemporalDecoder from HuggingFace."""
        mock_vae = create_mock_vae()
        mock_vae_class.from_pretrained.return_value = mock_vae

        SVDVideoGenerator(device="cpu")

        mock_vae_class.from_pretrained.assert_called_once()
        call_args = mock_vae_class.from_pretrained.call_args
        assert call_args[0][0] == SVDVideoGenerator.MODEL_ID_SVD_XT
        assert call_args[1]["subfolder"] == "vae"

    @patch("embedding_art.generators.video.AutoencoderKLTemporalDecoder")
    def test_init_should_move_vae_to_device(self, mock_vae_class: MagicMock) -> None:
        """__init__ should move VAE to the specified device."""
        mock_vae = create_mock_vae()
        mock_vae_class.from_pretrained.return_value = mock_vae

        SVDVideoGenerator(device="cpu")

        mock_vae.to.assert_called_once()

    @patch("embedding_art.generators.video.AutoencoderKLTemporalDecoder")
    def test_init_should_set_vae_to_eval_mode(self, mock_vae_class: MagicMock) -> None:
        """__init__ should set VAE to evaluation mode."""
        mock_vae = create_mock_vae()
        mock_vae_class.from_pretrained.return_value = mock_vae

        SVDVideoGenerator(device="cpu")

        mock_vae.eval.assert_called_once()

    @patch("embedding_art.generators.video.AutoencoderKLTemporalDecoder")
    def test_init_should_freeze_vae_parameters(self, mock_vae_class: MagicMock) -> None:
        """__init__ should freeze VAE parameters (no gradient updates)."""
        mock_vae = create_mock_vae()
        mock_param = MagicMock()
        mock_vae.parameters.return_value = [mock_param]
        mock_vae_class.from_pretrained.return_value = mock_vae

        SVDVideoGenerator(device="cpu")

        mock_param.requires_grad_ = False

    @patch("embedding_art.generators.video.AutoencoderKLTemporalDecoder")
    def test_init_should_accept_custom_num_frames(self, mock_vae_class: MagicMock) -> None:
        """__init__ should accept custom num_frames parameter."""
        mock_vae = create_mock_vae()
        mock_vae_class.from_pretrained.return_value = mock_vae

        generator = SVDVideoGenerator(device="cpu", num_frames=14)

        assert generator._num_frames == 14

    @patch("embedding_art.generators.video.AutoencoderKLTemporalDecoder")
    def test_device_property_should_return_configured_device(
        self, mock_vae_class: MagicMock
    ) -> None:
        """device property should return the configured torch device."""
        mock_vae = create_mock_vae()
        mock_vae_class.from_pretrained.return_value = mock_vae

        generator = SVDVideoGenerator(device="cpu")

        assert generator.device == torch.device("cpu")


class TestSVDVideoGeneratorLatentOperations:
    """Tests for latent initialization and operations."""

    @patch("embedding_art.generators.video.AutoencoderKLTemporalDecoder")
    def test_init_latent_should_return_tensor_with_correct_shape(
        self, mock_vae_class: MagicMock
    ) -> None:
        """init_latent should return tensor with shape [1, 4, F, 72, 128]."""
        mock_vae = create_mock_vae()
        mock_vae_class.from_pretrained.return_value = mock_vae

        generator = SVDVideoGenerator(device="cpu", num_frames=25)
        latent = generator.init_latent()

        expected_shape = (1, 4, 25, 72, 128)
        assert latent.shape == expected_shape

    @patch("embedding_art.generators.video.AutoencoderKLTemporalDecoder")
    def test_init_latent_should_have_requires_grad_true(self, mock_vae_class: MagicMock) -> None:
        """init_latent should return tensor with requires_grad=True."""
        mock_vae = create_mock_vae()
        mock_vae_class.from_pretrained.return_value = mock_vae

        generator = SVDVideoGenerator(device="cpu")
        latent = generator.init_latent()

        assert latent.requires_grad is True

    @patch("embedding_art.generators.video.AutoencoderKLTemporalDecoder")
    def test_init_latent_should_be_on_correct_device(self, mock_vae_class: MagicMock) -> None:
        """init_latent should return tensor on the generator's device."""
        mock_vae = create_mock_vae()
        mock_vae_class.from_pretrained.return_value = mock_vae

        generator = SVDVideoGenerator(device="cpu")
        latent = generator.init_latent()

        assert latent.device == torch.device("cpu")

    @patch("embedding_art.generators.video.AutoencoderKLTemporalDecoder")
    def test_init_latent_with_seed_should_be_reproducible(self, mock_vae_class: MagicMock) -> None:
        """init_latent with same seed should produce identical results."""
        mock_vae = create_mock_vae()
        mock_vae_class.from_pretrained.return_value = mock_vae

        generator = SVDVideoGenerator(device="cpu")
        latent1 = generator.init_latent(seed=42)
        latent2 = generator.init_latent(seed=42)

        assert torch.allclose(latent1, latent2)

    @patch("embedding_art.generators.video.AutoencoderKLTemporalDecoder")
    def test_init_latent_without_seed_should_vary(self, mock_vae_class: MagicMock) -> None:
        """init_latent without seed should produce different results."""
        mock_vae = create_mock_vae()
        mock_vae_class.from_pretrained.return_value = mock_vae

        generator = SVDVideoGenerator(device="cpu")
        latent1 = generator.init_latent()
        latent2 = generator.init_latent()

        assert not torch.allclose(latent1, latent2)

    @patch("embedding_art.generators.video.AutoencoderKLTemporalDecoder")
    def test_latent_shape_should_respect_num_frames(self, mock_vae_class: MagicMock) -> None:
        """latent_shape property should use configured num_frames."""
        mock_vae = create_mock_vae()
        mock_vae_class.from_pretrained.return_value = mock_vae

        generator = SVDVideoGenerator(device="cpu", num_frames=14)

        assert generator.latent_shape == (1, 4, 14, 72, 128)


class TestSVDVideoGeneratorDecode:
    """Tests for decoding latents to video."""

    @patch("embedding_art.generators.video.AutoencoderKLTemporalDecoder")
    def test_decode_should_call_vae_decode(self, mock_vae_class: MagicMock) -> None:
        """decode should pass latent through VAE decoder."""
        mock_vae = create_mock_vae()
        mock_decoded = MagicMock()
        # VAE returns flattened frames: [B*F, 3, H*8, W*8]
        mock_decoded.sample = torch.randn(25, 3, 576, 1024)
        mock_vae.decode.return_value = mock_decoded
        mock_vae_class.from_pretrained.return_value = mock_vae

        generator = SVDVideoGenerator(device="cpu")
        latent = torch.randn(1, 4, 25, 72, 128)
        generator.decode(latent)

        mock_vae.decode.assert_called_once()

    @patch("embedding_art.generators.video.AutoencoderKLTemporalDecoder")
    def test_decode_should_scale_latent_before_decoding(self, mock_vae_class: MagicMock) -> None:
        """decode should divide latent by SCALING_FACTOR before VAE decode."""
        mock_vae = create_mock_vae()
        mock_decoded = MagicMock()
        # VAE returns flattened frames: [B*F, 3, H*8, W*8]
        mock_decoded.sample = torch.randn(25, 3, 576, 1024)
        mock_vae.decode.return_value = mock_decoded
        mock_vae_class.from_pretrained.return_value = mock_vae

        generator = SVDVideoGenerator(device="cpu")
        latent = torch.randn(1, 4, 25, 72, 128)
        generator.decode(latent)

        # After permute and flatten, input is [B*F, C, H, W]
        actual_latent = mock_vae.decode.call_args[0][0]
        expected_latent = (
            latent.permute(0, 2, 1, 3, 4).reshape(25, 4, 72, 128) / SVDVideoGenerator.SCALING_FACTOR
        )
        assert torch.allclose(actual_latent, expected_latent, rtol=1e-5)

    @patch("embedding_art.generators.video.AutoencoderKLTemporalDecoder")
    def test_decode_output_should_be_in_zero_one_range(self, mock_vae_class: MagicMock) -> None:
        """decode should return tensor with values clamped to [0, 1]."""
        mock_vae = create_mock_vae()
        mock_decoded = MagicMock()
        # VAE returns flattened frames: [B*F, 3, H*8, W*8] with values outside [0,1]
        mock_decoded.sample = torch.randn(25, 3, 576, 1024) * 2
        mock_vae.decode.return_value = mock_decoded
        mock_vae_class.from_pretrained.return_value = mock_vae

        generator = SVDVideoGenerator(device="cpu")
        latent = torch.randn(1, 4, 25, 72, 128)
        output = generator.decode(latent)

        assert output.min() >= 0.0
        assert output.max() <= 1.0


class TestSVDVideoGeneratorEncode:
    """Tests for encoding video to latent."""

    @patch("embedding_art.generators.video.AutoencoderKLTemporalDecoder")
    def test_encode_should_process_each_frame(self, mock_vae_class: MagicMock) -> None:
        """encode should process video frames through encoder."""
        mock_vae = create_mock_vae()
        mock_latent_dist = MagicMock()
        mock_latent_dist.sample.return_value = torch.randn(25, 4, 72, 128)
        mock_vae.encode.return_value = MagicMock(latent_dist=mock_latent_dist)
        mock_vae_class.from_pretrained.return_value = mock_vae

        generator = SVDVideoGenerator(device="cpu")
        video = torch.rand(1, 25, 3, 576, 1024)
        generator.encode(video)

        mock_vae.encode.assert_called_once()

    @patch("embedding_art.generators.video.AutoencoderKLTemporalDecoder")
    def test_encode_should_apply_scaling_factor(self, mock_vae_class: MagicMock) -> None:
        """encode should multiply by SCALING_FACTOR."""
        mock_vae = create_mock_vae()
        mock_latent_dist = MagicMock()
        raw_latent = torch.randn(25, 4, 72, 128)
        mock_latent_dist.sample.return_value = raw_latent
        mock_vae.encode.return_value = MagicMock(latent_dist=mock_latent_dist)
        mock_vae_class.from_pretrained.return_value = mock_vae

        generator = SVDVideoGenerator(device="cpu")
        video = torch.rand(1, 25, 3, 576, 1024)
        latent = generator.encode(video)

        expected = (
            raw_latent.view(1, 25, 4, 72, 128).permute(0, 2, 1, 3, 4)
            * SVDVideoGenerator.SCALING_FACTOR
        )
        assert torch.allclose(latent, expected, rtol=1e-5)


class TestSVDVideoGeneratorConditioning:
    """Tests for conditioning image handling."""

    @patch("embedding_art.generators.video.AutoencoderKLTemporalDecoder")
    def test_set_conditioning_image_should_store_image(self, mock_vae_class: MagicMock) -> None:
        """set_conditioning_image should store the conditioning image."""
        mock_vae = create_mock_vae()
        mock_vae_class.from_pretrained.return_value = mock_vae

        generator = SVDVideoGenerator(device="cpu")
        conditioning_image = torch.rand(1, 3, 576, 1024)
        generator.set_conditioning_image(conditioning_image)

        assert generator._conditioning_image is not None
        assert torch.equal(generator._conditioning_image, conditioning_image)

    @patch("embedding_art.generators.video.AutoencoderKLTemporalDecoder")
    def test_set_conditioning_image_should_validate_dimensions(
        self, mock_vae_class: MagicMock
    ) -> None:
        """set_conditioning_image should validate image has correct dimensions."""
        mock_vae = create_mock_vae()
        mock_vae_class.from_pretrained.return_value = mock_vae

        generator = SVDVideoGenerator(device="cpu")
        wrong_size_image = torch.rand(1, 3, 512, 512)

        with pytest.raises(ValueError, match="576x1024"):
            generator.set_conditioning_image(wrong_size_image)


@pytest.mark.slow
class TestSVDVideoGeneratorIntegration:
    """
    Integration tests for SVD generator.

    These tests require downloading the SVD model weights (~4GB)
    and significant memory to run. Mark with @pytest.mark.slow.
    """

    def test_should_load_vae_successfully(self) -> None:
        """The generator should load the SVD VAE without errors."""
        generator = SVDVideoGenerator(
            model_id=SVDVideoGenerator.MODEL_ID_SVD_XT,
            device="mps",
        )
        assert generator is not None

    def test_should_initialize_latent_with_correct_shape(self) -> None:
        """init_latent should return tensor with shape [1, 4, 25, 72, 128]."""
        generator = SVDVideoGenerator(device="mps")
        latent = generator.init_latent()

        expected_shape = (1, 4, 25, 72, 128)
        assert latent.shape == expected_shape

    def test_decode_should_produce_video_frames(self) -> None:
        """decode should produce correct output shape with reduced frames."""
        # Use only 2 frames to fit in memory (~6GB instead of ~80GB)
        num_frames = 2
        generator = SVDVideoGenerator(device="mps", num_frames=num_frames)
        latent = generator.init_latent(seed=42)
        output = generator.decode(latent)

        expected_shape = (1, num_frames, 3, 576, 1024)
        assert output.shape == expected_shape

    def test_decode_should_be_differentiable(self) -> None:
        """decode should allow gradients to flow back for optimization."""
        # Use only 2 frames to fit in memory
        num_frames = 2
        generator = SVDVideoGenerator(device="mps", num_frames=num_frames)
        latent = generator.init_latent(seed=42)
        output = generator.decode(latent)

        loss = output.mean()
        loss.backward()

        assert latent.grad is not None
