"""
Tests for SVDVideoGenerator.

These tests verify the SVD generator stub class has the expected
properties and constants for Stable Video Diffusion integration.

Tests are marked with @pytest.mark.skip until the full implementation
is complete, as the stub raises NotImplementedError.
"""

import pytest

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

    def test_latent_shape_property_should_return_5d_tuple(self) -> None:
        """
        latent_shape should return [B, C, F, H, W] tuple.

        Expected: (1, 4, 25, 72, 128) for SVD-XT at 576x1024
        """
        generator = SVDVideoGenerator.__new__(SVDVideoGenerator)

        expected_shape = (1, 4, 25, 72, 128)
        assert generator.latent_shape == expected_shape

    def test_latent_shape_batch_dimension_should_be_one(self) -> None:
        """Batch dimension defaults to 1 for single video generation."""
        generator = SVDVideoGenerator.__new__(SVDVideoGenerator)
        batch_dim = generator.latent_shape[0]
        assert batch_dim == 1

    def test_latent_shape_channel_dimension_should_be_four(self) -> None:
        """Channel dimension should match LATENT_CHANNELS constant."""
        generator = SVDVideoGenerator.__new__(SVDVideoGenerator)
        channel_dim = generator.latent_shape[1]
        assert channel_dim == SVDVideoGenerator.LATENT_CHANNELS

    def test_latent_shape_frame_dimension_should_match_default(self) -> None:
        """Frame dimension should match DEFAULT_NUM_FRAMES constant."""
        generator = SVDVideoGenerator.__new__(SVDVideoGenerator)
        frame_dim = generator.latent_shape[2]
        assert frame_dim == SVDVideoGenerator.DEFAULT_NUM_FRAMES


class TestSVDVideoGeneratorOutputModality:
    """Tests for output_modality property."""

    def test_output_modality_should_be_video(self) -> None:
        """The output modality should be 'video'."""
        generator = SVDVideoGenerator.__new__(SVDVideoGenerator)
        assert generator.output_modality == "video"


class TestSVDVideoGeneratorStubBehavior:
    """Tests verifying stub implementation raises NotImplementedError."""

    def test_init_should_raise_not_implemented_error(self) -> None:
        """__init__ should raise NotImplementedError with guidance."""
        with pytest.raises(NotImplementedError) as exc_info:
            SVDVideoGenerator()

        error_message = str(exc_info.value)
        assert "research stub" in error_message.lower()
        assert "AutoencoderKLTemporalDecoder" in error_message

    def test_device_property_should_raise_not_implemented_error(self) -> None:
        """device property should raise NotImplementedError."""
        generator = SVDVideoGenerator.__new__(SVDVideoGenerator)

        with pytest.raises(NotImplementedError):
            _ = generator.device

    def test_init_latent_should_raise_not_implemented_error(self) -> None:
        """init_latent should raise NotImplementedError with guidance."""
        generator = SVDVideoGenerator.__new__(SVDVideoGenerator)

        with pytest.raises(NotImplementedError) as exc_info:
            generator.init_latent()

        error_message = str(exc_info.value)
        assert "init_latent" in error_message

    def test_decode_should_raise_not_implemented_error(self) -> None:
        """decode should raise NotImplementedError with guidance."""
        generator = SVDVideoGenerator.__new__(SVDVideoGenerator)

        with pytest.raises(NotImplementedError) as exc_info:
            generator.decode(None)

        error_message = str(exc_info.value)
        assert "decode" in error_message

    def test_encode_should_raise_not_implemented_error(self) -> None:
        """encode should raise NotImplementedError with guidance."""
        generator = SVDVideoGenerator.__new__(SVDVideoGenerator)

        with pytest.raises(NotImplementedError) as exc_info:
            generator.encode(None)

        error_message = str(exc_info.value)
        assert "encode" in error_message

    def test_set_conditioning_image_should_raise_not_implemented_error(self) -> None:
        """set_conditioning_image should raise NotImplementedError with guidance."""
        generator = SVDVideoGenerator.__new__(SVDVideoGenerator)

        with pytest.raises(NotImplementedError) as exc_info:
            generator.set_conditioning_image(None)

        error_message = str(exc_info.value)
        assert "set_conditioning_image" in error_message


@pytest.mark.skip(reason="Full implementation not yet complete")
class TestSVDVideoGeneratorIntegration:
    """
    Integration tests for SVD generator (skipped until implementation complete).

    These tests will require downloading the SVD model weights (~4GB)
    and significant memory to run.
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
        """decode should produce output shape [1, 25, 3, 576, 1024]."""
        generator = SVDVideoGenerator(device="mps")
        latent = generator.init_latent(seed=42)
        output = generator.decode(latent)

        expected_shape = (1, 25, 3, 576, 1024)
        assert output.shape == expected_shape

    def test_decode_should_be_differentiable(self) -> None:
        """decode should allow gradients to flow back for optimization."""
        generator = SVDVideoGenerator(device="mps")
        latent = generator.init_latent(seed=42)
        output = generator.decode(latent)

        loss = output.mean()
        loss.backward()

        assert latent.grad is not None
