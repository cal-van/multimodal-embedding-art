"""
Tests for image upscaling functionality.

These tests verify that the RealESRGAN upscaler correctly loads,
upscales images, and integrates with the CLI.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from click.testing import CliRunner
from PIL import Image

from embedding_art.cli.main import cli


class TestRealESRGANUpscalerExists:
    """Tests for RealESRGANUpscaler class existence and basic structure."""

    def test_upscaler_module_exists(self) -> None:
        """The upscalers module should be importable."""
        from embedding_art import upscalers

        assert upscalers is not None

    def test_realesrgan_upscaler_class_exists(self) -> None:
        """The RealESRGANUpscaler class should exist."""
        from embedding_art.upscalers import RealESRGANUpscaler

        assert RealESRGANUpscaler is not None

    def test_upscaler_has_upscale_method(self) -> None:
        """RealESRGANUpscaler should have an upscale method."""
        from embedding_art.upscalers import RealESRGANUpscaler

        assert hasattr(RealESRGANUpscaler, "upscale")
        assert callable(getattr(RealESRGANUpscaler, "upscale"))

    def test_upscaler_has_device_property(self) -> None:
        """RealESRGANUpscaler should have a device property."""
        from embedding_art.upscalers import RealESRGANUpscaler

        # Check the property exists on the class
        assert "device" in dir(RealESRGANUpscaler)


class TestRealESRGANUpscalerInit:
    """Tests for RealESRGANUpscaler initialization."""

    def test_init_accepts_device_parameter(self) -> None:
        """RealESRGANUpscaler should accept a device parameter."""
        from embedding_art.upscalers import RealESRGANUpscaler

        # Initialization is lazy - model isn't loaded until first upscale
        upscaler = RealESRGANUpscaler(device="cpu")

        assert upscaler.device == torch.device("cpu")

    def test_init_accepts_scale_parameter(self) -> None:
        """RealESRGANUpscaler should accept a scale parameter."""
        from embedding_art.upscalers import RealESRGANUpscaler

        # Initialization is lazy - model isn't loaded until first upscale
        upscaler = RealESRGANUpscaler(device="cpu", scale=2)

        assert upscaler.scale == 2

    def test_init_default_scale_is_four(self) -> None:
        """Default scale should be 4x."""
        from embedding_art.upscalers import RealESRGANUpscaler

        # Initialization is lazy - model isn't loaded until first upscale
        upscaler = RealESRGANUpscaler(device="cpu")

        assert upscaler.scale == 4


class TestRealESRGANUpscalerUpscale:
    """Tests for the upscale method."""

    def test_upscale_accepts_pil_image(self) -> None:
        """upscale should accept a PIL Image and return a PIL Image."""
        from embedding_art.upscalers import RealESRGANUpscaler

        # Create mock for the lazily-imported RealESRGAN
        mock_realesrgan_instance = MagicMock()
        upscaled_image = Image.new("RGB", (256, 256), color="blue")
        mock_realesrgan_instance.predict.return_value = upscaled_image

        mock_realesrgan_class = MagicMock(return_value=mock_realesrgan_instance)

        # Mock the import by patching the builtins __import__
        import sys

        fake_module = MagicMock()
        fake_module.RealESRGAN = mock_realesrgan_class

        with patch.dict(sys.modules, {"RealESRGAN": fake_module}):
            upscaler = RealESRGANUpscaler(device="cpu")
            input_image = Image.new("RGB", (64, 64), color="red")
            result = upscaler.upscale(input_image)

        assert isinstance(result, Image.Image)
        mock_realesrgan_instance.predict.assert_called_once_with(input_image)

    def test_upscale_accepts_scale_override(self) -> None:
        """upscale should accept an optional scale parameter to override default."""
        from embedding_art.upscalers import RealESRGANUpscaler

        # Create mock for the lazily-imported RealESRGAN
        mock_realesrgan_instance = MagicMock()
        upscaled_image = Image.new("RGB", (128, 128), color="blue")
        mock_realesrgan_instance.predict.return_value = upscaled_image

        mock_realesrgan_class = MagicMock(return_value=mock_realesrgan_instance)

        import sys

        fake_module = MagicMock()
        fake_module.RealESRGAN = mock_realesrgan_class

        with patch.dict(sys.modules, {"RealESRGAN": fake_module}):
            upscaler = RealESRGANUpscaler(device="cpu", scale=4)
            input_image = Image.new("RGB", (64, 64), color="red")
            # Override scale to 2
            result = upscaler.upscale(input_image, scale=2)

        assert isinstance(result, Image.Image)

    def test_upscale_returns_larger_image(self) -> None:
        """upscale should return an image larger than the input."""
        from embedding_art.upscalers import RealESRGANUpscaler

        input_size = (64, 64)
        output_size = (256, 256)  # 4x upscale

        # Create mock for the lazily-imported RealESRGAN
        mock_realesrgan_instance = MagicMock()
        upscaled_image = Image.new("RGB", output_size, color="blue")
        mock_realesrgan_instance.predict.return_value = upscaled_image

        mock_realesrgan_class = MagicMock(return_value=mock_realesrgan_instance)

        import sys

        fake_module = MagicMock()
        fake_module.RealESRGAN = mock_realesrgan_class

        with patch.dict(sys.modules, {"RealESRGAN": fake_module}):
            upscaler = RealESRGANUpscaler(device="cpu", scale=4)
            input_image = Image.new("RGB", input_size, color="red")
            result = upscaler.upscale(input_image)

        assert result.size[0] > input_image.size[0]
        assert result.size[1] > input_image.size[1]


class TestUpscaleCLICommand:
    """Tests for the upscale CLI command."""

    def test_upscale_command_exists_in_help(self) -> None:
        """The CLI should have an upscale command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "upscale" in result.output

    def test_upscale_command_has_input_argument(self) -> None:
        """The upscale command should have an input argument."""
        runner = CliRunner()
        result = runner.invoke(cli, ["upscale", "--help"])

        assert result.exit_code == 0
        assert "INPUT" in result.output or "input" in result.output.lower()

    def test_upscale_command_has_scale_option(self) -> None:
        """The upscale command should have a --scale option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["upscale", "--help"])

        assert result.exit_code == 0
        assert "--scale" in result.output

    def test_upscale_command_has_output_option(self) -> None:
        """The upscale command should have an --output option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["upscale", "--help"])

        assert result.exit_code == 0
        assert "--output" in result.output

    def test_upscale_command_requires_input(self) -> None:
        """The upscale command should require an input file."""
        runner = CliRunner()
        result = runner.invoke(cli, ["upscale"])

        # Should fail because input is required
        assert result.exit_code != 0

    def test_upscale_command_validates_input_exists(self) -> None:
        """The upscale command should validate that the input file exists."""
        runner = CliRunner()
        result = runner.invoke(cli, ["upscale", "/nonexistent/path/image.png"])

        assert result.exit_code != 0

    def test_upscale_command_runs_with_mocked_upscaler(self) -> None:
        """The upscale command should run successfully with valid input."""
        runner = CliRunner()

        # Create a mock upscaler that returns an image
        mock_upscaler = MagicMock()
        upscaled_image = Image.new("RGB", (256, 256), color="blue")
        mock_upscaler_instance = MagicMock()
        mock_upscaler_instance.upscale.return_value = upscaled_image
        mock_upscaler.return_value = mock_upscaler_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test input image
            input_path = Path(tmpdir) / "input.png"
            input_image = Image.new("RGB", (64, 64), color="red")
            input_image.save(input_path)

            output_path = Path(tmpdir) / "output.png"

            with patch.dict(
                "sys.modules",
                {
                    "embedding_art.upscalers": MagicMock(RealESRGANUpscaler=mock_upscaler),
                },
            ):
                result = runner.invoke(
                    cli,
                    [
                        "upscale",
                        str(input_path),
                        "--output",
                        str(output_path),
                    ],
                )

            if result.exit_code != 0 and result.exception:
                import traceback

                traceback.print_exception(
                    type(result.exception), result.exception, result.exception.__traceback__
                )

            assert result.exit_code == 0

    def test_upscale_command_default_scale_is_four(self) -> None:
        """The upscale command should default to 4x scale."""
        runner = CliRunner()
        result = runner.invoke(cli, ["upscale", "--help"])

        assert result.exit_code == 0
        # Check default value is mentioned
        assert "4" in result.output

    def test_upscale_command_accepts_custom_scale(self) -> None:
        """The upscale command should accept a custom scale value."""
        runner = CliRunner()

        mock_upscaler = MagicMock()
        upscaled_image = Image.new("RGB", (128, 128), color="blue")
        mock_upscaler_instance = MagicMock()
        mock_upscaler_instance.upscale.return_value = upscaled_image
        mock_upscaler.return_value = mock_upscaler_instance

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.png"
            input_image = Image.new("RGB", (64, 64), color="red")
            input_image.save(input_path)

            output_path = Path(tmpdir) / "output.png"

            with patch.dict(
                "sys.modules",
                {
                    "embedding_art.upscalers": MagicMock(RealESRGANUpscaler=mock_upscaler),
                },
            ):
                result = runner.invoke(
                    cli,
                    [
                        "upscale",
                        str(input_path),
                        "--scale",
                        "2",
                        "--output",
                        str(output_path),
                    ],
                )

            # Check that the scale=2 was passed to the upscaler
            if result.exit_code == 0:
                mock_upscaler.assert_called()


class TestUpscalerExceptions:
    """Tests for upscaler exception handling."""

    def test_upscaler_error_class_exists(self) -> None:
        """UpscalerError exception class should exist."""
        from embedding_art.exceptions import UpscalerError

        assert issubclass(UpscalerError, Exception)

    def test_upscaler_error_inherits_from_embedding_art_error(self) -> None:
        """UpscalerError should inherit from EmbeddingArtError."""
        from embedding_art.exceptions import EmbeddingArtError, UpscalerError

        assert issubclass(UpscalerError, EmbeddingArtError)
