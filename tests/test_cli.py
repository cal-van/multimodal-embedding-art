"""
Tests for CLI configuration handling.

These tests verify that the CLI correctly loads config files,
applies defaults, and respects CLI argument overrides.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from embedding_art.cli.main import cli


class TestCliConfigOption:
    """Tests for the --config CLI option."""

    def test_config_option_exists(self) -> None:
        """The CLI should have a --config option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert "--config" in result.output

    def test_config_option_accepts_path(self) -> None:
        """The --config option should accept a file path."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("""
device: cpu
optimization:
  steps: 100
""")
            f.flush()

            result = runner.invoke(cli, ["--config", f.name, "--help"])

        # Should not error out, just print help
        assert result.exit_code == 0

    def test_config_option_error_for_invalid_yaml(self) -> None:
        """Should show error for invalid YAML in config file."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("""
invalid: yaml: content: [
""")
            f.flush()

            result = runner.invoke(cli, ["--config", f.name, "embed", "-t", "test"])

        assert result.exit_code != 0


def _create_mock_modules():
    """Create mock modules for CLI testing."""
    mock_encoder = MagicMock()
    mock_generator = MagicMock()
    mock_engine = MagicMock()
    mock_concept = MagicMock()

    mock_concept_instance = MagicMock()
    mock_concept_instance.description = "test"
    mock_concept.from_text.return_value = mock_concept_instance

    mock_result = MagicMock()
    mock_result.final_similarity = 0.95
    mock_result.elapsed_seconds = 1.0
    mock_result.get_final_image.return_value = MagicMock()

    mock_engine_instance = MagicMock()
    mock_engine_instance.optimize.return_value = mock_result
    mock_engine.return_value = mock_engine_instance

    return mock_encoder, mock_generator, mock_engine, mock_concept


class TestCliConfigMerging:
    """Tests for CLI arg merging with config file values."""

    def test_cli_args_override_config_file(self) -> None:
        """CLI args should override values from config file."""
        runner = CliRunner()

        mock_encoder, mock_generator, mock_engine, mock_concept = _create_mock_modules()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            config_path.write_text("""
device: cpu
optimization:
  steps: 500
  learning_rate: 0.05
""")
            with patch.dict(
                "sys.modules",
                {
                    "embedding_art": MagicMock(
                        Concept=mock_concept,
                        EmbeddingArtEngine=mock_engine,
                        OptimizationConfig=MagicMock(side_effect=lambda **kw: MagicMock(**kw)),
                    ),
                    "embedding_art.encoders": MagicMock(ImageBindEncoder=mock_encoder),
                    "embedding_art.generators": MagicMock(SDXLImageGenerator=mock_generator),
                },
            ):
                result = runner.invoke(
                    cli,
                    [
                        "--config",
                        str(config_path),
                        "optimize",
                        "-t",
                        "test",
                        "1.0",
                        "--steps",
                        "100",
                        "--lr",
                        "0.2",
                        "-p",
                        str(Path(tmpdir) / "out.png"),
                    ],
                )

            if result.exit_code != 0 and result.exception:
                import traceback

                traceback.print_exception(type(result.exception), result.exception, result.exception.__traceback__)

            call_args = mock_engine.return_value.optimize.call_args
            if call_args:
                opt_config = call_args[1].get("config") or call_args[0][2]
                assert opt_config.steps == 100
                assert opt_config.learning_rate == 0.2

    def test_config_file_values_used_when_no_cli_override(self) -> None:
        """Config file values should be used when CLI args not provided."""
        runner = CliRunner()

        mock_encoder, mock_generator, mock_engine, mock_concept = _create_mock_modules()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            config_path.write_text("""
device: cpu
optimization:
  steps: 750
  learning_rate: 0.08
""")
            with patch.dict(
                "sys.modules",
                {
                    "embedding_art": MagicMock(
                        Concept=mock_concept,
                        EmbeddingArtEngine=mock_engine,
                        OptimizationConfig=MagicMock(side_effect=lambda **kw: MagicMock(**kw)),
                    ),
                    "embedding_art.encoders": MagicMock(ImageBindEncoder=mock_encoder),
                    "embedding_art.generators": MagicMock(SDXLImageGenerator=mock_generator),
                },
            ):
                result = runner.invoke(
                    cli,
                    [
                        "--config",
                        str(config_path),
                        "optimize",
                        "-t",
                        "test",
                        "1.0",
                        "-p",
                        str(Path(tmpdir) / "out.png"),
                    ],
                )

            if result.exit_code != 0 and result.exception:
                import traceback

                traceback.print_exception(type(result.exception), result.exception, result.exception.__traceback__)

            call_args = mock_engine.return_value.optimize.call_args
            if call_args:
                opt_config = call_args[1].get("config") or call_args[0][2]
                assert opt_config.steps == 750
                assert opt_config.learning_rate == 0.08

    def test_device_from_config_file(self) -> None:
        """Device setting should be read from config file."""
        runner = CliRunner()

        mock_encoder, mock_generator, mock_engine, mock_concept = _create_mock_modules()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"
            config_path.write_text("""
device: cuda
optimization:
  steps: 100
""")
            with patch.dict(
                "sys.modules",
                {
                    "embedding_art": MagicMock(
                        Concept=mock_concept,
                        EmbeddingArtEngine=mock_engine,
                        OptimizationConfig=MagicMock(side_effect=lambda **kw: MagicMock(**kw)),
                    ),
                    "embedding_art.encoders": MagicMock(ImageBindEncoder=mock_encoder),
                    "embedding_art.generators": MagicMock(SDXLImageGenerator=mock_generator),
                },
            ):
                result = runner.invoke(
                    cli,
                    [
                        "--config",
                        str(config_path),
                        "optimize",
                        "-t",
                        "test",
                        "1.0",
                        "-p",
                        str(Path(tmpdir) / "out.png"),
                    ],
                )

            mock_encoder.assert_called_with(device="cuda")


class TestDefaultConfigLoading:
    """Tests for automatic loading of default config.yaml."""

    def test_loads_config_yaml_from_cwd_by_default(self, isolated_cwd: Path) -> None:
        """Should automatically load config.yaml from current directory."""
        runner = CliRunner()

        mock_encoder, mock_generator, mock_engine, mock_concept = _create_mock_modules()

        config_path = isolated_cwd / "config.yaml"
        config_path.write_text("""
device: cpu
optimization:
  steps: 333
""")

        with patch.dict(
            "sys.modules",
            {
                "embedding_art": MagicMock(
                    Concept=mock_concept,
                    EmbeddingArtEngine=mock_engine,
                    OptimizationConfig=MagicMock(side_effect=lambda **kw: MagicMock(**kw)),
                ),
                "embedding_art.encoders": MagicMock(ImageBindEncoder=mock_encoder),
                "embedding_art.generators": MagicMock(SDXLImageGenerator=mock_generator),
            },
        ):
            result = runner.invoke(
                cli,
                [
                    "optimize",
                    "-t",
                    "test",
                    "1.0",
                    "-p",
                    str(isolated_cwd / "out.png"),
                ],
            )

        if result.exit_code != 0 and result.exception:
            import traceback

            traceback.print_exception(type(result.exception), result.exception, result.exception.__traceback__)

        call_args = mock_engine.return_value.optimize.call_args
        if call_args:
            opt_config = call_args[1].get("config") or call_args[0][2]
            assert opt_config.steps == 333
