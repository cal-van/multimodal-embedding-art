"""
Tests for batch processing CLI command.

These tests verify that the batch command correctly:
- Parses YAML batch files
- Executes jobs sequentially with proper configuration
- Shows progress across all jobs
- Handles errors gracefully
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from embedding_art.cli.main import cli


class TestBatchCommandExists:
    """Tests that the batch command is available and has expected options."""

    def test_batch_command_available_in_cli(self) -> None:
        """The CLI should have a batch command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "batch" in result.output

    def test_batch_command_has_help(self) -> None:
        """The batch command should have help text."""
        runner = CliRunner()
        result = runner.invoke(cli, ["batch", "--help"])

        assert result.exit_code == 0
        assert "batch" in result.output.lower()

    def test_batch_command_requires_file_argument(self) -> None:
        """The batch command should require a YAML file argument."""
        runner = CliRunner()
        result = runner.invoke(cli, ["batch"])

        # Should fail due to missing required argument
        assert result.exit_code != 0

    def test_batch_command_accepts_file_argument(self) -> None:
        """The batch command should accept a YAML file path."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("""
jobs: []
""")
            f.flush()

            result = runner.invoke(cli, ["batch", f.name])

        # Should not fail due to argument parsing (may fail for empty jobs)
        assert "Missing argument" not in result.output


class TestBatchFileParsing:
    """Tests for batch file YAML parsing."""

    def test_batch_file_must_have_jobs_key(self) -> None:
        """Batch file must contain a 'jobs' key."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("""
not_jobs:
  - target_text: [["test", 1.0]]
""")
            f.flush()

            result = runner.invoke(cli, ["batch", f.name])

        assert result.exit_code != 0
        assert "jobs" in result.output.lower()

    def test_batch_file_jobs_must_be_list(self) -> None:
        """Jobs must be a list, not a dict or scalar."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("""
jobs:
  target_text: [["test", 1.0]]
""")
            f.flush()

            result = runner.invoke(cli, ["batch", f.name])

        assert result.exit_code != 0

    def test_batch_file_empty_jobs_succeeds(self) -> None:
        """Empty jobs list should succeed (no-op)."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("""
jobs: []
""")
            f.flush()

            result = runner.invoke(cli, ["batch", f.name])

        assert result.exit_code == 0

    def test_batch_file_invalid_yaml_shows_error(self) -> None:
        """Invalid YAML should show a clear error."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("""
jobs: [
  invalid yaml structure
""")
            f.flush()

            result = runner.invoke(cli, ["batch", f.name])

        assert result.exit_code != 0

    def test_batch_file_nonexistent_shows_error(self) -> None:
        """Nonexistent batch file should show error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["batch", "/nonexistent/batch.yaml"])

        assert result.exit_code != 0

    def test_batch_job_requires_target(self) -> None:
        """Each job must have at least one target (text, image, or audio)."""
        runner = CliRunner()

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
            f.write("""
jobs:
  - output: outputs/test.png
""")
            f.flush()

            result = runner.invoke(cli, ["batch", f.name])

        assert result.exit_code != 0
        assert "target" in result.output.lower()


class TestBatchJobExecution:
    """Tests for batch job execution with mocked engine."""

    def _create_mock_modules(self):
        """Create mock modules for CLI testing."""
        mock_encoder = MagicMock()
        mock_generator = MagicMock()
        mock_engine = MagicMock()
        mock_concept = MagicMock()

        mock_concept_instance = MagicMock()
        mock_concept_instance.description = "test"
        mock_concept.from_text.return_value = mock_concept_instance
        mock_concept.combine.return_value = mock_concept_instance

        mock_result = MagicMock()
        mock_result.final_similarity = 0.95
        mock_result.elapsed_seconds = 1.0
        mock_result.get_final_image.return_value = MagicMock()

        mock_engine_instance = MagicMock()
        mock_engine_instance.optimize.return_value = mock_result
        mock_engine.return_value = mock_engine_instance

        return mock_encoder, mock_generator, mock_engine, mock_concept

    def test_batch_executes_single_job(self) -> None:
        """Batch should execute a single job successfully."""
        runner = CliRunner()
        mock_encoder, mock_generator, mock_engine, mock_concept = self._create_mock_modules()

        with tempfile.TemporaryDirectory() as tmpdir:
            batch_path = Path(tmpdir) / "batch.yaml"
            batch_path.write_text("""
jobs:
  - target_text: [["goldfish", 1.0]]
    output: outputs/goldfish.png
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
                result = runner.invoke(cli, ["batch", str(batch_path)])

            if result.exit_code != 0 and result.exception:
                import traceback

                traceback.print_exception(
                    type(result.exception), result.exception, result.exception.__traceback__
                )

            assert mock_engine.return_value.optimize.call_count == 1

    def test_batch_executes_multiple_jobs(self) -> None:
        """Batch should execute multiple jobs sequentially."""
        runner = CliRunner()
        mock_encoder, mock_generator, mock_engine, mock_concept = self._create_mock_modules()

        with tempfile.TemporaryDirectory() as tmpdir:
            batch_path = Path(tmpdir) / "batch.yaml"
            batch_path.write_text("""
jobs:
  - target_text: [["goldfish", 1.0]]
    output: outputs/goldfish.png
  - target_text: [["flamingo", 1.0]]
    output: outputs/flamingo.png
  - target_text: [["parrot", 1.0]]
    output: outputs/parrot.png
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
                result = runner.invoke(cli, ["batch", str(batch_path)])

            if result.exit_code != 0 and result.exception:
                import traceback

                traceback.print_exception(
                    type(result.exception), result.exception, result.exception.__traceback__
                )

            assert mock_engine.return_value.optimize.call_count == 3

    def test_batch_job_with_steps_override(self) -> None:
        """Job-level steps override should be passed to OptimizationConfig."""
        runner = CliRunner()
        mock_encoder, mock_generator, mock_engine, mock_concept = self._create_mock_modules()

        captured_configs = []

        def capture_config(**kw):
            config = MagicMock(**kw)
            captured_configs.append(kw)
            return config

        with tempfile.TemporaryDirectory() as tmpdir:
            batch_path = Path(tmpdir) / "batch.yaml"
            batch_path.write_text("""
jobs:
  - target_text: [["goldfish", 1.0]]
    output: outputs/goldfish.png
    steps: 500
""")

            with patch.dict(
                "sys.modules",
                {
                    "embedding_art": MagicMock(
                        Concept=mock_concept,
                        EmbeddingArtEngine=mock_engine,
                        OptimizationConfig=MagicMock(side_effect=capture_config),
                    ),
                    "embedding_art.encoders": MagicMock(ImageBindEncoder=mock_encoder),
                    "embedding_art.generators": MagicMock(SDXLImageGenerator=mock_generator),
                },
            ):
                result = runner.invoke(cli, ["batch", str(batch_path)])

            if result.exit_code != 0 and result.exception:
                import traceback

                traceback.print_exception(
                    type(result.exception), result.exception, result.exception.__traceback__
                )

            assert len(captured_configs) >= 1
            assert captured_configs[0].get("steps") == 500

    def test_batch_job_with_learning_rate_override(self) -> None:
        """Job-level learning_rate override should be passed to OptimizationConfig."""
        runner = CliRunner()
        mock_encoder, mock_generator, mock_engine, mock_concept = self._create_mock_modules()

        captured_configs = []

        def capture_config(**kw):
            config = MagicMock(**kw)
            captured_configs.append(kw)
            return config

        with tempfile.TemporaryDirectory() as tmpdir:
            batch_path = Path(tmpdir) / "batch.yaml"
            batch_path.write_text("""
jobs:
  - target_text: [["goldfish", 1.0]]
    output: outputs/goldfish.png
    learning_rate: 0.05
""")

            with patch.dict(
                "sys.modules",
                {
                    "embedding_art": MagicMock(
                        Concept=mock_concept,
                        EmbeddingArtEngine=mock_engine,
                        OptimizationConfig=MagicMock(side_effect=capture_config),
                    ),
                    "embedding_art.encoders": MagicMock(ImageBindEncoder=mock_encoder),
                    "embedding_art.generators": MagicMock(SDXLImageGenerator=mock_generator),
                },
            ):
                result = runner.invoke(cli, ["batch", str(batch_path)])

            if result.exit_code != 0 and result.exception:
                import traceback

                traceback.print_exception(
                    type(result.exception), result.exception, result.exception.__traceback__
                )

            assert len(captured_configs) >= 1
            assert captured_configs[0].get("learning_rate") == 0.05


class TestBatchProgress:
    """Tests for batch progress display."""

    def _create_mock_modules(self):
        """Create mock modules for CLI testing."""
        mock_encoder = MagicMock()
        mock_generator = MagicMock()
        mock_engine = MagicMock()
        mock_concept = MagicMock()

        mock_concept_instance = MagicMock()
        mock_concept_instance.description = "test"
        mock_concept.from_text.return_value = mock_concept_instance
        mock_concept.combine.return_value = mock_concept_instance

        mock_result = MagicMock()
        mock_result.final_similarity = 0.95
        mock_result.elapsed_seconds = 1.0
        mock_result.get_final_image.return_value = MagicMock()

        mock_engine_instance = MagicMock()
        mock_engine_instance.optimize.return_value = mock_result
        mock_engine.return_value = mock_engine_instance

        return mock_encoder, mock_generator, mock_engine, mock_concept

    def test_batch_shows_job_count(self) -> None:
        """Batch should display total number of jobs."""
        runner = CliRunner()
        mock_encoder, mock_generator, mock_engine, mock_concept = self._create_mock_modules()

        with tempfile.TemporaryDirectory() as tmpdir:
            batch_path = Path(tmpdir) / "batch.yaml"
            batch_path.write_text("""
jobs:
  - target_text: [["goldfish", 1.0]]
    output: outputs/goldfish.png
  - target_text: [["flamingo", 1.0]]
    output: outputs/flamingo.png
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
                result = runner.invoke(cli, ["batch", str(batch_path)])

            # Output should indicate how many jobs are being processed
            assert "2" in result.output

    def test_batch_shows_completion_message(self) -> None:
        """Batch should show completion message when done."""
        runner = CliRunner()
        mock_encoder, mock_generator, mock_engine, mock_concept = self._create_mock_modules()

        with tempfile.TemporaryDirectory() as tmpdir:
            batch_path = Path(tmpdir) / "batch.yaml"
            batch_path.write_text("""
jobs:
  - target_text: [["goldfish", 1.0]]
    output: outputs/goldfish.png
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
                result = runner.invoke(cli, ["batch", str(batch_path)])

            # Should indicate completion
            assert "complete" in result.output.lower() or "finished" in result.output.lower()


class TestBatchDryRun:
    """Tests for batch dry-run mode."""

    def test_batch_has_dry_run_option(self) -> None:
        """Batch command should have --dry-run flag."""
        runner = CliRunner()
        result = runner.invoke(cli, ["batch", "--help"])

        assert "--dry-run" in result.output

    def test_batch_dry_run_shows_jobs_without_executing(self) -> None:
        """Dry run should display jobs without running optimization."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as tmpdir:
            batch_path = Path(tmpdir) / "batch.yaml"
            batch_path.write_text("""
jobs:
  - target_text: [["goldfish", 1.0]]
    output: outputs/goldfish.png
  - target_text: [["flamingo", 1.0]]
    output: outputs/flamingo.png
    steps: 500
""")

            result = runner.invoke(cli, ["batch", str(batch_path), "--dry-run"])

        assert result.exit_code == 0
        assert "goldfish" in result.output
        assert "flamingo" in result.output
        # Should show the steps override
        assert "500" in result.output
        # Should NOT contain "Loading models" since we're in dry-run mode
        assert "Loading models" not in result.output
