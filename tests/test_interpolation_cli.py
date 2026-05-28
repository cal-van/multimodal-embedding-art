"""
Tests for the interpolate-v2 CLI command.

TDD red phase: structural/option tests — no real optimization is run.
"""

from __future__ import annotations

from click.testing import CliRunner


class TestInterpolateV2Command:
    """The interpolate-v2 command must be registered and have the expected options."""

    def _get_cli(self):
        from embedding_art.cli.main import cli

        return cli

    def test_command_is_registered(self):
        """interpolate-v2 must appear in the root CLI group."""
        cli = self._get_cli()
        assert (
            "interpolate-v2" in cli.commands
        ), f"interpolate-v2 not found in CLI commands: {list(cli.commands.keys())}"

    def test_help_text_is_accessible(self):
        """--help should not raise an error."""
        runner = CliRunner()
        cli = self._get_cli()
        result = runner.invoke(cli, ["interpolate-v2", "--help"])
        assert result.exit_code == 0, f"--help failed with: {result.output}"

    def test_option_concept_a_required(self):
        """Missing --concept-a / -a should produce a 'Missing' error."""
        runner = CliRunner()
        cli = self._get_cli()
        result = runner.invoke(cli, ["interpolate-v2", "-b", "flamingo"])
        # Click uses exit_code=2 for usage errors
        assert result.exit_code == 2
        assert "concept-a" in result.output.lower() or "missing" in result.output.lower()

    def test_option_concept_b_required(self):
        """Missing --concept-b / -b should produce a 'Missing' error."""
        runner = CliRunner()
        cli = self._get_cli()
        result = runner.invoke(cli, ["interpolate-v2", "-a", "goldfish"])
        assert result.exit_code == 2
        assert "concept-b" in result.output.lower() or "missing" in result.output.lower()

    def test_option_steps_default(self):
        """--help output should mention the default value of 10 for --steps."""
        runner = CliRunner()
        cli = self._get_cli()
        result = runner.invoke(cli, ["interpolate-v2", "--help"])
        assert "10" in result.output, "Expected default steps=10 in --help output"

    def test_option_output_modality_default(self):
        """--help output should show 'image' as the default output modality."""
        runner = CliRunner()
        cli = self._get_cli()
        result = runner.invoke(cli, ["interpolate-v2", "--help"])
        assert "image" in result.output

    def test_option_device_present(self):
        """--device option should be listed in --help."""
        runner = CliRunner()
        cli = self._get_cli()
        result = runner.invoke(cli, ["interpolate-v2", "--help"])
        assert "--device" in result.output

    def test_option_opt_steps_present(self):
        """--opt-steps option should be listed in --help."""
        runner = CliRunner()
        cli = self._get_cli()
        result = runner.invoke(cli, ["interpolate-v2", "--help"])
        assert "--opt-steps" in result.output

    def test_option_output_dir_present(self):
        """--output-dir option should be listed in --help."""
        runner = CliRunner()
        cli = self._get_cli()
        result = runner.invoke(cli, ["interpolate-v2", "--help"])
        assert "--output-dir" in result.output

    def test_option_encoder_present(self):
        """--encoder option should be listed in --help."""
        runner = CliRunner()
        cli = self._get_cli()
        result = runner.invoke(cli, ["interpolate-v2", "--help"])
        assert "--encoder" in result.output

    def test_short_options_a_and_b_listed(self):
        """-a and -b short options appear in --help."""
        runner = CliRunner()
        cli = self._get_cli()
        result = runner.invoke(cli, ["interpolate-v2", "--help"])
        assert "-a" in result.output
        assert "-b" in result.output
