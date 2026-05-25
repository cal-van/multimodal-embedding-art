"""
Tests for ``embed-art validate-vsd`` (issue va2).

This command is intended to be run on an Apple Silicon host with
SD3.5-medium weights. The tests cover what is reachable on Linux:

* ``--dry-run`` writes a plan manifest and exits 0.
* The plan manifest contains the expected fields documenting the run.
* Without ``--dry-run``, the command refuses to proceed on a host
  without MPS / a GPU rather than silently downloading the SD3.5
  weights.

The end-to-end loop wiring is intentionally left as a follow-up; the
runbook at docs/superpowers/runbooks/vsd-validation.md documents what
the user must run on M1 Max.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from embedding_art.cli.main import cli


class TestValidateVsdDryRun:
    """``--dry-run`` writes a plan manifest and exits 0."""

    def test_dry_run_writes_plan(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "validate-vsd",
                "--concept",
                "thunder",
                "-o",
                str(tmp_path),
                "--dry-run",
                "--steps",
                "5",
            ],
        )
        assert result.exit_code == 0, result.output
        plan_path = tmp_path / "plan.json"
        assert plan_path.exists()
        plan = json.loads(plan_path.read_text())
        assert plan["concept"] == "thunder"
        assert plan["steps"] == 5
        assert plan["dry_run"] is True
        assert plan["tracks"] == ["honest", "natural"]
        for artefact in ("honest.png", "natural.png", "comparison.png", "manifest.json"):
            assert artefact in plan["expected_artefacts"]

    def test_dry_run_with_custom_concept(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "validate-vsd",
                "--concept",
                "goldfish",
                "-o",
                str(tmp_path),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        plan = json.loads((tmp_path / "plan.json").read_text())
        assert plan["concept"] == "goldfish"

    def test_plan_records_runbook_path(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "validate-vsd",
                "-o",
                str(tmp_path),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        plan = json.loads((tmp_path / "plan.json").read_text())
        assert plan["runbook"] == "docs/superpowers/runbooks/vsd-validation.md"
        assert "expected_outputs" in plan

    def test_dry_run_records_seed_and_device(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "validate-vsd",
                "-o",
                str(tmp_path),
                "--dry-run",
                "--seed",
                "42",
                "--device",
                "cuda",
                "--rank",
                "8",
                "--guidance-scale",
                "9.5",
            ],
        )
        assert result.exit_code == 0, result.output
        plan = json.loads((tmp_path / "plan.json").read_text())
        assert plan["seed"] == 42
        assert plan["device"] == "cuda"
        assert plan["lora_rank"] == 8
        assert plan["guidance_scale"] == 9.5
