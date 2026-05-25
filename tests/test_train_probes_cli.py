"""Tests for the ``embed-art train-probes`` CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from click.testing import CliRunner

from embedding_art.cli.main import cli


def _build_mock_encoder() -> MagicMock:
    """Mock encoder that produces a deterministic 32-d embedding per text.

    Two-cluster construction: "animal"-prefixed texts cluster along
    axis 0, "vehicle"-prefixed cluster along axis 1. This gives the
    binary probe a real signal to learn.
    """

    def encode_text(text: str) -> torch.Tensor:
        emb = torch.zeros(1, 32)
        if "animal" in text:
            emb[0, 0] = 3.0 + 0.1 * len(text)
        elif "vehicle" in text:
            emb[0, 1] = 3.0 + 0.1 * len(text)
        elif "instrument" in text:
            emb[0, 2] = 3.0 + 0.1 * len(text)
        return emb

    encoder = MagicMock()
    encoder.encode_text.side_effect = encode_text
    return encoder


def test_train_probes_cli_writes_manifest_and_metrics(tmp_path: Path) -> None:
    manifest = {
        "encoder": "languagebind",
        "device": "cpu",
        "probes": [
            {
                "name": "is_animal",
                "multilabel": True,
                "examples": [
                    {"text": "animal goldfish", "labels": {"is_animal": 1.0}},
                    {"text": "animal flamingo", "labels": {"is_animal": 1.0}},
                    {"text": "animal lion", "labels": {"is_animal": 1.0}},
                    {"text": "animal horse", "labels": {"is_animal": 1.0}},
                    {"text": "vehicle car", "labels": {"is_animal": 0.0}},
                    {"text": "vehicle bicycle", "labels": {"is_animal": 0.0}},
                    {"text": "vehicle plane", "labels": {"is_animal": 0.0}},
                    {"text": "vehicle ship", "labels": {"is_animal": 0.0}},
                ],
            },
            {
                "name": "category",
                "multiclass": ["animal", "vehicle", "instrument"],
                "examples": [
                    {"text": "animal goldfish", "label": "animal"},
                    {"text": "animal flamingo", "label": "animal"},
                    {"text": "animal horse", "label": "animal"},
                    {"text": "vehicle car", "label": "vehicle"},
                    {"text": "vehicle bicycle", "label": "vehicle"},
                    {"text": "vehicle plane", "label": "vehicle"},
                    {"text": "instrument piano", "label": "instrument"},
                    {"text": "instrument guitar", "label": "instrument"},
                    {"text": "instrument drum", "label": "instrument"},
                ],
            },
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    out_dir = tmp_path / "probes_out"

    registry = MagicMock()
    registry.load.return_value = _build_mock_encoder()

    runner = CliRunner()
    with patch(
        "embedding_art.encoders.defaults.create_default_registry",
        return_value=registry,
    ):
        result = runner.invoke(
            cli,
            [
                "train-probes",
                "--manifest",
                str(manifest_path),
                "--out-dir",
                str(out_dir),
                "--epochs",
                "50",
                "--seed",
                "0",
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output

    # Both probes were persisted.
    index = json.loads((out_dir / "index.json").read_text())
    assert set(index.keys()) == {"is_animal", "category"}
    assert (out_dir / "is_animal.pt").is_file()
    assert (out_dir / "category.pt").is_file()

    # Metrics summary present and contains both probes.
    metrics = json.loads((out_dir / "training_metrics.json").read_text())
    assert set(metrics.keys()) == {"is_animal", "category"}
    assert metrics["is_animal"]["num_epochs"] == 50
    assert metrics["category"]["num_epochs"] == 50


def test_train_probes_cli_rejects_missing_manifest(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["train-probes", "--manifest", str(tmp_path / "nope.json")],
        catch_exceptions=False,
    )
    assert result.exit_code != 0
