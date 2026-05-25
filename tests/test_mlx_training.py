"""
Tests for the MLX SAE training port (issue gfi).

MLX is macOS-only. These tests cover what's reachable on Linux:

* :func:`mlx_available` returns ``False`` here.
* :func:`train_sae_mlx` raises :class:`MLXUnavailableError` when called
  without MLX installed.
* The ``embed-art sae train --backend mlx`` CLI falls back to the
  PyTorch loop with a warning rather than erroring out.

The actual MLX training path is exercised only on M1/M2 Max — there
is no way to validate the parameter updates on Linux because MLX is
not installable.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from click.testing import CliRunner

from embedding_art.cli.commands.sae import _resolve_backend
from embedding_art.cli.main import cli
from embedding_art.sae.mlx_training import (
    MLXUnavailableError,
    mlx_available,
    train_sae_mlx,
)


class TestMlxAvailability:
    def test_mlx_unavailable_on_linux(self) -> None:
        # MLX is macOS-only; Linux test runs always report False.
        assert mlx_available() is False

    def test_train_sae_mlx_raises_when_missing(self, tmp_path: Path) -> None:
        # Build a tiny embeddings file.
        emb_path = tmp_path / "embs.pt"
        torch.save({"embeddings": torch.randn(4, 8)}, emb_path)

        with pytest.raises(MLXUnavailableError):
            train_sae_mlx(
                embeddings_path=emb_path,
                output_path=tmp_path / "sae",
                embed_dim=8,
                n_features=16,
                k=2,
                n_iterations=10,
            )


class TestResolveBackend:
    """Backend resolver logic."""

    def test_torch_returns_torch(self) -> None:
        assert _resolve_backend("torch") == "torch"

    def test_mlx_when_missing_falls_back(self) -> None:
        with patch("embedding_art.sae.mlx_training.mlx_available", return_value=False):
            assert _resolve_backend("mlx") == "torch"

    def test_auto_picks_torch_when_mlx_missing(self) -> None:
        with patch("embedding_art.sae.mlx_training.mlx_available", return_value=False):
            assert _resolve_backend("auto") == "torch"

    def test_auto_picks_mlx_when_available(self) -> None:
        with patch("embedding_art.sae.mlx_training.mlx_available", return_value=True):
            assert _resolve_backend("auto") == "mlx"


class TestCliFallback:
    """The CLI falls back from MLX to torch on Linux without raising."""

    def test_sae_train_with_mlx_backend_falls_back(self, tmp_path: Path) -> None:
        # Build a tiny embeddings file the existing PyTorch loop can chew on.
        emb_path = tmp_path / "embs.pt"
        torch.save({"embeddings": torch.randn(32, 16)}, emb_path)
        out_dir = tmp_path / "out"

        runner = CliRunner()
        # Patch train_sae so we don't actually run 25k Adam steps in
        # this fast test \u2014 we only care that the CLI dispatches to
        # the right backend.
        with patch("embedding_art.sae.training.train_sae") as torch_train:
            torch_train.return_value = None
            result = runner.invoke(
                cli,
                [
                    "sae",
                    "train",
                    "--embeddings",
                    str(emb_path),
                    "--output",
                    str(out_dir),
                    "--features",
                    "16",
                    "--sparsity",
                    "2",
                    "--backend",
                    "mlx",
                ],
            )
            assert result.exit_code == 0, result.output
            torch_train.assert_called_once()
            assert "MLX requested but not installed" in result.output

    def test_sae_train_default_uses_torch(self, tmp_path: Path) -> None:
        emb_path = tmp_path / "embs.pt"
        torch.save({"embeddings": torch.randn(32, 16)}, emb_path)
        out_dir = tmp_path / "out"

        runner = CliRunner()
        with patch("embedding_art.sae.training.train_sae") as torch_train:
            torch_train.return_value = None
            result = runner.invoke(
                cli,
                [
                    "sae",
                    "train",
                    "--embeddings",
                    str(emb_path),
                    "--output",
                    str(out_dir),
                    "--features",
                    "16",
                    "--sparsity",
                    "2",
                ],
            )
            assert result.exit_code == 0, result.output
            torch_train.assert_called_once()
            assert "backend=torch" in result.output
