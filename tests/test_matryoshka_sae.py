"""Tests for ``embedding_art.sae.matryoshka.MatryoshkaSAE``."""

from __future__ import annotations

import pytest
import torch

from embedding_art.sae.matryoshka import MatryoshkaSAE, train_matryoshka_sae


class TestMatryoshkaSAEConstruction:
    def test_rejects_non_increasing_nests(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            MatryoshkaSAE(embed_dim=8, nested_sizes=[4, 4, 8], k=2)

    def test_rejects_k_larger_than_smallest_nest(self) -> None:
        with pytest.raises(ValueError, match="smallest nest"):
            MatryoshkaSAE(embed_dim=8, nested_sizes=[4, 16], k=8)

    def test_rejects_empty_nests(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            MatryoshkaSAE(embed_dim=8, nested_sizes=[], k=1)

    def test_nest_weights_default_to_uniform(self) -> None:
        sae = MatryoshkaSAE(embed_dim=8, nested_sizes=[4, 8, 16], k=2)
        assert sae.nest_weights.shape == (3,)
        assert torch.allclose(sae.nest_weights, torch.tensor([1 / 3, 1 / 3, 1 / 3]))

    def test_nest_weights_length_must_match(self) -> None:
        with pytest.raises(ValueError, match="nest_weights length"):
            MatryoshkaSAE(embed_dim=8, nested_sizes=[4, 8, 16], k=2, nest_weights=[0.5, 0.5])


class TestMatryoshkaSAEEncoding:
    def test_smaller_nest_zeroes_unreachable_features(self) -> None:
        """encode_nest(x, n_1) must produce activations only in the first n_1 columns."""
        sae = MatryoshkaSAE(embed_dim=4, nested_sizes=[2, 4, 8], k=2)
        x = torch.randn(3, 4)
        z = sae.encode_nest(x, nest_size=2)
        assert z.shape == (3, 8)
        # Columns 2: are all zero.
        assert torch.all(z[:, 2:] == 0)

    def test_forward_uses_largest_nest(self) -> None:
        sae = MatryoshkaSAE(embed_dim=4, nested_sizes=[2, 4, 8], k=2)
        x = torch.randn(3, 4)
        z = sae.forward(x)
        assert z.shape == (3, 8)
        # k=2 means exactly 2 non-zero features per row.
        non_zero = (z != 0).sum(dim=-1)
        assert torch.all(non_zero == 2)

    def test_decode_round_trip(self) -> None:
        sae = MatryoshkaSAE(embed_dim=8, nested_sizes=[4, 8], k=2)
        x = torch.randn(5, 8)
        z = sae.forward(x)
        x_hat = sae.decode(z)
        assert x_hat.shape == x.shape


class TestMatryoshkaLoss:
    def test_loss_is_scalar_and_finite(self) -> None:
        sae = MatryoshkaSAE(embed_dim=8, nested_sizes=[4, 8, 16], k=2)
        x = torch.randn(8, 8)
        loss = sae.matryoshka_loss(x)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_training_reduces_loss(self) -> None:
        """Confidence test: a tiny training run should reduce loss
        materially. Uses a deterministic seed and a small embedding
        matrix so the test stays under 100ms."""
        torch.manual_seed(0)
        embeddings = torch.randn(64, 8)
        sae = MatryoshkaSAE(embed_dim=8, nested_sizes=[2, 8], k=2)
        x = embeddings[:8]
        initial_loss = sae.matryoshka_loss(x).item()

        trained = train_matryoshka_sae(
            embeddings=embeddings,
            nested_sizes=[2, 8],
            k=2,
            batch_size=16,
            n_iterations=200,
            lr=5e-3,
        )
        final_loss = trained.matryoshka_loss(x).item()
        assert final_loss < initial_loss


class TestMatryoshkaSAEDecoderNorm:
    def test_normalize_decoder_unit_columns(self) -> None:
        sae = MatryoshkaSAE(embed_dim=8, nested_sizes=[4, 8], k=2)
        with torch.no_grad():
            sae.W_dec.data *= 10  # de-normalise
        sae.normalize_decoder()
        norms = sae.W_dec.norm(dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# ---------------------------------------------------------------------------
# Pipeline + CLI integration
# ---------------------------------------------------------------------------


class TestMatryoshkaTrainingPipeline:
    """Cover the production-grade ``train_matryoshka_sae_pipeline``."""

    def _write_embeddings(self, tmp_path, embed_dim: int = 16, n_samples: int = 64):
        path = tmp_path / "embeddings.pt"
        torch.manual_seed(0)
        torch.save({"embeddings": torch.randn(n_samples, embed_dim)}, path)
        return path

    def test_pipeline_saves_weights_and_meta(self, tmp_path) -> None:
        from embedding_art.sae.training import train_matryoshka_sae_pipeline

        embeddings_path = self._write_embeddings(tmp_path)
        output_dir = tmp_path / "sae"
        train_matryoshka_sae_pipeline(
            embeddings_path=embeddings_path,
            output_path=output_dir,
            embed_dim=16,
            nested_sizes=(4, 8, 16),
            k=2,
            n_iterations=4,
            batch_size=8,
            log_every=10_000,
        )
        assert (output_dir / "sae_weights.pt").exists()
        meta_path = output_dir / "sae_meta.json"
        assert meta_path.exists()

        import json

        meta = json.loads(meta_path.read_text())
        assert meta["kind"] == "matryoshka"
        assert meta["nested_sizes"] == [4, 8, 16]
        assert meta["k"] == 2
        assert meta["embed_dim"] == 16
        assert meta["n_features"] == 16

    def test_pipeline_rejects_embed_dim_mismatch(self, tmp_path) -> None:
        from embedding_art.sae.training import train_matryoshka_sae_pipeline

        embeddings_path = self._write_embeddings(tmp_path, embed_dim=16)
        with pytest.raises(ValueError, match="embed_dim mismatch"):
            train_matryoshka_sae_pipeline(
                embeddings_path=embeddings_path,
                output_path=tmp_path / "sae",
                embed_dim=32,
                nested_sizes=(4, 8),
                k=2,
                n_iterations=1,
            )

    def test_load_sae_from_dir_round_trips_matryoshka(self, tmp_path) -> None:
        from embedding_art.sae.matryoshka import MatryoshkaSAE
        from embedding_art.sae.training import (
            load_sae_from_dir,
            train_matryoshka_sae_pipeline,
        )

        embeddings_path = self._write_embeddings(tmp_path)
        output_dir = tmp_path / "sae"
        train_matryoshka_sae_pipeline(
            embeddings_path=embeddings_path,
            output_path=output_dir,
            embed_dim=16,
            nested_sizes=(4, 8, 16),
            k=2,
            n_iterations=4,
            batch_size=8,
            log_every=10_000,
        )
        loaded = load_sae_from_dir(output_dir)
        assert isinstance(loaded, MatryoshkaSAE)
        assert loaded.nested_sizes == [4, 8, 16]
        assert loaded.k == 2

    def test_load_sae_from_dir_falls_back_to_groupsparse(self, tmp_path) -> None:
        """Older sae_weights.pt without sae_meta.json should still load."""
        from embedding_art.sae.training import (
            GroupSparseSAE,
            load_sae_from_dir,
            train_sae,
        )

        embeddings_path = self._write_embeddings(tmp_path)
        output_dir = tmp_path / "sae"
        train_sae(
            embeddings_path=embeddings_path,
            output_path=output_dir,
            embed_dim=16,
            n_features=16,
            k=2,
            n_iterations=4,
            batch_size=8,
        )
        # No sae_meta.json was written by train_sae — that's the v3 layout.
        assert not (output_dir / "sae_meta.json").exists()
        loaded = load_sae_from_dir(output_dir)
        assert isinstance(loaded, GroupSparseSAE)


class TestMatryoshkaCLIWiring:
    """``embed-art sae train --sae-type matryoshka`` dispatch."""

    def test_cli_dispatches_to_matryoshka_pipeline(self, tmp_path, monkeypatch) -> None:
        """The CLI parses ``--sae-type matryoshka`` and forwards the right args.

        We patch the pipeline entry point so the test stays fast — the
        actual training loop is covered above by
        :class:`TestMatryoshkaTrainingPipeline`.
        """
        from click.testing import CliRunner

        from embedding_art.cli.commands.sae import train as train_cli

        embeddings_path = tmp_path / "embeddings.pt"
        torch.save({"embeddings": torch.randn(8, 16)}, embeddings_path)

        captured: dict = {}

        def fake_pipeline(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(
            "embedding_art.sae.training.train_matryoshka_sae_pipeline",
            fake_pipeline,
        )

        runner = CliRunner()
        result = runner.invoke(
            train_cli,
            [
                "--embeddings",
                str(embeddings_path),
                "--output",
                str(tmp_path / "sae"),
                "--sae-type",
                "matryoshka",
                "--nested-sizes",
                "4,8",
                "--sparsity",
                "2",
            ],
        )
        assert result.exit_code == 0, result.output
        assert captured["nested_sizes"] == (4, 8)
        assert captured["k"] == 2
        assert captured["embed_dim"] == 16

    def test_cli_rejects_bad_nested_sizes(self, tmp_path) -> None:
        from click.testing import CliRunner

        from embedding_art.cli.commands.sae import train as train_cli

        embeddings_path = tmp_path / "embeddings.pt"
        torch.save({"embeddings": torch.randn(8, 16)}, embeddings_path)

        runner = CliRunner()
        result = runner.invoke(
            train_cli,
            [
                "--embeddings",
                str(embeddings_path),
                "--output",
                str(tmp_path / "sae"),
                "--sae-type",
                "matryoshka",
                "--nested-sizes",
                "not-an-int",
                "--sparsity",
                "2",
            ],
        )
        assert result.exit_code != 0
        assert "nested-sizes" in result.output.lower() or "not-an-int" in result.output
