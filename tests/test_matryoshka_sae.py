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
