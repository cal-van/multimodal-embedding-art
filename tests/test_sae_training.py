"""
Unit tests for the SAE training pipeline.

All tests use small synthetic models and in-memory / tmp-path data so that no
real encoder, dataset, or GPU is required.

Sizes used throughout:
  embed_dim  = 32
  n_features = 64
  k          = 4
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from embedding_art.sae.training import (
    GroupSparseSAE,
    _get_vocab,
    collect_embeddings,
    label_features,
    train_sae,
)

# ---------------------------------------------------------------------------
# Shared constants / helpers
# ---------------------------------------------------------------------------

EMBED_DIM = 32
N_FEATURES = 64
K = 4


def _make_sae(**kwargs) -> GroupSparseSAE:
    """Return a small SAE with deterministic weights (not trained)."""
    defaults = dict(embed_dim=EMBED_DIM, n_features=N_FEATURES, k=K)
    defaults.update(kwargs)
    return GroupSparseSAE(**defaults)


def _make_batch(b: int = 4, embed_dim: int = EMBED_DIM, seed: int = 0) -> torch.Tensor:
    """Return a normalised batch of embeddings."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(b, embed_dim, generator=gen)
    return F.normalize(x, dim=-1)


def _mock_encoder(embed_dim: int = EMBED_DIM) -> MagicMock:
    """Return a mock encoder whose encode_image and encode_text return unit vectors."""
    enc = MagicMock()
    enc.embedding_dim = embed_dim

    def _encode_image(path):
        seed = hash(str(path)) % (2**32)
        gen = torch.Generator().manual_seed(seed)
        return F.normalize(torch.randn(1, embed_dim, generator=gen), dim=-1)

    def _encode_text(text):
        seed = hash(text) % (2**32)
        gen = torch.Generator().manual_seed(seed)
        return F.normalize(torch.randn(1, embed_dim, generator=gen), dim=-1)

    enc.encode_image.side_effect = _encode_image
    enc.encode_text.side_effect = _encode_text
    return enc


def _write_fake_images(directory: Path, n: int = 5) -> list[Path]:
    """Write n small PNG images to *directory* and return their paths."""
    paths: list[Path] = []
    for i in range(n):
        p = directory / f"img_{i:03d}.png"
        img = Image.new("RGB", (8, 8), color=(i * 20, i * 30, i * 40))
        img.save(p)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# GroupSparseSAE — construction
# ---------------------------------------------------------------------------


class TestGroupSparseSAEConstruction:
    """GroupSparseSAE initialises with the correct parameter shapes."""

    def test_w_enc_shape(self) -> None:
        sae = _make_sae()
        assert sae.W_enc.shape == (N_FEATURES, EMBED_DIM)

    def test_w_dec_shape(self) -> None:
        sae = _make_sae()
        assert sae.W_dec.shape == (EMBED_DIM, N_FEATURES)

    def test_bias_shape(self) -> None:
        sae = _make_sae()
        assert sae.bias.shape == (N_FEATURES,)

    def test_pre_biases_default_count(self) -> None:
        sae = _make_sae()
        assert len(sae.pre_biases) == 2  # default n_modality_biases=2

    def test_pre_biases_custom_count(self) -> None:
        sae = _make_sae(n_modality_biases=4)
        assert len(sae.pre_biases) == 4

    def test_pre_bias_shape(self) -> None:
        sae = _make_sae()
        for pb in sae.pre_biases:
            assert pb.shape == (EMBED_DIM,)

    def test_k_validation(self) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            GroupSparseSAE(embed_dim=EMBED_DIM, n_features=N_FEATURES, k=0)

    def test_n_modality_biases_validation(self) -> None:
        with pytest.raises(ValueError, match="n_modality_biases must be >= 1"):
            GroupSparseSAE(embed_dim=EMBED_DIM, n_features=N_FEATURES, k=K, n_modality_biases=0)

    def test_is_nn_module(self) -> None:
        import torch.nn as nn

        assert isinstance(_make_sae(), nn.Module)


# ---------------------------------------------------------------------------
# GroupSparseSAE — forward (encode)
# ---------------------------------------------------------------------------


class TestGroupSparseSAEForward:
    """GroupSparseSAE.forward() produces correctly shaped sparse activations."""

    def test_output_shape(self) -> None:
        sae = _make_sae()
        x = _make_batch(b=3)
        z = sae(x)
        assert z.shape == (3, N_FEATURES)

    def test_exactly_k_nonzero_per_row(self) -> None:
        sae = _make_sae()
        x = _make_batch(b=8)
        z = sae(x)
        nonzero_counts = (z != 0).sum(dim=-1)
        # Each row must have exactly k non-zero activations.
        assert (nonzero_counts == K).all(), nonzero_counts.tolist()

    def test_activations_are_non_negative(self) -> None:
        """Post-ReLU activations must be >= 0."""
        sae = _make_sae()
        x = _make_batch(b=4)
        z = sae(x)
        assert (z >= 0).all()

    def test_modality_idx_selects_pre_bias(self) -> None:
        """Different modality indices produce different outputs."""
        sae = _make_sae(n_modality_biases=2)
        # Initialise pre_biases to different values.
        with torch.no_grad():
            sae.pre_biases[0].fill_(0.0)
            sae.pre_biases[1].fill_(1.0)
        x = _make_batch(b=2)
        z0 = sae(x, modality_idx=0)
        z1 = sae(x, modality_idx=1)
        # They should not be identical.
        assert not torch.allclose(z0, z1)

    def test_k_larger_than_features_clips(self) -> None:
        """If k >= n_features, all post-ReLU activations are kept."""
        sae = GroupSparseSAE(embed_dim=8, n_features=4, k=10)
        x = F.normalize(torch.randn(1, 8), dim=-1)
        z = sae(x)
        assert z.shape == (1, 4)

    def test_single_sample_batch(self) -> None:
        sae = _make_sae()
        x = _make_batch(b=1)
        z = sae(x)
        assert z.shape == (1, N_FEATURES)


# ---------------------------------------------------------------------------
# GroupSparseSAE — decode
# ---------------------------------------------------------------------------


class TestGroupSparseSAEDecode:
    """GroupSparseSAE.decode() reconstructs embeddings with the correct shape."""

    def test_output_shape(self) -> None:
        sae = _make_sae()
        z = torch.zeros(3, N_FEATURES)
        x_hat = sae.decode(z)
        assert x_hat.shape == (3, EMBED_DIM)

    def test_zero_activations_give_pre_bias(self) -> None:
        """Decoding all-zero activations should return the pre_bias vector."""
        sae = _make_sae()
        with torch.no_grad():
            sae.pre_biases[0].fill_(0.5)
        z = torch.zeros(1, N_FEATURES)
        x_hat = sae.decode(z, modality_idx=0)
        expected = torch.full((1, EMBED_DIM), 0.5)
        assert torch.allclose(x_hat, expected, atol=1e-5)

    def test_encode_decode_round_trip_shape(self) -> None:
        sae = _make_sae()
        x = _make_batch(b=5)
        z = sae(x)
        x_hat = sae.decode(z)
        assert x_hat.shape == x.shape


# ---------------------------------------------------------------------------
# GroupSparseSAE — reconstruction_loss
# ---------------------------------------------------------------------------


class TestReconstructionLoss:
    """reconstruction_loss returns a non-negative scalar."""

    def test_loss_is_scalar(self) -> None:
        sae = _make_sae()
        x = _make_batch()
        z = sae(x)
        x_hat = sae.decode(z)
        loss = sae.reconstruction_loss(x, x_hat)
        assert loss.shape == ()

    def test_loss_is_non_negative(self) -> None:
        sae = _make_sae()
        x = _make_batch()
        z = sae(x)
        x_hat = sae.decode(z)
        loss = sae.reconstruction_loss(x, x_hat)
        assert loss.item() >= 0.0

    def test_perfect_reconstruction_gives_zero_loss(self) -> None:
        sae = _make_sae()
        x = _make_batch()
        loss = sae.reconstruction_loss(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# GroupSparseSAE — group_sparse_loss
# ---------------------------------------------------------------------------


class TestGroupSparseLoss:
    """group_sparse_loss implements the L_{2,1} norm correctly."""

    def test_loss_is_scalar(self) -> None:
        sae = _make_sae()
        z_x = torch.rand(4, N_FEATURES)
        z_y = torch.rand(4, N_FEATURES)
        loss = sae.group_sparse_loss(z_x, z_y)
        assert loss.shape == ()

    def test_loss_is_non_negative(self) -> None:
        sae = _make_sae()
        z_x = torch.rand(4, N_FEATURES)
        z_y = torch.rand(4, N_FEATURES)
        assert sae.group_sparse_loss(z_x, z_y).item() >= 0.0

    def test_zero_activations_give_near_zero_loss(self) -> None:
        sae = _make_sae()
        z = torch.zeros(4, N_FEATURES)
        loss = sae.group_sparse_loss(z, z)
        # sqrt(0 + 0 + eps) * n_features * batch_size — should be very small.
        assert loss.item() < 1.0

    def test_identical_activations_greater_than_zero_pair(self) -> None:
        """Non-zero activations must produce a positive loss."""
        sae = _make_sae()
        z = torch.ones(1, N_FEATURES)
        loss = sae.group_sparse_loss(z, z)
        assert loss.item() > 0.0

    def test_symmetry(self) -> None:
        """L_{2,1} is symmetric: loss(z_x, z_y) == loss(z_y, z_x)."""
        sae = _make_sae()
        gen = torch.Generator().manual_seed(7)
        z_x = torch.rand(4, N_FEATURES, generator=gen)
        z_y = torch.rand(4, N_FEATURES, generator=gen)
        assert sae.group_sparse_loss(z_x, z_y).item() == pytest.approx(
            sae.group_sparse_loss(z_y, z_x).item(), abs=1e-5
        )


# ---------------------------------------------------------------------------
# GroupSparseSAE — normalize_decoder
# ---------------------------------------------------------------------------


class TestNormalizeDecoder:
    """normalize_decoder() projects W_dec columns back to unit norm."""

    def test_columns_are_unit_norm_after_call(self) -> None:
        sae = _make_sae()
        # Manually break unit-norm constraint.
        with torch.no_grad():
            sae.W_dec.data *= 5.0

        sae.normalize_decoder()

        col_norms = sae.W_dec.data.norm(dim=0)  # [n_features]
        assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-5)

    def test_no_grad_side_effect(self) -> None:
        """normalize_decoder should not create graph nodes."""
        sae = _make_sae()
        sae.normalize_decoder()
        assert sae.W_dec.grad_fn is None


# ---------------------------------------------------------------------------
# collect_embeddings
# ---------------------------------------------------------------------------


class TestCollectEmbeddings:
    """collect_embeddings encodes images and saves a .pt file."""

    def test_creates_output_file(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "images"
        dataset_dir.mkdir()
        _write_fake_images(dataset_dir, n=3)

        out_file = tmp_path / "embeddings.pt"
        encoder = _mock_encoder()

        collect_embeddings(encoder, dataset_dir, out_file)

        assert out_file.exists()

    def test_saved_tensor_shape(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "images"
        dataset_dir.mkdir()
        _write_fake_images(dataset_dir, n=5)

        out_file = tmp_path / "embeddings.pt"
        encoder = _mock_encoder()

        collect_embeddings(encoder, dataset_dir, out_file)

        data = torch.load(out_file, weights_only=True)
        assert "embeddings" in data
        assert data["embeddings"].shape == (5, EMBED_DIM)

    def test_embeddings_are_normalised(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "images"
        dataset_dir.mkdir()
        _write_fake_images(dataset_dir, n=4)

        out_file = tmp_path / "embeddings.pt"
        encoder = _mock_encoder()

        collect_embeddings(encoder, dataset_dir, out_file)

        data = torch.load(out_file, weights_only=True)
        norms = data["embeddings"].norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_max_samples_limits_output(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "images"
        dataset_dir.mkdir()
        _write_fake_images(dataset_dir, n=10)

        out_file = tmp_path / "embeddings.pt"
        encoder = _mock_encoder()

        collect_embeddings(encoder, dataset_dir, out_file, max_samples=3)

        data = torch.load(out_file, weights_only=True)
        assert data["embeddings"].shape[0] == 3

    def test_missing_dataset_raises(self, tmp_path: Path) -> None:
        encoder = _mock_encoder()
        with pytest.raises(FileNotFoundError):
            collect_embeddings(encoder, tmp_path / "does_not_exist", tmp_path / "out.pt")

    def test_empty_dataset_raises(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "empty"
        dataset_dir.mkdir()
        encoder = _mock_encoder()
        with pytest.raises(ValueError, match="No image files found"):
            collect_embeddings(encoder, dataset_dir, tmp_path / "out.pt")

    def test_returns_resolved_path(self, tmp_path: Path) -> None:
        dataset_dir = tmp_path / "images"
        dataset_dir.mkdir()
        _write_fake_images(dataset_dir, n=2)
        out_file = tmp_path / "embeddings.pt"
        encoder = _mock_encoder()
        returned = collect_embeddings(encoder, dataset_dir, out_file)
        assert returned == out_file.resolve()


# ---------------------------------------------------------------------------
# train_sae
# ---------------------------------------------------------------------------


class TestTrainSAE:
    """train_sae runs a short training loop and saves weights."""

    def _save_fake_embeddings(self, path: Path, n: int = 50) -> None:
        """Write a small synthetic embeddings .pt file."""
        gen = torch.Generator().manual_seed(42)
        emb = F.normalize(torch.randn(n, EMBED_DIM, generator=gen), dim=-1)
        torch.save({"embeddings": emb}, path)

    def test_returns_group_sparse_sae(self, tmp_path: Path) -> None:
        emb_file = tmp_path / "emb.pt"
        self._save_fake_embeddings(emb_file)
        sae = train_sae(
            emb_file,
            tmp_path / "out",
            embed_dim=EMBED_DIM,
            n_features=N_FEATURES,
            k=K,
            n_iterations=5,
            batch_size=8,
        )
        assert isinstance(sae, GroupSparseSAE)

    def test_saves_weights_file(self, tmp_path: Path) -> None:
        emb_file = tmp_path / "emb.pt"
        self._save_fake_embeddings(emb_file)
        out_dir = tmp_path / "out"
        train_sae(
            emb_file,
            out_dir,
            embed_dim=EMBED_DIM,
            n_features=N_FEATURES,
            k=K,
            n_iterations=5,
            batch_size=8,
        )
        assert (out_dir / "sae_weights.pt").exists()

    def test_weights_are_loadable(self, tmp_path: Path) -> None:
        emb_file = tmp_path / "emb.pt"
        self._save_fake_embeddings(emb_file)
        out_dir = tmp_path / "out"
        train_sae(
            emb_file,
            out_dir,
            embed_dim=EMBED_DIM,
            n_features=N_FEATURES,
            k=K,
            n_iterations=5,
            batch_size=8,
        )
        state = torch.load(out_dir / "sae_weights.pt", weights_only=True)
        assert "W_enc" in state
        assert "W_dec" in state

    def test_embed_dim_mismatch_raises(self, tmp_path: Path) -> None:
        emb_file = tmp_path / "emb.pt"
        self._save_fake_embeddings(emb_file, n=20)
        with pytest.raises(ValueError, match="embed_dim mismatch"):
            train_sae(emb_file, tmp_path / "out", embed_dim=999, n_iterations=2, batch_size=4)

    def test_sae_in_eval_mode_after_training(self, tmp_path: Path) -> None:
        emb_file = tmp_path / "emb.pt"
        self._save_fake_embeddings(emb_file)
        sae = train_sae(
            emb_file,
            tmp_path / "out",
            embed_dim=EMBED_DIM,
            n_features=N_FEATURES,
            k=K,
            n_iterations=3,
            batch_size=8,
        )
        assert not sae.training


# ---------------------------------------------------------------------------
# label_features
# ---------------------------------------------------------------------------


class TestLabelFeatures:
    """label_features returns one label string per feature."""

    def _save_trained_sae(self, directory: Path) -> None:
        """Save a minimal SAE state dict so label_features can load it."""
        directory.mkdir(parents=True, exist_ok=True)
        sae = _make_sae()
        torch.save(sae.state_dict(), directory / "sae_weights.pt")

    def test_returns_list_of_strings(self, tmp_path: Path) -> None:
        sae_dir = tmp_path / "sae"
        self._save_trained_sae(sae_dir)
        encoder = _mock_encoder()
        labels = label_features(sae_dir, encoder, vocab_size=20)
        assert isinstance(labels, list)
        assert all(isinstance(lbl, str) for lbl in labels)

    def test_one_label_per_feature(self, tmp_path: Path) -> None:
        sae_dir = tmp_path / "sae"
        self._save_trained_sae(sae_dir)
        encoder = _mock_encoder()
        labels = label_features(sae_dir, encoder, vocab_size=20)
        assert len(labels) == N_FEATURES

    def test_labels_are_from_vocab(self, tmp_path: Path) -> None:
        sae_dir = tmp_path / "sae"
        self._save_trained_sae(sae_dir)
        encoder = _mock_encoder()
        vocab = _get_vocab(20)
        labels = label_features(sae_dir, encoder, vocab_size=20)
        for lbl in labels:
            assert lbl in vocab

    def test_accepts_direct_weights_path(self, tmp_path: Path) -> None:
        sae_dir = tmp_path / "sae"
        self._save_trained_sae(sae_dir)
        encoder = _mock_encoder()
        # Pass the weights file directly instead of the directory.
        labels = label_features(sae_dir / "sae_weights.pt", encoder, vocab_size=10)
        assert len(labels) == N_FEATURES

    def test_missing_weights_raises(self, tmp_path: Path) -> None:
        encoder = _mock_encoder()
        with pytest.raises(FileNotFoundError):
            label_features(tmp_path / "no_such_dir", encoder, vocab_size=5)


# ---------------------------------------------------------------------------
# _get_vocab helper
# ---------------------------------------------------------------------------


class TestGetVocab:
    """_get_vocab returns a deduplicated list capped at vocab_size."""

    def test_returns_list_of_strings(self) -> None:
        vocab = _get_vocab(10)
        assert isinstance(vocab, list)
        assert all(isinstance(w, str) for w in vocab)

    def test_length_capped_at_vocab_size(self) -> None:
        vocab = _get_vocab(5)
        assert len(vocab) == 5

    def test_no_duplicates(self) -> None:
        vocab = _get_vocab(200)
        assert len(vocab) == len(set(vocab))

    def test_returns_full_list_when_vocab_size_large(self) -> None:
        from embedding_art.sae.training import _VOCAB_BASE

        vocab = _get_vocab(10000)
        assert len(vocab) == len(set(_VOCAB_BASE))
