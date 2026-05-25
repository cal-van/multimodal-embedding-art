"""
Tests for the per-modality SAE stack (M3).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from embedding_art.sae.multimodal_stack import (
    CANONICAL_MODALITIES,
    MultimodalSAEStack,
    train_multimodal_sae_stack,
)


def _make_fake_sae(embed_dim: int, n_features: int) -> torch.nn.Module:
    """A minimal stand-in for GroupSparseSAE — just a linear encode."""

    class FakeSAE(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(embed_dim, n_features)

        def forward(self, x: torch.Tensor, modality_idx: int = 0) -> torch.Tensor:
            return torch.relu(self.encoder(x))

    return FakeSAE()


class TestMultimodalSAEStack:
    """Stack container basics."""

    def test_call_dispatches_to_correct_sae(self) -> None:
        embed_dim = 32
        n_features = 64
        saes = {
            "image": _make_fake_sae(embed_dim, n_features),
            "audio": _make_fake_sae(embed_dim, n_features),
        }
        stack = MultimodalSAEStack(saes=saes, embed_dim=embed_dim, n_features=n_features)

        x = torch.randn(1, embed_dim)
        out_image = stack(x, modality="image")
        out_audio = stack(x, modality="audio")

        assert out_image.shape == (1, n_features)
        assert out_audio.shape == (1, n_features)

    def test_unknown_modality_raises(self) -> None:
        stack = MultimodalSAEStack(
            saes={"image": _make_fake_sae(32, 64)}, embed_dim=32, n_features=64
        )
        with pytest.raises(KeyError):
            stack(torch.randn(1, 32), modality="audio")

    def test_decompose_returns_sparse_dict(self) -> None:
        stack = MultimodalSAEStack(
            saes={"image": _make_fake_sae(32, 64)}, embed_dim=32, n_features=64
        )
        result = stack.decompose(torch.randn(1, 32), modality="image", top_k=5)
        assert isinstance(result, dict)
        # All values should be floats > 0 (post-ReLU, only positive features kept).
        for v in result.values():
            assert v > 0.0

    def test_1d_embedding_auto_unsqueezed(self) -> None:
        stack = MultimodalSAEStack(
            saes={"image": _make_fake_sae(32, 64)}, embed_dim=32, n_features=64
        )
        out = stack(torch.randn(32), modality="image")
        assert out.shape == (1, 64)


class TestFromDirectory:
    """Loading a stack from disk."""

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            MultimodalSAEStack.from_directory(tmp_path / "nonexistent", embed_dim=32)

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            MultimodalSAEStack.from_directory(tmp_path, embed_dim=32)

    def test_loads_present_modalities(self, tmp_path: Path) -> None:
        from embedding_art.sae.training import GroupSparseSAE

        # Save fake checkpoints for image + audio only.
        for modality in ("image", "audio"):
            modality_dir = tmp_path / modality
            modality_dir.mkdir()
            sae = GroupSparseSAE(embed_dim=32, n_features=64, k=8)
            torch.save(sae.state_dict(), modality_dir / "sae_weights.pt")

        stack = MultimodalSAEStack.from_directory(tmp_path, embed_dim=32)
        assert "image" in stack.saes
        assert "audio" in stack.saes
        assert "video" not in stack.saes
        assert "text" not in stack.saes


class TestCanonicalModalities:
    """The canonical modality list matches LanguageBind + 'shared'."""

    def test_includes_all_languagebind_modalities(self) -> None:
        assert "image" in CANONICAL_MODALITIES
        assert "audio" in CANONICAL_MODALITIES
        assert "video" in CANONICAL_MODALITIES
        assert "text" in CANONICAL_MODALITIES

    def test_includes_shared(self) -> None:
        assert "shared" in CANONICAL_MODALITIES


class TestTrainMultimodalSAEStack:
    """train_multimodal_sae_stack orchestrates per-modality train_sae calls."""

    def test_calls_train_sae_per_modality(self, tmp_path: Path, monkeypatch) -> None:
        from embedding_art.sae import multimodal_stack as stack_module

        call_log: list[tuple[str, Path, Path]] = []

        def fake_train_sae(*, embeddings_path, output_path, **kwargs):
            call_log.append(("train_sae", embeddings_path, output_path))
            return _make_fake_sae(kwargs.get("embed_dim", 32), kwargs.get("n_features", 64))

        monkeypatch.setattr(stack_module, "train_sae", fake_train_sae, raising=False)
        # Also patch the imported reference inside the function.
        import embedding_art.sae.training as training_module

        monkeypatch.setattr(training_module, "train_sae", fake_train_sae, raising=False)

        embeddings_by_modality = {
            "image": tmp_path / "image.pt",
            "audio": tmp_path / "audio.pt",
        }
        # Don't actually need real .pt files because we mocked train_sae.

        stack = train_multimodal_sae_stack(
            embeddings_by_modality=embeddings_by_modality,
            output_root=tmp_path / "stack",
            embed_dim=32,
            n_features=64,
            k=8,
            n_iterations=10,
        )

        assert isinstance(stack, MultimodalSAEStack)
        assert set(stack.saes.keys()) == {"image", "audio"}
        # Both per-modality output dirs should have been created.
        assert (tmp_path / "stack" / "image").exists()
        assert (tmp_path / "stack" / "audio").exists()
        # train_sae was called twice.
        assert len(call_log) == 2
