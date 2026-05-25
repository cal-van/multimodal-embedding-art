"""Tests for ``collect_multimodal_embeddings`` and its CLI wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from click.testing import CliRunner

from embedding_art.cli.main import cli
from embedding_art.sae.training import collect_multimodal_embeddings


class _FakeMultimodalEncoder:
    """Encoder that emits a deterministic 8-d embedding per call."""

    def __init__(self, embed_dim: int = 8) -> None:
        self.embed_dim = embed_dim

    def _emb(self, key: str) -> torch.Tensor:
        torch.manual_seed(abs(hash(key)) % (2**31))
        return torch.randn(1, self.embed_dim)

    def encode_image(self, path) -> torch.Tensor:  # noqa: ANN001
        return self._emb(f"image:{path}")

    def encode_audio(self, path) -> torch.Tensor:  # noqa: ANN001
        return self._emb(f"audio:{path}")

    def encode_video(self, path) -> torch.Tensor:  # noqa: ANN001
        return self._emb(f"video:{path}")

    def encode_text(self, text: str) -> torch.Tensor:
        return self._emb(f"text:{text}")


def _populate_dataset(root: Path) -> None:
    """Create a mixed-content dataset on disk."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "a.jpg").write_bytes(b"\x00")
    (root / "b.png").write_bytes(b"\x00")
    (root / "c.wav").write_bytes(b"\x00")
    (root / "d.mp4").write_bytes(b"\x00")
    (root / "skip.txt").write_text("not a recognized media extension")
    (root / "texts.txt").write_text("goldfish\nthunder\nsilence\n")


class TestCollectMultimodalEmbeddings:
    def test_writes_one_pt_per_modality(self, tmp_path: Path) -> None:
        ds = tmp_path / "ds"
        out = tmp_path / "out"
        _populate_dataset(ds)
        encoder = _FakeMultimodalEncoder()

        saved = collect_multimodal_embeddings(encoder=encoder, dataset_path=ds, output_dir=out)

        assert set(saved.keys()) == {"image", "audio", "video", "text"}
        # Image file count == 2, audio == 1, video == 1, text lines == 3
        image_emb = torch.load(saved["image"])["embeddings"]
        audio_emb = torch.load(saved["audio"])["embeddings"]
        video_emb = torch.load(saved["video"])["embeddings"]
        text_emb = torch.load(saved["text"])["embeddings"]
        assert image_emb.shape == (2, 8)
        assert audio_emb.shape == (1, 8)
        assert video_emb.shape == (1, 8)
        assert text_emb.shape == (3, 8)
        # All normalised to unit length.
        assert torch.allclose(image_emb.norm(dim=-1), torch.ones(2), atol=1e-5)

    def test_subset_modalities_skips_others(self, tmp_path: Path) -> None:
        ds = tmp_path / "ds"
        out = tmp_path / "out"
        _populate_dataset(ds)
        saved = collect_multimodal_embeddings(
            encoder=_FakeMultimodalEncoder(),
            dataset_path=ds,
            output_dir=out,
            modalities=["image", "text"],
        )
        assert set(saved.keys()) == {"image", "text"}
        assert not (out / "audio_embeddings.pt").exists()
        assert not (out / "video_embeddings.pt").exists()

    def test_max_samples_caps_per_modality(self, tmp_path: Path) -> None:
        ds = tmp_path / "ds"
        out = tmp_path / "out"
        _populate_dataset(ds)
        saved = collect_multimodal_embeddings(
            encoder=_FakeMultimodalEncoder(),
            dataset_path=ds,
            output_dir=out,
            max_samples_per_modality=1,
        )
        # Image had 2 files; should be capped to 1.
        image_emb = torch.load(saved["image"])["embeddings"]
        text_emb = torch.load(saved["text"])["embeddings"]
        assert image_emb.shape[0] == 1
        assert text_emb.shape[0] == 1

    def test_missing_encode_method_skips_modality_silently(self, tmp_path: Path) -> None:
        """An encoder missing encode_audio doesn't crash; audio is just skipped."""
        ds = tmp_path / "ds"
        out = tmp_path / "out"
        _populate_dataset(ds)

        class ImageOnly:
            def encode_image(self, path):  # noqa: ANN001
                return torch.randn(1, 8)

        saved = collect_multimodal_embeddings(encoder=ImageOnly(), dataset_path=ds, output_dir=out)
        assert "image" in saved
        assert "audio" not in saved
        assert "video" not in saved

    def test_handles_empty_directory_gracefully(self, tmp_path: Path) -> None:
        ds = tmp_path / "ds"
        ds.mkdir()
        out = tmp_path / "out"
        saved = collect_multimodal_embeddings(
            encoder=_FakeMultimodalEncoder(), dataset_path=ds, output_dir=out
        )
        assert saved == {}


class TestCollectMultimodalCli:
    def test_cli_invokes_collect_multimodal(self, tmp_path: Path) -> None:
        ds = tmp_path / "ds"
        out = tmp_path / "out"
        _populate_dataset(ds)

        registry = MagicMock()
        registry.load.return_value = _FakeMultimodalEncoder()

        with patch(
            "embedding_art.encoders.defaults.create_default_registry", return_value=registry
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "sae",
                    "collect-multimodal",
                    "--dataset",
                    str(ds),
                    "--output-dir",
                    str(out),
                ],
            )

        assert result.exit_code == 0, result.output
        assert (out / "image_embeddings.pt").exists()
        assert (out / "text_embeddings.pt").exists()
        assert "image:" in result.output
        assert "text:" in result.output
