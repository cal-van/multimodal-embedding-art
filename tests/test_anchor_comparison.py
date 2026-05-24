"""
Tests for the anchor-comparison experiment (M9).

Uses small synthetic encoders so the tests run without LanguageBind weights.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from click.testing import CliRunner

from embedding_art.cli.main import cli
from embedding_art.experiments.anchor_comparison import (
    AnchorComparison,
    run_anchor_comparison,
)


@dataclass
class _FakeEncoder:
    """Tiny encoder for the anchor-comparison tests.

    Returns deterministic per-modality embeddings so tests can verify
    cross-modal cosine matrices exactly.
    """

    embed_dim: int = 32

    def encode_text(self, word: str) -> torch.Tensor:
        seed = abs(hash(("text", word))) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        emb = torch.randn(1, self.embed_dim, generator=gen)
        return emb / emb.norm(dim=-1, keepdim=True)

    def encode_image(self, path: str | Path) -> torch.Tensor:
        seed = abs(hash(("image", str(path)))) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        emb = torch.randn(1, self.embed_dim, generator=gen)
        return emb / emb.norm(dim=-1, keepdim=True)

    def encode_audio(self, path: str | Path) -> torch.Tensor:
        seed = abs(hash(("audio", str(path)))) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        emb = torch.randn(1, self.embed_dim, generator=gen)
        return emb / emb.norm(dim=-1, keepdim=True)

    def encode_video(self, path: str | Path) -> torch.Tensor:
        seed = abs(hash(("video", str(path)))) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        emb = torch.randn(1, self.embed_dim, generator=gen)
        return emb / emb.norm(dim=-1, keepdim=True)


class TestRunAnchorComparison:
    """run_anchor_comparison produces the expected structure."""

    def test_text_only_yields_one_modality(self) -> None:
        encoder = _FakeEncoder()
        result = run_anchor_comparison(
            concept_label="thunder",
            encoder=encoder,
            encoder_name="fake",
            text="thunder",
        )
        assert isinstance(result, AnchorComparison)
        assert result.concept_label == "thunder"
        assert "text" in result.modalities
        assert "image" not in result.modalities
        # Self-similarity is 1.0.
        assert result.cosine_matrix["text"]["text"] == pytest.approx(1.0)

    def test_all_four_modalities(self) -> None:
        encoder = _FakeEncoder()
        result = run_anchor_comparison(
            concept_label="thunder",
            encoder=encoder,
            encoder_name="fake",
            text="thunder",
            image_path="storm.jpg",
            audio_path="thunder.wav",
            video_path="lightning.mp4",
        )
        assert set(result.modalities.keys()) == {"text", "image", "audio", "video"}
        # Pairwise cosine matrix is dense.
        for m1 in ["text", "image", "audio", "video"]:
            for m2 in ["text", "image", "audio", "video"]:
                assert m1 in result.cosine_matrix
                assert m2 in result.cosine_matrix[m1]

    def test_cosine_matrix_symmetric_and_self_one(self) -> None:
        encoder = _FakeEncoder()
        result = run_anchor_comparison(
            concept_label="thunder",
            encoder=encoder,
            encoder_name="fake",
            text="thunder",
            image_path="storm.jpg",
        )
        # Diagonal == 1.0.
        for k in result.modalities:
            assert result.cosine_matrix[k][k] == pytest.approx(1.0)
        # Off-diagonal is symmetric.
        assert result.cosine_matrix["text"]["image"] == pytest.approx(
            result.cosine_matrix["image"]["text"]
        )

    def test_to_dict_drops_tensor_embedding(self) -> None:
        encoder = _FakeEncoder()
        result = run_anchor_comparison(
            concept_label="thunder",
            encoder=encoder,
            encoder_name="fake",
            text="thunder",
        )
        d = result.to_dict()
        assert "modalities" in d
        for record in d["modalities"].values():
            assert "embedding" not in record  # Tensor stripped.
            assert "embedding_dim" in record  # Replaced with shape info.
        # Dict is JSON-serialisable.
        json.dumps(d)

    def test_sae_overlap_when_sae_provided(self) -> None:
        encoder = _FakeEncoder()

        # Fake SAE returns sparse activations differing across modalities.
        def fake_sae(emb: torch.Tensor) -> torch.Tensor:
            activations = torch.zeros(1, 64)
            # Deterministic per-input activation pattern based on hash of values.
            seed = int(abs(emb.sum().item() * 1e4)) % 1000 + 1
            gen = torch.Generator().manual_seed(seed)
            indices = torch.randperm(64, generator=gen)[:10]
            activations[0, indices] = torch.rand(10, generator=gen) + 0.1
            return activations

        result = run_anchor_comparison(
            concept_label="thunder",
            encoder=encoder,
            encoder_name="fake",
            text="thunder",
            image_path="storm.jpg",
            sae=fake_sae,
            top_k_features=5,
        )
        assert result.sae_feature_overlap is not None
        assert "text" in result.sae_feature_overlap
        assert "image" in result.sae_feature_overlap["text"]
        cross = result.sae_feature_overlap["text"]["image"]
        assert "intersection" in cross
        assert "unique_to_a" in cross
        assert "unique_to_b" in cross
        assert "jaccard" in cross
        # Self-pair has 1.0 Jaccard and empty unique sets.
        self_pair = result.sae_feature_overlap["text"]["text"]
        assert self_pair["jaccard"] == pytest.approx(1.0)
        assert self_pair["unique_to_a"] == []

    def test_text_anchor_populated(self) -> None:
        encoder = _FakeEncoder()
        result = run_anchor_comparison(
            concept_label="thunder",
            encoder=encoder,
            encoder_name="fake",
            text="thunder",
            top_k_text=5,
        )
        anchor = result.modalities["text"]["text_anchor"]
        assert len(anchor) > 0
        assert all("word" in a and "similarity" in a for a in anchor)


class TestAnchorCompareCli:
    """The CLI command is registered and writes the expected artefacts."""

    def test_command_registered(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["anchor-compare", "--help"])
        assert result.exit_code == 0
        assert "anchor-compare" in result.output or "Compare a single" in result.output

    def test_requires_at_least_one_reference(self, tmp_path: Path) -> None:
        from embedding_art.cli.commands.anchor_compare import _anchor_compare_impl

        with patch("embedding_art.encoders.defaults.create_default_registry") as mock_registry:
            mock_registry.return_value.load.return_value = _FakeEncoder()

            with pytest.raises(Exception) as exc_info:
                _anchor_compare_impl(
                    concept_label="thunder",
                    output_dir=tmp_path,
                    text=None,
                    image_path=None,
                    audio_path=None,
                    video_path=None,
                    encoder_name="fake",
                    device="cpu",
                    sae_path=None,
                    top_k_text=5,
                    top_k_features=5,
                )
            assert "at least one" in str(exc_info.value).lower()

    def test_writes_json_and_md(self, tmp_path: Path) -> None:
        from embedding_art.cli.commands.anchor_compare import _anchor_compare_impl

        mock_registry = MagicMock()
        mock_registry.load.return_value = _FakeEncoder()
        with patch(
            "embedding_art.encoders.defaults.create_default_registry",
            return_value=mock_registry,
        ):
            _anchor_compare_impl(
                concept_label="thunder",
                output_dir=tmp_path,
                text="thunder",
                image_path=None,
                audio_path=None,
                video_path=None,
                encoder_name="fake",
                device="cpu",
                sae_path=None,
                top_k_text=5,
                top_k_features=5,
            )

        json_path = tmp_path / "anchor_comparison.json"
        md_path = tmp_path / "anchor_comparison.md"
        assert json_path.exists()
        assert md_path.exists()

        data = json.loads(json_path.read_text())
        assert data["concept_label"] == "thunder"
        assert "text" in data["modalities"]
        # Markdown card mentions the concept.
        assert "thunder" in md_path.read_text()
