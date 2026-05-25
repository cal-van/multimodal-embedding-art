"""Tests for ``embedding_art.sae.labelling``.

Exercises CosineLabeller wrapping behaviour and VLMLabeller's
graceful-fallback contract using a synthetic SAE checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from click.testing import CliRunner

from embedding_art.cli.main import cli
from embedding_art.sae.labelling import CosineLabeller, VLMLabeller


def _write_sae_checkpoint(directory: Path, n_features: int = 4, embed_dim: int = 8) -> Path:
    """Write a minimal sae_weights.pt that the labellers can read."""
    state = {
        "W_enc": torch.randn(n_features, embed_dim),
        "W_dec": torch.randn(embed_dim, n_features),
        "b_enc": torch.zeros(n_features),
        "b_dec": torch.zeros(embed_dim),
    }
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(state, directory / "sae_weights.pt")
    return directory


class _FakeEncoder:
    """An encoder whose word embeddings are deterministic per-input.

    Used to drive label_features without loading LanguageBind.
    """

    def __init__(self, embed_dim: int = 8) -> None:
        self.embed_dim = embed_dim
        torch.manual_seed(42)

    def encode_text(self, text: str) -> torch.Tensor:
        # Hash text → 32-bit seed → tensor.
        seed = abs(hash(text)) % (2**31)
        gen = torch.Generator()
        gen.manual_seed(seed)
        return torch.randn(1, self.embed_dim, generator=gen)


class TestCosineLabeller:
    def test_returns_one_label_per_feature(self, tmp_path: Path) -> None:
        sae_dir = _write_sae_checkpoint(tmp_path / "sae", n_features=5)
        labeller = CosineLabeller()
        labels = labeller.label_all(sae_path=sae_dir, encoder=_FakeEncoder())
        assert isinstance(labels, list)
        assert len(labels) == 5
        for label in labels:
            assert isinstance(label, str) and label  # non-empty


class TestVLMLabeller:
    def test_calls_llm_once_per_feature(self, tmp_path: Path) -> None:
        sae_dir = _write_sae_checkpoint(tmp_path / "sae", n_features=3)
        llm = MagicMock(return_value="thunder")
        labeller = VLMLabeller(llm_callable=llm)
        labels = labeller.label_all(sae_path=sae_dir, encoder=_FakeEncoder())
        assert labels == ["thunder", "thunder", "thunder"]
        assert llm.call_count == 3

    def test_falls_back_per_feature_on_llm_failure(self, tmp_path: Path) -> None:
        """When the LLM raises for a specific feature, we fall back to
        the cosine label for THAT feature only (others remain LLM-driven)."""
        sae_dir = _write_sae_checkpoint(tmp_path / "sae", n_features=3)
        call_count = {"n": 0}

        def flaky(prompt: str) -> str:
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("rate limited")
            return f"vlm_label_{call_count['n']}"

        labeller = VLMLabeller(llm_callable=flaky)
        labels = labeller.label_all(sae_path=sae_dir, encoder=_FakeEncoder())
        assert len(labels) == 3
        assert labels[0] == "vlm_label_1"
        assert labels[2] == "vlm_label_3"
        # feature 1's label is the cosine fallback — a non-empty string
        # from the built-in vocabulary, not the failed LLM label.
        assert labels[1] != "vlm_label_2"
        assert labels[1]  # non-empty

    def test_strips_quotes_and_whitespace_from_response(self, tmp_path: Path) -> None:
        sae_dir = _write_sae_checkpoint(tmp_path / "sae", n_features=1)

        def quoted_llm(prompt: str) -> str:
            return '  "electric storm"  '

        labels = VLMLabeller(llm_callable=quoted_llm).label_all(
            sae_path=sae_dir, encoder=_FakeEncoder()
        )
        assert labels == ["electric storm"]


class TestSaeAutoLabelCli:
    """``embed-art sae auto-label`` writes feature_labels.json."""

    def test_writes_default_output(self, tmp_path: Path) -> None:
        sae_dir = _write_sae_checkpoint(tmp_path / "sae", n_features=3)

        # Patch the encoder loading so we don't actually load LanguageBind.
        fake_encoder = _FakeEncoder()
        registry = MagicMock()
        registry.load.return_value = fake_encoder

        with patch(
            "embedding_art.encoders.defaults.create_default_registry", return_value=registry
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["sae", "auto-label", "--model", str(sae_dir)])

        assert result.exit_code == 0, result.output
        labels_file = sae_dir / "feature_labels.json"
        assert labels_file.exists()
        labels = json.loads(labels_file.read_text())
        assert set(labels.keys()) == {"0", "1", "2"}
        for v in labels.values():
            assert isinstance(v, str) and v

    def test_writes_custom_output_path(self, tmp_path: Path) -> None:
        sae_dir = _write_sae_checkpoint(tmp_path / "sae", n_features=2)
        out = tmp_path / "labels_v1.json"

        fake_encoder = _FakeEncoder()
        registry = MagicMock()
        registry.load.return_value = fake_encoder

        with patch(
            "embedding_art.encoders.defaults.create_default_registry", return_value=registry
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "sae",
                    "auto-label",
                    "--model",
                    str(sae_dir),
                    "--output",
                    str(out),
                ],
            )

        assert result.exit_code == 0, result.output
        assert out.exists()
        # The default location was NOT used.
        assert not (sae_dir / "feature_labels.json").exists()

    def test_use_vlm_falls_back_when_no_api_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With --use-vlm but no ANTHROPIC_API_KEY, the CLI still
        succeeds by falling back to the cosine labeller."""
        sae_dir = _write_sae_checkpoint(tmp_path / "sae", n_features=2)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        fake_encoder = _FakeEncoder()
        registry = MagicMock()
        registry.load.return_value = fake_encoder

        with patch(
            "embedding_art.encoders.defaults.create_default_registry", return_value=registry
        ):
            runner = CliRunner()
            result = runner.invoke(
                cli,
                ["sae", "auto-label", "--model", str(sae_dir), "--use-vlm"],
            )

        assert result.exit_code == 0, result.output
        assert "Falling back to cosine" in result.output


def _write_corpus(path: Path, sources: list[str], embed_dim: int = 8) -> Path:
    """Write a synthetic 'sae collect' corpus with both embeddings + sources."""
    import torch

    torch.manual_seed(0)
    embeddings = torch.randn(len(sources), embed_dim)
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    torch.save({"embeddings": embeddings, "sources": sources}, path)
    return path


class TestActivatingExamplesLabeller:
    def test_prompts_with_top_k_source_names(self, tmp_path: Path) -> None:
        from embedding_art.sae.labelling import ActivatingExamplesLabeller

        sae_dir = _write_sae_checkpoint(tmp_path / "sae", n_features=2, embed_dim=8)
        corpus = _write_corpus(
            tmp_path / "corpus.pt",
            [f"img_{i:03d}.png" for i in range(12)],
            embed_dim=8,
        )

        captured_prompts: list[str] = []

        def fake_llm(prompt: str) -> str:
            captured_prompts.append(prompt)
            return "thunder"

        labeller = ActivatingExamplesLabeller(
            corpus_path=corpus,
            llm_callable=fake_llm,
            top_k_examples=4,
        )
        labels = labeller.label_all(sae_path=sae_dir, encoder=_FakeEncoder())

        assert labels == ["thunder", "thunder"]
        assert len(captured_prompts) == 2
        # Every prompt should embed exactly 4 source filenames.
        for prompt in captured_prompts:
            assert "activates" in prompt
            source_mentions = sum(1 for i in range(12) if f"img_{i:03d}.png" in prompt)
            assert source_mentions == 4

    def test_falls_back_when_corpus_lacks_sources(self, tmp_path: Path) -> None:
        """Backwards-compat: a corpus saved by an older sae collect (no
        'sources' key) should not break; we degrade to cosine labels."""
        import torch

        from embedding_art.sae.labelling import ActivatingExamplesLabeller

        sae_dir = _write_sae_checkpoint(tmp_path / "sae", n_features=2, embed_dim=8)
        # Legacy corpus: embeddings only, no sources.
        corpus_path = tmp_path / "legacy_corpus.pt"
        torch.save({"embeddings": torch.randn(8, 8)}, corpus_path)

        llm = MagicMock(return_value="should-not-be-called")
        labeller = ActivatingExamplesLabeller(
            corpus_path=corpus_path,
            llm_callable=llm,
            top_k_examples=4,
        )
        labels = labeller.label_all(sae_path=sae_dir, encoder=_FakeEncoder())

        assert len(labels) == 2
        for label in labels:
            assert isinstance(label, str) and label
        # We did NOT call the LLM — fell back to cosine.
        llm.assert_not_called()

    def test_falls_back_per_feature_on_llm_failure(self, tmp_path: Path) -> None:
        from embedding_art.sae.labelling import ActivatingExamplesLabeller

        sae_dir = _write_sae_checkpoint(tmp_path / "sae", n_features=3, embed_dim=8)
        corpus = _write_corpus(
            tmp_path / "corpus.pt", [f"x_{i}.wav" for i in range(8)], embed_dim=8
        )

        call_count = {"n": 0}

        def flaky(prompt: str) -> str:
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("rate limited")
            return f"vlm_{call_count['n']}"

        labels = ActivatingExamplesLabeller(
            corpus_path=corpus,
            llm_callable=flaky,
            top_k_examples=3,
        ).label_all(sae_path=sae_dir, encoder=_FakeEncoder())
        assert len(labels) == 3
        assert labels[0] == "vlm_1"
        assert labels[2] == "vlm_3"
        assert labels[1] and labels[1] != "vlm_2"  # cosine fallback


class TestSaeCollectStoresSources:
    """Regression: ``collect_embeddings`` must store source paths so the
    activating-examples labeller works downstream."""

    def test_collect_embeddings_writes_sources(self, tmp_path: Path) -> None:
        import torch
        from PIL import Image

        from embedding_art.sae.training import collect_embeddings

        # Two trivial images.
        for i in range(2):
            img = Image.new("RGB", (32, 32), color=(i * 80, 0, 0))
            img.save(tmp_path / f"img_{i}.png")

        class _StubEncoder:
            embedding_dim = 8

            def encode_image(self, path):  # noqa: ANN001
                return torch.randn(1, 8)

        out = tmp_path / "corpus.pt"
        collect_embeddings(
            encoder=_StubEncoder(),
            dataset_path=tmp_path,
            output_path=out,
        )
        data = torch.load(out, weights_only=True)
        assert "sources" in data
        assert len(data["sources"]) == 2
        assert all(s.endswith(".png") for s in data["sources"])
