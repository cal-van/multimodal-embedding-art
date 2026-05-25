"""
Tests for ``EncoderActivationCache``.

Pure-Python; no encoder weights required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from embedding_art.experiments.activation_cache import EncoderActivationCache
from embedding_art.experiments.anchor_comparison import run_anchor_comparison


class TestEncoderActivationCache:
    """The cache short-circuits compute() on repeated identical inputs."""

    def test_text_cache_hit_skips_compute_on_second_call(self, tmp_path: Path) -> None:
        cache = EncoderActivationCache(tmp_path, encoder_id="fake")

        compute = MagicMock(return_value=torch.tensor([[0.1, 0.2, 0.3]]))
        result1 = cache.get_or_compute("text", "thunder", compute)
        result2 = cache.get_or_compute("text", "thunder", compute)

        # Second call should be a hit and NOT invoke compute again.
        assert compute.call_count == 1
        torch.testing.assert_close(result1, result2)
        assert cache.hits == 1
        assert cache.misses == 1

    def test_different_text_misses(self, tmp_path: Path) -> None:
        cache = EncoderActivationCache(tmp_path, encoder_id="fake")
        compute_a = MagicMock(return_value=torch.tensor([[1.0]]))
        compute_b = MagicMock(return_value=torch.tensor([[2.0]]))

        cache.get_or_compute("text", "thunder", compute_a)
        cache.get_or_compute("text", "lightning", compute_b)

        assert compute_a.call_count == 1
        assert compute_b.call_count == 1
        assert cache.hits == 0
        assert cache.misses == 2

    def test_file_reference_hashed_by_content_not_path(self, tmp_path: Path) -> None:
        """Two distinct files with identical content should collide."""
        cache = EncoderActivationCache(tmp_path / "cache", encoder_id="fake")

        file_a = tmp_path / "a.bin"
        file_b = tmp_path / "b.bin"
        # Identical bytes — distinct paths.
        for f in (file_a, file_b):
            f.write_bytes(b"\x01\x02\x03\x04")

        compute = MagicMock(return_value=torch.tensor([[42.0]]))
        cache.get_or_compute("image", file_a, compute)
        cache.get_or_compute("image", file_b, compute)

        assert compute.call_count == 1, "identical file contents should hit the cache"

    def test_different_file_content_misses(self, tmp_path: Path) -> None:
        cache = EncoderActivationCache(tmp_path / "cache", encoder_id="fake")

        file_a = tmp_path / "a.bin"
        file_b = tmp_path / "b.bin"
        file_a.write_bytes(b"hello")
        file_b.write_bytes(b"world")

        compute_a = MagicMock(return_value=torch.tensor([[1.0]]))
        compute_b = MagicMock(return_value=torch.tensor([[2.0]]))
        cache.get_or_compute("image", file_a, compute_a)
        cache.get_or_compute("image", file_b, compute_b)

        assert compute_a.call_count == 1
        assert compute_b.call_count == 1

    def test_cross_process_persistence(self, tmp_path: Path) -> None:
        """A new cache instance pointed at the same directory should hit."""
        cache1 = EncoderActivationCache(tmp_path, encoder_id="fake")
        compute1 = MagicMock(return_value=torch.tensor([[7.0, 8.0]]))
        cache1.get_or_compute("text", "thunder", compute1)

        # New instance — empty in-memory L1, but disk is populated.
        cache2 = EncoderActivationCache(tmp_path, encoder_id="fake")
        compute2 = MagicMock(return_value=torch.tensor([[9.0, 9.0]]))
        result = cache2.get_or_compute("text", "thunder", compute2)

        # Second instance should hit disk; compute2 should never be called.
        assert compute2.call_count == 0
        torch.testing.assert_close(result, torch.tensor([[7.0, 8.0]]))

    def test_encoder_id_scopes_the_cache(self, tmp_path: Path) -> None:
        """Different encoder ids are separate namespaces."""
        cache_a = EncoderActivationCache(tmp_path, encoder_id="languagebind")
        cache_b = EncoderActivationCache(tmp_path, encoder_id="imagebind")

        compute_a = MagicMock(return_value=torch.tensor([[1.0]]))
        compute_b = MagicMock(return_value=torch.tensor([[2.0]]))

        cache_a.get_or_compute("text", "thunder", compute_a)
        # Same content, different encoder id — must miss.
        cache_b.get_or_compute("text", "thunder", compute_b)

        assert compute_a.call_count == 1
        assert compute_b.call_count == 1

    def test_missing_file_falls_back_to_compute(self, tmp_path: Path) -> None:
        """If the reference cannot be hashed, we still return a result."""
        cache = EncoderActivationCache(tmp_path, encoder_id="fake")
        compute = MagicMock(return_value=torch.tensor([[3.14]]))
        # File doesn't exist — hashing will raise.
        nonexistent = tmp_path / "no-such-file.png"
        out = cache.get_or_compute("image", nonexistent, compute)
        assert compute.call_count == 1
        torch.testing.assert_close(out, torch.tensor([[3.14]]))


class TestAnchorComparisonWithCache:
    """``run_anchor_comparison`` short-circuits encoder forwards via cache."""

    def test_repeat_run_short_circuits_encoder(self, tmp_path: Path) -> None:
        # Counting encoder so we can verify the second run hits the cache.
        counter = {"n": 0}

        class CountingEncoder:
            embed_dim: int = 8

            def encode_text(self, word: str) -> torch.Tensor:
                counter["n"] += 1
                return torch.ones(1, 8)

        encoder = CountingEncoder()
        cache = EncoderActivationCache(tmp_path / "cache", encoder_id="fake")

        run_anchor_comparison(
            concept_label="thunder",
            encoder=encoder,
            encoder_name="fake",
            text="thunder",
            cache=cache,
        )
        first_count = counter["n"]

        run_anchor_comparison(
            concept_label="thunder",
            encoder=encoder,
            encoder_name="fake",
            text="thunder",
            cache=cache,
        )
        second_count = counter["n"]

        assert first_count >= 1, "first run must have called the encoder"
        assert second_count == first_count, (
            f"second run should hit the cache, but encoder was called "
            f"{second_count - first_count} additional time(s)"
        )

    def test_distinct_text_does_not_short_circuit(self, tmp_path: Path) -> None:
        counter = {"n": 0}

        class CountingEncoder:
            embed_dim: int = 8

            def encode_text(self, word: str) -> torch.Tensor:
                counter["n"] += 1
                return torch.ones(1, 8) * (1.0 + counter["n"])

        encoder = CountingEncoder()
        cache = EncoderActivationCache(tmp_path / "cache", encoder_id="fake")

        run_anchor_comparison(
            concept_label="thunder",
            encoder=encoder,
            encoder_name="fake",
            text="thunder",
            cache=cache,
        )
        run_anchor_comparison(
            concept_label="lightning",
            encoder=encoder,
            encoder_name="fake",
            text="lightning",
            cache=cache,
        )

        assert counter["n"] == 2, "distinct text references must each hit the encoder"


def test_hash_text_is_stable() -> None:
    """Same text -> same hash; small change -> different hash."""
    a = EncoderActivationCache.hash_text("thunder")
    b = EncoderActivationCache.hash_text("thunder")
    c = EncoderActivationCache.hash_text("thunders")
    assert a == b
    assert a != c


def test_hash_file_is_stable(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"\x00\x01\x02")
    a = EncoderActivationCache.hash_file(p)
    b = EncoderActivationCache.hash_file(p)
    assert a == b
    # Re-write with different content -> different hash.
    p.write_bytes(b"\x00\x01\x03")
    c = EncoderActivationCache.hash_file(p)
    assert a != c


def test_hash_reference_dispatches_on_modality(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello")
    text_hash = EncoderActivationCache.hash_reference("text", "hello")
    image_hash = EncoderActivationCache.hash_reference("image", p)
    # Different code paths (utf-8 bytes vs file bytes), happen to differ here.
    # Either way, both should be 64 hex chars.
    assert len(text_hash) == 64
    assert len(image_hash) == 64


@pytest.mark.parametrize("modality", ["image", "audio", "video"])
def test_path_layout_is_predictable(tmp_path: Path, modality: str) -> None:
    cache = EncoderActivationCache(tmp_path, encoder_id="fake-encoder")
    p = cache.path_for(modality, "abc123")
    assert p == tmp_path / "fake-encoder" / modality / "abc123.pt"
