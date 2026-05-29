"""
Content-addressed cache for canonical-encoder activations.

The anchor-comparison experiment (and any other path that re-encodes the
same reference repeatedly) is dominated by the encoder forward. A naive
re-run of ``embed-art anchor-compare`` on the same image / audio / video
files re-pays that cost every time, even though the inputs are bit-for-
bit identical.

This module provides a small disk + in-memory cache keyed on
``(encoder_id, modality, content_hash)``. ``content_hash`` is the SHA256
of the file contents for binary modalities (image / audio / video) and
the SHA256 of the UTF-8 bytes for ``text``.

Cache misses fall through to a user-supplied compute function; hits load
the previously-saved tensor from disk. Reads and writes are best-effort:
any I/O failure logs a warning and proceeds (compute on miss, return the
tensor on write-failure) — the cache is *strictly* a perf optimisation
and must never change result correctness.

Typical layout::

    <cache_dir>/<encoder_id>/<modality>/<sha256>.pt
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


_CHUNK = 64 * 1024


class EncoderActivationCache:
    """Disk-backed content-addressed cache for encoder activations.

    Args:
        cache_dir: Root directory for the cache. Created if missing.
        encoder_id: A string scoping the cache to a particular encoder
            (typically the encoder name plus, optionally, a version /
            checkpoint hash). Two encoders with the same id are assumed
            to produce identical activations for the same reference;
            collisions silently return stale data, so callers should
            include enough information in the id to avoid this.
    """

    def __init__(self, cache_dir: str | Path, encoder_id: str) -> None:
        self.cache_dir = Path(cache_dir)
        self.encoder_id = encoder_id
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Process-local L1 cache: avoid touching disk for the second hit
        # on the same reference within a single run.
        self._mem: dict[str, torch.Tensor] = {}
        # Lightweight counters useful for tests + logging.
        self.hits: int = 0
        self.misses: int = 0

    # ------------------------------------------------------------------
    # Hashing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def hash_text(text: str) -> str:
        """SHA256 over the UTF-8 bytes of ``text``."""
        h = hashlib.sha256()
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    @staticmethod
    def hash_file(path: str | Path) -> str:
        """SHA256 over the bytes of the file at ``path``."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(_CHUNK), b""):
                h.update(chunk)
        return h.hexdigest()

    @classmethod
    def hash_reference(cls, modality: str, ref: str | Path) -> str:
        """Hash a per-modality reference into a stable content key.

        For ``modality == "text"`` the reference is hashed as raw bytes;
        otherwise it is treated as a file path and hashed by content.
        """
        if modality == "text":
            return cls.hash_text(str(ref))
        return cls.hash_file(ref)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def path_for(self, modality: str, content_hash: str) -> Path:
        """Return the on-disk path for a given modality + content hash."""
        return self.cache_dir / self.encoder_id / modality / f"{content_hash}.pt"

    def get_or_compute(
        self,
        modality: str,
        ref: str | Path,
        compute: Callable[[], torch.Tensor],
    ) -> torch.Tensor:
        """Return the cached activation if present, otherwise compute + store.

        ``compute`` is called only on a cache miss. Its return value
        must be a ``torch.Tensor``; it is detached + moved to CPU before
        being stored. Callers should re-move to their preferred device
        on retrieval.
        """
        try:
            content_hash = self.hash_reference(modality, ref)
        except Exception as exc:
            # If we cannot hash (e.g. file missing, permission), fall
            # straight through to compute without caching — the cache
            # must never break the experiment.
            logger.warning(
                "Could not hash %s reference %r: %s — bypassing cache.",
                modality,
                ref,
                exc,
            )
            self.misses += 1
            return compute()

        mem_key = f"{modality}:{content_hash}"
        if mem_key in self._mem:
            self.hits += 1
            return self._mem[mem_key]

        on_disk = self.path_for(modality, content_hash)
        if on_disk.exists():
            try:
                tensor = torch.load(on_disk, map_location="cpu", weights_only=True)
                self._mem[mem_key] = tensor
                self.hits += 1
                return tensor
            except Exception as exc:
                logger.warning("Cache read failed for %s: %s — recomputing.", on_disk, exc)

        # Cache miss: compute, store, return.
        self.misses += 1
        tensor = compute()
        if not isinstance(tensor, torch.Tensor):
            # Strictly, the contract requires a tensor; but if a caller
            # returns something else we still hand it back rather than
            # break their flow. We just skip the cache write.
            return tensor

        tensor_cpu = tensor.detach().cpu()
        try:
            on_disk.parent.mkdir(parents=True, exist_ok=True)
            torch.save(tensor_cpu, on_disk)
        except Exception as exc:
            logger.warning("Cache write failed for %s: %s", on_disk, exc)
        self._mem[mem_key] = tensor_cpu
        return tensor_cpu
