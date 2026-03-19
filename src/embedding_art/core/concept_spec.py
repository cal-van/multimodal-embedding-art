"""
ConceptSpec — encoder-agnostic concept specification.

A ConceptSpec describes *what* to render without binding to any encoder.
This is the v2 entry point for expressing a concept: callers build a
ConceptSpec and hand it to an engine, which resolves it against a
concrete encoder to produce a Concept (embedding).

Design notes:
- Frozen dataclass: value-type semantics, hashable, safe to use as dict keys.
- All modality fields are optional; an empty spec is structurally valid.
- weight is a plain float (not validated here) so callers can express
  subtraction (negative) or amplification (>1) without extra ceremony.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConceptSpec:
    """What to render — not yet bound to any encoder."""

    text: str | None = None
    image: Path | None = None
    audio: Path | None = None
    video: Path | None = None
    weight: float = 1.0

    def describe(self) -> str:
        """Human-readable description of this spec."""
        parts = []
        if self.text is not None:
            parts.append(f'text:"{self.text}"')
        if self.image is not None:
            parts.append(f"image:{self.image.name}")
        if self.audio is not None:
            parts.append(f"audio:{self.audio.name}")
        if self.video is not None:
            parts.append(f"video:{self.video.name}")
        desc = " + ".join(parts) if parts else "empty"
        if self.weight != 1.0:
            desc = f"{self.weight}*({desc})"
        return desc

    def __repr__(self) -> str:
        return f"ConceptSpec({self.describe()})"
