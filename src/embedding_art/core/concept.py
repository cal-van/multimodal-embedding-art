"""
Concept: A point in multimodal embedding space.

Concepts are the targets for optimization. They can be created from any modality
(text, image, audio, video) and combined using arithmetic operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from embedding_art.encoders.base import Encoder


@dataclass
class Concept:
    """
    A point or direction in embedding space.

    Supports arithmetic operations for combining concepts:
        - Addition: concept_a + concept_b
        - Subtraction: concept_a - concept_b
        - Scalar multiplication: 0.5 * concept_a
        - Spherical interpolation: Concept.slerp(a, b, t)

    All embeddings are stored normalized on the unit hypersphere.
    """

    embedding: torch.Tensor  # Shape: [1, embed_dim], normalized
    description: str = ""

    def __post_init__(self):
        """Ensure embedding is normalized."""
        if self.embedding.dim() == 1:
            self.embedding = self.embedding.unsqueeze(0)
        self.embedding = F.normalize(self.embedding, dim=-1)

    @classmethod
    def from_text(cls, text: str, encoder: Encoder) -> Concept:
        """Create concept from text description."""
        embedding = encoder.encode_text(text)
        return cls(embedding=embedding, description=f'text:"{text}"')

    @classmethod
    def from_image(cls, image: Path | str, encoder: Encoder) -> Concept:
        """Create concept from image file."""
        path = Path(image)
        embedding = encoder.encode_image(path)
        return cls(embedding=embedding, description=f"image:{path.name}")

    @classmethod
    def from_audio(
        cls,
        audio: Path | str,
        encoder: Encoder,
        start: float = 0.0,
        duration: float = 2.0,
    ) -> Concept:
        """Create concept from audio file."""
        path = Path(audio)
        embedding = encoder.encode_audio(path, start=start, duration=duration)
        return cls(
            embedding=embedding, description=f"audio:{path.name}[{start}:{start+duration}]"
        )

    @classmethod
    def from_video(
        cls,
        video: Path | str,
        encoder: Encoder,
        timestamp: float = 0.0,
    ) -> Concept:
        """Create concept from video file at specific timestamp."""
        path = Path(video)
        embedding = encoder.encode_video(path, timestamp=timestamp)
        return cls(embedding=embedding, description=f"video:{path.name}@{timestamp}")

    @classmethod
    def from_embedding(
        cls,
        embedding: torch.Tensor,
        description: str = "raw_embedding",
    ) -> Concept:
        """Create concept from raw embedding tensor."""
        return cls(embedding=embedding, description=description)

    @classmethod
    def load(cls, path: Path | str) -> Concept:
        """Load concept from saved .pt file."""
        data = torch.load(path, weights_only=False)
        return cls(
            embedding=data["embedding"],
            description=data.get("description", str(path)),
        )

    def save(self, path: Path | str) -> None:
        """Save concept to .pt file."""
        torch.save(
            {
                "embedding": self.embedding,
                "description": self.description,
            },
            path,
        )

    def to(self, device: str | torch.device) -> Concept:
        """Move embedding to device."""
        return Concept(
            embedding=self.embedding.to(device),
            description=self.description,
        )

    # === Arithmetic operations ===

    def __add__(self, other: Concept) -> Concept:
        """Add two concepts (then re-normalize)."""
        combined = self.embedding + other.embedding
        return Concept(
            embedding=combined,
            description=f"({self.description} + {other.description})",
        )

    def __sub__(self, other: Concept) -> Concept:
        """Subtract concept (then re-normalize)."""
        combined = self.embedding - other.embedding
        return Concept(
            embedding=combined,
            description=f"({self.description} - {other.description})",
        )

    def __mul__(self, scalar: float) -> Concept:
        """Scale concept (applied before normalization in combinations)."""
        # Note: scaling a normalized vector then re-normalizing does nothing
        # This is for use in weighted combinations before final normalization
        return Concept(
            embedding=self.embedding * scalar,
            description=f"{scalar}*{self.description}",
        )

    def __rmul__(self, scalar: float) -> Concept:
        """Right multiply by scalar."""
        return self.__mul__(scalar)

    def __neg__(self) -> Concept:
        """Negate concept direction."""
        return Concept(
            embedding=-self.embedding,
            description=f"-{self.description}",
        )

    # === Combination methods ===

    @staticmethod
    def slerp(a: Concept, b: Concept, t: float) -> Concept:
        """
        Spherical linear interpolation between two concepts.

        Args:
            a: Start concept
            b: End concept
            t: Interpolation factor [0, 1]

        Returns:
            Interpolated concept on the hypersphere
        """
        # Ensure normalized
        v0 = F.normalize(a.embedding, dim=-1)
        v1 = F.normalize(b.embedding, dim=-1)

        # Compute angle between vectors
        dot = torch.clamp(torch.sum(v0 * v1, dim=-1, keepdim=True), -1.0, 1.0)
        theta = torch.acos(dot)

        # Handle near-parallel vectors
        if theta.abs() < 1e-6:
            return Concept(
                embedding=(1 - t) * v0 + t * v1,
                description=f"slerp({a.description}, {b.description}, {t})",
            )

        sin_theta = torch.sin(theta)
        s0 = torch.sin((1 - t) * theta) / sin_theta
        s1 = torch.sin(t * theta) / sin_theta

        interpolated = s0 * v0 + s1 * v1

        return Concept(
            embedding=interpolated,
            description=f"slerp({a.description}, {b.description}, {t})",
        )

    @staticmethod
    def combine(
        concepts: list[Concept],
        weights: list[float] | None = None,
    ) -> Concept:
        """
        Weighted combination of multiple concepts.

        Args:
            concepts: List of concepts to combine
            weights: Optional weights (default: equal weighting)

        Returns:
            Combined concept (normalized)
        """
        if not concepts:
            raise ValueError("Cannot combine empty list of concepts")

        if weights is None:
            weights = [1.0 / len(concepts)] * len(concepts)

        if len(weights) != len(concepts):
            raise ValueError("Number of weights must match number of concepts")

        # Weighted sum
        combined = sum(w * c.embedding for w, c in zip(weights, concepts))

        # Build description
        parts = [f"{w}*{c.description}" for w, c in zip(weights, concepts)]
        description = f"combine({', '.join(parts)})"

        return Concept(embedding=combined, description=description)

    # === Utility methods ===

    def similarity(self, other: Concept) -> float:
        """Compute cosine similarity with another concept."""
        sim = F.cosine_similarity(self.embedding, other.embedding, dim=-1)
        return sim.item()

    def __repr__(self) -> str:
        device = self.embedding.device
        dim = self.embedding.shape[-1]
        return f"Concept({self.description}, dim={dim}, device={device})"
