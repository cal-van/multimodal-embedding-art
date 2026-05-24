"""
Concept: A point in multimodal embedding space.

Concepts are the targets for optimization. They can be created from any modality
(text, image, audio, video) and combined using arithmetic operations.

v3 (aggressive-rewrite) addition: a Concept may carry an
:class:`~embedding_art.sae.lens.SAEDecomposition` alongside its dense embedding.
When present, the decomposition is the *interpretable* view of the concept —
a sparse set of named features with activation strengths — and arithmetic
operations compose decompositions in feature space (in addition to composing
dense embeddings as before).

The decomposition is **optional** — concepts created by encoders start without
one.  Calling :meth:`decompose` with a trained SAE produces the decomposition
and returns it; calling :meth:`with_decomposition` returns a new Concept that
carries the decomposition alongside the original embedding.  The
:meth:`from_decomposition` constructor builds a concept whose embedding is
derived from the decomposition (via SAE reconstruction + L2-normalisation).

Feature-space arithmetic is automatically used when both operands carry
decompositions.  The resulting concept carries *both* the combined embedding
(dense-space arithmetic, as before) and the combined decomposition (sparse
feature-space arithmetic).  These two states are maintained in parallel; the
embedding remains the authoritative target for optimisation, while the
decomposition is an interpretable annotation that tracks how the concept's
features evolve through algebraic operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812

if TYPE_CHECKING:
    from embedding_art.encoders.base import Encoder
    from embedding_art.sae.lens import SAEDecomposition, SAELens


@dataclass
class Concept:
    """
    A point or direction in embedding space, optionally annotated with a sparse
    feature decomposition.

    Supports arithmetic operations for combining concepts:
        - Addition: concept_a + concept_b
        - Subtraction: concept_a - concept_b
        - Scalar multiplication: 0.5 * concept_a
        - Spherical interpolation: Concept.slerp(a, b, t)

    All embeddings are stored normalized on the unit hypersphere.

    When a ``decomposition`` is present (set via :meth:`with_decomposition`,
    :meth:`from_decomposition`, or :meth:`from_features`), binary arithmetic
    also composes the sparse activation vectors.  The composed decomposition
    is carried on the result so downstream code can inspect how features combine.
    """

    embedding: torch.Tensor  # Shape: [1, embed_dim], normalized
    description: str = ""
    text_source: str | None = None  # Original text if created from text
    source_input: torch.Tensor | None = None  # Ephemeral raw input; never persisted
    decomposition: SAEDecomposition | None = field(default=None, repr=False)

    def __post_init__(self):
        """Ensure embedding is normalized."""
        if self.embedding.dim() == 1:
            self.embedding = self.embedding.unsqueeze(0)
        self.embedding = F.normalize(self.embedding, dim=-1)

    # === Properties ===

    @property
    def has_decomposition(self) -> bool:
        """Whether this concept carries an interpretable SAE decomposition."""
        return self.decomposition is not None

    # === Factory methods ===

    @classmethod
    def from_text(cls, text: str, encoder: Encoder) -> Concept:
        """Create concept from text description."""
        embedding = encoder.encode_text(text)
        return cls(embedding=embedding, description=f'text:"{text}"', text_source=text)

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
            embedding=embedding, description=f"audio:{path.name}[{start}:{start + duration}]"
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
    def from_decomposition(
        cls,
        decomposition: SAEDecomposition,
        sae: SAELens,
        description: str = "from_decomposition",
    ) -> Concept:
        """Build a concept whose embedding is derived from a SAE decomposition.

        The embedding is the L2-normalised reconstruction of the decomposition
        through the SAE decoder.  The decomposition is stored on the resulting
        concept so feature-space arithmetic is available.

        Parameters
        ----------
        decomposition:
            The sparse activation vector to decode.
        sae:
            The SAE whose decoder produces the dense embedding.
        description:
            Human-readable label for this concept.
        """
        embedding = sae.reconstruct(decomposition)
        return cls(
            embedding=F.normalize(embedding, dim=-1),
            description=description,
            decomposition=decomposition,
        )

    @classmethod
    def load(cls, path: Path | str) -> Concept:
        """Load concept from saved .pt file."""
        data = torch.load(path, weights_only=True)
        decomposition = None
        if "decomposition_activations" in data:
            from embedding_art.sae.lens import SAEDecomposition

            decomposition = SAEDecomposition(
                activations=data["decomposition_activations"],
                active_features=data.get("decomposition_active_features", {}),
                reconstruction_error=data.get("decomposition_error", 0.0),
            )
        return cls(
            embedding=data["embedding"],
            description=data.get("description", str(path)),
            decomposition=decomposition,
        )

    def save(self, path: Path | str) -> None:
        """Save concept to .pt file."""
        payload: dict = {
            "embedding": self.embedding,
            "description": self.description,
        }
        if self.decomposition is not None:
            payload["decomposition_activations"] = self.decomposition.activations
            payload["decomposition_active_features"] = self.decomposition.active_features
            payload["decomposition_error"] = self.decomposition.reconstruction_error
        torch.save(payload, path)

    def to(self, device: str | torch.device) -> Concept:
        """Move embedding (and source_input if present) to device."""
        new_decomp = None
        if self.decomposition is not None:
            from embedding_art.sae.lens import SAEDecomposition

            new_decomp = SAEDecomposition(
                activations=self.decomposition.activations.to(device),
                active_features=self.decomposition.active_features,
                reconstruction_error=self.decomposition.reconstruction_error,
            )
        return Concept(
            embedding=self.embedding.to(device),
            description=self.description,
            text_source=self.text_source,
            source_input=self.source_input.to(device) if self.source_input is not None else None,
            decomposition=new_decomp,
        )

    def with_decomposition(self, decomposition: SAEDecomposition) -> Concept:
        """Return a copy of this concept annotated with a SAE decomposition.

        The embedding is unchanged; the decomposition is an interpretable
        annotation layered on top.
        """
        return Concept(
            embedding=self.embedding.clone(),
            description=self.description,
            text_source=self.text_source,
            source_input=self.source_input,
            decomposition=decomposition,
        )

    # === Arithmetic operations ===

    @staticmethod
    def _compose_decompositions(
        a_decomp: SAEDecomposition | None,
        b_decomp: SAEDecomposition | None,
        op: str,
    ) -> SAEDecomposition | None:
        """Compose two decompositions via feature-space arithmetic.

        Returns None if either operand lacks a decomposition.  This ensures
        feature-space arithmetic is only performed when both sides are
        interpretable.
        """
        if a_decomp is None or b_decomp is None:
            return None
        from embedding_art.sae.lens import SAEDecomposition

        a_acts = a_decomp.activations
        b_acts = b_decomp.activations
        if op == "add":
            combined_acts = a_acts + b_acts
        elif op == "sub":
            combined_acts = a_acts - b_acts
        else:
            raise ValueError(f"Unknown op: {op!r}")

        # Clamp: SAE activations are post-ReLU, so should be >= 0.
        combined_acts = combined_acts.clamp(min=0.0)

        nonzero_indices = combined_acts[0].nonzero(as_tuple=True)[0].tolist()
        # Merge vocabularies from both operands for feature names.
        active_features: dict[str, float] = {}
        for idx in nonzero_indices:
            # Try to recover the feature name from either operand's active_features.
            # Since SAEDecomposition stores feature names keyed by name (not index),
            # we need the reverse lookup.  Fall back to index-based name.
            val = float(combined_acts[0, idx].item())
            name = f"feature_{idx}"
            # Search both operands' active features for this index's name.
            for feat_name, feat_val in a_decomp.active_features.items():
                # Match by checking if this index produces a similar activation
                if abs(a_acts[0, idx].item()) > 0 and feat_name not in active_features:
                    # The feature at this index in a_decomp was named feat_name
                    # if its activation matches.
                    if abs(a_acts[0, idx].item() - feat_val) < 1e-6:
                        name = feat_name
                        break
            else:
                for feat_name, feat_val in b_decomp.active_features.items():
                    if abs(b_acts[0, idx].item()) > 0:
                        if abs(b_acts[0, idx].item() - feat_val) < 1e-6:
                            name = feat_name
                            break
            active_features[name] = val

        return SAEDecomposition(
            activations=combined_acts,
            active_features=active_features,
            reconstruction_error=max(
                a_decomp.reconstruction_error,
                b_decomp.reconstruction_error,
            ),
        )

    def __add__(self, other: Concept) -> Concept:
        """Add two concepts (then re-normalize).

        When both concepts carry decompositions, the result also carries a
        decomposition whose activations are the element-wise sum (clamped to
        non-negative).
        """
        combined = self.embedding + other.embedding
        composed_decomp = self._compose_decompositions(
            self.decomposition, other.decomposition, "add"
        )
        return Concept(
            embedding=combined,
            description=f"({self.description} + {other.description})",
            decomposition=composed_decomp,
        )

    def __sub__(self, other: Concept) -> Concept:
        """Subtract concept (then re-normalize).

        When both concepts carry decompositions, the result also carries a
        decomposition whose activations are the element-wise difference (clamped
        to non-negative).
        """
        combined = self.embedding - other.embedding
        composed_decomp = self._compose_decompositions(
            self.decomposition, other.decomposition, "sub"
        )
        return Concept(
            embedding=combined,
            description=f"({self.description} - {other.description})",
            decomposition=composed_decomp,
        )

    def __mul__(self, scalar: float) -> Concept:
        """Scale concept (applied before normalization in combinations).

        When this concept carries a decomposition, the result's decomposition
        activations are scaled by the same factor (clamped to non-negative).
        """
        scaled_decomp = None
        if self.decomposition is not None:
            from embedding_art.sae.lens import SAEDecomposition

            scaled_acts = (self.decomposition.activations * scalar).clamp(min=0.0)
            active_features = {}
            for feat_name, feat_val in self.decomposition.active_features.items():
                scaled_val = feat_val * scalar
                if scaled_val > 0:
                    active_features[feat_name] = scaled_val
            scaled_decomp = SAEDecomposition(
                activations=scaled_acts,
                active_features=active_features,
                reconstruction_error=self.decomposition.reconstruction_error,
            )

        return Concept(
            embedding=self.embedding * scalar,
            description=f"{scalar}*{self.description}",
            decomposition=scaled_decomp,
        )

    def __rmul__(self, scalar: float) -> Concept:
        """Right multiply by scalar."""
        return self.__mul__(scalar)

    def __neg__(self) -> Concept:
        """Negate concept direction.

        Decomposition is dropped on negation because SAE activations are
        post-ReLU and must be non-negative; negating them is not meaningful.
        """
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

        # Interpolate decompositions if both have them.
        interp_decomp = None
        if a.decomposition is not None and b.decomposition is not None:
            from embedding_art.sae.lens import SAEDecomposition

            interp_acts = (
                (1 - t) * a.decomposition.activations + t * b.decomposition.activations
            ).clamp(min=0.0)
            # Merge feature names from both sides.
            all_names: dict[str, float] = {}
            for name, val in a.decomposition.active_features.items():
                all_names[name] = (1 - t) * val
            for name, val in b.decomposition.active_features.items():
                all_names[name] = all_names.get(name, 0.0) + t * val
            active_features = {n: v for n, v in all_names.items() if v > 1e-8}
            interp_decomp = SAEDecomposition(
                activations=interp_acts,
                active_features=active_features,
                reconstruction_error=max(
                    a.decomposition.reconstruction_error,
                    b.decomposition.reconstruction_error,
                ),
            )

        return Concept(
            embedding=interpolated,
            description=f"slerp({a.description}, {b.description}, {t})",
            decomposition=interp_decomp,
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

        # Combine decompositions if all concepts have them.
        combined_decomp = None
        if all(c.decomposition is not None for c in concepts):
            from embedding_art.sae.lens import SAEDecomposition

            combined_acts = sum(
                w * c.decomposition.activations for w, c in zip(weights, concepts)
            ).clamp(min=0.0)
            all_features: dict[str, float] = {}
            for w, c in zip(weights, concepts):
                for name, val in c.decomposition.active_features.items():
                    all_features[name] = all_features.get(name, 0.0) + w * val
            active_features = {n: v for n, v in all_features.items() if v > 1e-8}
            combined_decomp = SAEDecomposition(
                activations=combined_acts,
                active_features=active_features,
                reconstruction_error=max(c.decomposition.reconstruction_error for c in concepts),
            )

        return Concept(
            embedding=combined,
            description=description,
            decomposition=combined_decomp,
        )

    # === Utility methods ===

    def similarity(self, other: Concept) -> float:
        """Compute cosine similarity with another concept."""
        sim = F.cosine_similarity(self.embedding, other.embedding, dim=-1)
        return sim.item()

    # === SAE integration ===

    def decompose(self, sae: SAELens) -> SAEDecomposition:
        """
        Decompose this concept's embedding into interpretable SAE features.

        The returned ``SAEDecomposition`` contains the sparse activation vector,
        a human-readable ``active_features`` dict (feature name → activation
        strength), and the normalised reconstruction error.

        Parameters
        ----------
        sae:
            A trained :class:`~embedding_art.sae.lens.SAELens` instance whose
            ``embed_dim`` must match this concept's embedding dimension.

        Returns
        -------
        SAEDecomposition
            The sparse decomposition of this concept.
        """
        return sae.decompose(self.embedding)

    @classmethod
    def from_features(
        cls,
        sae: SAELens,
        features: dict[str, float],
    ) -> Concept:
        """
        Build a concept from named SAE features.

        Constructs a synthetic embedding by placing the given feature strengths
        into the sparse activation vector and decoding through the SAE's decoder
        matrix.  The result is L2-normalised so it sits on the unit hypersphere.

        The decomposition is stored on the resulting concept so feature-space
        arithmetic is available immediately.

        Parameters
        ----------
        sae:
            A trained :class:`~embedding_art.sae.lens.SAELens` instance.
        features:
            Mapping of feature name → activation strength.  Any feature not
            listed defaults to zero activation.

        Returns
        -------
        Concept
            A concept whose embedding is the normalised SAE reconstruction and
            whose decomposition is the supplied feature vector.

        Raises
        ------
        ~embedding_art.exceptions.FeatureNotFoundError
            If any key in *features* is not present in the SAE vocabulary.
        """
        from embedding_art.sae.lens import SAEDecomposition

        decomp = SAEDecomposition.from_dict(features, sae)
        embedding = sae.reconstruct(decomp)
        return cls(
            embedding=F.normalize(embedding, dim=-1),
            description=f"features:{features}",
            decomposition=decomp,
        )

    def __repr__(self) -> str:
        device = self.embedding.device
        dim = self.embedding.shape[-1]
        tag = ""
        if self.decomposition is not None:
            n_active = len(self.decomposition.active_features)
            tag = f", features={n_active}"
        return f"Concept({self.description}, dim={dim}, device={device}{tag})"
