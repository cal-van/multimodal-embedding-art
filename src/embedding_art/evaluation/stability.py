"""Seed-stability evaluation for the M8 evaluation card.

Re-running an optimisation with different seeds tells us whether the
showcase artefact is a stable feature of the embedding-space target
or an attractor reachable only from specific initialisations. This
module *summarises* the per-seed results — it does not run the
optimisations itself, so it stays dependency-light and instantly
testable.

The expected caller shape: run ``embed-art showcase`` N times with
different seeds, collect the resulting similarities and SAE feature
sets, then pass them here for a single :class:`SeedStabilityReport`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SeedStabilityReport:
    """Summary of how stable a showcase rendering is across seeds.

    Attributes:
        n_seeds: Number of independent runs that contributed.
        mean_similarity: Mean of ``final_similarity`` across runs.
        std_similarity: Sample standard deviation across runs.
        min_similarity: Worst-case run.
        max_similarity: Best-case run.
        feature_overlap_jaccard: Mean pairwise Jaccard overlap of the
            top-K SAE feature sets across runs. ``None`` when no
            per-run feature sets were supplied.
    """

    n_seeds: int
    mean_similarity: float
    std_similarity: float
    min_similarity: float
    max_similarity: float
    feature_overlap_jaccard: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "n_seeds": self.n_seeds,
            "mean_similarity": self.mean_similarity,
            "std_similarity": self.std_similarity,
            "min_similarity": self.min_similarity,
            "max_similarity": self.max_similarity,
            "feature_overlap_jaccard": self.feature_overlap_jaccard,
        }


def compute_seed_stability(
    *,
    similarities: list[float],
    feature_sets: list[set[int]] | None = None,
) -> SeedStabilityReport:
    """Summarise per-seed showcase results into a stability report.

    Args:
        similarities: Per-run ``final_similarity`` values (one per
            seed). Must contain at least 1 element.
        feature_sets: Optional per-run set of top-K SAE feature
            indices. When provided, ``feature_overlap_jaccard`` is the
            *mean pairwise* Jaccard over all ``C(n, 2)`` pairs.

    Raises:
        ValueError: When ``similarities`` is empty.
    """
    if not similarities:
        raise ValueError("similarities must contain at least one value")

    n = len(similarities)
    mean = sum(similarities) / n
    var = sum((s - mean) ** 2 for s in similarities) / max(n - 1, 1)
    std = math.sqrt(var)

    overlap: float | None = None
    if feature_sets is not None:
        if len(feature_sets) != n:
            raise ValueError(
                f"feature_sets length {len(feature_sets)} != " f"similarities length {n}"
            )
        if n < 2:
            overlap = 1.0
        else:
            pairs = 0
            total = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = feature_sets[i], feature_sets[j]
                    union = a | b
                    if union:
                        total += len(a & b) / len(union)
                    else:
                        total += 1.0
                    pairs += 1
            overlap = total / pairs

    return SeedStabilityReport(
        n_seeds=n,
        mean_similarity=mean,
        std_similarity=std,
        min_similarity=min(similarities),
        max_similarity=max(similarities),
        feature_overlap_jaccard=overlap,
    )
