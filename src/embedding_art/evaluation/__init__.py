"""Evaluation toolkit: cross-encoder probes, seed-stability, etc.

Anything that produces metrics about a render (rather than producing
the render itself) lives here. The pieces are deliberately small and
encoder-agnostic so they can be exercised in tests without real model
weights.
"""

from embedding_art.evaluation.probes import (
    ProbeReport,
    compute_cross_encoder_probes,
)
from embedding_art.evaluation.stability import (
    SeedStabilityReport,
    compute_seed_stability,
)

__all__ = [
    "ProbeReport",
    "SeedStabilityReport",
    "compute_cross_encoder_probes",
    "compute_seed_stability",
]
