"""
Types for tracking optimization progress and final render outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from embedding_art.core.config import OptimizationConfig


@dataclass
class LossBreakdown:
    """Every loss component tracked separately for logging/debugging."""

    total: torch.Tensor
    components: dict[str, torch.Tensor]


class OptimizationHistory:
    """Tracks all loss components across optimization steps."""

    def __init__(self) -> None:
        self.steps: list[int] = []
        self.loss_breakdowns: list[dict[str, float]] = []
        self.similarity_values: list[float] = []

    def record(self, step: int, breakdown: LossBreakdown) -> None:
        """Append one optimization step's losses to the history.

        The 'similarity' component is stored as its negation so that
        similarity_values always contains positive cosine similarity scores
        (optimization minimizes -cosine_sim, so the raw component is negative).
        """
        self.steps.append(step)
        self.loss_breakdowns.append({k: v.item() for k, v in breakdown.components.items()})
        if "similarity" in breakdown.components:
            self.similarity_values.append(-breakdown.components["similarity"].item())

    @property
    def final_similarity(self) -> float:
        """Return the most recently recorded cosine similarity, or 0.0 if empty."""
        return self.similarity_values[-1] if self.similarity_values else 0.0

    def __len__(self) -> int:
        return len(self.steps)


@dataclass
class RenderResult:
    """Output of any rendering strategy."""

    output: torch.Tensor
    history: OptimizationHistory
    encoder_name: str
    final_similarity: float
    config: OptimizationConfig | None = None
    checkpoints: list[Path] | None = None
