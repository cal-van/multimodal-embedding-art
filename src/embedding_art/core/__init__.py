"""Core components: Concept, Engine, Config, Loss, Strategies."""

from embedding_art.core.concept import Concept
from embedding_art.core.concept_spec import ConceptSpec
from embedding_art.core.config import AugmentationConfig, LossConfig, OptimizationConfig
from embedding_art.core.engine import EmbeddingArtEngine, OptimizationResult
from embedding_art.core.loss import CompositeLoss
from embedding_art.core.render_result import LossBreakdown, OptimizationHistory, RenderResult
from embedding_art.core.strategies import (
    DiffusionGuidanceStrategy,
    OptimizationStrategy,
    RenderingStrategy,
)

__all__ = [
    "Concept",
    "ConceptSpec",
    "OptimizationConfig",
    "AugmentationConfig",
    "LossConfig",
    "EmbeddingArtEngine",
    "OptimizationResult",
    "RenderResult",
    "OptimizationHistory",
    "LossBreakdown",
    "CompositeLoss",
    "RenderingStrategy",
    "OptimizationStrategy",
    "DiffusionGuidanceStrategy",
]
