"""
Embedding Art: Generate art by optimizing toward coordinates in multimodal embedding space.

Example:
    from embedding_art import EmbeddingArtEngine, Concept, OptimizationConfig
    from embedding_art.encoders import ImageBindEncoder
    from embedding_art.generators import SDXLImageGenerator

    encoder = ImageBindEncoder(device='mps')
    generator = SDXLImageGenerator(device='mps')

    engine = EmbeddingArtEngine(encoder)
    engine.register_generator('image', generator)

    target = Concept.from_text('goldfish', encoder)
    result = engine.optimize(target, 'image', OptimizationConfig(steps=500))

    result.get_final_image(generator).save('goldfish.png')
"""

from embedding_art.core import (
    AugmentationConfig,
    Concept,
    EmbeddingArtEngine,
    OptimizationConfig,
    OptimizationResult,
)
from embedding_art.regularizers import CompositeRegularizer

__version__ = "0.1.0"

__all__ = [
    # Core
    "Concept",
    "EmbeddingArtEngine",
    "OptimizationConfig",
    "AugmentationConfig",
    "OptimizationResult",
    # Regularizers
    "CompositeRegularizer",
]
