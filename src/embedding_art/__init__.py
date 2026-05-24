"""
Embedding Art: Generate art by optimizing toward coordinates in multimodal embedding space.

v2 Example (multi-encoder):
    from embedding_art import EmbeddingArtEngine, ConceptSpec, OptimizationConfig
    from embedding_art.encoders import create_default_registry

    registry = create_default_registry()
    engine = EmbeddingArtEngine.from_registry(registry, default_encoder="imagebind")
    engine.register_generator('image', image_generator)

    spec = ConceptSpec(text="goldfish")
    result = engine.render(spec, output_modality="image")
    result.output  # [1, 3, H, W] tensor

v1 Example (backward compatible):
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

from embedding_art import upscalers
from embedding_art.core import (
    AugmentationConfig,
    CompositeLoss,
    Concept,
    ConceptSpec,
    EmbeddingArtEngine,
    LossBreakdown,
    LossConfig,
    OptimizationConfig,
    OptimizationHistory,
    OptimizationResult,
    DiffusionGuidanceStrategy,
    OptimizationStrategy,
    RenderingStrategy,
    RenderResult,
)
from embedding_art.regularizers import CompositeRegularizer

try:
    from embedding_art.renderers import (
        DirectRenderer,
        IPAdapterRenderer,
        ProjectionDecoder,
        RawDecoder,
        TextRenderer,
    )
except ImportError:  # pragma: no cover
    pass

try:
    from embedding_art.probes import ActivationProbe, ModelState, StateRenderer
except ImportError:  # pragma: no cover
    pass

try:
    from embedding_art.sae import FeatureRenderer
except ImportError:  # pragma: no cover
    pass

__version__ = "0.2.0"

__all__ = [
    # Core
    "Concept",
    "ConceptSpec",
    "EmbeddingArtEngine",
    "OptimizationConfig",
    "AugmentationConfig",
    "LossConfig",
    "OptimizationResult",
    "RenderResult",
    "OptimizationHistory",
    "LossBreakdown",
    # Loss & Strategies
    "CompositeLoss",
    "RenderingStrategy",
    "OptimizationStrategy",
    "DiffusionGuidanceStrategy",
    # Regularizers
    "CompositeRegularizer",
    # Renderers
    "DirectRenderer",
    "RawDecoder",
    "ProjectionDecoder",
    "IPAdapterRenderer",
    "TextRenderer",
    # Probes
    "ActivationProbe",
    "ModelState",
    "StateRenderer",
    # SAE
    "FeatureRenderer",
    # Upscalers
    "upscalers",
]
