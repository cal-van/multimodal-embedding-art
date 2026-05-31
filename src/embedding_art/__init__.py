# ruff: noqa: E402
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

import sys
from pathlib import Path

# Automatically add LanguageBind to path if found as a sibling or local subdirectory
for path_cand in [
    Path(__file__).resolve().parents[2] / "LanguageBind",  # local subfolder in repo root
    Path.cwd() / "LanguageBind",  # local subfolder in CWD
    Path(__file__).resolve().parents[3] / "LanguageBind",  # sibling folder
    Path.cwd().parent / "LanguageBind",  # sibling folder in CWD parent
]:
    if path_cand.exists() and str(path_cand.resolve()) not in sys.path:
        sys.path.insert(0, str(path_cand.resolve()))
        break

# Dynamic compatibility patch: newer transformers versions lack _expand_mask in clip modeling
try:
    import torch
    import transformers.models.clip.modeling_clip as modeling_clip

    if not hasattr(modeling_clip, "_expand_mask"):

        def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: int | None = None):
            bsz, src_len = mask.size()
            tgt_len = tgt_len if tgt_len is not None else src_len
            expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
            inverted_mask = 1.0 - expanded_mask
            return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

        modeling_clip._expand_mask = _expand_mask

    # Newer torchaudio versions removed set_audio_backend
    import torchaudio

    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda backend: None
except Exception:
    pass


from embedding_art import upscalers
from embedding_art.core import (
    AugmentationConfig,
    CompositeLoss,
    Concept,
    ConceptSpec,
    DiffusionGuidanceStrategy,
    EmbeddingArtEngine,
    LossBreakdown,
    LossConfig,
    OptimizationConfig,
    OptimizationHistory,
    OptimizationResult,
    OptimizationStrategy,
    RenderingStrategy,
    RenderResult,
)
from embedding_art.regularizers import CompositeRegularizer

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
    # Upscalers
    "upscalers",
]
