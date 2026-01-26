# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Embedding Art generates images (audio/video planned) by optimizing generative model latents toward target coordinates in ImageBind's multimodal embedding space. This is feature visualization as art—extracting "platonic ideals" from neural networks.

**Core idea**: Take a concept (text, image, audio), encode it to embedding space, then gradient-descent a VAE latent until its decoded output maximizes cosine similarity with that embedding.

## Build & Development Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# ImageBind must be installed separately (required encoder)
git clone https://github.com/facebookresearch/ImageBind
cd ImageBind && pip install -e . && cd ..

# Linting
black --line-length 100 src/ tests/
ruff check src/ tests/

# Tests
pytest
pytest tests/test_concept.py -v  # Single file
pytest -k "test_slerp"           # Single test by name

# CLI
embed-art optimize -t "goldfish" 1.0 -o image
embed-art interpolate -a "goldfish" -b "flamingo" -s 10 -o image

# Beads issue tracking
bd ls                    # List issues
bd add "Issue title"     # Create issue
bd show <id>             # View issue details
```

## Architecture

### Data Flow

```
Input (text/image/audio) → ImageBind encoder → Target embedding (1024d)
                                                      ↓
Random latent → Generator (SDXL VAE) → Decoded image → ImageBind → Current embedding
     ↑                                                                    ↓
     └──────────────── Gradient update ←── Loss: -cosine_sim + regularization
```

### Package Structure

```
src/embedding_art/
├── __init__.py           # Public API exports
├── core/
│   ├── concept.py        # Concept class with embedding arithmetic
│   ├── config.py         # OptimizationConfig, AugmentationConfig
│   └── engine.py         # EmbeddingArtEngine, OptimizationResult
├── encoders/
│   ├── base.py           # Encoder protocol
│   └── imagebind.py      # ImageBind wrapper
├── generators/
│   ├── base.py           # Generator protocol
│   └── image.py          # SDXLImageGenerator
├── regularizers/
│   └── base.py           # TotalVariation, SpectralRegularizer, LatentNorm, CompositeRegularizer
└── cli/
    └── main.py           # Click CLI (embed-art command)
```

### Key Classes

**Concept** (`core/concept.py`)
- Wraps normalized embedding tensor with source description
- Arithmetic ops: `a + b`, `a - 0.3*b`, `Concept.slerp(a, b, t)`, `Concept.combine([...], weights)`
- Factory methods: `from_text()`, `from_image()`, `from_audio()`, `from_video()`

**EmbeddingArtEngine** (`core/engine.py`)
- Registers generators by modality name
- `optimize(target, output_modality, config)` → OptimizationResult
- `interpolation_series(a, b, steps)` for concept morphs

**OptimizationConfig** (`core/config.py`)
- `steps`, `learning_rate`, `optimizer`, `scheduler`
- `augmentation` settings (random_crop, random_flip)
- `checkpoint_every`, `seed`

**Regularizers** (`regularizers/base.py`)
- `TotalVariation`: Spatial smoothness
- `SpectralRegularizer`: Penalize high-frequency noise
- `LatentNorm`: Keep latent near typical VAE distribution
- `CompositeRegularizer.default_image()`, `.minimal()`, `.heavy()` presets

## Testing Strategy

**TDD is appropriate for:**
- `Concept` class: arithmetic operations, normalization, slerp math
- Regularizers: mathematical properties (TV, spectral, norm calculations)
- Config validation
- CLI argument parsing

**Integration tests for:**
- Full optimization loop (requires models - mark with `@pytest.mark.slow`)
- Encoder/generator round-trips

**Mock strategy:**
- Mock `ImageBindEncoder` with fake embeddings for unit tests
- Mock `Generator.decode()` with simple tensor transforms

## Hardware

Targets Apple Silicon (MPS backend). Memory budget ~6-8GB for ImageBind + SDXL VAE.

```python
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
```

## Success Metrics

- Cosine similarity >0.85 with target embedding
- Full optimization <30 minutes on M1 Max
- Cross-modal outputs embed within 0.90 similarity of each other
