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

## Testing Strategy: Red-Green-Refactor TDD

This project follows strict TDD with the Red-Green-Refactor cycle:

1. **Red**: Write a failing test that defines expected behavior
2. **Green**: Write minimal code to make the test pass
3. **Refactor**: Clean up while keeping tests green

### TDD Workflow

```bash
# 1. Write test first
pytest tests/test_concept.py::test_slerp_midpoint -v  # Should fail (Red)

# 2. Implement minimal code to pass
pytest tests/test_concept.py::test_slerp_midpoint -v  # Should pass (Green)

# 3. Refactor if needed, run all tests
pytest tests/test_concept.py -v  # All green
```

### What to TDD

**Unit tests (pure functions, no I/O):**
- `Concept` arithmetic: add, subtract, scalar multiply, slerp, combine
- `Concept` normalization: embeddings always unit norm after operations
- Regularizers: TotalVariation, SpectralRegularizer, LatentNorm math
- Config validation and defaults

**Integration tests (mark with `@pytest.mark.slow`):**
- Full optimization loop with real/mocked models
- Encoder/generator round-trips
- CLI end-to-end

### Mock Strategy

```python
# conftest.py fixtures
@pytest.fixture
def mock_encoder():
    """Returns deterministic embeddings for testing."""
    encoder = Mock(spec=Encoder)
    encoder.encode_text.return_value = torch.randn(1, 1024)
    return encoder

@pytest.fixture
def mock_generator():
    """Returns simple decoded tensors."""
    gen = Mock(spec=Generator)
    gen.latent_shape = (1, 4, 128, 128)
    gen.decode.return_value = torch.rand(1, 3, 1024, 1024)
    return gen
```

### Test File Organization

```
tests/
├── conftest.py          # Shared fixtures
├── test_concept.py      # Concept arithmetic and properties
├── test_regularizers.py # Regularizer math
├── test_config.py       # Config validation
├── test_engine.py       # Engine with mocked deps
└── test_integration.py  # @pytest.mark.slow real model tests
```

## Hardware

Targets M1 Max with 64GB unified memory. Plenty of headroom for all models simultaneously.

| Component | Memory |
|-----------|--------|
| ImageBind | ~3GB |
| SDXL VAE | ~1GB |
| AudioLDM 2 | ~2GB |
| SVD (video) | ~4GB |
| Optimization overhead | ~2GB |
| **Available headroom** | **~50GB** |

```python
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
```

## Success Metrics

- Cosine similarity >0.85 with target embedding
- Full optimization <30 minutes on M1 Max
- Cross-modal outputs embed within 0.90 similarity of each other
