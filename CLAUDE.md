# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Embedding Art renders model internal states directly as different modalities — images, audio, video, text. It visualizes what neural networks actually represent, from individual features to full embeddings, providing a true view of a model's learned concepts.

**Core idea**: Take a model's internal state (embedding, layer activation, SAE feature) and render it directly as output. Multiple rendering paths trade off between faithfulness and aesthetics — from raw linear projections (alien but faithful) to IP-Adapter conditioned diffusion (aesthetic but approximate) to optimization loops (highest quality, slowest).

**Three rendering approaches** (v3):
1. **Direct rendering** — single forward pass from embedding to output (RawDecoder, ProjectionDecoder, IPAdapterRenderer)
2. **State probing** — capture and render intermediate layer activations to show concept formation
3. **Feature visualization** — decompose embeddings via SAE and render individual features

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

# CLI — v3 direct rendering
embed-art direct -t "goldfish" 1.0 --renderer raw
embed-art direct -t "goldfish" 1.0 --renderer ip-adapter
embed-art probe -t "goldfish" 1.0 --encoder siglip2 --layers 0,6,12,18,26
embed-art feature-viz -t "goldfish" 1.0 --sae sae_models/siglip2.safetensors

# CLI — v1/v2 optimization
embed-art optimize -t "goldfish" 1.0 -o image
embed-art interpolate -a "goldfish" -b "flamingo" -s 10 -o image

# Beads issue tracking
bd ls                    # List issues
bd add "Issue title"     # Create issue
bd show <id>             # View issue details
```

## Architecture

### Data Flow (v3 — Direct Rendering)

```
Most faithful                                              Most aesthetic
     ↓                                                          ↓
┌──────────┐   ┌──────────┐   ┌───────────┐   ┌───────────────┐
│ Raw      │   │Projection│   │ IP-Adapter │   │ Optimization  │
│ Decode   │   │ Decoder  │   │ Guided     │   │ Loop (v1/v2)  │
│ (~0s)    │   │ (~0.1s)  │   │ (~10s)     │   │ (~5-30min)    │
└──────────┘   └──────────┘   └───────────┘   └───────────────┘
```

```
v3 direct path:  Input → Encoder → Embedding → DirectRenderer → Output
v3 state probe:  Input → Encoder layers 0..N → Per-layer render → Progression
v3 feature viz:  Input → Encoder → SAE decompose → Per-feature render → Grid
v1/v2 optimize:  Input → Encoder → Target → [gradient loop × 500] → Output
```

### Package Structure

```
src/embedding_art/
├── __init__.py           # Public API exports
├── core/
│   ├── concept.py        # Concept class with embedding arithmetic
│   ├── concept_spec.py   # Encoder-agnostic concept specification
│   ├── config.py         # OptimizationConfig, AugmentationConfig, LossConfig
│   ├── engine.py         # EmbeddingArtEngine (render, render_direct, render_state, render_features)
│   ├── loss.py           # CompositeLoss (cosine sim + feature matching + SAE + regularization)
│   ├── strategies.py     # OptimizationStrategy, DiffusionGuidanceStrategy
│   └── render_result.py  # RenderResult, OptimizationHistory, LossBreakdown
├── renderers/            # v3: Direct rendering pipeline
│   ├── base.py           # DirectRenderer protocol
│   ├── raw.py            # Linear embedding→pixel decoder (most faithful)
│   ├── projection.py     # Learned MLP decoder
│   ├── ip_adapter.py     # IP-Adapter conditioned diffusion
│   └── text.py           # Embedding→text via nearest neighbor
├── probes/               # v3: Model state capture
│   ├── activation_probe.py  # Hook-based multi-layer capture
│   └── state_renderer.py    # Render captured states per-layer
├── encoders/
│   ├── base.py           # Encoder protocol
│   ├── imagebind.py      # ImageBind wrapper (6 modalities, 1024d)
│   ├── siglip2.py        # SigLIP2 (text+image, 1152d, multi-layer)
│   ├── clap.py           # CLAP (text+audio, 512d)
│   └── registry.py       # EncoderRegistry with capability flags
├── generators/
│   ├── base.py           # Generator/LatentGenerator/DirectGenerator/DiffusionGenerator protocols
│   ├── image.py          # SDXLImageGenerator
│   ├── audio.py          # AudioLDM2 generator
│   ├── video.py          # SVD video generator
│   ├── inr.py            # Implicit Neural Representation (DirectGenerator)
│   └── diffusion.py      # Embedding-guided SDXL diffusion
├── sae/                  # Sparse autoencoder for interpretable features
│   ├── lens.py           # SAELens: decompose/reconstruct/manipulate
│   ├── training.py       # SAE training pipeline
│   └── feature_renderer.py  # v3: Per-feature visualization
├── regularizers/
│   └── base.py           # TotalVariation, SpectralRegularizer, LatentNorm, CompositeRegularizer
└── cli/
    ├── main.py           # Click CLI (embed-art command)
    └── commands/         # direct, probe, feature-viz, optimize, render, etc.
```

### Key Classes

**DirectRenderer** (`renderers/base.py`) — v3 protocol for single-pass rendering
- `RawDecoder`: Linear projection, no prior, most faithful
- `ProjectionDecoder`: Learned MLP to generator latent space
- `IPAdapterRenderer`: Pretrained adapter + SDXL diffusion
- `TextRenderer`: Nearest-neighbor text decode

**ActivationProbe** (`probes/activation_probe.py`) — captures multi-layer encoder states
**StateRenderer** (`probes/state_renderer.py`) — renders ModelState through any renderer
**FeatureRenderer** (`sae/feature_renderer.py`) — renders individual SAE features

**EmbeddingArtEngine** (`core/engine.py`)
- `render_direct(spec, renderer)` — v3 direct rendering (no optimization)
- `render_state(spec, renderer, layers)` — v3 multi-layer state rendering
- `render_features(spec, renderer, sae)` — v3 SAE feature visualization
- `render(spec, output_modality, config)` — v2 optimization-based rendering
- `optimize(target, output_modality, config)` — v1 backward-compatible

**Concept** (`core/concept.py`)
- Wraps normalized embedding tensor with source description
- Arithmetic ops: `a + b`, `a - 0.3*b`, `Concept.slerp(a, b, t)`, `Concept.combine([...], weights)`

**SAELens** (`sae/lens.py`)
- Decomposes opaque embeddings into named interpretable features
- `decompose()`, `reconstruct()`, `manipulate()` for feature-level control

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

- Direct rendering produces output in <1s (raw), <15s (IP-Adapter)
- Feature decomposition + rendering in <30s for top-10 features
- Multi-layer state probe captures all layers in single forward pass
- Optimization loop (legacy): cosine similarity >0.85 within 2000 steps
- Cross-modal outputs embed within 0.90 similarity of each other
