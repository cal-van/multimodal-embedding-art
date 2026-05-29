# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Embedding Art renders a single concept across all four modalities — image + audio + video + text — in **LanguageBind's shared 768-d embedding space**. This is feature visualisation as art: extracting "platonic ideals" from neural networks.

**Core idea**: Take a concept (text, image, audio, or video), encode it to a single shared embedding, then for each requested output modality gradient-descend a generator's latent until its decoded output maximises cosine similarity with that embedding. Output is a `showcase bundle`: image.png + audio.wav + video.mp4 + text-card.md + interpretation/evaluation manifest.

The headline command is `embed-art showcase`. The legacy `embed-art optimize` command runs the v1/v2 single-modality ImageBind pipeline and is kept only for back-compat — do not use for new work.

## Build & Development Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# LanguageBind is the canonical multimodal encoder (v3 aggressive-rewrite).
# Distributed as a research codebase without setup.py. Two-step install:
#   (a) Clone and add to PYTHONPATH:
git clone https://github.com/PKU-YuanGroup/LanguageBind ../LanguageBind
export PYTHONPATH="$PYTHONPATH:$(pwd)/../LanguageBind"
#   (b) Install LanguageBind's transitive Python deps via this repo's extras:
pip install -e ".[languagebind]"        # Linux / Windows
pip install -e ".[languagebind-macos]"  # Apple Silicon (uses eva-decord
                                        #   because upstream decord has no
                                        #   prebuilt arm64 wheels).

# ImageBind is the v1/v2 encoder, deprecated in v3 but kept for back-compat
git clone https://github.com/facebookresearch/ImageBind
cd ImageBind && pip install -e . && cd ..

# Linting
black --line-length 100 src/ tests/
ruff check src/ tests/

# Tests
pytest
pytest tests/test_concept.py -v  # Single file
pytest -k "test_slerp"           # Single test by name

# CLI — canonical v3 command. On Apple Silicon use the autocast + compile flags for 3-5x wall-time.
embed-art showcase -t "goldfish" -o outputs/goldfish/ \
  --autocast-dtype bf16 --compile-mode reduce-overhead

# Other v3 commands
embed-art interpolate -a "goldfish" -b "flamingo" -s 10 -o image
embed-art anchor-compare --text "thunder" --image storm.jpg --audio thunder.wav
embed-art sae train --sae-type matryoshka --nested-sizes 1024,4096,16384 \
  --embeddings outputs/embeddings.pt --output outputs/sae/
embed-art train-probes --probes-config probes.yaml --output outputs/probes/
embed-art profile --encoder languagebind --output traces/

# Legacy v2 single-modality pipeline. Kept for back-compat only — slow on MPS (no autocast, no compile)
# and superseded by `showcase`. Do not use for new work.
embed-art optimize -t "goldfish" 1.0 -o image
```

## Architecture

### Data Flow (v3 canonical, via `embed-art showcase`)

```
Input (text/image/audio/video) → LanguageBind encoder → Target embedding (768d, shared)
                                                            ↓
   ┌───────────────────────────────────────────────────────┴────────────────┐
   │  Per requested output modality, in turn:                                │
   │                                                                          │
   │   Random latent → Generator (SD3.5 / Stable Audio Open /                │
   │   ↑              LTX-Video) → Decoded output                            │
   │   │                                            ↓                        │
   │   │              LanguageBind (re-encode same modality, shared space) ←─┘
   │   │                                            ↓
   │   └──── Gradient update ←── Loss: -cosine_sim + PatchAlignment + text_anchor +
   │                                    regularization (TV + spectral + latent_norm)
   │                                    + heavy regularisation on the natural track
   │                                    (VSD/SDS prior planned, not yet wired)
   └──── Interpretation bundle (SAE + text-anchor + attribution + linear probes)
         + Evaluation card (cross-modal Jaccard + cross-encoder probes + seed-stability)
```

### Legacy v2 data flow (only via `embed-art optimize`)

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
│   ├── concept.py        # Concept class with embedding arithmetic + SAEDecomposition co-state
│   ├── config.py         # OptimizationConfig (autocast_dtype, compile_mode), AugmentationConfig, LossConfig
│   ├── engine.py         # EmbeddingArtEngine, OptimizationResult
│   ├── loss.py           # CompositeLoss: cosine + patch_alignment + text_anchor (VSD/SDS prior planned, not yet wired)
│   └── patch_alignment.py  # Multi-layer ViT patch-cosine alignment
├── encoders/
│   ├── base.py             # Encoder protocol (v3 duck-typed)
│   ├── languagebind.py     # LanguageBind (canonical v3 encoder, all 4 modalities)
│   ├── siglip2.py          # SigLIP 2 SO400M (image probe)
│   ├── clap.py             # CLAP (audio probe)
│   ├── imagebind.py        # ImageBind (deprecated v1/v2)
│   ├── defaults.py         # Encoder registry: languagebind, siglip2-so400m, clap-general, imagebind
│   └── registry.py         # EncoderRegistry
├── generators/
│   ├── base.py             # Generator protocol
│   ├── sd35.py             # SD3.5-medium VAE (v3 canonical image)
│   ├── stable_audio_open.py  # Stable Audio Open (v3 canonical audio)
│   ├── ltx_video.py        # LTX-Video (v3 canonical video)
│   ├── image.py            # SDXLImageGenerator (legacy v2)
│   ├── audio.py            # AudioLDM2 (legacy v2)
│   └── video.py            # SVD (legacy v2)
├── sae/
│   ├── base.py             # GroupSparseSAE + MatryoshkaSAE (nested-prefix)
│   ├── training.py         # train_*_sae_pipeline() + load_sae_from_dir() auto-detection
│   └── labelling.py        # CosineLabeller, VLMLabeller, ActivatingExamplesLabeller
├── interpretation/
│   └── bundle.py           # InterpretationBundle: text-anchor + SAE features + attribution + linear probes
├── evaluation/
│   ├── probes.py           # compute_cross_encoder_probes() — does another encoder agree?
│   └── stability.py        # compute_seed_stability() — variance across N re-runs
├── experiments/
│   ├── anchor_comparison.py  # Multi-modal anchor-compare (image+audio+video+text pairwise matrix)
│   ├── activation_cache.py   # Content-addressed encoder activation cache
│   └── text_anchor_sweep.py  # M4 empirical sweep harness
├── perf/
│   ├── compile.py          # torch.compile wrapper with silent fallback
│   ├── attention.py        # SDPA flash-attention routing for LanguageBind ViT
│   ├── diffusers_knobs.py  # QKV fusion + VAE tiling + attention slicing
│   ├── probe_registry.py   # CoreML/ANE conversion candidates (SigLIP2/CLAP/DINOv3)
│   └── mlx_sae.py          # MLX SAE-training backend port
├── priors/
│   └── sd35_score.py       # SDS + VSD loss math; LoRA phi-adapter injection
├── regularizers/
│   ├── base.py             # TotalVariation, SpectralRegularizer, LatentNorm, CompositeRegularizer
│   └── audio.py            # AudioTotalVariation
├── web/                    # FastAPI routes: jobs, /jobs/showcase, /experiments/anchor-compare
└── cli/
    ├── main.py             # Click CLI group
    └── commands/
        ├── showcase.py     # `embed-art showcase` — headline command
        ├── anchor_compare.py
        ├── sae.py          # `sae train`, `sae train-stack`, `sae auto-label`, `sae collect`
        ├── train_probes.py
        ├── compile_probes.py
        ├── profile.py
        ├── validate_vsd.py
        ├── text_anchor_sweep.py
        └── optimize.py     # Legacy v2 single-modality command (deprecated)
```

### Key Classes

**Concept** (`core/concept.py`)
- Wraps normalised embedding tensor with source description
- Co-primary state: optional `SAEDecomposition` (when both operands carry one, arithmetic composes in feature space too)
- Arithmetic ops: `a + b`, `a - 0.3*b`, `Concept.slerp(a, b, t)`, `Concept.combine([...], weights)`
- Factory methods: `from_text()`, `from_image()`, `from_audio()`, `from_video()`

**EmbeddingArtEngine** (`core/engine.py`)
- Registers generators by modality name
- `render(target, encoder_name, output_modality, config)` → `RenderResult` (v3 canonical)
- `optimize(target, output_modality, config)` → `OptimizationResult` (legacy v2 alias kept for back-compat)
- `interpolation_series(a, b, steps)` for concept morphs

**OptimizationConfig** (`core/config.py`)
- `steps`, `learning_rate`, `optimizer`, `scheduler`
- `autocast_dtype` (`fp32` / `fp16` / `bf16`) — Apple Silicon perf knob; backward stays fp32 for stability
- `compile_mode` (`none` / `default` / `reduce-overhead`) — `torch.compile` knob with silent fallback
- `augmentation` settings (random_crop, random_flip)
- `checkpoint_every`, `seed`

**LossConfig** (`core/config.py`)
- `similarity_weight`, `feature_matching_weight`, `text_anchor_weight`
- `regularization`: composite regulariser instance
- Showcase composes per-track LossConfigs (`honest` = minimal regularisation, `natural` = heavy regularisation; VSD/SDS prior planned, not yet wired — heavy regularisation is the current proxy)

**Regularizers** (`regularizers/base.py`)
- `TotalVariation`: Spatial smoothness
- `SpectralRegularizer`: Penalise high-frequency noise
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

Targets M1 Max with 64GB unified memory. Real-weight memory under v3 canonical backbones:

| Component | Memory |
|-----------|--------|
| LanguageBind (all 4 sub-encoders) | ~6GB |
| SD3.5-medium VAE | ~2GB |
| Stable Audio Open | ~3GB |
| LTX-Video | ~5GB |
| Optimisation overhead | ~3GB |
| **Total active (full showcase)** | **~19GB** |
| **Available headroom (64GB)** | **~45GB** |

```python
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
```

**Apple Silicon perf checklist** (validated knobs, all wired in v3):
- `--autocast-dtype bf16` — fp32 backward retained for stability
- `--compile-mode reduce-overhead` — silent fallback if MPS Inductor rejects an op
- `enable_sdpa=True` on `LanguageBindEncoder` (default on MPS/CUDA) — flash-attention via PyTorch's native SDPA
- diffusers `fuse_qkv_projections()` + `vae.enable_tiling()` + `enable_attention_slicing("max")` — auto-applied to SD3.5 / SAO / LTX-Video
- `torch.mps.empty_cache()` between modalities — pre-allocated step buffers reduce allocator churn
- CoreML/ANE for inference-only probes (gated behind `embed-art compile-probes` runbook)
- MLX SAE-training backend (gated behind `--backend mlx` runbook)

## Success Metrics

- **Cosine to target (convergence diagnostic, not a success metric).** This is the optimisation objective reported back to itself, so it's circular — a measure of whether the loop converged, not independent validation. Typical achieved values are ~0.15–0.45 with the current generator backbones, and that's expected: the generator manifold plus a cosine objective don't reach the ">0.85" an earlier spec assumed. Don't read a low cosine as failure.
- **Cross-encoder probe (the genuinely independent signal).** Re-encode the output with a *different* encoder (SigLIP2 for image, CLAP for audio) and check whether it agrees the artefact reads as the concept. Because it's a separate model from the one being optimised against, agreement here is the real validation — not the cosine above.
- **Cross-modal agreement.** What the code actually computes is the Jaccard overlap of the top-K text-anchor *words* between modalities — a lexical agreement proxy, not embedding cosine. A true cross-modal embedding-cosine metric (e.g. "outputs embed within 0.90 of each other") is *planned*, not yet computed.
- **Full optimization <30 minutes on M1 Max.**
