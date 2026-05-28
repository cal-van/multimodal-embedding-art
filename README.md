# Embedding Art

[![CI](https://github.com/cal-van/multimodal-embedding-art/actions/workflows/ci.yml/badge.svg)](https://github.com/cal-van/multimodal-embedding-art/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

Generate art by optimising toward coordinates in a shared multimodal embedding space.

This system extracts the "platonic ideals" that live inside neural networks — not "generate me a goldfish" but "show me the direction in representation space that means goldfish, across every modality, cranked to maximum."

## What it does

- Takes a concept (text, image, audio, or video) and encodes it to a shared 768-d **LanguageBind** embedding.
- Renders that single target across all four modalities — image + audio + video + text — in one orchestrated showcase.
- Optionally renders a dual track (`honest` = what the model thinks; `natural` = VSD-prior conditioned).
- Produces an interpretation bundle (text-anchor readout, SAE feature lists, attribution maps, linear-probe activations) and an evaluation card (cross-modal Jaccard agreement, cross-encoder probes, seed-stability) per run.

The outputs are what the model *thinks* these concepts look like — maximally activated representations, not training-data lookups.

## Quick Start

### 1. Install this package

```bash
git clone https://github.com/cal-van/multimodal-embedding-art
cd multimodal-embedding-art

python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### 2. Install LanguageBind (canonical encoder)

LanguageBind is **not on PyPI** — it's distributed as a research codebase without `setup.py`. Two-step install:

**2a. Clone the LanguageBind repo and add it to `PYTHONPATH`:**

```bash
# From the parent directory of this repo:
git clone https://github.com/PKU-YuanGroup/LanguageBind
export PYTHONPATH="$PYTHONPATH:$(pwd)/LanguageBind"
# Add that export to ~/.zshrc (macOS) or ~/.bashrc so it persists across shells.
```

**2b. Install LanguageBind's transitive Python deps via this repo's extras group:**

```bash
# Linux / Windows
pip install -e ".[languagebind]"

# Apple Silicon (uses eva-decord instead of decord — upstream decord has no
# prebuilt arm64 wheels and source builds frequently fail on M1/M2/M3).
pip install -e ".[languagebind-macos]"
```

> **eva-decord on macOS arm64**: `eva-decord` is a community-maintained fork of the upstream DMLC `decord` video reader that ships prebuilt macOS arm64 wheels. Drop-in compatible (same `import decord` module name). LanguageBind requires the video reader for its video sub-encoder, but the upstream package has no arm64 wheel — so on Apple Silicon you either accept a long source build (which often fails on recent macOS SDKs) or use the fork. We default to the fork on Apple Silicon for this reason. If you'd rather build upstream `decord` from source, run `pip install -e ".[languagebind]"` on Apple Silicon too.

### 3. Run the canonical showcase

```bash
# On Apple Silicon, recommended flags give ~3-5x wall-time improvement over fp32:
embed-art showcase -t 'goldfish' -o outputs/goldfish/ \
  --autocast-dtype bf16 \
  --compile-mode reduce-overhead
```

This writes `outputs/goldfish/image.png`, `audio.wav`, `video.mp4`, `text-card.md`, and `manifest.json` (interpretation bundle + evaluation card) into the output directory.

A minimal smoke test that doesn't load the heavy generators (good for verifying the LanguageBind install path works at all):

```bash
embed-art showcase -t 'thunder' -o outputs/thunder/ --modalities text --steps 10
```

### 4. Optional: install legacy ImageBind for v1/v2 artefacts

ImageBind is **deprecated in v3** but kept for back-compat with old artefacts:

```bash
git clone https://github.com/facebookresearch/ImageBind
cd ImageBind && pip install -e . && cd ..
```

## Examples

Runnable recipes for the main workflows live in [`examples/`](examples/). Each
starts with a fast smoke variant so you can verify your install before a long
run:

```bash
bash examples/01_showcase.sh        # one concept → all four modalities
python examples/02_concept_algebra.py  # add / subtract / interpolate concepts
bash examples/03_anchor_compare.sh  # cross-modal embedding agreement
bash examples/04_interpolate.sh     # render a morph between two concepts
```

See [`examples/README.md`](examples/README.md) for details and runtime costs.

## Headline command: `embed-art showcase`

```bash
embed-art showcase -t '<concept>' -o <output_dir>/ [flags]
```

Key flags (full list via `embed-art showcase --help`):

| Flag | Default | What it does |
|---|---|---|
| `--modalities image,audio,video,text` | all four | Which modalities to render. |
| `--steps 2000` | 2000 | Optimisation steps per modality. |
| `--autocast-dtype bf16` | `fp32` | fp16 / bf16 forward (fp32 backward). Big win on Apple Silicon. |
| `--compile-mode reduce-overhead` | `none` | `torch.compile` for the optimisation hot loop. Silent fallback if MPS Inductor balks. |
| `--tracks honest,natural` | `honest` | Render both tracks side-by-side. Natural track uses VSD prior. |
| `--image-backbone sd35` | `sd35` | `sd35` (default) or `sdxl` (legacy ablation). |
| `--audio-backbone stable-audio-open` | `stable-audio-open` | `stable-audio-open` (default) or `audioldm2` (legacy). |
| `--video-backbone ltx-video` | `ltx-video` | `ltx-video` (default) or `svd` (legacy). |
| `--probes siglip2-so400m,clap-general` | empty | Comma-separated cross-encoder probes for the M8 eval card. |
| `--seed-stability N` | 0 | Re-run N extra times with distinct seeds; aggregates per-modality variance into the manifest. |
| `--linear-probes-dir <path>` | none | Directory of a trained linear-probe manifest (see `embed-art train-probes`). |
| `--text-anchor-weight 0.25` | 0.0 | Auxiliary loss pulling the optimisation embedding toward the encoder's text projection. |

## Other useful CLI commands

```bash
# Render an interpolation series between two concepts.
embed-art interpolate -a "goldfish" -b "flamingo" -s 10 -o image -d outputs/interp/

# Multimodal anchor-comparison: encode the same concept through 1-4 modalities, report pairwise cosine.
embed-art anchor-compare --text "thunder" --image storm.jpg --audio thunder.wav

# Inspect / compare embeddings.
embed-art embed -t "goldfish" -s embeddings/goldfish.pt
embed-art compare embeddings/goldfish.pt embeddings/orange.pt

# Train per-modality SAEs (Matryoshka by default in v3, GroupSparse still available).
embed-art sae train --sae-type matryoshka --nested-sizes 1024,4096,16384 \
  --embeddings outputs/embeddings.pt --output outputs/sae/

# Train linear probes for the interpretation bundle's "linear probes" row.
embed-art train-probes --probes-config probes.yaml --output outputs/probes/

# Apple Silicon perf harness: Chrome trace + per-op timing of one forward+backward step.
embed-art profile --encoder languagebind --output traces/

# Web UI (FastAPI + Vite).
embed-art web --port 8000 --reload
# Frontend (separate terminal):
cd ui && npm install && npm run dev
```

## Concept algebra (Python API)

```python
from embedding_art import Concept
from embedding_art.encoders.languagebind import LanguageBindEncoder

encoder = LanguageBindEncoder(device="mps")

# Addition
fire_water = Concept.from_text("fire", encoder) + Concept.from_text("water", encoder)

# Weighted combination
sunset_ocean = 0.3 * Concept.from_text("sunset", encoder) + 0.7 * Concept.from_text("ocean", encoder)

# Subtraction
hairless = Concept.from_text("dog", encoder) - 0.3 * Concept.from_text("fur", encoder)

# Spherical interpolation
midpoint = Concept.slerp(concept_a, concept_b, t=0.5)

# Cross-modal (LanguageBind binds all four modalities into the same space)
thunder_purple = Concept.from_audio("thunder.wav", encoder) + 0.3 * Concept.from_text("purple", encoder)
```

In v3 the concept's `SAEDecomposition` is co-primary state with the embedding — arithmetic composes decompositions in feature space when both operands carry one.

## Legacy v1/v2 commands (deprecated)

Kept for back-compat with the original ImageBind-centric workflow. These run a single modality through the v2 optimisation loop and don't go through LanguageBind. **Do not use for new work** — they are 3-5× slower and produce single-modality artefacts only.

```bash
# Legacy single-modality optimisation (v2). Slow and modality-island; superseded by `showcase`.
embed-art optimize -t "goldfish" 1.0 -o image
embed-art optimize -t "fire" 0.5 -t "water" 0.5 -o image
embed-art optimize -a thunder.wav 1.0 -t "purple" 0.3 -o image -p thunder_purple.png
```

## Configuration

`OptimizationConfig` knobs that matter most in v3:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `steps` | 2000 | Optimisation iterations per modality |
| `learning_rate` | 0.1 | Step size |
| `autocast_dtype` | `fp32` | `fp16` / `bf16` recommended on Apple Silicon |
| `compile_mode` | `none` | `reduce-overhead` is the recommended `torch.compile` mode |
| `seed` | None | Random seed for reproducibility |
| `regularization.total_variation` | 0.01 | Spatial smoothness |
| `regularization.spectral` | 0.001 | High-frequency penalty |
| `regularization.latent_norm` | 0.1 | Keep latent in distribution |

## Hardware

Targets M1 Max with 64GB unified memory. Real-weight runtimes (autocast bf16 + compile reduce-overhead, 2000 steps):

| Component | Memory |
|-----------|--------|
| LanguageBind (all 4 encoders) | ~6GB |
| SD3.5-medium VAE | ~2GB |
| Stable Audio Open | ~3GB |
| LTX-Video | ~5GB |
| Optimisation overhead | ~3GB |
| **Total active** | **~19GB** |
| **Available headroom (64GB)** | **~45GB** |

Single-modality renders (`--modalities image` only) fit comfortably on 16GB Apple Silicon. Full four-modality dual-track runs benefit from 64GB.

## Project structure

```
embedding-art/
├── src/embedding_art/
│   ├── core/             # Concept, Engine, Config, Loss, PatchAlignment
│   ├── encoders/         # LanguageBind (canonical) + SigLIP2/CLAP/DINOv3 (probes) + ImageBind (legacy)
│   ├── generators/       # SD3.5 image, Stable Audio Open, LTX-Video, + legacy SDXL/AudioLDM2/SVD
│   ├── sae/              # GroupSparse + Matryoshka SAEs + activating-examples VLM labelling
│   ├── interpretation/   # InterpretationBundle: text-anchor, attribution, linear probes
│   ├── evaluation/       # Cross-encoder probes, seed-stability, eval card assembly
│   ├── experiments/      # anchor-compare, text-anchor sweep
│   ├── perf/             # torch.compile wrapper, SDPA routing, CoreML/ANE, MLX
│   ├── regularizers/     # Total variation, spectral, latent norm, audio TV
│   ├── web/              # FastAPI surface for the showcase + anchor-compare endpoints
│   └── cli/              # `embed-art` commands
├── ui/                   # React + TS frontend (Showcase, Anchor-Compare, Job pages)
├── examples/             # Runnable recipes for the main workflows
└── outputs/              # Generated outputs
```

## Roadmap

- [x] Image generation via SDXL VAE (legacy v2)
- [x] LanguageBind canonical encoder (v3)
- [x] SD3.5-medium / Stable Audio Open / LTX-Video backbones (v3)
- [x] Dual-track rendering (`honest` + `natural`)
- [x] Multi-modality `embed-art showcase` command
- [x] Interpretation bundle (text-anchor, SAE features, attribution, linear probes)
- [x] Cross-encoder evaluation card + seed-stability
- [x] Apple Silicon perf knobs (autocast, `torch.compile`, SDPA, diffusers QKV fusion)
- [x] CoreML/ANE probe path + MLX SAE training backend (infra; speedup measurement gated on M1 Max runbooks)
- [x] Web UI (Alpha) with showcase + anchor-comparison viewers
- [ ] M2b real-weight VSD validation on M1 Max (math + LoRA injection done; runtime gated on runbook)

## License

MIT
