# Multimodal Embedding Art System

## Project Overview

A system for generating art by optimizing generative model outputs toward target coordinates in a shared multimodal embedding space. The goal is to extract and visualize the "platonic ideals" that neural networks have learned—the maximally X thing according to the model's learned representations.

This is feature visualization as an artistic medium, extended across modalities.

---

## Product Requirements Document

### Vision

Create large-format, print-ready artwork that represents what a neural network *thinks* a concept looks like, sounds like, or moves like. Not human-curated outputs, but raw model-optimized representations—the direction in embedding space cranked to its maximum.

### Goals

1. **Multimodal input**: Accept any combination of text, image, audio, video as concept sources
2. **Multimodal output**: Generate images, audio, or video from the same target embedding
3. **Concept algebra**: Combine, interpolate, and negate embeddings to create novel concepts
4. **High resolution**: Output suitable for large-format printing (minimum 2048×2048 for images)
5. **Aligned outputs**: Outputs across modalities should re-embed to approximately the same coordinate
6. **Reproducibility**: Deterministic runs with seed control, checkpoint saving
7. **Local execution**: Run entirely on M1 Max MacBook Pro (64GB unified memory)

### Non-Goals

1. Real-time generation (optimization runs are expected to take minutes)
2. User-facing web interface (CLI and notebooks are sufficient)
3. Training or fine-tuning models (all models used frozen)
4. Photorealism or prompt-following in the traditional sense

### Success Criteria

- Generated image embeds within cosine similarity >0.85 of target embedding
- Image output at 1024×1024 native, upscalable to 2048×2048+
- Audio output at 16kHz+, 10+ seconds duration
- Full optimization run completes in <30 minutes on M1 Max
- Cross-modal outputs from same target embed within cosine similarity >0.90 of each other

---

## System Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT STAGE                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   [Text]──────┐                                                      │
│               │                                                      │
│   [Image]─────┼───► ImageBind Encoder ───► Target Embedding (1024d) │
│               │            ▲                       │                 │
│   [Audio]─────┤            │                       │                 │
│               │     (frozen weights)               │                 │
│   [Video]─────┘                                    │                 │
│                                                    │                 │
│   ┌────────────────────────────────────────────────┘                 │
│   │  Concept Algebra:                                                │
│   │  - Weighted addition: 0.6*A + 0.4*B                             │
│   │  - Subtraction: A - 0.3*B                                       │
│   │  - Spherical interpolation: slerp(A, B, t)                      │
│   │  - Normalization to unit sphere                                  │
│   └────────────────────────────────────────────────┐                 │
│                                                    ▼                 │
└─────────────────────────────────────────────────────────────────────┘
                                                     │
                                              Target Embedding
                                                     │
                                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      OPTIMIZATION STAGE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐        │
│   │   Random    │      │  Generator  │      │  ImageBind  │        │
│   │   Latent z  │─────►│   (frozen)  │─────►│  (frozen)   │        │
│   │             │      │             │      │             │        │
│   └─────────────┘      └─────────────┘      └─────────────┘        │
│          ▲                   │                     │                 │
│          │                   │                     ▼                 │
│          │                   │              Current Embedding        │
│          │                   │                     │                 │
│          │                   ▼                     ▼                 │
│          │            [Raw Output]     ┌───────────────────┐        │
│          │            (for preview)    │   Loss Function   │        │
│          │                             │                   │        │
│          │                             │ L = -cos_sim(     │        │
│          │                             │   current,        │        │
│          │                             │   target          │        │
│          │                             │ ) + λ*reg_loss    │        │
│          │                             └───────────────────┘        │
│          │                                      │                    │
│          │                                      ▼                    │
│          │                               ∂L/∂z (gradients)          │
│          │                                      │                    │
│          └──────────────────────────────────────┘                    │
│                      (update latent only)                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                                     │
                                          After N iterations
                                                     │
                                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        OUTPUT STAGE                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Optimized Latent ───► Generator Decode ───► Raw Output            │
│                                                      │               │
│                                                      ▼               │
│                                              ┌──────────────┐        │
│                                              │  Upscaling   │        │
│                                              │  (optional)  │        │
│                                              └──────────────┘        │
│                                                      │               │
│                                                      ▼               │
│                                              Final Output            │
│                                              - Image: PNG/TIFF       │
│                                              - Audio: WAV/FLAC       │
│                                              - Video: MP4            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EmbeddingArtEngine                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  EncoderModule   │  │ GeneratorModule  │  │ OptimizerModule  │  │
│  ├──────────────────┤  ├──────────────────┤  ├──────────────────┤  │
│  │                  │  │                  │  │                  │  │
│  │ - ImageBind      │  │ - SDXL VAE       │  │ - Adam/AdamW     │  │
│  │ - encode_text()  │  │ - AudioLDM VAE   │  │ - learning rate  │  │
│  │ - encode_image() │  │ - SVD VAE        │  │ - scheduler      │  │
│  │ - encode_audio() │  │                  │  │ - gradient clip  │  │
│  │ - encode_video() │  │ - decode()       │  │                  │  │
│  │                  │  │ - latent_shape() │  │                  │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  ConceptModule   │  │ RegularizerModule│  │  OutputModule    │  │
│  ├──────────────────┤  ├──────────────────┤  ├──────────────────┤  │
│  │                  │  │                  │  │                  │  │
│  │ - add()          │  │ - total_var()    │  │ - save_image()   │  │
│  │ - subtract()     │  │ - spectral()     │  │ - save_audio()   │  │
│  │ - slerp()        │  │ - latent_norm()  │  │ - save_video()   │  │
│  │ - normalize()    │  │ - blur_augment() │  │ - save_latent()  │  │
│  │ - from_file()    │  │                  │  │ - upscale()      │  │
│  │                  │  │                  │  │                  │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Technical Specifications

### Models

#### Encoder: ImageBind (Meta)

- **Purpose**: Unified embedding space for all modalities
- **Embedding dimension**: 1024
- **Supported modalities**: Image, text, audio, video, depth, thermal, IMU
- **Image input size**: 224×224
- **Audio input**: 2 seconds at 16kHz (spectrogram)
- **Weights**: `imagebind_huge` (~1.2GB)
- **License**: MIT

#### Image Generator: SDXL VAE

- **Purpose**: High-quality image synthesis from latent
- **Latent shape**: `[1, 4, 128, 128]` → decodes to 1024×1024
- **Latent distribution**: Gaussian, roughly N(0, 1)
- **Weights**: Part of `stabilityai/sdxl-vae` (~335MB)
- **License**: OpenRAIL++

#### Audio Generator: AudioLDM 2

- **Purpose**: Audio synthesis from latent
- **Latent shape**: `[1, 8, 256, 16]` → decodes to ~10s at 16kHz
- **Weights**: `cvssp/audioldm2` (~1.5GB)
- **License**: CC-BY-NC-SA 4.0 (non-commercial)

Alternative: **Stable Audio Open**
- **Latent shape**: Variable length
- **Output**: Up to 47s at 44.1kHz
- **License**: Stability AI Community License

#### Video Generator: Stable Video Diffusion (SVD) VAE

- **Purpose**: Video synthesis from latent
- **Latent shape**: `[1, 4, frames, 64, 64]` → decodes to frames×512×512
- **Typical frames**: 14-25
- **Weights**: `stabilityai/stable-video-diffusion` (~4GB)
- **License**: Stability AI Community License

### Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU/Accelerator | M1 Pro (16GB) | M1 Max (32GB+) |
| Unified Memory | 32GB | 64GB |
| Storage | 20GB (models) | 50GB+ (outputs) |
| macOS | 13.0+ | 14.0+ |

### Memory Budget (M1 Max 64GB)

| Component | VRAM (approx) |
|-----------|---------------|
| ImageBind | ~3GB |
| SDXL VAE | ~1GB |
| AudioLDM 2 | ~2GB |
| Optimization overhead | ~2GB |
| **Total active** | **~8GB** |

Plenty of headroom. Can load multiple generators simultaneously if needed.

### Dependencies

```toml
[project]
name = "embedding-art"
version = "0.1.0"
requires-python = ">=3.10"

[project.dependencies]
torch = ">=2.1.0"
torchvision = ">=0.16.0"
torchaudio = ">=2.1.0"
diffusers = ">=0.24.0"
transformers = ">=4.36.0"
accelerate = ">=0.25.0"
safetensors = ">=0.4.0"
einops = ">=0.7.0"
numpy = ">=1.24.0"
pillow = ">=10.0.0"
soundfile = ">=0.12.0"
librosa = ">=0.10.0"
scipy = ">=1.11.0"
tqdm = ">=4.66.0"
pyyaml = ">=6.0"
click = ">=8.1.0"
rich = ">=13.0.0"

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "ipython>=8.0.0",
    "jupyter>=1.0.0",
    "matplotlib>=3.8.0",
]
```

---

## API Design

### Core Classes

```python
# === Concept: A target in embedding space ===

class Concept:
    """
    Represents a point or region in ImageBind's embedding space.
    Supports arithmetic operations for combining concepts.
    """
    
    embedding: torch.Tensor  # Shape: [1, 1024], normalized
    source_description: str  # Human-readable description
    
    @classmethod
    def from_text(cls, text: str) -> "Concept": ...
    
    @classmethod
    def from_image(cls, path: Path | Image) -> "Concept": ...
    
    @classmethod
    def from_audio(cls, path: Path, start: float = 0, duration: float = 2) -> "Concept": ...
    
    @classmethod
    def from_video(cls, path: Path, timestamp: float = 0) -> "Concept": ...
    
    @classmethod
    def from_embedding(cls, embedding: torch.Tensor, description: str = "") -> "Concept": ...
    
    def __add__(self, other: "Concept") -> "Concept": ...
    def __sub__(self, other: "Concept") -> "Concept": ...
    def __mul__(self, scalar: float) -> "Concept": ...
    def __rmul__(self, scalar: float) -> "Concept": ...
    
    def normalize(self) -> "Concept": ...
    
    @staticmethod
    def slerp(a: "Concept", b: "Concept", t: float) -> "Concept":
        """Spherical linear interpolation between concepts."""
        ...
    
    @staticmethod
    def combine(concepts: list["Concept"], weights: list[float] | None = None) -> "Concept":
        """Weighted combination of multiple concepts."""
        ...


# === Generator: Output modality backends ===

class Generator(Protocol):
    """Protocol for output generators."""
    
    @property
    def latent_shape(self) -> tuple[int, ...]: ...
    
    @property
    def output_modality(self) -> str: ...  # "image", "audio", "video"
    
    def init_latent(self, seed: int | None = None) -> torch.Tensor:
        """Initialize random latent for optimization."""
        ...
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to output tensor."""
        ...
    
    def to_output(self, decoded: torch.Tensor) -> Image | AudioSegment | VideoClip:
        """Convert decoded tensor to output format."""
        ...


class ImageGenerator(Generator):
    """SDXL VAE-based image generator."""
    
    latent_shape = (1, 4, 128, 128)  # → 1024×1024
    output_modality = "image"
    
    def __init__(self, model_id: str = "stabilityai/sdxl-vae"): ...


class AudioGenerator(Generator):
    """AudioLDM 2-based audio generator."""
    
    latent_shape = (1, 8, 256, 16)  # → ~10s audio
    output_modality = "audio"
    
    def __init__(self, model_id: str = "cvssp/audioldm2"): ...


class VideoGenerator(Generator):
    """SVD VAE-based video generator."""
    
    latent_shape = (1, 4, 14, 64, 64)  # → 14 frames @ 512×512
    output_modality = "video"
    
    def __init__(self, model_id: str = "stabilityai/stable-video-diffusion"): ...


# === Regularizers: Keep optimization stable ===

class Regularizer(Protocol):
    """Protocol for regularization losses."""
    
    def __call__(self, latent: torch.Tensor, decoded: torch.Tensor | None = None) -> torch.Tensor:
        """Compute regularization loss."""
        ...


class TotalVariation(Regularizer):
    """Encourages spatial smoothness."""
    weight: float = 0.01


class SpectralRegularizer(Regularizer):
    """Penalizes high-frequency noise."""
    weight: float = 0.001


class LatentNorm(Regularizer):
    """Keeps latent near typical distribution."""
    weight: float = 0.1


# === Engine: Main optimization loop ===

@dataclass
class OptimizationConfig:
    """Configuration for optimization run."""
    
    steps: int = 2000
    learning_rate: float = 0.1
    optimizer: str = "adam"  # "adam", "adamw", "sgd"
    scheduler: str = "cosine"  # "cosine", "constant", "linear"
    
    # Regularization
    regularizers: list[Regularizer] = field(default_factory=list)
    
    # Augmentation during optimization
    random_crop: bool = True
    random_flip: bool = False
    color_jitter: bool = False
    
    # Checkpointing
    checkpoint_every: int = 100
    preview_every: int = 50
    
    # Reproducibility
    seed: int | None = None


@dataclass
class OptimizationResult:
    """Result of an optimization run."""
    
    final_latent: torch.Tensor
    final_output: Image | AudioSegment | VideoClip
    final_embedding: torch.Tensor
    target_embedding: torch.Tensor
    final_similarity: float
    loss_history: list[float]
    similarity_history: list[float]
    checkpoints: list[tuple[int, torch.Tensor]]  # (step, latent)
    config: OptimizationConfig
    elapsed_seconds: float


class EmbeddingArtEngine:
    """
    Main engine for optimizing outputs toward concept embeddings.
    """
    
    def __init__(
        self,
        encoder: ImageBindEncoder,
        device: str = "mps",
    ):
        self.encoder = encoder
        self.device = device
        self._generators: dict[str, Generator] = {}
    
    def register_generator(self, name: str, generator: Generator) -> None:
        """Register a generator for a modality."""
        ...
    
    def optimize(
        self,
        target: Concept,
        output_modality: str,  # "image", "audio", "video"
        config: OptimizationConfig | None = None,
        callback: Callable[[int, float, torch.Tensor], None] | None = None,
    ) -> OptimizationResult:
        """
        Optimize a latent to maximize similarity with target concept.
        
        Args:
            target: The concept to optimize toward
            output_modality: Which generator to use
            config: Optimization hyperparameters
            callback: Called each step with (step, loss, latent)
        
        Returns:
            OptimizationResult with final output and metadata
        """
        ...
    
    def batch_optimize(
        self,
        targets: list[Concept],
        output_modality: str,
        config: OptimizationConfig | None = None,
    ) -> list[OptimizationResult]:
        """Optimize multiple targets (same config, sequential)."""
        ...
    
    def interpolation_series(
        self,
        concept_a: Concept,
        concept_b: Concept,
        output_modality: str,
        steps: int = 10,
        config: OptimizationConfig | None = None,
    ) -> list[OptimizationResult]:
        """Generate outputs along interpolation between two concepts."""
        ...
```

### CLI Interface

```bash
# Basic usage
embed-art optimize \
    --target-text "goldfish" \
    --output image \
    --output-path ./outputs/goldfish.png

# Combined concepts
embed-art optimize \
    --target-text "ocean" 0.5 \
    --target-text "fire" 0.5 \
    --output image \
    --output-path ./outputs/ocean_fire.png

# Cross-modal: audio to image
embed-art optimize \
    --target-audio ./thunder.wav \
    --target-text "purple" 0.3 \
    --output image \
    --output-path ./outputs/purple_thunder.png

# Interpolation series
embed-art interpolate \
    --concept-a "goldfish" \
    --concept-b "flamingo" \
    --steps 20 \
    --output image \
    --output-dir ./outputs/fish_to_bird/

# Inspect embedding
embed-art embed \
    --input-text "goldfish" \
    --save-embedding ./embeddings/goldfish.pt

# Compare embeddings
embed-art compare \
    --embedding-a ./embeddings/goldfish.pt \
    --embedding-b ./embeddings/orange.pt
    
# Upscale result
embed-art upscale \
    --input ./outputs/goldfish.png \
    --scale 2 \
    --output ./outputs/goldfish_2x.png
```

### Configuration File

```yaml
# config.yaml

# Default optimization settings
optimization:
  steps: 2000
  learning_rate: 0.1
  optimizer: adamw
  scheduler: cosine
  seed: null  # null for random

# Regularization
regularization:
  total_variation: 0.01
  spectral: 0.001
  latent_norm: 0.1

# Augmentation during optimization
augmentation:
  random_crop: true
  crop_scale: [0.8, 1.0]
  random_flip: false
  color_jitter: false

# Output settings
output:
  checkpoint_every: 100
  preview_every: 50
  save_latents: true
  save_loss_history: true

# Models (paths or HuggingFace IDs)
models:
  encoder: facebook/imagebind_huge
  image_generator: stabilityai/sdxl-vae
  audio_generator: cvssp/audioldm2
  video_generator: stabilityai/stable-video-diffusion

# Hardware
device: mps  # or cuda, cpu
dtype: float32  # float16 can be unstable for optimization
```

---

## Project Structure

```
embedding-art/
├── pyproject.toml
├── README.md
├── config.yaml                 # Default configuration
│
├── src/
│   └── embedding_art/
│       ├── __init__.py
│       ├── __main__.py         # CLI entry point
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── concept.py      # Concept class and algebra
│       │   ├── engine.py       # Main optimization loop
│       │   └── config.py       # Configuration dataclasses
│       │
│       ├── encoders/
│       │   ├── __init__.py
│       │   ├── base.py         # Encoder protocol
│       │   └── imagebind.py    # ImageBind wrapper
│       │
│       ├── generators/
│       │   ├── __init__.py
│       │   ├── base.py         # Generator protocol
│       │   ├── image.py        # SDXL VAE
│       │   ├── audio.py        # AudioLDM 2
│       │   └── video.py        # SVD VAE
│       │
│       ├── regularizers/
│       │   ├── __init__.py
│       │   ├── total_variation.py
│       │   ├── spectral.py
│       │   └── latent_norm.py
│       │
│       ├── augmentation/
│       │   ├── __init__.py
│       │   └── transforms.py   # Differentiable augmentations
│       │
│       ├── output/
│       │   ├── __init__.py
│       │   ├── save.py         # Save to various formats
│       │   └── upscale.py      # Resolution upscaling
│       │
│       └── cli/
│           ├── __init__.py
│           ├── main.py         # Click CLI app
│           └── commands/
│               ├── optimize.py
│               ├── interpolate.py
│               ├── embed.py
│               └── compare.py
│
├── scripts/
│   ├── download_models.py      # Pre-download all model weights
│   └── benchmark.py            # Performance testing
│
├── notebooks/
│   ├── 01_quickstart.ipynb     # Basic usage examples
│   ├── 02_concept_algebra.ipynb
│   ├── 03_cross_modal.ipynb
│   └── 04_experiments.ipynb
│
├── tests/
│   ├── conftest.py
│   ├── test_concept.py
│   ├── test_engine.py
│   ├── test_generators.py
│   └── test_regularizers.py
│
└── outputs/                    # Generated outputs (gitignored)
    ├── images/
    ├── audio/
    ├── video/
    └── checkpoints/
```

---

## Implementation Plan

### Phase 1: Foundation (Week 1)

**Goal**: Basic image optimization working end-to-end

| Task | Details | Estimate |
|------|---------|----------|
| Project setup | pyproject.toml, directory structure, dev tools | 2h |
| ImageBind integration | Load model, encode text/image/audio | 4h |
| SDXL VAE integration | Load VAE, encode/decode, verify round-trip | 3h |
| Concept class | Basic embedding wrapper with arithmetic | 2h |
| Core optimization loop | Adam, cosine sim loss, gradient descent | 4h |
| Basic regularizers | Total variation, latent norm | 2h |
| CLI scaffold | Click app with `optimize` command | 2h |
| **Phase 1 Total** | | **~19h** |

**Deliverable**: Can run `embed-art optimize --target-text "goldfish" --output image` and get a result.

### Phase 2: Robustness (Week 2)

**Goal**: Stable, configurable optimization with good defaults

| Task | Details | Estimate |
|------|---------|----------|
| Augmentation pipeline | Differentiable crops, jitter during optimization | 3h |
| Spectral regularizer | Penalize high frequencies | 2h |
| Learning rate scheduling | Cosine, warmup, configurable | 2h |
| Checkpointing | Save/resume optimization, save previews | 3h |
| Configuration system | YAML config, CLI overrides | 2h |
| Seed handling | Reproducible runs | 1h |
| Logging & progress | Rich progress bars, loss curves | 2h |
| **Phase 2 Total** | | **~15h** |

**Deliverable**: Stable optimization with tunable hyperparameters, reproducible results.

### Phase 3: Multi-Modal (Week 3)

**Goal**: Audio and video output working

| Task | Details | Estimate |
|------|---------|----------|
| AudioLDM 2 integration | Load model, latent space exploration | 4h |
| Audio optimization loop | Adapt core loop for audio latents | 3h |
| Audio regularizers | Spectral smoothness for audio | 2h |
| SVD VAE integration | Load model, understand latent structure | 4h |
| Video optimization loop | Adapt core loop for video latents | 4h |
| Cross-modal testing | Verify alignment across modalities | 3h |
| **Phase 3 Total** | | **~20h** |

**Deliverable**: Can generate aligned image/audio/video from same target embedding.

### Phase 4: Polish & Scale (Week 4)

**Goal**: Print-ready outputs, tooling for exploration

| Task | Details | Estimate |
|------|---------|----------|
| Upscaling pipeline | Tiled diffusion or external upscaler | 4h |
| Interpolation tooling | Generate series between concepts | 3h |
| Batch processing | Multiple targets, grid outputs | 3h |
| Embedding inspector | Visualize, compare, save embeddings | 2h |
| Notebooks | Usage examples, experiments | 4h |
| Documentation | README, API docs | 3h |
| **Phase 4 Total** | | **~19h** |

**Deliverable**: Complete toolkit for embedding art generation.

---

## Experiments to Run

Once the system is working, these are interesting directions to explore:

### Concept Algebra

1. **Additive concepts**: `"fire" + "water"` — what does the model think reconciles these?
2. **Subtractive concepts**: `"dog" - "fur"` — can you remove attributes?
3. **Cross-modal addition**: `embed(jazz.wav) + embed("anger")` — emotional audio-visual fusion
4. **Negation chains**: `"art" - "human" - "paint"` — what's left?

### Optimization Dynamics

1. **Regularization sweep**: Same target, vary regularization from 0 to high — watch coherence emerge/collapse
2. **Learning rate extremes**: Very high LR gives you "raw activation" aesthetics
3. **Early stopping gallery**: Save outputs at step 10, 100, 500, 2000 — show optimization trajectory
4. **Multi-seed ensemble**: Same target, 10 seeds — how much variance?

### Cross-Modal Alignment

1. **Round-trip fidelity**: Image → embed → optimize audio → embed → compare to original
2. **Modality disagreement**: Find targets where image/audio converge to different embeddings
3. **Anchor modality**: Does optimizing from image-derived vs text-derived embeddings differ?

### Adversarial Aesthetics

1. **Minimal regularization**: Let the optimization find true maxima, embrace the noise
2. **Targeted neurons**: Instead of class logits, target specific internal features
3. **Negative optimization**: Find the *least* X thing possible

### Scale & Series

1. **Concept morphs**: 100-frame interpolation from "birth" to "death"
2. **Concept grids**: 10×10 grid varying two concept weights
3. **Audio-visual pairs**: Same target → image + audio side by side

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| ImageBind doesn't encode some concepts well | Medium | High | Test embedding quality early, have fallback to CLIP for text-image |
| Audio optimization unstable | Medium | Medium | AudioLDM's latent space may need different regularizers; extensive tuning |
| MPS backend issues | Low | Medium | Fall back to CPU for problematic ops; most PyTorch MPS issues resolved |
| Memory pressure with video | Medium | Low | Video is optional; can skip or reduce frame count |
| Optimization converges to adversarial garbage | High | Medium | This is expected without regularization; tune regularizers, accept some runs fail |
| SDXL VAE alone gives blurry results | Medium | Medium | VAE-only decoding is less sharp than full diffusion; may need to add refinement step |

---

## Open Questions

1. **VAE vs Full Diffusion**: SDXL VAE alone will decode latents, but the results won't be as crisp as running through the full diffusion process. Do we accept this, or add an optional "refinement" step that runs a few diffusion denoising steps?

2. **ImageBind audio length**: ImageBind only sees 2-second audio clips. For longer audio generation, do we optimize for a single 2s window and let the rest be coherent via the generator's priors? Or optimize multiple windows?

3. **Video frame consistency**: Similar question — optimize all frames toward the same target, or allow drift?

4. **Upscaling approach**: Tiled diffusion (run SDXL at overlapping tiles), external upscaler (Real-ESRGAN), or accept 1024×1024 as final?

5. **Alternative encoders**: Should we support CLIP as a fallback/comparison? It's image-text only but more widely used and understood.

---

## Success Metrics

### Technical

- [ ] Cosine similarity >0.85 achieved within 2000 steps for text concepts
- [ ] Cross-modal outputs (image/audio from same target) embed within 0.90 similarity
- [ ] Full optimization run <30 minutes on M1 Max
- [ ] Memory usage <20GB peak

### Artistic

- [ ] Generated images visually distinct from training data
- [ ] Concept algebra produces semantically meaningful results
- [ ] Outputs interesting enough to print and hang on wall

### Practical

- [ ] Can explain what the system does to a non-technical person
- [ ] Reproducible results with seed control
- [ ] CLI usable without reading source code
