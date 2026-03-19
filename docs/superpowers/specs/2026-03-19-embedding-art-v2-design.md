# Embedding Art v2: Multimodal Model State Renderer

**Date:** 2026-03-19
**Status:** Draft — awaiting user approval before implementation

## Vision

Take a model's internal state and render it directly as different modalities (text, image, video, audio). Go straight from an embedding to an output — whether or not that output is human-legible — to get a true view of the state of the model at a given time that represents a specific concept. Interpretability research meets generative art.

## Problems with Current Approach

The current system uses ImageBind (2023, unmaintained) + gradient descent on SDXL VAE latents to maximize cosine similarity with a target embedding. Issues:

1. **Slow** — hundreds of optimization steps per image
2. **Regularization-sensitive** — hand-tuned TV/spectral/norm weights dominate output aesthetics
3. **Indirect** — searching VAE latent space for something that re-encodes close to the embedding, not rendering the embedding itself
4. **Single encoder, single loss signal** — cosine similarity on the final 1024d embedding is a thin gradient signal
5. **No interpretability** — opaque 1024d vectors, no understanding of what dimensions mean
6. **Stale models** — ImageBind hasn't been updated since 2023; the field moves daily

## Three Core Innovations

### 1. Multi-Layer Feature Matching Loss (MIMIC-inspired)

Instead of cosine similarity on just the final embedding, match encoder feature statistics (mean, variance) at every intermediate layer. Based on MIMIC (arxiv 2508.07833).

**Why this matters:** A single cosine similarity on a 1024d vector provides thin gradients. Multi-layer matching gives the optimizer information from every stage of the encoder — low-level texture features, mid-level structure, high-level semantics. This is a dramatically richer gradient signal.

### 2. Group-Sparse SAE Decomposition (ICLR 2026)

Train sparse autoencoders (arxiv 2601.20028) on encoder embedding spaces to decompose opaque vectors into interpretable, cross-modal features. Each feature has a human-readable name ("golden", "aquatic", "scales").

**Why this matters:** Enables concept arithmetic in an interpretable basis. Instead of manipulating opaque dimensions, you can say "more golden, less cartoon" and have it mean something. Features are cross-modal — the same "golden" feature fires on text and images.

### 3. Dual Rendering Paths

- **VAE latent optimization** — fast, aesthetically pleasing, constrained to natural image manifold
- **INR/pixel optimization** — slow, maximally faithful to what the model actually represents, potentially alien/non-human outputs

**Why this matters:** The VAE constrains outputs to look "normal." For art that's fine. For interpretability, you want to see what the model *actually* represents, even if it looks weird. Both paths have value.

## Architecture

### Data Flow

```
ConceptSpec (encoder-agnostic)
    "a goldfish"
         │
    ┌────┴────────────────┐
    ▼                     ▼
Encoder A              Encoder B
(SigLIP 2)            (ImageBind)
    │                     │
    ▼                     ▼
Concept A              Concept B
(1152d)               (1024d)
    │                     │
    ├── SAE Lens ──┐      ├── SAE Lens ──┐
    ▼              ▼      ▼              ▼
Raw Embedding  Features  Raw Embedding  Features
    │                     │
    ▼                     ▼
RenderingStrategy      RenderingStrategy
    │                     │
    ▼                     ▼
Output A               Output B
```

### Section 1: Core Abstractions

#### Encoder-Agnostic Concept Specification

```python
@dataclass
class ConceptSpec:
    """What to render — not yet bound to any encoder."""
    text: str | None = None
    image: Path | None = None
    audio: Path | None = None
    video: Path | None = None
    weight: float = 1.0
```

Concepts are materialized per-encoder when rendering begins. This enables multi-encoder comparison without coupling concept creation to a specific embedding space.

#### Encoder Capabilities

```python
class EncoderCapability(Flag):
    TEXT = auto()
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()
    BACKPROP_OPTIMIZABLE = auto()      # Efficient backprop for optimization loops
    MULTI_LAYER_FEATURES = auto()      # Exposes intermediate layer activations

@dataclass
class EncoderCard:
    """Metadata descriptor — what a model can do, not how it does it."""
    name: str
    capabilities: EncoderCapability
    embedding_dim: int
    memory_estimate_mb: int
    backprop_cost: Literal["low", "medium", "high"]
```

#### Typed Layer Features

```python
@dataclass
class LayerFeatures:
    tensor: Tensor
    spatial: bool                    # Has spatial dims?
    shape_semantic: str              # "batch_tokens_dim" | "batch_channels_height_width"
    layer_name: str                  # Human-readable
```

#### Encoder Protocol

```python
class Encoder(Protocol):
    card: EncoderCard
    def encode(self, spec: ConceptSpec) -> Concept: ...
    def encode_for_optimization(self, tensor: Tensor) -> Embedding: ...
    def get_layer_features(self, tensor: Tensor) -> dict[int, LayerFeatures]: ...
    def unload(self) -> None: ...
```

#### SAE Lens (separate from encoder and generator)

```python
class SAELens:
    def decompose(self, embedding: Tensor) -> SAEDecomposition: ...
    def reconstruct(self, decomposition: SAEDecomposition) -> Tensor: ...
    def manipulate(self, decomposition: SAEDecomposition,
                   adjustments: dict[str, float]) -> SAEDecomposition: ...

@dataclass
class SAEDecomposition:
    activations: Tensor                    # [1, n_features] sparse
    active_features: dict[str, float]      # name -> activation strength
    reconstruction_error: float
```

#### Rendering Strategy Protocol

```python
class RenderingStrategy(Protocol):
    """Abstracts how we go from concept to output."""
    def render(self, target: Concept, generator: Generator,
               encoder: Encoder, config: Config) -> RenderResult: ...

class OptimizationStrategy(RenderingStrategy):
    """Current approach: gradient descent on latent/pixels."""
    ...

class DirectDecoderStrategy(RenderingStrategy):
    """Future: feedforward from embedding to output (IP-Adapter, UnCLIP)."""
    ...
```

#### Generator Protocols (split for two paradigms)

```python
class LatentGenerator(Protocol):
    """VAE/diffusion decode from latent space."""
    latent_shape: tuple
    def init_latent(self, seed: int) -> Tensor: ...
    def decode(self, latent: Tensor) -> Tensor: ...

class DirectGenerator(Protocol):
    """INR/pixel — the output itself is optimized."""
    def get_optimizable_parameters(self) -> list[Parameter]: ...
    def render(self) -> Tensor: ...
```

#### Feature Statistics (per-architecture)

```python
class FeatureStatisticsExtractor(Protocol):
    def extract(self, features: LayerFeatures) -> FeatureStatistics: ...

@dataclass
class FeatureStatistics:
    mean: Tensor
    std: Tensor

class ViTStatisticsExtractor(FeatureStatisticsExtractor):
    """For SigLIP, ImageBind — mean/std over patch tokens per feature dim."""
    ...

class CNNStatisticsExtractor(FeatureStatisticsExtractor):
    """For CNN-based encoders — per-channel stats over spatial dims."""
    ...
```

#### Simple Registry (no decorators)

```python
ENCODERS: dict[str, type[Encoder]] = {
    "siglip2-so400m": SigLIP2Encoder,
    "imagebind": ImageBindEncoder,
    "clap-general": CLAPEncoder,
    "lco-omni-3b": LCOOmniEncoder,
}

class EncoderRegistry:
    def list_available(self) -> list[EncoderCard]: ...
    def load(self, name: str) -> Encoder: ...          # Lazy load
    def get_for_modality(self, modality: str) -> list[str]: ...
    def can_fit(self, names: list[str], memory_budget_mb: int) -> bool: ...
```

### Section 2: Optimization Engine & Loss Architecture

#### The Engine

```python
class EmbeddingArtEngine:
    """Orchestrates rendering of concepts as multimodal outputs."""

    def __init__(self, registry: EncoderRegistry, device: str = "auto"):
        self.registry = registry
        self.generators: dict[str, LatentGenerator | DirectGenerator] = {}
        self.sae_lenses: dict[str, SAELens] = {}

    def render(
        self,
        spec: ConceptSpec,
        encoder: str,
        output_modality: str,
        strategy: RenderingStrategy | None = None,
        config: OptimizationConfig | None = None,
    ) -> RenderResult: ...

    def render_compare(
        self,
        spec: ConceptSpec,
        encoders: list[str],
        output_modality: str,
        config: OptimizationConfig | None = None,
    ) -> dict[str, RenderResult]: ...

    def render_interpolation(
        self,
        spec_a: ConceptSpec,
        spec_b: ConceptSpec,
        steps: int,
        encoder: str,
        output_modality: str,
    ) -> list[RenderResult]: ...
```

#### Composite Loss

```python
@dataclass
class LossConfig:
    similarity_weight: float = 1.0
    feature_matching_weight: float = 0.5
    feature_matching_layers: list[int] | str = "every_4th"  # or "all", or [0, 6, 12, 18, 26]
    sae_feature_weight: float = 0.0          # off by default until SAE trained
    sae_target_features: dict[str, float] | None = None
    regularization: CompositeRegularizer | None = None


class CompositeLoss:
    def __init__(self, config: LossConfig, encoder: Encoder,
                 sae: SAELens | None = None):
        self.config = config
        self.sae = sae
        self.stats_extractor = self._select_extractor(encoder)
        self.reference_stats: dict[int, FeatureStatistics] | None = None

    def calibrate(self, target_concept: Concept, encoder: Encoder) -> None:
        """Compute reference feature statistics from target input.
        Must be called before optimization begins."""
        if encoder.card.capabilities & EncoderCapability.MULTI_LAYER_FEATURES:
            layer_features = encoder.get_layer_features(target_concept.source_input)
            self.reference_stats = {
                idx: self.stats_extractor.extract(feats)
                for idx, feats in layer_features.items()
            }

    def __call__(self, current_output: Tensor, target: Concept,
                 encoder: Encoder, latent: Tensor | None = None) -> LossBreakdown:
        components = {}

        # 1. Cosine similarity on final embedding
        current_emb = encoder.encode_for_optimization(current_output)
        components["similarity"] = -F.cosine_similarity(
            current_emb, target.embedding, dim=-1
        ).mean() * self.config.similarity_weight

        # 2. Multi-layer feature matching
        if (self.config.feature_matching_weight > 0
            and self.reference_stats is not None):
            layer_features = encoder.get_layer_features(current_output)
            feat_loss = torch.tensor(0.0)
            for idx, feats in layer_features.items():
                current_stats = self.stats_extractor.extract(feats)
                ref_stats = self.reference_stats[idx]
                feat_loss += (
                    (current_stats.mean - ref_stats.mean).pow(2).sum() +
                    (current_stats.std - ref_stats.std).pow(2).sum()
                )
            components["feature_matching"] = feat_loss * self.config.feature_matching_weight

        # 3. SAE feature-space loss
        if self.config.sae_feature_weight > 0 and self.sae is not None:
            current_decomp = self.sae.decompose(current_emb)
            target_decomp = self.sae.decompose(target.embedding)
            components["sae_features"] = F.mse_loss(
                current_decomp.activations,
                target_decomp.activations,
            ) * self.config.sae_feature_weight

        # 4. Regularization
        if self.config.regularization and latent is not None:
            components["regularization"] = self.config.regularization(
                latent, current_output
            )

        total = sum(components.values())
        return LossBreakdown(total=total, components=components)


@dataclass
class LossBreakdown:
    """Every loss component tracked separately for logging/debugging."""
    total: Tensor
    components: dict[str, Tensor]
```

#### Optimization Loop

```python
class OptimizationStrategy(RenderingStrategy):
    def render(self, target: Concept, generator, encoder, config) -> RenderResult:
        loss_fn = CompositeLoss(config.loss, encoder, sae=self.sae)
        loss_fn.calibrate(target, encoder)

        if isinstance(generator, LatentGenerator):
            latent = generator.init_latent(config.seed)
            latent.requires_grad_(True)
            optimizable = [latent]
        elif isinstance(generator, DirectGenerator):
            optimizable = generator.get_optimizable_parameters()

        optimizer = self._build_optimizer(optimizable, config)
        scheduler = self._build_scheduler(optimizer, config)
        history = OptimizationHistory()

        for step in range(config.steps):
            optimizer.zero_grad()

            if isinstance(generator, LatentGenerator):
                output = generator.decode(latent)
            else:
                output = generator.render()

            if config.augmentation:
                output = self._augment(output, config.augmentation)

            breakdown = loss_fn(output, target, encoder,
                              latent if isinstance(generator, LatentGenerator) else None)
            breakdown.total.backward()

            torch.nn.utils.clip_grad_norm_(optimizable, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            history.record(step, breakdown)

            if config.callbacks:
                config.callbacks.on_step(step, breakdown, output)

        return RenderResult(
            output=output.detach(),
            history=history,
            encoder_name=encoder.card.name,
            final_similarity=history.final_similarity,
        )
```

### Section 3: SAE Integration

#### Lifecycle

```
Training (offline)  →  Artifact (.safetensors + vocab.json)  →  Inference (in pipeline)
```

Training is a separate offline process. The SAE is a pre-computed artifact per encoder, not trained during optimization.

#### SAELens Implementation

```python
class SAELens:
    def __init__(self, encoder_name: str, artifact_path: Path):
        self.W_enc: Tensor      # [embed_dim, n_features]
        self.W_dec: Tensor      # [n_features, embed_dim]
        self.bias: Tensor
        self.pre_bias: Tensor   # per-modality biases
        self.vocab: list[str]   # feature names from labeling
        self.k: int             # sparsity (top-k)

    def decompose(self, embedding: Tensor) -> SAEDecomposition:
        z = F.relu(self.W_enc @ (embedding - self.pre_bias) + self.bias)
        z = top_k(z, self.k)
        active = z.nonzero(as_tuple=True)[-1]
        return SAEDecomposition(
            activations=z,
            active_features={self.vocab[i]: z[0, i].item() for i in active},
            reconstruction_error=self._reconstruction_error(embedding, z),
        )

    def reconstruct(self, decomposition: SAEDecomposition) -> Tensor:
        return self.W_dec @ decomposition.activations.T + self.pre_bias

    def manipulate(self, decomposition: SAEDecomposition,
                   adjustments: dict[str, float]) -> SAEDecomposition:
        new_activations = decomposition.activations.clone()
        for name, value in adjustments.items():
            idx = self.vocab.index(name)
            new_activations[0, idx] = value
        return SAEDecomposition(activations=new_activations, ...)
```

#### Concept Integration

```python
class Concept:
    def decompose(self, sae: SAELens) -> SAEDecomposition:
        return sae.decompose(self.embedding)

    @classmethod
    def from_features(cls, sae: SAELens, features: dict[str, float],
                      encoder: Encoder) -> "Concept":
        decomp = SAEDecomposition.from_dict(features, sae)
        embedding = sae.reconstruct(decomp)
        return cls(embedding=F.normalize(embedding), source=f"features:{features}")
```

#### SAE CLI Commands

```bash
embed-art sae collect --encoder siglip2-so400m --dataset cc3m --output embeds/
embed-art sae train --embeddings embeds/ --features 16384 --sparsity 32 \
    --group-sparse --lambda 0.05 --output sae_models/siglip2.safetensors
embed-art sae label --model sae_models/siglip2.safetensors --encoder siglip2-so400m
embed-art sae inspect --model sae_models/siglip2.safetensors --top 20
```

### Section 4: CLI Evolution

```bash
# Backward compatible (existing commands work with default encoder)
embed-art optimize -t "goldfish" 1.0 -o image

# New: explicit encoder selection
embed-art render -t "goldfish" 1.0 -o image --encoder siglip2-so400m

# New: multi-encoder comparison
embed-art compare -t "goldfish" 1.0 -o image \
    --encoders siglip2-so400m,imagebind

# New: SAE-based concept construction
embed-art render --features "golden:0.8,aquatic:0.6,scales:0.4" \
    -o image --encoder siglip2-so400m --sae sae_models/siglip2.safetensors

# New: decompose a concept to see its features
embed-art decompose -t "goldfish" 1.0 --encoder siglip2-so400m \
    --sae sae_models/siglip2.safetensors

# Existing commands work unchanged
embed-art interpolate -a "goldfish" -b "flamingo" -s 10 -o image
```

## Recommended Default Encoders

| Model | Modalities | Dims | Memory | Backprop Cost | Role |
|-------|-----------|------|--------|---------------|------|
| **SigLIP 2 So400m** | text, image | 1152 | ~1.6GB | Low | Primary vision-language |
| **CLAP** | text, audio | 512 | ~1.2GB | Low | Primary audio-language |
| **ImageBind** | 6 modalities | 1024 | ~3GB | Low | Fallback, cross-modal |
| **LCO-Embed-Omni-3B** | text, img, audio, video | varies | ~10GB | High | Optional high-quality |

All fit on M1 Max 64GB with plenty of headroom. Multiple encoders can be loaded simultaneously for comparison (~11-15GB total for 3 encoders + generator).

## Research Foundations

| Innovation | Source | Key Insight |
|-----------|--------|-------------|
| Multi-layer feature matching | MIMIC (arxiv 2508.07833) | Match encoder feature statistics at every layer, not just final embedding |
| Group-Sparse SAE | ICLR 2026 (arxiv 2601.20028) | Cross-modal random masking + group-sparse regularization produces truly multimodal features |
| Dual rendering paths | Implicit Inversion (2025) + current VAE approach | INR provides natural frequency decomposition, eliminating hand-tuned regularizers |
| Platonic Representation Hypothesis | ICML 2024 | All models converge to similar representations — validates cross-model comparison |

## Migration Strategy (v1 → v2)

### Approach: Incremental with Compatibility Layer

v2 is NOT a big-bang rewrite. The old API is preserved behind adapters during migration:

**Phase 1 — New abstractions alongside old (non-breaking):**
- Add `ConceptSpec`, `EncoderCard`, `EncoderCapability`, `EncoderRegistry`, `LossConfig`, `LossBreakdown` as new modules
- Add `RenderingStrategy`, `OptimizationStrategy` as new modules
- Existing `Encoder` protocol gains `encode(spec)` as a convenience method that dispatches to existing `encode_text`/`encode_image`/`encode_audio` methods internally. Old methods remain.
- Existing `EmbeddingArtEngine` gains a `from_registry()` classmethod alongside the current `__init__(encoder)`. Both work.
- `Concept.from_text()` remains as-is. `ConceptSpec` is the new path for multi-encoder workflows.
- All existing tests continue to pass unchanged.

**Phase 2 — New engine and loss (additive):**
- `CompositeLoss` added alongside existing loss computation in engine
- New `render()`, `render_compare()`, `render_interpolation()` methods added to engine
- Old `optimize()` method preserved, delegates to `render()` internally
- New CLI commands (`render`, `compare`, `decompose`) added alongside existing `optimize`

**Phase 3 — New encoders (additive):**
- SigLIP 2, CLAP, LCO-Omni encoder implementations added
- ImageBind encoder adapted to new protocol (keeps old methods, adds new ones)
- Default encoder configurable (starts as ImageBind for backward compat, eventually SigLIP 2)

**Phase 4 — SAE integration (additive):**
- SAE training pipeline, SAELens, feature decomposition
- New CLI commands (`sae collect`, `sae train`, `sae label`, `sae inspect`)

**Phase 5 — Deprecation (eventual):**
- Old `optimize()` method marked deprecated, points to `render()`
- Old `encode_text`/`encode_image` methods marked deprecated, points to `encode(spec)`
- Old `EmbeddingArtEngine(encoder)` constructor marked deprecated

### File Change Impact

| File | Change Type | Phase |
|------|------------|-------|
| `core/concept.py` | Add `source_input` field, `ConceptSpec` class | 1 |
| `core/config.py` | Add `LossConfig` field to `OptimizationConfig` | 1 |
| `core/engine.py` | Add `from_registry()`, `render()`, `render_compare()` | 2 |
| `encoders/base.py` | Extend protocol with `encode(spec)`, `get_layer_features()`, `unload()` | 1 |
| `encoders/imagebind.py` | Adapt to extended protocol | 3 |
| `encoders/siglip2.py` | New file | 3 |
| `encoders/clap.py` | New file | 3 |
| `encoders/registry.py` | New file | 1 |
| `loss/composite.py` | New file (CompositeLoss, LossBreakdown) | 2 |
| `loss/feature_matching.py` | New file (FeatureStatisticsExtractor) | 2 |
| `sae/lens.py` | New file | 4 |
| `sae/training.py` | New file | 4 |
| `generators/inr.py` | New file (DirectGenerator for INR) | 2 |
| `cli/commands/render.py` | New file | 2 |
| `cli/commands/compare.py` | New file | 2 |
| `cli/commands/sae.py` | New file | 4 |

## Handling the `source_input` Problem

Multi-layer feature matching requires the original input tensor (image, audio) to compute per-layer reference statistics. The current `Concept` class discards this after encoding.

**Solution:** Add an optional `source_input` field to `Concept`:

```python
@dataclass
class Concept:
    embedding: Tensor              # Normalized embedding vector
    description: str               # Human-readable description
    source_input: Tensor | None = None  # Original input tensor, retained for multi-layer loss
```

- `source_input` is populated when creating concepts via `encode(spec)` or `from_text()`
- It is NOT serialized (not saved with `concept.save()`) — it's ephemeral, used only during the optimization session
- For text-only concepts, `source_input` is None (text encoders don't produce an image tensor to match against). Multi-layer feature matching gracefully degrades: calibration uses the target embedding only, not per-layer statistics.
- Memory note: a 384x384 float32 image tensor = ~1.7MB. Negligible vs model weights.

**`calibrate()` updated:**

```python
def calibrate(self, target: Concept, encoder: Encoder) -> None:
    if (target.source_input is not None
        and encoder.card.capabilities & EncoderCapability.MULTI_LAYER_FEATURES):
        layer_features = encoder.get_layer_features(target.source_input)
        self.reference_stats = {
            idx: self.stats_extractor.extract(feats)
            for idx, feats in layer_features.items()
        }
    # If source_input is None (text concept), feature matching is disabled
    # and only final-embedding cosine similarity is used
```

## Diffusion Guidance Strategy

The existing `SDXLDiffusionGenerator` implements a third rendering paradigm: diffusion denoising with per-step gradient guidance. This doesn't fit cleanly into `LatentGenerator` or `DirectGenerator`.

**Solution:** Add `DiffusionGuidanceStrategy` as a rendering strategy:

```python
class DiffusionGuidanceStrategy(RenderingStrategy):
    """Diffusion denoising with per-step embedding guidance.

    Unlike OptimizationStrategy (which runs a loop around a frozen generator),
    this injects gradients INTO the diffusion denoising loop at each timestep.
    The generator and the optimization are interleaved, not sequential.
    """

    def render(self, target: Concept, generator: DiffusionGenerator,
               encoder: Encoder, config: Config) -> RenderResult:
        # Generator runs its own denoising loop
        # At each step, it calls back to compute embedding loss
        # and applies gradients to the latent
        return generator.generate_guided(
            target_embedding=target.embedding,
            encoder=encoder,
            loss_fn=CompositeLoss(config.loss, encoder),
            config=config,
        )
```

The `DiffusionGenerator` protocol extends `LatentGenerator`:

```python
class DiffusionGenerator(LatentGenerator, Protocol):
    """Diffusion model with embedding-guided denoising."""
    def generate_guided(
        self,
        target_embedding: Tensor,
        encoder: Encoder,
        loss_fn: CompositeLoss,
        config: Config,
    ) -> RenderResult: ...
```

This preserves the existing `SDXLDiffusionGenerator` functionality while fitting it into the strategy pattern. The engine selects `DiffusionGuidanceStrategy` when the generator is a `DiffusionGenerator`.

## Error Handling

New exception types extending the existing hierarchy in `exceptions.py`:

```python
class EncoderNotFoundError(EmbeddingArtError):
    """Raised when an encoder name isn't in the registry."""

class EncoderCapabilityError(EmbeddingArtError):
    """Raised when an operation requires a capability the encoder lacks."""

class SAENotTrainedError(EmbeddingArtError):
    """Raised when SAE features are requested but no SAE artifact exists."""

class MemoryBudgetExceededError(EmbeddingArtError):
    """Raised when loading encoders would exceed memory budget."""

class FeatureNotFoundError(EmbeddingArtError):
    """Raised when an SAE feature name isn't in the vocabulary."""
```

**Behavior:**
- `EncoderRegistry.load("unknown")` → raises `EncoderNotFoundError`
- `CompositeLoss` with `feature_matching_weight > 0` on encoder without `MULTI_LAYER_FEATURES` → logs warning, silently degrades to final-embedding-only loss
- `SAELens.manipulate({"unknown_feature": 1.0})` → raises `FeatureNotFoundError`
- `EncoderRegistry.load()` when memory budget would be exceeded → raises `MemoryBudgetExceededError`
- `render()` with `sae_feature_weight > 0` but no SAE loaded → raises `SAENotTrainedError`

## Config Integration

`LossConfig` becomes a field on the existing `OptimizationConfig`:

```python
@dataclass
class OptimizationConfig:
    # Existing fields preserved
    steps: int = 500
    learning_rate: float = 0.01
    optimizer: str = "adam"
    scheduler: str = "cosine"
    augmentation: AugmentationConfig | None = None
    checkpoint_every: int = 50
    seed: int | None = None

    # New field
    loss: LossConfig = field(default_factory=LossConfig)
```

This is backward compatible — existing code that constructs `OptimizationConfig()` gets the default `LossConfig` (similarity-only, matching the current behavior).

## Memory Considerations for Multi-Layer Features

Multi-layer feature extraction stores intermediate ViT activations during backprop. Memory impact:

| Encoder | Layers | Per-Layer Size | Total Feature Memory |
|---------|--------|---------------|---------------------|
| SigLIP 2 So400m | 27 | ~70MB | ~1.9GB |
| ImageBind (ViT-H) | 32 | ~85MB | ~2.7GB |
| CLAP (HTSAT) | 4 | ~15MB | ~0.06GB |

**Mitigation:** The `feature_matching_layers` config supports `"all"`, `"every_4th"`, or explicit layer indices like `[0, 6, 12, 18, 26]`. Using every 4th layer reduces memory by ~75% with minimal quality loss. Default: `"every_4th"` for ViT encoders, `"all"` for small encoders.

## Known Limitations

1. **Reproducibility across encoder versions:** Same seed + same text + different encoder version = different output. Embeddings are encoder-specific. Checkpoints include encoder name and version.
2. **SAE reconstruction is lossy:** ~54-92% of dense embedding quality depending on dataset. SAE features are for interpretability and concept manipulation, not lossless round-trips.
3. **Text-only concepts can't use multi-layer feature matching:** No source image to extract features from. Feature matching only applies when the concept source includes an image/audio input, or when combined with a reference image dataset.
4. **MPS compatibility varies by encoder:** Each encoder needs MPS testing. Some operations may fall back to CPU. The `EncoderCard.backprop_cost` field should reflect MPS-specific performance, not GPU performance.
5. **Implicit Inversion reference:** arxiv 2505.23161 (May 2025). Uses frequency-aware INR optimized with only CLIP losses.

## Review Feedback (Incorporated)

### Round 1 — Architectural Review:
- **ConceptSpec pattern**: Encoder-agnostic specification that each encoder materializes independently
- **SAE as separate lens**: Not part of encoder or generator, sits between them
- **RenderingStrategy protocol**: Distinguishes optimization-based vs direct/feedforward rendering
- **Split generator protocols**: LatentGenerator (VAE) vs DirectGenerator (INR/pixel)
- **Per-architecture statistics extractors**: ViT patch tokens need different treatment than CNN feature maps
- **Explicit calibrate() step**: Reference statistics computed once before optimization
- **Resource management**: unload() on encoders, sequential mode for memory-constrained comparison
- **Simple dict registry**: No decorator magic for 3-4 encoders

### Round 2 — Spec Review:
- **Migration plan added**: Incremental 5-phase migration, no big-bang rewrite
- **`source_input` field added to Concept**: Retains original input tensor for multi-layer feature matching
- **Encoder protocol backward compat**: Old `encode_text`/`encode_image` methods preserved, `encode(spec)` dispatches to them
- **Engine constructor backward compat**: `from_registry()` classmethod added alongside existing `__init__(encoder)`
- **DiffusionGuidanceStrategy added**: Third rendering strategy for existing SDXLDiffusionGenerator
- **Error handling defined**: New exception types for registry, capability, SAE, and memory failures
- **LossConfig integrated into OptimizationConfig**: Backward compatible via default factory
- **Multi-layer memory quantified**: ~1.9-2.7GB for full ViT features, mitigated by `every_4th` default
- **SAEDecomposition field name fixed**: `activations` (not `feature_activations`)
- **Known limitations documented**: Reproducibility, SAE lossy reconstruction, text-only concepts, MPS compat

### Round 3 — Re-review:

**Fixes verified.** Remaining items addressed:

- **`ConceptSpec` now includes `video` field**
- **`feature_matching_layers` default changed to `"every_4th"`** to match mitigation section
- **`Concept.source_input` handling in arithmetic**: `source_input` is `None` for arithmetic results (`__add__`, `__sub__`, `slerp`, `combine`). Only populated on initial encoding. `to()` moves it to device. `save()`/`load()` do NOT serialize it (ephemeral).
- **`EncoderCard` compatibility shim**: In Phase 1, existing `ImageBindEncoder` gets a `card` property that synthesizes an `EncoderCard` from its existing `embedding_dim` and `device` properties. `CompositeLoss` checks `hasattr(encoder, 'card')` and falls back to final-embedding-only loss if `card` is missing. This allows Phase 2 to work before Phase 3 completes.
- **`DiffusionGuidanceStrategy` is a refactor of `SDXLDiffusionGenerator`**: Phase 2 includes refactoring the existing `generate()` method into `generate_guided()`. The old `generate()` is preserved as a thin wrapper. The key mapping: `imagebind_encoder` → `encoder`, `target_embedding` → `target.embedding`, `regularizers` → `loss_fn.config.regularization`, `callback` → `config.callbacks`.
- **`encode_for_optimization`**: Added to Encoder protocol in Phase 1 (currently exists on ImageBindEncoder but not in the base protocol)

## Missing Type Definitions

```python
@dataclass
class RenderResult:
    """Output of any rendering strategy."""
    output: Tensor                      # Decoded image/audio/video tensor
    history: OptimizationHistory        # Loss/similarity tracking
    encoder_name: str                   # Which encoder was used
    final_similarity: float             # Final cosine similarity with target
    config: OptimizationConfig          # Config used for this render
    checkpoints: list[Path] | None = None


class OptimizationHistory:
    """Tracks all loss components across optimization steps."""
    def __init__(self):
        self.steps: list[int] = []
        self.loss_breakdowns: list[dict[str, float]] = []  # component name → value
        self.similarity_values: list[float] = []

    def record(self, step: int, breakdown: LossBreakdown) -> None:
        self.steps.append(step)
        self.loss_breakdowns.append(
            {k: v.item() for k, v in breakdown.components.items()}
        )
        if "similarity" in breakdown.components:
            self.similarity_values.append(-breakdown.components["similarity"].item())

    @property
    def final_similarity(self) -> float:
        return self.similarity_values[-1] if self.similarity_values else 0.0
```

`RenderResult` replaces `OptimizationResult` from v1. The old class is preserved as an alias during migration:

```python
OptimizationResult = RenderResult  # Deprecated, use RenderResult
```
