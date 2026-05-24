# Embedding Art v3: Direct State Rendering

**Date:** 2026-05-24
**Status:** Implementation in progress

## The Problem with v1/v2

Both v1 and v2 share the same fundamental limitation: they **search** for outputs that match an embedding rather than **decoding** the embedding directly.

```
v1/v2 pipeline (indirect):
  Concept → Target Embedding → [optimization loop × 500-2000 steps] → Output
                                        ↑
                              This is the problem.
                              We're searching, not rendering.
```

The optimization loop introduces three distortions:
1. **Regularization bias** — TV, spectral, latent norm all push outputs toward "natural-looking" images, suppressing what the model actually represents
2. **Generator manifold constraint** — SDXL VAE can only produce images within its learned manifold, which is a tiny slice of what embedding space encodes
3. **Loss surface artifacts** — Cosine similarity has gradient pathologies near zero and at large magnitudes; optimization gets stuck in local minima

The result: we see what SDXL can produce that *happens to score well* on cosine similarity, not what the embedding *actually represents*.

## The New Vision

**Render model internal states directly as different modalities.** Go straight from an embedding to an output — whether or not that output is human-legible — to get a true view of what the model represents.

This means:
- **Direct decoding** as the primary path, optimization as a secondary/comparison tool
- **Multiple decoder fidelity levels** — from "faithful but alien" to "pretty but approximate"
- **Multi-layer rendering** — not just final embeddings, but intermediate states at every layer
- **Feature-level rendering** — individual SAE features rendered as their own outputs
- **Multi-modal output** from the same internal state — text, image, audio, video

## Architecture

### Core Abstraction: DirectRenderer

```python
class DirectRenderer(Protocol):
    """Renders an embedding directly to an output modality.

    Unlike RenderingStrategy (which optimizes toward an embedding),
    a DirectRenderer produces output in a single forward pass or
    a fixed number of diffusion steps — no gradient loop.
    """

    @property
    def output_modality(self) -> str: ...

    def render(self, embedding: Tensor, **kwargs) -> RenderResult: ...
```

### Rendering Hierarchy (fidelity spectrum)

```
Most faithful                                          Most aesthetic
(alien/abstract)                                    (human-legible)
     ↓                                                      ↓
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌───────────┐
│ Raw      │   │Projection│   │IP-Adapter │   │Optimization│
│ Decode   │   │ Decoder  │   │ Guided    │   │   Loop     │
│          │   │          │   │ Diffusion │   │  (v1/v2)   │
├──────────┤   ├──────────┤   ├──────────┤   ├───────────┤
│Linear map│   │Learned   │   │Pretrained │   │Gradient    │
│from embed│   │MLP from  │   │adapter +  │   │descent on  │
│to pixel  │   │embed to  │   │SDXL for   │   │latent with │
│space     │   │generator │   │conditioned│   │re-encoding │
│          │   │condition │   │generation │   │per step    │
│          │   │space     │   │           │   │            │
│No prior. │   │Weak prior│   │Strong     │   │Strongest   │
│Raw model │   │from      │   │diffusion  │   │prior +     │
│state.    │   │training  │   │prior.     │   │regularizers│
│          │   │data.     │   │           │   │            │
│~0s       │   │~0.1s     │   │~5-15s     │   │~5-30min    │
└──────────┘   └──────────┘   └──────────┘   └───────────┘
```

Each position on this spectrum answers a different question:
- **Raw Decode**: "What does this embedding literally look like as pixels?"
- **Projection Decoder**: "What image is closest to this embedding in a learned mapping?"
- **IP-Adapter Guided**: "What high-quality image best represents this embedding?"
- **Optimization Loop**: "What output maximizes similarity with this embedding?"

### Data Flow: Multi-Layer State Rendering

```
Input ("goldfish")
     │
     ▼
┌─────────────────────────────────────┐
│            Encoder (SigLIP2)        │
│                                     │
│  Layer 0  ─→  [768d activations]  ──┼──→ Probe → Render → "Layer 0 view"
│  Layer 6  ─→  [768d activations]  ──┼──→ Probe → Render → "Layer 6 view"
│  Layer 12 ─→  [768d activations]  ──┼──→ Probe → Render → "Layer 12 view"
│  Layer 18 ─→  [768d activations]  ──┼──→ Probe → Render → "Layer 18 view"
│  Layer 26 ─→  [1152d embedding]   ──┼──→ Probe → Render → "Final embedding view"
│                                     │
└─────────────────────────────────────┘
                                            │
                                            ▼
                                     Progression video:
                                     animate Layer 0 → 26
                                     showing concept formation
```

### Data Flow: SAE Feature Rendering

```
Embedding (1152d)
     │
     ▼
┌─────────────────────────────────┐
│         SAE Decomposition       │
│                                 │
│  Feature "golden"    = 2.3  ────┼──→ Render feature direction → Image
│  Feature "aquatic"   = 1.8  ────┼──→ Render feature direction → Image
│  Feature "scales"    = 1.5  ────┼──→ Render feature direction → Image
│  Feature "swimming"  = 0.9  ────┼──→ Render feature direction → Image
│  ...                            │
│  Reconstruction ────────────────┼──→ Render reconstruction → Image
│                                 │
└─────────────────────────────────┘
                                        │
                                        ▼
                                  Feature grid:
                                  composite showing what each
                                  active feature "looks like"
```

## Implementation Plan

### New Modules

```
src/embedding_art/
├── renderers/                    # NEW: Direct rendering pipeline
│   ├── __init__.py
│   ├── base.py                   # DirectRenderer protocol
│   ├── raw.py                    # Linear embedding→pixel decoder
│   ├── projection.py             # Learned MLP decoder
│   ├── ip_adapter.py             # IP-Adapter conditioned diffusion
│   └── text.py                   # Embedding→text via nearest neighbor
├── probes/                       # NEW: Model state capture
│   ├── __init__.py
│   ├── activation_probe.py       # Hook-based multi-layer capture
│   └── state_renderer.py         # Render captured states per-layer
├── sae/
│   ├── lens.py                   # (existing)
│   ├── training.py               # (existing)
│   └── feature_renderer.py       # NEW: Per-feature visualization
```

### Module Details

#### 1. `renderers/base.py` — DirectRenderer Protocol

```python
class DirectRenderer(Protocol):
    output_modality: str

    def render(self, embedding: Tensor, **kwargs) -> RenderResult: ...
```

#### 2. `renderers/raw.py` — RawDecoder

Linear projection from embedding space to pixel space. No learned prior, no
regularization. The most faithful (and likely least human-legible) view.

```python
class RawDecoder(DirectRenderer):
    """Project embedding directly to pixel space via learned linear map."""

    def __init__(self, embed_dim: int, output_shape: tuple):
        # Simple linear: embed_dim → prod(output_shape)
        self.projection = nn.Linear(embed_dim, prod(output_shape))

    def render(self, embedding: Tensor) -> RenderResult:
        pixels = self.projection(embedding).reshape(output_shape)
        return RenderResult(output=pixels, ...)
```

#### 3. `renderers/projection.py` — ProjectionDecoder

Learned MLP that maps from embedding space to generator conditioning space,
then uses the generator for final decode. Trained offline on (embedding, image)
pairs.

```python
class ProjectionDecoder(DirectRenderer):
    """MLP from embedding space to generator latent space."""

    def __init__(self, embed_dim: int, latent_dim: int, hidden_dims: list[int]):
        self.mlp = build_mlp(embed_dim, latent_dim, hidden_dims)
        self.generator = None  # set after construction

    def render(self, embedding: Tensor) -> RenderResult:
        latent = self.mlp(embedding)
        output = self.generator.decode(latent.reshape(self.generator.latent_shape))
        return RenderResult(output=output, ...)
```

#### 4. `renderers/ip_adapter.py` — IPAdapterRenderer

Uses the pretrained IP-Adapter to condition SDXL on an embedding vector.
Single forward pass through diffusion (50 steps, no optimization loop).

```python
class IPAdapterRenderer(DirectRenderer):
    """Embedding-conditioned image generation via IP-Adapter + SDXL."""

    def __init__(self, model_id: str = "h94/IP-Adapter"):
        # Load SDXL + IP-Adapter weights
        ...

    def render(self, embedding: Tensor,
               num_inference_steps: int = 50,
               guidance_scale: float = 7.5) -> RenderResult:
        # Inject embedding via IP-Adapter cross-attention
        # Run standard SDXL diffusion
        ...
```

#### 5. `renderers/text.py` — TextRenderer

Decodes embeddings to text using nearest-neighbor search against a precomputed
text embedding index. Returns the closest text descriptions.

```python
class TextRenderer(DirectRenderer):
    """Decode embedding to text via nearest-neighbor in text embedding space."""

    def __init__(self, encoder, vocabulary: list[str]):
        # Precompute text embeddings for vocabulary
        ...

    def render(self, embedding: Tensor, k: int = 5) -> RenderResult:
        # Find k nearest text embeddings
        # Return as concatenated description
        ...
```

#### 6. `probes/activation_probe.py` — ActivationProbe

Hook-based capture of intermediate activations from any encoder.

```python
class ActivationProbe:
    """Capture intermediate activations from encoder layers."""

    def capture(self, encoder, input_tensor: Tensor,
                layers: list[int] | None = None) -> ModelState:
        # Register forward hooks on specified layers
        # Run forward pass
        # Collect and return activations
        ...

@dataclass
class ModelState:
    layer_activations: dict[int, Tensor]  # layer_idx → activation
    final_embedding: Tensor
    encoder_name: str
    input_description: str
```

#### 7. `probes/state_renderer.py` — StateRenderer

Renders captured model states through any DirectRenderer.

```python
class StateRenderer:
    """Render captured model states as different modalities."""

    def render_layer(self, state: ModelState, layer: int,
                     renderer: DirectRenderer) -> RenderResult:
        activation = state.layer_activations[layer]
        return renderer.render(activation)

    def render_progression(self, state: ModelState,
                          renderer: DirectRenderer) -> list[RenderResult]:
        return [renderer.render(act) for act in state.layer_activations.values()]
```

#### 8. `sae/feature_renderer.py` — FeatureRenderer

Renders individual SAE features as images/audio/video.

```python
class FeatureRenderer:
    """Render individual SAE features as visual/audio outputs."""

    def render_feature(self, sae: SAELens, feature_idx: int,
                       activation: float,
                       renderer: DirectRenderer) -> RenderResult:
        # Create embedding from single feature direction
        direction = sae._W_dec[:, feature_idx]  # decoder column = feature direction
        embedding = F.normalize(direction.unsqueeze(0)) * activation
        return renderer.render(embedding)

    def render_decomposition(self, decomposition: SAEDecomposition,
                            sae: SAELens,
                            renderer: DirectRenderer,
                            max_features: int = 10) -> dict[str, RenderResult]:
        # Render top-k active features individually
        ...

    def render_feature_grid(self, sae: SAELens,
                           feature_indices: list[int],
                           renderer: DirectRenderer,
                           grid_cols: int = 4) -> Tensor:
        # Compose individual feature renders into a grid
        ...
```

### Engine Integration

```python
class EmbeddingArtEngine:
    # Existing methods preserved...

    def render_direct(
        self,
        spec: ConceptSpec,
        renderer: DirectRenderer,
        encoder: str | None = None,
    ) -> RenderResult:
        """Render a concept directly through a DirectRenderer (no optimization)."""
        enc = self._resolve_encoder(encoder)
        concept = enc.encode(spec)
        return renderer.render(concept.embedding)

    def render_state(
        self,
        spec: ConceptSpec,
        renderer: DirectRenderer,
        encoder: str | None = None,
        layers: list[int] | None = None,
    ) -> list[RenderResult]:
        """Render multi-layer states of encoding a concept."""
        enc = self._resolve_encoder(encoder)
        probe = ActivationProbe()
        state = probe.capture(enc, spec)
        sr = StateRenderer()
        return sr.render_progression(state, renderer)

    def render_features(
        self,
        spec: ConceptSpec,
        renderer: DirectRenderer,
        sae: SAELens,
        encoder: str | None = None,
        max_features: int = 10,
    ) -> dict[str, RenderResult]:
        """Render individual SAE features of a concept."""
        enc = self._resolve_encoder(encoder)
        concept = enc.encode(spec)
        decomposition = sae.decompose(concept.embedding)
        fr = FeatureRenderer()
        return fr.render_decomposition(decomposition, sae, renderer, max_features)
```

### CLI Commands

```bash
# Direct rendering (no optimization loop)
embed-art direct -t "goldfish" 1.0 --renderer raw --output image
embed-art direct -t "goldfish" 1.0 --renderer projection --output image
embed-art direct -t "goldfish" 1.0 --renderer ip-adapter --output image

# Multi-layer state rendering
embed-art probe -t "goldfish" 1.0 --encoder siglip2 --renderer raw \
    --layers 0,6,12,18,26 --output-dir ./states/

# Feature-level rendering
embed-art feature-viz -t "goldfish" 1.0 --sae sae_models/siglip2.safetensors \
    --renderer ip-adapter --max-features 10 --output-dir ./features/

# Text decoding
embed-art decode-text -t "goldfish" 1.0 --encoder imagebind --k 10

# Compare all renderers on same concept
embed-art compare-renderers -t "goldfish" 1.0 \
    --renderers raw,projection,ip-adapter,optimization \
    --output-dir ./comparison/
```

## Research Foundations

| Technique | Source | Application |
|-----------|--------|-------------|
| IP-Adapter | Ye et al. 2023, 22M params | Embedding-conditioned diffusion |
| RCDM | Bordes et al. 2022, Meta | Representation-conditioned diffusion |
| unCLIP/DALL-E 2 | Ramesh et al. 2022 | Prior + decoder from embeddings |
| SAE Feature Viz | Anthropic 2024, Templeton et al. | Per-feature rendering |
| MIMIC multi-layer | arxiv 2508.07833 | Layer-wise state capture |
| Implicit Inversion | arxiv 2505.23161 | INR from CLIP embeddings |
| Concept Bottleneck AE | CVPR 2025 | Interpretable generator control |
| V2A-Mapper | AAAI 2024 | Cross-modal embedding projection |
| Gradient Slingshots | 2024 | Cosine similarity pathologies |
| VISTA | Dec 2024 | Neural representation cartography |

## Relationship to v1/v2

v3 is additive — it doesn't remove the optimization loop, it adds direct
rendering as a first-class alternative. The engine gains new methods while
keeping all existing ones. The fidelity spectrum lets users choose their
tradeoff between faithfulness and aesthetics.

v1 optimization → still available via `embed-art optimize`
v2 multi-encoder → still available via `embed-art render`
v3 direct rendering → new via `embed-art direct`
v3 state probing → new via `embed-art probe`
v3 feature viz → new via `embed-art feature-viz`
