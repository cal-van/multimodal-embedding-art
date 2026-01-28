# Embedding Art

Generate art by optimizing toward coordinates in multimodal embedding space.

This system extracts the "platonic ideals" that live inside neural networks—not "generate me a goldfish" but "show me the direction in representation space that means goldfish, cranked to maximum."

## What it does

- Takes any combination of text, image, audio as input concepts
- Combines them using embedding arithmetic (addition, subtraction, interpolation)
- Optimizes a generative model's latent space to maximize similarity with the target embedding
- Outputs images (audio and video coming soon)

The outputs are what the model *thinks* these concepts look like—maximally activated representations, not training data.

## Quick Start

### 1. Install dependencies

```bash
# Clone this repo
cd embedding-art

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install package
pip install -e ".[dev]"
```

### 2. Install ImageBind

```bash
# ImageBind is required for the multimodal encoder
git clone https://github.com/facebookresearch/ImageBind
cd ImageBind
pip install -e .
cd ..
```

### 3. Run your first optimization

```bash
# Via CLI
embed-art optimize --target-text "goldfish" 1.0 --output image

# Or in Python
python -c "
from embedding_art import EmbeddingArtEngine, Concept, OptimizationConfig
from embedding_art.encoders.imagebind import ImageBindEncoder
from embedding_art.generators.image import SDXLImageGenerator

encoder = ImageBindEncoder(device='mps')
generator = SDXLImageGenerator(device='mps')

engine = EmbeddingArtEngine(encoder)
engine.register_generator('image', generator)

target = Concept.from_text('goldfish', encoder)
result = engine.optimize(target, 'image', OptimizationConfig(steps=500))

result.get_final_image(generator).save('goldfish.png')
print(f'Saved! Similarity: {result.final_similarity:.4f}')
"
```

## Concept Algebra

Combine concepts using arithmetic:

```python
# Addition
fire_water = Concept.from_text("fire", encoder) + Concept.from_text("water", encoder)

# Weighted combination
sunset_ocean = 0.3 * Concept.from_text("sunset", encoder) + 0.7 * Concept.from_text("ocean", encoder)

# Subtraction (remove attributes)
hairless = Concept.from_text("dog", encoder) - 0.3 * Concept.from_text("fur", encoder)

# Spherical interpolation
midpoint = Concept.slerp(concept_a, concept_b, t=0.5)

# Cross-modal combination
thunder_purple = Concept.from_audio("thunder.wav", encoder) + 0.3 * Concept.from_text("purple", encoder)
```

## CLI Reference

```bash
# Basic optimization
embed-art optimize -t "goldfish" 1.0 -o image

# Combined concepts
embed-art optimize -t "fire" 0.5 -t "water" 0.5 -o image

# Cross-modal (audio + text → image)
embed-art optimize -a thunder.wav 1.0 -t "purple" 0.3 -o image -p thunder_purple.png

# Interpolation series
embed-art interpolate -a "goldfish" -b "flamingo" -s 10 -o image -d outputs/interp/

# Inspect embeddings
embed-art embed -t "goldfish" -s embeddings/goldfish.pt
embed-art compare embeddings/goldfish.pt embeddings/orange.pt

# Visualization
embed-art visualize similarity checkpoint.pt
embed-art visualize embeddings -t "cat" -t "dog" -t "car" --method tsne

# Upscaling
embed-art upscale input.png --scale 4 --output upscaled.png

# Batch processing
embed-art batch run batch_config.yaml
```

## Configuration

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `steps` | 2000 | Optimization iterations |
| `learning_rate` | 0.1 | Step size |
| `seed` | None | Random seed for reproducibility |
| `regularization.total_variation` | 0.01 | Spatial smoothness |
| `regularization.spectral` | 0.001 | High-frequency penalty |
| `regularization.latent_norm` | 0.1 | Keep latent in distribution |

## Hardware

Tested on M1 Max MacBook Pro (64GB). Memory usage:
- ImageBind: ~3GB
- SDXL VAE: ~1GB  
- Optimization overhead: ~2GB
- **Total: ~6-8GB**

Should work fine on any Apple Silicon Mac with 16GB+ unified memory.

## Project Structure

```
embedding-art/
├── src/embedding_art/
│   ├── core/           # Concept, Engine, Config
│   ├── encoders/       # ImageBind wrapper
│   ├── generators/     # SDXL VAE, AudioLDM (planned)
│   ├── regularizers/   # Total variation, spectral, etc.
│   └── cli/            # Command line interface
├── notebooks/          # Jupyter notebooks for exploration
└── outputs/            # Generated outputs
```

## Roadmap

- [x] Image generation via SDXL VAE
- [x] Concept algebra (add, subtract, slerp)
- [x] CLI interface
- [x] Audio output via AudioLDM 2
- [x] Video output via SVD
- [x] Resolution upscaling
- [x] Batch processing
- [ ] Web UI

## License

MIT
