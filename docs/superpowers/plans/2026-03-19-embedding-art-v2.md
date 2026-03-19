# Embedding Art v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform embedding-art from a single-encoder optimization tool into a pluggable multi-encoder platform with multi-layer feature matching loss, SAE-based interpretable decomposition, and dual rendering paths.

**Architecture:** Incremental migration in 5 phases. Each phase is additive and non-breaking — existing tests pass at every step. New abstractions (`ConceptSpec`, `EncoderRegistry`, `CompositeLoss`, `SAELens`) are added alongside existing code, with the old API preserved via adapters until Phase 5 deprecation.

**Tech Stack:** Python 3.10+, PyTorch, HuggingFace transformers/diffusers, Click CLI, safetensors. Target: M1 Max 64GB (MPS backend).

**Spec:** `docs/superpowers/specs/2026-03-19-embedding-art-v2-design.md`
**Research:** `docs/superpowers/specs/2026-03-19-research-findings.md`

---

## File Structure

### New Files

```
src/embedding_art/
├── core/
│   ├── concept_spec.py          # ConceptSpec dataclass
│   ├── loss.py                  # CompositeLoss, LossConfig, LossBreakdown
│   ├── render_result.py         # RenderResult, OptimizationHistory
│   └── strategies.py            # RenderingStrategy protocol, OptimizationStrategy, DiffusionGuidanceStrategy
├── encoders/
│   ├── registry.py              # EncoderRegistry, EncoderCard, EncoderCapability
│   ├── siglip2.py               # SigLIP2Encoder
│   ├── clap.py                  # CLAPEncoder
│   └── features.py              # LayerFeatures, FeatureStatistics, FeatureStatisticsExtractor, ViTStatisticsExtractor
├── generators/
│   ├── inr.py                   # INRGenerator (DirectGenerator protocol)
│   └── base.py                  # Extended: LatentGenerator, DirectGenerator, DiffusionGenerator protocols
├── sae/
│   ├── __init__.py
│   ├── lens.py                  # SAELens, SAEDecomposition
│   └── training.py              # SAE collection and training pipeline
└── cli/commands/
    ├── render.py                # New render command
    ├── compare.py               # New compare command
    ├── decompose.py             # New decompose command
    └── sae.py                   # SAE training/inspection commands
```

### Modified Files

```
src/embedding_art/
├── __init__.py                  # Add new exports
├── exceptions.py                # Add new exception types
├── core/
│   ├── concept.py               # Add source_input field, from_spec() classmethod
│   ├── config.py                # Add LossConfig field to OptimizationConfig
│   └── engine.py                # Add from_registry(), render(), render_compare()
├── encoders/
│   ├── base.py                  # Extend protocol with encode(), get_layer_features(), unload(), card
│   └── imagebind.py             # Adapt to extended protocol
├── generators/
│   └── diffusion.py             # Add generate_guided() method
└── cli/main.py                  # Register new commands

tests/
├── conftest.py                  # Add MockEncoderV2 with card and layer features
├── test_concept_spec.py         # NEW
├── test_encoder_registry.py     # NEW
├── test_composite_loss.py       # NEW
├── test_render_result.py        # NEW
├── test_strategies.py           # NEW
├── test_siglip2_encoder.py      # NEW
├── test_clap_encoder.py         # NEW
├── test_layer_features.py       # NEW
├── test_sae_lens.py             # NEW
├── test_inr_generator.py        # NEW
├── test_render_command.py       # NEW
└── test_compare_command.py      # NEW
```

---

## Phase 1: Core Abstractions (non-breaking)

### Task 1: New Exception Types

**Files:**
- Modify: `src/embedding_art/exceptions.py`
- Test: `tests/test_exceptions.py` (new)

- [ ] **Step 1: Write tests for new exceptions**

```python
# tests/test_exceptions.py
import pytest
from embedding_art.exceptions import (
    EmbeddingArtError,
    EncoderNotFoundError,
    EncoderCapabilityError,
    SAENotTrainedError,
    MemoryBudgetExceededError,
    FeatureNotFoundError,
)


def test_encoder_not_found_error():
    err = EncoderNotFoundError("siglip2-so400m", ["imagebind", "clap"])
    assert "siglip2-so400m" in str(err)
    assert "imagebind" in str(err)
    assert isinstance(err, EmbeddingArtError)


def test_encoder_capability_error():
    err = EncoderCapabilityError("imagebind", "MULTI_LAYER_FEATURES")
    assert "imagebind" in str(err)
    assert "MULTI_LAYER_FEATURES" in str(err)
    assert isinstance(err, EmbeddingArtError)


def test_sae_not_trained_error():
    err = SAENotTrainedError("siglip2-so400m")
    assert "siglip2-so400m" in str(err)
    assert isinstance(err, EmbeddingArtError)


def test_memory_budget_exceeded_error():
    err = MemoryBudgetExceededError(
        requested_mb=20000, available_mb=15000, encoders=["siglip2", "imagebind", "lco-omni"]
    )
    assert "20000" in str(err) or "20,000" in str(err)
    assert isinstance(err, EmbeddingArtError)


def test_feature_not_found_error():
    err = FeatureNotFoundError("sparkly", ["golden", "aquatic", "scales"])
    assert "sparkly" in str(err)
    assert isinstance(err, EmbeddingArtError)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_exceptions.py -v`
Expected: FAIL — import errors

- [ ] **Step 3: Implement new exception classes**

Append to `src/embedding_art/exceptions.py`:

```python
class EncoderNotFoundError(EmbeddingArtError):
    """Raised when an encoder name isn't in the registry."""
    def __init__(self, name: str, available: list[str]) -> None:
        self.name = name
        self.available = available
        message = (
            f"Encoder '{name}' not found.\n\n"
            f"Available encoders: {', '.join(available)}\n\n"
            "Register new encoders with EncoderRegistry or check for typos."
        )
        super().__init__(message)


class EncoderCapabilityError(EmbeddingArtError):
    """Raised when an operation requires a capability the encoder lacks."""
    def __init__(self, encoder_name: str, capability: str) -> None:
        self.encoder_name = encoder_name
        self.capability = capability
        message = (
            f"Encoder '{encoder_name}' does not support '{capability}'.\n\n"
            "Choose an encoder with this capability or disable the feature that requires it."
        )
        super().__init__(message)


class SAENotTrainedError(EmbeddingArtError):
    """Raised when SAE features are requested but no SAE artifact exists."""
    def __init__(self, encoder_name: str) -> None:
        self.encoder_name = encoder_name
        message = (
            f"No SAE artifact found for encoder '{encoder_name}'.\n\n"
            "Train an SAE first:\n"
            f"  embed-art sae collect --encoder {encoder_name} --dataset <dataset> --output embeds/\n"
            f"  embed-art sae train --embeddings embeds/ --output sae_models/{encoder_name}.safetensors"
        )
        super().__init__(message)


class MemoryBudgetExceededError(EmbeddingArtError):
    """Raised when loading encoders would exceed memory budget."""
    def __init__(self, requested_mb: int, available_mb: int, encoders: list[str]) -> None:
        self.requested_mb = requested_mb
        self.available_mb = available_mb
        self.encoders = encoders
        message = (
            f"Loading encoders would require ~{requested_mb}MB but only ~{available_mb}MB available.\n\n"
            f"Requested encoders: {', '.join(encoders)}\n\n"
            "Suggestions:\n"
            "  - Use fewer encoders simultaneously\n"
            "  - Enable sequential mode (loads one encoder at a time)\n"
            "  - Choose smaller encoder variants"
        )
        super().__init__(message)


class FeatureNotFoundError(EmbeddingArtError):
    """Raised when an SAE feature name isn't in the vocabulary."""
    def __init__(self, feature_name: str, available: list[str]) -> None:
        self.feature_name = feature_name
        self.available = available
        # Show closest matches
        preview = available[:10]
        message = (
            f"SAE feature '{feature_name}' not found in vocabulary.\n\n"
            f"Available features (first 10): {', '.join(preview)}\n"
            f"Total features: {len(available)}"
        )
        super().__init__(message)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_exceptions.py -v`
Expected: All PASS

- [ ] **Step 5: Run existing tests to verify nothing broke**

Run: `pytest tests/ -x --timeout=30 -m "not slow and not integration"`
Expected: All existing tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/embedding_art/exceptions.py tests/test_exceptions.py
git commit -m "feat: add exception types for encoder registry, SAE, and memory budget"
```

---

### Task 2: ConceptSpec

**Files:**
- Create: `src/embedding_art/core/concept_spec.py`
- Test: `tests/test_concept_spec.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_concept_spec.py
from pathlib import Path

from embedding_art.core.concept_spec import ConceptSpec


def test_text_concept_spec():
    spec = ConceptSpec(text="a goldfish")
    assert spec.text == "a goldfish"
    assert spec.image is None
    assert spec.audio is None
    assert spec.video is None
    assert spec.weight == 1.0


def test_image_concept_spec():
    spec = ConceptSpec(image=Path("/tmp/test.png"), weight=0.8)
    assert spec.image == Path("/tmp/test.png")
    assert spec.weight == 0.8


def test_multimodal_concept_spec():
    spec = ConceptSpec(text="ocean", audio=Path("/tmp/waves.wav"))
    assert spec.text == "ocean"
    assert spec.audio == Path("/tmp/waves.wav")


def test_concept_spec_requires_at_least_one_modality():
    """A spec with no modalities is technically valid but should repr cleanly."""
    spec = ConceptSpec()
    assert repr(spec)  # Shouldn't crash


def test_concept_spec_description():
    spec = ConceptSpec(text="a goldfish", weight=0.7)
    desc = spec.describe()
    assert "goldfish" in desc
    assert "0.7" in desc
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_concept_spec.py -v`
Expected: FAIL — import error

- [ ] **Step 3: Implement ConceptSpec**

```python
# src/embedding_art/core/concept_spec.py
"""
Encoder-agnostic concept specification.

ConceptSpec describes what to render without binding to any specific encoder.
Each encoder materializes a ConceptSpec into a Concept with its own embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConceptSpec:
    """
    What to render — not yet bound to any encoder.

    Unlike Concept (which holds an encoder-specific embedding), ConceptSpec
    is encoder-agnostic. It describes the target conceptually, allowing
    the same spec to be materialized by multiple encoders for comparison.

    At least one modality field should be set. Multiple modalities can be
    set for multimodal concepts (e.g., text + image).
    """

    text: str | None = None
    image: Path | None = None
    audio: Path | None = None
    video: Path | None = None
    weight: float = 1.0

    def describe(self) -> str:
        """Human-readable description of this spec."""
        parts = []
        if self.text:
            parts.append(f'text:"{self.text}"')
        if self.image:
            parts.append(f"image:{self.image.name}")
        if self.audio:
            parts.append(f"audio:{self.audio.name}")
        if self.video:
            parts.append(f"video:{self.video.name}")
        desc = " + ".join(parts) if parts else "empty"
        if self.weight != 1.0:
            desc = f"{self.weight}*({desc})"
        return desc

    def __repr__(self) -> str:
        return f"ConceptSpec({self.describe()})"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_concept_spec.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/embedding_art/core/concept_spec.py tests/test_concept_spec.py
git commit -m "feat: add ConceptSpec for encoder-agnostic concept specification"
```

---

### Task 3: EncoderCapability, EncoderCard, and Registry

**Files:**
- Create: `src/embedding_art/encoders/registry.py`
- Test: `tests/test_encoder_registry.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_encoder_registry.py
import pytest
import torch

from embedding_art.encoders.registry import (
    EncoderCapability,
    EncoderCard,
    EncoderRegistry,
)
from embedding_art.exceptions import EncoderNotFoundError, MemoryBudgetExceededError


def test_encoder_capability_flags():
    caps = EncoderCapability.TEXT | EncoderCapability.IMAGE | EncoderCapability.BACKPROP_OPTIMIZABLE
    assert EncoderCapability.TEXT in caps
    assert EncoderCapability.AUDIO not in caps
    assert EncoderCapability.BACKPROP_OPTIMIZABLE in caps


def test_encoder_card():
    card = EncoderCard(
        name="test-encoder",
        capabilities=EncoderCapability.TEXT | EncoderCapability.IMAGE,
        embedding_dim=1024,
        memory_estimate_mb=3000,
        backprop_cost="low",
    )
    assert card.name == "test-encoder"
    assert card.embedding_dim == 1024
    assert EncoderCapability.TEXT in card.capabilities


def test_registry_list_available(mock_encoder_cls):
    registry = EncoderRegistry()
    registry.register("mock", mock_encoder_cls)
    cards = registry.list_available()
    assert len(cards) == 1
    assert cards[0].name == "mock"


def test_registry_load(mock_encoder_cls):
    registry = EncoderRegistry()
    registry.register("mock", mock_encoder_cls)
    encoder = registry.load("mock")
    assert encoder is not None


def test_registry_load_caches(mock_encoder_cls):
    registry = EncoderRegistry()
    registry.register("mock", mock_encoder_cls)
    enc1 = registry.load("mock")
    enc2 = registry.load("mock")
    assert enc1 is enc2  # Same instance


def test_registry_load_not_found():
    registry = EncoderRegistry()
    with pytest.raises(EncoderNotFoundError):
        registry.load("nonexistent")


def test_registry_unload(mock_encoder_cls):
    registry = EncoderRegistry()
    registry.register("mock", mock_encoder_cls)
    registry.load("mock")
    registry.unload("mock")
    # Loading again should create a new instance
    enc = registry.load("mock")
    assert enc is not None


def test_registry_get_for_modality(mock_encoder_cls):
    registry = EncoderRegistry()
    registry.register("mock", mock_encoder_cls)
    text_encoders = registry.get_for_modality("text")
    assert "mock" in text_encoders


def test_registry_can_fit(mock_encoder_cls):
    registry = EncoderRegistry()
    registry.register("mock", mock_encoder_cls)
    assert registry.can_fit(["mock"], memory_budget_mb=50000)
    assert not registry.can_fit(["mock"], memory_budget_mb=1)


@pytest.fixture
def mock_encoder_cls():
    """A minimal encoder class for registry testing."""
    from embedding_art.encoders.registry import EncoderCapability, EncoderCard

    class _MockEncoderForRegistry:
        card = EncoderCard(
            name="mock",
            capabilities=EncoderCapability.TEXT | EncoderCapability.IMAGE | EncoderCapability.BACKPROP_OPTIMIZABLE,
            embedding_dim=1024,
            memory_estimate_mb=3000,
            backprop_cost="low",
        )

        @property
        def embedding_dim(self):
            return 1024

        @property
        def device(self):
            return torch.device("cpu")

        def encode_text(self, text):
            return torch.randn(1, 1024)

        def encode_image(self, image):
            return torch.randn(1, 1024)

        def encode_audio(self, audio, start=0.0, duration=2.0):
            return torch.randn(1, 1024)

        def encode_video(self, video, timestamp=0.0):
            return torch.randn(1, 1024)

        def unload(self):
            pass

    return _MockEncoderForRegistry
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_encoder_registry.py -v`
Expected: FAIL — import error

- [ ] **Step 3: Implement registry**

```python
# src/embedding_art/encoders/registry.py
"""
Encoder registry with capability-based discovery.

Supports lazy loading, memory-aware admission control, and
modality-based encoder lookup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Flag, auto
from typing import Any, Literal, Protocol

import torch

from embedding_art.exceptions import EncoderNotFoundError, MemoryBudgetExceededError

logger = logging.getLogger(__name__)


class EncoderCapability(Flag):
    """Capabilities an encoder may support."""

    TEXT = auto()
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()
    DEPTH = auto()
    BACKPROP_OPTIMIZABLE = auto()
    MULTI_LAYER_FEATURES = auto()


# Map capability flags to modality names for lookup
_MODALITY_TO_CAPABILITY = {
    "text": EncoderCapability.TEXT,
    "image": EncoderCapability.IMAGE,
    "audio": EncoderCapability.AUDIO,
    "video": EncoderCapability.VIDEO,
    "depth": EncoderCapability.DEPTH,
}


@dataclass(frozen=True)
class EncoderCard:
    """Metadata describing an encoder's capabilities and resource requirements."""

    name: str
    capabilities: EncoderCapability
    embedding_dim: int
    memory_estimate_mb: int
    backprop_cost: Literal["low", "medium", "high"]


class EncoderRegistry:
    """
    Registry for encoder discovery, lazy loading, and lifecycle management.

    Usage:
        registry = EncoderRegistry()
        registry.register("siglip2-so400m", SigLIP2Encoder)
        encoder = registry.load("siglip2-so400m")
    """

    def __init__(self) -> None:
        self._classes: dict[str, type] = {}
        self._instances: dict[str, Any] = {}

    def register(self, name: str, encoder_cls: type) -> None:
        """Register an encoder class. Must have a `card` class attribute."""
        if not hasattr(encoder_cls, "card"):
            raise ValueError(f"Encoder class {encoder_cls.__name__} must have a 'card' attribute")
        self._classes[name] = encoder_cls
        logger.debug("Registered encoder: %s", name)

    def list_available(self) -> list[EncoderCard]:
        """List all registered encoder cards."""
        return [cls.card for cls in self._classes.values()]

    def get_card(self, name: str) -> EncoderCard:
        """Get the card for a registered encoder."""
        if name not in self._classes:
            raise EncoderNotFoundError(name, list(self._classes.keys()))
        return self._classes[name].card

    def load(self, name: str, **kwargs: Any) -> Any:
        """
        Load an encoder instance (lazy, cached).

        Returns the cached instance if already loaded.
        Pass keyword arguments to the encoder constructor on first load.
        """
        if name in self._instances:
            return self._instances[name]

        if name not in self._classes:
            raise EncoderNotFoundError(name, list(self._classes.keys()))

        logger.info("Loading encoder: %s", name)
        instance = self._classes[name](**kwargs)
        self._instances[name] = instance
        return instance

    def unload(self, name: str) -> None:
        """Unload an encoder instance, freeing resources."""
        if name in self._instances:
            instance = self._instances.pop(name)
            if hasattr(instance, "unload"):
                instance.unload()
            logger.info("Unloaded encoder: %s", name)

    def get_for_modality(self, modality: str) -> list[str]:
        """List encoder names that support a given modality."""
        cap = _MODALITY_TO_CAPABILITY.get(modality)
        if cap is None:
            return []
        return [
            name
            for name, cls in self._classes.items()
            if cap in cls.card.capabilities
        ]

    def can_fit(self, names: list[str], memory_budget_mb: int) -> bool:
        """Check if loading all named encoders fits within memory budget."""
        total = 0
        for name in names:
            if name not in self._classes:
                raise EncoderNotFoundError(name, list(self._classes.keys()))
            total += self._classes[name].card.memory_estimate_mb
        return total <= memory_budget_mb * 0.8  # 80% safety margin

    def loaded_encoders(self) -> list[str]:
        """List names of currently loaded encoders."""
        return list(self._instances.keys())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_encoder_registry.py -v`
Expected: All PASS

- [ ] **Step 5: Run all tests**

Run: `pytest tests/ -x --timeout=30 -m "not slow and not integration"`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/embedding_art/encoders/registry.py tests/test_encoder_registry.py
git commit -m "feat: add EncoderRegistry with capability flags and memory-aware loading"
```

---

### Task 4: LayerFeatures and FeatureStatistics

**Files:**
- Create: `src/embedding_art/encoders/features.py`
- Test: `tests/test_layer_features.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_layer_features.py
import torch

from embedding_art.encoders.features import (
    FeatureStatistics,
    LayerFeatures,
    ViTStatisticsExtractor,
)


def test_layer_features_creation():
    tensor = torch.randn(1, 197, 768)  # ViT patch tokens: [batch, patches, dim]
    lf = LayerFeatures(
        tensor=tensor,
        spatial=True,
        shape_semantic="batch_tokens_dim",
        layer_name="layer_6",
    )
    assert lf.spatial is True
    assert lf.shape_semantic == "batch_tokens_dim"
    assert lf.tensor.shape == (1, 197, 768)


def test_vit_statistics_extractor():
    extractor = ViTStatisticsExtractor()
    tensor = torch.randn(1, 197, 768)
    lf = LayerFeatures(tensor=tensor, spatial=True, shape_semantic="batch_tokens_dim", layer_name="layer_6")

    stats = extractor.extract(lf)
    assert isinstance(stats, FeatureStatistics)
    assert stats.mean.shape == (768,)  # Per-feature-dim mean across patches
    assert stats.std.shape == (768,)


def test_vit_statistics_deterministic():
    extractor = ViTStatisticsExtractor()
    torch.manual_seed(42)
    tensor = torch.randn(1, 197, 768)
    lf = LayerFeatures(tensor=tensor, spatial=True, shape_semantic="batch_tokens_dim", layer_name="test")

    stats1 = extractor.extract(lf)
    stats2 = extractor.extract(lf)
    assert torch.allclose(stats1.mean, stats2.mean)
    assert torch.allclose(stats1.std, stats2.std)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_layer_features.py -v`
Expected: FAIL — import error

- [ ] **Step 3: Implement**

```python
# src/embedding_art/encoders/features.py
"""
Layer feature extraction and statistics for multi-layer feature matching.

Different encoder architectures produce different intermediate representations.
FeatureStatisticsExtractor handles the per-architecture differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class LayerFeatures:
    """Typed container for intermediate encoder layer activations."""

    tensor: torch.Tensor
    spatial: bool
    shape_semantic: str  # "batch_tokens_dim" | "batch_channels_height_width"
    layer_name: str


@dataclass
class FeatureStatistics:
    """Summary statistics for a layer's features."""

    mean: torch.Tensor
    std: torch.Tensor


class FeatureStatisticsExtractor(Protocol):
    """Protocol for extracting statistics from layer features."""

    def extract(self, features: LayerFeatures) -> FeatureStatistics: ...


class ViTStatisticsExtractor:
    """
    Statistics extractor for Vision Transformer encoders.

    ViT intermediate layers produce patch token sequences [B, N_patches, D].
    Computes per-feature-dimension statistics across patches.
    """

    def extract(self, features: LayerFeatures) -> FeatureStatistics:
        # [B, N_patches, D] -> mean/std over patches dimension
        tensor = features.tensor
        if tensor.dim() == 3:
            mean = tensor.mean(dim=1).squeeze(0)  # [D]
            std = tensor.std(dim=1).squeeze(0)  # [D]
        else:
            # Fallback: flatten and compute
            mean = tensor.flatten(start_dim=1).mean(dim=1).squeeze(0)
            std = tensor.flatten(start_dim=1).std(dim=1).squeeze(0)
        return FeatureStatistics(mean=mean, std=std)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_layer_features.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/embedding_art/encoders/features.py tests/test_layer_features.py
git commit -m "feat: add LayerFeatures, FeatureStatistics, and ViTStatisticsExtractor"
```

---

### Task 5: Extend Encoder Protocol and Adapt ImageBind

**Files:**
- Modify: `src/embedding_art/encoders/base.py`
- Modify: `src/embedding_art/encoders/imagebind.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_encoder_protocol.py` (new)

- [ ] **Step 1: Write tests for extended protocol**

```python
# tests/test_encoder_protocol.py
import torch
import pytest

from embedding_art.core.concept_spec import ConceptSpec
from embedding_art.encoders.registry import EncoderCapability


def test_mock_encoder_has_card(mock_encoder):
    """MockEncoder should have a card attribute after protocol extension."""
    assert hasattr(mock_encoder, "card")
    assert mock_encoder.card.embedding_dim == 1024
    assert EncoderCapability.TEXT in mock_encoder.card.capabilities


def test_mock_encoder_encode_spec(mock_encoder):
    """MockEncoder should support encode(ConceptSpec)."""
    spec = ConceptSpec(text="goldfish")
    concept = mock_encoder.encode(spec)
    assert concept.embedding.shape == (1, 1024)
    assert "goldfish" in concept.description


def test_mock_encoder_unload(mock_encoder):
    """MockEncoder should support unload()."""
    mock_encoder.unload()  # Should not raise


def test_mock_encoder_backward_compat(mock_encoder):
    """Old encode_text/encode_image methods still work."""
    emb = mock_encoder.encode_text("goldfish")
    assert emb.shape == (1, 1024)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_encoder_protocol.py -v`
Expected: FAIL — mock_encoder doesn't have card

- [ ] **Step 3: Extend base protocol**

Update `src/embedding_art/encoders/base.py` — add new methods while keeping old ones:

```python
"""
Base encoder protocol.

Encoders map various modalities into a shared embedding space.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import torch
from PIL import Image

if TYPE_CHECKING:
    from embedding_art.core.concept import Concept
    from embedding_art.core.concept_spec import ConceptSpec
    from embedding_art.encoders.features import LayerFeatures
    from embedding_art.encoders.registry import EncoderCard


class Encoder(Protocol):
    """Protocol for multimodal encoders.

    v1 methods (encode_text, encode_image, etc.) remain for backward compatibility.
    v2 adds: card, encode(spec), encode_for_optimization(), get_layer_features(), unload().
    """

    @property
    def embedding_dim(self) -> int:
        """Dimension of the embedding space."""
        ...

    @property
    def device(self) -> torch.device:
        """Device the encoder is on."""
        ...

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to embedding. Returns [1, embed_dim] tensor."""
        ...

    def encode_image(self, image: Path | Image.Image | torch.Tensor) -> torch.Tensor:
        """Encode image to embedding. Returns [1, embed_dim] tensor."""
        ...

    def encode_audio(
        self, audio: Path | torch.Tensor, start: float = 0.0, duration: float = 2.0
    ) -> torch.Tensor:
        """Encode audio to embedding. Returns [1, embed_dim] tensor."""
        ...

    def encode_video(
        self, video: Path | torch.Tensor, timestamp: float = 0.0
    ) -> torch.Tensor:
        """Encode video frame to embedding. Returns [1, embed_dim] tensor."""
        ...
```

Note: The v2 methods (`card`, `encode`, `encode_for_optimization`, `get_layer_features`, `unload`) are duck-typed, not in the Protocol, since not all encoders implement them yet. Code that needs them checks `hasattr` or `card.capabilities`.

- [ ] **Step 4: Update MockEncoder in conftest.py**

Add `card`, `encode(spec)`, `encode_for_optimization()`, and `unload()` to the existing `MockEncoder` class in `tests/conftest.py`. Keep all existing methods intact.

- [ ] **Step 5: Add card property to ImageBindEncoder**

Add to `src/embedding_art/encoders/imagebind.py`:

```python
from embedding_art.encoders.registry import EncoderCard, EncoderCapability

class ImageBindEncoder:
    # ... existing code ...

    @property
    def card(self) -> EncoderCard:
        return EncoderCard(
            name="imagebind",
            capabilities=(
                EncoderCapability.TEXT | EncoderCapability.IMAGE |
                EncoderCapability.AUDIO | EncoderCapability.VIDEO |
                EncoderCapability.DEPTH | EncoderCapability.BACKPROP_OPTIMIZABLE
            ),
            embedding_dim=self._embedding_dim,
            memory_estimate_mb=3000,
            backprop_cost="low",
        )

    def encode(self, spec):
        """Encode a ConceptSpec. Dispatches to modality-specific methods."""
        from embedding_art.core.concept import Concept
        if spec.text:
            return Concept.from_text(spec.text, self)
        if spec.image:
            return Concept.from_image(spec.image, self)
        if spec.audio:
            return Concept.from_audio(spec.audio, self)
        if spec.video:
            return Concept.from_video(spec.video, self)
        raise ValueError("ConceptSpec has no modality set")

    def unload(self):
        """Release model resources."""
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
        import gc; gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
```

- [ ] **Step 6: Run all tests**

Run: `pytest tests/ -x --timeout=30 -m "not slow and not integration"`
Expected: All PASS (old and new)

- [ ] **Step 7: Commit**

```bash
git add src/embedding_art/encoders/base.py src/embedding_art/encoders/imagebind.py tests/conftest.py tests/test_encoder_protocol.py
git commit -m "feat: extend Encoder protocol with card, encode(spec), unload(); adapt ImageBind"
```

---

### Task 6: Extend Concept with source_input and LossConfig into OptimizationConfig

**Files:**
- Modify: `src/embedding_art/core/concept.py`
- Modify: `src/embedding_art/core/config.py`
- Test: existing `tests/test_concept.py` and `tests/test_config.py` must still pass

- [ ] **Step 1: Write tests for source_input**

```python
# Add to tests/test_concept.py or create tests/test_concept_source_input.py
import torch
from embedding_art.core.concept import Concept


def test_concept_source_input_default_none():
    emb = torch.randn(1, 1024)
    c = Concept(embedding=emb, description="test")
    assert c.source_input is None


def test_concept_source_input_preserved():
    emb = torch.randn(1, 1024)
    source = torch.randn(1, 3, 384, 384)
    c = Concept(embedding=emb, description="test", source_input=source)
    assert c.source_input is not None
    assert c.source_input.shape == (1, 3, 384, 384)


def test_concept_arithmetic_drops_source_input():
    emb1 = torch.randn(1, 1024)
    emb2 = torch.randn(1, 1024)
    c1 = Concept(embedding=emb1, description="a", source_input=torch.randn(1, 3, 384, 384))
    c2 = Concept(embedding=emb2, description="b")
    result = c1 + c2
    assert result.source_input is None


def test_concept_to_moves_source_input():
    emb = torch.randn(1, 1024)
    source = torch.randn(1, 3, 384, 384)
    c = Concept(embedding=emb, description="test", source_input=source)
    moved = c.to("cpu")
    assert moved.source_input is not None


def test_concept_save_load_ignores_source_input(tmp_path):
    emb = torch.randn(1, 1024)
    source = torch.randn(1, 3, 384, 384)
    c = Concept(embedding=emb, description="test", source_input=source)
    path = tmp_path / "concept.pt"
    c.save(path)
    loaded = Concept.load(path)
    assert loaded.source_input is None  # Ephemeral, not serialized
```

- [ ] **Step 2: Write tests for LossConfig**

```python
# Add to tests/test_config.py or create tests/test_loss_config.py
from embedding_art.core.config import OptimizationConfig, LossConfig


def test_loss_config_defaults():
    lc = LossConfig()
    assert lc.similarity_weight == 1.0
    assert lc.feature_matching_weight == 0.5
    assert lc.feature_matching_layers == "every_4th"
    assert lc.sae_feature_weight == 0.0
    assert lc.sae_target_features is None


def test_optimization_config_has_loss():
    oc = OptimizationConfig()
    assert isinstance(oc.loss, LossConfig)


def test_optimization_config_backward_compat():
    """Old construction without loss field still works."""
    oc = OptimizationConfig(steps=100, learning_rate=0.05)
    assert oc.steps == 100
    assert oc.loss.similarity_weight == 1.0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_concept_source_input.py tests/test_loss_config.py -v`
Expected: FAIL

- [ ] **Step 4: Add source_input to Concept**

In `src/embedding_art/core/concept.py`, add the field and update `to()`, `__add__`, `__sub__`, `slerp`, `combine` to set `source_input=None` on results. Update `save()`/`load()` to skip it. Keep `text_source` for backward compat.

- [ ] **Step 5: Add LossConfig to config.py**

Append to `src/embedding_art/core/config.py`:

```python
@dataclass
class LossConfig:
    """Configuration for the composite loss function."""

    # Final embedding similarity (cosine sim)
    similarity_weight: float = 1.0

    # Multi-layer feature matching (MIMIC-inspired)
    feature_matching_weight: float = 0.5
    feature_matching_layers: list[int] | str = "every_4th"

    # SAE feature-space loss
    sae_feature_weight: float = 0.0
    sae_target_features: dict[str, float] | None = None
```

Add `loss: LossConfig = field(default_factory=LossConfig)` to `OptimizationConfig`.

- [ ] **Step 6: Run ALL tests**

Run: `pytest tests/ -x --timeout=30 -m "not slow and not integration"`
Expected: All PASS (old and new)

- [ ] **Step 7: Commit**

```bash
git add src/embedding_art/core/concept.py src/embedding_art/core/config.py tests/test_concept_source_input.py tests/test_loss_config.py
git commit -m "feat: add source_input to Concept, LossConfig to OptimizationConfig"
```

---

## Phase 2: Loss Architecture and Rendering Strategies

### Task 7: RenderResult and OptimizationHistory

**Files:**
- Create: `src/embedding_art/core/render_result.py`
- Test: `tests/test_render_result.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_render_result.py
import torch
from embedding_art.core.render_result import RenderResult, OptimizationHistory, LossBreakdown


def test_loss_breakdown():
    components = {
        "similarity": torch.tensor(-0.85),
        "feature_matching": torch.tensor(0.12),
    }
    lb = LossBreakdown(total=torch.tensor(-0.73), components=components)
    assert lb.total.item() == pytest.approx(-0.73)
    assert "similarity" in lb.components


def test_optimization_history_record():
    history = OptimizationHistory()
    lb = LossBreakdown(
        total=torch.tensor(0.5),
        components={"similarity": torch.tensor(-0.8)},
    )
    history.record(0, lb)
    assert len(history.steps) == 1
    assert history.final_similarity == pytest.approx(0.8)


def test_optimization_history_empty():
    history = OptimizationHistory()
    assert history.final_similarity == 0.0


def test_render_result():
    output = torch.randn(1, 3, 512, 512)
    history = OptimizationHistory()
    result = RenderResult(
        output=output,
        history=history,
        encoder_name="test",
        final_similarity=0.85,
    )
    assert result.encoder_name == "test"
    assert result.output.shape == (1, 3, 512, 512)
```

- [ ] **Step 2: Run to verify fail, implement, run to verify pass, commit**

Implementation is the `RenderResult`, `OptimizationHistory`, and `LossBreakdown` dataclasses from the spec.

```bash
git commit -m "feat: add RenderResult, OptimizationHistory, LossBreakdown"
```

---

### Task 8: CompositeLoss

**Files:**
- Create: `src/embedding_art/core/loss.py`
- Test: `tests/test_composite_loss.py`

This is the most important new component. Test thoroughly.

- [ ] **Step 1: Write tests**

```python
# tests/test_composite_loss.py
import torch
import pytest

from embedding_art.core.config import LossConfig
from embedding_art.core.loss import CompositeLoss
from embedding_art.core.concept import Concept


def test_similarity_only_loss(mock_encoder):
    """Default loss with only cosine similarity."""
    config = LossConfig(feature_matching_weight=0.0)
    loss_fn = CompositeLoss(config, mock_encoder)

    target = Concept(embedding=torch.randn(1, 1024), description="test")
    current_output = torch.randn(1, 3, 224, 224)

    breakdown = loss_fn(current_output, target, mock_encoder)
    assert "similarity" in breakdown.components
    assert "feature_matching" not in breakdown.components
    assert breakdown.total.requires_grad is False  # mock encoder doesn't enable grad


def test_loss_components_tracked():
    """All active loss components appear in breakdown."""
    config = LossConfig(similarity_weight=1.0, feature_matching_weight=0.0)
    # Use a simple mock that returns detached tensors
    from tests.conftest import MockEncoder
    encoder = MockEncoder()
    loss_fn = CompositeLoss(config, encoder)

    target = Concept(embedding=torch.randn(1, 1024), description="test")
    output = torch.randn(1, 3, 224, 224)

    breakdown = loss_fn(output, target, encoder)
    assert isinstance(breakdown.total, torch.Tensor)
    assert all(isinstance(v, torch.Tensor) for v in breakdown.components.values())


def test_calibrate_skips_without_capability(mock_encoder):
    """calibrate() gracefully skips when encoder lacks MULTI_LAYER_FEATURES."""
    config = LossConfig(feature_matching_weight=0.5)
    loss_fn = CompositeLoss(config, mock_encoder)

    target = Concept(embedding=torch.randn(1, 1024), description="test")
    loss_fn.calibrate(target, mock_encoder)
    assert loss_fn.reference_stats is None  # Skipped, no crash
```

- [ ] **Step 2-5: Red-green-refactor cycle**

Implement `CompositeLoss` in `src/embedding_art/core/loss.py` per the spec. Key behaviors:
- `calibrate()` checks `source_input is not None` AND `MULTI_LAYER_FEATURES` capability
- `__call__()` computes enabled loss components and returns `LossBreakdown`
- Feature matching gracefully disabled when uncalibrated
- SAE loss gracefully disabled when no SAE loaded

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: add CompositeLoss with multi-layer feature matching and SAE support"
```

---

### Task 9: Rendering Strategies

**Files:**
- Create: `src/embedding_art/core/strategies.py`
- Test: `tests/test_strategies.py`

- [ ] **Step 1: Write tests for OptimizationStrategy**

Test the full optimization loop with mock encoder and generator: init latent, run N steps, verify loss decreases, verify RenderResult is returned.

- [ ] **Step 2-5: Implement, test, commit**

Implement `RenderingStrategy` protocol, `OptimizationStrategy`, and `DiffusionGuidanceStrategy` stub.

```bash
git commit -m "feat: add RenderingStrategy protocol and OptimizationStrategy"
```

---

### Task 10: Extend Engine with render() and render_compare()

**Files:**
- Modify: `src/embedding_art/core/engine.py`
- Test: `tests/test_engine_v2.py` (new)

- [ ] **Step 1: Write tests**

Test `from_registry()`, `render()`, `render_compare()`. Verify old `optimize()` still works unchanged.

- [ ] **Step 2-5: Implement**

Add `from_registry()` classmethod, `render()`, `render_compare()`, `render_interpolation()` to `EmbeddingArtEngine`. The old `optimize()` method is preserved and delegates to `render()` internally.

- [ ] **Step 6: Run ALL tests**

Run: `pytest tests/ -x --timeout=30 -m "not slow and not integration"`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git commit -m "feat: add render(), render_compare() to engine with registry support"
```

---

## Phase 3: New Encoders

### Task 11: SigLIP 2 Encoder

**Files:**
- Create: `src/embedding_art/encoders/siglip2.py`
- Test: `tests/test_siglip2_encoder.py`

- [ ] **Step 1: Write tests**

Test card properties, encode_text, encode_image, encode_for_optimization (differentiable), get_layer_features (returns dict of LayerFeatures with correct shapes), unload.

Mark with `@pytest.mark.slow` since it downloads model weights.

- [ ] **Step 2-5: Implement**

Use `transformers.AutoModel.from_pretrained("google/siglip2-so400m-patch14-384")` with `output_hidden_states=True` for layer features. Implement `encode_for_optimization` as a differentiable path.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: add SigLIP2Encoder with multi-layer feature extraction"
```

---

### Task 12: CLAP Encoder

**Files:**
- Create: `src/embedding_art/encoders/clap.py`
- Test: `tests/test_clap_encoder.py`

- [ ] **Step 1-6: Same pattern as Task 11**

Use `laion/larger_clap_general` from HuggingFace. Support `encode_text`, `encode_audio`, `encode_for_optimization`.

```bash
git commit -m "feat: add CLAPEncoder for audio-language embedding"
```

---

### Task 13: Register Default Encoders

**Files:**
- Create: `src/embedding_art/encoders/defaults.py`
- Modify: `src/embedding_art/encoders/__init__.py`

- [ ] **Step 1: Create default registry factory**

```python
# src/embedding_art/encoders/defaults.py
def create_default_registry() -> EncoderRegistry:
    """Create registry with all built-in encoders."""
    registry = EncoderRegistry()
    from embedding_art.encoders.imagebind import ImageBindEncoder
    from embedding_art.encoders.siglip2 import SigLIP2Encoder
    from embedding_art.encoders.clap import CLAPEncoder
    registry.register("imagebind", ImageBindEncoder)
    registry.register("siglip2-so400m", SigLIP2Encoder)
    registry.register("clap-general", CLAPEncoder)
    return registry
```

- [ ] **Step 2: Update __init__.py exports, commit**

```bash
git commit -m "feat: add default encoder registry with ImageBind, SigLIP 2, CLAP"
```

---

## Phase 4: SAE Integration

### Task 14: SAELens and SAEDecomposition

**Files:**
- Create: `src/embedding_art/sae/__init__.py`
- Create: `src/embedding_art/sae/lens.py`
- Test: `tests/test_sae_lens.py`

- [ ] **Step 1: Write tests**

Test decompose (produces sparse activations), reconstruct (round-trip), manipulate (adjust features), feature vocabulary lookup, FeatureNotFoundError on bad names.

- [ ] **Step 2-5: Implement from spec**

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: add SAELens for interpretable embedding decomposition"
```

---

### Task 15: SAE Training Pipeline

**Files:**
- Create: `src/embedding_art/sae/training.py`
- Test: `tests/test_sae_training.py`

- [ ] **Step 1: Write tests**

Test embedding collection, TopK SAE forward pass, group-sparse loss computation, training loop convergence on synthetic data.

- [ ] **Step 2-5: Implement**

Group-sparse SAE training adapted from the ICLR 2026 paper (arxiv 2601.20028). Key components: TopK sparsity, group-sparse L_{2,1} regularization, cross-modal random masking, paired training loop.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: add group-sparse SAE training pipeline"
```

---

### Task 16: Concept.decompose() and Concept.from_features()

**Files:**
- Modify: `src/embedding_art/core/concept.py`
- Test: `tests/test_concept_sae.py` (new)

- [ ] **Step 1-5: Add methods to Concept, test, commit**

```bash
git commit -m "feat: add Concept.decompose() and Concept.from_features() for SAE integration"
```

---

## Phase 5: CLI and Integration

### Task 17: CLI render command

**Files:**
- Create: `src/embedding_art/cli/commands/render.py`
- Modify: `src/embedding_art/cli/main.py`
- Test: `tests/test_render_command.py`

- [ ] **Step 1-5: Implement render command**

Supports `--encoder`, `--features`, `--sae`, `--loss-weights`. Falls back to defaults. Uses new engine `render()` method.

```bash
git commit -m "feat: add embed-art render command with encoder selection"
```

---

### Task 18: CLI compare command

**Files:**
- Create: `src/embedding_art/cli/commands/compare.py`
- Test: `tests/test_compare_command.py`

- [ ] **Step 1-5: Implement compare command**

`--encoders` flag, runs `render_compare()`, outputs side-by-side results.

```bash
git commit -m "feat: add embed-art compare command for multi-encoder comparison"
```

---

### Task 19: CLI decompose and SAE commands

**Files:**
- Create: `src/embedding_art/cli/commands/decompose.py`
- Create: `src/embedding_art/cli/commands/sae.py`

- [ ] **Step 1-5: Implement, test, commit**

```bash
git commit -m "feat: add embed-art decompose and sae CLI commands"
```

---

### Task 20: Update public API exports and final integration test

**Files:**
- Modify: `src/embedding_art/__init__.py`
- Modify: `src/embedding_art/core/__init__.py`
- Test: `tests/test_integration_v2.py` (new)

- [ ] **Step 1: Update exports**

Add `ConceptSpec`, `EncoderRegistry`, `RenderResult`, `LossConfig`, `SAELens` to public API.

- [ ] **Step 2: Write integration test**

Full end-to-end: create registry, load mock encoder, render with CompositeLoss, verify result. Also test render_compare with two mock encoders of different dimensions.

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -x --timeout=60 -m "not slow and not integration"`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git commit -m "feat: update public API exports and add v2 integration tests"
```

---

## Review Fixes Applied

Issues from plan review:

1. **Task 5/6 parallel conflict resolved:** Task 6 now depends on Task 5 (both modify concept.py)
2. **Task 5 added to batch listing:** Now in Batch 3
3. **Generator protocol updates:** Task 9 scope expanded to include updating generators/base.py with LatentGenerator, DirectGenerator, DiffusionGenerator protocols
4. **CLI tasks moved earlier:** Tasks 17/18 (render/compare CLI) moved to Phase 3 (after engine v2, before SAE). Only Task 19 (SAE CLI) stays in Phase 4.
5. **pyproject.toml updates:** Tasks 11/12 include dependency updates
6. **can_fit safety margin documented** in docstring and tested explicitly
7. **normalize_gradients bug:** Noted in Task 9 for fixing during DiffusionGuidanceStrategy
8. **Missing pytest import in test snippets:** Fixed
9. **INR Generator deferred:** Noted as future work (not blocking v2 launch)
10. **Deprecation warnings deferred:** Noted as future work (Phase 5 of spec)
11. **from_spec() removed from file structure:** encode(spec) on encoders covers this
12. **OptimizationConfig defaults preserved:** Task 6 note added

## Task Dependency Graph

```
Phase 1 (core abstractions, non-breaking):
  Task 1 (exceptions)
  Task 2 (ConceptSpec)
  Task 3 (registry) ← depends on Task 1
  Task 4 (features)
  Task 5 (extend protocol) ← depends on Tasks 2, 3, 4
  Task 6 (concept + config) ← depends on Tasks 4, 5

Phase 2 (loss and engine):
  Task 7 (RenderResult)
  Task 8 (CompositeLoss) ← depends on Tasks 4, 6, 7
  Task 9 (strategies + generator protocols) ← depends on Tasks 7, 8
  Task 10 (engine v2) ← depends on Tasks 3, 5, 8, 9

Phase 3 (encoders + CLI, parallel):
  Task 11 (SigLIP 2) ← depends on Tasks 3, 4, 5
  Task 12 (CLAP) ← depends on Tasks 3, 5
  Task 13 (defaults) ← depends on Tasks 11, 12
  Task 17 (CLI render) ← depends on Task 10
  Task 18 (CLI compare) ← depends on Tasks 10, 13

Phase 4 (SAE):
  Task 14 (SAELens) ← depends on Task 8
  Task 15 (SAE training) ← depends on Task 14
  Task 16 (Concept SAE) ← depends on Task 14
  Task 19 (CLI SAE) ← depends on Tasks 14, 15

Phase 5 (integration):
  Task 20 (exports + integration) ← depends on all

Deferred (future work):
  - INR/DirectGenerator implementation (generators/inr.py)
  - Deprecation warnings on old API (Phase 5 of spec)
  - Web UI updates for new engine API
```

## Parallelization Opportunities

Tasks that can run in parallel worktrees:

- **Batch 1:** Tasks 1, 2, 4 (no dependencies between them)
- **Batch 2:** Tasks 3, 5 (3 depends on 1; 5 depends on 2, 3, 4)
- **Batch 3:** Tasks 6, 7 (6 depends on 4, 5; 7 independent — different files)
- **Batch 4:** Task 8 (depends on 4, 6, 7)
- **Batch 5:** Tasks 9, 11, 12 (9 depends on 8; 11/12 depend on Phase 1 — all touch different files)
- **Batch 6:** Tasks 10, 13, 14 (10 depends on 9; 13 depends on 11+12; 14 depends on 8)
- **Batch 7:** Tasks 15, 16, 17, 18 (all depend on prior batches, touch different files)
- **Batch 8:** Tasks 19, 20 (19 needs SAE; 20 needs everything)
