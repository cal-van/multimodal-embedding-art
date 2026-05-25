"""
Configuration dataclasses for optimization runs.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

DEFAULT_CONFIG_FILENAME = "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        path: Path to the YAML config file. If None, looks for config.yaml
              in the current working directory.

    Returns:
        Dictionary containing the configuration. Returns empty dict if
        file doesn't exist or is empty.
    """
    if path is None:
        path = Path.cwd() / DEFAULT_CONFIG_FILENAME
    else:
        path = Path(path)

    if not path.exists():
        return {}

    with open(path) as f:
        content = yaml.safe_load(f)

    if content is None:
        return {}

    return content


@dataclass
class LossConfig:
    """Configuration for the composite loss function."""

    similarity_weight: float = 1.0
    feature_matching_weight: float = 0.5
    feature_matching_layers: list[int] | str = "every_4th"
    sae_feature_weight: float = 0.0
    sae_target_features: dict[str, float] | None = None
    # Text-anchor auxiliary loss (M4): when > 0 and either text_anchor_text
    # is set or the target concept carries a string source, the loss adds a
    # cosine-distance term pulling the current optimisation embedding toward
    # the encoder's text projection of that string. Exploits LanguageBind's
    # text-as-anchor design so that even image / audio / video targets get a
    # language-flavoured supervision signal.
    text_anchor_weight: float = 0.0
    text_anchor_text: str | None = None
    # Optional regularizer; typed as Any to avoid a circular import with
    # embedding_art.regularizers.  Pass a CompositeRegularizer (or any callable
    # matching the Regularizer protocol) here if regularization is desired.
    regularization: Any = None


@dataclass
class AugmentationConfig:
    """Configuration for augmentation during optimization."""

    random_crop: bool = True
    crop_scale: tuple[float, float] = (0.8, 1.0)
    random_flip: bool = False
    color_jitter: bool = False


@dataclass
class OptimizationConfig:
    """Configuration for an optimization run."""

    # Core parameters
    steps: int = 2000
    learning_rate: float = 0.1
    optimizer: Literal["adam", "adamw", "sgd"] = "adam"
    scheduler: Literal["constant", "cosine", "linear"] = "cosine"

    # Augmentation during optimization
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    # Checkpointing
    checkpoint_every: int = 100
    preview_every: int = 50

    # Reproducibility
    seed: int | None = None

    # Guidance
    guidance_scale: float = 0.0  # ImageBind guidance scale
    normalize_gradients: bool = False  # Whether to normalize gradients for stability

    # Mixed-precision: wraps the forward+loss section of each step in
    # ``torch.autocast(device_type=device, dtype=...)``.  Gradients are still
    # accumulated in fp32 by torch's autocast machinery, so stability is
    # preserved.  Defaults to ``"fp32"`` (no autocast) for safety; on M1 Max
    # ``"bf16"`` is the recommended setting and roughly halves wall-time.
    autocast_dtype: Literal["fp32", "fp16", "bf16"] = "fp32"

    # Apple Silicon perf knobs.  ``compile_mode`` wraps the encoder + loss
    # path in ``torch.compile`` when not ``"none"``.  ``"reduce-overhead"``
    # is the recommended setting on PyTorch 2.5+ MPS for a 1.5-2.5x speedup
    # on the optimisation hot loop; ``"default"`` is the safe baseline;
    # ``"max-autotune"`` is aggressive and brittle, use only after profiling.
    compile_mode: Literal["none", "default", "reduce-overhead", "max-autotune"] = "none"

    # ``empty_mps_cache_between_modalities`` calls ``torch.mps.empty_cache``
    # between the four showcase modalities to recover the MPS allocator's
    # working set.  Free win on M1 Max, no-op on non-MPS backends.
    empty_mps_cache_between_modalities: bool = True

    # Loss function
    loss: LossConfig = field(default_factory=LossConfig)

    def __post_init__(self):
        if isinstance(self.augmentation, dict):
            self.augmentation = AugmentationConfig(**self.augmentation)
