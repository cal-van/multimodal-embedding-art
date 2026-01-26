"""
Configuration dataclasses for optimization runs.
"""

from dataclasses import dataclass, field
from typing import Literal


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

    def __post_init__(self):
        if isinstance(self.augmentation, dict):
            self.augmentation = AugmentationConfig(**self.augmentation)
