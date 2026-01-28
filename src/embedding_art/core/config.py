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
