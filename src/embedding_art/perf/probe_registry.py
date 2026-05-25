"""
Registry of probe encoders that are CoreML-conversion candidates.

The cross-encoder probe path (re-encode rendered outputs with an
independent encoder and compare to the canonical embedding) is non-
differentiable. On Apple Silicon those forwards can run on the ANE via
CoreML for a 4-6x speedup. This module enumerates the probes we know
how to convert and exposes a uniform interface for the
``embed-art compile-probes`` / ``embed-art bench-probes`` commands.

Each entry produces:

* ``module``: a ``torch.nn.Module`` whose ``forward(x)`` takes the
  given example input shape and returns a fixed-shape embedding.
* ``example_input``: a dummy tensor in the expected shape.
* ``name``: stable identifier used as the ``.mlpackage`` filename.

The registry is intentionally small and explicit: each probe needs a
hand-written wrapper because each encoder exposes its inference path
slightly differently. Add new probes here as they're validated on real
M1/M2 Max hardware.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class ProbeSpec:
    """A registered probe encoder.

    Attributes:
        name: Stable id (used as the .mlpackage filename).
        modality: One of ``"image"`` / ``"audio"`` / ``"text"`` /
            ``"video"``. Drives expected input shape semantics.
        example_input_shape: Expected forward-pass input shape.
        loader: Returns ``(module, example_input)``. Called lazily so
            that loading a probe doesn't pay the cost of every probe.
    """

    name: str
    modality: str
    example_input_shape: tuple[int, ...]
    loader: Callable[[], tuple[torch.nn.Module, torch.Tensor]]


def _load_siglip2_image() -> tuple[torch.nn.Module, torch.Tensor]:
    """Wrap SigLIP 2's image encoder for tracing."""
    from embedding_art.encoders.siglip2 import SigLIP2Encoder

    encoder = SigLIP2Encoder(device="cpu")
    model: Any = encoder._model

    class _Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            features = self.model.get_image_features(pixel_values=pixel_values)
            return torch.nn.functional.normalize(features, dim=-1)

    wrapper = _Wrapper().eval()
    # SigLIP 2 SO400M expects 384x384.
    example = torch.randn(1, 3, 384, 384)
    return wrapper, example


def _load_clap_audio() -> tuple[torch.nn.Module, torch.Tensor]:
    """Wrap CLAP's audio encoder for tracing."""
    from embedding_art.encoders.clap import CLAPEncoder

    encoder = CLAPEncoder(device="cpu")
    model: Any = encoder._model

    class _Wrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model

        def forward(self, input_features: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            features = self.model.get_audio_features(input_features=input_features)
            return torch.nn.functional.normalize(features, dim=-1)

    wrapper = _Wrapper().eval()
    # CLAP HTSAT-base expects mel input ~ [B, 1001, 64].
    example = torch.randn(1, 1001, 64)
    return wrapper, example


REGISTRY: dict[str, ProbeSpec] = {
    "siglip2-image": ProbeSpec(
        name="siglip2-image",
        modality="image",
        example_input_shape=(1, 3, 384, 384),
        loader=_load_siglip2_image,
    ),
    "clap-audio": ProbeSpec(
        name="clap-audio",
        modality="audio",
        example_input_shape=(1, 1001, 64),
        loader=_load_clap_audio,
    ),
}


def available_probes() -> list[str]:
    """Names of probes registered for CoreML conversion."""
    return sorted(REGISTRY.keys())


def get_probe(name: str) -> ProbeSpec:
    """Look up a probe by name. Raises ``KeyError`` when missing."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown probe '{name}'. Available: {', '.join(available_probes())}")
    return REGISTRY[name]
