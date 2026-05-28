"""Base renderer protocol for direct embedding-to-output rendering."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch

from embedding_art.core.render_result import RenderResult


@runtime_checkable
class DirectRenderer(Protocol):
    """Renders an embedding directly to output without an optimization loop."""

    @property
    def output_modality(self) -> str: ...

    def render(self, embedding: torch.Tensor, **kwargs: Any) -> RenderResult: ...
