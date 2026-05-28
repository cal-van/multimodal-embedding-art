"""Raw linear projection from embedding space to pixel space."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from embedding_art.core.render_result import OptimizationHistory, RenderResult


class RawDecoder:
    """Linear projection from embedding space to pixel space.

    The most faithful (and likely least legible) view of what an embedding
    "looks like" -- a single learned linear map followed by sigmoid.
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        output_shape: tuple[int, ...] = (1, 3, 256, 256),
        device: str = "cpu",
    ) -> None:
        self._output_shape = output_shape
        self._device = torch.device(device)
        flat_size = math.prod(output_shape)
        self._projection = nn.Linear(embed_dim, flat_size, bias=True)
        nn.init.xavier_uniform_(self._projection.weight)
        self._projection = self._projection.to(self._device)

    @property
    def output_modality(self) -> str:
        return "image"

    def render(self, embedding: torch.Tensor, **kwargs: Any) -> RenderResult:
        embedding = embedding.to(self._device)
        if embedding.dim() == 2:
            if embedding.shape[0] != 1:
                raise ValueError(f"Expected single embedding, got batch of {embedding.shape[0]}")
            embedding = embedding.squeeze(0)
        flat = self._projection(embedding)
        output = torch.sigmoid(flat).reshape(self._output_shape)
        history = OptimizationHistory()
        return RenderResult(
            output=output,
            history=history,
            encoder_name="direct",
            final_similarity=0.0,
        )
