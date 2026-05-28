"""Learned MLP projection from embedding space to generator latent space."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from embedding_art.core.render_result import OptimizationHistory, RenderResult


class ProjectionDecoder:
    """Learned MLP from embedding space to a generator's latent space.

    Projects through an MLP with GELU activations and LayerNorm, then
    uses the generator to decode the predicted latent.
    """

    def __init__(
        self,
        embed_dim: int,
        generator: Any,
        hidden_dims: list[int] | None = None,
        device: str = "cpu",
    ) -> None:
        self._generator = generator
        self._device = torch.device(device)
        self._latent_shape = generator.latent_shape
        flat_latent = math.prod(self._latent_shape)

        if hidden_dims is None:
            hidden_dims = [2048, 2048]

        layers: list[nn.Module] = []
        in_dim = embed_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, flat_latent))

        self._mlp = nn.Sequential(*layers).to(self._device)

    @property
    def output_modality(self) -> str:
        return self._generator.output_modality

    def render(self, embedding: torch.Tensor, **kwargs: Any) -> RenderResult:
        embedding = embedding.to(self._device)
        if embedding.dim() == 2:
            if embedding.shape[0] != 1:
                raise ValueError(f"Expected single embedding, got batch of {embedding.shape[0]}")
            embedding = embedding.squeeze(0)
        flat_latent = self._mlp(embedding)
        latent = flat_latent.reshape(self._latent_shape)
        output = self._generator.decode(latent)
        history = OptimizationHistory()
        return RenderResult(
            output=output,
            history=history,
            encoder_name="direct",
            final_similarity=0.0,
        )

    def save(self, path: Path) -> None:
        torch.save(self._mlp.state_dict(), path)

    def load(self, path: Path) -> None:
        state_dict = torch.load(path, map_location=self._device, weights_only=True)
        self._mlp.load_state_dict(state_dict)
