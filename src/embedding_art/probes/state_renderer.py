from __future__ import annotations

import math
from typing import Any

import torch

from embedding_art.core.render_result import RenderResult
from embedding_art.probes.activation_probe import ModelState


class StateRenderer:
    """Renders captured ModelState through DirectRenderers."""

    def render_layer(self, state: ModelState, layer_idx: int, renderer: Any) -> RenderResult:
        """Render a single layer's activation."""
        activation = state.layer_activations[layer_idx]
        return renderer.render(activation)

    def render_progression(self, state: ModelState, renderer: Any) -> list[RenderResult]:
        """Render all captured layers as a sequence."""
        results = []
        for layer_idx in sorted(state.layer_activations.keys()):
            results.append(self.render_layer(state, layer_idx, renderer))
        return results

    def render_comparison(
        self, states: list[ModelState], layer_idx: int, renderer: Any
    ) -> list[RenderResult]:
        """Render the same layer from multiple encoder states for comparison."""
        results = []
        for state in states:
            results.append(self.render_layer(state, layer_idx, renderer))
        return results

    def compose_grid(self, results: list[RenderResult], grid_cols: int = 4) -> torch.Tensor:
        """Compose multiple render results into a grid image."""
        if not results:
            return torch.empty(0)
        outputs = [r.output for r in results]
        target_shape = outputs[0].shape
        n = len(outputs)
        rows = math.ceil(n / grid_cols)
        cols = min(n, grid_cols)
        c, h, w = target_shape[1], target_shape[2], target_shape[3]
        grid = torch.zeros(1, c, rows * h, cols * w, device=outputs[0].device)
        for i, out in enumerate(outputs):
            row = i // grid_cols
            col = i % grid_cols
            grid[:, :, row * h : (row + 1) * h, col * w : (col + 1) * w] = out
        return grid
