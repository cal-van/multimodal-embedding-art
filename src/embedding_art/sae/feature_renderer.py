"""Renders individual SAE features as visual or audio outputs.

Each SAE feature corresponds to a direction in embedding space (a column
of the decoder weight matrix). This module renders those directions through
a renderer to visualize what individual features "look like."
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from embedding_art.core.render_result import RenderResult
from embedding_art.exceptions import FeatureNotFoundError
from embedding_art.sae.lens import SAEDecomposition, SAELens


class FeatureRenderer:
    """Renders individual SAE features as visual or audio outputs.

    Each SAE feature corresponds to a direction in embedding space (a column
    of the decoder weight matrix). This class renders those directions through
    a renderer to visualize what individual features "look like."
    """

    def feature_direction(self, sae: SAELens, feature_idx: int) -> torch.Tensor:
        """Extract the normalized embedding direction for a single feature.

        Returns a [1, embed_dim] tensor -- the decoder column for this feature,
        normalized to unit length.
        """
        direction = sae.feature_direction(feature_idx)
        return F.normalize(direction, dim=-1)

    def render_feature(
        self,
        sae: SAELens,
        feature_idx: int,
        renderer: Any,
        activation: float = 1.0,
    ) -> RenderResult:
        """Render a single feature at a given activation strength.

        Extracts the feature's direction from the SAE decoder, scales it by
        the activation strength, and passes it through the renderer.
        """
        direction = self.feature_direction(sae, feature_idx)
        scaled = direction * activation
        return renderer.render(scaled)

    def render_feature_by_name(
        self,
        sae: SAELens,
        feature_name: str,
        renderer: Any,
        activation: float = 1.0,
    ) -> RenderResult:
        """Render a named feature. Raises FeatureNotFoundError if not in vocab."""
        feature_idx = sae.feature_index(feature_name)
        return self.render_feature(sae, feature_idx, renderer, activation=activation)

    def render_decomposition(
        self,
        decomposition: SAEDecomposition,
        sae: SAELens,
        renderer: Any,
        max_features: int = 10,
    ) -> dict[str, RenderResult]:
        """Render the top active features from a decomposition.

        Returns a dict mapping feature_name -> RenderResult for the
        top max_features most active features in the decomposition.
        """
        # Sort active_features by activation strength (descending)
        sorted_features = sorted(
            decomposition.active_features.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        top_features = sorted_features[:max_features]

        results: dict[str, RenderResult] = {}
        for name, act_strength in top_features:
            results[name] = self.render_feature_by_name(
                sae, name, renderer, activation=act_strength
            )
        return results

    def render_reconstruction(
        self,
        decomposition: SAEDecomposition,
        sae: SAELens,
        renderer: Any,
    ) -> RenderResult:
        """Render the SAE reconstruction of a decomposition.

        Reconstructs the embedding from the sparse features and renders it.
        This shows what the SAE "thinks" the concept looks like after
        lossy compression to sparse features.
        """
        reconstructed = sae.reconstruct(decomposition)
        return renderer.render(F.normalize(reconstructed, dim=-1))

    def render_feature_spectrum(
        self,
        sae: SAELens,
        feature_idx: int,
        renderer: Any,
        activations: list[float] | None = None,
    ) -> list[RenderResult]:
        """Render a single feature at multiple activation strengths.

        Useful for understanding how a feature's visual representation
        changes as its activation increases. Default activations: [0.1, 0.5, 1.0, 2.0, 5.0]
        """
        if activations is None:
            activations = [0.1, 0.5, 1.0, 2.0, 5.0]
        results: list[RenderResult] = []
        for act in activations:
            results.append(self.render_feature(sae, feature_idx, renderer, activation=act))
        return results

    def compose_feature_grid(
        self,
        renders: dict[str, RenderResult],
        grid_cols: int = 4,
    ) -> torch.Tensor:
        """Compose named feature renders into a labeled grid image.

        Arranges rendered features in a grid. Each cell shows the rendered
        feature output. Returns a single tensor combining all cells.
        """
        if not renders:
            raise ValueError("No renders provided to compose into a grid.")

        tensors = [result.output for result in renders.values()]

        # Determine the spatial size from the first tensor.
        # All tensors should have shape [1, C, H, W] or [C, H, W].
        ref = tensors[0]
        if ref.dim() == 4:
            _, channels, cell_h, cell_w = ref.shape
        else:
            channels, cell_h, cell_w = ref.shape

        n = len(tensors)
        grid_rows = (n + grid_cols - 1) // grid_cols

        # Create output grid
        grid = torch.zeros(1, channels, grid_rows * cell_h, grid_cols * cell_w)

        for idx, t in enumerate(tensors):
            # Ensure [C, H, W]
            cell = t.squeeze(0) if t.dim() == 4 else t
            row = idx // grid_cols
            col = idx % grid_cols
            y0 = row * cell_h
            x0 = col * cell_w
            grid[0, :, y0 : y0 + cell_h, x0 : x0 + cell_w] = cell

        return grid
