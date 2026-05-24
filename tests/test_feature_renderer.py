"""Tests for FeatureRenderer: rendering individual SAE features."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from embedding_art.core.render_result import OptimizationHistory, RenderResult
from embedding_art.exceptions import FeatureNotFoundError
from embedding_art.sae.feature_renderer import FeatureRenderer
from embedding_art.sae.lens import SAEDecomposition, SAELens


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_sae() -> SAELens:
    """Create a small SAELens for testing."""
    n_features = 8
    embed_dim = 16
    torch.manual_seed(0)
    W_enc = torch.randn(n_features, embed_dim)
    W_dec = torch.randn(embed_dim, n_features)
    bias = torch.zeros(n_features)
    pre_bias = torch.zeros(embed_dim)
    vocab = [f"feature_{i}" for i in range(n_features)]
    return SAELens.from_tensors(W_enc, W_dec, bias, pre_bias, vocab, k=3)


class MockDirectRenderer:
    """Mock renderer that records calls and returns dummy RenderResults."""

    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []
        self.output_modality = "image"

    def render(self, embedding: torch.Tensor, **kwargs: object) -> RenderResult:
        self.calls.append(embedding.clone())
        return RenderResult(
            output=torch.rand(1, 3, 32, 32),
            history=OptimizationHistory(),
            encoder_name="mock",
            final_similarity=0.0,
        )


@pytest.fixture
def mock_renderer() -> MockDirectRenderer:
    return MockDirectRenderer()


@pytest.fixture
def feature_renderer() -> FeatureRenderer:
    return FeatureRenderer()


# ---------------------------------------------------------------------------
# feature_direction tests
# ---------------------------------------------------------------------------


class TestFeatureDirection:
    def test_feature_direction_shape(
        self, feature_renderer: FeatureRenderer, test_sae: SAELens
    ) -> None:
        """feature_direction returns [1, embed_dim]."""
        direction = feature_renderer.feature_direction(test_sae, 0)
        assert direction.shape == (1, test_sae.embed_dim)

    def test_feature_direction_normalized(
        self, feature_renderer: FeatureRenderer, test_sae: SAELens
    ) -> None:
        """Result has unit norm."""
        direction = feature_renderer.feature_direction(test_sae, 3)
        norm = torch.norm(direction, dim=-1)
        assert torch.allclose(norm, torch.tensor([1.0]), atol=1e-6)

    def test_feature_direction_matches_decoder(
        self, feature_renderer: FeatureRenderer, test_sae: SAELens
    ) -> None:
        """Direction matches the normalized W_dec column."""
        idx = 5
        direction = feature_renderer.feature_direction(test_sae, idx)
        expected_col = test_sae._W_dec[:, idx]
        expected = F.normalize(expected_col.unsqueeze(0), dim=-1)
        assert torch.allclose(direction, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# render_feature tests
# ---------------------------------------------------------------------------


class TestRenderFeature:
    def test_render_feature_calls_renderer(
        self,
        feature_renderer: FeatureRenderer,
        test_sae: SAELens,
        mock_renderer: MockDirectRenderer,
    ) -> None:
        """renderer.render() is called with the correct embedding."""
        result = feature_renderer.render_feature(test_sae, 2, mock_renderer)
        assert len(mock_renderer.calls) == 1
        assert isinstance(result, RenderResult)

    def test_render_feature_scaled_by_activation(
        self,
        feature_renderer: FeatureRenderer,
        test_sae: SAELens,
        mock_renderer: MockDirectRenderer,
    ) -> None:
        """Embedding passed to renderer is scaled by the activation value."""
        activation = 3.5
        feature_renderer.render_feature(test_sae, 1, mock_renderer, activation=activation)
        direction = feature_renderer.feature_direction(test_sae, 1)
        expected = direction * activation
        assert torch.allclose(mock_renderer.calls[0], expected, atol=1e-6)

    def test_render_feature_default_activation(
        self,
        feature_renderer: FeatureRenderer,
        test_sae: SAELens,
        mock_renderer: MockDirectRenderer,
    ) -> None:
        """Default activation is 1.0 (unscaled direction)."""
        feature_renderer.render_feature(test_sae, 4, mock_renderer)
        direction = feature_renderer.feature_direction(test_sae, 4)
        # With activation=1.0, the passed embedding should equal the direction
        assert torch.allclose(mock_renderer.calls[0], direction, atol=1e-6)


# ---------------------------------------------------------------------------
# render_feature_by_name tests
# ---------------------------------------------------------------------------


class TestRenderFeatureByName:
    def test_render_feature_by_name(
        self,
        feature_renderer: FeatureRenderer,
        test_sae: SAELens,
        mock_renderer: MockDirectRenderer,
    ) -> None:
        """Finds a feature by name and renders it."""
        result = feature_renderer.render_feature_by_name(
            test_sae, "feature_3", mock_renderer, activation=2.0
        )
        assert isinstance(result, RenderResult)
        assert len(mock_renderer.calls) == 1
        # The embedding should match feature_3's direction scaled by 2.0
        direction = feature_renderer.feature_direction(test_sae, 3)
        expected = direction * 2.0
        assert torch.allclose(mock_renderer.calls[0], expected, atol=1e-6)

    def test_render_feature_by_name_not_found(
        self,
        feature_renderer: FeatureRenderer,
        test_sae: SAELens,
        mock_renderer: MockDirectRenderer,
    ) -> None:
        """Raises FeatureNotFoundError for unknown feature names."""
        with pytest.raises(FeatureNotFoundError):
            feature_renderer.render_feature_by_name(
                test_sae, "nonexistent_feature", mock_renderer
            )


# ---------------------------------------------------------------------------
# render_decomposition tests
# ---------------------------------------------------------------------------


class TestRenderDecomposition:
    def _make_decomposition(self, test_sae: SAELens) -> SAEDecomposition:
        """Create a decomposition with known active features."""
        return SAEDecomposition.from_dict(
            {
                "feature_0": 5.0,
                "feature_1": 3.0,
                "feature_2": 1.0,
                "feature_3": 4.0,
                "feature_4": 2.0,
            },
            test_sae,
        )

    def test_render_decomposition_returns_dict(
        self,
        feature_renderer: FeatureRenderer,
        test_sae: SAELens,
        mock_renderer: MockDirectRenderer,
    ) -> None:
        """Returns dict of feature_name -> RenderResult."""
        decomp = self._make_decomposition(test_sae)
        results = feature_renderer.render_decomposition(decomp, test_sae, mock_renderer)
        assert isinstance(results, dict)
        for name, result in results.items():
            assert isinstance(name, str)
            assert isinstance(result, RenderResult)

    def test_render_decomposition_top_k(
        self,
        feature_renderer: FeatureRenderer,
        test_sae: SAELens,
        mock_renderer: MockDirectRenderer,
    ) -> None:
        """Only renders top max_features features."""
        decomp = self._make_decomposition(test_sae)
        results = feature_renderer.render_decomposition(
            decomp, test_sae, mock_renderer, max_features=3
        )
        assert len(results) == 3
        # Verify these are the top-3 by activation
        assert set(results.keys()) == {"feature_0", "feature_3", "feature_1"}

    def test_render_decomposition_sorted_by_activation(
        self,
        feature_renderer: FeatureRenderer,
        test_sae: SAELens,
        mock_renderer: MockDirectRenderer,
    ) -> None:
        """Most active features come first (dict insertion order)."""
        decomp = self._make_decomposition(test_sae)
        results = feature_renderer.render_decomposition(
            decomp, test_sae, mock_renderer, max_features=5
        )
        keys = list(results.keys())
        # feature_0=5.0, feature_3=4.0, feature_1=3.0, feature_4=2.0, feature_2=1.0
        assert keys == ["feature_0", "feature_3", "feature_1", "feature_4", "feature_2"]


# ---------------------------------------------------------------------------
# render_reconstruction tests
# ---------------------------------------------------------------------------


class TestRenderReconstruction:
    def test_render_reconstruction_uses_sae_reconstruct(
        self,
        feature_renderer: FeatureRenderer,
        test_sae: SAELens,
        mock_renderer: MockDirectRenderer,
    ) -> None:
        """Calls sae.reconstruct() and passes result to renderer."""
        decomp = SAEDecomposition.from_dict({"feature_0": 1.0, "feature_1": 2.0}, test_sae)
        result = feature_renderer.render_reconstruction(decomp, test_sae, mock_renderer)
        assert isinstance(result, RenderResult)
        assert len(mock_renderer.calls) == 1

    def test_render_reconstruction_normalizes(
        self,
        feature_renderer: FeatureRenderer,
        test_sae: SAELens,
        mock_renderer: MockDirectRenderer,
    ) -> None:
        """Output embedding passed to renderer is normalized."""
        decomp = SAEDecomposition.from_dict({"feature_0": 1.0, "feature_2": 3.0}, test_sae)
        feature_renderer.render_reconstruction(decomp, test_sae, mock_renderer)
        passed_embedding = mock_renderer.calls[0]
        norm = torch.norm(passed_embedding, dim=-1)
        assert torch.allclose(norm, torch.tensor([1.0]), atol=1e-6)


# ---------------------------------------------------------------------------
# render_feature_spectrum tests
# ---------------------------------------------------------------------------


class TestRenderFeatureSpectrum:
    def test_render_feature_spectrum_default_activations(
        self,
        feature_renderer: FeatureRenderer,
        test_sae: SAELens,
        mock_renderer: MockDirectRenderer,
    ) -> None:
        """Renders at 5 default activation levels."""
        results = feature_renderer.render_feature_spectrum(test_sae, 0, mock_renderer)
        assert len(results) == 5
        assert len(mock_renderer.calls) == 5

    def test_render_feature_spectrum_custom_activations(
        self,
        feature_renderer: FeatureRenderer,
        test_sae: SAELens,
        mock_renderer: MockDirectRenderer,
    ) -> None:
        """Respects custom activation list."""
        custom = [0.5, 1.5, 3.0]
        results = feature_renderer.render_feature_spectrum(
            test_sae, 2, mock_renderer, activations=custom
        )
        assert len(results) == 3
        # Verify each call was scaled correctly
        direction = feature_renderer.feature_direction(test_sae, 2)
        for i, act in enumerate(custom):
            expected = direction * act
            assert torch.allclose(mock_renderer.calls[i], expected, atol=1e-6)

    def test_render_feature_spectrum_length(
        self,
        feature_renderer: FeatureRenderer,
        test_sae: SAELens,
        mock_renderer: MockDirectRenderer,
    ) -> None:
        """Returns correct number of results for custom activation list."""
        activations = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        results = feature_renderer.render_feature_spectrum(
            test_sae, 1, mock_renderer, activations=activations
        )
        assert len(results) == 7


# ---------------------------------------------------------------------------
# compose_feature_grid tests
# ---------------------------------------------------------------------------


class TestComposeFeatureGrid:
    def test_compose_feature_grid_shape(
        self,
        feature_renderer: FeatureRenderer,
        test_sae: SAELens,
        mock_renderer: MockDirectRenderer,
    ) -> None:
        """Grid has correct dimensions for given renders and grid_cols."""
        decomp = SAEDecomposition.from_dict(
            {"feature_0": 5.0, "feature_1": 3.0, "feature_2": 1.0, "feature_3": 4.0},
            test_sae,
        )
        renders = feature_renderer.render_decomposition(decomp, test_sae, mock_renderer)
        grid = feature_renderer.compose_feature_grid(renders, grid_cols=2)
        # 4 features, 2 cols -> 2 rows. Each cell is 32x32, channels=3
        assert grid.shape == (1, 3, 2 * 32, 2 * 32)

    def test_compose_feature_grid_single_feature(
        self, feature_renderer: FeatureRenderer, mock_renderer: MockDirectRenderer
    ) -> None:
        """Works with one feature."""
        single_render = {
            "only_feature": RenderResult(
                output=torch.rand(1, 3, 32, 32),
                history=OptimizationHistory(),
                encoder_name="mock",
                final_similarity=0.0,
            )
        }
        grid = feature_renderer.compose_feature_grid(single_render, grid_cols=4)
        # 1 feature, 4 cols -> 1 row
        assert grid.shape == (1, 3, 32, 4 * 32)
