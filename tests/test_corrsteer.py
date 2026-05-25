"""Tests for ``embedding_art.interpretation.corrsteer``.

Uses synthetic SAE callables so the tests run in milliseconds without
requiring trained weights.
"""

from __future__ import annotations

import torch

from embedding_art.interpretation.corrsteer import (
    compute_corrsteer,
    pearson_correlation_matrix,
)


class TestPearsonCorrelationMatrix:
    """Pure-function tests on the inner Pearson kernel."""

    def test_perfect_positive_correlation(self) -> None:
        activations = torch.arange(10, dtype=torch.float32).unsqueeze(1)  # [10, 1]
        similarities = torch.arange(10, dtype=torch.float32)  # [10]
        corr = pearson_correlation_matrix(activations, similarities)
        assert corr.shape == (1,)
        assert torch.allclose(corr, torch.tensor([1.0]), atol=1e-5)

    def test_perfect_negative_correlation(self) -> None:
        activations = torch.arange(10, dtype=torch.float32).unsqueeze(1)
        similarities = -torch.arange(10, dtype=torch.float32)
        corr = pearson_correlation_matrix(activations, similarities)
        assert torch.allclose(corr, torch.tensor([-1.0]), atol=1e-5)

    def test_constant_column_yields_zero_not_nan(self) -> None:
        """A feature that never activates (constant) should be 0, not NaN."""
        activations = torch.zeros((10, 3))
        similarities = torch.randn(10)
        corr = pearson_correlation_matrix(activations, similarities)
        assert corr.shape == (3,)
        assert torch.all(corr == 0.0)
        assert not torch.any(torch.isnan(corr))

    def test_mixed_columns(self) -> None:
        """Two columns: one tracks similarity, one is anti-correlated."""
        similarities = torch.linspace(0.0, 1.0, 50)
        col_aligned = similarities  # cor = 1
        col_anti = -similarities + 0.5  # cor = -1
        activations = torch.stack([col_aligned, col_anti], dim=1)
        corr = pearson_correlation_matrix(activations, similarities)
        assert corr[0].item() == pytest_approx(1.0)
        assert corr[1].item() == pytest_approx(-1.0)

    def test_single_sample_returns_zeros(self) -> None:
        activations = torch.randn(1, 5)
        similarities = torch.randn(1)
        corr = pearson_correlation_matrix(activations, similarities)
        assert corr.shape == (5,)
        assert torch.all(corr == 0.0)


class TestComputeCorrSteer:
    """``compute_corrsteer`` wires the SAE forward + Pearson together."""

    def test_returns_top_k_features_by_abs_correlation(self) -> None:
        # Build a synthetic SAE: feature 0 = mean, feature 1 = sum,
        # feature 2 = noise. Embeddings change linearly with t so f0
        # tracks t, f1 also tracks t, f2 doesn't.
        n_steps = 30
        d = 4
        embeddings = [torch.full((d,), float(t)) for t in range(n_steps)]
        similarity = [float(t) / n_steps for t in range(n_steps)]

        class SyntheticSAE:
            def __call__(self, x: torch.Tensor) -> torch.Tensor:
                # x: [N, D]; produce [N, 3] activations.
                col0 = x.mean(dim=1)
                col1 = x.sum(dim=1)
                # Stable random feature seeded on dim, NOT on the input,
                # so it is constant across steps → zero correlation.
                torch.manual_seed(0)
                col2 = torch.randn(x.shape[0])
                return torch.stack([col0, col1, col2], dim=1)

        result = compute_corrsteer(
            sae=SyntheticSAE(),
            embedding_history=embeddings,
            similarity_history=similarity,
            top_k=3,
        )
        assert len(result) >= 2
        feature_indices = [f for f, _ in result]
        # f0 and f1 are perfectly correlated with similarity → top 2
        assert 0 in feature_indices[:2]
        assert 1 in feature_indices[:2]

    def test_returns_empty_on_empty_history(self) -> None:
        result = compute_corrsteer(
            sae=lambda x: x,
            embedding_history=[],
            similarity_history=[],
            top_k=8,
        )
        assert result == []

    def test_swallows_sae_errors(self) -> None:
        """A broken SAE returns ``[]`` rather than raising."""

        def boom(x: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("synthetic failure")

        result = compute_corrsteer(
            sae=boom,
            embedding_history=[torch.randn(4) for _ in range(5)],
            similarity_history=[0.1, 0.2, 0.3, 0.4, 0.5],
        )
        assert result == []

    def test_accepts_stacked_tensor_history(self) -> None:
        """The function also accepts a pre-stacked ``[N, D]`` tensor."""
        embeddings = torch.randn(10, 4)
        similarity = torch.linspace(0.0, 1.0, 10)

        class IdentitySAE:
            """Returns the input itself, so corrcoef equals classic
            per-dimension Pearson against the similarity ramp."""

            def __call__(self, x: torch.Tensor) -> torch.Tensor:
                return x

        result = compute_corrsteer(
            sae=IdentitySAE(),
            embedding_history=embeddings,
            similarity_history=similarity,
            top_k=4,
        )
        assert len(result) <= 4
        # All entries are (int, float) pairs with finite correlations.
        for feat, corr in result:
            assert isinstance(feat, int)
            assert isinstance(corr, float)
            assert -1.0 - 1e-6 <= corr <= 1.0 + 1e-6


def pytest_approx(expected: float, abs: float = 1e-5) -> object:  # noqa: A002
    """Inline pytest.approx so the assertion line stays readable."""
    import pytest

    return pytest.approx(expected, abs=abs)
