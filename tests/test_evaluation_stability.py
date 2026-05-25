"""Tests for ``embedding_art.evaluation.stability``."""

from __future__ import annotations

import math

import pytest

from embedding_art.evaluation.stability import (
    SeedStabilityReport,
    compute_seed_stability,
)


class TestComputeSeedStability:
    def test_single_seed_is_zero_variance(self) -> None:
        report = compute_seed_stability(similarities=[0.92])
        assert report.n_seeds == 1
        assert report.mean_similarity == pytest.approx(0.92)
        assert report.std_similarity == 0.0
        assert report.min_similarity == 0.92
        assert report.max_similarity == 0.92
        assert report.feature_overlap_jaccard is None

    def test_multi_seed_basic_stats(self) -> None:
        sims = [0.80, 0.85, 0.90, 0.95]
        report = compute_seed_stability(similarities=sims)
        assert report.n_seeds == 4
        assert report.mean_similarity == pytest.approx(0.875)
        # Sample stddev (Bessel's correction)
        expected_std = math.sqrt(sum((s - 0.875) ** 2 for s in sims) / 3)
        assert report.std_similarity == pytest.approx(expected_std)
        assert report.min_similarity == 0.80
        assert report.max_similarity == 0.95

    def test_empty_similarities_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            compute_seed_stability(similarities=[])

    def test_feature_overlap_jaccard_perfect(self) -> None:
        sims = [0.9, 0.9, 0.9]
        feats = [{1, 2, 3}, {1, 2, 3}, {1, 2, 3}]
        report = compute_seed_stability(similarities=sims, feature_sets=feats)
        assert report.feature_overlap_jaccard == pytest.approx(1.0)

    def test_feature_overlap_jaccard_zero_disjoint(self) -> None:
        sims = [0.9, 0.9]
        feats = [{1, 2, 3}, {4, 5, 6}]
        report = compute_seed_stability(similarities=sims, feature_sets=feats)
        assert report.feature_overlap_jaccard == pytest.approx(0.0)

    def test_feature_overlap_jaccard_partial(self) -> None:
        # 3 seeds: pairwise Jaccards = 1/3, 1/3, 1/3 => mean = 1/3
        sims = [0.9, 0.9, 0.9]
        feats = [{1, 2}, {1, 3}, {1, 4}]
        report = compute_seed_stability(similarities=sims, feature_sets=feats)
        assert report.feature_overlap_jaccard == pytest.approx(1 / 3)

    def test_feature_sets_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length"):
            compute_seed_stability(similarities=[0.9, 0.9], feature_sets=[{1, 2}])

    def test_single_seed_with_feature_set_returns_one(self) -> None:
        report = compute_seed_stability(similarities=[0.9], feature_sets=[{1, 2, 3}])
        assert report.feature_overlap_jaccard == 1.0


class TestSeedStabilityReport:
    def test_to_dict_round_trip(self) -> None:
        r = SeedStabilityReport(
            n_seeds=3,
            mean_similarity=0.9,
            std_similarity=0.05,
            min_similarity=0.85,
            max_similarity=0.95,
            feature_overlap_jaccard=0.5,
        )
        assert r.to_dict() == {
            "n_seeds": 3,
            "mean_similarity": 0.9,
            "std_similarity": 0.05,
            "min_similarity": 0.85,
            "max_similarity": 0.95,
            "feature_overlap_jaccard": 0.5,
        }
