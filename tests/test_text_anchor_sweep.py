"""
Tests for the M4 text-anchor empirical sweep harness (issue iik).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from click.testing import CliRunner

from embedding_art.cli.main import cli
from embedding_art.experiments.text_anchor_sweep import (
    SweepPoint,
    SweepResult,
    _pareto_front,
    _recommend,
    run_text_anchor_sweep,
    write_report,
)


def _angled_pair(dim: int, angle_deg: float, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a (target, anchor) pair at the requested angular separation."""
    gen = torch.Generator().manual_seed(seed)
    target = torch.randn(dim, generator=gen)
    target = target / target.norm()
    raw = torch.randn(dim, generator=gen)
    raw = raw - (raw @ target) * target
    orth = raw / raw.norm()
    theta = math.radians(angle_deg)
    anchor = math.cos(theta) * target + math.sin(theta) * orth
    return target, anchor / anchor.norm()


class TestSweepBasic:
    """Basic invariants of the sweep harness."""

    def test_zero_anchor_weight_optimises_only_target(self) -> None:
        target, anchor = _angled_pair(64, 60.0)
        result = run_text_anchor_sweep(
            target_embedding=target,
            anchor_embedding=anchor,
            modality="image",
            similarity_weights=(1.0,),
            anchor_weights=(0.0,),
            steps=200,
        )
        assert len(result.points) == 1
        p = result.points[0]
        # With anchor_weight=0 and 200 steps, target_sim should be ~1.0.
        assert p.target_similarity > 0.99
        # Anchor_sim should be close to cos(60°)=0.5 (since target became
        # the target and target·anchor=cos(60°)).
        assert abs(p.anchor_similarity - 0.5) < 0.05

    def test_zero_similarity_weight_optimises_only_anchor(self) -> None:
        target, anchor = _angled_pair(64, 60.0)
        result = run_text_anchor_sweep(
            target_embedding=target,
            anchor_embedding=anchor,
            modality="image",
            similarity_weights=(0.0,),
            anchor_weights=(1.0,),
            steps=200,
        )
        p = result.points[0]
        # current ends up at anchor.
        assert p.anchor_similarity > 0.99
        assert abs(p.target_similarity - 0.5) < 0.05

    def test_pareto_frontier_is_monotonic(self) -> None:
        target, anchor = _angled_pair(64, 45.0)
        result = run_text_anchor_sweep(
            target_embedding=target,
            anchor_embedding=anchor,
            similarity_weights=(1.0,),
            anchor_weights=(0.0, 0.1, 0.5, 1.0, 2.0),
            steps=200,
        )
        # As anchor_weight increases monotonically, anchor_sim increases
        # and target_sim decreases.
        sorted_by_aw = sorted(result.points, key=lambda p: p.anchor_weight)
        for prev, curr in zip(sorted_by_aw, sorted_by_aw[1:]):
            assert curr.anchor_similarity >= prev.anchor_similarity - 1e-3
            assert curr.target_similarity <= prev.target_similarity + 1e-3

    def test_grid_size_is_outer_times_inner(self) -> None:
        target, anchor = _angled_pair(32, 45.0)
        result = run_text_anchor_sweep(
            target_embedding=target,
            anchor_embedding=anchor,
            similarity_weights=(0.5, 1.0),
            anchor_weights=(0.0, 0.5, 1.0),
            steps=50,
        )
        assert len(result.points) == 2 * 3

    def test_mismatched_shapes_raise(self) -> None:
        with pytest.raises(ValueError):
            run_text_anchor_sweep(
                target_embedding=torch.randn(32),
                anchor_embedding=torch.randn(64),
                steps=10,
            )


class TestParetoFront:
    """The Pareto-front utility."""

    def test_single_point_is_its_own_front(self) -> None:
        p = SweepPoint(1.0, 0.5, 0.9, 0.7, -1.6, 100)
        front = _pareto_front([p])
        assert front == [p]

    def test_dominated_point_excluded(self) -> None:
        # p1 dominates p2 on both axes.
        p1 = SweepPoint(1.0, 0.5, 0.9, 0.8, -1.7, 100)
        p2 = SweepPoint(1.0, 0.5, 0.8, 0.7, -1.5, 100)
        front = _pareto_front([p1, p2])
        assert p1 in front
        assert p2 not in front

    def test_trade_off_both_kept(self) -> None:
        p_high_target = SweepPoint(1.0, 0.0, 0.99, 0.50, -0.99, 100)
        p_high_anchor = SweepPoint(1.0, 2.0, 0.85, 0.95, -2.0, 100)
        front = _pareto_front([p_high_target, p_high_anchor])
        assert len(front) == 2

    def test_recommend_picks_balanced_point(self) -> None:
        # A clean three-point Pareto frontier: corner-corner-middle.
        # The recommender should pick the middle (max-min).
        pareto = [
            SweepPoint(1.0, 0.0, 1.00, 0.50, -1.0, 100),
            SweepPoint(1.0, 0.5, 0.90, 0.85, -1.5, 100),
            SweepPoint(1.0, 2.0, 0.50, 1.00, -1.5, 100),
        ]
        recommended = _recommend(pareto)
        assert recommended is not None
        assert recommended.anchor_weight == 0.5


class TestWriteReport:
    """The JSON + markdown reporting surface."""

    def test_write_report_emits_both_files(self, tmp_path: Path) -> None:
        target, anchor = _angled_pair(32, 45.0)
        result = run_text_anchor_sweep(
            target_embedding=target,
            anchor_embedding=anchor,
            modality="image",
            similarity_weights=(1.0,),
            anchor_weights=(0.0, 0.5, 1.0),
            steps=50,
        )
        json_path, md_path = write_report(result, tmp_path)
        assert json_path.exists()
        assert md_path.exists()

        data = json.loads(json_path.read_text())
        assert data["modality"] == "image"
        assert len(data["points"]) == 3
        assert "recommended" in data
        assert data["recommended"] is not None

        md = md_path.read_text()
        assert "text-anchor sweep" in md
        assert "Recommended default" in md
        assert "| sim_w | anchor_w |" in md


class TestSweepResultToDict:
    """SweepResult.to_dict produces JSON-safe data."""

    def test_to_dict_preserves_fields(self) -> None:
        result = SweepResult(
            modality="audio",
            points=[SweepPoint(1.0, 0.5, 0.9, 0.7, -1.6, 100)],
            pareto_front=[SweepPoint(1.0, 0.5, 0.9, 0.7, -1.6, 100)],
            recommended=SweepPoint(1.0, 0.5, 0.9, 0.7, -1.6, 100),
        )
        d = result.to_dict()
        assert d["modality"] == "audio"
        assert len(d["points"]) == 1
        assert d["recommended"]["anchor_weight"] == 0.5
        # Round-trip through JSON.
        round_tripped = json.loads(json.dumps(d))
        assert round_tripped == d


class TestCli:
    """CLI surface: synthetic sweep produces parseable JSON output."""

    def test_synthetic_sweep_runs_end_to_end(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "text-anchor-sweep",
                "-o",
                str(tmp_path),
                "--modality",
                "image",
                "--dim",
                "32",
                "--angular-separation",
                "45",
                "--steps",
                "50",
                "--anchor-weights",
                "0.0,0.5,1.0",
            ],
        )
        assert result.exit_code == 0, result.output
        sweep_json = tmp_path / "image_sweep.json"
        assert sweep_json.exists()
        data = json.loads(sweep_json.read_text())
        assert data["modality"] == "image"
        assert len(data["points"]) == 3
