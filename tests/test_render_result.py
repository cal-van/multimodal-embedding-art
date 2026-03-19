"""
Unit tests for LossBreakdown, OptimizationHistory, and RenderResult.

Tests verify:
- LossBreakdown stores total and per-component losses
- OptimizationHistory tracks steps, accumulates breakdowns, computes final similarity
- RenderResult wraps output with metadata and optional fields
"""

from pathlib import Path

import pytest
import torch

from embedding_art.core.config import OptimizationConfig
from embedding_art.core.render_result import (
    LossBreakdown,
    OptimizationHistory,
    RenderResult,
)

# =============================================================================
# LossBreakdown Tests
# =============================================================================


class TestLossBreakdownCreation:
    """Tests for creating a LossBreakdown and accessing its fields."""

    def test_stores_total_tensor(self) -> None:
        """LossBreakdown should expose the total loss tensor."""
        total = torch.tensor(0.75)
        breakdown = LossBreakdown(total=total, components={})

        assert breakdown.total is total

    def test_stores_components_dict(self) -> None:
        """LossBreakdown should expose the components dict."""
        components = {"similarity": torch.tensor(-0.8), "tv": torch.tensor(0.05)}
        breakdown = LossBreakdown(total=torch.tensor(1.0), components=components)

        assert breakdown.components is components

    def test_can_access_individual_component(self) -> None:
        """Individual loss components should be accessible by key."""
        similarity = torch.tensor(-0.82)
        breakdown = LossBreakdown(
            total=torch.tensor(0.9),
            components={"similarity": similarity, "latent_norm": torch.tensor(0.08)},
        )

        assert breakdown.components["similarity"] is similarity

    def test_empty_components_dict_is_valid(self) -> None:
        """LossBreakdown with no components should be constructable."""
        breakdown = LossBreakdown(total=torch.tensor(0.0), components={})

        assert breakdown.components == {}

    def test_total_and_components_are_independent_fields(self) -> None:
        """Total and components are separate fields, not derived from each other."""
        total = torch.tensor(2.0)
        components = {"a": torch.tensor(1.0), "b": torch.tensor(0.5)}
        breakdown = LossBreakdown(total=total, components=components)

        assert breakdown.total is total
        assert len(breakdown.components) == 2


# =============================================================================
# OptimizationHistory Tests
# =============================================================================


class TestOptimizationHistoryEmptyState:
    """Tests for the initial empty state of OptimizationHistory."""

    def test_len_is_zero_when_empty(self) -> None:
        """Empty history should have length 0."""
        history = OptimizationHistory()

        assert len(history) == 0

    def test_steps_is_empty_list(self) -> None:
        """Empty history should have an empty steps list."""
        history = OptimizationHistory()

        assert history.steps == []

    def test_loss_breakdowns_is_empty_list(self) -> None:
        """Empty history should have an empty loss_breakdowns list."""
        history = OptimizationHistory()

        assert history.loss_breakdowns == []

    def test_similarity_values_is_empty_list(self) -> None:
        """Empty history should have an empty similarity_values list."""
        history = OptimizationHistory()

        assert history.similarity_values == []

    def test_final_similarity_is_zero_when_empty(self) -> None:
        """final_similarity should return 0.0 when no steps have been recorded."""
        history = OptimizationHistory()

        assert history.final_similarity == 0.0


class TestOptimizationHistoryRecord:
    """Tests for recording steps into OptimizationHistory."""

    def test_record_appends_step_number(self) -> None:
        """record() should append the step number to steps list."""
        history = OptimizationHistory()
        breakdown = LossBreakdown(total=torch.tensor(1.0), components={})

        history.record(step=10, breakdown=breakdown)

        assert history.steps == [10]

    def test_record_increments_length(self) -> None:
        """len() should increase by one after each record() call."""
        history = OptimizationHistory()
        breakdown = LossBreakdown(total=torch.tensor(1.0), components={})

        history.record(step=0, breakdown=breakdown)

        assert len(history) == 1

    def test_record_stores_component_as_float(self) -> None:
        """record() should convert component tensors to Python floats."""
        history = OptimizationHistory()
        breakdown = LossBreakdown(
            total=torch.tensor(0.5),
            components={"tv": torch.tensor(0.12345)},
        )

        history.record(step=0, breakdown=breakdown)

        assert isinstance(history.loss_breakdowns[0]["tv"], float)

    def test_record_extracts_similarity_component(self) -> None:
        """record() should extract the similarity key and negate it."""
        history = OptimizationHistory()
        # Optimization minimizes -cosine_sim, so similarity component is negative.
        # History should store the positive similarity value (negated).
        breakdown = LossBreakdown(
            total=torch.tensor(0.2),
            components={"similarity": torch.tensor(-0.85)},
        )

        history.record(step=0, breakdown=breakdown)

        assert pytest.approx(history.similarity_values[0], abs=1e-6) == 0.85

    def test_record_without_similarity_key_does_not_append_similarity(self) -> None:
        """record() with no 'similarity' key should not append to similarity_values."""
        history = OptimizationHistory()
        breakdown = LossBreakdown(
            total=torch.tensor(0.3),
            components={"tv": torch.tensor(0.1)},
        )

        history.record(step=0, breakdown=breakdown)

        assert history.similarity_values == []

    def test_record_stores_all_components_in_loss_breakdowns(self) -> None:
        """record() should store all component names and their float values."""
        history = OptimizationHistory()
        breakdown = LossBreakdown(
            total=torch.tensor(1.0),
            components={
                "similarity": torch.tensor(-0.7),
                "tv": torch.tensor(0.05),
                "latent_norm": torch.tensor(0.25),
            },
        )

        history.record(step=5, breakdown=breakdown)

        recorded = history.loss_breakdowns[0]
        assert set(recorded.keys()) == {"similarity", "tv", "latent_norm"}
        assert isinstance(recorded["tv"], float)
        assert isinstance(recorded["latent_norm"], float)


class TestOptimizationHistoryMultiStep:
    """Tests for multi-step recording behavior."""

    def test_multi_step_recording_accumulates_steps(self) -> None:
        """Recording multiple steps should accumulate all step numbers."""
        history = OptimizationHistory()

        for step in [0, 10, 20]:
            history.record(
                step=step,
                breakdown=LossBreakdown(
                    total=torch.tensor(1.0),
                    components={"similarity": torch.tensor(-0.5)},
                ),
            )

        assert history.steps == [0, 10, 20]

    def test_multi_step_recording_accumulates_similarity_values(self) -> None:
        """Recording multiple steps should accumulate all similarity values."""
        history = OptimizationHistory()
        sim_tensors = [torch.tensor(-0.5), torch.tensor(-0.7), torch.tensor(-0.9)]

        for i, sim in enumerate(sim_tensors):
            history.record(
                step=i,
                breakdown=LossBreakdown(
                    total=torch.tensor(1.0),
                    components={"similarity": sim},
                ),
            )

        assert len(history.similarity_values) == 3

    def test_final_similarity_reflects_last_recorded_value(self) -> None:
        """final_similarity should return the most recently recorded similarity."""
        history = OptimizationHistory()

        history.record(
            step=0,
            breakdown=LossBreakdown(
                total=torch.tensor(1.0),
                components={"similarity": torch.tensor(-0.5)},
            ),
        )
        history.record(
            step=1,
            breakdown=LossBreakdown(
                total=torch.tensor(0.6),
                components={"similarity": torch.tensor(-0.9)},
            ),
        )

        assert pytest.approx(history.final_similarity, abs=1e-6) == 0.9

    def test_len_equals_number_of_recorded_steps(self) -> None:
        """len() should equal the total number of record() calls made."""
        history = OptimizationHistory()

        for i in range(7):
            history.record(
                step=i,
                breakdown=LossBreakdown(total=torch.tensor(0.1), components={}),
            )

        assert len(history) == 7


# =============================================================================
# RenderResult Tests
# =============================================================================


class TestRenderResultCreation:
    """Tests for creating a RenderResult with all required fields."""

    def test_stores_output_tensor(self) -> None:
        """RenderResult should store the output tensor."""
        output = torch.rand(1, 3, 64, 64)
        history = OptimizationHistory()
        result = RenderResult(
            output=output,
            history=history,
            encoder_name="imagebind",
            final_similarity=0.87,
        )

        assert result.output is output

    def test_stores_history(self) -> None:
        """RenderResult should store the OptimizationHistory."""
        history = OptimizationHistory()
        result = RenderResult(
            output=torch.rand(1, 3, 64, 64),
            history=history,
            encoder_name="imagebind",
            final_similarity=0.87,
        )

        assert result.history is history

    def test_stores_encoder_name(self) -> None:
        """RenderResult should store the encoder name string."""
        result = RenderResult(
            output=torch.rand(1, 3, 64, 64),
            history=OptimizationHistory(),
            encoder_name="imagebind",
            final_similarity=0.87,
        )

        assert result.encoder_name == "imagebind"

    def test_stores_final_similarity(self) -> None:
        """RenderResult should store the final cosine similarity."""
        result = RenderResult(
            output=torch.rand(1, 3, 64, 64),
            history=OptimizationHistory(),
            encoder_name="imagebind",
            final_similarity=0.92,
        )

        assert result.final_similarity == 0.92

    def test_config_defaults_to_none(self) -> None:
        """config field should default to None when not provided."""
        result = RenderResult(
            output=torch.rand(1, 3, 64, 64),
            history=OptimizationHistory(),
            encoder_name="imagebind",
            final_similarity=0.87,
        )

        assert result.config is None

    def test_checkpoints_defaults_to_none(self) -> None:
        """checkpoints field should default to None when not provided."""
        result = RenderResult(
            output=torch.rand(1, 3, 64, 64),
            history=OptimizationHistory(),
            encoder_name="imagebind",
            final_similarity=0.87,
        )

        assert result.checkpoints is None

    def test_can_store_config(self) -> None:
        """RenderResult should accept an OptimizationConfig when provided."""
        config = OptimizationConfig(steps=500)
        result = RenderResult(
            output=torch.rand(1, 3, 64, 64),
            history=OptimizationHistory(),
            encoder_name="imagebind",
            final_similarity=0.87,
            config=config,
        )

        assert result.config is config
        assert result.config.steps == 500

    def test_can_store_checkpoints(self) -> None:
        """RenderResult should accept a list of checkpoint paths when provided."""
        checkpoints = [Path("/tmp/ckpt_0.pt"), Path("/tmp/ckpt_100.pt")]
        result = RenderResult(
            output=torch.rand(1, 3, 64, 64),
            history=OptimizationHistory(),
            encoder_name="imagebind",
            final_similarity=0.87,
            checkpoints=checkpoints,
        )

        assert result.checkpoints == checkpoints


# =============================================================================
# Backward Compatibility Alias Tests
# =============================================================================


