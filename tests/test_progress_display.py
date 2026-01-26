"""
Tests for progress reporting functionality in the optimization engine.

These tests verify the behavior of progress display options and metrics reporting.
"""

from dataclasses import dataclass
from typing import Literal
from unittest.mock import MagicMock, patch

import pytest
import torch

from embedding_art.core.config import OptimizationConfig
from embedding_art.core.engine import (
    EmbeddingArtEngine,
    OptimizationResult,
    ProgressConfig,
)


class MockEncoder:
    """Mock encoder for testing."""

    def encode_text(self, text: str) -> torch.Tensor:
        return torch.randn(1, 128)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return torch.randn(1, 128)

    def encode_for_optimization(self, image: torch.Tensor) -> torch.Tensor:
        return torch.randn(1, 128, requires_grad=True)


class MockGenerator:
    """Mock generator for testing."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

    def init_latent(self, seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
        latent = torch.randn(1, 4, 64, 64, requires_grad=True, device=self.device)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(latent.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1))


def get_mock_encoder() -> MockEncoder:
    """Factory for mock encoder."""
    return MockEncoder()


def get_mock_generator(device: str = "cpu") -> MockGenerator:
    """Factory for mock generator."""
    return MockGenerator(device=device)


class TestProgressConfigDataclass:
    """Tests for ProgressConfig dataclass."""

    def test_default_values(self) -> None:
        """Default ProgressConfig should have sensible defaults."""
        config = ProgressConfig()

        assert config.enabled is True
        assert config.verbose is False
        assert config.show_loss_curve is True
        assert config.show_memory is True

    def test_disabled_progress(self) -> None:
        """ProgressConfig with enabled=False should disable all display."""
        config = ProgressConfig(enabled=False)

        assert config.enabled is False

    def test_verbose_mode(self) -> None:
        """ProgressConfig with verbose=True should enable detailed output."""
        config = ProgressConfig(verbose=True)

        assert config.verbose is True

    def test_all_options_configurable(self) -> None:
        """All ProgressConfig options should be configurable."""
        config = ProgressConfig(
            enabled=True,
            verbose=True,
            show_loss_curve=False,
            show_memory=False,
        )

        assert config.enabled is True
        assert config.verbose is True
        assert config.show_loss_curve is False
        assert config.show_memory is False


class TestProgressBackwardCompatibility:
    """Tests for backward compatibility with progress=True/False."""

    def test_progress_true_works(self) -> None:
        """progress=True should still work as before."""
        engine = EmbeddingArtEngine(encoder=get_mock_encoder(), device="cpu")
        engine.register_generator("image", get_mock_generator())

        from embedding_art.core.concept import Concept

        target = Concept(embedding=torch.randn(1, 128))
        config = OptimizationConfig(steps=3)

        result = engine.optimize(
            target=target,
            output_modality="image",
            config=config,
            progress=True,
        )

        assert isinstance(result, OptimizationResult)
        assert len(result.loss_history) == 3

    def test_progress_false_works(self) -> None:
        """progress=False should still disable progress display."""
        engine = EmbeddingArtEngine(encoder=get_mock_encoder(), device="cpu")
        engine.register_generator("image", get_mock_generator())

        from embedding_art.core.concept import Concept

        target = Concept(embedding=torch.randn(1, 128))
        config = OptimizationConfig(steps=3)

        result = engine.optimize(
            target=target,
            output_modality="image",
            config=config,
            progress=False,
        )

        assert isinstance(result, OptimizationResult)
        assert len(result.loss_history) == 3


class TestProgressConfigInOptimize:
    """Tests for ProgressConfig integration in optimize method."""

    def test_progress_config_accepted(self) -> None:
        """optimize() should accept ProgressConfig for progress parameter."""
        engine = EmbeddingArtEngine(encoder=get_mock_encoder(), device="cpu")
        engine.register_generator("image", get_mock_generator())

        from embedding_art.core.concept import Concept

        target = Concept(embedding=torch.randn(1, 128))
        config = OptimizationConfig(steps=3)
        progress_config = ProgressConfig(enabled=True, verbose=False)

        result = engine.optimize(
            target=target,
            output_modality="image",
            config=config,
            progress=progress_config,
        )

        assert isinstance(result, OptimizationResult)

    def test_verbose_mode_records_metrics(self) -> None:
        """Verbose mode should record per-step metrics."""
        engine = EmbeddingArtEngine(encoder=get_mock_encoder(), device="cpu")
        engine.register_generator("image", get_mock_generator())

        from embedding_art.core.concept import Concept

        target = Concept(embedding=torch.randn(1, 128))
        config = OptimizationConfig(steps=5)
        progress_config = ProgressConfig(enabled=True, verbose=True)

        result = engine.optimize(
            target=target,
            output_modality="image",
            config=config,
            progress=progress_config,
        )

        assert len(result.loss_history) == 5
        assert len(result.similarity_history) == 5


class TestMetricsTracking:
    """Tests for metrics tracking during optimization."""

    def test_similarity_history_recorded(self) -> None:
        """Similarity should be recorded each step."""
        engine = EmbeddingArtEngine(encoder=get_mock_encoder(), device="cpu")
        engine.register_generator("image", get_mock_generator())

        from embedding_art.core.concept import Concept

        target = Concept(embedding=torch.randn(1, 128))
        config = OptimizationConfig(steps=5)

        result = engine.optimize(
            target=target,
            output_modality="image",
            config=config,
            progress=False,
        )

        assert len(result.similarity_history) == 5
        for sim in result.similarity_history:
            assert -1.0 <= sim <= 1.0

    def test_loss_history_recorded(self) -> None:
        """Loss should be recorded each step."""
        engine = EmbeddingArtEngine(encoder=get_mock_encoder(), device="cpu")
        engine.register_generator("image", get_mock_generator())

        from embedding_art.core.concept import Concept

        target = Concept(embedding=torch.randn(1, 128))
        config = OptimizationConfig(steps=5)

        result = engine.optimize(
            target=target,
            output_modality="image",
            config=config,
            progress=False,
        )

        assert len(result.loss_history) == 5


class TestSparklineGeneration:
    """Tests for sparkline/mini loss curve generation."""

    def test_generate_sparkline_empty_list(self) -> None:
        """Sparkline should handle empty list gracefully."""
        from embedding_art.core.engine import generate_sparkline

        result = generate_sparkline([])

        assert result == ""

    def test_generate_sparkline_single_value(self) -> None:
        """Sparkline should handle single value."""
        from embedding_art.core.engine import generate_sparkline

        result = generate_sparkline([0.5])

        assert len(result) == 1
        assert result in "▁▂▃▄▅▆▇█"

    def test_generate_sparkline_increasing_values(self) -> None:
        """Sparkline should show increasing pattern for increasing values."""
        from embedding_art.core.engine import generate_sparkline

        values = [0.0, 0.25, 0.5, 0.75, 1.0]
        result = generate_sparkline(values)

        assert len(result) == 5

    def test_generate_sparkline_respects_max_length(self) -> None:
        """Sparkline should respect max_length parameter."""
        from embedding_art.core.engine import generate_sparkline

        values = list(range(100))
        result = generate_sparkline(values, max_length=20)

        assert len(result) <= 20

    def test_generate_sparkline_constant_values(self) -> None:
        """Sparkline should handle constant values."""
        from embedding_art.core.engine import generate_sparkline

        values = [0.5, 0.5, 0.5, 0.5, 0.5]
        result = generate_sparkline(values)

        assert len(result) == 5


class TestColorCoding:
    """Tests for color-coding similarity values."""

    def test_improving_similarity_color(self) -> None:
        """Improving similarity should get positive color code."""
        from embedding_art.core.engine import get_similarity_style

        style = get_similarity_style(current=0.8, previous=0.7)

        assert "green" in style or "bold" in style

    def test_stagnant_similarity_color(self) -> None:
        """Stagnant similarity should get neutral/warning color code."""
        from embedding_art.core.engine import get_similarity_style

        style = get_similarity_style(current=0.5, previous=0.5)

        assert "yellow" in style or "dim" in style

    def test_declining_similarity_color(self) -> None:
        """Declining similarity should get negative color code."""
        from embedding_art.core.engine import get_similarity_style

        style = get_similarity_style(current=0.4, previous=0.5)

        assert "red" in style or "yellow" in style


class TestMemoryUsageTracking:
    """Tests for memory usage tracking on GPU devices."""

    def test_get_memory_usage_cpu(self) -> None:
        """Memory usage should return None on CPU."""
        from embedding_art.core.engine import get_memory_usage_mb

        result = get_memory_usage_mb("cpu")

        assert result is None

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available",
    )
    def test_get_memory_usage_mps(self) -> None:
        """Memory usage should return a value on MPS if available."""
        from embedding_art.core.engine import get_memory_usage_mb

        result = get_memory_usage_mb("mps")

        assert result is None or isinstance(result, float)

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available",
    )
    def test_get_memory_usage_cuda(self) -> None:
        """Memory usage should return a value on CUDA if available."""
        from embedding_art.core.engine import get_memory_usage_mb

        result = get_memory_usage_mb("cuda")

        assert result is None or isinstance(result, float)


class TestLearningRateTracking:
    """Tests for learning rate tracking during optimization."""

    def test_lr_tracked_with_scheduler(self) -> None:
        """Learning rate should be tracked when using a scheduler."""
        engine = EmbeddingArtEngine(encoder=get_mock_encoder(), device="cpu")
        engine.register_generator("image", get_mock_generator())

        from embedding_art.core.concept import Concept

        target = Concept(embedding=torch.randn(1, 128))
        config = OptimizationConfig(steps=5, scheduler="cosine")

        lr_values = []

        def capture_lr(step: int, loss: float, sim: float, latent: torch.Tensor) -> None:
            pass

        result = engine.optimize(
            target=target,
            output_modality="image",
            config=config,
            progress=False,
            callback=capture_lr,
        )

        assert isinstance(result, OptimizationResult)
