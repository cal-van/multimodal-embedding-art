"""
Tests for v1 API deprecation warnings.

TDD red phase: these tests are written before any deprecation warning exists.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn.functional as F

from tests.conftest import MockEncoder, MockGenerator


class TestOptimizeDeprecationWarning:
    """EmbeddingArtEngine.optimize() should emit a DeprecationWarning."""

    def _make_engine(self):
        from embedding_art.core.engine import EmbeddingArtEngine

        encoder = MockEncoder(embedding_dim=1024, device="cpu")
        gen = MockGenerator(
            latent_shape=(1, 4, 8, 8),
            output_modality="image",
            output_shape=(1, 3, 16, 16),
            device="cpu",
        )
        engine = EmbeddingArtEngine(encoder=encoder, device="cpu")
        engine.register_generator("image", gen)
        return engine, encoder

    def test_optimize_emits_deprecation_warning(self):
        """Calling engine.optimize() must raise DeprecationWarning."""
        from embedding_art.core.concept import Concept
        from embedding_art.core.config import OptimizationConfig

        engine, encoder = self._make_engine()

        target_emb = F.normalize(torch.randn(1, 1024), dim=-1)
        target = Concept(embedding=target_emb, description="test")

        config = OptimizationConfig(steps=1)

        with pytest.warns(DeprecationWarning, match="optimize"):
            engine.optimize(target=target, output_modality="image", config=config)

    def test_optimize_deprecation_message_mentions_render(self):
        """The warning message should mention the replacement: render()."""
        from embedding_art.core.concept import Concept
        from embedding_art.core.config import OptimizationConfig

        engine, encoder = self._make_engine()

        target_emb = F.normalize(torch.randn(1, 1024), dim=-1)
        target = Concept(embedding=target_emb, description="test")
        config = OptimizationConfig(steps=1)

        with pytest.warns(DeprecationWarning) as record:
            engine.optimize(target=target, output_modality="image", config=config)

        # At least one warning should mention "render"
        messages = [str(w.message) for w in record]
        assert any(
            "render" in msg.lower() for msg in messages
        ), f"Expected warning to mention 'render', got: {messages}"

    def test_optimize_still_returns_result(self):
        """Despite the warning, optimize() should still return a valid result."""
        from embedding_art.core.concept import Concept
        from embedding_art.core.config import OptimizationConfig
        from embedding_art.core.engine import OptimizationResult

        engine, encoder = self._make_engine()

        target_emb = F.normalize(torch.randn(1, 1024), dim=-1)
        target = Concept(embedding=target_emb, description="test")
        config = OptimizationConfig(steps=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            result = engine.optimize(target=target, output_modality="image", config=config)

        assert isinstance(result, OptimizationResult)

    def test_optimize_warning_stacklevel_points_to_caller(self):
        """The warning's filename should not be engine.py (stacklevel=2)."""
        from embedding_art.core.concept import Concept
        from embedding_art.core.config import OptimizationConfig

        engine, encoder = self._make_engine()

        target_emb = F.normalize(torch.randn(1, 1024), dim=-1)
        target = Concept(embedding=target_emb, description="test")
        config = OptimizationConfig(steps=1)

        with pytest.warns(DeprecationWarning) as record:
            engine.optimize(target=target, output_modality="image", config=config)

        # stacklevel=2 means the warning filename should point here, not engine.py
        for w in record:
            assert "engine.py" not in str(
                w.filename
            ), f"Warning points to engine.py — stacklevel is wrong: {w.filename}"
