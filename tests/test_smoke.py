"""
Smoke tests to verify basic imports work correctly.

These tests don't verify behavior - they just ensure the package
structure is correct and all public modules are importable.
"""


class TestPackageImports:
    """Verify all public modules can be imported."""

    def test_top_level_imports(self):
        """Should import core types from package root."""
        from embedding_art import (
            AugmentationConfig,
            CompositeRegularizer,
            Concept,
            EmbeddingArtEngine,
            OptimizationConfig,
            OptimizationResult,
        )

        assert Concept is not None
        assert EmbeddingArtEngine is not None
        assert OptimizationConfig is not None
        assert AugmentationConfig is not None
        assert OptimizationResult is not None
        assert CompositeRegularizer is not None

    def test_encoder_protocol_import(self):
        """Should import encoder protocol from encoders module."""
        from embedding_art.encoders.base import Encoder

        assert Encoder is not None

    def test_generator_protocol_import(self):
        """Should import generator protocol from generators module."""
        from embedding_art.generators.base import Generator

        assert Generator is not None

    def test_regularizer_imports(self):
        """Should import regularizers from regularizers module."""
        from embedding_art.regularizers import (
            CompositeRegularizer,
            LatentNorm,
            Regularizer,
            SpectralRegularizer,
            TotalVariation,
        )

        assert Regularizer is not None
        assert TotalVariation is not None
        assert SpectralRegularizer is not None
        assert LatentNorm is not None
        assert CompositeRegularizer is not None

    def test_core_module_imports(self):
        """Should import from core module directly."""
        from embedding_art.core import (
            AugmentationConfig,
            Concept,
            EmbeddingArtEngine,
            OptimizationConfig,
            OptimizationResult,
        )

        assert Concept is not None
        assert EmbeddingArtEngine is not None
        assert OptimizationConfig is not None
        assert AugmentationConfig is not None
        assert OptimizationResult is not None


class TestTorchAvailable:
    """Verify PyTorch is available and working."""

    def test_torch_import(self):
        """Should import torch."""
        import torch

        assert torch is not None

    def test_torch_tensor_creation(self):
        """Should create basic tensors."""
        import torch

        tensor = torch.zeros(3, 3)
        assert tensor.shape == (3, 3)
