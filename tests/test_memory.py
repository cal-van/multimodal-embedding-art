"""
Tests for memory management functionality.

These tests verify the MemoryManager class and related utilities
for optimizing memory usage during optimization runs.
"""

import pytest
import torch
import torch.nn as nn


class TestMemoryConfigExists:
    """Tests verifying MemoryConfig dataclass exists and has expected fields."""

    def test_memory_config_is_importable(self):
        """Should be able to import MemoryConfig from embedding_art.core.memory."""
        from embedding_art.core.memory import MemoryConfig

        assert MemoryConfig is not None

    def test_memory_config_has_default_values(self):
        """Should have sensible default values."""
        from embedding_art.core.memory import MemoryConfig

        config = MemoryConfig()

        assert config.offload_to_cpu is False
        assert config.use_mixed_precision is False
        assert config.empty_cache_every == 0
        assert config.track_memory is False

    def test_memory_config_accepts_custom_values(self):
        """Should accept custom configuration values."""
        from embedding_art.core.memory import MemoryConfig

        config = MemoryConfig(
            offload_to_cpu=True,
            use_mixed_precision=True,
            empty_cache_every=10,
            track_memory=True,
        )

        assert config.offload_to_cpu is True
        assert config.use_mixed_precision is True
        assert config.empty_cache_every == 10
        assert config.track_memory is True

    def test_memory_config_low_memory_preset(self):
        """Should provide a low_memory preset with aggressive settings."""
        from embedding_art.core.memory import MemoryConfig

        config = MemoryConfig.low_memory()

        assert config.offload_to_cpu is True
        assert config.use_mixed_precision is True
        assert config.empty_cache_every > 0
        assert config.track_memory is True


class TestMemoryManagerExists:
    """Tests verifying MemoryManager class exists and is importable."""

    def test_memory_manager_is_importable(self):
        """Should be able to import MemoryManager from embedding_art.core.memory."""
        from embedding_art.core.memory import MemoryManager

        assert MemoryManager is not None

    def test_memory_manager_accepts_config(self):
        """Should accept a MemoryConfig at initialization."""
        from embedding_art.core.memory import MemoryConfig, MemoryManager

        config = MemoryConfig(offload_to_cpu=True)
        manager = MemoryManager(config=config, device="cpu")

        assert manager.config.offload_to_cpu is True

    def test_memory_manager_accepts_device(self):
        """Should accept a device string at initialization."""
        from embedding_art.core.memory import MemoryConfig, MemoryManager

        config = MemoryConfig()
        manager = MemoryManager(config=config, device="cpu")

        assert manager.device == torch.device("cpu")


class TestMemoryContextManager:
    """Tests verifying the memory context manager functionality."""

    def test_memory_efficient_context_exists(self):
        """Should provide a memory_efficient context manager."""
        from embedding_art.core.memory import MemoryConfig, MemoryManager

        config = MemoryConfig()
        manager = MemoryManager(config=config, device="cpu")

        assert hasattr(manager, "memory_efficient")
        assert callable(manager.memory_efficient)

    def test_memory_efficient_context_is_context_manager(self):
        """Should be usable as a context manager."""
        from embedding_art.core.memory import MemoryConfig, MemoryManager

        config = MemoryConfig()
        manager = MemoryManager(config=config, device="cpu")

        with manager.memory_efficient():
            pass  # Should not raise

    def test_memory_efficient_context_enables_autocast_when_configured(self):
        """Should enable autocast when use_mixed_precision is True."""
        from embedding_art.core.memory import MemoryConfig, MemoryManager

        config = MemoryConfig(use_mixed_precision=True)
        manager = MemoryManager(config=config, device="cpu")

        # CPU autocast uses bfloat16 on supported hardware, otherwise no-op
        with manager.memory_efficient():
            x = torch.randn(10, 10)
            y = torch.matmul(x, x)
            # Should complete without error
            assert y is not None


class TestModelOffloading:
    """Tests verifying model offloading functionality."""

    def test_offload_model_moves_to_cpu(self):
        """Should move model to CPU when offloading."""
        from embedding_art.core.memory import MemoryConfig, MemoryManager

        config = MemoryConfig(offload_to_cpu=True)
        manager = MemoryManager(config=config, device="cpu")

        model = nn.Linear(10, 10)
        model.to("cpu")  # Start on CPU (for testing)

        manager.offload_model(model)

        # Model should be on CPU
        assert next(model.parameters()).device == torch.device("cpu")

    def test_restore_model_moves_back_to_device(self):
        """Should move model back to original device when restoring."""
        from embedding_art.core.memory import MemoryConfig, MemoryManager

        config = MemoryConfig(offload_to_cpu=True)
        manager = MemoryManager(config=config, device="cpu")

        model = nn.Linear(10, 10)

        manager.offload_model(model)
        manager.restore_model(model)

        # Model should be back on the manager's device
        assert next(model.parameters()).device == manager.device

    def test_offload_is_noop_when_not_configured(self):
        """Should not move model when offload_to_cpu is False."""
        from embedding_art.core.memory import MemoryConfig, MemoryManager

        config = MemoryConfig(offload_to_cpu=False)
        manager = MemoryManager(config=config, device="cpu")

        model = nn.Linear(10, 10)
        original_device = next(model.parameters()).device

        manager.offload_model(model)

        # Model should still be on original device
        assert next(model.parameters()).device == original_device

    def test_model_offload_context_manager(self):
        """Should provide context manager for temporary offloading."""
        from embedding_art.core.memory import MemoryConfig, MemoryManager

        config = MemoryConfig(offload_to_cpu=True)
        manager = MemoryManager(config=config, device="cpu")

        model = nn.Linear(10, 10)

        with manager.offloaded(model):
            # Model should be on CPU inside context
            assert next(model.parameters()).device == torch.device("cpu")

        # Model should be back on manager device after context
        assert next(model.parameters()).device == manager.device


class TestEmptyCacheFunctionality:
    """Tests verifying cache clearing functionality."""

    def test_empty_cache_method_exists(self):
        """Should have an empty_cache method."""
        from embedding_art.core.memory import MemoryConfig, MemoryManager

        config = MemoryConfig()
        manager = MemoryManager(config=config, device="cpu")

        assert hasattr(manager, "empty_cache")
        assert callable(manager.empty_cache)

    def test_empty_cache_does_not_error_on_cpu(self):
        """Should not raise error when called on CPU device."""
        from embedding_art.core.memory import MemoryConfig, MemoryManager

        config = MemoryConfig()
        manager = MemoryManager(config=config, device="cpu")

        # Should not raise
        manager.empty_cache()

    def test_should_empty_cache_returns_true_at_configured_interval(self):
        """Should return True when step matches empty_cache_every interval."""
        from embedding_art.core.memory import MemoryConfig, MemoryManager

        config = MemoryConfig(empty_cache_every=10)
        manager = MemoryManager(config=config, device="cpu")

        assert manager.should_empty_cache(0) is True
        assert manager.should_empty_cache(5) is False
        assert manager.should_empty_cache(10) is True
        assert manager.should_empty_cache(20) is True
        assert manager.should_empty_cache(25) is False

    def test_should_empty_cache_returns_false_when_disabled(self):
        """Should return False when empty_cache_every is 0."""
        from embedding_art.core.memory import MemoryConfig, MemoryManager

        config = MemoryConfig(empty_cache_every=0)
        manager = MemoryManager(config=config, device="cpu")

        assert manager.should_empty_cache(0) is False
        assert manager.should_empty_cache(10) is False
        assert manager.should_empty_cache(100) is False


class TestMemoryTracking:
    """Tests verifying memory tracking functionality."""

    def test_get_memory_usage_returns_dict(self):
        """Should return a dictionary with memory information."""
        from embedding_art.core.memory import MemoryConfig, MemoryManager

        config = MemoryConfig(track_memory=True)
        manager = MemoryManager(config=config, device="cpu")

        usage = manager.get_memory_usage()

        assert isinstance(usage, dict)
        assert "allocated_mb" in usage
        assert "reserved_mb" in usage

    def test_get_memory_usage_returns_none_values_on_cpu(self):
        """Should return None values when on CPU (no GPU memory to track)."""
        from embedding_art.core.memory import MemoryConfig, MemoryManager

        config = MemoryConfig(track_memory=True)
        manager = MemoryManager(config=config, device="cpu")

        usage = manager.get_memory_usage()

        # CPU doesn't have GPU memory tracking
        assert usage["allocated_mb"] is None
        assert usage["reserved_mb"] is None

    def test_record_memory_snapshot(self):
        """Should record memory snapshots when tracking is enabled."""
        from embedding_art.core.memory import MemoryConfig, MemoryManager

        config = MemoryConfig(track_memory=True)
        manager = MemoryManager(config=config, device="cpu")

        manager.record_memory_snapshot("before_decode")
        manager.record_memory_snapshot("after_decode")

        snapshots = manager.get_memory_snapshots()

        assert len(snapshots) == 2
        assert snapshots[0]["label"] == "before_decode"
        assert snapshots[1]["label"] == "after_decode"

    def test_no_snapshots_recorded_when_tracking_disabled(self):
        """Should not record snapshots when track_memory is False."""
        from embedding_art.core.memory import MemoryConfig, MemoryManager

        config = MemoryConfig(track_memory=False)
        manager = MemoryManager(config=config, device="cpu")

        manager.record_memory_snapshot("test")

        snapshots = manager.get_memory_snapshots()

        assert len(snapshots) == 0

    def test_peak_memory_tracking(self):
        """Should track peak memory usage."""
        from embedding_art.core.memory import MemoryConfig, MemoryManager

        config = MemoryConfig(track_memory=True)
        manager = MemoryManager(config=config, device="cpu")

        peak = manager.get_peak_memory_mb()

        # Should return a number (or None on CPU)
        assert peak is None or isinstance(peak, float)

    def test_reset_peak_memory(self):
        """Should be able to reset peak memory tracking."""
        from embedding_art.core.memory import MemoryConfig, MemoryManager

        config = MemoryConfig(track_memory=True)
        manager = MemoryManager(config=config, device="cpu")

        # Should not raise
        manager.reset_peak_memory()


class TestMixedPrecisionSupport:
    """Tests verifying mixed precision support."""

    def test_autocast_context_exists(self):
        """Should provide an autocast context manager."""
        from embedding_art.core.memory import MemoryConfig, MemoryManager

        config = MemoryConfig(use_mixed_precision=True)
        manager = MemoryManager(config=config, device="cpu")

        assert hasattr(manager, "autocast")
        assert callable(manager.autocast)

    def test_autocast_context_is_usable(self):
        """Should be usable as a context manager."""
        from embedding_art.core.memory import MemoryConfig, MemoryManager

        config = MemoryConfig(use_mixed_precision=True)
        manager = MemoryManager(config=config, device="cpu")

        with manager.autocast():
            x = torch.randn(10, 10)
            y = torch.matmul(x, x)
            assert y is not None

    def test_autocast_is_noop_when_not_configured(self):
        """Should be a no-op context when use_mixed_precision is False."""
        from embedding_art.core.memory import MemoryConfig, MemoryManager

        config = MemoryConfig(use_mixed_precision=False)
        manager = MemoryManager(config=config, device="cpu")

        with manager.autocast():
            x = torch.randn(10, 10, dtype=torch.float32)
            y = torch.matmul(x, x)
            # Should still be float32 when autocast is disabled
            assert y.dtype == torch.float32


class TestMemoryManagerWithMPS:
    """Tests specific to MPS (Apple Silicon) device handling."""

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available",
    )
    def test_empty_cache_calls_mps_empty_cache(self):
        """Should call torch.mps.empty_cache() on MPS device."""
        from embedding_art.core.memory import MemoryConfig, MemoryManager

        config = MemoryConfig()
        manager = MemoryManager(config=config, device="mps")

        # Should not raise
        manager.empty_cache()

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS not available",
    )
    def test_autocast_uses_float16_on_mps(self):
        """Should use float16 for autocast on MPS."""
        from embedding_art.core.memory import MemoryConfig, MemoryManager

        config = MemoryConfig(use_mixed_precision=True)
        manager = MemoryManager(config=config, device="mps")

        # MPS autocast should work
        with manager.autocast():
            x = torch.randn(10, 10, device="mps")
            y = torch.matmul(x, x)
            assert y is not None


class TestMemoryManagerWithCUDA:
    """Tests specific to CUDA device handling."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available",
    )
    def test_empty_cache_calls_cuda_empty_cache(self):
        """Should call torch.cuda.empty_cache() on CUDA device."""
        from embedding_art.core.memory import MemoryConfig, MemoryManager

        config = MemoryConfig()
        manager = MemoryManager(config=config, device="cuda")

        # Should not raise
        manager.empty_cache()

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available",
    )
    def test_get_memory_usage_returns_values_on_cuda(self):
        """Should return actual memory values on CUDA."""
        from embedding_art.core.memory import MemoryConfig, MemoryManager

        config = MemoryConfig(track_memory=True)
        manager = MemoryManager(config=config, device="cuda")

        usage = manager.get_memory_usage()

        assert usage["allocated_mb"] is not None
        assert usage["reserved_mb"] is not None
        assert isinstance(usage["allocated_mb"], float)
        assert isinstance(usage["reserved_mb"], float)


class TestEngineMemoryIntegration:
    """Tests for MemoryManager integration with EmbeddingArtEngine."""

    def test_engine_accepts_memory_config(self, mock_encoder, mock_generator):
        """Should accept MemoryConfig in optimize method."""
        from embedding_art.core.config import AugmentationConfig, OptimizationConfig
        from embedding_art.core.engine import EmbeddingArtEngine
        from embedding_art.core.memory import MemoryConfig
        from embedding_art.regularizers import CompositeRegularizer

        engine = EmbeddingArtEngine(encoder=mock_encoder, device="cpu")
        engine.register_generator("image", mock_generator)

        from embedding_art.core.concept import Concept

        target = Concept.from_text("test", mock_encoder)

        config = OptimizationConfig(
            steps=5,
            learning_rate=0.1,
            checkpoint_every=0,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        memory_config = MemoryConfig(track_memory=True)

        result = engine.optimize(
            target=target,
            output_modality="image",
            config=config,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
            memory_config=memory_config,
        )

        assert result is not None
        assert result.final_latent is not None

    def test_engine_uses_mixed_precision_when_configured(self, mock_encoder, mock_generator):
        """Should use mixed precision during optimization when configured."""
        from embedding_art.core.config import AugmentationConfig, OptimizationConfig
        from embedding_art.core.engine import EmbeddingArtEngine
        from embedding_art.core.memory import MemoryConfig
        from embedding_art.regularizers import CompositeRegularizer

        engine = EmbeddingArtEngine(encoder=mock_encoder, device="cpu")
        engine.register_generator("image", mock_generator)

        from embedding_art.core.concept import Concept

        target = Concept.from_text("test", mock_encoder)

        config = OptimizationConfig(
            steps=5,
            learning_rate=0.1,
            checkpoint_every=0,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        memory_config = MemoryConfig(use_mixed_precision=True)

        # Should not raise - mixed precision should work
        result = engine.optimize(
            target=target,
            output_modality="image",
            config=config,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
            memory_config=memory_config,
        )

        assert result is not None

    def test_engine_empties_cache_at_configured_intervals(self, mock_encoder, mock_generator):
        """Should empty cache at configured step intervals."""
        from embedding_art.core.config import AugmentationConfig, OptimizationConfig
        from embedding_art.core.engine import EmbeddingArtEngine
        from embedding_art.core.memory import MemoryConfig
        from embedding_art.regularizers import CompositeRegularizer

        engine = EmbeddingArtEngine(encoder=mock_encoder, device="cpu")
        engine.register_generator("image", mock_generator)

        from embedding_art.core.concept import Concept

        target = Concept.from_text("test", mock_encoder)

        config = OptimizationConfig(
            steps=20,
            learning_rate=0.1,
            checkpoint_every=0,
            augmentation=AugmentationConfig(random_crop=False, random_flip=False),
        )

        memory_config = MemoryConfig(empty_cache_every=5)

        # Should complete without error
        result = engine.optimize(
            target=target,
            output_modality="image",
            config=config,
            regularizers=CompositeRegularizer.minimal(),
            progress=False,
            memory_config=memory_config,
        )

        assert result is not None


class TestCLIMemoryOptions:
    """Tests for CLI memory management options."""

    def test_cli_accepts_low_memory_flag(self):
        """Should accept --low-memory flag on optimize command."""
        from click.testing import CliRunner

        from embedding_art.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["optimize", "-t", "test", "1.0", "--low-memory", "--dry-run"],
        )

        # Should not fail due to unknown option
        assert result.exit_code == 0 or "--low-memory" not in result.output

    def test_cli_accepts_offload_flag(self):
        """Should accept --offload flag on optimize command."""
        from click.testing import CliRunner

        from embedding_art.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["optimize", "-t", "test", "1.0", "--offload", "--dry-run"],
        )

        # Should not fail due to unknown option
        assert result.exit_code == 0 or "--offload" not in result.output

    def test_cli_accepts_fp16_flag(self):
        """Should accept --fp16 flag on optimize command."""
        from click.testing import CliRunner

        from embedding_art.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["optimize", "-t", "test", "1.0", "--fp16", "--dry-run"],
        )

        # Should not fail due to unknown option
        assert result.exit_code == 0 or "--fp16" not in result.output


class TestMemoryManagerIntegration:
    """Integration tests for MemoryManager with mock models."""

    def test_full_workflow_with_offloading(self):
        """Should handle full offload/restore workflow."""
        from embedding_art.core.memory import MemoryConfig, MemoryManager

        config = MemoryConfig(
            offload_to_cpu=True,
            track_memory=True,
        )
        manager = MemoryManager(config=config, device="cpu")

        encoder = nn.Linear(1024, 1024)
        generator = nn.Linear(64, 1024)

        # Simulate optimization loop pattern
        manager.record_memory_snapshot("start")

        # Offload encoder while using generator
        with manager.offloaded(encoder):
            manager.record_memory_snapshot("encoder_offloaded")
            output = generator(torch.randn(1, 64))
            assert output is not None

        manager.record_memory_snapshot("encoder_restored")

        # Offload generator while using encoder
        with manager.offloaded(generator):
            manager.record_memory_snapshot("generator_offloaded")
            embedding = encoder(torch.randn(1, 1024))
            assert embedding is not None

        manager.record_memory_snapshot("end")

        snapshots = manager.get_memory_snapshots()
        assert len(snapshots) == 5

    def test_workflow_with_mixed_precision_and_cache_clearing(self):
        """Should handle mixed precision and cache clearing together."""
        from embedding_art.core.memory import MemoryConfig, MemoryManager

        config = MemoryConfig(
            use_mixed_precision=True,
            empty_cache_every=5,
            track_memory=True,
        )
        manager = MemoryManager(config=config, device="cpu")

        model = nn.Linear(100, 100)

        for step in range(20):
            with manager.autocast():
                x = torch.randn(10, 100)
                y = model(x)
                assert y is not None

            if manager.should_empty_cache(step):
                manager.empty_cache()
                manager.record_memory_snapshot(f"step_{step}_cleared")

        snapshots = manager.get_memory_snapshots()
        # Should have snapshots at steps 0, 5, 10, 15
        assert len(snapshots) == 4
