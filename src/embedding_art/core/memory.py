"""
Memory management utilities for embedding-art.

Provides tools for optimizing memory usage during optimization runs,
including model offloading, mixed precision support, and memory tracking.
"""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from embedding_art.exceptions import OutOfMemoryError


DEFAULT_MEMORY_LIMIT_MB = 48 * 1024  # 48GB default limit


@dataclass
class MemoryConfig:
    """Configuration for memory management during optimization."""

    offload_to_cpu: bool = False
    use_mixed_precision: bool = False
    empty_cache_every: int = 0
    track_memory: bool = False
    memory_limit_mb: float = DEFAULT_MEMORY_LIMIT_MB

    @classmethod
    def low_memory(cls) -> "MemoryConfig":
        """
        Create a configuration optimized for low memory usage.

        Enables all memory-saving features with aggressive cache clearing.
        """
        return cls(
            offload_to_cpu=True,
            use_mixed_precision=True,
            empty_cache_every=10,
            track_memory=True,
            memory_limit_mb=4 * 1024,  # Lower limit for low-memory mode
        )


class MemoryManager:
    """
    Manages memory during optimization runs.

    Provides utilities for:
    - Model offloading (move models to CPU when not in use)
    - Mixed precision inference (torch.autocast)
    - Cache clearing (torch.mps.empty_cache / torch.cuda.empty_cache)
    - Memory usage tracking and snapshots
    """

    def __init__(
        self,
        config: MemoryConfig,
        device: str,
    ) -> None:
        self._config = config
        self._device = torch.device(device)
        self._device_type = self._get_device_type()
        self._snapshots: list[dict[str, Any]] = []

    def check_memory_limit(self) -> None:
        """
        Check if memory usage exceeds the configured limit.

        Raises:
            OutOfMemoryError: If usage exceeds limit.
        """
        if self._device_type == "cuda" and torch.cuda.is_available():
            current = torch.cuda.memory_allocated() / (1024 * 1024)
        elif self._device_type == "mps" and torch.backends.mps.is_available():
            try:
                current = torch.mps.current_allocated_memory() / (1024 * 1024)
            except AttributeError:
                return
        else:
            return

        if current > self._config.memory_limit_mb:
            raise OutOfMemoryError(
                operation="memory_check",
                device=str(self._device),
                original_error=RuntimeError(
                    f"Memory limit exceeded: {current:.1f}MB > {self._config.memory_limit_mb:.1f}MB"
                ),
            )

    @property
    def config(self) -> MemoryConfig:
        """Get the memory configuration."""
        return self._config

    @property
    def device(self) -> torch.device:
        """Get the target device."""
        return self._device

    def _get_device_type(self) -> str:
        """Extract device type from device string."""
        device_str = str(self._device)
        if ":" in device_str:
            return device_str.split(":")[0]
        return device_str

    @contextmanager
    def memory_efficient(self) -> Generator[None, None, None]:
        """
        Context manager for memory-efficient operations.

        Enables autocast if use_mixed_precision is configured.
        """
        if self._config.use_mixed_precision:
            with self.autocast():
                yield
        else:
            yield

    @contextmanager
    def autocast(self) -> Generator[None, None, None]:
        """
        Context manager for automatic mixed precision.

        Uses the appropriate dtype for the device:
        - CUDA: float16
        - MPS: float16
        - CPU: bfloat16 (if supported) or no-op
        """
        if not self._config.use_mixed_precision:
            yield
            return

        if self._device_type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                yield
        elif self._device_type == "mps":
            # MPS autocast uses float16
            with torch.autocast(device_type="mps", dtype=torch.float16):
                yield
        elif self._device_type == "cpu":
            # CPU autocast - may be no-op on unsupported hardware
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                yield
        else:
            yield

    def offload_model(self, model: nn.Module) -> None:
        """
        Move a model to CPU to free GPU memory.

        Only performs the move if offload_to_cpu is configured.

        Args:
            model: The PyTorch model to offload.
        """
        if not self._config.offload_to_cpu:
            return

        model.to("cpu")

    def restore_model(self, model: nn.Module) -> None:
        """
        Move a model back to the target device.

        Args:
            model: The PyTorch model to restore.
        """
        model.to(self._device)

    @contextmanager
    def offloaded(self, model: nn.Module) -> Generator[None, None, None]:
        """
        Context manager for temporarily offloading a model to CPU.

        The model is moved to CPU on entry and restored to the
        target device on exit.

        Args:
            model: The PyTorch model to offload.

        Yields:
            None
        """
        self.offload_model(model)
        try:
            yield
        finally:
            self.restore_model(model)

    def empty_cache(self) -> None:
        """
        Clear the GPU cache to free unused memory.

        Calls the appropriate empty_cache function based on device type:
        - CUDA: torch.cuda.empty_cache()
        - MPS: torch.mps.empty_cache()
        - CPU: no-op
        """
        if self._device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif self._device_type == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def should_empty_cache(self, step: int) -> bool:
        """
        Check if cache should be emptied at this step.

        Args:
            step: The current optimization step.

        Returns:
            True if cache should be emptied, False otherwise.
        """
        if self._config.empty_cache_every == 0:
            return False

        return step % self._config.empty_cache_every == 0

    def get_memory_usage(self) -> dict[str, float | None]:
        """
        Get current memory usage information.

        Returns:
            Dictionary with memory usage in MB:
            - allocated_mb: Currently allocated memory
            - reserved_mb: Total reserved memory (including cached)
        """
        if self._device_type == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)
            return {
                "allocated_mb": allocated,
                "reserved_mb": reserved,
            }
        elif self._device_type == "mps" and torch.backends.mps.is_available():
            try:
                allocated = torch.mps.current_allocated_memory() / (1024 * 1024)
                # MPS doesn't have reserved memory tracking
                return {
                    "allocated_mb": allocated,
                    "reserved_mb": allocated,
                }
            except AttributeError:
                return {
                    "allocated_mb": None,
                    "reserved_mb": None,
                }

        return {
            "allocated_mb": None,
            "reserved_mb": None,
        }

    def record_memory_snapshot(self, label: str) -> None:
        """
        Record a memory snapshot with a label.

        Only records if track_memory is enabled.

        Args:
            label: A descriptive label for this snapshot.
        """
        if not self._config.track_memory:
            return
        
        self.check_memory_limit()

        usage = self.get_memory_usage()
        self._snapshots.append(
            {
                "label": label,
                "allocated_mb": usage["allocated_mb"],
                "reserved_mb": usage["reserved_mb"],
            }
        )

    def get_memory_snapshots(self) -> list[dict[str, Any]]:
        """
        Get all recorded memory snapshots.

        Returns:
            List of snapshot dictionaries with label and memory values.
        """
        return self._snapshots.copy()

    def get_peak_memory_mb(self) -> float | None:
        """
        Get peak memory usage in MB.

        Returns:
            Peak memory in MB, or None if not available (e.g., on CPU).
        """
        if self._device_type == "cuda" and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        elif self._device_type == "mps" and torch.backends.mps.is_available():
            try:
                # MPS doesn't have peak tracking, return current as approximation
                return torch.mps.current_allocated_memory() / (1024 * 1024)
            except AttributeError:
                return None

        return None

    def reset_peak_memory(self) -> None:
        """
        Reset peak memory tracking.

        Only has an effect on CUDA devices.
        """
        if self._device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
