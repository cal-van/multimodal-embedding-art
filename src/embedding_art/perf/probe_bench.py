"""
Benchmark helpers for PyTorch-vs-CoreML probe inference.

Provides a small, dependency-light helper that times a callable over N
forward passes (with M warmup iterations) and returns the median +
percentiles. Used by both the ``embed-art bench-probes`` command and
the unit tests.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class TimingStats:
    """Summary statistics from a benchmark run."""

    name: str
    n_iters: int
    n_warmup: int
    median_ms: float
    p10_ms: float
    p90_ms: float
    mean_ms: float
    stddev_ms: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "name": self.name,
            "n_iters": self.n_iters,
            "n_warmup": self.n_warmup,
            "median_ms": round(self.median_ms, 3),
            "p10_ms": round(self.p10_ms, 3),
            "p90_ms": round(self.p90_ms, 3),
            "mean_ms": round(self.mean_ms, 3),
            "stddev_ms": round(self.stddev_ms, 3),
        }


def time_callable(
    fn: Callable[[], object],
    *,
    name: str,
    n_warmup: int = 3,
    n_iters: int = 20,
) -> TimingStats:
    """Time ``fn`` over ``n_warmup + n_iters`` calls, return aggregates.

    Warmup calls are discarded. ``fn`` should be a zero-arg callable
    that wraps the actual forward pass (e.g. ``lambda: model(x)``).
    """
    for _ in range(n_warmup):
        fn()

    durations_ms: list[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        durations_ms.append((time.perf_counter() - t0) * 1000.0)

    durations_ms.sort()
    median = statistics.median(durations_ms)
    p10 = durations_ms[max(0, int(0.1 * n_iters) - 1)]
    p90 = durations_ms[min(n_iters - 1, int(0.9 * n_iters))]
    mean = statistics.fmean(durations_ms)
    stddev = statistics.pstdev(durations_ms) if n_iters > 1 else 0.0

    return TimingStats(
        name=name,
        n_iters=n_iters,
        n_warmup=n_warmup,
        median_ms=median,
        p10_ms=p10,
        p90_ms=p90,
        mean_ms=mean,
        stddev_ms=stddev,
    )


def speedup(baseline: TimingStats, candidate: TimingStats) -> float:
    """Return ``baseline.median / candidate.median``.

    ``> 1.0`` → candidate is faster than baseline.
    """
    if candidate.median_ms <= 0.0:
        return float("inf")
    return baseline.median_ms / candidate.median_ms
