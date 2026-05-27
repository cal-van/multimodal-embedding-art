"""
Progress display logic for optimization.

Encapsulates Rich visualization and simple progress reporting to decouple
presentation from the optimization engine.
"""

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"


@dataclass
class ProgressConfig:
    """Configuration for progress display during optimization."""

    enabled: bool = True
    verbose: bool = False
    show_loss_curve: bool = True
    show_memory: bool = True


def generate_sparkline(values: list[float], max_length: int = 30) -> str:
    """Generate a sparkline string from a list of values."""
    if not values:
        return ""

    if len(values) > max_length:
        step = len(values) / max_length
        sampled = [values[int(i * step)] for i in range(max_length)]
        values = sampled

    min_val = min(values)
    max_val = max(values)
    value_range = max_val - min_val

    if value_range == 0:
        return SPARKLINE_CHARS[len(SPARKLINE_CHARS) // 2] * len(values)

    result = []
    for val in values:
        normalized = (val - min_val) / value_range
        index = int(normalized * (len(SPARKLINE_CHARS) - 1))
        index = max(0, min(len(SPARKLINE_CHARS) - 1, index))
        result.append(SPARKLINE_CHARS[index])

    return "".join(result)


def get_similarity_style(current: float, previous: float | None = None) -> str:
    """Get Rich style string for similarity value based on trend."""
    if previous is None:
        return "bold white"

    delta = current - previous
    threshold = 0.001

    if delta > threshold:
        return "bold green"
    elif delta < -threshold:
        return "red"
    else:
        return "yellow dim"


def get_memory_usage_mb(device: str) -> float | None:
    """Get current memory usage in MB for GPU devices."""
    device_type = device.split(":")[0] if ":" in device else device

    if device_type == "cuda" and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    elif device_type == "mps" and torch.backends.mps.is_available():
        try:
            return torch.mps.current_allocated_memory() / (1024 * 1024)
        except AttributeError:
            return None
    return None


class ProgressObserver(Protocol):
    """Protocol for observing optimization progress."""

    def on_start(self, total_steps: int, description: str = "Optimizing...") -> None: ...

    def on_step(
        self,
        step: int,
        loss: float,
        similarity: float,
        lr: float | None = None,
        loss_history: list[float] | None = None,
        similarity_history: list[float] | None = None,
        device: str | None = None,
    ) -> None: ...

    def on_finish(self) -> None: ...


class NoOpProgressObserver:
    """Observer that does nothing."""

    def on_start(self, total_steps: int, description: str = "Optimizing...") -> None:
        pass

    def on_step(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_finish(self) -> None:
        pass


class SimpleProgressObserver:
    """Simple progress bar using Rich."""

    def __init__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        )
        self.task_id = None
        self.total_steps = 0

    def on_start(self, total_steps: int, description: str = "Optimizing...") -> None:
        self.total_steps = total_steps
        self.progress.start()
        self.task_id = self.progress.add_task(description, total=total_steps)

    def on_step(
        self,
        step: int,
        loss: float,
        similarity: float,
        lr: float | None = None,
        loss_history: list[float] | None = None,
        similarity_history: list[float] | None = None,
        device: str | None = None,
    ) -> None:
        self.progress.update(self.task_id, completed=step + 1, description=f"sim={similarity:.4f}")

    def on_finish(self) -> None:
        self.progress.stop()


class VerboseProgressObserver:
    """Rich live display with metrics and sparklines."""

    def __init__(self, config: ProgressConfig):
        self.config = config
        self.console = Console()
        self.live = Live(console=self.console, refresh_per_second=4)
        self.total_steps = 0
        self.description = ""

    def on_start(self, total_steps: int, description: str = "Optimizing...") -> None:
        self.total_steps = total_steps
        self.description = description
        self.live.start()

    def on_finish(self) -> None:
        self.live.stop()

    def on_step(
        self,
        step: int,
        loss: float,
        similarity: float,
        lr: float | None = None,
        loss_history: list[float] | None = None,
        similarity_history: list[float] | None = None,
        device: str | None = None,
    ) -> None:
        display = self._build_display(
            step=step + 1,
            total_steps=self.total_steps,
            sim_val=similarity,
            loss_val=loss,
            lr=lr,
            loss_history=loss_history,
            similarity_history=similarity_history,
            device=device,
        )
        self.live.update(Panel(display, title="Optimization Progress", border_style="blue"))

    def _build_display(
        self,
        step: int,
        total_steps: int,
        sim_val: float,
        loss_val: float,
        lr: float | None,
        loss_history: list[float] | None,
        similarity_history: list[float] | None,
        device: str | None,
    ) -> Group:
        progress_bar = Progress(
            SpinnerColumn(),
            TextColumn(f"[bold]{self.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        )
        progress_bar.add_task(self.description, completed=step, total=total_steps)

        metrics_table = Table.grid(padding=(0, 2))
        metrics_table.add_column("Label", style="dim")
        metrics_table.add_column("Value")

        prev_sim = None
        if similarity_history and len(similarity_history) >= 2:
            prev_sim = similarity_history[-2]

        sim_style = get_similarity_style(sim_val, prev_sim)
        sim_text = Text(f"{sim_val:.4f}", style=sim_style)
        metrics_table.add_row("Similarity:", sim_text)
        metrics_table.add_row("Loss:", f"{loss_val:.4f}")

        if lr is not None:
            metrics_table.add_row("Learning Rate:", f"{lr:.2e}")

        if self.config.show_memory and device:
            mem_mb = get_memory_usage_mb(device)
            if mem_mb is not None:
                metrics_table.add_row("Memory:", f"{mem_mb:.1f} MB")

        elements: list = [progress_bar, metrics_table]

        if self.config.show_loss_curve and loss_history:
            sparkline = generate_sparkline(loss_history)
            loss_curve_text = Text(f"Loss: {sparkline}", style="cyan")
            elements.append(loss_curve_text)

            if similarity_history:
                sim_sparkline = generate_sparkline(similarity_history)
                sim_curve_text = Text(f"Sim:  {sim_sparkline}", style="green")
                elements.append(sim_curve_text)

        return Group(*elements)
