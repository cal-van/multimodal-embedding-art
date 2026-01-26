"""
Core optimization engine.

This is the main loop that optimizes a latent to maximize similarity
between its decoded output and a target concept embedding.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import time
from typing import Any, Callable

import torch
import torch.nn.functional as F
from PIL import Image
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

from embedding_art.core.concept import Concept
from embedding_art.core.config import OptimizationConfig
from embedding_art.encoders.base import Encoder
from embedding_art.generators.base import Generator
from embedding_art.regularizers import CompositeRegularizer, Regularizer

SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"


@dataclass
class ProgressConfig:
    """Configuration for progress display during optimization."""

    enabled: bool = True
    verbose: bool = False
    show_loss_curve: bool = True
    show_memory: bool = True


def generate_sparkline(values: list[float], max_length: int = 30) -> str:
    """
    Generate a sparkline string from a list of values.

    Args:
        values: List of numeric values to visualize.
        max_length: Maximum length of the sparkline.

    Returns:
        A string of sparkline characters representing the values.
    """
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
    """
    Get Rich style string for similarity value based on trend.

    Args:
        current: Current similarity value.
        previous: Previous similarity value (for trend detection).

    Returns:
        A Rich style string (e.g., "bold green", "yellow", "red").
    """
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
    """
    Get current memory usage in MB for GPU devices.

    Args:
        device: The device string ("cpu", "mps", "cuda", etc.).

    Returns:
        Memory usage in MB, or None if not available/applicable.
    """
    device_type = device.split(":")[0] if ":" in device else device

    if device_type == "cuda" and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    elif device_type == "mps" and torch.backends.mps.is_available():
        try:
            return torch.mps.current_allocated_memory() / (1024 * 1024)
        except AttributeError:
            return None
    return None


@dataclass
class OptimizationResult:
    """Result of an optimization run."""

    # Final outputs
    final_latent: torch.Tensor
    final_embedding: torch.Tensor
    target_embedding: torch.Tensor

    # Metrics
    final_similarity: float
    loss_history: list[float] = field(default_factory=list)
    similarity_history: list[float] = field(default_factory=list)

    # Checkpoints (step, latent tensor)
    checkpoints: list[tuple[int, torch.Tensor]] = field(default_factory=list)

    # Metadata
    config: OptimizationConfig | None = None
    elapsed_seconds: float = 0.0

    def get_final_image(self, generator: Generator) -> Image.Image:
        """Decode final latent to PIL Image."""
        with torch.no_grad():
            decoded = generator.decode(self.final_latent)
            # Convert to PIL
            img_array = decoded[0].permute(1, 2, 0).cpu().numpy()
            img_array = (img_array * 255).clip(0, 255).astype("uint8")
            return Image.fromarray(img_array)


class EmbeddingArtEngine:
    """
    Main engine for optimizing outputs toward concept embeddings.

    Usage:
        engine = EmbeddingArtEngine(encoder)
        engine.register_generator("image", image_generator)

        target = Concept.from_text("goldfish", encoder)
        result = engine.optimize(target, "image")

        result.get_final_image(image_generator).save("goldfish.png")
    """

    def __init__(
        self,
        encoder: Encoder,
        device: str = "mps",
    ):
        self.encoder = encoder
        self.device = torch.device(device)
        self._generators: dict[str, Generator] = {}

    def register_generator(self, name: str, generator: Generator) -> None:
        """Register a generator for a modality."""
        self._generators[name] = generator

    def get_generator(self, name: str) -> Generator:
        """Get a registered generator."""
        if name not in self._generators:
            available = list(self._generators.keys())
            raise KeyError(f"Generator '{name}' not found. Available: {available}")
        return self._generators[name]

    def optimize(
        self,
        target: Concept,
        output_modality: str,
        config: OptimizationConfig | None = None,
        regularizers: list[Regularizer] | CompositeRegularizer | None = None,
        callback: Callable[[int, float, float, torch.Tensor], None] | None = None,
        progress: bool | ProgressConfig = True,
        checkpoint_dir: Path | str | None = None,
        resume_from: Path | str | None = None,
    ) -> OptimizationResult:
        """
        Optimize a latent to maximize similarity with target concept.

        Args:
            target: The concept to optimize toward
            output_modality: Which generator to use ("image", "audio", "video")
            config: Optimization hyperparameters
            regularizers: Regularization losses (default: CompositeRegularizer.default_image())
            callback: Called each step with (step, loss, similarity, latent)
            progress: Show progress bar. Can be bool or ProgressConfig for fine-grained control.
            checkpoint_dir: Directory to save checkpoints. If None, no checkpoints are saved to disk.
            resume_from: Path to a checkpoint file to resume from.

        Returns:
            OptimizationResult with final output and metadata
        """
        config = config or OptimizationConfig()
        generator = self.get_generator(output_modality)

        progress_config = self._normalize_progress_config(progress)

        # Setup regularizers
        if regularizers is None:
            regularizers = CompositeRegularizer.default_image()
        elif isinstance(regularizers, list):
            regularizers = CompositeRegularizer(regularizers)

        # Move target to device
        target_embedding = target.embedding.to(self.device)

        # Initialize latent and optimizer
        start_step = 0

        if resume_from is not None:
            # Resume from checkpoint
            checkpoint = self._load_checkpoint(resume_from)
            latent = checkpoint["latent"].to(self.device).requires_grad_(True)
            start_step = checkpoint["step"]
            optimizer = self._create_optimizer(latent, config)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        else:
            # Fresh start
            latent = generator.init_latent(seed=config.seed)
            optimizer = self._create_optimizer(latent, config)

        scheduler = self._create_scheduler(optimizer, config)

        # Advance scheduler to correct position if resuming
        # The warning about calling scheduler.step() before optimizer.step() is expected
        # here since we're fast-forwarding the scheduler to match the checkpoint position
        if resume_from is not None and scheduler is not None:
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`",
                )
                for _ in range(start_step):
                    scheduler.step()

        # Convert checkpoint_dir to Path if provided
        checkpoint_path: Path | None = None
        if checkpoint_dir is not None:
            checkpoint_path = Path(checkpoint_dir)

        # Tracking
        loss_history: list[float] = []
        similarity_history: list[float] = []
        checkpoints: list[tuple[int, torch.Tensor]] = []

        start_time = time()

        if progress_config.enabled and progress_config.verbose:
            elapsed = self._run_verbose_optimization(
                latent=latent,
                optimizer=optimizer,
                scheduler=scheduler,
                generator=generator,
                regularizers=regularizers,
                target_embedding=target_embedding,
                output_modality=output_modality,
                config=config,
                progress_config=progress_config,
                loss_history=loss_history,
                similarity_history=similarity_history,
                checkpoints=checkpoints,
                callback=callback,
                start_time=start_time,
                checkpoint_dir=checkpoint_path,
                start_step=start_step,
            )
        else:
            elapsed = self._run_simple_optimization(
                latent=latent,
                optimizer=optimizer,
                scheduler=scheduler,
                generator=generator,
                regularizers=regularizers,
                target_embedding=target_embedding,
                output_modality=output_modality,
                config=config,
                progress_config=progress_config,
                loss_history=loss_history,
                similarity_history=similarity_history,
                checkpoints=checkpoints,
                callback=callback,
                start_time=start_time,
                checkpoint_dir=checkpoint_path,
                start_step=start_step,
            )

        # Final encoding (without augmentation)
        with torch.no_grad():
            decoded = generator.decode(latent)
            final_embedding = self._encode_for_modality(decoded, output_modality)
            final_similarity = (
                F.cosine_similarity(final_embedding, target_embedding, dim=-1).mean().item()
            )

        return OptimizationResult(
            final_latent=latent.detach(),
            final_embedding=final_embedding.detach(),
            target_embedding=target_embedding.detach(),
            final_similarity=final_similarity,
            loss_history=loss_history,
            similarity_history=similarity_history,
            checkpoints=checkpoints,
            config=config,
            elapsed_seconds=elapsed,
        )

    def _encode_for_modality(
        self,
        output: torch.Tensor,
        modality: str,
    ) -> torch.Tensor:
        """Encode output tensor back through the encoder."""
        if modality == "image":
            # ImageBind expects specific preprocessing
            if hasattr(self.encoder, "encode_for_optimization"):
                return self.encoder.encode_for_optimization(output)
            else:
                return self.encoder.encode_image(output)
        elif modality == "audio":
            return self.encoder.encode_audio(output)
        elif modality == "video":
            return self.encoder.encode_video(output)
        else:
            raise ValueError(f"Unknown modality: {modality}")

    def _augment(
        self,
        decoded: torch.Tensor,
        config: OptimizationConfig,
    ) -> torch.Tensor:
        """Apply augmentation during optimization."""
        aug_config = config.augmentation

        if not aug_config.random_crop and not aug_config.random_flip:
            return decoded

        augmented = decoded

        if aug_config.random_crop:
            # Random crop and resize back
            scale = torch.empty(1).uniform_(*aug_config.crop_scale).item()
            h, w = decoded.shape[-2:]
            crop_h, crop_w = int(h * scale), int(w * scale)

            top = torch.randint(0, h - crop_h + 1, (1,)).item()
            left = torch.randint(0, w - crop_w + 1, (1,)).item()

            augmented = augmented[:, :, top : top + crop_h, left : left + crop_w]
            augmented = F.interpolate(augmented, size=(h, w), mode="bilinear", align_corners=False)

        if aug_config.random_flip and torch.rand(1).item() > 0.5:
            augmented = torch.flip(augmented, dims=[-1])

        return augmented

    def _create_optimizer(
        self,
        latent: torch.Tensor,
        config: OptimizationConfig,
    ) -> torch.optim.Optimizer:
        """Create optimizer for latent."""
        if config.optimizer == "adam":
            return torch.optim.Adam([latent], lr=config.learning_rate)
        elif config.optimizer == "adamw":
            return torch.optim.AdamW([latent], lr=config.learning_rate)
        elif config.optimizer == "sgd":
            return torch.optim.SGD([latent], lr=config.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")

    def _create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizationConfig,
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Create learning rate scheduler."""
        if config.scheduler == "constant":
            return None
        elif config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.steps, eta_min=config.learning_rate * 0.01
            )
        elif config.scheduler == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=config.steps,
            )
        else:
            raise ValueError(f"Unknown scheduler: {config.scheduler}")

    def _normalize_progress_config(
        self,
        progress: bool | ProgressConfig,
    ) -> ProgressConfig:
        """Convert bool or ProgressConfig to ProgressConfig."""
        if isinstance(progress, ProgressConfig):
            return progress
        return ProgressConfig(enabled=progress)

    def _save_checkpoint(
        self,
        checkpoint_dir: Path,
        step: int,
        latent: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        config: OptimizationConfig,
    ) -> None:
        """Save a checkpoint to disk."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"checkpoint_step_{step:04d}.pt"

        checkpoint_data: dict[str, Any] = {
            "latent": latent.detach().clone(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "config": asdict(config),
        }
        torch.save(checkpoint_data, checkpoint_path)

    def _load_checkpoint(self, checkpoint_path: Path | str) -> dict[str, Any]:
        """Load a checkpoint from disk."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return torch.load(path, weights_only=False)

    def _run_simple_optimization(
        self,
        latent: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        generator: Generator,
        regularizers: CompositeRegularizer,
        target_embedding: torch.Tensor,
        output_modality: str,
        config: OptimizationConfig,
        progress_config: ProgressConfig,
        loss_history: list[float],
        similarity_history: list[float],
        checkpoints: list[tuple[int, torch.Tensor]],
        callback: Callable[[int, float, float, torch.Tensor], None] | None,
        start_time: float,
        checkpoint_dir: Path | None = None,
        start_step: int = 0,
    ) -> float:
        """Run optimization with simple progress bar."""
        progress_ctx = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            disable=not progress_config.enabled,
        )

        remaining_steps = config.steps - start_step

        with progress_ctx as pbar:
            task = pbar.add_task("Optimizing...", total=remaining_steps)

            for step in range(start_step, config.steps):
                loss_val, sim_val = self._optimization_step(
                    step=step,
                    latent=latent,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    generator=generator,
                    regularizers=regularizers,
                    target_embedding=target_embedding,
                    output_modality=output_modality,
                    config=config,
                    loss_history=loss_history,
                    similarity_history=similarity_history,
                    checkpoints=checkpoints,
                    callback=callback,
                    checkpoint_dir=checkpoint_dir,
                )

                pbar.update(task, advance=1, description=f"sim={sim_val:.4f}")

        return time() - start_time

    def _run_verbose_optimization(
        self,
        latent: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        generator: Generator,
        regularizers: CompositeRegularizer,
        target_embedding: torch.Tensor,
        output_modality: str,
        config: OptimizationConfig,
        progress_config: ProgressConfig,
        loss_history: list[float],
        similarity_history: list[float],
        checkpoints: list[tuple[int, torch.Tensor]],
        callback: Callable[[int, float, float, torch.Tensor], None] | None,
        start_time: float,
        checkpoint_dir: Path | None = None,
        start_step: int = 0,
    ) -> float:
        """Run optimization with verbose Rich display."""
        console = Console()
        device_str = str(self.device)

        def build_display(
            step: int,
            total_steps: int,
            sim_val: float,
            loss_val: float,
            lr: float | None,
        ) -> Group:
            """Build the Rich display group."""
            progress_bar = Progress(
                SpinnerColumn(),
                TextColumn("[bold]{task.description}"),
                BarColumn(bar_width=40),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            )
            progress_bar.add_task("Optimizing...", completed=step, total=total_steps)

            metrics_table = Table.grid(padding=(0, 2))
            metrics_table.add_column("Label", style="dim")
            metrics_table.add_column("Value")

            prev_sim = similarity_history[-2] if len(similarity_history) >= 2 else None
            sim_style = get_similarity_style(sim_val, prev_sim)
            sim_text = Text(f"{sim_val:.4f}", style=sim_style)
            metrics_table.add_row("Similarity:", sim_text)
            metrics_table.add_row("Loss:", f"{loss_val:.4f}")

            if lr is not None:
                metrics_table.add_row("Learning Rate:", f"{lr:.2e}")

            if progress_config.show_memory:
                mem_mb = get_memory_usage_mb(device_str)
                if mem_mb is not None:
                    metrics_table.add_row("Memory:", f"{mem_mb:.1f} MB")

            elements: list = [progress_bar, metrics_table]

            if progress_config.show_loss_curve and loss_history:
                sparkline = generate_sparkline(loss_history)
                loss_curve_text = Text(f"Loss: {sparkline}", style="cyan")
                elements.append(loss_curve_text)

                if similarity_history:
                    sim_sparkline = generate_sparkline(similarity_history)
                    sim_curve_text = Text(f"Sim:  {sim_sparkline}", style="green")
                    elements.append(sim_curve_text)

            return Group(*elements)

        with Live(console=console, refresh_per_second=4) as live:
            for step in range(start_step, config.steps):
                loss_val, sim_val = self._optimization_step(
                    step=step,
                    latent=latent,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    generator=generator,
                    regularizers=regularizers,
                    target_embedding=target_embedding,
                    output_modality=output_modality,
                    config=config,
                    loss_history=loss_history,
                    similarity_history=similarity_history,
                    checkpoints=checkpoints,
                    callback=callback,
                    checkpoint_dir=checkpoint_dir,
                )

                lr = None
                if scheduler is not None:
                    lr = scheduler.get_last_lr()[0]

                display = build_display(
                    step=step + 1,
                    total_steps=config.steps,
                    sim_val=sim_val,
                    loss_val=loss_val,
                    lr=lr,
                )
                live.update(Panel(display, title="Optimization Progress", border_style="blue"))

        return time() - start_time

    def _optimization_step(
        self,
        step: int,
        latent: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        generator: Generator,
        regularizers: CompositeRegularizer,
        target_embedding: torch.Tensor,
        output_modality: str,
        config: OptimizationConfig,
        loss_history: list[float],
        similarity_history: list[float],
        checkpoints: list[tuple[int, torch.Tensor]],
        callback: Callable[[int, float, float, torch.Tensor], None] | None,
        checkpoint_dir: Path | None = None,
    ) -> tuple[float, float]:
        """Execute a single optimization step."""
        optimizer.zero_grad()

        decoded = generator.decode(latent)
        augmented = self._augment(decoded, config)
        current_embedding = self._encode_for_modality(augmented, output_modality)

        similarity = F.cosine_similarity(current_embedding, target_embedding, dim=-1).mean()
        similarity_loss = -similarity
        reg_loss = regularizers(latent, decoded)
        loss = similarity_loss + reg_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_([latent], max_norm=1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_val = loss.item()
        sim_val = similarity.item()
        loss_history.append(loss_val)
        similarity_history.append(sim_val)

        if config.checkpoint_every and step % config.checkpoint_every == 0:
            checkpoints.append((step, latent.detach().clone()))
            # Save to disk if checkpoint_dir is provided
            if checkpoint_dir is not None:
                self._save_checkpoint(checkpoint_dir, step, latent, optimizer, config)

        if callback is not None:
            callback(step, loss_val, sim_val, latent)

        return loss_val, sim_val

    def interpolation_series(
        self,
        concept_a: Concept,
        concept_b: Concept,
        output_modality: str,
        steps: int = 10,
        config: OptimizationConfig | None = None,
        progress: bool = True,
    ) -> list[OptimizationResult]:
        """
        Generate outputs along interpolation between two concepts.

        Args:
            concept_a: Starting concept
            concept_b: Ending concept
            output_modality: Which generator to use
            steps: Number of interpolation points
            config: Optimization config for each point

        Returns:
            List of OptimizationResult for each interpolation point
        """
        results = []

        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0.5
            interpolated = Concept.slerp(concept_a, concept_b, t)

            result = self.optimize(
                target=interpolated,
                output_modality=output_modality,
                config=config,
                progress=progress,
            )
            results.append(result)

        return results
