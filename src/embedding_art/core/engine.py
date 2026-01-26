"""
Core optimization engine.

This is the main loop that optimizes a latent to maximize similarity
between its decoded output and a target concept embedding.
"""

from dataclasses import dataclass, field
from time import time
from typing import Callable

import torch
import torch.nn.functional as F
from PIL import Image
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from embedding_art.core.concept import Concept
from embedding_art.core.config import OptimizationConfig
from embedding_art.encoders.base import Encoder
from embedding_art.generators.base import Generator
from embedding_art.regularizers import CompositeRegularizer, Regularizer


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
        progress: bool = True,
    ) -> OptimizationResult:
        """
        Optimize a latent to maximize similarity with target concept.

        Args:
            target: The concept to optimize toward
            output_modality: Which generator to use ("image", "audio", "video")
            config: Optimization hyperparameters
            regularizers: Regularization losses (default: CompositeRegularizer.default_image())
            callback: Called each step with (step, loss, similarity, latent)
            progress: Show progress bar

        Returns:
            OptimizationResult with final output and metadata
        """
        config = config or OptimizationConfig()
        generator = self.get_generator(output_modality)

        # Setup regularizers
        if regularizers is None:
            regularizers = CompositeRegularizer.default_image()
        elif isinstance(regularizers, list):
            regularizers = CompositeRegularizer(regularizers)

        # Move target to device
        target_embedding = target.embedding.to(self.device)

        # Initialize latent
        latent = generator.init_latent(seed=config.seed)

        # Setup optimizer
        optimizer = self._create_optimizer(latent, config)
        scheduler = self._create_scheduler(optimizer, config)

        # Tracking
        loss_history = []
        similarity_history = []
        checkpoints = []

        start_time = time()

        # Progress bar
        progress_ctx = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            disable=not progress,
        )

        with progress_ctx as pbar:
            task = pbar.add_task("Optimizing...", total=config.steps)

            for step in range(config.steps):
                optimizer.zero_grad()

                # Decode latent to output
                decoded = generator.decode(latent)

                # Apply augmentation if configured
                augmented = self._augment(decoded, config)

                # Encode output to embedding space
                current_embedding = self._encode_for_modality(augmented, output_modality)

                # Compute similarity loss (negative because we maximize)
                similarity = F.cosine_similarity(
                    current_embedding, target_embedding, dim=-1
                ).mean()
                similarity_loss = -similarity

                # Compute regularization loss
                reg_loss = regularizers(latent, decoded)

                # Total loss
                loss = similarity_loss + reg_loss

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_([latent], max_norm=1.0)

                # Update
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                # Track metrics
                loss_val = loss.item()
                sim_val = similarity.item()
                loss_history.append(loss_val)
                similarity_history.append(sim_val)

                # Checkpointing
                if config.checkpoint_every and step % config.checkpoint_every == 0:
                    checkpoints.append((step, latent.detach().clone()))

                # Callback
                if callback is not None:
                    callback(step, loss_val, sim_val, latent)

                # Update progress bar
                pbar.update(task, advance=1, description=f"sim={sim_val:.4f}")

        elapsed = time() - start_time

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
            augmented = F.interpolate(
                augmented, size=(h, w), mode="bilinear", align_corners=False
            )

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
