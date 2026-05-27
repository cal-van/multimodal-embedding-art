"""
Core optimization engine.

This is the main loop that optimizes a latent to maximize similarity
between its decoded output and a target concept embedding.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import time
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from embedding_art.core.concept import Concept
from embedding_art.core.config import OptimizationConfig
from embedding_art.core.memory import MemoryConfig, MemoryManager
from embedding_art.core.progress import (
    NoOpProgressObserver,
    ProgressConfig,
    ProgressObserver,
    SimpleProgressObserver,
    VerboseProgressObserver,
)
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
            img = decoded[0].detach().cpu().permute(1, 2, 0).numpy()
            img_array = (img * 255).clip(0, 255).astype("uint8")
            return Image.fromarray(img_array)

    def get_final_audio(self, generator: Generator) -> tuple[int, Any]:
        """
        Decode final latent to audio.

        Returns:
            (sample_rate, waveform_numpy)
        """
        with torch.no_grad():
            audio = generator.decode(self.final_latent)
            # [B, Samples] -> [Samples]
            audio_np = audio[0].cpu().numpy()
            return generator.SAMPLE_RATE, audio_np

    def get_final_video(self, generator: Generator) -> list[Image.Image]:
        """
        Decode final latent to video frames.

        Returns:
            List of PIL frames in order.
        """
        with torch.no_grad():
            decoded = generator.decode(self.final_latent)

        if decoded.ndim != 5:
            raise ValueError("Decoded video must have shape [B, F, C, H, W]")

        frames = []
        for frame in decoded[0].detach().cpu():
            frame = frame.permute(1, 2, 0).clamp(0, 1)
            frame_np = (frame * 255).to(torch.uint8).numpy()
            frames.append(Image.fromarray(frame_np))

        return frames


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
        encoder: Encoder | None,
        device: str = "mps",
    ):
        self.encoder = encoder
        self.device = torch.device(device)
        self._generators: dict[str, Generator] = {}

    # ------------------------------------------------------------------
    # v2 factory
    # ------------------------------------------------------------------

    @classmethod
    def from_registry(
        cls,
        registry: Any,
        default_encoder: str | None = None,
        device: str = "mps",
    ) -> "EmbeddingArtEngine":
        """Create an engine from an EncoderRegistry.

        Args:
            registry: An ``EncoderRegistry`` instance.
            default_encoder: Name of the encoder to pre-load as ``self.encoder``.
                When ``None``, ``self.encoder`` is left as ``None`` and callers
                must pass ``encoder_name`` to :meth:`render`.
            device: PyTorch device string.

        Returns:
            Configured ``EmbeddingArtEngine`` with ``_registry`` set.
        """
        encoder = None
        if default_encoder is not None:
            encoder = registry.load(default_encoder)
        instance = cls(encoder=encoder, device=device)
        instance._registry = registry
        return instance

    # ------------------------------------------------------------------
    # v2 rendering API
    # ------------------------------------------------------------------

    def render(
        self,
        spec: Any,
        encoder_name: str | None = None,
        output_modality: str = "image",
        strategy: Any = None,
        config: OptimizationConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        """Render a ConceptSpec (or an already-resolved Concept) via the v2 pipeline.

        The v2 pipeline delegates the optimization loop to an
        :class:`~embedding_art.core.strategies.OptimizationStrategy`, keeping the
        engine itself thin.

        Args:
            spec: A :class:`~embedding_art.core.concept_spec.ConceptSpec` **or** an
                already-resolved :class:`~embedding_art.core.concept.Concept`.
            encoder_name: Name of the encoder to load from ``self._registry``.
                When ``None``, ``self.encoder`` is used as the fallback.
            output_modality: Generator modality key (e.g. ``"image"``).
            strategy: An object satisfying the ``RenderingStrategy`` protocol.
                Defaults to a freshly constructed ``OptimizationStrategy``.
            config: Optimization hyper-parameters.  Defaults to
                ``OptimizationConfig()`` when ``None``.
            **kwargs: Extra arguments forwarded to the strategy's ``render`` method.

        Returns:
            :class:`~embedding_art.core.render_result.RenderResult` produced by
            the strategy.

        Raises:
            ValueError: When no encoder is available (neither ``encoder_name``
                resolves via the registry nor ``self.encoder`` is set).
        """
        from embedding_art.core.concept_spec import ConceptSpec
        from embedding_art.core.strategies import OptimizationStrategy

        config = config or OptimizationConfig()

        # Resolve encoder
        if encoder_name is not None and hasattr(self, "_registry"):
            encoder = self._registry.load(encoder_name)
        elif self.encoder is not None:
            encoder = self.encoder
        else:
            raise ValueError(
                "No encoder available. Provide encoder_name or set a default encoder "
                "via from_registry(default_encoder=...) or by passing encoder to __init__."
            )

        # Materialize concept from spec when needed
        if isinstance(spec, ConceptSpec) and hasattr(encoder, "encode"):
            target = encoder.encode(spec)
        else:
            target = spec  # Already a Concept

        generator = self.get_generator(output_modality)

        if strategy is None:
            strategy = OptimizationStrategy()

        return strategy.render(target, generator, encoder, config, **kwargs)

    def render_compare(
        self,
        spec: Any,
        encoder_names: list[str],
        output_modality: str = "image",
        config: OptimizationConfig | None = None,
        sequential: bool = False,
    ) -> dict[str, Any]:
        """Render the same concept with multiple encoders for comparison.

        Each encoder produces an independent :class:`~embedding_art.core.render_result.RenderResult`,
        useful for studying how different encoders perceive the same concept.

        Args:
            spec: Concept specification or resolved Concept.
            encoder_names: List of registry encoder names to compare.
            output_modality: Generator modality key.
            config: Shared optimization configuration for all renders.
            sequential: Reserved for future parallel execution control.
                Currently all renders are sequential regardless of this flag.

        Returns:
            Dict mapping each encoder name to its ``RenderResult``.
        """
        results: dict[str, Any] = {}
        for name in encoder_names:
            results[name] = self.render(
                spec,
                encoder_name=name,
                output_modality=output_modality,
                config=config,
            )
        return results

    def render_interpolation(
        self,
        spec_a: Any,
        spec_b: Any,
        steps: int = 10,
        encoder_name: str | None = None,
        output_modality: str = "image",
        config: OptimizationConfig | None = None,
    ) -> list[Any]:
        """Render a slerp interpolation between two concepts.

        Constructs ``steps`` evenly-spaced interpolation points between the
        two concepts using spherical linear interpolation (slerp) and renders
        each one independently via :meth:`render`.  The endpoints (t=0 and
        t=1) are always included.

        Args:
            spec_a: Starting concept — a ``ConceptSpec`` or resolved
                ``Concept``.
            spec_b: Ending concept — a ``ConceptSpec`` or resolved
                ``Concept``.
            steps: Number of interpolation points (including both endpoints).
                Must be >= 1.  When ``steps=1`` the midpoint (t=0.5) is used.
            encoder_name: Name of the encoder to load from
                ``self._registry``.  Falls back to ``self.encoder`` when
                ``None``.
            output_modality: Generator modality key (e.g. ``"image"``).
            config: Shared optimization configuration for all renders.
                Defaults to ``OptimizationConfig()`` when ``None``.

        Returns:
            List of ``RenderResult`` objects, one per interpolation step, in
            order from ``spec_a`` (t=0) to ``spec_b`` (t=1).

        Raises:
            ValueError: When no encoder is available.
        """
        from embedding_art.core.concept import Concept
        from embedding_art.core.concept_spec import ConceptSpec
        from embedding_art.core.strategies import OptimizationStrategy

        config = config or OptimizationConfig()

        # Resolve encoder
        if encoder_name is not None and hasattr(self, "_registry"):
            encoder = self._registry.load(encoder_name)
        elif self.encoder is not None:
            encoder = self.encoder
        else:
            raise ValueError(
                "No encoder available. Provide encoder_name or set a default encoder "
                "via from_registry(default_encoder=...) or by passing encoder to __init__."
            )

        # Materialize concepts from specs when needed
        if isinstance(spec_a, ConceptSpec) and hasattr(encoder, "encode"):
            concept_a = encoder.encode(spec_a)
        else:
            concept_a = spec_a

        if isinstance(spec_b, ConceptSpec) and hasattr(encoder, "encode"):
            concept_b = encoder.encode(spec_b)
        else:
            concept_b = spec_b

        generator = self.get_generator(output_modality)
        strategy = OptimizationStrategy()
        results = []

        for i in range(steps):
            t = i / max(steps - 1, 1)
            interpolated = Concept.slerp(concept_a, concept_b, t)
            result = strategy.render(interpolated, generator, encoder, config)
            results.append(result)

        return results

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
        memory_config: MemoryConfig | None = None,
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
            memory_config: Memory management configuration for optimization.

        Returns:
            OptimizationResult with final output and metadata
        """
        import warnings

        warnings.warn(
            "EmbeddingArtEngine.optimize() is deprecated. Use engine.render() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        config = config or OptimizationConfig()
        generator = self.get_generator(output_modality)

        # Check if generator supports direct generation (Diffusion Pipelines)
        if hasattr(generator, "generate"):
            return self._run_diffusion_generation(
                generator=generator,
                target=target,
                output_modality=output_modality,
                config=config,
                regularizers=regularizers,
                callback=callback,
            )

        # Setup optimization components
        memory_manager: MemoryManager | None = None
        if memory_config is not None:
            memory_manager = MemoryManager(config=memory_config, device=str(self.device))

        progress_config = self._normalize_progress_config(progress)

        # Setup progress observer
        observer: ProgressObserver
        if not progress_config.enabled:
            observer = NoOpProgressObserver()
        elif progress_config.verbose:
            observer = VerboseProgressObserver(progress_config)
        else:
            observer = SimpleProgressObserver()

        # Setup regularizers
        if regularizers is None:
            if output_modality == "audio":
                regularizers = CompositeRegularizer.default_audio()
            elif output_modality == "video":
                regularizers = CompositeRegularizer.default_video()
            else:
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

        observer.on_start(total_steps=config.steps)

        try:
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
                    checkpoint_dir=checkpoint_path,
                    memory_manager=memory_manager,
                )

                current_lr = None
                if scheduler is not None:
                    current_lr = scheduler.get_last_lr()[0]

                observer.on_step(
                    step=step,
                    loss=loss_val,
                    similarity=sim_val,
                    lr=current_lr,
                    loss_history=loss_history,
                    similarity_history=similarity_history,
                    device=str(self.device),
                )
        finally:
            observer.on_finish()

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

    # Constants
    DEFAULT_CFG_SCALE = 7.5
    DEFAULT_STEPS = 30
    DEFAULT_DUMMY_LATENT_SHAPE = (1, 4, 64, 64)

    def _run_diffusion_generation(
        self,
        generator: Generator,
        target: Concept,
        output_modality: str,
        config: OptimizationConfig,
        regularizers: Any,
        callback: Callable | None,
    ) -> OptimizationResult:
        """Run generation using a diffusion pipeline (SDXL, AudioLDM, SVD)."""
        prompt = target.text_source or target.description
        if target.text_source is None:
            # If no text source (e.g. audio concept), use description
            prompt = target.description

        # Start tracking
        loss_history: list[float] = []
        similarity_history: list[float] = []
        start_time = time()

        # Helper for callback adaptation
        def generation_callback(step: int, loss: float, sim: float, latent: torch.Tensor | None):
            if callback:
                # Provide empty tensor if latent is None to match signature
                callback(step, loss, sim, latent if latent is not None else torch.tensor([]))

        # Check ImageBind configuration
        imagebind_scale = getattr(config, "guidance_scale", 0.0)

        # If ImageBind guidance is active, we want PURE concept visualization.
        # We disable the Text Prompt (human bias) and Text CFG to let ImageBind drive.
        if imagebind_scale > 0:
            prompt = ""
            text_guidance_scale = 0.0  # Disable standard CFG
        else:
            text_guidance_scale = self.DEFAULT_CFG_SCALE

        params = {
            "prompt": prompt,
            "num_inference_steps": config.steps if config.steps > 0 else self.DEFAULT_STEPS,
            "guidance_scale": text_guidance_scale,
            "callback": generation_callback,
            "imagebind_encoder": self.encoder,
            "target_embedding": target.embedding.to(self.device),
            "imagebind_guidance_scale": imagebind_scale,
            "regularizers": regularizers,
            "normalize_gradients": getattr(config, "normalize_gradients", False),
        }

        # Call generator
        final_image = generator.generate(**params)

        elapsed = time() - start_time

        # Convert PIL to Tensor [1, 3, H, W]
        img_tensor = TF.to_tensor(final_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            final_embedding = self._encode_for_modality(img_tensor, output_modality)
            final_similarity = (
                F.cosine_similarity(final_embedding, target.embedding.to(self.device), dim=-1)
                .mean()
                .item()
            )

        # VAE Encode result to get a valid latent if possible
        if hasattr(generator, "pipeline") and hasattr(generator.pipeline, "vae"):
            # Scale to [-1, 1] for VAE
            img_normalized = img_tensor * 2.0 - 1.0
            latent_dist = generator.pipeline.vae.encode(img_normalized).latent_dist
            latent = latent_dist.sample() * generator.pipeline.vae.config.scaling_factor
        else:
            latent = torch.zeros(self.DEFAULT_DUMMY_LATENT_SHAPE)  # Dummy

        return OptimizationResult(
            final_latent=latent,
            final_embedding=final_embedding.detach(),
            target_embedding=target.embedding.detach(),
            final_similarity=final_similarity,
            loss_history=loss_history,
            similarity_history=similarity_history,
            elapsed_seconds=elapsed,
            config=config,
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

        # Audio or other non-spatial modalities shouldn't be augmented with spatial ops
        # ImageBind audio input is [B, T], images are [B, C, H, W], video is [B, F, C, H, W]
        if decoded.ndim not in (4, 5):
            return decoded

        augmented = decoded
        is_video = decoded.ndim == 5

        if aug_config.random_crop:
            # Random crop and resize back
            scale = torch.empty(1).uniform_(*aug_config.crop_scale).item()
            h, w = decoded.shape[-2:]
            crop_h, crop_w = int(h * scale), int(w * scale)

            top = torch.randint(0, h - crop_h + 1, (1,)).item()
            left = torch.randint(0, w - crop_w + 1, (1,)).item()

            if is_video:
                augmented = augmented[:, :, :, top : top + crop_h, left : left + crop_w]
                batch_size, num_frames, channels, _, _ = augmented.shape
                augmented = augmented.reshape(batch_size * num_frames, channels, crop_h, crop_w)
                augmented = F.interpolate(
                    augmented, size=(h, w), mode="bilinear", align_corners=False
                )
                augmented = augmented.reshape(batch_size, num_frames, channels, h, w)
            else:
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
        return torch.load(path, weights_only=True)

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
        memory_manager: MemoryManager | None = None,
    ) -> tuple[float, float]:
        """Execute a single optimization step."""
        optimizer.zero_grad()

        # Use memory-efficient context if configured
        if memory_manager is not None:
            with memory_manager.memory_efficient():
                decoded = generator.decode(latent)
                augmented = self._augment(decoded, config)
                current_embedding = self._encode_for_modality(augmented, output_modality)
        else:
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

        if memory_manager is not None:
            memory_manager.check_memory_limit()
            if memory_manager.should_empty_cache(step):
                memory_manager.empty_cache()

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
