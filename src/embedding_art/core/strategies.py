"""
Rendering strategies for embedding optimization.

A *rendering strategy* is the algorithm that turns a target Concept into a
RenderResult — it owns the optimization loop, the optimizer, and the
scheduler.

Two public surfaces are provided:

* ``RenderingStrategy`` — a ``@runtime_checkable`` Protocol that any strategy
  must satisfy.  Callers should type-hint against this, not against concrete
  classes.
* ``OptimizationStrategy`` — the standard gradient-descent implementation.
  It supports both ``LatentGenerator`` (optimize a latent tensor) and
  ``DirectGenerator`` (optimize parameters directly held by the generator).

Design reference: docs/superpowers/specs/2026-03-19-embedding-art-v2-design.md
Section 3: Rendering Strategies
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch
import torch.nn.functional as F  # noqa: N812

from embedding_art.core.config import AugmentationConfig, OptimizationConfig
from embedding_art.core.loss import CompositeLoss
from embedding_art.core.render_result import OptimizationHistory, RenderResult

if TYPE_CHECKING:
    from embedding_art.core.concept import Concept


@runtime_checkable
class RenderingStrategy(Protocol):
    """Protocol for how we go from concept to output.

    Any object that implements ``render(target, generator, encoder, config)``
    and returns a ``RenderResult`` satisfies this protocol.
    """

    def render(
        self,
        target: Concept,
        generator: Any,
        encoder: Any,
        config: OptimizationConfig,
    ) -> RenderResult:
        """Render *target* into an output using *generator* and *encoder*.

        Args:
            target: The concept embedding to optimize toward.
            generator: A ``LatentGenerator`` or ``DirectGenerator`` instance.
            encoder: An encoder whose ``encode_for_optimization`` is called
                     each step to compute the loss.
            config: Full optimization configuration.

        Returns:
            ``RenderResult`` containing the final output tensor and full
            optimization history.
        """
        ...


class OptimizationStrategy:
    """Gradient descent on generator parameters toward a target embedding.

    Supports two generator modes detected at runtime:

    * **Latent mode** — generator has ``init_latent`` / ``decode``.  A latent
      tensor is created once, then differentiated through ``decode`` each step.
    * **Direct mode** — generator has ``get_optimizable_parameters`` /
      ``render``.  The generator's own parameters are optimized directly.

    Args:
        sae: Optional sparse autoencoder for SAE feature-space loss.  Passed
             through to ``CompositeLoss`` unchanged.
    """

    def __init__(self, sae: Any = None) -> None:
        self.sae = sae

    # ------------------------------------------------------------------
    # Public API (satisfies RenderingStrategy protocol)
    # ------------------------------------------------------------------

    def render(
        self,
        target: Concept,
        generator: Any,
        encoder: Any,
        config: OptimizationConfig,
        callback: Callable[[int, Any, torch.Tensor], None] | None = None,
        checkpoint_dir: str | Path | None = None,
        resume_from: str | Path | None = None,
    ) -> RenderResult:
        """Run the optimization loop and return the final result.

        Args:
            target: Target concept.
            generator: LatentGenerator or DirectGenerator.
            encoder: Encoder used for loss computation.
            config: Optimization configuration.
            callback: Optional callable invoked each step as
                ``callback(step, breakdown, output)``.
            checkpoint_dir: Directory in which to save checkpoint ``.pt`` files.
                Files are written when ``config.checkpoint_every > 0`` and this
                argument is not ``None``.
            resume_from: Path to a checkpoint file produced by a previous run.
                When supplied, the latent and optimizer state are restored and
                the loop starts from ``checkpoint['step'] + 1``.

        Returns:
            RenderResult with final output and full history.
        """
        loss_fn = CompositeLoss(config.loss, encoder, sae=self.sae)
        loss_fn.calibrate(target, encoder)

        # Apple Silicon perf: optionally wrap the loss callable in
        # ``torch.compile``. ``reduce-overhead`` is the recommended
        # setting on PyTorch 2.5+ MPS and gives 1.5-2.5x on the
        # optimisation hot loop. Silent fallback to the original
        # callable on platforms / PyTorch versions where compile fails.
        compile_mode = getattr(config, "compile_mode", "none")
        if compile_mode and compile_mode != "none":
            from embedding_art.perf import compile_module

            loss_fn = compile_module(loss_fn, mode=compile_mode)

        # Decide which mode to use based on available interface.
        is_direct = hasattr(generator, "get_optimizable_parameters")

        if is_direct:
            optimizable = generator.get_optimizable_parameters()
            latent = None
        else:
            latent = generator.init_latent(config.seed)
            latent.requires_grad_(True)
            optimizable = [latent]

        optimizer = self._build_optimizer(optimizable, config)
        scheduler = self._build_scheduler(optimizer, config)
        history = OptimizationHistory()

        # --- Resume from checkpoint -------------------------------------------
        start_step = 0
        if resume_from is not None:
            checkpoint = torch.load(resume_from, weights_only=True)
            if latent is not None:
                latent.data = checkpoint["latent"]
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_step = checkpoint["step"] + 1

        # --- Checkpoint directory setup ---------------------------------------
        ckpt_dir: Path | None = Path(checkpoint_dir) if checkpoint_dir is not None else None

        output: torch.Tensor | None = None

        # Resolve autocast context. fp32 is a no-op so we don't pay the
        # `torch.autocast` enter/exit cost in that case.
        autocast_ctx, autocast_dtype = self._resolve_autocast(encoder, config.autocast_dtype)

        for step in range(start_step, config.steps):
            optimizer.zero_grad()

            with autocast_ctx:
                if is_direct:
                    output = generator.render()
                else:
                    output = generator.decode(latent)  # type: ignore[arg-type]

                # Apply augmentation before loss computation.
                augmented = self._augment(output, config.augmentation)

                breakdown = loss_fn(augmented, target, encoder, latent)

            # Backward always happens outside autocast: gradients are kept
            # in fp32 by autocast machinery, but issuing .backward() inside
            # the context is unnecessary and slightly slower.
            breakdown.total.backward()

            torch.nn.utils.clip_grad_norm_(optimizable, max_norm=1.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            history.record(step, breakdown)

            if callback is not None:
                callback(step, breakdown, output)

            # --- Save checkpoint ---------------------------------------------
            if (
                ckpt_dir is not None
                and config.checkpoint_every
                and step % config.checkpoint_every == 0
            ):
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "latent": latent.detach().clone() if latent is not None else None,
                        "optimizer_state": optimizer.state_dict(),
                        "step": step,
                    },
                    ckpt_dir / f"checkpoint_{step:06d}.pt",
                )

        final_output = output.detach() if output is not None else torch.zeros(1)

        encoder_name = self._encoder_name(encoder)

        return RenderResult(
            output=final_output,
            history=history,
            encoder_name=encoder_name,
            final_similarity=history.final_similarity,
            config=config,
        )

    # ------------------------------------------------------------------
    # Optimizer / scheduler builders (exposed for unit testing)
    # ------------------------------------------------------------------

    def _build_optimizer(
        self, params: list[torch.Tensor], config: OptimizationConfig
    ) -> torch.optim.Optimizer:
        """Construct an optimizer from *config*.

        Falls back to ``Adam`` for any unrecognised optimizer name so that
        callers using custom subclasses of ``OptimizationConfig`` don't crash.

        Args:
            params: List of tensors / parameters to optimise.
            config: Configuration providing ``optimizer`` and ``learning_rate``.

        Returns:
            Configured ``torch.optim.Optimizer`` instance.
        """
        lr = config.learning_rate
        if config.optimizer == "adam":
            return torch.optim.Adam(params, lr=lr)
        if config.optimizer == "adamw":
            return torch.optim.AdamW(params, lr=lr)
        if config.optimizer == "sgd":
            return torch.optim.SGD(params, lr=lr)
        # Safe default for unexpected values
        return torch.optim.Adam(params, lr=lr)

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer, config: OptimizationConfig
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Construct an LR scheduler from *config*, or ``None`` for constant.

        Args:
            optimizer: The optimizer whose LR the scheduler will adjust.
            config: Configuration providing ``scheduler`` and ``steps``.

        Returns:
            An LR scheduler instance, or ``None`` when ``scheduler="constant"``.
        """
        if config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.steps)
        if config.scheduler == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.01,
                total_iters=config.steps,
            )
        return None  # "constant" or any other value

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _augment(self, output: torch.Tensor, aug_config: AugmentationConfig) -> torch.Tensor:
        """Apply random augmentations for robust optimization.

        Only 4-D image tensors ``[B, C, H, W]`` and 5-D video tensors
        ``[B, F, C, H, W]`` are augmented.  All other shapes (e.g. 1-D audio)
        are returned unchanged.  When both ``random_crop`` and ``random_flip``
        are disabled the tensor is returned as-is.

        Args:
            output: Decoded output from the generator.
            aug_config: Augmentation configuration.

        Returns:
            Augmented tensor with the same shape as *output*.
        """
        if not aug_config.random_crop and not aug_config.random_flip:
            return output

        # Only augment spatial tensors.
        if output.ndim not in (4, 5):
            return output

        augmented = output
        is_video = output.ndim == 5

        if aug_config.random_crop:
            scale = torch.empty(1).uniform_(*aug_config.crop_scale).item()
            h, w = output.shape[-2:]
            new_h, new_w = int(h * scale), int(w * scale)

            top = torch.randint(0, max(h - new_h, 1), (1,)).item()
            left = torch.randint(0, max(w - new_w, 1), (1,)).item()

            if is_video:
                augmented = augmented[:, :, :, top : top + new_h, left : left + new_w]
                b, f, c, _, _ = augmented.shape
                augmented = augmented.reshape(b * f, c, new_h, new_w)
                augmented = F.interpolate(
                    augmented, size=(h, w), mode="bilinear", align_corners=False
                )
                augmented = augmented.reshape(b, f, c, h, w)
            else:
                augmented = augmented[..., top : top + new_h, left : left + new_w]
                augmented = F.interpolate(
                    augmented, size=(h, w), mode="bilinear", align_corners=False
                )

        if aug_config.random_flip and torch.rand(1).item() > 0.5:
            augmented = torch.flip(augmented, dims=[-1])

        return augmented

    @staticmethod
    def _encoder_name(encoder: Any) -> str:
        """Extract a human-readable name from *encoder* if available."""
        return _encoder_name(encoder)

    def _resolve_autocast(self, encoder: Any, dtype_name: str) -> tuple[Any, torch.dtype | None]:
        """Build the autocast context manager for the requested dtype.

        Returns a tuple of (``context_manager``, ``dtype``). When
        ``dtype_name == "fp32"`` the context manager is a no-op
        ``contextlib.nullcontext`` and the dtype is ``None`` — callers
        therefore do not pay any overhead for the default case.

        ``device_type`` is inferred from the encoder's ``device`` attribute
        when available. ``cuda`` and ``mps`` both support autocast in recent
        PyTorch (2.4+); on ``cpu`` only bf16 is meaningfully supported.

        Unknown dtype strings raise ``ValueError``.
        """
        from contextlib import nullcontext

        if dtype_name == "fp32":
            return nullcontext(), None

        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(dtype_name)
        if dtype is None:
            raise ValueError(
                f"Unknown autocast_dtype '{dtype_name}'. " "Expected one of 'fp32', 'fp16', 'bf16'."
            )

        device_obj = getattr(encoder, "device", None)
        device_type = "cpu"
        if device_obj is not None:
            device_type = (
                device_obj.type if isinstance(device_obj, torch.device) else str(device_obj)
            )
            # ``str(torch.device("cuda:0"))`` is ``"cuda:0"`` — strip the index.
            device_type = device_type.split(":", 1)[0]

        return torch.autocast(device_type=device_type, dtype=dtype), dtype


def _encoder_name(encoder: Any) -> str:
    """Extract a human-readable name from an encoder if available."""
    if hasattr(encoder, "card") and encoder.card is not None:
        return encoder.card.name
    return "unknown"


class DiffusionGuidanceStrategy:
    """Diffusion denoising with per-step embedding guidance.

    Unlike ``OptimizationStrategy`` (which runs a loop around a frozen generator),
    this injects gradients INTO the diffusion denoising loop at each timestep.
    The generator and the optimization are interleaved, not sequential.

    The strategy delegates the denoising loop to the generator via
    ``generate_guided()``, which is responsible for injecting the embedding
    loss at each diffusion timestep.  This keeps the strategy thin and the
    generator in control of its own scheduling.

    Args:
        sae: Optional sparse autoencoder for SAE feature-space loss.  Passed
             through to ``CompositeLoss`` unchanged.
    """

    def __init__(self, sae: Any = None) -> None:
        self.sae = sae

    def render(
        self,
        target: Concept,
        generator: Any,
        encoder: Any,
        config: OptimizationConfig,
    ) -> RenderResult:
        """Delegate to the generator's guided denoising loop.

        The generator is expected to implement ``generate_guided()`` which
        runs the full diffusion process with per-step embedding feedback.
        If the generator returns a raw tensor rather than a ``RenderResult``,
        it is wrapped into one with an empty history.

        Args:
            target: Target concept embedding to guide toward.
            generator: A generator implementing ``generate_guided()``.
            encoder: Encoder used to compute the embedding loss at each step.
            config: Optimization configuration forwarded to the generator.

        Returns:
            ``RenderResult`` from the guided generation.
        """
        loss_fn = CompositeLoss(config.loss, encoder, sae=self.sae)
        loss_fn.calibrate(target, encoder)

        # Delegate to the generator's guided generation.
        # The generator manages its own denoising loop with per-step guidance.
        result = generator.generate_guided(
            target_embedding=target.embedding,
            encoder=encoder,
            loss_fn=loss_fn,
            config=config,
        )

        # If generator returns a raw image tensor, wrap in RenderResult.
        if not isinstance(result, RenderResult):
            encoder_name = _encoder_name(encoder)
            output = result if isinstance(result, torch.Tensor) else torch.zeros(1)
            return RenderResult(
                output=output,
                history=OptimizationHistory(),
                encoder_name=encoder_name,
                final_similarity=0.0,
                config=config,
            )

        return result
