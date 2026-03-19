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

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

from embedding_art.core.config import OptimizationConfig
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
    ) -> RenderResult:
        """Run the optimization loop and return the final result.

        Args:
            target: Target concept.
            generator: LatentGenerator or DirectGenerator.
            encoder: Encoder used for loss computation.
            config: Optimization configuration.

        Returns:
            RenderResult with final output and full history.
        """
        loss_fn = CompositeLoss(config.loss, encoder, sae=self.sae)
        loss_fn.calibrate(target, encoder)

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

        output: torch.Tensor | None = None

        for step in range(config.steps):
            optimizer.zero_grad()

            if is_direct:
                output = generator.render()
            else:
                output = generator.decode(latent)  # type: ignore[arg-type]

            breakdown = loss_fn(output, target, encoder, latent)
            breakdown.total.backward()

            torch.nn.utils.clip_grad_norm_(optimizable, max_norm=1.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            history.record(step, breakdown)

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

    @staticmethod
    def _encoder_name(encoder: Any) -> str:
        """Extract a human-readable name from *encoder* if available."""
        return _encoder_name(encoder)


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
        target: "Concept",
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
