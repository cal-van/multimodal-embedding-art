"""
Score Distillation Sampling (SDS) — the baseline diffusion-prior loss.

Per :paper:`DreamFusion`, the SDS gradient is::

    ∇_θ L_SDS(θ) = E_{t, ε} [ ω(t) * (ε_φ(x_t, t, y) - ε) * ∂x/∂θ ]

with ``x_t = α_t * x + σ_t * ε``. In practice this is implemented as a
pseudo-loss:

    L_SDS = MSE(x, x - stop_grad(ε_φ(x_t, t, y) - ε) * step)

so that ``autograd.backward`` produces the correct gradient on ``θ``.

Limitations
-----------
SDS is mode-seeking: averaging across timesteps biases the optimisation
toward the marginal mode of the diffusion prior. This produces the
'natural but over-smoothed' look of plain SDS optimisations. VSD
:class:`embedding_art.diffusion_priors.vsd.VSDLoss` mitigates this; SDS
is retained as a baseline / fallback / ablation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class SDSLoss:
    """Score-distillation sampling loss as a callable.

    Args:
        score_fn: Frozen score network. Signature
            ``score_fn(x_t, t, conditioning) -> ε_pred``. Must accept a
            batched ``x_t`` and a scalar / batched timestep ``t``.
        sigma_schedule: Function ``t -> σ_t`` giving the standard
            deviation of the forward diffusion process at timestep
            ``t``. Many parameterisations use ``σ_t = √(1 - α_bar_t)``.
        alpha_schedule: Function ``t -> α_t``; signal-coefficient of
            ``x_t = α_t * x + σ_t * ε``.
        timestep_range: Inclusive range from which timesteps are sampled
            during the SDS loop. Defaults to ``(0.02, 0.98)`` (typical
            DreamFusion choice; avoids extreme timesteps where the prior
            is uninformative).
        weighting_fn: Optional ``t -> ω(t)``. Defaults to ``ω(t) = 1`` (the
            'simplified' SDS recommended in subsequent works for stability).
        guidance_scale: Classifier-free guidance scale applied to the
            score evaluation. Default 7.5 follows SD usage.
    """

    score_fn: Callable[..., torch.Tensor]
    sigma_schedule: Callable[[torch.Tensor], torch.Tensor]
    alpha_schedule: Callable[[torch.Tensor], torch.Tensor]
    timestep_range: tuple[float, float] = (0.02, 0.98)
    weighting_fn: Callable[[torch.Tensor], torch.Tensor] | None = None
    guidance_scale: float = 7.5
    _rng: Any = field(default=None, init=False, repr=False)

    def seed(self, value: int) -> None:
        """Set a deterministic RNG for unit tests."""
        self._rng = torch.Generator(device="cpu").manual_seed(value)

    def __call__(
        self,
        x: torch.Tensor,
        conditioning: Any,
        *,
        timestep: float | None = None,
    ) -> torch.Tensor:
        """Compute the SDS pseudo-loss for ``x``.

        Args:
            x: Current generator output (in the diffusion model's input
                space, typically latents). Shape ``[B, ...]``. Must
                require gradient.
            conditioning: Whatever the teacher score network expects as
                its conditioning argument (e.g. text-encoded tokens for
                SD; the embedding-art target embedding for v3).
            timestep: Optional fixed timestep ``t ∈ [0, 1]`` (handy for
                tests). When omitted a uniformly random timestep is
                drawn from ``timestep_range``.

        Returns:
            Scalar pseudo-loss. Backprop produces the SDS gradient on
            ``x`` (and thence on whatever upstream parameters produced
            ``x``).
        """
        if not x.requires_grad:
            raise ValueError("SDSLoss expects x.requires_grad=True")

        t = self._sample_timestep(x, timestep)
        alpha_t = self.alpha_schedule(t)
        sigma_t = self.sigma_schedule(t)
        eps = self._sample_noise(x)

        x_t = alpha_t * x + sigma_t * eps

        with torch.no_grad():
            eps_pred = self.score_fn(x_t, t, conditioning)

        weight = self._broadcast_weight(t, x)

        grad = (weight * (eps_pred - eps)).detach()
        loss = (x * grad).sum() / x.shape[0]
        return loss

    def _broadcast_weight(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.weighting_fn is not None:
            w = self.weighting_fn(t)
        else:
            w = torch.ones_like(t)
        # Broadcast to x's shape: [B, 1, 1, ...] (or scalar batch dim).
        while w.dim() < x.dim():
            w = w.unsqueeze(-1)
        return w

    def _sample_timestep(self, x: torch.Tensor, override: float | None) -> torch.Tensor:
        if override is not None:
            return torch.full((x.shape[0],), float(override), device=x.device)
        lo, hi = self.timestep_range
        if self._rng is not None:
            sample = torch.rand((x.shape[0],), generator=self._rng).to(x.device)
        else:
            sample = torch.rand((x.shape[0],), device=x.device)
        return lo + (hi - lo) * sample

    def _sample_noise(self, x: torch.Tensor) -> torch.Tensor:
        if self._rng is not None:
            return torch.randn(x.shape, generator=self._rng).to(x.device)
        return torch.randn_like(x)
