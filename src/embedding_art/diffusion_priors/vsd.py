"""
Variational Score Distillation (VSD).

Per ProlificDreamer (Wang et al. 2023), VSD replaces the constant
'random ε' term of SDS with a learnable score model ``ε_lora`` that
absorbs the per-concept distribution. The VSD gradient is::

    ∇_θ L_VSD(θ) = E_{t, ε} [ ω(t) * (ε_φ(x_t, t, y) - ε_lora(x_t, t, y)) * ∂x/∂θ ]

with two parameter sets updated alternately:

1. The generator parameters ``θ`` (driven by the VSD pseudo-loss).
2. The LoRA adapter parameters ``φ_lora`` (driven by a standard diffusion
   loss: ``MSE(ε_lora(x_t, t, y), ε)``).

The result is a 'natural' track that stays high-fidelity without
collapsing to the marginal mode of the diffusion prior — which is the
specific failure mode that produces 2018-flavoured DeepDream aesthetics
out of plain SDS optimisations on modern backbones.

This module is backbone-agnostic. ``teacher_score_fn`` and
``student_score_fn`` are callables; the actual SD3.5 / Stable-Audio-Open
/ LTX-Video integrations live in their respective generator modules.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812


@dataclass
class VSDLoss:
    """Variational Score Distillation loss with paired generator + LoRA updates.

    Args:
        teacher_score_fn: Frozen teacher score network.
            ``teacher_score_fn(x_t, t, conditioning) -> ε_pred``.
        student_score_fn: LoRA-modified student score network. Same
            signature; its parameters are updated by
            :meth:`student_loss`.
        sigma_schedule: ``t -> σ_t`` (forward-diffusion std).
        alpha_schedule: ``t -> α_t`` (forward-diffusion signal coeff.).
        timestep_range: ``(lo, hi)`` inclusive sampling range.
        weighting_fn: Optional ``t -> ω(t)``. Defaults to 1.
        guidance_scale: CFG scale for the teacher evaluation.
        student_steps_per_generator_step: How many LoRA updates to take
            for every generator step. The ProlificDreamer paper uses 1;
            higher values can stabilise the student.
    """

    teacher_score_fn: Callable[..., torch.Tensor]
    student_score_fn: Callable[..., torch.Tensor]
    sigma_schedule: Callable[[torch.Tensor], torch.Tensor]
    alpha_schedule: Callable[[torch.Tensor], torch.Tensor]
    timestep_range: tuple[float, float] = (0.02, 0.98)
    weighting_fn: Callable[[torch.Tensor], torch.Tensor] | None = None
    guidance_scale: float = 7.5
    student_steps_per_generator_step: int = 1
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
        """Compute the VSD pseudo-loss for the generator.

        Args:
            x: Current generator output. Shape ``[B, ...]``. Must
                require gradient.
            conditioning: Conditioning argument forwarded to both score
                functions.
            timestep: Optional fixed timestep (for tests).

        Returns:
            Scalar pseudo-loss whose backward produces the VSD gradient
            on ``x``.
        """
        if not x.requires_grad:
            raise ValueError("VSDLoss expects x.requires_grad=True")

        t = self._sample_timestep(x, timestep)
        alpha_t = self.alpha_schedule(t)
        sigma_t = self.sigma_schedule(t)
        eps = self._sample_noise(x)

        x_t = alpha_t * x + sigma_t * eps

        with torch.no_grad():
            eps_teacher = self.teacher_score_fn(x_t, t, conditioning)
            eps_student = self.student_score_fn(x_t, t, conditioning)

        weight = self._broadcast_weight(t, x)

        grad = (weight * (eps_teacher - eps_student)).detach()
        loss = (x * grad).sum() / x.shape[0]
        return loss

    def _broadcast_weight(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.weighting_fn is not None:
            w = self.weighting_fn(t)
        else:
            w = torch.ones_like(t)
        while w.dim() < x.dim():
            w = w.unsqueeze(-1)
        return w

    def student_loss(
        self,
        x: torch.Tensor,
        conditioning: Any,
        *,
        timestep: float | None = None,
    ) -> torch.Tensor:
        """Compute the LoRA student's standard diffusion loss.

        This is the *real* loss the LoRA is trained on: predict the
        injected noise given ``x_t``. It is NOT a pseudo-loss; calling
        ``.backward()`` updates the LoRA's actual parameters.

        Args:
            x: Current generator output (no gradient required, but
                may be detached as a defensive measure).
            conditioning: Conditioning for the student.
            timestep: Optional fixed timestep (for tests).

        Returns:
            Scalar MSE loss. Caller backprops into LoRA params.
        """
        x = x.detach()
        t = self._sample_timestep(x, timestep)
        alpha_t = self.alpha_schedule(t)
        sigma_t = self.sigma_schedule(t)
        eps = self._sample_noise(x)
        x_t = alpha_t * x + sigma_t * eps

        eps_pred = self.student_score_fn(x_t, t, conditioning)
        return F.mse_loss(eps_pred, eps)

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
