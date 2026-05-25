"""Minimal hand-rolled LoRA layer + injection helpers for VSD.

LoRA (Low-Rank Adaptation, Hu et al. 2021) decomposes a weight update
as ``ΔW = B · A`` where ``A ∈ R^{r×d_in}`` and ``B ∈ R^{d_out×r}`` with
``r ≪ min(d_in, d_out)``. The modified linear forward is::

    y = W · x  +  scale · (B · (A · x))

This module provides:

* :class:`LoRALinear` — wraps an existing ``nn.Linear`` and adds the
  LoRA update. The base linear's parameters are frozen; only the LoRA
  parameters are trainable. The wrapper has an ``enabled`` flag so
  callers can route the same transformer through both "base score
  function" (LoRA off) and "phi score function" (LoRA on) without
  duplicating weights.
* :func:`inject_lora_into_transformer` — walks a transformer module
  tree and replaces every attention projection submodule (q_proj /
  k_proj / v_proj / out_proj, or to_q / to_k / to_v / to_out, the two
  conventions diffusers transformers use) with a :class:`LoRALinear`
  wrapper. Returns the list of injected wrappers so callers can
  collect their trainable parameters for the optimizer.
* :func:`lora_parameters` — iterate trainable LoRA parameters from a
  module tree.
* :func:`set_lora_enabled` — toggle every :class:`LoRALinear` under a
  module on/off; used to share a transformer between base and phi
  adapters.

The injection is *additive* — when ``enabled=False`` the wrappers
are functionally identical to the original linear, so existing
inference paths through the transformer remain correct.

No external dependencies required (no ``peft``, no ``diffusers``-
specific imports). Works on any ``nn.Module`` tree that exposes
attention modules with the standard projection submodule names.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from typing import Any

import torch
import torch.nn as nn

# Standard attention projection attribute names across two conventions:
# - HF / CLIP shape: q_proj / k_proj / v_proj / out_proj
# - diffusers / SD3 shape: to_q / to_k / to_v / to_out (to_out is a Sequential)
_PROJECTION_NAMES = ("q_proj", "k_proj", "v_proj", "out_proj", "to_q", "to_k", "to_v", "to_out")


class LoRALinear(nn.Module):
    """LoRA-wrapped ``nn.Linear``.

    Args:
        base: The ``nn.Linear`` to wrap. Its parameters are frozen on
            wrap.
        rank: LoRA rank. Default 4. Smaller is cheaper but less
            expressive.
        alpha: LoRA scaling factor. The effective update is multiplied
            by ``alpha / rank`` (the standard LoRA scaling
            convention). Default 4.0.
        dropout: Optional dropout on the LoRA branch. Default 0.0.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int = 4,
        alpha: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / rank
        in_features = base.in_features
        out_features = base.out_features
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.base(x)
        if not self.enabled:
            return out
        delta = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + self.scale * delta

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def lora_state(self) -> dict[str, torch.Tensor]:
        """Return only the LoRA state (for cheap snapshotting)."""
        return {"lora_A": self.lora_A.detach().clone(), "lora_B": self.lora_B.detach().clone()}


def _replace_projection(
    parent: nn.Module, attr_name: str, rank: int, alpha: float
) -> LoRALinear | None:
    """Replace ``parent.<attr_name>`` with a :class:`LoRALinear` wrapper.

    Returns the new wrapper, or ``None`` when the attribute isn't a
    plain ``nn.Linear`` (some diffusers ``to_out`` are ``nn.Sequential``
    holding a Linear at index 0).
    """
    target = getattr(parent, attr_name, None)
    if isinstance(target, nn.Linear):
        wrapped = LoRALinear(target, rank=rank, alpha=alpha)
        setattr(parent, attr_name, wrapped)
        return wrapped
    if isinstance(target, nn.Sequential) and len(target) > 0 and isinstance(target[0], nn.Linear):
        # diffusers ``to_out`` is ``Sequential(Linear, Dropout)`` —
        # wrap the inner Linear, leave the dropout in place.
        wrapped = LoRALinear(target[0], rank=rank, alpha=alpha)
        target[0] = wrapped
        return wrapped
    return None


def inject_lora_into_transformer(
    transformer: nn.Module,
    *,
    rank: int = 4,
    alpha: float = 4.0,
    projection_names: Iterable[str] = _PROJECTION_NAMES,
) -> list[LoRALinear]:
    """Inject LoRA adapters into every attention projection in the tree.

    Args:
        transformer: Root module to walk.
        rank: LoRA rank.
        alpha: LoRA scaling factor.
        projection_names: Attribute names to look for and wrap. The
            default covers both HF CLIP attention (``q_proj`` / ...)
            and diffusers ``Attention`` (``to_q`` / ...) shapes.

    Returns:
        List of injected :class:`LoRALinear` wrappers. Their
        ``.lora_A`` and ``.lora_B`` parameters are the trainable
        LoRA state for the optimizer.
    """
    injected: list[LoRALinear] = []
    for module in transformer.modules():
        for name in projection_names:
            wrapped = _replace_projection(module, name, rank=rank, alpha=alpha)
            if wrapped is not None:
                injected.append(wrapped)
    return injected


def lora_parameters(root: nn.Module) -> Iterator[nn.Parameter]:
    """Yield every trainable LoRA parameter in the module tree."""
    for module in root.modules():
        if isinstance(module, LoRALinear):
            yield module.lora_A
            yield module.lora_B


def set_lora_enabled(root: nn.Module, enabled: bool) -> int:
    """Toggle every :class:`LoRALinear` under ``root``.

    Returns the number of wrappers toggled. Useful for routing a
    shared transformer through two adapters (one with LoRA active,
    one without).
    """
    n = 0
    for module in root.modules():
        if isinstance(module, LoRALinear):
            module.enabled = enabled
            n += 1
    return n


class lora_disabled:  # noqa: N801 — intentional context-manager-style lowercase
    """Context manager that disables every LoRA wrapper inside ``root``
    for the duration of the ``with`` block, then restores.

    Usage::

        with lora_disabled(transformer):
            # base score evaluation
            base_pred = transformer(x_t, ...)
        # back to LoRA-on
    """

    def __init__(self, root: nn.Module) -> None:
        self._root = root
        self._prev: list[tuple[LoRALinear, bool]] = []

    def __enter__(self) -> lora_disabled:
        self._prev = [(m, m.enabled) for m in self._root.modules() if isinstance(m, LoRALinear)]
        for m, _ in self._prev:
            m.enabled = False
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        for m, prev in self._prev:
            m.enabled = prev
        self._prev = []
