"""
Patch-level alignment loss against LanguageBind ViT hooks (M2c / v3).

Adds per-token cosine similarity between target and current patch tokens at
multiple ViT layers, in addition to the pooled embedding alignment that the
v1/v2 ``CompositeLoss`` already implements.

Why patch-level
---------------
The pooled embedding alignment forces the **average** of all patches to match
the target. This is a weak signal: it permits an output whose pooled mean is
right but whose per-region representation is wrong (e.g. a generic
'goldfish-coloured texture' that pools to ~goldfish, but where no individual
patch actually looks like a fish part).

Patch-level alignment supplies the **per-position** signal that the pooled
loss lacks. The target image is encoded once at multiple ViT depths; during
optimisation, the current rendering's patch tokens at the same depths are
compared **per-token** to the target's. The result is a much stronger geometric
constraint on the output — every spatial location is asked to look like the
target's same-position location at every layer.

For the LanguageBind canonical encoder, the patch tokens at each layer have
shape ``[B, N_patches + 1, D]`` (CLS token + 256 patch tokens for the
ViT-L/14 used in LanguageBind's image encoder). The CLS token is dropped
before the per-token cosine; only the spatial patches participate.

When patch-level alignment is not applicable
--------------------------------------------
Patch-level alignment requires the target concept to carry a ``source_input``
image (so we can extract its patch tokens). For text-only or audio-only
targets, there are no target patches to compare to; ``PatchAlignmentLoss``
silently disables itself in that case (returns 0.0). The pooled cosine
alignment and the multi-layer mean/std statistics still run.

Multi-layer alignment
---------------------
By default this loss computes the per-token cosine at every layer
``get_layer_features`` returns, then averages across layers. This implements
the 'multi-layer feature alignment' clause in the M2 spec from the same
extraction calls — no separate multi-layer loss class is needed.

Design reference: docs/superpowers/plans/2026-05-24-first-principles.md
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

from embedding_art.encoders.features import LayerFeatures

logger = logging.getLogger(__name__)


class PatchAlignmentLoss:
    """Per-token cosine similarity between target and current ViT patch tokens.

    Calibration extracts patch tokens from the target image at each configured
    layer; on every optimisation step the current rendering's patch tokens are
    extracted and compared per-token to the calibration tokens.

    The loss returned is ``1 - mean(cosine_sim)`` averaged across layers (so
    lower is better, and the value lies in ``[0, 2]``).

    Parameters
    ----------
    drop_cls_token:
        Whether to drop the first token (CLS) before computing per-token
        cosine. Defaults to True for ViT-style encoders where the first token
        is the CLS token. Set to False for encoders that emit only spatial
        tokens.
    layer_subset:
        Optional list of layer indices to use; defaults to all layers
        returned by the encoder's ``get_layer_features``.
    """

    def __init__(
        self,
        drop_cls_token: bool = True,
        layer_subset: list[int] | None = None,
    ) -> None:
        self.drop_cls_token = drop_cls_token
        self.layer_subset = layer_subset
        self.reference_tokens: dict[int, torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, target_image: torch.Tensor, encoder: Any) -> None:
        """Extract reference patch tokens from the target image.

        Must be called once before the optimisation loop. Skipped silently
        (and the loss disabled) if the encoder doesn't expose
        ``get_layer_features``.

        Args:
            target_image: ``[B, 3, H, W]`` or ``[3, H, W]`` float tensor in
                [0, 1]. The target image whose patch tokens supply the
                per-layer alignment reference.
            encoder: The encoder used during optimisation; must expose
                ``get_layer_features``.
        """
        if not hasattr(encoder, "get_layer_features"):
            logger.info(
                "PatchAlignmentLoss disabled: encoder %s has no " "get_layer_features method.",
                type(encoder).__name__,
            )
            self.reference_tokens = None
            return

        with torch.no_grad():
            try:
                layer_features = encoder.get_layer_features(target_image)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "PatchAlignmentLoss disabled: get_layer_features raised %s",
                    exc,
                )
                self.reference_tokens = None
                return

        self.reference_tokens = {}
        for idx, lf in layer_features.items():
            if self.layer_subset is not None and idx not in self.layer_subset:
                continue
            tokens = self._extract_patch_tokens(lf)
            if tokens is None:
                continue
            self.reference_tokens[idx] = tokens.detach().clone()

        if not self.reference_tokens:
            logger.info(
                "PatchAlignmentLoss disabled: no usable patch-token layers in "
                "encoder output (got %d layers, none of shape "
                "[B, N, D]).",
                len(layer_features),
            )
            self.reference_tokens = None

    # ------------------------------------------------------------------
    # Per-step computation
    # ------------------------------------------------------------------

    def __call__(self, current_image: torch.Tensor, encoder: Any) -> torch.Tensor:
        """Compute the per-token cosine loss for the current rendering.

        Args:
            current_image: ``[B, 3, H, W]`` float tensor in [0, 1] — the
                rendering's current output. Must be differentiable.
            encoder: The same encoder passed to ``calibrate``.

        Returns:
            Scalar loss value, ``1 - mean_layer mean_patch cos_sim``. Returns
            zero if the loss is disabled (no calibration data).
        """
        if not self.is_active:
            return torch.zeros((), device=current_image.device, dtype=current_image.dtype)

        layer_features = encoder.get_layer_features(current_image)

        per_layer_losses: list[torch.Tensor] = []
        for idx, lf in layer_features.items():
            if idx not in self.reference_tokens:
                continue
            cur_tokens = self._extract_patch_tokens(lf)
            if cur_tokens is None:
                continue
            ref_tokens = self.reference_tokens[idx].to(cur_tokens.device, cur_tokens.dtype)
            # Cosine similarity per patch position: [B, N_patches]
            sim = F.cosine_similarity(cur_tokens, ref_tokens, dim=-1)
            # Average over patches and batch.
            per_layer_losses.append(1.0 - sim.mean())

        if not per_layer_losses:
            return torch.zeros((), device=current_image.device, dtype=current_image.dtype)

        return torch.stack(per_layer_losses).mean()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """Whether calibration has produced usable reference tokens."""
        return self.reference_tokens is not None and len(self.reference_tokens) > 0

    def _extract_patch_tokens(self, lf: LayerFeatures) -> torch.Tensor | None:
        """Return the patch tokens from a ``LayerFeatures``, or ``None`` if
        the layer's tensor doesn't have the expected ``[B, N, D]`` shape."""
        tensor = lf.tensor
        if tensor.dim() != 3:
            return None
        if self.drop_cls_token and tensor.shape[1] > 1:
            tensor = tensor[:, 1:, :]
        return tensor
