"""
Gradient attribution — modality-agnostic saliency map.

For any output tensor (image, audio waveform, video frame stack) and any
encoder that exposes ``encode_for_optimization``, computes the gradient of
cosine similarity (current embedding ↔ target embedding) with respect to
the input. The magnitude of the gradient at each spatial / temporal
position is the saliency: 'how much would moving this part of the output
change the alignment to the concept?'

This is the simplest interpretability tool but the most universal — it
works on any differentiable encoder, any modality. The first-principles
doc preferred classical ViT-specific attribution (attention rollout +
integrated gradients) for LanguageBind's ViT encoders, but gradient
attribution is the universal fallback that also works for the audio (CNN)
and video (3D CNN) encoders.

Output: a tensor with the same shape as the input, where each value is
the absolute gradient magnitude (so the caller can render it as a
single-channel heatmap regardless of input channel count).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812


def gradient_attribution(
    output: torch.Tensor,
    target_embedding: torch.Tensor,
    encoder: Any,
) -> torch.Tensor:
    """Compute pixel-/sample-level saliency by backprop through the encoder.

    Args:
        output: The final rendered output (image, audio, video). Will be
            detached and copied to allow independent gradient computation.
        target_embedding: The target embedding in the encoder's space.
            Shape ``[D]`` or ``[1, D]``. Will be unit-normalised.
        encoder: Any encoder exposing ``encode_for_optimization(tensor)``.

    Returns:
        Tensor with the same shape as ``output`` whose values are the
        absolute gradient magnitude. Caller decides how to reduce across
        channels for visualisation.
    """
    if target_embedding.dim() == 1:
        target_embedding = target_embedding.unsqueeze(0)
    target_norm = F.normalize(target_embedding, dim=-1)

    output_grad = output.detach().clone().requires_grad_(True)

    current_emb = encoder.encode_for_optimization(output_grad)
    if current_emb.dim() == 1:
        current_emb = current_emb.unsqueeze(0)
    current_norm = F.normalize(current_emb, dim=-1)

    sim = F.cosine_similarity(current_norm, target_norm, dim=-1).mean()

    grad = torch.autograd.grad(sim, output_grad, retain_graph=False)[0]

    return grad.detach().abs()
