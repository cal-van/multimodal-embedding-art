"""
Per-modality SAE stack (M3).

LanguageBind embeds image, audio, video, and text into a single 768-d
space, but the *distribution* of points within that space differs by
modality — images cluster around visual concepts, audio around timbral
ones, etc. A single SAE trained on a mix of modalities ends up with
features that are weighted by whichever modality is over-represented in
the corpus.

The multimodal SAE stack trains one SAE per modality (image / audio /
video / text) plus one on the mean-pooled-across-modalities
representation. The delta between per-modality and shared decompositions
reveals modality-specific vs modality-agnostic features.

This module provides:

* :class:`MultimodalSAEStack` — runtime container that holds the per-modality
  SAEs and dispatches based on modality string.
* :func:`train_multimodal_sae_stack` — orchestrator that loops over the
  four per-modality datasets, training each SAE separately using the
  existing single-modality ``train_sae`` infrastructure.

The actual SAE class is the existing :class:`GroupSparseSAE` from
``embedding_art.sae.training`` — no new SAE math, just orchestration on
top of the already-shipped TopK + group-sparse implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

CANONICAL_MODALITIES = ("image", "audio", "video", "text", "shared")


@dataclass
class MultimodalSAEStack:
    """Container holding one SAE per LanguageBind modality.

    Attributes
    ----------
    saes:
        Dict from modality name → :class:`GroupSparseSAE`. The 'shared'
        key holds the SAE trained on the mean-pooled-across-modalities
        representation, which lives in the same canonical space and can
        be used as a modality-agnostic decomposition.
    embed_dim:
        Dimensionality of the canonical encoder embedding (768 for
        LanguageBind).
    n_features:
        SAE bottleneck width (same across all modalities for
        comparability).
    """

    saes: dict[str, Any]
    embed_dim: int
    n_features: int

    def __call__(self, embedding: torch.Tensor, modality: str) -> torch.Tensor:
        """Encode an embedding using the appropriate per-modality SAE.

        Args:
            embedding: Shape ``[B, D]`` (or ``[D]`` — auto-unsqueezed).
            modality: One of ``"image"``, ``"audio"``, ``"video"``,
                ``"text"``, or ``"shared"``.

        Returns:
            Sparse activation tensor ``[B, n_features]``.
        """
        if modality not in self.saes:
            available = sorted(self.saes.keys())
            raise KeyError(f"Modality '{modality}' not in stack; have {available}")
        sae = self.saes[modality]
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        return sae(embedding, modality_idx=0)

    def decompose(
        self, embedding: torch.Tensor, modality: str, top_k: int = 32
    ) -> dict[int, float]:
        """Return the top-K active feature indices and activations.

        Convenience wrapper that turns the dense activation vector into
        a sparse dict suitable for the :class:`InterpretationBundle`.
        """
        activations = self(embedding, modality).detach().cpu().squeeze(0)
        active_indices = (activations > 0).nonzero(as_tuple=True)[0]
        if len(active_indices) == 0:
            return {}
        active_values = activations[active_indices]
        top_k = min(top_k, len(active_values))
        top_values, sort_indices = active_values.topk(top_k)
        top_feature_indices = active_indices[sort_indices]
        return {
            int(feat_idx): float(val)
            for feat_idx, val in zip(top_feature_indices.tolist(), top_values.tolist())
        }

    @classmethod
    def from_directory(cls, root: Path | str, embed_dim: int = 768) -> MultimodalSAEStack:
        """Load a stack from a directory containing one ``<modality>/sae_weights.pt`` per modality.

        Args:
            root: Directory containing ``image/``, ``audio/``, ``video/``,
                ``text/``, optionally ``shared/`` subdirectories.
            embed_dim: Canonical embedding dimension.

        Returns:
            Populated stack. Modalities whose checkpoint is missing are
            skipped (with a warning).
        """
        from embedding_art.sae.training import GroupSparseSAE

        root = Path(root)
        saes: dict[str, Any] = {}
        n_features = None
        for modality in CANONICAL_MODALITIES:
            weights_path = root / modality / "sae_weights.pt"
            if not weights_path.exists():
                logger.warning(
                    "SAE checkpoint missing for modality %s at %s", modality, weights_path
                )
                continue
            state = torch.load(weights_path, weights_only=True)
            w_enc = state["W_enc"]
            if w_enc.shape[1] != embed_dim:
                raise ValueError(
                    f"embed_dim mismatch for {modality}: expected {embed_dim}, "
                    f"checkpoint has {w_enc.shape[1]}"
                )
            current_features = w_enc.shape[0]
            if n_features is None:
                n_features = current_features
            elif n_features != current_features:
                logger.warning(
                    "n_features mismatch in stack: %s has %d, others have %d",
                    modality,
                    current_features,
                    n_features,
                )

            sae = GroupSparseSAE(embed_dim=embed_dim, n_features=current_features, k=32)
            sae.load_state_dict(state)
            sae.eval()
            saes[modality] = sae

        if not saes:
            raise FileNotFoundError(f"No SAE checkpoints found under {root}")

        return cls(saes=saes, embed_dim=embed_dim, n_features=n_features or 0)


def train_multimodal_sae_stack(
    *,
    embeddings_by_modality: dict[str, Path],
    output_root: Path | str,
    embed_dim: int = 768,
    n_features: int = 4096,
    k: int = 32,
    lambda_gs: float = 0.05,
    n_iterations: int = 25000,
    lr: float = 1e-3,
    batch_size: int = 128,
) -> MultimodalSAEStack:
    """Train one SAE per modality on the supplied per-modality embeddings files.

    Args:
        embeddings_by_modality: Dict from modality string to the path of
            an embeddings ``.pt`` file produced by
            :func:`embedding_art.sae.training.collect_embeddings`. Each
            file holds ``{"embeddings": tensor[N, embed_dim]}``.
        output_root: Directory under which per-modality subdirectories
            will be created (e.g. ``image/sae_weights.pt``,
            ``audio/sae_weights.pt``).
        embed_dim: Canonical embedding dimensionality (768 for
            LanguageBind).
        n_features: SAE bottleneck width. Same across modalities for
            comparability.
        k: TopK sparsity (active features per input).
        lambda_gs: Group-sparse loss weight.
        n_iterations: Per-modality gradient steps.
        lr: Adam learning rate.
        batch_size: Per-modality minibatch size.

    Returns:
        A :class:`MultimodalSAEStack` containing the trained SAEs.
    """
    from embedding_art.sae.training import train_sae

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    saes: dict[str, Any] = {}
    for modality, embeddings_path in embeddings_by_modality.items():
        if modality not in CANONICAL_MODALITIES:
            logger.warning(
                "Unknown modality '%s' — accepted modalities are %s",
                modality,
                CANONICAL_MODALITIES,
            )
        out_dir = output_root / modality
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Training SAE for modality '%s' → %s", modality, out_dir)

        sae = train_sae(
            embeddings_path=embeddings_path,
            output_path=out_dir,
            embed_dim=embed_dim,
            n_features=n_features,
            k=k,
            lambda_gs=lambda_gs,
            batch_size=batch_size,
            n_iterations=n_iterations,
            lr=lr,
        )
        saes[modality] = sae

    return MultimodalSAEStack(saes=saes, embed_dim=embed_dim, n_features=n_features)
