"""
CompositeLoss: multi-signal loss function for embedding optimization.

Combines four optional components:
  1. Cosine similarity on the final embedding (always active)
  2. Multi-layer feature matching — mean/std statistics per encoder layer
  3. SAE feature-space loss — MSE between sparse decompositions
  4. Regularization — existing spatial/spectral regularizer system

The caller must invoke ``calibrate()`` before the first optimization step
when multi-layer feature matching is desired.  If the encoder does not
expose ``MULTI_LAYER_FEATURES`` or the target concept carries no
``source_input``, calibration is silently skipped and feature matching
is disabled for that run.

Design reference: docs/superpowers/specs/2026-03-19-embedding-art-v2-design.md
Section 2: Optimization Engine & Loss Architecture
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from embedding_art.core.concept import Concept
from embedding_art.core.config import LossConfig
from embedding_art.core.render_result import LossBreakdown
from embedding_art.encoders.features import FeatureStatistics, ViTStatisticsExtractor
from embedding_art.encoders.registry import EncoderCapability

logger = logging.getLogger(__name__)


class CompositeLoss:
    """Composite loss for gradient-based embedding optimization.

    Args:
        config: Weights and settings for each loss component.
        encoder: The encoder used during optimization.  Used here to decide
            which ``FeatureStatisticsExtractor`` is appropriate.
        sae: Optional SAE lens for feature-space loss.  When ``None`` the
            ``sae_features`` component is omitted even if
            ``config.sae_feature_weight > 0``.
    """

    def __init__(self, config: LossConfig, encoder, sae=None) -> None:
        self.config = config
        self.sae = sae
        self.stats_extractor = self._select_extractor(encoder)
        self.reference_stats: dict[int, FeatureStatistics] | None = None
        # Cached text-anchor embedding (computed lazily on first call).
        self._text_anchor_embedding: torch.Tensor | None = None
        self._text_anchor_source: str | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_extractor(encoder):
        """Return a FeatureStatisticsExtractor appropriate for *encoder*.

        Currently uses ``ViTStatisticsExtractor`` for any encoder that
        advertises ``MULTI_LAYER_FEATURES``.  Returns ``None`` for encoders
        that don't expose intermediate layer activations — the feature
        matching path will simply be skipped at calibration time.
        """
        if not hasattr(encoder, "card"):
            return None
        if EncoderCapability.MULTI_LAYER_FEATURES in encoder.card.capabilities:
            return ViTStatisticsExtractor()
        return None

    def _resolve_anchor_text(self, target: Concept) -> str | None:
        """Return the text to use for the text-anchor auxiliary loss.

        Precedence: ``config.text_anchor_text`` wins; otherwise we fall
        back to a string ``source`` on the target Concept (e.g. when the
        concept was created via ``Concept.from_text("storm")``).
        """
        if self.config.text_anchor_text:
            return self.config.text_anchor_text
        source = getattr(target, "source", None)
        if isinstance(source, str) and source.strip():
            return source.strip()
        return None

    def _get_text_anchor_embedding(self, encoder, anchor_text: str | None) -> torch.Tensor | None:
        """Encode ``anchor_text`` to its text-projection embedding.

        Cached per CompositeLoss instance to avoid re-encoding the same
        string on every step. Returns ``None`` when no text could be
        derived or the encoder lacks ``encode_text``.
        """
        if not anchor_text:
            return None
        if self._text_anchor_embedding is not None and self._text_anchor_source == anchor_text:
            return self._text_anchor_embedding
        if not hasattr(encoder, "encode_text"):
            return None
        try:
            with torch.no_grad():
                emb = encoder.encode_text(anchor_text).detach()
        except Exception as exc:
            logger.warning("text-anchor encoding failed for %r: %s", anchor_text, exc)
            return None
        self._text_anchor_embedding = emb
        self._text_anchor_source = anchor_text
        return emb

    @staticmethod
    def _encode(encoder, tensor: torch.Tensor) -> torch.Tensor:
        """Call the encoder's optimization-oriented encoding method.

        Prefers ``encode_for_optimization`` (introduced in v2 protocol) and
        falls back to ``encode_image`` for legacy encoders.
        """
        if hasattr(encoder, "encode_for_optimization"):
            return encoder.encode_for_optimization(tensor)
        return encoder.encode_image(tensor)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(self, target: Concept, encoder) -> None:
        """Compute reference feature statistics from *target*'s source input.

        This must be called once before the optimization loop begins.  If the
        target concept has no ``source_input`` tensor (e.g. it was created
        from text), or the encoder does not support multi-layer features,
        ``reference_stats`` remains ``None`` and feature matching is silently
        disabled.

        Args:
            target: The target concept.  ``target.source_input`` must be a
                tensor for calibration to succeed.
            encoder: The encoder to extract layer features from.
        """
        if target.source_input is None:
            return
        if not hasattr(encoder, "card"):
            return
        if EncoderCapability.MULTI_LAYER_FEATURES not in encoder.card.capabilities:
            return
        if self.stats_extractor is None:
            return

        layer_features = encoder.get_layer_features(target.source_input)
        self.reference_stats = {
            idx: self.stats_extractor.extract(feats) for idx, feats in layer_features.items()
        }

    def __call__(
        self,
        current_output: torch.Tensor,
        target: Concept,
        encoder,
        latent: torch.Tensor | None = None,
    ) -> LossBreakdown:
        """Compute composite loss for one optimization step.

        Args:
            current_output: Decoded output tensor, e.g. ``[B, C, H, W]`` image.
            target: Target concept embedding to optimize toward.
            encoder: Encoder used to re-encode *current_output*.
            latent: Optional latent tensor, passed to regularization if present.

        Returns:
            ``LossBreakdown`` with ``total`` (scalar) and per-component tensors.
        """
        components: dict[str, torch.Tensor] = {}

        # 1. Cosine similarity on final embedding — always computed.
        current_emb = self._encode(encoder, current_output)
        sim = F.cosine_similarity(current_emb, target.embedding, dim=-1).mean()
        components["similarity"] = -sim * self.config.similarity_weight

        # 2. Multi-layer feature matching.
        if self.config.feature_matching_weight > 0 and self.reference_stats is not None:
            layer_features = encoder.get_layer_features(current_output)
            feat_loss = torch.tensor(0.0, device=current_output.device)
            for idx, feats in layer_features.items():
                if idx not in self.reference_stats:
                    continue
                current_stats = self.stats_extractor.extract(feats)
                ref_stats = self.reference_stats[idx]
                feat_loss = feat_loss + (
                    (current_stats.mean - ref_stats.mean).pow(2).sum()
                    + (current_stats.std - ref_stats.std).pow(2).sum()
                )
            components["feature_matching"] = feat_loss * self.config.feature_matching_weight

        # 3. SAE feature-space loss.
        if self.config.sae_feature_weight > 0 and self.sae is not None:
            current_decomp = self.sae.decompose(current_emb)
            target_decomp = self.sae.decompose(target.embedding)
            components["sae_features"] = (
                F.mse_loss(current_decomp.activations, target_decomp.activations)
                * self.config.sae_feature_weight
            )

        # 4. Text-anchor auxiliary loss (M4).
        if self.config.text_anchor_weight > 0:
            anchor_text = self._resolve_anchor_text(target)
            anchor_emb = self._get_text_anchor_embedding(encoder, anchor_text)
            if anchor_emb is not None:
                anchor_sim = F.cosine_similarity(
                    current_emb, anchor_emb.to(current_emb.device), dim=-1
                ).mean()
                components["text_anchor"] = -anchor_sim * self.config.text_anchor_weight

        # 5. Regularization (uses existing regularizer system).
        if self.config.regularization is not None and latent is not None:
            components["regularization"] = self.config.regularization(latent, current_output)

        total: torch.Tensor = sum(components.values())  # type: ignore[assignment]
        return LossBreakdown(total=total, components=components)
