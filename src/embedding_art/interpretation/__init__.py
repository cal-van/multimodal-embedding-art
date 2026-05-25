"""
Interpretation bundle (M7): tools for explaining what a rendered concept
actually represents in the canonical LanguageBind embedding space.

The bundle assembles four complementary signals on every rendered output:

* :func:`text_anchor_readout` — for the given embedding, the top-K
  vocabulary words whose text embedding lands nearest in the shared
  LanguageBind space. Exploits LanguageBind's text-as-anchor design:
  every modality is aligned to language, so projecting the rendered
  embedding back into language gives a human-readable label set.

* :func:`gradient_attribution` — pixel-space saliency map computed via
  the gradient of cosine similarity (current embedding ↔ target
  embedding) with respect to the input tensor. Universal across
  modalities (works on images, audio waveforms, video frames).

* :class:`SAEDecomposition` (from ``core.concept``) — when a trained SAE
  is supplied, decomposes the current embedding into a sparse set of
  feature activations. The features are then optionally labelled via
  :func:`embedding_art.sae.training.label_features`.

* CorrSteer-style trajectory analysis — when a per-step optimisation
  history is available, correlates SAE feature activations with the
  cosine-similarity trajectory to identify *which* features actually
  drove the optimisation. Returns the top-K features by correlation.

These four signals are bundled into an :class:`InterpretationBundle`
which is serialised into the showcase ``manifest.json`` and rendered
into a human-readable section of the text card.

Design reference: docs/superpowers/plans/2026-05-24-first-principles.md
"""

from embedding_art.interpretation.attribution import gradient_attribution
from embedding_art.interpretation.bundle import (
    InterpretationBundle,
    run_interpretation,
)
from embedding_art.interpretation.corrsteer import (
    compute_corrsteer,
    pearson_correlation_matrix,
)
from embedding_art.interpretation.text_anchor import text_anchor_readout

__all__ = [
    "InterpretationBundle",
    "compute_corrsteer",
    "gradient_attribution",
    "pearson_correlation_matrix",
    "run_interpretation",
    "text_anchor_readout",
]
