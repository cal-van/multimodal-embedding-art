"""
Experiments (M9): research artefacts that go beyond rendering a concept.

The anchor-comparison experiment encodes the same concept through each
modality of the canonical LanguageBind encoder, then reports how those
encodings differ. This is a direct experimental observation of the
'Platonic representation hypothesis' in LanguageBind: do the same
concept's text / image / audio / video encodings actually land at the
same point in the shared 768-d space, or does each modality carve out
a slightly different region?

Outputs include:
* The four embeddings (text, image, audio, video) in the canonical space.
* Pairwise cosine similarities (the agreement matrix).
* Pairwise SAE-decomposition deltas — when a trained SAE is supplied,
  decompose each embedding and report which features are unique to each
  modality.
* Top-K text-anchor readouts per modality (so the caller can see
  qualitatively how the same concept reads differently from each side).
"""

from embedding_art.experiments.activation_cache import EncoderActivationCache
from embedding_art.experiments.anchor_comparison import (
    AnchorComparison,
    run_anchor_comparison,
)

__all__ = [
    "AnchorComparison",
    "EncoderActivationCache",
    "run_anchor_comparison",
]
