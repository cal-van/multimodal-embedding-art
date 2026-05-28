#!/usr/bin/env bash
# Anchor comparison: encode the *same* concept through several modalities and
# measure how close the resulting embeddings land in the shared space. This is
# the quickest way to see LanguageBind's cross-modal binding in action — no
# generation, just encoding, so it runs in seconds.
#
# Output: <output-dir>/anchor_comparison.json (pairwise cosine matrix + per-
# modality top text anchors).
set -euo pipefail

# Text-only baseline — always runnable, no assets required.
embed-art anchor-compare \
  --label thunder \
  --text 'thunder' \
  --output-dir outputs/examples/anchor-thunder/

# Cross-modal — point at your own files to see how an image / audio clip of a
# concept embeds relative to its text description. Replace the paths below.
#
#   embed-art anchor-compare \
#     --label thunder \
#     --text 'thunder' \
#     --image path/to/storm.jpg \
#     --audio path/to/thunder.wav \
#     --output-dir outputs/examples/anchor-thunder-multimodal/ \
#     --top-k-text 10
#
# A high cross-modal cosine (e.g. image-vs-audio of the same concept) is the
# property that makes the four-modality `showcase` coherent.
