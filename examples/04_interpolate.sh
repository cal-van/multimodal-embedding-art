#!/usr/bin/env bash
# Interpolation: render a morph series between two concepts by walking a
# straight path between their embeddings and rendering an image at each step.
# Great for seeing how the model's notion of "goldfish" deforms into "flamingo".
#
# Output: <output-dir>/ with one image per interpolation point.
set -euo pipefail

# SMOKE — 3 points, few optimisation steps each. Confirms the path works.
embed-art interpolate \
  --concept-a 'goldfish' \
  --concept-b 'flamingo' \
  --steps 3 \
  --opt-steps 50 \
  --output image \
  --output-dir outputs/examples/interp-smoke/

echo "Smoke done. The full series below takes a few minutes."

# FULL — 10 points, default optimisation budget per point.
embed-art interpolate \
  --concept-a 'goldfish' \
  --concept-b 'flamingo' \
  --steps 10 \
  --output image \
  --output-dir outputs/examples/interp-goldfish-flamingo/
