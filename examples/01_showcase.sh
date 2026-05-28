#!/usr/bin/env bash
# The headline command: render one concept across all four modalities
# (image + audio + video + text) from a single shared LanguageBind embedding.
#
# Output bundle: <output-dir>/image.png, audio.wav, video.mp4, text-card.md,
# and manifest.json (interpretation bundle + evaluation card).
set -euo pipefail

# 1. SMOKE TEST — text only, 10 steps. Verifies the install path without
#    loading the heavy image/audio/video generators. Finishes in seconds.
embed-art showcase \
  --target-text 'thunder' \
  --output-dir outputs/examples/thunder-smoke/ \
  --modalities text \
  --steps 10

echo "Smoke test done. For the real thing, run the full showcase below."

# 2. FULL SHOWCASE — all four modalities. Apple Silicon perf flags give a
#    ~3-5x wall-time win over fp32. Expect tens of minutes.
embed-art showcase \
  --target-text 'goldfish' \
  --output-dir outputs/examples/goldfish/ \
  --autocast-dtype bf16 \
  --compile-mode reduce-overhead \
  --seed 0

# Variations to try:
#   --modalities image            # just the image, fits comfortably on 16GB
#   --tracks honest,natural       # render both the "honest" and VSD-prior tracks
#   --probes siglip2-so400m,clap-general   # add cross-encoder agreement to the eval card
#   --seed-stability 3            # re-run 3x with different seeds, report variance
