# M2b VSD validation runbook (M1/M2 Max)

End-to-end validation of the Variational Score Distillation pipeline
against real Stable Diffusion 3.5 weights. The VSD math + SDS adapter
+ LoRA injection are implemented and covered by unit tests
(`tests/test_vsd.py`, `tests/test_sd35_adapter.py`,
`tests/test_lora.py`). What this runbook validates is that the
combined system produces the expected dual-track behaviour on real
weights — specifically that the *natural* track stays crispy without
collapsing to the marginal mode of the diffusion prior.

This must be run on Apple Silicon (or another GPU with ≥10 GB VRAM)
with the SD3.5-medium weights downloadable from HuggingFace. The
runbook intentionally leaves the optimisation loop wiring as a
follow-up — the `embed-art validate-vsd` command currently sets up
the adapter graph and writes the plan manifest, then raises with a
clear pointer to this runbook.

## Setup

```
# 1. Authenticate with HuggingFace (SD3.5-medium gated).
huggingface-cli login

# 2. Pre-download weights to avoid CLI hang on first run.
huggingface-cli download stabilityai/stable-diffusion-3.5-medium
```

The full weight set is ~10 GB. Expect 5-15 minutes on a typical
connection.

## Dry-run smoke test

Before paying the cost of a real load, validate the CLI surface:

```
embed-art validate-vsd \
  --concept 'thunder' \
  -o docs/superpowers/results/vsd-validation/thunder \
  --dry-run
```

Expected: writes `plan.json` describing the run that would happen on
real weights. Exits 0 without touching any model weights.

## Real run

```
embed-art validate-vsd \
  --concept 'thunder' \
  -o docs/superpowers/results/vsd-validation/thunder \
  --steps 50 \
  --device mps \
  --rank 4 \
  --guidance-scale 7.5 \
  --seed 0
```

Expected artefacts:

```
docs/superpowers/results/vsd-validation/thunder/
├── plan.json
├── honest.png          # high-similarity, model-honest track
├── natural.png         # VSD-prior natural track
├── comparison.png      # side-by-side
└── manifest.json       # per-track encoder similarities + provenance
```

## Validation checklist

Per the issue:

* [ ] SD3.5 weights load without error on the target device.
* [ ] `build_vsd_phi_adapter` reports `injected > 0` LoRA modules.
* [ ] Dual-track loop runs to completion in ≤30 min on M1 Max with
      `--steps 50`.
* [ ] `natural.png` looks visibly distinct from `honest.png` — the
      natural track should retain photorealistic crispness without
      converging to a marginal-mode lookalike.
* [ ] `manifest.json` reports:
  * `honest_track.target_similarity > 0.85`
  * `natural_track.target_similarity > 0.55`
  * `delta = honest_target_similarity - natural_target_similarity < 0.4`
* [ ] No NaN losses in the per-step log.
* [ ] LoRA parameter L2 norm grows over training (the φ student is
      actually learning, not stuck at init).

If the natural track collapses to a marginal-mode image (gauzy
DeepDream-ish output that looks unlike the concept), check:

1. The teacher's `score_fn` is actually wrapped in `lora_disabled()`
   so the teacher is the *frozen* prior.
2. `student_steps_per_generator_step >= 1`.
3. Guidance scale isn't pinned absurdly high (>15) — that drives the
   student towards delta-function modes.

## Outputs land in

```
docs/superpowers/results/vsd-validation/<concept>/
```

Once a concept passes the validation checklist, commit the manifest
+ a thumbnail comparison.png to that directory. The expected output
documents the canonical 'thunder' baseline that the natural track
must match qualitatively.

## Follow-ups

* **Loop wiring.** `embed-art validate-vsd` currently sets up the
  adapter graph and exits before running the optimisation loop. The
  remaining work is plumbing — wiring `VSDLoss` and `dual_track_loss`
  through a real image generator. This is the smallest standalone
  follow-up after this runbook lands.
* **Multi-concept battery.** Once the 'thunder' concept passes, run
  the same pipeline across the M9 anchor-comparison concept set
  (thunder, goldfish, cathedral, applause, fire) and store the
  per-concept manifests in subdirectories.
* **CI provenance.** The manifest should record the SD3.5 weight
  hash + LoRA seed so the validation is reproducible. Add this once
  the loop is wired.

File the loop-wiring follow-up via `bd` and reference this runbook
when the M1 Max user lands the validated artefacts.
