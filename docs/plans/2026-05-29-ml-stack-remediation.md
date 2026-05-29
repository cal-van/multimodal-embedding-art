# ML-stack remediation plan (2026-05-29)

From `docs/reviews/2026-05-29-ml-stack-expert-review.md`. Executed in waves; full `pytest` + `ruff` (and `tsc`/`build` for FE) as the gate after each; commit per wave. Backend waves share files (loss/engine/strategies/languagebind/showcase) so run sequentially; frontend copy is delegated.

Principle: implement everything that is **high-confidence correct and verifiable by the test suite**. For items that require real-weight validation or large risky refactors, do the safe part and track the remainder as a beads issue — do NOT ship unvalidated ML paths (that would recreate the silent-wrongness the review flagged).

## Wave A — P0 correctness: modality dispatch + double-ViT
- `engine.render` → pass `output_modality` to `strategy.render`.
- `OptimizationStrategy.render` → accept `output_modality`, pass to `CompositeLoss`.
- `CompositeLoss` → take `modality`; `_encode` dispatches: image→`encode_for_optimization`, video→`encode_video_for_optimization`, text→`encode_text_for_optimization`, audio→`encode_audio_for_optimization`.
- `LanguageBindEncoder.encode_audio_for_optimization`: raise a clear `EncoderError` ("differentiable audio optimisation not yet validated") — fail LOUD instead of silently mis-encoding. (Validated differentiable mel path tracked separately.)
- Double-ViT: add `encode_with_layers(tensor)` returning `(projected_normalized, hidden_states)` from ONE `vision_model(..., output_hidden_states=True)`; `CompositeLoss` uses it when feature-matching is active instead of two forwards.
- Tests: per-modality dispatch (mock encoder, assert correct method called); audio raises; single-forward used when feature-matching on.

## Wave B — performance (safe, verifiable)
- `stable_audio_open.py`, `ltx_video.py`: load `subfolder="vae"` only (mirror `sd35.py`), not the full pipeline.
- `showcase.py`: release the previous modality's generator (pop from engine, `del`, `gc.collect()`) BEFORE `empty_mps_cache()`.
- `perf/__init__.py`: default `dynamic=False`; on MPS prefer `mode="default"`; keep call-time + first-call guard. Drop `max-autotune` from the CLI choice list (or gate to CUDA).

## Wave C — evaluation honesty
- `showcase.py _compute_evaluation_card`: compute the REAL pairwise cross-modal cosine between per-modality final embeddings (the metric the README claims), as the primary cross-modal block; relabel word-Jaccard as secondary "lexical overlap".
- Add a `--baseline-random` option that re-runs the card against a random unit-norm target so users can see the discriminative gap.
- Keep cross-encoder probes opt-in (heavy) but document them as the gold-standard independent check and surface prominently in the card when present.
- Persist achieved per-modality cosine honestly; stop implying >0.85.

## Wave D — docs + interpretability honesty (docs by me; FE copy by subagent)
- `README.md` + `CLAUDE.md`: reframe "what the model sees" → "what this generator can produce that this encoder scores highly"; disclose real cosine range (~0.14–0.41 observed); fix the "natural = VSD prior" claim → "heavy regularisation (VSD planned)"; fix the cross-modal "0.90 cosine" claim to match the implemented metric; relabel the success-metric section (cosine is a convergence diagnostic, the cross-encoder probe is the validation).
- FE (subagent): `ShowcasePage.tsx` `realismCaption`/lede — honest reframe of honest/natural; note honest may be adversarial-leaning.

## Wave E — architecture (safe high-value)
- `languagebind.py encode(spec)`: use real `ConceptSpec` fields (`text/image/audio/video`) not `spec.modality/value`.
- `job_manager.get_engine()`: point web at the v3 registry/LanguageBind + SD3.5/SAO/LTX (match CLI), not the deprecated imagebind/SDXL stack.
- Add WARNING logs on silent skips (registry import skip, feature-matching calibration skip, compile fallback) and record skipped components in the manifest.

## Deferred → beads issues (need real-weight validation or large risky refactor)
1. Validated differentiable AUDIO optimisation path (mel transform matching LanguageBind's processor).
2. Transformation-robustness ensemble (multi-view averaging) + diversity objective — Olah's top interpretability rec.
3. Wire `PatchAlignmentLoss` into `CompositeLoss` (needs weight tuning + validation) or delete.
4. Collapse the three optimisation loops (`engine.optimize`, `_run_diffusion_generation`, `OptimizationStrategy`) into one.
5. Land the VSD/SDS prior for the natural track.

## Gate
`pytest -q && ruff check src tests` ; `cd ui && npx tsc --noEmit && npm run build && npx vitest run`.
