# Implementation plan: review fixes (2026-05-29)

Derived from `docs/reviews/2026-05-29-realism-dial-and-ui-redesign.md`. Executed in waves; verify after each (tsc + build + vitest for FE, pytest + ruff for BE). Commit per wave.

## Wave 1 — Correctness (P0/P1), done by hand
1. **Manifest contract (P0).** Rewrite `api.ts` `ShowcaseManifest`/`ShowcaseModalityRender`/`ShowcaseTrackSummary` to the real backend schema (`modalities` object of `{path, final_similarity, backbone, interpretation}`; dual-track top = `{tracks, per_track:{<t>:{path, manifest, summary}}}`, no top-level `modalities`/`renders`). Rewrite `ShowcaseManifestView` to read `manifest.modalities` (object) for single-track and `per_track[t].summary` for dual-track. Guard the modalities label line. Add a fixture test (`ui/src/test`) loading a committed real manifest and asserting the consumed keys render without throwing.
2. **Perf default (P1).** `jobs.py` + `job_manager.py`: web `compile_mode` default `"none"`. Keep `bf16`. Add a comment documenting the intentional bf16 CLI(fp32)/web(bf16) divergence.
3. **Security validators (P1/P6).** Add `@field_validator` allowlists for `compile_mode` + `autocast_dtype` on `CreateShowcaseRequest`. Add `math.isfinite` guard to realism on CLI + web. `weights_only=True` on `linear_probes.py:310` + `activation_cache.py:140`. Restrict `sae_path` under cwd/outputs (best-effort).
4. **Tests (P1).** Add: web compile_mode+realism forwarding regression test; exact endpoint regularizer-weight assertions; web realism/tracks 422 validator tests; patch-assert interpolate_loss_config used in the realism render.

## Wave 2 — Accessibility (P1), parallel subagent + my CSS
5. `index.css`: add `--text-label` (~#7d8590) and apply to `.eyebrow`/`.field-label`/`.nav-item .idx`/placeholder; mobile `@media (max-width:700px)` collapsing the rail + reducing console padding; `input[type=range]:focus-visible` thumb ring; bump touch-target padding on `.chip`/`.preset`/`.nav-item`/`button`/slider thumb hit area; add a `.pulse` class and cover inline pulses + transitions in the reduced-motion block.
6. ShowcasePage: `aria-label`+`aria-valuetext`+min/max on both sliders; `aria-pressed` on modality chips; associate the realism `<span>` label. JobPage/ManifestView: meaningful `alt`; swap inline `animation:'pulse...'` for the `.pulse` class. AnchorCompare: `aria-pressed` on mode chips; fix double-`<label>` in FileSlot.

## Wave 3 — Product elevation (P2), parallel subagent
7. ShowcasePage: wrap autocast/compile/seed in an "Advanced" `<details>`; promote steps as a "Quick preview ↔ Full quality" control; add a runtime-estimate line near submit; rewrite the lede to lead with outcome; rename "(ablation)" → "(comparison)".
8. Sidebar: move "Single-modality" below the rule with a "legacy" tag.
9. AnchorCompare matrix: two-branch cosine heat color (negatives visible).
10. Backend: surface `realism` in `JobResponse`; add web validator/CLI warning when realism + explicit tracks both set.

## Out of scope this pass (parking lot)
- Full CLI/web param parity (seed_stability/text_anchor_weight/probes/linear_probes on web).
- JobPage logs virtualization; compile fallback hot-loop hardening + dynamic=False.

## Verification gate
`cd ui && npx tsc --noEmit && npm run build && npx vitest run` ; `pytest -q && ruff check src` . Browser screenshot of Showcase (mobile + desktop) + a real manifest render in JobPage.
