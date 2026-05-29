# Review: realism dial + perf defaults + "Latent Instrument" UI redesign

**Date:** 2026-05-29
**Scope:** commits `673d0c4` (realism dial + perf) and `4d5793c` (UI redesign) vs `9d246cd`. 16 files, +1975/−1015.
**Method:** comprehensive-review + expert-review run as a unified 7-agent parallel panel (Performance[opus], Architect[opus], Test Quality, Frontend Correctness, Accessibility, Product, Security). All findings below were **independently validated against the actual code** before inclusion; two reviewer claims were dismissed as incorrect (noted at the end).

**Verdict:** NEEDS FIXES — one validated P0 (the canonical showcase output cannot render in the web UI), plus a perf-default regression and a cluster of public-release a11y/security/test gaps. None block the *Python pipeline*; the P0 blocks the *web display* of showcase results.

---

## Scorecard

| Dimension | Verdict | Headline |
|-----------|---------|----------|
| Performance | NEEDS FIX | `reduce-overhead` compile as a web *default* likely slows the common 200-step run; bf16 default is sound. |
| Architecture | NEEDS FIX | Manifest contract mismatch (P0); CLI/web perf defaults silently diverge; realism silently overrides tracks. |
| Test Quality | NEEDS FIX | The exact bug just fixed (compile_mode forwarding) has no regression test; endpoint test asserts counts not weights. |
| Frontend Correctness | NEEDS FIX | ManifestView crashes on real manifests (P0). WebSocket/data logic correctly preserved. |
| Accessibility | NEEDS FIX | No mobile layout (<700px unusable); realism slider invisible to AT; `--text-faint` labels fail AA contrast. |
| Product | ELEVATE | Compute knobs crowd the creative path; no runtime estimate; legacy page over-promoted in nav. |
| Security | HARDEN | Web `compile_mode`/`autocast_dtype` lack allowlists; `sae_path` unrestricted; two `torch.load` lack `weights_only`. |

---

## P0 — Must fix (core feature broken)

**1. `ShowcaseManifestView` cannot render a real showcase manifest.**
- `ui/src/components/ShowcaseManifestView.tsx:88` calls `manifest.modalities.join(', ')`. A real single-track manifest (`outputs/test_run_2/manifest.json`) has `modalities` as an **object** (`{image: {...}, audio: {...}}`), and the dual-track top manifest has **no** `modalities` key at all → `.join` throws, crashing the component.
- The per-modality renders live under `manifest.modalities` (fields: `path`, `final_similarity`, `backbone`, `interpretation`), but the view branches on `manifest.renders` (line 100) which the backend never writes → falls through to "Manifest contains no renders yet."
- `ShowcaseModalityRender` in `api.ts` expects `output_file`/`similarity`; backend writes `path`/`final_similarity`. `ShowcaseTrackSummary` expects `modalities`/`similarity_summary`/`output_dir`; backend `per_track[t]` is `{path, manifest, summary}` where `summary` is `{<mod>: {path, final_similarity, backbone}, _evaluation}`.
- **Pre-existing** (api.ts was always wrong; the redesign preserved the broken contract). **Fix:** reconcile `api.ts` types + `ShowcaseManifestView` with the real schema; add a fixture test that loads a committed real `manifest.json` and asserts the consumed keys.

---

## P1 — Should fix

**2. Revert web `compile_mode` default to `"none"` (keep bf16).** `perf/__init__.py:55` `compile_module` defaults `mode="reduce-overhead", dynamic=True`; it's invoked once per modality per track inside `OptimizationStrategy.render()`, so a default web run pays 4 (or 8) separate Inductor warmups amortized over only 200 steps, and `reduce-overhead`'s CUDA-graph layer is largely inert on MPS. The silent fallback only guards `torch.compile()` call-time, not the lazy first-call compile in the hot loop. Net: the default likely makes first runs *slower*. bf16 autocast is SAFE (fp32 backward retained — verified at `strategies.py:184`). Reverting compile→none also removes the compile half of finding #3.

**3. CLI/web perf defaults silently diverge.** CLI defaults `fp32`/`none` (`showcase.py:212,221`); web defaults `bf16`/`reduce-overhead`. The system's stated invariant is CLI/web lockstep. Decision: keep an **intentional, documented** divergence (web = interactive M1 Max → bf16; CLI = reproducible/ablation → fp32) and align compile to `none` on both. Add a code comment so it's not silent.

**4. Test gaps (cheap, high value).**
- MISSING: regression test that web `compile_mode` **and** `realism` reach `_showcase_impl` (the exact bug just fixed — `job_manager.py:493-496`).
- WEAK: `test_endpoints_match_named_tracks` checks regularizer *count* + sim/fm only; a drift in `_TV_WEIGHTS`/`_SPECTRAL_WEIGHTS`/`_LATENT_NORM_WEIGHTS` passes silently. Assert exact weights vs `minimal()`/`heavy()`.
- MISSING: web `CreateShowcaseRequest` realism + tracks validator tests (422).
- WEAK: `test_realism_renders_single_labelled_track` uses text-only modality, so it never exercises `interpolate_loss_config` being *used*. Patch-and-assert it's called with the right realism.

**5. Accessibility (public-release blockers).**
- **No mobile layout** — `.layout` is a fixed `232px 1fr` grid with no `@media`; at 375px the console is ~47px wide. Add a ≤700px breakpoint collapsing the rail. (WCAG 1.4.10)
- **Realism slider invisible to AT** — no `aria-label`/`aria-valuetext`/value association. Add `aria-label` + `aria-valuetext={caption.title}` + min/max; same for steps slider. (4.1.2)
- **Contrast** — `--text-faint` #5a606b at 2.96:1 fails AA for every `.eyebrow`/`.field-label`/`.nav-item .idx`/placeholder. Introduce a `--text-label` (~#7d8590, ≥4.5:1) for label roles. (1.4.3)
- Slider has no focus-visible ring (2.4.7); touch targets (chips/presets/nav/buttons/thumb) below 44px (2.5.5); inline `animation:pulse` not covered by the reduced-motion block; modality/mode chips lack `aria-pressed`; generic alt text on result images.

**6. Security (calibrated for local-first + public release).**
- Web `compile_mode`/`autocast_dtype` are bare `str` with no allowlist (CLI uses `click.Choice`) — arbitrary string reaches `torch.compile`. Add `@field_validator`s.
- `sae_path` accepts arbitrary filesystem paths → local file read on a non-local deploy. Restrict under `outputs/` (or a configured dir).
- `torch.load` without `weights_only=True` at `evaluation/linear_probes.py:310` and `experiments/activation_cache.py:140` (CLI-only today, but repo is going public — pickle RCE footgun). Add `weights_only=True`.
- Add explicit `math.isfinite` guard on realism (defense-in-depth; current NaN rejection relies on a subtle comparison).

---

## P2 — Nice to have / product elevation

7. Realism silently overrides `tracks` — add a web validator/CLI warning when both are set deliberately.
8. Surface `realism` in `JobResponse` (`_map_job_to_response`) so polling clients can read the dial position.
9. AnchorCompare cosine matrix clamps negatives to transparent (`AnchorComparePage.tsx:435`) — restore a two-branch heat color (it already exists in `ShowcaseManifestView.cosineHeatColor`).
10. **Product: progressive disclosure** — collapse compute knobs (autocast/compile/seed) behind an "Advanced" `<details>`; keep concept → dial → steps → run prominent.
11. **Product: runtime estimate** near the submit button; reconcile the UI steps default (200) vs README (2000) with a "quick preview ↔ full quality" framing.
12. **Product: IA** — demote "Single-modality" below the nav rule with a "legacy" tag; rewrite the Showcase lede to lead with the outcome, not "LanguageBind embedding space"; rename generator "(ablation)" → "(comparison)".
13. Cap/virtualize the JobPage logs panel and compute per-line color once (unbounded DOM growth over long runs).

## Parking lot
- Full CLI/web param parity (`seed_stability`, `text_anchor_weight`, `probes`, `linear_probes_dir` are CLI-only; the UI even renders eval blocks the web can never fill). Plumb through or document.
- `torch.compile` fallback hardening (catch hot-loop first-call failures) + per-modality `dynamic=False`.

## Dismissed reviewer claims (validated as incorrect)
- "Gallery hides non-image thumbnails (regression)" — backend saves raw `.wav`/`.mp4` as `result_url`; the old code rendered broken `<img src=.wav>`. The new glyph placeholder is an **improvement**, not a regression.
- "ManifestView `isHonest` includes('honest') bug" — harmless today; single-realism uses the flat layout, not `per_track`. Tracked as P2 #—(hardening only).
