# Test Plan — PR #1 linear-probes wiring in `ShowcaseManifestView`

## Scope

The PR is large (25 commits), but most of its claims (LanguageBind canonical encoding, SD3.5
generation, dual-track rendering, real LoRA VSD, multi-modal anchor-compare end-to-end) require
real model weights (~10GB) plus M1 Max hardware to validate meaningfully. None of those are
available on this Linux test box.

The single new user-visible UI deliverable that *is* fully testable here is the **linear-probes
row** added to each modality's interpretation block in `ShowcaseManifestView`. That row only
appears when the manifest's per-modality `interpretation.linear_probes` field is populated, so
a synthetic manifest is sufficient to prove the rendering path works.

This is the right test target because:
- It exercises the actual code that landed in commit `f8bc705`
  (`ui/src/components/ShowcaseManifestView.tsx:193-250`).
- A broken implementation would produce a visibly different result — either no row at all,
  or formatting bugs (e.g. the multi-class probe rendering as `[object Object]`).
- It does not require any model weights or GPU.

## Test target

Single end-to-end UI flow:

1. Navigate to a route that renders `<ShowcaseManifestView />` against a synthetic manifest
   containing `linear_probes` on every modality, then verify each row is rendered with the
   correct formatting.

A temporary test-only route `/test-manifest` has been added to `ui/src/App.tsx` for this. It
mounts `ShowcaseManifestView` directly with a hardcoded manifest URL pointing at
`/outputs/showcase/synth-test/manifest.json`, served by the FastAPI `StaticFiles` mount on
`/outputs/`. The route will be reverted before the session ends.

## Synthetic manifest

`outputs/showcase/synth-test/manifest.json` is hand-written with three modalities carrying
linear probes:

- **image**: `is_weather_concept = 0.91` (binary float), `scene_category = {indoor: 0.04, outdoor: 0.88, abstract: 0.08}` (multi-class dict)
- **audio**: `is_weather_concept = 0.84`, `scene_category = {indoor: 0.03, outdoor: 0.91, abstract: 0.06}`
- **video**: `is_weather_concept = 0.78`, `scene_category = {indoor: 0.05, outdoor: 0.86, abstract: 0.09}`
- **text**: no interpretation block (correct — text card is markdown only)

## Test steps

### Test 1: linear-probes row renders with correct content for binary + multi-class probes

**Setup**: Backend at `127.0.0.1:8000`, Vite at `127.0.0.1:5173`. Both already running.

1. Navigate browser to `http://127.0.0.1:5173/test-manifest`.

2. Wait for the manifest fetch to complete (the "Loading showcase manifest…" placeholder is
   replaced by three modality cards: image / audio / video).

3. **Assertion A — image card linear-probes row**

   - Expected: under the image modality card, beneath the "text-anchor:" and "SAE features:" lines,
     there is a row labelled `linear probes:` whose content reads literally:

       `is_weather_concept=0.91 · scene_category: outdoor (0.88)`

   - The `=0.91` confirms the binary-probe branch in `formatProbeReading` is taking the `number`
     case correctly.
   - The `scene_category: outdoor (0.88)` confirms the multi-class branch is taking the argmax
     correctly (`outdoor` has the highest value at 0.88).
   - The middle separator is `·` (middle dot), per `ShowcaseManifestView.tsx:228`.

   - **What a broken implementation would look like** (so we know this is a meaningful check):
     - No `linear probes:` row at all (renders the modality but skips the probes branch).
     - `is_weather_concept=[object Object]` (forgot the `typeof value === 'number'` branch).
     - `scene_category: indoor (0.04)` (took the wrong entry — first instead of argmax).
     - Raw JSON dumped instead of formatted (the `Object.entries(...).map(...)` failed).

4. **Assertion B — audio card linear-probes row**

   - Expected: under the audio modality card, the linear-probes row reads:

       `is_weather_concept=0.84 · scene_category: outdoor (0.91)`

   - Different numbers and a different argmax probability than the image card; confirms the row
     is per-modality and not accidentally shared.

5. **Assertion C — video card linear-probes row**

   - Expected: under the video modality card, the linear-probes row reads:

       `is_weather_concept=0.78 · scene_category: outdoor (0.86)`

6. **Assertion D — text card omits linear-probes (because its manifest entry has no interpretation block)**

   - Expected: there is no `linear probes:` row under the text modality card.

7. **Assertion E — text-anchor and SAE-features rows continue to render (regression)**

   - Expected: the image card still shows a `text-anchor:` row including the words `storm`,
     `lightning`, `crackling` (with similarities formatted to 2 decimal places), and a
     `SAE features:` row including `weather: storm features` and `high-frequency edges`.
   - This proves the probes addition did not break the existing rows.

8. Capture a single full-window screenshot of `/test-manifest` showing all three modality cards
   with their `linear probes:` rows side by side.

## Why this is adversarial

- The two formatting branches (`number` vs `Record<string, number>`) are distinct code paths in
  `formatProbeReading` (`ShowcaseManifestView.tsx:235-250`). One synthetic manifest exercises both.
- The argmax logic is non-trivial — picking `outdoor (0.88)` over `indoor (0.04)` and
  `abstract (0.08)` requires the reducer at lines 245-248 to compare correctly. A broken
  implementation (e.g. picking the first entry, or last entry) would show `indoor` or `abstract`
  in the rendered string.
- The per-modality population (different scores for image / audio / video) catches accidental
  state-sharing or off-by-one in the modality iteration.
- The text card having no probes catches a "probes are global, not per-modality" bug.

## What this test does NOT cover (acknowledged blind spots, escalated up front)

1. **Real linear-probe training + showcase end-to-end**. The `embed-art train-probes` CLI is
   covered by fast unit tests; the showcase CLI + `--linear-probes-dir` round-trip is covered
   by `tests/test_showcase_cli.py::TestShowcaseLinearProbes`. But no end-to-end run on a real
   M1 Max with real LanguageBind + SD3.5 has been performed. This is a follow-up.
2. **Multi-modal anchor-compare UI**. Backend handler requires LanguageBind weights; not testable
   here without ~4GB of downloads and minutes of model load. The UI page renders without a
   backend response, but the round-trip is out of scope.
3. **VSD natural track**. Math is unit-tested; real-weight wall-time on M1 Max is filed as a `bd`
   issue.

## Cleanup after testing

- Remove the temporary `/test-manifest` route from `ui/src/App.tsx`.
- Remove the `outputs/showcase/synth-test/` directory.
- Stop the backend and Vite dev server.
