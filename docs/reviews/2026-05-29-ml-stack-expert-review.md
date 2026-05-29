# Expert Review: full ML stack

**Date:** 2026-05-29
**Panel (personas channeling specific lenses):** Chris Olah (interpretability methodology), Soumith Chintala (ML systems architecture), Horace He (PyTorch/MPS performance), Been Kim (evaluation rigor).
**Scope:** the core ML pipeline — encoders, generators, engine/loss/regularizers, SAE, interpretation, evaluation, diffusion priors, perf layer (~15K LoC under `src/embedding_art`, excluding web/cli).
**Project stage:** alpha, prepping for public release.

## Executive summary

The instinct — render a concept across modalities by optimising toward a shared multimodal embedding — is genuinely novel and the scaffolding (registry, protocols, lazy-load, perf knobs) is well-built. But the panel converged on a hard conclusion: **as it stands the project is a compelling generative-art pipeline wearing interpretability's vocabulary, and several flagship behaviours are silently broken or overclaimed.** One **P0 correctness bug** (audio/video are encoded through the image encoder) was confirmed by direct code read. The evaluation card is **circular** (it reports the optimisation objective back as a success metric) and the headline ">0.85 cosine" target is **missed ~2x by every on-disk example** with no disclosure. The "honest" interpretability track uses minimal regularisation that classic feature-viz says produces *adversarial* artefacts, not faithful ones. And two pieces of described machinery — `PatchAlignmentLoss` and the VSD/SDS prior — are **dead/unwired** despite docs claiming they're active.

**Overall health: At Risk** (for the interpretability/scientific claims and the cross-modal feature). The art output and architecture bones are salvageable with focused work.

## Scorecard

| Lens | Expert | Verdict | Headline |
|------|--------|---------|----------|
| Interpretability | Olah | Needs reframing | Artefact is co-authored by the frozen generator manifold; "honest" low-reg track is the *least* trustworthy; key machinery is dead code. |
| Architecture | Chintala | At risk (1 P0) | Modality dropped before the loss → audio/video encoded as images; 3 parallel optimisation loops; contract lives in `hasattr`. |
| Performance | He | Needs fixes | "~19GB" is fiction (full pipelines loaded then discarded; generators never released); compile defaults wrong on MPS; double ViT forward per step. |
| Evaluation | Kim | At risk | Success metric is circular; 0.85 target missed ~2x undisclosed; only independent check is off by default; no random baseline anywhere. |

---

## Cross-expert themes

**Theme 1 — "Claims exceed reality" (all four experts).** Olah: "what the model sees" attributes to LanguageBind something co-authored by SD3.5. Kim: the eval card reports the objective back to itself and misses its own target 2x. Chintala: VSD claimed in README but unwired; modality silently mis-handled. He: the "3-5x" / "~19GB" performance claims are CUDA folklore / false on MPS. **The unifying fix is honesty:** reframe claims to match what the code actually does, and make the metrics independent.

**Theme 2 — "Built but not wired" (Olah + Chintala).** `PatchAlignmentLoss` (the multi-layer geometric constraint), the VSD/SDS prior subsystem, `encode_video_for_optimization`, and the v3 `encode(spec)` path on the canonical encoder are all present but not actually invoked by the flagship path. Impressive surface area, partially inert.

**Theme 3 — "Silent fallbacks stack into undebuggable success" (Chintala + He + Olah).** Registry import-skip, feature-matching calibration skip (always skipped for text targets), compile fallback, modality mis-dispatch — a showcase can report "completed" while two modalities are mis-encoded, one loss term is inert, and compile silently did nothing.

---

## P0 — Must fix (confirmed correctness bug)

**1. Audio & video are optimised against the IMAGE encoder.** `engine.render` (`core/engine.py:213-218`) computes the right generator from `output_modality` but calls `strategy.render(target, generator, encoder, config)` **without** the modality. `OptimizationStrategy.render` has no modality param; it builds `CompositeLoss` and calls `loss_fn(...)`, whose `_encode` (`core/loss.py:114-122,177`) only ever calls `encode_for_optimization` — which hard-loads the **image** model and expects `[B,3,H,W]` (`encoders/languagebind.py:560-592`). A correct `encode_video_for_optimization` exists (`:594-632`) but is **never called**; `encode_audio` raises `NotImplementedError` for tensor input (`:485-492`) and has no `_for_optimization` variant. **Consequence:** the flagship `showcase` either errors on the 5D video tensor or mis-encodes audio/video through the image ViT — every audio/video `final_similarity` and the cross-modal agreement matrix are invalid. The legacy `engine.optimize._encode_for_modality` (`:614-631`) already dispatches correctly; the v3 rewrite regressed it.
**Fix:** thread `output_modality` through `render → strategy → CompositeLoss._encode` and dispatch to `encode_for_optimization` / `encode_audio_for_optimization` / `encode_video_for_optimization`. Implement the missing differentiable audio path. Add a per-modality test asserting the correct encoder method is called.

---

## P1 — Should fix

**Evaluation (Kim):**
- **Circular headline metric.** `cosine > 0.85` is the loss reported back as a score (`showcase.py:1450-1454`, `CLAUDE.md` Success Metrics). Stop presenting self-cosine as success; label it a convergence diagnostic.
- **Undisclosed 2x miss.** Every on-disk run is far below 0.85: `outputs/test_run_2` = 0.139; curated gallery tops out ~0.35–0.41. Disclose the real achieved range or drop the threshold.
- **Promote the cross-encoder probe to default + headline.** It's the only genuinely independent signal (a *different* encoder agreeing — `evaluation/probes.py:61-139`), implemented correctly, but OFF by default (`--probes ""`, `showcase.py:271-281`). Default it to `siglip2-so400m,clap-general`.
- **Claimed≠implemented cross-modal metric.** README claims "outputs within 0.90 cosine of each other"; the code computes Jaccard overlap of top-K anchor *words* (`showcase.py:1461-1480`). Compute the actual pairwise embedding cosine (trivial — embeddings are in hand); relabel word-Jaccard secondary.
- **No baseline anywhere.** Add a random/shuffled-target eval card column. If "thunder" and noise produce the same numbers, the metrics measure nothing. Turn seed-stability on by default (≥3).

**Interpretability (Olah):**
- **Reframe the dial / the headline claim.** The artefact is "what this generator can produce that this encoder scores highly," not "what the model sees." The low-reg "honest" track is the most likely to be adversarial-for-LanguageBind; relabel it ("maximal activation, may be adversarial") or invert the framing. Run the sd35-vs-sdxl backbone ablation to quantify generator dominance.
- **Add transformation-robustness ensemble + a diversity objective** — the single highest-value feature-viz technique missing. Currently one jittered view per step (`strategies.py:177`), no diversity → single-mode, adversarial-leaning maximiser.
- **Wire in `PatchAlignmentLoss` or delete it** (`core/patch_alignment.py` is never referenced). **Fix the VSD claim**: README/`--tracks` help say "natural = VSD prior"; it's actually heavy regularisation (`diffusion_priors/*` unwired).

**Performance (He):**
- **Memory bombs.** `stable_audio_open.py:99` and `ltx_video.py:99` load the *full* multi-GB pipeline then keep only `.vae` — load `subfolder="vae"` like `sd35.py` does. Generators are never released between modalities (all VAEs + LanguageBind resident at once); `empty_mps_cache()` (`showcase.py:653`) frees nothing because refs are live — drop the prev generator + `gc.collect()` *before* emptying.
- **Double ViT forward per step.** On the default honest track, `loss.py:177` encodes for cosine, then `:183` runs the entire ViT again for hidden states. One `vision_model(..., output_hidden_states=True)` yields both — ~30-40% step-time cut.
- **Compile defaults wrong for MPS.** `reduce-overhead` (CUDA-graphs, inert on MPS) + `dynamic=True` (pessimises fixed shapes) at `perf/__init__.py:56-61`; compiling the whole `CompositeLoss` graph-breaks on Python branches. Default `dynamic=False`/`mode="default"`, compile the ViT not the loss, drop `max-autotune` from the MPS choices. SDPA flash routing is the one lever that actually works on MPS — keep it.

## P2 / Architecture (Chintala)

- Collapse the **three** optimisation loops (`engine.optimize`, `_run_diffusion_generation`, `OptimizationStrategy`) into one; make `optimize()` a thin shim. Dedup the verbatim-copied `_augment`.
- Make the **Encoder Protocol the real contract** (declare `encode_for_optimization`, `get_layer_features`, `card`, `unload`); drive routing off `card.capabilities`, not scattered `hasattr`.
- Two incompatible `ConceptSpec` dialects: LanguageBind's `encode(spec)` reads `spec.modality/value` (don't exist); SigLIP2 reads the real `text/image/...` fields → `encode(spec)` is broken for the canonical encoder (`encoders/languagebind.py:730-755`).
- The **web stack is pinned to the deprecated v1 pipeline** (`job_manager.get_engine()` hardcodes `imagebind` + SDXL/AudioLDM/SVD) — web and CLI run different pipelines. Point it at the v3 registry.
- Log on every silent skip (registry import, calibration, compile) and record skipped components in the manifest so a "green" run is verifiable.

---

## Detailed expert reports
The four full expert write-ups (with persona framing and complete finding tables) were produced in the review run and are summarised above. Re-run the panel to regenerate verbatim if needed.

## Recommended sequencing
1. **P0 modality dispatch** — the flagship feature is silently broken; nothing else matters until cross-modal similarities are real.
2. **Evaluation honesty** (default cross-encoder probe + random baseline + disclose real cosine range + fix claimed-vs-implemented metric) — cheap, and it's what makes the project credible to a skeptical reader.
3. **Reframe interpretability claims** + wire-or-delete PatchAlignment/VSD — removes the overclaim risk before public release.
4. **Perf: memory release + single ViT forward** — makes a real showcase actually fit the budget and run faster.
5. **Architecture consolidation** — pays down the "three loops / hasattr contract" debt so model-swapping is genuinely easy.
