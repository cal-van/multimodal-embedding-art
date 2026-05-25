# Gap Analysis & Proposed Next Steps

**Date:** 2026-05-24
**Author:** Devin (for Callum)
**Status:** Draft — awaiting scope confirmation before any code changes
**Companion docs:**
- `docs/superpowers/specs/2026-03-19-embedding-art-v2-design.md` — existing v2 design (still relevant; this builds on it)
- `docs/superpowers/specs/2026-03-19-research-findings.md` — research as of mid-March 2026

---

## 1. Where we are now

The v2 refactor on `main` is substantial and mostly **infrastructure** rather than **artefacts**. It puts the right abstractions in place, but the things the vision actually wants to *produce* — interpretable feature decompositions, multi-modality side-by-side renders, quantitative evaluation, learned regularisers — aren't there yet.

### 1a. What is shipped (and works against the vision)

| Capability | Status | Files |
|---|---|---|
| Concept algebra (add/sub/scalar/slerp) + `Concept.source_input` for layer matching | shipped | `core/concept.py`, `core/concept_spec.py` |
| Encoder registry with `EncoderCard` / `EncoderCapability` | shipped | `encoders/registry.py`, `encoders/defaults.py` |
| SigLIP 2 (text+image, multi-layer features), CLAP (text+audio), ImageBind | shipped | `encoders/siglip2.py`, `encoders/clap.py`, `encoders/imagebind.py` |
| `CompositeLoss` with cosine + multi-layer feature matching + SAE feature loss + reg | shipped | `core/loss.py`, `encoders/features.py` |
| `OptimizationStrategy` + `DiffusionGuidanceStrategy` + augment/callback/checkpoint | shipped | `core/strategies.py`, `generators/diffusion.py` |
| INR generator (Fourier-features MLP) | shipped (basic) | `generators/inr.py` |
| SAE training pipeline (TopK, group-sparse, per-modality pre-biases) + `SAELens` | shipped | `sae/training.py`, `sae/lens.py` |
| CLI: `render`, `compare`, `decompose`, `sae collect/train/label/inspect`, `interpolate-v2` | shipped | `cli/commands/*.py` |
| Web UI scaffold (FastAPI + React) | shipped | `src/embedding_art/web/`, `ui/` |
| Unit/integration test suite for the above | shipped | `tests/test_*.py` |

### 1b. What is specced but not yet built (per `2026-03-19-embedding-art-v2.md`)

The v2 plan has 67 checkboxes; the *code* of phases 1–5 is largely in place, but several mission-critical *artefacts and validations* are not:

| Gap | Why it matters | Effort |
|---|---|---|
| **No trained SAE checkpoint shipped.** Pipeline exists; no `.safetensors` produced from a real corpus. `decompose` and SAE-feature loss can't run end-to-end. | The entire interpretability story depends on this. | M |
| **No feature labelling artefact** — `label_features` exists in code but no vocab.json has been generated and committed for any encoder. | Without labels, "active features" are just integers. Vision says human-readable names. | S |
| **No CLAP SAE.** Group-sparse design requires paired image+audio embeddings; only the image side is wired into `collect_embeddings`. | Cross-modal SAE — the load-bearing claim of the group-sparse paper — is currently unverifiable. | M |
| **No cross-modal alignment validation.** PROJECT.md target is "Cross-modal outputs from same target embed within 0.90." There's no test or script that measures this. | Without this number, the "side-by-side same concept" art deliverable has no quality bar. | S |
| **No evaluation harness.** Callum's own "what I'd do differently" note: "I'd build the evaluation alongside the generation rather than after it." Still not done. | The regulariser-legibility complaint can only be answered with numbers. | M |
| **SVD video generator is in the codebase but slow/flaky.** Not exercised by `render`/`compare` at the same fidelity as image. | Four-modality showcase needs working video. | M |

### 1c. What is NOT in the v2 spec at all (the real gap relative to the vision)

These are the directions that the vision document asks for but neither the shipped code nor the v2 plan addresses:

1. **Quantitative evaluation harness** — held-out similarity, seed-stability, regulariser-ablation grid. Callum's own retrospective calls this out. (`What I'd do differently`, paragraph 1.)
2. **Regulariser legibility** — the TV / spectral / latent-norm weights are still magic numbers tuned by feel. Vision wants them derived, not empirical.
3. **Interpretability *of an optimisation run*** — after we render `goldfish`, we don't expose *which SAE features fired hardest*, *which were already there at init*, *what the final embedding does to its k-nearest training-set embeddings*. The optimisation is a black box even with the SAE shipped.
4. **The four-modality showcase output bundle** — there is no CLI command that takes one concept and produces image + audio + video + the feature-decomposition card, all in one bundled output for a single concept. `render` does one modality. `compare` does multiple encoders, not multiple modalities. The art deliverable Callum described has nowhere to live yet.
5. **Cross-encoder / Platonic-representation experiment** — we have multiple encoders but no instrument to *measure* whether they all converge on the same concept geometry. (i.e. crosscoders.)
6. **Spatial attribution of SAE features in the output.** Even with a trained SAE, we can say "feature 1487 is highly active" but not "here is the patch of the goldfish image that is making feature 1487 fire". Without that, the SAE is interpretable in name only.
7. **Diversity / ensemble feature visualisation.** Olah-style multi-transform robust feature visualisation is well-established; we're optimising a single latent and getting one image per concept. Adversarial-noise failure mode is real.
8. **Natural-image / on-manifold prior** as a *learned* replacement for hand-tuned TV/spectral/latent-norm.

These are the areas where new (Apr–May 2026) techniques are actually applicable.

---

## 2. New research since `2026-03-19-research-findings.md`

The existing research doc is excellent through mid-March 2026. Below are findings that have appeared (or that I missed in the original sweep) and that change the design recommendation.

### 2a. Matryoshka SAE (MSAE) — `arxiv 2502.21214`, ICML 2025; reproducibility paper TMLR-2026

> Learns hierarchical SAE representations at multiple granularities simultaneously by training nested prefixes of the dictionary. 0.99 cosine reconstruction with 80% sparsity on CLIP; ~120 named semantic concepts extracted.

**Implication for us:** the current `GroupSparseSAE` in `sae/training.py` is fine but pays the standard sparsity-vs-reconstruction tradeoff. MSAE Pareto-dominates it for our use case, and the *hierarchy* is itself useful for the art deliverable: render at the coarse prefix to get the "platonic ideal", render at the fine prefix to get the nuanced variant of the concept. Independent reproduction confirms the gains are real and code (`github.com/WolodjaZ/MSAE`) is Apache-2.0.

**Recommendation:** add `MatryoshkaSAE` as an additional trainable class in `sae/training.py`, behind a flag (`--variant matryoshka`). Keep `GroupSparseSAE` for cross-modal.

### 2b. Crosscoders (Anthropic, Feb 2025; active library `oclivegriffin/crosscode`, 2025-06)

> A single SAE-like dictionary that operates on the *concatenation* of activations from multiple sources (layers, models, encoders) and learns features shared across all of them.

**Implication for us:** this is the right instrument for the Platonic Representation question. Train a crosscoder over `[ImageBind-1024 ; SigLIP2-1152 ; CLAP-512]` on the *same* set of paired text/image/audio concepts. Features the crosscoder finds in common are *cross-model* features — the closest empirical evidence we can give for or against the Platonic claim, and a direct interpretability deliverable.

**Recommendation:** add `Crosscoder` as a third sparse-decoder variant alongside SAE / MSAE. Output an HTML report of shared-vs-unique features per encoder. This is novel territory — there's no public crosscoder for multimodal encoders yet that I can find.

### 2c. Implicit Inversion turns CLIP into a Decoder (ICLR 2026, `arxiv 2505.23161`)

> Frequency-stratified INR + adversarially robust initialisation + Orthogonal Procrustes alignment between local text and image embeddings + a "blending loss" that anchors outputs to natural-image statistics. Operates with CLIP frozen and no learned decoder.

**Implication for us:** the existing `INRGenerator` is a plain Fourier-features MLP. The 2026 paper's three upgrades — frequency stratification across layers, adversarial init, and the blending loss — are *exactly* the regulariser-legibility fix Callum's retrospective asks for. They are derived from the structure of the problem rather than hand-tuned. The blending loss in particular replaces the brittle latent-norm + spectral combo with one principled term.

**Recommendation:** port the three techniques into a new `INRGeneratorV2`. Keep the existing INR for ablation comparisons.

### 2d. Generative Latent Prior (GLP) — Luo et al., ICML 2026 (`arxiv 2602.06964`)

> Train a small diffusion model on a corpus of activations from a frozen model; use it as a learned *on-manifold prior* for steering / scalar probing / optimisation.

**Implication for us:** this is the most ambitious option for the regulariser-legibility gap. Train a diffusion model on SDXL VAE latents (or on ImageBind embeddings themselves). Then during optimisation we add an SDS-style loss that says "stay on the manifold of latents the diffusion model thinks are plausible." This *learns* the equivalent of TV + spectral + latent-norm from data, so the regulariser is derived rather than empirical. Heaviest option; would also be the strongest scientific result.

**Recommendation:** scope as a stretch goal. Worth it if Callum wants this to read as a research artefact; skippable if the priority is the four-modality showcase.

### 2e. Matryoshka & MSAE-style + automated labelling (Bricken-style)

> "Feature dashboards" methodology from Anthropic/Bricken: for each SAE feature, find the top-N maximally activating *training inputs*; an LM (cheap call) writes a label from a few examples.

**Implication for us:** the existing `label_features` in `sae/training.py` labels features by cosine similarity to a fixed vocabulary. That's brittle. The Bricken approach — show the LM the top-20 maximally activating images for the feature and let it generate the label — is more robust, has been validated by independent reproductions in early 2026 (`learnmechinterp.com/topics/sae-interpretability`), and yields better feature names for the art deliverable.

**Recommendation:** swap `label_features` to use a vision-language model (a cheap MiniCPM-V or Gemini call) given the top-20 examples per feature. Adds a one-time API cost but produces much better labels for the showcase.

### 2f. AudioSAE — Aparin et al., Feb 2026

> SAEs across all encoder layers of Whisper / HuBERT. >50% feature stability across seeds. Concepts erasable by removing 19–27% of features. EEG-correlated features.

**Implication for us:** validates SAE-on-audio in general (so our CLAP SAE plan is on solid footing) and provides a stability metric we can borrow for the evaluation harness ("≥X% of features must be stable across seeds for the SAE to be considered usable").

### 2g. Circuit Tracing in VLMs / OmniTrace (CVPR 2026 / `arxiv 2604.13073`)

> Transcoders + attribution graphs for multimodal models. OmniTrace is plug-and-play across audio+image+video.

**Implication for us:** lets us produce *attribution maps* over the generated image showing which spatial regions are responsible for the most-active SAE features at the end of optimisation. This is the missing "interpret the run" deliverable from §1c.4.

**Recommendation:** add a `interpret` CLI command. Inputs: a finished `RenderResult`. Outputs: an HTML page with the image, the top-K SAE features with names + their activation curves over the optimisation run, and an attribution map per feature.

### 2h. CorrSteer (ICML 2026)

> At generation/optimisation time, correlate SAE feature activations with task outcome; intervene only on features that both correlate and demonstrably move the metric.

**Implication for us:** this is the right way to identify *which features drove the render*. Log SAE activations every step; at the end, correlate each feature's trajectory with cosine-similarity-to-target trajectory. Features with high positive correlation are "the features the model needed to activate to satisfy the concept." That list, sorted, *is* the interpretability artefact for each render.

**Recommendation:** wire this into the new `interpret` command.

### 2i. SEM — Sparse Embedding Modulation (CVPR 2026)

> Use a SAE on CLIP *text* embeddings, manipulate sparse features, project back. Demonstrated for post-hoc debiasing.

**Implication for us:** validates `Concept.from_features()` (already in the spec; relies on a trained SAE). No new code, but reinforces that the design is sound.

---

## 3. Proposed work, prioritised

I've organised proposals into three tiers. Each item is tagged `[research]`, `[art]`, or `[both]`. The default reading is "do tier 1, then ask Callum which of tier 2 to take next." Tier 3 is research stretch goals that I'd flag for a longer thread.

### Tier 1 — Make the existing v2 system actually deliver the vision (1–2 weeks)

These are the smallest changes that turn the current codebase into something that *produces the vision's deliverables*. No new research techniques, just finishing what's started.

1. **Train and ship a SAE checkpoint for ImageBind and SigLIP 2.** `[both]`
   `sae collect` + `sae train` + `sae label` on a public dataset (CC3M-subset or COCO). Commit the artefacts (`sae_models/imagebind-cc3m.safetensors`, `sae_models/siglip2-cc3m.safetensors`) + their vocabs. Without this, every SAE-using feature is dead.

2. **Add `gallery render` CLI: one concept → image + audio + video + decomposition card.** `[art]`
   New command that takes a `ConceptSpec`, runs `render` on all three output modalities (image/audio/video) with the same target embedding, plus runs `decompose` on the target, plus produces an HTML index page bundling all of it. This *is* the four-modality showcase artefact.

3. **Add evaluation harness with three metrics.** `[research]`
   - Held-out similarity: encode a paraphrase of the prompt with a *different* encoder; report cosine sim.
   - Seed-stability: same prompt, three seeds; report variance in similarity and pairwise output similarity.
   - Cross-modal alignment: render same concept in image / audio / video; report cosine sim between the re-encoded outputs.
   New module `evaluation/`. New CLI `embed-art evaluate`. Gates the regulariser-legibility work because the only honest way to "make regularisers legible" is to have numbers to ablate against.

4. **Audio side of the group-sparse SAE.** `[research]`
   Extend `collect_embeddings` to accept image+audio paired datasets (AudioCaps style). Train the group-sparse SAE with cross-modal masking actually enabled (currently the code path runs but no audio embeddings are produced). Ship a CLAP-paired SAE checkpoint.

5. **Patch the noisy bits called out in `CODE_REVIEW.md`** that are still on `main` (the duplicate line in memory.py, the dead `is_video` branch in engine.py, the magic-number CFG scales, the `Concept.to()` text_source dropout). `[both]`
   These were P1 in January; the v2 refactor mostly preserved them.

### Tier 2 — Apr–May 2026 techniques (2–3 weeks beyond tier 1)

6. **MSAE variant of the SAE trainer.** `[research]`
   Add `MatryoshkaSAE` next to `GroupSparseSAE`. `sae train --variant matryoshka`. Compare reconstruction-vs-sparsity Pareto on the same corpus. The hierarchical structure also unlocks "coarse vs fine concept" rendering in the art deliverable.

7. **`interpret` CLI command** combining CorrSteer + an attribution map. `[both]`
   Inputs: a `RenderResult` from a prior render. Outputs:
   - SAE decomposition of the final image's embedding (already possible via `decompose`).
   - **Top-K features by correlation between feature activation and similarity-to-target over the optimisation history** (CorrSteer-style — the new bit).
   - For each top feature, a heatmap on the generated image showing where in the image that feature fires (OmniTrace / Diffusion-CAM style for our optimisation setting — we can hook the same backward pass we use for cosine similarity).
   - All bundled into one HTML page.
   This is the "you can actually see what the model represents" surface from the vision.

8. **`INRGeneratorV2` with the three Implicit Inversion fixes.** `[both]`
   - Stratify Fourier frequencies across MLP layers (low freq early, high freq late).
   - Adversarially robust initialisation (a few steps of FGSM-style perturbation on the input before training).
   - Blending loss term that pulls toward natural-image statistics computed from a held-out image corpus.
   Goal: produce INR renders that compete with the VAE renders on quality, *without* the hand-tuned regulariser triplet. This directly attacks the regulariser-legibility gap.

9. **Improved feature labelling using VLM captions over top-K activating examples.** `[both]`
   Replace cosine-similarity-to-vocab labelling with a VLM call. Gives much better feature names; small one-time API cost.

10. **Cross-modal alignment metric → "side-by-side concept" gate.** `[both]`
    Extend §3.3 with a target threshold (PROJECT.md says 0.90). Gate the `gallery render` artefacts: only ship a showcase bundle if cross-modal similarity passes the gate. Forces honesty.

### Tier 3 — Research stretch goals (multiple weeks each, scope each separately)

11. **Multimodal crosscoder** over `[ImageBind ; SigLIP2 ; CLAP]`. `[research]`
    Novel territory. Outputs: a feature dictionary with per-encoder masks showing which encoders carry each feature. Direct empirical evidence on the Platonic Representation Hypothesis for these three encoders. Worthy of a write-up.

12. **Learned on-manifold prior (GLP-style) replacing TV/spectral/latent-norm.** `[research]`
    Train a small diffusion model on a corpus of SDXL latents (or ImageBind embeddings). Use score-distillation guidance during optimisation. If this works, the regulariser-legibility gap is *closed* — the prior is derived from data, the empirical knobs disappear. Biggest scientific payoff, biggest implementation cost.

13. **Diversity-objective ensemble feature visualisation** (Olah/Schubert 2017 redux). `[research]`
    Optimise a *batch* of K latents jointly, with a pairwise-diversity term in addition to the per-latent similarity objective. Yields K parallel "what does goldfish look like" outputs that exhibit the *range* of the concept rather than one specific representative. Better art, plus better evidence the concept isn't a one-off lucky basin.

14. **Interactive embedding-space exploration UI** (the "interactive layer is deferred to the lab" item from the vision). `[both]`
    Extend the existing React UI: PCA/UMAP projection of a precomputed set of concept embeddings into 2D/3D, click a point, see the optimised render. Drag between two points to slerp. Requires tier-1 done first and probably tier-2 #7.

---

## 4. Recommended starting slice

If I had to pick one slice to ship first, it would be:

> **Tier 1 items 1, 2, 3** (SAE checkpoint, `gallery render`, evaluation harness), then **tier 2 item 7** (`interpret` command).

Rationale:
- Items 1 + 2 produce the *art deliverable* you described: a single concept rendered as image + audio + video + a feature card. Without #1, #2 is just three independently optimised outputs with no interpretability layer; without #2, #1 is a `.safetensors` file with no surface.
- Item 3 unlocks any honest comparison between regulariser settings, encoder choices, optimisation tricks — the entire research-quality side of the project depends on it.
- Item 7 (`interpret`) is the smallest credible "deeper understanding of internal model states" deliverable. It turns each render into a documented interpretability artefact.

After that, the natural follow-ups are:
- Tier 2 #8 (INR v2) — directly answers the regulariser-legibility complaint with a derived alternative.
- Tier 3 #11 (crosscoder) — the most interesting novel research thread.

I'd skip GLP (tier 3 #12) unless you specifically want to push this toward a paper-grade artefact; the implementation cost is large and tier-2 #8 captures most of the practical benefit.

---

## 5. Open questions before I start coding

1. **Dataset for SAE training.** Tier 1 #1 needs a corpus. Options:
   - **CC3M subset** (Apache 2.0, ~10K images). Cheap to download; legal cleanly.
   - **COCO 2017** (Apache 2.0, ~118K images, well-known).
   - **LAION subset** (more diverse, larger license question).
   Default I'd pick: CC3M-10K. Sound good or do you have a preferred corpus on disk already?

2. **VLM for feature labelling (tier 2 #9).** Options: a local MiniCPM-V (no API cost, slower), Gemini 2 (API key needed), Claude 3.7 vision via Anthropic key. Any preference?

3. **Scope confirmation for the first PR.**
   Am I right that the first PR should be **tier 1 items 1–3 + tier 2 item 7**, and that tier 2 items 6/8/9/10 and tier 3 land in separate PRs after we've seen the first one?

4. **Compute target.**
   - The shipped code targets M1 Max 64GB. SAE training (item 1) on CC3M-10K is fine there.
   - Crosscoder training (tier 3 #11) is heavier but still tractable.
   - GLP (tier 3 #12) wants a real GPU. If you want that one, where do you want it to run?

5. **Treatment of `bd` (beads) issue tracking.**
   The repo uses beads. Want me to file beads issues for each agreed-on proposal so progress is tracked there in addition to the PR?

I'll wait for your answers (or a "yes to all defaults, proceed") before changing any code.
