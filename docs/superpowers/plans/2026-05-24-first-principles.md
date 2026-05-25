# First-Principles Refinement & Tradeoffs

**Date:** 2026-05-24
**Author:** Devin (for Callum)
**Status:** RESOLVED — committed direction. Implementation underway.
**Supersedes (for design):** parts of `docs/superpowers/specs/2026-03-19-embedding-art-v2-design.md`
**Companion (still relevant):** `docs/superpowers/plans/2026-05-24-gap-analysis.md`

## Resolved decisions (2026-05-24, final)

| Decision | Resolved as | Notes |
|---|---|---|
| **Canonical encoder** | **LanguageBind (ICLR 2024)** | All four modalities (image+audio+video+text) in one shared space anchored on language. OmniBind ruled out — no video. WAVE ruled out for first PR — backprop through Qwen2.5-Omni is 10–50× slower per step on M1 Max; revisit as upgrade later. |
| Dual rendering (honest + natural) | **Yes, both 2026-crispy.** | INR-as-generator dropped — 2018 DeepDream aesthetics not acceptable. Differentiation via VSD-weight balance in the same SD3.5 latent pipeline. |
| Aggressive rewrite vs incremental | **Aggressive rewrite.** | No backwards compatibility constraint. One feature branch, one PR series. |
| Compute target | **M1 Max only.** | LanguageBind huge variants + SD3.5-medium + Stable Audio Open + LTX-Video + MSAE stack ≈ 30GB load, headroom comfortable. |
| Image backbone | **Stable Diffusion 3.5 Medium** | SDXL VAE dropped. Flow-matching latent, sharper, permissive Stability Community License. Flux.1-schnell optional/stretch. |
| Diffusion prior loss | **Variational Score Distillation (VSD)** | Trains a small LoRA absorbing per-concept distribution to avoid SDS mode collapse. CSD as faster alternative. |
| Audio backbone | **Stable Audio Open 1.0** | MIT, 44.1kHz, 47-sec clips. |
| Video backbone | **LTX-Video 0.9.5** | Open RAIL-M, ~2B, ~5-sec clips at 24fps, M1 Max friendly. |
| Patch-level feature alignment loss | **LanguageBind ViT patch tokens** (no DINOv3). | LanguageBind's ViT-Huge exposes patches natively; DINOv3 would be redundant. Multi-layer MIMIC-style alignment via existing `ViTStatisticsExtractor` infrastructure. |
| Cross-modal-target patch loss fallback | **Pooled cosine + multi-layer feature alignment when target and current modality differ.** | Patch-level alignment is enabled only when modalities match. |
| SAE variant | **Matryoshka SAE (MSAE) stack** | Per-modality MSAE (image/audio/video/text on LanguageBind embeddings) + one shared MSAE on mean-pooled multimodal representation. Delta between per-modality and shared SAEs = modality-specific vs modality-agnostic features. Replaces GroupSparseSAE. |
| Crosscoder | **Demoted to tier-3 research instrument.** | Over [LanguageBind, SigLIP2, CLAP] as Platonic-representation experiment. Not load-bearing for showcase (LanguageBind already provides shared space). |
| Hand-tuned regularizers | **Dropped from default path.** | TV/spectral/latent-norm replaced by VSD prior + frequency stratification + blending loss against natural-image statistics. |
| Feature labelling | **VLM caption over top-K activating examples** (Anthropic API). | Replaces cosine-to-vocab matching. |
| **Text-anchor trick (NEW)** | **Yes — LanguageBind-specific addition.** | (a) During any-modality rendering, project current embedding to text-space; log human-readable interpretability signal at every step. (b) Optional auxiliary text-anchor loss against the target's text-projection. (c) Mid-render "current ≈ {'storm','crackling','electric'}" readouts. |
| **Linear-probe verification (NEW)** | **Yes — added to evaluation bundle.** | Cheap linear probes on LanguageBind canonical space ("is this an animal", "danger-coded", etc.); probe activations become part of the interpretation card. |
| **Anchor-comparison experiment (NEW)** | **Yes — added as M9 research deliverable.** | Encode same concept via text / image / audio / video; compare shared-MSAE decompositions; the delta IS the modality-specific carve-out. Clean Platonic-representation experiment unique to LanguageBind-style anchored architectures. |
| Attribution method | **Classical ViT attribution (attention rollout + integrated gradients).** | OmniTrace-style VLM circuit tracing dropped — designed for decoder VLMs, doesn't apply to CLIP-style contrastive encoders. Classical methods are well-suited to LanguageBind's ViT structure. |
| Web UI | **Reduced to job submission + bundle viewer.** | Architectural rewrite invalidates current frontend; rebuild after artefact pipeline is solid. |
| LatentGenerator / DirectGenerator split | **Collapsed to single `Generator.render()` protocol.** | Backwards compat split is no longer needed. |
| `optimize` CLI command | **Dropped.** | `render` is the only entrypoint. |
| ImageBind 1.0 encoder | **Dropped entirely.** | LanguageBind is its direct successor — covers same modalities, MIT licensed, better quality. |
| DINOv3 | **Dropped from default; kept as optional eval probe.** | LanguageBind ViT provides patches; DINOv3 is redundant in default path. |

## 0. TL;DR

If we strip the constraint of "v2 spec was already approved" and ask "what should this system be in May 2026, given leading techniques," six architectural choices change from the current/planned design:

| Choice | Current/planned | First-principles answer | Δ to ultimate output |
|---|---|---|---|
| **Canonical concept representation** | Dense embedding vector | Sparse feature decomposition (SAE / crosscoder) is primary; embedding is derived | Showcase is the *decomposition* with renderings as views; not 4 renderings stitched together |
| **Default generator** | SDXL VAE latent | INR + score-distillation diffusion prior | Outputs become more honest "what the model thinks" — possibly *less* photo-real, definitely more interpretable |
| **Primary loss** | Cosine on pooled embedding + mean/std layer-stats | Patch-token alignment + SAE-feature alignment + SDS prior | Avoids texture-matching failure mode; needs paired-modality SAE to be trained first |
| **Regularisers** | Hand-tuned TV + spectral + latent-norm | Learned / derived (SDS, blending loss, frequency stratification) | Removes the regulariser-legibility complaint; introduces SDS noise risk |
| **Interpretation** | Optional post-step (`decompose`) | Default bundle every render produces: decomposition + per-step trajectory + attribution maps + cross-encoder verification | Every artefact ships with a self-contained interpretability dossier |
| **Cross-encoder geometry** | Per-encoder SAEs, optional `compare` | Multimodal crosscoder is the central object | Direct empirical answer to Platonic-representation question; novel research; heavy compute |

There are six structural tradeoffs in those choices that I want you to weigh in on before I commit. The four most consequential are at the top of §3.

---

## 1. First-principles critique by layer

I'll walk each layer of the system from first principles, then say what changes.

### 1.1 The encoder

**Premise:** the encoder defines "the model's concept of X." Choice of encoder *is* a choice of what the project means by "the model."

The v2 plan adopts a registry of dual-encoders (SigLIP 2, CLAP, ImageBind) plus optional LLM-based encoders for quality checks. That's right. But two things go further:

- **Multiple encoders should not be alternative perspectives on a concept; they should jointly define it.** If "goldfish" means something different to SigLIP than to ImageBind, the right primitive is *the intersection*, not the disjunction. That's what a crosscoder is.
- **The 1024-d pooled embedding is a lossy summary.** Patch-level / token-level features carry the information cosine-similarity throws away. Multi-layer feature matching (already in v2) is a step in this direction, but the v2 implementation matches *mean/std* statistics, which is Gatys-style texture matching — fine for style transfer, lossy for semantics. Earth-Mover distance over patch tokens, or direct token-set similarity, would be tighter.

**Change:** crosscoder over multiple encoders is the canonical concept space. Each encoder is a *view* on the canonical space, not an island.

### 1.2 The concept representation

**Premise:** how concepts are stored determines what operations on them mean.

Vector arithmetic on the dense embedding is the project's signature framing. It's a great pedagogical artefact ("look, fire + water = useful") and it's mathematically clean. But once you have an SAE, vector arithmetic in dense space is *strictly worse* than feature arithmetic in sparse space:

- Adding two normalized vectors averages every dimension; adding two sparse-feature decompositions unions the *named* features.
- Subtraction in dense space is a regularizing nudge; in sparse space it's "remove this specific named feature."
- Interpolation in dense space is geometric (slerp); in sparse space it's "morph the feature mixture."

**Change:** `Concept` carries `decomposition: SAEDecomposition` as its canonical state. `Concept.embedding` is a *property* computed from the decomposition. All arithmetic operations are defined in feature space. This is what SEM (CVPR 2026) does for debiasing, and it's the right primitive.

> **Caveat:** until SAEs are trained for every encoder we use, dense embedding remains the actual storage. That means the order of operations matters: train the SAEs first, then refactor `Concept`. Otherwise we paint ourselves into a corner.

### 1.3 The generator

**Premise:** the generator should render "what the model thinks the concept looks like."

The VAE latent approach has been the de facto choice because (a) it's fast, (b) the VAE prior keeps outputs natural-looking. But this is *aesthetic bias dressed up as engineering*. The VAE doesn't render the model's representation; it renders the *closest natural image* whose embedding matches the model's representation. The vision document is explicit that we want the *former*.

INR / pixel-space optimization is honest, but adversarial-noise failure modes are real, which is why the field abandoned it. The Implicit Inversion paper (ICLR 2026) revives it with three derived techniques:
1. Stratify Fourier feature frequencies across MLP layers (coarse-to-fine generation as a structural property)
2. Adversarially robust initialization (the latent starts somewhere the encoder doesn't catastrophically misread)
3. A "blending loss" anchoring outputs to natural-image statistics computed from a corpus (a learned alternative to TV + spectral + latent-norm)

This *replaces* the hand-tuned regularizer triplet with derived terms. It's exactly the response to your retrospective.

Going further, Score Distillation Sampling (SDS) from a frozen diffusion model is the leading approach for "natural-image prior as a loss term" — used in DreamFusion, ProlificDreamer, all the text-to-3D work. The Generative Latent Prior paper (ICML 2026) extends this idea to activation-space priors. For our use case, plain SDS from a frozen SDXL would already be a strong learned prior, no additional training needed.

**Change:** the canonical generator is an INR + SDS prior. The VAE path stays as a fast/aesthetic alternative, with a clear honest label.

### 1.4 The optimization loop & loss

**Premise:** the loss specifies what counts as "matching the concept."

The v2 plan's composite loss is:
- Cosine similarity on pooled embedding (always on)
- Multi-layer feature matching via mean/std statistics
- SAE feature MSE
- Hand-tuned regularizers

First-principles critique:
- **Pooled cosine** is "match the final answer." Equivalent to matching logits in classification — the classic activation-maximization failure mode where you get adversarial noise that has the right top-1 class. Multi-layer feature matching mitigates this but doesn't fix it.
- **Mean/std stats matching is texture-matching.** Gatys-style. Captures texture distributions per channel; loses *which* patch is *which* part of the object. For "goldfish," it tells the optimizer to match goldfish-textures *somewhere* in the image, not to make a goldfish-shaped goldfish.
- **SAE MSE in dense feature space** matches *what features are active*, not *to what degree*. Threshold-mismatch (target wants feature 1487 at 0.8, current has it at 0.4) gets the same gradient as direction-mismatch.
- **Hand-tuned regularizers** are the explicit gap you flagged.

The right loss, from first principles, is patch-token alignment in the embedding space (preserves spatial structure) + SAE feature-direction alignment (preserves what concepts are active) + SDS prior (replaces hand-tuned regularizers with a learned natural-image manifold).

**Change:** loss = patch-token alignment + SAE feature-direction loss + SDS prior. Cosine on pooled embedding is a *metric* (logged at every step) but is *not* the optimization signal. Hand-tuned regularizers go away.

### 1.5 Evaluation

**Premise:** without numbers we have no honest comparison.

Your retrospective and my prior gap analysis both call this out. First principles: evaluation should test the *thesis* of the project, which is "this rendering is what the model represents for this concept." Tests of the thesis:

1. **Cross-encoder agreement.** Encode the rendering with a *different* encoder than the one we optimized against. If the rendering really represents the concept, the alternative encoder should also place it close to its own embedding for that concept.
2. **Cross-modal agreement.** The image, audio, and video for the same target should re-encode to vectors near each other in their shared space.
3. **Seed stability.** Three runs from different seeds should produce outputs whose pairwise similarity (in some external metric) exceeds a threshold.
4. **Regulariser ablation grid.** Held-fixed everything else, vary one regularizer weight; report the curve. Makes regularizers legible at last.
5. **Human evaluation.** Pairwise A/B vs SDXL-from-text. Are these distinguishable? Does the audience prefer one?

These are not "nice to have." They're the difference between an essay-grade artefact and a research-grade one. You said you wanted both.

**Change:** evaluation is a first-class subsystem. Every artefact ships with its evaluation card.

### 1.6 Interpretation

**Premise:** "deeper understanding of internal model states" is goal 1. Therefore interpretation must be the default surface, not an optional analysis step.

Today, after a render, you have an image. You don't have:
- Which features fired hardest in the final output, named.
- Which features the optimizer activated *first* vs *last* (the trajectory says what the model considered "easy" vs "hard" about the concept).
- Where in the output each top feature spatially lives (attribution maps).
- Whether another encoder agrees about what's in the image.
- The delta between target decomposition and realised decomposition.

That last one is the most powerful: when "goldfish" is rendered, *which features did the model fail to express*? That's the most honest interpretability artefact in the project — it tells you the model's blind spots about the concept.

**Change:** every render emits an interpretability bundle by default. Image is one view of the bundle.

### 1.7 The art deliverable

**Premise:** "showcase pieces with a single concept represented side by side in image, audio, video, and text."

The naive read: render the concept in each modality, lay them out side by side. That's what `gallery render` in my prior plan does.

The deeper read: the concept *itself* is a feature decomposition. The four modalities are renderings *of the same decomposition*. The artefact exhibited is the decomposition; the four images/sounds/videos are different visualizations of the same underlying thing.

This is also the only honest answer to "cross-modal agreement." Independent optimization in four modalities won't agree by default; if you want them to agree, they need a shared substrate. The crosscoder feature decomposition is that substrate.

**Change:** the showcase deliverable for a concept is (decomposition + four renderings + cross-modal agreement matrix), with the decomposition as the central exhibited object.

---

## 2. The refined target system

Putting it all together, the ideal system in one diagram:

```
          ConceptSpec ("thunder", "goldfish", thunder.wav, ...)
                 │
                 ▼
       ┌────────────────────┐
       │   Crosscoder /     │   <─ shared dictionary trained over
       │   Multimodal SAE   │      [SigLIP2, CLAP, ImageBind]
       └────────────────────┘
                 │
       (sparse feature decomposition: ~16K dict, top-K active)
                 │
                 ▼
       ┌────────────────────┐
       │  Concept algebra   │   <─ add / subtract / slerp in feature space
       │  (in feature space)│
       └────────────────────┘
                 │
        ┌────────┼─────────┬─────────┐
        ▼        ▼         ▼         ▼
     image    audio      video    text-card
   (INR+SDS) (AudioLDM) (SVD)   (feature manifest)
        │        │         │         │
        └────────┴─────────┴─────────┘
                 │
                 ▼
       ┌────────────────────┐
       │  Interpretation    │
       │  bundle per render │   <─ traj + attribution + cross-encoder check
       └────────────────────┘
                 │
                 ▼
       ┌────────────────────┐
       │  Evaluation gate   │   <─ cross-encoder, cross-modal, stability
       └────────────────────┘
                 │
                 ▼
            Showcase artefact (decomposition + 4 renderings)
```

The crosscoder is the linchpin. Without it: per-encoder SAEs work but the "side-by-side same concept" deliverable can't be guaranteed (modalities will drift). With it: the four renderings literally share a substrate.

---

## 3. Tradeoffs you should weigh in on (most consequential first)

These are decisions where the leading-techniques answer materially changes what the ultimate output looks like, in ways that involve genuine value judgments — not just engineering tradeoffs.

### Tradeoff A — "Honest aliens" vs "Beautiful naturalism"
**The choice:** INR + SDS as default generator, OR keep VAE latent as default.

- **INR + SDS** renders what the model represents. Outputs may look strange, even adversarial-flavoured (think DeepDream-era aesthetics — fractal, recursive, sometimes psychedelic). For research, this is the honest answer: the model's representation of "goldfish" is what comes out, not a *photo* of a goldfish. For art, this aesthetic has a specific, divisive character — it can be striking and "of the model," but it doesn't read as "a goldfish" to a casual viewer.
- **VAE latent (current default)** renders the closest natural-looking image whose embedding matches. Outputs are photo-realistic. The VAE's training distribution is doing most of the aesthetic work — you're rendering "the natural image the model would identify as goldfish," not "the model's representation of goldfish." But the outputs look like things people recognize, which makes the side-by-side showcase legible.

**Impact on the ultimate output:**
- Pick INR+SDS as default → showcase looks like Inceptionism (Mordvintsev-style). The art piece is *about* what the model thinks. Decisively different from prompt-to-image generators.
- Pick VAE latent as default → showcase looks like a series of nice SDXL pictures. Closer to existing prompt-to-image work, less obviously about the model's internals.

**My recommendation, if I had to pick one:** ship BOTH, but make INR+SDS the *headline* default for the research/showcase, and VAE the *fast/legible* alternative used when speed matters or when an audience needs a recognizable image. The compare-side-by-side itself becomes interesting: "VAE thinks goldfish looks like *this*; INR thinks the model's idea of goldfish is *this*." That dual rendering is more compelling than either alone.

### Tradeoff B — Crosscoder centrality vs per-encoder SAEs
**The choice:** Train one multimodal crosscoder over [SigLIP2, CLAP, ImageBind] as the canonical concept space, OR keep per-encoder SAEs with `compare` for side-by-side.

- **Crosscoder** is novel (no public multimodal crosscoders yet) and the right primitive for "shared concept across encoders." Risk: features might not align well, you might end up with a low-quality dictionary if the encoders' spaces are too different. Compute: medium-heavy training run.
- **Per-encoder SAEs** are established (per the ICLR 2026 group-sparse paper). Lower risk, easier to debug. Loss: cross-modal "side-by-side same concept" pieces have no guaranteed shared substrate; they will drift.

**Impact on the ultimate output:**
- Pick crosscoder → the showcase artefact for "thunder" is a single named decomposition like `{storm: 0.8, low-frequency: 0.4, fear: 0.2}` and four renderings of *that exact decomposition*. The decomposition is the artefact; the renderings are views.
- Pick per-encoder SAEs → the showcase is "here's goldfish in SigLIP, here's goldfish in CLAP, here's goldfish in ImageBind, here's how their decompositions compare." Less unified artefact but more comparison-focused — better suited to a research write-up about how different encoders see the same concept.

**My recommendation, if I had to pick one:** ship per-encoder SAEs in the first PR (lower risk, faster artefacts) and treat the crosscoder as the second project — promised but not committed. The crosscoder is also more interesting *after* you've seen what the per-encoder SAEs look like, because you can interpret crosscoder features in terms of per-encoder features.

### Tradeoff C — Patch-level loss vs pooled-cosine loss
**The choice:** Replace pooled cosine with patch-token-level alignment loss as the optimization signal.

- **Patch-level** preserves spatial structure. Goldfish-shaped rendering instead of goldfish-textured noise. Much richer gradient signal. The leading dense-feature techniques (DINOv2/v3, Patchwise-Retrieval WACV'26, ICLR 2026 CLS/Patch interaction work) all converge on this.
- **Pooled cosine** matches the encoder's actual training objective. Simpler. Known well. But it's the source of the texture-failure-mode in current activation-maximization work.

**Impact on the ultimate output:**
- Pick patch-level → outputs have correct *shape* / *arrangement*, not just correct *texture distribution*. This is a quality jump for the image deliverable. But: requires encoder hooks we don't have wired in fully (only SigLIP2's are hooked); CLAP and ImageBind would need patch-level hooks added. And: patch-level loss optimizing toward features that don't survive pooling is wasted gradient — pooled cosine is what the encoder actually thinks. The risk is the patch-level loss leads you toward something that looks right at the patch level but the encoder doesn't "see."
- Pick pooled cosine → simpler, more in tune with encoder, but lower-quality outputs.

**My recommendation, if I had to pick one:** patch-level loss as primary, pooled cosine as a *logged metric*, with a hyperparameter to interpolate between them so we can ablate. Low risk because we can always fall back. The leading-techniques call here is decisive.

### Tradeoff D — Time-to-first-artefact vs ultimate cleanliness
**The choice:** Aggressive rewrite to the refined architecture, OR incremental progression through tier 1 → tier 2 → tier 3 of my prior plan.

- **Aggressive rewrite:** ~4–6 weeks before any complete artefact. Cleaner final code. Adopts leading techniques at the architectural level. Risk: a long stretch with no visible artefacts.
- **Incremental:** ~1–2 weeks to first showcase. Carries existing technical debt forward. Some refactoring inevitable later. Lower risk per step.

**Impact on the ultimate output:**
- Same eventual state. Only the path differs.
- Aggressive rewrite produces a "complete" first artefact (decomposition + INR-with-SDS + interpretation bundle + four modalities + evaluation) on first ship.
- Incremental produces a partial showcase early, with the deeper-research surface arriving in follow-on PRs.

**My recommendation, if I had to pick one:** hybrid. Aggressive rewrite of the *architecture* (so we don't paint ourselves into corners), incremental shipping of the *artefacts*. Concretely: in the first PR, do the architectural refactor (Concept-decomposition-first, INR+SDS as default generator, patch-level loss, interpretation-by-default) AND ship the first artefact (one trained MSAE, one fully-rendered showcase concept with all four modalities). Subsequent PRs ship more concepts, more encoders, the crosscoder, the evaluation harness, etc.

---

## 4. Smaller tradeoffs (defaults I'd commit to without asking)

These are calls where the leading-techniques answer is clear enough that I'd just do them, but I'm flagging them for visibility:

| Decision | Default | Why |
|---|---|---|
| Replace `GroupSparseSAE` with Matryoshka SAE (MSAE) | Yes | Pareto dominates on sparsity vs reconstruction; provides hierarchy that doubles as art-vs-research toggle |
| Replace cosine-similarity-to-vocab feature labelling with VLM caption over top-K activating examples | Yes | Vocab matching gives shallow labels; VLM gives semantic ones. One-time inference cost. |
| Drop the LatentGenerator/DirectGenerator split; replace with a single `Generator.render() -> Tensor` protocol | Yes | The split exists for backwards compat with the existing engine. Without backwards compat, one protocol is enough. |
| Drop `optimize` CLI command in favor of `render` | Yes | `optimize` is the legacy name; `render` is the new one. Without backwards compat, drop the alias. |
| Drop the React Web UI for now (or reduce to job submission only) | Probably | The existing UI ties to the old API; rebuilding it is scope-expensive. The interpretation bundle can be HTML files served statically until we know what the right interactive surface is. |
| Use a public CC0/CC-BY paired dataset (CC3M + AudioCaps + WebVid subsets, ~10K each) for SAE training | Yes | Free, license-clean, fits on disk, will produce defensible artefacts. |
| Use Anthropic Claude or Gemini for VLM-based feature labelling | Yes (pick one — Anthropic preferred for consistency with your dev tooling) | Either works. Cost is negligible (~$5 for all features across all encoders). |
| Add the regulariser-ablation grid as part of evaluation | Yes | This is the *direct* response to your retrospective; without it the regulariser legibility complaint stays unanswered. |
| Drop hand-tuned TV/spectral/latent-norm regularizers in the INR+SDS path | Yes | They're replaced by SDS prior + blending loss + frequency stratification. They stay on the VAE path. |
| Replace mean/std statistics extractor with patch-token-set alignment (Earth Mover or InfoNCE) | Yes | See tradeoff C. |
| Make beads (`bd`) tickets for each shipped artefact | Yes | Consistent with the AGENTS.md workflow. |

## 5. Risks I want to surface

Independent of which tradeoffs you choose:

1. **SDS gradient signal is mode-seeking.** Outputs from SDS-guided optimization tend to collapse to "the most typical thing in the diffusion model's training distribution." If we use SDS as the regularizer, we risk every concept rendering as a vaguely-similar SDXL-generic image. Mitigation: balance SDS weight against feature-alignment weight aggressively, and use variational SDS variants (ProlificDreamer-style) which are less mode-collapsing. This is the biggest non-architectural risk.

2. **Crosscoder features may not align well across our encoders.** SigLIP2 and CLAP train on very different objectives; their patch-token spaces may not have enough shared structure for a crosscoder to find unified features. If this fails, we fall back to per-encoder SAEs and lose the unified artefact. Mitigation: train per-encoder SAEs first; if their features are highly correlated, crosscoder is likely to work; if not, we know to be cautious.

3. **MPS / M1 Max limits.** Patch-level loss + INR + SDS all require keeping more activations and gradients in memory than VAE-cosine. Could break the M1 Max 64GB budget on multi-encoder setups. Mitigation: gradient checkpointing, lower-resolution INR (512x512 is fine for showcase), and the option of cloud GPU for crosscoder training only.

4. **The interpretability deliverable depends on labelled features being meaningful.** If our SAE features end up polysemantic (firing on unrelated concepts), the named labels are misleading. Mitigation: Bricken-style automated interpretation tests (does ablating feature X change behaviour for inputs that activate it?), reported as a quality gate.

5. **The art piece may be less legible.** INR+SDS outputs read as "AI art from 2018." For some audiences this is exactly the point ("this is what the model thinks"); for others it looks dated. Mitigation: ship both INR+SDS and VAE renders side by side; let the contrast itself be the artefact.

6. **Evaluation could trap us in metric-chasing.** The point of the project is interpretive depth, not benchmark-topping. If we let cross-encoder cosine drive every design call, we lose the "alien outputs are good" stance. Mitigation: evaluation reports numbers but does not gate creative direction.

7. **Per-concept compute time.** Each render in the refined system runs (a) optimization with three loss terms, (b) cross-encoder verification, (c) attribution map extraction, (d) optional second-pass with different generator. Estimate: 30–60 min per concept on M1 Max. For a showcase of, say, 10 concepts × 4 modalities × 2 generators, that's many hours. Not a blocker but worth pricing in.

---

## 6. The questions I need answers to

Reduced from my prior 5 to 4 — the ones that actually affect direction:

1. **Tradeoff A (honest aliens vs natural):** Default generator INR+SDS, with VAE as alternative? Or VAE default with INR+SDS as alternative? My recommendation: ship both, INR+SDS as headline, VAE as fast/legible alternative.

2. **Tradeoff B (crosscoder centrality):** Per-encoder SAEs in first PR, crosscoder in second? Or commit to crosscoder up front?

3. **Tradeoff D (time-to-artefact vs cleanliness):** Aggressive architectural rewrite in first PR with one fully-rendered showcase concept? Or incremental tier-1 → tier-2 → tier-3 progression?

4. **Compute target.** Will this work be all M1 Max, or do you want me to use a cloud GPU for the SAE / crosscoder training runs? If cloud: do you have one already, or should I set one up?

Once you give me the four answers, I'll commit to a direction and start a single feature branch.
