# M4 text-anchor empirical sweep (issue iik)

Sweeps the `(similarity_weight, text_anchor_weight)` grid for the M4
auxiliary loss described in `core/loss.py`. The sweep uses a synthetic
decoder-free harness in `experiments/text_anchor_sweep.py` so it runs
in seconds on CPU and characterises the *pure* loss surface — no SD3.5,
SAO, LTX-Video, or LanguageBind weights are required.

## What the harness measures

A free embedding `current ∈ R^D` is optimised with Adam against::

    loss = -sim_w  · cos(current, target)
         - anc_w · cos(current, anchor)

with `target` and `anchor` sampled at a controllable angular
separation. After convergence we record both `cos(current, target)`
and `cos(current, anchor)`. Per-modality angular separations were
chosen to reflect the empirical separation between concept embeddings
and their text projections under LanguageBind:

| modality | angular sep (synthetic) | rationale                                                  |
| -------- | ----------------------- | ---------------------------------------------------------- |
| text     | 15°                     | anchor is (nearly) the target; sanity baseline             |
| image    | 45°                     | typical image↔text-projection separation in LanguageBind   |
| audio    | 60°                     | audio is the loosest-aligned modality in LanguageBind      |
| video    | 50°                     | video sits between image and audio                         |

These are *priors*. The real numbers should be measured on M1 Max with
LanguageBind weights via:

```
embed-art text-anchor-sweep \
  --encoder languagebind \
  --target-text "$CONCEPT" \
  --anchor-text "$CONCEPT" \
  -o docs/superpowers/results/text-anchor-sweep/real \
  --modality image
```

(The same concept name is used for both — the relevant angular separation
on real weights is between the image-projection of the concept and the
text-projection. To probe other modalities, supply paired concept refs.)

## Recommended defaults

Across all four synthetic separations, the Pareto frontier is monotonic
and the max-min elbow (the point that maximises the smaller of the two
normalised similarities) lands at `anchor_w = 0.5`:

| modality | recommended `text_anchor_weight` | target sim @ elbow | anchor sim @ elbow |
| -------- | -------------------------------: | -----------------: | -----------------: |
| image    | 0.500                            | 0.9675             | 0.8629             |
| audio    | 0.500                            | 0.9449             | 0.7559             |
| video    | 0.500                            | 0.9605             | 0.8306             |
| text     | 0.500                            | 0.9962             | 0.9848             |

Interpretation:

* The loss surface is perfectly symmetric in `(sim_w, anc_w)` — the
  Pareto frontier mirrors across `anc_w = sim_w`. With `sim_w = 1.0`
  fixed, `anc_w = 1.0` is exactly the symmetric midpoint, and the
  max-min elbow is the highest-anc-weight point that keeps
  `target_sim ≥ anchor_sim`. That comes out at `anc_w = 0.5` in our
  grid for every modality.
* In practice we recommend **`text_anchor_weight = 0.25`** as the
  showcase default, not 0.5. The reasoning: the synthetic harness has
  no regularisation, no patch-alignment loss, and no decoder coupling,
  so it sees the *cleanest* trade-off. In the real loss the anchor
  signal competes against feature-matching and regularisation, so a
  more conservative anchor weight preserves target fidelity. 0.25 is
  the highest grid point where the synthetic harness keeps
  `target_sim > 0.98` for the image/video angular regime.
* For **text targets** specifically, the auxiliary loss is mostly
  vestigial (target == anchor by construction); we keep
  `text_anchor_weight = 0.0` for text-only showcase runs.

## Per-modality detail

* [image_sweep.md](image_sweep.md)
* [audio_sweep.md](audio_sweep.md)
* [video_sweep.md](video_sweep.md)
* [text_sweep.md](text_sweep.md)

## Surfacing through the CLI

* `embed-art text-anchor-sweep -o <dir> [--modality image|audio|video|text]`
  runs the sweep.
* `embed-art showcase ... --text-anchor-weight 0.25` enables the
  auxiliary loss on the honest + natural tracks. The default value is
  0.0 (preserve v3 baseline behaviour); set explicitly if you want the
  anchor signal.

## Follow-up — real-weight calibration

The synthetic harness validates the loss math and gives well-defined
recommended defaults. To replace these with empirical numbers from a
true LanguageBind run, on M1 Max:

```
# Repeat per modality with paired references.
embed-art text-anchor-sweep \
  --encoder languagebind --device mps \
  --target-text "thunder" --anchor-text "thunder" \
  -o docs/superpowers/results/text-anchor-sweep/real --modality image
```

This will overwrite the synthetic numbers with real ones. The
recommended-default selection logic in the harness is encoder-agnostic.
