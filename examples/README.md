# Examples

Runnable recipes for the main `embed-art` workflows. Each script is small,
commented, and safe to copy-paste. They assume you've completed the install in
the top-level [README](../README.md) — including cloning **LanguageBind** and
putting it on your `PYTHONPATH`.

> **Heads up on runtime.** The headline `showcase` flow loads multi-GB
> generators (SD3.5, Stable Audio Open, LTX-Video) and gradient-descends a
> latent for thousands of steps. A full four-modality run takes tens of minutes
> on an M1 Max. Every script here starts with a fast, low-step *smoke* variant
> so you can confirm your install works before committing to a long run.

| Script | What it shows | Cost |
|---|---|---|
| [`01_showcase.sh`](01_showcase.sh) | The headline command — one concept rendered across all four modalities | minutes → tens of minutes |
| [`02_concept_algebra.py`](02_concept_algebra.py) | The Python API: add / subtract / interpolate concepts in embedding space | seconds (encode only) |
| [`03_anchor_compare.sh`](03_anchor_compare.sh) | Encode one concept through several modalities and compare the embeddings | seconds |
| [`04_interpolate.sh`](04_interpolate.sh) | Render a morph series between two concepts | minutes |

Run any script from the repo root, e.g.:

```bash
bash examples/01_showcase.sh
python examples/02_concept_algebra.py
```

Add `--device cpu` to any command if you're not on Apple Silicon / CUDA (much
slower, but works for smoke tests).
