# Apple Silicon Performance Roadmap (M1/M2 Max)

Status: **shipped** (commits f0f9cd1, 574e0de, and this one).

This document describes the perf knobs available for the
aggressive-rewrite showcase on Apple Silicon, the expected impact of
each, and the harness used to measure them.

## TL;DR

```bash
# Recommended M1 Max showcase command.
embed-art showcase \
  -t "thunder" \
  -o outputs/thunder/ \
  --autocast-dtype bf16 \
  --compile-mode reduce-overhead
```

* `--autocast-dtype bf16` — runs the forward + loss section in bf16
  autocast, ~2x throughput on MPS, with fp32 master weights and
  gradients (PyTorch handles the dtype boundary).
* `--compile-mode reduce-overhead` — wraps the loss callable in
  `torch.compile` (PyTorch 2.5+ MPS Inductor). 1.5-2.5x on the hot
  loop. Silent fallback to identity on platforms where compile fails.

These two flags alone should deliver ~3-5x wall-time improvement on
the M1 Max optimisation hot loop.

## Knobs by layer

### Optimisation hot loop

| Knob | Flag / Config | Default | Expected impact |
|---|---|---|---|
| Mixed precision | `--autocast-dtype bf16` | fp32 | ~2x throughput on MPS |
| Loss compilation | `--compile-mode reduce-overhead` | none | 1.5-2.5x on the hot loop |
| MPS cache flush | `empty_mps_cache_between_modalities=True` | True | recovers 30%+ memory between modalities |

### LanguageBind encoder

| Knob | Trigger | Expected impact |
|---|---|---|
| SDPA flash attention | auto on MPS/CUDA | ~2x attention throughput |
| Lazy modality loading | auto | text-only runs skip image/audio/video weights (~3GB saved) |
| Text-anchor disk cache | auto | vocab embeddings persist across runs |

### Diffusers pipelines (SD3.5 / Stable Audio Open / LTX-Video)

Applied via `embedding_art.perf.apply_diffusers_perf_knobs`. Each
knob is best-effort and silently skipped when the pipeline doesn't
expose it.

| Knob | Method | Expected impact |
|---|---|---|
| VAE tiling | `vae.enable_tiling()` | load-bearing — keeps 1024x1024 / 47s-audio / 720p-video VAE decodes inside the M1 Max unified-memory budget |
| QKV fusion | `pipe.fuse_qkv_projections()` | ~10-15% on MMDiT-heavy pipelines (SD3.5 full path) |
| Attention slicing | `pipe.enable_attention_slicing("max")` | reduces peak VRAM without changing math |

### Probe encoders (inference only)

* CoreML / ANE for SigLIP2 + CLAP cross-encoder probes is *not yet
  implemented*; the infrastructure is sketched in
  `embedding_art.perf.PerfProfile` and the planned API lives behind
  an opt-in `--ane` flag, but conversion is gated on running on the
  target M1/M2 Max machine.
* See: tasks tracked in the issue list.

## Measurement: `embed-art profile`

The new `embed-art profile` CLI runs a representative forward + loss
+ backward step under `torch.profiler` and emits a Chrome trace +
per-op summary table.

```bash
# Synthetic encoder (default) — completes in <1s on CPU/MPS.
embed-art profile \
  --output outputs/profile/synthetic \
  --device mps \
  --steps 10 \
  --autocast-dtype bf16 \
  --compile-mode reduce-overhead

# Real LanguageBind image encoder (downloads weights on first run).
embed-art profile \
  --output outputs/profile/lb-image \
  --device mps \
  --encoder languagebind
```

Outputs:

* `<output>/trace.json` — Chrome trace, drop into `chrome://tracing/`
  or perfetto.dev to visualise timelines.
* `<output>/summary.txt` — per-op self-CPU and self-MPS time,
  truncated to the top-K hotspots.

## Internal API

All knobs live in `embedding_art.perf` and are dependency-free
(`import embedding_art.perf` does not import any heavy ML library).
The public surface:

```python
from embedding_art.perf import (
    mps_available,                 # bool — guard MPS-specific paths
    empty_mps_cache,               # no-op on non-MPS
    compile_module,                # torch.compile with silent fallback
    patch_attention_to_sdpa,       # monkey-patch ViT attention block
    apply_diffusers_perf_knobs,    # QKV fusion + attention slicing + VAE tiling
    PerfProfile,                   # torch.profiler context manager
    default_compile_callback,      # config-string -> compile-or-identity
)
```

Every helper falls back silently when its target platform/version
isn't available. The hot-loop never branches on perf availability —
it always calls through the same function reference, which may or
may not be `torch.compile`-wrapped.

## Validation status

* ✅ Apple Silicon perf module + tests (commit f0f9cd1)
* ✅ SDPA flash-attention routing for LanguageBind ViT (commit 574e0de)
* ✅ Profiler harness CLI (this commit)
* ⏳ Real-weight throughput numbers — pending M1 Max validation run

The validation run is currently blocked on the M2b SD3.5 score
adapter integration; until real SD3.5 weights are loaded the
`compile-mode` impact on the full showcase loop can't be measured.
