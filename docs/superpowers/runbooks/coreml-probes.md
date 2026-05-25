# CoreML probe runbook (M1/M2 Max)

This runbook covers the end-to-end validation of the CoreML probe path
on Apple Silicon. The conversion infrastructure ships in `perf/coreml.py`
and the registry of CoreML-convertible probes lives in
`perf/probe_registry.py`. The CLIs `embed-art compile-probes` and
`embed-art bench-probes` drive the conversion + benchmark loop.

This must be run on macOS with `coremltools` installed; on Linux the
commands degrade to a no-op that records a clear "skipped" reason in
the report JSON.

## Setup

```
pip install coremltools  # macOS only
```

The required model weights (SigLIP 2 SO400M, CLAP HTSAT) are pulled by
Hugging Face on first probe load. Expect ~3 GB total.

## Compile

```
embed-art compile-probes \
  -o .cache/coreml-probes \
  --probes all \
  --compute-units ALL
```

Expected output: `.cache/coreml-probes/{siglip2-image,clap-audio}.mlpackage/`
plus a `compile_report.json` summarising success per probe.

Use `--compute-units CPU_AND_NE` to force ANE-or-CPU (no GPU) — useful
when you specifically want to validate ANE residency.

The compile step is idempotent. Re-running without `--force` reuses
existing artefacts.

## Benchmark

```
embed-art bench-probes \
  --probes-dir .cache/coreml-probes \
  --probes all \
  --n-iters 50 \
  --n-warmup 5 \
  --output .cache/coreml-probes/bench_report.json
```

Expected output (M1 Max numbers; ranges based on Apple's published ANE
throughput, to be re-measured):

| probe         | pytorch (cpu/mps) | coreml (ANE)  | speedup |
| ------------- | ----------------- | ------------- | ------- |
| siglip2-image | ~120-180 ms       | ~25-40 ms     | 4-6x    |
| clap-audio    | ~80-120 ms        | ~20-35 ms     | 3-5x    |

These numbers are guidance; the harness writes the actual values to
the report JSON. The expected speedup range is wide because CoreML's
ANE scheduling depends on model shape and quantisation: a 4-6x speedup
is consistent with Apple's published numbers for ViT-class probes.

## Validation checklist

After running both commands the following should be true on M1/M2:

* [ ] `.cache/coreml-probes/siglip2-image.mlpackage/` exists and is
      non-empty.
* [ ] `.cache/coreml-probes/clap-audio.mlpackage/` exists and is
      non-empty.
* [ ] `compile_report.json` shows `ok: true` for every requested
      probe.
* [ ] `bench_report.json` reports `coreml_speedup_vs_pytorch > 1.5`
      for both probes (anything below 1.5x suggests ANE was not
      engaged — try `--compute-units CPU_AND_NE`).
* [ ] `numpy.allclose(pytorch_output, coreml_output, atol=1e-3)` for
      the same dummy input — verifies semantic equivalence, not just
      latency.

The last item is **not** automated by `bench-probes` yet because it
requires the model-specific output-extraction logic; see the
"correctness sweep" follow-up below.

## Follow-ups

* **Correctness sweep.** Add a `--check-correctness` flag to
  `bench-probes` that asserts allclose between the two backends on a
  shared input. Not yet wired because each probe needs custom output
  key extraction (CoreML output names depend on `ct.convert`'s naming).
* **Half-precision conversion.** `compile_probe_to_coreml` currently
  uses float32. Adding `--precision fp16` could double throughput on
  ANE. Investigate after the float32 baseline is validated.
* **Cache integration.** The compiled `.mlpackage`s should be the
  default backend for the cross-encoder evaluation card; wire this
  through `evaluation/probes.py` once correctness is confirmed.

When the validation checklist above passes, file a follow-up via `bd`
documenting the measured numbers and check the runbook into
`docs/superpowers/results/coreml-probes/`.
