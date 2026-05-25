# MLX SAE training runbook (M1/M2 Max)

Port of the SAE training loop to Apple's MLX framework for ~2-3x
throughput on Apple Silicon. macOS-only.

## Setup

```
pip install mlx
```

On a Mac with `mlx` installed, ``embed-art sae train --backend mlx``
takes the same arguments as the canonical PyTorch path and writes a
``sae_weights.pt`` that loads cleanly into ``GroupSparseSAE`` for
downstream use.

## Train

```
embed-art sae train \
  --embeddings .cache/sae/image_embeddings.pt \
  --output .cache/sae/image_mlx \
  --features 16384 \
  --sparsity 32 \
  --backend mlx
```

Expected throughput on M1 Max (16384-feature SAE, 64 GB unified memory):

| backend  | iters/sec | wall time @ 25k steps |
| -------- | --------: | --------------------: |
| torch+mps | ~120-160 | ~3-4 min              |
| mlx      | ~280-380  | ~1-2 min              |

These are guidance numbers; the actual values must be measured on real
hardware. The factor-2-3 speedup is consistent with MLX's published
matmul benchmarks for the [B, D] @ [D, F] shape SAE training spends
most of its time on.

## Backend selection

* ``--backend torch`` — canonical PyTorch path (default).
* ``--backend mlx`` — MLX path. On Linux falls back to torch with a
  warning so copy-pasted commands don't blow up.
* ``--backend auto`` — picks MLX when available, otherwise torch.

## Numerical equivalence

The MLX path uses the same architecture and the same loss formulation
as ``GroupSparseSAE``. The saved ``state_dict`` keys match
character-for-character, so the trained artefact can be loaded with:

```python
from embedding_art.sae.training import GroupSparseSAE
sae = GroupSparseSAE(embed_dim=768, n_features=16384, k=32)
sae.load_state_dict(torch.load("sae_weights.pt"))
```

Note: at this stage we explicitly do **not** claim bit-identical
reconstruction loss between MLX and torch runs because:

1. RNG seeds use different streams (mlx.random vs torch).
2. MLX's Adam may differ in epsilon handling vs torch.optim.Adam.
3. Floating-point reduction order on GPU vs unified-memory MLX
   tensors will produce micro-differences.

What we DO claim is structural equivalence (same architecture, same
state-dict shapes, same loss formulation) and quality equivalence
(final reconstruction MSE and feature-activation statistics should
match within a few percent of the torch baseline).

## Validation checklist

After running the MLX training loop on M1/M2:

* [ ] ``sae_weights.pt`` exists in the output directory.
* [ ] ``torch.load(weights_path)`` returns a dict with keys
      ``W_enc``, ``W_dec``, ``bias``, ``pre_biases.0``, ``pre_biases.1``.
* [ ] All tensors are float32 CPU tensors.
* [ ] ``GroupSparseSAE.load_state_dict(...)`` succeeds without
      ``RuntimeError`` about missing or unexpected keys.
* [ ] ``embed-art sae inspect`` on the MLX-trained checkpoint produces
      a sensible feature activation histogram (most features near
      zero, k features active per input).
* [ ] Wall-time measured against the torch+mps baseline shows
      ≥1.5x speedup. <1.5x suggests the MLX path is not engaging
      Apple's GPU as expected — check `mx.default_device()`.

## Follow-ups

* **Bench command.** Add ``embed-art sae bench-backend`` that runs a
  short (1000-step) training session under both backends and reports
  iters/sec + final MSE. Useful as a regression guard once MLX is
  validated.
* **Multi-modality.** The MLX loop currently uses ``modality_idx=0``
  only. Extend to the full multi-pre-bias path once the single-
  modality numbers are validated.
* **MLX optimisers.** The current loop uses ``mlx.optimizers.Adam``.
  MLX also ships ``AdamW`` and ``SGD``; route the existing ``--lr`` /
  ``--optimizer`` knobs through ``train_sae_mlx`` when the canonical
  path adds them.

When the MLX speedup is validated, file a follow-up via ``bd`` with
the measured numbers and check the runbook into
``docs/superpowers/results/mlx-sae-training/``.
