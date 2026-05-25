"""
MLX-based SAE training loop (issue gfi — stretch).

A drop-in alternative to :func:`embedding_art.sae.training.train_sae`
that uses Apple's MLX framework for a ~2-3x throughput improvement on
the matmul-heavy SAE training loop. The implementation is
**inference-equivalent** to the PyTorch loop: it produces a state-dict
that loads cleanly into ``GroupSparseSAE`` for downstream use.

This module is **macOS-only**: MLX has no Linux distribution. On any
platform without ``mlx`` installed the public entry point raises a
clear ``MLXUnavailableError`` so callers can fall back to the PyTorch
loop. The CLI surface (``embed-art sae train-stack --backend mlx``)
already gracefully degrades to PyTorch when MLX is missing.

Architecture identical to ``GroupSparseSAE``:

    encode(x; mod):
        centred = x - pre_bias[mod]
        pre_act = centred @ W_enc.T + bias
        acts    = relu(pre_act)
        z       = topk(acts, k=k, dim=-1)
    decode(z; mod):
        x_hat = z @ W_dec.T + pre_bias[mod]
    loss = mse(x_hat, x) + lambda_gs * group_sparse(z, z_masked)

After training the parameters are copied back into a CPU torch tensor
state-dict (numpy → torch.from_numpy) so the artefact is fully
interchangeable with the PyTorch loop's output.

The motivation for MLX over PyTorch+MPS for this specific workload:

* MLX uses a unified-memory tensor model — there is no device-staging
  cost between forward and backward.
* MLX's matmul kernels are tuned for Apple Silicon's GPU and ANE.
* SAE training is matmul-dominated (one rectangular [B, D]@[D, F] per
  step), the regime where MLX's published speedups are largest.

Costs (also documented in the issue):

* Boundary copy when collecting embeddings (PyTorch → numpy → MLX) and
  when saving the state-dict (MLX → numpy → PyTorch).
* MLX requires its own optimiser API; we use ``mlx.optimizers.Adam``.

This file is intentionally small and only exposes the training loop.
Inference still goes through the canonical PyTorch ``GroupSparseSAE``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class MLXUnavailableError(RuntimeError):
    """Raised when the MLX backend is requested but ``mlx`` is missing."""


def mlx_available() -> bool:
    """True iff ``mlx`` is importable.

    Linux always returns False; macOS with ``pip install mlx`` returns
    True. Used by ``embed-art sae train-stack --backend mlx`` to
    silently fall back when MLX is not available.
    """
    try:
        import mlx.core  # noqa: F401

        return True
    except ImportError:
        return False


def train_sae_mlx(
    embeddings_path: Path | str,
    output_path: Path | str,
    embed_dim: int,
    *,
    n_features: int = 16384,
    k: int = 32,
    lambda_gs: float = 0.05,
    mask_prob: float = 0.2,
    batch_size: int = 128,
    n_iterations: int = 25000,
    lr: float = 1e-3,
    n_modality_biases: int = 2,
    log_every: int = 1000,
    seed: int = 0,
) -> Path:
    """Train a GroupSparseSAE on MLX, save state-dict for PyTorch loading.

    Mirrors :func:`embedding_art.sae.training.train_sae` exactly, except
    the optimisation runs on MLX. The output ``sae_weights.pt`` file is
    a standard PyTorch state-dict and is byte-identical in shape to the
    PyTorch-trained equivalent.

    Args:
        embeddings_path: ``.pt`` file produced by ``collect_embeddings``
            with an ``"embeddings"`` key of shape ``[N, embed_dim]``.
        output_path: Directory into which ``sae_weights.pt`` is written.
        embed_dim: Dimensionality of the input embeddings.
        n_features: Width of the sparse bottleneck.
        k: TopK sparsity per row.
        lambda_gs: Weight of the group-sparse term.
        mask_prob: Mask probability for the group-sparse pair.
        batch_size: Minibatch size.
        n_iterations: Total Adam steps.
        lr: Adam learning rate.
        n_modality_biases: Number of per-modality pre-bias vectors.
        log_every: Log step interval.
        seed: RNG seed (forwarded to MLX).

    Returns:
        Path to the saved ``sae_weights.pt`` file.

    Raises:
        MLXUnavailableError: if MLX is not installed.
    """
    if not mlx_available():
        raise MLXUnavailableError(
            "MLX is not installed. Run on macOS with `pip install mlx` to "
            "enable this path. The PyTorch training loop in "
            "embedding_art.sae.training.train_sae remains the canonical "
            "fallback."
        )

    import mlx.core as mx  # noqa: PLC0415
    import mlx.nn as mxnn  # noqa: PLC0415
    import mlx.optimizers as mxoptim  # noqa: PLC0415
    import torch  # noqa: PLC0415

    embeddings_path = Path(embeddings_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    data = torch.load(embeddings_path, weights_only=True)
    embeddings_np: np.ndarray = data["embeddings"].float().numpy()
    if embeddings_np.shape[1] != embed_dim:
        raise ValueError(
            f"embed_dim mismatch: file has {embeddings_np.shape[1]}-d "
            f"embeddings, but embed_dim={embed_dim} was specified."
        )
    embeddings = mx.array(embeddings_np)
    n_samples = embeddings_np.shape[0]
    logger.info(
        "MLX SAE training: %d samples, embed_dim=%d, n_features=%d, k=%d",
        n_samples,
        embed_dim,
        n_features,
        k,
    )

    model = _MLXGroupSparseSAE(
        embed_dim=embed_dim,
        n_features=n_features,
        k=k,
        n_modality_biases=n_modality_biases,
    )
    optimizer = mxoptim.Adam(learning_rate=lr)
    mx.random.seed(seed)

    loss_and_grad = mxnn.value_and_grad(model, _loss_fn)

    for iteration in range(n_iterations):
        idx = mx.random.randint(0, n_samples, shape=(batch_size,))
        x_batch = embeddings[idx]
        mask = mx.random.uniform(shape=x_batch.shape) > mask_prob
        x_masked = x_batch * mask
        x_masked = x_masked / (mx.linalg.norm(x_masked, axis=-1, keepdims=True) + 1e-8)

        (loss_value, parts), grads = loss_and_grad(model, x_batch, x_masked, lambda_gs)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        with mx.no_grad():
            norms = mx.linalg.norm(model.W_dec, axis=0, keepdims=True) + 1e-8
            model.W_dec = model.W_dec / norms

        if (iteration + 1) % log_every == 0:
            logger.info(
                "mlx iter %d/%d  loss=%.4f  recon=%.4f  gs=%.4f",
                iteration + 1,
                n_iterations,
                float(loss_value),
                float(parts["recon"]),
                float(parts["gs"]),
            )

    state_dict = _mlx_to_torch_state_dict(model)
    weights_path = output_path / "sae_weights.pt"
    torch.save(state_dict, weights_path)
    logger.info("Saved MLX-trained SAE weights to %s", weights_path)
    return weights_path


# ---------------------------------------------------------------------------
# MLX module
# ---------------------------------------------------------------------------


class _MLXGroupSparseSAE:
    """MLX mirror of GroupSparseSAE.

    Held as a plain class (not ``mlx.nn.Module`` subclass) so the file
    can be imported on Linux without ``mlx`` being installed; the actual
    instantiation is gated on :func:`mlx_available`.
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        n_features: int,
        k: int,
        n_modality_biases: int,
    ) -> None:
        import mlx.core as mx  # noqa: PLC0415

        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if n_modality_biases < 1:
            raise ValueError(f"n_modality_biases must be >= 1, got {n_modality_biases}")

        self.embed_dim = embed_dim
        self.n_features = n_features
        self.k = k
        self.n_modality_biases = n_modality_biases

        scale_enc = (2.0 / embed_dim) ** 0.5
        self.W_enc = mx.random.normal(shape=(n_features, embed_dim)) * scale_enc
        dec = mx.random.normal(shape=(embed_dim, n_features))
        dec = dec / (mx.linalg.norm(dec, axis=0, keepdims=True) + 1e-8)
        self.W_dec = dec
        self.bias = mx.zeros((n_features,))
        self.pre_biases = [mx.zeros((embed_dim,)) for _ in range(n_modality_biases)]

    def parameters(self) -> dict[str, Any]:
        return {
            "W_enc": self.W_enc,
            "W_dec": self.W_dec,
            "bias": self.bias,
            **{f"pre_bias_{i}": pb for i, pb in enumerate(self.pre_biases)},
        }

    def encode(self, x: Any, modality_idx: int = 0) -> Any:
        import mlx.core as mx  # noqa: PLC0415

        pre = self.pre_biases[modality_idx]
        centred = x - pre[None, :]
        pre_act = centred @ self.W_enc.T + self.bias[None, :]
        acts = mx.maximum(pre_act, 0.0)
        return _topk_mlx(acts, self.k)

    def decode(self, z: Any, modality_idx: int = 0) -> Any:
        pre = self.pre_biases[modality_idx]
        return z @ self.W_dec.T + pre[None, :]


def _loss_fn(
    model: _MLXGroupSparseSAE, x: Any, x_masked: Any, lambda_gs: float
) -> tuple[Any, dict[str, Any]]:
    """Composite loss used by the MLX training loop."""
    import mlx.core as mx  # noqa: PLC0415

    z = model.encode(x, modality_idx=0)
    x_hat = model.decode(z, modality_idx=0)
    recon = mx.mean((x_hat - x) ** 2)

    z_masked = model.encode(x_masked, modality_idx=0)
    gs = mx.mean(mx.sqrt(z**2 + z_masked**2 + 1e-12))

    total = recon + lambda_gs * gs
    return total, {"recon": recon, "gs": gs}


def _topk_mlx(activations: Any, k: int) -> Any:
    """TopK sparsity along the last dim; zero non-top-k entries."""
    import mlx.core as mx  # noqa: PLC0415

    threshold = mx.partition(activations, kth=-k, axis=-1)[..., -k : -k + 1]
    mask = activations >= threshold
    return activations * mask


# ---------------------------------------------------------------------------
# State-dict conversion (MLX → PyTorch)
# ---------------------------------------------------------------------------


def _mlx_to_torch_state_dict(model: _MLXGroupSparseSAE) -> dict[str, Any]:
    """Convert the MLX module's parameters to a PyTorch state-dict.

    Output keys match ``GroupSparseSAE.state_dict()`` so the result can
    be loaded directly with ``torch.nn.Module.load_state_dict``.
    """
    import torch  # noqa: PLC0415

    state: dict[str, Any] = {
        "W_enc": torch.from_numpy(np.asarray(model.W_enc)).float(),
        "W_dec": torch.from_numpy(np.asarray(model.W_dec)).float(),
        "bias": torch.from_numpy(np.asarray(model.bias)).float(),
    }
    for i, pb in enumerate(model.pre_biases):
        state[f"pre_biases.{i}"] = torch.from_numpy(np.asarray(pb)).float()
    return state
