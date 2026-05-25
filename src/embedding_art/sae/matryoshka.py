"""
Matryoshka SAE (Bussmann et al. 2024 / Kissane et al. 2025).

A nested-prefix Sparse Autoencoder: feature indices ``[0:n_1] ⊂ [0:n_2]
⊂ ... ⊂ [0:n_K]`` form *valid sub-decompositions* on their own. The
training loss is the sum (or weighted sum) of reconstruction MSE
evaluated at every nested width. This makes every prefix a usable SAE,
yielding a single artefact that Pareto-dominates a standard SAE at any
fixed width below the full one.

Two practical wins:

* Pick any width at inference time. Same checkpoint serves "fast,
  imprecise" via the smallest nest and "slow, exhaustive" via the
  largest, without retraining.
* Better feature granularity at small widths. The smallest nest is
  forced to span coarse semantics; larger nests then specialise.

Implementation notes:

* TopK sparsity is applied *within* each nest (the first ``n_i``
  feature columns), so the per-nest active count stays bounded and
  comparable.
* The decoder weight matrix is *shared* across nests — only the active
  feature columns differ. This is what makes the nests genuinely
  nested rather than independent SAEs glued together.
* No per-modality biases here. Matryoshka SAE is a structural choice
  about feature ordering; per-modality biases can be added in a
  subclass if needed (mirroring :class:`GroupSparseSAE`).

References:
    Bussmann, B. et al. (2024). Learning Multi-Level Features with
    Matryoshka Sparse Autoencoders. arXiv:2412.17744.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class MatryoshkaSAE(nn.Module):
    """Nested-prefix SAE.

    Args:
        embed_dim: Dimensionality of the input embedding.
        nested_sizes: Strictly-increasing list of nest widths. The
            largest entry is the full feature count. Every entry must
            be ``>= k``.
        k: TopK sparsity applied *within each nest*. The same ``k``
            is used at every nest; deeper nests get *more* available
            features to choose from but still keep only ``k`` active.
        nest_weights: Optional per-nest weighting for the reconstruction
            loss. Defaults to uniform weighting.

    Attributes:
        W_enc: Encoder weights, shape ``[max_features, embed_dim]``.
        W_dec: Decoder weights, shape ``[embed_dim, max_features]``.
        bias: Per-feature encoder bias.
    """

    def __init__(
        self,
        embed_dim: int,
        nested_sizes: Sequence[int],
        k: int = 32,
        nest_weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        nested = list(nested_sizes)
        if not nested:
            raise ValueError("nested_sizes must be non-empty")
        for prev, cur in zip(nested, nested[1:]):
            if cur <= prev:
                raise ValueError(f"nested_sizes must be strictly increasing, got {nested}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if k > nested[0]:
            raise ValueError(
                f"k={k} exceeds the smallest nest size {nested[0]}; "
                "every nest must contain at least k features"
            )
        if nest_weights is not None and len(nest_weights) != len(nested):
            raise ValueError(
                f"nest_weights length {len(nest_weights)} != " f"len(nested_sizes)={len(nested)}"
            )

        self.embed_dim = embed_dim
        self.nested_sizes: list[int] = nested
        self.n_features = nested[-1]
        self.k = k

        if nest_weights is None:
            uniform = 1.0 / len(nested)
            self.register_buffer(
                "nest_weights",
                torch.tensor([uniform] * len(nested), dtype=torch.float32),
            )
        else:
            self.register_buffer(
                "nest_weights",
                torch.tensor(list(nest_weights), dtype=torch.float32),
            )

        self.W_enc = nn.Parameter(torch.empty(self.n_features, embed_dim))
        self.W_dec = nn.Parameter(torch.empty(embed_dim, self.n_features))
        self.bias = nn.Parameter(torch.zeros(self.n_features))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.W_enc, nonlinearity="relu")
        nn.init.normal_(self.W_dec)
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    # ------------------------------------------------------------------
    # Forward / nest-specific operations
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the largest-nest sparse activations.

        For finer control use :meth:`encode_nest`. ``forward`` returns
        the full-width activation vector so it slot-replaces a
        :class:`GroupSparseSAE` for downstream consumers (e.g. the
        interpretation bundle).
        """
        return self.encode_nest(x, self.nested_sizes[-1])

    def encode_nest(self, x: torch.Tensor, nest_size: int) -> torch.Tensor:
        """TopK-encode ``x`` within the first ``nest_size`` features.

        Returns a ``[B, n_features]`` tensor with the active features
        confined to the first ``nest_size`` columns (so a downstream
        decoder@W_dec multiplication naturally uses only the relevant
        decoder columns).
        """
        if nest_size > self.n_features:
            raise ValueError(f"nest_size {nest_size} exceeds max features {self.n_features}")
        pre_act = x @ self.W_enc[:nest_size].T + self.bias[:nest_size].unsqueeze(0)
        acts = F.relu(pre_act)
        sparse_nest = self._topk(acts)
        # Pad to full width so caller-side decoder shapes are stable.
        if nest_size == self.n_features:
            return sparse_nest
        pad = torch.zeros(
            x.shape[0],
            self.n_features - nest_size,
            dtype=sparse_nest.dtype,
            device=sparse_nest.device,
        )
        return torch.cat([sparse_nest, pad], dim=-1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode full-width activations back to embedding space."""
        return z @ self.W_dec.T

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def matryoshka_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Sum of per-nest reconstruction MSE losses.

        For each nest, encode within that nest, decode, and accumulate
        ``nest_weights[i] * MSE(x, x_hat)``.
        """
        loss = x.new_zeros(())
        for weight, nest_size in zip(self.nest_weights, self.nested_sizes):
            z = self.encode_nest(x, nest_size)
            x_hat = self.decode(z)
            loss = loss + weight * F.mse_loss(x_hat, x)
        return loss

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _topk(self, acts: torch.Tensor) -> torch.Tensor:
        k = min(self.k, acts.shape[-1])
        topk_values, topk_indices = torch.topk(acts, k=k, dim=-1)
        sparse = torch.zeros_like(acts)
        sparse.scatter_(-1, topk_indices, topk_values)
        return sparse

    def normalize_decoder(self) -> None:
        """Re-normalise decoder columns to unit norm.

        Should be called once per training step after the optimiser
        step; identical convention to :class:`GroupSparseSAE`.
        """
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)


def train_matryoshka_sae(
    embeddings: torch.Tensor,
    *,
    nested_sizes: Sequence[int],
    k: int = 32,
    batch_size: int = 128,
    n_iterations: int = 1000,
    lr: float = 1e-3,
    nest_weights: Sequence[float] | None = None,
    device: str | torch.device = "cpu",
) -> MatryoshkaSAE:
    """Train a :class:`MatryoshkaSAE` on a pre-collected embedding tensor.

    Returns the trained module. Intentionally lightweight (no
    checkpointing, no LR schedule, no dataset abstraction) — the
    point of this function is to be the minimum-viable trainer that
    test suites and tutorials can call. Production training goes via
    a richer pipeline in :mod:`embedding_art.sae.training`.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got {tuple(embeddings.shape)}")

    embed_dim = embeddings.shape[1]
    sae = MatryoshkaSAE(
        embed_dim=embed_dim,
        nested_sizes=nested_sizes,
        k=k,
        nest_weights=nest_weights,
    ).to(device)
    embeddings = embeddings.to(device)

    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    n_samples = embeddings.shape[0]

    for _ in range(n_iterations):
        idx = torch.randint(0, n_samples, (min(batch_size, n_samples),), device=device)
        batch = embeddings[idx]
        loss = sae.matryoshka_loss(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sae.normalize_decoder()

    return sae
