"""
SAELens: Sparse Autoencoder wrapper for interpretable embedding decomposition.

An SAE maps a dense embedding (e.g. 1024-d from ImageBind) to a sparse set of
named features, then reconstructs the original embedding from those features.
This lets us inspect *which* learned concepts are active for any given input,
and surgically adjust them before re-optimizing.

Architecture
------------
  encode:  z = ReLU(W_enc @ (x - pre_bias) + bias)   → topK sparsification
  decode:  x̂ = W_dec @ z

Where:
  x         — input embedding  [1, embed_dim]
  pre_bias  — subtracted before encoding (learned centre of the data)
  W_enc     — encoder weight matrix  [n_features, embed_dim]
  bias      — encoder bias           [n_features]
  W_dec     — decoder weight matrix  [embed_dim, n_features]
  z         — sparse feature activations  [1, n_features]
  k         — number of active features to keep (TopK sparsification)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from embedding_art.exceptions import FeatureNotFoundError

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# SAEDecomposition
# ---------------------------------------------------------------------------


@dataclass
class SAEDecomposition:
    """
    The result of decomposing an embedding through a Sparse Autoencoder.

    Attributes
    ----------
    activations:
        Sparse feature activation tensor of shape [1, n_features].
        Most entries are zero; only the top-k are non-zero.
    active_features:
        Mapping of feature_name → activation_strength for the non-zero entries.
        This is the human-readable view of the decomposition.
    reconstruction_error:
        L2 distance between the original embedding and the SAE reconstruction,
        normalised by the original embedding's norm.  Lower is better.
    """

    activations: torch.Tensor  # [1, n_features]
    active_features: dict[str, float]
    reconstruction_error: float

    @classmethod
    def from_dict(
        cls,
        features: dict[str, float],
        sae: SAELens,
    ) -> SAEDecomposition:
        """
        Construct a SAEDecomposition from a mapping of feature names to strengths.

        Useful for building synthetic decompositions in tests or for the
        ``manipulate`` workflow where you want to start from named features
        rather than a raw activation tensor.

        Parameters
        ----------
        features:
            Dict mapping feature name → desired activation strength.
            Features not in this dict are set to zero.
        sae:
            The SAELens instance whose vocabulary defines the feature indices.

        Returns
        -------
        SAEDecomposition
            A decomposition whose ``activations`` tensor reflects the supplied
            feature strengths and whose ``active_features`` mirrors ``features``
            (pruned to only the entries with non-zero strength).

        Raises
        ------
        FeatureNotFoundError
            If any key in ``features`` is not present in the SAE vocabulary.
        """
        # Validate all feature names up-front so the error is clear.
        unknown = [name for name in features if name not in sae._vocab_to_idx]
        if unknown:
            raise FeatureNotFoundError(unknown[0], list(sae._vocab_to_idx.keys()))

        activations = torch.zeros(1, sae.n_features, dtype=torch.float32)
        for name, strength in features.items():
            idx = sae._vocab_to_idx[name]
            activations[0, idx] = strength

        active = {name: strength for name, strength in features.items() if strength != 0.0}

        return cls(
            activations=activations,
            active_features=active,
            reconstruction_error=0.0,
        )


# ---------------------------------------------------------------------------
# SAELens
# ---------------------------------------------------------------------------


class SAELens:
    """
    Wrapper around a pre-trained Sparse Autoencoder for a given encoder.

    The SAE maps dense embeddings to a sparse, human-labelled feature space,
    enabling interpretable decomposition and surgical manipulation of concepts.

    Parameters
    ----------
    encoder_name:
        Identifier of the encoder this SAE was trained on (e.g. "imagebind").
    artifact_path:
        Path to the directory (or file root) containing:
        - ``weights.safetensors`` — encoder/decoder weight matrices
        - ``vocab.json``          — list mapping feature index → name
    """

    def __init__(self, encoder_name: str, artifact_path: Path) -> None:
        self.encoder_name = encoder_name
        self.artifact_path = Path(artifact_path)

        weights_path = self.artifact_path / "weights.safetensors"
        vocab_path = self.artifact_path / "vocab.json"

        # Load vocabulary first so we can build the index map.
        with vocab_path.open() as fh:
            vocab: list[str] = json.load(fh)

        self._vocab: list[str] = vocab
        self._vocab_to_idx: dict[str, int] = {name: i for i, name in enumerate(vocab)}

        # Load weight tensors from safetensors.
        try:
            from safetensors.torch import load_file
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'safetensors' package is required to load SAE weights.\n"
                "Install it with: pip install safetensors"
            ) from exc

        tensors = load_file(str(weights_path), device="cpu")
        self._W_enc: torch.Tensor = tensors["W_enc"]  # [n_features, embed_dim]
        self._W_dec: torch.Tensor = tensors["W_dec"]  # [embed_dim, n_features]
        self._bias: torch.Tensor = tensors["bias"]  # [n_features]
        self._pre_bias: torch.Tensor = tensors["pre_bias"]  # [embed_dim]
        self._k: int = int(tensors["k"].item())

        self._validate_shapes()

    # ------------------------------------------------------------------
    # Alternate constructor (for testing / programmatic use)
    # ------------------------------------------------------------------

    @classmethod
    def from_tensors(
        cls,
        W_enc: torch.Tensor,
        W_dec: torch.Tensor,
        bias: torch.Tensor,
        pre_bias: torch.Tensor,
        vocab: list[str],
        k: int,
    ) -> SAELens:
        """
        Build an SAELens directly from tensors, bypassing file I/O.

        This is the primary entry point for tests and for programmatic
        construction of SAE instances without writing files to disk.

        Parameters
        ----------
        W_enc:   [n_features, embed_dim] — encoder weight matrix
        W_dec:   [embed_dim, n_features] — decoder weight matrix
        bias:    [n_features]            — encoder bias
        pre_bias:[embed_dim]             — input centering bias
        vocab:   list of n_features feature names
        k:       number of active features to retain (TopK)

        Returns
        -------
        SAELens instance ready for decompose/reconstruct/manipulate.
        """
        instance = cls.__new__(cls)
        instance.encoder_name = "from_tensors"
        instance.artifact_path = Path(".")

        instance._W_enc = W_enc.float()
        instance._W_dec = W_dec.float()
        instance._bias = bias.float()
        instance._pre_bias = pre_bias.float()
        instance._k = k
        instance._vocab = list(vocab)
        instance._vocab_to_idx = {name: i for i, name in enumerate(vocab)}

        instance._validate_shapes()
        return instance

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_features(self) -> int:
        """Number of SAE features (width of the sparse layer)."""
        return self._W_enc.shape[0]

    @property
    def embed_dim(self) -> int:
        """Dimensionality of the input embedding space."""
        return self._W_enc.shape[1]

    @property
    def k(self) -> int:
        """Number of features kept non-zero after TopK sparsification."""
        return self._k

    @property
    def vocab(self) -> list[str]:
        """Feature names in index order."""
        return list(self._vocab)

    def feature_direction(self, feature_idx: int) -> torch.Tensor:
        """Return the decoder direction for a feature as a [1, embed_dim] tensor."""
        return self._W_dec[:, feature_idx].unsqueeze(0)

    def feature_index(self, name: str) -> int:
        """Return the index of a named feature, or raise FeatureNotFoundError."""
        if name not in self._vocab_to_idx:
            raise FeatureNotFoundError(name, list(self._vocab_to_idx.keys()))
        return self._vocab_to_idx[name]

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def decompose(self, embedding: torch.Tensor) -> SAEDecomposition:
        """
        Encode an embedding into a sparse feature decomposition.

        The encoding pipeline is:
          1. Subtract pre_bias (centre the input)
          2. Linear projection: W_enc @ centred + bias
          3. ReLU activation
          4. TopK sparsification — zero out all but the k largest activations
          5. Compute reconstruction and reconstruction error

        Parameters
        ----------
        embedding:
            Input embedding tensor. Accepted shapes:
            - [embed_dim]     — will be unsqueezed to [1, embed_dim]
            - [1, embed_dim]  — used as-is

        Returns
        -------
        SAEDecomposition with activations, active_features, and reconstruction_error.
        """
        x = embedding.float()
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, embed_dim]

        # Step 1: centre
        centred = x - self._pre_bias.unsqueeze(0)  # [1, embed_dim]

        # Step 2 & 3: encode + ReLU
        pre_act = centred @ self._W_enc.T + self._bias.unsqueeze(0)  # [1, n_features]
        acts = F.relu(pre_act)  # [1, n_features]

        # Step 4: TopK sparsification
        sparse_acts = self._topk(acts)  # [1, n_features]

        # Step 5: reconstruction error
        reconstruction = self.reconstruct(
            SAEDecomposition(
                activations=sparse_acts,
                active_features={},
                reconstruction_error=0.0,
            )
        )
        error = float(
            torch.norm(x - reconstruction).item() / (torch.norm(x).item() + 1e-8)
        )

        # Build the active_features dict
        nonzero_indices = sparse_acts[0].nonzero(as_tuple=True)[0].tolist()
        active_features: dict[str, float] = {
            self._vocab[idx]: float(sparse_acts[0, idx].item()) for idx in nonzero_indices
        }

        return SAEDecomposition(
            activations=sparse_acts,
            active_features=active_features,
            reconstruction_error=error,
        )

    def reconstruct(self, decomposition: SAEDecomposition) -> torch.Tensor:
        """
        Decode a SAEDecomposition back into embedding space.

        Parameters
        ----------
        decomposition:
            A decomposition whose ``activations`` tensor has shape [1, n_features].

        Returns
        -------
        Reconstructed embedding tensor of shape [1, embed_dim].
        """
        z = decomposition.activations.float()  # [1, n_features]
        reconstruction = z @ self._W_dec.T  # [1, embed_dim]
        reconstruction = reconstruction + self._pre_bias.unsqueeze(0)
        return reconstruction

    def manipulate(
        self,
        decomposition: SAEDecomposition,
        adjustments: dict[str, float],
    ) -> SAEDecomposition:
        """
        Return a new SAEDecomposition with named features adjusted by delta values.

        Each entry in ``adjustments`` is *added* to the current activation for
        that feature (positive values amplify, negative values suppress).
        Resulting activations are clamped to >= 0 so they remain valid ReLU
        outputs.  The ``active_features`` dict and ``reconstruction_error`` on
        the returned decomposition are recomputed from the modified activations.

        Parameters
        ----------
        decomposition:
            The starting decomposition to modify.
        adjustments:
            Dict mapping feature_name → delta.  Every key must appear in the
            SAE vocabulary; unknown names raise FeatureNotFoundError.

        Returns
        -------
        A new SAEDecomposition reflecting the adjusted activations.

        Raises
        ------
        FeatureNotFoundError
            If any key in ``adjustments`` is not in the SAE vocabulary.
        """
        unknown = [name for name in adjustments if name not in self._vocab_to_idx]
        if unknown:
            raise FeatureNotFoundError(unknown[0], list(self._vocab_to_idx.keys()))

        new_acts = decomposition.activations.clone()  # [1, n_features]

        for name, delta in adjustments.items():
            idx = self._vocab_to_idx[name]
            new_acts[0, idx] = new_acts[0, idx] + delta

        # Clamp: SAE activations are post-ReLU, so must be >= 0.
        new_acts = new_acts.clamp(min=0.0)

        # Rebuild active_features from the modified tensor.
        nonzero_indices = new_acts[0].nonzero(as_tuple=True)[0].tolist()
        active_features: dict[str, float] = {
            self._vocab[idx]: float(new_acts[0, idx].item()) for idx in nonzero_indices
        }

        return SAEDecomposition(
            activations=new_acts,
            active_features=active_features,
            reconstruction_error=decomposition.reconstruction_error,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _topk(self, acts: torch.Tensor) -> torch.Tensor:
        """
        Zero out all but the k largest activations in a [1, n_features] tensor.

        Parameters
        ----------
        acts:
            Pre-sparsified activations of shape [1, n_features].

        Returns
        -------
        Sparse activation tensor of the same shape.
        """
        k = min(self._k, acts.shape[-1])
        topk_values, topk_indices = torch.topk(acts, k=k, dim=-1)
        sparse = torch.zeros_like(acts)
        sparse.scatter_(-1, topk_indices, topk_values)
        return sparse

    def _validate_shapes(self) -> None:
        """Raise ValueError if the loaded tensors have incompatible shapes."""
        n, d = self._W_enc.shape
        if self._W_dec.shape != (d, n):
            raise ValueError(
                f"Shape mismatch: W_enc is [{n}, {d}] but W_dec is {list(self._W_dec.shape)} "
                f"(expected [{d}, {n}])"
            )
        if self._bias.shape != (n,):
            raise ValueError(
                f"Shape mismatch: bias is {list(self._bias.shape)}, expected [{n}]"
            )
        if self._pre_bias.shape != (d,):
            raise ValueError(
                f"Shape mismatch: pre_bias is {list(self._pre_bias.shape)}, expected [{d}]"
            )
        if len(self._vocab) != n:
            raise ValueError(
                f"Vocabulary length {len(self._vocab)} does not match "
                f"n_features {n} from W_enc shape"
            )
        if self._k < 1:
            raise ValueError(f"k must be >= 1, got {self._k}")
