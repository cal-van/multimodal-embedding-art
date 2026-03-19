"""
SAE Training Pipeline for Embedding Art.

Trains a GroupSparseSAE over a corpus of embeddings produced by any multimodal
encoder.  The pipeline has four stages:

  1. ``collect_embeddings`` — encode image files to a cached .pt tensor file.
  2. ``GroupSparseSAE``     — the SAE model with TopK sparsity and a group-sparse
                             cross-modal loss term.
  3. ``train_sae``          — training loop that learns interpretable features.
  4. ``label_features``     — assign a human-readable text label to every feature
                             by finding the vocabulary word with the highest
                             cosine similarity to each decoder column.

Architecture
------------
  encode:  z = TopK(ReLU(W_enc @ (x - pre_bias[modality]) + bias))
  decode:  x̂ = W_dec @ z

Group sparse loss encourages paired cross-modal embeddings (image + audio of the
same concept) to activate the *same* set of features rather than disjoint ones:

  L_gs = sum_i sqrt(z_x_i^2 + z_y_i^2)

This is the L_{2,1} norm across the two activation vectors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from embedding_art.encoders.base import Encoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage 1: Collect embeddings
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def collect_embeddings(
    encoder: Encoder,
    dataset_path: Path | str,
    output_path: Path | str,
    batch_size: int = 32,
    max_samples: int | None = None,
) -> Path:
    """
    Encode all image files in *dataset_path* and save the result as a .pt file.

    The saved file is a dict with key ``"embeddings"`` whose value is a float32
    tensor of shape ``[N, embed_dim]``.  Each row is the L2-normalised embedding
    for one image.

    Parameters
    ----------
    encoder:
        Any object that implements ``encode_image(path) -> Tensor[1, D]``.
    dataset_path:
        Directory that is searched recursively for image files.
    output_path:
        Destination .pt file.  Parent directories are created if needed.
    batch_size:
        Number of images to encode before logging progress.  Has no effect on
        correctness — embeddings are accumulated incrementally.
    max_samples:
        If given, stop after encoding this many images.

    Returns
    -------
    Path
        The resolved path of the saved .pt file.

    Raises
    ------
    FileNotFoundError
        If *dataset_path* does not exist.
    ValueError
        If no image files are found in *dataset_path*.
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    # Collect image paths (recursively, deterministic order).
    image_paths = sorted(
        p for p in dataset_path.rglob("*") if p.suffix.lower() in _IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise ValueError(f"No image files found in {dataset_path}")

    if max_samples is not None:
        image_paths = image_paths[:max_samples]

    logger.info("Encoding %d images from %s …", len(image_paths), dataset_path)

    all_embeddings: list[torch.Tensor] = []

    for i, img_path in enumerate(image_paths):
        try:
            emb = encoder.encode_image(img_path)  # [1, D]
        except Exception:
            logger.warning("Failed to encode %s — skipping.", img_path, exc_info=True)
            continue

        all_embeddings.append(emb.detach().cpu().float())

        if (i + 1) % batch_size == 0:
            logger.info("  … %d / %d done", i + 1, len(image_paths))

    if not all_embeddings:
        raise ValueError("All image encodings failed — nothing to save.")

    embeddings = torch.cat(all_embeddings, dim=0)  # [N, D]
    embeddings = F.normalize(embeddings, dim=-1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": embeddings}, output_path)

    logger.info("Saved %d embeddings to %s", embeddings.shape[0], output_path)
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Stage 2: GroupSparseSAE model
# ---------------------------------------------------------------------------


class GroupSparseSAE(nn.Module):
    """
    Sparse Autoencoder with TopK sparsity and per-modality input biases.

    The encoder projects a centred embedding into a high-dimensional feature
    space, keeps only the k largest activations (TopK sparsity), then the
    decoder projects back to embedding space.

    Per-modality pre-biases (``pre_biases``) allow the SAE to handle small
    distributional shifts between modalities (e.g. image vs. audio embeddings
    from ImageBind) without separate models.

    Parameters
    ----------
    embed_dim:
        Dimensionality of the input/output embeddings (e.g. 1024 for ImageBind).
    n_features:
        Width of the sparse bottleneck layer.  Typically 8–16× ``embed_dim``.
    k:
        Number of features kept non-zero after TopK sparsification.
    n_modality_biases:
        Number of per-modality pre-bias vectors to allocate.  Index 0 is used
        by default; set ``modality_idx`` in ``forward`` / ``decode`` to select.
    """

    def __init__(
        self,
        embed_dim: int,
        n_features: int,
        k: int,
        n_modality_biases: int = 2,
    ) -> None:
        super().__init__()

        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if n_modality_biases < 1:
            raise ValueError(f"n_modality_biases must be >= 1, got {n_modality_biases}")

        self.embed_dim = embed_dim
        self.n_features = n_features
        self.k = k

        # Encoder / decoder weight matrices.
        self.W_enc: nn.Parameter = nn.Parameter(torch.empty(n_features, embed_dim))
        self.W_dec: nn.Parameter = nn.Parameter(torch.empty(embed_dim, n_features))

        # Encoder bias (one per feature).
        self.bias: nn.Parameter = nn.Parameter(torch.zeros(n_features))

        # Per-modality pre-biases (subtract before encoding, add after decoding).
        self.pre_biases: nn.ParameterList = nn.ParameterList(
            [nn.Parameter(torch.zeros(embed_dim)) for _ in range(n_modality_biases)]
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Kaiming-uniform encoder; unit-column decoder."""
        nn.init.kaiming_uniform_(self.W_enc, nonlinearity="relu")

        # Initialise decoder with random unit columns — the standard approach
        # for SAE dictionaries, matching SAELens convention.
        nn.init.normal_(self.W_dec)
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    # ------------------------------------------------------------------
    # Forward / decode
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, modality_idx: int = 0) -> torch.Tensor:
        """
        Encode *x* to a sparse feature vector.

        Parameters
        ----------
        x:
            Input embedding of shape ``[B, embed_dim]`` or ``[1, embed_dim]``.
        modality_idx:
            Which pre_bias to subtract before encoding.

        Returns
        -------
        Sparse activation tensor of shape ``[B, n_features]`` with exactly
        *k* non-zero entries per row.
        """
        pre_bias = self.pre_biases[modality_idx]
        centred = x - pre_bias.unsqueeze(0)  # [B, embed_dim]

        pre_act = centred @ self.W_enc.T + self.bias.unsqueeze(0)  # [B, n_features]
        acts = F.relu(pre_act)  # [B, n_features]

        return self._topk(acts)  # [B, n_features]

    def decode(self, z: torch.Tensor, modality_idx: int = 0) -> torch.Tensor:
        """
        Decode sparse activations back to embedding space.

        Parameters
        ----------
        z:
            Sparse activation tensor of shape ``[B, n_features]``.
        modality_idx:
            Which pre_bias to add after decoding.

        Returns
        -------
        Reconstructed embedding of shape ``[B, embed_dim]``.
        """
        pre_bias = self.pre_biases[modality_idx]
        reconstruction = z @ self.W_dec.T  # [B, embed_dim]
        return reconstruction + pre_bias.unsqueeze(0)

    # ------------------------------------------------------------------
    # Loss functions
    # ------------------------------------------------------------------

    def reconstruction_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        Mean squared error between original and reconstructed embeddings.

        Parameters
        ----------
        x:
            Original embedding batch, shape ``[B, embed_dim]``.
        x_hat:
            Reconstructed embedding batch, shape ``[B, embed_dim]``.

        Returns
        -------
        Scalar MSE loss tensor.
        """
        return F.mse_loss(x_hat, x)

    def group_sparse_loss(self, z_x: torch.Tensor, z_y: torch.Tensor) -> torch.Tensor:
        """
        L_{2,1} group-sparse loss across two activation vectors.

        Encourages paired embeddings (e.g. image and audio of the same concept)
        to activate the same set of features:

            L = sum_i sqrt(z_x_i^2 + z_y_i^2)

        where the sum is over all n_features and the batch.

        Parameters
        ----------
        z_x:
            Activation tensor for modality X, shape ``[B, n_features]``.
        z_y:
            Activation tensor for modality Y, shape ``[B, n_features]``.

        Returns
        -------
        Scalar group-sparse loss tensor.
        """
        # Compute the per-feature L2 norm across the two modalities, then sum.
        return torch.sqrt(z_x.pow(2) + z_y.pow(2) + 1e-8).sum()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _topk(self, acts: torch.Tensor) -> torch.Tensor:
        """
        Zero out all but the k largest activations in each row.

        Parameters
        ----------
        acts:
            Pre-sparsified activations of shape ``[B, n_features]``.

        Returns
        -------
        Sparse activation tensor of the same shape.
        """
        k = min(self.k, acts.shape[-1])
        topk_values, topk_indices = torch.topk(acts, k=k, dim=-1)
        sparse = torch.zeros_like(acts)
        sparse.scatter_(-1, topk_indices, topk_values)
        return sparse

    def normalize_decoder(self) -> None:
        """
        Re-normalise decoder columns to unit norm after each gradient step.

        This prevents the model from "cheating" by making decoder columns
        very small (which would trivially minimise reconstruction error).
        Should be called once per training iteration, after the optimiser step.
        """
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)


# ---------------------------------------------------------------------------
# Stage 3: Training loop
# ---------------------------------------------------------------------------


def train_sae(
    embeddings_path: Path | str,
    output_path: Path | str,
    embed_dim: int,
    n_features: int = 16384,
    k: int = 32,
    lambda_gs: float = 0.05,
    mask_prob: float = 0.2,
    batch_size: int = 128,
    n_iterations: int = 25000,
    lr: float = 1e-3,
) -> GroupSparseSAE:
    """
    Train a GroupSparseSAE on a pre-collected embeddings file.

    The training uses a single-modality dataset (no paired cross-modal data is
    required).  The group-sparse loss is computed between the original embedding
    and a randomly masked copy, which encourages the SAE to activate overlapping
    features for semantically similar (but noisy) inputs.

    Parameters
    ----------
    embeddings_path:
        Path to the .pt file produced by ``collect_embeddings``.  Must contain
        an ``"embeddings"`` key with tensor of shape ``[N, embed_dim]``.
    output_path:
        Directory where the trained SAE will be saved:
        - ``sae_weights.pt`` — full state dict
    embed_dim:
        Dimensionality of the embedding space.
    n_features:
        Width of the SAE bottleneck.
    k:
        TopK sparsity — number of active features per input.
    lambda_gs:
        Weight of the group-sparse loss term.
    mask_prob:
        Probability of zeroing each dimension when creating the masked copy
        used for the group-sparse loss.
    batch_size:
        Training mini-batch size.
    n_iterations:
        Total number of gradient steps.
    lr:
        Adam learning rate.

    Returns
    -------
    GroupSparseSAE
        The trained model (also saved to *output_path*).
    """
    embeddings_path = Path(embeddings_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load embeddings.
    data = torch.load(embeddings_path, weights_only=True)
    embeddings: torch.Tensor = data["embeddings"].float()  # [N, embed_dim]

    if embeddings.shape[1] != embed_dim:
        raise ValueError(
            f"embed_dim mismatch: file has {embeddings.shape[1]}-d embeddings, "
            f"but embed_dim={embed_dim} was specified."
        )

    n_samples = embeddings.shape[0]
    logger.info(
        "Training SAE: %d samples, embed_dim=%d, n_features=%d, k=%d",
        n_samples,
        embed_dim,
        n_features,
        k,
    )

    sae = GroupSparseSAE(embed_dim=embed_dim, n_features=n_features, k=k)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    sae.train()

    for iteration in range(n_iterations):
        # Sample a random mini-batch.
        indices = torch.randint(0, n_samples, (batch_size,))
        x = embeddings[indices]  # [B, embed_dim]

        # Forward pass — encode and decode.
        z = sae(x, modality_idx=0)  # [B, n_features]
        x_hat = sae.decode(z, modality_idx=0)  # [B, embed_dim]

        # Reconstruction loss.
        loss_recon = sae.reconstruction_loss(x, x_hat)

        # Group-sparse loss: compare activations of x vs. a randomly masked copy.
        mask = (torch.rand_like(x) > mask_prob).float()
        x_masked = F.normalize(x * mask, dim=-1)
        z_masked = sae(x_masked, modality_idx=0)
        loss_gs = sae.group_sparse_loss(z, z_masked)

        loss = loss_recon + lambda_gs * loss_gs

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Keep decoder columns unit-normed.
        sae.normalize_decoder()

        if (iteration + 1) % 1000 == 0:
            logger.info(
                "iter %d/%d  loss=%.4f  recon=%.4f  gs=%.4f",
                iteration + 1,
                n_iterations,
                loss.item(),
                loss_recon.item(),
                loss_gs.item(),
            )

    sae.eval()

    # Save weights.
    weights_path = output_path / "sae_weights.pt"
    torch.save(sae.state_dict(), weights_path)
    logger.info("Saved SAE weights to %s", weights_path)

    return sae


# ---------------------------------------------------------------------------
# Stage 4: Label features
# ---------------------------------------------------------------------------


def label_features(
    sae_path: Path | str,
    encoder: Encoder,
    vocab_size: int = 15000,
) -> list[str]:
    """
    Assign a human-readable label to every SAE feature.

    Each feature is labelled with the word from a built-in vocabulary whose
    text embedding has the highest cosine similarity to that feature's decoder
    column (i.e. the direction in embedding space the feature responds to).

    The vocabulary is drawn from a fixed list of common English words.  When
    the encoder returns very similar embeddings for multiple words, some features
    may share a label; this is expected behaviour.

    Parameters
    ----------
    sae_path:
        Path to the directory containing ``sae_weights.pt`` and a
        ``sae_config.pt`` file saved by ``train_sae``, OR directly the path
        to ``sae_weights.pt`` itself.
    encoder:
        The same encoder used to collect the training embeddings.  Used here
        to embed vocabulary words.
    vocab_size:
        Maximum vocabulary size.  Words are drawn from a built-in list; if
        ``vocab_size`` exceeds the list length the full list is used.

    Returns
    -------
    list[str]
        One label per SAE feature, in feature-index order.
    """
    sae_path = Path(sae_path)

    # Allow pointing at the directory or directly at the weights file.
    if sae_path.is_dir():
        weights_file = sae_path / "sae_weights.pt"
    else:
        weights_file = sae_path

    if not weights_file.exists():
        raise FileNotFoundError(f"SAE weights file not found: {weights_file}")

    state = torch.load(weights_file, weights_only=True)

    # Recover embed_dim and n_features from W_enc shape.
    W_enc = state["W_enc"]  # [n_features, embed_dim]
    W_dec = state["W_dec"]  # [embed_dim, n_features]
    n_features = W_dec.shape[1]

    logger.info("Labelling %d features …", n_features)

    # ---------------------------------------------------------------
    # Build the vocabulary.
    # ---------------------------------------------------------------
    vocab = _get_vocab(vocab_size)

    logger.info("Encoding %d vocabulary words …", len(vocab))

    # Encode vocabulary words — accumulate [V, embed_dim].
    word_embeddings: list[torch.Tensor] = []
    for word in vocab:
        try:
            emb = encoder.encode_text(word)  # [1, D]
        except Exception:
            logger.warning("Could not encode word '%s' — skipping.", word, exc_info=True)
            word_embeddings.append(torch.zeros(1, W_dec.shape[0]))
            continue
        word_embeddings.append(emb.detach().cpu().float())

    V = torch.cat(word_embeddings, dim=0)  # [V, embed_dim]
    V = F.normalize(V, dim=-1)

    # ---------------------------------------------------------------
    # Cosine similarity between decoder columns and word embeddings.
    # decoder column j is W_dec[:, j] — the direction feature j decodes to.
    # ---------------------------------------------------------------
    # W_dec: [embed_dim, n_features]  →  columns: [n_features, embed_dim]
    decoder_cols = W_dec.T.float()  # [n_features, embed_dim]
    decoder_cols = F.normalize(decoder_cols, dim=-1)

    # Similarity matrix: [n_features, V]
    sim = decoder_cols @ V.T  # [n_features, V]

    best_word_indices = sim.argmax(dim=-1).tolist()  # [n_features]
    labels = [vocab[idx] for idx in best_word_indices]

    logger.info("Feature labelling complete.")
    return labels


# ---------------------------------------------------------------------------
# Internal vocabulary helper
# ---------------------------------------------------------------------------

# A compact but broad list of common English words covering concrete nouns,
# emotions, textures, colours, actions, and nature terms — similar to the kind
# of vocabulary used in CLIP/DINO probing studies.
_VOCAB_BASE: list[str] = [
    # colours
    "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown",
    "black", "white", "grey", "cyan", "magenta", "crimson", "turquoise",
    # animals
    "dog", "cat", "bird", "fish", "horse", "lion", "tiger", "elephant",
    "bear", "wolf", "fox", "rabbit", "deer", "eagle", "owl", "shark",
    "whale", "dolphin", "monkey", "gorilla", "zebra", "giraffe", "penguin",
    "frog", "snake", "turtle", "bee", "butterfly", "spider", "ant",
    # nature / landscape
    "ocean", "river", "lake", "mountain", "forest", "desert", "valley",
    "beach", "sunset", "sunrise", "storm", "snow", "rain", "fog", "fire",
    "cloud", "sky", "tree", "flower", "grass", "rock", "sand", "leaf",
    "wave", "waterfall", "cave", "island", "volcano",
    # textures / materials
    "wood", "metal", "glass", "stone", "fabric", "leather", "plastic",
    "silk", "velvet", "marble", "concrete", "rust", "crystal", "foam",
    # emotions / moods
    "joy", "sadness", "anger", "fear", "surprise", "disgust", "love",
    "hope", "peace", "energy", "calm", "chaos", "mystery", "nostalgia",
    # food
    "apple", "banana", "strawberry", "orange", "lemon", "grape", "bread",
    "cheese", "coffee", "tea", "chocolate", "cake", "pizza",
    # objects
    "book", "chair", "table", "door", "window", "clock", "lamp", "mirror",
    "phone", "camera", "music", "art", "painting", "sculpture",
    # places
    "city", "village", "market", "garden", "park", "bridge", "road",
    "station", "harbour", "castle", "temple", "church",
    # actions / abstract
    "running", "flying", "swimming", "dancing", "sleeping", "falling",
    "glowing", "growing", "spinning", "melting", "floating", "exploding",
    # sensory
    "loud", "quiet", "bright", "dark", "warm", "cold", "smooth", "rough",
    "sharp", "soft", "heavy", "light", "fast", "slow",
]


def _get_vocab(vocab_size: int) -> list[str]:
    """
    Return a deduplicated vocabulary list capped at *vocab_size* entries.

    If *vocab_size* exceeds the built-in list length the full list is returned.
    """
    seen: set[str] = set()
    result: list[str] = []
    for word in _VOCAB_BASE:
        if word not in seen:
            seen.add(word)
            result.append(word)
            if len(result) >= vocab_size:
                break
    return result
