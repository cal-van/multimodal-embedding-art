"""
Visualization utilities for embedding-art.

Provides functions for visualizing embeddings, optimization progress,
and similarity relationships between concepts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from embedding_art.core.concept import Concept


def plot_similarity_history(
    similarity_history: list[float],
    loss_history: list[float] | None = None,
    title: str | None = None,
) -> Figure:
    """
    Plot similarity values over optimization steps.

    Args:
        similarity_history: List of cosine similarity values per step.
        loss_history: Optional list of loss values to plot on secondary axis.
        title: Optional title for the plot.

    Returns:
        Matplotlib Figure object.

    Raises:
        ValueError: If similarity_history is empty.
    """
    if not similarity_history:
        raise ValueError("similarity_history cannot be empty")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    steps = range(len(similarity_history))
    ax1.plot(steps, similarity_history, color="tab:blue", label="Similarity", linewidth=2)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Cosine Similarity", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, alpha=0.3)

    if loss_history is not None:
        ax2 = ax1.twinx()
        ax2.plot(steps, loss_history, color="tab:red", label="Loss", linewidth=2, linestyle="--")
        ax2.set_ylabel("Loss", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    else:
        ax1.legend(loc="lower right")

    if title:
        ax1.set_title(title)

    fig.tight_layout()
    return fig


def compute_similarity_matrix(concepts: list[Concept]) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix for a list of concepts.

    Args:
        concepts: List of Concept objects.

    Returns:
        NumPy array of shape (n, n) containing pairwise similarities.
    """
    n = len(concepts)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            matrix[i, j] = concepts[i].similarity(concepts[j])

    return matrix


def plot_embeddings_2d(
    concepts: list[Concept],
    method: str = "tsne",
    title: str | None = None,
) -> Figure:
    """
    Create a 2D projection of concept embeddings.

    Args:
        concepts: List of Concept objects to visualize.
        method: Dimensionality reduction method ("tsne" or "pca").
        title: Optional title for the plot.

    Returns:
        Matplotlib Figure object.

    Raises:
        ValueError: If fewer than 2 concepts provided.
        ValueError: If unknown method specified.
    """
    if len(concepts) < 2:
        raise ValueError("Need at least 2 concepts for 2D projection")

    # Stack embeddings into a matrix
    embeddings = torch.cat([c.embedding for c in concepts], dim=0).cpu().numpy()

    # Reduce dimensionality
    if method == "tsne":
        from sklearn.manifold import TSNE

        # Perplexity must be less than n_samples
        perplexity = min(30, len(concepts) - 1)
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        coords_2d = reducer.fit_transform(embeddings)
    elif method == "pca":
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2, random_state=42)
        coords_2d = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'tsne' or 'pca'.")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot points
    ax.scatter(coords_2d[:, 0], coords_2d[:, 1], s=100, alpha=0.7)

    # Add labels
    for i, concept in enumerate(concepts):
        ax.annotate(
            concept.description,
            (coords_2d[i, 0], coords_2d[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
        )

    ax.set_xlabel(f"{method.upper()} Component 1")
    ax.set_ylabel(f"{method.upper()} Component 2")

    if title:
        ax.set_title(title)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


def plot_distance_matrix(
    concepts: list[Concept],
    title: str | None = None,
) -> Figure:
    """
    Create a heatmap of pairwise cosine similarities between concepts.

    Args:
        concepts: List of Concept objects.
        title: Optional title for the plot.

    Returns:
        Matplotlib Figure object.

    Raises:
        ValueError: If fewer than 2 concepts provided.
    """
    if len(concepts) < 2:
        raise ValueError("Need at least 2 concepts for distance matrix")

    # Compute similarity matrix
    matrix = compute_similarity_matrix(concepts)
    labels = [c.description for c in concepts]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create heatmap
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-1, vmax=1)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Cosine Similarity")

    # Configure ticks and labels
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # Add value annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            text_color = "white" if abs(matrix[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color=text_color)

    if title:
        ax.set_title(title)

    fig.tight_layout()
    return fig
