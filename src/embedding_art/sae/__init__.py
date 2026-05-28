"""Sparse Autoencoder integration for interpretable embedding decomposition."""

from embedding_art.sae.feature_renderer import FeatureRenderer
from embedding_art.sae.lens import SAEDecomposition, SAELens
from embedding_art.sae.matryoshka import MatryoshkaSAE, train_matryoshka_sae

__all__ = [
    "FeatureRenderer",
    "MatryoshkaSAE",
    "SAEDecomposition",
    "SAELens",
    "train_matryoshka_sae",
]
