"""Sparse Autoencoder integration for interpretable embedding decomposition."""

from embedding_art.sae.feature_renderer import FeatureRenderer
from embedding_art.sae.lens import SAEDecomposition, SAELens

__all__ = ["SAELens", "SAEDecomposition", "FeatureRenderer"]
