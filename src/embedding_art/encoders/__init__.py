"""Multimodal encoders."""

from embedding_art.encoders.base import Encoder
from embedding_art.encoders.imagebind import ImageBindEncoder

__all__ = [
    "Encoder",
    "ImageBindEncoder",
]
