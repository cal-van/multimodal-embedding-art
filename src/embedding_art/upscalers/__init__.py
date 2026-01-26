"""
Image upscalers for embedding-art.

Provides upscaling functionality to increase output resolution beyond
the native generator resolution. Useful for producing 2048x2048+ outputs
from 1024x1024 generated images.
"""

from embedding_art.upscalers.realesrgan import RealESRGANUpscaler

__all__ = [
    "RealESRGANUpscaler",
]
