"""Concept algebra: do arithmetic on concepts in the shared embedding space.

LanguageBind binds text, image, audio, and video into one 768-d space, so you
can add, subtract, scale, and interpolate concepts regardless of modality. This
script only *encodes* concepts (no generation), so it runs in seconds once the
encoder weights are cached.

Run from the repo root:
    python examples/02_concept_algebra.py
"""

from __future__ import annotations

import torch

from embedding_art import Concept
from embedding_art.encoders.languagebind import LanguageBindEncoder

device = "mps" if torch.backends.mps.is_available() else "cpu"
encoder = LanguageBindEncoder(device=device)


def sim(a: Concept, b: Concept) -> float:
    """Cosine similarity between two (unit-norm) concept embeddings."""
    return torch.cosine_similarity(a.embedding, b.embedding, dim=-1).item()


# --- Addition: blend two concepts ------------------------------------------
fire = Concept.from_text("fire", encoder)
water = Concept.from_text("water", encoder)
fire_water = fire + water
print(f"fire+water  vs fire : {sim(fire_water, fire):.3f}")
print(f"fire+water  vs water: {sim(fire_water, water):.3f}")

# --- Weighted combination --------------------------------------------------
sunset_ocean = 0.3 * Concept.from_text("sunset", encoder) + 0.7 * Concept.from_text(
    "ocean", encoder
)
print(f"0.3*sunset + 0.7*ocean leans ocean: {sim(sunset_ocean, water):.3f}")

# --- Subtraction: remove an attribute --------------------------------------
dog = Concept.from_text("dog", encoder)
hairless_dog = dog - 0.3 * Concept.from_text("fur", encoder)
print(f"dog - 0.3*fur still dog-ish: {sim(hairless_dog, dog):.3f}")

# --- Spherical interpolation: walk between two concepts --------------------
midpoint = Concept.slerp(fire, water, t=0.5)
print(f"slerp(fire, water, 0.5) midpoint sim to each: "
      f"{sim(midpoint, fire):.3f} / {sim(midpoint, water):.3f}")

# Any of these composed Concepts can be handed straight to
# `EmbeddingArtEngine.render(...)` to turn the coordinate into art — see the
# showcase example for the rendering side.
