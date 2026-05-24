"""Text renderer via nearest-neighbor search in a precomputed vocabulary."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from dataclasses import dataclass, field

from embedding_art.core.render_result import OptimizationHistory, RenderResult


@dataclass
class TextRenderResult(RenderResult):
    """RenderResult extended with matched text descriptions."""

    descriptions: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

DEFAULT_VOCABULARY = [
    # Animals
    "cat", "dog", "bird", "fish", "horse", "elephant", "lion", "tiger", "bear", "wolf",
    "eagle", "owl", "butterfly", "snake", "dolphin", "whale", "penguin", "rabbit", "deer",
    "fox", "shark", "octopus", "frog", "turtle", "bee",
    # Objects
    "car", "bicycle", "airplane", "boat", "train", "clock", "lamp", "chair", "table", "book",
    "guitar", "piano", "violin", "drum", "camera", "phone", "computer", "television", "mirror",
    "umbrella", "key", "candle", "bottle", "cup", "hat",
    # Nature
    "mountain", "ocean", "river", "forest", "desert", "waterfall", "volcano", "glacier",
    "meadow", "canyon", "island", "lake", "cave", "sunset", "sunrise", "rainbow", "storm",
    "lightning", "snowflake", "coral reef",
    # Food
    "apple", "banana", "orange", "strawberry", "pizza", "bread", "cake", "chocolate",
    "coffee", "wine", "cheese", "sushi", "pasta", "ice cream", "honey",
    # Colors and visual
    "red", "blue", "green", "yellow", "purple", "magenta", "pink", "black", "white", "gold",
    "silver", "bronze", "turquoise", "crimson", "indigo",
    # Textures and materials
    "wood", "metal", "glass", "stone", "silk", "velvet", "leather", "marble", "crystal",
    "sand", "moss", "rust", "smoke", "fog", "ice",
    # Architecture
    "castle", "bridge", "tower", "cathedral", "lighthouse", "pyramid", "skyscraper",
    "temple", "cottage", "palace",
    # Emotions and abstract
    "joy", "sadness", "anger", "fear", "love", "hope", "peace", "chaos", "freedom",
    "solitude", "nostalgia", "wonder", "mystery", "serenity", "melancholy",
    # Abstract concepts
    "time", "space", "infinity", "gravity", "energy", "light", "shadow", "dream",
    "memory", "silence", "music", "rhythm", "harmony", "balance", "symmetry",
    # Actions and scenes
    "dancing", "flying", "swimming", "running", "sleeping", "singing", "painting",
    "reading", "cooking", "exploring",
    # Celestial
    "sun", "moon", "star", "galaxy", "nebula", "comet", "planet", "aurora", "eclipse",
    "constellation",
    # Patterns
    "spiral", "fractal", "wave", "circle", "triangle", "grid", "maze", "mosaic",
    "kaleidoscope", "mandala",
    # Seasons and weather
    "spring", "summer", "autumn", "winter", "rain", "snow", "wind", "cloud", "thunder",
    "mist",
    # Plants
    "flower", "tree", "rose", "sunflower", "lotus", "bamboo", "cactus", "fern", "mushroom",
    "vine",
]


class TextRenderer:
    """Decodes embeddings to text descriptions via nearest-neighbor search."""

    def __init__(
        self,
        encoder: Any,
        vocabulary: list[str] | None = None,
        device: str = "cpu",
    ) -> None:
        self._encoder = encoder
        self._device = torch.device(device)
        self._vocab = vocabulary if vocabulary is not None else DEFAULT_VOCABULARY
        self._text_embeddings: torch.Tensor | None = None

    @property
    def output_modality(self) -> str:
        return "text"

    def build_index(self) -> None:
        embeddings = []
        for word in self._vocab:
            emb = self._encoder.encode_text(word)
            embeddings.append(F.normalize(emb, dim=-1))
        self._text_embeddings = torch.cat(embeddings, dim=0).to(self._device)

    def render(self, embedding: torch.Tensor, k: int = 5, **kwargs: Any) -> RenderResult:
        if self._text_embeddings is None:
            self.build_index()
        embedding = F.normalize(embedding.to(self._device), dim=-1)
        if embedding.dim() == 2:
            embedding = embedding.squeeze(0)
        similarities = torch.mv(self._text_embeddings, embedding)
        top_k = torch.topk(similarities, min(k, len(self._vocab)))
        top_indices = top_k.indices.tolist()
        top_scores = top_k.values
        descriptions = [self._vocab[i] for i in top_indices]
        history = OptimizationHistory()
        return TextRenderResult(
            output=top_scores.unsqueeze(0),
            history=history,
            encoder_name="text_nn",
            final_similarity=float(top_scores[0]),
            config=None,
            checkpoints=None,
            descriptions=descriptions,
            scores=[float(s) for s in top_scores],
        )
