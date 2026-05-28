from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from embedding_art.core.concept import Concept
from embedding_art.core.concept_spec import ConceptSpec


@dataclass
class ModelState:
    """Captured internal state of a model at a point in time."""

    layer_activations: dict[int, torch.Tensor]
    final_embedding: torch.Tensor
    encoder_name: str
    input_description: str
    layer_names: dict[int, str] | None = None


class ActivationProbe:
    """Captures intermediate activations from encoder forward passes."""

    def _extract_layer_activations(
        self, layer_features: dict[int, Any]
    ) -> tuple[dict[int, torch.Tensor], dict[int, str]]:
        """Extract mean-pooled activations and layer names from LayerFeatures."""
        activations: dict[int, torch.Tensor] = {}
        names: dict[int, str] = {}
        for idx, feat in layer_features.items():
            tensor = feat.tensor
            if tensor.dim() == 4:
                pooled = tensor.mean(dim=(2, 3))
            elif tensor.dim() == 3:
                pooled = tensor.mean(dim=1)
            else:
                pooled = tensor.flatten(start_dim=1) if tensor.dim() > 1 else tensor.unsqueeze(0)
            activations[idx] = pooled
            names[idx] = feat.layer_name
        return activations, names

    def capture_from_concept(self, encoder: Any, concept: Concept) -> ModelState:
        """Capture model state from an already-encoded concept."""
        encoder_name = encoder.card.name if hasattr(encoder, "card") else "unknown"
        if concept.source_input is not None and hasattr(encoder, "get_layer_features"):
            layer_features = encoder.get_layer_features(concept.source_input)
            activations, names = self._extract_layer_activations(layer_features)
            return ModelState(
                layer_activations=activations,
                final_embedding=concept.embedding,
                encoder_name=encoder_name,
                input_description=concept.description,
                layer_names=names,
            )
        return ModelState(
            layer_activations={},
            final_embedding=concept.embedding,
            encoder_name=encoder_name,
            input_description=concept.description,
        )

    def capture_from_spec(self, encoder: Any, spec: ConceptSpec) -> ModelState:
        """Encode a spec and capture all intermediate states."""
        concept = encoder.encode(spec)
        return self.capture_from_concept(encoder, concept)

    def capture_from_tensor(
        self, encoder: Any, tensor: torch.Tensor, description: str = ""
    ) -> ModelState:
        """Capture states from a raw input tensor."""
        encoder_name = encoder.card.name if hasattr(encoder, "card") else "unknown"
        if hasattr(encoder, "get_layer_features"):
            layer_features = encoder.get_layer_features(tensor)
            activations, names = self._extract_layer_activations(layer_features)
            embedding = encoder.encode_for_optimization(tensor)
            return ModelState(
                layer_activations=activations,
                final_embedding=embedding,
                encoder_name=encoder_name,
                input_description=description,
                layer_names=names,
            )
        embedding = encoder.encode_for_optimization(tensor)
        return ModelState(
            layer_activations={},
            final_embedding=embedding,
            encoder_name=encoder_name,
            input_description=description,
        )
