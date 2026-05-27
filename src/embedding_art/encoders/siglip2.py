"""
SigLIP 2 encoder for vision-language embedding.

Wraps ``google/siglip2-so400m-patch14-384`` from HuggingFace transformers and
implements the v2 Encoder duck-typed interface:

* ``card`` — class-level ``EncoderCard`` with capability flags.
* ``encode_text`` / ``encode_image`` — standard modality entry-points.
* ``encode_for_optimization`` — differentiable image path for gradient-descent loops.
* ``get_layer_features`` — every-4th-layer intermediate activations for multi-layer loss.
* ``encode(ConceptSpec)`` — single-method dispatch.
* ``unload`` — release weights and clear device cache.

Requires: ``pip install transformers``
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from PIL import Image

from embedding_art.encoders.features import LayerFeatures
from embedding_art.encoders.registry import EncoderCapability, EncoderCard
from embedding_art.exceptions import EncoderError, ModelLoadError

if TYPE_CHECKING:
    from embedding_art.core.concept import Concept
    from embedding_art.core.concept_spec import ConceptSpec

logger = logging.getLogger(__name__)

MODEL_NAME = "google/siglip2-so400m-patch14-384"
EMBEDDING_DIM = 1152

# SigLIP 2 uses standard ImageNet normalisation statistics.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Native input resolution expected by this checkpoint.
_INPUT_SIZE = 384


class SigLIP2Encoder:
    """SigLIP 2 So400m encoder for text and image embedding.

    Implements the v2 duck-typed encoder interface.  The ``card`` class attribute
    is intentionally placed on the class (not a property) so that the registry can
    read it via ``cls.card`` before any instantiation.

    Multi-layer feature extraction is supported via ``get_layer_features``, which
    returns every 4th hidden state from the vision transformer (including the final
    layer) as :class:`~embedding_art.encoders.features.LayerFeatures` objects.

    Args:
        device: PyTorch device string (e.g. ``"mps"``, ``"cpu"``, ``"cuda"``).
        model_name: HuggingFace model identifier.  Override only for testing.
    """

    # Class-level card — readable on the class before instantiation.
    # A property would only work on instances, causing AttributeError in the registry.
    card: EncoderCard = EncoderCard(
        name="siglip2-so400m",
        capabilities=(
            EncoderCapability.TEXT
            | EncoderCapability.IMAGE
            | EncoderCapability.BACKPROP_OPTIMIZABLE
            | EncoderCapability.MULTI_LAYER_FEATURES
        ),
        embedding_dim=EMBEDDING_DIM,
        memory_estimate_mb=1600,
        backprop_cost=1.0,
    )

    def __init__(self, device: str = "mps", model_name: str = MODEL_NAME) -> None:
        self._device = torch.device(device)
        self._model_name = model_name
        self._model: Any = None
        self._processor: Any = None
        self._tokenizer: Any = None
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            from transformers import AutoModel, AutoProcessor, AutoTokenizer

            logger.info("Loading SigLIP 2 model: %s", self._model_name)
            self._model = (
                AutoModel.from_pretrained(self._model_name, trust_remote_code=True)
                .to(self._device)
                .eval()
            )
            self._processor = AutoProcessor.from_pretrained(self._model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            logger.info("SigLIP 2 loaded successfully")
        except Exception as e:
            raise ModelLoadError("siglip2-so400m", e) from e

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        return EMBEDDING_DIM

    @property
    def device(self) -> torch.device:
        return self._device

    # ------------------------------------------------------------------
    # Modality encoders
    # ------------------------------------------------------------------

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to a unit-normalised [1, 1152] embedding.

        Args:
            text: Input string.

        Returns:
            Float tensor of shape ``[1, 1152]``.

        Raises:
            EncoderError: On any tokenisation or forward-pass failure.
        """
        try:
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self._device)
            with torch.no_grad():
                text_features = self._model.get_text_features(**inputs)
            return F.normalize(text_features, dim=-1)
        except Exception as e:
            raise EncoderError("text", e) from e

    def encode_image(self, image: Path | Image.Image | torch.Tensor) -> torch.Tensor:
        """Encode an image to a unit-normalised [1, 1152] embedding.

        Accepts a PIL ``Image``, a filesystem ``Path``, or a ``[B, 3, H, W]``
        tensor.  Tensor inputs are routed through :meth:`encode_for_optimization`
        so that the caller can choose whether to retain gradients.

        Args:
            image: PIL image, path to an image file, or float tensor in [0, 1].

        Returns:
            Float tensor of shape ``[1, 1152]``.

        Raises:
            EncoderError: On any preprocessing or forward-pass failure.
        """
        try:
            if isinstance(image, torch.Tensor):
                return self.encode_for_optimization(image)

            if isinstance(image, (str, Path)):
                image = Image.open(image).convert("RGB")

            inputs = self._processor(images=image, return_tensors="pt").to(self._device)
            with torch.no_grad():
                image_features = self._model.get_image_features(**inputs)
            return F.normalize(image_features, dim=-1)
        except Exception as e:
            raise EncoderError("image", e) from e

    def encode_audio(
        self,
        audio: Path | torch.Tensor,
        start: float = 0.0,
        duration: float = 2.0,
    ) -> torch.Tensor:
        """Not supported by SigLIP 2.

        Raises:
            EncoderError: Always, with an explanatory message.
        """
        raise EncoderError("audio", NotImplementedError("SigLIP 2 does not support audio"))

    def encode_video(
        self,
        video: Path | torch.Tensor,
        timestamp: float = 0.0,
    ) -> torch.Tensor:
        """Not supported by SigLIP 2.

        Raises:
            EncoderError: Always, with an explanatory message.
        """
        raise EncoderError("video", NotImplementedError("SigLIP 2 does not support video"))

    # ------------------------------------------------------------------
    # v2 duck-typed interface
    # ------------------------------------------------------------------

    def encode_for_optimization(self, tensor: torch.Tensor) -> torch.Tensor:
        """Differentiable image encoding for gradient-descent optimisation loops.

        Gradients flow through this method — do *not* wrap the call in
        ``torch.no_grad()`` if you need them.

        The input is ImageNet-normalised, resized to 384×384 if necessary, and
        forwarded through the vision encoder.  If the model exposes a
        ``visual_projection`` head it is applied before normalisation.

        Args:
            tensor: ``[B, 3, H, W]`` float tensor with pixel values in [0, 1].

        Returns:
            ``[B, 1152]`` unit-normalised float tensor.
        """
        mean = torch.tensor(_IMAGENET_MEAN, device=tensor.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=tensor.device).view(1, 3, 1, 1)
        normalized = (tensor - mean) / std

        if normalized.shape[-2:] != (_INPUT_SIZE, _INPUT_SIZE):
            normalized = F.interpolate(
                normalized,
                size=(_INPUT_SIZE, _INPUT_SIZE),
                mode="bilinear",
                align_corners=False,
            )

        vision_outputs = self._model.vision_model(
            pixel_values=normalized,
            output_hidden_states=False,
        )

        pooled = vision_outputs.pooler_output
        if pooled is None:
            pooled = vision_outputs.last_hidden_state.mean(dim=1)

        if hasattr(self._model, "visual_projection"):
            pooled = self._model.visual_projection(pooled)

        return F.normalize(pooled, dim=-1)

    def get_layer_features(self, tensor: torch.Tensor) -> dict[int, LayerFeatures]:
        """Extract intermediate vision transformer activations.

        Returns every 4th hidden state (0, 4, 8, …) and always appends the
        final layer regardless of whether it falls on a 4-step boundary.  This
        gives a compact but representative cross-layer view for multi-layer loss
        computation while avoiding redundant computation.

        Args:
            tensor: ``[B, 3, H, W]`` or ``[3, H, W]`` float tensor in [0, 1].

        Returns:
            Mapping from layer index → :class:`~embedding_art.encoders.features.LayerFeatures`.
            Each tensor has shape ``[B, N_patches, D]``.
        """
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        mean = torch.tensor(_IMAGENET_MEAN, device=tensor.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=tensor.device).view(1, 3, 1, 1)
        normalized = (tensor - mean) / std

        if normalized.shape[-2:] != (_INPUT_SIZE, _INPUT_SIZE):
            normalized = F.interpolate(
                normalized,
                size=(_INPUT_SIZE, _INPUT_SIZE),
                mode="bilinear",
                align_corners=False,
            )

        # Use forward hooks to capture hidden states because some transformers
        # versions don't wire output_hidden_states through SiglipEncoder properly.
        captured_states: list[torch.Tensor] = []
        hooks = []
        for layer in self._model.vision_model.encoder.layers:

            def _hook(module, input, output, states=captured_states):
                out = output[0] if isinstance(output, tuple) else output
                states.append(out)

            hooks.append(layer.register_forward_hook(_hook))

        try:
            self._model.vision_model(pixel_values=normalized)
        finally:
            for h in hooks:
                h.remove()

        hidden_states = captured_states
        n_layers = len(hidden_states)

        # Every 4th layer, always including the final one.
        layer_indices = list(range(0, n_layers, 4))
        if (n_layers - 1) not in layer_indices:
            layer_indices.append(n_layers - 1)

        result: dict[int, LayerFeatures] = {}
        for idx in layer_indices:
            result[idx] = LayerFeatures(
                tensor=hidden_states[idx],
                spatial=True,
                shape_semantic="batch_tokens_dim",
                layer_name=f"layer_{idx}",
            )

        return result

    def encode(self, spec: ConceptSpec) -> Concept:
        """Dispatch a :class:`~embedding_art.core.concept_spec.ConceptSpec` to the right encoder.

        Supports ``text`` and ``image`` modalities.  When both are present the
        embeddings are averaged and re-normalised.

        Args:
            spec: Concept specification.

        Returns:
            A :class:`~embedding_art.core.concept.Concept` whose embedding is
            derived from the chosen modality (or modality combination).

        Raises:
            ValueError: If no supported modality (text or image) is set on *spec*.
        """
        from embedding_art.core.concept import Concept

        if spec.text is not None and spec.image is not None:
            # Combined text + image: average embeddings in the shared space.
            text_emb = self.encode_text(spec.text)
            img_emb = self.encode_image(spec.image)
            embedding = F.normalize(text_emb + img_emb, dim=-1)
            source_input = self._load_source_tensor(spec.image)
            desc = f'text:"{spec.text}" + image:{Path(spec.image).name}'
            return Concept(embedding=embedding, description=desc, source_input=source_input)

        if spec.text is not None:
            embedding = self.encode_text(spec.text)
            return Concept(embedding=embedding, description=f'text:"{spec.text}"')

        if spec.image is not None:
            embedding = self.encode_image(spec.image)
            source_input = self._load_source_tensor(spec.image)
            return Concept(
                embedding=embedding,
                description=f"image:{Path(spec.image).name}",
                source_input=source_input,
            )

        raise ValueError(
            "ConceptSpec has no supported modality for SigLIP 2 — "
            "at least one of text or image must be provided"
        )

    def unload(self) -> None:
        """Release model weights from memory.

        Sets ``_model``, ``_processor``, and ``_tokenizer`` to ``None``, then
        triggers garbage collection and clears the device cache (MPS or CUDA).
        """
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        gc.collect()
        if self._device.type == "mps":
            torch.mps.empty_cache()
        elif self._device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_source_tensor(self, image: Path | Any) -> torch.Tensor | None:
        """Load an image file as a ``[1, 3, H, W]`` float tensor, or return None."""
        try:
            import torchvision.transforms.functional as TF

            pil_img = Image.open(image).convert("RGB")
            return TF.to_tensor(pil_img).unsqueeze(0)
        except Exception:
            return None
