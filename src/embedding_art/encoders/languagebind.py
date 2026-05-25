"""
LanguageBind encoder — canonical multimodal encoder for the v3 aggressive-rewrite.

LanguageBind (ICLR 2024, PKU-YuanGroup) is the ImageBind successor that binds
N modalities (image, audio, video, depth, thermal, text) into a single shared
embedding space anchored on language. Unlike SigLIP 2 (text + image only) and
CLAP (text + audio only), LanguageBind is *truly* multimodal — a single point
in its embedding space can be reached from any modality, and the four-modality
showcase (image / audio / video / text rendered from the same target) is
structurally possible.

This wrapper exposes the v2 ``Encoder`` duck-typed interface:

* ``card`` — class-level ``EncoderCard`` with ALL FOUR modality capabilities.
* ``encode_text`` / ``encode_image`` / ``encode_audio`` / ``encode_video`` —
  modality entry-points; each lazy-loads only the per-modality CLIP-style ViT
  needed to satisfy the call.
* ``encode_for_optimization`` — differentiable image path for gradient-descent
  loops (extends naturally to audio/video via ``encode_for_optimization_audio``,
  ``encode_for_optimization_video``).
* ``encode_text_for_optimization`` — differentiable text path used by the
  text-anchor trick (project current embedding back to text-space during
  rendering).
* ``get_layer_features`` — multi-layer ViT activations for MIMIC-style
  multi-layer alignment loss.
* ``encode(ConceptSpec)`` — single-method dispatch.
* ``unload`` — release all loaded modality weights.

Architecture
------------
LanguageBind ships per-modality HuggingFace checkpoints
(``LanguageBind/LanguageBind_Image``, ``LanguageBind/LanguageBind_Video_FT``,
``LanguageBind/LanguageBind_Audio_FT``). Each checkpoint contains:

* A CLIP-style vision/audio/video ViT (``vision_model``).
* A linear projection head (``visual_projection``) to the shared space.
* A CLIP-style text encoder (``text_model``) and projection (``text_projection``).
* A logit-scale parameter (``logit_scale``).

All checkpoints share the SAME text encoder architecture and weights, so we
load text from whichever modality model is loaded first. Modality models are
lazy-loaded — calling ``encode_image`` only loads the image checkpoint.

Embedding space
---------------
LanguageBind is built on OpenCLIP ViT-L/14 → 768-d shared embedding space.

Installation
------------
LanguageBind is distributed as a research codebase without ``setup.py``; the
``languagebind`` Python package must be installed from source::

    git clone https://github.com/PKU-YuanGroup/LanguageBind ../LanguageBind
    cd ../LanguageBind && pip install -e . && cd ..

(See ``CLAUDE.md`` for the canonical setup procedure; we install LanguageBind
the same way ImageBind is installed.)

If the ``languagebind`` package is not importable this encoder raises
``ModelLoadError`` with a helpful message; it does NOT silently fall back to a
plain HF CLIP load, because the LoRA / PEFT adapters baked into LanguageBind
checkpoints are essential to the shared-space alignment property.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image

from embedding_art.encoders.features import LayerFeatures
from embedding_art.encoders.registry import EncoderCapability, EncoderCard
from embedding_art.exceptions import EncoderError, ModelLoadError

if TYPE_CHECKING:
    from embedding_art.core.concept import Concept
    from embedding_art.core.concept_spec import ConceptSpec

logger = logging.getLogger(__name__)

# Map our modality names → LanguageBind HuggingFace checkpoint identifiers.
_MODALITY_CHECKPOINTS: dict[str, str] = {
    "image": "LanguageBind/LanguageBind_Image",
    "video": "LanguageBind/LanguageBind_Video_FT",
    "audio": "LanguageBind/LanguageBind_Audio_FT",
}

# Shared embedding dimension across all LanguageBind modality checkpoints
# (built on OpenCLIP ViT-L/14 → 768-d projection head).
EMBEDDING_DIM = 768

# ImageNet normalisation statistics — LanguageBind image/video use these.
_IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
_IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)

# Native input resolution for image/video frames.
_INPUT_SIZE = 224

# LanguageBind tokenizer max length (matches upstream inference.py).
_TOKEN_MAX_LENGTH = 77

# Modality names accepted by the modality kwarg of ``_load_modality``.
ModalityName = Literal["image", "video", "audio"]


class LanguageBindEncoder:
    """LanguageBind canonical multimodal encoder.

    Implements the v2 duck-typed encoder interface and serves as the *canonical*
    encoder for the aggressive-rewrite architecture — all four-modality
    showcase targets live in this encoder's 768-d shared space.

    Per-modality LanguageBind checkpoints are lazy-loaded: calling
    ``encode_image`` for the first time triggers a download of the
    LanguageBind_Image checkpoint (~1.2GB), then subsequent calls reuse the
    cached model. Calling ``encode_audio`` separately triggers the
    LanguageBind_Audio_FT checkpoint (~700MB), and so on.

    The text encoder is shared across all modality checkpoints; the first
    loaded modality also supplies the text encoder.

    Args:
        device: PyTorch device string (e.g. ``"mps"``, ``"cpu"``, ``"cuda"``).
        cache_dir: Where to cache LanguageBind weights. Defaults to
            ``~/.cache/embedding_art/languagebind``.
        eager_load: Modalities to eagerly load at construction time. Default
            ``("image",)`` loads only the image checkpoint up-front; pass
            ``("image", "audio", "video")`` to load all up-front (uses ~3GB).
    """

    card: EncoderCard = EncoderCard(
        name="languagebind",
        capabilities=(
            EncoderCapability.TEXT
            | EncoderCapability.IMAGE
            | EncoderCapability.AUDIO
            | EncoderCapability.VIDEO
            | EncoderCapability.BACKPROP_OPTIMIZABLE
            | EncoderCapability.MULTI_LAYER_FEATURES
        ),
        embedding_dim=EMBEDDING_DIM,
        memory_estimate_mb=3500,  # ~1.2 (image) + ~1.5 (video) + ~0.7 (audio)
        backprop_cost=1.0,
    )

    def __init__(
        self,
        device: str = "mps",
        cache_dir: str | Path | None = None,
        eager_load: tuple[str, ...] = ("image",),
        enable_sdpa: bool | None = None,
    ) -> None:
        self._device = torch.device(device)
        self._cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else Path.home() / ".cache" / "embedding_art" / "languagebind"
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded per-modality models keyed by modality name.
        # Each value is a dict with keys 'model', 'processor', 'tokenizer'
        # (tokenizer is shared via the first loaded modality).
        self._modality_models: dict[str, dict[str, Any]] = {}
        self._tokenizer: Any = None

        # SDPA defaults: enabled when running on MPS or CUDA (where the
        # flash-style kernel lives); off on CPU to avoid pointless
        # rewrap. Passing an explicit bool overrides.
        if enable_sdpa is None:
            enable_sdpa = self._device.type in {"mps", "cuda"}
        self._enable_sdpa = enable_sdpa

        for modality in eager_load:
            self._load_modality(modality)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_modality(self, modality: ModalityName) -> None:
        """Load a single modality's LanguageBind checkpoint.

        Idempotent — calling twice with the same modality is a no-op after the
        first call. The text encoder is sourced from the first-loaded modality.

        Raises:
            ModelLoadError: If the ``languagebind`` package isn't installed
                or the checkpoint fails to download/load.
        """
        if modality in self._modality_models:
            return
        if modality not in _MODALITY_CHECKPOINTS:
            raise ModelLoadError(
                f"languagebind-{modality}",
                ValueError(
                    f"Unknown modality {modality!r}; supported: " f"{sorted(_MODALITY_CHECKPOINTS)}"
                ),
            )

        checkpoint = _MODALITY_CHECKPOINTS[modality]
        try:
            from languagebind import (  # type: ignore[import-not-found]
                LanguageBindAudio,
                LanguageBindAudioProcessor,
                LanguageBindImage,
                LanguageBindImageProcessor,
                LanguageBindImageTokenizer,
                LanguageBindVideo,
                LanguageBindVideoProcessor,
            )
        except ImportError as e:
            raise ModelLoadError(
                "languagebind",
                ImportError(
                    f"LanguageBind import failed ({e}).\n"
                    "\n"
                    "LanguageBind is not installed. It's NOT on PyPI \u2014 distributed as a\n"
                    "research codebase without setup.py. Two-step install:\n"
                    "\n"
                    "  1. Clone and add to PYTHONPATH (from this repo's parent dir):\n"
                    "       git clone https://github.com/PKU-YuanGroup/LanguageBind\n"
                    '       export PYTHONPATH="$PYTHONPATH:$(pwd)/LanguageBind"\n'
                    "     Add that export to ~/.zshrc (macOS) or ~/.bashrc so it\n"
                    "     persists across shells.\n"
                    "\n"
                    "  2. Install its transitive Python deps via this repo's extras:\n"
                    '       pip install -e ".[languagebind]"        # Linux / Windows\n'
                    '       pip install -e ".[languagebind-macos]"  # Apple Silicon\n'
                    "     (macOS arm64 uses eva-decord because upstream decord has\n"
                    "     no prebuilt arm64 wheels.)\n"
                    "\n"
                    "See README.md / CLAUDE.md for the full guide."
                ),
            ) from e

        model_cls = {
            "image": LanguageBindImage,
            "video": LanguageBindVideo,
            "audio": LanguageBindAudio,
        }[modality]
        processor_cls = {
            "image": LanguageBindImageProcessor,
            "video": LanguageBindVideoProcessor,
            "audio": LanguageBindAudioProcessor,
        }[modality]

        try:
            logger.info("Loading LanguageBind-%s from %s", modality, checkpoint)
            model = (
                model_cls.from_pretrained(checkpoint, cache_dir=str(self._cache_dir))
                .to(self._device)
                .eval()
            )
            processor = processor_cls(model.config)
            self._modality_models[modality] = {
                "model": model,
                "processor": processor,
            }
            if self._enable_sdpa:
                n_patched = self._patch_clip_attention(model)
                if n_patched:
                    logger.info(
                        "LanguageBind-%s: routed %d attention blocks to SDPA",
                        modality,
                        n_patched,
                    )

            # Source the text tokenizer from the first-loaded modality —
            # tokenizers are identical across modalities (all built on CLIP).
            if self._tokenizer is None:
                self._tokenizer = LanguageBindImageTokenizer.from_pretrained(
                    "LanguageBind/LanguageBind_Image",
                    cache_dir=str(self._cache_dir / "tokenizer"),
                )
            logger.info("LanguageBind-%s loaded successfully", modality)
        except Exception as e:
            raise ModelLoadError(f"languagebind-{modality}", e) from e

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
        """Encode text to a unit-normalised ``[1, 768]`` embedding.

        The text encoder is sourced from the first loaded modality (default
        ``image``); a modality is loaded automatically if none has been loaded
        yet.

        Args:
            text: Input string (truncated to 77 tokens).

        Returns:
            Float tensor of shape ``[1, 768]``.

        Raises:
            EncoderError: On any tokenisation or forward-pass failure.
        """
        if not self._modality_models:
            self._load_modality("image")
        modality_key = next(iter(self._modality_models))
        model = self._modality_models[modality_key]["model"]
        try:
            inputs = self._tokenizer(
                text,
                max_length=_TOKEN_MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                text_outputs = model.text_model(**inputs)
                pooled = (
                    text_outputs[1]
                    if isinstance(text_outputs, tuple)
                    else text_outputs.pooler_output
                )
                projected = model.text_projection(pooled)
            return F.normalize(projected, dim=-1)
        except Exception as e:
            raise EncoderError("text", e) from e

    def encode_text_batch(self, texts: list[str], batch_size: int = 128) -> torch.Tensor:
        """Batch-encode a list of texts to a ``[N, 768]`` unit-normalised tensor.

        Materially faster than calling :meth:`encode_text` in a Python loop:
        the tokenizer batches and the transformer forward sees the whole
        batch at once. For the 1500-word text-anchor vocabulary this is the
        difference between ~30 s and ~1 s on M1 Max.

        Args:
            texts: Input strings; each truncated to 77 tokens.
            batch_size: Forward-pass chunk size. ``128`` keeps memory bounded
                even on M1 Max without sacrificing throughput. Set lower if
                memory pressure surfaces.

        Returns:
            Float tensor of shape ``[len(texts), 768]``.

        Raises:
            EncoderError: On any tokenisation or forward-pass failure.
        """
        if not texts:
            return torch.empty(0, EMBEDDING_DIM, device=self._device)

        if not self._modality_models:
            self._load_modality("image")
        modality_key = next(iter(self._modality_models))
        model = self._modality_models[modality_key]["model"]

        outputs: list[torch.Tensor] = []
        try:
            for start in range(0, len(texts), batch_size):
                chunk = texts[start : start + batch_size]
                inputs = self._tokenizer(
                    chunk,
                    max_length=_TOKEN_MAX_LENGTH,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).to(self._device)
                with torch.no_grad():
                    text_outputs = model.text_model(**inputs)
                    pooled = (
                        text_outputs[1]
                        if isinstance(text_outputs, tuple)
                        else text_outputs.pooler_output
                    )
                    projected = model.text_projection(pooled)
                outputs.append(F.normalize(projected, dim=-1))
        except Exception as e:
            raise EncoderError("text-batch", e) from e
        return torch.cat(outputs, dim=0)

    def encode_image(self, image: Path | Image.Image | torch.Tensor) -> torch.Tensor:
        """Encode an image to a unit-normalised ``[1, 768]`` embedding.

        Args:
            image: PIL image, filesystem path, or a ``[B, 3, H, W]`` float
                tensor in [0, 1]. Tensor inputs route through
                :meth:`encode_for_optimization`.

        Returns:
            Float tensor of shape ``[1, 768]``.

        Raises:
            EncoderError: On any preprocessing or forward-pass failure.
        """
        self._load_modality("image")
        try:
            if isinstance(image, torch.Tensor):
                return self.encode_for_optimization(image)
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert("RGB")

            processor = self._modality_models["image"]["processor"]
            model = self._modality_models["image"]["model"]
            inputs = processor(image, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                vision_outputs = model.vision_model(**inputs)
                pooled = (
                    vision_outputs[1]
                    if isinstance(vision_outputs, tuple)
                    else vision_outputs.pooler_output
                )
                projected = model.visual_projection(pooled)
            return F.normalize(projected, dim=-1)
        except Exception as e:
            raise EncoderError("image", e) from e

    def encode_audio(
        self,
        audio: Path | torch.Tensor,
        start: float = 0.0,
        duration: float = 10.0,
    ) -> torch.Tensor:
        """Encode audio to a unit-normalised ``[1, 768]`` embedding.

        Args:
            audio: Path to an audio file, or a 1-D float waveform tensor.
            start: Start offset in seconds when loading from a file
                (currently unused; passed for protocol compatibility).
            duration: Clip duration in seconds (LanguageBind expects ~10s
                clips).

        Returns:
            Float tensor of shape ``[1, 768]``.

        Raises:
            EncoderError: On any preprocessing or forward-pass failure.
        """
        self._load_modality("audio")
        try:
            processor = self._modality_models["audio"]["processor"]
            model = self._modality_models["audio"]["model"]
            if isinstance(audio, (str, Path)):
                inputs = processor([str(audio)], return_tensors="pt")
            else:
                # Tensor path is not currently supported by the LanguageBind
                # audio processor; user must pass a path.
                raise NotImplementedError(
                    "encode_audio currently requires a filesystem path. "
                    "Tensor-input support is a follow-up to integrate with "
                    "Stable Audio Open's render pipeline."
                )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                audio_outputs = model.vision_model(**inputs)
                pooled = (
                    audio_outputs[1]
                    if isinstance(audio_outputs, tuple)
                    else audio_outputs.pooler_output
                )
                projected = model.visual_projection(pooled)
            return F.normalize(projected, dim=-1)
        except EncoderError:
            raise
        except Exception as e:
            raise EncoderError("audio", e) from e

    def encode_video(
        self,
        video: Path | torch.Tensor,
        timestamp: float = 0.0,
    ) -> torch.Tensor:
        """Encode a video clip to a unit-normalised ``[1, 768]`` embedding.

        Args:
            video: Path to a video file (mp4/mov/etc.), or a
                ``[B, F, 3, H, W]`` float tensor in [0, 1].
            timestamp: Start offset in seconds (currently unused; passed for
                protocol compatibility).

        Returns:
            Float tensor of shape ``[1, 768]``.

        Raises:
            EncoderError: On any preprocessing or forward-pass failure.
        """
        self._load_modality("video")
        try:
            processor = self._modality_models["video"]["processor"]
            model = self._modality_models["video"]["model"]
            if isinstance(video, (str, Path)):
                inputs = processor([str(video)], return_tensors="pt")
            elif isinstance(video, torch.Tensor):
                # Differentiable tensor path — bypass processor.
                return self.encode_video_for_optimization(video)
            else:
                raise EncoderError(
                    "video",
                    TypeError(f"Unsupported video input type: {type(video).__name__}"),
                )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                video_outputs = model.vision_model(**inputs)
                pooled = (
                    video_outputs[1]
                    if isinstance(video_outputs, tuple)
                    else video_outputs.pooler_output
                )
                projected = model.visual_projection(pooled)
            return F.normalize(projected, dim=-1)
        except EncoderError:
            raise
        except Exception as e:
            raise EncoderError("video", e) from e

    # ------------------------------------------------------------------
    # v2 duck-typed interface — differentiable paths
    # ------------------------------------------------------------------

    def encode_for_optimization(self, tensor: torch.Tensor) -> torch.Tensor:
        """Differentiable image encoding for gradient-descent loops.

        Gradients flow through this method — do not wrap in ``torch.no_grad()``
        when used inside an optimisation step.

        Args:
            tensor: ``[B, 3, H, W]`` float tensor with pixel values in [0, 1].

        Returns:
            ``[B, 768]`` unit-normalised float tensor.
        """
        self._load_modality("image")
        model = self._modality_models["image"]["model"]

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

        vision_outputs = model.vision_model(pixel_values=normalized)
        pooled = (
            vision_outputs[1] if isinstance(vision_outputs, tuple) else vision_outputs.pooler_output
        )
        projected = model.visual_projection(pooled)
        return F.normalize(projected, dim=-1)

    def encode_video_for_optimization(self, tensor: torch.Tensor) -> torch.Tensor:
        """Differentiable video encoding for gradient-descent loops.

        Args:
            tensor: ``[B, F, 3, H, W]`` float tensor in [0, 1].

        Returns:
            ``[B, 768]`` unit-normalised float tensor.
        """
        self._load_modality("video")
        model = self._modality_models["video"]["model"]

        if tensor.dim() != 5:
            raise EncoderError(
                "video",
                ValueError(f"Expected 5D tensor [B, F, 3, H, W], got shape {list(tensor.shape)}"),
            )

        mean = torch.tensor(_IMAGENET_MEAN, device=tensor.device).view(1, 1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=tensor.device).view(1, 1, 3, 1, 1)
        normalized = (tensor - mean) / std

        if normalized.shape[-2:] != (_INPUT_SIZE, _INPUT_SIZE):
            b, f, c, h, w = normalized.shape
            flat = normalized.view(b * f, c, h, w)
            flat = F.interpolate(
                flat,
                size=(_INPUT_SIZE, _INPUT_SIZE),
                mode="bilinear",
                align_corners=False,
            )
            normalized = flat.view(b, f, c, _INPUT_SIZE, _INPUT_SIZE)

        vision_outputs = model.vision_model(pixel_values=normalized)
        pooled = (
            vision_outputs[1] if isinstance(vision_outputs, tuple) else vision_outputs.pooler_output
        )
        projected = model.visual_projection(pooled)
        return F.normalize(projected, dim=-1)

    def encode_text_for_optimization(
        self, token_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Differentiable text encoding for the text-anchor trick.

        Used by ``embedding_art.loss.text_anchor`` to project a current
        embedding back to language space during rendering — gradients flow
        through the text encoder so the text-anchor loss is well-defined.

        Args:
            token_ids: ``[B, L]`` long tensor of token ids.
            attention_mask: Optional ``[B, L]`` long tensor; defaults to
                all-ones if not provided.

        Returns:
            ``[B, 768]`` unit-normalised float tensor.
        """
        if not self._modality_models:
            self._load_modality("image")
        modality_key = next(iter(self._modality_models))
        model = self._modality_models[modality_key]["model"]
        if attention_mask is None:
            attention_mask = torch.ones_like(token_ids)
        text_outputs = model.text_model(input_ids=token_ids, attention_mask=attention_mask)
        pooled = text_outputs[1] if isinstance(text_outputs, tuple) else text_outputs.pooler_output
        projected = model.text_projection(pooled)
        return F.normalize(projected, dim=-1)

    # ------------------------------------------------------------------
    # Multi-layer feature extraction
    # ------------------------------------------------------------------

    def get_layer_features(self, tensor: torch.Tensor) -> dict[int, LayerFeatures]:
        """Extract intermediate ViT activations for multi-layer alignment loss.

        Returns every 4th hidden state (0, 4, 8, …) and always appends the
        final layer regardless of whether it falls on a 4-step boundary.

        Args:
            tensor: ``[B, 3, H, W]`` or ``[3, H, W]`` float tensor in [0, 1].

        Returns:
            Mapping from layer index → ``LayerFeatures`` whose ``tokens`` field
            has shape ``[B, N_patches + 1, D]`` (CLS + patch tokens).
        """
        self._load_modality("image")
        model = self._modality_models["image"]["model"]

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

        outputs = model.vision_model(
            pixel_values=normalized,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states  # tuple of [B, N, D] per layer
        if hidden_states is None:
            raise EncoderError(
                "image",
                RuntimeError(
                    "vision_model did not return hidden_states; "
                    "model may not support multi-layer feature extraction."
                ),
            )

        total_layers = len(hidden_states)
        selected_indices = list(range(0, total_layers, 4))
        if (total_layers - 1) not in selected_indices:
            selected_indices.append(total_layers - 1)

        result: dict[int, LayerFeatures] = {}
        for idx in selected_indices:
            result[idx] = LayerFeatures(
                tensor=hidden_states[idx],
                spatial=True,
                shape_semantic="batch_tokens_dim",
                layer_name=f"layer_{idx}",
            )
        return result

    # ------------------------------------------------------------------
    # ConceptSpec dispatch
    # ------------------------------------------------------------------

    def encode(self, spec: ConceptSpec) -> Concept:
        """Encode a :class:`ConceptSpec` to a :class:`Concept`.

        Dispatches on ``spec.modality`` to the appropriate ``encode_<modality>``.
        Lazy-loads the per-modality LanguageBind checkpoint on first use.
        """
        from embedding_art.core.concept import Concept

        modality = spec.modality
        if modality == "text":
            emb = self.encode_text(spec.value)
        elif modality == "image":
            emb = self.encode_image(spec.value)
        elif modality == "audio":
            emb = self.encode_audio(Path(spec.value))
        elif modality == "video":
            emb = self.encode_video(Path(spec.value))
        else:
            raise EncoderError(
                modality,
                ValueError(
                    f"Unsupported modality {modality!r}; LanguageBind supports "
                    "text / image / audio / video."
                ),
            )
        return Concept(embedding=emb, description=str(spec.value))

    # ------------------------------------------------------------------
    # Apple Silicon perf
    # ------------------------------------------------------------------

    def enable_sdpa_attention(self) -> dict[str, int]:
        """Re-route attention blocks to ``scaled_dot_product_attention``.

        Walks every loaded modality's transformer and replaces any
        attention block of the standard CLIP shape (``q_proj``,
        ``k_proj``, ``v_proj``, ``out_proj``, ``num_heads``) with an
        SDPA-backed forward. On PyTorch 2.4+ MPS this dispatches to a
        flash-attention-style kernel and gives ~2x attention
        throughput.

        Returns:
            Map ``{modality: n_blocks_patched}``. Modalities with zero
            patches usually mean the model already uses SDPA natively
            or has an unrecognised attention layout.
        """
        report: dict[str, int] = {}
        for modality, bundle in self._modality_models.items():
            model = bundle.get("model")
            if model is None:
                continue
            report[modality] = self._patch_clip_attention(model)
        return report

    @staticmethod
    def _patch_clip_attention(root: Any) -> int:
        """Patch every CLIP-shaped attention block under ``root``.

        Returns the number of blocks patched.
        """

        def _has_clip_attention_shape(m: Any) -> bool:
            return (
                hasattr(m, "q_proj")
                and hasattr(m, "k_proj")
                and hasattr(m, "v_proj")
                and hasattr(m, "out_proj")
                and hasattr(m, "num_heads")
            )

        n_patched = 0
        for module in root.modules():
            if not _has_clip_attention_shape(module):
                continue

            def _make_forward(block: Any) -> Any:
                num_heads = block.num_heads

                def _sdpa_forward(
                    hidden_states: torch.Tensor,
                    attention_mask: Any = None,
                    causal_attention_mask: Any = None,
                    output_attentions: bool = False,
                    **kwargs: Any,
                ) -> Any:
                    b, n, d = hidden_states.shape
                    head_dim = d // num_heads
                    q = block.q_proj(hidden_states).view(b, n, num_heads, head_dim).transpose(1, 2)
                    k = block.k_proj(hidden_states).view(b, n, num_heads, head_dim).transpose(1, 2)
                    v = block.v_proj(hidden_states).view(b, n, num_heads, head_dim).transpose(1, 2)
                    # Causal/attention masks are forwarded directly;
                    # SDPA handles both. ``causal_attention_mask`` is
                    # CLIP's name for the lookahead mask in the text
                    # tower.
                    mask = (
                        causal_attention_mask
                        if causal_attention_mask is not None
                        else attention_mask
                    )
                    attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                    attn = attn.transpose(1, 2).contiguous().view(b, n, d)
                    out = block.out_proj(attn)
                    if output_attentions:
                        # The HF API expects (attn_output, attn_weights).
                        # SDPA doesn't return weights; return None to
                        # match the contract.
                        return out, None
                    return out, None

                return _sdpa_forward

            module.forward = _make_forward(module)  # type: ignore[assignment]
            n_patched += 1
        return n_patched

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def unload(self, modality: str | None = None) -> None:
        """Release loaded LanguageBind weights.

        Args:
            modality: If provided, unload only that modality's model. Otherwise
                unload all loaded modalities.
        """
        if modality is None:
            self._modality_models.clear()
            self._tokenizer = None
        elif modality in self._modality_models:
            del self._modality_models[modality]
        gc.collect()
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        elif self._device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
