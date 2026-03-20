"""
CLAP encoder for audio-language embedding.

Wraps ``laion/larger_clap_general`` from HuggingFace transformers and
implements the v2 Encoder duck-typed interface:

* ``card`` — class-level ``EncoderCard`` with capability flags.
* ``encode_text`` / ``encode_audio`` — standard modality entry-points.
* ``encode_for_optimization`` — differentiable audio path for gradient-descent loops.
* ``encode(ConceptSpec)`` — single-method dispatch.
* ``unload`` — release weights and clear device cache.

Requires: ``pip install transformers librosa``
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from embedding_art.encoders.registry import EncoderCapability, EncoderCard
from embedding_art.exceptions import EncoderError, ModelLoadError

if TYPE_CHECKING:
    from embedding_art.core.concept import Concept
    from embedding_art.core.concept_spec import ConceptSpec

logger = logging.getLogger(__name__)

MODEL_NAME = "laion/larger_clap_general"
EMBEDDING_DIM = 512

# CLAP's audio feature extractor expects 48 kHz mono audio.
_CLAP_SAMPLE_RATE = 48_000


class CLAPEncoder:
    """LAION CLAP encoder for text and audio embedding.

    Implements the v2 duck-typed encoder interface.  The ``card`` class attribute
    is intentionally placed on the class (not a property) so that the registry can
    read it via ``cls.card`` before any instantiation.

    Args:
        device: PyTorch device string (e.g. ``"mps"``, ``"cpu"``, ``"cuda"``).
        model_name: HuggingFace model identifier.  Override only for testing.
    """

    # Class-level card — readable on the class before instantiation.
    # A property would only work on instances, causing AttributeError in the registry.
    card: EncoderCard = EncoderCard(
        name="clap-general",
        capabilities=(
            EncoderCapability.TEXT
            | EncoderCapability.AUDIO
            | EncoderCapability.BACKPROP_OPTIMIZABLE
        ),
        embedding_dim=EMBEDDING_DIM,
        memory_estimate_mb=1200,
        backprop_cost=1.0,
    )

    def __init__(self, device: str = "mps", model_name: str = MODEL_NAME) -> None:
        self._device = torch.device(device)
        self._model_name = model_name
        self._model: Any = None
        self._processor: Any = None
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            from transformers import ClapModel, ClapProcessor

            logger.info("Loading CLAP model: %s", self._model_name)
            self._model = (
                ClapModel.from_pretrained(self._model_name)
                .to(self._device)
                .eval()
            )
            self._processor = ClapProcessor.from_pretrained(self._model_name)
            logger.info("CLAP loaded successfully")
        except Exception as e:
            raise ModelLoadError(self._model_name, e) from e

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
        """Encode text to a unit-normalised [1, 512] embedding.

        Args:
            text: Input string.

        Returns:
            Float tensor of shape ``[1, 512]``.

        Raises:
            EncoderError: On any tokenisation or forward-pass failure.
        """
        try:
            inputs = self._processor(
                text=text,
                return_tensors="pt",
                padding=True,
            ).to(self._device)
            with torch.no_grad():
                text_features = self._model.get_text_features(**inputs)
            return F.normalize(text_features, dim=-1)
        except Exception as e:
            raise EncoderError("text", e) from e

    def encode_audio(
        self,
        audio: Path | torch.Tensor,
        start: float = 0.0,
        duration: float = 2.0,
    ) -> torch.Tensor:
        """Encode audio to a unit-normalised [1, 512] embedding.

        Accepts a filesystem path (loaded via librosa at 48 kHz) or a
        pre-loaded waveform tensor.

        Args:
            audio: Path to an audio file, or a 1-D float waveform tensor.
            start: Start offset in seconds when loading from a file.
            duration: Clip duration in seconds when loading from a file.

        Returns:
            Float tensor of shape ``[1, 512]``.

        Raises:
            EncoderError: On any preprocessing or forward-pass failure.
        """
        try:
            if isinstance(audio, (str, Path)):
                import librosa

                waveform, sr = librosa.load(
                    str(audio),
                    sr=_CLAP_SAMPLE_RATE,
                    offset=start,
                    duration=duration,
                )
            elif isinstance(audio, torch.Tensor):
                waveform = audio.cpu().numpy()
                if waveform.ndim > 1:
                    waveform = waveform.squeeze()
                sr = _CLAP_SAMPLE_RATE
            else:
                raise ValueError(f"Unsupported audio type: {type(audio)}")

            inputs = self._processor(
                audio=waveform,
                sampling_rate=sr,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                audio_features = self._model.get_audio_features(**inputs)
            return F.normalize(audio_features, dim=-1)
        except EncoderError:
            raise
        except Exception as e:
            raise EncoderError("audio", e) from e

    def encode_image(self, image: Any) -> torch.Tensor:
        """Not supported by CLAP.

        Raises:
            EncoderError: Always, with an explanatory message.
        """
        raise EncoderError("image", NotImplementedError("CLAP does not support images"))

    def encode_video(self, video: Any, timestamp: float = 0.0) -> torch.Tensor:
        """Not supported by CLAP.

        Raises:
            EncoderError: Always, with an explanatory message.
        """
        raise EncoderError("video", NotImplementedError("CLAP does not support video"))

    # ------------------------------------------------------------------
    # v2 duck-typed interface
    # ------------------------------------------------------------------

    def encode_for_optimization(self, tensor: torch.Tensor) -> torch.Tensor:
        """Differentiable audio encoding for gradient-descent optimisation loops.

        Gradients flow through this method — do *not* wrap the call in
        ``torch.no_grad()`` if you need them.

        The input is a mel-spectrogram tensor in the format expected by CLAP's
        audio model (typically produced by the AudioLDM generator).  It is
        forwarded through the audio encoder, projected, and unit-normalised.

        Args:
            tensor: Mel-spectrogram tensor with shape ``[B, n_mels, T]`` or
                whatever shape ``ClapAudioModel`` expects as ``input_features``.

        Returns:
            ``[B, 512]`` unit-normalised float tensor.
        """
        audio_outputs = self._model.audio_model(input_features=tensor)
        pooled = audio_outputs.pooler_output
        if hasattr(self._model, "audio_projection"):
            pooled = self._model.audio_projection(pooled)
        return F.normalize(pooled, dim=-1)

    def encode(self, spec: ConceptSpec) -> Concept:
        """Dispatch a :class:`~embedding_art.core.concept_spec.ConceptSpec` to the right encoder.

        Supports ``text`` and ``audio`` modalities.  When both are present the
        embeddings are averaged and re-normalised.

        Args:
            spec: Concept specification.

        Returns:
            A :class:`~embedding_art.core.concept.Concept` whose embedding is
            derived from the chosen modality (or modality combination).

        Raises:
            ValueError: If no supported modality (text or audio) is set on *spec*.
        """
        from embedding_art.core.concept import Concept

        if spec.text is not None and spec.audio is not None:
            text_emb = self.encode_text(spec.text)
            audio_emb = self.encode_audio(spec.audio)
            embedding = F.normalize(text_emb + audio_emb, dim=-1)
            desc = f'text:"{spec.text}" + audio:{spec.audio.name}'
            return Concept(embedding=embedding, description=desc)

        if spec.text is not None:
            embedding = self.encode_text(spec.text)
            return Concept(embedding=embedding, description=f'text:"{spec.text}"')

        if spec.audio is not None:
            embedding = self.encode_audio(spec.audio)
            return Concept(embedding=embedding, description=f"audio:{spec.audio.name}")

        raise ValueError(
            "ConceptSpec has no supported modality for CLAP — "
            "at least one of text or audio must be provided"
        )

    def unload(self) -> None:
        """Release model weights from memory.

        Sets ``_model`` and ``_processor`` to ``None``, then triggers garbage
        collection and clears the device cache (MPS or CUDA).
        """
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None

        gc.collect()
        if self._device.type == "mps":
            torch.mps.empty_cache()
        elif self._device.type == "cuda":
            torch.cuda.empty_cache()
