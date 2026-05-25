"""
Custom exceptions for embedding-art.

These provide user-friendly error messages with actionable suggestions.
"""


class EmbeddingArtError(Exception):
    """Base exception for embedding-art errors."""

    pass


class ImageBindNotInstalledError(EmbeddingArtError):
    """Raised when ImageBind is not installed."""

    def __init__(self) -> None:
        message = (
            "ImageBind is not installed.\n\n"
            "To install ImageBind:\n"
            "  git clone https://github.com/facebookresearch/ImageBind\n"
            "  cd ImageBind\n"
            "  pip install -e .\n\n"
            "Note: ImageBind requires PyTorch with CUDA or MPS support.\n"
            "See https://github.com/facebookresearch/ImageBind for requirements."
        )
        super().__init__(message)


class ModelLoadError(EmbeddingArtError):
    """Raised when a model fails to load."""

    def __init__(self, model_name: str, original_error: Exception) -> None:
        self.model_name = model_name
        self.original_error = original_error

        # When the underlying failure is a missing Python module, the network /
        # cache suggestions are actively misleading — surface the import error
        # verbatim instead so the user reads the install instructions baked
        # into the wrapper's ImportError message.
        if isinstance(original_error, ImportError):
            message = f"Failed to load model '{model_name}'.\n\n" f"{original_error}"
        else:
            message = (
                f"Failed to load model '{model_name}'.\n\n"
                f"Original error: {original_error}\n\n"
                "Possible causes:\n"
                "  - Network connection issues during download\n"
                "  - Insufficient disk space for model weights\n"
                "  - Corrupted model cache\n\n"
                "Suggestions:\n"
                "  - Check your internet connection\n"
                "  - Clear the model cache: rm -rf ~/.cache/huggingface/hub\n"
                "  - Try again with a stable connection"
            )
        super().__init__(message)


class OutOfMemoryError(EmbeddingArtError):
    """Raised when GPU runs out of memory."""

    def __init__(
        self,
        operation: str,
        device: str,
        original_error: Exception,
    ) -> None:
        self.operation = operation
        self.device = device
        self.original_error = original_error

        if device == "mps":
            device_name = "Apple Silicon GPU"
            clear_cache_suggestion = "  - Restart Python to clear MPS memory"
        elif device.startswith("cuda"):
            device_name = "NVIDIA GPU"
            clear_cache_suggestion = "  - Run torch.cuda.empty_cache() or restart Python"
        else:
            device_name = device
            clear_cache_suggestion = "  - Restart Python to clear memory"

        message = (
            f"Out of memory on {device_name} during {operation}.\n\n"
            f"Original error: {original_error}\n\n"
            "Suggestions:\n"
            "  - Reduce batch size if processing multiple items\n"
            "  - Use a smaller image resolution\n"
            "  - Close other GPU-intensive applications\n"
            f"{clear_cache_suggestion}\n"
            "  - Try using CPU instead: --device cpu (slower but uses system RAM)"
        )
        super().__init__(message)


class InvalidConfigError(EmbeddingArtError):
    """Raised when configuration is invalid."""

    def __init__(self, field: str, value: object, reason: str) -> None:
        self.field = field
        self.value = value
        self.reason = reason

        message = f"Invalid configuration for '{field}'.\n\nValue: {value}\nReason: {reason}"
        super().__init__(message)


class EncoderError(EmbeddingArtError):
    """Raised when encoding fails."""

    def __init__(self, modality: str, original_error: Exception) -> None:
        self.modality = modality
        self.original_error = original_error

        message = f"Failed to encode {modality} input.\n\nOriginal error: {original_error}"
        super().__init__(message)


class GeneratorError(EmbeddingArtError):
    """Raised when generation/decoding fails."""

    def __init__(self, operation: str, original_error: Exception) -> None:
        self.operation = operation
        self.original_error = original_error

        message = f"Failed during {operation}.\n\nOriginal error: {original_error}"
        super().__init__(message)


class UpscalerError(EmbeddingArtError):
    """Raised when upscaling fails."""

    def __init__(self, operation: str, original_error: Exception) -> None:
        self.operation = operation
        self.original_error = original_error

        message = (
            f"Failed during {operation}.\n\n"
            f"Original error: {original_error}\n\n"
            "Suggestions:\n"
            "  - Ensure the input image is a valid RGB image\n"
            "  - Try reducing the scale factor\n"
            "  - Check that Real-ESRGAN is installed: pip install py-real-esrgan"
        )
        super().__init__(message)


class EncoderNotFoundError(EmbeddingArtError):
    """Raised when a requested encoder name is not in the registry."""

    def __init__(self, name: str, available: list[str]) -> None:
        self.name = name
        self.available = available

        if available:
            available_str = "\n".join(f"  - {enc}" for enc in available)
            suggestion = f"Available encoders:\n{available_str}"
        else:
            suggestion = "No encoders are currently registered. Check your installation."

        message = (
            f"Encoder '{name}' not found.\n\n"
            f"{suggestion}\n\n"
            "Suggestions:\n"
            "  - Check the encoder name for typos\n"
            "  - Ensure the encoder is installed and registered before use"
        )
        super().__init__(message)


class EncoderCapabilityError(EmbeddingArtError):
    """Raised when an operation requires a capability the encoder does not support."""

    def __init__(self, encoder_name: str, capability: str) -> None:
        self.encoder_name = encoder_name
        self.capability = capability

        message = (
            f"Encoder '{encoder_name}' does not support capability '{capability}'.\n\n"
            "Suggestions:\n"
            "  - Use an encoder that supports this capability\n"
            "  - Check the encoder's documentation for supported modalities\n"
            "  - ImageBind supports text, image, audio, video, depth, and thermal inputs"
        )
        super().__init__(message)


class SAENotTrainedError(EmbeddingArtError):
    """Raised when SAE features are requested but no trained SAE artifact exists."""

    def __init__(self, encoder_name: str) -> None:
        self.encoder_name = encoder_name

        message = (
            f"No trained SAE artifact found for encoder '{encoder_name}'.\n\n"
            "Sparse autoencoder (SAE) features require a pre-trained artifact.\n\n"
            "Suggestions:\n"
            "  - Train an SAE on this encoder first: embed-art train-sae --encoder "
            f"{encoder_name}\n"
            "  - Download a pre-trained SAE artifact if one is available\n"
            "  - Use a different encoder that has a trained SAE artifact"
        )
        super().__init__(message)


class MemoryBudgetExceededError(EmbeddingArtError):
    """Raised when loading the requested encoders would exceed the memory budget."""

    def __init__(self, requested_mb: int, available_mb: int, encoders: list[str]) -> None:
        self.requested_mb = requested_mb
        self.available_mb = available_mb
        self.encoders = encoders

        if encoders:
            encoders_str = ", ".join(encoders)
            encoders_line = f"Encoders requested: {encoders_str}\n"
        else:
            encoders_line = ""

        overage_mb = requested_mb - available_mb
        message = (
            f"Memory budget exceeded: requested {requested_mb} MB, "
            f"but only {available_mb} MB is available "
            f"({overage_mb} MB over budget).\n\n"
            f"{encoders_line}"
            "Suggestions:\n"
            "  - Load fewer encoders simultaneously\n"
            "  - Increase the memory budget with --memory-budget\n"
            "  - Use a machine with more available RAM or VRAM\n"
            "  - Unload unused models before loading new ones"
        )
        super().__init__(message)


class FeatureNotFoundError(EmbeddingArtError):
    """Raised when an SAE feature name is not in the vocabulary."""

    # Show at most this many suggestions to keep the message readable.
    _MAX_SUGGESTIONS = 10

    def __init__(self, feature_name: str, available: list[str]) -> None:
        self.feature_name = feature_name
        self.available = available

        if available:
            shown = available[: self._MAX_SUGGESTIONS]
            available_str = "\n".join(f"  - {f}" for f in shown)
            remainder = len(available) - len(shown)
            truncation = f"\n  ... and {remainder} more" if remainder else ""
            suggestion = f"Known features (sample):\n{available_str}{truncation}"
        else:
            suggestion = "No features are available in the current SAE vocabulary."

        message = (
            f"Feature '{feature_name}' not found in the SAE vocabulary.\n\n"
            f"{suggestion}\n\n"
            "Suggestions:\n"
            "  - Check the feature name for typos\n"
            "  - Use embed-art list-features to browse available features\n"
            "  - Ensure you are using the correct SAE artifact for this encoder"
        )
        super().__init__(message)
