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

        message = (
            f"Invalid configuration for '{field}'.\n\n" f"Value: {value}\n" f"Reason: {reason}"
        )
        super().__init__(message)


class EncoderError(EmbeddingArtError):
    """Raised when encoding fails."""

    def __init__(self, modality: str, original_error: Exception) -> None:
        self.modality = modality
        self.original_error = original_error

        message = f"Failed to encode {modality} input.\n\n" f"Original error: {original_error}"
        super().__init__(message)


class GeneratorError(EmbeddingArtError):
    """Raised when generation/decoding fails."""

    def __init__(self, operation: str, original_error: Exception) -> None:
        self.operation = operation
        self.original_error = original_error

        message = f"Failed during {operation}.\n\n" f"Original error: {original_error}"
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
