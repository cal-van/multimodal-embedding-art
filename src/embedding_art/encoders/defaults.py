"""Default encoder registry with all built-in encoders."""
from embedding_art.encoders.registry import EncoderRegistry


def create_default_registry() -> EncoderRegistry:
    """Create registry with all built-in encoders.

    Imports are deferred so missing optional dependencies
    (like ImageBind) don't prevent the registry from loading.
    """
    registry = EncoderRegistry()

    # Always available via the existing codebase
    try:
        from embedding_art.encoders.imagebind import ImageBindEncoder

        registry.register("imagebind", ImageBindEncoder)
    except ImportError:
        pass

    # Future encoders registered here as they're implemented:
    # try:
    #     from embedding_art.encoders.siglip2 import SigLIP2Encoder
    #     registry.register("siglip2-so400m", SigLIP2Encoder)
    # except ImportError:
    #     pass

    return registry
