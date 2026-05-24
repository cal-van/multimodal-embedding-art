"""Default encoder registry with all built-in encoders.

Encoder roles in the v3 (aggressive-rewrite) architecture:

* ``languagebind`` — **canonical** multimodal encoder (ICLR 2024). All four
  modalities (text + image + audio + video) bind into a single shared 768-d
  space anchored on language. Used as the canonical encoder for showcase
  targets so all four-modality renderings share a coordinate.
* ``siglip2-so400m`` — per-modality **quality probe** (text + image only,
  higher per-modality fidelity than LanguageBind's image encoder). Used at
  evaluation time to verify the canonical encoder's geometry isn't lying.
* ``clap-general`` — per-modality **quality probe** for audio.
* ``imagebind`` — **deprecated**, kept for backwards-compatible loading of
  v1/v2 artefacts only; new work targets ``languagebind``.
"""

from embedding_art.encoders.registry import EncoderRegistry


def create_default_registry() -> EncoderRegistry:
    """Create registry with all built-in encoders.

    Imports are deferred so missing optional dependencies don't prevent the
    registry from loading. ``languagebind`` is registered first so it is the
    canonical encoder when callers ask for the default.
    """
    registry = EncoderRegistry()

    try:
        from embedding_art.encoders.languagebind import LanguageBindEncoder

        registry.register("languagebind", LanguageBindEncoder)
    except ImportError:
        pass

    try:
        from embedding_art.encoders.imagebind import ImageBindEncoder

        registry.register("imagebind", ImageBindEncoder)
    except ImportError:
        pass

    try:
        from embedding_art.encoders.siglip2 import SigLIP2Encoder

        registry.register("siglip2-so400m", SigLIP2Encoder)
    except ImportError:
        pass

    try:
        from embedding_art.encoders.clap import CLAPEncoder

        registry.register("clap-general", CLAPEncoder)
    except ImportError:
        pass

    return registry
