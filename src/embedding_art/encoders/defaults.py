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

import logging

from embedding_art.encoders.registry import EncoderRegistry

logger = logging.getLogger(__name__)


def create_default_registry() -> EncoderRegistry:
    """Create registry with all built-in encoders.

    Imports are deferred so missing optional dependencies don't prevent the
    registry from loading. ``languagebind`` is registered first so it is the
    canonical encoder when callers ask for the default.

    A failed import is logged at WARNING (not silently swallowed): otherwise a
    later ``registry.load("languagebind")`` surfaces a confusing "not
    registered" error that hides the real ImportError (missing extra, bad
    PYTHONPATH, etc.).
    """
    registry = EncoderRegistry()

    _candidates = [
        ("languagebind", "embedding_art.encoders.languagebind", "LanguageBindEncoder"),
        ("imagebind", "embedding_art.encoders.imagebind", "ImageBindEncoder"),
        ("siglip2-so400m", "embedding_art.encoders.siglip2", "SigLIP2Encoder"),
        ("clap-general", "embedding_art.encoders.clap", "CLAPEncoder"),
    ]
    import importlib

    for name, module_path, class_name in _candidates:
        try:
            module = importlib.import_module(module_path)
            registry.register(name, getattr(module, class_name))
        except ImportError as exc:
            logger.warning(
                "Encoder %r not registered — import failed (install its extras / "
                "check PYTHONPATH): %s",
                name,
                exc,
            )

    return registry
