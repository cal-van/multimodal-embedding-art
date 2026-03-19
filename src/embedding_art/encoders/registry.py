"""
Encoder registry with capability flags, lazy loading, and memory-aware admission control.

Design notes
------------
* ``EncoderCapability`` is a ``Flag`` enum so capabilities compose naturally with
  bitwise OR and membership is tested with the ``in`` operator.
* ``EncoderCard`` is a frozen dataclass — value-equal, hashable, immutable.
* ``EncoderRegistry`` is a plain class.  It never imports model weights itself;
  callers control *when* loading happens by calling ``load()``.
* The 80 % safety margin in ``can_fit`` is applied to the *budget*, not to the
  estimates.  So ``effective_budget = budget * 0.80`` and we check
  ``sum(estimates) <= effective_budget``.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any

from embedding_art.exceptions import EncoderNotFoundError

# ---------------------------------------------------------------------------
# EncoderCapability
# ---------------------------------------------------------------------------


class EncoderCapability(enum.Flag):
    """Capability flags for encoders.

    Flags compose with ``|``::

        caps = EncoderCapability.TEXT | EncoderCapability.IMAGE
        EncoderCapability.TEXT in caps  # True
        EncoderCapability.AUDIO in caps  # False
    """

    TEXT = enum.auto()
    IMAGE = enum.auto()
    AUDIO = enum.auto()
    VIDEO = enum.auto()
    DEPTH = enum.auto()
    BACKPROP_OPTIMIZABLE = enum.auto()
    MULTI_LAYER_FEATURES = enum.auto()


# ---------------------------------------------------------------------------
# EncoderCard
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class EncoderCard:
    """Metadata describing a registered encoder.

    Attributes:
        name: Registry key used to look up this encoder.
        capabilities: Bitfield of ``EncoderCapability`` flags.
        embedding_dim: Dimensionality of the output embedding vector.
        memory_estimate_mb: Approximate peak memory footprint in megabytes.
        backprop_cost: Relative cost of running a backward pass (arbitrary units;
            higher = more expensive).
    """

    name: str
    capabilities: EncoderCapability
    embedding_dim: int
    memory_estimate_mb: int
    backprop_cost: float


# ---------------------------------------------------------------------------
# EncoderRegistry
# ---------------------------------------------------------------------------

# Map lower-case modality name → corresponding capability flag.
_MODALITY_TO_CAPABILITY: dict[str, EncoderCapability] = {
    "text": EncoderCapability.TEXT,
    "image": EncoderCapability.IMAGE,
    "audio": EncoderCapability.AUDIO,
    "video": EncoderCapability.VIDEO,
    "depth": EncoderCapability.DEPTH,
}

# Safety margin applied to the memory budget before comparison.
_MEMORY_SAFETY_MARGIN = 0.80


class EncoderRegistry:
    """Registry for named encoder classes with lazy instantiation and caching.

    Usage::

        registry = EncoderRegistry()
        registry.register("imagebind", ImageBindEncoder)
        encoder = registry.load("imagebind", device="mps")
        registry.unload("imagebind")
    """

    def __init__(self) -> None:
        # name → encoder class (must have a ``card: EncoderCard`` class attribute)
        self._classes: dict[str, type] = {}
        # name → live instance (populated on first load, cleared on unload)
        self._instances: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, encoder_cls: type) -> None:
        """Register *encoder_cls* under *name*.

        The class must expose a ``card`` class attribute of type ``EncoderCard``.
        Re-registering the same name overwrites the previous registration and
        clears any cached instance.

        Args:
            name: Registry key.  Must match ``encoder_cls.card.name`` by convention
                (not enforced so callers can alias encoders).
            encoder_cls: Class to instantiate on ``load()``.
        """
        self._classes[name] = encoder_cls
        self._instances.pop(name, None)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_available(self) -> list[EncoderCard]:
        """Return the ``EncoderCard`` for every registered encoder.

        Order matches registration order (Python 3.7+ dict insertion order).
        """
        return [cls.card for cls in self._classes.values()]

    def get_card(self, name: str) -> EncoderCard:
        """Return the ``EncoderCard`` for *name*.

        Raises:
            EncoderNotFoundError: If *name* has not been registered.
        """
        if name not in self._classes:
            raise EncoderNotFoundError(name, list(self._classes.keys()))
        return self._classes[name].card

    def loaded_encoders(self) -> list[str]:
        """Return the names of all currently loaded (cached) encoders."""
        return list(self._instances.keys())

    # ------------------------------------------------------------------
    # Loading / unloading
    # ------------------------------------------------------------------

    def load(self, name: str, **kwargs: Any) -> Any:
        """Return a loaded encoder instance, creating it if necessary.

        The instance is cached after the first call; subsequent calls with the
        same *name* return the same object regardless of *kwargs*.

        Args:
            name: Registered encoder name.
            **kwargs: Passed to the encoder constructor on first load only.

        Returns:
            The encoder instance.

        Raises:
            EncoderNotFoundError: If *name* has not been registered.
        """
        if name not in self._classes:
            raise EncoderNotFoundError(name, list(self._classes.keys()))
        if name not in self._instances:
            self._instances[name] = self._classes[name](**kwargs)
        return self._instances[name]

    def unload(self, name: str) -> None:
        """Remove the cached instance for *name*, freeing its memory.

        Calling ``unload`` on a name that was never loaded (or never registered)
        is a harmless no-op.

        Args:
            name: Registered encoder name to evict from the cache.
        """
        self._instances.pop(name, None)

    # ------------------------------------------------------------------
    # Modality helpers
    # ------------------------------------------------------------------

    def get_for_modality(self, modality: str) -> list[str]:
        """Return the names of all registered encoders that support *modality*.

        The comparison is case-insensitive.  Unknown modalities return an empty
        list rather than raising.

        Args:
            modality: One of ``"text"``, ``"image"``, ``"audio"``, ``"video"``,
                ``"depth"`` (case-insensitive).

        Returns:
            List of registered encoder names whose capability flags include the
            flag corresponding to *modality*.
        """
        capability = _MODALITY_TO_CAPABILITY.get(modality.lower())
        if capability is None:
            return []
        return [name for name, cls in self._classes.items() if capability in cls.card.capabilities]

    # ------------------------------------------------------------------
    # Memory budget
    # ------------------------------------------------------------------

    def can_fit(self, names: list[str], memory_budget_mb: int | float) -> bool:
        """Check whether loading *names* would fit within *memory_budget_mb*.

        An 80 % safety margin is applied: the effective budget is
        ``memory_budget_mb * 0.80``.

        Args:
            names: Encoder names to evaluate.
            memory_budget_mb: Total memory available in megabytes.

        Returns:
            ``True`` if the combined ``memory_estimate_mb`` of all named encoders
            is at most ``memory_budget_mb * 0.80``.

        Raises:
            EncoderNotFoundError: If any name in *names* is not registered.
        """
        if not names:
            return True

        # Validate all names first so we give a clear error before any arithmetic.
        for name in names:
            if name not in self._classes:
                raise EncoderNotFoundError(name, list(self._classes.keys()))

        total_mb = sum(self._classes[name].card.memory_estimate_mb for name in names)
        effective_budget = memory_budget_mb * _MEMORY_SAFETY_MARGIN
        return total_mb <= effective_budget
