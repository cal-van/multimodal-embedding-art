"""
Unit tests for the source_input field on Concept.

source_input is an optional tensor that holds the raw encoder input
(e.g. a pixel tensor for an image concept). It is ephemeral: it moves
with to(), but is never persisted through save/load and is always
stripped by arithmetic operations.
"""

import tempfile
from pathlib import Path

import torch

from embedding_art.core.concept import Concept

# =============================================================================
# Helpers
# =============================================================================


def make_concept(source_input: torch.Tensor | None = None) -> Concept:
    """Return a minimal Concept with an optional source_input."""
    return Concept(
        embedding=torch.tensor([[1.0, 0.0, 0.0]]),
        description="test",
        source_input=source_input,
    )


def make_source_tensor() -> torch.Tensor:
    """Return a small dummy source tensor (e.g. a 3x4x4 pixel patch)."""
    return torch.ones(3, 4, 4)


# =============================================================================
# Default value
# =============================================================================


class TestSourceInputDefault:
    """source_input should default to None when not provided."""

    def test_source_input_defaults_to_none(self) -> None:
        """Concept constructed without source_input should have source_input=None."""
        concept = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0]]))

        assert concept.source_input is None

    def test_source_input_none_when_only_description_provided(self) -> None:
        """Providing only description should still leave source_input as None."""
        concept = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0]]), description="foo")

        assert concept.source_input is None


# =============================================================================
# Explicit value preserved
# =============================================================================


class TestSourceInputPreserved:
    """source_input should be stored as-is when set explicitly."""

    def test_source_input_is_stored_when_set(self) -> None:
        """Concept should store the provided source_input tensor."""
        source = make_source_tensor()
        concept = make_concept(source_input=source)

        assert concept.source_input is source

    def test_source_input_not_modified_by_post_init(self) -> None:
        """__post_init__ must not alter the source_input tensor."""
        source = torch.tensor([1.0, 2.0, 3.0])
        concept = make_concept(source_input=source)

        assert torch.equal(concept.source_input, source)


# =============================================================================
# to() device movement
# =============================================================================


class TestSourceInputToDevice:
    """to() should move source_input to the target device when present."""

    def test_to_moves_source_input_when_present(self) -> None:
        """to() should return a Concept with source_input on the new device."""
        source = make_source_tensor()
        concept = make_concept(source_input=source)

        moved = concept.to("cpu")

        assert moved.source_input is not None
        assert moved.source_input.device == torch.device("cpu")

    def test_to_preserves_none_source_input(self) -> None:
        """to() on a Concept with source_input=None should keep it None."""
        concept = make_concept(source_input=None)

        moved = concept.to("cpu")

        assert moved.source_input is None

    def test_to_source_input_has_same_values_after_move(self) -> None:
        """Source tensor values should be unchanged after to()."""
        source = make_source_tensor()
        concept = make_concept(source_input=source)

        moved = concept.to("cpu")

        assert torch.equal(moved.source_input, source)


# =============================================================================
# save / load — source_input is ephemeral
# =============================================================================


class TestSourceInputNotPersisted:
    """source_input must not be serialised into the .pt checkpoint."""

    def test_save_load_drops_source_input(self) -> None:
        """Loaded Concept should have source_input=None even if original had one."""
        source = make_source_tensor()
        concept = make_concept(source_input=source)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "concept.pt"
            concept.save(path)
            loaded = Concept.load(path)

        assert loaded.source_input is None

    def test_save_load_preserves_embedding_despite_source_input(self) -> None:
        """Embedding should round-trip correctly even when source_input is present."""
        source = make_source_tensor()
        concept = make_concept(source_input=source)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "concept.pt"
            concept.save(path)
            loaded = Concept.load(path)

        assert torch.allclose(loaded.embedding, concept.embedding)


# =============================================================================
# Arithmetic drops source_input
# =============================================================================


class TestSourceInputDroppedByArithmetic:
    """All arithmetic / combination operations must set source_input=None on the result."""

    def test_add_drops_source_input(self) -> None:
        """Adding two concepts should produce a result with source_input=None."""
        a = make_concept(source_input=make_source_tensor())
        b = make_concept(source_input=make_source_tensor())

        result = a + b

        assert result.source_input is None

    def test_sub_drops_source_input(self) -> None:
        """Subtracting concepts should produce a result with source_input=None."""
        a = make_concept(source_input=make_source_tensor())
        b = make_concept(source_input=make_source_tensor())

        result = a - b

        assert result.source_input is None

    def test_mul_drops_source_input(self) -> None:
        """Scalar multiplication should produce a result with source_input=None."""
        a = make_concept(source_input=make_source_tensor())

        result = a * 0.5

        assert result.source_input is None

    def test_rmul_drops_source_input(self) -> None:
        """Right scalar multiplication should produce a result with source_input=None."""
        a = make_concept(source_input=make_source_tensor())

        result = 0.5 * a

        assert result.source_input is None

    def test_neg_drops_source_input(self) -> None:
        """Negation should produce a result with source_input=None."""
        a = make_concept(source_input=make_source_tensor())

        result = -a

        assert result.source_input is None

    def test_slerp_drops_source_input(self) -> None:
        """slerp should produce a result with source_input=None."""
        a = make_concept(source_input=make_source_tensor())
        b = make_concept(source_input=make_source_tensor())

        result = Concept.slerp(a, b, 0.5)

        assert result.source_input is None

    def test_combine_drops_source_input(self) -> None:
        """combine() should produce a result with source_input=None."""
        a = make_concept(source_input=make_source_tensor())
        b = make_concept(source_input=make_source_tensor())

        result = Concept.combine([a, b])

        assert result.source_input is None

    def test_arithmetic_on_concept_without_source_input_still_returns_none(self) -> None:
        """Operations on concepts with source_input=None should also return None."""
        a = make_concept(source_input=None)
        b = make_concept(source_input=None)

        result = a + b

        assert result.source_input is None
