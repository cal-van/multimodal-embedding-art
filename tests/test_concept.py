"""
Unit tests for the Concept class.

Tests verify the behavior of embedding operations, arithmetic,
interpolation, and persistence.
"""

import math
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from embedding_art.core.concept import Concept


class TestConceptPostInit:
    """Tests for __post_init__ normalization and tensor shape handling."""

    def test_1d_tensor_becomes_2d(self) -> None:
        """1D embedding tensor should be reshaped to 2D [1, embed_dim]."""
        embedding_1d = torch.tensor([1.0, 0.0, 0.0])
        concept = Concept(embedding=embedding_1d)

        assert concept.embedding.dim() == 2
        assert concept.embedding.shape == (1, 3)

    def test_2d_tensor_remains_2d(self) -> None:
        """2D embedding tensor should remain 2D."""
        embedding_2d = torch.tensor([[1.0, 0.0, 0.0]])
        concept = Concept(embedding=embedding_2d)

        assert concept.embedding.dim() == 2
        assert concept.embedding.shape == (1, 3)

    def test_embedding_is_normalized(self) -> None:
        """Embedding should be L2-normalized after construction."""
        unnormalized = torch.tensor([[3.0, 4.0, 0.0]])
        concept = Concept(embedding=unnormalized)

        norm = torch.norm(concept.embedding, dim=-1)
        assert torch.allclose(norm, torch.tensor([1.0]))

    def test_already_normalized_embedding_unchanged(self) -> None:
        """Already normalized embedding should remain normalized."""
        normalized = torch.tensor([[0.6, 0.8, 0.0]])  # 0.6^2 + 0.8^2 = 1
        concept = Concept(embedding=normalized)

        expected = F.normalize(normalized, dim=-1)
        assert torch.allclose(concept.embedding, expected)

    def test_description_is_stored(self) -> None:
        """Description should be stored as provided."""
        embedding = torch.tensor([1.0, 0.0, 0.0])
        description = "test description"
        concept = Concept(embedding=embedding, description=description)

        assert concept.description == description

    def test_default_description_is_empty_string(self) -> None:
        """Default description should be empty string."""
        embedding = torch.tensor([1.0, 0.0, 0.0])
        concept = Concept(embedding=embedding)

        assert concept.description == ""


class TestConceptAddition:
    """Tests for __add__ operation."""

    def test_addition_combines_embeddings(self) -> None:
        """Adding two concepts should sum their embeddings then normalize."""
        a = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0]]))
        b = Concept(embedding=torch.tensor([[0.0, 1.0, 0.0]]))

        result = a + b

        # The sum [1, 1, 0] normalized is [1/sqrt(2), 1/sqrt(2), 0]
        expected = F.normalize(torch.tensor([[1.0, 1.0, 0.0]]), dim=-1)
        assert torch.allclose(result.embedding, expected, atol=1e-6)

    def test_addition_result_is_normalized(self) -> None:
        """Result of addition should always be normalized."""
        a = Concept(embedding=torch.tensor([[3.0, 0.0, 0.0]]))
        b = Concept(embedding=torch.tensor([[0.0, 4.0, 0.0]]))

        result = a + b

        norm = torch.norm(result.embedding, dim=-1)
        assert torch.allclose(norm, torch.tensor([1.0]))

    def test_addition_description_format(self) -> None:
        """Addition should create combined description."""
        a = Concept(embedding=torch.tensor([1.0, 0.0, 0.0]), description="A")
        b = Concept(embedding=torch.tensor([0.0, 1.0, 0.0]), description="B")

        result = a + b

        assert result.description == "(A + B)"


class TestConceptSubtraction:
    """Tests for __sub__ operation."""

    def test_subtraction_subtracts_embeddings(self) -> None:
        """Subtracting concepts should subtract embeddings then normalize."""
        a = Concept(embedding=torch.tensor([[1.0, 1.0, 0.0]]))
        b = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0]]))

        result = a - b

        # After normalization of a and b, then subtraction:
        # a normalized = [1/sqrt(2), 1/sqrt(2), 0]
        # b normalized = [1, 0, 0]
        # difference = [1/sqrt(2) - 1, 1/sqrt(2), 0]
        # Then normalized again
        expected_unnorm = a.embedding - b.embedding
        expected = F.normalize(expected_unnorm, dim=-1)
        assert torch.allclose(result.embedding, expected, atol=1e-6)

    def test_subtraction_result_is_normalized(self) -> None:
        """Result of subtraction should always be normalized."""
        a = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0]]))
        b = Concept(embedding=torch.tensor([[0.0, 1.0, 0.0]]))

        result = a - b

        norm = torch.norm(result.embedding, dim=-1)
        assert torch.allclose(norm, torch.tensor([1.0]))

    def test_subtraction_description_format(self) -> None:
        """Subtraction should create combined description."""
        a = Concept(embedding=torch.tensor([1.0, 0.0, 0.0]), description="A")
        b = Concept(embedding=torch.tensor([0.0, 1.0, 0.0]), description="B")

        result = a - b

        assert result.description == "(A - B)"


class TestConceptMultiplication:
    """Tests for __mul__ and __rmul__ operations."""

    def test_scalar_multiplication(self) -> None:
        """Multiplying concept by scalar should scale then normalize."""
        concept = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0]]))

        result = concept * 2.0

        # Scaling then normalizing a unit vector gives the same unit vector
        # (because normalization removes the scale factor)
        norm = torch.norm(result.embedding, dim=-1)
        assert torch.allclose(norm, torch.tensor([1.0]))

    def test_right_multiplication(self) -> None:
        """Right multiplication should work the same as left."""
        concept = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0]]))

        result = 2.0 * concept

        norm = torch.norm(result.embedding, dim=-1)
        assert torch.allclose(norm, torch.tensor([1.0]))

    def test_multiplication_description_format(self) -> None:
        """Multiplication should update description."""
        concept = Concept(embedding=torch.tensor([1.0, 0.0, 0.0]), description="A")

        result = 0.5 * concept

        assert result.description == "0.5*A"


class TestConceptNegation:
    """Tests for __neg__ operation."""

    def test_negation_flips_direction(self) -> None:
        """Negating a concept should flip its direction."""
        concept = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0]]))

        result = -concept

        expected = torch.tensor([[-1.0, 0.0, 0.0]])
        assert torch.allclose(result.embedding, expected)

    def test_negation_result_is_normalized(self) -> None:
        """Result of negation should be normalized."""
        concept = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0]]))

        result = -concept

        norm = torch.norm(result.embedding, dim=-1)
        assert torch.allclose(norm, torch.tensor([1.0]))

    def test_negation_description_format(self) -> None:
        """Negation should update description."""
        concept = Concept(embedding=torch.tensor([1.0, 0.0, 0.0]), description="A")

        result = -concept

        assert result.description == "-A"

    def test_double_negation_returns_original_direction(self) -> None:
        """Double negation should return original direction."""
        concept = Concept(embedding=torch.tensor([[0.6, 0.8, 0.0]]))

        result = -(-concept)

        assert torch.allclose(result.embedding, concept.embedding, atol=1e-6)


class TestConceptSlerp:
    """Tests for spherical linear interpolation."""

    def test_slerp_t_zero_returns_a(self) -> None:
        """slerp with t=0 should return the first concept."""
        a = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0]]), description="A")
        b = Concept(embedding=torch.tensor([[0.0, 1.0, 0.0]]), description="B")

        result = Concept.slerp(a, b, 0.0)

        assert torch.allclose(result.embedding, a.embedding, atol=1e-6)

    def test_slerp_t_one_returns_b(self) -> None:
        """slerp with t=1 should return the second concept."""
        a = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0]]), description="A")
        b = Concept(embedding=torch.tensor([[0.0, 1.0, 0.0]]), description="B")

        result = Concept.slerp(a, b, 1.0)

        assert torch.allclose(result.embedding, b.embedding, atol=1e-6)

    def test_slerp_t_half_is_midpoint(self) -> None:
        """slerp with t=0.5 should be at the midpoint on the sphere."""
        a = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0]]), description="A")
        b = Concept(embedding=torch.tensor([[0.0, 1.0, 0.0]]), description="B")

        result = Concept.slerp(a, b, 0.5)

        # Midpoint of 90-degree arc should be at 45 degrees
        expected = F.normalize(torch.tensor([[1.0, 1.0, 0.0]]), dim=-1)
        assert torch.allclose(result.embedding, expected, atol=1e-5)

    def test_slerp_result_is_normalized(self) -> None:
        """slerp result should always be on the unit sphere."""
        a = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0]]))
        b = Concept(embedding=torch.tensor([[0.0, 1.0, 0.0]]))

        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = Concept.slerp(a, b, t)
            norm = torch.norm(result.embedding, dim=-1)
            assert torch.allclose(norm, torch.tensor([1.0]), atol=1e-6)

    def test_slerp_handles_parallel_vectors(self) -> None:
        """slerp should handle nearly parallel vectors (linear interpolation fallback)."""
        a = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0]]), description="A")
        b = Concept(embedding=torch.tensor([[1.0, 1e-8, 0.0]]), description="B")

        result = Concept.slerp(a, b, 0.5)

        # Should not raise and should return normalized result
        norm = torch.norm(result.embedding, dim=-1)
        assert torch.allclose(norm, torch.tensor([1.0]), atol=1e-6)

    def test_slerp_handles_identical_vectors(self) -> None:
        """slerp should handle identical vectors."""
        a = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0]]), description="A")
        b = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0]]), description="B")

        result = Concept.slerp(a, b, 0.5)

        assert torch.allclose(result.embedding, a.embedding, atol=1e-6)

    def test_slerp_description_format(self) -> None:
        """slerp should create descriptive description."""
        a = Concept(embedding=torch.tensor([1.0, 0.0, 0.0]), description="A")
        b = Concept(embedding=torch.tensor([0.0, 1.0, 0.0]), description="B")

        result = Concept.slerp(a, b, 0.5)

        assert result.description == "slerp(A, B, 0.5)"


class TestConceptCombine:
    """Tests for weighted combination of concepts."""

    def test_combine_with_equal_weights(self) -> None:
        """Combining concepts without weights should use equal weights."""
        a = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0]]))
        b = Concept(embedding=torch.tensor([[0.0, 1.0, 0.0]]))

        result = Concept.combine([a, b])

        # Equal weights means average direction, then normalized
        expected = F.normalize(torch.tensor([[0.5, 0.5, 0.0]]), dim=-1)
        assert torch.allclose(result.embedding, expected, atol=1e-6)

    def test_combine_with_custom_weights(self) -> None:
        """Combining concepts with custom weights."""
        a = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0]]))
        b = Concept(embedding=torch.tensor([[0.0, 1.0, 0.0]]))

        result = Concept.combine([a, b], weights=[0.8, 0.2])

        # Weighted sum then normalized
        expected = F.normalize(torch.tensor([[0.8, 0.2, 0.0]]), dim=-1)
        assert torch.allclose(result.embedding, expected, atol=1e-6)

    def test_combine_result_is_normalized(self) -> None:
        """Combined result should always be normalized."""
        concepts = [
            Concept(embedding=torch.tensor([[1.0, 0.0, 0.0]])),
            Concept(embedding=torch.tensor([[0.0, 1.0, 0.0]])),
            Concept(embedding=torch.tensor([[0.0, 0.0, 1.0]])),
        ]

        result = Concept.combine(concepts, weights=[0.5, 0.3, 0.2])

        norm = torch.norm(result.embedding, dim=-1)
        assert torch.allclose(norm, torch.tensor([1.0]))

    def test_combine_empty_list_raises(self) -> None:
        """Combining empty list should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot combine empty list"):
            Concept.combine([])

    def test_combine_mismatched_weights_raises(self) -> None:
        """Mismatched weights and concepts should raise ValueError."""
        concepts = [
            Concept(embedding=torch.tensor([[1.0, 0.0, 0.0]])),
            Concept(embedding=torch.tensor([[0.0, 1.0, 0.0]])),
        ]

        with pytest.raises(ValueError, match="Number of weights must match"):
            Concept.combine(concepts, weights=[1.0])

    def test_combine_single_concept(self) -> None:
        """Combining single concept should return equivalent concept."""
        original = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0]]))

        result = Concept.combine([original])

        assert torch.allclose(result.embedding, original.embedding, atol=1e-6)


class TestConceptSimilarity:
    """Tests for cosine similarity computation."""

    def test_similarity_identical_concepts(self) -> None:
        """Identical concepts should have similarity 1.0."""
        a = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0]]))
        b = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0]]))

        similarity = a.similarity(b)

        assert math.isclose(similarity, 1.0, abs_tol=1e-6)

    def test_similarity_opposite_concepts(self) -> None:
        """Opposite concepts should have similarity -1.0."""
        a = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0]]))
        b = Concept(embedding=torch.tensor([[-1.0, 0.0, 0.0]]))

        similarity = a.similarity(b)

        assert math.isclose(similarity, -1.0, abs_tol=1e-6)

    def test_similarity_orthogonal_concepts(self) -> None:
        """Orthogonal concepts should have similarity 0.0."""
        a = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0]]))
        b = Concept(embedding=torch.tensor([[0.0, 1.0, 0.0]]))

        similarity = a.similarity(b)

        assert math.isclose(similarity, 0.0, abs_tol=1e-6)

    def test_similarity_returns_float(self) -> None:
        """Similarity should return a Python float, not a tensor."""
        a = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0]]))
        b = Concept(embedding=torch.tensor([[0.5, 0.5, 0.0]]))

        similarity = a.similarity(b)

        assert isinstance(similarity, float)


class TestConceptSaveLoad:
    """Tests for save and load operations."""

    def test_save_and_load_roundtrip(self) -> None:
        """Saving and loading should preserve embedding and description."""
        original = Concept(
            embedding=torch.tensor([[0.6, 0.8, 0.0]]),
            description="test concept",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "concept.pt"
            original.save(path)
            loaded = Concept.load(path)

        assert torch.allclose(loaded.embedding, original.embedding)
        assert loaded.description == original.description

    def test_save_creates_file(self) -> None:
        """Save should create a file at the specified path."""
        concept = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0]]))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "concept.pt"
            concept.save(path)

            assert path.exists()

    def test_load_normalizes_embedding(self) -> None:
        """Loaded embedding should be normalized."""
        original = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0]]))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "concept.pt"
            original.save(path)
            loaded = Concept.load(path)

        norm = torch.norm(loaded.embedding, dim=-1)
        assert torch.allclose(norm, torch.tensor([1.0]))

    def test_load_with_string_path(self) -> None:
        """Load should accept string path."""
        original = Concept(
            embedding=torch.tensor([[1.0, 0.0, 0.0]]),
            description="test",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "concept.pt")
            original.save(path)
            loaded = Concept.load(path)

        assert torch.allclose(loaded.embedding, original.embedding)


class TestConceptTo:
    """Tests for device movement."""

    def test_to_returns_new_concept(self) -> None:
        """to() should return a new Concept, not modify in place."""
        original = Concept(
            embedding=torch.tensor([[1.0, 0.0, 0.0]]),
            description="test",
        )

        result = original.to("cpu")

        assert result is not original

    def test_to_preserves_description(self) -> None:
        """to() should preserve the description."""
        original = Concept(
            embedding=torch.tensor([[1.0, 0.0, 0.0]]),
            description="test description",
        )

        result = original.to("cpu")

        assert result.description == original.description

    def test_to_preserves_embedding_values(self) -> None:
        """to() should preserve embedding values."""
        original = Concept(embedding=torch.tensor([[0.6, 0.8, 0.0]]))

        result = original.to("cpu")

        assert torch.allclose(result.embedding.cpu(), original.embedding.cpu())

    def test_to_changes_device(self) -> None:
        """to() should move embedding to specified device."""
        original = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0]]))

        result = original.to("cpu")

        assert result.embedding.device.type == "cpu"


class TestConceptRepr:
    """Tests for string representation."""

    def test_repr_contains_description(self) -> None:
        """__repr__ should include the description."""
        concept = Concept(
            embedding=torch.tensor([[1.0, 0.0, 0.0]]),
            description="my concept",
        )

        repr_str = repr(concept)

        assert "my concept" in repr_str

    def test_repr_contains_dimension(self) -> None:
        """__repr__ should include the embedding dimension."""
        concept = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]))

        repr_str = repr(concept)

        assert "dim=5" in repr_str

    def test_repr_contains_device(self) -> None:
        """__repr__ should include the device."""
        concept = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0]]))

        repr_str = repr(concept)

        assert "device=" in repr_str


# ---------------------------------------------------------------------------
# Concept-as-decomposition tests (v3 aggressive-rewrite addition)
# ---------------------------------------------------------------------------


def _toy_sae(embed_dim: int = 4, n_features: int = 6, k: int = 2):
    """Build a tiny deterministic SAELens for tests.

    The toy SAE has identity-style behaviour: the first ``embed_dim`` columns
    of W_dec are unit vectors so reconstruct(features) ≈ features (truncated).
    """
    from embedding_art.sae.lens import SAELens

    W_enc = torch.eye(n_features, embed_dim)  # noqa: N806
    W_dec = torch.eye(embed_dim, n_features)  # noqa: N806
    bias = torch.zeros(n_features)
    pre_bias = torch.zeros(embed_dim)
    vocab = [f"feat_{i}" for i in range(n_features)]
    return SAELens.from_tensors(
        W_enc=W_enc, W_dec=W_dec, bias=bias, pre_bias=pre_bias, vocab=vocab, k=k
    )


class TestConceptHasDecomposition:
    """``has_decomposition`` reflects whether an SAE annotation is attached."""

    def test_default_concept_has_no_decomposition(self) -> None:
        concept = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        assert concept.has_decomposition is False
        assert concept.decomposition is None

    def test_concept_with_decomposition_reports_true(self) -> None:
        sae = _toy_sae()
        concept = Concept.from_features(sae, {"feat_0": 1.0, "feat_1": 0.5})
        assert concept.has_decomposition is True
        assert concept.decomposition is not None


class TestConceptFromDecomposition:
    """``from_decomposition`` builds a concept whose embedding is derived from
    the SAE reconstruction."""

    def test_from_decomposition_carries_decomposition(self) -> None:
        sae = _toy_sae()
        from embedding_art.sae.lens import SAEDecomposition

        acts = torch.zeros(1, 6)
        acts[0, 0] = 1.0
        acts[0, 1] = 0.5
        decomp = SAEDecomposition(
            activations=acts,
            active_features={"feat_0": 1.0, "feat_1": 0.5},
            reconstruction_error=0.0,
        )
        concept = Concept.from_decomposition(decomp, sae, description="test")
        assert concept.has_decomposition is True
        assert concept.description == "test"

    def test_from_decomposition_embedding_is_normalized(self) -> None:
        sae = _toy_sae()
        from embedding_art.sae.lens import SAEDecomposition

        acts = torch.zeros(1, 6)
        acts[0, 0] = 3.0
        acts[0, 1] = 4.0
        decomp = SAEDecomposition(activations=acts, active_features={}, reconstruction_error=0.0)
        concept = Concept.from_decomposition(decomp, sae)
        norm = torch.norm(concept.embedding, dim=-1)
        assert torch.allclose(norm, torch.tensor([1.0]), atol=1e-5)


class TestConceptWithDecomposition:
    """``with_decomposition`` adds an SAE annotation without changing the
    embedding."""

    def test_with_decomposition_preserves_embedding(self) -> None:
        sae = _toy_sae()
        concept = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        decomp = concept.decompose(sae)
        annotated = concept.with_decomposition(decomp)
        assert torch.allclose(annotated.embedding, concept.embedding)

    def test_with_decomposition_adds_decomposition(self) -> None:
        sae = _toy_sae()
        concept = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        decomp = concept.decompose(sae)
        annotated = concept.with_decomposition(decomp)
        assert annotated.has_decomposition is True


class TestConceptFeatureSpaceArithmetic:
    """Arithmetic composes decompositions when both operands have them."""

    def test_addition_composes_decompositions(self) -> None:
        sae = _toy_sae()
        a = Concept.from_features(sae, {"feat_0": 1.0})
        b = Concept.from_features(sae, {"feat_1": 1.0})
        result = a + b
        assert result.has_decomposition is True
        # feat_0 and feat_1 should both be active in the sum
        assert "feat_0" in result.decomposition.active_features
        assert "feat_1" in result.decomposition.active_features

    def test_addition_without_decomposition_does_not_synthesize_one(self) -> None:
        a = Concept(embedding=torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        b = Concept(embedding=torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
        result = a + b
        assert result.has_decomposition is False

    def test_mixed_addition_drops_decomposition(self) -> None:
        """When only one operand has a decomposition, the result has none."""
        sae = _toy_sae()
        a = Concept.from_features(sae, {"feat_0": 1.0})
        b = Concept(embedding=torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
        result = a + b
        assert result.has_decomposition is False

    def test_subtraction_composes_decompositions(self) -> None:
        sae = _toy_sae(n_features=6, k=4)
        a = Concept.from_features(sae, {"feat_0": 1.0, "feat_1": 1.0})
        b = Concept.from_features(sae, {"feat_1": 1.0})
        result = a - b
        assert result.has_decomposition is True

    def test_scalar_multiplication_scales_decomposition(self) -> None:
        sae = _toy_sae()
        concept = Concept.from_features(sae, {"feat_0": 1.0, "feat_1": 0.5})
        scaled = 2.0 * concept
        assert scaled.has_decomposition is True
        # Scaled activations should be doubled.
        feats = scaled.decomposition.active_features
        assert feats.get("feat_0", 0) > 1.5  # ~2.0
        assert feats.get("feat_1", 0) > 0.5  # ~1.0

    def test_negation_drops_decomposition(self) -> None:
        """SAE activations are post-ReLU and can't be meaningfully negated."""
        sae = _toy_sae()
        concept = Concept.from_features(sae, {"feat_0": 1.0})
        negated = -concept
        assert negated.has_decomposition is False


class TestConceptSlerpWithDecomposition:
    """slerp interpolates decompositions in feature space when both operands
    have them."""

    def test_slerp_carries_decomposition(self) -> None:
        sae = _toy_sae()
        a = Concept.from_features(sae, {"feat_0": 1.0})
        b = Concept.from_features(sae, {"feat_1": 1.0})
        mid = Concept.slerp(a, b, t=0.5)
        assert mid.has_decomposition is True


class TestConceptSaveLoadWithDecomposition:
    """save / load roundtrips preserve the decomposition annotation."""

    def test_save_load_preserves_decomposition(self) -> None:
        sae = _toy_sae()
        concept = Concept.from_features(sae, {"feat_0": 1.0, "feat_1": 0.5})
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fh:
            path = Path(fh.name)
        try:
            concept.save(path)
            loaded = Concept.load(path)
            assert loaded.has_decomposition is True
            assert "feat_0" in loaded.decomposition.active_features
        finally:
            path.unlink(missing_ok=True)


class TestConceptReprWithDecomposition:
    """``__repr__`` mentions the feature count when a decomposition is present."""

    def test_repr_mentions_features_when_decomposition_present(self) -> None:
        sae = _toy_sae()
        concept = Concept.from_features(sae, {"feat_0": 1.0, "feat_1": 0.5})
        repr_str = repr(concept)
        assert "features=" in repr_str
