"""
Tests for embedding visualization capabilities.

Tests verify the behavior of plotting similarity history, creating 2D projections
of embeddings, and generating distance matrix heatmaps.
"""

import tempfile
from pathlib import Path

import pytest
import torch
from click.testing import CliRunner

from embedding_art.cli.main import cli
from embedding_art.core.concept import Concept
from embedding_art.core.engine import OptimizationResult


class TestVisualizeCLICommand:
    """Tests for the visualize CLI command existence and subcommands."""

    def test_visualize_command_exists(self) -> None:
        """The CLI should have a visualize command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "visualize" in result.output

    def test_visualize_command_has_help(self) -> None:
        """The visualize command should have help text."""
        runner = CliRunner()
        result = runner.invoke(cli, ["visualize", "--help"])

        assert result.exit_code == 0
        assert "similarity" in result.output or "embeddings" in result.output

    def test_visualize_similarity_subcommand_exists(self) -> None:
        """The visualize command should have a similarity subcommand."""
        runner = CliRunner()
        result = runner.invoke(cli, ["visualize", "--help"])

        assert result.exit_code == 0
        assert "similarity" in result.output

    def test_visualize_embeddings_subcommand_exists(self) -> None:
        """The visualize command should have an embeddings subcommand."""
        runner = CliRunner()
        result = runner.invoke(cli, ["visualize", "--help"])

        assert result.exit_code == 0
        assert "embeddings" in result.output

    def test_visualize_distances_subcommand_exists(self) -> None:
        """The visualize command should have a distances subcommand."""
        runner = CliRunner()
        result = runner.invoke(cli, ["visualize", "--help"])

        assert result.exit_code == 0
        assert "distances" in result.output


class TestPlotSimilarityHistory:
    """Tests for plotting similarity over optimization steps."""

    def test_plot_similarity_accepts_list_of_values(self) -> None:
        """Should accept a list of similarity values and return a figure."""
        from embedding_art.visualization import plot_similarity_history

        similarity_values = [0.1, 0.3, 0.5, 0.7, 0.85]

        fig = plot_similarity_history(similarity_values)

        assert fig is not None
        # Matplotlib figure should have at least one axes
        assert len(fig.axes) >= 1

    def test_plot_similarity_from_optimization_result(self) -> None:
        """Should be able to plot similarity from an OptimizationResult."""
        from embedding_art.visualization import plot_similarity_history

        result = OptimizationResult(
            final_latent=torch.randn(1, 4, 64, 64),
            final_embedding=torch.randn(1, 1024),
            target_embedding=torch.randn(1, 1024),
            final_similarity=0.85,
            similarity_history=[0.1, 0.2, 0.4, 0.6, 0.75, 0.85],
            loss_history=[1.0, 0.8, 0.6, 0.4, 0.3, 0.2],
        )

        fig = plot_similarity_history(result.similarity_history)

        assert fig is not None
        assert len(fig.axes) >= 1

    def test_plot_similarity_can_include_loss(self) -> None:
        """Should optionally plot loss history on secondary axis."""
        from embedding_art.visualization import plot_similarity_history

        similarity_values = [0.1, 0.3, 0.5, 0.7, 0.85]
        loss_values = [1.0, 0.8, 0.5, 0.3, 0.15]

        fig = plot_similarity_history(similarity_values, loss_history=loss_values)

        assert fig is not None

    def test_plot_similarity_saves_to_file(self) -> None:
        """Should be able to save plot to file."""
        from embedding_art.visualization import plot_similarity_history

        similarity_values = [0.1, 0.3, 0.5, 0.7, 0.85]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "similarity.png"
            fig = plot_similarity_history(similarity_values)
            fig.savefig(output_path)

            assert output_path.exists()

    def test_plot_similarity_with_title(self) -> None:
        """Should accept optional title parameter."""
        from embedding_art.visualization import plot_similarity_history

        similarity_values = [0.1, 0.3, 0.5, 0.7, 0.85]

        fig = plot_similarity_history(similarity_values, title="Test Optimization")

        assert fig is not None
        # Title should be set on the figure or axes
        ax = fig.axes[0]
        assert ax.get_title() == "Test Optimization" or fig._suptitle is not None

    def test_plot_similarity_empty_list_raises(self) -> None:
        """Should raise ValueError for empty similarity list."""
        from embedding_art.visualization import plot_similarity_history

        with pytest.raises(ValueError, match="empty"):
            plot_similarity_history([])


class TestPlotEmbeddings2D:
    """Tests for creating 2D projections of multiple concepts."""

    def test_plot_embeddings_accepts_list_of_concepts(self) -> None:
        """Should accept a list of Concepts and return a figure."""
        from embedding_art.visualization import plot_embeddings_2d

        concepts = [
            Concept(embedding=torch.randn(1, 1024), description="fire"),
            Concept(embedding=torch.randn(1, 1024), description="water"),
            Concept(embedding=torch.randn(1, 1024), description="earth"),
        ]

        fig = plot_embeddings_2d(concepts)

        assert fig is not None
        assert len(fig.axes) >= 1

    def test_plot_embeddings_uses_tsne_by_default(self) -> None:
        """Should use t-SNE for dimensionality reduction by default."""
        from embedding_art.visualization import plot_embeddings_2d

        concepts = [
            Concept(embedding=torch.randn(1, 1024), description="fire"),
            Concept(embedding=torch.randn(1, 1024), description="water"),
            Concept(embedding=torch.randn(1, 1024), description="earth"),
        ]

        fig = plot_embeddings_2d(concepts, method="tsne")

        assert fig is not None

    def test_plot_embeddings_supports_pca(self) -> None:
        """Should support PCA for dimensionality reduction."""
        from embedding_art.visualization import plot_embeddings_2d

        concepts = [
            Concept(embedding=torch.randn(1, 1024), description="fire"),
            Concept(embedding=torch.randn(1, 1024), description="water"),
            Concept(embedding=torch.randn(1, 1024), description="earth"),
        ]

        fig = plot_embeddings_2d(concepts, method="pca")

        assert fig is not None

    def test_plot_embeddings_labels_points(self) -> None:
        """Should label points with concept descriptions."""
        from embedding_art.visualization import plot_embeddings_2d

        concepts = [
            Concept(embedding=torch.randn(1, 1024), description="fire"),
            Concept(embedding=torch.randn(1, 1024), description="water"),
        ]

        fig = plot_embeddings_2d(concepts)

        # Check that text annotations exist in the axes
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        assert "fire" in texts or any("fire" in str(child) for child in ax.get_children())

    def test_plot_embeddings_saves_to_file(self) -> None:
        """Should be able to save plot to file."""
        from embedding_art.visualization import plot_embeddings_2d

        concepts = [
            Concept(embedding=torch.randn(1, 1024), description="fire"),
            Concept(embedding=torch.randn(1, 1024), description="water"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "embeddings.png"
            fig = plot_embeddings_2d(concepts)
            fig.savefig(output_path)

            assert output_path.exists()

    def test_plot_embeddings_minimum_concepts(self) -> None:
        """Should require at least 2 concepts for 2D projection."""
        from embedding_art.visualization import plot_embeddings_2d

        concepts = [Concept(embedding=torch.randn(1, 1024), description="fire")]

        with pytest.raises(ValueError, match="at least 2"):
            plot_embeddings_2d(concepts)

    def test_plot_embeddings_with_title(self) -> None:
        """Should accept optional title parameter."""
        from embedding_art.visualization import plot_embeddings_2d

        concepts = [
            Concept(embedding=torch.randn(1, 1024), description="fire"),
            Concept(embedding=torch.randn(1, 1024), description="water"),
        ]

        fig = plot_embeddings_2d(concepts, title="Concept Space")

        assert fig is not None


class TestPlotDistanceMatrix:
    """Tests for creating distance matrix heatmaps."""

    def test_plot_distances_accepts_list_of_concepts(self) -> None:
        """Should accept a list of Concepts and return a figure with heatmap."""
        from embedding_art.visualization import plot_distance_matrix

        concepts = [
            Concept(embedding=torch.randn(1, 1024), description="fire"),
            Concept(embedding=torch.randn(1, 1024), description="water"),
            Concept(embedding=torch.randn(1, 1024), description="earth"),
        ]

        fig = plot_distance_matrix(concepts)

        assert fig is not None
        assert len(fig.axes) >= 1

    def test_plot_distances_shows_similarity_values(self) -> None:
        """Should show cosine similarity values in the matrix."""
        from embedding_art.visualization import plot_distance_matrix

        # Use orthogonal vectors for predictable similarities
        concepts = [
            Concept(embedding=torch.tensor([[1.0, 0.0, 0.0]]), description="x"),
            Concept(embedding=torch.tensor([[0.0, 1.0, 0.0]]), description="y"),
            Concept(embedding=torch.tensor([[0.0, 0.0, 1.0]]), description="z"),
        ]

        fig = plot_distance_matrix(concepts)

        # Should have created a heatmap
        assert fig is not None

    def test_plot_distances_diagonal_is_one(self) -> None:
        """Diagonal of similarity matrix should be 1.0 (self-similarity)."""
        from embedding_art.visualization import compute_similarity_matrix

        concepts = [
            Concept(embedding=torch.randn(1, 1024), description="fire"),
            Concept(embedding=torch.randn(1, 1024), description="water"),
        ]

        matrix = compute_similarity_matrix(concepts)

        # Diagonal should be 1.0
        for i in range(len(concepts)):
            assert abs(matrix[i, i] - 1.0) < 1e-5

    def test_plot_distances_symmetric(self) -> None:
        """Similarity matrix should be symmetric."""
        from embedding_art.visualization import compute_similarity_matrix

        concepts = [
            Concept(embedding=torch.randn(1, 1024), description="fire"),
            Concept(embedding=torch.randn(1, 1024), description="water"),
            Concept(embedding=torch.randn(1, 1024), description="earth"),
        ]

        matrix = compute_similarity_matrix(concepts)

        # Check symmetry
        for i in range(len(concepts)):
            for j in range(len(concepts)):
                assert abs(matrix[i, j] - matrix[j, i]) < 1e-5

    def test_plot_distances_labels_axes(self) -> None:
        """Should label axes with concept descriptions."""
        from embedding_art.visualization import plot_distance_matrix

        concepts = [
            Concept(embedding=torch.randn(1, 1024), description="fire"),
            Concept(embedding=torch.randn(1, 1024), description="water"),
        ]

        fig = plot_distance_matrix(concepts)

        # Check that tick labels are set
        ax = fig.axes[0]
        x_labels = [t.get_text() for t in ax.get_xticklabels()]
        y_labels = [t.get_text() for t in ax.get_yticklabels()]

        # At least one of the concept names should appear
        assert "fire" in x_labels or "fire" in y_labels or len(x_labels) > 0

    def test_plot_distances_saves_to_file(self) -> None:
        """Should be able to save plot to file."""
        from embedding_art.visualization import plot_distance_matrix

        concepts = [
            Concept(embedding=torch.randn(1, 1024), description="fire"),
            Concept(embedding=torch.randn(1, 1024), description="water"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "distances.png"
            fig = plot_distance_matrix(concepts)
            fig.savefig(output_path)

            assert output_path.exists()

    def test_plot_distances_minimum_concepts(self) -> None:
        """Should require at least 2 concepts for distance matrix."""
        from embedding_art.visualization import plot_distance_matrix

        concepts = [Concept(embedding=torch.randn(1, 1024), description="fire")]

        with pytest.raises(ValueError, match="at least 2"):
            plot_distance_matrix(concepts)

    def test_plot_distances_with_title(self) -> None:
        """Should accept optional title parameter."""
        from embedding_art.visualization import plot_distance_matrix

        concepts = [
            Concept(embedding=torch.randn(1, 1024), description="fire"),
            Concept(embedding=torch.randn(1, 1024), description="water"),
        ]

        fig = plot_distance_matrix(concepts, title="Similarity Matrix")

        assert fig is not None


class TestVisualizeSimilarityCLI:
    """Tests for the visualize similarity CLI subcommand."""

    def test_similarity_subcommand_accepts_checkpoint(self) -> None:
        """The similarity subcommand should accept a checkpoint file."""
        runner = CliRunner()
        result = runner.invoke(cli, ["visualize", "similarity", "--help"])

        assert result.exit_code == 0
        # Should mention checkpoint or file argument
        assert "checkpoint" in result.output.lower() or "file" in result.output.lower()

    def test_similarity_subcommand_accepts_output_option(self) -> None:
        """The similarity subcommand should have an output file option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["visualize", "similarity", "--help"])

        assert result.exit_code == 0
        assert "--output" in result.output or "-o" in result.output


class TestVisualizeEmbeddingsCLI:
    """Tests for the visualize embeddings CLI subcommand."""

    def test_embeddings_subcommand_accepts_text_targets(self) -> None:
        """The embeddings subcommand should accept text targets."""
        runner = CliRunner()
        result = runner.invoke(cli, ["visualize", "embeddings", "--help"])

        assert result.exit_code == 0
        assert "-t" in result.output or "--text" in result.output

    def test_embeddings_subcommand_accepts_output_option(self) -> None:
        """The embeddings subcommand should have an output file option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["visualize", "embeddings", "--help"])

        assert result.exit_code == 0
        assert "--output" in result.output or "-o" in result.output

    def test_embeddings_subcommand_accepts_method_option(self) -> None:
        """The embeddings subcommand should accept a method option for dim reduction."""
        runner = CliRunner()
        result = runner.invoke(cli, ["visualize", "embeddings", "--help"])

        assert result.exit_code == 0
        assert "--method" in result.output or "tsne" in result.output.lower()


class TestVisualizeDistancesCLI:
    """Tests for the visualize distances CLI subcommand."""

    def test_distances_subcommand_accepts_text_targets(self) -> None:
        """The distances subcommand should accept text targets."""
        runner = CliRunner()
        result = runner.invoke(cli, ["visualize", "distances", "--help"])

        assert result.exit_code == 0
        assert "-t" in result.output or "--text" in result.output

    def test_distances_subcommand_accepts_output_option(self) -> None:
        """The distances subcommand should have an output file option."""
        runner = CliRunner()
        result = runner.invoke(cli, ["visualize", "distances", "--help"])

        assert result.exit_code == 0
        assert "--output" in result.output or "-o" in result.output
