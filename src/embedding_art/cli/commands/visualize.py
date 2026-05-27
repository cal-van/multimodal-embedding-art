import sys
from pathlib import Path

import click

from embedding_art.cli.utils import console, handle_exception


@click.group()
def visualize() -> None:
    """Visualize embeddings and optimization progress."""
    pass


@visualize.command()
@click.argument("checkpoint", type=click.Path(exists=True))
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Output file path for the plot (default: show interactively)",
)
@click.option(
    "--show-loss",
    is_flag=True,
    default=False,
    help="Include loss history on secondary axis",
)
@click.option(
    "--title",
    type=str,
    default=None,
    help="Title for the plot",
)
@click.pass_context
def similarity(
    ctx, checkpoint: str, output: str | None, show_loss: bool, title: str | None
) -> None:
    """Plot similarity history from an optimization checkpoint."""
    debug_mode = ctx.obj.get("debug", False) if ctx.obj else False

    try:
        import torch
        from embedding_art.visualization import plot_similarity_history

        # Load checkpoint
        checkpoint_data = torch.load(checkpoint, weights_only=True)

        # Extract similarity history
        if "similarity_history" not in checkpoint_data:
            raise click.UsageError("Checkpoint does not contain similarity_history")

        similarity_history = checkpoint_data["similarity_history"]
        loss_history = checkpoint_data.get("loss_history") if show_loss else None

        # Create plot
        fig = plot_similarity_history(
            similarity_history,
            loss_history=loss_history,
            title=title,
        )

        # Save or show
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150)
            console.print(f"[bold green]Saved plot to {output_path}[/bold green]")
        else:
            import matplotlib.pyplot as plt

            plt.show()

    except click.UsageError:
        raise
    except Exception as e:
        handle_exception(e, debug_mode)
        sys.exit(1)


@visualize.command()
@click.option(
    "-t",
    "--text",
    multiple=True,
    type=str,
    help="Text concept to include (can be specified multiple times)",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Output file path for the plot (default: show interactively)",
)
@click.option(
    "--method",
    type=click.Choice(["tsne", "pca"]),
    default="tsne",
    help="Dimensionality reduction method (default: tsne)",
)
@click.option(
    "--title",
    type=str,
    default=None,
    help="Title for the plot",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device to use for encoding (default: mps or from config)",
)
@click.pass_context
def embeddings(
    ctx,
    text: tuple[str, ...],
    output: str | None,
    method: str,
    title: str | None,
    device: str | None,
) -> None:
    """Create a 2D projection of concept embeddings."""
    loaded_config = ctx.obj.get("config", {}) if ctx.obj else {}
    debug_mode = ctx.obj.get("debug", False) if ctx.obj else False

    if len(text) < 2:
        raise click.UsageError("Need at least 2 text concepts for 2D projection")

    try:
        from embedding_art import Concept
        from embedding_art.encoders import ImageBindEncoder
        from embedding_art.visualization import plot_embeddings_2d

        # Get device from config or default
        device = device if device is not None else loaded_config.get("device", "mps")

        console.print("[bold]Loading encoder...[/bold]")
        encoder = ImageBindEncoder(device=device)

        # Create concepts
        console.print(f"[bold]Encoding {len(text)} concepts...[/bold]")
        concepts = [Concept.from_text(t, encoder) for t in text]

        # Create plot
        console.print(f"[bold]Creating {method.upper()} projection...[/bold]")
        fig = plot_embeddings_2d(concepts, method=method, title=title)

        # Save or show
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150)
            console.print(f"[bold green]Saved plot to {output_path}[/bold green]")
        else:
            import matplotlib.pyplot as plt

            plt.show()

    except click.UsageError:
        raise
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled[/yellow]")
        sys.exit(1)
    except Exception as e:
        handle_exception(e, debug_mode)
        sys.exit(1)


@visualize.command()
@click.option(
    "-t",
    "--text",
    multiple=True,
    type=str,
    help="Text concept to include (can be specified multiple times)",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Output file path for the plot (default: show interactively)",
)
@click.option(
    "--title",
    type=str,
    default=None,
    help="Title for the plot",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device to use for encoding (default: mps or from config)",
)
@click.pass_context
def distances(
    ctx,
    text: tuple[str, ...],
    output: str | None,
    title: str | None,
    device: str | None,
) -> None:
    """Create a distance matrix heatmap showing pairwise similarities."""
    loaded_config = ctx.obj.get("config", {}) if ctx.obj else {}
    debug_mode = ctx.obj.get("debug", False) if ctx.obj else False

    if len(text) < 2:
        raise click.UsageError("Need at least 2 text concepts for distance matrix")

    try:
        from embedding_art import Concept
        from embedding_art.encoders import ImageBindEncoder
        from embedding_art.visualization import plot_distance_matrix

        # Get device from config or default
        device = device if device is not None else loaded_config.get("device", "mps")

        console.print("[bold]Loading encoder...[/bold]")
        encoder = ImageBindEncoder(device=device)

        # Create concepts
        console.print(f"[bold]Encoding {len(text)} concepts...[/bold]")
        concepts = [Concept.from_text(t, encoder) for t in text]

        # Create plot
        console.print("[bold]Computing similarity matrix...[/bold]")
        fig = plot_distance_matrix(concepts, title=title)

        # Save or show
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150)
            console.print(f"[bold green]Saved plot to {output_path}[/bold green]")
        else:
            import matplotlib.pyplot as plt

            plt.show()

    except click.UsageError:
        raise
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled[/yellow]")
        sys.exit(1)
    except Exception as e:
        handle_exception(e, debug_mode)
        sys.exit(1)
