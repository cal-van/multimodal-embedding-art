import sys
from pathlib import Path

import click
from rich.table import Table

from embedding_art.cli.utils import console, handle_exception


def _print_grid_dry_run_1d(
    concepts: list[tuple[str, float]],
    cols: int,
    output_path: str | None,
    device: str,
    opt_steps: int,
) -> None:
    """Print dry run summary for 1D grid generation."""
    table = Table(title="1D Grid Dry Run Configuration", show_header=False, box=None)
    table.add_column("Setting", style="bold cyan")
    table.add_column("Value", style="white")

    table.add_row("Mode", "1D Interpolation")
    table.add_row("Columns", str(cols))
    table.add_row("Device", device)
    table.add_row("Optimization Steps", str(opt_steps))

    if output_path:
        table.add_row("Output Path", output_path)
    else:
        table.add_row("Output Path", "(auto-generated)")

    console.print(table)
    console.print()

    concepts_table = Table(title="Concepts to Interpolate", show_header=True)
    concepts_table.add_column("Endpoint", style="bold")
    concepts_table.add_column("Text", style="white")
    concepts_table.add_column("Weight", style="cyan")

    for i, (text, weight) in enumerate(concepts):
        endpoint = "Start" if i == 0 else "End" if i == len(concepts) - 1 else f"Point {i}"
        concepts_table.add_row(endpoint, text, str(weight))

    console.print(concepts_table)
    console.print()
    console.print("[dim]Dry run complete. No models were loaded.[/dim]")


def _print_grid_dry_run_2d(
    corners: tuple[str, str, str, str],
    size: int,
    output_path: str | None,
    device: str,
    opt_steps: int,
) -> None:
    """Print dry run summary for 2D grid generation."""
    table = Table(title="2D Grid Dry Run Configuration", show_header=False, box=None)
    table.add_column("Setting", style="bold cyan")
    table.add_column("Value", style="white")

    table.add_row("Mode", "2D Interpolation")
    table.add_row("Grid Size", f"{size}x{size}")
    table.add_row("Total Images", str(size * size))
    table.add_row("Device", device)
    table.add_row("Optimization Steps", str(opt_steps))

    if output_path:
        table.add_row("Output Path", output_path)
    else:
        table.add_row("Output Path", "(auto-generated)")

    console.print(table)
    console.print()

    corners_table = Table(title="Corner Concepts", show_header=True)
    corners_table.add_column("Position", style="bold")
    corners_table.add_column("Concept", style="white")

    corners_table.add_row("Top-Left", corners[0])
    corners_table.add_row("Top-Right", corners[1])
    corners_table.add_row("Bottom-Left", corners[2])
    corners_table.add_row("Bottom-Right", corners[3])

    console.print(corners_table)
    console.print()
    console.print("[dim]Dry run complete. No models were loaded.[/dim]")


@click.command()
@click.option(
    "-t",
    "--target-text",
    multiple=True,
    nargs=2,
    type=(str, float),
    help="Text concept and weight for 1D interpolation (use twice for start/end)",
)
@click.option(
    "--corners",
    nargs=4,
    type=str,
    default=None,
    help="Four corner concepts for 2D interpolation (top-left, top-right, bottom-left, bottom-right)",
)
@click.option(
    "--cols",
    type=int,
    default=5,
    help="Number of columns for 1D grid (default: 5)",
)
@click.option(
    "--size",
    type=int,
    default=3,
    help="Grid size for 2D interpolation (creates size x size grid, default: 3)",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Output file path for the grid image",
)
@click.option(
    "--opt-steps",
    type=int,
    default=500,
    help="Optimization steps per grid cell (default: 500)",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device to use (default: mps or from config)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would be done without running optimization",
)
@click.pass_context
def grid(
    ctx: click.Context,
    target_text: tuple[tuple[str, float], ...],
    corners: tuple[str, str, str, str] | None,
    cols: int,
    size: int,
    output: str | None,
    opt_steps: int,
    device: str | None,
    dry_run: bool,
) -> None:
    """Generate a grid of images by interpolating between concepts."""
    # Get config/debug from context
    loaded_config = ctx.obj.get("config", {}) if ctx.obj else {}
    debug_mode = ctx.obj.get("debug", False) if ctx.obj else False

    # Validate inputs
    has_targets = len(target_text) > 0
    has_corners = corners is not None

    if not has_targets and not has_corners:
        raise click.UsageError("Either -t/--target-text (for 1D) or --corners (for 2D) is required")

    if has_targets and has_corners:
        raise click.UsageError("Cannot use both -t/--target-text and --corners. Choose one mode.")

    if has_targets and len(target_text) < 2:
        raise click.UsageError("1D grid requires at least 2 concepts (use -t twice)")

    if has_corners and len(corners) != 4:
        raise click.UsageError("--corners requires exactly 4 concepts")

    # Get device from config or default
    device = device if device is not None else loaded_config.get("device", "mps")

    if dry_run:
        if has_targets:
            _print_grid_dry_run_1d(
                concepts=list(target_text),
                cols=cols,
                output_path=output,
                device=device,
                opt_steps=opt_steps,
            )
        else:
            _print_grid_dry_run_2d(
                corners=corners,
                size=size,
                output_path=output,
                device=device,
                opt_steps=opt_steps,
            )
        return

    try:
        from embedding_art import Concept, EmbeddingArtEngine, OptimizationConfig
        from embedding_art.core.grid import interpolate_1d_grid, interpolate_2d_grid, stitch_grid
        from embedding_art.encoders import ImageBindEncoder
        from embedding_art.generators import SDXLImageGenerator

        console.print("[bold]Loading models...[/bold]")

        encoder = ImageBindEncoder(device=device)
        generator = SDXLImageGenerator(device=device)

        engine = EmbeddingArtEngine(encoder, device=device)
        engine.register_generator("image", generator)

        config = OptimizationConfig(steps=opt_steps)

        if has_targets:
            # 1D grid mode
            console.print(f"[bold]Generating 1D grid with {cols} columns...[/bold]")

            # Build concepts (use first and last for interpolation, ignore weights for now)
            concept_start = Concept.from_text(target_text[0][0], encoder)
            concept_end = Concept.from_text(target_text[-1][0], encoder)

            console.print(
                f"[dim]Interpolating: {concept_start.description} -> {concept_end.description}[/dim]"
            )

            concepts = interpolate_1d_grid(concept_start, concept_end, cols)

            # Optimize each concept
            images = []
            for i, concept in enumerate(concepts):
                console.print(f"[cyan]Optimizing cell {i + 1}/{cols}...[/cyan]")
                result = engine.optimize(concept, "image", config, progress=True)
                images.append(result.get_final_image(generator))
                console.print(f"[green]  Similarity: {result.final_similarity:.4f}[/green]")

            # Stitch into single row
            grid_image = stitch_grid([images])

        else:
            # 2D grid mode
            console.print(f"[bold]Generating {size}x{size} grid...[/bold]")

            # Build corner concepts
            # corners is not None because has_corners checks it
            assert corners is not None
            corner_concepts = [Concept.from_text(c, encoder) for c in corners]

            console.print(
                f"[dim]Corners: {', '.join(c.description for c in corner_concepts)}[/dim]"
            )

            concepts_grid = interpolate_2d_grid(corner_concepts, size)

            # Optimize each concept in the grid
            images_grid = []
            total = size * size
            count = 0
            for row_idx, row in enumerate(concepts_grid):
                row_images = []
                for col_idx, concept in enumerate(row):
                    count += 1
                    console.print(
                        f"[cyan]Optimizing cell ({row_idx + 1},{col_idx + 1}) [{count}/{total}]...[/cyan]"
                    )
                    result = engine.optimize(concept, "image", config, progress=True)
                    row_images.append(result.get_final_image(generator))
                    console.print(f"[green]  Similarity: {result.final_similarity:.4f}[/green]")
                images_grid.append(row_images)

            # Stitch into grid
            grid_image = stitch_grid(images_grid)

        # Save output
        if output is None:
            if has_targets:
                output = "outputs/grid_1d.png"
            else:
                output = "outputs/grid_2d.png"

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        grid_image.save(output_path)

        console.print(f"[bold green]Saved grid to {output_path}[/bold green]")
        console.print(f"[dim]Grid size: {grid_image.size[0]}x{grid_image.size[1]}[/dim]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Grid generation cancelled[/yellow]")
        sys.exit(1)
    except Exception as e:
        handle_exception(e, debug_mode)
        sys.exit(1)
