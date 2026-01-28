"""
CLI utility functions.
"""

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from PIL import Image

console = Console()
DEFAULT_VIDEO_FPS = 8

def handle_exception(e: Exception, debug_mode: bool = False) -> None:
    """Handle an exception with user-friendly output."""
    from embedding_art.exceptions import (
        EmbeddingArtError,
        ImageBindNotInstalledError,
        ModelLoadError,
        OutOfMemoryError,
        UpscalerError,
    )

    if debug_mode:
        console.print_exception()
        return

    if isinstance(e, ImageBindNotInstalledError):
        console.print(f"[bold red]Error:[/bold red] {e}")
    elif isinstance(e, OutOfMemoryError):
        console.print(f"[bold red]Out of Memory:[/bold red] {e}")
    elif isinstance(e, ModelLoadError):
        console.print(f"[bold red]Model Load Failed:[/bold red] {e}")
    elif isinstance(e, UpscalerError):
        console.print(f"[bold red]Upscaling Failed:[/bold red] {e}")
    elif isinstance(e, EmbeddingArtError):
        console.print(f"[bold red]Error:[/bold red] {e}")
    elif isinstance(e, click.UsageError):
        raise e
    else:
        console.print(f"[bold red]Unexpected Error:[/bold red] {type(e).__name__}: {e}")
        console.print("[dim]Use --debug for full traceback[/dim]")


def save_video(frames: list[Image.Image], output_path: Path, fps: int = DEFAULT_VIDEO_FPS) -> None:
    """Save a list of PIL images as a video/GIF."""
    if not frames:
        raise ValueError("No frames to save")

    duration_ms = int(1000 / fps) if fps > 0 else 125
    first_frame = frames[0].convert("RGB")
    extra_frames = [frame.convert("RGB") for frame in frames[1:]]
    first_frame.save(
        output_path,
        save_all=True,
        append_images=extra_frames,
        duration=duration_ms,
        loop=0,
    )


def print_dry_run_summary(
    target_text: tuple[tuple[str, float], ...],
    target_image: tuple[tuple[str, float], ...],
    target_audio: tuple[tuple[str, float], ...],
    output_modality: str,
    output_path: str | None,
    steps: int,
    lr: float,
    seed: int | None,
    device: str,
) -> None:
    """Print a summary of what optimization would do without running it."""
    table = Table(title="Dry Run Configuration", show_header=False, box=None)
    table.add_column("Setting", style="bold cyan")
    table.add_column("Value", style="white")

    table.add_row("Device", device)
    table.add_row("Steps", str(steps))
    table.add_row("Learning Rate", str(lr))
    table.add_row("Output Modality", output_modality)

    if output_path:
        table.add_row("Output Path", output_path)
    else:
        table.add_row("Output Path", "(auto-generated)")

    if seed is not None:
        table.add_row("Seed", str(seed))
    else:
        table.add_row("Seed", "(random)")

    console.print(table)
    console.print()

    # Print target concepts
    concepts_table = Table(title="Target Concepts", show_header=True)
    concepts_table.add_column("Type", style="bold")
    concepts_table.add_column("Value", style="white")
    concepts_table.add_column("Weight", style="cyan")

    for text, weight in target_text:
        concepts_table.add_row("Text", text, str(weight))

    for path, weight in target_image:
        concepts_table.add_row("Image", str(path), str(weight))

    for path, weight in target_audio:
        concepts_table.add_row("Audio", str(path), str(weight))

    console.print(concepts_table)
    console.print()
    console.print("[dim]Dry run complete. No models were loaded.[/dim]")
