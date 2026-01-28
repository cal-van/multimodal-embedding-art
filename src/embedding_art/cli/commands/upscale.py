import sys
from pathlib import Path

import click
from PIL import Image

from embedding_art.cli.utils import (
    console, 
    handle_exception
)


@click.command()
@click.argument("input", type=click.Path(exists=True))
@click.option(
    "-s",
    "--scale",
    type=click.Choice(["2", "4"]),
    default="4",
    help="Upscaling factor (default: 4)",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Output file path (default: input_upscaled.png)",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device to use (default: mps or from config)",
)
@click.pass_context
def upscale(ctx, input: str, scale: str, output: str | None, device: str | None) -> None:
    """Upscale an image using Real-ESRGAN."""
    loaded_config = ctx.obj.get("config", {}) if ctx.obj else {}
    debug_mode = ctx.obj.get("debug", False) if ctx.obj else False

    input_path = Path(input)

    # Determine output path
    if output is None:
        output_path = input_path.parent / f"{input_path.stem}_upscaled{input_path.suffix}"
    else:
        output_path = Path(output)

    # Get device from config or default
    device = device if device is not None else loaded_config.get("device", "mps")
    scale_int = int(scale)

    try:
        # Import inside try/except to catch failures cleanly
        from embedding_art.upscalers import RealESRGANUpscaler

        console.print(f"[bold]Loading Real-ESRGAN ({scale_int}x)...[/bold]")

        upscaler = RealESRGANUpscaler(device=device, scale=scale_int)

        console.print(f"[bold]Upscaling:[/bold] {input_path}")

        # Load and upscale
        image = Image.open(input_path).convert("RGB")
        upscaled = upscaler.upscale(image)

        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        upscaled.save(output_path)

        console.print(f"[bold green]Saved to {output_path}[/bold green]")
        console.print(
            f"[dim]Input: {image.size[0]}x{image.size[1]} -> Output: {upscaled.size[0]}x{upscaled.size[1]}[/dim]"
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]Upscaling cancelled[/yellow]")
        sys.exit(1)
    except Exception as e:
        handle_exception(e, debug_mode)
        sys.exit(1)
