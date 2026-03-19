"""
Command-line interface for embedding-art.

Usage:
    embed-art optimize --target-text "goldfish" 1.0 --output image
    embed-art interpolate -a "goldfish" -b "flamingo" -s 10 -o image
"""

import sys
import click
import yaml

from embedding_art.core.config import load_config
from embedding_art.cli.commands.optimize import optimize
from embedding_art.cli.commands.interpolate import interpolate
from embedding_art.cli.commands.grid import grid
from embedding_art.cli.commands.embed import embed, compare
from embedding_art.cli.commands.batch import batch
from embedding_art.cli.commands.upscale import upscale
from embedding_art.cli.commands.visualize import visualize
from embedding_art.cli.commands.web import web
from embedding_art.cli.commands.render import render
from embedding_art.cli.commands.compare import compare as compare_cmd
from embedding_art.cli.commands.decompose import decompose
from embedding_art.cli.commands.sae import sae


@click.group()
@click.version_option(version="0.1.0")
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Show full tracebacks on errors",
)
@click.option(
    "--config",
    type=click.Path(exists=False),
    default=None,
    help="Path to YAML config file (default: config.yaml in current directory)",
)
@click.pass_context
def cli(ctx: click.Context, debug: bool, config: str | None) -> None:
    """Generate art by optimizing toward coordinates in multimodal embedding space."""
    # Ensure ctx.obj exists (it might be None)
    ctx.ensure_object(dict)
    
    ctx.obj["debug"] = debug

    # Load config file
    try:
        loaded_config = load_config(config)
        ctx.obj["config"] = loaded_config
    except yaml.YAMLError as e:
        raise click.UsageError(f"Invalid YAML in config file: {e}") from e


# Register commands
cli.add_command(optimize)
cli.add_command(interpolate)
cli.add_command(grid)
cli.add_command(embed)
cli.add_command(compare)
cli.add_command(batch)
cli.add_command(upscale)
cli.add_command(visualize)
cli.add_command(web)
cli.add_command(render)
cli.add_command(compare_cmd, name="compare-encoders")
cli.add_command(decompose)
cli.add_command(sae)


if __name__ == "__main__":
    cli()
