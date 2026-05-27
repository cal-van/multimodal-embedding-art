import click
import uvicorn

from embedding_art.cli.utils import console


@click.command()
@click.option(
    "--port",
    type=int,
    default=8000,
    help="Port to run the server on (default: 8000)",
)
@click.option(
    "--host",
    type=str,
    default="127.0.0.1",
    help="Host to run the server on (default: 127.0.0.1)",
)
@click.option(
    "--reload",
    is_flag=True,
    default=False,
    help="Enable auto-reload for development",
)
def web(port: int, host: str, reload: bool) -> None:
    """Start the Web UI server."""
    console.print(f"[bold green]Starting Web UI on http://{host}:{port}[/bold green]")

    uvicorn.run(
        "embedding_art.web.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
