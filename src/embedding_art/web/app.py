from contextlib import asynccontextmanager
from typing import AsyncGenerator
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from rich.console import Console

from embedding_art.web.routes import jobs

console = Console()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan."""
    console.print("[bold green]Web UI Backend Starting...[/bold green]")
    yield
    console.print("[bold yellow]Web UI Backend Shutting Down...[/bold yellow]")


app = FastAPI(
    title="Embedding Art UI",
    description="Web interface for Multimodal Embedding Art",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS Configuration
# Restrict origins for security (no wildcard with credentials)
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",  # Vite default
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs.router)

# Ensure outputs directory exists
OUTPUTS_DIR = os.path.join(os.getcwd(), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Mount outputs directory
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "embedding-art"}


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint info."""
    return {
        "message": "Embedding Art API",
        "docs": "/docs",
        "health": "/health",
    }
