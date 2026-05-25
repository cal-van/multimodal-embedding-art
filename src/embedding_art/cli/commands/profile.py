"""``embed-art profile`` — Apple Silicon perf profiling harness.

Runs a representative forward + loss + backward step under
``torch.profiler`` and emits a Chrome trace + per-op summary table.

The harness uses *synthetic* tensors and a no-weight ``Linear`` proxy
encoder by default so the measurement runs in seconds rather than
loading real model weights. Real encoder profiling is opt-in via
``--encoder languagebind`` (requires LanguageBind installed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import torch

from embedding_art.cli.utils import console, handle_exception


@click.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="outputs/profile",
    show_default=True,
    help="Directory to write the Chrome trace and summary table.",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "mps", "cuda"]),
    default="mps" if torch.backends.mps.is_available() else "cpu",
    show_default=True,
    help="Device to profile on. 'mps' is the canonical M1/M2 Max target.",
)
@click.option(
    "--steps",
    type=int,
    default=10,
    show_default=True,
    help="Number of forward+backward iterations to profile.",
)
@click.option(
    "--compile-mode",
    type=click.Choice(["none", "default", "reduce-overhead", "max-autotune"]),
    default="none",
    show_default=True,
    help="Wrap the loss in torch.compile to measure the speedup.",
)
@click.option(
    "--autocast-dtype",
    type=click.Choice(["fp32", "fp16", "bf16"]),
    default="fp32",
    show_default=True,
    help="Mixed-precision dtype for the forward+loss section.",
)
@click.option(
    "--encoder",
    type=click.Choice(["synthetic", "languagebind"]),
    default="synthetic",
    show_default=True,
    help="Encoder to profile. 'synthetic' uses a no-weight Linear proxy "
    "and runs in seconds. 'languagebind' loads the real image encoder.",
)
@click.option(
    "--top-k",
    type=int,
    default=20,
    show_default=True,
    help="Number of top hotspot ops to include in the summary table.",
)
@click.pass_context
def profile(
    ctx: click.Context,
    output: str,
    device: str,
    steps: int,
    compile_mode: str,
    autocast_dtype: str,
    encoder: str,
    top_k: int,
) -> None:
    """Profile a forward+backward step under torch.profiler."""
    debug_mode = ctx.obj.get("debug", False) if ctx.obj else False

    try:
        _profile_impl(
            output_dir=Path(output),
            device=device,
            steps=steps,
            compile_mode=compile_mode,
            autocast_dtype=autocast_dtype,
            encoder_name=encoder,
            top_k=top_k,
        )
    except Exception as e:
        handle_exception(e, debug_mode)
        sys.exit(1)


def _profile_impl(
    *,
    output_dir: Path,
    device: str,
    steps: int,
    compile_mode: str,
    autocast_dtype: str,
    encoder_name: str,
    top_k: int,
) -> None:
    from embedding_art.perf import PerfProfile, compile_module, empty_mps_cache

    output_dir.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(device)
    console.print(f"[bold cyan]Profiling on {device} for {steps} steps[/bold cyan]")

    embed_dim = 768
    seq_len = 256

    if encoder_name == "synthetic":
        encoder = torch.nn.Linear(embed_dim, embed_dim).to(torch_device).eval()

        def encode(x: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.normalize(encoder(x.mean(dim=1)), dim=-1)

    elif encoder_name == "languagebind":
        from embedding_art.encoders.languagebind import LanguageBindEncoder

        lb = LanguageBindEncoder(device=device, eager_load=("image",))

        def encode(x: torch.Tensor) -> torch.Tensor:
            return lb.encode_for_optimization(x)

    else:  # pragma: no cover - click already validates
        raise ValueError(f"Unknown encoder {encoder_name}")

    target = torch.nn.functional.normalize(torch.randn(1, embed_dim, device=torch_device), dim=-1)

    def loss_step(x: torch.Tensor) -> torch.Tensor:
        emb = encode(x)
        return -(emb * target).sum(dim=-1).mean()

    compiled_loss = (
        compile_module(loss_step, mode=compile_mode) if compile_mode != "none" else loss_step
    )

    autocast_dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    autocast_torch_dtype = autocast_dtype_map[autocast_dtype]
    autocast_enabled = autocast_dtype != "fp32"
    autocast_device_type = "cuda" if device == "cuda" else "cpu"
    if device == "mps":
        autocast_device_type = "cpu"  # torch.autocast on MPS routes via CPU dtype

    # Warm-up: trigger any lazy compilation outside the timed window.
    x_warm = torch.randn(1, seq_len, embed_dim, device=torch_device, requires_grad=True)
    try:
        _ = compiled_loss(x_warm)
    except Exception as exc:  # pragma: no cover - defensive
        console.print(f"[yellow]warmup failed: {exc}[/yellow]")

    with PerfProfile(
        output_dir=output_dir,
        record_shapes=False,
        with_stack=False,
    ) as prof:
        for _ in range(steps):
            x = torch.randn(1, seq_len, embed_dim, device=torch_device, requires_grad=True)
            if autocast_enabled:
                with torch.autocast(device_type=autocast_device_type, dtype=autocast_torch_dtype):
                    loss = compiled_loss(x)
            else:
                loss = compiled_loss(x)
            loss.backward()

        summary = prof.summary(row_limit=top_k)

    empty_mps_cache()

    summary_path = output_dir / "summary.txt"
    summary_path.write_text(summary)
    console.print("[bold green]Profile complete[/bold green]")
    console.print(f"Trace : {output_dir / 'trace.json'}")
    console.print(f"Summary: {summary_path}")
