"""
CLI command for v2 interpolation between two concepts.

Renders a slerp interpolation series using the v2 engine and registry,
saving each frame as a PNG image.
"""

from __future__ import annotations

import click


@click.command("interpolate-v2")
@click.option(
    "-a", "--concept-a", required=True, help="First concept text (start of interpolation)"
)
@click.option("-b", "--concept-b", required=True, help="Second concept text (end of interpolation)")
@click.option("-s", "--steps", default=10, show_default=True, help="Number of interpolation steps")
@click.option(
    "-o",
    "--output",
    "output_modality",
    default="image",
    show_default=True,
    help="Output modality",
)
@click.option(
    "--encoder",
    "encoder_name",
    default="imagebind",
    show_default=True,
    help="Encoder name from the default registry",
)
@click.option(
    "--output-dir",
    default="outputs/interpolation",
    show_default=True,
    help="Directory in which to save output frames",
)
@click.option(
    "--opt-steps",
    default=500,
    show_default=True,
    help="Optimization steps per interpolation frame",
)
@click.option(
    "--device",
    default="mps",
    show_default=True,
    help="PyTorch device (e.g. mps, cuda, cpu)",
)
def interpolate_v2(
    concept_a: str,
    concept_b: str,
    steps: int,
    output_modality: str,
    encoder_name: str,
    output_dir: str,
    opt_steps: int,
    device: str,
) -> None:
    """Generate an interpolation series between two concepts using the v2 engine."""
    import datetime
    from pathlib import Path

    from torchvision.transforms.functional import to_pil_image

    from embedding_art.core.concept_spec import ConceptSpec
    from embedding_art.core.config import OptimizationConfig
    from embedding_art.core.engine import EmbeddingArtEngine
    from embedding_art.encoders.defaults import create_default_registry
    from embedding_art.generators.image import SDXLImageGenerator

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir)
    output_path = output_path.parent / f"{output_path.name}_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)

    registry = create_default_registry()
    engine = EmbeddingArtEngine.from_registry(registry, default_encoder=encoder_name, device=device)
    engine.register_generator(output_modality, SDXLImageGenerator(device=device))

    spec_a = ConceptSpec(text=concept_a)
    spec_b = ConceptSpec(text=concept_b)
    config = OptimizationConfig(steps=opt_steps)

    click.echo(f"Interpolating '{concept_a}' → '{concept_b}' in {steps} steps...")

    results = engine.render_interpolation(
        spec_a,
        spec_b,
        steps=steps,
        encoder_name=encoder_name,
        output_modality=output_modality,
        config=config,
    )

    for i, result in enumerate(results):
        t = i / max(steps - 1, 1)
        img = to_pil_image(result.output.squeeze(0).clamp(0, 1))
        img.save(output_path / f"frame_{i:04d}_t{t:.3f}.png")
        click.echo(
            f"  Frame {i}/{steps - 1} (t={t:.3f}) — similarity: {result.final_similarity:.4f}"
        )

    click.echo(f"Done. {steps} frames saved to {output_path}/")
