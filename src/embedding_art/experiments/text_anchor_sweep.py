"""
M4 text-anchor empirical sweep (issue iik).

The text-anchor auxiliary loss (``LossConfig.text_anchor_weight``) adds a
cosine-distance term pulling the current optimisation embedding toward
the encoder's text projection of a concept label. The open question is
*how much* anchor signal to mix in. Too little → no auxiliary value;
too much → the optimisation collapses to the language anchor and
abandons the actual target.

This module provides a synthetic, decoder-free sweep harness that
characterises the trade-off curve directly in embedding space. The
forward model is a deliberate idealisation::

    current = learnable embedding  (initialised from a Gaussian)
    loss    = -sim_w * cos(current, target) - anchor_w * cos(current, anchor)

We sweep ``anchor_w`` over a user-supplied grid (with ``sim_w`` fixed by
default), run Adam to convergence, and report::

* The final ``cos(current, target)`` and ``cos(current, anchor)``.
* The Pareto frontier of the two objectives.
* The 'elbow' point — our recommended default — defined as the Pareto
  point that maximises the *minimum* of the two normalised similarities
  (a simple, conservative pick).

This harness runs in O(seconds) on CPU and validates the loss math
exactly without committing us to a particular encoder / decoder. Real-
weight validation against LanguageBind + SD3.5 belongs to a separate
M1-Max-bound script and is referenced from this module's docs.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

logger = logging.getLogger(__name__)


@dataclass
class SweepPoint:
    """One (similarity_weight, anchor_weight) measurement."""

    similarity_weight: float
    anchor_weight: float
    target_similarity: float
    anchor_similarity: float
    final_loss: float
    steps_taken: int

    def dominates(self, other: SweepPoint) -> bool:
        """Pareto dominance on (target_sim, anchor_sim) — higher is better."""
        better_or_equal = (
            self.target_similarity >= other.target_similarity
            and self.anchor_similarity >= other.anchor_similarity
        )
        strictly_better = (
            self.target_similarity > other.target_similarity
            or self.anchor_similarity > other.anchor_similarity
        )
        return better_or_equal and strictly_better


@dataclass
class SweepResult:
    """Result of one sweep, including the recommended default."""

    modality: str
    points: list[SweepPoint] = field(default_factory=list)
    pareto_front: list[SweepPoint] = field(default_factory=list)
    recommended: SweepPoint | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "points": [asdict(p) for p in self.points],
            "pareto_front": [asdict(p) for p in self.pareto_front],
            "recommended": asdict(self.recommended) if self.recommended else None,
        }


def run_text_anchor_sweep(
    *,
    target_embedding: torch.Tensor,
    anchor_embedding: torch.Tensor,
    modality: str = "image",
    similarity_weights: Sequence[float] = (1.0,),
    anchor_weights: Sequence[float] = (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
    steps: int = 400,
    learning_rate: float = 0.05,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> SweepResult:
    """Sweep ``anchor_weight`` against fixed ``similarity_weight``\\(s\\).

    Args:
        target_embedding: The 'true' concept embedding, shape ``(D,)`` or
            ``(1, D)``. Need not be unit norm — we normalise internally.
        anchor_embedding: The text-projection embedding of the concept
            label, same shape as ``target_embedding``.
        modality: Label for the result record (e.g. ``"image"``).
        similarity_weights: Outer grid of similarity weights to sweep
            (default keeps it fixed at 1.0).
        anchor_weights: Inner grid of anchor weights.
        steps: Adam steps per grid point.
        learning_rate: Adam learning rate.
        seed: RNG seed for initialisation.
        device: Torch device.

    Returns:
        Populated :class:`SweepResult` with all grid points, the Pareto
        front, and a recommended default.
    """
    device = torch.device(device)
    target = _prep(target_embedding, device)
    anchor = _prep(anchor_embedding, device)
    if target.shape != anchor.shape:
        raise ValueError(
            f"target {tuple(target.shape)} and anchor {tuple(anchor.shape)} "
            "must have the same shape"
        )

    points: list[SweepPoint] = []
    for sim_w in similarity_weights:
        for anc_w in anchor_weights:
            point = _optimise_one(
                target=target,
                anchor=anchor,
                similarity_weight=float(sim_w),
                anchor_weight=float(anc_w),
                steps=steps,
                learning_rate=learning_rate,
                seed=seed,
                device=device,
            )
            points.append(point)

    pareto = _pareto_front(points)
    recommended = _recommend(pareto)
    return SweepResult(
        modality=modality, points=points, pareto_front=pareto, recommended=recommended
    )


def _prep(emb: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Reshape to ``(D,)`` and unit-normalise."""
    emb = emb.detach().to(device).float()
    if emb.dim() == 2:
        emb = emb.squeeze(0)
    if emb.dim() != 1:
        raise ValueError(f"expected 1D or 2D embedding, got shape {tuple(emb.shape)}")
    return F.normalize(emb, dim=-1)


def _optimise_one(
    *,
    target: torch.Tensor,
    anchor: torch.Tensor,
    similarity_weight: float,
    anchor_weight: float,
    steps: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> SweepPoint:
    """Run Adam on a free embedding for one (sim_w, anc_w) point."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    init = torch.randn(target.shape[0], generator=gen)
    current = init.to(device).clone().requires_grad_(True)

    opt = torch.optim.Adam([current], lr=learning_rate)
    last_loss = float("nan")
    for _ in range(steps):
        opt.zero_grad()
        normed = F.normalize(current, dim=-1)
        loss = (
            -similarity_weight
            * F.cosine_similarity(normed.unsqueeze(0), target.unsqueeze(0), dim=-1).squeeze()
            - anchor_weight
            * F.cosine_similarity(normed.unsqueeze(0), anchor.unsqueeze(0), dim=-1).squeeze()
        )
        loss.backward()
        opt.step()
        last_loss = float(loss.detach())

    with torch.no_grad():
        normed = F.normalize(current, dim=-1)
        target_sim = float(F.cosine_similarity(normed.unsqueeze(0), target.unsqueeze(0)).item())
        anchor_sim = float(F.cosine_similarity(normed.unsqueeze(0), anchor.unsqueeze(0)).item())

    return SweepPoint(
        similarity_weight=similarity_weight,
        anchor_weight=anchor_weight,
        target_similarity=target_sim,
        anchor_similarity=anchor_sim,
        final_loss=last_loss,
        steps_taken=steps,
    )


def _pareto_front(points: list[SweepPoint]) -> list[SweepPoint]:
    """Return the non-dominated subset of ``points``."""
    front: list[SweepPoint] = []
    for p in points:
        dominated = any(other.dominates(p) for other in points if other is not p)
        if not dominated:
            front.append(p)
    # Stable order: highest target_similarity first.
    front.sort(key=lambda x: x.target_similarity, reverse=True)
    return front


def _recommend(pareto: list[SweepPoint]) -> SweepPoint | None:
    """Pick the 'elbow' of the Pareto frontier.

    We maximise the *minimum* of the two normalised similarities. This
    is a max-min (Rawlsian) choice that conservatively prefers points
    that do reasonably well on *both* objectives over points that ace
    one at the cost of the other.
    """
    if not pareto:
        return None
    # Normalise both objectives to [0, 1] over the frontier.
    target_sims = [p.target_similarity for p in pareto]
    anchor_sims = [p.anchor_similarity for p in pareto]
    t_min, t_max = min(target_sims), max(target_sims)
    a_min, a_max = min(anchor_sims), max(anchor_sims)
    t_range = max(t_max - t_min, 1e-9)
    a_range = max(a_max - a_min, 1e-9)

    def score(p: SweepPoint) -> float:
        t = (p.target_similarity - t_min) / t_range
        a = (p.anchor_similarity - a_min) / a_range
        return min(t, a)

    return max(pareto, key=score)


def write_report(result: SweepResult, output_dir: str | Path) -> tuple[Path, Path]:
    """Write the sweep result to JSON + a human-readable markdown card."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{result.modality}_sweep.json"
    json_path.write_text(json.dumps(result.to_dict(), indent=2))

    md_path = output_dir / f"{result.modality}_sweep.md"
    md_path.write_text(_render_markdown(result))
    return json_path, md_path


def _render_markdown(result: SweepResult) -> str:
    lines: list[str] = []
    lines.append(f"# text-anchor sweep — {result.modality}\n")
    if result.recommended is not None:
        r = result.recommended
        lines.append(
            f"**Recommended default:** `text_anchor_weight = {r.anchor_weight:.3f}` "
            f"at `similarity_weight = {r.similarity_weight:.3f}`\n"
        )
        lines.append(
            f"- target similarity: {r.target_similarity:.4f}\n"
            f"- anchor similarity: {r.anchor_similarity:.4f}\n"
        )

    lines.append("\n## Pareto frontier\n")
    lines.append("| sim_w | anchor_w | target sim | anchor sim |")
    lines.append("| --- | --- | --- | --- |")
    for p in result.pareto_front:
        lines.append(
            f"| {p.similarity_weight:.3f} | {p.anchor_weight:.3f} | "
            f"{p.target_similarity:.4f} | {p.anchor_similarity:.4f} |"
        )

    lines.append("\n## All grid points\n")
    lines.append("| sim_w | anchor_w | target sim | anchor sim |")
    lines.append("| --- | --- | --- | --- |")
    for p in sorted(result.points, key=lambda x: (x.similarity_weight, x.anchor_weight)):
        lines.append(
            f"| {p.similarity_weight:.3f} | {p.anchor_weight:.3f} | "
            f"{p.target_similarity:.4f} | {p.anchor_similarity:.4f} |"
        )

    return "\n".join(lines) + "\n"
