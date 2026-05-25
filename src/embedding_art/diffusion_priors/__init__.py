"""
Diffusion-prior losses (M2b).

This package implements the score-distillation family of losses for using
a frozen diffusion model as a *prior* over the generator's output:

* :class:`SDSLoss` — vanilla Score Distillation Sampling (Poole et al. 2022).
  Mode-seeking, used as a baseline and as the 'natural' track when VSD is
  unavailable.
* :class:`VSDLoss` — Variational Score Distillation (Wang et al. 2023).
  Trains a small LoRA adapter on top of the diffusion prior to absorb the
  per-concept distribution; avoids the mode collapse of SDS and is the
  default 'natural' track for v3.

Both classes are deliberately backbone-agnostic: they receive a
``score_fn`` (the frozen teacher network) and, in the VSD case, a
``student_score_fn`` (the LoRA-modified student). The actual integration
with ``StableDiffusion3Pipeline`` or any other diffusion model lives in
the generator (e.g. :class:`SD35ImageGenerator`); the loss objects only
need callable score functions and a noise schedule.

References
----------
* SDS: ``DreamFusion: Text-to-3D using 2D Diffusion`` (Poole et al. 2022).
* VSD: ``ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation
  with Variational Score Distillation`` (Wang et al. 2023).
"""

from embedding_art.diffusion_priors.dual_track import (
    DualTrackConfig,
    dual_track_loss,
    make_dual_track_tracks,
)
from embedding_art.diffusion_priors.sd35_adapter import (
    SD35DiffusionAdapter,
    build_vsd_phi_adapter,
)
from embedding_art.diffusion_priors.sds import SDSLoss
from embedding_art.diffusion_priors.vsd import VSDLoss

__all__ = [
    "DualTrackConfig",
    "SD35DiffusionAdapter",
    "SDSLoss",
    "VSDLoss",
    "build_vsd_phi_adapter",
    "dual_track_loss",
    "make_dual_track_tracks",
]
