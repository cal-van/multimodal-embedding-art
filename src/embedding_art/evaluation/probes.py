"""Cross-encoder probes for the M8 evaluation card.

The motivation: a render that satisfies LanguageBind isn't trustworthy
on its own — LanguageBind could be hallucinating about its own
agreement. Re-encoding the rendered output with a *different* encoder
(SigLIP 2, CLAP, DINOv3) and reporting the cosine similarity to that
encoder's projection of the target text is the cheapest cross-check
that the showcase isn't tautological.

This module is encoder-agnostic: it operates on the duck-typed
``encode_*`` interfaces and reports a single :class:`ProbeReport`
per (modality, probe) pair. The caller chooses which probes to run.

Failure handling: any per-probe error is captured and reported but
never propagated. A probe that can't be loaded simply yields an
empty entry — the eval card still renders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn.functional as F  # noqa: N812


@dataclass(frozen=True)
class ProbeReport:
    """Cross-encoder probe result for a single (modality, probe) pair.

    Attributes:
        modality: ``"image"``, ``"audio"``, or ``"video"``.
        probe: Name of the probe encoder (e.g. ``"siglip2"``).
        similarity: Cosine similarity between probe-encoded output and
            probe-encoded target text. ``None`` when the probe can't
            be run (e.g. no ``encode_text``).
        error: Human-readable error if anything went wrong, else ``None``.
    """

    modality: str
    probe: str
    similarity: float | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "probe": self.probe,
            "similarity": self.similarity,
            "error": self.error,
        }


_MODALITY_TO_ENCODE_METHOD = {
    "image": "encode_image",
    "audio": "encode_audio",
    "video": "encode_video",
}


def compute_cross_encoder_probes(
    *,
    target_text: str | None,
    modality_outputs: dict[str, Any],
    probes: dict[str, Any],
) -> list[ProbeReport]:
    """Re-encode each modality output with each probe and report similarity.

    Args:
        target_text: The natural-language target the showcase rendered
            against. When ``None``, every probe report is skipped (we
            have nothing to compare against).
        modality_outputs: Mapping ``{modality: output}`` where
            ``output`` is whatever the probe's ``encode_<modality>``
            method accepts (typically a ``Path`` or a decoded tensor).
        probes: Mapping ``{name: encoder_instance}``. Each encoder must
            expose ``encode_text`` plus at least one ``encode_image`` /
            ``encode_audio`` / ``encode_video`` method.

    Returns:
        A list of :class:`ProbeReport`, one per (modality, probe) pair
        where the probe has the required ``encode_<modality>`` method.
        Probes lacking the right method are silently skipped (no error
        report) because they're not applicable.
    """
    if not target_text or not modality_outputs or not probes:
        return []

    reports: list[ProbeReport] = []
    for probe_name, probe in probes.items():
        if not hasattr(probe, "encode_text"):
            reports.append(
                ProbeReport(
                    modality="*",
                    probe=probe_name,
                    similarity=None,
                    error="probe has no encode_text",
                )
            )
            continue
        try:
            text_emb = probe.encode_text(target_text).detach()
        except Exception as exc:
            reports.append(
                ProbeReport(
                    modality="*",
                    probe=probe_name,
                    similarity=None,
                    error=f"encode_text failed: {exc!r}",
                )
            )
            continue

        for modality, output in modality_outputs.items():
            method_name = _MODALITY_TO_ENCODE_METHOD.get(modality)
            if method_name is None or not hasattr(probe, method_name):
                # Probe doesn't cover this modality. Not an error.
                continue
            try:
                output_emb = getattr(probe, method_name)(output).detach()
            except Exception as exc:
                reports.append(
                    ProbeReport(
                        modality=modality,
                        probe=probe_name,
                        similarity=None,
                        error=f"{method_name} failed: {exc!r}",
                    )
                )
                continue
            sim = float(
                F.cosine_similarity(
                    text_emb.flatten().unsqueeze(0),
                    output_emb.flatten().unsqueeze(0),
                    dim=-1,
                ).item()
            )
            reports.append(ProbeReport(modality=modality, probe=probe_name, similarity=sim))
    return reports


def reports_to_table(reports: list[ProbeReport]) -> dict[str, dict[str, float | None]]:
    """Pivot a list of reports into ``{modality: {probe: similarity}}``.

    Probes that errored out per-modality are omitted from the table;
    they remain visible in the raw list of reports.
    """
    table: dict[str, dict[str, float | None]] = {}
    for r in reports:
        if r.error is not None:
            continue
        table.setdefault(r.modality, {})[r.probe] = r.similarity
    return table
