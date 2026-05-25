"""Tests for ``embedding_art.evaluation.probes``."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F  # noqa: F401,N812

from embedding_art.evaluation.probes import (
    ProbeReport,
    compute_cross_encoder_probes,
    reports_to_table,
)


@pytest.fixture()
def deterministic_probe() -> MagicMock:
    """A probe that returns the same embedding regardless of input.

    Useful for asserting that compute_cross_encoder_probes calls the
    right methods with the right arguments without caring about
    numeric content.
    """
    probe = MagicMock()
    fixed_emb = F.normalize(torch.randn(1, 256), dim=-1)
    probe.encode_text.return_value = fixed_emb
    probe.encode_image.return_value = fixed_emb
    probe.encode_audio.return_value = fixed_emb
    return probe


class TestComputeCrossEncoderProbes:
    def test_empty_inputs_return_empty_list(self) -> None:
        assert compute_cross_encoder_probes(target_text=None, modality_outputs={}, probes={}) == []
        assert (
            compute_cross_encoder_probes(target_text="storm", modality_outputs={}, probes={}) == []
        )

    def test_one_probe_one_modality(self, deterministic_probe: MagicMock) -> None:
        reports = compute_cross_encoder_probes(
            target_text="storm",
            modality_outputs={"image": "fake/path.png"},
            probes={"siglip2": deterministic_probe},
        )
        assert len(reports) == 1
        r = reports[0]
        assert r.modality == "image"
        assert r.probe == "siglip2"
        assert r.error is None
        # Two identical (normalised) embeddings => cosine sim == 1
        assert r.similarity == pytest.approx(1.0, abs=1e-5)

        deterministic_probe.encode_text.assert_called_once_with("storm")
        deterministic_probe.encode_image.assert_called_once_with("fake/path.png")

    def test_probe_without_encode_text_reports_error(self) -> None:
        bad = MagicMock(spec=["encode_image"])
        reports = compute_cross_encoder_probes(
            target_text="storm",
            modality_outputs={"image": "x.png"},
            probes={"broken": bad},
        )
        assert len(reports) == 1
        assert reports[0].error == "probe has no encode_text"
        assert reports[0].similarity is None

    def test_modality_not_supported_is_silently_skipped(self) -> None:
        # SigLIP2-shaped probe — image+text only, no audio.
        image_only = MagicMock(spec=["encode_text", "encode_image"])
        image_only.encode_text.return_value = F.normalize(torch.randn(1, 256), dim=-1)
        image_only.encode_image.return_value = F.normalize(torch.randn(1, 256), dim=-1)
        reports = compute_cross_encoder_probes(
            target_text="storm",
            modality_outputs={"image": "a.png", "audio": "b.wav"},
            probes={"siglip2": image_only},
        )
        # Only the image report is present — the audio modality is
        # outside SigLIP2's coverage, no error.
        assert {(r.modality, r.probe) for r in reports} == {("image", "siglip2")}

    def test_encode_text_failure_reports_error_and_skips_probe(self) -> None:
        broken = MagicMock()
        broken.encode_text.side_effect = RuntimeError("api down")
        reports = compute_cross_encoder_probes(
            target_text="storm",
            modality_outputs={"image": "x.png"},
            probes={"broken": broken},
        )
        assert len(reports) == 1
        assert reports[0].error is not None
        assert "encode_text failed" in reports[0].error

    def test_modality_encode_failure_reports_per_modality(
        self, deterministic_probe: MagicMock
    ) -> None:
        deterministic_probe.encode_image.side_effect = RuntimeError("bad image")
        reports = compute_cross_encoder_probes(
            target_text="storm",
            modality_outputs={"image": "x.png", "audio": "y.wav"},
            probes={"probe": deterministic_probe},
        )
        by_modality = {r.modality: r for r in reports}
        assert by_modality["image"].error is not None
        assert "encode_image failed" in by_modality["image"].error
        # Audio still succeeds because failure was scoped to one method.
        assert by_modality["audio"].error is None


class TestReportsToTable:
    def test_pivots_into_modality_probe_dict(self) -> None:
        reports = [
            ProbeReport(modality="image", probe="siglip2", similarity=0.91),
            ProbeReport(modality="audio", probe="clap", similarity=0.87),
            ProbeReport(modality="image", probe="dinov3", similarity=0.79),
        ]
        table = reports_to_table(reports)
        assert table == {
            "image": {"siglip2": 0.91, "dinov3": 0.79},
            "audio": {"clap": 0.87},
        }

    def test_omits_errored_reports_from_table(self) -> None:
        reports = [
            ProbeReport(modality="image", probe="siglip2", similarity=0.91),
            ProbeReport(modality="audio", probe="clap", similarity=None, error="oops"),
        ]
        table = reports_to_table(reports)
        assert table == {"image": {"siglip2": 0.91}}


class TestProbeReportSerialization:
    def test_to_dict_round_trip(self) -> None:
        r = ProbeReport(modality="image", probe="siglip2", similarity=0.5)
        assert r.to_dict() == {
            "modality": "image",
            "probe": "siglip2",
            "similarity": 0.5,
            "error": None,
        }
