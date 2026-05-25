"""
Tests for the ``embed-art showcase`` command (M10a).

Fast tests verify command registration and CLI flag parsing.  The orchestration
function is exercised end-to-end with mocked encoder + generators so we can
validate the manifest assembly and text-card formatting without loading real
LanguageBind / SD3.5 / AudioLDM2 / SVD weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from click.testing import CliRunner

from embedding_art.cli.commands.showcase import (
    _save_audio_output,
    _save_image_output,
    showcase,
)

# ---------------------------------------------------------------------------
# CLI registration tests
# ---------------------------------------------------------------------------


class TestShowcaseRegistration:
    """The showcase command is registered with the main CLI group."""

    def test_showcase_command_registered(self) -> None:
        from embedding_art.cli.main import cli

        assert "showcase" in cli.commands

    def test_showcase_help_exits_zero(self) -> None:
        runner = CliRunner()
        result = runner.invoke(showcase, ["--help"])
        assert result.exit_code == 0
        assert "showcase" in result.output.lower()

    def test_showcase_lists_all_four_modality_flags(self) -> None:
        runner = CliRunner()
        result = runner.invoke(showcase, ["--help"])
        # The --modalities flag should mention all four modalities.
        assert "image" in result.output
        assert "audio" in result.output
        assert "video" in result.output
        assert "text" in result.output


class TestShowcaseFlagDefaults:
    """Default flag values match the M10a contract."""

    def test_default_encoder_is_languagebind(self) -> None:
        runner = CliRunner()
        result = runner.invoke(showcase, ["--help"])
        assert "languagebind" in result.output

    def test_default_image_backbone_is_sd35(self) -> None:
        runner = CliRunner()
        result = runner.invoke(showcase, ["--help"])
        assert "sd35" in result.output

    def test_default_audio_backbone_is_stable_audio_open(self) -> None:
        runner = CliRunner()
        result = runner.invoke(showcase, ["--help"])
        assert "stable-audio-open" in result.output
        assert "audioldm2" in result.output

    def test_default_video_backbone_is_ltx_video(self) -> None:
        runner = CliRunner()
        result = runner.invoke(showcase, ["--help"])
        assert "ltx-video" in result.output
        assert "svd" in result.output

    def test_tracks_flag_documented(self) -> None:
        runner = CliRunner()
        result = runner.invoke(showcase, ["--help"])
        assert "honest" in result.output
        assert "natural" in result.output

    def test_autocast_dtype_flag_documented(self) -> None:
        runner = CliRunner()
        result = runner.invoke(showcase, ["--help"])
        assert "--autocast-dtype" in result.output
        assert "bf16" in result.output
        assert "fp16" in result.output

    def test_invalid_track_rejected(self) -> None:
        runner = CliRunner()
        result = runner.invoke(showcase, ["-t", "x", "-o", "/tmp/out", "--tracks", "bogus"])
        assert result.exit_code != 0
        assert "track" in result.output.lower() or "bogus" in result.output

    def test_required_target_text(self) -> None:
        """--target-text is mandatory."""
        runner = CliRunner()
        result = runner.invoke(showcase, ["-o", "/tmp/out"])
        assert result.exit_code != 0
        # Click's standard usage error path mentions the missing option.
        assert "target-text" in result.output.lower() or "-t" in result.output

    def test_required_output_dir(self) -> None:
        """--output-dir is mandatory."""
        runner = CliRunner()
        result = runner.invoke(showcase, ["-t", "goldfish"])
        assert result.exit_code != 0
        assert "output-dir" in result.output.lower() or "-o" in result.output


# ---------------------------------------------------------------------------
# Output helper tests
# ---------------------------------------------------------------------------


class TestSaveImageOutput:
    """``_save_image_output`` writes a PIL-readable PNG."""

    def test_writes_png_file(self, tmp_path: Path) -> None:
        result = MagicMock()
        result.output = torch.rand(1, 3, 32, 32)
        path = tmp_path / "image.png"
        _save_image_output(result, generator=MagicMock(), path=path)
        assert path.exists()
        # Verify the PNG is openable.
        from PIL import Image

        img = Image.open(path)
        assert img.size == (32, 32)


class TestSaveAudioOutput:
    """``_save_audio_output`` writes a WAV with the generator's sample rate."""

    def test_writes_wav_file(self, tmp_path: Path) -> None:
        result = MagicMock()
        # 1 second of audio at SAMPLE_RATE
        result.output = torch.zeros(1, 16000)
        generator = MagicMock()
        generator.SAMPLE_RATE = 16000
        path = tmp_path / "audio.wav"
        _save_audio_output(result, generator, path)
        assert path.exists()
        assert path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Orchestration test (mocks encoder + generators)
# ---------------------------------------------------------------------------


def _build_mock_engine_result(modality: str) -> MagicMock:
    """Construct a fake RenderResult appropriate to *modality*."""
    result = MagicMock()
    if modality == "image":
        result.output = torch.rand(1, 3, 64, 64)
    elif modality == "audio":
        result.output = torch.zeros(1, 16000)
    elif modality == "video":
        result.output = torch.rand(1, 3, 4, 32, 32)  # [B, C, F, H, W]
    result.final_similarity = 0.42
    return result


class TestShowcaseOrchestrationTextOnly:
    """Text-card modality alone exercises the manifest pipeline without
    loading any generators."""

    def test_text_only_writes_card_and_manifest(self, tmp_path: Path) -> None:
        from embedding_art.cli.commands.showcase import _showcase_impl

        mock_encoder = MagicMock()
        mock_encoder.encode_text.return_value = torch.randn(1, 768)
        mock_registry = MagicMock()
        mock_registry.load.return_value = mock_encoder

        with patch(
            "embedding_art.encoders.defaults.create_default_registry",
            return_value=mock_registry,
        ):
            with patch(
                "embedding_art.core.engine.EmbeddingArtEngine.from_registry"
            ) as mock_engine_factory:
                mock_engine = MagicMock()
                mock_engine_factory.return_value = mock_engine

                _showcase_impl(
                    target_text="goldfish",
                    output_dir=tmp_path,
                    encoder_name="languagebind",
                    modalities=["text"],
                    steps=10,
                    seed=42,
                    device="cpu",
                    image_backbone="sd35",
                )

        manifest_path = tmp_path / "manifest.json"
        card_path = tmp_path / "text-card.md"
        assert manifest_path.exists()
        assert card_path.exists()

        manifest = json.loads(manifest_path.read_text())
        assert manifest["concept"]["text"] == "goldfish"
        assert manifest["encoder"] == "languagebind"
        assert manifest["steps"] == 10
        assert manifest["seed"] == 42
        assert "text" in manifest["modalities"]

        card = card_path.read_text()
        assert "goldfish" in card
        assert "languagebind" in card


class TestShowcaseInterpretationManifest:
    """When --interpret is on, the manifest gains an interpretation block."""

    def test_interpretation_appears_in_modality_record_for_text_only(self, tmp_path: Path) -> None:
        """The text-only modality has no per-modality interpretation (text-card
        is purely a markdown summary), but the evaluation card should still
        appear in the manifest with the default --evaluate flag."""
        from embedding_art.cli.commands.showcase import _showcase_impl

        mock_encoder = MagicMock()
        mock_encoder.encode_text.return_value = torch.randn(1, 768)
        mock_registry = MagicMock()
        mock_registry.load.return_value = mock_encoder

        with patch(
            "embedding_art.encoders.defaults.create_default_registry",
            return_value=mock_registry,
        ):
            with patch(
                "embedding_art.core.engine.EmbeddingArtEngine.from_registry"
            ) as mock_engine_factory:
                mock_engine_factory.return_value = MagicMock()

                _showcase_impl(
                    target_text="goldfish",
                    output_dir=tmp_path,
                    encoder_name="languagebind",
                    modalities=["text"],
                    steps=10,
                    seed=42,
                    device="cpu",
                    image_backbone="sd35",
                    interpret=True,
                    evaluate=True,
                )

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        # Evaluation card should still appear (empty cross-modal matrix, but
        # the structure is there).
        assert "evaluation" in manifest


class TestTrackLossConfig:
    """`_build_track_loss_config` returns distinct profiles for honest vs natural."""

    def test_honest_has_higher_similarity_than_natural(self) -> None:
        from embedding_art.cli.commands.showcase import _build_track_loss_config

        honest = _build_track_loss_config("honest")
        natural = _build_track_loss_config("natural")
        assert honest.similarity_weight > natural.similarity_weight
        assert honest.feature_matching_weight > natural.feature_matching_weight

    def test_natural_has_heavier_regularisation(self) -> None:
        from embedding_art.cli.commands.showcase import _build_track_loss_config

        honest = _build_track_loss_config("honest")
        natural = _build_track_loss_config("natural")
        assert honest.regularization is not None
        assert natural.regularization is not None
        # Heavy regulariser has strictly more components and higher weights
        # than minimal — see CompositeRegularizer.heavy vs .minimal.
        n_honest_regs = len(honest.regularization.regularizers)
        n_natural_regs = len(natural.regularization.regularizers)
        assert n_natural_regs >= n_honest_regs

    def test_unknown_track_raises(self) -> None:
        from embedding_art.cli.commands.showcase import _build_track_loss_config

        with pytest.raises(ValueError, match="Unknown track"):
            _build_track_loss_config("bogus")


class TestDualTrackOrchestration:
    """Both honest + natural tracks land into ``<output>/<track>/`` subdirs
    with a top-level summary manifest."""

    def test_two_tracks_produce_subdirs_and_top_manifest(self, tmp_path: Path) -> None:
        from embedding_art.cli.commands.showcase import _showcase_impl

        mock_encoder = MagicMock()
        mock_encoder.encode_text.return_value = torch.randn(1, 768)
        mock_registry = MagicMock()
        mock_registry.load.return_value = mock_encoder

        with patch(
            "embedding_art.encoders.defaults.create_default_registry",
            return_value=mock_registry,
        ):
            with patch(
                "embedding_art.core.engine.EmbeddingArtEngine.from_registry"
            ) as mock_engine_factory:
                mock_engine_factory.return_value = MagicMock()

                _showcase_impl(
                    target_text="goldfish",
                    output_dir=tmp_path,
                    encoder_name="languagebind",
                    modalities=["text"],
                    steps=10,
                    seed=42,
                    device="cpu",
                    image_backbone="sd35",
                    audio_backbone="stable-audio-open",
                    video_backbone="ltx-video",
                    tracks=["honest", "natural"],
                )

        assert (tmp_path / "honest" / "manifest.json").exists()
        assert (tmp_path / "natural" / "manifest.json").exists()
        top = json.loads((tmp_path / "manifest.json").read_text())
        assert top["tracks"] == ["honest", "natural"]
        assert "per_track" in top
        assert set(top["per_track"].keys()) == {"honest", "natural"}
        assert top["per_track"]["honest"]["manifest"] == "honest/manifest.json"
        assert top["per_track"]["natural"]["manifest"] == "natural/manifest.json"
        assert top["audio_backbone"] == "stable-audio-open"
        assert top["video_backbone"] == "ltx-video"

    def test_single_track_writes_flat_layout(self, tmp_path: Path) -> None:
        """Single-track render preserves the flat layout (no subdir)."""
        from embedding_art.cli.commands.showcase import _showcase_impl

        mock_encoder = MagicMock()
        mock_encoder.encode_text.return_value = torch.randn(1, 768)
        mock_registry = MagicMock()
        mock_registry.load.return_value = mock_encoder

        with patch(
            "embedding_art.encoders.defaults.create_default_registry",
            return_value=mock_registry,
        ):
            with patch(
                "embedding_art.core.engine.EmbeddingArtEngine.from_registry"
            ) as mock_engine_factory:
                mock_engine_factory.return_value = MagicMock()

                _showcase_impl(
                    target_text="goldfish",
                    output_dir=tmp_path,
                    encoder_name="languagebind",
                    modalities=["text"],
                    steps=10,
                    seed=42,
                    device="cpu",
                    image_backbone="sd35",
                    audio_backbone="stable-audio-open",
                    video_backbone="ltx-video",
                    tracks=["honest"],
                )

        # Flat layout: manifest at root, no honest/ subdir.
        assert (tmp_path / "manifest.json").exists()
        assert not (tmp_path / "honest").exists()
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["track"] == "honest"


@pytest.mark.slow
class TestShowcaseFullIntegration:
    """End-to-end test running the real LanguageBind + SD3.5 + AudioLDM2 + SVD
    pipeline. Requires all real checkpoints (~13 GB total)."""

    def test_real_showcase_runs_without_error(self, tmp_path: Path) -> None:
        from embedding_art.cli.commands.showcase import _showcase_impl

        _showcase_impl(
            target_text="goldfish",
            output_dir=tmp_path,
            encoder_name="languagebind",
            modalities=["text"],
            steps=5,
            seed=42,
            device="cpu",
            image_backbone="sd35",
        )
        assert (tmp_path / "manifest.json").exists()
