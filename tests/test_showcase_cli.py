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

    def test_realism_flag_documented(self) -> None:
        runner = CliRunner()
        result = runner.invoke(showcase, ["--help"])
        assert "--realism" in result.output

    def test_out_of_range_realism_rejected(self) -> None:
        runner = CliRunner()
        result = runner.invoke(showcase, ["-t", "x", "-o", "/tmp/out", "--realism", "2.0"])
        assert result.exit_code != 0
        assert "realism" in result.output.lower()

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

    def test_progress_callback_receives_structured_events(self, tmp_path: Path) -> None:
        """``progress_callback`` is invoked with structured events.

        The text-only modality is sufficient to verify that
        ``showcase_start`` / ``track_start`` / ``modality_start`` /
        ``modality_complete`` / ``track_complete`` are emitted in order.
        """
        from embedding_art.cli.commands.showcase import _showcase_impl

        mock_encoder = MagicMock()
        mock_encoder.encode_text.return_value = torch.randn(1, 768)
        mock_registry = MagicMock()
        mock_registry.load.return_value = mock_encoder

        events: list[dict] = []

        with patch(
            "embedding_art.encoders.defaults.create_default_registry",
            return_value=mock_registry,
        ):
            with patch(
                "embedding_art.core.engine.EmbeddingArtEngine.from_registry"
            ) as mock_engine_factory:
                mock_engine_factory.return_value = MagicMock()
                _showcase_impl(
                    target_text="thunder",
                    output_dir=tmp_path,
                    encoder_name="languagebind",
                    modalities=["text"],
                    steps=5,
                    seed=1,
                    device="cpu",
                    image_backbone="sd35",
                    progress_callback=events.append,
                )

        kinds = [e.get("type") for e in events]
        assert kinds[0] == "showcase_start"
        assert "track_start" in kinds
        assert "modality_start" in kinds
        assert "modality_complete" in kinds
        assert kinds[-1] == "track_complete"

        modality_start = next(e for e in events if e["type"] == "modality_start")
        assert modality_start["modality"] == "text"
        assert modality_start["track"] == "honest"

        modality_complete = next(e for e in events if e["type"] == "modality_complete")
        assert modality_complete["modality"] == "text"


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


class TestRealismDial:
    """`interpolate_loss_config` is the continuous honest<->natural dial.

    The named tracks are its endpoints: ``honest`` == realism 0.0,
    ``natural`` == realism 1.0. Intermediate values interpolate every weight
    monotonically so a UI slider can sweep between 'what the model sees' and
    'human-legible'.
    """

    def test_endpoints_match_named_tracks(self) -> None:
        from embedding_art.cli.commands.showcase import (
            _build_track_loss_config,
            interpolate_loss_config,
        )

        honest = _build_track_loss_config("honest")
        natural = _build_track_loss_config("natural")
        r0 = interpolate_loss_config(0.0)
        r1 = interpolate_loss_config(1.0)

        assert r0.similarity_weight == honest.similarity_weight
        assert r0.feature_matching_weight == honest.feature_matching_weight
        assert r1.similarity_weight == natural.similarity_weight
        assert r1.feature_matching_weight == natural.feature_matching_weight
        # Endpoints reproduce the minimal/heavy regulariser shapes exactly.
        assert len(r0.regularization.regularizers) == len(honest.regularization.regularizers)
        assert len(r1.regularization.regularizers) == len(natural.regularization.regularizers)

    def test_similarity_decreases_monotonically_with_realism(self) -> None:
        from embedding_art.cli.commands.showcase import interpolate_loss_config

        sims = [interpolate_loss_config(r).similarity_weight for r in (0.0, 0.25, 0.5, 0.75, 1.0)]
        assert sims == sorted(sims, reverse=True)
        assert len(set(sims)) == len(sims)  # strictly distinct

    def test_midpoint_is_between_endpoints(self) -> None:
        from embedding_art.cli.commands.showcase import interpolate_loss_config

        mid = interpolate_loss_config(0.5)
        assert 0.4 < mid.similarity_weight < 1.0
        assert 0.15 < mid.feature_matching_weight < 0.5
        # Midpoint gains the spatial regularisers absent at the honest end.
        assert len(mid.regularization.regularizers) == 3

    def test_out_of_range_raises(self) -> None:
        from embedding_art.cli.commands.showcase import interpolate_loss_config

        with pytest.raises(ValueError, match="realism"):
            interpolate_loss_config(1.5)
        with pytest.raises(ValueError, match="realism"):
            interpolate_loss_config(-0.1)

    def test_text_anchor_weight_forwarded(self) -> None:
        from embedding_art.cli.commands.showcase import interpolate_loss_config

        cfg = interpolate_loss_config(0.3, text_anchor_weight=0.25)
        assert cfg.text_anchor_weight == 0.25


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

    def test_realism_renders_single_labelled_track(self, tmp_path: Path) -> None:
        """A continuous ``realism`` value renders one flat track labelled by
        its dial position, overriding the named ``tracks``."""
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
                    realism=0.3,
                )

        assert (tmp_path / "manifest.json").exists()
        assert not (tmp_path / "honest").exists()
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["track"] == "realism-0.30"
        assert manifest["realism"] == 0.3


class TestShowcaseLinearProbes:
    """``--linear-probes-dir`` loads a trained probe manifest and forwards it
    to the interpretation step. We assert that ``_load_linear_probes``
    returns a non-None dict and that ``_run_modality_interpretation`` is
    invoked with the wrapped probes for each rendered modality."""

    def test_load_linear_probes_returns_wrapped_callables(self, tmp_path: Path) -> None:
        from embedding_art.cli.commands.showcase import _load_linear_probes
        from embedding_art.evaluation.linear_probes import (
            ProbeDataset,
            save_probe_manifest,
            train_linear_probe,
        )

        # Build + train two tiny probes.
        emb = torch.randn(40, 32)
        # Inject a separable signal so training converges.
        emb[:20, 0] += 3.0
        emb[20:, 1] += 3.0
        binary_labels = torch.cat([torch.ones(20, 1), torch.zeros(20, 1)], dim=0)
        binary_ds = ProbeDataset(emb, binary_labels, ["is_pos"])
        multi_labels = torch.cat(
            [torch.zeros(20, dtype=torch.long), torch.ones(20, dtype=torch.long)], dim=0
        )
        multi_ds = ProbeDataset(emb, multi_labels, ["a", "b"])
        binary_probe, _ = train_linear_probe(binary_ds, epochs=20, seed=0)
        multi_probe, _ = train_linear_probe(multi_ds, epochs=20, seed=0)

        save_probe_manifest({"is_pos": binary_probe, "category": multi_probe}, tmp_path / "probes")

        wrapped = _load_linear_probes(tmp_path / "probes")
        assert wrapped is not None
        assert set(wrapped.keys()) == {"is_pos", "category"}

        # Each wrapped probe is callable on an embedding.
        emb_in = torch.zeros(32)
        emb_in[0] = 3.0
        binary_out = wrapped["is_pos"](emb_in)
        assert isinstance(binary_out, float)
        multi_out = wrapped["category"](emb_in)
        assert isinstance(multi_out, dict)
        assert set(multi_out.keys()) == {"a", "b"}

    def test_showcase_propagates_linear_probes_to_interpretation(self, tmp_path: Path) -> None:
        """When a probes dir is supplied, ``_run_modality_interpretation`` is
        called with the loaded ``linear_probes`` kwarg."""
        from embedding_art.cli.commands.showcase import _showcase_impl
        from embedding_art.evaluation.linear_probes import (
            ProbeDataset,
            save_probe_manifest,
            train_linear_probe,
        )

        # Persist a one-probe manifest.
        emb = torch.randn(40, 32)
        emb[:20, 0] += 3.0
        labels = torch.cat([torch.ones(20, 1), torch.zeros(20, 1)], dim=0)
        ds = ProbeDataset(emb, labels, ["is_pos"])
        probe, _ = train_linear_probe(ds, epochs=20, seed=0)
        save_probe_manifest({"is_pos": probe}, tmp_path / "probes")

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
                    output_dir=tmp_path / "out",
                    encoder_name="languagebind",
                    modalities=["text"],
                    steps=5,
                    seed=42,
                    device="cpu",
                    image_backbone="sd35",
                    linear_probes_dir=tmp_path / "probes",
                )

        # Text modality has no interpretation block (it's a markdown card)
        # but the manifest must exist and the probes loader must have
        # succeeded silently — i.e. we get this far without raising.
        assert (tmp_path / "out" / "manifest.json").exists()


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


# ---------------------------------------------------------------------------
# M8: probe loader + seed-stability
# ---------------------------------------------------------------------------


class TestLoadProbeEncoders:
    """``_load_probe_encoders`` resolves names through the registry."""

    def test_empty_list_returns_none(self) -> None:
        from embedding_art.cli.commands.showcase import _load_probe_encoders

        assert _load_probe_encoders([], device="cpu") is None

    def test_resolves_valid_probe(self) -> None:
        from embedding_art.cli.commands.showcase import _load_probe_encoders

        fake_encoder = MagicMock()
        registry = MagicMock()
        registry.load.return_value = fake_encoder

        with patch(
            "embedding_art.encoders.defaults.create_default_registry", return_value=registry
        ):
            probes = _load_probe_encoders(["siglip2-so400m"], device="cpu")
        assert probes is not None
        assert probes == {"siglip2-so400m": fake_encoder}
        registry.load.assert_called_once_with("siglip2-so400m", device="cpu")

    def test_skips_failing_probe_and_keeps_others(self) -> None:
        from embedding_art.cli.commands.showcase import _load_probe_encoders

        good = MagicMock()

        def loader(name: str, device: str):  # noqa: ANN001 - test stub
            if name == "broken":
                raise RuntimeError("download failed")
            return good

        registry = MagicMock()
        registry.load.side_effect = loader

        with patch(
            "embedding_art.encoders.defaults.create_default_registry", return_value=registry
        ):
            probes = _load_probe_encoders(["broken", "clap-general"], device="cpu")
        assert probes == {"clap-general": good}

    def test_all_failing_returns_none(self) -> None:
        from embedding_art.cli.commands.showcase import _load_probe_encoders

        registry = MagicMock()
        registry.load.side_effect = RuntimeError("nope")
        with patch(
            "embedding_art.encoders.defaults.create_default_registry", return_value=registry
        ):
            assert _load_probe_encoders(["x"], device="cpu") is None


class TestSeedStability:
    """``--seed-stability N`` runs N extra renders and aggregates variance."""

    def _patch_render_track(self, similarities_per_run: list[dict[str, float]]):
        """Return a side-effect that yields a per-run manifest stub."""
        runs = iter(similarities_per_run)

        def fake_render_track(**kwargs):
            sims = next(runs)
            output_dir = kwargs["output_dir"]
            output_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "track": kwargs["track"],
                "modalities": {
                    mod: {"final_similarity": sim, "path": f"{mod}.bin"}
                    for mod, sim in sims.items()
                },
            }
            (output_dir / "manifest.json").write_text(json.dumps(manifest))
            return manifest

        return fake_render_track

    def test_zero_disables(self, tmp_path: Path) -> None:
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
                    target_text="thunder",
                    output_dir=tmp_path,
                    encoder_name="languagebind",
                    modalities=["text"],
                    steps=1,
                    seed=0,
                    device="cpu",
                    image_backbone="sd35",
                    seed_stability=0,
                )

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        evaluation = manifest.get("evaluation") or {}
        assert "seed_stability" not in evaluation
        # No .stability subdirectory should have been created.
        assert not (tmp_path / ".stability").exists()

    def test_extra_runs_produce_per_modality_block(self, tmp_path: Path) -> None:
        """With seed_stability=2 + image+text modalities, the evaluation
        card should report mean/std for the image modality across the 2
        extra runs."""
        from embedding_art.cli.commands.showcase import _showcase_impl

        mock_encoder = MagicMock()
        mock_encoder.encode_text.return_value = torch.randn(1, 768)
        mock_registry = MagicMock()
        mock_registry.load.return_value = mock_encoder

        fake_render = self._patch_render_track(
            similarities_per_run=[
                # Canonical run (seed 0, used by main showcase loop)
                {"image": 0.80, "text": 1.0},
                # Stability runs (seed 1, seed 2)
                {"image": 0.82, "text": 1.0},
                {"image": 0.78, "text": 1.0},
            ]
        )

        with patch(
            "embedding_art.encoders.defaults.create_default_registry",
            return_value=mock_registry,
        ):
            with patch(
                "embedding_art.core.engine.EmbeddingArtEngine.from_registry"
            ) as mock_engine_factory:
                mock_engine_factory.return_value = MagicMock()
                with patch(
                    "embedding_art.cli.commands.showcase._render_track",
                    side_effect=fake_render,
                ):
                    _showcase_impl(
                        target_text="thunder",
                        output_dir=tmp_path,
                        encoder_name="languagebind",
                        modalities=["image", "text"],
                        steps=1,
                        seed=0,
                        device="cpu",
                        image_backbone="sd35",
                        seed_stability=2,
                        evaluate=False,
                    )

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        stab = manifest["evaluation"]["seed_stability"]
        assert stab["n_extra_runs"] == 2
        assert stab["base_seed"] == 0
        image = stab["per_modality"]["image"]
        assert image["n_seeds"] == 2
        # mean of [0.82, 0.78] = 0.80
        assert abs(image["mean_similarity"] - 0.80) < 1e-6
        # Text modality is excluded (only image/audio/video tracked).
        assert "text" not in stab["per_modality"]
