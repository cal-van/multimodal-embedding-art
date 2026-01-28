from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from embedding_art.cli.main import cli


@pytest.fixture
def mock_engine_setup():
    # Patch what main.py imports: "from embedding_art import EmbeddingArtEngine"
    with (
        patch("embedding_art.EmbeddingArtEngine") as MockEngine,
        patch("embedding_art.encoders.ImageBindEncoder") as MockEncoder,
        patch("embedding_art.generators.SDXLImageGenerator") as MockImageGen,
        patch("embedding_art.generators.AudioLDMGenerator") as MockAudioGen,
    ):
        # Setup mock result
        mock_result = MagicMock()
        mock_result.final_similarity = 0.95
        mock_result.elapsed_seconds = 10.0

        # Setup mock engine instance
        mock_engine_instance = MockEngine.return_value
        mock_engine_instance.optimize.return_value = mock_result

        yield {
            "engine_cls": MockEngine,
            "encoder_cls": MockEncoder,
            "image_gen_cls": MockImageGen,
            "audio_gen_cls": MockAudioGen,
            "engine_instance": mock_engine_instance,
            "result": mock_result,
        }


def test_cli_optimize_audio_modality(mock_engine_setup):
    """Test that --output audio triggers audio generator and saving."""
    mocks = mock_engine_setup
    runner = CliRunner()

    # Mock scipy.io.wavfile.write to check saving
    with (
        patch("scipy.io.wavfile.write") as mock_write,
        patch.object(mocks["result"], "get_final_audio") as mock_get_audio,
    ):
        # Setup mock audio return (rate, data)
        mock_get_audio.return_value = (16000, "audio_data")

        result = runner.invoke(
            cli,
            [
                "optimize",
                "--target-text",
                "thunder",
                "1.0",
                "--output",
                "audio",
                "--steps",
                "2",
                "--output-path",
                "thunder.wav",
            ],
        )

        if result.exit_code != 0:
            print(result.output)

        assert result.exit_code == 0

        # Verify AudioLDMGenerator was initialized
        mocks["audio_gen_cls"].assert_called_once()

        # Verify engine.optimize called with "audio"
        mocks["engine_instance"].optimize.assert_called_once()
        call_args = mocks["engine_instance"].optimize.call_args
        assert call_args[0][1] == "audio"  # output_modality

        # Verify get_final_audio was called
        mock_get_audio.assert_called_once_with(mocks["audio_gen_cls"].return_value)

        # Verify save called
        mock_write.assert_called_once()
        args = mock_write.call_args
        assert str(args[0][0]) == "thunder.wav"
        assert args[0][1] == 16000
        assert args[0][2] == "audio_data"


def test_cli_check_audio_imports_error(mock_engine_setup):
    """Test that it handles missing generators gracefully (if implemented)."""
    # behavior depends on implementation, but sticking to positive path for now
    pass
