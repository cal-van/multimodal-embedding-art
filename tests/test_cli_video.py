from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from PIL import Image

from embedding_art.cli.main import cli


@pytest.fixture
def mock_engine_setup():
    with (
        patch("embedding_art.EmbeddingArtEngine") as MockEngine,
        patch("embedding_art.encoders.ImageBindEncoder") as MockEncoder,
        patch("embedding_art.generators.SDXLImageGenerator") as MockImageGen,
        patch("embedding_art.generators.AudioLDMGenerator") as MockAudioGen,
        patch("embedding_art.generators.SVDVideoGenerator") as MockVideoGen,
    ):
        mock_result = MagicMock()
        mock_result.final_similarity = 0.95
        mock_result.elapsed_seconds = 10.0

        mock_engine_instance = MockEngine.return_value
        mock_engine_instance.optimize.return_value = mock_result

        yield {
            "engine_cls": MockEngine,
            "encoder_cls": MockEncoder,
            "image_gen_cls": MockImageGen,
            "audio_gen_cls": MockAudioGen,
            "video_gen_cls": MockVideoGen,
            "engine_instance": mock_engine_instance,
            "result": mock_result,
        }


def test_cli_optimize_video_modality(mock_engine_setup):
    mocks = mock_engine_setup
    runner = CliRunner()

    frames = [Image.new("RGB", (8, 8), color=(10, 20, 30)) for _ in range(3)]
    mocks["result"].get_final_video.return_value = frames

    with patch("embedding_art.cli.main._save_video") as mock_save:
        result = runner.invoke(
            cli,
            [
                "optimize",
                "--target-text",
                "storm",
                "1.0",
                "--output",
                "video",
                "--steps",
                "2",
                "--output-path",
                "storm.gif",
            ],
        )

    if result.exit_code != 0:
        print(result.output)

    assert result.exit_code == 0
    mocks["video_gen_cls"].assert_called_once()

    mocks["engine_instance"].optimize.assert_called_once()
    call_args = mocks["engine_instance"].optimize.call_args
    assert call_args[0][1] == "video"

    mocks["result"].get_final_video.assert_called_once_with(mocks["video_gen_cls"].return_value)

    mock_save.assert_called_once()
    save_args = mock_save.call_args[0]
    assert save_args[0] == frames
    assert str(save_args[1]) == "storm.gif"
