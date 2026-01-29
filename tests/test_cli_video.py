from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from PIL import Image

from embedding_art.cli.main import cli


@pytest.fixture
def mock_engine_setup():
    with (
        patch("embedding_art.EmbeddingArtEngine") as mock_engine,
        patch("embedding_art.encoders.ImageBindEncoder") as mock_encoder,
        patch("embedding_art.generators.SDXLImageGenerator") as mock_image_gen,
        patch("embedding_art.generators.AudioLDMGenerator") as mock_audio_gen,
        patch("embedding_art.generators.SVDVideoGenerator") as mock_video_gen,
    ):
        mock_result = MagicMock()
        mock_result.final_similarity = 0.95
        mock_result.elapsed_seconds = 10.0

        mock_engine_instance = mock_engine.return_value
        mock_engine_instance.optimize.return_value = mock_result

        yield {
            "engine_cls": mock_engine,
            "encoder_cls": mock_encoder,
            "image_gen_cls": mock_image_gen,
            "audio_gen_cls": mock_audio_gen,
            "video_gen_cls": mock_video_gen,
            "engine_instance": mock_engine_instance,
            "result": mock_result,
        }


def test_cli_optimize_video_modality(mock_engine_setup):
    mocks = mock_engine_setup
    runner = CliRunner()

    frames = [Image.new("RGB", (8, 8), color=(10, 20, 30)) for _ in range(3)]
    mocks["result"].get_final_video.return_value = frames

    with patch("embedding_art.cli.commands.optimize.save_video") as mock_save:
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
