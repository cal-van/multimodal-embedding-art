from unittest.mock import MagicMock

import torch

from embedding_art.core.concept import Concept
from embedding_art.core.config import OptimizationConfig
from embedding_art.core.engine import EmbeddingArtEngine, OptimizationResult
from embedding_art.encoders.base import Encoder
from embedding_art.generators.audio import AudioLDMGenerator
from embedding_art.regularizers.base import CompositeRegularizer


class MockEncoder(Encoder):
    def encode_text(self, text):
        return torch.randn(1, 1024)

    def encode_image(self, image):
        return torch.randn(1, 1024)

    def encode_audio(self, audio):
        # Expecting audio input
        return torch.randn(1, 1024)


class MockAudioLDMGenerator(AudioLDMGenerator):
    """
    Subclass that mocks out the heavy lifting but keeps the structure.
    """

    def __init__(self, device="cpu"):
        self._device = torch.device(device)
        self._audio_length = 10.24
        # Standard shapes for tests
        self._latent_width = 64
        # We don't call super().__init__ to avoid loading weights

        # Mock VAE and Vocoder
        self.vae = MagicMock()
        self.vocoder = MagicMock()
        self.SCALING_FACTOR = 1.0

    def init_latent(self, seed=None):
        # Return a real tensor so gradients works
        return torch.randn(1, 8, 16, 64, device=self._device, requires_grad=True)

    def decode(self, latent):
        # Fake decoding: just return a tensor of correct audio shape
        # shape: [B, Samples]
        # 10.24s * 16000Hz = 163840 samples
        batch_size = latent.shape[0]
        return torch.randn(batch_size, 163840, device=self._device, requires_grad=True)

    def encode(self, mel):
        return torch.randn(1, 8, 16, 64)


def test_audio_optimization_loop_structure():
    """
    Test that the engine successfully drives the audio generator
    for a few steps without crashing.
    """
    encoder = MockEncoder()
    engine = EmbeddingArtEngine(encoder=encoder, device="cpu")

    generator = MockAudioLDMGenerator(device="cpu")
    engine.register_generator("audio", generator)

    target = Concept(embedding=torch.randn(1, 1024), description="test target")

    config = OptimizationConfig(
        steps=2,
        learning_rate=0.1,
        optimizer="adam",
        scheduler="constant",
        augmentation=MagicMock(),  # Disable augmentation logic if possible or ensure defaults work
    )
    # Ensure defaults are fine
    config.augmentation.random_crop = False
    config.augmentation.random_flip = False

    result = engine.optimize(
        target=target,
        output_modality="audio",
        config=config,
        regularizers=CompositeRegularizer.default_audio(),
        progress=False,
    )

    assert isinstance(result, OptimizationResult)
    assert len(result.loss_history) == 2
    assert result.final_latent.shape == (1, 8, 16, 64)
    # Check if similarity exists
    assert -1.0 <= result.final_similarity <= 1.0
