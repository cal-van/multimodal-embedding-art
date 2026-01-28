"""
AudioLDM 2 audio generator.

Uses the VAE from AudioLDM 2 to decode latents into mel spectrograms,
which are then converted to audio waveforms via a vocoder.

Research Notes (AudioLDM 2 Architecture)
=========================================

AudioLDM 2 is a text-to-audio latent diffusion model (LDM) that works in the
mel-spectrogram domain. Key architectural details:

Latent Space
------------
- The VAE compresses mel-spectrograms X in R^(T x F) into latents z in R^(C x T/r x F/r)
- T = time dimension (frames), F = frequency dimension (mel bins)
- C = latent channels, r = compression ratio (downsampling factor)
- Default compression ratio r=4 achieves good quality with reasonable compute
- Latent channels C=8 is used with r=4 compression

Mel-Spectrogram Configuration
-----------------------------
- 64 mel-frequency bins (F=64)
- 16 kHz sample rate for vocoder output (feature extractor uses 48kHz for input)
- Mel spectrogram is 2D: [height=mel_bins, width=time_frames]
- VAE input has 1 channel (single mel spectrogram)

VAE Scale Factor
----------------
- vae_scale_factor = 4 (determined by pipeline)
- This determines spatial downsampling from mel spectrogram to latent
- With block_out_channels=(128, 256, 512), scale_factor=4

Latent Shape
------------
- Shape: [batch, channels=8, height, width]
- Height = mel_bins / vae_scale_factor = 64 / 4 = 16
- Width = time_frames / vae_scale_factor (varies with audio length)
- For 10.24s audio at 16kHz with hop_size=160: width = 256 / 4 = 64
- Example latent shape for 10s audio: [1, 8, 16, 64]

UNet Configuration
------------------
- sample_size from UNet config determines base latent dimensions
- audio_length_in_s = sample_size * vae_scale_factor * vocoder_upsample_factor
- Default audio length: 10.24 seconds

Vocoder
-------
- SpeechT5HifiGan converts mel spectrograms to waveforms
- upsample_rates = [5, 4, 2, 2, 2] -> product = 160
- Expects input shape [batch, time_frames, mel_bins]
- Output sample rate: 16 kHz

Latent Distribution
-------------------
- Latents are sampled from N(0, 1) Gaussian distribution
- VAE applies scaling factor 0.4110932946205139 (different from SD's 0.18215)

Memory Considerations
---------------------
- Cross-attention memory scales with sequence length (width) squared
- Long audio = wide latents = quadratic memory growth
- 10s audio is typical; longer requires chunking or more memory

References
----------
- Paper: "AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining"
  https://arxiv.org/abs/2308.05734
- HuggingFace Diffusers: https://huggingface.co/docs/diffusers/en/api/pipelines/audioldm2
- Model: https://huggingface.co/cvssp/audioldm2
"""

import torch
from diffusers import AudioLDM2Pipeline

from embedding_art.exceptions import (
    GeneratorError,
    ModelLoadError,
    OutOfMemoryError,
)


class AudioLDMGenerator:
    """
    Audio generator using AudioLDM 2's VAE.

    Decodes 8-channel latent tensors into mel spectrograms, then uses a vocoder
    to convert to audio waveforms.

    Latent shape [1, 8, 16, W] -> Mel [1, 1, 64, W*4] -> Audio waveform

    Where W depends on desired audio length:
    - W = (audio_length_in_s * sample_rate) / (hop_size * vae_scale_factor)
    - For 10.24s at 16kHz with hop_size=160: W = (10.24 * 16000) / (160 * 4) = 256

    Architecture Notes
    ------------------
    AudioLDM 2 operates on mel-spectrograms rather than raw audio:

    1. Audio -> Mel spectrogram (via STFT + mel filterbank)
    2. Mel spectrogram -> VAE latent (compression)
    3. Diffusion in latent space
    4. VAE latent -> Mel spectrogram (decompression)
    5. Mel spectrogram -> Audio (vocoder)

    For optimization, we work in step 2-4, gradient-descending the latent
    to maximize embedding similarity with the target.

    The VAE uses:
    - 1 input channel (mel spectrogram is 2D grayscale-like)
    - 8 latent channels
    - Compression ratio of 4 in both dimensions
    - 64 mel bins for frequency resolution

    Vocoder (SpeechT5HifiGan) expects:
    - Input shape [batch, time_frames, mel_bins=64]
    - Note: This requires transposing VAE output from [B, 1, 64, T] to [B, T, 64]

    Attributes
    ----------
    LATENT_CHANNELS : int
        Number of channels in the latent representation (8).
    LATENT_HEIGHT : int
        Height of latent (mel bins / 4 = 16 for 64 mel bins).
    MEL_CHANNELS : int
        Number of mel-frequency bins (64).
    SAMPLE_RATE : int
        Audio sample rate in Hz (16000).
    VAE_SCALE_FACTOR : int
        Spatial downsampling factor from mel to latent (4).
    SCALING_FACTOR : float
        Latent scaling factor for VAE (0.4110932946205139).
    DEFAULT_AUDIO_LENGTH : float
        Default audio length in seconds (10.24).
    """

    # Latent configuration
    LATENT_CHANNELS = 8
    LATENT_HEIGHT = 16  # 64 mel bins / 4 compression

    # Mel spectrogram configuration
    MEL_CHANNELS = 64  # Number of mel-frequency bins

    # Audio configuration
    SAMPLE_RATE = 16000  # Hz (vocoder output sample rate)
    DEFAULT_AUDIO_LENGTH = 10.24  # seconds

    # VAE configuration
    VAE_SCALE_FACTOR = 4  # Compression ratio
    SCALING_FACTOR = 0.4110932946205139  # AudioLDM2 VAE scaling factor

    # Vocoder configuration
    VOCODER_UPSAMPLE_FACTOR = 160  # 5 * 4 * 2 * 2 * 2

    # Derived constants
    # hop_size typically 160 for 16kHz (10ms hop)
    HOP_SIZE = 160

    def __init__(
        self,
        model_id: str = "cvssp/audioldm2",
        device: str = "mps",
        audio_length_in_s: float = 10.24,
    ):
        """
        Initialize AudioLDM generator.

        Parameters
        ----------
        model_id : str
            HuggingFace model identifier. Options:
            - "cvssp/audioldm2" (base, 350M UNet)
            - "cvssp/audioldm2-large" (750M UNet)
            - "cvssp/audioldm2-music" (music-specialized)
        device : str
            Device to run on ("mps", "cuda", "cpu").
        audio_length_in_s : float
            Target audio length in seconds. Default 10.24s.
            Longer audio requires more memory (quadratic in width).
        """
        self._device = torch.device(device)
        self._model_id = model_id
        self._audio_length = audio_length_in_s

        # Calculate latent width based on audio length
        # time_frames = audio_length * sample_rate / hop_size
        # latent_width = time_frames / vae_scale_factor
        time_frames = int(audio_length_in_s * self.SAMPLE_RATE / self.HOP_SIZE)
        self._latent_width = time_frames // self.VAE_SCALE_FACTOR

        # Load full pipeline to extract VAE and vocoder
        try:
            pipeline = AudioLDM2Pipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise OutOfMemoryError(
                    operation="loading AudioLDM2 pipeline",
                    device=device,
                    original_error=e,
                ) from e
            raise ModelLoadError(model_name=model_id, original_error=e) from e
        except OSError as e:
            raise ModelLoadError(model_name=model_id, original_error=e) from e

        # Extract and keep only VAE and vocoder (discard the rest to save memory)
        self.vae = pipeline.vae
        self.vocoder = pipeline.vocoder

        # Move to device
        try:
            self.vae.to(self._device)
            self.vocoder.to(self._device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise OutOfMemoryError(
                    operation="moving AudioLDM2 components to device",
                    device=device,
                    original_error=e,
                ) from e
            raise

        # Set to eval mode and freeze parameters
        self.vae.eval()
        self.vocoder.eval()

        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.vocoder.parameters():
            param.requires_grad = False

        # Clean up remaining pipeline components
        del pipeline

    @property
    def latent_shape(self) -> tuple[int, ...]:
        """
        Shape of the latent tensor this generator accepts.

        Returns [batch=1, channels=8, height=16, width] where width
        depends on the configured audio length.
        """
        return (1, self.LATENT_CHANNELS, self.LATENT_HEIGHT, self._latent_width)

    @property
    def output_modality(self) -> str:
        """Output modality: 'audio'."""
        return "audio"

    @property
    def device(self) -> torch.device:
        """Device the generator is on."""
        return self._device

    @property
    def audio_length(self) -> float:
        """Configured audio length in seconds."""
        return self._audio_length

    def init_latent(self, seed: int | None = None) -> torch.Tensor:
        """
        Initialize a random latent for optimization.

        Returns a tensor with requires_grad=True, sampled from N(0, 1)
        which is the typical latent distribution for VAEs.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        torch.Tensor
            Latent tensor of shape [1, 8, 16, W] with requires_grad=True.
        """
        if seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(seed)
        else:
            generator = None

        latent = torch.randn(
            self.latent_shape,
            device=self._device,
            dtype=torch.float32,
            generator=generator,
        )

        latent.requires_grad_(True)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to audio waveform tensor.

        The decoding pipeline is:
        1. Scale latent by 1/SCALING_FACTOR
        2. Decode through VAE to mel spectrogram
        3. Convert mel spectrogram to waveform via vocoder

        Parameters
        ----------
        latent : torch.Tensor
            Latent tensor of shape [B, 8, 16, W].

        Returns
        -------
        torch.Tensor
            Audio waveform tensor of shape [B, samples] where
            samples = audio_length_in_s * SAMPLE_RATE.
        """
        try:
            # First decode to mel spectrogram
            mel = self.decode_to_mel(latent)

            # Reshape for vocoder: [B, 1, mels, time] -> [B, time, mels]
            mel_for_vocoder = mel.squeeze(1).transpose(1, 2)

            # Convert mel to waveform via vocoder
            waveform = self.vocoder(mel_for_vocoder)

            return waveform
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise OutOfMemoryError(
                    operation="decoding latent to audio",
                    device=str(self._device),
                    original_error=e,
                ) from e
            raise GeneratorError(operation="audio decode", original_error=e) from e

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Encode mel spectrogram to latent (for round-trip testing).

        Note: This method encodes a mel spectrogram directly, not raw audio.
        Converting raw audio to mel spectrogram requires additional preprocessing
        that is outside the scope of this generator.

        Parameters
        ----------
        mel : torch.Tensor
            Mel spectrogram tensor of shape [B, 1, 64, T] where T is time frames.

        Returns
        -------
        torch.Tensor
            Latent tensor of shape [B, 8, 16, T//4].
        """
        try:
            # Encode mel spectrogram through VAE
            with torch.no_grad():
                latent_dist = self.vae.encode(mel).latent_dist
                latent = latent_dist.sample()

            # Apply scaling factor
            latent = latent * self.SCALING_FACTOR

            return latent
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise OutOfMemoryError(
                    operation="encoding mel spectrogram to latent",
                    device=str(self._device),
                    original_error=e,
                ) from e
            raise GeneratorError(operation="VAE encode", original_error=e) from e

    def decode_to_mel(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to mel spectrogram (intermediate step).

        Useful for visualization and debugging.

        Parameters
        ----------
        latent : torch.Tensor
            Latent tensor of shape [B, 8, 16, W].

        Returns
        -------
        torch.Tensor
            Mel spectrogram tensor of shape [B, 1, 64, W*4].
        """
        try:
            # Scale latent (AudioLDM2 VAE expects scaled latents)
            scaled_latent = latent / self.SCALING_FACTOR

            # Decode through VAE to mel spectrogram
            mel = self.vae.decode(scaled_latent).sample

            return mel
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise OutOfMemoryError(
                    operation="decoding latent to mel spectrogram",
                    device=str(self._device),
                    original_error=e,
                ) from e
            raise GeneratorError(operation="VAE decode to mel", original_error=e) from e
