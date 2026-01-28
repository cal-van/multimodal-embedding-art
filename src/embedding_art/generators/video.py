"""
Stable Video Diffusion (SVD) video generator.

Uses the VAE from Stable Video Diffusion to decode latents into video frames.

Architecture Research (from HuggingFace diffusers and Stability AI):
==================================================================

Stable Video Diffusion (SVD) is a latent video diffusion model that generates
video from an image conditioning. It extends Stable Diffusion 2.1's architecture
with temporal convolutions and attention layers.

Model Variants:
--------------
- SVD (img2vid): 14 frames at 576x1024 resolution
- SVD-XT (img2vid-xt): 25 frames at 576x1024 resolution
- SVD 1.1 (img2vid-xt-1-1): 25 frames at 1024x576 (landscape)

VAE Architecture (AutoencoderKLTemporalDecoder):
-----------------------------------------------
- Encoder: Frozen 2D encoder from SD 2.1 image VAE
- Decoder: Temporal decoder with 3D convolutions for frame coherence
- The temporal decoder enables feature interaction across frames,
  significantly reducing flicker artifacts

Latent Space:
------------
- Shape: [B, C, F, H, W] where:
  - B = batch size
  - C = 4 latent channels (same as SD/SDXL)
  - F = number of frames (14 for SVD, 25 for SVD-XT)
  - H = height // 8 (spatial downscale factor)
  - W = width // 8 (spatial downscale factor)
- For 576x1024 input: latent shape is [B, 4, F, 72, 128]
- Scaling factor: 0.18215 (inherited from SD 2.1)
- Latent distribution: ~N(0, 1), values typically in [-3, 3]

Conditioning Requirements:
-------------------------
- REQUIRED: Conditioning image (context frame) at same resolution as output
- The model is image-to-video only; text conditioning is NOT supported
- Uses CLIP ViT-H-14 image encoder for conditioning embeddings

Temporal Handling:
-----------------
- No temporal compression in the VAE (1:1 frame ratio in latent space)
- All frames denoised simultaneously via temporal attention
- f8-decoder provides temporal consistency (reduces flicker)

Memory Requirements:
------------------
- Model weights: ~4GB
- Generation: Additional ~2-4GB for 25 frames at 576x1024
- Total recommended: 8-12GB VRAM for comfortable generation

References:
----------
- Paper: "Stable Video Diffusion: Scaling Latent Video Diffusion Models
  to Large Datasets" (arXiv:2311.15127)
- HuggingFace: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
- Diffusers: https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/svd
"""

import torch
from diffusers.models import AutoencoderKLTemporalDecoder

from embedding_art.exceptions import (
    GeneratorError,
    ModelLoadError,
    OutOfMemoryError,
)


class SVDVideoGenerator:
    """
    Video generator using Stable Video Diffusion's VAE.

    Decodes 5D latent tensors into video frame sequences.
    Latent shape [B, 4, F, 72, 128] -> Video [B, F, 3, 576, 1024]

    SVD uses a temporal decoder that processes all frames together,
    enabling temporal consistency through 3D convolutions. Unlike
    the image VAE, the video decoder cannot process frames independently.

    Attributes:
        LATENT_CHANNELS: Number of channels in latent space (4)
        LATENT_HEIGHT: Height of latent tensor for 576px output (72)
        LATENT_WIDTH: Width of latent tensor for 1024px output (128)
        SCALING_FACTOR: VAE scaling factor from SD 2.1 (0.18215)
        SPATIAL_SCALE_FACTOR: Spatial downscaling ratio (8)
        DEFAULT_NUM_FRAMES: Default frame count (25 for XT variant)
        OUTPUT_HEIGHT: Default output video height (576)
        OUTPUT_WIDTH: Default output video width (1024)

    Note:
        This generator requires a conditioning image to function.
        Pure latent-to-video generation without conditioning produces
        incoherent results. The conditioning image provides semantic
        guidance for the generated video content.
    """

    LATENT_CHANNELS = 4
    LATENT_HEIGHT = 72  # 576 // 8
    LATENT_WIDTH = 128  # 1024 // 8
    SCALING_FACTOR = 0.18215
    SPATIAL_SCALE_FACTOR = 8
    DEFAULT_NUM_FRAMES = 25  # SVD-XT default
    OUTPUT_HEIGHT = 576
    OUTPUT_WIDTH = 1024

    # Model identifiers
    MODEL_ID_SVD = "stabilityai/stable-video-diffusion-img2vid"  # 14 frames
    MODEL_ID_SVD_XT = "stabilityai/stable-video-diffusion-img2vid-xt"  # 25 frames
    MODEL_ID_SVD_XT_1_1 = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"  # 25 frames, improved

    def __init__(
        self,
        model_id: str = MODEL_ID_SVD_XT,
        device: str = "mps",
        num_frames: int = DEFAULT_NUM_FRAMES,
    ):
        """
        Initialize the SVD video generator.

        Args:
            model_id: HuggingFace model identifier for SVD variant
            device: Device to run on ('mps', 'cuda', or 'cpu')
            num_frames: Number of frames to generate (14 for SVD, 25 for SVD-XT)

        Note:
            SVD requires AutoencoderKLTemporalDecoder from diffusers.
            The temporal decoder has different behavior than standard
            AutoencoderKL - it processes the frame dimension with 3D
            convolutions for temporal coherence.
        """
        self._device = torch.device(device)
        self._model_id = model_id
        self._num_frames = num_frames
        self._conditioning_image: torch.Tensor | None = None

        try:
            self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
                model_id,
                subfolder="vae",
                torch_dtype=torch.float32,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise OutOfMemoryError(
                    operation="loading SVD VAE model",
                    device=device,
                    original_error=e,
                ) from e
            raise ModelLoadError(model_name=model_id, original_error=e) from e
        except OSError as e:
            raise ModelLoadError(model_name=model_id, original_error=e) from e

        try:
            self.vae.to(self._device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise OutOfMemoryError(
                    operation="moving SVD VAE to device",
                    device=device,
                    original_error=e,
                ) from e
            raise

        self.vae.eval()

        for param in self.vae.parameters():
            param.requires_grad = False

    @property
    def latent_shape(self) -> tuple[int, ...]:
        """
        Shape of the latent tensor this generator accepts.

        Returns:
            Tuple of (batch, channels, frames, height, width)
            Default: (1, 4, 25, 72, 128) for SVD-XT at 576x1024
        """
        return (
            1,
            self.LATENT_CHANNELS,
            self._num_frames,
            self.LATENT_HEIGHT,
            self.LATENT_WIDTH,
        )

    @property
    def output_modality(self) -> str:
        """Output modality identifier."""
        return "video"

    @property
    def device(self) -> torch.device:
        """Device the generator is on."""
        return self._device

    def init_latent(self, seed: int | None = None) -> torch.Tensor:
        """
        Initialize a random latent for optimization.

        Creates a 5D tensor [B, C, F, H, W] sampled from N(0, 1),
        which matches the expected latent distribution for the SVD VAE.

        Args:
            seed: Optional random seed for reproducibility

        Returns:
            Latent tensor with requires_grad=True, shape [1, 4, F, 72, 128]

        Note:
            For embedding optimization, the latent should be initialized
            with proper scaling. The SVD VAE expects latents scaled by
            SCALING_FACTOR (0.18215) before decoding.
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

    def decode(
        self,
        latent: torch.Tensor,
        conditioning_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Decode latent to video tensor.

        Args:
            latent: Tensor of shape [B, 4, F, H, W]
            conditioning_image: Optional conditioning image [B, 3, 576, 1024].
                SVD is designed for image-to-video generation and typically
                requires a conditioning image for coherent output.

        Returns:
            Video tensor [B, F, 3, 576, 1024] with values in [0, 1]

        Note:
            The temporal decoder processes all frames together through 3D
            convolutions. This is different from processing frames individually
            through a 2D decoder, and is what enables temporal consistency.

            Decoding steps:
            1. Scale latent: scaled = latent / SCALING_FACTOR
            2. Reshape for decoder if needed
            3. Pass through AutoencoderKLTemporalDecoder
            4. Normalize output to [0, 1]
        """
        try:
            scaled_latent = latent / self.SCALING_FACTOR

            # Latent shape: [B, C, F, H, W] -> need [B*F, C, H, W] for 2D VAE
            batch_size, channels, num_frames, height, width = scaled_latent.shape
            # Permute to [B, F, C, H, W] then flatten batch and frames
            latent_permuted = scaled_latent.permute(0, 2, 1, 3, 4)
            latent_flat = latent_permuted.reshape(batch_size * num_frames, channels, height, width)

            # Decode through VAE (temporal decoder needs num_frames)
            decoded_flat = self.vae.decode(latent_flat, num_frames=num_frames).sample

            # Reshape back to video: [B*F, 3, H*8, W*8] -> [B, F, 3, H*8, W*8]
            _, out_channels, out_height, out_width = decoded_flat.shape
            decoded = decoded_flat.view(batch_size, num_frames, out_channels, out_height, out_width)

            decoded = (decoded + 1) / 2
            decoded = decoded.clamp(0, 1)

            return decoded
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise OutOfMemoryError(
                    operation="decoding latent to video",
                    device=str(self._device),
                    original_error=e,
                ) from e
            raise GeneratorError(operation="VAE decode", original_error=e) from e

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode video to latent (for round-trip testing).

        Args:
            video: Tensor of shape [B, F, 3, H, W] with values in [0, 1]

        Returns:
            Latent tensor [B, 4, F, H//8, W//8]

        Note:
            SVD uses a frozen 2D encoder from SD 2.1, meaning it encodes
            each frame independently. The temporal coherence comes from
            the decoder, not the encoder.

            Encoding steps:
            1. Normalize video to [-1, 1]
            2. Encode each frame through 2D encoder
            3. Stack into temporal dimension
            4. Apply SCALING_FACTOR
        """
        try:
            batch_size, num_frames, channels, height, width = video.shape

            video_flat = video.view(batch_size * num_frames, channels, height, width)

            video_normalized = video_flat * 2 - 1

            with torch.no_grad():
                latent_dist = self.vae.encode(video_normalized).latent_dist
                latent_flat = latent_dist.sample()

            latent = latent_flat.view(batch_size, num_frames, 4, height // 8, width // 8)
            latent = latent.permute(0, 2, 1, 3, 4)

            latent = latent * self.SCALING_FACTOR

            return latent
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise OutOfMemoryError(
                    operation="encoding video to latent",
                    device=str(self._device),
                    original_error=e,
                ) from e
            raise GeneratorError(operation="VAE encode", original_error=e) from e

    def set_conditioning_image(self, image: torch.Tensor) -> None:
        """
        Set the conditioning image for video generation.

        SVD is an image-to-video model that requires a conditioning
        frame to guide generation. This method stores the conditioning
        image for use during decode().

        Args:
            image: Conditioning image tensor [B, 3, 576, 1024] or PIL Image

        Raises:
            ValueError: If image dimensions are incorrect.

        Note:
            The conditioning image is encoded via CLIP ViT-H-14 and
            injected into the UNet via cross-attention during generation.
            For direct VAE-only generation, the conditioning mechanism
            differs from full pipeline usage.
        """
        if image.shape[-2:] != (self.OUTPUT_HEIGHT, self.OUTPUT_WIDTH):
            raise ValueError(
                f"Conditioning image must be {self.OUTPUT_HEIGHT}x{self.OUTPUT_WIDTH}, "
                f"got {image.shape[-2]}x{image.shape[-1]}"
            )

        self._conditioning_image = image
