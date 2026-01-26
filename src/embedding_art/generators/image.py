"""
SDXL VAE image generator.

Uses the VAE from Stable Diffusion XL to decode latents into images.
The VAE alone (without the full diffusion process) gives faster results
but with somewhat less sharpness than full diffusion.
"""

import torch
from diffusers import AutoencoderKL


class SDXLImageGenerator:
    """
    Image generator using SDXL's VAE.

    Decodes 4-channel latent tensors into RGB images.
    Latent shape [1, 4, 128, 128] → Image [1, 3, 1024, 1024]
    """

    LATENT_CHANNELS = 4
    LATENT_SIZE = 128  # Decodes to 1024x1024
    OUTPUT_SIZE = 1024
    SCALING_FACTOR = 0.13025  # SDXL VAE scaling factor

    def __init__(
        self,
        model_id: str = "stabilityai/sdxl-vae",
        device: str = "mps",
    ):
        self._device = torch.device(device)

        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # float32 for MPS stability
        )
        self.vae.to(self._device)
        self.vae.eval()

        # Freeze VAE parameters
        for param in self.vae.parameters():
            param.requires_grad = False

    @property
    def latent_shape(self) -> tuple[int, ...]:
        return (1, self.LATENT_CHANNELS, self.LATENT_SIZE, self.LATENT_SIZE)

    @property
    def output_modality(self) -> str:
        return "image"

    @property
    def device(self) -> torch.device:
        return self._device

    def init_latent(self, seed: int | None = None) -> torch.Tensor:
        """
        Initialize a random latent for optimization.

        Returns a tensor with requires_grad=True, sampled from N(0, 1)
        which is the typical latent distribution for VAEs.
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
        Decode latent to image tensor.

        Args:
            latent: Tensor of shape [B, 4, 128, 128]

        Returns:
            Image tensor [B, 3, 1024, 1024] with values in [0, 1]
        """
        # Scale latent (SDXL VAE expects scaled latents)
        scaled_latent = latent / self.SCALING_FACTOR

        # Decode through VAE
        decoded = self.vae.decode(scaled_latent).sample

        # Clamp to [0, 1] range
        decoded = (decoded + 1) / 2  # VAE outputs [-1, 1]
        decoded = decoded.clamp(0, 1)

        return decoded

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent (for round-trip testing).

        Args:
            image: Tensor of shape [B, 3, H, W] with values in [0, 1]

        Returns:
            Latent tensor [B, 4, H//8, W//8]
        """
        # Scale to [-1, 1]
        image = image * 2 - 1

        # Encode
        with torch.no_grad():
            latent_dist = self.vae.encode(image).latent_dist
            latent = latent_dist.sample()

        # Apply scaling
        latent = latent * self.SCALING_FACTOR

        return latent
