import torch
import torch.nn.functional as F
import gc
import logging
from diffusers import StableDiffusionXLPipeline
import PIL.Image

# Type checking imports
from typing import TYPE_CHECKING, List, Optional, Union, Callable, Any

# Conditional import to avoid runtime circular dependency if not needed immediately
try:
    from embedding_art.encoders.imagebind import ModalityType, ImageBindEncoder
except ImportError:
    ModalityType = Any  # Fallback if ImageBind not available
    ImageBindEncoder = Any

if TYPE_CHECKING:
    from embedding_art.encoders.imagebind import ImageBindEncoder

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024
DEFAULT_SEED_SCALE = 0.1
DEFAULT_VAE_SCALING_FACTOR = (
    0.13025  # Standard SDXL VAE scaling factor usually, but access via config is better
)

# Guidance Constants
GUIDANCE_TIMESTEP_RATIO = 0.8
GRADIENT_CLIP_MIN = -0.1
GRADIENT_CLIP_MAX = 0.1
GRADIENT_STEPS = 1
GUIDANCE_IMAGE_SIZE = (224, 224)


class SDXLDiffusionGenerator:
    def __init__(self, device: str = "cpu"):
        self._device = torch.device(device)  # Internal storage as torch.device
        logger.info(f"Loading SDXL Pipeline on {device}...")

        # Load SDXL
        # User accepted "SDXL" generally. Using SDXL Base 1.0.
        model_id = DEFAULT_MODEL_ID

        # Optimization for M1 Max (64GB): Use float32 to avoid "Input type (float) and bias type (Half)" errors.
        # FP32 is safer and fits in 64GB RAM.
        dtype = torch.float32
        variant = None

        if device == "cuda":
            dtype = torch.float16
            variant = "fp16"

        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_id, torch_dtype=dtype, variant=variant, use_safetensors=True
        ).to(
            self.device
        )  # Uses property

        # Optimize for Mac/M1
        if device == "mps":
            # MPS specific optimizations
            self.pipeline.enable_attention_slicing()
            self.pipeline.enable_vae_slicing()
            self.pipeline.enable_vae_tiling()

    @property
    def device(self) -> torch.device:
        return self._device

    def generate(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = DEFAULT_INFERENCE_STEPS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE,  # Text CFG
        imagebind_encoder: Optional["ImageBindEncoder"] = None,
        target_embedding: Optional[torch.Tensor] = None,
        imagebind_guidance_scale: float = 0.0,  # How much to force ImageBind concept
        regularizers: Optional[Any] = None,  # Regularizers
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        normalize_gradients: bool = False,
        **kwargs,
    ) -> PIL.Image.Image:
        """
        Generate an image using SDXL, optionally guided by ImageBind.
        """

        # If no ImageBind guidance, just use standard pipeline
        if imagebind_guidance_scale <= 0 or imagebind_encoder is None or target_embedding is None:
            logger.info("Running Standard SDXL Generation...")
            return self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                callback_on_step_end=self._adapt_callback(callback) if callback else None,
            ).images[0]

        # Custom Guidance Loop Implementation
        logger.info(
            f"Running ImageBind Guided SDXL Generation (scale={imagebind_guidance_scale})..."
        )

        try:
            # 0. Setup
            height = kwargs.get("height", DEFAULT_HEIGHT)
            width = kwargs.get("width", DEFAULT_WIDTH)
            original_size = (height, width)
            target_size = (height, width)

            # 1. Encode Text Prompts (SDXL requires 2 text encoders)
            # We use the pipeline's helper to encode
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = self.pipeline.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                device=self.device,
            )

            # 2. Prepare Timesteps
            self.pipeline.scheduler.set_timesteps(num_inference_steps, device=self.device)
            timesteps = self.pipeline.scheduler.timesteps

            # 3. Prepare Latents
            num_channels_latents = self.pipeline.unet.config.in_channels
            latents = self.pipeline.prepare_latents(
                1, num_channels_latents, height, width, prompt_embeds.dtype, self.device, None
            )

            # 4. Prepare Added Time Ids (for SDXL)
            add_text_embeds = pooled_prompt_embeds

            if self.pipeline.text_encoder_2 is None:
                text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
            else:
                text_encoder_projection_dim = self.pipeline.text_encoder_2.config.projection_dim

            # Ensure add_time_ids is on correct device
            add_time_ids = self.pipeline._get_add_time_ids(
                original_size,
                (0, 0),
                target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            ).to(self.device)

            # 5. Denoising Loop
            with self.pipeline.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # Standard Diffusion Step (No Grad)
                    with torch.no_grad():
                        # Expand latents for CFG
                        latent_model_input = (
                            torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
                        )
                        latent_model_input = self.pipeline.scheduler.scale_model_input(
                            latent_model_input, t
                        )

                        # Predict noise
                        added_cond_kwargs = {
                            "text_embeds": add_text_embeds,
                            "time_ids": add_time_ids,
                        }
                        if guidance_scale > 1.0:
                            added_cond_kwargs["text_embeds"] = torch.cat(
                                [negative_pooled_prompt_embeds, add_text_embeds]
                            )
                            added_cond_kwargs["time_ids"] = torch.cat([add_time_ids, add_time_ids])

                        noise_pred = self.pipeline.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=(
                                torch.cat([negative_prompt_embeds, prompt_embeds])
                                if guidance_scale > 1.0
                                else prompt_embeds
                            ),
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]

                        # Perform CFG
                        if guidance_scale > 1.0:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (
                                noise_pred_text - noise_pred_uncond
                            )

                        # Compute previous noisy sample x_t -> x_{t-1}
                        # Standard scheduler step
                        latents = self.pipeline.scheduler.step(
                            noise_pred, t, latents, return_dict=False
                        )[0]

                    # --- IMAGEBIND GUIDANCE ---
                    # Determine when to guide. Early steps define structure, later steps define texture.
                    # Assessing similarity effectively requires some structure.
                    # Let's guide for the first 80% of steps.
                    should_guide = imagebind_guidance_scale > 0 and i < (
                        num_inference_steps * GUIDANCE_TIMESTEP_RATIO
                    )

                    loss_val = 0.0
                    sim_val = 0.0

                    if should_guide:
                        for k in range(GRADIENT_STEPS):
                            # 1. Enable gradients
                            latents = latents.detach().requires_grad_(True)

                            # 2. Decode current latent approximation
                            # VAE decode is expensive, so we do it with gradients tracked
                            latents_in = (
                                latents.to(self.pipeline.vae.dtype)
                                / self.pipeline.vae.config.scaling_factor
                            )
                            image = self.pipeline.vae.decode(latents_in).sample  # [1, 3, H, W]

                            # 3. ImageBind Encode
                            # ImageBind expects [1, 3, 224, 224] for vision
                            # Resize to 224x224
                            image_224 = F.interpolate(
                                image, size=GUIDANCE_IMAGE_SIZE, mode="bicubic", align_corners=False
                            )

                            # 4. Calculate Loss
                            embeds = imagebind_encoder.model({ModalityType.VISION: image_224})[
                                ModalityType.VISION
                            ]
                            current_embedding = F.normalize(embeds, dim=-1)

                            # Cosine Distance
                            sim = F.cosine_similarity(
                                current_embedding,
                                target_embedding.to(current_embedding.device),
                                dim=-1,
                            ).mean()
                            loss = 1.0 - sim

                            # Regularization Loss (TV, etc.) to prevent noise
                            if regularizers:
                                reg_loss = regularizers(latents, image)
                                loss = loss + reg_loss

                            # 5. Backward
                            # Calculate raw gradient
                            # 5. Backward / Calculate Gradient
                            grad = torch.autograd.grad(loss, latents)[0]

                            # 6. Apply Gradient Logic
                            if normalize_gradients:
                                # Normalized Gradient Ascent: Direction * Step Size
                                # We treat 'imagebind_guidance_scale' as a step size roughly scaled.
                                # Standard approx: 1000 scale -> 1.0 step size.
                                # Normalize vector to length 1, then multiply.
                                if grad is not None:
                                    grad = F.normalize(grad, dim=-1) * (
                                        imagebind_guidance_scale * 0.002
                                    )
                            else:
                                # Standard Gradient Ascent: Raw Gradient * Scale
                                # Clamped to prevent explosions
                                grad = torch.clamp(grad, GRADIENT_CLIP_MIN, GRADIENT_CLIP_MAX)
                                grad = grad * imagebind_guidance_scale

                            # Update Latents
                            latents = latents - grad

                            # Track metrics
                            loss_val = loss.item()
                            sim_val = sim.item()

                            # Cleanup VAE graph
                            del image, image_224, embeds, current_embedding, grad

                        logger.info(f"Step {i}: Sim={sim_val:.4f} Loss={loss_val:.4f}")

                    # Call callback
                    if callback:
                        callback(i, loss_val, sim_val, latents.detach())  # send detached latents
                    progress_bar.update()

                    # Cleanup to save memory
                    gc.collect()
                    if str(self.device).startswith("mps"):
                        torch.mps.empty_cache()

            # Final decode
            image = self.decode(latents)

            # Convert to PIL
            image_pil = self.pipeline.image_processor.postprocess(image, output_type="pil")[0]

            # Cleanup final
            del latents, prompt_embeds, negative_prompt_embeds, add_text_embeds, add_time_ids
            gc.collect()
            if str(self.device).startswith("mps"):
                torch.mps.empty_cache()

            return image_pil

        except Exception:
            logger.error("Generation failed", exc_info=True)
            raise

    def _adapt_callback(self, callback):
        # Adapt diffusers callback to our engine callback signature
        # diffusers: callback(pipe, step, timestep, callback_kwargs)
        def wrapped(pipe, step, timestep, callback_kwargs):
            # We don't have easy access to 'loss' or 'sim' here yet in standard loop
            # So just send progress
            callback(step, 0.0, 0.0, None)
            return callback_kwargs

        return wrapped

    def init_latent(
        self, seed: int | None = None, scale: float = DEFAULT_SEED_SCALE
    ) -> torch.Tensor:
        """
        Initialize a random latent.
        Required by Engine interface, though not heavily used in 'generate' mode.
        """
        # SDXL latent shape: [1, 4, 128, 128] typically for 1024x1024
        shape = (1, 4, 128, 128)

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        latent = (
            torch.randn(
                shape,
                device=self.device,
                dtype=torch.float32,  # Use float32 on MPS
                generator=generator,
            )
            * scale
        )

        latent.requires_grad_(True)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to images.
        Required by Engine.OptimizationResult.
        """
        # SDXL VAE scaling
        # Ensure latent is correct dtype for VAE
        latent = latent.to(self.pipeline.vae.dtype)

        # Unscale
        latent = latent / self.pipeline.vae.config.scaling_factor

        with torch.no_grad():
            image = self.pipeline.vae.decode(latent).sample

        return image  # [B, 3, H, W]
