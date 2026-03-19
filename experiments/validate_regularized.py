#!/usr/bin/env python3
"""
Pixel optimization with Lucent-style regularization.

Techniques from Distill article "Feature Visualization":
1. Jitter - random translation before each step
2. Scale - random scaling (0.95-1.05)
3. Rotation - random small rotation
4. Total Variation - penalize high-frequency noise
5. Fourier preconditioning - optimize in decorrelated space
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import math
from pathlib import Path

from src.embedding_art.encoders.imagebind import ImageBindEncoder


def random_jitter(img: torch.Tensor, max_shift: int = 16) -> torch.Tensor:
    """Randomly shift image by up to max_shift pixels."""
    _, _, h, w = img.shape
    shift_x = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    shift_y = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    return torch.roll(img, shifts=(shift_y, shift_x), dims=(2, 3))


def random_scale(img: torch.Tensor, min_scale: float = 0.95, max_scale: float = 1.05) -> torch.Tensor:
    """Randomly scale image."""
    scale = torch.empty(1).uniform_(min_scale, max_scale).item()
    _, _, h, w = img.shape
    new_h, new_w = int(h * scale), int(w * scale)
    scaled = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)
    # Crop or pad back to original size
    if scale > 1:
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        return scaled[:, :, start_h:start_h+h, start_w:start_w+w]
    else:
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        return F.pad(scaled, (pad_w, w - new_w - pad_w, pad_h, h - new_h - pad_h), mode='reflect')


def random_rotate(img: torch.Tensor, max_angle: float = 5.0) -> torch.Tensor:
    """Randomly rotate image by up to max_angle degrees."""
    angle = torch.empty(1).uniform_(-max_angle, max_angle).item()
    angle_rad = math.radians(angle)
    
    # Build rotation matrix
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    theta = torch.tensor([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0]
    ], dtype=img.dtype, device=img.device).unsqueeze(0)
    
    grid = F.affine_grid(theta, img.shape, align_corners=False)
    return F.grid_sample(img, grid, mode='bilinear', padding_mode='reflection', align_corners=False)


def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    """Compute total variation loss to penalize high-frequency content."""
    diff_h = img[:, :, 1:, :] - img[:, :, :-1, :]
    diff_w = img[:, :, :, 1:] - img[:, :, :, :-1]
    return diff_h.abs().mean() + diff_w.abs().mean()


def blur_gradients(grad: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Apply Gaussian blur to gradients (frequency penalization)."""
    kernel_size = int(4 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size, device=grad.device, dtype=grad.dtype) - kernel_size // 2
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Create 2D kernel
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    kernel = kernel_2d.expand(3, 1, -1, -1)
    
    # Apply blur
    padding = kernel_size // 2
    return F.conv2d(grad, kernel, padding=padding, groups=3)


def optimize_with_regularization(
    target_text: str = "a goldfish",
    steps: int = 300,
    lr: float = 0.05,
    tv_weight: float = 0.01,
    device: str = "mps",
    output_dir: str = "outputs",
):
    """Optimize with Lucent-style transformations."""
    
    print(f"Target: '{target_text}'")
    print(f"Steps: {steps}, LR: {lr}, TV Weight: {tv_weight}")
    print("Regularization: Jitter + Scale + Rotate + TV + Gradient Blur")
    print("-" * 50)
    
    # Load ImageBind
    encoder = ImageBindEncoder(device=device)
    
    # Get target embedding
    target_embedding = encoder.encode_text([target_text])
    target_embedding = F.normalize(target_embedding, dim=-1)
    
    # Initialize with grey + small noise
    # Use slightly larger canvas for jitter headroom
    canvas_size = 224 + 32  # Extra for jitter without edge artifacts
    img = torch.ones(1, 3, canvas_size, canvas_size, device=device) * 0.5
    img = img + torch.randn_like(img) * 0.01
    img.requires_grad_(True)
    
    optimizer = torch.optim.Adam([img], lr=lr)
    
    best_sim = -1.0
    best_img = None
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # Apply transformations (before gradient computation)
        # NOTE: Rotation disabled due to MPS grid_sampler backward limitation
        img_t = img.clone()
        img_t = random_jitter(img_t, max_shift=8)
        img_t = random_scale(img_t, 0.95, 1.05)
        # img_t = random_rotate(img_t, 5.0)  # Disabled on MPS
        
        # Crop to 224x224 for ImageBind
        offset = (canvas_size - 224) // 2
        img_cropped = img_t[:, :, offset:offset+224, offset:offset+224]
        
        # Clamp to valid range
        img_clamped = torch.clamp(img_cropped, 0, 1)
        
        # Encode
        img_embedding = encoder.encode_for_optimization(img_clamped)
        img_embedding = F.normalize(img_embedding, dim=-1)
        
        # Similarity loss
        similarity = F.cosine_similarity(img_embedding, target_embedding, dim=-1)
        
        # Total loss = -similarity + TV regularization
        tv_loss = total_variation_loss(img_clamped)
        loss = -similarity.mean() + tv_weight * tv_loss
        
        loss.backward()
        
        # Blur gradients before update (frequency penalization)
        if img.grad is not None:
            img.grad.data = blur_gradients(img.grad.data, sigma=1.0)
        
        optimizer.step()
        
        # Track best (use center crop for evaluation)
        with torch.no_grad():
            eval_crop = torch.clamp(img[:, :, offset:offset+224, offset:offset+224], 0, 1)
            eval_embed = encoder.encode_for_optimization(eval_crop)
            eval_embed = F.normalize(eval_embed, dim=-1)
            eval_sim = F.cosine_similarity(eval_embed, target_embedding, dim=-1).item()
            
            if eval_sim > best_sim:
                best_sim = eval_sim
                best_img = eval_crop.detach().clone()
        
        if step % 30 == 0 or step == steps - 1:
            print(f"Step {step:4d} | Sim: {eval_sim:.4f} | Best: {best_sim:.4f} | TV: {tv_loss.item():.4f}")
    
    print("-" * 50)
    print(f"FINAL: Best similarity = {best_sim:.4f}")
    
    # Save result
    if best_img is not None:
        img_np = best_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        
        # Sanitize filename
        safe_name = target_text.replace(" ", "_").replace("'", "")[:30]
        output_path = Path(output_dir) / f"regularized_{safe_name}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pil_img.save(output_path)
        print(f"Saved: {output_path}")
    
    return best_sim


if __name__ == "__main__":
    targets = [
        "a goldfish",
        "a dog barking",
        "ocean waves",
    ]
    
    results = {}
    for target in targets:
        print(f"\n{'='*60}")
        sim = optimize_with_regularization(
            target_text=target,
            steps=300,
            lr=0.05,
            tv_weight=0.01,
        )
        results[target] = sim
    
    print(f"\n{'='*60}")
    print("SUMMARY (WITH REGULARIZATION):")
    for target, sim in results.items():
        print(f"  '{target}': {sim:.4f}")
    
    avg_sim = sum(results.values()) / len(results)
    print(f"\nAverage similarity: {avg_sim:.4f}")
