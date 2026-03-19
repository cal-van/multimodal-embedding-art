#!/usr/bin/env python3
"""
Fourier-parameterized pixel optimization.

Instead of optimizing RGB pixels directly, we optimize in Fourier space
with color decorrelation. This naturally suppresses high-frequency noise.

Based on Distill's "Feature Visualization" article recommendations.
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path

from src.embedding_art.encoders.imagebind import ImageBindEncoder


# ImageNet color correlation matrix (from Lucid)
COLOR_CORRELATION_SVD_SQRT = torch.tensor([
    [0.26, 0.09, 0.02],
    [0.27, 0.00, -0.05],
    [0.27, -0.09, 0.03]
]).T


def rfft2d_freqs(h: int, w: int) -> torch.Tensor:
    """Compute 2D FFT frequencies."""
    fy = torch.fft.fftfreq(h)[:, None]
    fx = torch.fft.fftfreq(w)[None, :w // 2 + 1]
    return torch.sqrt(fx**2 + fy**2)


class FourierImage(torch.nn.Module):
    """
    Parameterize image in Fourier space with color decorrelation.
    
    This produces much smoother gradients and naturally suppresses
    high-frequency noise without explicit regularization.
    """
    
    def __init__(self, h: int = 224, w: int = 224, device: str = "mps"):
        super().__init__()
        self.h = h
        self.w = w
        self.device = device
        
        # Frequency scaling (lower = smoother, higher = sharper)
        freqs = rfft2d_freqs(h, w).to(device)
        self.register_buffer("scale", 1.0 / (1e-4 + freqs))  # Inverse frequency weighting
        
        # Color decorrelation matrix
        self.register_buffer("color_corr", COLOR_CORRELATION_SVD_SQRT.to(device))
        
        # Initialize in Fourier space (complex coefficients)
        # Shape: (3, h, w//2+1) for rfft2
        init_spectrum = torch.randn(3, h, w // 2 + 1, dtype=torch.cfloat, device=device) * 0.01
        self.spectrum = torch.nn.Parameter(init_spectrum)
    
    def forward(self) -> torch.Tensor:
        """Convert Fourier coefficients to RGB image."""
        # Apply frequency scaling
        scaled_spectrum = self.spectrum * self.scale
        
        # Inverse FFT to spatial domain
        spatial = torch.fft.irfft2(scaled_spectrum, s=(self.h, self.w))
        
        # Apply color correlation (decorrelate -> RGB)
        # spatial: (3, h, w) -> (h, w, 3) for matmul -> (3, h, w)
        spatial = spatial.permute(1, 2, 0)  # (h, w, 3)
        spatial = torch.matmul(spatial, self.color_corr.T)  # (h, w, 3)
        spatial = spatial.permute(2, 0, 1)  # (3, h, w)
        
        # Sigmoid to [0, 1]
        img = torch.sigmoid(spatial)
        
        return img.unsqueeze(0)  # (1, 3, h, w)


def random_jitter(img: torch.Tensor, max_shift: int = 8) -> torch.Tensor:
    """Randomly shift image."""
    shift_x = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    shift_y = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    return torch.roll(img, shifts=(shift_y, shift_x), dims=(2, 3))


def optimize_fourier(
    target_text: str = "a goldfish",
    steps: int = 500,
    lr: float = 0.1,
    device: str = "mps",
    output_dir: str = "outputs",
):
    """Optimize in Fourier-parameterized space."""
    
    print(f"Target: '{target_text}'")
    print(f"Steps: {steps}, LR: {lr}")
    print("Method: Fourier parameterization + color decorrelation")
    print("-" * 50)
    
    # Load ImageBind
    encoder = ImageBindEncoder(device=device)
    
    # Get target embedding
    target_embedding = encoder.encode_text([target_text])
    target_embedding = F.normalize(target_embedding, dim=-1)
    
    # Create Fourier-parameterized image
    fourier_img = FourierImage(h=224, w=224, device=device)
    
    optimizer = torch.optim.Adam(fourier_img.parameters(), lr=lr)
    
    best_sim = -1.0
    best_img = None
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # Generate image from Fourier params
        img = fourier_img()
        
        # Apply jitter for robustness
        img_jittered = random_jitter(img, max_shift=8)
        
        # Encode
        img_embedding = encoder.encode_for_optimization(img_jittered)
        img_embedding = F.normalize(img_embedding, dim=-1)
        
        # Similarity loss
        similarity = F.cosine_similarity(img_embedding, target_embedding, dim=-1)
        loss = -similarity.mean()
        
        loss.backward()
        optimizer.step()
        
        # Track best
        sim_val = similarity.item()
        if sim_val > best_sim:
            best_sim = sim_val
            best_img = img.detach().clone()
        
        if step % 50 == 0 or step == steps - 1:
            print(f"Step {step:4d} | Sim: {sim_val:.4f} | Best: {best_sim:.4f}")
    
    print("-" * 50)
    print(f"FINAL: Best similarity = {best_sim:.4f}")
    
    # Save result
    if best_img is not None:
        img_np = best_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        
        safe_name = target_text.replace(" ", "_").replace("'", "")[:30]
        output_path = Path(output_dir) / f"fourier_{safe_name}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pil_img.save(output_path)
        print(f"Saved: {output_path}")
    
    return best_sim


if __name__ == "__main__":
    targets = ["a goldfish", "a dog barking", "ocean waves"]
    
    results = {}
    for target in targets:
        print(f"\n{'='*60}")
        sim = optimize_fourier(target_text=target, steps=500, lr=0.1)
        results[target] = sim
    
    print(f"\n{'='*60}")
    print("SUMMARY (FOURIER PARAMETERIZATION):")
    for target, sim in results.items():
        print(f"  '{target}': {sim:.4f}")
    print(f"\nAverage: {sum(results.values()) / len(results):.4f}")
