#!/usr/bin/env python3
"""
StyleGAN2 latent space optimization with ImageBind guidance.

Instead of optimizing pixels directly, we optimize in StyleGAN2's
W latent space. The GAN's learned prior naturally produces clean,
coherent images.

This is similar to StyleCLIP but using ImageBind instead of CLIP.
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path

try:
    # Try to import stylegan2-ada-pytorch
    import dnnlib
    import legacy
    STYLEGAN_AVAILABLE = True
except ImportError:
    STYLEGAN_AVAILABLE = False
    print("StyleGAN2 not available. Install with: pip install ninja")

from src.embedding_art.encoders.imagebind import ImageBindEncoder


def load_stylegan2(model_path: str = None, device: str = "mps"):
    """Load pretrained StyleGAN2 model."""
    if not STYLEGAN_AVAILABLE:
        raise ImportError("StyleGAN2 not available")
    
    if model_path is None:
        # Default to FFHQ model from NVIDIA
        # Download from: https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
        model_path = "pretrained/stylegan2-ffhq-config-f.pkl"
    
    print(f"Loading StyleGAN2 from: {model_path}")
    with dnnlib.util.open_url(model_path) as f:
        G = legacy.load_network_pkl(f)['G_ema']
    
    return G.to(device).eval()


def optimize_stylegan_latent(
    target_text: str = "a goldfish",
    steps: int = 300,
    lr: float = 0.1,
    device: str = "mps",
    output_dir: str = "outputs",
    stylegan_path: str = None,
):
    """Optimize StyleGAN2 latent to match ImageBind embedding."""
    
    print(f"Target: '{target_text}'")
    print(f"Steps: {steps}, LR: {lr}")
    print("Method: StyleGAN2 W-space latent optimization")
    print("-" * 50)
    
    # Load models
    encoder = ImageBindEncoder(device=device)
    
    try:
        G = load_stylegan2(stylegan_path, device)
    except Exception as e:
        print(f"ERROR: Could not load StyleGAN2: {e}")
        print("\nTo use StyleGAN2 optimization, download a pretrained model:")
        print("  mkdir -p pretrained")
        print("  wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl -O pretrained/stylegan2-ffhq-config-f.pkl")
        return None
    
    # Get target embedding
    target_embedding = encoder.encode_text([target_text])
    target_embedding = F.normalize(target_embedding, dim=-1)
    
    # Initialize W latent (StyleGAN2 uses 512-dim W space)
    # W+ space: (1, num_layers, 512) for per-layer control
    # W space: (1, 512) broadcast to all layers
    w_dim = G.w_dim
    num_ws = G.num_ws
    
    # Start from mean W
    w = G.mapping.w_avg.clone().unsqueeze(0).unsqueeze(0).repeat(1, num_ws, 1)
    w = w + torch.randn_like(w) * 0.01  # Small noise
    w.requires_grad_(True)
    
    optimizer = torch.optim.Adam([w], lr=lr)
    
    best_sim = -1.0
    best_img = None
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # Generate image from latent
        with torch.no_grad():
            img = G.synthesis(w, noise_mode='const')
        
        # StyleGAN outputs [-1, 1], convert to [0, 1]
        img = (img + 1) / 2
        
        # Resize to 224x224 for ImageBind
        img_resized = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
        img_resized = torch.clamp(img_resized, 0, 1)
        
        # Make it differentiable by re-running synthesis with grad
        img_grad = G.synthesis(w, noise_mode='const')
        img_grad = (img_grad + 1) / 2
        img_grad = F.interpolate(img_grad, size=(224, 224), mode='bilinear', align_corners=False)
        img_grad = torch.clamp(img_grad, 0, 1)
        
        # Encode
        img_embedding = encoder.encode_for_optimization(img_grad)
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
        
        if step % 30 == 0 or step == steps - 1:
            print(f"Step {step:4d} | Sim: {sim_val:.4f} | Best: {best_sim:.4f}")
    
    print("-" * 50)
    print(f"FINAL: Best similarity = {best_sim:.4f}")
    
    # Save result
    if best_img is not None:
        img_np = best_img[0].permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        
        safe_name = target_text.replace(" ", "_").replace("'", "")[:30]
        output_path = Path(output_dir) / f"stylegan_{safe_name}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pil_img.save(output_path)
        print(f"Saved: {output_path}")
    
    return best_sim


if __name__ == "__main__":
    if not STYLEGAN_AVAILABLE:
        print("\n" + "="*60)
        print("StyleGAN2-ADA-PyTorch is not installed.")
        print("="*60)
        print("\nTo install, clone the repo and add to path:")
        print("  git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git")
        print("  # Then download a pretrained model:")
        print("  mkdir -p pretrained")
        print("  wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl -O pretrained/ffhq.pkl")
        print("\nAlternatively, use the Fourier approach which requires no extra dependencies.")
    else:
        targets = ["a beautiful face", "a dog", "ocean waves"]
        
        results = {}
        for target in targets:
            print(f"\n{'='*60}")
            sim = optimize_stylegan_latent(target_text=target, steps=300, lr=0.1)
            if sim is not None:
                results[target] = sim
        
        if results:
            print(f"\n{'='*60}")
            print("SUMMARY (STYLEGAN LATENT):")
            for target, sim in results.items():
                print(f"  '{target}': {sim:.4f}")
            print(f"\nAverage: {sum(results.values()) / len(results):.4f}")
