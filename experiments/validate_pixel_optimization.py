#!/usr/bin/env python3
"""
Quick validation experiment for direct pixel optimization through ImageBind.

This tests whether we can achieve high embedding similarity by optimizing
an image tensor directly via gradient ascent through ImageBind.
"""

import sys
sys.path.insert(0, ".")

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

# Import our ImageBind wrapper
from src.embedding_art.encoders.imagebind import ImageBindEncoder


def validate_pixel_optimization(
    target_text: str = "a goldfish",
    steps: int = 200,
    lr: float = 0.1,
    device: str = "mps",
):
    """Test direct pixel optimization through ImageBind."""
    
    print(f"Target: '{target_text}'")
    print(f"Steps: {steps}, LR: {lr}, Device: {device}")
    print("-" * 50)
    
    # Load ImageBind
    encoder = ImageBindEncoder(device=device)
    
    # Get target embedding from text
    target_embedding = encoder.encode_text([target_text])
    target_embedding = F.normalize(target_embedding, dim=-1)
    print(f"Target embedding shape: {target_embedding.shape}")
    
    # Initialize random image (ImageBind expects 224x224)
    # Start with grey + small noise to avoid saturation
    img = torch.ones(1, 3, 224, 224, device=device) * 0.5
    img = img + torch.randn_like(img) * 0.01
    img.requires_grad_(True)
    
    # Simple SGD (Adam can overshoot badly for this)
    optimizer = torch.optim.Adam([img], lr=lr)
    
    best_sim = -1.0
    best_img = None
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # Clamp to valid range before forward pass
        img_clamped = torch.clamp(img, 0, 1)
        
        # Encode the image through ImageBind (differentiable)
        img_embedding = encoder.encode_for_optimization(img_clamped)
        img_embedding = F.normalize(img_embedding, dim=-1)
        
        # Cosine similarity (we want to maximize this)
        similarity = F.cosine_similarity(img_embedding, target_embedding, dim=-1)
        
        # Loss is negative similarity (we minimize loss = maximize similarity)
        loss = -similarity.mean()
        
        # Backprop
        loss.backward()
        
        # Update
        optimizer.step()
        
        # Track best
        sim_val = similarity.item()
        if sim_val > best_sim:
            best_sim = sim_val
            best_img = img_clamped.detach().clone()
        
        # Log progress
        if step % 20 == 0 or step == steps - 1:
            print(f"Step {step:4d} | Similarity: {sim_val:.4f} | Best: {best_sim:.4f}")
    
    print("-" * 50)
    print(f"FINAL: Best similarity = {best_sim:.4f}")
    
    # Save the best image
    if best_img is not None:
        img_np = best_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        output_path = "outputs/pixel_optimization_test.png"
        pil_img.save(output_path)
        print(f"Saved result to: {output_path}")
    
    return best_sim


if __name__ == "__main__":
    # Run with different targets to see consistency
    targets = [
        "a goldfish",
        "a dog barking",
        "ocean waves",
    ]
    
    results = {}
    for target in targets:
        print(f"\n{'='*60}")
        sim = validate_pixel_optimization(target_text=target, steps=100)
        results[target] = sim
    
    print(f"\n{'='*60}")
    print("SUMMARY:")
    for target, sim in results.items():
        print(f"  '{target}': {sim:.4f}")
    
    avg_sim = sum(results.values()) / len(results)
    print(f"\nAverage similarity: {avg_sim:.4f}")
    
    if avg_sim > 0.7:
        print("\n✅ VALIDATED: Direct pixel optimization achieves high similarity!")
    elif avg_sim > 0.5:
        print("\n⚠️  MODERATE: Works but may need more tuning or regularization.")
    else:
        print("\n❌ ISSUE: Low similarity - gradient flow may be problematic.")
