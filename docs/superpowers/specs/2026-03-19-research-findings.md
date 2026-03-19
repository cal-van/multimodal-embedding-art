# Research Findings: Embedding Art v2

**Date:** 2026-03-19

## Key Papers & Projects

### Group-Sparse Autoencoders for Multimodal Embeddings (ICLR 2026)
- **Paper:** arxiv 2601.20028
- **Authors:** Chiraag Kaushik, Davis Barch, Andrea Fanelli
- **Key idea:** Standard SAEs on multimodal embeddings produce "split dictionaries" (features are unimodal). Group-sparse regularization + cross-modal random masking produces truly cross-modal features.
- **Architecture:** TopK SAE with shared W_enc/W_dec, separate per-modality pre-biases, group-sparse L_{2,1} loss, shared random binary mask
- **Tested on:** CLIP ViT-B/16 (512d) and LAION CLAP (512d)
- **Results:** ~3,200 multimodal features (vs ~600 for standard SAE) from 8,192 dictionary
- **Limitation:** Reconstruction quality degrades (~54% of dense on ImageNet zero-shot, ~92% on CIFAR-10)
- **No public code.** Re-implement from paper, adapting from `saprmarks/dictionary_learning`

### MIMIC: Multimodal Inversion for Model Interpretation (updated March 2026)
- **Paper:** arxiv 2508.07833
- **Code:** github.com/anaekin/MIMIC (Apache 2.0)
- **Key idea:** Optimize raw pixels to invert VLM internal states. No decoder — pixels ARE the output.
- **Key technique:** Multi-layer ViT feature statistics matching (mean, variance per layer) + LLM token cross-entropy
- **v2 update:** Expanded from LLaVA-only to 7+ VLMs (Gemma, Qwen3-VL, DeepSeek-VL)
- **Our take:** Use the multi-layer feature matching loss, but apply it to our VAE/INR optimization pipeline for better image quality

### Gemini Embedding 2 (Google, March 2026)
- **3,072d** natively multimodal (text, image, audio, video, docs)
- **API-only** — no open weights, no local execution, no gradient backpropagation
- **Verdict:** Unusable for optimization-based approaches. Dead end for our core use case.

### LUCID-SAE (February 2026)
- **Paper:** arxiv 2602.07311
- Unified vision-language SAE with optimal transport matching
- Patch-level grounding

### Platonic Representation Hypothesis (ICML 2024, active 2026 debate)
- All sufficiently capable models converge toward the same representation of reality
- Validates cross-model comparison — different encoders should produce increasingly similar results
- Quanta Magazine feature: January 2026

## Encoder Landscape (March 2026)

### Best Options (differentiable, open weights, fits M1 Max 64GB)

| Model | Modalities | Dims | Params | Memory | Release | License |
|-------|-----------|------|--------|--------|---------|---------|
| SigLIP 2 So400m | text, image | 1152 | 400M | ~1.6GB | Feb 2025 | Apache 2.0 |
| Qwen3-VL-Embedding-2B | text, image, video | 2048 | 2B | ~4GB | Jan 2026 | Apache 2.0 |
| NVIDIA llama-nemotron-embed-vl-1b-v2 | text, image | 2048 | ~1.7B | ~3.4GB | Mar 2026 | NVIDIA Open |
| LCO-Embedding-Omni-3B | text, img, audio, video | varies | ~5B | ~10GB | Oct 2025 | Apache 2.0 |
| Omni-Embed-Nemotron-3B | text, img, audio, video | 2048 | 4.7B | ~9.4GB | Oct 2025 | Noncommercial |
| ImageBind | 6 modalities | 1024 | ~1.2B | ~3GB | 2023 | MIT |
| LanguageBind | 6 modalities | 768 | ~300M/mod | ~1.2GB/mod | 2023-2024 | MIT |
| CLAP (LAION) | text, audio | 512 | ~300M | ~1.2GB | 2022 | Apache 2.0 |
| Nomic Embed Vision v1.5 | text, image | 768 | 93M | ~0.4GB | 2024 | Apache 2.0 |

### Key Insight: Dual-encoder vs LLM-based

Dual-encoders (SigLIP 2, ImageBind, CLAP) are 5-25x cheaper per backward pass than LLM-based encoders (Qwen3-VL, LCO-Omni). For optimization loops running 500+ steps, this is the difference between 5 minutes and an hour. Use dual-encoders for optimization, LLM-based for single-pass quality checks.

## Techniques Surveyed

### From Anthropic Interpretability
- Sparse autoencoders on Claude 3 Sonnet (Scaling Monosemanticity, May 2024)
- Genuinely multimodal features — "Golden Gate Bridge" fires on text AND images
- Circuit tracing / attribution graphs (May 2025)
- No direct embedding-to-image rendering — text-centric visualization

### Feature Visualization Evolution
- Generator-based activation maximization (our approach)
- Implicit Neural Representations (INR) as alternative to VAE latents
- Concept Bottleneck Models for interpretable intermediate representations

### Cross-Modal Rendering
- AudioLDM / CLAP for audio from embeddings
- ImageBind-as-guidance in diffusion (Xing et al. 2024) — validates our approach
- Brain decoding (EEG-to-image, NeurIPS 2024) — architecturally identical to our approach

### Representation Engineering
- Steering vectors — extract meaningful directions from activation space
- Could render "what does the truthfulness direction look like?"
- Cross-modal steering works (text vectors steer visual understanding)

## SolidGoldMagikarp Connection
- LessWrong article about anomalous "glitch tokens" in GPT
- Found by clustering token embeddings — same technique as our embedding space exploration
- Embedding geometry has structure: centroids, clusters, dead zones
- Nobody has rendered glitch token embeddings as images/audio — opportunity for our project
- Gradient descent on token embeddings parallels our optimization approach
