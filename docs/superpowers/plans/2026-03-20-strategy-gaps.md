# Strategy Gaps Implementation Plan

**Goal:** Add missing features to OptimizationStrategy: augmentation, callbacks, checkpoints. Then validate SigLIP2 and CLAP encoders with real model weights.

**Approach:** Extend the strategy's render() loop with the features from v1's engine._optimization_step(), then run slow tests.

---

## Tasks

### Task 1: Add augmentation to OptimizationStrategy
- Port augmentation from v1 engine._optimization_step() (random crop, flip)
- Apply after decode, before loss computation
- Respect config.augmentation settings

### Task 2: Add callback support to OptimizationStrategy
- Add optional `callback` parameter to render()
- Call with (step, breakdown, output) at each step
- Wire into CompositeLoss for per-step metric reporting

### Task 3: Add checkpoint support to OptimizationStrategy
- Save latent + optimizer state every config.checkpoint_every steps
- Support resume_from parameter

### Task 4: Add progress tracking to render()
- Use existing ProgressConfig system from core/progress.py
- Rich progress bar with loss components

### Task 5: Validate SigLIP2 with real weights (@pytest.mark.slow)
- Run the slow tests against actual model
- Fix any MPS compatibility issues
- Verify multi-layer feature extraction produces correct shapes

### Task 6: Validate CLAP with real weights (@pytest.mark.slow)
- Run the slow tests against actual model
- Fix any audio processing issues
- Verify encode_audio with real WAV file
