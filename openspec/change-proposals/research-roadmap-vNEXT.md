# Research Roadmap vNEXT (Milestone 2026.05.170)

**Date:** 2026-05-14
**Status:** DRAFT

## 1. Context and Retrospective

Milestone `.169` successfully laid the groundwork across multiple parallel tracks:
- **Hardware Sovereignty:** Proved PolarFire SoC passive execution of the pure-Python verifier subset.
- **Phase 4 Active Inference:** Scaled continuous active inference to $n=16, 32$, evaluating the $\Delta\alpha$ scaling limits.
- **THRML Bias Investigation:** Differentiated between finite-N and systematic bias in the `carnot` vs `thrml` joint underestimate.
- **Phase 1 Ship Track:** Successfully executed a dry-run PyPI build and Twine verification.

However, three major gaps remain between our current architecture and the PRD vision:
1. **Continuous Self-Learning (Gap 1):** The autoresearch loop is currently constrained to searching the architecture space; it lacks a native gradient-based fine-tuning mechanism (like Energy-Based Fine-Tuning) to incrementally update the LLM solver component directly against the energy landscape.
2. **Hard Constraint Satisfaction in Continuous Space (Gap 2):** While Ising and Boolean verifiers provide hard discrete constraints, the Phase 3 continuous models rely on soft penalties and Langevin sampling. Recent 2025/2026 literature (e.g., CASAL) introduces primal-dual split augmented sampling to guarantee hard physical and mathematical constraints during generative modeling.
3. **Phase 2 Asymptotic Hardware Execution (Gap 3):** The transition to discrete SB RTL (KV260) or Extropic TSU must move beyond simulated bounds and record true latency to establish the performance baseline needed before Phase 3 training.

## 2. Milestone Objectives

Milestone `.170` aims to close these gaps by injecting 2025/2026 state-of-the-art EBM methods into the Carnot framework, while shipping mature Phase 1 components.

### Objective 1: Implement CASAL for Hard Constraints
Integrate strictly constrained generative modeling via Split Augmented Langevin Sampling (CASAL) to replace soft-penalty Langevin samplers, providing zero-false-accept guarantees in continuous EBMs.

### Objective 2: Continuous Self-Learning with EBFT
Implement the Energy-Based Fine-Tuning (EBFT) feature-matching objective to allow the EBM to generate semantic feedback for the `unsloth/gemma-4-26B-A4B-it-GGUF` and `unsloth/Qwen3.6-35B-A3B-GGUF` solvers, enabling unsupervised self-improvement.

### Objective 3: Accelerate KAEMEnergy via SineKAN
Replace the B-spline inverse-transform sampling in the KAEMEnergy fast-path with periodic sine grids (SineKAN) or Feature-Enriched KANs (FEKAN), targeting a 10x inference speedup.

## 3. Phase Descriptions and Task Graph

The 12 experiments in this milestone are structured across four phases.

### Phase 1: Foundation, Ship, and Fixes
- **Exp 1685:** PyPI Publish Actual (`twine upload`).
- **Exp 1686:** THRML bias correction (implementing fixes found in .169).
- **Exp 1687:** KV260 Vivado Synthesis & bitfile generation retry.

### Phase 2: Continuous Hard Constraints (CASAL)
- **Exp 1688:** CASAL Primal-Dual sampler implementation in `carnot.samplers`.
- **Exp 1689:** CASAL vs MCMC Langevin verification on $n=16, 32$ continuous landscapes.
- **Exp 1690:** Integration of CASAL as a Phase 3 continuous fast-path verifier.

### Phase 3: Continuous Self-Learning (EBFT)
- **Exp 1691:** Implement the EBFT loss function in JAX.
- **Exp 1692:** E2E Autoresearch loop trial: fine-tune the solver parameter subset using EBFT on held-out constraints (using `unsloth/gemma-4-26B-A4B-it-GGUF`).
- **Exp 1693:** Transpile the successful JAX EBFT/CASAL structures to `carnot-samplers` (Rust).

### Phase 4: KAN Acceleration & Retro
- **Exp 1694:** Implement SineKAN/FEKAN substitute for KAEMEnergy splines.
- **Exp 1695:** Benchmark SineKAN vs baseline KAEMEnergy.
- **Exp 1696:** Milestone .170 Retrospective.

## 4. Hardware Requirements
- **Local:** Dual RTX 3090 (for running Qwen 3.6 35B and Gemma 4 31B GGUFs).
- **Edge:** KV260 board (if available, for Exp 1687 execution).
- **Compute:** CPU-bound tasks rely on standard parallelism. JAX targets CPU or ROCm/CUDA.
