# Milestone 2026.05.207: NeSy-EBM Integration, Continuous Latent Reasoning, and Thermodynamic Benchmarks

**Status:** Proposed  
**Date:** 2026-05-16  

## 1. Context and Motivation

Milestone `2026.05.206` established foundational blocks for KAN Symbolization, SMT Verification Integration, EBM-CoT Latent Calibration, and DTM Simulation. However, the overarching PRD vision of replacing autoregressive models with zero-false-accept continuous latent planners remains incomplete. Recent literature (early 2026), particularly the Kona (Energy-Based Reasoning Models) and NeSy-EBM breakthroughs, demonstrates that treating reasoning as a continuous global Lagrangian optimization problem solves hard constraints (like Sudoku) at a 96.2% success rate—far beyond the capabilities of frontier LLMs.

This milestone closes the three largest gaps to the PRD vision:
1. **Continuous Latent Planning (Kona-Parity):** Connecting the gradient-based blocks and EBM-CoT adapters into a full continuous multi-step reasoning loop.
2. **Continuous Self-Learning (FR-11):** Integrating dynamic resolution and ActFocus token-level reweighting to solve catastrophic forgetting during EBM updates.
3. **Hardware Integration & Benchmarking:** Upgrading the DTM simulation to a stochastic gradient lattice random walk (SGLRW) and running a live GSM8K benchmark across the dual RTX 3090 cluster.

## 2. Architecture Impact

This milestone significantly evolves the Phase 3 (Continuous EBM) and Phase 4 (Constraint Resolution) architectures:
- **Neuro-Symbolic Encoding:** Hard constraints (from Phase 1 SMT logic) are encoded directly into the neural architecture as differentiable energy penalties via Maximum A Posteriori (MAP) inference.
- **Continuous Generative Trajectories:** `unsloth/Qwen3.6-35B-A3B-GGUF` and `unsloth/gemma-4-31B-it-GGUF` will be used as the decoding interface, mapping the continuous latent optimization (Energy Minimization) back to discrete tokens.
- **Thermodynamic Sampler Backend:** The `SamplerBackend` abstraction gains an SGLRW module that emulates physical stochastic hardware, preparing Carnot for future Extropic TSU integration.

## 3. Phase Descriptions

### Phase 0: Activation
- **Exp 2070:** Archive .206, activate .207.

### Phase 1: Kona-Style Continuous Latent Reasoning
- **Exp 2071:** Implement a global Lagrangian energy function that sums local symbolic constraint potentials.
- **Exp 2072:** Apply the Lagrangian optimizer to solve hard Sudoku using Kona's methodology, bypassing autoregressive generation.
- **Exp 2073:** Route the optimized latent thought representations back into local SOTA GGUFs for sequence decoding.

### Phase 2: Neuro-Symbolic Energy-Based Models (NeSy-EBM)
- **Exp 2074:** Create a symbolic encoder module that maps deterministic constraints into differentiable energy penalty tensors.
- **Exp 2075:** Implement Maximum A Posteriori (MAP) inference alternating training for the NeSy framework.
- **Exp 2076:** Connect Phase 2 MAP inference outputs to the KAN4CBC SMT verifier from .206, proving zero-false-accepts on a neuro-symbolic toy problem.

### Phase 3: Continuous Self-Learning (CSL) Loop
- **Exp 2077:** Implement ActFocus token-level energy reweighting for better credit assignment in multi-turn logic.
- **Exp 2078:** Implement dynamic resolution scaling for continuous EBM learning.
- **Exp 2079:** Build the end-to-end FR-11 CSL pipeline utilizing `unsloth/gemma-4-26B-A4B-it-GGUF`.

### Phase 4: Hardware Benchmarks and DTM
- **Exp 2080:** Upgrade the DTM simulation to a JAX-optimized hardware-ready stochastic gradient lattice random walk (SGLRW) sampler.
- **Exp 2081:** Run the SGLRW sampler on the dual RTX 3090 setup against a 100-sample GSM8K partition.

## 4. Hardware Requirements
- **Local SOTA GGUFs:** Tested on CPU/GPU depending on VRAM availability.
- **Dual RTX 3090:** Required for Phase 4 (Exp 2081) benchmark runs. If local CUDA execution fails, fallback to CPU benchmarking is acceptable for CI.

## 5. Dependency Graph
```text
Exp 2070 (Activation)
  |-- Phase 1 (Kona)
  |      |-- Exp 2071 --> Exp 2072 --> Exp 2073
  |-- Phase 2 (NeSy-EBM)
  |      |-- Exp 2074 --> Exp 2075 --> Exp 2076
  |-- Phase 3 (CSL Loop)
  |      |-- Exp 2077 --> Exp 2078 --> Exp 2079
  |-- Phase 4 (Hardware Benchmarks)
         |-- Exp 2080 --> Exp 2081
```