# Carnot Milestone 2026.05.213: Process-Reward Energy, Ising Constraints, and Test-Time Compute

**Date:** 2026-05-17
**Status:** PROPOSED

## 1. Context and Previous Milestone Proofs

Milestone 2026.05.212 successfully delivered **Hardware-Assisted DAB (Discrete Auto-Regressive Biasing)**, **Substrate Shifting CSL**, and the **Continuous Latent Reasoning (CLR) Verifier Bridge**. We proved that mapping energy landscapes onto simulated LUT architectures can massively accelerate guided decoding, and that Substrate Shifting effectively prevents mode collapse during continuous learning.

However, three major gaps remain between our current state and the PRD vision:
1. **Outcome vs. Process Energy:** Our energy guidance is heavily weighted on terminal outcomes. True O1-style continuous reasoning requires step-level Energy-Based Process Reward Models (PREMs).
2. **Discrete Logic Bottleneck:** The CLR verifier bridge still relies on discrete SAT/SMT structures, limiting hardware acceleration. We need to translate logical constraints directly into Ising spin Hamiltonians.
3. **Static Generation Budgets:** Test-time compute is static. The EBM's energy variance should dynamically gate the generation budget, expanding search only where necessary.

## 2. Milestone 2026.05.213 Objectives

This milestone aims to:
- Introduce **Process-Reward Energy Models (PREMs)** for step-level energy guidance.
- Implement **Discrete-to-Ising Translation** to map hard constraints into native thermodynamic landscapes.
- Build a **Dynamic Test-Time Compute (TTC) Controller** driven by EBM partition function variance.
- Feed PREM intrinsic rewards back into **Continuous Self-Learning (CSL)**.

## 3. Architecture Phase Descriptions

### Phase 1: Process-Reward Energy Models (PREMs)
Shift the evaluation of energy from solely the final outcome to every intermediate reasoning step. By applying an EBM to step-wise latent states, we enable dense, continuous guidance (O1-style) for generative decoding.

### Phase 2: Thermodynamic Ising Constraints
Translate discrete constraint formulations (Z3/SAT) into Ising Hamiltonians. This formulation natively maps onto our ALPS (Annealed Langevin Posterior Sampling) module, allowing fully differentiable, continuous constraint satisfaction.

### Phase 3: Dynamic Test-Time Compute & CSL Integration
Utilize the uncertainty (variance) of the PREM landscape to dynamically scale the number of decoding steps. If the energy landscape is smooth, we sample directly; if highly multimodal, we increase the TTC budget. This phase also closes the loop by using PREM energy as an intrinsic reward for CSL, encouraging exploratory sampling.

### Phase 4: E2E Benchmarking & Retrospective
Integrate PREMs, Ising constraints, and TTC into a unified pipeline and benchmark on GSM8K and constrained logic tasks using SOTA local GGUF models.

## 4. Hardware Requirements
- SOTA local models: `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, `unsloth/gemma-4-26B-A4B-it-GGUF`
- GPU: Minimum 24GB VRAM for unsloth MoE variants.
- CPU/RAM: High-throughput memory for ALPS MCMC chains and Ising reductions.