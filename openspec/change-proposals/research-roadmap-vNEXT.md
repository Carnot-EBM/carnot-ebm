# Research Roadmap vNEXT (Milestone 2026.05.139)

**Title:** Phase-16: Formal KAN Verification, EBM-Guided Reasoning, and Thermodynamic Denoising Simulation
**Status:** DRAFT

## 1. Context and Prior Milestone (2026.05.138)
The previous milestone (.138) completed the Phase-15 tasks, which integrated Symbolic-KAN, tested energy-based fine-tuning concepts, and finalized the formal orchestration. The pipeline demonstrated stability but highlighted three key gaps:
1. **Continuous Self-Learning (FR-11):** The continuous learning pipeline generated constraints but lacked a fully automated and verified DPO loop to train on generated correct/incorrect pairs without forgetting.
2. **KAN Formal Verification:** While Symbolic-KAN was integrated, we lack a formal mathematical guarantee (e.g., via Mixed Integer Linear Programming and Piecewise Affine abstractions) for the bounds of these networks.
3. **Hardware / Sampling Capabilities:** The `thrml` software provides a path to Denoising Thermodynamic Models (DTM), but we haven't leveraged it to perform EDDP benchmarking for diffusion-like constrained sampling.

## 2. Milestone Objectives
- **Objective 1 (Reasoning):** Introduce an EBT-style partial-trace energy scoring prototype to evaluate intermediate reasoning steps and feed them into a DPO loop.
- **Objective 2 (Learning):** Train a DPO adapter using 2,000 verified reasoning traces derived autonomously, fully realizing the FR-11 autonomous self-learning loop.
- **Objective 3 (Verification):** Build a Piecewise Affine (PWA) abstraction for KAEMEnergy and encode it as a MILP problem to verify bounds.
- **Objective 4 (Hardware/Sampling):** Simulate Denoising Thermodynamic Models (DTM) with `thrml` and compute the Energy-Delay-Deficiency Product (EDDP).

## 3. Phase Descriptions

### Phase 1: Energy-Guided Reasoning & Continuous Self-Learning
We build a dataset of 2,000 verified traces using local SOTA models, guided by partial-trace energy scoring, and apply Direct Preference Optimization (DPO) to fine-tune the reasoning adapter.
- Exp 1799: Implement partial-trace energy scoring prototype.
- Exp 1800: Generate 2000 verified reasoning traces.
- Exp 1801: DPO training on verified pairs.
- Exp 1802: Evaluate continuous self-learning non-forgetting and soundness.

### Phase 2: Formal Verification of KAN layers
To trust the KAN energy tier, we verify its bounds mathematically using MILP.
- Exp 1803: Piecewise affine (PWA) abstraction of KAEMEnergy.
- Exp 1804: MILP encoding of PWA abstractions.
- Exp 1805: End-to-end formal verification smoke test on KAEM.

### Phase 3: Hardware-Accelerated Simulation
We simulate the physics-based probabilistic computing layer to prepare for future Extropic TSU integration.
- Exp 1806: Denoising Thermodynamic Model (DTM) simulation using `thrml`.
- Exp 1807: EDDP benchmarking of the DTM simulator vs MCMC.

### Phase 4: Capstone E2E Validation and Retro
We run the full verifying-repair pipeline across our mandated SOTA GGUF models using the new DPO-tuned adapter, KAN verifier, and partial-trace energy scores.
- Exp 1808: Capstone E2E with Qwen3.6-35B-A3B.
- Exp 1809: Capstone E2E with Gemma4-31B-it.
- Exp 1810: Capstone E2E with Gemma4-26B-A4B-it.
- Exp 1811: Milestone Retrospective.

## 4. Hardware Requirements
- Dual RTX 3090 CUDA for the SOTA GGUF inference and DPO training.
- CPU for MILP solvers (Z3/PySAT) and `thrml` simulation.

## 5. Dependency Graph
Phase 1 tasks are sequential (1799 -> 1800 -> 1801 -> 1802). Phase 2 tasks are sequential (1803 -> 1804 -> 1805). Phase 3 tasks are sequential (1806 -> 1807). Phase 4 capstones require Phase 1 and 2 to be complete.