# Research Roadmap v2026.05.203: Continuous Self-Learning, HardNet++ Constraints, and Symbolic-KANs

## 1. Context and Previous Milestone
Milestone `2026.05.202` completed the post-Bash-failure audit, PyPI re-check, CoT2-Meta retry, and citation-sweep cadence. The infrastructure is now stable, and we have successfully integrated local SOTA GGUFs (`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`). The primary gaps remaining to achieve the Carnot PRD vision are:
1. **Continuous Self-Learning:** Moving beyond static datasets to in-situ model updates via streaming Recursive Logic Subsystem (RLS) feedback.
2. **Hard Constraints in Continuous Latent Space:** Transitioning from soft Lagrangian penalties to deterministic, hard nonlinear constraint enforcement.
3. **Hardware Acceleration:** Efficient execution of verification constraints using LUTs directly on FPGAs (KV260).

## 2. Recent ArXiv Findings (2025-2026) Incorporated
- **EBTs (Gladstone 2025) / Kona (Bodnia 2026):** Energy-Based Transformers allow continuous latent reasoning traces, optimizing for energy minimization over AR sampling.
- **HardNet++ (2026):** Differentiable layers using damped local linearizations for hard nonlinear constraint enforcement.
- **Symbolic-KANs & KANELÉ (2026):** Kolmogorov-Arnold Networks mapped to splines and LUTs for high-speed hardware evaluation and symbolic logic embedding.
- **Interleaved Gibbs Diffusion (IGD) (2025):** Mixed discrete-continuous samplers.

## 3. Architecture Phase Design

This milestone focuses on unifying EBT reasoning loops with KAN-based hard constraints.

### Phase 1: Continuous Self-Learning Foundation (Tasks 1-3)
Establishing the RLS feedback loop to generate dense verification signals and integrating the EBT continuous objective with SOTA GGUFs for in-situ updates.

### Phase 2: KANs for Efficient Constraints (Tasks 4-6)
Prototyping Symbolic-KANs to encode discrete verification logic into continuous spaces, and developing Codex-generated KAGNN verifiers.

### Phase 3: HardNet++ & IGD Integration (Tasks 7-10)
Integrating HardNet++'s differentiable layers into the Carnot-Gibbs tier to ensure exact constraint adherence, tested via the new IGD sampler.

### Phase 4: Multi-Scale Hardware Prototyping (Tasks 11-13)
Synthesizing KANELÉ LUT-based logic for the KV260 FPGA and orchestrating a hardware-in-the-loop inference test combining macro-reasoning (GPU) with micro-constraints (FPGA).

## 4. Hardware Requirements
- **Local GPUs:** RTX 3090/4090 required for SOTA GGUFs (`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-26B-A4B-it-GGUF`).
- **FPGA:** KV260 board with `oss-cad-suite` for hardware synthesis.
