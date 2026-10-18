# Carnot Research Roadmap: v2026.05.125

**Milestone Title:** Phase-3 Energy-Based Reasoning, Certified Self-Learning, and Energy-Guided Decoding
**Status:** DRAFT
**Author:** Carnot Research Planning Agent
**Date:** 2026-05-09

## What the Previous Milestone (.124) Proved

Milestone `.124` advanced the foundational components for latent space navigation and hardware abstraction:
1. **Adaptive Energy Landscapes:** Successfully configured dynamic energy landscapes (Exp 1624).
2. **KANELÉ Hardware Synthesis:** Completed LUT-mapping logic synthesis and latency accounting for KANELÉ (Exp 1621, 1623).
3. **EBM vs LLM Task Allocation:** Validated a router for dispatching tasks between explicit EBM solvers and LLMs (Exp 1625).
4. **Nabla-Reasoner Prototype:** Proved that continuous latent optimization is a viable path, although convergence issues blocked live SOTA integration (Exp 1616, 1617).

## Biggest Gaps to PRD Vision

1. **Robust Multi-Step Latent Reasoning:** The Nabla-Reasoner failed to converge during live SOTA validation. Carnot must achieve stable energy-guided latent trajectory optimization before Phase 3 can ship.
2. **Continual Learning without Catastrophic Forgetting:** FR-11 query-time memory policies need a verifiable mechanism to ensure that promoting new policies doesn't break established constraint invariants.
3. **Energy-Guided Generation Guarantees:** We need tighter coupling between the verifier's explicit energy scores and the LLM's decoding process to structurally prevent hallucinations without post-hoc rejection loops.

## Milestone .125 Phases

### Phase 1: EBRMs and Latent Trajectory Convergence
Focuses on repairing the continuous latent optimizer and testing Energy-Based Reasoning Models (EBRMs) using local SOTA GGUF models (`unsloth/Qwen3.6-35B-A3B-GGUF` and `unsloth/gemma-4-31B-it-GGUF`).
- **Exp 1627:** Nabla-Reasoner convergence debugging and learning rate sweep.
- **Exp 1628:** EBRM-style continuous latent trace scoring prototype.
- **Exp 1629:** Live SOTA validation of EBRM trajectory optimization.

### Phase 2: Continual Learning and Certified Updates
Implements "Self-Modeling Generative Intelligence" (SMGI) principles to guarantee non-forgetting in FR-11.
- **Exp 1630:** LTLZinc temporal constraint benchmark expansion for non-forgetting checks.
- **Exp 1631:** Integrate SMGI "certified update" logic into the FR-11 pipeline.
- **Exp 1632:** FR-11 query-time continuous self-learning with the non-forgetting gate.

### Phase 3: Constraint Satisfaction & Guided Decoding
Applies Pi-net style differentiable projections and Energy-Guided Decoding directly to the mandated GGUF models.
- **Exp 1633:** Pi-net style differentiable projection layer prototype.
- **Exp 1634:** Compare Pi-net projection against T-SKM on CCTU constraints.
- **Exp 1635:** ConsFormer-style refiner prototype for FoVer CSPs.
- **Exp 1636:** Energy-Guided Decoding implementation using `unsloth/gemma-4-26B-A4B-it-GGUF`.

### Phase 4: Architecture & Hardware Unblocking
Finishes the hardware validation delayed in .124 and cleans up the architecture.
- **Exp 1637:** Vivado linting integration preflight check.
- **Exp 1638:** KANELÉ RTL simulation on synthesized LUT mappings.
- **Exp 1639:** Milestone 125 Retrospective.

## Dependency Graph
```
Exp 1627 ──> Exp 1628 ──> Exp 1629
Exp 1630 ──> Exp 1631 ──> Exp 1632
Exp 1633 ──> Exp 1634 ──> Exp 1635 ──> Exp 1636
Exp 1637 ──> Exp 1638
```

## Hardware Requirements
- **Local SOTA inference:** Dual RTX 3090 GPUs.
- **FPGA Simulation:** Vivado 2023.2 locally installed.
