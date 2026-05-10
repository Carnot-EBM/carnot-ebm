# Carnot Research Roadmap: Milestone 2026.05.130 (Phase 7)

## Title: Continuous Constraint Discovery, Certified KArAt, and Deep Energy Decoding

**Status:** Proposed
**Author:** Carnot Planning Agent
**Date:** 2026-05-10

## 1. Context and Previous Milestone (.129)
Milestone 2026.05.129 successfully introduced Energy-Driven Steering (EDS), CRANE Decoding, and Kolmogorov-Arnold Attention (KArAt). However, key gaps remain in realizing the PRD's vision of autonomous directed self-learning. While the FR-11 continuous learning loop exists, it relies on manually designed constraints. Furthermore, the newly introduced KArAt attention lacks formal verification bounds, limiting its deployment in critical reasoning pipelines. 

## 2. Milestone Objectives
This milestone addresses the three largest remaining gaps:
1. **Continuous Self-Learning via Self-Play:** Move beyond manual constraint extraction by enabling the EBM to autonomously discover new constraints from failed LLM reasoning traces.
2. **Formal Verification of KArAt:** Leverage recent advances in KAN verification (e.g., MILP abstractions and algebraic geometry, arXiv:2602.06737) to certify KArAt attention blocks.
3. **Deep Energy-Guided Decoding:** Integrate the Nabla-Reasoner continuous latent optimization into the primary decoding loop for Test-Time Scaling (ETS).

## 3. Phase Breakdown

### Phase 1: Continuous Constraint Discovery (Self-Learning)
Implement a self-play loop where mandated SOTA models generate reasoning traces, the EBCN identifies coherence violations, and an automated extractor transpiles these failures into new NSVIF DSL constraints added to the FR-11 ledger.

### Phase 2: Formal Verification of KArAt Attention
Build on the PWA (Piecewise Affine) KAN abstractions to support KArAt. Encode the attention splines as MILP problems to verify Lipschitz bounds and prevent attention collapse during long-horizon reasoning. Incorporate Constraint-Informed KAN (CIKAN) regularizers.

### Phase 3: Deep Energy-Guided Decoding
Expand the Nabla-Reasoner from a prototype into a production Test-Time Scaling (ETS) decoder, directly guiding continuous latent states to satisfy EBCN bounds during generation.

### Phase 4: Emulation & Retrospective
Push the KV260 Vivado integration toward cycle-accurate simulation for larger Potts constraints and conclude with an operational retrospective.

## 4. Hardware Requirements
- **Local SOTA Testing:** Dual RTX 3090s for running `unsloth/Qwen3.6-35B-A3B-GGUF` and `unsloth/gemma-4-31B-it-GGUF`.
- **FPGA Simulation:** Vivado 2023.2+ simulator for cycle-accurate RTL verification (no physical KV260 board required yet).

## 5. Dependency Graph
- Exp 1683 (Self-Play Prototype) -> Exp 1684 (FR-11 Ledger)
- Exp 1686 (PWA KArAt) -> Exp 1687 (MILP Verification)
- Exp 1690 (Deep ETS) -> Exp 1694 (Full Pipeline Live SOTA)