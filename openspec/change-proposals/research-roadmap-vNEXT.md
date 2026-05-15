# Research Roadmap: Milestone 2026.05.180

## Theme: Inference-Time Energy Optimization, Symbolic Verification, and Self-Distillation

**Status:** PLANNED
**Milestone:** 2026.05.180

### 1. Executive Summary
Building on the successful Phase 4 Bijection Integration and CEM Substrate in milestone .179, this milestone closes the three largest gaps to our PRD vision:
1. **Inference-Time Optimization:** Standard autoregressive models fail on hard constraints. We will adapt Energy-Based Transformer (EBT) techniques (arXiv:2507.02092) and TTT-Discover (arXiv:2601.16175) to perform gradient-based energy minimization at test time.
2. **Symbolic Verification Integration:** Inspired by Logical Intelligence's Kona/Aleph architecture, we will prototype a bridge to Lean 4 for formal symbolic verification of extracted constraints, moving beyond pure Ising/KAN checks for mathematical domains.
3. **Continuous Self-Learning Stability:** Utilizing recent findings on EBMs for continual learning (arXiv:2601.19897) and EB-SLE, we will implement an FR-11 self-distillation loop that prevents catastrophic forgetting.

### 2. Architecture Impact
- **Test-Time Training (TTT) Router:** A new pipeline stage that dynamically allocates compute (optimization steps) based on the initial energy of the LLM output.
- **Lean 4 Verifier Backend:** A new `VerifierBackend` complementing `IsingEBM` and `KAN` for strict formal logic constraints.
- **Self-Distillation Memory:** Updates the FR-11 continuous self-learning loop to maintain an energy-based memory of past constraints to prevent forgetting.

### 3. Phases and Experiments

**Phase 0: Foundation & Archival**
- exp1735: Archive .179, setup .180 metrics and tracking.

**Phase 1: Inference-Time Energy Optimization (EBT & TTT)**
- exp1736: EBT-style Gradient Refinement Loop Prototype
- exp1737: Entropic Utility Search Prototype (TTT-Discover)
- exp1738: SOTA EBT/TTT Evaluation on GSM8K

**Phase 2: Symbolic Verification Bridge (Kona/Aleph inspired)**
- exp1739: Lean 4 Verifier Backend Prototype
- exp1740: Symbolic Verification on Expert Sudoku

**Phase 3: Continuous Self-Learning & Self-Distillation**
- exp1741: FR-11 Live Policy Promotion with Self-Distillation (Continuous Self-Learning)
- exp1742: EB-SLE Reward Hacking Prevention Prototype

**Phase 4: Hardware & Retrospective**
- exp1743: TSU/FPGA Hardware Accounting for Lean 4 pre-processing
- exp1744: Milestone .180 Retrospective
