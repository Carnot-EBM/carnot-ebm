# Milestone 2026.05.219: Primal-Dual Decoding, Formal KANs, and Robust CSL

## Goal
Advance the Carnot EBM framework by integrating discrete-domain energy decoding, formal constraint verification, and resilient continuous self-learning, recovering from the blockages in `.218`.

## What Previous Milestone Proved
Milestone `.218` attempted to deploy inference-time EORM verification and KAN LUT-mapping for FPGAs. The operations surfaced hard constraints and repeated `DOOMED_RERUN` blocks in schematic integration (SCG-MEM) and LUT configuration. The primary insight is that continuous space operations require hard constraint boundary verification, and discrete decoding needs efficient Lagrangian multipliers to avoid exponential overhead. 

## Architectural Design

### Phase 1: Core Inference (Discrete Guided Decoding & Coherence)
The first phase integrates Primal-Dual Guided Decoding directly into the generation logits. Instead of Gumbel-softmax outer loops, it uses Lagrangian multipliers for inference-time adaptation. Simultaneously, an Energy-Based Constraint Network (EBCN) state-space model is added to score structural coherence text-wide, moving away from local Regex.

### Phase 2: KAN Formal Synthesis & Hardware Parity
We address the zero-false-accept requirements for Phase 2 hardware mapping. By incorporating Lipschitz-regularization (LipKAN) and SMT-solvers for Control Barrier Certificates (KAN4CBC), we can formally prove constraint bounds before synthesis. The KANELÉ FPGA BRAM/LUT bitstream will be re-attempted utilizing this safer verified topology.

### Phase 3: Continuous Self-Learning (FR-11)
To prevent catastrophic forgetting observed in past self-learning tiers, we implement a dynamic Lipschitz-bound update mechanism within the Continuous Self-Learning loop. This ensures that new constraint representations (from JEPA/Crosscoder) do not overwrite fundamental logical invariants.

### Phase 4: Capstone Evaluation
Live GPU evaluation combining the EBCN structural score, Primal-Dual token logit modifiers, and robust self-learning models using SOTA GGUF models.

## Phase Structure
- **Phase 0:** Archival of `.218` and setup of `.219`.
- **Phase 1:** Primal-Dual Decoding & Energy-Based Constraint Networks.
- **Phase 2:** Formal verification of KANs (LipKAN & KAN4CBC) and KV260 hardware synthesis retry.
- **Phase 3:** Lipschitz-regulated Continuous Self-Learning (FR-11).
- **Phase 4:** E2E GPU Benchmark & Retro.

## Dependency Graph
- Phase 1 tasks depend on Phase 0 setup.
- Phase 2 KAN hardware synthesis depends on KAN formal verification.
- Phase 3 CSL relies on Phase 2's Lipschitz boundary constraints.
- Phase 4 depends on all core implementations succeeding.

## Hardware Requirements
- Dual RTX 3090 (48GB total) for the SOTA GGUF inferences (`unsloth/Qwen3.6-35B-A3B-GGUF` or `unsloth/gemma-4-31B-it-GGUF`).
- Local KV260 FPGA board (target) / OSS-CAD-Suite for Synthesis Verification.
