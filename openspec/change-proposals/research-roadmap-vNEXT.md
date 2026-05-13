# Research Roadmap vNEXT (2026.05.164)

**Milestone:** 2026.05.164
**Status:** Planned
**Focus:** Energy-Based Transformers, Differentiable Hard Constraints, and Schema-Constrained Inference

## Executive Summary

Milestone `.163` successfully implemented SMT-based constraint extraction and KAN piecewise-affine (PWA) MILP verification, establishing a bridge between real instruction-tuned responses and rigorous mathematical bounds. However, significant gaps remain between the current pipeline and the long-term vision of autonomous, continuous self-learning via energy landscapes (Phase 3). 

Recent arXiv literature (April/May 2026) underscores that "hard" constraint projection layers ($\Pi$Net, HardNet++) and Energy-Based Transformers (EBTs) are the prevailing state-of-the-art for embedding constraints directly into neural generation. Furthermore, training-free energy-guided decoding techniques such as Energy-Guided Test-Time Scaling (ETS) and Schema-Constrained Generation for Agent Memory (SCG-MEM) provide immediate pathways for applying constraint energy to inference loops without RLHF.

This milestone will:
1. Rebuild and scale constraint enforcement around differentiable hard constraint projections.
2. Introduce a proof-of-concept EBT module for generative reasoning.
3. Integrate SCG-MEM and ETS to upgrade Carnot’s Continuous Self-Learning Tier 2 (Memory Pattern Constraint Addition).

## Biggest Gaps to PRD Vision

1. **Verification-Generation Disconnect:** Carnot currently relies on post-hoc MCMC/Ising sampling to verify constraint satisfaction. The PRD goal of zero-false-accept continuous reasoning requires differentiable hard constraints ($\Pi$Net / HardNet++) or energy-guided decoders (EBT) integrated directly into the LLM loop.
2. **Brittle Constraint Memory (Tier 2):** Pattern extraction relies on hand-crafted or generalized rules. Applying Schema-Constrained Generation (SCG-MEM) will enforce strict structural alignment on Trace2Skill operations, resolving formatting hallucinations in memory storage.
3. **NPU / Hardware Activation:** The XDNA NPU toolchain was attempted in `.163` but needs verified models deployed to it. KAN hardware metrics (RM/BOP/NABS) must also be computed to ensure PWA/MILP abstraction scales to Extropic TSU / FPGA budgets.

## Architecture Evolution

```mermaid
graph TD
    A[Instruction-Tuned LLM] --> B{Draft Output}
    B --> C[SMT/LLM Extractor]
    C --> D[SCG-MEM / ETS Constraints]
    D --> E{Energy-Based Decoder}
    E --> F[$\Pi$Net Projection Layer]
    F --> G[Valid, Low-Energy Output]
    G --> H[Continuous Self-Learning (Tier 2 Memory)]
```

## Milestone Phases

### Phase 1: Hard Constraints and Schema Enforcement
We begin by adapting the recent $\Pi$Net and HardNet++ architectures to Carnot's symbolic verifiers. Simultaneously, we implement SCG-MEM for our Continuous Learning trace pipelines to ensure strict memory integrity.

### Phase 2: Energy-Based Transformer Prototype
Implementing a baseline EBT (Energy-Based Transformer) to model inference as gradient descent over an energy landscape. This aligns with Phase 3 foundation goals.

### Phase 3: Hardware & Pre-flight 
Following up on `.163`, we push the JEPA model to the NPU and perform necessary hardware resource counting on the new MILP abstractions.

### Phase 4: Retro & Docs
Synthesize findings, log constraint efficacy, and update spec traceability.

## Hardware Requirements
- Dual RTX 3090 (for SOTA GGUF generation)
- AMD XDNA NPU (for JEPA model offloading)
