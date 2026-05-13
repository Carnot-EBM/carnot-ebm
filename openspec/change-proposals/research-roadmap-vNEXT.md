# Research Roadmap vNEXT (Milestone 2026.05.155)

**Milestone:** 2026.05.155
**Title:** Energy-Native Trajectory Optimization, Continual Verification Skills, and Scalable Hardware Accounting
**Status:** DRAFT

## 1. What Previous Milestone Proved

Milestone `.154` established baselines for Continuous Latent Generative paths via COLD decoding, DOMINO speculative masking, and deep solver constraints using RUN-CSP and DeepSaDe. The integration proved that continuous reasoning and deep MaxSMT constraints could be modeled, but we still need scalable hardware accounting for these tiers and a robust continuous skill acquisition mechanism without forgetting.

## 2. Milestone Objectives

1. **Continuous Latent Representation:** Expand Energy-Guided Decoding and ConsFormer iterative refinement, targeting Phase 3 PRD goals.
2. **Continual Self-Learning:** Implement Routing without Forgetting and Audited Skill-Graph updates to fix FR-11 retention collapse.
3. **Hardware Accounting:** Translate KAN implementations into LUT-based models (KANELÉ) and integrate p-dit abstraction checks without claiming unauthenticated board execution.

## 3. Architecture Overview

```text
User Instruction
      │
      ▼
┌─────────────────────────────────┐
│ Reasoning-Time Extraction (ROCE)│  <-- Residual Drift Ledger
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ Energy-Guided Continuous Latent │  <-- COLD / ConsFormer / FAR layers
│ Generation (Mandated GGUFs)     │
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ Multi-turn FR-11 Skill Graph    │  <-- Routing without Forgetting
│ (No-Forgetting Promotion)       │
└────────────────┬────────────────┘
                 ▼
┌─────────────────────────────────┐
│ Deterministic Verifiers         │  <-- KANELÉ LUT Accounting, Curie-Weiss
└─────────────────────────────────┘
```

## 4. Phase Descriptions

### Phase 1: Architecture and Diagnostic Foundations
- **Exp 1982:** Continuous Latent Generative layer setup inspired by FAR.
- **Exp 1983:** Apply Energy-Guided Decoding on SOTA models for object hallucination mitigation.
- **Exp 1984:** Build LUT-based representation for the S2KAN tier (KANELÉ hardware accounting).

### Phase 2: Continual Learning and Skill Acquisition
- **Exp 1985:** ConsFormer refinement loop evaluation against deterministic solver baselines.
- **Exp 1986:** Validator-tree promotion ledger with no-forgetting checks (Routing without Forgetting).
- **Exp 1987:** Structure Snowballing guardrail task across constrained paths.
- **Exp 1988:** Audited skill-graph self-improvement via verifier-backed replay.

### Phase 3: Deep Solvers and Hardware Alignment
- **Exp 1989:** Graph preconditioning and p-dit accounting preflight.
- **Exp 1990:** PVF/Glauber metadata roundtrip over validators (interface audit).
- **Exp 1991:** Corrected Curie-Weiss parity (hardware_execution_claim=false).
- **Exp 1992:** Residual-drift ledger over compiled ROCE validator trees.

### Phase 4: Synthesis
- **Exp 1993:** Tri-SOTA E2E Integration v10.
- **Exp 1994:** Pre-retro Audit.
- **Exp 1995:** Retrospective and `.156` Planning.

## 5. Hardware Requirements

No new hardware acquisition is required. The dual RTX 3090 CUDA local SOTA runtime is expected to be functioning for the mandated GGUF models. KANELÉ and p-dit tasks are restricted to logic-level and resource accounting only (`hardware_execution_claim=false`).
