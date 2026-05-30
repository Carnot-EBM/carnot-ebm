# Carnot Research Roadmap — Milestone 2026.05.314

**Milestone:** 2026.05.314
**Title:** Latent Energy Spills, Neural Uncertainty Phase Transitions, and Kona Global Reasoning
**Status:** PROPOSED
**Date:** 2026-05-30
**Experiment IDs:** exp3403–exp3415

---

## What Milestone .313 Proved

Milestone 2026.05.313 successfully advanced constraints and FR-11, but missed key tests and hardware deployment:

1. **GateMate N16 Bitstream Failure** (exp3392): The GateMate N16 bootstrap fix was unspecified/failed.
2. **Proximal-Gradient Constraints** (exp3393): Completed successfully.
3. **Continuous Learning** (exp3395, exp3396, exp3401): Energy-Based Replay Selection achieved better nonforgetting compared to random. CAS updates were verified. FR-11 stress test complete.
4. **LogicVault** (exp3399): Checked long context facts successfully.
5. **Missed Task**: The Kona-style global optimization benchmark (exp3394) did not execute.

## Three Biggest Gaps for .314

1. **GateMate Hardware Deployment**: We still lack a successful GateMate N16 bootstrap and board-local execution.
2. **True Continuous Global Reasoning (Kona style)**: We must adapt to the 2026 Kona-style continuous latent space global optimization for reasoning traces, escaping pure autoregressive dead-ends.
3. **Training-Free Latent Diagnostics**: We lack integration of the latest 2025/2026 EBM diagnostic findings, specifically measuring Latent Energy Spills and Neural Uncertainty Principle (NUP) phase transitions to detect hallucinations without retraining.

## New Research Opportunities (Late 2025 - Early 2026)

- **Training-Free Latent Energy Spills (2025)**: Detects hallucinations by measuring energy spills in latent representations.
- **Neural Uncertainty Principle (Mar 2026)**: Uses the Ising model to map hallucinations and adversarial fragility to phase transitions.
- **Abductive Reasoning as CSP (Nov 2025)**: Reformulates CoT verification as a Constraint Satisfaction Problem.
- **Structural Enforcement (Dec 2025)**: Defines "Semantic Violation Cost" to reject smooth falsehoods.
- **VGS Decoding (Mar 2026)**: Grounding-guided decoding overriding language priors.

---

## Architecture After .314

VerifyRepairPipeline (post-.314 target)
│
├── Kona-Style Global Reasoner (exp3408) ── solves constraint landscapes holistically
│
├── Training-Free Diagnostics 
│   ├── Latent Energy Spills (exp3406)
│   └── NUP Phase Transition Metric (exp3405)
│
├── Structural Enforcement / Abductive CSP (exp3407, exp3409)
│
└── FR-11 Self-Learning (exp3410)
    └── Continual Learning with Latent Energy Spills

---

## Phase Structure

### Phase A: Admin + Hardware Fix (exp3403-exp3404)
- exp3403: Archive .313 and activate .314
- exp3404: GateMate N16 Bootstrap fix (replaces exp3392)

### Phase B: Latent Space Diagnostics (exp3405-exp3407)
- exp3405: Neural Uncertainty Principle (NUP) Ising Phase Transition Metric.
- exp3406: Training-free Latent Energy Spills Detection (EBM).
- exp3407: Abductive Reasoning Constraint Satisfaction Problem (CSP) Pipeline.

### Phase C: True Continuous Global Reasoning & FR-11 (exp3408-exp3410)
- exp3408: Kona-Style Global Optimization Reasoning Benchmark (Sudoku).
- exp3409: Structural Enforcement "Semantic Violation Cost" integration.
- exp3410: FR-11 Continual Learning with Latent Energy Spills.

### Phase D: Evaluation and Validation (exp3411-exp3415)
- exp3411: EBM-CoT Verification scaling using unsloth/Qwen3.6-35B-A3B-GGUF.
- exp3412: VGS-style Grounded Decoding applied to constraints.
- exp3413: Cross-Corpus Evidence Matrix v39.
- exp3414: FR-11 Stress Test with NUP and Latent Spills.
- exp3415: Capstone v314.

---

## Dependency Graph

- A(exp3403 Admin) -> B(exp3404 GateMate)
- A -> C(exp3405 NUP)
- A -> D(exp3406 Spills)
- A -> E(exp3408 Kona)
- C -> F(exp3410 FR-11 Spills)
- D -> F
- D -> G(exp3407 CSP)
- E -> H(exp3409 Structural Cost)
- A -> I(exp3411 EBM-CoT)
- A -> J(exp3412 VGS)
- F -> K(exp3414 Stress Test)
- B -> L(exp3413 Matrix)
- G -> L
- H -> L
- I -> L
- J -> L
- K -> L
- L -> M(exp3415 Capstone)

## Hardware Requirements

- **GPU**: RTX 3090 required for inference and latent extraction.
- **Hardware Boards**: GateMate board required for exp3404.
