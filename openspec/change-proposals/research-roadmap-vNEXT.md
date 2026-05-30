# Carnot Research Roadmap — Milestone 2026.05.313

**Milestone:** 2026.05.313
**Title:** Proximal-Gradient Constraints, GateMate Recovery, and Energy-Guided Replay
**Status:** PROPOSED
**Date:** 2026-05-29
**Experiment IDs:** exp3391–exp3402

---

## What Milestone .312 Proved

Milestone 2026.05.312 successfully validated several components but hit a critical hardware failure:

1. **KV260 Hardware Execution Verified** (exp3381): Measured true hardware latency versus CPU.
2. **GateMate N16 Bitstream Failure** (exp3382): Failed due to `artifact_not_updated_past_bootstrap`. Conductor terminated because the script crashed before writing the artifact.
3. **EBM-CoT Trajectory Monitor works** (exp3383): An Interwhen-style verifier differentiates early commitment hallucinations.
4. **PEM and CAffNet Prototypes** (exp3384, exp3385): Prototyped Parallel Energy Minimization and Differentiable Constraint enforcement.
5. **FR-11 Locality-Aware Nonforgetting** (exp3386): Memory rollback and updates validated.
6. **LogicVault CDCL** (exp3388): Clause learning logic integrated.

## Three Biggest Gaps for .313

1. **GateMate Flashing Pipeline Fails**: The GateMate deployment must succeed; we need the bootstrap fix to properly write the artifact before the test routines can crash the agent loop.
2. **True Continuous Global Reasoning (Kona style)**: We must adapt to the 2026 Kona-style continuous latent space global optimization for reasoning traces, escaping pure autoregressive dead-ends.
3. **Continuous Learning Selection Bottleneck**: FR-11 lacks an energy-based replay selection mechanism to pick high-value samples (derived from Hopfield memory concepts).

## New Research Opportunities (Late 2025 - Early 2026)

- **Proximal-Gradient Constraint Networks (2026)**: Enforcing constraints directly inside the network using proximal descent.
- **Kona Global Optimization (2026)**: Global energy optimization over traces to avoid autoregressive dead ends.
- **Compress-Add-Smooth (CAS) Diffusion**: Formalizing continuous memory additions.

---

## Architecture After .313

VerifyRepairPipeline (post-.313 target)
│
├── Proximal-Gradient Constraints (exp3393) ── strictly enforces logical bounds on output
│
├── Kona-Style Global Reasoner (exp3394) ── solves constraint landscapes holistically
│
├── EBM-CoT Monitor (exp3397) ── aborts early upon energy spikes
│
└── FR-11 Self-Learning (exp3401)
    ├── Energy-Based Replay Selection (exp3395)
    └── CAS Diffusion Memory Updates (exp3396)

---

## Phase Structure

### Phase A: Admin + Hardware Fix (exp3391-exp3392)
- exp3391: Archive .312 and activate .313
- exp3392: GateMate N16 Bootstrap fix (replaces exp3382)

### Phase B: Differentiable Constraints + Kona Strategy (exp3393-exp3394)
- exp3393: Proximal-Gradient Constraint Layer integration.
- exp3394: Emulate Kona's global inference procedure on Sudoku boards.

### Phase C: FR-11 Continuous Learning Upgrades (exp3395-exp3396)
- exp3395: Energy-Based Replay Selection for FR-11.
- exp3396: CAS Diffusion algorithm for constraints.

### Phase D: Pipeline Integration and Evaluation (exp3397-exp3401)
- exp3397: EBM-CoT Verification Pipeline on Live GGUFs.
- exp3398: CAffNet Layer Out-of-Distribution Robustness.
- exp3399: LogicVault Multi-Turn Context Expansion.
- exp3400: Cross-Corpus Evidence Matrix v38.
- exp3401: FR-11 Continuous Learning End-to-End Stress Test.

### Phase E: Capstone (exp3402)
- exp3402: Capstone v313 aggregation.

---

## Dependency Graph

- A(exp3391 Admin) -> B(exp3392 GateMate)
- A -> C(exp3393 Proximal-Gradient)
- A -> D(exp3394 Kona)
- A -> E(exp3395 Energy Replay)
- E -> F(exp3396 CAS Diffusion)
- F -> G(exp3401 FR-11 Stress Test)
- C -> H(exp3397 EBM-CoT Live)
- D -> I(exp3399 LogicVault)
- B -> J(exp3400 Matrix)
- G -> J
- H -> J
- I -> J
- J -> K(exp3402 Capstone)

## Hardware Requirements

- **GPU**: RTX 3090 required for exp3393, exp3394, exp3395, exp3397, exp3399, exp3401.
- **Hardware Boards**: GateMate board required for exp3392.
