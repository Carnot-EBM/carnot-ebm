# Carnot Research Roadmap: Milestone 2026.05.183
**Date:** 2026-05-15
**Milestone:** 2026.05.183
**Status:** DRAFT

## 1. What Previous Milestone (2026.05.182) Proved
- **Measurement-Level Rescue:** `exp1745` investigated the Phase 4 per-step alpha disaggregation after `exp1741` proved the infimum `alpha_t'` was completely scale-invariant across substrates. The scale invariance suggests the aggregation metric was fundamentally hiding the substrate effect.
- **QAOD/NLA TPR Collapse:** `exp1746` diagnosed the 0.73 to 0.47 True Positive Rate collapse in the QAOD vs NLA head-to-head. It confirmed a corpus mismatch / label-noise issue.
- **EBT Mode Collapse:** `exp1747` investigated the suspicious 128% energy decrease in the EBT gradient refinement loop, pointing to a mode-collapse where the energy function was unbounded below.
- **Phase 1 Ship-Track:** The HuggingFace mirror attempt (`exp1748`) remained a critical, often-stranded carry-forward.

## 2. Architecture & Strategic Shifts
Our primary gaps between current state and the PRD vision are:
1. **Measurement of Constraint Reasoning:** Phase 4 needs a new measurement paradigm since `alpha_t'` is scale-invariant. We are pivoting to **Thermodynamically Constrained Neural Generation** metrics based on recent 2026 literature.
2. **Robust EBM Optimization:** We must bound the energy descent in EBTs to prevent mode collapse (the 128% decrease anomaly).
3. **Continuous Self-Learning (FR-11):** Memory growth and self-learning loops suffer from mode collapse over time. We will introduce **Dynamic Resolution** to stabilize continuous policy updates.

## 3. Phase Descriptions

### Phase 0: Carry Forwards & Fixes
- Retry HuggingFace publication with honest fallback.
- Sync QAOD/NLA corpus and rerun head-to-head correctly.

### Phase 1: Bounded Energy Descent
- Enforce hard bounds on the EBT gradient refinement loop to prevent mode collapse.

### Phase 2: Thermodynamic Constrained Measurement
- Implement a new thermodynamic penalty metric for Phase 4.
- Sweep substrate scaling (n=8/16/32/64) using this new metric to finally prove scale dependence.

### Phase 3: SOTA GGUF Thermo-Decoding
- Apply thermodynamic penalty sampling to the mandated SOTA local GGUF models.

### Phase 4: Continuous Self-Learning (FR-11)
- Apply Dynamic Resolution during continuous learning to guarantee no soundness mistakes or mode collapse.

### Phase 5: Hardware & Retrospective
- Perform no-synthesis substrate-aware KAN accounting for the KV260.
- Milestone retrospective.

## 4. Hardware Requirements
- Dual RTX 3090 (for running mandated SOTA GGUF models).
- CPU/Simulator for KAN and Thermodynamic metric evaluations. No Vivado or KV260 board-execution required for this milestone.
