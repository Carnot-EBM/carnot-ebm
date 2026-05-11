# Carnot Research Roadmap vNEXT: Phase-13 — Energy-Based Fine-Tuning, ROCE, and Hierarchical Constraints

**Created:** 2026-05-10
**Milestone:** 2026.05.136
**Status:** Planned (activates when milestone 2026.05.135 completes)
**Supersedes:** research-roadmap-vNEXT.md (milestone 2026.05.135)
**Informed by:** Experiments 1746-1758, EBFT, HRM, ROCE, HILED literature.

## What Phase-12 Proved

Phase-12 proved that EqM latency overhead could be reduced via sparse updates, Symbolic-KAN structures can effectively map to constraint-informed verification, and multi-agent frameworks raise the verification bound on formal benchmarks. However, scaling these into a unified, continuous self-learning system requires addressing the extraction bottleneck, sequence-level tuning, and memory bloat.

**The gap:** 
1. **Extraction bottleneck:** We need dynamic, open-world constraint elicitation (ROCE) instead of fixed schemas.
2. **Training feedback:** Continuous self-learning needs sequence-level energy objectives (EBFT) to reduce the reliance on expensive external verifiers.
3. **Memory management:** The continual learning buffer bloats indefinitely without semantic pruning.

## vNEXT Architecture: Dynamic Constraints & Sequence Energy

```
User Prompt (Open World)
     │
     ▼
┌──────────────────────┐
│ ROCE Extractor       │ → Extracts verifiable constraints dynamically
│ (Exp 1763)           │
└──────────────────────┘
     │
     ▼
┌──────────────────────┐
│ Hierarchical         │ → Maps abstract constraints to concrete rules
│ Reasoning (HRM)      │ → (Exp 1764)
└──────────────────────┘
     │
     ▼
┌──────────────────────┐
│ EBFT Self-Learning   │ → Continuous learning via sequence-level energy
│ (Exp 1759-1760)      │ → Semantic memory pruning (Exp 1761)
└──────────────────────┘
     │
     ▼
┌──────────────────────┐
│ Live SOTA Inference  │ → unsloth/Qwen3.6-35B-A3B-GGUF
│ + Hardware-in-Loop   │ → unsloth/gemma-4-31B-it-GGUF
└──────────────────────┘
```

## Phase 1: Energy-Based Fine-Tuning (EBFT) & Self-Learning

### Exp 1759: EBFT Sequence-Level Objective Implementation
Implement Energy-Based Fine-Tuning using sequence-level feature matching without external verifiers.
- **Deliverable:** `python/carnot/training/ebft_objective.py`

### Exp 1760: Continual Self-Learning with EBFT
Apply EBFT to the continuous self-learning pipeline to measure calibration stability.
- **Deliverable:** `scripts/experiment_1760_ebft_continual.py`

### Exp 1761: Semantic Pruning in Continual EBM Learning
Implement semantic pruning for the FR-11 memory buffer to prevent context saturation.
- **Deliverable:** `python/carnot/pipeline/semantic_pruning.py`

### Exp 1762: Continuous Stability Test on LTLZinc
Evaluate the pruned, EBFT-trained self-learning loop on the expanded LTLZinc dataset.
- **Deliverable:** `results/experiment_1762_stability.json`

## Phase 2: Open Constraint Elicitation (ROCE) & Hierarchical Models

### Exp 1763: Reasoning-Time Open Constraint Elicitation (ROCE)
Implement dynamic extraction of verifiable logical constraints from unstructured user prompts.
- **Deliverable:** `python/carnot/pipeline/roce_extractor.py`

### Exp 1764: Hierarchical Reasoning Model (HRM) Constraint Integration
Add abstract-to-detailed execution layering within the verifier to handle complex ROCE constraints.
- **Deliverable:** `python/carnot/models/hrm_verifier.py`

### Exp 1765: Evaluate ROCE + HRM
Run open-world reasoning tasks using the integrated ROCE + HRM stack.
- **Deliverable:** `scripts/experiment_1765_roce_hrm.py`

## Phase 3: Hardware-in-the-Loop & Live SOTA Execution

### Exp 1766: Hardware-In-The-Loop Energy Decoding (HILED)
Prototype offloading energy scoring/sampling to an external/simulated FPGA over AXI.
- **Deliverable:** `python/carnot/inference/hiled_decoder.py`

### Exp 1767: Full E2E Pipeline with Qwen3.6-35B-A3B
Test the complete vNEXT pipeline on the flagship MoE.
- **Deliverable:** `scripts/experiment_1767_e2e_qwen.py`

### Exp 1768: Full E2E Pipeline with Gemma4-31B-it
Test the complete vNEXT pipeline on the flagship dense model.
- **Deliverable:** `scripts/experiment_1768_e2e_gemma31.py`

### Exp 1769: Full E2E Pipeline with Gemma4-26B-A4B-it
Test the complete vNEXT pipeline on the middle MoE model.
- **Deliverable:** `scripts/experiment_1769_e2e_gemma26.py`

## Phase 4: Operations

### Exp 1770: Milestone .136 Retrospective
Aggregate the results and honestly document the gaps.
- **Deliverable:** `scripts/experiment_1770_retro.py`

## Dependencies
- Phase 1 must complete before Phase 2.
- Phase 3 relies on models and extractors built in Phases 1 and 2.
- All live inference must use the listed SOTA GGUF models.
