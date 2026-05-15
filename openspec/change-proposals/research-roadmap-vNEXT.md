# Research Roadmap vNEXT (Milestone 2026.05.187)

**Title:** Pretest Recovery, Fast-Slow Variant Scale-Up, and EBM Continual Learning
**Milestone:** 2026.05.187
**Author:** Planning Agent
**Date:** 2026-05-15

## 1. Context and Previous Milestone Reality
Milestone `.186` encountered a critical cascade failure: every task was skipped due to a broken pre-test environment (2 failing tests). The immediate priority is restoring the test harness integrity before any further research can proceed. Once unblocked, we must execute the delayed scale-up of the Fast-Slow variant (arXiv:2605.12484) and integrate recent 2026 advancements in Energy-Based Continual Learning (LSEBMCL) to address persistent PRD goals.

## 2. The 3 Biggest Gaps to PRD Vision
1. **Broken Autonomous Loop:** The core automated pipeline is blocked by test failures. Carnot's autonomous research cannot progress until the 100% test pass requirement is restored.
2. **Phase 4 Thermodynamic Validation vs Fast-Slow Training:** The PRD requires hardware-accelerated sampling and continual self-learning. The newly identified Fast-Slow Training (FST) variant aligns perfectly with Carnot's verifier-summary architecture, but its scale-up on SOTA local GGUF models remains stalled.
3. **Catastrophic Forgetting in FR-11:** Carnot's continuous self-learning track (FR-11) needs robust mitigation against forgetting. Recent 2026 papers on Latent Space EBMs (LSEBMCL) offer a direct path to generative replay via Langevin dynamics without mode collapse.

## 3. Phase Descriptions
### Phase 0: Infrastructure Recovery
Restore the test harness to a 100% passing state and execute the blocked `.186` retro alongside the delayed PyPI release.
### Phase 1: Fast-Slow Variant Scale-Up
Implement and scale the Fast-Slow Training architecture using `unsloth/Qwen3.6-35B-A3B-GGUF` and `unsloth/gemma-4-31B-it-GGUF`.
### Phase 2: EBM-Driven Continual Learning
Integrate LSEBMCL and Hybrid Energy-Distance mechanisms into the FR-11 loop to solve catastrophic forgetting.
### Phase 3: Verification & Audits
Execute QAOD vs NLA head-to-head evaluations, audit the findings, measure token-level energy telemetry, and conclude the milestone.

## 4. Hardware Requirements
- Dual GPU (RTX 3090/4090 class) for SOTA GGUF inference (Qwen3.6-35B, Gemma-4-31B).
- CPU for test recovery and symbolic validation.
