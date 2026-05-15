# Research Roadmap vNEXT (Milestone 2026.05.188)

**Title:** Test Harness Recovery, KAEM Integration, and System 2 Energy-Based Transformers
**Milestone:** 2026.05.188
**Author:** Planning Agent
**Date:** 2026-05-15

## 1. Context and Previous Milestone Reality
Milestone `.187` experienced a catastrophic cascade failure due to broken pre-tests, leading to `Gemini CLI error: Wall-clock+idle timeout` during the initial phase. Consequently, nearly all downstream tasks were skipped (`GATE_BLOCK` or `DOOMED_RERUN_BLOCK`). Our first imperative is to restore the integrity of the test harness. Once the autonomous loop is unblocked, we must carry forward the stalled Fast-Slow Variant experiments and aggressively integrate the latest 2025-2026 ArXiv findings—specifically Kolmogorov-Arnold Energy Models (KAEM) and Energy-Based Transformers (EBT)—to modernize Carnot's core capabilities.

## 2. The 3 Biggest Gaps to PRD Vision
1. **Broken Autonomous Loop:** The core automated pipeline remains blocked by test failures. Carnot's autonomous research cannot progress until the 100% test pass requirement is restored and verified.
2. **"Black Box" MLP Energy Functions:** The PRD emphasizes interpretability and exact inference, yet we rely on opaque MLP-based energy landscapes that require slow MCMC. The recent KAEM (Kolmogorov-Arnold Energy Model) paradigm provides an exact-inference alternative using univariate splines.
3. **Catastrophic Forgetting & Lack of System 2 Thinking:** Our continuous self-learning track (FR-11) suffers from catastrophic forgetting, and our verifier lacks true iterative refinement. LSEBMCL (Latent Space EBM Continual Learning) and EBTs (arXiv:2507.02092) offer proven mechanisms for generative replay and iterative "System 2" energy minimization.

## 3. Phase Descriptions

### Phase 0: Infrastructure Recovery
**Focus:** Unblock the conductor.
We must fix the broken pretests with Opus-level supervision and conduct a deep retrospective of the `.187` timeout failure.

### Phase 1: Fast-Slow Variant Scale-Up & Continual Learning
**Focus:** Executing delayed .187 priorities and adding generative replay.
We will scale the Fast-Slow variant on SOTA GGUFs (`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`) and implement LSEBMCL (arXiv:2501.05495) to prevent catastrophic forgetting via pseudo-exemplar sampling.

### Phase 2: Kolmogorov-Arnold Energy Models (KAEM) & System 2 EBT
**Focus:** Architectural modernization based on late-2025/2026 literature.
We will prototype KAEM, replacing MLP energies with univariate KART splines to eliminate MCMC during prior sampling. Concurrently, we will implement Energy-Based Transformer iterative verification (System 2 thinking) using `unsloth/gemma-4-26B-A4B-it-GGUF`.

### Phase 3: Kona EBRM Integration & Milestone Audit
**Focus:** Continuous latent reasoning and safety.
We will explore Kona-style Energy-Based Reasoning Models (EBRM) with local energy edits, apply Lyapunov Control Barrier Functions (arXiv:2605.05530) for safety guarantees, and perform comprehensive audits.

## 4. Hardware Requirements
- **Compute:** Dual GPU (RTX 3090/4090 class) for SOTA GGUF inference (Qwen3.6-35B, Gemma-4-31B) and iterative EBT minimization.
- **Verification:** Fast CPU for KAEM spline processing and Python/Rust test suite execution.

## 5. Dependency Graph
```text
[Phase 0] exp1798 (Pretests)
   |--> exp1799 (.187 Retro)
   |--> [Phase 1] exp1800 (Fast-Slow Proto) --> exp1801 (Fast-Slow SOTA)
   |--> [Phase 1] exp1802 (LSEBMCL)
   |--> [Phase 2] exp1803 (KAEM Proto) --> exp1804 (KAEM Vis)
   |--> [Phase 2] exp1805 (EBT System 2)
   |--> [Phase 3] exp1806 (Kona EBRM)
   |--> [Phase 3] exp1807 (Lyapunov Safety)
```
