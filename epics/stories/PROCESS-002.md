# PROCESS-002 — Validate CUDA ORT batch_size >= 32 Crossover (Deferred from 2026.04.21)

**Status:** Open
**Origin:** Operational Retrospective 2026.04.21, action item RETRO-2026-04-20-D
**Carry-over from:** 2026.04.20 retro (original: Exp 259 finding, milestone 2026.04.20)
**Consecutive milestones deferred:** 2

## Problem

Exp 259 showed CPU ORT outperforms CUDA ORT 5.49× at batch_size=1 for the 9→1
linear gate.  The recommended hybrid strategy (CPU ORT gate + GPU LLM inference +
batched queries at batch_size=8) was never standardized.  The CUDA ORT crossover
point at batch_size >= 32 has never been validated.  Every experiment script that
uses the PredictiveVerifier still invokes ORT without the hybrid routing.

## Acceptance Criteria

- [ ] A new script `scripts/experiment_NNN_cuda_ort_batch_test.py` tests CUDA ORT
  at batch_size in [1, 4, 8, 16, 32, 64] and records latency per batch.
- [ ] The script identifies the crossover batch_size where CUDA ORT matches or
  beats CPU ORT.
- [ ] Results written to `results/experiment_NNN_cuda_ort_batch_results.json`.
- [ ] PredictiveVerifier updated (or documented) to use CPU ORT for
  batch_size < crossover and CUDA ORT for batch_size >= crossover.
- [ ] At least 15 tests cover the crossover logic; 100% coverage on the new module.

## Why This Matters

GPU inference latency is the dominant cost in live GSM8K / HumanEval benchmarks.
The hybrid routing identified in Exp 259 could reduce per-question latency by ~5×
at small batch sizes.  Without validating the crossover, every new experiment
runs suboptimally.

## Suggested Next Steps

1. Write `scripts/experiment_NNN_cuda_ort_batch_test.py` per the acceptance
   criteria above.
2. Add `hybrid_ort_routing` helper to `python/carnot/verifier.py` or equivalent.
3. Update the scaffold template in `_bmad/architecture.md` to show the hybrid
   pattern as the default ORT usage.
