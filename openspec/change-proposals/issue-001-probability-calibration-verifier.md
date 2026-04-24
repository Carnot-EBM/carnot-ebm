# Probability-calibration verifier for LLM-produced P(event) claims

**Status:** Draft change proposal.
**Origin:** [GitHub issue #1](https://github.com/Carnot-EBM/carnot-ebm/issues/1) (2026-04-24).
**Target milestone:** 2026.04.63 or later.
**Priority:** Medium. New verifier class, not blocking existing work.
**Depends on:** extractor layer (for evidence-atom extraction) + Ising sampler (for energy scoring).

## Summary

Add `ProbabilityCalibrationVerifier` as an opt-in verifier slotting between
EORM (tier 1) and the Ising sampler (tier 2), or as a side-car callable by the
extractor. Scores how tightly a reasoning chain supports an explicit
probability claim. Distinct from current EORM/Ising scoring — those measure
contradiction/consistency; calibration measures whether `P(X) = 0.62`
matches what the chain's evidence atoms imply.

See issue #1 for full motivation (weather forecast, reliability estimate,
medical triage, risk-scoring use cases where calibrated confidence matters
more than contradiction-freedom).

## Proposed experiments

### Exp A — `ProbabilityCalibrationVerifier` primitive + evidence-atom extractor

**Deliverable:** `python/carnot/verify/probability_calibration.py` +
`results/experiment_<N>_probability_calibration_primitive.json`.

**What it does:** Extract evidence atoms from a chain (reference counts,
base rates, cited subpopulations, cited mechanisms) and score the claim
`P(event | context) = p` against the implied range. Returns an energy score
and optional `[p_lo, p_hi]`.

**Acceptance gates:**

1. On a synthetic 100-item corpus (50 well-calibrated, 50 mis-calibrated by
   ≥ 0.2), AUROC ≥ 0.85 separating the two.
2. Per-claim latency ≤ 50 ms at p95 on CPU.
3. `honest_verdict` enum: `calibration_verifier_ships`,
   `calibration_auroc_below_gate`, `calibration_latency_above_budget`.

### Exp B — Live-data calibration corpus

**Deliverable:** 200 real probability claims from Qwen/Gemma CoT on
probability-laden prompts (weather, base-rate, triage scenarios), labelled
with graded ground-truth ranges.

### Exp C — Cascade wire-in

Opt-in flag on `VerifyRepairPipeline`. Per-claim overhead ≤ 50 ms; false-cutoff
on calibration-neutral outputs ≤ 1%.
