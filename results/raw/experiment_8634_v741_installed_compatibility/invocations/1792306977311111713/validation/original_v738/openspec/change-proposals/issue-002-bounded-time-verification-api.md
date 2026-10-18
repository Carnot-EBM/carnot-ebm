# Bounded-time verification API (budget_ms parameter)

**Status:** Draft change proposal.
**Origin:** [GitHub issue #2](https://github.com/Carnot-EBM/carnot-ebm/issues/2) (2026-04-24).
**Target milestone:** 2026.04.63. High leverage — unblocks CI-gating,
  IDE-embedded, edge-deployment use cases.
**Priority:** High. Small API change, broad impact.
**Depends on:** nothing new — cascade already has natural early-exit.

## Summary

Add `budget_ms: int | None = None` to `VerifyRepairPipeline.verify()` and
`verify_and_repair()`. If the budget expires mid-cascade, return the
highest-tier partial result reached so far with metadata
(`budget_exhausted`, `tier_reached`, `ms_consumed`). Today the cascade's
worst-case latency is unbounded; a hard candidate can push through tier 0c
(SinkProbe) → tier 0d (HalluField) → tier 1 (KAN/EORM) → tier 2 (JEPA-Reasoner)
→ tier 3 (Ising/SMT) with repair loops, producing wall-times measured in
seconds. Predictable upper bound per call matters more than best-possible
verdict for latency-constrained deployments.

See issue #2 for the full rationale and semantics.

## Proposed experiments

### Exp A — `budget_ms` primitive on `VerifyRepairPipeline.verify()`

**Deliverable:** API change in `python/carnot/pipeline/verify_repair.py` +
`results/experiment_<N>_budget_ms_primitive.json`.

**Acceptance gates:**

1. `verify(question, candidate, budget_ms=500)` returns within 500 ± 50 ms
   on p99 across a 200-candidate test mix of easy / medium / hard inputs.
2. When budget expires, returned verdict carries `budget_exhausted=True` and
   a valid `tier_reached` integer.
3. Honest-verdict on the verifier artefact: `budget_respected_across_mix` /
   `budget_violated_on_hard` / `budget_respected_but_always_fails_safe`.

### Exp B — Budget-aware `verify_and_repair()`

Extends Exp A to the repair path. Repair loops get their own nested budget;
parent budget shrinks through each repair attempt.

### Exp C — Budget-autotune heuristic

Optional helper that, given observed per-tier latency distributions,
recommends a `budget_ms` for a given p99 target. Non-blocking.

## Risks

- **Partial verdicts give false confidence.** If the early-exit tiers are
  permissive and tier 3 rarely catches their misses, `budget_exhausted=True`
  returning the tier-2 verdict may look "passed" when tier-3 would have
  failed it. Mitigation: `budget_exhausted=True` → verdict label is
  `abstain` by default, opt-in flag to honour the partial verdict.
- **Gaming the budget.** Downstream consumers pick the loosest budget they
  can get away with. Mitigation: log distribution of `budget_ms` values in
  production telemetry; alert on regressions.
