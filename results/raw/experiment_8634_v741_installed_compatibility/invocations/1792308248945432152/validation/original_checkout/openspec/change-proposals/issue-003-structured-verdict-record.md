# Structured verdict record (VerdictRecord dataclass)

**Status:** Draft change proposal.
**Origin:** [GitHub issue #3](https://github.com/Carnot-EBM/carnot-ebm/issues/3) (2026-04-24).
**Target milestone:** 2026.04.63. Pairs with issue #2 (budget_ms) — both extend the pipeline's public API surface.
**Priority:** High. Enables downstream routing, audit logging, confidence-weighted gating.
**Depends on:** nothing structural; expose fields already available internally.

## Summary

Promote `VerifyRepairPipeline.verify()`'s return type to a documented dataclass:

```python
@dataclass
class VerdictRecord:
    verdict: Literal["pass", "fail", "abstain"]
    energy: float
    calibrated_confidence: float    # monotonic in energy, comparable across tiers
    tier_reached: int               # 0..3
    rationale: str                  # short structured reason code
    ms_consumed: float
    budget_exhausted: bool = False
    evidence: dict[str, Any] | None = None  # optional per-tier trace
```

Currently consumers introspect internal state or re-run checks to recover
most of this. Standardizing the record unblocks: routing (send fail to human
review), confidence-weighted downstream gating, audit logging, autotuning
budgets from observed `ms_consumed`.

See issue #3 for the downstream-consumer use cases that motivate each field.

## Proposed experiments

### Exp A — `VerdictRecord` dataclass + calibration primitive

**Deliverable:** `python/carnot/pipeline/verdict_record.py` +
`results/experiment_<N>_verdict_record.json`.

**What it does:**

1. Define the dataclass with all 8 fields.
2. Fit a per-tier isotonic regression from historical
   `(energy, observed_correctness)` pairs — produces the
   `calibrated_confidence` mapping.
3. Populate the `rationale` field with a structured reason code (small
   enum, per-tier: e.g. `sink_skip`, `eorm_low_energy`, `ising_unsat`,
   `budget_exhausted_at_tier_1`).

**Acceptance gates:**

1. On a 500-sample test mix, `calibrated_confidence` is monotonic in
   `energy` (Kendall's τ ≥ 0.9 within each tier).
2. Per-tier `calibrated_confidence` values are on a comparable [0, 1]
   scale — a tier-1 0.7 and a tier-3 0.7 mean roughly the same
   probability of correctness (cross-tier reliability diagram curve).
3. `rationale` enum has no `other`/`unknown` placeholder on any real
   trace — every exit point names itself.

### Exp B — Backwards-compatibility shim

The existing return type (simpler tuple/dict) must still work for one
milestone. Add a `_legacy_format()` method and deprecate loudly.

## Risks

- **Calibration drifts.** The isotonic regression is trained once. As models
  evolve, the mapping becomes stale. Mitigation: schedule recalibration as
  a conductor-visible experiment every N milestones.
- **Over-reliance on `rationale`.** Short reason codes are a compression;
  consumers may start routing on them as if they were rich explanations.
  Mitigation: document `rationale` as a hint not a contract; require
  `evidence` for anything production-critical.
