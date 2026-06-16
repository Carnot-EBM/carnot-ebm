# Audit: the .397 capstone (exp4301) BLOCKED and reports spurious all-False — the real scorecard

**Outer-loop audit, 2026-06-16.** exp4301 reports `headline_outcome: blocked_v397_artifacts_missing`
and defaulted EVERY scorecard boolean to False — `cross_generator_moat_closes: False`,
`in_generation_moat_holds: False`, `efficiency_pareto_hardened: False`, `paper_ready: None`.
**This is a capstone robustness bug, NOT the real result.** Do not read the .397 scorecard as
"everything failed."

## Root cause: one missing artifact blocked the whole aggregation

`missing_upstream_artifacts: [{artifact_key: 4294_efficiency, experiment_id: 4294, reason: missing}]`.
exp4294 (the C1 efficiency-harden "strong judge") **never produced an artifact** — C1 failed 3×
(2h wall-clock cap + a `strong_judge.py` bug) and was 3-fail-skipped. The capstone treats ANY
missing required upstream as a hard block and defaults all booleans to False — even for axes whose
artifacts ARE present and conclusive. That is the bug: a missing *efficiency* artifact should not
zero out the *cross-generator* verdict.

## The real .397 scorecard (from the intact source artifacts)

| Axis | Truth | Source |
|---|---|---|
| **Cross-generator moat** | ✅ **CLOSED** (legit, non-degenerate) — `cross_generator_delta +0.50`, CI95 [0.29,0.71], vote@1=0.25, oracle@K=0.75 (real headroom), passes the non-degenerate guards | exp4291 (present) |
| Partial-state diffusion scorer | ✅ built, leak-free (AUROC 0.966 — yellow-flag, leak re-check advised) | exp4292 |
| In-generation moat | ❌ **NOT held** — degenerate controls (rfg=unguided=entrgi=0.3 no-ops); correctly flagged + excluded | exp4293 (quarantined) |
| Efficiency Pareto hardened | ❓ **UNRESOLVED** — C1/exp4294 failed 3× (never ran) | exp4294 (MISSING) |
| Self-learning (tier-2 fixed) | see exp4295 | exp4295 |
| ARC | total_levels per exp4296 | exp4296 |

So the headline is the OPPOSITE of the capstone's all-False: the **cross-generator axis CLOSED**
(the last open axis of the selection moat). The only genuine negatives are the in-generation moat
(degenerate controls) and the still-open efficiency-harden (failed task, not a measured null).

## Recommendations

1. **The capstone code should aggregate AVAILABLE artifacts and report missing ones as per-axis
   gaps**, not hard-block + default-all-False. A missing artifact for one axis must not zero out a
   conclusive verdict on another. (Robustness fix in the capstone module; the conductor imports it,
   so it lands on the next restart.)
2. **.398 planner: read the TRUE scorecard above, not the blocked exp4301.** The cross-generator
   axis is CLOSED; do not re-open it as if it failed. The open items are the efficiency-harden
   (re-scope C1 so it fits the window — it doomed-looped on the 2h cap) and the in-generation moat
   (needs differentiated controls per the exp4293 audit).
3. `paper_ready` was not computed for .397 (capstone blocked) — it remains as last computed in .395
   (`True`, within-distribution selection claim). The cross-generator close STRENGTHENS that claim
   but the capstone didn't record it.

## Provenance
- exp4301 `results/experiment_4301_capstone_v397.json` (blocked)
- exp4291 (cross-generator, the real win) + exp4293 audit
  (`docs/research-notes/exp4293-in-generation-moat-degenerate-controls-audit-2026-06-16.md`)
