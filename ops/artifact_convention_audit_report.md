# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 7 |
| AGGREGATE_ONLY | 1 |

## experiment_7020_counterexample_belief_ledger.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The deterministic belief ledger passed all readiness checks, including construction, safety, belief-state coverage, leakage prevention, and fresh-process replay, without claiming future utility.

## WHAT IS MISSING
nothing; `"gate_check_summary"` records every check’s expected and observed values, while `"per_event_results"` provides event-level rows.

## THE CHECK A READER CANNOT DO
none

## experiment_7021_prospective_belief_utility.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Prospective utility for the counterexample-belief arm was not demonstrated because its improvement over recency-only did not satisfy the paired-interval gate.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7022_belief_ledger_cold_audit.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The belief ledger is shadow-safe but not promotable: `counterexample_belief` reportedly beat both controls in aggregate, but failed the nonnegative paired-interval gate.

## WHAT IS MISSING
The per-unit `per_decision_results`, `action_ranking_rows`, and `paired_delta_rows` are missing; they are only named under `cited_upstream_artifacts`, while `aggregate_recomputation_rows` contains arm aggregates and pooled paired deltas. The blocker itself is diagnosed in `gate_check_summary.failed_check`.

## THE CHECK A READER CANNOT DO
Did `counterexample_belief` improve action-ranking accuracy broadly across all three clusters, or was the aggregate advantage driven by one cluster or degenerate controls?

## experiment_7023_belief_query_api.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The deterministic bounded, game-blind belief-query API is ready because all acceptance checks passed across the recorded fixtures.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7024_belief_aware_e3_selector.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The belief-aware E3 selector is correctly wired into the live policy path, influences supported rankings, abstains safely on invalid or uncertain evidence, and leaves the disabled default path unchanged.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7025_belief_shadow_live_trace.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The live belief-shadow trace was blocked because `live_trace_execution` raised `ValueError: invalid ARC evaluation provenance: model_filename must be one GGUF filename`.

## WHAT IS MISSING
nothing; `verdict_class`, `honest_verdict`, and `gate_check_summary.failed_check` identify the blocker, while `gate_check_summary.observed_value` records the exact error.

## THE CHECK A READER CANNOT DO
none

## experiment_7026_held_mechanic_belief_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `exp7025-belief-shadow-live-trace.belief_shadow_trace_ready_score` was 0 instead of the required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7027_v615_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V615 capstone is blocked because required live evidence is absent.

## WHAT IS MISSING
nothing; `"gate_check_summary"` identifies `"required_live_upstream_complete"` as failed and records exact `"expected_value"` and `"observed_value"` entries, while `"scientific_gap_summary"` gives the underlying live-trace error.

## THE CHECK A READER CANNOT DO
none
