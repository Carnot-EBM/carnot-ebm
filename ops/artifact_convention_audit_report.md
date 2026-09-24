# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 6 |
| AGGREGATE_ONLY | 1 |
| CANNOT_DETERMINE | 1 |

## experiment_7620_evaluation_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked at the conductor pre-gate layer because 4 of 9 upstream gate checks failed, beginning with `exp7617-schema-pilot.evidence_transport_ready_score` returning 0 instead of 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7624_v665_evidence_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_10012_gate_usefulness.json

**CANNOT_DETERMINE**

> Audit-integrity guard: quoted field(s) ['pair_id', 'candidate_id', 'scorable_rows', 'gate_passed'] do not appear in the artifact, so this verdict was downgraded and must not be acted on.

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Gate acceptance and rejection performance across filtering conditions differs between the BUDGET_150K and LIVE_SCORED arms, supporting the narrowed claim that lp85 exact-1.0 acceptance rests on one scorable held-out row.

## WHAT IS MISSING
Per-pair or per-unit rows containing individual unit identifiers, individual metric values, and scorable row indicators (e.g., `pair_id`, `candidate_id`, `scorable_rows`, `gate_passed`). While `appendix_gate_tables_by_arm` breaks down categories into `accepted_and_negative`, `accepted_and_positive`, `rejected_negative`, `rejected_positive`, `n_measured_pairs`, `n_pairs`, and `unmeasured`, it records only aggregate frequency counts and no unit-level data.

## THE CHECK A READER CANNOT DO
A reader cannot determine which specific engine pair accounts for the single `accepted_and_positive` count under `live_exact_1.0`, nor verify whether that acceptance actually rests on a single scorable held-out row for `lp85`.

## experiment_7625_v665_arc_supervisor_transfer.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
No supervisor redirects fired across the evaluated games and arms, establishing an honest null result with no causal treatment benefit (`complete_null_no_firings_nothing_to_refine`).

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7626_v665_native_service.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The native service protocol is ready with three-arm parity and durability demonstrated, but no speed or scientific benefit was found.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7627_v665_native_cost.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The native fixed total-cost comparative gate was met, although the separate 10× target was not.

## WHAT IS MISSING
Per-unit primary-comparison data: the top-level `paired_timing_rows` or `rows` containing each arm’s absolute cost for all 120 paired blocks. The present `acceptance_gate_results` and `honest_verdict` report the result, while `instrumentation_rows` explicitly cover only telemetry controls outside comparator selection.

## THE CHECK A READER CANNOT DO
Did the claimed total-cost advantage occur broadly across paired blocks and strata, or was it driven by a few extreme measurements?

## experiment_7628_v665_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The capstone is blocked because evidence transport was not ready and three required scientific-producer artifacts were missing.

## WHAT IS MISSING
nothing; `"gate_check_summary"` records each failed `"check"` with `"field"`, `"expected"`, `"observed"`, `"operator"`, `"path"`, and `"upstream"`.

## THE CHECK A READER CANNOT DO
none

## experiment_10013_planner_dedup_tiebreak.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The planner dedup/tiebreak experiment completed measurement of OFF, HUD_DEDUP, and HUD_DEDUP+TIEBREAK, with `guard_1_passed` reported as true.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
