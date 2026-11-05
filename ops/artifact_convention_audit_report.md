# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 4 |
| CANNOT_DETERMINE | 4 |

## experiment_8845_v757_indexed_witness.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The indexed witness evaluator preserves reference decisions across the full roster and passes semantic-readiness gates.

## WHAT IS MISSING
The remainder of "parity_rows": the supplied artifact ends mid-row, while "completed_count" reports 7200 and "acceptance_gates" reports success.

## THE CHECK A READER CANNOT DO
Does every intended unit have matching "indexed_decision" and "reference_decision"?
No rerun performed; truncation prevents checking the full roster.

## experiment_8846_v757_future_benefit_audit.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The audit reports a completed null future-benefit result in `"honest_verdict": "complete_null_future_benefit_audit"`.

## WHAT IS MISSING
Per-stream, per-arm Brier metrics for all 24 streams are not visible. `"per_stream_results"` is named in `"field_principles"`, but the artifact truncates inside `"influence_rows"` at `"estimate_without_s`, before those results can be inspected.

## THE CHECK A READER CANNOT DO
What were the Brier scores of `categorical_combined` and `categorical_frozen` on each of the 24 streams?

The unprovided remainder cannot be checked; it may contain the needed rows.

## experiment_8847_v757_indexed_cost.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The evaluation met gates for at least 20% paid-operation reduction, completed-decision parity, and coverage loss and semantic false-accept increase each bounded by 1%.

## WHAT IS MISSING
Matched per-unit paid-operation counts, decisions, coverage outcomes, and semantic false-accept outcomes for the compared arms. “acceptance_gates” records “passed”: true; visible “complete_request_rows” provide identifiers, timings, and completion status. The artifact ends mid-row, preventing assessment of whether the remaining rows supply the missing metrics.

## THE CHECK A READER CANNOT DO
Do matched per-unit measurements reproduce the operation-reduction and quality gates marked as passed?

The truncated remainder could not be checked.

## experiment_8848_v757_memory_size_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact reports completed cost measurements for 60 windows across ten paired blocks, with per-unit costs and paired differences, and explicitly makes no benefit claim.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

Execution was not rerun or independently verified; this assessment covers the supplied record only.

## experiment_8849_qwen_witness_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Source-span extraction was blocked because `runtime_repair_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8850_arc_outcome_transfer.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The audit was blocked because `arc_new_outcomes_available_score` was 0, while the gate required 1, as recorded in `gate_check_summary` and `gates_evaluated`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8851_v757_hardware_operation_budget.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Hardware acquisition should be deferred because no complete, supported physical workload benefit has been measured.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

Assessment uses only the supplied excerpt; no experiment was re-run, and physical acceleration remains unmeasured.

## experiment_8852_v757_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion reports complete accounting and owned validation passed, while natural generalization failed and activation was not authorized.

## WHAT IS MISSING
The artifact’s remainder: it cuts off inside "branch_conclusions" → "reduction" → "draw_ids", so the complete conclusions and any supporting per-unit metrics are unavailable.

## THE CHECK A READER CANNOT DO
Does the complete artifact include per-unit metrics supporting its comparative claims, if any? The truncated remainder cannot be checked and may contain that evidence.
