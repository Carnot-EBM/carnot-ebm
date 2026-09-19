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

## experiment_7401_v649_online_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The online audit was blocked because two required producer artifacts were missing or unreadable.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7402_v649_proposal_capture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the `one_owned_rtx3090_slot` precondition failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7403_v649_synthetic_memory.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The efficacy gates passed because the reported paid-query and full-cost ratio CI95 upper bounds beat their thresholds against both persistent comparator arms.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7404_live_memory.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because 2 of 6 upstream gates failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7405_v649_proof_audit.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The synthetic cohort passed the efficacy gate because both persistent methods achieved acceptable aggregate full-cost and paid-query ratio confidence bounds.

## WHAT IS MISSING
Per-stream or per-formula-family arm-level cost and paid-query measurements underlying `"full_cost_ratio_ci95_upper"` and `"paid_query_ratio_ci95_upper"`; `"erasure_witness_rows"` contains per-request diagnostic booleans but not those comparative metric values.

## THE CHECK A READER CANNOT DO
Did the reported efficacy hold broadly across the 32 independent stream groups, or was it driven by a few outliers or units with no headroom?

## experiment_7406_v649_arc_generalization.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was completed but disqualified because required evidence checks failed.

## WHAT IS MISSING
nothing; `"gate_check_summary.failed_checks"` identifies both failures and records each `"check"`, `"artifact_field"`, `"expected"`, `"observed"`, and `"upstream"` value.

## THE CHECK A READER CANNOT DO
none

## experiment_7407_v649_service_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The vectorized NumPy arm provides a positive full-service benefit over the scalar NumPy arm.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7408_v649_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The capstone claims completion with fourteen honest dispositions, while combined scientific benefit was not established and several branches remained blocked.

## WHAT IS MISSING
The artifact is truncated mid-value inside `"field_principles"`, so the actual `"gate_check_summary"` and any per-unit metric rows supporting the positive `"claim_matrix"` entries cannot be found; only summaries such as `"acceptance_gate_results"` and `"continuation_rows"` are visible.

## THE CHECK A READER CANNOT DO
Do later, omitted fields contain per-unit evidence showing that the claimed synthetic-memory and host-service benefits were broad effects rather than aggregate results driven by outliers or degenerate controls?
