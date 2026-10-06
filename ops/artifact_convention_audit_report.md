# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 3 |
| CANNOT_DETERMINE | 5 |

## experiment_8196_v708_selective_sealed_evaluation.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8197_v708_selective_decision_audit.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8198_calibrated_online_memory.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked at the conductor pre-gate because upstream dependency `exp8193-learning-qualification` failed two gate checks (`calibrated_memory_ready_score` and `stream_input_ready_score` both observed 0 when 1 was expected).

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8200_v708_request_trace_census.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8201_observed_service_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked at pre-gate because upstream artifact `exp8200-request-trace-census` had `request_trace_ready_score` equal to 0 instead of the expected 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8202_v708_arc_supervisor_frontier.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8203_v708_hardware_decision_boundary.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8204_v708_capstone.json

**CANNOT_DETERMINE**

reviewer call failed
