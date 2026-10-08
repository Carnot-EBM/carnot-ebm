# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 2 |
| CANNOT_DETERMINE | 6 |

## experiment_8263_v714_protocol_conformance.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8264_v714_evidence_view_canary.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8265_fit_view_capture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked at conductor pre-gate because two upstream canary gate checks in `exp8264-evidence-view-canary` failed (`view_canary_ready_score` and `fit_capture_budget_ready_score` were 0 instead of expected 1).

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8266_tune_view_capture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked at conductor_pre_gate because upstream dependency exp8264-evidence-view-canary failed two gate checks (view_canary_ready_score and tune_capture_budget_ready_score both observed 0 instead of expected 1).

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8272_v714_arc_outcome_frontier.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8273_v714_kv260_evidence_cost_boundary.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8274_v714_gatemate_physical_delta.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8275_v714_capstone.json

**CANNOT_DETERMINE**

reviewer call failed
