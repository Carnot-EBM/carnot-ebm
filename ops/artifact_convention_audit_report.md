# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 8 |

## experiment_7632_fit_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked before execution at the conductor pre-gate because four upstream gate checks failed on exp7631-schema-pilot.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7633_online_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because four of nine upstream gates failed.

## WHAT IS MISSING
nothing; `gates_evaluated` records each failed check and its `actual` and `expected` values, and `gate_check_summary` identifies the first failure.

## THE CHECK A READER CANNOT DO
none

## experiment_7634_evaluation_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked at the conductor pre-gate because upstream prerequisites failed, specifically that `exp7631-schema-pilot` had an `evidence_transport_ready_score` of 0 instead of 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7638_v666_evidence_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7639_v666_arc_goal_dedup.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment completed, but planner goal guard validation failed, so it claims no probability benefit.

## WHAT IS MISSING
nothing; `rows` records per-unit arm metrics, and `gate_check_summary.failed_operational_checks` names `validity` and `readiness`.

## THE CHECK A READER CANNOT DO
none

## experiment_7640_arc_wrapper_generalization.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked before measurement because all three upstream gates failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7641_v666_native_consumer.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The native consumer is ready: all 12 integration conditions passed, with no new probability, cost, or speed benefit claimed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7642_v666_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The v666 capstone milestone is blocked because required upstream external scientific producers are missing and conductor pre-gates failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
