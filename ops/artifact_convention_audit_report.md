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

## experiment_8113_radial_decision_fit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked at conductor pre-gate because upstream check `exp8112-fit-source-capture.fit_capture_ready_score` returned 0 instead of the expected 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8116_v702_independent_online_memory.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8117_learning_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked prior to execution by an unmet pre-gate condition because upstream dependency `exp8116-independent-online-memory` had `learning_trajectory_ready_score` equal to 0 instead of the expected 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8118_v702_fresh_acquisition_cost.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8119_v702_batched_service_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8120_v702_arc_supervisor_delta.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8121_v702_hardware_batch_boundary.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8122_v702_capstone.json

**CANNOT_DETERMINE**

reviewer call failed
