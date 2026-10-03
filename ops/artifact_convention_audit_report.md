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

## experiment_8046_v697_branch_protocols.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8047_fit_score_capture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked prior to execution because upstream dependency exp8045-scorer-workspace failed the verdict_class gate with an observed value of "circular_positive" instead of "positive" or "null".

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8051_v697_feedback_constrained_learning.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8052_v697_learning_benefit_audit.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8053_v697_guarded_transaction_cost.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8054_v697_arc_supervisor_delta.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8055_v697_hardware_guard_boundary.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8056_v697_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims that the V697 capstone execution is blocked (`honest_verdict`: "complete_blocked_v697_capstone") due to unmet upstream acceptance gates and prerequisite contract failures.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
