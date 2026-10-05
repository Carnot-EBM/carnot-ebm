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

## experiment_8168_sentence_energy_fit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked at the pre-gate layer because upstream dependency `exp8167-fit-sentence-capture` failed the requirement `fit_trainable_score == 1` with an observed value of 0.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8171_v706_released_feedback_learning.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8172_v706_learning_benefit_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Hypothesis H2 (error-center versus fixed-public-center) failed to demonstrate an advantage over the control arm, reporting an honest null with `h2_passed: false` and zero gain.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8173_v706_service_validation.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8174_v706_complete_request_cost.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8175_v706_arc_supervisor_delta.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8176_v706_hardware_workload_boundary.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8177_v706_capstone.json

**CANNOT_DETERMINE**

reviewer call failed
