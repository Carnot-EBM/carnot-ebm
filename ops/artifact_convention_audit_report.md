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

## experiment_8248_v713_evidence_intervention_methods.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8249_v713_evidence_view_kernel.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8250_evidence_view_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked prior to execution because the upstream gate check on exp8249-evidence-view-kernel.view_kernel_ready_score failed (observed 0, expected 1).

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8257_v713_arc_outcome_frontier.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8258_v713_kv260_evidence_cost_boundary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run was blocked (`"honest_verdict": "complete_blocked_capture"`) due to missing upstream V713 artifacts, with no device speedup or learning benefit achieved, while numerical CPU fallback preserves actions across tested fixed-point fixtures.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8259_v713_polarfire_dispatch_qualification.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8260_v713_gatemate_physical_delta.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
GateMate hardware execution remains blocked because no authenticated physical change has been recorded since Experiment 8246.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8261_v713_capstone.json

**CANNOT_DETERMINE**

reviewer call failed
