# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 1 |
| CANNOT_DETERMINE | 7 |

## experiment_8289_v715_capstone.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8290_v716_runtime_localization.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8291_v716_dependency_scoped_admission.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8292_evidence_view_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked at conductor pre-gate because upstream dependency `exp8290-runtime-localization` failed the `cuda_context_ready_score` gate check (observed 0, expected 1).

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8300_v716_arc_outcome_frontier.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8301_v716_kv260_evidence_cost_boundary.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8302_v716_gatemate_physical_delta.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8303_v716_capstone.json

**CANNOT_DETERMINE**

reviewer call failed
