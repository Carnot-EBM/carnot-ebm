# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 5 |
| CANNOT_DETERMINE | 3 |

## experiment_8008_v694_conditioned_energy_fit.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8009_development_decisions.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked at conductor pre-gate because upstream experiment 8008 failed required gate checks (`conditioned_fit_ready_score` was 0 vs expected 1, and `verdict_class` was 'blocked').

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8010_v694_source_intervention_protocol.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8011_v694_qwen_source_sensitivity.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8012_budgeted_online_updates.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8014_v694_arc_supervisor_delta.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8016_v694_hardware_update_boundary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run is blocked with no qualified update trajectory or device benefit due to failed upstream gating prerequisites.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8017_v694_capstone.json

**CANNOT_DETERMINE**

reviewer call failed
