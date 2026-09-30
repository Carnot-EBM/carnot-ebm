# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 7 |
| CANNOT_DETERMINE | 1 |

## experiment_7904_v686_training_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The training qualification run is disqualified due to validation gate failures on `e2e_016_fixture` and `e2e_016_replay`, with no comparative benefit claimed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7904_v686_training_qualification.json.validators.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7905_v686_intervention_qualification.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_7906_energy_fit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked prior to execution because the upstream artifact failed three pre-gate checks.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7908_qwen_sufficiency.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked at the conductor pre-gate because upstream experiment exp7905 failed qualification checks (`intervention_protocol_ready_score` was 0 instead of 1, and `verdict_class` was "disqualified").

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7911_v686_arc_supervisor_delta.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run was disqualified during terminal verification due to failed validation checks, with no causal improvement or new game solves claimed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7913_v686_hardware_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7914_v686_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims that capstone execution is blocked with zero readiness due to failed upstream dependencies and unmet gate checks.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
