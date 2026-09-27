# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 8 |

## experiment_7731_development_decisions.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7732_v673_causal_admission.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
One growth proposal was accepted with a mean Brier reduction of 0.35 across eight frozen units; the overall validity gate failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7733_continuous_set_learning.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run was blocked because four of six upstream gates failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7734_v673_independent_evidence_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run is blocked from activation (`complete_blocked_required_v673_evidence`) because required upstream dependencies Exp7731 and Exp7733 failed eligibility checks.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7735_v673_arc_organic_visits.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7736_arc_organic_measurement.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked prior to execution because upstream experiment exp7735-arc-organic-visits failed required pre-gate checks.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7737_set_service_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked at conductor pre-gate because upstream experiment `exp7730-set-energy-fit` failed required gate checks.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7738_v673_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The v673 capstone run was blocked because required upstream evidence artifacts failed eligibility gate checks.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
