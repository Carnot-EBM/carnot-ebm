# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 8 |

## experiment_7566_v661_energy_fit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The energy fitting task completed and produced a ready frozen bundle, while predictive benefit remained unmeasured and no positive empirical claim was made.

## WHAT IS MISSING
nothing; `"positive_claim": false`, `"predictive_benefit_measured": false`, and `"gate_check_summary"` records `"passed": true` with no failed checks.

## THE CHECK A READER CANNOT DO
none

## experiment_7567_v661_source_evaluation.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The source-contrast candidate showed no supported probability or decision benefit, and fresh confirmatory claims were disallowed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7568_continuous_recalibration.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked at the conductor pre-gate because upstream gate checks failed on exp7561-recalibration-prototype.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7569_v661_decision_learning_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims an honest null for source evaluation with zero qualified benefit, alongside a learning branch blocked by an upstream recalibration gate failure.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7570_v661_arc_live_lineage.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7571_v661_portable_calibration.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_10009_b2_induction_gate_measurement_v3.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims that code-only completion lengths were not uniformly capped, although 2 of 42 completed responses reached the 4,096-token limit.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7572_v661_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims the milestone evidence is disqualified with zero empirical benefit due to upstream gate and validation failures across recalibration and ARC lineage tasks.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
