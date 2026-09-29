# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 8 |

## experiment_7867_v683_natural_runtime.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7868_v683_intervention_protocol.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7874_v683_arc_supervisor_delta.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7876_v683_hardware_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
This is receipt-only historical accounting: it reports zero current device executions, no measured hardware speedup, and a disqualified verdict.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7877_v683_independent_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The audit was completed but disqualified because required v683 validation and upstream readiness checks failed.

## WHAT IS MISSING
nothing; `gate_check_summary` records the failed checks with their `expected` and `observed` values.

## THE CHECK A READER CANNOT DO
none

## experiment_7878_v683_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims that the v683 milestone capstone is disqualified from execution readiness (`complete_disqualified_required_v683_validation`) due to failed prerequisite experiment gate checks and validation command timeouts.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7869_energy_fit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7871_qwen_sufficiency.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked at the conductor pre-gate because 5 of 6 upstream gate checks failed, beginning with exp7866-source-boundary.source_boundary_ready_score observing 0 instead of 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
