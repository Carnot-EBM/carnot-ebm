# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 8 |

## experiment_7839_v681_intervention_protocol.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was disqualified because the required validation check `changed_coverage.passed` failed with an observed value of false against expected true.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7845_v681_arc_supervisor_delta.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The audit found no new supervisor outcomes and was disqualified after required validation failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7847_v681_hardware_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7848_v681_length_shortcut.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The length model had a 0.007424 mean Brier improvement over the prevalence baseline on 64 development families, but the run was disqualified by required validation failures.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7849_v681_independent_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Required V681 science was blocked because upstream readiness and validation checks failed and several science producers were missing.

## WHAT IS MISSING
nothing; `gate_check_summary` records the failed checks with `artifact_field`, `expected`, and `observed` values.

## THE CHECK A READER CANNOT DO
none

## experiment_7850_v681_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The milestone is blocked because required evidence and prerequisite checks did not qualify.

## WHAT IS MISSING
nothing; `gate_check_summary` and `preconditions_checked.failed_operands` name the failed checks and record their expected and observed values.

## THE CHECK A READER CANNOT DO
none

## experiment_7840_energy_fit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because two upstream gates failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7842_qwen_counter_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
