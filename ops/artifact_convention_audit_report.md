# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 8 |

## experiment_7798_view_energy_fit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked from executing because 5 of 7 upstream gate checks failed, starting with `sentence_protocol_ready_score` observing 0 instead of 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7800_v678_counter_evidence_protocol.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was disqualified because a required validation check failed.

## WHAT IS MISSING
nothing; `gate_check_summary` identifies `full_python_suite.exit_code` as the failed check, with `observed: -15` against `expected: 0`.

## THE CHECK A READER CANNOT DO
none

## experiment_7801_qwen_counter_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked at the conductor pre-gate because upstream checks on experiment 7800 failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7803_v678_arc_runner_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The runner qualification was disqualified because the required validation check `full_python_suite` failed with an observed return code of -15 against an expected value of 0.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7804_arc_organic_measurement.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7806_v678_hardware_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
No qualified local whole-service board benefit was measured, so acquisition is deferred and the result is disqualified by failed required checks.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7807_v678_independent_evidence_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims disqualification (`verdict_class: "disqualified"`) due to missing upstream science producers and failed required validation commands.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7808_v678_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V678 capstone is blocked due to missing and disqualified required upstream evidence.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
