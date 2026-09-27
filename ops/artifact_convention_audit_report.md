# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 8 |

## experiment_7770_v676_qwen_runner_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The CPU fixture completed its protocol, but the result was disqualified for readiness and claims no semantic benefit.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7771_view_energy_fit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked at the conductor pre-gate because 6 of 9 upstream qualification gate checks failed, starting with `sentence_protocol_ready_score` in upstream `exp7768-source-view-qualification` observing 0 instead of 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7773_qwen_event_confidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked before comparison because two upstream gates failed.

## WHAT IS MISSING
nothing; `gates_evaluated` records each check, its expected and actual values, and whether it passed.

## THE CHECK A READER CANNOT DO
none

## experiment_7775_v676_independent_evidence_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The audit was blocked because the required Exp7772 and Exp7774 producer artifacts were missing.

## WHAT IS MISSING
nothing; `gate_check_summary` names both failed `producer_path` checks and records `observed: "missing"`.

## THE CHECK A READER CANNOT DO
none

## experiment_7776_v676_arc_runner_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The runner qualification was disqualified because required validation checks failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7777_arc_organic_measurement.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7779_v676_hardware_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7780_v676_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The v676 capstone is disqualified, with readiness 0 and validity false.

## WHAT IS MISSING
nothing; `gate_check_summary` records the failed checks, expected values, and observed values, while `rows` identifies the affected producers.

## THE CHECK A READER CANNOT DO
none
