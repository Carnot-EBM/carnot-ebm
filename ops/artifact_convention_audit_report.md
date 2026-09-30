# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 8 |

## experiment_7894_v685_energy_fit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run completed but was disqualified by required-check failures, with `energy_fit_ready_score` of 0.

## WHAT IS MISSING
nothing; `historical_required_failures` names failed checks and records their exit codes, and `rows` contains per-unit metrics.

## THE CHECK A READER CANNOT DO
none

## experiment_7895_decision_abstention.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked at the conductor pre-gate because upstream dependency exp7894-energy-fit failed readiness and verdict gate checks.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7896_qwen_sufficiency.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked at the conductor pre-gate because upstream dependency `exp7893-intervention-protocol` failed gate checks (`intervention_protocol_ready_score` was 0 instead of 1, and `verdict_class` was `disqualified`).

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7897_causal_acquisition.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked at conductor pre-gate because upstream dependency exp7894 failed its readiness and verdict gates.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7899_v685_arc_supervisor_delta.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7900_service_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked at the conductor pre-gate because upstream experiment exp7894-energy-fit failed its readiness gate checks.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7901_v685_hardware_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run was disqualified because required checks failed; it claims no measured hardware benefit.

## WHAT IS MISSING
nothing — `observed_child_commands` identifies the failed checks and records their `actual_exit` and `expected_exit` values, even though `gate_check_summary` is empty.

## THE CHECK A READER CANNOT DO
none

## experiment_7902_v685_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The capstone is disqualified because required upstream evidence and validation checks failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
