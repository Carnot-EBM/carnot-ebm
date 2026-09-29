# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 8 |

## experiment_7879_v684_contract_methods.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The validation run is disqualified and fails the contract gate because the staged authority file `research-roadmap-next.yaml` is missing.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7880_v684_source_boundary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 7880 was disqualified because the required coverage check failed: it measured 22% against a 100% threshold.

## WHAT IS MISSING
nothing; `gate_check_summary` and `observed_child_commands` identify `coverage_report`, its failed status, exit code, measured coverage, and threshold.

## THE CHECK A READER CANNOT DO
none

## experiment_7881_v684_intervention_protocol.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run was disqualified because a required validation check failed (`coverage_report.passed` observed false vs. expected true).

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7882_energy_fit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7884_qwen_sufficiency.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run was blocked because four of six upstream gates failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7887_v684_arc_supervisor_delta.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7889_v684_hardware_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run claims no current hardware execution or measured hardware advantage, and reports disqualification due to required checks.

## WHAT IS MISSING
nothing; `board_rows` records the board statuses, `board_rows.GateMate.blocker` records `0xffffffff`, and `historical_required_failures` and `historical_failures` identify failed checks and observed results.

## THE CHECK A READER CANNOT DO
none

## experiment_7890_v684_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The capstone is disqualified with readiness score 0 because required validation and upstream gates failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
