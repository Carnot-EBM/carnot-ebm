# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 8 |

## experiment_7853_v682_natural_runtime.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked before natural measurement because upstream precondition checks on experiment 7852 failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7854_v682_intervention_protocol.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact records a scripted fixture run with failed validity and readiness gates and makes no measured sufficiency claim.

## WHAT IS MISSING
nothing; `acceptance_gate_results` gives the gate outcomes, and `fixture_request_rows` records per-fixture, per-arm results and failure statuses.

## THE CHECK A READER CANNOT DO
none

## experiment_7855_energy_fit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked at the conductor pre-gate because four of six upstream prerequisite gate checks failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7857_qwen_sufficiency.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked at the conductor pre-gate because five of six upstream dependency gate checks failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7860_v682_arc_supervisor_delta.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
No new supervisor outcomes, causal action savings, or level solves were observed across the candidate sources (honest null).

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7862_v682_hardware_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact reports a null result: it accounts for historical board evidence but records no new device execution or measured hardware advantage.

## WHAT IS MISSING
nothing; `rows` and `board_rows` give board-level records, and GateMate’s `blocker` records the observed value `0xffffffff`.

## THE CHECK A READER CANNOT DO
none

## experiment_7863_v682_independent_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The milestone 2026.09.682 validation audit is blocked from execution because required upstream science producer artifacts failed prerequisite checks or are missing.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7864_v682_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The v682 capstone task is blocked on required upstream science producer experiments.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
