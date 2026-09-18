# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 7 |
| AGGREGATE_ONLY | 1 |

## experiment_7371_v647_proof_boundary.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The synthetic comparative-efficacy gate passed across the five experimental arms.

## WHAT IS MISSING
Per-unit, per-arm metric rows for each formula stream, family, seed, or condition are missing; `"acceptance_gate_results"` records `"synthetic_comparative_efficacy"` as `"observed": true` and `"passed": true`, and asserts `"complete_five_arm_rows"` passed, but the displayed `"erasure_witness_rows"` contain no five-arm comparative metrics.

## THE CHECK A READER CANNOT DO
Did the claimed comparative advantage occur broadly across formula streams, or was the aggregate gate driven by a few outliers or degenerate controls?

## experiment_7372_v647_qwen_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The completed canary was disqualified because internal artifact validation failed, specifically a `qwen_assignment_transport_ready_score_mismatch`.

## WHAT IS MISSING
nothing; `raw_call_rows` provides per-call outcomes, while `gate_check_summary`, `acceptance_gate_results`, and `internal_validation_errors` identify the failed checks and observed values.

## THE CHECK A READER CANNOT DO
none

## experiment_7373_proposal_capture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because 3 of 6 prerequisite gates failed, beginning with `qwen_assignment_transport_ready_score` being 0 when 1 was required.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7376_v647_arc_outcomes.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run completed but was disqualified because six required gates failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7377_v647_ising_law.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
All terminal acceptance gates passed, including preservation of the source law under proof assistance and detection of intentional law changes.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7378_v647_ising_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run failed the terminal validation gate (`terminal_validation_passed`) and was not automatically promoted.

## WHAT IS MISSING
nothing; `acceptance_gate_results` identifies the failed check with `observed: false`, and `cell_results` provides per-cell evidence.

## THE CHECK A READER CANNOT DO
none

## experiment_7379_v647_hardware_envelope.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because no qualifying operator-authored GateMate physical-state change was recorded after Exp6559, while unavailable placement inputs support no speed claim.

## WHAT IS MISSING
nothing; `gate_check_summary.failures`, `first_failure`, `honest_verdict`, `board_rows.error`, and `placement_envelope_rows.error` identify the failed checks and observed values.

## THE CHECK A READER CANNOT DO
none

## experiment_7380_v647_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
All twelve V647 dispositions were accounted for, but the capstone was disqualified because required producer validation failed and the proof-memory measurement and audit were pre-gated.

## WHAT IS MISSING
nothing; `"gate_check_summary"`, `"acceptance_gate_results"`, and per-task `"disposition_rows"` record the failed checks, expected values, observed values, and affected upstream tasks.

## THE CHECK A READER CANNOT DO
none
