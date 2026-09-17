# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 8 |

## experiment_7347_v645_plan_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
All four model calls produced usable plans and met the plan-transport readiness gate.

## WHAT IS MISSING
nothing; per-call evidence appears in `"rows"` and `"raw_call_manifest.calls"`, while `"gate_check_summary"` records the expected and observed gate values.

## THE CHECK A READER CANNOT DO
none

## experiment_7348_v645_plan_capture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The result is blocked because `"terminal_validation"` failed: `"expected_value": true` but `"observed_value": false`.

## WHAT IS MISSING
nothing; `"gate_check_summary"` identifies `"failed_check": "terminal_validation"` and records the expected and observed values, while `"evaluator_rows"` provides per-call outcomes.

## THE CHECK A READER CANNOT DO
none

## experiment_7349_prospective_learning.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because 5 of 6 prerequisite gates failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7351_v645_acquisition_prototype.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The prototype is disqualified because required validation and the cost-value gate failed.

## WHAT IS MISSING
nothing; per-unit comparative data appear in `"rows"`, while `"acceptance_gate_results"`, `"gate_check_summary"`, and `"repository_health.current_observation"` identify the failed checks and observed values.

## THE CHECK A READER CANNOT DO
none

## experiment_7352_acquisition_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because 2 of 3 prerequisite gates failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7354_v645_arc_transfer.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment completed but was disqualified because the `causal_feedback_value` and `required_validation` gates failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7355_v645_board_state.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
GateMate work is blocked because no qualifying operator-authored physical-state change was recorded after Exp6559, while three board dispositions are complete and hardware readiness, value, and promotion remain zero.

## WHAT IS MISSING
nothing; `board_rows` provides per-board records, and `gate_check_summary.failures` plus `gate_check_summary.first_failure` identify the failed check, field, expected value, and observed value.

## THE CHECK A READER CANNOT DO
none

## experiment_7356_v645_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
All fourteen task dispositions are represented, but required V645 science is unavailable and promotion remains blocked.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
