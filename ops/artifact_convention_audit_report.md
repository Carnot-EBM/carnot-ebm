# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 8 |

## experiment_7385_v648_decision_training.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The calibration value-reduction gate failed, with a score of 0, because `brier_ci_below_both_controls` was false while the other efficacy checks passed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7386_v648_online_decisions.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The acceptance gate was not met because `required_validation`, `independent_safety_readers`, and `online_learning_value` each observed `false` against an expected `true`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7387_decision_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because 3 of 6 upstream gates failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7388_proposal_capture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because 4 of 6 prerequisite gates failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7391_arc_generalization.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because all three prerequisite gates failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7392_v648_ising_reduction.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The reduction was disqualified because `capability_e2e_passed` was false, while exact-law preservation and quality gates for nonempty archived cells passed but the original all-cell gate failed.

## WHAT IS MISSING
nothing; `"rows"` contains per-unit formula/beta/condition outcomes, and `"gate_check_summary"` identifies the blocking failure with `"check": "capability_e2e_passed"`, `"expected": true`, and `"observed": false`.

## THE CHECK A READER CANNOT DO
none

## experiment_7393_v648_hardware_placement.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task is blocked because no qualifying post-Exp6559 GateMate physical-state change was recorded, and incomplete placement-stage measurements support no hardware-speed claim.

## WHAT IS MISSING
nothing; `"gate_check_summary.failures"`, `"honest_verdict"`, `"board_rows[].error"`, `"placement_envelope_rows[].failed_field"`, and `"observed_value"` identify the failed checks and observed values.

## THE CHECK A READER CANNOT DO
none

## experiment_7394_v648_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
All fourteen V648 tasks were accounted for, but required science and validation gates failed, so promotion, deployment, and publication were not authorized.

## WHAT IS MISSING
nothing; `"disposition_rows"` provides per-task outcomes and `"gate_check_summary.failures"` records the failed checks with expected and observed values.

## THE CHECK A READER CANNOT DO
none
