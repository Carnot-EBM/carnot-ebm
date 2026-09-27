# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 6 |
| BLOCKED_WITHOUT_DIAGNOSTIC | 1 |
| CANNOT_DETERMINE | 1 |

## experiment_7757_view_energy_fit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked at the conductor pre-gate because 6 of 9 upstream gate checks failed, starting with exp7754 sentence_protocol_ready_score returning 0 instead of 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7759_v675_qwen_evidence_views.json

**BLOCKED_WITHOUT_DIAGNOSTIC**

## VERDICT
BLOCKED_WITHOUT_DIAGNOSTIC

## WHAT THE CLAIM IS
The 72-row diagnostic completed but was disqualified because required checks did not pass.

## WHAT IS MISSING
The failed required check’s name and observed value. `honest_verdict` says `complete_disqualified_required_checks`, `acceptance_gate_results.readiness.measured_operands.required_checks_passed` is `false`, and `gate_check_summary` is empty.

## THE CHECK A READER CANNOT DO
Which required check caused the disqualification, and what value did it observe?

## experiment_7760_v675_online_runner.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The synthetic delayed-feedback fixture completed and met its readiness and validity gates; natural benefit is unmeasured.

## WHAT IS MISSING
The supplied artifact ends mid-entry in `"rows"`. It reports `"acceptance_gate_results"` and `"online_runtime_ready_score"`, but the remaining per-unit rows needed to check those results are not available.

## THE CHECK A READER CANNOT DO
Do the outcomes across all measured units support the reported readiness result?

## experiment_7762_v675_independent_evidence_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment is disqualified and blocked because required upstream producer artifacts for Exp7758 and Exp7761 are missing.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7763_v675_arc_runner_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run was disqualified because required runner validation failed.

## WHAT IS MISSING
nothing; `gate_check_summary` identifies `full_pytest` as failed, with `observed: 2` against `expected: 0`.

## THE CHECK A READER CANNOT DO
none

## experiment_7764_arc_organic_measurement.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked prior to execution because upstream prerequisite exp7763 failed its qualification gate checks (organic_runner_ready_score was 0 instead of 1, and verdict_class was disqualified).

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7765_v675_service_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7766_v675_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V675 capstone validation is disqualified with a completion score of 0 and failed acceptance gates due to upstream validation and producer check failures.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
