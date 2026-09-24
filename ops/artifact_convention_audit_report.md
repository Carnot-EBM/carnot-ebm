# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 8 |

## experiment_7605_fit_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked at the conductor pre-gate because upstream task `exp7604-evidence-pilot` failed the gate check with an `evidence_transport_ready_score` of 0 against an expected value of 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7606_test_online_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
No claim; execution was blocked at conductor_pre_gate because upstream gate checks failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7610_v664_evidence_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The audit is blocked because required upstream evidence producers in the v664 evidence chain are missing or ineligible, with no comparative claims made.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7611_v664_arc_matched_support.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Protocol fixture readiness is established, but empirical benefit was not established (an honest null due to zero selected matched keys).

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7612_v664_arc_history_measurement.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked (`verdict_class`: "blocked") due to a failed upstream precondition check (`exp7611_protocol`) on the matched protocol path from experiment 7611.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7613_v664_service_attribution.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The empirical accelerator-benefit gate failed, preserving the conclusion `complete_null_rust_consumer_ready_speed_gate_failed`.

## WHAT IS MISSING
nothing; `acceptance_gate_results` records the failed check with `expected`, `observed`, and `passed`, while `consumer_stage_rows` provides per-unit measurements keyed by `pair_id`, `seed`, `unit_id`, and `arm`; the GateMate block also records `error`, `exact_missing_receipt`, and `last_diagnostic`.

## THE CHECK A READER CANNOT DO
none

## experiment_7614_v664_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment makes no positive comparative claim (`positive_claim`: false) and is blocked by missing and failing upstream prerequisites (`honest_verdict`: "complete_blocked_required_v664_external_evidence").

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_10012_gate_usefulness.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment completed gate-usefulness measurement with 3 informative windows and 0 unrecoverable cases.

## WHAT IS MISSING
nothing; the aggregate `"gate_tables_by_arm"` are backed by `"per_pair_rows"` containing per-pair `"arm_status"`, reasons, and execution diagnostics.

## THE CHECK A READER CANNOT DO
none
