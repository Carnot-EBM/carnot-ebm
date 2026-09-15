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

## experiment_7307_v642_batch_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The canary was blocked before model work because upstream experiment `exp7306-batch-fixture` had `verdict_class` equal to `disqualified`.

## WHAT IS MISSING
nothing; `gate_check_summary` records `"failed_check": "upstream_terminal_class"`, `"field": "verdict_class"`, `"observed_value": "disqualified"`, and `"expected_value": "not blocked or disqualified"`.

## THE CHECK A READER CANNOT DO
none

## experiment_7308_batch_measurement.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `exp7307-batch-canary.batch_canary_ready_score` was 0 instead of the required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7310_v642_factor_prototype.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The bounded factor-local revision fixture is ready, while prospective efficacy remains unmeasured.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7311_v642_factor_learning.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The factor-local retained-witness arm reduced future and non-feedback error versus both reset controls, although the false-accept and recurrence gates failed.

## WHAT IS MISSING
Per-stream treatment and control metric values keyed by `stream_id`, `seed`, and `stratum`; `"comparison_rows"` contains estimates, confidence intervals, and anonymous `"paired_differences"`, while `"feedback_update_rows"` does not contain the compared outcome metrics.

## THE CHECK A READER CANNOT DO
For each stream, were zero or favorable differences caused by genuine improvement, identical arms, or one arm already being pinned at a metric floor or ceiling?

## experiment_7312_v642_factor_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Retained factor witnesses improved some future-error metrics but failed the promotion contract because `recurrence_error_vs_frozen` and `false_accept_vs_local_reset` failed.

## WHAT IS MISSING
nothing; per-unit metrics are recorded in `rows` and `independent_stream_intervals.paired_differences`, while failed checks and observed values appear in `acceptance_gate_results` and `honest_verdict`.

## THE CHECK A READER CANNOT DO
none

## experiment_7313_v642_cost_envelope.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The measured and counterfactual evidence failed the 10× target, the warm group-16 1.5× gate, and the technique-warrant check, so further implementation was deferred.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7314_v642_board_continuity.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Three authenticated board dispositions and exact next conditions were recorded: KV260 FPGA-fabric graduation and PolarFire CPU dispatch remain preserved, while GateMate is blocked because no qualifying changed-physical-state receipt exists.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7315_v642_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The capstone is blocked because Exp7309 batch-audit evidence is absent, while ARC tool use and factor value are null and the remaining receipts do not establish scientific efficacy.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
