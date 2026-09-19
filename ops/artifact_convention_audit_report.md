# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 5 |
| AGGREGATE_ONLY | 2 |
| CANNOT_DETERMINE | 1 |

## experiment_7413_v650_source_calibration.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The completed experiment found no registered decision benefit because the scientific-benefit gate failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7414_v650_selected_feedback.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The registered online-value gate was not met because the simultaneous Brier, log-loss, coverage, and action-risk requirement failed.

## WHAT IS MISSING
nothing; `"acceptance_gate_results"` identifies the failed check and observed value, while `"condition_reports"` and `"feedback_event_rows"` provide condition-level and per-observation metrics.

## THE CHECK A READER CANNOT DO
none

## experiment_7415_v650_decision_audit.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The static and online branches are complete nulls with no jointly registered scientific value.

## WHAT IS MISSING
The supplied JSON is truncated inside `"phase_spans"`; although `"field_principles"` describes a `"rows"` field, the actual per-unit `"rows"` data are not visible, so per-seed/cell metrics underlying `"static_reduction"`, `"online_reduction"`, and `"acceptance_gate_results"` cannot be confirmed present or absent.

## THE CHECK A READER CANNOT DO
Do the null results hold across individual seeds and conditions, or are they artifacts of pooled aggregates hiding outliers, degenerate controls, or floor/ceiling cases?

## experiment_7416_v650_anchored_extraction.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the `one_owned_rtx3090_slot` precondition failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7417_extraction_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because 2 of 3 upstream gates failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7418_v650_revision_memory.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Both persistent arms passed the paid-query reduction gate but failed the total-cost benefit gate.

## WHAT IS MISSING
Per-unit rows by `"arm"`, `"stream_id"`, `"condition"`, and `"request_id"` containing the paid-query and total-cost measurements used to calculate the four aggregate CI bounds in `"acceptance_gate_results"`; the present `"erasure_witness_rows"` establish individual erasure witnesses but not those benefit ratios.

## THE CHECK A READER CANNOT DO
Were the paid-query reductions and excessive total costs broad across units, or driven by a small number of outlier streams or requests?

## experiment_7419_v650_precision_placement.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The int8 arm achieved no registered full-service speed benefit because `"int8_total_time_ratio_upper_paired_95"` was 1.0377982860931962, failing the `< 1.0` gate.

## WHAT IS MISSING
Per-unit `"timing_rows"` containing each paired block’s arm, batch size, and full-service timing; only the aggregate gate value appears in `"acceptance_gate_results"`, while `"field_principles"` merely describes `"timing_rows"` and `"timing_summary"`.

## THE CHECK A READER CANNOT DO
Was the failed speed gate broad across the paired timing blocks, or driven by a few extreme measurements?

## experiment_7420_v650_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The capstone completed all twelve dispositions but remained blocked because required extraction science was unavailable.

## WHAT IS MISSING
nothing; `"gate_check_summary.failures"` records the failed checks, expected and observed values, fields, paths, and upstream tasks.

## THE CHECK A READER CANNOT DO
none
