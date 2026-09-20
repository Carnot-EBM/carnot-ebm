# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 6 |
| AGGREGATE_ONLY | 2 |

## experiment_7439_v652_certified_decisions.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The run completed validly but showed no registered decision benefit, despite aggregate noninferiority to controls.

## WHAT IS MISSING
The actual per-unit `"rows"` or `"probability_row_shards"` containing each source group’s arm, seed, condition, prediction, label, coverage, utility, Brier score, and log loss are missing; only aggregate `"independent_reduction.metrics"`, `"paired_coverage_intervals"`, and certificate counts are present.

## THE CHECK A READER CANNOT DO
Were the reported arm comparisons and zero coverage deltas consistent across the 450 paired source groups, or caused by degenerate controls or a small number of influential units?

## experiment_7440_v652_mixture_learning.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment completed but found insufficient online benefit because multiple scientific-benefit, safety, coverage, and negative-control gates failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7441_v652_decision_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The online-learning branch is disqualified because prediction-time expert evidence is missing and weight-update replay is incomplete, while the static evidence supports only a null, non-deployment conclusion.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7442_v652_span_capture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The span-capture experiment was blocked with a null result because runtime validation failed and the development gate did not open.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7443_v652_span_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The audit completed with a null, development-only verdict because producer runtime integrity failed and no sealed evaluation was attempted.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7444_v652_arc_supervisor_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The two archived episodes produced no supervisor firings, so they provide no arm-effect evidence and justify no policy change.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7445_v652_hardware_envelope.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The hardware envelope is a null result: the measured 0.37696 persistence fraction limits idealized acceleration to 2.65×, so the 100× condition fails and requires persistence-orchestration redesign.

## WHAT IS MISSING
Per-block timing rows for the 30 paired blocks underlying each comparative service condition and per-unit measurements underlying `"observed_unaccelerated_fraction"`; only `"whole_service_time_ratio"`, confidence intervals, `"paired_blocks": 30`, and aggregate Amdahl values are present. The GateMate blocker is adequately diagnosed in `"gate_check_summary"`.

## THE CHECK A READER CANNOT DO
Were the null service comparisons and 0.37696 persistence fraction broad across the paired measurements, or driven by a few outlier or degenerate blocks?

## experiment_7446_v652_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The capstone completed all 13 task dispositions but disqualified the required V652 science because three upstream scientific-validity checks observed `"valid": false`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
