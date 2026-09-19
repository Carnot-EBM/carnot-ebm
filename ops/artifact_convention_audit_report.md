# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 4 |
| AGGREGATE_ONLY | 4 |

## experiment_7426_v651_static_decisions.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The decision-value gate failed solely because certified coverage was 0.0 rather than at least 0.25, while the comparative Brier-improvement and non-worse-log-loss checks passed.

## WHAT IS MISSING
Per-unit metric rows keyed by `arm`, `seed`, and `condition`, containing at least Brier score, log loss, and coverage; `"detailed_rows_present"` and `"detail_row_directory"` assert those rows exist elsewhere, but the artifact itself provides only pooled/domain aggregates and checkpoint metadata.

## THE CHECK A READER CANNOT DO
Did every seed and condition show Brier improvement over the controls, or was the reported `"simultaneous_brier_improvement": true` driven by a small number of unusually favorable units?

## experiment_7427_v651_randomized_feedback.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The experiment completed validly but found no registered online value because the scientific-benefit gate failed.

## WHAT IS MISSING
Per-unit arm/control metric rows and paired Brier-delta/interval rows are missing; `"condition_reports"` contains only condition-level aggregates, while `"feedback_event_rows"` and `"checkpoint_lineage"` provide only shard manifests, row counts, and hashes—not row contents.

## THE CHECK A READER CANNOT DO
Did `"upper_brier_deltas_below_zero": false` reflect broadly non-improving results across units, or a small number of outliers or degenerate control units?

## experiment_7428_v651_decision_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
All listed acceptance gates passed, including branch validity, required validation, attack rejection, and disabled promotion.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7429_v651_anchored_capture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The capture was gated because it produced 0 of 3 required usable development outputs, had an extraction-value score of 0 instead of 1, and failed terminal-reader validation.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7430_extraction_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because all three upstream gates failed, beginning with `extraction_capture_complete_score` being 0 instead of the required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7431_v651_arc_live_sentinel.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The live ARC sentinel reached the model and completed both scheduled units, while making no comparative efficacy claim.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7432_v651_update_placement.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The sparse fixed-point update showed no registered complete-service speed benefit because the `paired_whole_service_speed` gate failed at every reported batch size.

## WHAT IS MISSING
Per-unit paired timing rows underlying `timing_summary.ci95_upper`; `gate_check_summary` identifies the failed check and observed values, and `rows` provides event-level quality data, but no `timing_rows` are present.

## THE CHECK A READER CANNOT DO
Were the unfavorable timing results broad across paired repetitions, or driven by a few outliers or degenerate control measurements?

## experiment_7433_v651_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The artifact claims the capstone is complete but disqualified, while reporting comparative speed-gate and benefit verdicts across experimental arms.

## WHAT IS MISSING
Per-paired-block metric rows underlying `"complete_service_costs"`—specifically each block’s arm and baseline whole-service timings or ratio—are missing; only `"whole_service_time_ratio"`, `"ci95_lower"`, `"ci95_upper"`, `"paired_blocks": 30`, and `"speed_gate_passed"` are present. `"raw_rows_available": true` and external `"observed_path"` references do not include those rows in this artifact.

## THE CHECK A READER CANNOT DO
Were the reported speed-gate failures consistent across the 30 paired blocks, or driven by one or a few extreme blocks?
