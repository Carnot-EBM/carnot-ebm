# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 4 |
| AGGREGATE_ONLY | 3 |
| CANNOT_DETERMINE | 1 |

## experiment_7321_v643_batch_measurement.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The batch measurement was capture-complete but failed the value gate, first because its accuracy lower bound versus direct generation was −0.296875 against a required −0.02.

## WHAT IS MISSING
The artifact is truncated mid-field, so the actual `"per_source_group_results"` and `"rows"` cannot be found; they appear only inside `"field_principles"`. The visible `"gate_check_summary"` does provide the failed check, expected value, and observed value.

## THE CHECK A READER CANNOT DO
Do the per-source-group or per-row results show a broad accuracy degradation, or is the failed aggregate gate driven by a small number of outliers?

## experiment_7322_v643_batch_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The completed audit claims batch promotion failed because the preregistered accuracy, coverage, and speedup lower-bound gates were not met.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7323_v643_addition_prototype.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The bounded addition fixture is ready under shared Boolean executor authority, while held-out efficacy was not executed.

## WHAT IS MISSING
nothing; comparative per-request evidence is present in `"rows"`, and check outcomes and observed values are recorded in `"acceptance_gate_results"` and `"gate_check_summary"`.

## THE CHECK A READER CANNOT DO
none

## experiment_7324_v643_addition_learning.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Persistent structural acquisition reduced primary oracle calls versus reset and exact-cache controls while preserving utility and coverage, so the acceptance gates were met.

## WHAT IS MISSING
Per-stream arm-level values for `"primary_oracle_calls"`, `"mean_utility_fraction"`, and `"coverage_rate"` are missing; `"comparison_rows"` contains only aggregate `"estimate"`, `"ci95_lower"`, `"ci95_upper"`, and `"paired_stream_count"` values, while `"constraint_update_rows"` does not provide those comparative metrics.

## THE CHECK A READER CANNOT DO
Did the oracle-work advantage occur broadly across the 24 paired streams, or was the pooled ratio driven by a few outlier or degenerate streams?

## experiment_7325_v643_addition_audit.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Persistent structural acquisition used fewer total queries than reset and exact-cache controls without reducing utility, coverage, or feasibility.

## WHAT IS MISSING
Per-stream or per-request arm-level metric rows for `total_query_attempts`, `mean_utility_fraction`, `feasibility_coverage`, and feasibility outcomes; `"independent_comparison_rows"` contains only pooled estimates and confidence intervals, while the promised top-level `"rows"` field is absent.

## THE CHECK A READER CANNOT DO
Did query savings occur broadly across the 24 streams, or were the pooled ratios driven by a few outliers or degenerate control streams?

## experiment_7326_v643_constraint_kernel.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Rust achieved bit-exact parity with Python, but failed the predeclared 10× speedup gate at every tested batch size.

## WHAT IS MISSING
nothing; `"kernel_rows.cost"` provides 90 paired per-block timing rows, `"kernel_rows.parity"` provides per-fixture comparisons, and `"acceptance_gate_results.speedup_lower_ci95"` records the failed check, expected threshold, observed values, and pass status.

## THE CHECK A READER CANNOT DO
none

## experiment_7327_v643_board_continuity.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because no qualifying operator-authored GateMate physical-state change was recorded after Exp6559, while prior KV260 fabric and PolarFire CPU evidence remained preserved.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7328_v643_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The capstone claims structural addition reduced query work, Rust achieved bit-exact parity but missed the 10× cost gate, batch value gates failed, and overall readiness remains blocked by a missing GateMate physical-state receipt.

## WHAT IS MISSING
Per-unit metric rows for the comparative claims—especially each of the 24 paired structural-learning streams, each batch comparison unit, and each of the 90 paired Rust cost rows; `"claim_matrix"`, `"paired_stream_count"`, `"paired_cost_rows"`, `"paired_speedup_intervals"`, and aggregate `"metrics"` are present, while the GateMate blocker is fully diagnosed in `"gate_check_summary"`.

## THE CHECK A READER CANNOT DO
Was the reported structural-learning query reduction broad across the 24 paired streams, or driven by a few outliers or units with unequal headroom?
