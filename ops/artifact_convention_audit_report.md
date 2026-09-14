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

## experiment_7294_v641_reuse_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The reuse audit completed but promotion was disqualified because reuse failed the full-cost speedup and positive-gain gates despite passing accuracy, coverage, parity, and integrity checks.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7295_v641_mixture_prototype.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The bounded fixed-share fixture passed its mechanics and acceptance gates; prospective efficacy is explicitly not claimed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7296_v641_mixture_learning.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The fixed-share mixture passed the shuffled-label and safety controls but failed several efficacy and recurrence-harm acceptance gates.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7297_v641_mixture_audit.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The artifact claims several comparative acceptance gates passed—including true feedback beating shuffled controls—while other comparisons failed their gates.

## WHAT IS MISSING
Per-unit comparative metric rows for each stream, seed, cell, or condition, including both arms’ metric values; `"acceptance_gate_results"` contains only aggregate `"observed"` bounds and `"pass"`/`"passed"` verdicts, while `"causal_change_rows"` records change events but no arm-level outcome metrics.

## THE CHECK A READER CANNOT DO
Were the reported improvements broad across evaluation units, or driven by a few outliers or degenerate control rows?

## experiment_7298_v641_snapshot_journal.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The persistent full-snapshot acknowledgment protocol passed, including parity across three native-controller arms and seeded crash recovery.

## WHAT IS MISSING
The artifact is truncated mid-record, so the complete `"rows"` field—promised by `"field_principles.rows"`—cannot be found or confirmed; `"native_controller_parity"` contains only aggregate values (`"arms": 3`, `"failures": 0`), although per-unit `"crash_control_rows"` are present for crash recovery.

## THE CHECK A READER CANNOT DO
Did each of the three controller arms match every endpoint and event order individually, or is that claim supported only by the aggregate zero-failure count?

## experiment_7299_v641_snapshot_cost.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Persistent SQLite full snapshots failed the frozen V640 deployment bounds, specifically because the fixed group-16 burst-throughput gate and the 10× full-boundary target were not met.

## WHAT IS MISSING
Per-seed, per-arm throughput and latency measurements—such as the described but absent `"per_run_results"` or `"rows"` fields. `"independent_raw_reducer"` contains only aggregate means, confidence intervals, and row counts, while `"independent_recovery_rows"` contains recovery metrics rather than the performance values underlying the comparative gates.

## THE CHECK A READER CANNOT DO
Were the eight paired-seed group-16 throughput results broadly below the 1.5× gate, or was the failed lower confidence bound driven by one outlier seed?

## experiment_7300_v641_board_continuity.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Three per-board dispositions are complete: KV260 preserved FPGA-fabric execution, PolarFire preserved CPU dispatch rather than FPGA sampling, and GateMate is blocked because no qualifying post-Exp6559 physical-state-change receipt exists.

## WHAT IS MISSING
nothing; per-unit evidence appears in `"rows"` and `"board_rows"`, while the GateMate failure is identified in `"gate_check_summary.board_blocks"` with `"field"`, `"expected_value"`, and `"observed_value"` and is repeated in the GateMate row’s `"failed_value"` and `"error"`.

## THE CHECK A READER CANNOT DO
none

## experiment_7301_v641_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The capstone is blocked by quarantined ARC evidence, while the completed reuse, mixture, and storage branches failed their scientific-value gates.

## WHAT IS MISSING
The top-level scientific `"rows"` containing each unit’s arm, seed, metric, cost, error, abstention, and censoring are missing; `"contract_rows"` cover only Markdown/YAML agreement, while `"audit_score_rows"` and `"gate_check_summary"` record aggregate gate outcomes. The blocker diagnostic itself is present in `"gate_check_summary"`.

## THE CHECK A READER CANNOT DO
Were the failed reuse, mixture, and storage comparisons broad across units, or driven by outliers, degenerate controls, or units with no headroom?
