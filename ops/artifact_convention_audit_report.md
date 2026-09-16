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

## experiment_7331_learning_adapter.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because both upstream gates failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7332_plan_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because both prerequisite gates failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7336_v644_arc_resume.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was disqualified because affected scoped validation failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7337_arc_transfer.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because both prerequisite gates failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7339_v644_native_binding.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The imported in-process Rust binding preserves Python schedule semantics exactly and completed the fixed readiness protocol, with no speed or learning-value claim.

## WHAT IS MISSING
nothing; `"parity_rows"` supplies per-fixture Python and Rust outputs plus `"matched"`, while `"acceptance_gate_results"` and `"gate_check_summary"` record each gate’s observed value and pass status.

## THE CHECK A READER CANNOT DO
none

## experiment_7340_v644_native_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
All 90 paired benchmark blocks completed with exact parity, but the native 10× speed gate failed at every tested batch size.

## WHAT IS MISSING
nothing; `"rows"` records per-unit measurements, while `"acceptance_gate_results.native_ten_x"` records the threshold, observed CI lower bounds, and failed status.

## THE CHECK A READER CANNOT DO
none

## experiment_7341_v644_board_continuity.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
GateMate promotion is blocked because no qualifying operator-authored physical-state-change receipt exists after Exp6559, while all three board dispositions are complete and hardware readiness remains zero.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7342_v644_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The capstone claims required science is blocked and that the native in-process boundary failed the 10× throughput gate.

## WHAT IS MISSING
Per-block benchmark rows for each request size and comparison arm are missing; only aggregate `"throughput_ratio_intervals"`, `"paired_blocks": 90`, and `"rows": 270` counts are present, while `"contract_rows"` contains contract checks rather than measured benchmark units.

## THE CHECK A READER CANNOT DO
A reader cannot recompute the throughput ratios and confidence bounds or determine whether the failed 10× gate reflects a broad per-block effect or a few anomalous measurements.
