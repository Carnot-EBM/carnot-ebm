# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 1 |
| AGGREGATE_ONLY | 1 |
| CANNOT_DETERMINE | 6 |

## experiment_8120_v702_arc_supervisor_delta.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8121_v702_hardware_batch_boundary.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8122_v702_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment is blocked due to failed precondition gates (verdict: "complete_blocked_design_exact_task_contract") and asserts no comparative scientific gain for hypotheses H1 or H2.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8123_v703_contract_custody.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8132_v703_service_cost.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8133_v703_arc_reader_frontier.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8134_v703_hardware_service_boundary.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The artifact claims that exact decision parity is satisfied across implementations and that a 100x speedup is ruled out by Amdahl outer bound ceilings across all tested arms and conditions.

## WHAT IS MISSING
Per-unit execution, parity, and timing rows for the 150 units per condition (`"units": 150`) or 11,070 completed runs (`"completed_count": 11070`). Present fields provide only aggregate totals in `"amdahl_bounds"` (`"retained_ns"`, `"total_ns"`, `"outer_ceiling"`), boolean gate summaries in `"acceptance_gates"` (`"exact_decision_parity": true`, `"host_component_rows": true`), and high-level board metadata in `"board_rows"`.

## THE CHECK A READER CANNOT DO
A reader cannot verify whether exact decision parity held on every individual center/seed unit or whether the pooled Amdahl retained and total times were driven by outlier transactions versus a consistent profile across the 150 units.

## experiment_8135_v703_capstone.json

**CANNOT_DETERMINE**

reviewer call failed
