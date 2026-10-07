# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 2 |
| AGGREGATE_ONLY | 1 |
| CANNOT_DETERMINE | 5 |

## experiment_8212_v709_memory_benefit_audit.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8213_v709_prospective_request_recorder.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8214_v709_prospective_service_measurement.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Native atomic batch processing achieves no speedup over Python durable batch processing (paired ratio ~0.997, 95% CI [0.984, 1.011]), resulting in a complete null verdict that fails NFR-01.

## WHAT IS MISSING
Per-unit paired observation rows (such as `request_rows`, `rows`, or `stage_cost_rows`, all defined in `field_principles` but absent from the payload) showing individual timings and outcomes for each request across both arms. Present fields record only summary aggregates: `complete_workload_totals`, `paired_ratio_ci95`, `completed_count` (21), `failed_count` (3), `intended_count` (24), `all24_slot_wall_s`, and `amdahl_ceiling`.

## THE CHECK A READER CANNOT DO
A reader cannot determine whether the paired ratio estimate of ~0.997 reflects genuine unit-level parity across all 21 completed sources or whether individual request latencies diverged widely and were masked by aggregation, floor/ceiling effects, or the 3 failed requests.

## experiment_8215_v709_arc_authoritative_frontier.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8216_v709_hardware_workload_obligations.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8217_v709_capstone.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8218_v710_contract_replay_qualification.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_8219_v710_utility_patch_methods.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
