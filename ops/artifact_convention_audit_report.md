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

## experiment_7097_v623_contract_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V623 task contract conforms because all 12 task rows and eight gate clauses passed, producing `"v623_task_contract_conforms_score": 1`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7098_v623_sota_ingestion.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V623 SOTA ingestion completed, passed its coverage gate, and produced no new adoption delta.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7099_v623_adapter_withheld_preflight.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact is truncated before its headline verdict, so the claim cannot be identified.

## WHAT IS MISSING
The actual `"gate_check_summary"`, `"adapter_withheld_live_path_ready_score"`, `"verdict_class"`, and `"honest_verdict"` values are missing; only their descriptions appear in `"field_principles"`, while per-unit data are present in `"per_game_results"` and `"rows"`.

## THE CHECK A READER CANNOT DO
Did the final verdict report success, a null result, or a blocked gate—and, if blocked, which check failed at what observed value?

## experiment_7100_adapter_withheld_arc_loo_measurement.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `adapter_withheld_live_path_ready_score` was 0 when the gate required it to equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7105_v623_exact_constraint_stream.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7106_v623_procedural_memory_csl.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The `delayed_procedural` arm outperformed all four comparison arms, with positive 95% confidence intervals.

## WHAT IS MISSING
Per-unit outcome rows containing each unit’s arm, capacity, and correctness/accuracy metric; `capacity_rows` and `confidence_interval_rows` contain only aggregates, while `decision_rows` records choices and pre-feedback `score` values, not outcome correctness.

## THE CHECK A READER CANNOT DO
Did `delayed_procedural` improve broadly across units, or were its pooled accuracy and confidence intervals driven by a small number of unusually favorable units?

## experiment_7107_v623_continual_memory_cold_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The continual-memory cold audit is ready, with the delayed-procedural arm outperforming the comparison arms at every recorded capacity.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7108_v623_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The delayed-commit procedural-memory mechanism produced positive comparative evidence and should advance to held-out replication.

## WHAT IS MISSING
The actual per-unit A/B metric rows for experiment 7106—presumably `"self_learning_rows"`—are missing; `"per_unit_presence_rows"` reports only `row_count: 720`, while `"headline_recomputation_rows"` provides only aggregate readiness scores.

## THE CHECK A READER CANNOT DO
Did the procedural-memory arm outperform its control broadly across the 720 units, or was the claimed benefit driven by a few outliers or degenerate controls?
