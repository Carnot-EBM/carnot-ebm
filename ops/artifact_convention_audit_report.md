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

## experiment_3377_archive_v310_activate_v311.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The archive is complete and milestone 2026.05.311 is ready for activation.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_3392_archive_v311_activate_v312.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The archive of milestone 2026.05.311 is complete, and milestone 2026.05.312 is activated and ready.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_833_constraint_delta_root_cause.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The root cause is a missing write path: verification found a violation, but `"n_store_write_calls"` remained 0 despite 10 retrieval calls.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6582_gemma4_31b_flagship_source_shard.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The four-unit Gemma4-31B source shard completed successfully and met the readiness gate with `"gemma4_31b_family_source_shard_ready_score": 1.0`.

## WHAT IS MISSING
The per-unit `"rows"` containing each unit’s raw output, runtime, failure status, token counts, timing, and cost metrics are missing; only aggregate counts in `"aggregate_row_recomputation"`, hashes in `"checkpoint_receipts"`, and limited `"parser_diagnostic_rows"` are present.

## THE CHECK A READER CANNOT DO
Did every one of the four source units independently satisfy all readiness criteria, or does the aggregate conceal a degenerate or anomalous unit?

## experiment_6583_gemma4_26b_a4b_flagship_source_shard.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The Gemma4-26B-A4B source shard completed and met the readiness gate with `"gemma4_26b_a4b_family_source_shard_ready_score": 1.0`.

## WHAT IS MISSING
The per-unit `"rows"` containing each source unit’s readiness checks and metric values are missing; only aggregate counts in `"aggregate_row_recomputation"` and non-metric `"checkpoint_receipts"` and `"parser_diagnostic_rows"` are present.

## THE CHECK A READER CANNOT DO
Did every one of the four source units independently satisfy all readiness conditions, or did the aggregate score conceal a degenerate or exceptional unit?

## experiment_6585_v573_terminal_recovery_and_execution_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V572 terminal states and three Exp6584 hard-limit attempts were successfully replayed, making the V573 execution contract ready without creating a scientific claim.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6586_isolated_full_suite_truth_baseline.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked because the `pytest_receipt` check failed in the isolated environment.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6587_v573_constraint_first_method_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The source, fixture, router, metric, and exact-authority contracts are complete and ready before any V573 model outcomes exist.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
