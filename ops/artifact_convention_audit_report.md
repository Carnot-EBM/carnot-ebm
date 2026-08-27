# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 6 |
| AGGREGATE_ONLY | 1 |
| CANNOT_DETERMINE | 1 |

## experiment_6651_failure_localized_suffix_regeneration.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `regeneration_headroom_count` was 2, below the required minimum of 8.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6653_state_grounded_repair_memory_fixture.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
`aggregate_row_recomputation.all_checks_passed` claims that all listed checks passed across 48 events, six attacks, and six transitions.

## WHAT IS MISSING
The artifact is truncated inside `event_rows`; although `event_count` is 48, the complete 48 rows—and any remaining transition and rollback rows supporting `transition_count` and `rollback_receipt_count`—cannot be found.

## THE CHECK A READER CANNOT DO
Do all 48 event rows and all six transition and rollback receipts actually satisfy every check marked `true`?

## experiment_6654_prospective_repair_memory_evolution.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The verified-memory arm outperformed the context-only arm in pooled prequential exact yield (0.2037 versus 0.1481), including a +0.6667 future-event yield difference.

## WHAT IS MISSING
nothing; `arm_order_event_rows` records per-event `arm`, `event_id`, `order_id`, and `exact_outcome`, alongside `aggregate_row_recomputation` and `recomputed_prospective_metrics`.

## THE CHECK A READER CANNOT DO
none

## experiment_6655_repair_memory_safety_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The safety replay passed and point estimates were reproduced, but the order-level interval includes zero, so the result is narrowed to this fixture.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6656_arc_trace_automaton_live_loo.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing; the artifact only records per-action rows in `"accepted_training_trace_rows"`, including `"run_id"`, `"action_index"`, `"pre_action_features"`, and `"next_outcome"`, with no comparative or blocked verdict present

## THE CHECK A READER CANNOT DO
none

## experiment_6657_bounded_treewidth_ising_reference.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Bounded exact-reference readiness was not established because the required full Python suite and spec-coverage checks were missing.

## WHAT IS MISSING
nothing; `"gate_check_summary.failed_checks"` identifies `"tests"` and records `"observed_value": ["full_python_suite", "spec_coverage"]`, while `"per_unit_rows"` records fixture-level results.

## THE CHECK A READER CANNOT DO
none

## experiment_6658_thermodynamic_schedule_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `ising_reference_ready` was observed as `false` but required to equal `true`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6659_v580_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Two-step advisory verification beat one-step by catching 8 of 8 paired errors without false rejects.

## WHAT IS MISSING
Pair-level outcome rows for all eight semantic twins, including each pair ID, verifier condition, catch result, and false-reject result; `"claim_classification_rows.evidence"` and `"headline_recomputation.verifier_units"` contain only arm aggregates, while `"per_unit_rows"` contains task-inventory rows rather than per-pair metrics.

## THE CHECK A READER CANNOT DO
Did each of the eight pairs actually show the claimed one-step-versus-two-step improvement without any clean member being falsely rejected?
