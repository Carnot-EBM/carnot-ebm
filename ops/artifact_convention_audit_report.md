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

## experiment_6654_prospective_repair_memory_evolution.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The verified-memory arm achieved higher prequential exact yield than the context-only arm overall and on eligible future events.

## WHAT IS MISSING
nothing; `"arm_order_event_rows"` records per-event `"arm"`, `"order_id"`, `"event_id"`, and `"exact_outcome"`, while `"aggregate_row_recomputation"` records the derived comparisons.

## THE CHECK A READER CANNOT DO
none

## experiment_6655_repair_memory_safety_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The safety replay passed and point estimates were reproduced, but the order-level interval includes zero, so the result is limited to this fixture.

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
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6657_bounded_treewidth_ising_reference.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Bounded exact-reference readiness was not established because the tests check failed, with `full_python_suite` and `spec_coverage` missing.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6658_thermodynamic_schedule_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `ising_reference_ready` was observed as `false` when the gate expected `true`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6659_v580_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
V580 produced valid admission and direct-corpus evidence, and two-step verification beat one-step on eight paired errors, while memory showed no general benefit and ARC and Ising remained blocked.

## WHAT IS MISSING
Pair-level verifier outcome rows showing each twin’s result under each arm, including catch and false-reject metrics; `"per_unit_rows"` contains task-level availability receipts, while `"headline_recomputation.verifier_units"` contains only arm aggregates.

## THE CHECK A READER CANNOT DO
Did two-step verification improve outcomes across all eight pairs, or was the aggregate advantage driven by only a subset of pairs or degenerate one-step results?

## experiment_6661_triggered_tail_fixture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The deterministic fixture is ready, with all recorded row, outcome, balance, and leakage checks passing; it makes no model-quality claim.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6662_triggered_structured_tail_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `triggered_tail_fixture_ready` was `false` when the gate expected `true`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
