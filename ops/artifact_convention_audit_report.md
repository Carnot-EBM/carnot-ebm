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

## experiment_6451_typed_fact_grounding_fixed_policy_logic_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `sota_corpus_ready_score` was 0.0 instead of the required 1.0.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6453_held_verifier_budget_allocation_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `sota_corpus_ready_score` was `0.0`, failing the required equality to `1.0`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6455_prospective_verifier_bounded_factor_weight_csl.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Verifier-bounded updates improved future exact yield over comparison arms by a mean paired delta of 1.0 across 69 distinct future units.

## WHAT IS MISSING
The actual `"per_unit_rows"` containing each unit’s arm-level exact-yield metrics and paired delta are missing; only aggregate claims in `"effects_and_uncertainty_over_distinct_future_units"` and metadata about 216 rows in `"aggregate_row_recomputation"` are present, while `"raw_pool_receipts"` contain hashes and paths rather than outcome metrics.

## THE CHECK A READER CANNOT DO
Did all 69 future units improve, or was the reported pooled improvement driven by outliers, degenerate controls, or units pinned at metric bounds?

## experiment_6456_corrupt_feedback_held_restart_csl_replication.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Clean and governed learning improved held-unit exact yield over frozen weights while safely containing all corrupt feedback, meeting the readiness gate with a score of 1.0.

## WHAT IS MISSING
The actual `"per_unit_rows"` array containing each held unit’s model, arm, and outcome metric is missing; only aggregate claims such as `"effects_and_uncertainty_over_distinct_held_units"`, `"future_exact_yield_delta"`, and `"aggregate_row_recomputation"` are written down, while `"device_and_runner_receipts.raw_pool_receipts"` provides hashes and paths but no comparative outcomes.

## THE CHECK A READER CANNOT DO
Were the reported clean and governed improvements broad across held units, or driven by a few outliers or units pinned at metric bounds?

## experiment_6457_independent_verifier_bounded_csl_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The independent verifier-bounded CSL audit did not grant eligibility because `raw_outputs_unique_and_partitions_disjoint` and `zero_current_critical_findings` failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6458_arc_representation_objective_generalization_ab.checkpoints.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact is truncated before any headline claim or verdict is recorded.

## WHAT IS MISSING
The complete artifact, including any top-level verdict, aggregate metrics, and `"gate_check_summary"`; only per-cell fields such as `"row_id"`, `"arm"`, `"seed"`, and `"policy_influence"` are visible before the JSON cuts off mid-field.

## THE CHECK A READER CANNOT DO
Did the experiment ultimately claim that one arm beat another, meet a gate, or become blocked—and what recorded evidence supported that verdict?

## experiment_6458_arc_representation_objective_generalization_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The audit completed but readiness was not achieved because the frozen safety roster regressed.

## WHAT IS MISSING
nothing; `"per_unit_rows"` records unit-level metrics, while `"gate_check_summary.failed_gates"` identifies `"frozen_safety_roster_not_regressed"` and `"regressed_games"` identifies `"g50t"`.

## THE CHECK A READER CANNOT DO
none

## experiment_6459_v555_adversarial_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V555 terminal handoff was blocked because queue integrity failed at `prior_failure_validation`, with `gate_audit_passed` observed as false.

## WHAT IS MISSING
nothing; `gate_check_summary` records `failed_check`, `expected_condition`, `observed_value`, `prior_failure_violations`, and `gate_audit_failure_details`.

## THE CHECK A READER CANNOT DO
none
