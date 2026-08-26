# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 4 |
| AGGREGATE_ONLY | 2 |
| CANNOT_DETERMINE | 2 |

## experiment_6611_live_arc_invariant_projection.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Selected invariant projection had no effect on held exact-next-frame prediction, with no solve claim.

## WHAT IS MISSING
Complete `"per_unit_rows"` for every game, transition, seed, and arm; the artifact truncates during the first `"no_projection"` row, while `"held_arm_summary"` contains only aggregates. `"gate_check_summary"` is present and says `"blocked": false`.

## THE CHECK A READER CANNOT DO
Were errors identical row-by-row across all three arms, or did improvements and regressions cancel to produce equal aggregate means?

## experiment_6612_spectral_k_block_scale_rust_parity.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
`spectral_k_block_rust` beat `sequential_gibbs`, qualifying as a `software_win` with positive transition-efficiency and wall/charged-time gains.

## WHAT IS MISSING
The actual `per_unit_rows` containing metrics for every fixture, seed, and arm are missing; only aggregate `efficiency_summary.arm_means` and `efficiency_summary.rows` are present, despite references to `per_unit_rows` in `reducer`, `field_principles`, and `field_provenance`.

## THE CHECK A READER CANNOT DO
Did the Rust arm improve broadly across the 60 matched units, or was the reported mean gain driven by a few extreme rows?

## experiment_6613_invariant_memory_lifecycle.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The invariant-memory lifecycle passed its conformance checks and is ready, while making no utility claim.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6614_prospective_invariant_self_learning.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact claims continuous self-learning is not ready because held-future benefit gates and the tests gate failed.

## WHAT IS MISSING
The artifact is truncated mid-record, so the referenced `"per_unit_rows"`, `"gate_check_summary"`, `"honest_verdict"`, and `"status"` fields cannot be found or confirmed; the visible `"acceptance_gate_rows"` does diagnose the tests failure as `"expected": true`, `"observed": false`, `"passed": false`.

## THE CHECK A READER CANNOT DO
Did governed online memory broadly fail to outperform both controls across individual events and seeds, or was each failed benefit gate driven by only a few units?

## experiment_6605_qwen36_direct_headroom.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
All ten attack conditions failed closed with a `candidate_integrity_ready_score` of 0.0, matching the `expected_integrity_ready_score` of 0.0.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6615_v576_independent_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The capstone claims live projection and prospective self-learning produced zero benefit versus their controls, while decoding remained blocked.

## WHAT IS MISSING
Actual metric-bearing experimental rows are missing: `"per_unit_rows"` contains only task/comparative-group receipts and `"row_store_counts"`, while `"live_projection_replay.arm_summaries"` and `"continuous_learning_replay.held_future_benefit_over_static"`/`"held_future_benefit_over_shuffled"` provide only aggregates.

## THE CHECK A READER CANNOT DO
Were the reported zero comparative effects consistent across games and held-future pairs, or produced by offsetting wins, losses, outliers, or units with no headroom?

## experiment_6616_v577_execution_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The execution contract is blocked because the expected Exp6616–Exp6628 YAML task contracts are not all present.

## WHAT IS MISSING
nothing; `"gate_check_summary.failed_checks"` records the failed `"exact_task_set"` and `"model_policy"` checks with their `"expected"` and `"observed"` values, including the missing task IDs.

## THE CHECK A READER CANNOT DO
none

## experiment_6617_gpu_lease_phase_receipts.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `execution_contract_ready_score` was 0.0 instead of the required 1.0.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
