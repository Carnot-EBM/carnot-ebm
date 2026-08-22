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

## experiment_6498_csl_independent_audit.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The audit claims all critical attacks are closed and recommends `medium_bounded` capacity with total held-future utility `7.0`.

## WHAT IS MISSING
The artifact is truncated mid-row after `"arm_id": "always_updat`, so the complete per-unit future-utility rows and any final verdict or blocker diagnostic cannot be found; only aggregate fields such as `"future_metric_row_count": 28`, `"attacks_closed": true`, and `"recommended_capacity"` are visible.

## THE CHECK A READER CANNOT DO
Did `medium_bounded` outperform the other capacities broadly across future units, or does its reported utility depend on a small number of outlier rows?

## experiment_6499_arc_energy_progress_alignment.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The energy signal failed the readiness gate because it showed no positive incremental alignment beyond controls and was not directionally stable across games.

## WHAT IS MISSING
The actual `"rows"` or `"per_unit_rows"` containing each prefix/game/seed/horizon’s energy signal, progress outcome, and control metrics; only aggregated `"incremental_alignment_rows"`, binned `"calibration_rows"`, `"leave_one_game_out_rows"`, and a `"row_checksum"` are present.

## THE CHECK A READER CANNOT DO
Was the negative/null alignment broadly present across the 400 observations, or driven by a few outlying prefixes, games, seeds, or horizons?

## experiment_6500_gated_default_off_live_arc_policy_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `arc_energy_alignment_ready_score` was `0.0` instead of the required `1.0`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6501_v560_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The capstone claims its row recomputation confirms, among other outcomes, that exp6490’s best learned head beat the analytical baseline.

## WHAT IS MISSING
Per-unit held-result rows containing each unit’s identifier, outcome, and metric or prediction for every compared head; only aggregate `"head_metric_rows"`, `"held_row_count"`, and `"best_learned_beats_analytical"` are present.

## THE CHECK A READER CANNOT DO
Did the learned head’s advantage occur broadly across held units and family cells, or was it driven by a few outliers or degenerate cells?

## experiment_6502_v560_retirement_v561_lineage_lock.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V560 retirement ledger is complete and the V561 lineage-lock gate passed, permitting only the specified fresh exact-SAT/CSP lineage.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6503_v561_source_delta_method_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V561 source-delta method contract met its readiness gate, with all required receipts, boundaries, local-test targets, dependency constraints, and protected-file checks satisfied.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6504_exact_structural_benchmark_commitment.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims that the exact structural benchmark was completed and committed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6505_sota_formal_challenge_mutations.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment completed with a null result: all three model requests were quarantined, producing zero accepted mutations and a `challenge_pool_ready_score` of 0.0.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
