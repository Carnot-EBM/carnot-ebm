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

## experiment_6493_gated_decomposed_trajectory_energy_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because both prerequisite gates failed: each observed readiness score was 0.0 instead of the required 1.0.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6495_restarted_factor_pool_controller.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The restarted factor-pool controller achieved a readiness score of 1.0, with all required row checks passing and all eight attacks failing closed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6496_continuous_factor_learning.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Execution completed, but continuous self-learning readiness was zero because all 16 arm-opportunity rows failed exact admission and produced no durable writes.

## WHAT IS MISSING
nothing; per-unit evidence appears in `"event_rows"`, `"exact_admission_rows"`, `"decision_action_rows"`, and `"evidence_update_rows"`, including diagnostic `"reason"` and `"compile_reason"` values such as `"empty_response"` and `"json_decode_error"`.

## THE CHECK A READER CANNOT DO
none

## experiment_6497_factor_pool_support_stress.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The capacity-stress evaluation recommends `"medium_bounded"` and claims capacity, safety, lifecycle, support, and stress checks passed despite one negative-transfer regression.

## WHAT IS MISSING
nothing; `"aggregate_row_recomputation"` is backed by per-event, per-capacity `"capacity_arm_rows"` containing metrics such as `"capacity_respected"`, `"exact_admission_passed"`, `"immediate_gain_charged"`, and `"no_write_reason"`.

## THE CHECK A READER CANNOT DO
none

## experiment_6498_csl_independent_audit.json

**CANNOT_DETERMINE**

> Audit-integrity guard: quoted field(s) ['future_metric_rows'] do not appear in the artifact, so this verdict was downgraded and must not be acted on.

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The artifact claims `medium_bounded` is the recommended capacity, with support preserved and total held-future utility of 7.0.

## WHAT IS MISSING
Per-unit `future_metric_rows` containing the held-future utility for every `capacity_id` and `event_id`; only `"future_metric_row_count": 28` and the aggregate `"recommended_capacity"` result are present.

## THE CHECK A READER CANNOT DO
Did `medium_bounded` outperform the competing capacities broadly across stress events, or was its recommendation driven by one outlier event?

## experiment_6499_arc_energy_progress_alignment.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The energy signal failed the readiness gate because it lacked positive incremental alignment and leave-one-game-out directional stability, yielding an honest null result.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6500_gated_default_off_live_arc_policy_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `arc_energy_alignment_ready_score` was observed as `0.0`, failing the requirement that it equal `1.0`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6501_v560_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The artifact claims row-recomputed comparative results, including that exp6490’s learned head beat the analytical baseline and that exp6497’s medium bounded capacity was the recommended stress configuration.

## WHAT IS MISSING
The underlying per-unit metric rows are missing: `"head_metric_rows"` contains only arm-level aggregates, while `"capacity_arm_row_count"`, `"future_support_row_count"`, `"future_utility_row_count"`, and `"negative_transfer_row_count"` are counts rather than the actual rows. The blocked cases do have diagnostics in `"blocked_diagnostic_rows"` with `"failed_field"`, `"failed_expected"`, and `"failed_observed"`.

## THE CHECK A READER CANNOT DO
Was exp6490’s reported 0.880952-versus-0.811625 advantage broad across held units, or driven by a few units, degenerate controls, or floor/ceiling cases?
