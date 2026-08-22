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

## experiment_3343_verifier_diversity_reaudit_after_axis_v3.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
Diversity improved after remediation, with higher `"lambda_min_sigma_after"` and `"effective_k_after"`, no remaining collapsed pairs, and `"diversity_remediation_passed": true`.

## WHAT IS MISSING
Per-case rows for all `"n_cases": 1000`, including before-and-after diversity metrics and collapsed-pair status for each case; only pooled fields such as `"lambda_min_sigma_before"`, `"lambda_min_sigma_after"`, `"effective_k_before"`, and `"effective_k_after"` are present.

## THE CHECK A READER CANNOT DO
Did diversity improve broadly across the 1,000 cases, or were the reported aggregate gains driven by a few outliers or degenerate cases?

## experiment_2824_cross_corpus_verifier_matrix.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The production verifier outperformed architecture-only for tier_memory on FoVer by 0.25, while the other reported comparisons were tied.

## WHAT IS MISSING
Per-unit rows for each game, seed, cell, or condition underlying the aggregate `"production"`, `"architecture_only"`, and `"delta"` values in `"verifier_corpus_dual_matrix"`.

## THE CHECK A READER CANNOT DO
Was the tier_memory FoVer improvement broad across units, or caused by one outlier or units pinned at a floor or ceiling?

## experiment_3361_archive_v309_activate_v310.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The archive is complete and milestone 2026.05.310 is ready for activation.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

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

## experiment_833_constraint_delta_root_cause.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The embedding constraint store’s write path is missing: verification performed 10 retrievals but zero writes.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_3392_archive_v311_activate_v312.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The archive is complete and milestone 2026.05.311 was archived with 2026.05.312 activated and ready.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6506_v561_evidence_corrigendum_v562_lineage_lock.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact claims the V562 exact branch passed its lineage/readiness gate with `"v562_exact_branch_ready_score": 1.0`.

## WHAT IS MISSING
The complete `"per_unit_rows"` array is missing: `"exp6504_row_recomputation"` reports 480 instance rows, but the supplied artifact cuts off mid-row at instance `exp6504:random_3cnf:031` and is not complete JSON.

## THE CHECK A READER CANNOT DO
Do all 480 instance rows actually have matching regenerated labels and passing replays, as the reported readiness aggregate claims?

## experiment_6508_analytical_branch_refocus_ab.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the required upstream artifact `exp6507-exact-branch-counterfactual-dataset` was not found, so its `branch_counterfactual_dataset_ready_score == 1.0` gate failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
