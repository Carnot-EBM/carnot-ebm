# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 4 |
| CANNOT_DETERMINE | 4 |

## experiment_8868_v759_conditional_risk_methods.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The conditional-risk methods are ready, with no paired coverage benefit across 40 independent replicates.

## WHAT IS MISSING
The supplied artifact cuts off inside `preconditions_checked.references`. `paired_replicate_differences` records 40 zeros, but per-arm coverage and risk rows are not visible; `expected_row_counts.rows` lists 3,360 expected rows.

## THE CHECK A READER CANNOT DO
Did the zero differences reflect positive coverage in both arms or both arms accepting nothing?

The truncated remainder was unavailable for checking; it may contain the needed rows.

## experiment_8869_matched_memory_methods.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 8869 was blocked because upstream `diagnosis_ready_score` was 0, failing the required equality to 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8870_v759_conditional_risk_evaluation.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
No calibration-family violations occurred across 40 replicates.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8871_v759_size_aware_verifier.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The installed size-aware verifier passes acceptance gates, including exact parity with reference verifiers.

## WHAT IS MISSING
The artifact ends mid-entry in "operation_accounting_rows"; "semantic_parity_rows" and "rows" appear in "field_principles", but their actual contents are not visible.

## THE CHECK A READER CANNOT DO
Does each completed unit match both reference verifiers?

The truncated remainder was not checked; it may contain the required evidence.

## experiment_8874_v759_size_aware_complete_cost.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The recorded acceptance gates pass, but mixture-speed benefit gates fail for both boundaries and the cold size-8 latency-regression gate fails.

## WHAT IS MISSING
The artifact stops mid-field inside "censoring_rows". "acceptance_gates" and "benefit_gates" contain booleans; visible per-unit rows cover only censored cold size-64 cases. Completed per-unit timings underlying the comparative gates are not visible, and truncation prevents determining whether they appear later.

## THE CHECK A READER CANNOT DO
Do paired completed size-8 cold timings support the reported failure of "size8_latency_regression_upper_at_most_five_percent", which the truncated artifact leaves unchecked?

## experiment_8875_qwen_source_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `runtime_repair_ready_score` was 0 while the gate required 1.

## WHAT IS MISSING
nothing — `gate_check_summary` and `gates_evaluated` identify the failed check, observed value, and required value.

## THE CHECK A READER CANNOT DO
none

## experiment_8876_v759_hardware_operation_budget.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Acquisition is deferred because physical acceleration is unmeasured, while upstream completion is blocked by missing V759 producer inputs.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

Not checked: external evidence or the truncated remainder; this verdict covers only the supplied text.

## experiment_8877_v759_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion reports accounting and owned-validation gates passed, natural generalization failed, and activation unauthorized.

## WHAT IS MISSING
The artifact is truncated inside "branch_conclusions[2].reduction.world_ids". The remaining conclusions and any per-unit measurements are unavailable; visible "world_ids", "calibration_replicate_ids", and "constructed_measurement_count" provide identifiers and counts, not metric rows.

## THE CHECK A READER CANNOT DO
Does the unseen remainder support any comparative claim with per-unit measurements? That content was not checked; its absence prevents a definitive classification.
