# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 3 |
| AGGREGATE_ONLY | 1 |
| CANNOT_DETERMINE | 4 |

## experiment_7210_span_capture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `exp7209-span-canary.span_canary_ready_score` was 0 instead of the required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7212_v635_refinement_fixture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The refinement fixture is ready, while learning value remains unmeasured and prior V634 null results remain unpromoted.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7213_v635_refinement_learning.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The primary learning gate and secondary compiled-deployment gate failed, and version-space superiority was not claimed.

## WHAT IS MISSING
The artifact is truncated inside `"commit_deletion_rows"` at `"arm":"witness_q`, so its remaining fields are unavailable; although `"primary_criteria"` and `"secondary_deployment_criteria"` record failed checks and observed values, it is impossible to determine whether later per-unit comparison rows are present.

## THE CHECK A READER CANNOT DO
Do per-unit results supporting the aggregate arm-comparison statistics appear in the missing remainder of the artifact?

## experiment_7214_v635_refinement_cold_audit.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
cannot determine because the artifact is truncated mid-row before any headline verdict or summary appears

## WHAT IS MISSING
The remainder of the artifact, including any headline/verdict and blocker diagnostic fields; `"causal_control_rows"`, `"passed"`, `"metric"`, `"seed"`, and `"unit_id"` are present, but the JSON ends during a `"prospective_control_error"` row.

## THE CHECK A READER CANNOT DO
Does the artifact’s final headline claim follow from the complete per-seed results and recorded gate diagnostics?

## experiment_7215_v635_down_up_prototype.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact claims that all 90 finite transition cells and all three mutation controls passed, certifying the elementary CPU kernel.

## WHAT IS MISSING
The complete `"rows"` array containing all 90 transition-cell records is missing because the artifact ends mid-row; `"honest_verdict"`, `"mutation_rows"`, and `"gate_check_summary"` are present.

## THE CHECK A READER CANNOT DO
Did each of the claimed 90 transition cells individually pass its stationarity, detailed-balance, stochasticity, and empirical-comparison checks?

## experiment_7216_v635_down_up_quality.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion reports `"down_up_comparison_complete_score": 1` and `"down_up_value_score": 0`, but the artifact ends before any definitive headline verdict.

## WHAT IS MISSING
The complete artifact, including any final verdict, `"gate_check_summary"`, comparison rows, and per-unit arm metrics; only `"exact_authority_rows"` and aggregate score fields are visible before the JSON truncates mid-row.

## THE CHECK A READER CANNOT DO
Does the completed artifact claim a comparative gate was met, and if so, do paired per-unit results support that claim?

## experiment_7217_v635_abi_board_readiness.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The selected interpreter successfully imported and executed the compiled sampler, its explicit transition outputs matched the recorded Exp7187 authority, and serialized state survived restoration in a second process.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7218_v635_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The artifact claims the refinement mechanism should be retired because its primary gate failed and the strong version-space arm performed better.

## WHAT IS MISSING
Actual per-unit metric rows for the compared refinement and version-space arms are missing; only conclusions in `"branch_decisions"` and planned `"per_unit_rows": true` declarations in `"contract_receipt.tasks"` are present.

## THE CHECK A READER CANNOT DO
Were the version-space arm’s gains consistent across units, or driven by an outlier or degenerate control units?
