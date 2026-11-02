# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 1 |
| CANNOT_DETERMINE | 7 |

## experiment_8729_v748_equal_budget_verifier.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible "arm_rows" reports higher defect detection for targeted (130/150, 86.7%) than baseline (120/150, 80%).

## WHAT IS MISSING
The artifact ends mid-entry in "cited_upstream_artifacts". Visible "arm_rows" contains only aggregates; the missing remainder prevents determining whether paired per-unit detection outcomes are recorded elsewhere in the artifact.

## THE CHECK A READER CANNOT DO
What were the baseline and targeted detection results for each defective evaluation unit?

## experiment_8730_qwen_self_score_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 8730 was blocked because upstream `runtime_repair_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8731_v748_resident_sparse_prototype.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
All five "acceptance_gates" passed, including "exact_native_parity".

## WHAT IS MISSING
The artifact cuts off mid-"ack" in "boundary_rows"; the remaining per-unit parity results and any concluding verdict or diagnostic are unavailable.

## THE CHECK A READER CANNOT DO
Does "exact_native_parity": true hold across every head and sequence, beyond the partial head-0 boundary results shown?

## experiment_8732_v748_resident_update_cost.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible fragment records failed acceptance gates but does not show a headline claim or overall verdict.

## WHAT IS MISSING
The remainder of the artifact: it ends mid-string inside "cited_upstream_artifacts". "acceptance_gates" identifies failed checks and "branch_gate_values" records zero scores, so BLOCKED_WITHOUT_DIAGNOSTIC is not supported.

## THE CHECK A READER CANNOT DO
Does the completed artifact assert a comparative result supported by per-unit measurements?

## experiment_8733_v748_installed_cost_attribution.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
“acceptance_gates” reports that process completion and low-overhead attribution both passed.

## WHAT IS MISSING
The supplied artifact truncates inside “component_span_rows” at “parent_d”. Per-process measurements are present, but the complete rows and the overhead value and threshold supporting “low_overhead_attribution” cannot be inspected.

## THE CHECK A READER CANNOT DO
Does the complete set of measurements underlying “completed_count”: 40 support the reported low-overhead gate pass?

## experiment_8734_v748_arc_outcome_frontier.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible artifact claims readiness through `"arc_frontier_ready_score": 1` and `"authority.ready": true`.

## WHAT IS MISSING
The artifact cuts off inside `"authority.rows"` during the exp8727 `"task_object"`, leaving the remaining evidence and final verdict unavailable; the visible `"authority.gate_check_summary"` is empty.

## THE CHECK A READER CANNOT DO
Do all intended units have recorded passing checks supporting the readiness claim?

## experiment_8735_v748_hardware_cost_boundary.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion claims `"authority.ready": true`, but the artifact is truncated before its final outcome.

## WHAT IS MISSING
The JSON ends inside `"gated_on"` at `"op":`; the final `"honest_verdict"`, `"verdict_class"`, and remaining results are unavailable. `"authority.rows"` and `"authority.gate_check_summary": []` are visible.

## THE CHECK A READER CANNOT DO
Does the final verdict have the per-unit evidence or blocker diagnostic needed to check it?

## experiment_8736_v748_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible excerpt claims an independent reproducer passed the published FoVer AUROC and learning-contribution intervals.

## WHAT IS MISSING
The artifact is truncated inside "code_config_hashes". "G2.detail" gives aggregate intervals, but any subsequent per-unit metrics or final verdict are unavailable. "acceptance_gates" and "branch_gate_values" do record failed checks and values.

## THE CHECK A READER CANNOT DO
Was the reported learning contribution broadly shared across evaluated units or driven by a few outliers?
