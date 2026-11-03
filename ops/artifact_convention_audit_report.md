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

## experiment_8739_learning_union.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Qualification was blocked because both readiness gates observed 0 where 1 was required, as recorded in "gates_evaluated".

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8742_verifier_family_transfer.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 8742 was blocked because upstream `verifier_transfer_methods_ready_score` was 0, failing the required equality to 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8743_v749_single_extraction_prototype.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible rows report 128 baseline extractions versus 64 reuse extractions in each listed condition, but the headline verdict is not visible.

## WHAT IS MISSING
The artifact cuts off inside "frozen_8744_workload.requests"; actual "honest_verdict", "verdict_class", and "gate_check_summary" values are unavailable—their names appear only in "field_principles". "feature_extraction_count_rows" supplies condition-level counts, and "acceptance_gates" explicitly records "full_python_suite" as false.

## THE CHECK A READER CANNOT DO
Does the artifact’s final headline verdict follow from its recorded results and failed acceptance check?

## experiment_8744_single_extraction_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `feature_reuse_ready_score` was 0, failing the required equality to 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8745_v749_bounded_durable_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The 90-cell comparison remained unexecuted because “gate_check_summary” records 0 middle-window dense renormalization writes against a requirement of at least 1; no comparative benefit is claimed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8746_v749_arc_outcome_frontier.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible artifact asserts `arc_frontier_ready_score=1` and `authority.ready=true`.

## WHAT IS MISSING
The artifact cuts off inside `authority.tasks[3].prompt`; any terminal verdict and remaining outcome evidence are unavailable. Visible `authority.rows` contain only `task_contract` metrics.

## THE CHECK A READER CANNOT DO
Does `arc_frontier_ready_score=1` represent administrative readiness or a scientific gate supported by per-unit outcomes?

## experiment_8747_v749_hardware_cost_boundary.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible fragment says acquisition priority did not change because no compatible physical cost evidence was admitted.

## WHAT IS MISSING
The artifact cuts off inside "authority.tasks[].prompt", before the actual result-level "honest_verdict" or "verdict_class" is visible. "authority.rows" contain task-contract metrics; "acquisition_priority_delta.reason" does provide a diagnostic reason.

## THE CHECK A READER CANNOT DO
Does the completed artifact claim a measured comparative result, report a blocked outcome, or merely document an acquisition decision?

## experiment_8748_v749_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
“G2.detail” claims independent reproduction of FoVer AUROC and learning contribution within published confidence intervals.

## WHAT IS MISSING
The artifact cuts off mid-value inside “code_config_hashes”; “G2.detail” provides aggregate intervals, but any later per-unit results, final verdict, or blocker diagnostics are unavailable.

## THE CHECK A READER CANNOT DO
Was the positive FoVer learning contribution broadly distributed across units or driven by a few outliers?
