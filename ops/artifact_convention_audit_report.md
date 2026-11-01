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

## experiment_8681_v745_contract_methods.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
All 14 administrative contract objects match and authority mutation checks pass, while methods readiness is zero with "owned_validation" and "sealed_methods" both false.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8682_frozen_learning_core.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Qualification was blocked because `methods_ready_score` was 0 while the gate required 1, as recorded in `gates_evaluated` and `gate_check_summary`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8684_relational_evidence_controls.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 8684 was blocked because upstream `methods_ready_score` was 0, failing the required value of 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8688_installed_batch_prototype.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `exp8681-contract-methods.methods_ready_score` was 0, failing the required value of 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8691_v745_arc_censoring_decision.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible “arm_support_rows” recommend retaining all four arms unchanged, with no uncensored efficacy support.

## WHAT IS MISSING
The artifact’s remainder: “censoring_rows” truncates inside “reset_owner_cause,” leaving the remaining records and any overall verdict or diagnostic unavailable.

## THE CHECK A READER CANNOT DO
Does “censored_count”: 16 match sixteen documented per-unit censored outcomes?

## experiment_8692_v745_spline_precision_feasibility.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
"acceptance_gates" reports all four gates passed, including "primary_32_24".

## WHAT IS MISSING
The supplied artifact cuts off inside "cpu_operation_spans". Criteria and per-unit measurements supporting "primary_32_24" are not visible; "accumulator_bounds" records per-head bounds but does not explain that gate.

## THE CHECK A READER CANNOT DO
Which per-unit measurements, evaluated against which thresholds, establish that "primary_32_24" passed?

## experiment_8693_v745_hardware_operation_boundary.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion reports integer-precision admission while physical execution and complete-cost readiness remain unmet.

## WHAT IS MISSING
The final headline claim/verdict: the artifact cuts off mid-string in "exact_union_receipt.original_node_ids". Visible "board_status_rows" and "conditional_ceiling_rows" do record missing evidence and reasons.

## THE CHECK A READER CANNOT DO
Does the final verdict claim a comparative benefit, or only document readiness boundaries?

## experiment_8694_v745_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
“G2.detail” claims independent reproduction of the FoVer AUROC and positive learning contribution within published CIs.

## WHAT IS MISSING
FoVer per-unit condition/control metrics are not visible: “G2.detail” gives aggregate summaries, while “arc_decision_rows.horizon_rows” contains ARC observations. The supplied JSON ends mid-row, so whether the remainder contains the required comparisons is unknown.

## THE CHECK A READER CANNOT DO
Is the positive FoVer learning contribution broad across units or driven by a few outliers?
