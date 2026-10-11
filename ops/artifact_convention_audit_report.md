# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 5 |
| CANNOT_DETERMINE | 3 |

## experiment_8410_v724_cuda_reader_qualification.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible fragment reports `"cached_reader": true`, but ends mid-prompt before showing a terminal headline verdict.

## WHAT IS MISSING
The remainder of the artifact, including terminal `"honest_verdict"` and `"verdict_class"` values and outcome rows or diagnostics. `"acceptance_gates"`, `"environment_delta_rows"`, and `"contract_rows"` are present; `"execution_authority.gate_check_summary"` is empty.

## THE CHECK A READER CANNOT DO
Does the terminal verdict assert a comparative result without supporting per-unit measurements, report an undiagnosed blocker, or do neither?

## experiment_8411_bounded_qwen_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The Qwen evidence canary was blocked because `runtime_changed_score` and `cuda_context_ready_score` were both 0, while each gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8412_v724_arc_panel_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Panel qualification completed, but scientific benefit and independent generalization were not established.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8413_v724_board_operation_boundary.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion reports unmet board-acceptance gates, but no final verdict or comparative headline claim is visible.

## WHAT IS MISSING
The complete headline claim/verdict: the artifact cuts off mid-entry in "cited_upstream_artifacts". Visible "acceptance_gates" records failed checks, and "board_operation_rows" includes "reason" and "missing_reason", so blocker diagnostics are present.

## THE CHECK A READER CANNOT DO
Does the omitted remainder claim a comparative benefit, and provide per-unit evidence supporting it?

## experiment_8414_v724_gatemate_obligation.json

**CHECKABLE**

## VERDICT

CHECKABLE

## WHAT THE CLAIM IS

The obligation is recorded, while historical authentication and hardware execution remain blocked by missing exact source bytes and physical/device evidence.

## WHAT IS MISSING

nothing

## THE CHECK A READER CANNOT DO

none

## experiment_8415_v724_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
H1 reports a negative descriptive mean cost gain for spline34 versus RBF34, with geometry unqualified.

## WHAT IS MISSING
Complete "paired_cost_rows": slot 100 is truncated and slots 101–128 are absent, despite "intended_count": 128 and "bootstrap_summary" reporting "mean_gain". The geometry diagnostic is present in "qualification_reason" and "residuals".

## THE CHECK A READER CANNOT DO
Do the 128 per-unit cost differences reproduce the reported all-intended mean gain of −0.00390625?

## experiment_8416_v725_contract_methods.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Methods are frozen, but the contract is blocked by a missing independent design binding, only 4 of 13 task objects being present, and a missing runtime-repair receipt, all diagnosed in "gate_check_summary".

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8417_criterion_feature_panel.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `methods_ready_score` was 0, failing the required equality check against 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
