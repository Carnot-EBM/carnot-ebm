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

## experiment_8393_python_transaction_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 8393 was blocked because upstream `direct_state_ready_score` was 0, failing the required equality check against 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8395_label_criterion_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 8395 was blocked because `released_label_panel_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8396_v723_cuda_failure_cause.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible "acceptance_gates" are all false, but the artifact is truncated before its headline verdict.

## WHAT IS MISSING
The final verdict and any blocker diagnostics, such as "gate_check_summary"; the visible "acceptance_gates" contain only false booleans.

## THE CHECK A READER CANNOT DO
Does the complete artifact report a blocked outcome and identify the failed check and observed value?

## experiment_8397_bounded_qwen_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The bounded Qwen canary was blocked because all three upstream qualification gates observed 0 instead of the required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8398_v723_arc_generalization_panel.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The panel was disqualified with all eight intended episodes unstarted; no comparative result is claimed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8399_v723_board_operation_boundary.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion reports unmet kernel-compatibility, transport-completeness, and scientific-benefit gates.

## WHAT IS MISSING
The artifact’s remainder: it cuts off inside "cited_upstream_artifacts", leaving any headline verdict and comparative results unseen. "acceptance_gates" and "board_obligations.kv260.missing" already record failed checks and missing prerequisites.

## THE CHECK A READER CANNOT DO
Does the omitted remainder make a comparative claim without per-unit metrics?

## experiment_8400_v723_gatemate_continuity.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The continuity obligation is recorded, while historical authentication and hardware execution remain blocked by two missing sources and four unmet physical prerequisites.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8401_v723_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
H1 reports a negative mean cost gain for spline34 versus RBF34: −0.00390625 across 128 intended units.

## WHAT IS MISSING
Complete `"paired_cost_rows"`: slot 100 truncates at `"gain_l"`, and slots 101–128 are absent despite `"intended_count": 128`. Their `"spline_cost"`, `"comparator_cost"`, and `"qualified"` values are needed. The optimizer blocker has a recorded `"qualification_reason"`.

## THE CHECK A READER CANNOT DO
Do the paired costs across all 128 units reproduce the reported `"mean_gain"` of −0.00390625?
