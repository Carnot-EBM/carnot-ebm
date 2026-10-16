# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 6 |
| CANNOT_DETERMINE | 2 |

## experiment_8556_causal_brier_stream.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `online_adapter_ready_score` was 0, failing the required value of 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8558_v735_nine_input_native_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The native implementation passes the listed acceptance gates, including Python parity, supported by paired per-unit results in "complete_request_rows".

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8559_v735_complete_native_cost.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Native execution is reported ready, with complete throughput approximately 1.010× Python’s.

## WHAT IS MISSING
The artifact truncates inside "component_span_rows", mid-"start_ns". "cell_statistics" provides batch summaries, but the remaining measurements and any final verdict or diagnostics are unavailable.

## THE CHECK A READER CANNOT DO
Do complete-boundary measurements for both arms across repeats support the reported "complete_throughput_ratio"?

## experiment_8560_qwen_evidence_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The canary was blocked because upstream `runtime_repair_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8561_v735_arc_panel_replay_qualification.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion reports failed qualification gates but contains no explicit headline verdict or comparative claim.

## WHAT IS MISSING
The artifact cuts off inside "child_coverage_receipts.shards". "acceptance_gates.owned_coverage", "acceptance_gates.terminal_replay", and "child_coverage_receipts.passed" are false, but the remainder containing any final verdict or blocker diagnostic is unavailable.

## THE CHECK A READER CANNOT DO
Does the missing remainder explain which check caused qualification to fail and what value it observed?

## experiment_8562_arc_capability_panel.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `arc_panel_execution_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8563_v735_hardware_operation_boundary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Hardware deployment and acquisition remain unqualified because compatible installed operations and qualified device costs are missing.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8564_v735_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact reports administrative completion, with documented blocked tasks and no comparative scientific benefit claimed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
