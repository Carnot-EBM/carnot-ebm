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

## experiment_8542_canonical_update_accounting.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `historical_learning_validation_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8545_v734_nine_input_native_qualification.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The excerpt reports Python/native request parity and failed acceptance gates, but the headline verdict is not visible.

## WHAT IS MISSING
The remainder of the artifact: it cuts off mid-string inside "complete_request_rows". The visible "acceptance_gates" already identifies "all_required_contracts": false and "owned_validation": false, so blocker information is present.

## THE CHECK A READER CANNOT DO
Does the missing headline verdict claim an additional comparative result requiring metrics beyond the displayed parity rows?

## experiment_8546_complete_native_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `nine_input_native_ready_score` was 0, failing the requirement that it equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8547_qwen_evidence_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The canary was blocked because `runtime_repair_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8548_v734_arc_panel_replay_qualification.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion reports execution readiness of 0, with “owned_coverage” and “terminal_replay” gates false.

## WHAT IS MISSING
The artifact ends mid-array inside “child_coverage_receipts.shards[].executed_lines”; its remaining contents and final verdict are unavailable. “acceptance_gates” already identifies failed checks, so missing blocker diagnostics cannot be alleged.

## THE CHECK A READER CANNOT DO
Does the final verdict make a comparative claim supported by per-unit results?

## experiment_8549_arc_capability_panel.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `arc_panel_execution_ready_score` was 0 and the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8550_v734_hardware_operation_boundary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Hardware deployment and acquisition are gated by missing qualified cost evidence and compatible installed operations, with failed checks and observed values recorded in "gate_check_summary".

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8551_v734_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact reports administrative completion while scientific generalization remains unqualified and listed tasks are blocked by identified prerequisite failures.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
