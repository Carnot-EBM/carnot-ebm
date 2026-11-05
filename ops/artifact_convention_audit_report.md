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

## experiment_8831_v756_ordered_evidence_cost.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The paid-operation-reduction gate failed its 0.20 threshold while all other listed acceptance gates passed.

## WHAT IS MISSING
The remainder of "decision_rows" and observed statistics underlying "acceptance_gates": the supplied artifact ends mid-row, preventing reconciliation with "completed_count": 4320.

## THE CHECK A READER CANNOT DO
Do the complete paired results justify the reported failure of the 0.20 paid-operation-reduction gate?

## experiment_8832_v756_selective_availability.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
The artifact is truncated mid-value at `"bootstrap_control_rows"` → `"calibration_rows"` → `"metrics"` → `"brier"`, leaving any headline conclusion and remaining evidence unavailable.

## THE CHECK A READER CANNOT DO
Does the complete artifact make a comparative or blocked claim with supporting rows or a blocker diagnostic?

## experiment_8833_causal_objective_streams.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 8833 was blocked because upstream `memory_execution_ready_score` was 0, failing the requirement that it equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8835_v756_complete_adaptation_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run completed 90 adaptation-cost windows across three arms without establishing a scientific speed gain.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8836_qwen_witness_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `runtime_repair_ready_score` was 0, failing the required value of 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8837_arc_outcome_transfer.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The assessment was blocked because upstream `arc_new_outcomes_available_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8838_v756_hardware_operation_budget.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The mapping audit passes, but physical acceleration is not established, with board operations recorded as host-only or blocked by a JTAG result of 0xffffffff.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8839_v756_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The categorical candidate clears the development Brier improvement gate in all five seeds, but activation remains unauthorized because natural generalization is absent.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
