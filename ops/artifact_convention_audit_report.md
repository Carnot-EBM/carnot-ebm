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

## experiment_8491_objective_benefit_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `objective_stream_ready_score` was 0 while the gate required 1.

## WHAT IS MISSING
nothing — `gate_check_summary`, `failed_field`, `failed_expected`, `failed_observed`, and `gates_evaluated` identify the failed check and its values.

## THE CHECK A READER CANNOT DO
none

## experiment_8492_v730_publication_cost_model.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The publication cost model passes engineering readiness gates and reports conditional speed ceilings, without claiming measured append speedup.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8493_v730_append_store_qualification.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
“acceptance_gates” marks all five checks true, but the excerpt ends before an overall headline result is visible.

## WHAT IS MISSING
The artifact’s remaining contents, including its headline result or verdict: the supplied JSON cuts off inside “acknowledgment_ledger.rows[].response[]” after “changed”.

## THE CHECK A READER CANNOT DO
Does the complete artifact support its headline outcome with per-unit comparison evidence or identify a blocking check?

## experiment_8494_v730_append_service_cost.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Append reports pooled speedups over snapshot of 2.3386× at batch 1 and 1.4663× at batch 8, while receiving a disqualified verdict.

## WHAT IS MISSING
Complete batch-8 entries in "paired_repeat_rows", including each repeat’s "append_wall_s" and "snapshot_wall_s": the supplied artifact truncates during repeat 1, while "cell_results" reports five completed pairs.

## THE CHECK A READER CANNOT DO
Does the reported batch-8 speedup reflect consistent improvement across all five pairs, or one unusually favorable repeat?

## experiment_8495_bounded_qwen_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `runtime_repair_ready_score` was 0, failing the required equality to 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8496_v730_arc_panel_qualification.json

**CANNOT_DETERMINE**

## VERDICT

CANNOT_DETERMINE

## WHAT THE CLAIM IS

The visible "acceptance_gates" report that redirect support and scientific benefit failed.

## WHAT IS MISSING

The artifact ends mid-row in "descriptive_old_subset"; complete per-unit results for "completed_count": 16 and any final verdict or "gate_check_summary" are not visible.

## THE CHECK A READER CANNOT DO

Do the results across all 16 episodes support the failed "scientific_benefit" gate?

## experiment_8497_v730_board_operation_boundary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
NPU/TSU qualification is not met, with "compatible_operations" and "complete_transfer_device_clocks" both false under "acceptance_gates".

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8498_v730_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion reports administrative acceptance while withholding independent scientific generalization, but the artifact is truncated before its headline verdict.

## WHAT IS MISSING
The remainder of the artifact: actual values for "honest_verdict", "scientific_claims", and "rows" are unavailable; their names appear in "field_principles". The visible "gate_check_summary" supplies failed checks and observed values, so blocker diagnostics are present.

## THE CHECK A READER CANNOT DO
Does the complete artifact include per-unit metrics supporting the historical FoVer learning-contribution claim?
