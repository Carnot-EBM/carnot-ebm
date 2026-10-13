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

## experiment_8462_brier_online_learning.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream "objective_kernel_ready_score" was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8464_v728_persistence_localization.json

**CANNOT_DETERMINE**

## VERDICT

CANNOT_DETERMINE

## WHAT THE CLAIM IS

All seven "acceptance_gates" are reported as passed.

## WHAT IS MISSING

The artifact cuts off inside "historical_model_provenance". Actual "honest_verdict", "verdict_class", "paired_repeat_rows", "rows", and "stage_rows" are not visible; those names appear only in "field_principles". "gate_check_summary" is empty, but no blocked verdict is visible.

## THE CHECK A READER CANNOT DO

Do per-unit measurements support the reported acceptance gates, or are those gates supported only by aggregates?

## experiment_8465_v728_journal_equivalence.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
“acceptance_gates” marks five checks true, but the final headline claim is not visible.

## WHAT IS MISSING
The artifact ends mid-row in “acknowledgment_ledger”; the remaining evidence and final verdict are unavailable.

## THE CHECK A READER CANNOT DO
Does the final verdict make a comparative claim or report a blocker supported by the remaining evidence?

## experiment_8466_v728_journal_service_cost.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible results report pooled speedups of 1.0448× and 0.9112× for batches 1 and 8, with both speed acceptance gates false.

## WHAT IS MISSING
The artifact cuts off inside "historical_model_provenance". "cell_results" provides aggregate metrics for five pairs per cell; actual "paired_repeat_rows", "rows", and "honest_verdict" values are not visible—their names appear only in "field_principles".

## THE CHECK A READER CANNOT DO
Did most paired runs support each reported "pooled_speedup", or did one outlier drive it?

## experiment_8467_bounded_qwen_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because both readiness gates observed 0 when they required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8468_v728_arc_generalization_panel.json

**CHECKABLE**

## VERDICT

CHECKABLE

## WHAT THE CLAIM IS

The panel was disqualified with all 16 episodes unstarted and no scientific benefit established.

## WHAT IS MISSING

nothing

## THE CHECK A READER CANNOT DO

none

## experiment_8469_v728_board_operation_boundary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Neither NPU nor TSU is qualified, with "compatible_operations" and "complete_transfer_device_clocks" recorded as false in "acceptance_gates".

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8470_v728_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The headline claim cannot be established from this truncated artifact.

## WHAT IS MISSING
Actual values for `"honest_verdict"`, `"scientific_claims"`, and `"rows"`: they are named in `"field_principles"`, but the supplied JSON ends mid-`"historical_model_provenance"` before providing them; `"gate_check_summary"` does provide blocker diagnostics.

## THE CHECK A READER CANNOT DO
Does the final headline assert a comparative result supported by per-unit metrics, or only aggregate summaries?
