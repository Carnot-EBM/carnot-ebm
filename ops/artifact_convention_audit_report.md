# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 6 |
| BLOCKED_WITHOUT_DIAGNOSTIC | 1 |
| CANNOT_DETERMINE | 1 |

## experiment_8816_v755_selective_evidence_methods.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The prospective selective-evidence protocol is complete and ready; “scientific_benefit_measured” is false, and no current comparative benefit or blocked verdict is asserted.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8817_categorical_memory_prototype.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Qualification was blocked because upstream `objective_candidate_ready_score` was 0, failing the required `== 1` gate.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8818_v755_evidence_decisive_coverage.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
“acceptance_gates” reports that coverage gain failed while the other three gates passed.

## WHAT IS MISSING
The artifact cuts off inside “exact_reference_rows”. Those visible rows cover “unrestricted_exact_reference”; “comparator_selection.menu” contains calibration aggregates. Paired evaluation rows recording “unit_id”, “arm”, and “metric” for the compared arms are not visible, and truncation prevents determining whether they appear later.

## THE CHECK A READER CANNOT DO
Do the compared arms’ per-unit evaluation results substantiate “acceptance_gates.coverage_gain”: false?

## experiment_8819_v755_consensus_selective_energy.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment completed without establishing selective benefit, with every recorded calibration threshold ineligible and "benefit_passed" false.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8823_qwen_witness_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `runtime_repair_ready_score` was 0 rather than the required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8824_arc_outcome_transfer.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The assessment was blocked because upstream `arc_new_outcomes_available_score` was 0, failing the required equality to 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8825_v755_hardware_operation_map.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The static three-board operation audit is complete, with no speedup claim and explicit blockers recorded in "gate_check_summary".

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8826_v755_capstone.json

**BLOCKED_WITHOUT_DIAGNOSTIC**

## VERDICT
BLOCKED_WITHOUT_DIAGNOSTIC

## WHAT THE CLAIM IS
The capstone reports a selective-benefit null and declares several branches blocked.

## WHAT IS MISSING
For `"exp8817-categorical-memory-prototype"`, the failed check and its observed value are missing: `"verdict_class": "blocked"` and `"disposition": "conductor_pre_gate"` identify no specific prerequisite. `"gate_check_summary"` appears only in `"field_principles"`, without diagnostic entries in the supplied artifact.

## THE CHECK A READER CANNOT DO
Which prerequisite blocked exp8817, and what observed value caused it to fail?
