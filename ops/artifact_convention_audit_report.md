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

## experiment_8583_learning_benefit_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 8583 was blocked because upstream `causal_learning_rows_ready_score` was 0, failing the required value of 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8584_v737_resident_complete_cost.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The fragment reports 240 completed measurements, but the headline verdict is not visible.

## WHAT IS MISSING
The artifact ends mid-string inside "field_principles.honest_verdict"; the actual verdict and supporting results are unavailable. "acceptance_gates" and "completed_count" are present but do not establish a comparative claim or blocked outcome.

## THE CHECK A READER CANNOT DO
Does the completed artifact claim a comparative benefit, report a blocker, or make no claim?

## experiment_8585_v737_hallutruth_release_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The 4,000-row release audit completed as a null, making no comparative performance claim and documenting reasons for failed eligibility gates.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8586_qwen_evidence_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run was blocked because canary_inputs_ready_score and runtime_repair_ready_score each equaled 0 when their gates required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8587_v737_arc_publication_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
ARC publication and terminal replay qualification passed, with no scientific-benefit claim and with per-panel and per-check receipts recorded.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8588_v737_arc_capability_panel.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The arm-support and useful-efficiency gates are unmet, and no scientific reasoning benefit is claimed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8589_v737_resident_hardware_boundary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Resident acceleration is unmeasured and acquisition is unauthorized because qualified resident costs and compatible device execution are missing, with failed checks and observed values recorded in "gate_check_summary".

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8590_v737_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible excerpt reports administrative completion and ARC-panel qualification, with learning benefit and resident cost unqualified.

## WHAT IS MISSING
The artifact cuts off inside "historical_model_provenance"; its top-level verdict and any remaining supporting rows are unavailable. "acceptance_gates" is present. Blocker diagnostics are present in "blocked_inputs" → "source_gates", including failed checks and expected/observed values.

## THE CHECK A READER CANNOT DO
Does the reported ARC-panel qualification have supporting per-unit metrics in the complete artifact?
