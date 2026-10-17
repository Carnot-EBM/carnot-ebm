# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 3 |
| CANNOT_DETERMINE | 5 |

## experiment_8593_v738_resident_provenance_qualification.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The headline claim cannot be identified because the supplied artifact is truncated.

## WHAT IS MISSING
The JSON cuts off inside a "provenance_reference" path within "classifier_reproduction_rows"; the remaining results and final verdict are unavailable. Visible "acceptance_gates" and build "receipts" do not establish the headline conclusion.

## THE CHECK A READER CANNOT DO
Does the final conclusion report a comparison supported by per-unit measurements, a diagnosed blocker, or merely a receipt?

## experiment_8594_causal_brier_stream.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `learning_execution_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8596_v738_resident_complete_cost.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion reports that input, measurement-completeness, and validation gates passed.

## WHAT IS MISSING
The artifact is truncated mid-string inside "field_principles". "acceptance_gates" and "completed_count" are present, but the remaining conclusion and measurement evidence are unavailable; per-unit rows cannot be declared absent.

## THE CHECK A READER CANNOT DO
Does the complete artifact make a comparative claim supported by per-unit measurements?

## experiment_8597_v738_medexpert_evidence_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The MedExpert evidence audit completed, but evaluation is blocked because required release rows were not acquired, with failed checks and observed values recorded in "gate_check_summary".

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8598_qwen_evidence_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run was blocked because upstream `runtime_repair_ready_score` was 0, failing the requirement that it equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8599_v738_arc_outcome_identifiability.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion reports all listed acceptance gates passed; the headline research claim cannot be identified.

## WHAT IS MISSING
The remainder of the artifact: "action_receipt_rows" cuts off mid-record at "index": 71. The visible "acceptance_gates" and action receipts do not establish an overall research claim or verdict.

## THE CHECK A READER CANNOT DO
Does the completed artifact claim a comparative improvement, and provide per-unit metrics supporting it?

## experiment_8600_v738_resident_hardware_boundary.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Resident Python removes cold-start overhead, with a reported throughput-ratio confidence interval of 11.46–11.70 versus cold Python.

## WHAT IS MISSING
The supplied artifact cuts off inside "historical_model_provenance". Per-repeat cold/resident throughput and latency measurements supporting "measured_resident_acceleration" are not visible; "conditional_ceiling_rows" records derived ceilings. The missing remainder could contain those measurements. "gate_check_summary" does diagnose the hardware blockers.

## THE CHECK A READER CANNOT DO
Did resident Python improve throughput consistently across the ten paired repeats, or did a few repeats drive the reported advantage?

## experiment_8601_v738_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact claims administrative completion and resident-cost qualification, while ARC and learning-benefit qualification remain false.

## WHAT IS MISSING
Per-unit cost measurements supporting `"qualified_resident_cost": true` are not visible; the artifact cuts off mid-`"document_sha256"` within `"blocked_inputs"`. Recorded blockers have diagnostics in fields including `"source_gates"` and `"cascade_log_line"`, so they do not establish BLOCKED_WITHOUT_DIAGNOSTIC.

## THE CHECK A READER CANNOT DO
Which measured units and cost values justify `"qualified_resident_cost": true`?
