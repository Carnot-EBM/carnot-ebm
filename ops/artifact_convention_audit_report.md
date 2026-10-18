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

## experiment_8632_complete_source_comparison.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream "streaming_extractor_ready_score" was 0, while the gate required 1, as recorded in "gates_evaluated" and "gate_check_summary".

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8633_v741_installed_lean_prototype.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The installed prototype meets its readiness gate, including compatibility checks for 64 requests in each arithmetic arm.

## WHAT IS MISSING
The supplied artifact cuts off inside "historical_model_provenance". "installed_prototype_ready_score" is 1 and "completed_count" is 128, but "prototype_parity_rows" and "rows" appear only as descriptions in "field_principles"; their actual contents are not supplied.

## THE CHECK A READER CANNOT DO
Did each request in each arithmetic arm match its original-worker response?

## experiment_8634_v741_installed_compatibility.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion reports failed installed-compatibility gates, but the current invocation’s headline verdict is not shown.

## WHAT IS MISSING
The current invocation’s actual "honest_verdict" and "verdict_class" values: their names appear in "field_principles", but the artifact truncates inside "historical_model_provenance". "acceptance_gates" identifies failures, and "historical_8609.gate_check_summary" supplies a historical blocker diagnostic, so an unexplained blockage is not established.

## THE CHECK A READER CANNOT DO
Does the current invocation’s final claim require comparative evidence beyond the compatibility checks shown?

## experiment_8635_installed_complete_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `installed_measurement_ready_score` was 0, failing the required value of 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8636_qwen_evidence_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The run was blocked because `runtime_repair_ready_score` was 0 while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8637_v741_arc_observer_transfer.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible "acceptance_gates" entries all report success, but the headline research conclusion is not visible.

## WHAT IS MISSING
The remaining artifact: "actual_redirect_rows" cuts off inside the fourth row’s "sequence", preventing inspection of any subsequent claim, verdict, or comparison results.

## THE CHECK A READER CANNOT DO
Does the complete artifact supply per-unit evidence or a blocker diagnosis supporting its headline verdict?

## experiment_8638_v741_hardware_operation_delta.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
No new hardware operation cost boundary is admitted because required execution and cost-evidence gates failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8639_v741_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible “branch_outcomes” reports an ARC null and several blocked or disqualified branches, but the overall headline verdict is not shown.

## WHAT IS MISSING
The artifact’s remainder: it cuts off mid-“sha256” inside “continuation_decisions”, so any later verdict or per-unit measurements cannot be assessed. “blocked_inputs” already records diagnostic “checks” with “observed”, “expected”, and “passed” values.

## THE CHECK A READER CANNOT DO
Does the ARC “qualified_recording_null_no_policy_benefit” result have per-game comparative metrics supporting it?
