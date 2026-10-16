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

## experiment_8569_causal_brier_stream.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Training was blocked because both upstream readiness checks returned 0 against a required value of 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8571_v736_resident_inference_prototype.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible text reports a historical timing null, but the artifact’s headline verdict is not shown.

## WHAT IS MISSING
The artifact cuts off inside "historical_cost_null.operation_rows". Actual values for "honest_verdict", "verdict_class", "prototype_ready_score", and "required_checks_passed" are absent; these names appear only in "field_principles".

## THE CHECK A READER CANNOT DO
Does the artifact’s final verdict report successful prototype qualification or a blocker?

## experiment_8572_v736_resident_inference_qualification.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible fragment claims clock partitioning passed, but the overall headline claim cannot be identified.

## WHAT IS MISSING
The artifact is truncated mid-entry in "clock_partition_receipt.sessions[].intervals". "acceptance_gates" is present, but the overall verdict and any later comparative results or blocker diagnostics are unavailable.

## THE CHECK A READER CANNOT DO
Does the complete artifact provide evidence sufficient to check its headline verdict?

## experiment_8573_resident_complete_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 8573 was blocked because upstream `methods_ready_score` was 0, failing the required value of 1, as recorded in `gate_check_summary` and `gates_evaluated`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8574_qwen_evidence_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The canary was blocked because `methods_ready_score` and `runtime_repair_ready_score` each equaled 0 while their gates required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8575_arc_publication_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Execution was blocked because upstream `exp8565-contract-methods.methods_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8577_v736_resident_hardware_boundary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Hardware acquisition remains unauthorized because resident CPU benefit is unmeasured, with failed qualification checks documented in "gate_check_summary".

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8578_v736_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible excerpt reports administrative completion with several acceptance gates false, but the headline verdict is not shown.

## WHAT IS MISSING
The artifact is truncated mid-string inside "historical_model_provenance". The remaining verdict and any blocker diagnostics are unavailable; the visible "blocked_inputs" entry has "checks": [], while "downstream_consumer.gates" records a passing check.

## THE CHECK A READER CANNOT DO
Does the complete artifact explain which failed check caused the "external_blocked_producer" disposition?
