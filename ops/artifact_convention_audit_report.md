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

## experiment_8605_causal_brier_stream.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 8605 was blocked because both prerequisite gates observed readiness scores of 0 when 1 was required.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8608_v739_lean_resident_prototype.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible excerpt reports passed prototype acceptance gates, but the artifact ends before its overall claim or verdict appears.

## WHAT IS MISSING
The remainder of the artifact: it cuts off mid-entry in "cited_upstream_artifacts". "accepted_benchmark_claim" is false and "acceptance_gates" records passing checks, but the overall verdict and any subsequent experimental results are unavailable.

## THE CHECK A READER CANNOT DO
Does the artifact’s overall verdict assert a comparative result, report a blocker, or merely acknowledge prototype readiness?

## experiment_8609_v739_lean_resident_qualification.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible excerpt reports passing validation commands, but no experiment headline claim is visible.

## WHAT IS MISSING
The complete artifact: it ends mid-"path" within "acceptance_gates" → "historical_v738_replay" → "source_files". The visible "passed", "actual_exit", and "expected_exit" fields describe validation receipts; the experiment’s final verdict is unavailable.

## THE CHECK A READER CANNOT DO
Does the experiment’s final verdict assert a comparative result or report a blocker, and what recorded evidence supports it?

## experiment_8610_lean_resident_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `lean_measurement_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8611_qwen_evidence_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `runtime_repair_ready_score` was 0 while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8612_arc_outcome_observer.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `exp8602-contract-methods.methods_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8613_v739_hardware_operation_boundary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Hardware acceleration is not established, and acquisition is not recommended pending compatible device execution and complete cost evidence.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8614_v739_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact reports administrative completion while research qualification remains blocked, with failed checks and observed values documented.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
