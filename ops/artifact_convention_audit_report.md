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

## experiment_8318_v718_contract_replay.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The headline claim cannot be identified from the supplied, truncated artifact.

## WHAT IS MISSING
The remainder of the artifact, including its final verdict and any "gate_check_summary"; the text ends mid-"snapshot_sha256" inside "cited_upstream_artifacts". The visible "acceptance_gates" are false, but "adversarial_findings" records specific failures, so missing blocker diagnostics cannot be inferred.

## THE CHECK A READER CANNOT DO
Does the artifact’s final verdict make a comparative claim or report a blocker without a diagnostic?

## experiment_8319_local_evidence_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 8319 was blocked because upstream "history_reader_ready_score" was 0, failing the required equality check against 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8320_sentence_spline_fit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream `cached_support_ready_score` was 0 instead of the required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8326_runtime_reader_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Qualification was blocked because upstream `history_reader_ready_score` was 0, while the gate required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8328_v718_arc_supervisor_frontier.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
The artifact ends mid-entry in "coverage_statement_counts"; the remainder containing any final verdict, headline claim, or blocker diagnostic is unavailable. "acceptance_gates.owned_checks" is false and "arm_support_rows" is empty, but neither establishes a comparative claim or blocked verdict.

## THE CHECK A READER CANNOT DO
Does the complete artifact report a comparative result, a block with a recorded reason, or neither?

## experiment_8329_v718_kv260_workload_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Accelerator benefit remains unproved because no compatible operation is available, and CPU cost qualification is blocked by three missing upstream artifacts recorded in "gate_check_summary" with "expected": true and "observed": false.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8330_v718_gatemate_change_ledger.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
GateMate reopening remains blocked pending documented physical change, with historical IDCODE `0xffffffff` and no current hardware execution.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8331_v718_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
“H1.status” and “H2.status” report “blocked_unmeasured”; no comparative result is visible.

## WHAT IS MISSING
The artifact’s remainder: it stops mid-array inside “arc_support.coverage_statement_counts”. “acceptance_gates” already records “independent_science”: false and “owned_validation”: false, so blocker diagnostics are not wholly absent.

## THE CHECK A READER CANNOT DO
Does the omitted remainder assert a comparative result without per-unit supporting rows?
