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

## experiment_8318_v718_contract_replay.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion records failed acceptance gates and replay checks, but no headline verdict is visible.

## WHAT IS MISSING
The remainder of the artifact: it cuts off mid-value inside "cited_upstream_artifacts". "acceptance_gates" and "adversarial_findings" are present, but the final claim and any subsequent results or blocker summary are unavailable.

## THE CHECK A READER CANNOT DO
Does the complete artifact make a comparative claim without per-unit results?

## experiment_8319_local_evidence_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked at the conductor pre-gate because upstream gate check `history_reader_ready_score == 1` failed with an observed value of 0.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8320_sentence_spline_fit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was blocked at conductor pre-gate because upstream dependency `exp8318-contract-replay` had a `cached_support_ready_score` of 0 instead of the expected 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8326_runtime_reader_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked at the pre-gate check because upstream task exp8318-contract-replay recorded history_reader_ready_score=0 instead of the expected 1.

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
The remainder of the artifact: it ends mid-list inside "coverage_statement_counts". "acceptance_gates" records "owned_checks": false, but the artifact’s own final verdict is not visible.

## THE CHECK A READER CANNOT DO
Does the missing remainder declare a comparative result or a blocked verdict with a recorded diagnostic?

## experiment_8329_v718_kv260_workload_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
CPU cost qualification is blocked by three missing upstream artifacts, and accelerator benefit remains unproved because no compatible operation is available.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8330_v718_gatemate_change_ledger.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
GateMate remains blocked because a dated physical-change receipt is absent and the historical IDCODE is `0xffffffff`; no current hardware execution occurred.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_8331_v718_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
H1 and H2 are “blocked_unmeasured,” with “acceptance_gates” recording “independent_science” and “owned_validation” as false; no measured arm advantage is claimed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
