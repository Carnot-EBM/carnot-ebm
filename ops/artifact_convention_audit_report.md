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

## experiment_7983_reserved_decisions.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_7984_v692_evidence_ablation.json

**CANNOT_DETERMINE**

reviewer call failed

## experiment_7985_delayed_acquisition.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7987_issued_confidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked at the conductor pre-gate stage because upstream dependency exp7981-qwen-stream-capture failed two gate criteria.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7988_v692_arc_supervisor_delta.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7989_v692_service_cost.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact makes no comparative claim; it is a scaffolding receipt recording upstream branch readiness and dependency validation checks.

## WHAT IS MISSING
nothing. Evaluated present fields `acceptance_gate_results`, `branch_gate_check_summary`, `branch_readiness`, `acquisition_setup`, and `cited_upstream_artifacts`, which record complete per-check expected, observed, and pass/fail diagnostics.

## THE CHECK A READER CANNOT DO
none

## experiment_7990_v692_hardware_evidence.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Execution is blocked due to failed upstream custody or service checks (`honest_verdict`: "complete_blocked_required_custody_or_service"), with no hardware speedup claimed (`hardware_speedup_claimed`: false).

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7991_v692_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The capstone execution is blocked with zero readiness (`capstone_execution_ready_score`: 0) and failed acceptance gates due to multiple upstream check failures.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
