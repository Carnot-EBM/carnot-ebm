# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 7 |
| BLOCKED_WITHOUT_DIAGNOSTIC | 1 |

## experiment_6921_arc_dynamic_supervisor_banked_credit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The audit found no eligible banked credit (`"banked_credit_eligible_score": 0`) after reconciling redirect credit against banked replay.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6922_v605_independent_capstone.json

**BLOCKED_WITHOUT_DIAGNOSTIC**

## VERDICT
BLOCKED_WITHOUT_DIAGNOSTIC

## WHAT THE CLAIM IS
The V605 evidence synthesis is complete, with task exp6911 classified as blocked.

## WHAT IS MISSING
For exp6911, the failed check name and observed value are missing: `"declared_verdict_class": "blocked"` and `"evidence_state": "blocked"` are present, but `"gate_outcomes": []`; the overall `"gate_check_summary"` has `"failed_check": null`.

## THE CHECK A READER CANNOT DO
Which check blocked exp6911, and what observed value caused it to fail?

## experiment_6923_v606_lifecycle_evidence_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V606 lifecycle evidence contract audit completed but was blocked because the executable YAML contains only 4 of the expected 14 tasks and multiple related contracts failed.

## WHAT IS MISSING
nothing; `honest_verdict`, `gate_check_summary.failed_check`, and `gate_check_summary.failed_checks` record the failed checks plus their `expected` and `observed` values.

## THE CHECK A READER CANNOT DO
none

## experiment_6924_task_runtime_receipt_adoption.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The reusable task-runtime-receipt adoption path passed deterministic CPU validation and is ready, without making a comparative scientific claim.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6925_v606_sota_ingestion.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V606 SOTA ingestion completed its gate of 15 terminal source families and 15 terminal candidates, with one new ISM compatibility finding.

## WHAT IS MISSING
nothing; `"gate_check_summary"` records expected and observed values, while `"query_rows"`, `"candidate_rows"`, and `"ledger_append_rows"` provide the underlying unit-level records.

## THE CHECK A READER CANNOT DO
none

## experiment_6926_span_first_relation_fixture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The span-relation fixture passed all qualification gates, including exact ASP-effect parity across its per-fixture rows.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6927_v607_literature_delta.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V607 literature-delta review completed all gates and produced one verified metadata/code-state correction.

## WHAT IS MISSING
nothing; `"gate_check_summary"` records the expected and observed counts, while `"query_rows"`, `"candidate_rows"`, and `"ledger_append_rows"` provide the corresponding per-source, per-candidate, and correction evidence.

## THE CHECK A READER CANNOT DO
none

## experiment_6928_sota_runtime_receipt_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The runtime receipt qualified three specified models as successfully executed sequentially with task-owned dual-CUDA inference, complete teardown, and an accepted fresh-process recheck.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
