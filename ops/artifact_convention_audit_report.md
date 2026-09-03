# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 4 |
| BLOCKED_WITHOUT_DIAGNOSTIC | 1 |
| CANNOT_DETERMINE | 3 |

## experiment_6919_exact_prefix_viability_fixture.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact is truncated before any headline claim or verdict is recorded.

## WHAT IS MISSING
The remainder of the artifact, including any headline/verdict and gate summary; only `"ambiguous_rows"` and a truncated `"branch_factor_rows"` are present.

## THE CHECK A READER CANNOT DO
Does the complete artifact claim a comparative win or blocked verdict, and do its recorded rows support that claim?

## experiment_6920_sota_exact_guided_relation_generation.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
The artifact is truncated inside `"candidate_rows"` after `"timed_out": false`; the remainder and any top-level verdict, claim, or `"gate_check_summary"` fields are unavailable, while `"arm_budget_rows"` and a partial `"candidate_rows"` are present.

## THE CHECK A READER CANNOT DO
Was a comparative or blocked verdict recorded later in the artifact, and if blocked, which check failed at what value?

## experiment_6921_arc_dynamic_supervisor_banked_credit.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no claim identifiable because the artifact is truncated mid-row

## WHAT IS MISSING
The complete artifact, including any headline/verdict and `gate_check_summary`; the provided text ends inside `dedupe_rows`, while present fields such as `actions_to_progress_rows`, `applied_receipt_rows`, `banked_level_transition_rows`, and `censored_rows` do not state a headline claim.

## THE CHECK A READER CANNOT DO
Does the missing remainder make a comparative claim or blocked verdict, and if so, does it provide the required per-unit metrics or blocker diagnostic?

## experiment_6922_v605_independent_capstone.json

**BLOCKED_WITHOUT_DIAGNOSTIC**

## VERDICT
BLOCKED_WITHOUT_DIAGNOSTIC

## WHAT THE CLAIM IS
V605 evidence synthesis is complete without promoting unsupported scientific claims.

## WHAT IS MISSING
For `exp6911-v605-document-yaml-evidence-contract`, the artifact records `"verdict_class": "blocked"` and `"evidence_state": "blocked"`, but its `"gate_outcomes"` is empty and no field identifies the failed check or observed value; the top-level `"gate_check_summary"` instead says all synthesis checks passed.

## THE CHECK A READER CANNOT DO
What specific check blocked Exp6911, and what observed value caused it to fail?

## experiment_6923_v606_lifecycle_evidence_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V606 lifecycle evidence contract completed in a blocked state because multiple contract checks failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6924_task_runtime_receipt_adoption.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The deterministic CPU fixture passed fresh validation, making task-runtime-receipt adoption ready without asserting a scientific result.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6925_v606_sota_ingestion.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V606 SOTA ingestion completed across 15 source families and 15 named candidates, yielding one new compatibility finding about ISM’s hosted-model dependencies.

## WHAT IS MISSING
nothing; `"gate_check_summary"` records expected and observed counts, while `"query_rows"`, `"candidate_rows"`, `"compatibility_rows"`, and `"ledger_append_rows"` provide the underlying per-source and per-candidate evidence.

## THE CHECK A READER CANNOT DO
none

## experiment_6926_span_first_relation_fixture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The fixture-readiness gate passed, including exact expected ASP effects and parity between the primary and independent solvers across the recorded fixtures.

## WHAT IS MISSING
nothing; `"fixture_rows"` and `"asp_effect_rows"` provide per-fixture evidence, while `"gate_check_summary"` records each check’s `"expected"`, `"observed"`, and `"passed"` values.

## THE CHECK A READER CANNOT DO
none
