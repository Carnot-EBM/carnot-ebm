# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 8 |

## experiment_6926_span_first_relation_fixture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The span-relation fixture qualification gate passed with all expected per-fixture effects, solver parity checks, and coverage checks satisfied.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6927_v607_literature_delta.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V607 literature-delta review completed all 15 source-family checks and 10 named-candidate checks, producing one verified metadata and code-state correction.

## WHAT IS MISSING
nothing; the aggregate `gate_check_summary` is supported by per-unit `query_rows`, `candidate_rows`, and the correction in `ledger_append_rows`.

## THE CHECK A READER CANNOT DO
none

## experiment_6928_sota_runtime_receipt_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims that all three specified models produced qualified, task-owned local dual-CUDA runtime receipts with sequential lifecycles and successful fresh-process verification.

## WHAT IS MISSING
nothing; per-model evidence appears in `"model_rows"`, phase-level evidence in `"rows"`, and independent verification in `"fresh_process_recheck_rows"`.

## THE CHECK A READER CANNOT DO
none

## experiment_6941_v608_source_delta.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The bounded source audit completed all gates and found no material post-marker facts requiring ledger additions.

## WHAT IS MISSING
nothing; `"gate_check_summary"` records expected and observed counts with `"failed_check": null`, while `"query_rows"`, `"candidate_rows"`, and `"accepted_finding_rows"` provide unit-level outcomes.

## THE CHECK A READER CANNOT DO
none

## experiment_6942_v608_contract_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V608 contract preflight was blocked because `"bounded_scopes"` passed only 10 rows instead of all rows, and two lint commands exited with code 1.

## WHAT IS MISSING
nothing; `"gate_check_summary"` records `"failed_check"`, `"expected"`, and `"observed"`, while `"lint_command_rows"` provides command-level exit codes and output.

## THE CHECK A READER CANNOT DO
none

## experiment_6943_verifier_density_prefix_corpus.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `v608_execution_contract_ready_score` was 0 instead of the required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6948_arc_branch_corpus.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `exp6942-v608-contract-preflight.v608_execution_contract_ready_score` was `0`, while the gate required it to equal `1`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6952_v608_capstone.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V608 capstone contract audit completed, but the science remained incomplete because the execution-contract preflight failed and downstream tasks were blocked.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
