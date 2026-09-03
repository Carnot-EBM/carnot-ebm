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

## experiment_6913_relation_source_tuple_qualification.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The headline claim cannot be determined because the artifact is truncated before its verdict and results.

## WHAT IS MISSING
The actual `"rows"`, `"gate_check_summary"`, `"verdict_class"`, and `"honest_verdict"` values are missing; only their descriptions in `"field_principles"` and a partial `"receipt_cell_identity_set"` are present.

## THE CHECK A READER CANNOT DO
Did the qualification gates pass based on the recorded per-cell outcomes, rather than aggregates alone?

## experiment_6914_relation_asp_isomorphic_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
All five evaluated arms were disqualified by the qualification checks.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6915_qualified_relation_event_bank.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no claim identifiable because the artifact is truncated mid-record

## WHAT IS MISSING
The remainder of the artifact, including any verdict/headline and comparative metric or blocker summary; `"admitted_event_rows"` and `"eligibility_rows"` are present, but the JSON ends inside an `"eligibility_rows"` record.

## THE CHECK A READER CANNOT DO
Does the complete artifact claim a comparative result or blocked verdict, and does it provide the corresponding per-unit metrics or failure diagnostic?

## experiment_6916_isomorphic_prospective_relation_stream.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because both upstream gates failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6919_exact_prefix_viability_fixture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6920_sota_exact_guided_relation_generation.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact is truncated before any headline claim or verdict is recorded.

## WHAT IS MISSING
A complete artifact containing the headline/verdict and outcome metrics; `"arm_budget_rows"` and `"candidate_rows"` are present, but the JSON ends mid-row and no comparative result or blocker diagnosis is visible.

## THE CHECK A READER CANNOT DO
Did `"guided_frontier"` outperform `"unguided_best_of_k"` on the per-cell outcome metric?

## experiment_6921_arc_dynamic_supervisor_banked_credit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
No audited redirect qualifies for causal banked-progress credit, reflected by `"banked_credit_eligible_score": 0`.

## WHAT IS MISSING
nothing; `"actions_to_progress_rows"`, `"censored_rows"`, `"banked_level_transition_rows"`, and `"competing_redirect_rows"` provide row-level diagnostics.

## THE CHECK A READER CANNOT DO
none

## experiment_6922_v605_independent_capstone.json

**BLOCKED_WITHOUT_DIAGNOSTIC**

## VERDICT
BLOCKED_WITHOUT_DIAGNOSTIC

## WHAT THE CLAIM IS
V605 evidence synthesis is complete without promoting unsupported scientific claims.

## WHAT IS MISSING
For `exp6911-v605-document-yaml-evidence-contract`, `"verdict_class": "blocked"` is present, but `"gate_outcomes"` is empty and no failed check, expected value, or observed value explains the block; the artifact-level `"gate_check_summary"` only reports successful synthesis checks.

## THE CHECK A READER CANNOT DO
Which specific check blocked exp6911, and what value did that check observe?
