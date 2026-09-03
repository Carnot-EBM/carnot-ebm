# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 3 |
| BLOCKED_WITHOUT_DIAGNOSTIC | 1 |
| CANNOT_DETERMINE | 4 |

## experiment_6913_relation_source_tuple_qualification.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact’s headline claim cannot be identified because the supplied JSON ends mid-record.

## WHAT IS MISSING
The actual `"rows"`, `"gate_check_summary"`, `"source_tuple_shard_ready_score"`, `"verdict_class"`, and `"honest_verdict"` values are missing; only their descriptions in `"field_principles"` are visible, alongside part of `"preconditions_checked"`.

## THE CHECK A READER CANNOT DO
Did the recorded per-cell results support the final qualification verdict and every reported gate decision?

## experiment_6914_relation_asp_isomorphic_qualification.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
All five evaluated arms were disqualified based on their qualification metrics.

## WHAT IS MISSING
The artifact is truncated mid-entry, so it is impossible to determine whether complete per-cell rows exist for every arm; `"arm_summary_rows"` contains aggregate `"qualification_decision"` values, while the visible `"asp_compilation_rows"` cover only part of one arm.

## THE CHECK A READER CANNOT DO
Do the per-cell outcomes for every arm reproduce each reported disqualification, or are some decisions supported only by aggregate summaries?

## experiment_6915_qualified_relation_event_bank.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The headline claim cannot be determined because the supplied artifact is truncated mid-`eligibility_rows`.

## WHAT IS MISSING
The complete artifact tail, including any headline verdict/claim and `gate_check_summary`; only `admitted_event_rows` and a partial `eligibility_rows` are present.

## THE CHECK A READER CANNOT DO
Does the complete artifact make a comparative claim or report a blocked verdict with a diagnostic?

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
Cannot determine because the artifact is truncated mid-`candidate_rows` entry.

## WHAT IS MISSING
The remainder of `"candidate_rows"` and any subsequent verdict, comparative-summary, or gate-diagnostic fields; only `"arm_budget_rows"` and an incomplete `"candidate_rows"` are visible.

## THE CHECK A READER CANNOT DO
Does the complete artifact claim that one arm beat another or that execution was blocked, and does it contain the per-cell metrics or blocker diagnostic needed to verify that claim?

## experiment_6921_arc_dynamic_supervisor_banked_credit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The audit completed and found no eligible banked credit (`"arc_supervisor_audit_complete_score": 1`, `"banked_credit_eligible_score": 0`).

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6922_v605_independent_capstone.json

**BLOCKED_WITHOUT_DIAGNOSTIC**

## VERDICT
BLOCKED_WITHOUT_DIAGNOSTIC

## WHAT THE CLAIM IS
V605 evidence synthesis is complete without promoting unsupported scientific claims.

## WHAT IS MISSING
For `exp6911-v605-document-yaml-evidence-contract`, the failed check and its observed value are missing: `"verdict_class": "blocked"` and `"evidence_state": "blocked"` are present, but `"gate_outcomes": []`; the artifact-level `"gate_check_summary"` only reports that all synthesis checks passed.

## THE CHECK A READER CANNOT DO
What specific check blocked exp6911, and what value did that check observe?
