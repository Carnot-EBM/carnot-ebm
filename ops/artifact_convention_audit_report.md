# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 5 |
| AGGREGATE_ONLY | 1 |
| CANNOT_DETERMINE | 2 |

## experiment_6872_bounded_reliability_controller_quarantine.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact is truncated before any explicit headline claim or verdict can be identified.

## WHAT IS MISSING
The JSON ends mid-value inside `"admitted_update_rows"`; the remainder—including any `"verdict"`, `"gate_check_summary"`, and complete per-unit rows supporting `"abstention_rate_by_arm"` and `"action_distribution_by_arm"`—cannot be found.

## THE CHECK A READER CANNOT DO
Does the omitted portion contain complete per-unit evidence for the arm comparisons or a diagnostic for any blocked verdict?

## experiment_6873_prospective_sealed_self_learning_audit.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact is truncated before any explicit headline claim or verdict appears.

## WHAT IS MISSING
The remainder of the JSON, including any verdict or gate fields; visible fields include `"action_distribution_by_arm"`, `"action_entropy_by_arm"`, `"admitted_useful_updates_by_arm"`, and `"delayed_correction_rows"`.

## THE CHECK A READER CANNOT DO
Did the complete artifact claim a comparative gate was met or that the task was blocked, and what evidence supported that verdict?

## experiment_6874_v602_evidence_substrate_manifest_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact claims Exp6874 completed but was blocked because V602 document/YAML parity, gate-contract validation, and evidence-contract readiness checks failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6875_text_anchored_relation_asp_fixture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream field `v602_evidence_contract_ready_score` was 0 but had to equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6885_v603_executable_manifest_branch_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V603 executable-manifest branch contract is blocked because the document declares 13 tasks while the executable YAML contains only 4, causing multiple contract checks to fail.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6886_enoki_exact_relation_fixture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6887_three_family_relation_proposal_corpus.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing; the present `"abstention_rows"` record per-cell `"status"` and `"reason"` diagnostics.

## THE CHECK A READER CANNOT DO
none

## experiment_6888_independent_relation_qualification.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
`rule:anchored_lexical_v1` is the only eligible arm, passing every threshold while all four competing arms fail five thresholds.

## WHAT IS MISSING
Per-unit `rows`, `span_metric_rows`, `tuple_metric_rows`, `parse_coverage_rows`, and `perturbation_rows` underlying the aggregate `observed` values in `eligible_arm_rows`; only aggregates such as `family_rows` and `abstention_rows`, plus per-fixture `contradiction_rows` and `asp_compilation_rows`, are present.

## THE CHECK A READER CANNOT DO
Were the passing span F1, tuple precision/recall, coverage, and family-floor results broad across the 90 qualified relation events, or driven by a small subset of units, duplicates, or units with no headroom?
