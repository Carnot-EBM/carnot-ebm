# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 3 |
| AGGREGATE_ONLY | 1 |
| CANNOT_DETERMINE | 4 |

## experiment_6866_canonical_tokenizer_binding_requalification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The correction was wrapper-schema-only because the archived and live canonical payload SHA-256 hashes match for each of the three model rows.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6867_tokenizer_aware_semantic_preregistration_v2.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
The artifact is truncated mid-entry after `"score_identity": "sha256:27cb43b`; the complete top-level record, including any verdict, comparative summary, or blocker fields, is missing, while `"accepted_cell_manifest"` is present.

## THE CHECK A READER CANNOT DO
A reader cannot determine whether the omitted portion contains a comparative claim or a blocked verdict with—or without—a diagnostic.

## experiment_6868_three_family_semantic_scoring_stream_v2.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible fragment claims `"checkpoint_manifest.complete": true`, but contains no ascertainable experiment headline claim.

## WHAT IS MISSING
The artifact is truncated inside `"checkpoint_manifest.expected_identities"` and lacks the remainder containing any headline/verdict, gate diagnostics, and per-unit metric rows; only metadata such as `"expected_cell_count"`, `"completed_models"`, and `"calibration_score_manifest.row_count"` is visible.

## THE CHECK A READER CANNOT DO
Does the complete artifact make a comparative or blocked claim, and if so, does it include the per-unit results or failed-check diagnostic needed to verify it?

## experiment_6869_calibration_only_paired_semantic_rule.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The pooled semantic contrast improved by 0.10284, with a 95% BCa bootstrap interval of 0.07928 to 0.12502.

## WHAT IS MISSING
Per-`semantic_group_identity` metric or contrast rows for each arm/model; only aggregate `"bootstrap_rows"` with `"estimate"`, `"lower_bound"`, `"upper_bound"`, `"n_clusters"`, and `"n_groups"` are present.

## THE CHECK A READER CANNOT DO
Was the positive pooled effect broadly shared across semantic groups, or driven by a few outliers or groups with degenerate controls or no headroom?

## experiment_6870_sealed_independent_semantic_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The semantic compatibility audit was blocked because `semantic_contrast_rule_ready_score` was 0 instead of the required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6871_observable_reliability_opportunity_stream.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6872_bounded_reliability_controller_quarantine.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible fragment states no headline claim, but the artifact is truncated before any final claim or verdict can be determined.

## WHAT IS MISSING
The complete JSON remainder, including any `"verdict"` and `"gate_check_summary"` fields; only `"abstention_rate_by_arm"`, `"action_distribution_by_arm"`, `"action_entropy_by_arm"`, and an incomplete `"admitted_update_rows"` array are visible.

## THE CHECK A READER CANNOT DO
Does the complete artifact declare a comparative result or blocked gate, and if blocked, identify the failed check and observed value?

## experiment_6873_prospective_sealed_self_learning_audit.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no claim is visible because the artifact ends mid-record before any headline verdict or conclusion.

## WHAT IS MISSING
The complete artifact, including any verdict or claim fields; the supplied JSON truncates inside `"delayed_correction_rows"` at `"tombstone_present"`. Although aggregate fields such as `"action_distribution_by_arm"`, `"action_entropy_by_arm"`, and `"admitted_useful_updates_by_arm"` are present, it is impossible to determine whether they support a later comparative claim.

## THE CHECK A READER CANNOT DO
Does the artifact ultimately claim that an arm beat another arm, met a gate, or was blocked—and, if so, does it provide the required per-unit evidence or blocker diagnostic?
