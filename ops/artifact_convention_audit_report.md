# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 6 |
| AGGREGATE_ONLY | 1 |
| CANNOT_DETERMINE | 1 |

## experiment_6857_dynamic_live_arc_receipt_router.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The receipt router completed successfully, all declared checks passed, and it made no solve or comparative-effect claim.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6858_supervisor_counterfactual_credit_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the upstream `supervisor_headroom_ready_score` was 0 but the gate required it to equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6859_first_party_tool_gap_receipt_wiring.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The first-party tool-gap receipt contract is complete and ready, while no live effect or solve is claimed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6860_v599_independent_capstone.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The artifact claims positive held-future effects and identifies `unsloth/gemma-4-31B-it-GGUF` as a positive-margin model, while concluding that no scientific branch advanced.

## WHAT IS MISSING
Per-unit compatibility-margin and held-future treatment/control outcome rows are missing; `"rows"` contains only task-state, disposition, aggregate metric, consistency, and gate rows, while `"model_metrics"` and `"held_future_effect"` provide aggregates. The blocked branches are adequately diagnosed in `"gate_check_summary"` with `"failed_checks"` and `"observed"` values.

## THE CHECK A READER CANNOT DO
Was the model’s positive mean margin broad across conditions, or driven by a few outliers despite its 26 wins and 30 losses?

## experiment_6861_v600_branch_retirement_evidence_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V600 branch-retirement evidence contract is ready, with all four recorded readiness checks passing.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6862_dual_side_semantic_contrast_bank.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The 100-group dual-side semantic contrast bank passed every readiness gate and is ready for Exp6863.

## WHAT IS MISSING
The artifact is truncated mid-record; `"field_principles"` names `"rows"`, `"semantic_contrast_group_manifest"`, `"semantic_mutation_rows"`, `"solution_side_check_rows"`, and `"structure_side_check_rows"`, but their actual values are not visible, while only aggregate results appear in `"accepted_contrast_group_count"` and `"gate_check_summary"`.

## THE CHECK A READER CANNOT DO
Did every one of the 100 accepted groups independently pass both authority checks and the semantic-mutation check, rather than the readiness verdict relying only on aggregate summaries?

## experiment_6863_tokenizer_aware_semantic_contrast_preregistration.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The preregistration was blocked because both Gemma tokenizer hashes drifted and only the Qwen tokenizer satisfied the native-tokenizer requirement; no scientific effect was claimed.

## WHAT IS MISSING
nothing—the `"gate_check_summary"` records `"failed_checks"` with each check’s `"expected"` and `"observed"` values, while `"scientific_effect_claimed": false`.

## THE CHECK A READER CANNOT DO
none

## experiment_6864_three_family_semantic_contrast_scoring_stream.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because the upstream `semantic_contrast_preregistration_ready_score` was 0 instead of the required 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
