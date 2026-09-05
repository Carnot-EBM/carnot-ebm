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

## experiment_6968_arc_post_refit_induction_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The induction audit was blocked because the immutable transition source check failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7009_v614_source_contract_preflight.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V614 preflight was disqualified because the Markdown/YAML task contracts mismatch: `"expected_task_count"` is 14 while `"observed_task_count"` is 7.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7010_arc_eval_provenance_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The ARC evaluation provenance contract is ready because all recorded producer, consumer, rejection, round-trip, and row-level gates passed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7011_v614_sota_ingestion.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The selected sources were completely ingested and mapped, and no relevant primary or first-party artifact change was proved after the V614 marker.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_7012_exact_intervention_pair_fixture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The clean and changed intervention pairs were correctly distinguished as equivalent and non-equivalent, with terminal agreement and exact counterexamples.

## WHAT IS MISSING
nothing; per-unit evidence appears in `"authority_witness_rows"`, keyed by `"block_id"`, with `"clean_label"`, `"changed_label"`, `"passed"`, and `"exact_counterexamples"`.

## THE CHECK A READER CANNOT DO
none

## experiment_7013_three_family_intervention_surface.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no claim is visible in the supplied, truncated artifact

## WHAT IS MISSING
The artifact ends mid-`condition_rows` entry, so the complete top-level verdict/headline and any `gate_check_summary` or blocked-status diagnostic cannot be found; per-unit fields such as `pair_id`, `scientific_condition`, and `normalized_sequence_log_likelihood` are present.

## THE CHECK A READER CANNOT DO
Does the complete artifact make a comparative or blocked headline claim, and if so, do the recorded rows or diagnostics support it?

## experiment_7014_causal_feature_cold_audit.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The causal audit completed, but the feature bank is not ready because none of the three model families showed identifiable positive causal directions.

## WHAT IS MISSING
The per-unit signed metrics needed to support the aggregate `"mean"` and confidence intervals in `"family_identifiability_rows"`—specifically the actual `"per_pair_results"` or `"rows"` referenced only inside `"field_principles"` but not recorded in the artifact.

## THE CHECK A READER CANNOT DO
Were the non-identifiable family results broad across all 24 blocks, or driven by a few extreme or direction-reversing pairs?

## experiment_7015_pair_centered_pwa_kan.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `causal_feature_bank_ready_score` was 0 but was required to equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
