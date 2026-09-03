# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 7 |
| AGGREGATE_ONLY | 1 |

## experiment_6885_v603_executable_manifest_branch_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V603 executable-manifest branch contract is blocked because the executable YAML contains only 4 of the 13 documented tasks and multiple dependent contract checks fail.

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
nothing; `"claim_boundary"` explicitly records `"enoki_accuracy_claimed": false`, `"encoder_loaded": false`, and `"llm_inference_count": 0`, while `"asset_revision_rows"` provides per-asset checks.

## THE CHECK A READER CANNOT DO
none

## experiment_6887_three_family_relation_proposal_corpus.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing; the present `"abstention_rows"` records per-cell `"arm"`, `"cell_identity"`, `"fixture_id"`, `"status"`, and `"reason"` diagnostics.

## THE CHECK A READER CANNOT DO
none

## experiment_6888_independent_relation_qualification.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
`rule:anchored_lexical_v1` passed the qualification gate while all four model/encoder arms failed.

## WHAT IS MISSING
Per-unit span, tuple, parse-coverage, and perturbation metric rows underlying the aggregate `observed`, `qualified_event_count`, and `family_rows` values; only unrelated per-fixture `asp_compilation_rows` and `contradiction_rows` are present.

## THE CHECK A READER CANNOT DO
Did the passing arm’s reported span F1 and tuple precision reflect broad success across individual relation events, or were the aggregates driven by a small subset of units?

## experiment_6898_v604_evidence_admissibility_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V604 evidence-admissibility audit completed but was blocked because required sources and multiple manifest-contract checks failed.

## WHAT IS MISSING
nothing; `"gate_check_summary"` records `"failed_check": "preconditions"` and supplies expected and observed values, while `"hard_failures"` identifies every other failed check.

## THE CHECK A READER CANNOT DO
none

## experiment_6899_live_relation_acquisition_canary.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The live relation-acquisition canary completed successfully and met every readiness gate.

## WHAT IS MISSING
nothing; `"gate_check_summary"` records each check’s `"expected"`, `"observed"`, and `"passed"` values, while `"generated_token_rows"`, `"output_byte_rows"`, and `"parse_attempt_rows"` provide per-cell evidence.

## THE CHECK A READER CANNOT DO
none

## experiment_6900_authentic_anchored_relation_corpus.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
nothing; the artifact includes per-unit `"cell_manifest"` rows with `"cell_identity"`, `"seed"`, `"fixture_id"`, and `"parse_rows"`, and contains no blocked verdict or comparative claim.

## THE CHECK A READER CANNOT DO
none

## experiment_6901_independent_model_relation_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The task was gated because `unflagged_exp6900` and `fresh_adversarial_critical_count` failed.

## WHAT IS MISSING
nothing; `gate_check_summary.checks` and `adversarial_admission_rows` record each failed check’s `expected`, `observed`, and `passed` values.

## THE CHECK A READER CANNOT DO
none
