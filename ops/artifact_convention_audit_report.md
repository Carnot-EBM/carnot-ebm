# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 3 |
| AGGREGATE_ONLY | 3 |
| CANNOT_DETERMINE | 2 |

## experiment_6868_three_family_semantic_scoring_stream_v2.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
no claim

## WHAT IS MISSING
The artifact is truncated mid-value after `"expected_identities"`; despite visible fields such as `"checkpoint_manifest.complete": true`, `"expected_cell_count": 880`, and `"calibration_score_manifest.row_count": 448`, the remaining top-level verdict, comparative metrics, per-unit rows, and any `"gate_check_summary"` cannot be inspected.

## THE CHECK A READER CANNOT DO
Does the omitted remainder make a comparative or blocked headline claim, and if so, does it include the necessary per-unit evidence or blocker diagnostic?

## experiment_6869_calibration_only_paired_semantic_rule.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The pooled comparison shows a positive effect of 0.10283877494589307 with a 95% bootstrap interval of 0.07928079080704799 to 0.12502085885913594.

## WHAT IS MISSING
Per-`semantic_group_identity` metric rows containing each arm’s score and paired effect; `"bootstrap_rows"` contains only aggregate estimates and intervals, while `"calibration_access_log"` records data access rather than unit-level outcomes.

## THE CHECK A READER CANNOT DO
Was the pooled improvement broad across semantic groups, or driven by a few outliers, degenerate controls, or units pinned at a floor or ceiling?

## experiment_6870_sealed_independent_semantic_audit.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because `semantic_contrast_rule_ready_score` was 0 when the gate required it to equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6871_observable_reliability_opportunity_stream.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
No headline claim is visible in the supplied, truncated artifact.

## WHAT IS MISSING
The complete artifact, including any headline/verdict and result rows; only `"action_manifest"`, `"bounded_update_contract"`, and part of `"chronological_order_manifest"` are present before the JSON cuts off.

## THE CHECK A READER CANNOT DO
Does the omitted portion make a comparative claim or report a blocked verdict, and if so, does it include per-unit metrics or a blocker diagnostic?

## experiment_6872_bounded_reliability_controller_quarantine.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The arms differ in abstention rate and action distribution, including lower abstention for `exact_quarantine` than `bounded_update`.

## WHAT IS MISSING
Complete per-unit action rows for every arm and condition; `"abstention_rate_by_arm"` and `"action_distribution_by_arm"` are aggregates, while `"admitted_update_rows"` includes only admitted writes and omits the abstained, read-only, and no-memory units needed to verify them.

## THE CHECK A READER CANNOT DO
Was `exact_quarantine`’s lower pooled abstention rate broad across models, seeds, and conditions, or driven by a small subset of units?

## experiment_6873_prospective_sealed_self_learning_audit.json

**AGGREGATE_ONLY**

## VERDICT
AGGREGATE_ONLY

## WHAT THE CLAIM IS
The arms differ in behavior and useful learning, with `"v599_unsafe_reference"` recording 1,360 admitted useful updates versus 260 for each update arm and zero for the controls.

## WHAT IS MISSING
Per-unit rows underlying `"admitted_useful_updates_by_arm"`, `"action_distribution_by_arm"`, and `"action_entropy_by_arm"` for every arm; `"delayed_correction_rows"` is present but contains correction checks rather than the comparative metrics and shows only `"frozen_no_memory"` rows.

## THE CHECK A READER CANNOT DO
Is the apparent advantage in admitted useful updates broad across units, or driven by a small number of outliers or degenerate conditions?

## experiment_6874_v602_evidence_substrate_manifest_contract.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The V602 evidence-substrate manifest contract is blocked because document/YAML parity and related gate-contract readiness checks failed.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_6875_text_anchored_relation_asp_fixture.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The experiment was blocked because upstream field `v602_evidence_contract_ready_score` was observed as 0 but required to equal 1.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none
