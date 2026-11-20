# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 2 |
| CANNOT_DETERMINE | 6 |

## experiment_9208_v784_contract.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact's headline claim cannot be read because the JSON is truncated before `honest_verdict` and `verdict_class`, though the visible fields (`experiment_id`: 9208, `contract_ready_score`: 0, `claim_scope`) indicate an administrative contract-binding experiment with a zero contract-readiness score.

## WHAT IS MISSING
The JSON payload cuts off mid-string inside `full_task_objects[1].prompt` (ending abruptly without closing quotes, brackets, or braces), omitting every top-level field alphabetically after `full_task_objects`. Based on the schema keys listed in `field_principles`, the missing fields needed to audit the artifact are `honest_verdict`, `verdict_class`, `gate_check_summary`, `rows`, `required_checks_passed`, `validation_receipts`, `preconditions_checked`, `owned_statement_coverage`, `source_ready_score`, `static_methods_ready_score`, and `online_methods_ready_score` (only earlier fields—including `MODEL_SPECS`, `acceptance_gates`, `activation_authorized`, `attempted_count`, `claim_scope`, `completed_count`, `contract_ready_score`, `diagnostic_protocol`, `entropy_reference_protocol`, `experiment_id`, `failed_count`, `field_principles`, and the beginning of `full_task_objects`—are present).

## THE CHECK A READER CANNOT DO
Why is `contract_ready_score` equal to `0` despite `completed_count` being `14` and `failed_count` being `0`—what verdict (`honest_verdict` / `verdict_class`) did Exp9208 actually emit, and do `gate_check_summary` and `rows` record the specific failing check, observed value, and per-unit outcomes?

## experiment_9209_corpus_prefilter.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Exp9209 was blocked at the `conductor_pre_gate` layer because both upstream gate checks on `exp9208-contract` (`contract_ready_score` and `replay_authority_ready_score`) failed with observed value `0` against expected value `1`.

## WHAT IS MISSING
nothing — all required blocker diagnostics are present across `status`, `honest_verdict`, `blocked_at_layer`, `blocked_reason`, `failed_upstream`, `failed_field`, `failed_operator`, `failed_expected`, `failed_observed`, `failed_evidence_path`, `failed_evidence_sha256`, `blocked_diagnostic_contract`, `gate_check_summary`, and per-gate entries in `gates_evaluated`.

## THE CHECK A READER CANNOT DO
none

## experiment_9211_v784_label_ambiguity.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Based on `claim_scope` and `acceptance_gates`, the artifact intends to establish authenticated custody of 900 LPHB responses, a v2 taxonomy target-identifiability audit reporting zero identifiable SI/CF rows, an exact affine Brier label-ambiguity endpoint reducer with 900-response masks, and passing validation checks, but its actual headline verdict is cut off by truncation.

## WHAT IS MISSING
The artifact JSON is truncated mid-object inside `gate_check_summary` (right after `"check": "upstream_gate",` on entry 20), cutting off the remainder of `gate_check_summary` and every subsequent top-level field listed in `field_principles`: `honest_verdict`, `verdict_class`, `required_checks_passed`, `label_ambiguity_ready_score`, `target_identifiability_ready_score`, `rows`, `target_identifiability_rows`, `target_identifiability_summary`, `target_identifiability_controls`, `owned_statement_coverage`, `validation_receipts`, `scientific_effect_measured`, `support_counts`, and `source_references`. Only `MODEL_SPECS`, `acceptance_gates`, `ambiguity_masks_reference`, `attempted_count`, `censor_reason`, `censored_count`, `claim_scope`, `code_config_hashes`, `completed_count`, `current_llm_invocations`, `current_work_receipts`, `duration_s`, `effective_independent_groups`, `endpoint_control_rows`, `excluded_count`, `experiment_id`, `exposure_scope`, `failed_count`, `field_principles`, `flagged_adversarial`, and the first 19 complete entries of `gate_check_summary` are present. (Additionally, in the `empirical_TEST100_mask_fixture` entry of `endpoint_control_rows`, `probabilities` contains only 3 of the 300 units listed in `min_assignment` and `max_assignment`, so its `a1_contrast`, `exact_min`, and `exact_max` cannot be recomputed from the artifact alone.)

## THE CHECK A READER CANNOT DO
Did this experiment pass its gates or get blocked, and do the truncated `gate_check_summary`, `rows`, `target_identifiability_rows`, and `validation_receipts` fields actually contain per-unit evidence (or a failure diagnostic if blocked) rather than aggregate-only claims or an unexplained block?

## experiment_9221_v784_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible fragment reports that Phase-1 FoVer gates `G1`–`G4` all passed and `arc_summary` achieved 124/124 operator-list equality with mismatches reduced from 22 to 0 while `activation_authorized` and `arc_summary.required_checks_passed` remain `false`, but the artifact's overall headline claim and top-level verdict are cut off by truncation.

## WHAT IS MISSING
The JSON artifact is truncated mid-string inside the 22nd element of `arc_summary.rows` (ending at `"dagger_off_path_data_collection`), cutting off both the rest of the per-unit table and every top-level key alphabetically after `arc_summary`. Specifically, while `G1`, `G2`, `G3`, `G4`, `MODEL_SPECS`, `acceptance_gates`, `activation_authorized`, and partial `arc_summary` (`completed`, `denominator`, `equality_numerator`, `regression_result`, `required_checks_passed`, `consumer_reachability`, and `rows` entries `routing-000` through `routing-020`) are present, a reader is missing:
1. The remaining 103 rows of `arc_summary.rows` (`routing-021` through `routing-123`), which leaves all 72 `legacy_controls` rows and 18 of the 22 `before_mismatch: true` rows claimed in `arc_summary.regression_result` unobserved.
2. Any top-level `verdict`, `summary`, `status`, or `gate_check_summary` fields that sort alphabetically after `arc_summary`, so a reader cannot see the artifact's final verdict, whether it is declared blocked, or which of the nine `acceptance_gates` (`CES`, `ambiguity`, `contract`, `corpus`, `forwards`, `independence`, `learning`, `static`, `validation`) passed or failed to produce `activation_authorized: false`.
3. Any per-unit FoVer rows backing the comparative CI claims asserted in `G2` (`condition_A (production) AUROC in [0.9027, 0.9235]` and `learning_contribution in [0.0125, 0.0245]`), which are only summarized via external reference (`G1.source`: `experiment_2850_fover_dual_condition_integrity_v4.json`) in the visible prefix.

## THE CHECK A READER CANNOT DO
Whether the truncated remainder of the artifact actually contains a top-level verdict and diagnostics (`gate_check_summary`) explaining `activation_authorized: false`, and whether the missing 103 units of `arc_summary.rows` (including all 72 `legacy_controls`) truly reproduce `arc_summary.equality_numerator: 124` and `arc_summary.regression_result.after_mismatches: 0`.

## experiment_9222_v785_contract.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The headline claim cannot be read because the JSON artifact is truncated before its verdict fields, leaving only partial administrative fields showing `experiment_id` 9222 with `contract_ready_score` 0, `attempted_count` 5, `completed_count` 0, `failed_count` 5, `censored_count` 9, and three passing `consumer_shard_rows` under an administrative `claim_scope`.

## WHAT IS MISSING
The artifact is truncated mid-string inside `full_task_objects` (at the second task's `prompt`, right after `- validation_receipts: commands, log`), cutting off every top-level key alphabetically after `full_task_objects`. Looking at the fields that ARE present—`experiment_id`, `claim_scope`, `acceptance_gates`, `contract_ready_score`, `attempted_count`, `completed_count`, `failed_count`, `censored_count`, `consumer_shard_rows`, `field_principles`, and the beginning of `full_task_objects`—`field_principles` confirms the full schema includes the following missing fields that a reader needs to audit the artifact: `honest_verdict`, `verdict_class`, `gate_check_summary`, `rows`, `required_checks_passed`, `validation_receipts`, `preconditions_checked`, `protocol_owner_rows`, `replay_authority_ready_score`, `source_ready_score`, `static_methods_ready_score`, and `online_methods_ready_score`.

## THE CHECK A READER CANNOT DO
Given that `contract_ready_score` is 0 with `failed_count` 5 and `censored_count` 9 even though all three `consumer_shard_rows` have `passed: true`, what verdict did Exp9222 actually declare, and did it record which specific check(s) or unit(s) failed in `gate_check_summary` and `rows`?

## experiment_9223_corpus_prefilter.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 9223 was blocked at the `conductor_pre_gate` layer because 1 of 2 upstream pre-gate checks failed (`exp9222-contract.contract_ready_score` was `0` instead of the expected `1`).

## WHAT IS MISSING
nothing (inspected `status`, `honest_verdict`, `blocked_at_layer`, `blocked_reason`, `failed_upstream`, `failed_field`, `failed_operator`, `failed_expected`, `failed_observed`, `failed_evidence_path`, `failed_evidence_sha256`, `blocked_diagnostic_contract`, `gate_check_summary`, and per-gate entries in `gates_evaluated`, all of which record the exact failing check, expected vs. observed values, and upstream artifact hash).

## THE CHECK A READER CANNOT DO
none

## experiment_9225_v785_selective_deferral_protocol.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact's headline verdict and claim cannot be determined because the JSON is truncated mid-object before `honest_verdict` and `verdict_class`, though the visible `claim_scope` and `acceptance_gates` restrict its scope to source-custody receipts, frozen 10-stream selective-deferral/IPW controls, and label-free 57-column cubic B-spline support-density diagnostics.

## WHAT IS MISSING
The artifact is truncated mid-string inside `gate_check_summary` (cutting off at `"expected": "sha256:96c654f6ec04ecf532a8b1947f71bda819cf`). While `MODEL_SPECS`, `acceptance_gates`, `attempted_count`, `censor_reason`, `censored_count`, `claim_scope`, `classifier_reference`, `code_config_hashes`, `completed_count`, `current_llm_invocations`, `current_work_receipts`, `duration_s`, `effective_independent_groups`, `estimator_control_rows`, `excluded_count`, `experiment_id`, `exposure_scope`, `failed_count`, `field_principles`, and `flagged_adversarial` are present, the truncation removes the remainder of `gate_check_summary` and every subsequent field named in `field_principles`, including `honest_verdict`, `verdict_class`, `required_checks_passed`, `rows`, `spline_control_rows`, `selective_deferral_ready_score`, `spline_support_diagnostic_ready_score`, `owned_statement_coverage`, `support_counts`, and `validation_receipts`.

## THE CHECK A READER CANNOT DO
Did this run actually pass all gates or end up blocked (and if blocked, with or without a diagnostic), and do `rows` and `spline_control_rows` actually contain per-unit data verifying the 10-stream (50 qids / 150 responses) selective-deferral masks and 57-column cubic B-spline support-density controls?

## experiment_9240_v786_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact's top-level headline claim and verdict are truncated away, though the visible prefix reports passing FoVer reproducibility gates (`G1`–`G4`, including `condition_A` AUROC in `[0.9027, 0.9235]` and `learning_contribution` in `[0.0125, 0.0245]`), `activation_authorized: false`, and an ARC routing regression summary (`arc_summary`) claiming `124/124` operator-list equality (`before_mismatches: 22` reduced to `after_mismatches: 0`) with `required_checks_passed: false`.

## WHAT IS MISSING
The JSON is truncated mid-string inside `arc_summary.rows[20].reference` (`"online_warm_action_effect_controller_`), cutting off: (1) the remaining 104 of 124 per-unit rows in `arc_summary.rows` (only `routing-000` through `routing-019` are complete, showing 3 complete `before_mismatch: true` rows plus 1 partial row out of `regression_result.before_mismatches: 22`, and 0 of `regression_result.legacy_controls: 72`); (2) every top-level key alphabetically after `arc_summary`, including any top-level `verdict`, `status`, `summary`, or `gate_check_summary` explaining why `activation_authorized` and `arc_summary.required_checks_passed` are `false`; (3) any pass/fail results or per-unit rows for the 12 gate definitions listed in `acceptance_gates` (`CES`, `ambiguity`, `conformal`, `contract`, `corpus`, `forwards`, `independence`, `learning`, `selective_deferral`, `spline_support`, `static`, `validation`); and (4) any in-artifact per-unit data backing the aggregate AUROC / `learning_contribution` intervals cited in `G1`, `G2`, and `G4` (which only point to external `experiment_2850_fover_dual_condition_integrity_v4.json`).

## THE CHECK A READER CANNOT DO
A reader cannot determine what the artifact's actual top-level verdict is, whether `arc_summary`'s claimed `124/124` equality and `22 -> 0` mismatch reduction actually hold across the missing 104 rows (including all 72 `legacy_controls`), or whether a `gate_check_summary` later in the truncated JSON explains why `activation_authorized` and `arc_summary.required_checks_passed` are `false`.
