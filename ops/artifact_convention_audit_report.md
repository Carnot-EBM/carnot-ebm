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

## experiment_9207_v783_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible prefix of this truncated JSON reports that gates `G1`–`G4` pass (citing FoVer production AUROC in `[0.9027, 0.9235]` and `learning_contribution` in `[0.0125, 0.0245]`), records `activation_authorized: false` and `arc_summary.required_checks_passed: false`, and claims `arc_summary.regression_result` improved from `before_mismatches: 22` to `after_mismatches: 0` across `124` units, but cuts off before any top-level verdict or summary section appears.

## WHAT IS MISSING
Only top-level keys `G1`, `G2`, `G3`, `G4`, `MODEL_SPECS`, `acceptance_gates`, `activation_authorized`, and the beginning of `arc_summary` (`compatibility_matrix`, `completed`, `consumer_reachability`, `current_ARC_experiment_scheduled`, `denominator`, `equality_numerator`, `historical_experiment_id`, `live_path_observed`, `regression_result`, `required_checks_passed`, and the first 21 complete entries `routing-000` through `routing-020` of `arc_summary.rows`) are present before the file truncates mid-string (`"programmatic_expert_`) inside the 22nd row. Because keys are serialized in alphabetical order, the artifact is missing:
- The remaining 103 per-unit entries of `arc_summary.rows` needed to check `completed: 124`, `denominator: 124`, `equality_numerator: 124`, `before_mismatches: 22`, `after_mismatches: 0`, and `legacy_controls: 72`.
- All top-level keys after `arc_summary` (such as `verdict`, `status`, `summary`, or `gate_check_summary`), making it impossible to see the artifact's actual top-level verdict or whether a failure diagnostic exists for `activation_authorized: false` and `arc_summary.required_checks_passed: false`.
- Any per-unit FoVer evaluation rows (if present later in the file) supporting the aggregate `G1`/`G2` AUROC and `learning_contribution` CIs currently attributed only to external file `experiment_2850_fover_dual_condition_integrity_v4.json`.

## THE CHECK A READER CANNOT DO
Does the truncated remainder of the JSON contain the artifact's overall verdict, a diagnostic field (such as `gate_check_summary`) specifying which check caused `activation_authorized` and `arc_summary.required_checks_passed` to be `false`, and the remaining 103 entries of `arc_summary.rows` (and any per-unit FoVer rows) needed to verify the claimed aggregate counts and gate passes?

## experiment_9208_v784_contract.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact is truncated mid-JSON before its verdict appears, so its headline claim cannot be read, though visible fields (`experiment_id: 9208`, `contract_ready_score: 0`, `attempted_count: 14`, `completed_count: 14`, `failed_count: 0`, `claim_scope`) describe an administrative contract/replay-qualification run with `contract_ready_score: 0`.

## WHAT IS MISSING
The JSON payload is cut off mid-string inside `full_task_objects` (in the second task's `prompt` for `exp9209-corpus-prefilter`), omitting every alphabetically subsequent top-level field that `field_principles` states the artifact schema contains. While `MODEL_SPECS`, `acceptance_gates`, `activation_authorized`, `archive_lag`, `attempted_count`, `censored_count`, `claim_scope`, `code_config_hashes`, `completed_count`, `contract_ready_score`, `current_science_paper_ready`, `diagnostic_protocol`, `duration_s`, `effective_independent_groups`, `entropy_reference_protocol`, `excluded_count`, `experiment_id`, `failed_count`, `field_principles`, `flagged_adversarial`, and the beginning of `full_task_objects` are present, the artifact is missing `honest_verdict`, `verdict_class`, `gate_check_summary`, `rows`, `required_checks_passed`, `preconditions_checked`, `validation_receipts`, `previous_outcome_rows`, `source_ready_score`, `static_methods_ready_score`, `online_methods_ready_score`, and `replay_authority_ready_score` (required by `full_task_objects[0].prompt`).

## THE CHECK A READER CANNOT DO
A reader cannot determine whether Exp9208 was blocked, disqualified, or non-blocked—nor which specific gate, operand, or per-unit check caused `contract_ready_score: 0` despite `attempted_count: 14`, `completed_count: 14`, and `failed_count: 0`—because `honest_verdict`, `verdict_class`, `gate_check_summary`, `required_checks_passed`, and `rows` are truncated away.

## experiment_9209_corpus_prefilter.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 9209 was blocked at the `conductor_pre_gate` layer because 2 of 2 upstream gates on `exp9208-contract` (`contract_ready_score` and `replay_authority_ready_score`) failed with observed value `0` instead of expected `1`.

## WHAT IS MISSING
nothing — all required blocker diagnostics are present in `status`, `honest_verdict`, `blocked_reason`, `gate_check_summary`, `blocked_diagnostic_contract`, `gates_evaluated`, `failed_evidence_path`, `failed_evidence_sha256`, and `blocked_at_layer`.

## THE CHECK A READER CANNOT DO
none

## experiment_9211_v784_label_ambiguity.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Per `claim_scope.allowed_claims` and `acceptance_gates`, the artifact scopes itself to authenticated LPHB 900-response custody, a v2 taxonomy target-identifiability audit reporting zero identifiable SI/CF rows, and exact affine Brier label-ambiguity bounds with no LLM load or generalization claim, but its actual headline verdict is cut off by JSON truncation.

## WHAT IS MISSING
The artifact is truncated mid-object inside `gate_check_summary` (on the 20th entry, `exp9180` / `required_checks_passed`). While `MODEL_SPECS`, `acceptance_gates`, `ambiguity_masks_reference`, `claim_scope`, `completed_count`, `censor_reason`, `endpoint_control_rows`, and `field_principles` are present, every top-level field alphabetically after `gate_check_summary` that is listed in `field_principles` is missing from the text, including:
- `honest_verdict`, `verdict_class`, `required_checks_passed`, `label_ambiguity_ready_score`, `target_identifiability_ready_score`, and `scientific_effect_measured`
- the remaining entries of `gate_check_summary`, plus `preconditions_checked`, `validation_receipts`, `validation_scope_manifest`, and `owned_statement_coverage`
- `rows`, `target_identifiability_rows`, `target_identifiability_summary`, `target_identifiability_controls`, and `support_counts`

(Additionally, within the visible `endpoint_control_rows` item `empirical_TEST100_mask_fixture`, `probabilities` lists only 3 of the 300 units present in `min_assignment` and `max_assignment`, and the 900-response masks are only pointed to externally via `ambiguity_masks_reference`.)

## THE CHECK A READER CANNOT DO
Whether `honest_verdict` and `verdict_class` report a pass, a comparative claim, or a blocker—and whether the truncated remainder of `gate_check_summary`, `rows`, `target_identifiability_rows`, and `validation_receipts` actually substantiate that verdict with per-unit records or failing-check diagnostics.

## experiment_9221_v784_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible prefix reports passing FoVer reproducibility checks (`G1`–`G4`), `activation_authorized: false`, and an ARC routing regression fix (`arc_summary.equality_numerator: 124` of `denominator: 124`, reducing `before_mismatches: 22` to `after_mismatches: 0` while `required_checks_passed: false`), but the artifact is truncated mid-JSON before its overall headline claim or verdict appears.

## WHAT IS MISSING
The JSON is cut off mid-string inside the 22nd element of `arc_summary.rows` (`"dagger_off_path_data_collection`), leaving only the alphabetically ordered top-level keys `G1`, `G2`, `G3`, `G4`, `MODEL_SPECS`, `acceptance_gates`, `activation_authorized`, and an incomplete `arc_summary`. A reader is missing:
- The remaining 103 entries of `arc_summary.rows` (only 21 complete rows, `routing-000` through `routing-020`, are present out of `arc_summary.completed: 124` and `arc_summary.denominator: 124`; all 72 `arc_summary.regression_result.legacy_controls`, 31 of the 52 `recognized_total` rows, and 18 of the 22 `before_mismatches` are cut off).
- Every top-level field alphabetically after `arc_summary`, including any overall headline `verdict`/`status`/`summary`, any `gate_check_summary` needed to verify why `activation_authorized` is `false`, and any evaluation results or per-unit data corresponding to `acceptance_gates` or the FoVer `condition_A` AUROC / `learning_contribution` CIs summarized in `G1`–`G2`.

## THE CHECK A READER CANNOT DO
What is the artifact's actual top-level verdict (and its `gate_check_summary` if blocked), and do the 103 truncated rows of `arc_summary.rows`—including all 72 `legacy_controls` and 18 unseen `before_mismatches`—actually satisfy `after == reference` with `passed: true` and zero `after_mismatches`?

## experiment_9222_v785_contract.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact's headline claim and verdict cannot be read because the JSON is truncated mid-string inside `full_task_objects`, though the visible fields show an administrative contract run (`experiment_id`: 9222, `claim_scope.independent_generalization_score`: 0) with `contract_ready_score`: 0, `completed_count`: 0, `failed_count`: 5, and `censored_count`: 9.

## WHAT IS MISSING
The artifact JSON is truncated inside `full_task_objects` (ending mid-line at `- validation_receipts: commands, log`), cutting off every top-level key alphabetically after `full_task_objects`. While `experiment_id`, `acceptance_gates`, `claim_scope`, `attempted_count`, `completed_count`, `failed_count`, `censored_count`, `contract_ready_score`, `consumer_shard_rows`, `diagnostic_protocol`, `entropy_reference_protocol`, and `field_principles` are present, the following fields—all declared in `field_principles` as part of this artifact—are missing from the text: `honest_verdict`, `verdict_class`, `gate_check_summary`, `rows`, `previous_outcome_rows`, `protocol_owner_rows`, `required_checks_passed`, `validation_receipts`, `owned_statement_coverage`, `preconditions_checked`, `replay_authority_ready_score`, `source_ready_score`, `static_methods_ready_score`, `online_methods_ready_score`, and `sample_size_budget`.

## THE CHECK A READER CANNOT DO
Which verdict (`honest_verdict`/`verdict_class`) Exp9222 actually declared, which specific checks and observed values caused `contract_ready_score: 0`, `failed_count: 5`, and `censored_count: 9` (given that all three visible `consumer_shard_rows` have `passed: true` and `exit_code: 0`), and whether the cut-off `gate_check_summary` and `rows` record diagnostics for them?

## experiment_9223_corpus_prefilter.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 9223 was blocked at the `conductor_pre_gate` layer because 1 of 2 upstream pre-gates failed (`exp9222-contract.contract_ready_score` observed `0` when expecting `== 1`).

## WHAT IS MISSING
nothing (inspected `status`, `honest_verdict`, `blocked_reason`, `failed_upstream`, `failed_field`, `failed_operator`, `failed_expected`, `failed_observed`, `failed_evidence_path`, `failed_evidence_sha256`, `blocked_diagnostic_contract`, `gate_check_summary`, `gates_evaluated`, and `blocked_at_layer`, which record both evaluated gates, exact expected vs. observed values, and the upstream artifact path and SHA-256 hash).

## THE CHECK A READER CANNOT DO
none

## experiment_9225_v785_selective_deferral_protocol.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact's actual headline verdict is truncated before it appears, though `claim_scope` and `acceptance_gates` frame the experiment as qualifying source custody, frozen 10-stream selective-deferral Horvitz-Thompson IPW controls, and a label-free 57-column cubic B-spline support-density classifier (with no empirical LLM or generalization claim).

## WHAT IS MISSING
The JSON artifact is truncated mid-object inside `gate_check_summary` (at `"expected": "sha256:96c654f6ec04ecf532a8b1947f71bda819cf`). While `acceptance_gates`, `claim_scope`, `estimator_control_rows`, `attempted_count`, `completed_count`, `failed_count`, `field_principles`, and the prefix of `gate_check_summary` are present, every top-level field alphabetically after `gate_check_summary` that is listed in `field_principles` is cut off—specifically `honest_verdict`, `verdict_class`, `required_checks_passed`, `selective_deferral_ready_score`, `spline_support_diagnostic_ready_score`, `rows`, `spline_control_rows`, `support_counts`, `owned_statement_coverage`, `validation_receipts`, and the remainder of `gate_check_summary`.

## THE CHECK A READER CANNOT DO
A reader cannot determine whether `honest_verdict` / `verdict_class` reported a pass or a block, nor inspect `rows`, `spline_control_rows`, `validation_receipts`, or the rest of `gate_check_summary` to verify whether any gate failed (and if so, with a diagnostic) or whether per-unit spline and selective-deferral rows actually back the claimed readiness scores.
