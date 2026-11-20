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

## experiment_9221_v784_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact is truncated mid-JSON before any top-level verdict or headline section appears, while its visible prefix asserts passing FoVer ship gates (`G1`–`G4`, including `condition_A` AUROC in `[0.9027, 0.9235]` and `learning_contribution` in `[0.0125, 0.0245]`), records `activation_authorized: false` and `arc_summary.required_checks_passed: false`, and claims `arc_summary.regression_result` reduced 22 `before_mismatches` to 0 `after_mismatches` across 124 units.

## WHAT IS MISSING
The JSON is cut off mid-string inside the 22nd entry of `arc_summary.rows` (after `"dagger_off_path_data_collection`). Present top-level fields are only `G1`, `G2`, `G3`, `G4`, `MODEL_SPECS`, `acceptance_gates`, `activation_authorized`, and the beginning of `arc_summary` (including `compatibility_matrix`, `completed`, `consumer_reachability`, `current_ARC_experiment_scheduled`, `denominator`, `equality_numerator`, `historical_experiment_id`, `live_path_observed`, `regression_result`, `required_checks_passed`, and rows `routing-000` through `routing-020`). Missing fields are:
- All top-level fields alphabetically after `arc_summary` — including the artifact's actual `verdict`/`status`, headline claim, `gate_check_summary` (needed to tell whether `activation_authorized: false` and `arc_summary.required_checks_passed: false` represent a diagnosed or undiagnosed blocker), and any other experiment/result blocks.
- The remaining 103 of 124 per-unit entries in `arc_summary.rows` (only 21 complete rows appear, covering 3 of the claimed 22 `before_mismatches` and 0 of the 72 `legacy_controls`).
- Any per-unit data supporting the comparative `learning_contribution` and AUROC intervals in `G2` (which only references external file `experiment_2850_fover_dual_condition_integrity_v4.json`, if not present in the truncated tail).

## THE CHECK A READER CANNOT DO
Do the truncated 103 rows of `arc_summary.rows` and the cut-off top-level sections after `arc_summary` (such as `verdict` and `gate_check_summary`) confirm the artifact's headline verdict, blocker diagnostics, and `regression_result` totals (`before_mismatches: 22`, `after_mismatches: 0`, `legacy_controls: 72`, `equality_numerator: 124`)?

## experiment_9222_v785_contract.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact's headline verdict cannot be read because the JSON is truncated mid-stream inside `full_task_objects`, though the visible prefix of `experiment_id: 9222` shows an administrative contract/replay task reporting `contract_ready_score: 0`, `attempted_count: 5`, `completed_count: 0`, `failed_count: 5`, and `censored_count: 9`.

## WHAT IS MISSING
The JSON payload is truncated alphabetically inside `full_task_objects[1].prompt` (ending at `"validation_receipts: commands, log`), cutting off every key after `full_task_objects`. While `experiment_id`, `acceptance_gates`, `claim_scope`, `attempted_count`, `completed_count`, `failed_count`, `censored_count`, `contract_ready_score`, `consumer_shard_rows`, `code_config_hashes`, and `field_principles` are present—and `field_principles` explicitly indicates that the full schema contains later keys—the artifact is missing `honest_verdict`, `verdict_class`, `gate_check_summary`, `rows`, `validation_receipts`, `required_checks_passed`, `replay_authority_ready_score`, `protocol_owner_rows`, and `preconditions_checked`.

## THE CHECK A READER CANNOT DO
Why `contract_ready_score` is `0` with `failed_count: 5` and `censored_count: 9` (despite all three visible `consumer_shard_rows` having `passed: true` and `exit_code: 0`), what `honest_verdict` and `verdict_class` were issued, and whether `gate_check_summary` and `rows` actually record the failed checks, observed values, and per-unit results?

## experiment_9223_corpus_prefilter.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 9223 was blocked at `conductor_pre_gate` (`status`: `"blocked"`, `honest_verdict`: `"blocked_gate_check_failed"`) because 1 of 2 upstream gates failed—specifically `exp9222-contract.contract_ready_score` observed `0` when expecting `1`, as recorded across `gate_check_summary`, `blocked_reason`, `blocked_diagnostic_contract`, and `gates_evaluated`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9225_v785_selective_deferral_protocol.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Based on `claim_scope` and `acceptance_gates`, the experiment seeks to qualify upstream Exp9097/9126/9132/9211 custody, frozen 10-stream selective-deferral feedback masks with Horvitz-Thompson IPW risk controls, and a label-free 57-column cubic B-spline support-density classifier (explicitly disclaiming any LLM load, verifier benefit, or learning effect), but the artifact's actual headline verdict is cut off by truncation.

## WHAT IS MISSING
The artifact JSON is truncated mid-value inside `gate_check_summary` (ending abruptly at `"expected": "sha256:96c654f6ec04ecf532a8b1947f71bda819cf`). While `acceptance_gates`, `attempted_count` (150), `completed_count` (150), `failed_count` (0), `claim_scope`, `estimator_control_rows` (5 fixture controls), `field_principles`, and the beginning of `gate_check_summary` are present, every field enumerated in `field_principles` alphabetically after `gate_check_summary` is missing from the artifact text—including the remainder of `gate_check_summary`, `honest_verdict`, `verdict_class`, `required_checks_passed`, `selective_deferral_ready_score`, `spline_support_diagnostic_ready_score`, `scientific_effect_measured`, `rows`, `spline_control_rows`, `support_counts`, `owned_statement_coverage`, and `validation_receipts`.

## THE CHECK A READER CANNOT DO
Whether this run passed or was blocked (and if blocked, whether the tail of `gate_check_summary` recorded the failing check), and whether the truncated `rows`, `spline_control_rows`, and `validation_receipts` fields actually contain per-unit evidence backing the selective-deferral mask, B-spline support, and validation gates.

## experiment_9240_v786_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible portion of this truncated artifact reports passing FoVer reproducibility gates (`G1`–`G4`, citing `condition_A` AUROC in `[0.9027, 0.9235]` and `learning_contribution` in `[0.0125, 0.0245]`), `activation_authorized: false`, and an ARC routing regression reducing mismatches from `22` to `0` across `124` units (`arc_summary.regression_result`), while the top-level verdict and all subsequent sections are cut off.

## WHAT IS MISSING
The JSON is truncated mid-string inside `arc_summary.rows[20].reference` (`"online_warm_action_effect_controller_`), after alphabetically ordered top-level keys `G1`, `G2`, `G3`, `G4`, `MODEL_SPECS`, `acceptance_gates`, `activation_authorized`, and the start of `arc_summary` (which contains `compatibility_matrix`, `completed`, `consumer_reachability`, `current_ARC_experiment_scheduled`, `denominator`, `equality_numerator`, `historical_experiment_id`, `live_path_observed`, `regression_result`, `required_checks_passed`, and `rows`). As a result, a reader is missing:
- The remainder of `arc_summary.rows`: only `routing-000` through `routing-019` are complete, `routing-020` is cut off, and 103 further rows needed to account for `completed: 124`, `denominator: 124`, `equality_numerator: 124`, `before_mismatches: 22`, `after_mismatches: 0`, `legacy_controls: 72`, and `recognized_total: 52` are absent.
- All top-level fields alphabetically after `arc_summary`, including any `verdict`, `status`, `headline`/`summary`, `gate_check_summary` (needed to see whether `activation_authorized: false` and `arc_summary.required_checks_passed: false` constitute a block with or without diagnostic values), and any per-unit FoVer data backing `G2`'s `condition_A` AUROC and `learning_contribution` intervals (only the external filename `experiment_2850_fover_dual_condition_integrity_v4.json` in `G1`/`G4` `source` is present).

## THE CHECK A READER CANNOT DO
Do the missing 104 rows of `arc_summary.rows` actually satisfy `equality_numerator: 124`, `before_mismatches: 22`, and `after_mismatches: 0`, and do the truncated top-level keys after `arc_summary` include a final `verdict`, a `gate_check_summary` explaining `activation_authorized: false` and `arc_summary.required_checks_passed: false`, or per-unit rows verifying `G2`'s `condition_A` AUROC and `learning_contribution` claims?

## experiment_9241_v787_contract.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact's headline verdict is cut off by truncation, leaving only partial administrative metadata for `experiment_id` 9241 showing `contract_ready_score: 0`, `completed_count: 0`, `failed_count: 7`, `censored_count: 7` (out of `attempted_count: 7`), and six passing `consumer_shard_rows`.

## WHAT IS MISSING
The JSON artifact is truncated mid-string inside `full_task_objects` (at `exp9242-corpus-prefilter`'s `prompt`, ending at `Confirm REQ-REPORT-924`), cutting off every top-level key alphabetically after `full_task_objects`. Although `field_principles` lists them alongside present fields (`acceptance_gates`, `attempted_count`, `censored_count`, `claim_scope`, `completed_count`, `consumer_shard_rows`, `contract_ready_score`, `duration_s`, `experiment_id`, `failed_count`, `full_task_objects`), the actual values for `honest_verdict`, `verdict_class`, `gate_check_summary`, `rows`, `required_checks_passed`, `preconditions_checked`, `validation_receipts`, `replay_authority_ready_score`, `source_ready_score`, `static_methods_ready_score`, `online_methods_ready_score`, and `protocol_owner_rows` are missing from the provided text.

## THE CHECK A READER CANNOT DO
What headline verdict (`honest_verdict` / `verdict_class`) did Exp9241 actually record, and did `gate_check_summary` and `rows` document which specific check or operand failed to produce `contract_ready_score: 0`, `failed_count: 7`, and `censored_count: 7` despite all six `consumer_shard_rows` reporting `passed: true`?

## experiment_9242_corpus_prefilter.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 9242 was blocked at `conductor_pre_gate` because both upstream `exp9241-contract` gate checks (`contract_ready_score` and `replay_authority_ready_score`) evaluated to `0` instead of the expected `1`, as recorded in `gate_check_summary`, `blocked_diagnostic_contract`, and `gates_evaluated`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9244_v787_conformal_risk_protocol.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact's headline claim cannot be determined because the JSON is truncated mid-object before any verdict, summary, or gate outcome fields appear (only the five gate definitions in `acceptance_gates` and partial CA-OCRC fields are present).

## WHAT IS MISSING
Only the alphabetically early top-level keys `MODEL_SPECS`, `acceptance_gates`, `attempted_count`, `ca_ocrc_control_rows`, and an incomplete `ca_ocrc_protocol_reference` (truncated inside `proxy_delay_seeds` during `stream_07`, before reaching `proxy_delay_seed_count` of 150, ending on an unclosed `{`) are present. Every top-level field after `ca_ocrc_protocol_reference` is missing: the overall verdict/status field, `gate_check_summary` (or any pass/fail/blocked outcomes and diagnostics for the gates listed under `acceptance_gates`: `ca_ocrc_method`, `conformal_risk_method`, `provenance_ledger_method`, `source_custody`, and `validation`), and all empirical results or per-unit rows for the 2000-trial conformal risk comparison (`conformal_risk_method`) and the 900-response ledger (`provenance_ledger_method`).

## THE CHECK A READER CANNOT DO
Did this run report a passing comparative claim, a null result, or a blocked gate—and if it claimed `conformal_risk_method` or the other gates in `acceptance_gates` were met (or blocked), do the truncated sections contain per-unit rows or blocker diagnostics to verify that outcome?
