# Artifact convention audit

Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:
a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.

This audit never edits an artifact and never blocks anything. It surfaces; the operator
decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on
evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CHECKABLE | 3 |
| CANNOT_DETERMINE | 5 |

## experiment_9241_v787_contract.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact's headline verdict cannot be read because the JSON is truncated mid-string inside `full_task_objects` before any verdict fields appear, though the visible administrative fields (`experiment_id: 9241`, `contract_ready_score: 0`, `attempted_count: 7`, `completed_count: 0`, `failed_count: 7`, `censored_count: 7`, `claim_scope`) indicate an administrative contract and replay-qualification run that failed to achieve contract readiness.

## WHAT IS MISSING
Because the JSON is alphabetically ordered and truncated mid-string inside `full_task_objects` (at `"Confirm REQ-REPORT-924"` in the `exp9242-corpus-prefilter` entry), every key after `full_task_objects` is cut off—specifically `honest_verdict`, `verdict_class`, `gate_check_summary`, `rows`, `required_checks_passed`, `replay_authority_ready_score`, `source_ready_score`, `static_methods_ready_score`, `online_methods_ready_score`, and `validation_receipts`. The fields that ARE present—including `experiment_id`, `contract_ready_score`, `attempted_count`, `completed_count`, `failed_count`, `censored_count`, `excluded_count`, `consumer_shard_rows` (all six shards with `passed: true` and `exit_code: 0`), `claim_scope`, `acceptance_gates`, `full_task_objects`, and `field_principles` (which explicitly lists `gate_check_summary`, `honest_verdict`, `verdict_class`, `rows`, `required_checks_passed`, `replay_authority_ready_score`, and `validation_receipts`)—confirm those fields belong to the artifact schema but were truncated away before they could be audited.

## THE CHECK A READER CANNOT DO
Given that `contract_ready_score` is `0` and `failed_count`/`censored_count` are both `7` (`completed_count: 0`) while all six `consumer_shard_rows` show `passed: true`, a reader cannot determine what `honest_verdict`/`verdict_class` the artifact actually declares or whether `gate_check_summary` and `rows` record which specific check(s) failed and what observed values triggered the failure.

## experiment_9242_corpus_prefilter.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Experiment 9242 was blocked at `conductor_pre_gate` (`honest_verdict`: `blocked_gate_check_failed`) because 2 of 2 upstream `exp9241-contract` gates failed (`contract_ready_score` and `replay_authority_ready_score`, each observed as `0` against expected `1`).

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9244_v787_conformal_risk_protocol.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact's headline claim cannot be determined because the JSON is truncated mid-object before any verdict, summary, or gate-evaluation fields appear, showing only five gate specifications in `acceptance_gates` and 12 `passed: true` condition rows in `ca_ocrc_control_rows`.

## WHAT IS MISSING
The artifact contains only the alphabetically early top-level keys `MODEL_SPECS`, `acceptance_gates` (`ca_ocrc_method`, `conformal_risk_method`, `provenance_ledger_method`, `source_custody`, `validation`), `attempted_count`, `ca_ocrc_control_rows`, and `ca_ocrc_protocol_reference`, cutting off after 116 of 150 entries in `ca_ocrc_protocol_reference.proxy_delay_seeds` on an unclosed `{`. Missing because of truncation are: the overall verdict/status/headline-claim field, the gate-evaluation summary (e.g., `gate_check_summary` or per-gate pass/block diagnostics), and all result fields and per-unit rows for the remaining four gates defined in `acceptance_gates` (`conformal_risk_method`, `provenance_ledger_method`, `source_custody`, and `validation`).

## THE CHECK A READER CANNOT DO
Did the experiment pass, fail, or block—and for the comparative `conformal_risk_method` gate (cluster-valid bounds vs. naive i.i.d. sibling bounds across 2000 trials) and other remaining gates, did the artifact record per-unit evidence or a blocking diagnostic?

## experiment_9255_v788_contract.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
Cannot be determined because the JSON artifact is truncated mid-key inside `ca_ocrc_protocol_reference.proxy_delay_seeds`, omitting any headline claim, verdict, or outcome fields that would appear later in the document.

## WHAT IS MISSING
The artifact contains only alphabetically early top-level keys (`MODEL_SPECS`, `acceptance_gates`, `activation_authorized`, `ambiguity_masks_reference`, `archive_lag`, `attempted_count`, and a partial `ca_ocrc_protocol_reference` cut off at `"t` in `proxy_delay_seeds`). It is missing the remainder of the JSON payload—specifically any `verdict`/`status` field, `gate_check_summary` or blocker diagnostic fields, comparative metrics/gate outcomes, and any per-unit result rows that would sort alphabetically after `ca_ocrc_protocol_reference`.

## THE CHECK A READER CANNOT DO
Whether this artifact makes a comparative or gate-met claim backed by per-unit rows, reports a blocked verdict (with or without a diagnostic explaining which check failed and what value it observed), or makes no claim at all.

## experiment_9256_v788_corpus_prefilter.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The artifact is truncated before any headline verdict or summary appears, though `acceptance_gates` reports three corpus-prefilter gates passed (`complete_path_traversal`, `exact_semantic_parity`, `stable_candidate_bytes`) and one failed (`all_required_checks_and_coverage`), while `claim_scope.allowed_claims` limits claims to "Recorded failed qualification operands and completed controls only."

## WHAT IS MISSING
The JSON is truncated mid-string inside `corpus_manifest_reference.ordered_paths` (at `".experiment_6542_drift_bench_external_intake_v2.tx/shards/e068c9e346f812111eb964bdb8152de`), cutting off the rest of the file after the alphabetically ordered keys that ARE present (`MODEL_SPECS`, `acceptance_gates`, `attempted_count`, `candidate_counts`, `censored_count`, `claim_scope`, `code_config_hashes`, `completed_count`, and the start of `corpus_manifest_reference`). Because everything alphabetically after `corpus_manifest_reference` is missing, a reader cannot find the headline `verdict`/`status` field, `gate_check_summary` (or equivalent diagnostic showing which check failed in `acceptance_gates.all_required_checks_and_coverage` and what value was observed), or any per-unit result/control rows supporting `acceptance_gates.exact_semantic_parity` and the 41 `attempted_count`/`completed_count` items.

## THE CHECK A READER CANNOT DO
Which specific check(s) in `acceptance_gates.all_required_checks_and_coverage` (e.g., owned tests, V782 controls, E2E-018/021, Ruff, strict mypy, spec coverage, or 100% statement coverage) failed, what values were observed, and whether the untruncated artifact actually recorded those diagnostics and per-unit control results?

## experiment_9257_forward_qualification.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
Exp9257 was blocked at `blocked_at_layer` (`conductor_pre_gate`) with `honest_verdict` `blocked_gate_check_failed` because the upstream gate `exp9256-corpus-prefilter.corpus_scan_ready_score` evaluated to `0` (`int`) instead of the expected `1` (`int`), as documented in `gate_check_summary`, `gates_evaluated`, and `blocked_diagnostic_contract`.

## WHAT IS MISSING
nothing

## THE CHECK A READER CANNOT DO
none

## experiment_9258_v788_victris_energy_bot.json

**CHECKABLE**

## VERDICT
CHECKABLE

## WHAT THE CLAIM IS
The artifact makes no comparative simulator win-rate claim (`scientific_effect_measured: false`, `heldout_matches_played: 0`) and instead reports a blocked run (`verdict_class: "blocked"`, `honest_verdict: "complete_blocked_arena_sim_wheel_missing"`) due to the missing `arena_sim` wheel, claiming only offline harness controls (sub-10ms tick latency, shout isolation, DEV/HELDOUT seed disjointness, and passing owned validation checks).

## WHAT IS MISSING
nothing — the blocker diagnostics are explicitly recorded in `gate_check_summary` (`arena_sim_module_import` with `expected: true`, `observed: false`, `passed: false`; and `arena_sim_wheel_sha256` with `observed: "missing_file"`, `passed: false`) and `preconditions_checked` (`resource: "arena_sim_simulator"`, `available: false`), while the completed offline checks include per-unit synthetic rows in `rows` (`victris_synth_00` through `victris_synth_23`), per-candidate latency bounds in `tick_latency_rows` and `archive_versions`, and check receipts in `validation_receipts` and `site_rules_checks`.

## THE CHECK A READER CANNOT DO
none

## experiment_9268_v788_capstone.json

**CANNOT_DETERMINE**

## VERDICT
CANNOT_DETERMINE

## WHAT THE CLAIM IS
The visible prefix reports `G1`–`G4` passing (attesting external FoVer headline reproducibility), `activation_authorized: false`, and `arc_summary` routing equality of `124/124` (`before_mismatches: 22` to `after_mismatches: 0`) with `required_checks_passed: false`, but the artifact is truncated mid-JSON before any top-level verdict or headline claim appears.

## WHAT IS MISSING
The artifact is cut off mid-string inside `arc_summary.rows[20]` (`unit_id: "routing-020"`, field `reference` ending at `"programmatic_expert_trust_weighting_`). Because top-level keys are sorted alphabetically (only `G1`, `G2`, `G3`, `G4`, `MODEL_SPECS`, `acceptance_gates`, `activation_authorized`, and part of `arc_summary` are present), a reader cannot find:
- Any top-level `verdict`, `status`, `summary`, `claim`, or `gate_check_summary` fields that would appear after `arc_summary` and state what the headline outcome is and why `activation_authorized` is `false` and `arc_summary.required_checks_passed` is `false` (beyond `arc_summary.consumer_reachability` noting `live_path_observed: false` and `win_reached: false`).
- The remaining 103 rows (`routing-021` through `routing-123`) of `arc_summary.rows` needed to verify `arc_summary.completed` (`124`), `arc_summary.equality_numerator` (`124`), `arc_summary.denominator` (`124`), and `arc_summary.regression_result` (`before_mismatches: 22`, `after_mismatches: 0`, `legacy_controls: 72`, `recognized_total: 52`).
- Any evaluation results or per-unit rows corresponding to the 13 gates listed in `acceptance_gates` (`CES`, `ambiguity`, `conformal`, `contract`, `corpus`, `forwards`, `independence`, `learning`, `selective_deferral`, `spline_support`, `static`, `validation`, `victris`) or the FoVer AUROC/learning-contribution intervals cited in `G1`/`G2` (which only point to external `experiment_2850_fover_dual_condition_integrity_v4.json`).

## THE CHECK A READER CANNOT DO
What is the artifact's actual headline verdict—and if it is blocked (`activation_authorized: false`, `arc_summary.required_checks_passed: false`), does the truncated tail contain a `gate_check_summary` identifying the failing check and observed value, and do the missing 103 rows of `arc_summary.rows` actually reconcile to `equality_numerator: 124`, `before_mismatches: 22`, and `after_mismatches: 0`?
