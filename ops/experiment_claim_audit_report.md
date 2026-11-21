# Experiment claim-refutation audit

One question per artifact: what would REFUTE the headline claim, and was that
checked? Fabrication is out of scope (adversarial_verify covers it); this audit
targets claims that are true by construction, circular, in-sample, baseline-weak,
or contradicted by their own rows.

This audit never edits an artifact and never blocks anything. It surfaces; the
operator decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity
guard rest on evidence the reviewer could not have read -- do NOT act on them.

| verdict | count |
|---|---|
| CLAIM_SUPPORTED | 1 |
| NO_CLAIM | 7 |

## experiment_9241_v787_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or scientific headline claim in this artifact to refute. It is an infrastructure contract-binding and receipt-replay scaffolding artifact (`honest_verdict = "complete_blocked_v787_contract"`) that runs zero model invocations, explicitly disclaims scientific benefit and independent generalization, and records a blocked contract gate (`contract_ready_score: 0`, `completed_count: 0`, `failed_count: 7`, `censored_count: 7`). If it were misread as claiming a passing 14-task contract binding or any empirical method gain, that would be refuted by `contract_ready_score: 0`, `completed_count: 0`, `failed_count: 7`, and the absence of any empirical method comparison or generalization measurement. Note also that the provided artifact text is truncated inside `historical_input_roots`, omitting later fields such as `rows`, `replay_authority_ready_score`, and `validation_receipts`.

## WAS THAT CHECKED
Not applicable for a comparative scientific claim, as none is made. For the administrative scaffolding itself, the artifact checked contract input hashes (`gate_check_summary`) and six split consumer test shards (`consumer_shard_rows`, all passing with exit code 0), and recorded its own blocked contract status (`contract_ready_score: 0`, `completed_count: 0`, `failed_count: 7`, `activation_authorized: false`, `current_science_paper_ready: false`); fields after `historical_input_roots` (including `rows`, `validation_receipts`, and `replay_authority_ready_score`) are absent from the truncated excerpt.

## EVIDENCE
- `"track": "infrastructure"`
- `"title": "Exp9241: Qualify split V782-V786 consumer replay and bind fourteen V787 tasks"`
- `"scope": "Administrative contract and original methods only; no measured human benefit or ARC solve. Cannot close GAP-ORACLE-DISTINCT or GAP-DETECTOR-AUROC-4208."`
- `"condition": "unmeasured; fixtures and historical receipts grant no independent benefit"`
- `"MODEL_SPECS": []`
- `"current_llm_invocations": 0`
- `"generated_tokens": 0`
- `"effective_independent_groups": 0`
- `"independent_generalization_score": 0`
- `"exposure_scope": "exposed_development"`
- `"contract_ready_score": 0`
- `"attempted_count": 7`
- `"completed_count": 0`
- `"failed_count": 7`
- `"censored_count": 7`
- `"activation_authorized": false`
- `"current_science_paper_ready": false`
- `"external_publication_authorized": false`

## RECOMMENDATION
KEEP

## experiment_9242_corpus_prefilter.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This is a blocked-gate receipt artifact rather than an experimental result: Exp9242 ("Qualify a lossless byte prefilter for the complete corpus walk") never executed, so it makes no empirical or comparative claim about a prefilter. Treating the receipt itself as a factual status record, it would be refuted if the entries in `gates_evaluated` showed `passed: true` or `actual: 1` while `status` reported `"blocked"`, or if the artifact asserted any positive qualification outcome despite `duration_s: 0.0` and `blocked_at_layer: "conductor_pre_gate"`. Because the experiment did not run, there are no experimental rows, comparators, or prefilter metrics to evaluate.

## WAS THAT CHECKED
Yes, for internal consistency of the gate receipt: `gates_evaluated` lists 2 of 2 upstream gates on `exp9241-contract` (`contract_ready_score` and `replay_authority_ready_score`), both with `expected: 1`, `actual: 0`, and `passed: false`, matching `blocked_reason`, `blocked_diagnostic_contract`, `gate_check_summary`, `status: "blocked"`, and `honest_verdict: "blocked_gate_check_failed"`. No experimental method, baseline comparison, or corpus-walk prefilter qualification was executed or checked in this artifact (`duration_s: 0.0`).

## EVIDENCE
- `"schema": "blocked_gate_check_v1"`
- `"status": "blocked"`
- `"duration_s": 0.0`
- `"honest_verdict": "blocked_gate_check_failed"`
- `"blocked_at_layer": "conductor_pre_gate"`
- `"gate_check_summary": "gate-unsat(final): 2 of 2 gate(s) failed; first failure: exp9241-contract.contract_ready_score (actual=0 == expected=1)"`
- `"upstream": "exp9241-contract"`
- `"artifact_field": "contract_ready_score"`
- `"artifact_field": "replay_authority_ready_score"`
- `"expected": 1`
- `"actual": 0`
- `"passed": false`

## RECOMMENDATION
KEEP

## experiment_9244_v787_conformal_risk_protocol.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Because this is a protocol-sealing, custody, and provenance-ledger receipt with a null headline verdict (`complete_null_v787_conformal_risk_protocol`) and zero model invocations, it makes no empirical comparative claim about verifier benefit, learning gain, or generalization. What would refute its narrower protocol-qualification and accounting receipt would be: (1) any positive empirical or generalization verdict despite loading no models; (2) cluster-valid conformal bounds dropping below 0.95 coverage at rho >= 0.7, failing monotonicity across alpha_risk, or returning a numeric bound at zero accepted qids; (3) randomized proxy-delay IPW showing nonzero exact bias or failing to bracket corrupted feedback for epsilon in [0.0, 0.05]; or (4) missing/excluded responses, invented span labels, model-judge overrides, or failed upstream SHA-256 custody checks in the 900-response ledger. Note that the artifact lacks any empirical multi-arm evaluation on real predictions (intentionally deferred until after protocol sealing), and this excerpt elides 9 of 12 `ca_ocrc_control_rows`, 126 of 129 `gate_check_summary` entries, and 897 of 900 `rows`.

## WAS THAT CHECKED
Yes, for all protocol-qualification and accounting invariants, and no empirical comparative claim was tested because none is made. Specifically: zero model work and zero generalization are explicitly recorded in `MODEL_SPECS`, `model_invocation_counts`, `claim_scope`, and `honest_verdict`; cluster-valid coverage vs naive i.i.d. sibling coverage is checked in all 8 `coverage_control_rows`; edge behaviors (zero accepted qids, monotone alpha-risk grid, all-errors unity) are checked in `edge_control_rows`; IPW unbiasedness and corrupted-feedback bracketing are checked in `ca_ocrc_control_rows` (3 of 12 shown, 9 elided); complete 900-response / 300-qid provenance accounting across `TRAIN150`, `CAL50`, and `TEST100` is checked in `provenance_summary` and `provenance_ledger_reference`; upstream custody hashes are checked in `gate_check_summary` (3 of 129 shown, 126 elided); and unrelated repository-wide test failure is transparently separated in `repository_health` rather than hidden. The artifact lacks an empirical benchmark against real rival verifiers or models because it is strictly pre-test protocol scaffolding.

## EVIDENCE
`"honest_verdict": "complete_null_v787_conformal_risk_protocol"`
`"MODEL_SPECS": []`
`"current_llm_invocations": 0`
`"inference_substrate_class": "no_model_load"`
`"model_invocation_status": "not_requested_empty_model_specs"`
`"independent_generalization_score": 0`
`"exposure_scope": "exposed_development_only"`
`"scope": "Qid-cluster conformal risk bounds, CA-OCRC capacity-constrained delayed-feedback IPW tracking and Mycelium provenance accounting only; no LLM load, verifier benefit, learning effect or independent generalization."`
`"principle": "Capacity-constrained delayed-feedback IPW tracking qualifies a shadow protocol, not empirical learning gain."`
`"principle": "Finite-sample cluster-valid risk bounds qualify a descriptive audit method, not primary verifier gain."`
`"principle": "Complete provenance-typology-granularity accounting must seal before reading TEST predictions."`
`"conformal_risk_ready_score": 1`, `"ca_ocrc_ready_score": 1`, `"provenance_audit_ready_score": 1`
`"arm": "mycelium_provenance_conformal_ca_ocrc"`
`"intended_count": 900`, `"attempted_count": 900`, `"completed_count": 900`, `"excluded_count": 0`, `"failed_count": 0`, `"censored_count": 0`
`"answer_level_present": 900`, `"span_level_absent": 900`, `"span_level_present": 0`, `"invented_span_labels": 0`, `"model_judge_overrides": 0`
`"cluster_valid_coverage": 0.999`, `"naive_sibling_coverage": 0.8615`, `"cluster_valid_coverage": 0.975`, `"naive_sibling_coverage": 0.791`
`"exact_ipw_risk": 0.3`, `"ipw_risk_bias": 0.0`, `"naive_risk": 0.0`, `"naive_risk_bias": -0.3`, `"corrupted_risk_lower": 0.27`, `"corrupted_risk_upper": 0.37`
`"required_checks_passed": true`, `"passed": false`, `"note": "Unrelated repository-wide suite recorded separately from owned/affected checks"`

## RECOMMENDATION
KEEP

## experiment_9255_v788_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This is an administrative contract, protocol-binding, and receipt-replay scaffolding artifact rather than a comparative scientific experiment, so it makes no scientific or comparative claim to refute. If its administrative readiness receipt were audited, it would be falsified by any failing consumer replay shard (`exit_code` nonzero or `passed` false), any SHA-256 mismatch in `gate_check_summary`, any failed or excluded task among the 14 bound task objects, or failing coverage/replay checks in the truncated tail of the artifact (such as `owned_statement_coverage`, `replay_authority_ready_score`, `required_checks_passed`, `validation_receipts`, or `rows`).

## WAS THAT CHECKED
Yes for the visible administrative checks, and not applicable for scientific comparison because no comparative claim is made. In the visible artifact, all seven `consumer_shard_rows` pass with `exit_code: 0` and `passed: true`, the visible `gate_check_summary` hash checks pass (with 1,675 additional entries elided), task counts show 14 attempted / 14 completed / 0 failed / 0 excluded / 0 censored, and the pre-pass confirms `honest_verdict = "complete_null_v788_contract"` with `verifier_is_oracle = false` and no circularity flags. Note that the artifact text is truncated inside `historical_input_roots`, so keys appearing later alphabetically (`owned_statement_coverage`, `replay_authority_ready_score`, `required_checks_passed`, `validation_receipts`, `rows`) are not present in the provided excerpt.

## EVIDENCE
- `"experiment_id": 9255`
- `"track": "infrastructure"`
- `"inference_substrate_class": "no_model_load"`
- `"MODEL_SPECS": []`
- `"current_llm_invocations": 0`
- `"generated_tokens": 0`
- `"effective_independent_groups": 0`
- `"independent_generalization_score": 0`
- `"current_science_paper_ready": false`
- `"activation_authorized": false`
- `"external_publication_authorized": false`
- `"scope": "Administrative contract and original methods only; no measured human benefit or ARC solve. Cannot close GAP-ORACLE-DISTINCT or GAP-DETECTOR-AUROC-4208."`
- `"allowed_claims"` lists `"administrative task binding"`, `"authenticated original human and method authority"`, and `"frozen secondary label sensitivity protocol"`
- `"scientific_benefit"` has `"condition": "unmeasured; fixtures and historical receipts grant no independent benefit"`
- `"contract_ready_score": 1` with `"contract_ready_score": "Exact task equality is independent of scientific availability."`
- `"attempted_count": 14`, `"completed_count": 14`, `"failed_count": 0`, `"excluded_count": 0`, `"censored_count": 0`
- All seven entries in `"consumer_shard_rows"` have `"exit_code": 0` and `"passed": true`

## RECOMMENDATION
KEEP

## experiment_9256_v788_corpus_prefilter.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
The v788 five-sentinel byte corpus prefilter fails readiness qualification (`complete_disqualified_v788_corpus_prefilter`, `corpus_scan_ready_score = 0`) because required checks and coverage did not pass, while recording completed corpus traversal, candidate-byte stability, and exact semantic parity/mutation controls only.

## WHAT WOULD REFUTE IT
1. The disqualification verdict would be refuted if all four acceptance gates actually passed (`all_required_checks_and_coverage.passed == true` and `required_checks_passed == true`), or conversely if the artifact claimed readiness or generalization despite a failed gate.
2. The completed-controls sub-claim would be refuted by any classification discrepancy between the prefilter and `original_decoder` (`parity_discrepancies > 0` or `parity_match == false`), any undetected sentinel or Unicode-escape mutation (`mutation_detected == false` or `missed_positive_count == 0`), incomplete corpus traversal (`visited_files < total_files` or `unfinished_files > 0`), or concurrent file mutation (`changed_files > 0` or `candidate_bytes_stable == false`).
3. Although `verifier_is_oracle` is true, that would only render a positive verifier-value claim circular; here no positive readiness, speedup, or generalization claim is made (and no model was loaded). Full verification of the exact failing test/check inside `all_required_checks_and_coverage` and of timing comparisons is not directly visible because `validation_receipts` and `stratum_cost_comparison` are truncated at the end of the artifact, and the 744,675 `skipped_eof_negative` corpus files were not individually re-decoded by `original_decoder` beyond the frozen parity fixtures, decoded corpus candidates, and sampled cost-model rows.

## WAS THAT CHECKED
Yes. This is an honest disqualified/null verdict backed by the artifact's own gate evaluations and control rows:
- Qualification failure was checked and triggered: `acceptance_gates.all_required_checks_and_coverage` evaluated to `passed: false` and `required_checks_passed` is `false`, setting `honest_verdict` to `"complete_disqualified_v788_corpus_prefilter"`, `corpus_scan_ready_score` to `0`, `independent_generalization_score` to `0`, and restricting `claim_scope.allowed_claims` to recorded failed qualification operands and completed controls under `"exposed_development_only"`. (The specific failing receipt entries in `validation_receipts` and 11 elided `gate_check_summary` entries are not shown due to truncation, though `owned_statement_coverage.rows` also records unexecuted lines in `v755_branch_runner.py`.)
- Parity and refutability of the controls were genuinely checked: `parity_and_mutation_controls` ran 30 frozen fixtures (`parity_discrepancies: 0`, `semantic_parity_exact: true`) and 6 mutation controls deleting each of the 5 sentinels and the `\u` Unicode-escape fallback, all 6 of which broke parity (`missed_positive_count` of 9, 4, 2, 2, 2, and 1; `all_mutations_detected: true`), proving the parity check could have failed.
- Completeness, stability, and row accounting were checked: all 764,150 corpus files were visited (`unfinished_files: 0`, `changed_files: 0`, `candidate_bytes_stable: true`, 19,475 decoded, 744,675 `safely_negative`, 289 positive `checked`, 0 `offenders`), and all 41 intended evaluation rows completed with 0 failed, 0 censored, and 0 excluded.
- Real rival and exposure checks: the prefilter was evaluated directly against `original_decoder` (not a trivial sanity-check baseline), no LLM was invoked (`current_llm_invocations: 0`), and no held-out generalization is claimed.

## EVIDENCE
- `honest_verdict`: `"complete_disqualified_v788_corpus_prefilter"`
- `corpus_scan_ready_score`: `0`
- `independent_generalization_score`: `0`
- `required_checks_passed`: `false`
- `claim_scope`: `allowed_claims` contains `"Recorded failed qualification operands and completed controls only."`, `exposure_scope` is `"exposed_development_only"`, `independent_generalization_score` is `0`
- `acceptance_gates`: `all_required_checks_and_coverage` has `passed` = `false`; `complete_path_traversal`, `exact_semantic_parity`, and `stable_candidate_bytes` have `passed` = `true`
- `parity_and_mutation_controls`: `fixture_count` = `30`, `parity_discrepancies` = `0`, `semantic_parity_exact` = `true`, `all_mutations_detected` = `true`; `mutation_controls` all have `mutation_detected` = `true` with `missed_positive_count` values `9`, `4`, `2`, `2`, `2`, and `1`
- `corpus_manifest_reference` and `candidate_counts`: `total_files` = `764150`, `visited_files` = `764150`, `unfinished_files` = `0`, `changed_files` = `0`, `candidate_bytes_stable` = `true`, `decoded_count` = `19475`, `safely_negative_eof_count` = `744675`, `checked` = `289`, `offenders` = `[]`
- `sample_size_budget` and counts: `intended_count` = `41`, `attempted_count` = `41`, `completed_count` = `41`, `failed_count` = `0`, `censored_count` = `0`, `excluded_count` = `0`
- `current_llm_invocations` = `0`, `model_invocation_status` = `"not_requested_no_model_load"`

## RECOMMENDATION
KEEP

## experiment_9257_forward_qualification.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This is a pre-gate receipt artifact recording that Experiment 9257 never ran because an upstream gate failed; it makes no experimental or comparative claim. The only procedural assertion it records—that the pre-gate check failed—would be refuted if `gates_evaluated` reported `passed: true` or `actual == expected`, or if the referenced upstream artifact at `failed_evidence_path` (which is external to this receipt and not included here) actually contained `corpus_scan_ready_score == 1`.

## WAS THAT CHECKED
Yes, for internal consistency of the receipt: `gates_evaluated` records 1 of 1 gate evaluated with `actual` = `0`, `expected` = `1`, `op` = `"=="`, and `passed` = `false`, which matches `blocked_diagnostic_contract`, `gate_check_summary`, `status`, and `honest_verdict`. No experimental comparison or validation run occurred (`duration_s` is `0.0`), and the external upstream artifact file itself is not present here to cross-check.

## EVIDENCE
- `"schema": "blocked_gate_check_v1"`
- `"status": "blocked"`
- `"duration_s": 0.0`
- `"honest_verdict": "blocked_gate_check_failed"`
- `"blocked_at_layer": "conductor_pre_gate"`
- `"gate_check_summary": "gate-unsat(final): 1 of 1 gate(s) failed; first failure: exp9256-corpus-prefilter.corpus_scan_ready_score (actual=0 == expected=1)"`
- `"expected": 1`
- `"actual": 0`
- `"passed": false`

## RECOMMENDATION
KEEP

## experiment_9258_v788_victris_energy_bot.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This is a blocked scaffolding and harness-readiness receipt making no comparative or scientific performance claim (`scientific_effect_measured` is `false`, `verdict_class` is `"blocked"`). Its narrow engineering and blocked-status receipts would be refuted only if the `arena_sim` module or wheel were actually present while reported missing, if any archive candidate violated the sub-10ms tick latency target or 1000ms budget, if shout isolation or `DEV_SEEDS`/`HELDOUT_SEEDS` disjointness failed, if any of the 6 owned validation checks or 100% owned statement coverage failed, or if simulator match outcomes were reported despite the missing wheel. Any future comparative claim that the Gibbs energy bot outperforms `v0_shoot_baseline` or `ts-starter` would require installing the missing `arena_sim-0.83.0.whl` simulator and running the 200 currently censored matches, and would be refuted if the bot tied or lost to `v0_shoot_baseline` or `ts-starter` on held-out seeds.

## WAS THAT CHECKED
Yes for the scaffolding, latency, site-rules, validation, and missing-simulator receipts: `preconditions_checked` and `gate_check_summary` verified the missing `arena_sim` module/wheel; `tick_latency_rows` measured all 6 candidates over 24 synthetic ticks; `site_rules_checks` verified seed disjointness and shout isolation; and `validation_receipts` and `owned_statement_coverage` verified all 6 owned checks with 100% owned statement coverage. No comparative gameplay evaluation was checked because the environment lacks `arena_sim-0.83.0.whl`, leaving all 200 intended simulator matches censored and all dev/held-out win rates `null`; the 24 synthetic rows (`synthetic_tactical_verification`) are explicitly marked censored for the missing wheel and are not used to assert a scientific effect.

## EVIDENCE
- `honest_verdict`: `"complete_blocked_arena_sim_wheel_missing"`
- `verdict_class`: `"blocked"`
- `scientific_effect_measured`: `false`
- `inference_substrate_class`: `"blocked_no_run"`
- `independent_generalization_score`: `0`
- `victris_harness_ready_score`: `1`
- `arena_sim_installed`: `false`
- `heldout_matches_played`: `0`
- `claim_scope` / `allowed_claims`: `"deterministic sub-10ms GibbsModel-scale state-action energy scorer for Victris Arena"`, `"strict untrusted opponent shout isolation and DEV_SEEDS (0..999) / HELDOUT_SEEDS (10000..10999) disjointness"`, `"honest blocked_arena_sim_wheel_missing report with zero fabricated simulator matches"`
- `sample_size_budget`: `intended` `200`, `censored` `200`, `attempted` `24`, `completed` `24`
- `gate_check_summary`: `check` `"arena_sim_module_import"` with `observed` `false` and `passed` `false`; `check` `"arena_sim_wheel_sha256"` with `observed` `"missing_file"` and `passed` `false`
- `heldout_summary`: `opponent` `"ts-starter"` and `"random"`, both with `n` `0`, `win_rate` `null`, `baseline_win_rate` `null`, `ci95` `null`
- `archive_versions`: `dev_win_rate` `null`, `dev_ci95` `null` across all 6 candidates
- `rows`: `condition` `"synthetic_tactical_verification"`, `censor_reason` `"arena_sim_wheel_missing"`
- `tick_latency_rows`: `max_ms` up to `0.5478`, `within_10ms_target` `true`, `within_1000ms_budget` `true`
- `site_rules_checks`: `dev_heldout_disjoint` `true`, `opponent_shout_ignored` `true`, `game_repo_modified` `false`, `live_ladder_contacted` `false`
- `required_checks_passed`: `true`; `owned_statement_coverage` `percent` `100.0`

## RECOMMENDATION
KEEP

## experiment_9268_v788_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This is an upstream-aggregation accounting and milestone reconciliation receipt (`complete_blocked_declared_gate` per pre-pass) that explicitly disavows any scientific, comparative, or generalization claim. Its accounting posture would only be falsified if the artifact asserted a positive comparative result, authorized activation, or claimed primary-gate rescue from secondary or blocked branches despite zero attempted scientific answers and failed upstream gates.

## WAS THAT CHECKED
Yes, across every summary block present in the artifact (note that the JSON payload is truncated inside `cost_summary`, so the top-level `honest_verdict` field is only visible via the pre-pass note `honest_verdict = "complete_blocked_declared_gate"` and `inference_substrate = "aggregation_from_upstream_artifacts"`):
- **No model execution or generalization claim**: `MODEL_SPECS` is empty, `collection_partition` reports 0 attempted scientific answers and a 900-answer execution deficit, `activation_authorized` is `false`, and `claim_scope` restricts claims to administrative outcomes while setting `independent_generalization_score` to `0` on the exposed LPHB900/300 development panel.
- **No comparative win or silent baseline failure**: All three `ces_summary` comparators (`mean_entropy`, `max_entropy`, `probability_length_error_probability`) are `unmeasured` with `denominator: 0`, all `conformal_risk_summary` and `ca_ocrc_summary` branches are `blocked` with `denominator: 0` and `primary_gate_rescue: false`, and all `cost_summary` hardware rows shown have `physical_probe_performed: false`.
- **Failed and disqualified upstream branches reported honestly**: `corpus_scan_summary` is `disqualified` (`corpus_scan_ready_score: 0` due to full-suite timeout), `arc_summary` is `disqualified` with `value_added: false` and `win_reached: false`, and Exp9258's simulator evaluation is explicitly `blocked` on the missing `arena_sim` wheel.

## EVIDENCE
- `"MODEL_SPECS": []`
- `"activation_authorized": false`
- `"attempted_scientific_answers": 0`
- `"execution_deficit_answers": 900`
- `"execution_only": true`
- `"independent_generalization_score": 0`
- `"exposure_scope": "Original exposed LPHB900/300 development panel"`
- `"principle": "Frozen conditions establish accounting only; secondary evidence never rescues primary science"`
- `"unmeasured human science, secondary CES, descriptive ambiguity/deferral/spline-support/conformal/CA-OCRC branches, and blocked Victris simulator matches"`
- `"corpus_scan_ready_score": 0`
- `"disqualification_reason": "full_python_suite_timed_out_at_599.87s_exit_minus_9"`
- `"status": "unmeasured"`
- `"censor_reason": "missing_reference_and_scientific_predictions"`
- `"status": "blocked"`
- `"censor_reason": "missing_frozen_scientific_predictions"`
- `"primary_gate_rescue": false`
- `"verdict_class": "disqualified"`
- `"value_added": false`
- `"honest_verdict": "complete_blocked_arena_sim_wheel_missing"`
- `"physical_probe_performed": false`

## RECOMMENDATION
KEEP
