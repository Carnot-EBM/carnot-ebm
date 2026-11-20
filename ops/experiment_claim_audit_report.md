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

## experiment_9221_v784_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This artifact is a milestone capstone aggregation and gate-block receipt (`honest_verdict = "complete_blocked_declared_gate"` per the pre-pass) rather than a comparative scientific claim. Because no method superiority, generalization, or scientific effect is claimed, there is no scientific hypothesis to refute; the artifact lacks executed scientific predictions, live model invocations, comparative CES measurements, hardware probes, and independent held-out evaluation. Its administrative accounting would only be refuted if retained upstream artifacts actually passed their declared gates (e.g., `exp9208-contract` having `contract_ready_score == 1` and `replay_authority_ready_score == 1`) or if executed scientific runs were misreported as skipped/unstarted.

## WAS THAT CHECKED
Yes, at the administrative receipt level. Declared gate operands and upstream SHA-256 custody checks are explicitly logged in `declared_gate_rows`, `corpus_scan_summary.evidence.gate_check_summary`, and `confirmed_branch_results[0].gate_check_summary`, showing why downstream branches were blocked (`exp9208-contract` observed `0` against expected `1` on `contract_ready_score` and `replay_authority_ready_score`, and `exp9209-corpus-prefilter` had `corpus_scan_ready_score` missing). Every scientific or comparative branch (`ces_summary`, `collection_partition`, `cost_summary`, `corpus_scan_summary`) is explicitly reported as unmeasured, unstarted, or blocked, the only qualified branch (`Exp9211`) is an explicit `null` with `no_model_load`, and the historical `arc_summary` control is explicitly marked `disqualified` with `value_added: false`. Note that the provided JSON excerpt is truncated inside `field_principles`, so trailing top-level keys (such as `outcome_rows`, `gate_failure_rows`, and the top-level `honest_verdict` field itself) are absent from the excerpt, though the mechanical pre-pass reports `honest_verdict = "complete_blocked_declared_gate"`.

## EVIDENCE
- `"activation_authorized": false`
- `"current_science_paper_ready": false`
- `"external_publication_authorized": false`
- `"MODEL_SPECS": []`
- `"current_llm_invocations": 0`
- `"effective_independent_groups": 0`
- `"independent_generalization_score": 0`
- `"scientific_success_inferred": false`
- `"executed": 3`
- `"skipped": 11`
- `"attempted_scientific_answers": 0`
- `"execution_only": true`
- `"unstarted": 3963`
- `"status": "blocked"`
- `"status": "unmeasured"`
- `"verdict_class": "disqualified"`
- `"value_added": false`
- `"verdict_class": "null"`
- `"fourteen terminal administrative outcomes"`
- `"unmeasured human science, secondary CES and two label sensitivity branches"`
- `"LPHB is exposed development; no independent human evaluation or generalization. Historical graph access and source authority remain blocked."`
- `"Hashed primitive denominators bind accounting; administrative readiness grants no new scientific claim."`

## RECOMMENDATION
KEEP

## experiment_9222_v785_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This is an administrative contract, protocol-binding, and receipt-replay scaffolding artifact (`honest_verdict = "complete_blocked_v785_contract"` in the pre-pass notes) that explicitly disclaims any scientific, generalization, or comparative method claim. There is no comparative headline claim to refute. Even treating its administrative status as a receipt claim, it would only be contradicted if the underlying validation checks and per-unit rows contradicted the recorded blocked outcome (for example, if it claimed readiness or scientific benefit despite failed checks, or if the missing/truncated `rows`, `validation_receipts`, `required_checks_passed`, and elided `gate_check_summary` entries showed zero failures while reporting `failed_count: 5` and `contract_ready_score: 0`).

## WAS THAT CHECKED
No comparative or scientific claim was tested because none is made: no model was loaded or invoked (`current_llm_invocations: 0`, `generated_tokens: 0`, `effective_independent_groups: 0`), and all publication/activation flags are `false`. On the administrative receipt side, the visible portion of the artifact records three passing pytest shards under `consumer_shard_rows` (`passed: true`, `exit_code: 0`) and three passing SHA-256 `contract_input` checks in `gate_check_summary`, while reporting `attempted_count: 5`, `completed_count: 0`, `failed_count: 5`, `censored_count: 9`, and `contract_ready_score: 0`. Note that the artifact snippet is truncated inside `historical_model_provenance` (and elides 1,625 entries in `gate_check_summary`), so the downstream `rows`, `validation_receipts`, `required_checks_passed`, and `honest_verdict` fields are not present in the artifact text to audit which five checks failed; however, the pre-pass notes confirm `honest_verdict = "complete_blocked_v785_contract"`, `verifier_is_oracle = false`, and `inference_substrate = "cpu_cached_authority_and_receipt_replay"`.

## EVIDENCE
- `"experiment_id": 9222`, `"track": "infrastructure"`, `"title": "Exp9222: Qualify partitioned consumer replay and bind fourteen V785 tasks"`
- `"claim_scope"` -> `"scope": "Administrative contract and original methods only; no measured human benefit or ARC solve. Cannot close GAP-ORACLE-DISTINCT or GAP-DETECTOR-AUROC-4208."`
- `"claim_scope"` -> `"allowed_claims"`: `"administrative task binding"`, `"authenticated original human and method authority"`, `"frozen secondary label sensitivity protocol"`
- `"claim_scope"` -> `"independent_generalization_score": 0`, `"exposure_scope": "Original LPHB900 answers/300 qids remain exposed development"`
- `"acceptance_gates"` -> `"scientific_benefit"` -> `"condition": "unmeasured; fixtures and historical receipts grant no independent benefit"`
- `"contract_ready_score": 0`, `"attempted_count": 5`, `"completed_count": 0`, `"failed_count": 5`, `"censored_count": 9`, `"excluded_count": 0`
- `"MODEL_SPECS": []`, `"current_llm_invocations": 0`, `"generated_tokens": 0`, `"effective_independent_groups": 0`
- `"activation_authorized": false`, `"current_science_paper_ready": false`, `"external_publication_authorized": false`
- `"consumer_shard_rows"` -> `"shard_name": "direct_consumers_core"`, `"shard_name": "direct_consumers_v782_v783"`, `"shard_name": "direct_consumers_v784"` (all `"passed": true`, `"exit_code": 0`)

## RECOMMENDATION
KEEP

## experiment_9223_corpus_prefilter.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This is a conductor pre-gate receipt recording that the experiment never ran because an upstream prerequisite gate failed; it makes no experimental or comparative claim about the lossless byte prefilter. The receipt's own status report would be refuted only if all entries in `gates_evaluated` had `passed: true` while still reporting a block, if the `failed_*` fields contradicted `gates_evaluated`, or if a positive completion verdict were emitted despite the failed gate.

## WAS THAT CHECKED
Yes, for receipt consistency: `gates_evaluated` lists both upstream checks, showing `contract_ready_score` failed (`actual: 0` vs `expected: 1`, `passed: false`) while `replay_authority_ready_score` passed (`actual: 1` vs `expected: 1`, `passed: true`), which matches `gate_check_summary`, `blocked_diagnostic_contract`, and `honest_verdict`. No experimental evaluation occurred (`duration_s: 0.0`), so there are no corpus-walk rows, prefilter measurements, or baselines to audit.

## EVIDENCE
- `"schema": "blocked_gate_check_v1"`
- `"status": "blocked"`
- `"honest_verdict": "blocked_gate_check_failed"`
- `"blocked_at_layer": "conductor_pre_gate"`
- `"duration_s": 0.0`
- `"gate_check_summary": "gate-unsat(final): 1 of 2 gate(s) failed; first failure: exp9222-contract.contract_ready_score (actual=0 == expected=1)"`
- `"artifact_field": "contract_ready_score"`, `"expected": 1`, `"actual": 0`, `"passed": false`
- `"artifact_field": "replay_authority_ready_score"`, `"expected": 1`, `"actual": 1`, `"passed": true`

## RECOMMENDATION
KEEP

## experiment_9225_v785_selective_deferral_protocol.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Because this is a protocol-freeze, custody, and method-qualification receipt artifact making no comparative empirical claim, there is no scientific effect claim to refute. Its narrower readiness and integrity assertions would be refuted if upstream custody hashes mismatched, if `estimator_control_rows` showed Horvitz-Thompson IPW expectation diverging from `true_fixture_risk` for `alpha > 0` or claiming identifiability at `alpha = 0`, if `spline_control_rows` violated partition-of-unity or read labels, if stream audit-mask rows had failures/exclusions/censoring, or if owned validation/coverage checks failed. Any empirical claim of deferral gain, verifier benefit, or generalization would require actual model runs and held-out outcome evaluation against serious baselines, which this artifact lacks and explicitly disclaims.

## WAS THAT CHECKED
Yes for protocol readiness and mathematical controls; no empirical evaluation was run or claimed. Upstream custody was checked in `gate_check_summary`, estimator unbiasedness and `alpha = 0` unidentifiability were checked on synthetic fixtures in `estimator_control_rows`, label-free B-spline invariants were checked in `spline_control_rows`, pre-outcome stream audit draws were checked across all 150 units in `rows`, and owned code checks/coverage were verified in `validation_receipts` and `owned_statement_coverage` (with unrelated repository-wide test failure transparently segregated in `repository_health`). Model loading, verifier benefit, learning effects, and out-of-sample generalization were not checked because the artifact explicitly scopes them out.

## EVIDENCE
- `honest_verdict`: `"complete_null_v785_selective_deferral_protocol"`
- `scientific_effect_measured`: `false`
- `independent_generalization_score`: `0`
- `MODEL_SPECS`: `[]`, `current_llm_invocations`: `0`, `model_invocation_status`: `"not_requested_empty_model_specs"`, `inference_substrate_class`: `"no_model_load"`
- `claim_scope` -> `scope`: `"Selective-deferral feedback masks, inverse-propensity estimators and label-free cubic B-spline support-density diagnostics only; no LLM load, verifier benefit, learning effect or independent generalization."`
- `acceptance_gates` -> `selective_deferral_method` -> `principle`: `"Audited-deferral masks and inverse-propensity controls qualify a counterfactual method, not primary learning gain."`
- `rows`: single arm `arm`: `"selective_deferral_stream_mask"`, `metric`: `"audit_draw_u"`, `completed_count`: `150`, `failed_count`: `0`, `censored_count`: `0`, `excluded_count`: `0`
- `estimator_control_rows`: all 5 controls have `passed`: `true`, with `exact_ipw_brier_expectation`: `0.14500000000000002` matching `true_fixture_risk`: `0.14500000000000002` for `alpha`: `0.1` and `0.25`, and `identifiability_status`: `"unidentifiable"` at `alpha`: `0.0`
- `spline_control_rows`: all 4 controls have `labels_read`: `false`, `max_partition_of_unity_error`: `1.1102230246251565e-16`, `passed`: `true`
- `selective_deferral_ready_score`: `1`, `spline_support_diagnostic_ready_score`: `1`, `required_checks_passed`: `true`

## RECOMMENDATION
KEEP

## experiment_9240_v786_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Because this artifact is an administrative milestone capstone and gate-accounting receipt (`complete_blocked_declared_gate`) that explicitly disclaims any new scientific, comparative, generalization, activation, or publication claim, there is no comparative scientific claim to refute. Its administrative accounting would only be contradicted if its own rows showed a promoted positive scientific/generalization verdict despite blocked gates, if upstream declared-gate artifacts (`contract_ready_score`, `replay_authority_ready_score`, `corpus_scan_ready_score`) had actually existed and passed while being reported as missing/blocked, or if the truncated tail of the artifact (fields listed in `field_principles` after `forward_qualification_summary`, such as `task_outcome_rows`, `gate_failure_rows`, `unmet_gates`, and `reconciliation_ready_score`) asserted a positive claim not visible in the provided excerpt.

## WAS THAT CHECKED
Yes, for all visible gate-accounting and branch-status sections: `current_milestone_accounting`, `declared_gate_rows`, `corpus_scan_summary`, `forward_qualification_summary`, `conformal_risk_summary`, `ces_summary`, `collection_partition`, `cost_summary`, `arc_summary`, and `claim_scope` consistently record missing upstream artifacts, scheduler skips, zero executed scientific predictions, disqualified historical offline ARC controls with zero value added, and no confirmed branch results. No comparative scientific hypothesis test was run because upstream prerequisites were blocked, and the truncated tail fields after `forward_qualification_summary` (`task_outcome_rows`, `gate_failure_rows`, `unmet_gates`, `reconciliation_ready_score`) could not be inspected from this excerpt.

## EVIDENCE
- `"honest_verdict": "complete_blocked_declared_gate"`
- `"confirmed_branch_results": []`
- `"activation_authorized": false`
- `"current_science_paper_ready": false`
- `"external_publication_authorized": false`
- `"independent_generalization_score": 0`
- `"current_llm_invocations": 0`
- `"executed": 1`, `"skipped": 13`, `"scientific_success_inferred": false`
- `"attempted_scientific_answers": 0`, `"execution_deficit_answers": 900`
- `"attempted": 0`, `"completed": 0`, `"unstarted": 3963`
- `"fourteen terminal V786 administrative outcomes"`
- `"unmeasured human science, secondary CES, and descriptive ambiguity/deferral/spline-support/conformal branches"`
- `"Hashed primitive denominators bind accounting; administrative readiness grants no new scientific claim."`
- `"Exact retained operands and executed receipts support accounting; absent science stays blocked"`
- `"operand_status": "missing_file"`, `"passed": false`
- `"status": "unmeasured"`
- `"verdict_class": "disqualified"`, `"value_added": false`

## RECOMMENDATION
KEEP

## experiment_9241_v787_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative scientific or methodological claim to refute: this is an infrastructure/contract-binding and receipt-replay scaffolding artifact that explicitly disclaims any measured scientific benefit or generalization and reports a blocked administrative outcome (`complete_blocked_v787_contract`, `contract_ready_score: 0`). The only thing that could contradict this administrative receipt would be if the underlying per-item `rows` or `validation_receipts` showed all 14 contract objects and gates passing despite the blocked status, or if the artifact claimed readiness without actually running its contract and replay checks.

## WAS THAT CHECKED
No comparative evaluation was run, because none is claimed: no model was loaded, zero LLM invocations occurred, and zero independent scientific groups were evaluated. On the administrative scaffolding side, six consumer test shards (`consumer_shard_rows`) and SHA256 contract-input checks (`gate_check_summary`, with 1645 entries elided) were executed, and the contract readiness score was evaluated to `0` (`attempted_count: 7`, `failed_count: 7`, `censored_count: 7`, `completed_count: 0`, with only 7 entries in `full_task_objects` against the 14-object contract condition). Because the artifact excerpt is truncated mid-`historical_input_roots`, the artifact's `rows`, `validation_receipts`, `required_checks_passed`, and top-level `honest_verdict` fields are not visible in the excerpt (the pre-pass notes supply `honest_verdict = "complete_blocked_v787_contract"`).

## EVIDENCE
- `"track": "infrastructure"`
- `"inference_substrate_class": "no_model_load"`
- `"MODEL_SPECS": []`
- `"current_llm_invocations": 0`
- `"effective_independent_groups": 0`
- `"independent_generalization_score": 0`
- `"contract_ready_score": 0`
- `"attempted_count": 7`
- `"completed_count": 0`
- `"failed_count": 7`
- `"censored_count": 7`
- `"activation_authorized": false`
- `"current_science_paper_ready": false`
- `"external_publication_authorized": false`
- `"scope": "Administrative contract and original methods only; no measured human benefit or ARC solve. Cannot close GAP-ORACLE-DISTINCT or GAP-DETECTOR-AUROC-4208."`
- `"condition": "unmeasured; fixtures and historical receipts grant no independent benefit"`

## RECOMMENDATION
KEEP

## experiment_9242_corpus_prefilter.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This artifact is a pre-gate blocking receipt and makes no experimental or comparative claim about qualifying a lossless byte prefilter. Its blocking record would only be contradicted if one or both evaluated upstream gates had `passed: true` (i.e., `actual` equal to `expected` value `1`) while still reporting a blocked gate failure, or if the artifact reported executed prefilter measurements contradicting a `blocked` status.

## WAS THAT CHECKED
Yes, for the gate-receipt itself: `gates_evaluated` lists both checked upstream gates (`contract_ready_score` and `replay_authority_ready_score` on `exp9241-contract`), each showing `actual: 0`, `expected: 1`, and `passed: false`, which is consistent with `gate_check_summary`, `blocked_diagnostic_contract`, `duration_s: 0.0`, and `status: "blocked"`. No experimental evaluation, corpus walk, or comparator baseline was run because execution halted at `conductor_pre_gate`.

## EVIDENCE
- `"schema": "blocked_gate_check_v1"`
- `"status": "blocked"`
- `"duration_s": 0.0`
- `"honest_verdict": "blocked_gate_check_failed"`
- `"blocked_at_layer": "conductor_pre_gate"`
- `"gate_check_summary": "gate-unsat(final): 2 of 2 gate(s) failed; first failure: exp9241-contract.contract_ready_score (actual=0 == expected=1)"`
- `"artifact_field": "contract_ready_score"`
- `"artifact_field": "replay_authority_ready_score"`
- `"expected": 1`
- `"actual": 0`
- `"passed": false`

## RECOMMENDATION
KEEP

## experiment_9244_v787_conformal_risk_protocol.json

**CLAIM_SUPPORTED**

## VERDICT
CLAIM_SUPPORTED

## THE HEADLINE CLAIM
This protocol-qualification artifact reports an honest null on empirical model, verifier, learning, or generalization effects (`complete_null_v787_conformal_risk_protocol`) while claiming only that its qid-cluster conformal risk bounds (vs. naive i.i.d. sibling bounds), CA-OCRC randomized delayed-feedback IPW tracker (vs. biased drop-on-full tracking), 900-response Mycelium provenance ledger, and upstream custody checks satisfy their pre-specified readiness gates.

## WHAT WOULD REFUTE IT
The claim would be refuted by any of the following observations in the artifact's own data:
1. Any assertion of empirical verifier benefit, learning gain, or independent generalization despite zero model loads and exposed `TEST100` label accounting.
2. In `coverage_control_rows`: `cluster_valid_coverage < 0.95` for `clopper_pearson_cluster` or `betting_e_process_cluster` at `rho >= 0.7`, or `naive_sibling_coverage >= 0.95` at `rho >= 0.7` (showing the simulation failed to stress sibling correlation), or any `passed: false`.
3. In `edge_control_rows`: finite bounds instead of `null`/`unavailable` at `accepted_qids = 0`, non-monotone slack or admissibility across `alpha_risk` in `[0.05, 0.1, 0.15]`, or upper bounds below `1.0` when all accepted qids are errors.
4. In `ca_ocrc_control_rows`: non-zero `ipw_risk_bias`, large `mc_ipw_risk_bias`, a `[corrupted_risk_lower, corrupted_risk_upper]` envelope failing to bracket `true_risk`, `debiased_corrupted_ipw_risk != true_risk`, or drop-on-full comparators matching IPW unbiasedness.
5. In `provenance_summary`, `rows`, or `gate_check_summary`: non-zero `span_level_present`, `invented_span_labels`, `model_judge_overrides`, `failed_count`, `censored_count`, or `excluded_count`; split counts not summing to 900 responses / 300 qids; or failed upstream SHA-256 custody checks.
6. Unchecked/missing visibility in this excerpt: 9 of 12 `ca_ocrc_control_rows`, 126 of 129 `gate_check_summary` entries, and most of the 900 `rows` are elided or truncated, so any hidden failure in those elided rows would refute the corresponding readiness gate.

## WAS THAT CHECKED
Yes, across the artifact's own control tables, ledger summaries, and scope declarations:
- Empirical scope and leakage (Q2/Q3): Explicitly checked and bounded; no LLM is loaded (`MODEL_SPECS` is empty, `current_llm_invocations` is 0), `verifier_is_oracle` is false, and although `TEST100` human-label counts appear in the provenance ledger, the artifact explicitly restricts scope to `exposed_development_only` and sets `independent_generalization_score` to 0 rather than claiming held-out generalization.
- Conformal risk coverage and edge cases (Q1/Q4): Checked in `coverage_control_rows` (all 8 rows shown across `rho` in `{0.0, 0.7, 0.85, 1.0}` with 2000 trials each) where cluster-valid bounds stay in `[0.975, 1.0]` while the naive i.i.d. sibling comparator drops to `0.791–0.8615` at `rho >= 0.7`, and in `edge_control_rows` (all 3 rows shown), all passing.
- CA-OCRC delayed-feedback IPW tracking (Q1/Q4): Checked in `ca_ocrc_control_rows` (3 of 12 rows visible, 9 elided) showing exact IPW bias `0.0`, small Monte Carlo bias (`-0.00125` to `+0.00925`), corruption bracketing `[0.27, 0.37]` around `0.3` with exact debiasing `0.3`, against biased `naive_risk` (`0.0`) and `deterministic_ipw_risk` (`0.4–0.5`).
- Provenance accounting, custody, and row validity (Q5): Checked in `provenance_summary` (exact 900-response / 300-qid reconciliation across `TRAIN150`, `CAL50`, `TEST100` with zero exclusions/failures/censors and zero invented span labels or model-judge overrides), `gate_check_summary` (3 of 129 SHA-256 checks visible, 126 elided), and `owned_statement_coverage` (100% on the 3 owned files). The failing global pytest suite in `repository_health` (`passed: false`, `exit_code: 1`) is explicitly logged and separated from the owned/affected validation gate rather than hidden. Note that this artifact does not inspect elided rows directly in the truncated JSON excerpt, nor does it test any real model predictions (which it explicitly disclaims).

## EVIDENCE
- `"honest_verdict": "complete_null_v787_conformal_risk_protocol"`
- `"independent_generalization_score": 0`
- `"exposure_scope": "exposed_development_only"`
- `"label_exposure_scope": "exposed_development_only"`
- `"MODEL_SPECS": []`, `"current_llm_invocations": 0`, `"inference_substrate_class": "no_model_load"`
- `"scope": "Qid-cluster conformal risk bounds, CA-OCRC capacity-constrained delayed-feedback IPW tracking and Mycelium provenance accounting only; no LLM load, verifier benefit, learning effect or independent generalization."`
- `"conformal_risk_ready_score": 1`, `"ca_ocrc_ready_score": 1`, `"provenance_audit_ready_score": 1`, `"required_checks_passed": true`
- `coverage_control_rows`: `"trials": 2000`, `"nominal_target": 0.95`, `"rho": 0.7` (`"cluster_valid_coverage": 0.999`, `"cluster_valid_coverage": 1.0`, `"naive_sibling_coverage": 0.8615`), `"rho": 0.85` (`"cluster_valid_coverage": 0.993`, `"cluster_valid_coverage": 1.0`, `"naive_sibling_coverage": 0.85`), `"rho": 1.0` (`"cluster_valid_coverage": 0.975`, `"cluster_valid_coverage": 1.0`, `"naive_sibling_coverage": 0.791`), `"passed": true`
- `edge_control_rows`: `"control_id": "zero_accepted_qids_unavailable"`, `"status": "unavailable"`, `"clopper_pearson_cluster_upper": null`, `"betting_e_process_cluster_upper": null`; `"control_id": "zero_errors_monotone_alpha_risk_grid"`, `"clopper_pearson_cluster_upper": 0.058155079116972264`, `"betting_e_process_cluster_upper": 0.1099178754885417`; `"control_id": "all_errors_upper_bound_unity"`, `"clopper_pearson_cluster_upper": 1.0`, `"betting_e_process_cluster_upper": 1.0`
- `ca_ocrc_control_rows`: `"true_risk": 0.3`, `"exact_ipw_risk": 0.3`, `"ipw_risk_bias": 0.0`, `"mc_ipw_risk_bias": 0.008083333333333331`, `"mc_ipw_risk_bias": -0.0012499999999999734`, `"mc_ipw_risk_bias": 0.009250000000000036`, `"naive_risk": 0.0`, `"naive_risk_bias": -0.3`, `"deterministic_ipw_risk": 0.5`, `"deterministic_ipw_risk": 0.4`, `"corrupted_epsilon": 0.05`, `"corrupted_ipw_risk": 0.32`, `"corrupted_risk_lower": 0.27`, `"corrupted_risk_upper": 0.37`, `"debiased_corrupted_ipw_risk": 0.3`, `"passed": true`
- `provenance_summary` and top-level counts: `"intended_count": 900`, `"attempted_count": 900`, `"completed_count": 900`, `"failed_count": 0`, `"censored_count": 0`, `"excluded_count": 0`, `"effective_independent_groups": 300`, `"answer_level_present": 900`, `"span_level_absent": 900`, `"span_level_present": 0`, `"invented_span_labels": 0`, `"model_judge_overrides": 0`
- `owned_statement_coverage` and `repository_health`: `"complete": true`, `"percent": 100.0`, `"passed": false`, `"note": "Unrelated repository-wide suite recorded separately from owned/affected checks"`

## RECOMMENDATION
KEEP
