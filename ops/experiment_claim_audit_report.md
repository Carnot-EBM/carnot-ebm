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
| NO_CLAIM | 8 |

## experiment_9207_v783_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Because this artifact is an administrative capstone and receipt aggregation documenting a gate-blocked milestone rather than making a comparative scientific or generalization claim, there is no scientific claim to refute. What would contradict its administrative accounting status (`complete_blocked_declared_gate`) would be rows showing that the declared upstream gates actually passed (`contract_ready_score == 1`, `corpus_scan_ready_score == 1`, `source_ready_score == 1`), that scientific evaluations actually ran and produced confirmed branch results, or that the artifact claimed positive scientific value despite unmeasured downstream branches. All comparative scientific evaluations (CES vs. entropy and probability-length baselines, lossless corpus-prefilter parity, forward validation obligations, service-cost measurements, and live ARC solve evidence) were not run and remain absent due to the upstream gate block.

## WAS THAT CHECKED
Yes for the administrative gate-block accounting: the artifact records the failed declared-gate operands and scheduler skips in `declared_gate_rows`, `corpus_scan_summary`, and `forward_qualification_summary`, reports zero attempted scientific answers/requests across `collection_partition`, `ces_summary`, and `cost_summary`, and preserves the historical `arc_summary` as disqualified with no value added. No comparative scientific claim was tested because the upstream gate failure prevented execution of the corpus-prefilter, forward-validation, collection, CES, and cost branches, and the artifact explicitly disclaims any new scientific claim.

## EVIDENCE
- `"honest_verdict": "complete_blocked_declared_gate"`
- `"allowed_claims"`: `"fourteen terminal administrative outcomes"`, `"authenticated historical failures"`, `"unmeasured human science, secondary CES and two label sensitivity branches"`
- `"Hashed primitive denominators bind accounting; administrative readiness grants no new scientific claim."`
- `"confirmed_branch_results": []`
- `"independent_generalization_score": 0`
- `"activation_authorized": false`
- `"current_science_paper_ready": false`
- `"external_publication_authorized": false`
- `"executed": 2`, `"skipped": 12`, `"scientific_success_inferred": false`
- `"current_llm_invocations": 0`, `"attempted_scientific_answers": 0`, `"unstarted": 3963`
- `"artifact_field": "contract_ready_score"`, `"expected": 1`, `"observed": 0`, `"passed": false`
- `"status": "blocked"`
- `"status": "unmeasured"`, `"censor_reason": "missing_reference_and_scientific_predictions"`
- `"verdict_class": "disqualified"`, `"value_added": false`

## RECOMMENDATION
KEEP

## experiment_9208_v784_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative scientific or empirical claim to refute: this is an administrative contract-binding and receipt-replay scaffolding artifact (`honest_verdict = "complete_disqualified_v784_contract"` in the mechanical pre-pass) that executes no model runs, evaluates no rival methods, and explicitly disclaims scientific benefit, independent generalization, publication readiness, and contract readiness. For its non-claim / disqualified administrative status to be contradicted, the artifact would have to assert a positive scientific/method claim or readiness (`contract_ready_score = 1`, `activation_authorized = true`, or `current_science_paper_ready = true`), or its omitted rows and gate checks would have to contradict the recorded disqualification.

## WAS THAT CHECKED
Yes for the absence of any scientific or positive readiness claim: the visible artifact fields explicitly record zero model loads/invocations, zero independent groups, zero generalization score, zero contract readiness score, `false` authorization/paper-readiness flags, `exposed_development` scope, and an unmeasured scientific-benefit gate. For the underlying administrative disqualification itself, the specific failing gate records are not visible in the provided excerpt because 1,585 entries of `gate_check_summary` are elided and the artifact truncates inside `historical_model_provenance` before `rows`, `validation_receipts`, and `required_checks_passed`; however, because the artifact makes no comparative claim and claims no readiness pass, it remains a scaffolding/receipt artifact (`NO_CLAIM`).

## EVIDENCE
- `"experiment_id": 9208`
- `"title": "Exp9208: Qualify protocol-owner replay and bind fourteen current tasks"`
- `"track": "infrastructure"`
- `"inference_substrate_class": "no_model_load"`
- `"MODEL_SPECS": []`
- `"current_llm_invocations": 0`
- `"generated_tokens": 0`
- `"effective_independent_groups": 0`
- `"contract_ready_score": 0`
- `"activation_authorized": false`
- `"current_science_paper_ready": false`
- `"external_publication_authorized": false`
- `"exposure_scope": "exposed_development"`
- `"independent_generalization_score": 0`
- `"scope": "Administrative contract and original methods only; no measured human benefit or ARC solve. Cannot close GAP-ORACLE-DISTINCT or GAP-DETECTOR-AUROC-4208."`
- `"allowed_claims"`: `"administrative task binding"`, `"authenticated original human and method authority"`, `"frozen secondary label sensitivity protocol"`
- `"scientific_benefit"` / `"condition": "unmeasured; fixtures and historical receipts grant no independent benefit"`

## RECOMMENDATION
KEEP

## experiment_9209_corpus_prefilter.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This is a pre-gate block receipt recording that Experiment 9209 did not execute because two upstream precondition gates failed; there is no empirical or comparative claim to refute. The receipt itself would only be contradicted if its own `gates_evaluated` rows showed passing gates while reporting a blocked status, or if it asserted a positive experimental result despite `duration_s` being zero and the experiment never running.

## WAS THAT CHECKED
Yes. Both entries in `gates_evaluated` (`contract_ready_score` and `replay_authority_ready_score` on `exp9208-contract`) were checked against `expected` (`1`) with `op` `"=="`, both observed `0`, and both recorded `passed` as `false`, matching `gate_check_summary`, `status`, and `honest_verdict`. No prefilter evaluation or corpus walk was run or claimed.

## EVIDENCE
`"schema": "blocked_gate_check_v1"`, `"status": "blocked"`, `"honest_verdict": "blocked_gate_check_failed"`, `"blocked_at_layer": "conductor_pre_gate"`, `"duration_s": 0.0`, `"gate_check_summary": "gate-unsat(final): 2 of 2 gate(s) failed; first failure: exp9208-contract.contract_ready_score (actual=0 == expected=1)"`, `"artifact_field": "contract_ready_score"`, `"artifact_field": "replay_authority_ready_score"`, `"expected": 1`, `"actual": 0`, `"passed": false`

## RECOMMENDATION
KEEP

## experiment_9211_v784_label_ambiguity.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This is a pre-prediction scaffolding, data-custody, and target-identifiability receipt artifact making no comparative scientific, model, or verifier claim (`MODEL_SPECS` is empty, `current_llm_invocations` is 0, `scientific_effect_measured` is false, and `honest_verdict` is a null token). For its bounded non-comparative audit and readiness receipts, refutation would occur if: (1) any authenticated upstream SHA-256 hash check failed; (2) the LPHB source release actually contained an explicit human expressed-uncertainty column or >=2 same-generator equivalent-prompt re-elicitations contradicting the zero identifiable SI/CF and expressed-uncertainty counts; (3) the affine Brier endpoint reducer (`reducer_min`/`reducer_max`) disagreed with exhaustive binary enumeration (`exact_min`/`exact_max`) or allowed per-observation label splitting across the 10-stream shared-label control; or (4) any owned validation check failed or rows were silently dropped/censored.

## WAS THAT CHECKED
Yes, for the bounded custody, target-identifiability, and reducer-correctness receipts: immutable-byte SHA-256 checks are recorded in `gate_check_summary`, all 900 responses across 300 independent qids are audited across 3 v2 targets in `target_identifiability_rows` and `target_identifiability_summary` alongside 5 negative/edge controls in `target_identifiability_controls`, reducer endpoints are checked against exact enumeration and a 10-stream shared-label control in `endpoint_control_rows`, and the unrelated failing repo-wide suite is honestly separated under `repository_health` (`passed: false`) without being claimed as passing. No comparative model or verifier evaluation was run or claimed (and the artifact lacks any model predictions, rival baselines, or held-out generalization metrics, which it explicitly disclaims). Note that the provided excerpt elides additional rows in `endpoint_control_rows`, `gate_check_summary`, `rows`, `source_references`, and `target_identifiability_rows` and is truncated mid-object in the 5th `validation_receipts` entry, so the 6th owned validation receipt is not visible in this excerpt.

## EVIDENCE
- `"honest_verdict": "complete_null_v784_label_ambiguity"`
- `"scientific_effect_measured": false`
- `"MODEL_SPECS": []`
- `"current_llm_invocations": 0`
- `"inference_substrate_class": "no_model_load"`
- `"model_invocation_status": "not_requested_empty_model_specs"`
- `"independent_generalization_score": 0`
- `"exposure_scope": "exposed_development_only"`
- `"scope": "Target identifiability and exact label-ambiguity bounds only; no LLM load, verifier benefit, learning effect or independent generalization."`
- `"principle": "Repeated measurements share one underlying label; fixture probabilities establish no verifier benefit."`
- `"role": "evaluator_only_after_prediction_seal"`
- `"category_metrics": "unavailable: zero identifiable SI/CF or expressed-uncertainty rows in original human release"`
- `"identifiable_factual_error_rows": 900`
- `"identifiable_expressed_uncertainty_rows": 0`
- `"identifiable_si_cf_rows": 0`
- `"label_ambiguity_ready_score": 1`
- `"target_identifiability_ready_score": 1`
- `"required_checks_passed": true`
- `"intended_count": 900`, `"attempted_count": 900`, `"completed_count": 900`, `"failed_count": 0`, `"censored_count": 0`, `"excluded_count": 0`
- `"observed_agreement_count": 271`, `"observed_disagreement_count": 29`, `"missing_a2_count": 600`
- `"control_id": "exhaustive_small_panel_observed_disagreement"`, `"exact_min": -0.22999999999999995`, `"reducer_min": -0.22999999999999998`, `"exact_max": 0.17`, `"reducer_max": 0.16999999999999998`, `"passed": true`
- `"control_id": "ten_stream_shared_label_retention"`, `"exact_min": -0.04000000000000001`, `"reducer_min": -0.03999999999999999`, `"forbidden_per_observation_min": -0.6800000000000002`, `"passed": true`

## RECOMMENDATION
KEEP

## experiment_9221_v784_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or scientific performance claim to refute; this is an administrative capstone aggregation and gate-block receipt (`complete_blocked_declared_gate`) that explicitly withholds scientific, activation, and publication authorization. Its administrative accounting would be contradicted if the declared upstream prerequisite gates (`contract_ready_score`, `replay_authority_ready_score`, `corpus_scan_ready_score`) had actually passed, if scientific model evaluations had been executed and omitted from the counts, or if any scientific authorization flag had been set true despite zero measured scientific trials.

## WAS THAT CHECKED
Yes, for administrative gate accounting: `declared_gate_rows`, `corpus_scan_summary`, `current_milestone_accounting`, `collection_partition`, `ces_summary`, `cost_summary`, `arc_summary`, and `confirmed_branch_results` check and record the failed upstream gate operands (`observed: 0` vs `expected: 1`, and `missing_field`), zero scientific invocations/answers, the disqualified offline ARC scope, and the single model-free null qualification (`experiment_9211`). No comparative scientific hypothesis was evaluated because all scientific branches remain blocked and unmeasured; note also that top-level JSON keys after `retired_schedule` are truncated in the provided excerpt (with top-level `honest_verdict` supplied by the mechanical pre-pass notes).

## EVIDENCE
- `"field_principles"` states `"Hashed primitive denominators bind accounting; administrative readiness grants no new scientific claim."` and `"Exact retained operands and executed receipts support accounting; absent science stays blocked"`
- `"claim_scope"` lists `"allowed_claims"` as `"fourteen terminal administrative outcomes"`, `"authenticated historical V782/V783 failures and Exp9208 replay-authority disqualification"`, `"independently qualified Exp9211 target-identifiability and affine label-ambiguity method without model predictions"`, and `"unmeasured human science, secondary CES and two label sensitivity branches"`, with `"independent_generalization_score": 0`
- `"activation_authorized": false`, `"current_science_paper_ready": false`, `"external_publication_authorized": false`, `"scientific_success_inferred": false`
- `"MODEL_SPECS": []`, `"current_llm_invocations": 0`, `"effective_independent_groups": 0`
- `"current_milestone_accounting"` records `"executed": 3`, `"skipped": 11`, `"scheduler_artifact_skips": 1`, `"scheduler_log_skips": 10`
- `"collection_partition"` records `"attempted_scientific_answers": 0`, `"execution_deficit_answers": 900`, `"execution_only": true`
- `"ces_summary"` records `"complete_score": 0`, `"secondary_signal_score": 0`, `"reference_status": "missing_file"`, and `"status": "unmeasured"` with `"passed": false` across `"ranking_bound_rows"`
- `"cost_summary"` records `"scientific_request_counts"` with `"attempted": 0`, `"completed": 0`, `"intended": 3963`, `"unstarted": 3963`
- `"corpus_scan_summary"` records `"status": "blocked"`, `"qualified_checker_correction": false`, and `"honest_verdict": "complete_blocked_declared_gate"`
- `"declared_gate_rows"` records `"contract_ready_score"` (`"expected": 1`, `"observed": 0`, `"passed": false`), `"replay_authority_ready_score"` (`"expected": 1`, `"observed": 0`, `"passed": false`), and `"corpus_scan_ready_score"` (`"expected": 1`, `"observed": "missing_field"`, `"passed": false`)
- `"arc_summary"` records `"verdict_class": "disqualified"`, `"required_checks_passed": false`, `"live_path_observed": false`, `"value_added": false`, and `"scope": "Offline deterministic method controls; no independent solve evidence"`
- `"confirmed_branch_results"` contains only `"experiment_id": 9211` with `"honest_verdict": "complete_null_v784_label_ambiguity"` and `"verdict_class": "null"`

## RECOMMENDATION
KEEP

## experiment_9222_v785_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or scientific headline claim to refute: this is an administrative contract-binding and receipt-replay scaffolding artifact (`complete_blocked_v785_contract`) that performs no model inference and explicitly disclaims scientific benefit, independent generalization, and readiness. Taken strictly as an administrative receipt, it would only be contradicted if it asserted contract readiness, activation, publication authorization, or scientific benefit despite its five failed and nine censored checks, or if the truncated downstream `rows` and `validation_receipts` sections contradicted the reported counts and blocked verdict.

## WAS THAT CHECKED
Yes, against all five review questions within the visible artifact and pre-pass notes:
1. Could it have failed: Yes—no comparative arm or model evaluation was executed, and the administrative contract gate explicitly failed/blocked (`contract_ready_score` is 0, 5 attempted / 5 failed / 0 completed / 9 censored, with pre-pass `honest_verdict = "complete_blocked_v785_contract"`), while the three visible `consumer_shard_rows` passed.
2. Oracle: Pre-pass reports `verifier_is_oracle = false`, and the artifact makes no verifier value claim (explicitly disclaiming closure of `GAP-ORACLE-DISTINCT` and `GAP-DETECTOR-AUROC-4208`).
3. Model exposure: No model was loaded or invoked (`MODEL_SPECS` is empty, zero LLM invocations and tokens), exposure is explicitly labeled `exposed_development`, and `independent_generalization_score` is 0.
4. Rival baseline: Not applicable—no method comparison is run; `diagnostic_protocol` and `entropy_reference_protocol` are frozen specifications only.
5. Row accounting: Top-level counts report zero exclusions (`excluded_count` is 0) alongside 5 failures and 9 censored items. Note that the provided artifact text is truncated inside `historical_model_provenance`, so downstream fields including `rows`, `validation_receipts`, `replay_authority_ready_score`, and the in-body `honest_verdict` are absent from the excerpt and could not be inspected row by row; however, because the artifact makes no scientific or comparative claim and reports a blocked administrative state, it falls under `NO_CLAIM`.

## EVIDENCE
- `"experiment_id": 9222`
- `"track": "infrastructure"`
- `"title": "Exp9222: Qualify partitioned consumer replay and bind fourteen V785 tasks"`
- `"scope": "Administrative contract and original methods only; no measured human benefit or ARC solve. Cannot close GAP-ORACLE-DISTINCT or GAP-DETECTOR-AUROC-4208."`
- `"allowed_claims"` contains `"administrative task binding"`, `"authenticated original human and method authority"`, and `"frozen secondary label sensitivity protocol"`
- `"condition": "unmeasured; fixtures and historical receipts grant no independent benefit"` (under `"scientific_benefit"`)
- `"contract_ready_score": 0`
- `"attempted_count": 5`, `"completed_count": 0`, `"failed_count": 5`, `"censored_count": 9`, `"excluded_count": 0`
- `"activation_authorized": false`, `"current_science_paper_ready": false`, `"external_publication_authorized": false`
- `"MODEL_SPECS": []`, `"current_llm_invocations": 0`, `"generated_tokens": 0`, `"effective_independent_groups": 0`, `"independent_generalization_score": 0`
- `"exposure_scope": "exposed_development"`
- `"consumer_shard_rows"` reports `"passed": true` and `"exit_code": 0` for `"shard_name": "direct_consumers_core"`, `"shard_name": "direct_consumers_v782_v783"`, and `"shard_name": "direct_consumers_v784"`

## RECOMMENDATION
KEEP

## experiment_9223_corpus_prefilter.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This is a pre-gate receipt artifact; the corpus prefilter experiment in the title never executed and no comparative or scientific claim is made. The receipt's own operational status (blocked at pre-gate) would be refuted only if all entries in `gates_evaluated` reported `passed` as `true` or if `failed_observed` satisfied `failed_operator` against `failed_expected`.

## WAS THAT CHECKED
Yes, in `gates_evaluated`: 1 of 2 upstream gates (`contract_ready_score` on `exp9222-contract`) failed with `actual` = 0 against `expected` = 1 (`passed` = false), while the second gate (`replay_authority_ready_score`) passed (`actual` = 1, `expected` = 1, `passed` = true). No prefilter qualification, baseline comparison, or corpus metrics were run because execution stopped at `conductor_pre_gate`.

## EVIDENCE
- `"schema": "blocked_gate_check_v1"`
- `"status": "blocked"`
- `"duration_s": 0.0`
- `"honest_verdict": "blocked_gate_check_failed"`
- `"blocked_at_layer": "conductor_pre_gate"`
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
There is no comparative or empirical scientific claim to refute: this is a pre-outcome protocol, custody, and method-qualification scaffolding artifact that explicitly disavows any model load, verifier benefit, learning effect, or generalization. Its operational readiness and protocol-qualification receipts would be falsified if upstream SHA256 custody hashes failed to match, if the synthetic Horvitz-Thompson IPW controls were biased for `alpha > 0` or failed to report `alpha = 0` as unidentifiable, if the cubic B-spline controls read labels or violated partition-of-unity, clipping, constant-feature, or monotone-transform invariants, if any of the 150 pre-outcome audit-draw rows were missing/failed/censored, or if any of the 6 owned validation checks or 100% owned-file statement coverage failed.

## WAS THAT CHECKED
Yes, for all protocol and scaffolding readiness conditions: upstream byte-hash custody (`gate_check_summary`), synthetic IPW unbiasedness and `alpha = 0` unidentifiability (`estimator_control_rows`), label-free B-spline partition-of-unity and invariance (`spline_control_rows`), completion of all 150 single-arm audit-draw rows (`rows`, `sample_size_budget`), and the 6 owned validation checks plus 100% owned-file coverage (`validation_receipts`, `owned_statement_coverage`) were checked and passed, while the failing unrelated repository-wide test suite is separately disclosed (`repository_health`). No empirical model evaluation, held-out generalization test, or rival-method comparison on real outcomes was run, which is absent by design because no comparative scientific claim is made. Note that the provided excerpt elides 118 entries in `gate_check_summary`, 147 entries in `rows`, and truncates inside the 6th `validation_receipts` entry (`spec_coverage`, after its `passed` and `exit_code` fields).

## EVIDENCE
- `honest_verdict`: `"complete_null_v785_selective_deferral_protocol"`
- `scientific_effect_measured`: `false`
- `independent_generalization_score`: `0`
- `claim_scope` / `scope`: `"Selective-deferral feedback masks, inverse-propensity estimators and label-free cubic B-spline support-density diagnostics only; no LLM load, verifier benefit, learning effect or independent generalization."`
- `acceptance_gates` / `selective_deferral_method` / `principle`: `"Audited-deferral masks and inverse-propensity controls qualify a counterfactual method, not primary learning gain."`
- `MODEL_SPECS`: `[]`, `current_llm_invocations`: `0`, `inference_substrate_class`: `"no_model_load"`, `model_invocation_status`: `"not_requested_empty_model_specs"`
- `exposure_scope`: `"exposed_development_only"`, `label_exposure_scope`: `"exposed_development_only"`
- `selective_deferral_ready_score`: `1`, `spline_support_diagnostic_ready_score`: `1`, `required_checks_passed`: `true`
- `intended_count`: `150`, `attempted_count`: `150`, `completed_count`: `150`, `failed_count`: `0`, `excluded_count`: `0`, `censored_count`: `0`
- `rows`: `arm` `"selective_deferral_stream_mask"`, `metric` `"audit_draw_u"`, `status` `"completed"`, `censor_reason` `null`, `exclusion_reason` `null`
- `estimator_control_rows`: `passed` `true` for `"full_release_omniscient_reference"`, `"escalation_only_selection_bias_and_unidentifiability"` (`identifiability_status` `"unidentifiable"`), `"escalation_only_catastrophic_accepted_drift_unobservable"` (`identifiability_status` `"unidentifiable"`), `"audited_deferral_alpha_0.10_ipw_unbiased"` (`exact_ipw_brier_expectation` `0.14500000000000002`, `true_fixture_risk` `0.14500000000000002`, `mc_ipw_brier_mean` `0.14818708333333333`), and `"audited_deferral_alpha_0.25_ipw_unbiased"` (`mc_ipw_brier_mean` `0.14560183333333335`)
- `spline_control_rows`: `passed` `true`, `labels_read` `false`, `basis_columns` `57`, `max_partition_of_unity_error` `1.1102230246251565e-16`
- `owned_statement_coverage`: `complete` `true`, `percent` `100.0`
- `repository_health`: `passed` `false`, `exit_code` `1`, `note` `"Unrelated repository-wide suite recorded separately from owned/affected checks"`

## RECOMMENDATION
KEEP
