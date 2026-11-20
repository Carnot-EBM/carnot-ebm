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

## experiment_9208_v784_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no comparative or scientific claim to refute: this is an administrative contract-binding and receipt-replay scaffolding artifact (`experiment_9208_v784_contract.json`) that explicitly disclaims any measured scientific benefit, generalization, activation, or publication readiness. Even on its own administrative readiness axis, it reports zero readiness (`contract_ready_score: 0`) and is disqualified (`complete_disqualified_v784_contract` in the pre-pass notes); that administrative disqualification would only be contradicted if every required contract, replay-authority, and validation gate in its omitted check logs had actually passed.

## WAS THAT CHECKED
No comparative hypothesis was evaluated because none is made: no model was loaded or invoked, zero tokens were generated, zero independent groups were evaluated, and `claim_scope` restricts the artifact to administrative task binding and frozen protocol/receipt authority. For the administrative disqualification itself, the first three shown `gate_check_summary` entries pass, while the remaining 1,585 `gate_check_summary` entries and the later JSON keys (`rows`, `required_checks_passed`, `validation_receipts`, and top-level `honest_verdict`) are elided or truncated after `historical_model_provenance`, so the exact failing administrative gate is absent from this excerpt.

## EVIDENCE
- `"track": "infrastructure"`
- `"title": "Exp9208: Qualify protocol-owner replay and bind fourteen current tasks"`
- `"inference_substrate_class": "no_model_load"`
- `"scope": "Administrative contract and original methods only; no measured human benefit or ARC solve. Cannot close GAP-ORACLE-DISTINCT or GAP-DETECTOR-AUROC-4208."`
- `"allowed_claims"` contains `"administrative task binding"`, `"authenticated original human and method authority"`, and `"frozen secondary label sensitivity protocol"`
- `"scientific_benefit"` has `"condition": "unmeasured; fixtures and historical receipts grant no independent benefit"`
- `"contract_ready_score": 0`
- `"activation_authorized": false`
- `"current_science_paper_ready": false`
- `"external_publication_authorized": false`
- `"independent_generalization_score": 0`
- `"exposure_scope": "exposed_development"`
- `"MODEL_SPECS": []`
- `"current_llm_invocations": 0`
- `"generated_tokens": 0`
- `"effective_independent_groups": 0`

## RECOMMENDATION
KEEP

## experiment_9209_corpus_prefilter.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This is a conductor pre-gate receipt recording that experiment 9209 did not execute, not an experimental or comparative claim about qualifying a lossless byte prefilter. The receipt itself would only be falsified if the recorded gate evaluations contradicted the blocked status (for example, if gates had `passed: true` or `actual: 1` while reporting `blocked_gate_check_failed`), or if the referenced upstream file `experiment_9208_v784_contract.json` actually contained `1` for `contract_ready_score` and `replay_authority_ready_score` despite this receipt recording `0`.

## WAS THAT CHECKED
Yes for internal consistency: `gates_evaluated` lists 2 of 2 upstream gates with `actual: 0`, `expected: 1`, and `passed: false`, which directly matches `gate_check_summary`, `blocked_diagnostic_contract`, `status: "blocked"`, and `honest_verdict: "blocked_gate_check_failed"`. No for external verification of the upstream file contents (the artifact provides `artifact_path` and `artifact_sha256` but not the upstream file itself), and no prefilter evaluation occurred because execution stopped at `conductor_pre_gate` (`duration_s: 0.0`).

## EVIDENCE
- `"schema": "blocked_gate_check_v1"`
- `"status": "blocked"`
- `"duration_s": 0.0`
- `"honest_verdict": "blocked_gate_check_failed"`
- `"blocked_at_layer": "conductor_pre_gate"`
- `"gate_check_summary": "gate-unsat(final): 2 of 2 gate(s) failed; first failure: exp9208-contract.contract_ready_score (actual=0 == expected=1)"`
- `"artifact_field": "contract_ready_score"`
- `"artifact_field": "replay_authority_ready_score"`
- `"expected": 1`
- `"actual": 0`
- `"passed": false`

## RECOMMENDATION
KEEP

## experiment_9211_v784_label_ambiguity.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This is a pre-prediction scaffolding, custody, and data-audit receipt artifact that explicitly measures no scientific effect and makes no comparative or generalization claim (no models loaded, no verifier evaluated, and no comparator arms run on real data). Its recorded readiness and audit assertions would be falsified if an upstream custody SHA-256 failed, if the LPHB release actually contained identifiable SI/CF or expressed-uncertainty columns (contradicting zero identifiable rows), if the affine Brier contrast reducer disagreed with exact brute-force endpoints or broke shared-label constraints across repeated streams, or if any owned validation check or coverage gate failed. Because no model predictions are scored, the artifact naturally lacks empirical verifier-vs-baseline comparisons and held-out generalization tests (and the provided excerpt is truncated inside the fifth validation receipt, cutting off the sixth receipt and trailing fields).

## WAS THAT CHECKED
Yes, for every scaffolding and audit precondition it records: upstream artifact hashes are checked in `gate_check_summary`, all 900 responses across 3 v2 taxonomy targets are audited in `target_identifiability_rows` and `target_identifiability_summary` alongside negative controls in `target_identifiability_controls`, all 900 response masks and annotator counts are tallied with zero exclusions in `rows` and `support_counts`, the affine Brier reducer is verified against exact brute-force enumeration and a 10-stream shared-label control in `endpoint_control_rows`, and owned checks/coverage are recorded in `validation_receipts` and `owned_statement_coverage` (with the unrelated failing global suite disclosed separately in `repository_health`). Empirical comparative baselines and held-out model evaluation were not checked because none were run or claimed.

## EVIDENCE
- `"honest_verdict": "complete_null_v784_label_ambiguity"`
- `"scientific_effect_measured": false`
- `"MODEL_SPECS": []`
- `"current_llm_invocations": 0`
- `"inference_substrate": "cpu_cached_human_annotation_reduction"`
- `"inference_substrate_class": "no_model_load"`
- `"model_invocation_status": "not_requested_empty_model_specs"`
- `"independent_generalization_score": 0`
- `"exposure_scope": "exposed_development_only"`
- `"scope": "Target identifiability and exact label-ambiguity bounds only; no LLM load, verifier benefit, learning effect or independent generalization."`
- `"role": "evaluator_only_after_prediction_seal"`
- `"principle": "Repeated measurements share one underlying label; fixture probabilities establish no verifier benefit."`
- `"label_ambiguity_ready_score": 1`
- `"target_identifiability_ready_score": 1`
- `"identifiable_factual_error_rows": 900`
- `"identifiable_expressed_uncertainty_rows": 0`
- `"identifiable_si_cf_rows": 0`
- `"category_metrics": "unavailable: zero identifiable SI/CF or expressed-uncertainty rows in original human release"`
- `"attempted_count": 900`, `"completed_count": 900`, `"excluded_count": 0`, `"failed_count": 0`, `"censored_count": 0`
- `"required_checks_passed": true`

## RECOMMENDATION
KEEP

## experiment_9221_v784_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This artifact is a milestone capstone accounting and gate-receipt aggregation (`complete_blocked_declared_gate`) that explicitly disclaims any new scientific or comparative claim, so there is no scientific headline to refute. Its administrative accounting would only be contradicted if any retained upstream receipt showed that the declared gates (`contract_ready_score == 1`, `replay_authority_ready_score == 1`, `corpus_scan_ready_score == 1`) actually passed, that scientific model runs were executed (`attempted > 0` or `current_llm_invocations > 0`), or if the truncated tail of the JSON (fields alphabetically after `field_principles`, such as `outcome_rows`, `gate_failure_rows`, or top-level `honest_verdict`) asserted a positive scientific or comparative claim inconsistent with the blocked/unmeasured summaries.

## WAS THAT CHECKED
Yes, within the visible portion of the artifact and the mechanical pre-pass. The artifact checks and records the upstream gate operands (`declared_gate_rows`, `corpus_scan_summary`), zero executed scientific work (`collection_partition`, `cost_summary`, `ces_summary`, `current_llm_invocations`, `MODEL_SPECS`), explicit disqualification of offline ARC routing controls (`arc_summary`), and explicit scoping of the single qualified null/no-model-load branch (`confirmed_branch_results`, `claim_scope`), while setting all publication and activation flags to false. Because the JSON excerpt is truncated inside `field_principles`, fields after `field_principles` (such as `outcome_rows`, `gate_failure_rows`, and top-level `required_checks_passed`) could not be directly inspected here, though the mechanical pre-pass confirms `honest_verdict = "complete_blocked_declared_gate"` and `inference_substrate = "aggregation_from_upstream_artifacts"`.

## EVIDENCE
- `"activation_authorized": false`, `"current_science_paper_ready": false`, `"external_publication_authorized": false`
- `"MODEL_SPECS": []`, `"current_llm_invocations": 0`, `"effective_independent_groups": 0`, `"independent_generalization_score": 0`
- `"Hashed primitive denominators bind accounting; administrative readiness grants no new scientific claim."`
- `"Frozen conditions establish accounting only; secondary evidence never rescues primary science"`
- `"allowed_claims"` limited to `"fourteen terminal administrative outcomes"`, `"authenticated historical V782/V783 failures and Exp9208 replay-authority disqualification"`, `"independently qualified Exp9211 target-identifiability and affine label-ambiguity method without model predictions"`, and `"unmeasured human science, secondary CES and two label sensitivity branches"`
- `"executed": 3`, `"skipped": 11`, `"scientific_success_inferred": false`
- `"attempted_scientific_answers": 0`, `"execution_deficit_answers": 900`
- `"attempted": 0`, `"completed": 0`, `"intended": 3963`, `"unstarted": 3963`
- `"honest_verdict": "complete_blocked_declared_gate"` backed by `"artifact_field": "contract_ready_score"` (`"observed": 0`, `"passed": false`), `"artifact_field": "replay_authority_ready_score"` (`"observed": 0`, `"passed": false`), and `"artifact_field": "corpus_scan_ready_score"` (`"observed": "missing_field"`, `"passed": false`)
- `"verdict_class": "disqualified"` and `"value_added": false` in `arc_summary`; `"status": "unmeasured"` and `"secondary_signal_score": 0` in `ces_summary`; `"honest_verdict": "complete_null_v784_label_ambiguity"` in `confirmed_branch_results`

## RECOMMENDATION
KEEP

## experiment_9222_v785_contract.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
This is an infrastructure scaffolding and receipt-replay contract artifact that makes no comparative, empirical, or scientific value claim, and its administrative readiness status is explicitly blocked (`complete_blocked_v785_contract` in the pre-pass notes) with `contract_ready_score: 0`. The record would only be contradicted if the artifact asserted scientific benefit, held-out generalization, or contract/activation readiness despite zero completed obligations, five failures, nine censored slots, and zero model invocations, or if it silently excluded failing checks to manufacture a pass. (Note that the provided artifact text is truncated inside `historical_model_provenance`, so downstream fields such as `rows`, `validation_receipts`, `required_checks_passed`, and `replay_authority_ready_score` are absent from the excerpt.)

## WAS THAT CHECKED
Yes, at the contract-metadata and mechanical pre-pass level. The artifact explicitly restricts itself to administrative task binding and authenticated historical authority with zero independent generalization and unmeasured scientific benefit, runs no models (`current_llm_invocations: 0`, `MODEL_SPECS: []`), carries `verifier_is_oracle = false`, excludes no checks (`excluded_count: 0`), and records a non-ready/blocked outcome (`contract_ready_score: 0`, `completed_count: 0`, `failed_count: 5`, `censored_count: 9`, `activation_authorized: false`, `current_science_paper_ready: false`, `external_publication_authorized: false`, and pre-pass `honest_verdict = "complete_blocked_v785_contract"`). The three visible `consumer_shard_rows` passed (`exit_code: 0`, `passed: true`) without overriding the blocked overall score, confirming failure was possible and not masked. Per-row failure details (`rows`, `validation_receipts`, `replay_authority_ready_score`, and the 1,625 elided entries of `gate_check_summary`) could not be inspected because of excerpt truncation, but no comparative claim exists to evaluate against them.

## EVIDENCE
- `"experiment_id": 9222`
- `"track": "infrastructure"`
- `"title": "Exp9222: Qualify partitioned consumer replay and bind fourteen V785 tasks"`
- `"allowed_claims"` -> `"administrative task binding"`, `"authenticated original human and method authority"`, `"frozen secondary label sensitivity protocol"`
- `"scope": "Administrative contract and original methods only; no measured human benefit or ARC solve. Cannot close GAP-ORACLE-DISTINCT or GAP-DETECTOR-AUROC-4208."`
- `"scientific_benefit"` -> `"condition": "unmeasured; fixtures and historical receipts grant no independent benefit"`
- `"exposure_scope": "exposed_development"`
- `"independent_generalization_score": 0`
- `"effective_independent_groups": 0`
- `"MODEL_SPECS": []`
- `"current_llm_invocations": 0`
- `"generated_tokens": 0`
- `"contract_ready_score": 0`
- `"attempted_count": 5`
- `"completed_count": 0`
- `"failed_count": 5`
- `"censored_count": 9`
- `"excluded_count": 0`
- `"activation_authorized": false`
- `"current_science_paper_ready": false`
- `"external_publication_authorized": false`

## RECOMMENDATION
KEEP

## experiment_9223_corpus_prefilter.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
There is no empirical or comparative claim to refute because the experiment did not run; this is a pre-gate receipt artifact. The receipt's only internal assertion—that execution was blocked because an upstream gate failed—would be contradicted only if all evaluated gates in `gates_evaluated` had `"passed": true`, or if the observed value matched the expected value for the cited failing gate. No corpus-walk, prefilter, or comparator measurements are present in this artifact.

## WAS THAT CHECKED
Yes, for receipt consistency: `gates_evaluated` lists both checked gates from `exp9222-contract` and confirms one passed (`replay_authority_ready_score`: actual 1 == expected 1) and one failed (`contract_ready_score`: actual 0 != expected 1), matching `blocked_reason`, `blocked_diagnostic_contract`, and `gate_check_summary`. No experimental checks on the lossless byte prefilter itself were run because execution stopped at `conductor_pre_gate`.

## EVIDENCE
- `"schema": "blocked_gate_check_v1"`
- `"status": "blocked"`
- `"duration_s": 0.0`
- `"honest_verdict": "blocked_gate_check_failed"`
- `"blocked_at_layer": "conductor_pre_gate"`
- `"gate_check_summary": "gate-unsat(final): 1 of 2 gate(s) failed; first failure: exp9222-contract.contract_ready_score (actual=0 == expected=1)"`
- `"artifact_field": "contract_ready_score"` with `"expected": 1`, `"actual": 0`, `"passed": false`
- `"artifact_field": "replay_authority_ready_score"` with `"expected": 1`, `"actual": 1`, `"passed": true`

## RECOMMENDATION
KEEP

## experiment_9225_v785_selective_deferral_protocol.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Because this is a protocol, mask-freezing, and custody-receipt artifact rather than a comparative empirical experiment, there is no comparative scientific claim to refute. What would refute its protocol-readiness and custody receipts would be: (1) any mismatched upstream SHA256 custody hash or missing source receipt, (2) biased Horvitz-Thompson IPW expectations for `alpha > 0` or failure to flag `alpha = 0` as unidentifiable in the synthetic estimator fixtures, (3) B-spline partition-of-unity errors, label leakage (`labels_read: true`), or invariance failures in the spline controls, (4) failed owned validation checks or incomplete owned statement coverage, or (5) any hidden positive empirical or generalization verdict despite zero model invocations and no evaluated rival arms. The artifact lacks model runs, comparative experimental arms, and held-out outcome scoring by design.

## WAS THAT CHECKED
Yes, for protocol and receipt integrity—no comparative empirical claim was tested because none is made. Upstream custody hashes are checked in `gate_check_summary`, synthetic IPW unbiasedness and `alpha = 0` unidentifiability are checked in `estimator_control_rows` (all 5 `passed: true`), B-spline partition-of-unity and label-free controls are checked in `spline_control_rows` (all 4 `passed: true`, `labels_read: false`), and owned unit/consumer/lint/type/spec checks are recorded in `validation_receipts` (all 6 `passed: true`) alongside `owned_statement_coverage` (`percent: 100.0` across the 3 owned files in `files`). Unrelated global test failures are recorded separately in `repository_health` (`passed: false`). Comparative model evaluation, rival baselines, and out-of-sample generalization were not checked and are explicitly excluded from scope.

## EVIDENCE
- `honest_verdict`: `"complete_null_v785_selective_deferral_protocol"`
- `scientific_effect_measured`: `false`
- `MODEL_SPECS`: `[]`, `current_llm_invocations`: `0`, `inference_substrate_class`: `"no_model_load"`, `model_invocation_status`: `"not_requested_empty_model_specs"`
- `independent_generalization_score`: `0`, `exposure_scope`: `"exposed_development_only"`, `label_exposure_scope`: `"exposed_development_only"`
- `claim_scope` / `scope`: `"Selective-deferral feedback masks, inverse-propensity estimators and label-free cubic B-spline support-density diagnostics only; no LLM load, verifier benefit, learning effect or independent generalization."`
- `rows`: `"arm": "selective_deferral_stream_mask"`, `"metric": "audit_draw_u"`, `"status": "completed"`, `"censor_reason": null`, `"exclusion_reason": null` (`intended_count`: `150`, `attempted_count`: `150`, `completed_count`: `150`, `failed_count`: `0`, `censored_count`: `0`, `excluded_count`: `0`)
- `estimator_control_rows`: `"passed": true` on `"full_release_omniscient_reference"`, `"escalation_only_selection_bias_and_unidentifiability"` (`"identifiability_status": "unidentifiable"`, `"ipw_estimate": null`), `"escalation_only_catastrophic_accepted_drift_unobservable"` (`"identifiability_status": "unidentifiable"`, `"ipw_estimate": null`), `"audited_deferral_alpha_0.10_ipw_unbiased"` (`"exact_ipw_brier_expectation": 0.14500000000000002`, `"true_fixture_risk": 0.14500000000000002`), and `"audited_deferral_alpha_0.25_ipw_unbiased"` (`"exact_ipw_brier_expectation": 0.14500000000000002`, `"true_fixture_risk": 0.14500000000000002`)
- `spline_control_rows`: `"passed": true`, `"labels_read": false`, `"basis_columns": 57`, `"max_partition_of_unity_error": 1.1102230246251565e-16` across all 4 control rows
- `required_checks_passed`: `true`, `selective_deferral_ready_score`: `1`, `spline_support_diagnostic_ready_score`: `1`
- `repository_health`: `"passed": false`, `"note": "Unrelated repository-wide suite recorded separately from owned/affected checks"`

## RECOMMENDATION
KEEP

## experiment_9240_v786_capstone.json

**NO_CLAIM**

## VERDICT
NO_CLAIM

## THE HEADLINE CLAIM
no claim

## WHAT WOULD REFUTE IT
Because this artifact is a milestone-level administrative accounting and receipt-aggregation capstone making no comparative scientific claim, there is no positive empirical claim to refute. Its administrative-blocked accounting would be contradicted only if its own rows showed passing declared gates, uncounted scientific or LLM executions, or confirmed/publication-ready scientific branches while still reporting a blocked verdict. For any future scientific claim on these branches (CES ranking, conformal risk, lossless corpus prefilter, forward qualification, or ARC solving), refutation would require actually running the unexecuted protocols against real rivals (`mean_entropy`, `max_entropy`, `probability_length_error_probability`, or live ARC environments) on unexposed evaluation data rather than the exposed LPHB development panel. Note also that the provided artifact excerpt is truncated inside `forward_qualification_summary`, omitting fields alphabetically after it (such as `outcome_rows`, `reconciliation_ready_score`, and `task_outcome_rows`).

## WAS THAT CHECKED
Yes, at the administrative-accounting level across the visible gate receipts and branch summaries (`declared_gate_rows`, `current_milestone_accounting`, `collection_partition`, `cost_summary`, `ces_summary`, `conformal_risk_summary`, `corpus_scan_summary`, `forward_qualification_summary`, and `arc_summary`). Every declared gate row shown failed on missing upstream files, 13 of 14 scheduled milestone tasks were skipped at the scheduler, zero LLM invocations and zero scientific answers/requests were executed, the offline ARC routing control was explicitly marked disqualified with no value added, and no comparative or generalization claim is asserted. No empirical model evaluation, baseline comparison, or held-out generalization test was performed in this artifact.

## EVIDENCE
- `"Hashed primitive denominators bind accounting; administrative readiness grants no new scientific claim."`
- `"fourteen terminal V786 administrative outcomes"`
- `"unmeasured human science, secondary CES, and descriptive ambiguity/deferral/spline-support/conformal branches"`
- `"confirmed_branch_results": []`
- `"current_science_paper_ready": false`
- `"external_publication_authorized": false`
- `"activation_authorized": false`
- `"scientific_success_inferred": false`
- `"executed": 1`, `"skipped": 13`, `"scheduler_log_skips": 13`, `"scheduler_pretest_log_skips": 2`, `"scheduler_upstream_retired_log_skips": 11`
- `"MODEL_SPECS": []`, `"current_llm_invocations": 0`
- `"attempted_scientific_answers": 0`, `"execution_deficit_answers": 900`
- `"attempted": 0`, `"completed": 0`, `"intended": 3963`, `"unstarted": 3963`
- `"independent_generalization_score": 0`, `"effective_independent_groups": 0`
- `"exposure_scope": "LPHB is exposed development; no independent human evaluation or generalization. Historical graph access and source authority remain blocked."`
- `"honest_verdict": "complete_blocked_declared_gate"`, `"observed": "missing_file"`, `"passed": false`
- `"reference_status": "missing_file"`, `"status": "unmeasured"`, `"secondary_signal_score": 0`
- `"verdict_class": "disqualified"`, `"required_checks_passed": false`, `"value_added": false`, `"win_reached": false`, `"scope": "Offline deterministic method controls; no independent solve evidence"`

## RECOMMENDATION
KEEP
