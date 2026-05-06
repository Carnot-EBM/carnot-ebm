# Milestone .110 Carry-Forward Manifest

Prior milestone: `2026.04.109`
Target milestone: `2026.04.110`
Run date: `20260506`

| track | prior evidence | .110 task | gate rule | retire-if-same-verdict rule |
|---|---|---|---|---|
| Repair executor v2 / full pipeline scale-up | exp1414-certificate-llm-repair-executor-v1 (results/experiment_1414_certificate_llm_repair_executor_v1.json): complete_repair_executor_no_successful_repairs [repaired_cases_successful=0, repaired_case_success_rate=0.0]; exp1419-fullscale-pipeline-v3-repair-executor (results/experiment_1419_fullscale_pipeline_v3_repair_executor.json): not_headline_full_pipeline_below_0_40 [cases_evaluated=200, repaired_cases_successful=0, full_pipeline_pass_rate=0.305] | exp1427-repair-executor-rejection-ledger, exp1428-dccd-schema-constrained-repair-v2, exp1429-mcmc-constrained-repair-candidate-search, exp1430-prm-guided-repair-selector, exp1431-fullscale-pipeline-v4-micro-gated | exp1431 stays gated until exp1428 proves repaired_case_success_rate > 0.0 and exp1430 is ready; no 200-case rerun before micro evidence. | Retire the exact zero-accepted-repair path if repair v2 again produces complete_repair_executor_no_successful_repairs or the exp1419 verdict. |
| DVI v3 nonforgetting | exp1415-dvi-v3-1508-fresh-cases (results/experiment_1415_dvi_v3_1508_fresh_cases.json): dvi_v3_blocked_nonforgetting_below_gate [nonforgetting_rate=0.968604, dvi_v3_deployed=False] | exp1432-dvi-v3-nonforgetting-replay-balanced | Deploy only when dvi_v3_deployed=true, nonforgetting_rate >= 0.99, and AUROC does not regress below DVI v2. | Retire this DVI v3 repair variant if the same dvi_v3_blocked_nonforgetting_below_gate verdict recurs. |
| FR-11 v6 continuous self-learning | exp1418-fr11-self-learning-v6-dvi-v3 (retro-only evidence): gate_blocked_upstream_dvi_v3_not_deployed [source metrics unavailable] | exp1433-fr11-self-learning-v6-dvi-v3-gated | Launch only after exp1432 records dvi_v3_deployed=true. | Do not retire FR-11 on an upstream DVI gate block; keep it gated until deployable DVI evidence exists. |
| DPO provenance | exp1420-dpo-verified-pairs-1508 (results/experiment_1420_dpo_verified_pairs_1508.json): gguf_dpo_unsupported_reranker_fallback_measured [dpo_full_finetune_performed=False, headline_result_allowed=False] | exp1435-dpo-headline-provenance-audit | Headline provenance requires direct local adapter/fine-tune support; otherwise the track is relabeled reranker-only. | If direct local DPO remains unsupported, retire headline DPO wording and carry only the reranker benchmark. |
| Full Python suite and spec-coverage debt | exp1421-test-suite-execution-debt-v1 (results/experiment_1421_test_suite_execution_debt_v1.json): focused_embedding_store_runtime_failures_fixed_collection_clean_targeted_tests_green_100pct_store_coverage_full_suite_and_preexisting_spec_coverage_debt_remain [remaining_debt=True] | exp1426-test-suite-remaining-debt-cluster-map | Diagnostic scope only: map remaining clusters and recommend one next bounded fix without reopening the fixed embedding-store cluster. | Do not retire the whole test-debt track on another red full-suite result; split by named failure cluster. |
| PRM label completion | exp1423-process-reward-model-v1-fover-1508 (results/experiment_1423_process_reward_model_v1_fover_1508.json): prmv1_trained_on_available_step_labels_with_478_promoted_traces_missing_local_labels [training_traces_used=1030, missing_trace_labels=478] | exp1434-fover-prm-label-completion-v2 | Fill at least 478 missing labels or write an exact residual blocker ledger before any 1508-trace PRM claim. | Retire the 1508-trace PRM claim if labels remain missing without a blocker ledger; keep measured PRM v1 as partial evidence only. |

## Forbidden Exact Reruns

- exp1419 200-case full-scale pipeline rerun without nonzero accepted repair evidence is forbidden. Required unlock evidence: exp1428.repaired_case_success_rate > 0.0 and downstream validation uses the exp1431 micro-gated path before any larger scale run.

## No-Change Confirmation

- scripts/research_conductor.py: no activation-audit changes needed
- research-roadmap.yaml: no activation-audit changes needed
