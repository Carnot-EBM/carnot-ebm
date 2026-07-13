# Anomaly Escalations — human-triage queue (frame-violating anomalies the loop did NOT auto-reconcile)

Per Deep Think P3 / Anomaly-Escalation (scripts/anomaly_escalation.py). Each entry is a result the loop would normally silently reconcile as a dead-end, but which trips a frame-violating signal (infra false-negative / fabrication flag / invariant regression). **Human triage: dead-end or breadcrumb?**

## experiment_3680_dependency_aware_dual_condition_integrity.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - INVARIANT regression: production_auroc_dependency_aware_vs_frozen_headline_delta=0.012228 drifted from frozen 0.9131
- artifact: results/experiment_3680_dependency_aware_dual_condition_integrity.json

## experiment_3681_g2_reproducer_prep_operator_refreeze_package.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_3681_g2_reproducer_prep_operator_refreeze_package.json

## experiment_3728_bounded_checkpointed_train_ebt_and_ar.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - method may not have genuinely run (cumulative_steps_trained=0, a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_3728_bounded_checkpointed_train_ebt_and_ar.json

## experiment_3735_bounded_train_chunk2_resume.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - method may not have genuinely run (cumulative_steps_trained=0, a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_3735_bounded_train_chunk2_resume.json

## experiment_2090_crane_humaneval.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_2090_crane_humaneval.json

## experiment_911_drift_probe_tier0i.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_911_drift_probe_tier0i.json

## experiment_2481_capstone_v239.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_2481_capstone_v239.json

## experiment_1877_artifact_contract_normalization.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_1877_artifact_contract_normalization.json

## experiment_2101_interwhen.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_2101_interwhen.json

## experiment_2110_casal_pinet.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['GATE_PASSED_WITHOUT_DATA'] on a non-negative verdict
- artifact: results/experiment_2110_casal_pinet.json

## experiment_3778_fr11_self_learning_v18_tier2_constraint_memory.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - INVARIANT regression: auroc_within_frozen_ci=True drifted from frozen 0.9131
- artifact: results/experiment_3778_fr11_self_learning_v18_tier2_constraint_memory.json

## experiment_3813_fr11_v21_fast_path_robustness.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - INVARIANT regression: effective_auroc_in_frozen_ci=False drifted from frozen 0.9131
- artifact: results/experiment_3813_fr11_v21_fast_path_robustness.json

## experiment_3814_publication_gate_regression_confirmation.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - INVARIANT regression: frozen_fover_auroc_unchanged=True drifted from frozen 0.9131
- artifact: results/experiment_3814_publication_gate_regression_confirmation.json

## experiment_3819_latent_symbol_bridge.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_3819_latent_symbol_bridge.json

## experiment_3821_latent_symbol_bridge_unblocked.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_3821_latent_symbol_bridge_unblocked.json

## experiment_3834_archive_v352_activate_v353.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - INVARIANT regression: frozen_fover_auroc_unchanged=True drifted from frozen 0.9131
- artifact: results/experiment_3834_archive_v352_activate_v353.json

## experiment_3840_publication_gate_regression_confirmation.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - INVARIANT regression: frozen_fover_auroc_unchanged=True drifted from frozen 0.9131
- artifact: results/experiment_3840_publication_gate_regression_confirmation.json

## experiment_3839_edlm_kill_gate.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_3839_edlm_kill_gate.json

## experiment_3843.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - INVARIANT regression: frozen_fover_auroc_unchanged=True drifted from frozen 0.9131
- artifact: results/experiment_3843.json

## experiment_3844_verifier_error_independence_scissor_at_scale.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_3844_verifier_error_independence_scissor_at_scale.json

## experiment_3857_archive_v355_activate_v356.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - INVARIANT regression: frozen_fover_auroc_unchanged=True drifted from frozen 0.9131
- artifact: results/experiment_3857_archive_v355_activate_v356.json

## experiment_3863_graph_verifier_facts_complementarity_v2.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_3863_graph_verifier_facts_complementarity_v2.json

## experiment_3865_ldt_lattice_margin_sharpening_v2.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - INVARIANT regression: frozen_fover_auroc_unchanged=True drifted from frozen 0.9131
- artifact: results/experiment_3865_ldt_lattice_margin_sharpening_v2.json

## experiment_3868_capstone_v356.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - INVARIANT regression: frozen_fover_auroc_unchanged=True drifted from frozen 0.9131
- artifact: results/experiment_3868_capstone_v356.json

## experiment_3885_moat_scissor_in_distribution.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_3885_moat_scissor_in_distribution.json

## experiment_3886_graph_grounding_fact_verifier_defabricated.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_3886_graph_grounding_fact_verifier_defabricated.json

## experiment_3896_graph_grounding_verifier_harness.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_3896_graph_grounding_verifier_harness.json

## experiment_3905_cost_instrumented_verify_harness.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_3905_cost_instrumented_verify_harness.json

## experiment_3922_hardware_continuity_consolidated.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_3922_hardware_continuity_consolidated.json

## experiment_efficiency_panel_boosted_draft.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_efficiency_panel_boosted_draft.json

## experiment_efficiency_panel_boosted_boosted_draft.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_efficiency_panel_boosted_boosted_draft.json

## experiment_3926_valid_efficiency_head_to_head.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_3926_valid_efficiency_head_to_head.json

## experiment_3928_moat_scissor_replication.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_3928_moat_scissor_replication.json

## experiment_4022_decentralization_gated.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4022_decentralization_gated.json

## experiment_4031_offarc_transfer_smoke_raw.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4031_offarc_transfer_smoke_raw.json

## experiment_4031_offarc_transfer_build.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4031_offarc_transfer_build.json

## experiment_4037_decentralization_stronger_base_smoke.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4037_decentralization_stronger_base_smoke.json

## experiment_4044_offarc_transfer_power_smoke_raw.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4044_offarc_transfer_power_smoke_raw.json

## experiment_4044_offarc_transfer_power_build.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4044_offarc_transfer_power_build.json

## experiment_4048_decentralization_moe_base_smoke.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4048_decentralization_moe_base_smoke.json

## experiment_4047_decentralization_moe_base_build.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4047_decentralization_moe_base_build.json

## experiment_4056_offarc_power_evalplus_build.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_4056_offarc_power_evalplus_build.json

## experiment_4077_verifier_reward_rft_corpus_build.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_4077_verifier_reward_rft_corpus_build.json

## experiment_4078_verifier_reward_rft_train_launch.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_4078_verifier_reward_rft_train_launch.json

## experiment_4080_sudoku_rft_positive_control.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4080_sudoku_rft_positive_control.json

## experiment_4083_verifier_registry_gaps_hygiene.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_4083_verifier_registry_gaps_hygiene.json

## experiment_4095_verifier_registry_gaps_hygiene.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4095_verifier_registry_gaps_hygiene.json

## experiment_4100_trm_verifier_rft_conditional.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4100_trm_verifier_rft_conditional.json

## experiment_4109_carnot_verifier_graft_sudoku.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4109_carnot_verifier_graft_sudoku.json

## experiment_4116_sudoku_extreme_resume_pass1.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_4116_sudoku_extreme_resume_pass1.json

## experiment_4119_carnot_verifier_graft_sudoku.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT', 'GATE_PASSED_WITHOUT_DATA'] on a non-negative verdict
- artifact: results/experiment_4119_carnot_verifier_graft_sudoku.json

## experiment_4120_thirteenth_game_explore_first.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['GATE_PASSED_WITHOUT_DATA'] on a non-negative verdict
- artifact: results/experiment_4120_thirteenth_game_explore_first.json

## experiment_4122_verifier_registry_gaps_hygiene.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4122_verifier_registry_gaps_hygiene.json

## experiment_4128_carnot_verifier_graft_sudoku.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT', 'GATE_PASSED_WITHOUT_DATA'] on a non-negative verdict
- artifact: results/experiment_4128_carnot_verifier_graft_sudoku.json

## experiment_4131_verifier_registry_gaps_hygiene.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4131_verifier_registry_gaps_hygiene.json

## experiment_4135_sudoku_accumulate_pass1_fixed_lr.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4135_sudoku_accumulate_pass1_fixed_lr.json

## experiment_4136_sudoku_accumulate_pass2_fixed_lr.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT', 'GATE_PASSED_WITHOUT_DATA'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_4136_sudoku_accumulate_pass2_fixed_lr.json

## experiment_4137_sudoku_accumulate_pass3_fixed_lr.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT', 'GATE_PASSED_WITHOUT_DATA'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_4137_sudoku_accumulate_pass3_fixed_lr.json

## experiment_4138_sudoku_accumulate_pass4_convergence_check.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_4138_sudoku_accumulate_pass4_convergence_check.json

## experiment_4146_sudoku_accumulate_pass1_epochfix.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_4146_sudoku_accumulate_pass1_epochfix.json

## experiment_4147_sudoku_accumulate_pass2.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT', 'GATE_PASSED_WITHOUT_DATA'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_4147_sudoku_accumulate_pass2.json

## experiment_4148_sudoku_accumulate_pass3.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT', 'GATE_PASSED_WITHOUT_DATA'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_4148_sudoku_accumulate_pass3.json

## experiment_4149_sudoku_accumulate_pass4_convergence.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_4149_sudoku_accumulate_pass4_convergence.json

## experiment_4150_decisive_verifier_graft_sudoku.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4150_decisive_verifier_graft_sudoku.json

## experiment_4151_arc_incremental_progress.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['GATE_PASSED_WITHOUT_DATA'] on a non-negative verdict
- artifact: results/experiment_4151_arc_incremental_progress.json

## experiment_4153_verifier_registry_gaps_hygiene.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_4153_verifier_registry_gaps_hygiene.json

## experiment_4156_archive_v384_activate_v385.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_4156_archive_v384_activate_v385.json

## experiment_4157_baseline_harvest_contiguous_continue.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_4157_baseline_harvest_contiguous_continue.json

## experiment_4158_verifier_rerank_recovery_moat.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4158_verifier_rerank_recovery_moat.json

## experiment_4159_decisive_verifier_reward_graft.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4159_decisive_verifier_reward_graft.json

## experiment_4163_verifier_registry_gaps_hygiene.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_4163_verifier_registry_gaps_hygiene.json

## experiment_4169_arc_incremental_progress.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['GATE_PASSED_WITHOUT_DATA'] on a non-negative verdict
- artifact: results/experiment_4169_arc_incremental_progress.json

## experiment_4178_gap3_stage1_model_native_arc_energy.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_4178_gap3_stage1_model_native_arc_energy.json

## experiment_4188_sovereign_local_generator_gap4_self_distill.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4188_sovereign_local_generator_gap4_self_distill.json

## experiment_4193_verifier_registry_gaps_hygiene.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4193_verifier_registry_gaps_hygiene.json

## experiment_4197_verifier_reward_phase0_headroom_harness_build.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4197_verifier_reward_phase0_headroom_harness_build.json

## experiment_4198_verifier_reward_3arm_rft_launch.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4198_verifier_reward_3arm_rft_launch.json

## experiment_4200_certified_arc_corpus_distill_lift.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4200_certified_arc_corpus_distill_lift.json

## diffusiongemma_energy_prior_gguf.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/diffusiongemma_energy_prior_gguf.json

## experiment_4184_archive_v387_activate_v388.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['CIRCULAR_MOAT_OVERCLAIM'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_4184_archive_v387_activate_v388.json

## diffusiongemma_energy_prior_extracted.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/diffusiongemma_energy_prior_extracted.json

## experiment_4211_verifier_as_reward_finish_synchronous.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4211_verifier_as_reward_finish_synchronous.json

## oracle_distinct_diffusion_surprisal_detector.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/oracle_distinct_diffusion_surprisal_detector.json

## experiment_4212_certified_arc_corpus_distill_lift.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4212_certified_arc_corpus_distill_lift.json

## experiment_4216_verifier_registry_gaps_hygiene.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_4216_verifier_registry_gaps_hygiene.json

## experiment_4222_verifier_reward_lora_harness_fix_smoke.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4222_verifier_reward_lora_harness_fix_smoke.json

## experiment_4223_verifier_as_reward_3arm_synchronous.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4223_verifier_as_reward_3arm_synchronous.json

## experiment_4234_verifier_reward_lora_harness_real_training_smoke.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_4234_verifier_reward_lora_harness_real_training_smoke.json

## experiment_4247_verifier_reward_offline_harness_retire_livelora.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_4247_verifier_reward_offline_harness_retire_livelora.json

## experiment_4260_diffusiongemma_energy_guided_preflight.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_4260_diffusiongemma_energy_guided_preflight.json

## experiment_4266_verifier_registry_gaps_hygiene.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4266_verifier_registry_gaps_hygiene.json

## experiment_4307_arc_incremental_progress_new_game.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['GATE_PASSED_WITHOUT_DATA'] on a non-negative verdict
- artifact: results/experiment_4307_arc_incremental_progress_new_game.json

## experiment_4310_verifier_registry_gaps_hygiene.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4310_verifier_registry_gaps_hygiene.json

## arc_e3_agent_cd82.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/arc_e3_agent_cd82.json

## arc_gap_fill_su15.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/arc_gap_fill_su15.json

## arc_gap_fill_r11l.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/arc_gap_fill_r11l.json

## experiment_4342_self_learning_action_role_cross_game_encoder.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_4342_self_learning_action_role_cross_game_encoder.json

## experiment_4346_capstone_v401.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['CIRCULAR_MOAT_OVERCLAIM'] on a non-negative verdict
- artifact: results/experiment_4346_capstone_v401.json

## experiment_4347_archive_v401_activate_v402.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['CIRCULAR_MOAT_OVERCLAIM'] on a non-negative verdict
- artifact: results/experiment_4347_archive_v401_activate_v402.json

## experiment_4410_registry_gaps_hygiene_gap4_guard.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_4410_registry_gaps_hygiene_gap4_guard.json

## experiment_4421_config_rule_solve_unseen.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4421_config_rule_solve_unseen.json

## experiment_4425_config_rule_vocabulary_transfer.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4425_config_rule_vocabulary_transfer.json

## experiment_4433_example_conditioned_win_induction.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4433_example_conditioned_win_induction.json

## experiment_4472_variant_generic_transfer_benchmark_v4.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_4472_variant_generic_transfer_benchmark_v4.json

## experiment_4482_nocov_default_lint.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_4482_nocov_default_lint.json

## experiment_4490_human_replay_frame_change_predictor.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_4490_human_replay_frame_change_predictor.json

## experiment_4526_integration_8game_gate.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4526_integration_8game_gate.json

## experiment_4534_energy_trust_next_level_routing.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4534_energy_trust_next_level_routing.json

## experiment_4560_integration_8game_gate.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4560_integration_8game_gate.json

## experiment_4563_positive_control_failed_guard.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_4563_positive_control_failed_guard.json

## experiment_1035_dualgpu_rocm_v3.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_1035_dualgpu_rocm_v3.json

## experiment_3734_fix_harness_and_bounded_train_chunk1.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_3734_fix_harness_and_bounded_train_chunk1.json

## experiment_3787_p1_discrete_search_adjudication_v3_retry.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_3787_p1_discrete_search_adjudication_v3_retry.json

## experiment_4590_capstone_v423.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_4590_capstone_v423.json

## experiment_hazard_l3_calibration.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['ARC_OUTER_LOOP_SOLVE'] on a non-negative verdict
- artifact: results/experiment_hazard_l3_calibration.json

## experiment_hazard_l3_calibration_tu93_L2.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['ARC_OUTER_LOOP_SOLVE'] on a non-negative verdict
- artifact: results/experiment_hazard_l3_calibration_tu93_L2.json

## experiment_4604_world_model_trust_energy.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4604_world_model_trust_energy.json

## experiment_4610_world_model_trust_pass_rate_metric.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['WORLD_MODEL_TRUST_DEGENERACY'] on a non-negative verdict
- artifact: results/experiment_4610_world_model_trust_pass_rate_metric.json

## arc_loop_solve_lf52.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['ARC_OUTER_LOOP_SOLVE'] on a non-negative verdict
- artifact: results/arc_loop_solve_lf52.json

## arc_loop_solve_r11l.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['ARC_OUTER_LOOP_SOLVE'] on a non-negative verdict
- artifact: results/arc_loop_solve_r11l.json

## arc_loop_solve_vc33.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['ARC_OUTER_LOOP_SOLVE'] on a non-negative verdict
- artifact: results/arc_loop_solve_vc33.json

## experiment_4723_capstone_v434.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_4723_capstone_v434.json

## experiment_4735_capstone_v435.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_4735_capstone_v435.json

## experiment_4749_structured_engine_vs_freeform.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - FALSE_NEGATIVE_RISK flag ['FALSE_NEGATIVE_RISK'] on a null/negative verdict — verify a positive control ran
- artifact: results/experiment_4749_structured_engine_vs_freeform.json

## experiment_4752_held_out_first_win_readiness.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4752_held_out_first_win_readiness.json

## experiment_4753_persist_transfer.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_4753_persist_transfer.json

## experiment_4754_submitted_agent_config.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4754_submitted_agent_config.json

## experiment_4791_structural_energy_s2_offpath_trust_gate.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DEGENERATE_CANDIDATE_POOL'] on a non-negative verdict
- artifact: results/experiment_4791_structural_energy_s2_offpath_trust_gate.json

## experiment_4801_structural_energy_s2v2_diverse_trust_gate.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DEGENERATE_CANDIDATE_POOL'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_4801_structural_energy_s2v2_diverse_trust_gate.json

## experiment_4861_generation_wall_fork_probe.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_4861_generation_wall_fork_probe.json

## experiment_4883_inducer_ceiling_ab.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4883_inducer_ceiling_ab.json

## experiment_4917_heldout_first_win_readiness.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_4917_heldout_first_win_readiness.json

## arc_ige_llm_go_explore_ab.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/arc_ige_llm_go_explore_ab.json

## arc_relational_mask_deepen_ab.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['perception-overclaim'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/arc_relational_mask_deepen_ab.json

## experiment_4938_self_play_verifier_checkpoint.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_4938_self_play_verifier_checkpoint.json

## experiment_headway_lp85_capture.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['ARC_OUTER_LOOP_SOLVE'] on a non-negative verdict
- artifact: results/experiment_headway_lp85_capture.json

## experiment_5003_lora_ebm_scorer_musr.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_5003_lora_ebm_scorer_musr.json

## experiment_5004_uprm_replication.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_5004_uprm_replication.json

## experiment_5006_moat_second_corpus.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5006_moat_second_corpus.json

## experiment_4162_sota_ingestion_verifier_moat_guidance.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['MOAT_CLAIM_RIGOR'] on a non-negative verdict
- artifact: results/experiment_4162_sota_ingestion_verifier_moat_guidance.json

## experiment_4170_sota_ingestion_verifier_moat_guidance.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['MOAT_CLAIM_RIGOR'] on a non-negative verdict
- artifact: results/experiment_4170_sota_ingestion_verifier_moat_guidance.json

## experiment_4967_capstone_v457.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['MOAT_CLAIM_RIGOR'] on a non-negative verdict
- artifact: results/experiment_4967_capstone_v457.json

## experiment_4978_capstone_v458.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['MOAT_CLAIM_RIGOR'] on a non-negative verdict
- artifact: results/experiment_4978_capstone_v458.json

## experiment_4989_capstone_v459.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['MOAT_CLAIM_RIGOR'] on a non-negative verdict
- artifact: results/experiment_4989_capstone_v459.json

## experiment_5000_capstone_v460.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['MOAT_CLAIM_RIGOR'] on a non-negative verdict
- artifact: results/experiment_5000_capstone_v460.json

## experiment_5008_moat_oracle_distinct_lint.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['MOAT_CLAIM_RIGOR'] on a non-negative verdict
- artifact: results/experiment_5008_moat_oracle_distinct_lint.json

## experiment_5043_sota_gguf_judge_preflight.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_5043_sota_gguf_judge_preflight.json

## experiment_5044_second_corpus_candidate_cache.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5044_second_corpus_candidate_cache.json

## experiment_5046_vpr_process_reward_repair.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5046_vpr_process_reward_repair.json

## experiment_5049_second_corpus_confirmation.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5049_second_corpus_confirmation.json

## experiment_5051_verifier_trace_self_learning.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5051_verifier_trace_self_learning.json

## experiment_5054_arc_live_path_self_discovery.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT', 'LEVER_EXERCISE_EVIDENCE_DEGENERATE'] on a non-negative verdict
- artifact: results/experiment_5054_arc_live_path_self_discovery.json

## experiment_5057_gate_state_preflight_v465.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5057_gate_state_preflight_v465.json

## experiment_5058_sota_candidate_refresh_inwriting.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5058_sota_candidate_refresh_inwriting.json

## experiment_5059_d1_sota_refresh_audit.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5059_d1_sota_refresh_audit.json

## experiment_5070_sota_ingestion_backfill_v466.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5070_sota_ingestion_backfill_v466.json

## experiment_5085_llamacpp_logprob_endpoint_bringup_v467.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5085_llamacpp_logprob_endpoint_bringup_v467.json

## experiment_5088_temporal_consistency_prm_v467.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT', 'MOAT_CLAIM_RIGOR'] on a non-negative verdict
- artifact: results/experiment_5088_temporal_consistency_prm_v467.json

## experiment_5089_pbit_guided_cdcl_bridge_v467.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['MOAT_CLAIM_RIGOR'] on a non-negative verdict
- artifact: results/experiment_5089_pbit_guided_cdcl_bridge_v467.json

## experiment_5090_static_csr_constrained_decoding_v467.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5090_static_csr_constrained_decoding_v467.json

## experiment_5099_beaver_prefix_bound_verifier_v468.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5099_beaver_prefix_bound_verifier_v468.json

## experiment_5100_constrainprompt_code_assurance_v468.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5100_constrainprompt_code_assurance_v468.json

## experiment_5104_constrained_decoding_semantic_risk_audit_v468.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5104_constrained_decoding_semantic_risk_audit_v468.json

## experiment_5105_fr11_severa_guarded_memory_v468.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5105_fr11_severa_guarded_memory_v468.json

## experiment_5111_fover_in_domain_pool_v469.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_5111_fover_in_domain_pool_v469.json

## experiment_5119_sota_endpoint_rootcause_v469.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_5119_sota_endpoint_rootcause_v469.json

## experiment_5121_capstone_v469.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_5121_capstone_v469.json

## experiment_5123_v470_source_scope_audit.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5123_v470_source_scope_audit.json

## experiment_5125_structured_reasoning_pool_v470.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5125_structured_reasoning_pool_v470.json

## experiment_5126_distributional_energy_ranker_v470.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5126_distributional_energy_ranker_v470.json

## experiment_5135_v471_source_scope_audit.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5135_v471_source_scope_audit.json

## experiment_5138_ets_ebd_guided_decoding_v471.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_5138_ets_ebd_guided_decoding_v471.json

## experiment_5143_openskill_k2v_self_learning_v471.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5143_openskill_k2v_self_learning_v471.json

## experiment_5145_capstone_v471.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_5145_capstone_v471.json

## experiment_5161_gap4_sandbox_smoke.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5161_gap4_sandbox_smoke.json

## experiment_5161_gap4_protocol_execution_pilot_v473.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5161_gap4_protocol_execution_pilot_v473.json

## experiment_5173_diffusiongemma_energy_guided_diffusion_pilot_v474.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['CIRCULAR_MOAT_OVERCLAIM', 'MOAT_CLAIM_RIGOR'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_5173_diffusiongemma_energy_guided_diffusion_pilot_v474.json

## experiment_5178_hidden_state_verifier_pilot_v474.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5178_hidden_state_verifier_pilot_v474.json

## experiment_5211_gap4_sota_local_candidate_expansion_v477.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5211_gap4_sota_local_candidate_expansion_v477.json

## experiment_5215_arc_paw_amortization_gate_v477.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5215_arc_paw_amortization_gate_v477.json

## experiment_5226_veribmc_local_solver_feedback_pilot_v478.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5226_veribmc_local_solver_feedback_pilot_v478.json

## experiment_5241_arc_gated_live_patch_attempt_v479.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5241_arc_gated_live_patch_attempt_v479.json

## experiment_5256_capstone_v480.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_5256_capstone_v480.json

## experiment_5262_solver_grounded_constraint_extraction_v481.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5262_solver_grounded_constraint_extraction_v481.json

## experiment_5263_neuron_attention_energy_hallucination_probe_v481.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5263_neuron_attention_energy_hallucination_probe_v481.json

## experiment_5274_solver_constraint_extraction_retry_gated_v482.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_5274_solver_constraint_extraction_retry_gated_v482.json

## experiment_5296_sota_source_delta_v484.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5296_sota_source_delta_v484.json

## experiment_5301_ebt_spectral_step_control_diagnostic_v484.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5301_ebt_spectral_step_control_diagnostic_v484.json

## experiment_5318_smt_hint_validation_protocol_v485.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_5318_smt_hint_validation_protocol_v485.json

## experiment_5323_native_gguf_backend_flag_bisect_v486.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5323_native_gguf_backend_flag_bisect_v486.json

## experiment_5331_internal_energy_receipt_harness_v486.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5331_internal_energy_receipt_harness_v486.json

## experiment_5338_structured_output_protocol_calibration_v487.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5338_structured_output_protocol_calibration_v487.json

## experiment_5353_tokenprob_feature_audit_corrigendum_v488.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5353_tokenprob_feature_audit_corrigendum_v488.json

## experiment_5360_arc_perception_salience_levelup_attempt_v488.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_5360_arc_perception_salience_levelup_attempt_v488.json

## experiment_5480_arc_live_salience_levelup_v497.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_5480_arc_live_salience_levelup_v497.json

## experiment_5494_arc_live_trajectory_levelup_v498.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_5494_arc_live_trajectory_levelup_v498.json

## experiment_5506_hardware_multiboard_receipts_v499.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_5506_hardware_multiboard_receipts_v499.json

## experiment_5508_arc_live_perception_generation_levelup_v499.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_5508_arc_live_perception_generation_levelup_v499.json

## experiment_5521_arc_live_action_diverse_levelup.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_5521_arc_live_action_diverse_levelup.json

## experiment_5527_sota_hard_soft_panel_v2.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
- artifact: results/experiment_5527_sota_hard_soft_panel_v2.json

## experiment_5532_hardware_receipt_parser_repeatability.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5532_hardware_receipt_parser_repeatability.json

## experiment_5533_arc_strategy_routing_precheck.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5533_arc_strategy_routing_precheck.json

## experiment_5534_arc_strategy_routed_levelup.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag(s) ['DURATION_TOO_SHORT'] on a negative verdict (possible infra/fabrication artifact masquerading as a finding)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_5534_arc_strategy_routed_levelup.json

## experiment_5536_transition_v502.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_5536_transition_v502.json

## experiment_5548_arc_clean_live_levelup.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_5548_arc_clean_live_levelup.json

## experiment_5550_transition_v503.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_5550_transition_v503.json

## experiment_5562_arc_fsm_live_levelup.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_5562_arc_fsm_live_levelup.json

## experiment_5564_transition_v504.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - method may not have genuinely run (a precondition was False (method may have been infra-blocked)) — a 'bounded' verdict here could be an infra false-negative, not a real result (cf. Thesis-A exp3728)
- artifact: results/experiment_5564_transition_v504.json

## experiment_5588_tier3_induction_live_path_sanity_check.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5588_tier3_induction_live_path_sanity_check.json

## experiment_5589_tier3_induction_normal_budget_capability_check.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5589_tier3_induction_normal_budget_capability_check.json

## experiment_5593_goal_predicate_consistency_offline_sim_prototype.json
- Flagged a FRAME-VIOLATING ANOMALY (NOT auto-reconciled — human triage: dead-end or breadcrumb?)
  - CRITICAL adversarial flag ['DURATION_TOO_SHORT'] on a non-negative verdict
- artifact: results/experiment_5593_goal_predicate_consistency_offline_sim_prototype.json
