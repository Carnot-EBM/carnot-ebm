# Self-Learning Lineage Decision

Run date: 20260507
Decision: Narrow headline pivot selected
Honest verdict: self_learning_headline_pivot_selected_exp1447_verified_growth_only

## Evidence

- Continuous Self-Learning remains a core architectural goal in
  `research-program.md`, but headline wording needs persisted growth rather
  than replay-only improvement.
- Exp 1433 reported `fr11_self_learning_v6_dvi_v3_no_positive_growth_non_headline` with
  `self_learning_delta_overall=0` and
  `headline_result_allowed=False`.
- Exp 1447 reported `fr11_v7_positive_verified_growth_persisted_without_forgetting` with
  `exp1447_delta_overall=156`,
  `memory_entries_added=156`, and
  `nonforgetting_rate=1.0`.
- Exp 1449 reported `ltlzinc_temporal_adapter_ready_verified_cases_only_no_training` with
  `temporal_cases_generated=24`. It is a
  supporting benchmark feed, not a standalone headline claim.

## Non-Headline Lineage Pattern

- Exp 1273: smoke_only_not_headline (SIGNAL)
- Exp 1281: milestone_99_12_of_14_criteria_met (SIGNAL)
- Exp 1288: online_verifier_feedback_neutral_non_headline (SIGNAL)
- Exp 1295: milestone_100_5_of_14_criteria_met (SIGNAL)
- Exp 1303: online_memory_policy_improved_non_headline (SIGNAL)
- Exp 1315: cerce_nonforgetting_preserved_improved_non_headline (SIGNAL)
- Exp 1344: failure_type_memory_policy_dvi_ready_replay_non_headline (NOISE)
- Exp 1350: milestone_104_9_of_12_criteria_met_carryforward_required (SIGNAL)
- Exp 1358: verifier_selected_memory_replay_only_dvi_ready_non_headline (SIGNAL)
- Exp 1362: publication_hold_active_local_evidence_does_not_support_ebt_arm_kona_or_hardware_claims (SIGNAL)
- Exp 1364: milestone_105_carryforward_confirms_thinking_mode_missing_structural_tag_blocker_terminal_certificate_required_semantic_chain_gated_self_... (SIGNAL)
- Exp 1389: milestone_107_13_of_14_criteria_met_arxiv_not_submitted_hold_lift_recommended_not_confirmed_grpo_no_improvement_fr11_dvi_only (NOISE)
- Exp 1402: milestone_108_12_of_13_criteria_met_arxiv_submission_not_attempted_pipeline_semantic_fixed_full_pipeline_below_headline_grpo_retired_bipr... (NOISE)
- Exp 1424: milestone_109_10_of_13_criteria_met_threshold_met_but_repair_dvi_fr11_and_pipeline_carry_forward (SIGNAL)
- Exp 1433: fr11_self_learning_v6_dvi_v3_no_positive_growth_non_headline (SIGNAL)
- Exp 1438: milestone_110_12_of_14_criteria_met_threshold_met_repair_dvi_prm_dpo_latent_positive_fr11_growth_and_rtl_source_carry_forward (SIGNAL)

## Next Allowed Shape

`next_allowed_experiment_shape` in the Exp 1459 artifact is the governing
boundary. Only the Exp 1447 verified memory-policy mechanism may advance as a
narrow headline pivot, and it must report the required metrics plus
`nonforgetting_rate >= 0.99`. Replay-only, adapter-only,
and broad self-learning claims remain outside headline scope.
