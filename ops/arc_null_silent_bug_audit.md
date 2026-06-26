# ARC Null Silent-Bug Audit

- Verdict: `complete: silent_bug_audit_12_nulls_5_must_reopen`
- Nulls audited: `12`
- Must reopen: `5`
- A4 tautology verdict: `online_driver_arms_degenerate (no-op, must reopen)`
- Go-Explore `_frame_grid` fix confirmed: `True`

## Per-Null Verdicts

| Null | Verdict | Silent signatures | Evidence |
|---|---|---|---|
| `experiment_4628_dense_curiosity_progress_loop` | `trustworthy_null` | - | dense_curiosity_enabled_attempts=25, prediction_error_events=4691, edge_values=2849, state_coverage_delta=2.0 |
| `experiment_4640_goal_energy_generation_live` | `silent_bug_must_reopen` | no_op_goal_energy_cached_frame, byte_identical_goal_energy_and_baseline | goal_energy_measurement cloned 25 baseline attempts with goal_energy_neutral_on_cached_frame=True; uniform arm also cloned baseline |
| `experiment_4653_energy_fitness_qd_generation_live` | `silent_bug_must_reopen` | byte_identical_qd_search_random_arms, unchanged_candidate_pool | energy_qd/search/random arms share 4 identical attempts except arm label; winner_generated_count=0 with no distinct QD pool evidence |
| `experiment_4664_l2_goal_predicate_induction_live` | `trustworthy_null` | - | single-exemplar goal predicate evaluated: goal_predicate_satisfiable={'lp85': False, 'sc25': False}, l2_plan_len={'lp85': 0, 'sc25': 0}, lp85_bare_control=True |
| `experiment_4676_hierarchical_subgoal_search_live` | `silent_bug_must_reopen` | empty_subgoal_decomposition, subgoal_arm_noop | subgoal_decomposition=[] and per_subgoal_reachable=[]; target arm levels=[0, 0, 0] |
| `experiment_4688_controllable_novelty_proposal_policy_live` | `trustworthy_null` | - | controllable_novelty candidate_scores=4641, observed_effects=194, rnd_updates=194 |
| `experiment_4700_object_centric_perception_proposal_live` | `trustworthy_null` | - | object_centric_pool_nonempty; coverage=1.0, candidate_counts=[186, 186, 148, 148], augmented_candidates=6913, candidate_scores=8477 |
| `experiment_4701_amortized_exploration_prior_go_explore_live` | `silent_bug_must_reopen` | dead_go_explore_archive, empty_candidate_generation_pool | go_explore_archive observations=0, stored_cells=0, actions_injected=0, prefixes_injected=0, with_prior_total_steps=0 |
| `experiment_4710_arms_summary` | `trustworthy_null` | - | online_cnn_observed_4737_fits_935_errors_0; first_win rates flat at 0.04 but CNN exercise counters are positive |
| `experiment_4712_perception_grounded_l2_goal_lp85` | `trustworthy_null` | - | structural detector ran: slots=77, piece_count=2, goal_count=42, goal_predicate_satisfiable=False |
| `experiment_4713_surface_present_winner_verifier_ranker` | `trustworthy_null` | - | winner_present_coverage=1.0, surfacing_samples=235, fit_count=156, positive_samples=181, candidate_scores=22684 |
| `experiment_4715_online_action_learning_driver_corrected` | `silent_bug_must_reopen` | byte_identical_online_driver_arms, a4_all_arms_0_04_tautology | frozen_first_win=online_scratch_first_win=online_warm_first_win=0.04, delta=0.0, flagged_adversarial=True; goal_free_probe archive_cells=[4, 64] |

## Prioritized Reopen Recommendations

- `P0` `online_action_learning_driver`: reopen_as_435_A1 - The .434 A4 artifact has frozen/scratch/warm first-win all equal to 0.04 with zero delta and flagged_adversarial=true; the faithful driver was not validly distinguished.
- `P1` `amortized_prior_go_explore_archive`: rerun_after_frame_grid_fix_with_positive_archive_cells - The archive arm reported zero observations, zero cells, and zero prefix injections, matching the confirmed (1,64,64) frame-grid no-op.
- `P2` `goal_energy_generation_live`: rerun_only_if_goal_energy_scores_real_candidate_states - The goal-energy arm cloned cached baseline attempts and marked them goal_energy_neutral_on_cached_frame, so generation was not exercised.
- `P3` `energy_fitness_qd_generation`: rerun_with_distinct_qd_and_random_mutation_candidate_pools - The QD, random-QD, and search arms are byte-identical except for arm labels.
- `P4` `hierarchical_subgoal_search`: rerun_only_after_nonempty_subgoal_decomposition_gate - The subgoal-search arm emitted an empty subgoal decomposition and no reachable subgoal evidence.

## Preconditions

```json
{
  "arc_go_explore_error": "",
  "arc_go_explore_importable": true,
  "go_explore_frame_grid_fix_confirmed": true,
  "missing_artifacts": [],
  "missing_modules": [],
  "module_files_present": true,
  "null_artifacts_present": true,
  "ok": true,
  "resolved_aliases": {
    "results/experiment_4710_online_action_learning_arms_summary.json": "results/experiment_4710_arms_summary.json"
  }
}
```

## Experiment 4765 .438 ARC Null Silent-Bug Audit

- Verdict: `complete_arc_null_silent_bug_audit_3_nulls_1_reopen`
- Nulls audited: `3`
- Silent bugs found: `1`
- Inference substrate: `aggregation_from_upstream_artifacts`

| Null | Verdict | Silent signatures | Evidence |
|---|---|---|---|
| `experiment_4761_structural_energy_s0_core_bet_probe` | `silent_bug_must_reopen` | `s0_origin_probe_leak` | s0_candidate_rows=463<br>origin_probe_rows=926<br>origin_probe_auroc=0.732793<br>near_miss_negative_fraction=0.924188<br>in_sample_auroc=0.846405 |
| `experiment_4762_levelup_attempt` | `trustworthy_null` | - | levelup_attempts=6<br>exercised_attempts=6<br>reproduced_existing=5<br>timed_no_gate=1 |
| `experiment_4764_heldout_first_win_readiness` | `trustworthy_null` | - | heldout_first_win_rate=0.04<br>first_win_baseline=0.04<br>heldout_attempts=100<br>positive_control_passed=True<br>parity_test_green=True<br>null_delta_methodology_note_present=True |

## Experiment 4775 .439 ARC Null Silent-Bug Audit

- Verdict: `complete_arc_null_silent_bug_audit_3_nulls_1_reopen`
- Nulls audited: `3`
- S0' leak controls fired: `False`
- Silent bugs found: `1`
- Inference substrate: `aggregation_from_upstream_artifacts`

| Null | Verdict | Silent signatures | Evidence |
|---|---|---|---|
| `experiment_4771_structural_energy_s0prime_origin_matched` | `silent_bug_must_reopen` | `s0prime_origin_probe_not_refit`, `s0prime_corrigendum_pending` | s0prime_candidate_rows=463<br>s0prime_n_pos=186<br>s0prime_n_neg=277<br>s0prime_contributing_games_with_both_classes=16<br>s0prime_single_class_games_skipped=2<br>origin_matched=True<br>origin_probe_status=origin_matched_single_origin_all_induced<br>origin_probe_refit_on_origin_matched_data=False<br>origin_probe_auroc=0.5<br>shuffled_label_control_auroc=0.503309<br>shuffled_label_resamples=128<br>shuffled_label_module_permutation_loo=True |
| `experiment_4772_levelup_attempt` | `trustworthy_null` | - | levelup_attempts=1<br>exercised_attempts=1<br>solution_label_count=11<br>reproduced_existing=1<br>new_levels_banked=0<br>reproduced_levels=0 |
| `experiment_4774_heldout_first_win_readiness` | `trustworthy_null` | - | heldout_first_win_rate=0.04<br>first_win_baseline=0.04<br>heldout_attempts=100<br>positive_control_passed=True<br>parity_test_green=True<br>null_delta_methodology_note_present=True |

## Experiment 4785 .440 ARC Null Silent-Bug Audit

- Verdict: `complete_arc_null_silent_bug_audit_3_nulls_1_reopen`
- Nulls audited: `3`
- S1 controls fired: `False`
- Silent bugs found: `1`
- Inference substrate: `aggregation_from_upstream_artifacts`

| Null | Verdict | Silent signatures | Evidence |
|---|---|---|---|
| `experiment_4781_structural_energy_s1_contrastive_landscape` | `silent_bug_must_reopen` | `s1_origin_probe_not_refit` | s1_candidate_rows=463<br>s1_n_pos=186<br>s1_n_neg=277<br>origin_matched=True<br>origin_probe_status=origin_matched_single_origin_all_induced<br>origin_probe_refit_on_origin_matched_data=False<br>origin_probe_auroc=0.5<br>shuffled_label_control_auroc=0.493356<br>shuffled_label_resamples=16<br>shuffled_label_module_permutation_loo=True<br>denoising_direction_executed=True<br>denoising_direction_agreement=0.622339<br>n_seeds=10<br>distinct_seed_count=10 |
| `experiment_4782_levelup_attempt` | `trustworthy_null` | - | levelup_attempts=1<br>exercised_attempts=1<br>solution_label_count=42<br>reproduced_existing=1<br>new_levels_banked=0<br>reproduced_levels=0 |
| `experiment_4784_heldout_first_win_readiness` | `trustworthy_null` | - | heldout_first_win_rate=0.04<br>first_win_baseline=0.04<br>heldout_attempts=100<br>positive_control_passed=True<br>parity_test_green=True<br>null_delta_methodology_note_present=True |
## Experiment 4795 .441 ARC Null Silent-Bug Audit

- Verdict: `complete_arc_null_silent_bug_audit_3_nulls_0_reopen`
- Nulls audited: `3`
- S2 live path reachable confirmed: `True`
- Silent bugs found: `0`
- Inference substrate: `aggregation_from_upstream_artifacts`

| Null | Verdict | Silent signatures | Evidence |
|---|---|---|---|
| `experiment_4791_structural_energy_s2_offpath_trust_gate` | `trustworthy_null` | - | lint_passed=True<br>artifact_live_path_reachable=True<br>artifact_lint_precondition_passed=True<br>e3_policy_imports_trust_energy=True<br>real_binary_accuracy_control=True<br>identical_candidate_sets=True<br>binary_selection_reconstructed=True<br>checked_games=5 |
| `experiment_4792_levelup_attempt` | `trustworthy_null` | - | levelup_attempts=2<br>solution_label_count=81<br>reproduction_gates=2<br>existing_depth_reproduced=2<br>new_depth_claims=0<br>new_levels_banked=0<br>offline_reproduced=False |
| `experiment_4794_heldout_first_win_readiness` | `trustworthy_null` | - | heldout_first_win_rate=0.04<br>first_win_baseline=0.04<br>heldout_attempts=100<br>positive_control_passed=True<br>parity_test_green=True<br>null_delta_methodology_note_present=True<br>inference_substrate=aggregation_from_upstream_artifacts<br>proxy_cache_used=True |

### S2 Control Check

- Real binary control: `True`
- Identical candidate sets: `True`
- Binary selection reconstructed: `True`
- Checked games: `5`
## Experiment 4805 .442 ARC Null Silent-Bug Audit

- Verdict: `complete_arc_null_silent_bug_audit_3_nulls_1_reopen`
- Nulls audited: `3`
- S2-v2 candidate pool diverse: `False`
- Silent bugs found: `1`
- Inference substrate: `aggregation_from_upstream_artifacts`

| Null | Verdict | Silent signatures | Evidence |
|---|---|---|---|
| `experiment_4801_structural_energy_s2v2_diverse_trust_gate` | `silent_bug_must_reopen` | `s2v2_degenerate_candidate_pool` | degenerate_candidate_pool_flagged=True<br>n_effective_games=5<br>artifact_n_effective_games=5<br>min_heldout_games=5<br>effective_game_floor_met=True<br>per_game_selections_logged=True<br>live_path_reachable_confirmed=True |
| `experiment_4802_levelup_attempt` | `trustworthy_null` | - | levelup_attempts=21<br>solution_label_count=596<br>reproduction_gates=15<br>existing_depth_reproduced=16<br>new_depth_claims=0<br>new_levels_banked=0<br>offline_reproduced=False<br>target_game=bp35 |
| `experiment_4804_heldout_first_win_readiness` | `trustworthy_null` | - | heldout_first_win_rate=0.04<br>first_win_baseline=0.04<br>heldout_attempts=100<br>positive_control_passed=True<br>parity_test_green=True<br>null_delta_methodology_note_present=True<br>inference_substrate=aggregation_from_upstream_artifacts<br>proxy_cache_used=True |

### S2-v2 Diversity Check

- DEGENERATE_CANDIDATE_POOL fired: `True`
- Effective games: `5`
- Minimum held-out games: `5`
- Per-game selections logged: `True`
- Live-path reachable confirmed: `True`
