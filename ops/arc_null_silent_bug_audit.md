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
## Experiment 4825 .444 ARC Null Silent-Bug Audit

- Verdict: `complete_arc_null_silent_bug_audit_3_nulls_0_reopen`
- Nulls audited: `3`
- S3 controls verified: `True`
- S3 guidance exercised: `True`
- Silent bugs found: `0`
- Inference substrate: `aggregation_from_upstream_artifacts`

| Null | Verdict | Silent signatures | Evidence |
|---|---|---|---|
| `experiment_4821_structural_energy_s3_generation_lift` | `trustworthy_null` | - | matched_lambda0_control=True<br>same_games_seeds_budget=True<br>n_headroom_games=24<br>positive_control_passed=True<br>new_levels_not_in_bare_pool=True<br>live_path_reachable=True<br>s3_guidance_exercised=True<br>differing_headroom_games=18 |
| `experiment_4822_levelup_attempt` | `trustworthy_null` | - | levelup_attempts=19<br>solution_label_count=832<br>reproduction_gates=19<br>existing_depth_reproduced=19<br>new_depth_claims=0<br>new_levels_banked=0<br>offline_reproduced=False<br>target_game=ka59 |
| `experiment_4824_heldout_first_win_readiness` | `trustworthy_null` | - | heldout_first_win_rate=0.04<br>first_win_baseline=0.04<br>heldout_attempts=100<br>positive_control_passed=True<br>parity_test_green=True<br>null_delta_methodology_note_present=True<br>inference_substrate=aggregation_from_upstream_artifacts<br>proxy_cache_used=True |

### S3 Control Check

- Matched lambda-zero control: `True`
- Same games/seeds/budget: `True`
- Headroom games: `24`
- Reachable-winner positive control: `True`
- New levels not in bare pool: `True`
- Live-path reachable: `True`
- Guidance exercised: `True`
- Differing headroom games: `18`
## Experiment 4835 .445 ARC Null Silent-Bug Audit

- Verdict: `complete_arc_null_silent_bug_audit_3_nulls_0_reopen`
- Nulls audited: `3`
- A1 archive alive and prior exercised: `True`
- Silent bugs found: `0`
- Inference substrate: `aggregation_from_upstream_artifacts`

| Null | Verdict | Silent signatures | Evidence |
|---|---|---|---|
| `experiment_4831_amortized_incontext_exploration_prior_live` | `trustworthy_null` | - | archive_observations=2<br>archive_stored_cells=2<br>prefixes_injected=1<br>prior_changed_proposals=True<br>proposal_changes=1<br>imitation_heldout_games=1<br>heldout_not_in_distillation_set=True<br>first_win_rate_with_prior=0<br>first_win_rate_no_prior_ablation=0 |
| `experiment_4832_levelup_attempt` | `trustworthy_null` | - | levelup_attempts=19<br>solution_label_count=832<br>reproduction_gates=19<br>existing_depth_reproduced=19<br>new_depth_claims=0<br>new_levels_banked=0<br>offline_reproduced=False<br>target_game=ka59 |
| `experiment_4834_heldout_first_win_readiness` | `trustworthy_null` | - | heldout_first_win_rate=0.04<br>first_win_baseline=0.04<br>heldout_attempts=100<br>positive_control_passed=True<br>parity_test_green=True<br>null_delta_methodology_note_present=True<br>inference_substrate=aggregation_from_upstream_artifacts<br>proxy_cache_used=True |

### A1 Control Check

- Archive observations: `2`
- Archive stored cells: `2`
- Prefixes injected: `1`
- Prior changed proposals: `True`
- Imitation control held-out split: `True`
## Experiment 4845 .446 A1 Perception Probe Audit

- Verdict: `complete_a1_perception_probe_audit_genuinely_exercised`
- a1_genuinely_exercised: `True`
- Non-test reasons: `-`
- Inference substrate: `aggregation_from_upstream_artifacts`

| Check | Passed | Detail |
|---|---:|---|
| `measured_on_real_frames` | `True` | `{"artifact_measured_on_real_frames": true, "bad_games": []}` |
| `tracker_changed_vs_baseline` | `True` | `{"deltas": {"lp85": 0.04985, "r11l": 0.192308, "tu93": 0.06639}, "distinct_delta_count": 3, "missing_numeric_games": [], "nonzero_delta_games": ["lp85", "r11l", "tu93"]}` |
| `positive_control_and_recovery_claim` | `True` | `{"artifact_games_with_recovery": 1, "claimed_recovery_matches_rows": true, "complete_claimed": true, "goal_persistence": 0.5, "goal_track_id": 39, "player_motion": 220.624731, "player_track_id": 9, "positive_control_passed": true, "recovered_count": 1, "recovered_games": ["r11l"], "should_be_success": false, "success_claimed": false, "verdict_matches_numbers": true}` |
| `live_path_and_provenance` | `True` | `{"arc_orphan_solver_lint_passed": true, "artifact_live_path_reachable": true, "not_live_agent_self_discovery": true, "solve_provenance": "development_proxy"}` |
| `summarizer_and_adversarial_verify` | `True` | `{"adversarial_flag_count": 0, "adversarial_loaded": true, "summarizer_returncode": 0}` |

- Per-game deltas: `{'lp85': 0.04985, 'r11l': 0.192308, 'tu93': 0.06639}`
- Source checksum: `sha256:f34b2628f231f7ccd67147963c40e486acd8890a18c26e0182b43dac889bd313`
## Experiment 4855 .447 A1 Generation Diagnostic Audit

- Verdict: `complete_a1_generation_diagnostic_audited`
- a1_genuinely_diagnostic: `True`
- Non-diagnostic reasons: `-`
- Inference substrate: `aggregation_from_upstream_artifacts`

| Check | Passed | Detail |
|---|---:|---|
| `proposer_blind_to_banked_answer` | `True` | `{"allowed_only_in_classify_game_coverage": true, "artifact_flag": true, "disallowed_refs": [], "function_present": true, "winning_prefix_refs": 1}` |
| `positive_control` | `True` | `{"adaptered": true, "artifact_positive_control_covered": true, "coverage_bucket": "COVERED", "matched_winning_prefix_len": 18, "positive_control_game": "tu93", "reached_l1_win": true, "winning_prefix_len": 18}` |
| `bucket_distribution` | `True` | `{"bucket_counts": {"COVERED": 1, "NEVER_ENUMERATED": 9}, "claimed_dominant_bucket": "NEVER_ENUMERATED", "computed_dominant_bucket": "NEVER_ENUMERATED", "honest_verdict": "complete_generation_wall_never_enumerated_dominant", "invalid_games": [], "n_games_measured": 10, "per_game_count": 10}` |
| `live_path_and_provenance` | `True` | `{"arc_orphan_solver_lint_passed": true, "artifact_live_path_reachable": true, "development_proxy": true, "solve_provenance": "development_proxy"}` |
| `summarizer_and_adversarial_verify` | `True` | `{"adversarial_flag_count": 0, "adversarial_loaded": true, "summarizer_returncode": 0}` |

- Source artifact checksum: `sha256:a49c15b54ff935ac736d4a580fdf4639ac019377b251da04cb2cc7501f5cbaff`
- Source script checksum: `sha256:0af6642f1fa0a6e9aa2b4554df30fa3f35e61b76946de647d37e6c8a8fd8cb72`
## Experiment 4865 .448 A1 Fork Probe Audit

- Verdict: `complete_a1_fork_probe_non_test_positive_control_not_migrated_positive_control_row_missing_positive_control_not_covered`
- a1_genuinely_diagnostic: `False`
- Non-diagnostic reasons: `positive_control_not_migrated, positive_control_row_missing, positive_control_not_covered, positive_control_low_accuracy, n_games_measured_below_3, live_path_unreachable`
- Inference substrate: `aggregation_from_upstream_artifacts`

| Check | Passed | Detail |
|---|---:|---|
| `planner_blind_to_banked_answer` | `True` | `{"allowed_only_in_classify_planned_pool": true, "artifact_flag": true, "disallowed_refs": [], "function_present": true, "winning_prefix_refs": 1}` |
| `positive_control` | `False` | `{"artifact_positive_control_migrated": false, "engine_heldout_accuracy": null, "high_accuracy_threshold": 0.5, "planned_bucket": null, "positive_control_game": "tu93", "row_game": null}` |
| `numbers_match_fork` | `False` | `{"computed_coverage_migration_count": 0, "computed_fork_verdict": null, "computed_median_engine_heldout_accuracy": null, "invalid_games": [], "per_game_count": 0, "reported_coverage_migration_count": 0, "reported_fork_verdict": null, "reported_median_engine_heldout_accuracy": null, "reported_n_games_measured": 0}` |
| `live_path_and_provenance` | `False` | `{"arc_orphan_solver_lint_passed": true, "artifact_live_path_reachable": false, "development_proxy": true, "solve_provenance": "development_proxy", "source_live_path": {"calls": ["_collect_cold_policy_transitions", "max", "list", "int", "policy._induce_and_plan", "e3.load_engine", "live_methods.append", "repr", "_heldout_transitions", "sum", "ord", "score", "float", "float", "e3.WorldModelVerifier", "e3.plan_in_model", "live_methods.append", "int", "int", "repr", "normalize_sequence", "_execute_plan_reaches_l1", "classify_planned_pool", "row.update", "round", "round", "len", "len", "int", "list", "float", "float", "getattr"], "function_present": true, "induce_and_plan_called": true, "load_engine_called": true, "passed": true, "plan_in_model_called": true}, "source_live_path_called": true}` |
| `summarizer_and_adversarial_verify` | `False` | `{"adversarial_flag_count": 2, "adversarial_loaded": true, "summarizer_returncode": 2}` |

- Source artifact checksum: `sha256:fe40e39d556c549cd113e4c6850b78263c73cd856060c028e6cbebd28db906d3`
- Source script checksum: `sha256:c0e890c4540d72e7656d2ece898bdde7162c27e3d9e369fc66a01269a95fd998`
## Experiment 4876 A1/A1b Audit

- Verdict: `complete_a1_a1b_audited`
- a1_genuinely_diagnostic: `False`
- a1_ran_live_on_gpu0: `True`
- a1b_delta_trustworthy: `True`
- A1 reasons: `positive_control_not_migrated, positive_control_not_covered, positive_control_low_accuracy, fork_verdict_missing`
- A1b reasons: `-`
- A1b residuals: `a1b_positive_control_failed`
- Inference substrate: `aggregation_from_upstream_artifacts`

| Check | Passed | Detail |
|---|---:|---|
| `a1_live_gpu` | `True` | `{"duration_floor_s": 60.0, "duration_s": 587.0132269859314, "flagged_adversarial": false, "generator_backend": "gpu0_cuda"}` |
| `a1_planner_blind_to_banked_answer` | `True` | `{"allowed_only_in_classify_planned_pool": true, "artifact_flag": true, "disallowed_refs": [], "function_present": true, "winning_prefix_refs": 1}` |
| `a1_positive_control` | `False` | `{"artifact_positive_control_migrated": false, "engine_heldout_accuracy": 0.0, "high_accuracy_threshold": 0.5, "planned_bucket": "NEVER_ENUMERATED", "positive_control_game": "tu93", "row_game": "tu93"}` |
| `a1_numbers_match_fork` | `False` | `{"computed_coverage_migration_count": 0, "computed_fork_verdict": "INDUCER_CEILING", "computed_median_engine_heldout_accuracy": 0.0, "invalid_games": [], "per_game_count": 9, "reported_coverage_migration_count": 0, "reported_fork_verdict": null, "reported_median_engine_heldout_accuracy": 0.0, "reported_n_games_measured": 9}` |
| `a1_live_path` | `True` | `{"arc_orphan_solver_lint_passed": true, "artifact_live_path_reachable": true, "solve_provenance": "development_proxy", "source_live_path": {"calls": ["_collect_cold_policy_transitions", "list", "int", "_heldout_transitions", "normalize_sequence", "classify_planned_pool", "row.update", "policy._induce_and_plan", "e3.load_engine", "live_methods.append", "score", "float", "float", "_execute_plan_reaches_l1", "max", "e3.plan_in_model", "live_methods.append", "round", "round", "len", "len", "int", "list", "repr", "sum", "e3.WorldModelVerifier", "float", "float", "getattr", "int", "int", "repr", "ord"], "function_present": true, "induce_and_plan_called": true, "load_engine_called": true, "passed": true, "plan_in_model_called": true}, "source_live_path_called": true}` |
| `a1_summarizer_and_adversarial_verify` | `True` | `{"adversarial_flag_count": 0, "adversarial_loaded": true, "summarizer_returncode": 0}` |
| `a1b_delta` | `True` | `{"adversarial_flag_count": 1, "artifact_delta_on_truly_heldout_split": true, "circular_moat_flag_count": 0, "computed_ci95": [0.0, 0.0], "computed_median": 0.0, "live_path_reachable": true, "n_games_measured": 9, "oracle_distinct": true, "positive_control_passed": false, "reported_ci95": [0.0, 0.0], "reported_median": 0.0, "split_disjoint": true, "status": "ran", "verifier_is_oracle": false}` |

- A1 artifact checksum: `sha256:f223196660e020cef86c8f3901aedebe269012ab3eee7c1c9be6e4cea9bac081`
- A1 script checksum: `sha256:3a96f5b124687b1562ec0e1578e810852adebf85034150f50e4b389b506049bf`
- A1b artifact checksum: `sha256:0afdde35fa4ca01fd551d6df3136e2f4cff5b7d7b694df969c035e89d9b03bd8`
## Experiment 4887 A1/A1b Audit

- Verdict: `complete_a1_a1b_audited`
- a1_genuinely_diagnostic: `True`
- a1_positive_control_non_degenerate_confirmed: `True`
- a1_delta_on_heldout_disjoint_confirmed: `True`
- planner_blind_confirmed: `True`
- numbers_match_fork: `True`
- a1b_ab_trustworthy: `False`
- A1 reasons: `-`
- A1b reasons: `a1b_flagged_adversarial_stamp, a1b_adversarial_verify_flagged, a1b_duration_below_live_floor`
- Inference substrate: `aggregation_from_upstream_artifacts`

| Check | Passed | Detail |
|---|---:|---|
| `a1_live_gpu` | `True` | `{"duration_floor_s": 60.0, "duration_s": 168.14787244796753, "flagged_adversarial": false, "generator_backend": "gpu0_cuda"}` |
| `a1_positive_control` | `True` | `{"artifact_positive_control_non_degenerate": true, "cell_recall": 0.177143, "positive_control_game": "tu93", "row_game": "tu93"}` |
| `a1_heldout_disjoint_delta` | `True` | `{"artifact_delta_on_truly_heldout_split": true, "bad_games": [], "checked_games": ["cd82", "cn04", "ls20", "m0r0", "r11l", "sk48", "sp80", "su15", "wa30"]}` |
| `a1_planner_blind_to_banked_answer` | `True` | `{"allowed_only_in_classify_planned_pool": true, "artifact_flag": true, "disallowed_refs": [], "function_present": true, "winning_prefix_refs": 1}` |
| `a1_numbers_match_fork` | `True` | `{"computed_ci95": [-0.177673, 0.0], "computed_coverage_migration_count": 0, "computed_engine_cell_recall_median": 0.727273, "computed_fork_verdict": "INDUCER_CEILING_HARD", "computed_median_delta": -0.008699, "invalid_games": [], "never_enumerated_games": 9, "per_game_count": 9, "reported_ci95": [-0.177673, 0.0], "reported_coverage_migration_count": 0, "reported_engine_cell_recall_median": 0.727273, "reported_fork_verdict": "INDUCER_CEILING_HARD", "reported_median_delta": -0.008699, "reported_n_games_measured": 9}` |
| `a1_summarizer_and_adversarial_verify` | `True` | `{"adversarial_flag_count": 0, "adversarial_loaded": true, "summarizer_returncode": 0}` |
| `a1_oracle_distinct` | `True` | `{"circular_moat_flag_count": 0, "verifier_is_oracle": false}` |
| `a1_live_path` | `True` | `{"arc_orphan_solver_lint_passed": true, "artifact_live_path_reachable": true, "solve_provenance": "development_proxy", "source_live_path": {"calls": ["a1._collect_cold_policy_transitions", "list", "cold.get", "_fit_warm_ttt_model", "DynamicsValueAdapter.fit", "adapter.wrap", "e3.collect_transitions", "list", "score_graded_engine", "score_graded_engine", "float", "float", "a1.classify_planned_pool", "classification.update", "int", "sum", "np.asarray", "_plan_with_adapted_engine", "int", "max", "cold.get", "int", "a1._execute_plan_reaches_l1", "round", "round", "round", "round", "round", "_transition_ids", "_transition_ids", "_transition_ids", "len", "len", "len", "ord", "np.asarray", "float", "float", "str", "int"], "classification_path_called": true, "dynamics_value_adapter_called": true, "function_present": true, "passed": true, "plan_path_called": true}, "source_live_path_called": true}` |
| `a1b_ab_fairness` | `False` | `{"a1_games": ["cd82", "cn04", "ls20", "m0r0", "r11l", "sk48", "sp80", "su15", "wa30"], "adversarial_flag_count": 1, "bad_delta_games": [], "bad_split_games": [], "duration_floor_s": 60.0, "duration_s": 13.697947978973389, "live_path_reachable": true, "local_games": ["cd82", "cn04", "ls20", "m0r0", "r11l", "sk48", "sp80", "su15", "wa30"], "oracle_distinct": true, "reference_games": ["cd82", "cn04", "ls20", "m0r0", "r11l", "sk48", "sp80", "su15", "wa30"], "reference_lane_is_ceiling_only": true, "same_games_as_a1": true, "same_split_as_a1": true, "status": "ran", "verifier_is_oracle": false}` |

- A1 artifact checksum: `sha256:6a0f5e5552d56b267cb79f1b9414289514ae65bd3f6ff5653a98237b1b3c6333`
- A1 script checksum: `sha256:b678bbb29a94dca56189acfe144ddf9a69968784e7d4969a1fd90749b2d52e27`
- A1b artifact checksum: `sha256:e7b2a79e8d5e0393574b641be54e50ee9d3ab6491096cf18ce822e8286febb21`
