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
