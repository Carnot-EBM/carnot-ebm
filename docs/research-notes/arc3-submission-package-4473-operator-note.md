# ARC-AGI-3 Operator Submission Package Refresh (Exp 4473)

- Artifact: `results/experiment_4473_submission_package_prep_refresh.json`
- Ready for operator submission: `True`
- Revalidated package levels: `45`
- Grew vs Exp 4460/.412: `True`
- Prior submitted baseline: `13`
- Submitted by this task: `False`

Operator checklist:
- Review this refreshed JSON artifact and the package_manifest rows before submitting.
- Confirm total_reproduced_levels_in_package=45 is greater than prior baseline 13.
- Confirm grew_vs_412 reflects whether total_reproduced_levels_in_package>39.
- Run scripts/arc3_live_submit.py only as the operator; this prep task did not submit.
- Package contains 22 replayable games with cached action sequences.
- After any operator live validation, record the resulting scorecard separately.

Package manifest:
- r11l: L1, actions=4, env_match_basis=prior_live_submission_confirmed, source=results/experiment_4296_arc_incremental_progress_new_game.json
- ls20: L1, actions=13, env_match_basis=prior_live_submission_confirmed, source=results/experiment_4285_arc_incremental_progress_new_game.json
- wa30: L1, actions=33, env_match_basis=prior_live_submission_confirmed, source=results/experiment_4275_arc_incremental_progress_new_game.json
- s5i5: L1, actions=13, env_match_basis=offline_env_file_present, source=results/experiment_4421_config_rule_solve_unseen.json
- lp85: L5, actions=52, env_match_basis=prior_live_submission_confirmed, source=results/experiment_4372_e3_deeper_high_headroom_games.json
- sc25: L5, actions=102, env_match_basis=offline_env_file_present, source=results/experiment_4468_bank_sc25_provisional_levels.json
- cd82: L1, actions=5, env_match_basis=prior_live_submission_confirmed, source=results/arc_explore_trajectory_cd82.json
- sp80: L1, actions=4, env_match_basis=prior_live_submission_confirmed, source=results/arc_explore_trajectory_sp80.json
- su15: L1, actions=7, env_match_basis=prior_live_submission_confirmed, source=results/arc_explore_trajectory_su15.json
- tu93: L5, actions=93, env_match_basis=prior_live_submission_confirmed, source=results/experiment_4361_e3_deeper_high_headroom_games.json+TU93_L5_SUFFIX_ACTIONS
- tn36: L7, actions=102, env_match_basis=offline_env_file_present, source=results/experiment_4372_e3_deeper_high_headroom_games.json
- cn04: L1, actions=13, env_match_basis=prior_live_submission_confirmed, source=results/arc_explore_trajectory_cn04.json
- m0r0: L1, actions=15, env_match_basis=prior_live_submission_confirmed, source=results/arc_explore_trajectory_m0r0.json
- sk48: L1, actions=14, env_match_basis=prior_live_submission_confirmed, source=results/arc_explore_trajectory_sk48.json
- ar25: L1, actions=15, env_match_basis=offline_env_file_present, source=python/carnot/experiment_4339_e3_explore_verify_plan_ar25.py
- ka59: L1, actions=11, env_match_basis=offline_env_file_present, source=python/carnot/experiment_4350_e3_explore_verify_plan_ka59.py
- ft09: L1, actions=4, env_match_basis=offline_env_file_present, source=results/experiment_4363_e3_mechanic_limited_tails_tr87_ft09.json
- tr87: L6, actions=127, env_match_basis=offline_env_file_present, source=results/arc_loop_solve_tr87.json
- vc33: L1, actions=3, env_match_basis=offline_env_file_present, source=results/experiment_4446_drive_generic_first_contact_bank.json
- g50t: L1, actions=17, env_match_basis=offline_env_file_present, source=results/experiment_4443_bank_g50t_example_conditioned_win.json
- dc22: L1, actions=20, env_match_basis=offline_env_file_present, source=results/experiment_4467_solve_dc22_cegis_nocov.json
- sb26: L1, actions=9, env_match_basis=offline_env_file_present, source=results/experiment_4470_color_match_slot_operator_solve_sb26.json
