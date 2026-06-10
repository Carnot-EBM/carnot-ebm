# Phase 4 Active Inference

## Implementation Status
- Phase 4 Active Inference (Simulation): Implemented

## Requirements

### REQ-PHASE4-001: MLD Simulation
**Description:** The system SHALL simulate MLD steps on an Ising substrate, computing `mu_P` and `alpha_t` per step.

### REQ-PHASE4-002: Verifier Ensemble Stability
**Description:** The verifier-as-free-energy ensemble SHALL maintain `inf_t alpha_t > 0.10` with `k=6` verifiers, and collapse to `< 0.05` with `k=1`.

## Scenarios

### SCENARIO-PHASE4-1
**Description:** Running 100 MLD steps with `k=6` and `k=1` on `n=8` substrate yields a `delta_alpha > 0.05`.

### REQ-PHASE4-003: MLD Substrate Scaling
**Description:** The system SHALL simulate MLD steps on scaling Ising substrates (n=8, n=16, n=32) and track `delta_alpha` scaling behavior to identify potential substrate size collapse points.

### SCENARIO-PHASE4-2
**Description:** Running scaling simulation on n=8, 16, 32 yields recorded delta_alpha scaling points, correctly identifying if and where delta_alpha collapses below 0.05.

### REQ-PHASE4-004: Verifier Ablation Audit
**Description:** The system SHALL support running a random-verifier injection ablation audit to test if `delta_alpha` genuinely depends on verifier content.

### SCENARIO-PHASE4-3
**Description:** Running a 4-cell random-verifier ablation audit correctly computes `delta_alpha` and bootstrap CIs for each fraction, and sets the `monotonic_decay_observed` and `artifact_detected` flags based on the results.

### REQ-PHASE4-005: Maximum-Caliber Alpha_t Replacement
**Description:** The system SHALL implement `alpha_t'` derived from the maximum-caliber formulation (prediction error), where `alpha_t'` monotonically decays as random verifiers replace real verifiers in the ensemble.

### SCENARIO-PHASE4-4
**Description:** Running a 4-cell random-verifier ablation audit using the maximum-caliber `alpha_t'` results in monotonic decay of `delta_alpha` (`monotonic_decay_observed=true`) and absence of the bijection-invariance artifact (`artifact_detected=false`).

### REQ-PHASE4-CANONICAL-DECISION
**Description:** The system SHALL produce a canonical decision artifact documenting the transition from alpha_t metrics to the Fast-Slow Variant empirical metric.

### SCENARIO-DECISION-ARTIFACT-GENERATION
**Description:** Running the decision script outputs a valid carnot.phase4_canonical_decision.v2 JSON artifact confirming the retirement of thermodynamic metrics.

### REQ-PHASE4-006: ARC-AGI-3 Synthetic Harness Scaffold
**Description:** The system SHALL provide an importable ARC-AGI-3 harness scaffold that runs without GPU or live model dependencies, checks `import carnot.verify` before declaring readiness, exposes a tiny synthetic grid environment, encodes observations into verifier inputs, uses the cheap energy verifier as a verifier-as-router score for candidate actions, prunes low-scoring actions, escalates only when all candidates score poorly, and records that the scaffold is synthetic rather than a real ARC-AGI-3 benchmark result.

### SCENARIO-PHASE4-006
**Description:** Running the scaffold on the tiny synthetic grid task selects the verifier-preferred action, prunes at least one low-scoring candidate, solves the task, writes the Exp 3919 readiness artifact with bare scalar readiness fields, and reports no ARC-AGI-3 benchmark performance claim.

### REQ-PHASE4-007: ARC-AGI-3 Synthetic Action-Efficiency Measurement
**Description:** The system SHALL measure verifier-as-pruner action efficiency on a richer synthetic ARC-AGI-3-style environment with deterministic transitions, a larger discrete action space, a known goal, and at least 30 episodes, comparing cheap energy-verifier action selection against a no-verifier random/greedy baseline without claiming real ARC-AGI-3 benchmark performance.

### SCENARIO-PHASE4-007
**Description:** Running Exp 3929 records mean actions-to-solve for verifier and baseline arms, bootstrap CI for the baseline/verifier action-efficiency ratio, solve rates for both arms, methodology fields, real ARC-AGI-3 access preflight reachability, and an honest verdict that reports either verifier-router help only when the CI lower bound is above 1.0 or a no-advantage synthetic finding otherwise.

### REQ-PHASE4-008: Active Data Codex Nonspatial Sweep
**Description:** The system SHALL test whether the "active data -> codex program synthesis -> consistency-energy verification" pipeline generalizes across the 6 non-spatial games. It MUST collect active transitions, run codex program synthesis with the consistency-energy refactor-from-best loop (bounded to <=3 iterations), and grade the best program on a common held-out test. It MUST report a per-game table including best held-out consistency energy, whether it is trustworthy (<=0.15), and the vc33 baseline diff.

### SCENARIO-PHASE4-008
**Description:** Running Exp 3947 produces a valid artifact with fields `n_trustworthy_at_0.15`, `per_game_best_energy`, `total_codex_calls`, `total_codex_seconds`, and an honest verdict.

### SCENARIO-PHASE4-008-3968
**Description:** Running Exp 3968 re-attempts the active-data codex non-spatial sweep and writes `results/experiment_3968_active_codex_nonspatial_sweep.json` with `n_trustworthy_at_0.15`, `per_game_best_energy`, `total_codex_calls`, `total_codex_seconds`, `markov_vs_hidden_split`, `random_seed`, `honest_verdict`, `duration_s`, and `inference_substrate`; the run must block honestly with `blocked_codex_unavailable` or `blocked_arc_offline_env_unavailable` when either required substrate is absent.

### REQ-PHASE4-009: ARC-AGI-3 Goal Predicate Induction
**Description:** The system SHALL induce a grid-grounded goal predicate from observed level-up transitions for non-spatial ARC-AGI-3 games. It MUST collect transitions where `levels_completed` increments, requiring at least 2 examples to avoid mis-induction. It MUST evaluate the precision and recall of the induced predicate on held-out win vs. non-win states.

### SCENARIO-PHASE4-009
**Description:** Running Exp 3956 produces a valid artifact with fields `goal_predicate_precision`, `goal_predicate_recall`, `n_level_ups_observed`, `games_covered`, `random_seed`, `honest_verdict`, `duration_s`, and `inference_substrate`.

### REQ-PHASE4-012: ARC-AGI-3 M3 Efficiency Real Games
**Description:** The system SHALL measure the efficiency transfer of the verifier action-pruner on solved real ARC-AGI-3 games. It MUST load prior real game solve artifacts (e.g. from Exp 3946, 3953, 3954), simulate action evaluations with and without the verifier pruner over the required path to solve, bootstrap 95% CIs for the efficiency ratio of actions taken, and output the result artifact `results/experiment_3959_m3_efficiency_real_games.json` to prove if the pruner meaningfully cuts the search space on real games.

### SCENARIO-PHASE4-012
**Description:** Running Exp 3959 produces a valid artifact with fields `ci95_with`, `ci95_without`, `efficiency_ratio_with_over_without`, `cis_non_overlapping_pruner_helps`, `n_solved_levels_measured`, and `honest_verdict`. The verdict is honest and reports blocked if no real solved games are available.

### REQ-PHASE4-013: ARC-AGI-3 r11l Incremental Real-Env Solve
**Description:** The system SHALL extend the real-env-confirmed r11l solve monotonically beyond the prior L1 artifact by reusing the induced select-place mechanic, re-perceiving piece-target pairs after each level-up, stopping after L3 or the first unsolved level, and writing `results/experiment_3964_r11l_incremental_l2.json` with bare artifact fields for total levels solved, newly solved levels beyond L1, per-level actions, baseline action references, first failed level, real-env confirmation, random seed, duration, inference substrate, and terminal-prefix honest verdict.

### SCENARIO-PHASE4-013
**Description:** Running Exp 3964 against the offline r11l environment either writes `blocked_arc_offline_env_unavailable` when the offline ARC environment cannot load, or confirms each solved level through `levels_completed`, records actions and baseline references per solved level, reports at least the prior L1 total honestly, and never attempts a full all-level solve beyond the scoped L2/L3 target.

### REQ-PHASE4-014: ARC-AGI-3 lp85 Incremental Real-Env Solve
**Description:** The system SHALL extend the real-env-confirmed lp85 solve monotonically beyond the prior L1 artifact by reusing the Exp 3954 permutation-via-button-click mechanic, re-perceiving the lp85 button configuration after each level-up, stopping after L2 or the first unsolved level, and writing `results/experiment_3965_lp85_incremental_l2.json` with bare fields for total levels solved, newly solved levels beyond L1, per-level actions, baseline action references, whether the induced mechanic held at L2, real-env confirmation, random seed, duration, inference substrate, and a terminal-prefix honest verdict.

### SCENARIO-PHASE4-014
**Description:** Running Exp 3965 against the offline lp85 environment either writes `blocked_arc_offline_env_unavailable` when the offline ARC environment cannot load, or confirms each solved level through `levels_completed`, records the per-level action counts and `results/arc_agi3_access_probe.json` baseline references for every solved lp85 level, reports only one new level beyond L1 when L2 is solved, and does not continue beyond the scoped L2 target.

### REQ-PHASE4-015: ARC-AGI-3 ArcMemo Cross-Game Concept Memory
**Description:** The system SHALL run an ArcMemo-style cross-game memory experiment over an ordered sequence of at least three non-spatial ARC-AGI-3 games. It MUST compare a no-memory induction arm against a with-memory arm at equal active-data budget per game, seed concept-level records from the real-env-confirmed r11l select/place and lp85 permutation solves, retrieve concept records for later games, and write `results/experiment_3970_cross_game_arcmemo_transfer.json` with bare fields for transfer win, per-game call counts, per-game held-out consistency energy, stored concept count, reused concept count, random seed, duration, inference substrate, and a terminal-prefix honest verdict.

### SCENARIO-PHASE4-015
**Description:** Running Exp 3970 either blocks honestly with `blocked_arc_offline_env_unavailable` when the offline ARC environment cannot load, blocks with `blocked_codex_unavailable` only when fresh Codex synthesis is required and unavailable, or completes with concrete concept-reuse evidence showing which stored concepts were retrieved on later games and whether the with-memory arm reached trustworthy energy (`<=0.15`) with fewer calls or lower energy than the no-memory arm.

### REQ-PHASE4-016: ARC-AGI-3 M4 Offline Quota-Gate Readiness
**Description:** The system SHALL prepare an operator-only ARC-AGI-3 scored-run readiness package without submitting online. It MUST register a `hybrid` policy in `scripts/experiments/arc3_offline_eval.py` that replays real-env-confirmed solved-game mechanics for banked solved games and falls back to no-induction object-click exploration elsewhere. It MUST evaluate the hybrid, random, and object-click policies over the same start-here game list and budget settings, compare hybrid offline solved levels against the prior submitted Carnot run of 0 and the same-run no-induction baselines, report documented SOTA context for EWM 58.12% RHAE, Graph-Explore 30/52 levels, and frontier LLMs below 0.4%, and write `results/experiment_3971_m4_offline_quota_gate.json` with bare fields `hybrid_accuracy_levels_solved`, `baseline_accuracy_levels_solved`, `hybrid_efficiency_ratio`, `quota_gate_cleared`, `scored_run_ready_for_operator`, `random_seed`, `honest_verdict`, `duration_s`, and `inference_substrate`.

### SCENARIO-PHASE4-016
**Description:** Running Exp 3971 either blocks honestly with `blocked_arc_offline_env_unavailable` when the offline ARC environment cannot load, or completes offline with the readiness verdict. The verdict starts with `success:` only when the hybrid beats both prior 0 and the measured no-induction baseline on the same game set; otherwise it starts with `complete: quota_gate_not_cleared_...` and records the gap. The artifact MUST set `submitted_to_leaderboard=false` and MUST NOT run an online or scored ARC-AGI-3 submission.

### REQ-PHASE4-017: ARC-AGI-3 Execution-Guided World-Model Synthesis
**Description:** The system SHALL run Exp 3979 as an execution-guided world-model synthesis sweep for the six Exp 3968 non-spatial games. The inducer MUST keep only candidate programs that exactly replay their accepted observed transitions, MUST compose an exact replay table with a generalizing transition predictor, MUST grade the best accepted program on a common held-out test via consistency energy, MUST first pass the vc33 positive-control gate at held-out energy `<=0.05`, and MUST report consistency energy as prediction trustworthiness rather than mechanistic understanding. The terminal artifact `results/experiment_3979_world_model_gen_execution_guided.json` MUST include bare fields `positive_control_passed`, `n_trustworthy_at_0.15`, `per_game_best_energy`, `beats_exp3968`, `total_synthesis_calls`, `total_synthesis_seconds`, `random_seed`, `honest_verdict`, `duration_s`, and `inference_substrate`.

### SCENARIO-PHASE4-017
**Description:** Running Exp 3979 either blocks honestly with `blocked_arc_offline_env_unavailable`, `blocked_codex_unavailable` if fresh Codex synthesis is explicitly requested and unavailable, or `blocked_positive_control_failed`, or writes a complete/success artifact after evaluating all six non-spatial games. A passing run MUST cross-reference Exp 3968 per-game energies, the vc33 0.005 control, and the determinism probe hidden-state split, and MUST set `beats_exp3968=true` only when execution-guided synthesis yields more trustworthy models than Exp 3968.

### REQ-PHASE4-018: ARC-AGI-3 Per-Level Re-Induction Increment
**Description:** The system SHALL run Exp 3980 as a one-game incremental ARC-AGI-3 level-up attempt that first reads the Exp 3964/3965 L2 stall diagnostics, chooses the most promising single game, re-perceives the L2 board after the banked L1 level-up, re-induces a level-local rule from L2 observations instead of assuming the L1 mechanic transfers, executes the chosen L2 action plan in the real offline environment, confirms every level-up through `levels_completed`, and stops after the first failed level. The terminal artifact `results/experiment_3980_incremental_levels_reinduction.json` MUST include bare fields `ACCURACY_levels_solved`, `new_levels_solved_this_task`, `reinduction_found_different_rule`, `game_advanced`, `per_level_actions`, `baseline_actions_ref`, `real_env_confirmed`, `random_seed`, `honest_verdict`, `duration_s`, and `inference_substrate`.

### SCENARIO-PHASE4-018
**Description:** Running Exp 3980 either writes `blocked_arc_offline_env_unavailable` when the offline Arcade cannot load, or reports an honest terminal-prefix artifact for the selected game. A successful L2 advance MUST use `success: reinduction_advanced_<game>_to_L<n>` and record whether L2 required a different rule than L1; an unsolved attempt MUST use `complete: l2_wall_holds_<game>_<reason>` and record the first failed level without chasing unrelated games or all remaining levels.

### REQ-PHASE4-019: ARC-AGI-3 Fourth Game First Solve
**Description:** The system SHALL run Exp 3981 as a targeted first-solve attempt for a fourth distinct ARC-AGI-3 game, choosing a non-spatial candidate from {tn36, su15, dc22} empirically based on L0 baseline budget and inducibility. The agent MUST actively collect transitions, deterministically perceive objects and target locations, induce the mechanic and goal predicate from observation or level-up, and plan and execute moves in the offline environment. The terminal artifact `results/experiment_3981_fourth_game_first_solve.json` MUST include bare fields `ACCURACY_levels_solved`, `game_solved`, `first_solve_at_action`, `induced_mechanic`, `games_attempted`, `real_env_confirmed`, `random_seed`, `honest_verdict`, `duration_s`, and `inference_substrate`.

### SCENARIO-PHASE4-019
**Description:** Running Exp 3981 either writes `blocked_arc_offline_env_unavailable` when the offline Arcade cannot load, or reports an honest terminal-prefix artifact tracking the fourth game attempt. A successful L1 solve MUST use `success: <game>_first_solve_levels1_...` and record the action count; a 0-solve attempt MUST use `complete: fourth_game_no_solve_<reason>` and record the induced mechanics for all attempted candidates.

### REQ-PHASE4-020: ARC-AGI-3 ArcMemo Solve-Loop Transfer
**Description:** The system SHALL run Exp 3982 as an ArcMemo solve-loop transfer experiment over a held-out ARC-AGI-3 game. It MUST build a concept-level memory from banked real-env-confirmed solved-game mechanics, require a positive-control shared concept family across at least two games before claiming transfer, compare cold-start solving against concept-memory-seeded solving at equal offline perception, count real `env.step` actions and reset attempts to the first `levels_completed` increment in each arm, record the concrete concept retrieved and reused, and write `results/experiment_3982_arcmemo_solve_transfer.json` with bare fields `solve_transfer_win`, `actions_cold_start`, `actions_with_memory`, `attempts_cold_start`, `attempts_with_memory`, `concept_reused`, `positive_control_shared_structure`, `real_env_confirmed`, `random_seed`, `honest_verdict`, `duration_s`, and `inference_substrate`.

### SCENARIO-PHASE4-020
**Description:** Running Exp 3982 either writes `blocked_arc_offline_env_unavailable` when the offline Arcade cannot load, reports `complete: arcmemo_solve_no_transfer_positive_control_failed` when fewer than two banked games share a concept family, or completes an honest two-arm solve-loop comparison. A transfer win MUST use `success: arcmemo_solve_transfer_<a_cold>to<a_mem>_actions` only when the memory-seeded arm solves in the real offline env and uses fewer actions or reset attempts than cold start; otherwise it MUST use `complete: arcmemo_solve_no_transfer_<reason>` while preserving measured action and attempt counts.

### REQ-PHASE4-021: ARC-AGI-3 Verifier-Validated Per-Level Re-Induction
**Description:** The system SHALL run Exp 3992 as a one-game ARC-AGI-3 incremental level-up attempt that reads Exp 3980's L2 diagnosis, chooses the most promising single game, re-perceives each post-level-up board, re-induces level-local candidate rules, validates competing candidates against held-out L2/L3 copied-env transitions with a GAP-4-style executed-consistency verifier before spending real-env actions, commits only the validated candidate, confirms every level-up through the real environment's `levels_completed` counter, attempts L3 only after an L2 advance, and does not chase all remaining levels. The terminal artifact `results/experiment_3992_incremental_levels_verifier_validated.json` MUST include bare fields `ACCURACY_levels_solved`, `new_levels_solved_this_task`, `verifier_validated_the_rule`, `actions_saved_vs_openloop`, `game_advanced`, `per_level_actions`, `baseline_actions_ref`, `real_env_confirmed`, `random_seed`, `honest_verdict`, `duration_s`, and `inference_substrate`.

### SCENARIO-PHASE4-021
**Description:** Running Exp 3992 either writes `blocked_arc_offline_env_unavailable` when the offline Arcade cannot load, or reports an honest terminal-prefix artifact for the selected game. A successful verifier-validated advance MUST use `success: verifier_validated_reinduction_advanced_<game>_to_L<n>` and record the chosen candidate's executed-consistency energy before real actions were committed; an unsolved attempt MUST use `complete: l2_wall_holds_<game>_<reason>` and record whether verifier validation rejected the L2 rule without spending the open-loop actions from Exp 3980.

### REQ-PHASE4-022: ARC-AGI-3 ArcMemo Solve Transfer V2
**Description:** The system SHALL run Exp 3994 as a second ArcMemo solve-loop transfer check against a genuinely new fourth-game target from Exp 3993 when that artifact solved a fourth game, or against re-held-out `sc25` when Exp 3993 is absent or no-solve. The experiment MUST first confirm the offline Arcade precondition, build concept memory records from banked real-env-confirmed solved-game mechanics using `{name, when_it_applies, effect}`, require a positive-control shared concept family across at least two banked concepts before attempting transfer, compare cold-start solving against concept-memory-seeded solving at equal perception, count real `env.step` actions and reset attempts to the first `levels_completed` increment in each arm, record the concrete concept retrieved and reused, and write `results/experiment_3994_arcmemo_solve_transfer_v2.json` with bare fields `solve_transfer_win`, `actions_cold_start`, `actions_with_memory`, `attempts_cold_start`, `attempts_with_memory`, `target_game`, `concept_reused`, `positive_control_shared_structure`, `real_env_confirmed`, `random_seed`, `honest_verdict`, `duration_s`, and `inference_substrate`.

### SCENARIO-PHASE4-022
**Description:** Running Exp 3994 either writes `blocked_arc_offline_env_unavailable` when the offline Arcade cannot load, reports `complete: arcmemo_solve_no_transfer_positive_control_failed` when fewer than two banked concepts share structure, stops honestly when no replayable retrieved concept exists for the chosen target, or completes an honest two-arm solve-loop comparison. A transfer win MUST use `success: arcmemo_solve_transfer_v2_<a_cold>to<a_mem>_actions` only when both arms are confirmed by the real offline environment and the memory-seeded arm uses fewer actions or reset attempts than cold start; otherwise it MUST use `complete: arcmemo_solve_no_transfer_to_new_game_<reason>` while preserving measured action and attempt counts.

### REQ-PHASE4-023: ARC-AGI-3 Verifier-Validated Frontier Scaling
**Description:** The system SHALL run Exp 4003 as a scoped ARC-AGI-3 frontier-scaling attempt over the banked `r11l`, `lp85`, and `sc25` frontiers. It MUST replay or re-establish the banked state (`r11l` L3, `lp85` L1, `sc25` L1), re-perceive only the next scoped frontier levels, re-induce level-local candidate rules, validate candidates with GAP-4 executed-consistency energy before spending real-env actions, commit only validated candidates, confirm every level-up through the real environment's `levels_completed` counter, and stop each game at the first failed level rather than chasing all remaining levels. The terminal artifact `results/experiment_4003_scale_level_frontier.json` MUST include bare fields `ACCURACY_total_levels_solved`, `new_levels_this_task`, `per_game_max_level`, `verifier_validated_count`, `actions_saved_vs_openloop`, `per_level_actions`, `baseline_actions_ref`, `real_env_confirmed`, `random_seed`, `honest_verdict`, `duration_s`, and `inference_substrate`.

### SCENARIO-PHASE4-023
**Description:** Running Exp 4003 either writes `blocked_arc_offline_env_unavailable` when the offline Arcade cannot load, or reports an honest terminal-prefix artifact for the three scoped games. A successful frontier advance MUST use `success: scaled_level_frontier_<game>_to_L<n>_total<m>` and record per-game action counts and verifier validations; a no-advance run MUST use `complete: level_frontier_holds_<reason>` and record why `r11l`, `lp85`, and `sc25` stalled without spending actions on verifier-rejected candidates.

### REQ-PHASE4-024: ARC-AGI-3 Fourth Game Explore-First Verifier Pruning
**Description:** The system SHALL run Exp 4004 as a fourth distinct ARC-AGI-3 game solve attempt that fixes Exp 3993's prune-before-induce failure. It MUST first confirm the offline Arcade precondition, choose among `{tn36, su15, dc22}` by smallest L0 baseline action count while rejecting candidates whose survey classification is PSPACE-like spatial planning, actively explore with a fixed positive transition-observation budget before any verifier pruning, induce a grounded dynamics model from observed real-environment transitions, then apply the GAP-4-style executed-consistency verifier as an action pruner only after induction. The terminal artifact `results/experiment_4004_fourth_game_explore_first.json` MUST include bare fields `ACCURACY_levels_solved`, `game_solved`, `exploration_actions_used`, `dynamics_induced`, `first_solve_at_action`, `actions_vs_baseline`, `induced_mechanic`, `games_attempted`, `real_env_confirmed`, `random_seed`, `honest_verdict`, `duration_s`, and `inference_substrate`.

### SCENARIO-PHASE4-024
**Description:** Running Exp 4004 either writes `blocked_arc_offline_env_unavailable` when the offline Arcade cannot load, or records every attempted non-spatial candidate and its observed dynamics before pruning. A successful L1 solve MUST use `success: fourth_game_solved_<game>_at_action<n>`, set `dynamics_induced=true`, set `exploration_actions_used>0`, and confirm the solve through `levels_completed`; a no-solve run MUST use `complete: fourth_game_no_solve_<reason>` while preserving the observed transitions, induced mechanic, and verifier-pruner decisions for each attempt.

### REQ-PHASE4-025: ARC-AGI-3 ArcMemo Solve Transfer V3
**Description:** The system SHALL run Exp 4005 as an ArcMemo solve-loop transfer check against the new content produced in the current milestone, preferring Exp 4004's fourth-game solve, otherwise Exp 4003's higher r11l frontier advance, otherwise a re-held-out solved level. The experiment MUST first confirm the offline Arcade precondition, build concept memory records from banked solved-game mechanics using `{name, when_it_applies, effect}`, require a positive-control shared concept family across at least two banked concepts before attempting transfer, compare cold-start solving against concept-memory-seeded solving at equal offline perception, count real `env.step` actions and reset attempts to the first `levels_completed` increment in each arm, record the concrete concept retrieved and reused, and write `results/experiment_4005_arcmemo_solve_transfer_v3.json` with bare fields `solve_transfer_win`, `actions_cold_start`, `actions_with_memory`, `attempts_cold_start`, `attempts_with_memory`, `target_game`, `concept_reused`, `positive_control_shared_structure`, `real_env_confirmed`, `random_seed`, `honest_verdict`, `duration_s`, and `inference_substrate`.

### SCENARIO-PHASE4-025
**Description:** Running Exp 4005 either writes `blocked_arc_offline_env_unavailable` when the offline Arcade cannot load, reports `complete: arcmemo_solve_no_transfer_positive_control_failed` when fewer than two banked concepts share structure with the target, stops honestly when no replayable retrieved concept exists for the chosen target, or completes an honest two-arm solve-loop comparison. A transfer win MUST use `success: arcmemo_solve_transfer_v3_<a_cold>to<a_mem>_actions` only when both arms are confirmed by the real offline environment and the memory-seeded arm uses fewer actions or reset attempts than cold start; otherwise it MUST use `complete: arcmemo_solve_no_transfer_to_new_content_<reason>` while preserving measured action and attempt counts.
