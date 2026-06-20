# ARC World-Model Trust Energy Capability Specification

**Capability:** arc-world-model-trust-energy
**Version:** 0.1.0
**Status:** Draft

## Overview

Defines the oracle-distinct trust energy used to rank executable ARC-AGI-3
world-model candidates by held-out transition generalization. The capability is
for hidden-state games where executing a proposed simulator is not an oracle for
whether the induced latent mechanic will generalize.

## Requirements

### REQ-ARC-WMTE-4491: Held-Out Trust Energy Ranking

The repository SHALL expose a deterministic world-model trust-energy module for
experiment 4491. Given recorded transitions and at least one candidate engine,
the module SHALL split transitions into observed-prefix and held-out-suffix
parts, score each candidate on both splits, and rank candidates by a calibrated
energy whose target is held-out generalization rather than first observed-prefix
accuracy above a fixed threshold. The module SHALL also report the legacy
`first-clears-0.5` baseline selection so a strict improvement or honest null is
auditable.

### REQ-ARC-WMTE-4492: Oracle-Distinct Artifact Contract

Experiment 4491 SHALL write
`results/experiment_4491_world_model_trust_energy.json` with bare top-level
fields for `honest_verdict`, `inference_substrate`,
`preconditions_checked`, `verifier_is_oracle`, `hidden_state_games_n`,
`trust_energy_pick_rate`, `baseline_pick_rate`, `positive_control_passed`,
`false_negative_risk_guard`, and `selected_candidates`. The artifact SHALL set
`verifier_is_oracle=false` for the hidden-state trust-energy claim and SHALL
record the exact preconditions checked before scoring. Required field principles
SHALL be included for `honest_verdict`, `inference_substrate`, and
`preconditions_checked`.

### REQ-ARC-WMTE-4493: Positive Control and Null Guard

The experiment SHALL include a Markov positive control where transition
execution can adjudicate the best candidate, proving the harness can detect a
real win. If the trust energy does not beat the baseline on hidden-state
candidates, the artifact SHALL still be complete and SHALL report the null
honestly via `false_negative_risk_guard` instead of fabricating an improvement.

### REQ-ARC-WMTE-4494: Live Hidden-State Gate Replacement

The live `E3AgentPolicy` SHALL use the trust-energy selector for hidden-state
world-model candidates, replacing the binary `WorldModelVerifier.accuracy < 0.5`
gate. Markov/non-hidden-state games MAY retain the cheap execution-grounded
accuracy check. A one-candidate hidden-state pool SHALL still pass through the
same selector so the live path and offline experiment share one ranking rule.

### REQ-ARC-WMTE-4495: HUD/Register State and GOAL Predicate Accountability

Experiment 4493 SHALL extend the E3 hidden-state contract from a grid-only
snapshot to an explicit `(grid, registers)` state. The register payload SHALL
carry the induced scalar needed by the held-out L2 stalls, including
`hud_count` for ka59 and the ar25 undo-stack depth when that hidden stack is
available. Candidate `is_level_complete` checks SHALL be allowed to read the
registered state, and the experiment SHALL separately report
`goal_predicate_heldout_score` by comparing the registered GOAL predicate
against recorded level-up transitions rather than scoring dynamics alone.

Experiment 4493 SHALL write
`results/experiment_4493_hud_register_deepen.json` with bare top-level fields
for `honest_verdict`, `inference_substrate`, `preconditions_checked`,
`offline_reproduced`, `reproduced_levels`, `goal_predicate_heldout_score`,
`grid_only_goal_predicate_heldout_score`, `register_state_contract`,
`candidate_reproduction_attempts`, and `residual_blockers`. Required field
principles SHALL be included for:

- `honest_verdict`: MUST start with terminal prefix complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_ (Verdict Terminal-Prefix Discipline).
- `inference_substrate`: explicit (live_llm_inference | verifier_ensemble_against_cached_candidates | aggregation_from_upstream_artifacts) so adversarial_verify applies the right duration floor.
- `preconditions_checked`: records WHICH resources were verified; pre-empts silent-missing-resource fabrication.

A success artifact SHALL require
`offline_reproduced=true` and `reproduced_levels >= 1`; otherwise the artifact
SHALL report a terminal honest residual without fabricating an L2 reproduction.

### REQ-ARC-WMTE-4503: Incremental HUD/Register L2 Deepening

Experiment 4503 SHALL carry the 4493 HUD/register contract forward into the
E3 planner state. The planner state SHALL be explicit `(grid, registers)`,
candidate `is_level_complete` checks SHALL receive that registered state, and
the ka59 state SHALL include an induced `hud_count` scalar while ar25 MAY carry
`undo_stack_depth` when replay data exposes it. The experiment SHALL attempt
one incremental deepening for ar25 or ka59 from the prior L1 reproduction to
L2, and SHALL count only newly reproduced levels beyond L1 as progress.

Experiment 4503 SHALL write
`results/experiment_4503_hud_register_deepen_l2.json` with bare top-level
fields for `honest_verdict`, `inference_substrate`, `preconditions_checked`,
`offline_reproduced`, `reproduced_levels`, `target_game`,
`register_state_contract`, `goal_predicate_heldout_score`,
`grid_only_goal_predicate_heldout_score`, `reproduction_gate`,
`solution_labels`, and `residual_blockers`. Required field principles SHALL be
included for:

- `honest_verdict`: MUST start with terminal prefix complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_.
- `inference_substrate`: explicit substrate so adversarial_verify applies the right duration floor.
- `preconditions_checked`: records WHICH resources were verified; pre-empts silent-missing-resource fabrication.

A success artifact SHALL require `offline_reproduced=true` and
`reproduced_levels >= 1`, where `reproduced_levels` counts a new reproduced
level beyond L1. If the L2 gate remains blocked, the artifact SHALL emit the
residual blocker instead of fabricating an L2 reproduction.

### REQ-ARC-WMTE-4496: Adapter-Routed L1 to L2 Deepening

Experiment 4494 SHALL register one tractable game adapter for a clean L1 game
from `cd82`, `lf52`, `r11l`, or `ls20`. The adapter SHALL derive its
win-predicate and action model from the offline environment state, including
palette/action coordinates when applicable, and SHALL NOT hardcode level
coordinates as the mechanic. The verifier-routed offline solver SHALL use that
adapter to attempt one incremental deepening from the prior L1 reproduction to
L2, and the reproduction gate SHALL replay the adapter labels against the
offline environment before any success claim.

Experiment 4494 SHALL write
`results/experiment_4494_adapter_deepen_l2.json` with bare top-level fields for
`honest_verdict`, `inference_substrate`, `preconditions_checked`,
`offline_reproduced`, `reproduced_levels`, `adapter_registered`,
`target_game`, `reproduction_gate`, `solution_labels`, and
`residual_blockers`. Required field principles SHALL be included for:

- `honest_verdict`: MUST start with terminal prefix complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_ (Verdict Terminal-Prefix Discipline).
- `inference_substrate`: explicit (live_llm_inference | verifier_ensemble_against_cached_candidates | aggregation_from_upstream_artifacts) so adversarial_verify applies the right duration floor.
- `preconditions_checked`: records WHICH resources were verified; pre-empts silent-missing-resource fabrication.

A success artifact SHALL require `offline_reproduced=true` and
`reproduced_levels >= 1`, where `reproduced_levels` counts newly reproduced
levels beyond the prior L1 solve. If no L2 mechanic is reverse-engineerable, the
artifact SHALL report the reverse-engineering delta in `residual_blockers`
without fabricating an L2 reproduction.

### REQ-ARC-WMTE-4504: Adapter-Routed L2 Result Refresh

Experiment 4504 SHALL pick exactly one tractable L1 game from `cd82`, `lf52`,
`r11l`, or `ls20`, use its registered `GameAdapter`, and run the
verifier-routed offline solver plus reproduction gate to deepen from the prior
L1 solve to target level 2. The selected adapter SHALL derive its action labels,
win-predicate distance, and state key from the current offline environment
state, including palette/action coordinates when applicable, and SHALL NOT
hardcode level coordinates as the mechanic.

Experiment 4504 SHALL write
`results/experiment_4504_adapter_deepen_l2.json` with bare top-level fields for
`honest_verdict`, `inference_substrate`, `preconditions_checked`,
`offline_reproduced`, `reproduced_levels`, `adapter_registered`,
`target_game`, `reproduction_gate`, `solution_labels`, and
`residual_blockers`. Required field principles SHALL be included for:

- `honest_verdict`: MUST start with terminal prefix complete:/complete_/success:/success_/passed:/passed_/shipped:/shipped_ (Verdict Terminal-Prefix Discipline).
- `inference_substrate`: explicit (live_llm_inference | verifier_ensemble_against_cached_candidates | aggregation_from_upstream_artifacts) so adversarial_verify applies the right duration floor.
- `preconditions_checked`: records WHICH resources were verified; pre-empts silent-missing-resource fabrication.

A success artifact SHALL require `offline_reproduced=true` and
`reproduced_levels >= 1`, where `reproduced_levels` counts newly reproduced
levels beyond the prior L1 solve. If no L2 mechanic is reverse-engineerable, the
artifact SHALL report the reverse-engineering delta in `residual_blockers`
without fabricating an L2 reproduction.

### REQ-ARC-WMTE-4515: Graph-Explore L1 to L2 Level-Up Attempt Guarantee

Every ARC-sprint roadmap SHALL include at least one level-up attempt that banks
a new reproducible level. Experiment 4515 SHALL pick exactly one graph-explore
family L1 game from `su15`, `sp80`, `cn04`, `m0r0`, or `sk48`, SHALL NOT target
the HUD-register-stall games `ka59` or `ar25`, and SHALL derive the target L2
win-predicate from visible offline environment state before any success claim.
The selected game SHALL register or extend a `GameAdapter` using
`branch_mode='fresh_env'` so verifier-routed candidate paths are evaluated under
the same fresh reset semantics used by the reproduction gate.

Experiment 4515 SHALL write
`results/experiment_4515_deepen_graph_explore_l2.json` with bare top-level
fields for `honest_verdict`, `inference_substrate`,
`preconditions_checked`, `offline_reproduced`, `reproduced_levels`,
`target_game`, `reproducibility_checksum`, `adapter_registered`,
`reproduction_gate`, `solution_labels`, and `residual_blockers`. Required field
principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; e.g. success: <game>_L2_offline_reproduced OR complete: <game>_l2_honest_residual."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade reproduce, no LLM load (1s floor)."
- `offline_reproduced`: principle "a solve not reproducible offline is wasted effort -- only reproduced levels count (ARC Solve Reproducibility)."
- `reproduced_levels`: principle "the banked level count for the game (must be 2 for a successful deepen)."
- `target_game`: principle "the deepen target -- a NEW graph-explore game, not the HUD-register-stall ka59/ar25."
- `reproducibility_checksum`: principle "content-addressed hash of the reproduced replay -- the integrity gate against count inflation."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

A success artifact SHALL require `target_game=m0r0`,
`offline_reproduced=true`, `reproduced_levels=2`, a registered adapter with
`branch_mode='fresh_env'`, a reproduction gate whose `reached_level` is at least
2, and a content-addressed `reproducibility_checksum`. If L2 does not reproduce,
the artifact SHALL report `complete: m0r0_l2_honest_residual` and record the
dead-end in `residual_blockers` without incrementing the banked level count.

### REQ-ARC-WMTE-4524: CORE L1 to L2 Barrier Diagnosis and Lever Sweep

Experiment 4524 SHALL diagnose the live frame-only L1 to L2 barrier on at least
one CORE game with known deeper levels, with `lp85` as the canonical target, and
SHALL measure the available deeper-level levers rather than action-trimming
surrogates. The workflow SHALL run the offline arcade precondition, record the
baseline CORE per-level efficiency control `2.0074`, measure per-game deepest
levels reached for the control and each lever, and classify the terminal barrier
as one of `depth_cap`, `missing_mechanic`, `new_win_condition`,
`induction_not_engaged`, `budget_exhausted`, or `explored_out`.

Experiment 4524 SHALL write
`results/experiment_4524_reach_deeper_levels.json` with bare top-level fields
for `honest_verdict`, `inference_substrate`, `core_efficiency_baseline`,
`core_efficiency_best`, `deepest_level_reached_per_core_game`,
`barrier_diagnosis`, `levers_tried`, `offline_reproduced`, `random_seed`,
`reproducibility_checksum`, and `preconditions_checked`. Required field
principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: <game>_reached_L2_core_efficiency_<n>_above_2.0074 OR complete: l1_l2_barrier_diagnosed_<root_cause>_honest_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade search + world-model/verifier routing (no headline GGUF load); if the LLM induction tier is invoked, declare live_llm_inference + add the model precondition."
- `core_efficiency_baseline`: principle "2.0074 -- the REAL per-level metric control (NOT median actions, which is retired as a score lever)."
- `core_efficiency_best`: principle "the HEADLINE -- did any lever raise per-level efficiency by reaching a deeper level."
- `deepest_level_reached_per_core_game`: principle "best_level per CORE game WITH each lever -- the direct evidence of solving MORE levels (the score lever)."
- `barrier_diagnosis`: principle "the concrete, actionable root cause of the L1->L2 stall (depth-cap / missing mechanic / new win-condition / induction-not-engaged) -- the deliverable when L2 is not reached, and the input to the next milestone."
- `levers_tried`: principle "each L1->L2 lever (deeper search / world-model induction / verifier routing) with its measured effect -- no assumed wins."
- `offline_reproduced`: principle "any new level reached must offline-reproduce (arc_solver_kit.reproduce) to count -- not a one-off live fluke."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

A success artifact SHALL require some CORE game to reach L2 under at least one
lever, `core_efficiency_best > 2.0074`, no CORE game losing a previously reached
level, and `offline_reproduced=true` for the newly reached level. Otherwise the
artifact SHALL report a terminal honest null with the concrete barrier root cause
and SHALL NOT claim action-count reductions as a score win.

## Scenarios

### SCENARIO-ARC-WMTE-4491: Held-Out Ranking Beats First-Clears Baseline

Given candidate engines where the first engine clears 0.5 accuracy on the
observed prefix but fails the held-out suffix, and a later engine generalizes to
the held-out suffix
When the trust-energy selector ranks the candidates
Then it selects the held-out-generalizing engine and records that the baseline
selected the first prefix-clearing engine.

### SCENARIO-ARC-WMTE-4492: Oracle-Distinct Artifact Is Stable JSON

Given cached candidate scorecards and successful import/torch preconditions
When experiment 4491 writes its terminal artifact
Then the artifact is valid JSON with terminal-prefixed `honest_verdict`,
`inference_substrate=verifier_ensemble_against_cached_candidates`,
`verifier_is_oracle=false`, a positive-control result, and explicit
`preconditions_checked`.

### SCENARIO-ARC-WMTE-4493: Hidden-State Null Is Honest

Given hidden-state candidate scorecards where trust energy ties or trails the
baseline
When experiment 4491 writes its terminal artifact
Then it reports a complete null verdict, keeps `verifier_is_oracle=false`, and
sets `false_negative_risk_guard` to the positive-control-backed null state.

### SCENARIO-ARC-WMTE-4494: Registered GOAL Predicate Is Scored Against Level-Ups

Given held-out ka59 or ar25 states whose visible grid dynamics can be predicted
but whose completion condition depends on a latent register
When experiment 4493 scores the candidate GOAL predicate
Then the registered `(grid, registers)` predicate is compared with recorded
level-up transitions, the grid-only baseline score is reported separately, and
the artifact only claims a deepened L2 reproduction when offline replay reaches
at least one new level beyond L1.

### SCENARIO-ARC-WMTE-4503: HUD/Register State Gates One New L2 Level

Given an ar25 or ka59 L1 trajectory and a candidate L2 tail whose completion
predicate depends on a latent register
When experiment 4503 evaluates the registered state and runs the offline
reproduction gate for target level 2
Then `is_level_complete` reads `(grid, registers)`, the artifact records the
selected `target_game`, `solution_labels`, and `reproduction_gate`, and success
is reported only when offline replay reproduces at least one new level beyond
L1.

### SCENARIO-ARC-WMTE-4495: One Adapter Deepens a Frozen L1 Trajectory to L2

Given a clean L1 trajectory for a selected tractable game and an offline
environment exposing the game's target state and available action controls
When experiment 4494 registers the game adapter and runs verifier-routed
deepening to target level 2
Then the terminal artifact records explicit preconditions, the selected
`target_game`, a registered adapter, the adapter-produced `solution_labels`, and
only reports success when the offline reproduction gate reaches at least one new
level beyond L1.

### SCENARIO-ARC-WMTE-4504: cd82 Adapter Reproduces One New L2 Level

Given the registered cd82 adapter and an offline environment exposing palette
sprites, the canvas, and the target sprite
When experiment 4504 runs verifier-routed deepening to target level 2
Then the adapter-produced labels are replayed by the offline reproduction gate,
the terminal artifact records `target_game=cd82`, `adapter_registered=true`,
`offline_reproduced=true`, and `reproduced_levels >= 1`, and success is not
reported unless the gate reaches a new level beyond L1.

### SCENARIO-ARC-WMTE-4515: m0r0 Adapter Banks L2 Offline

Given the prior reproduced m0r0 L1 trajectory and visible L2 environment state
with two `pikgci` player sprites, `wahtyt` blockers, and `spswjz` hazard cells
When experiment 4515 registers the m0r0 `GameAdapter`, derives the coalescence
win-predicate from those visible sprites, and runs `OfflineSolver.solve_level`
from the prior L1 prefix
Then the full L1+L2 label plan is replayed through `arc_solver_kit.reproduce`,
the terminal artifact records `target_game=m0r0`,
`offline_reproduced=true`, `reproduced_levels=2`, and a stable
`reproducibility_checksum`, and success is not reported unless the offline
reproduction gate reaches level 2.

### SCENARIO-ARC-WMTE-4524: lp85 L1 to L2 Barrier Is Measured

Given the offline arcade precondition passes and lp85 has known deeper levels
When experiment 4524 runs the control, deeper-search, world-model induction, and
verifier-routing lever measurements
Then the terminal artifact records the CORE per-level efficiency baseline,
deepest level reached per CORE game for each lever, a concrete L1 to L2 barrier
diagnosis, and reports success only if a CORE game reaches L2 without losing a
CORE level and offline reproduction confirms the new level.
