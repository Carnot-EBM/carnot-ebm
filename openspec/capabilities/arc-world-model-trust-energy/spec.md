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
`core_efficiency_best`, `efficiency_delta`,
`deepest_level_reached_per_core_game`, `barrier_diagnosis`, `levers_tried`,
`offline_reproduced`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`. When `efficiency_delta == 0.0`, the artifact SHALL
also include `null_delta_methodology_note` explaining that baseline equals best
because no lever reached a deeper level, not because of a measurement bug.
Required field
principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: <game>_reached_L2_core_efficiency_<n>_above_2.0074 OR complete: l1_l2_barrier_diagnosed_<root_cause>_honest_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade search + world-model/verifier routing (no headline GGUF load); if the LLM induction tier is invoked, declare live_llm_inference + add the model precondition."
- `core_efficiency_baseline`: principle "2.0074 -- the REAL per-level metric control (NOT median actions, which is retired as a score lever)."
- `core_efficiency_best`: principle "the HEADLINE -- did any lever raise per-level efficiency by reaching a deeper level."
- `efficiency_delta`: principle "core_efficiency_best - core_efficiency_baseline, emitted explicitly so a null (delta 0.0) is annotated, not a control==best TAUTOLOGY false-positive."
- `null_delta_methodology_note`: principle "required when efficiency_delta is 0.0; states that equal baseline/best means no lever moved per-level efficiency, not a measurement bug."
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
and SHALL NOT claim action-count reductions as a score win. The honest-null
artifact SHALL emit `efficiency_delta=0.0` and the
`null_delta_methodology_note` so aggregation can distinguish the annotated
control-vs-best null from an unannotated metric collision.

### REQ-ARC-WMTE-4533: Per-Level Goal Re-Induction Gate

The live `E3AgentPolicy` and `StepwiseExplorer` SHALL treat an increment in
`frame.levels_completed` as a new goal-acquisition episode when
`target_levels > 1`. On every detected level-up, the policy SHALL clear stale
induction state, re-run the executable world-model / DSL induction path against
post-transition frames, and route subsequent frontier search with the newly
induced level-conditioned goal predicate as a bias while preserving depth as
the no-regression primary key. Stall-triggered induction SHALL remain available,
but it SHALL NOT be the only trigger for induction after a won level.

Experiment 4533 SHALL sweep `target_levels` over `{1, 2, 3, 8}` on at least
`lp85` and `m0r0`, report the CORE-preservation check over
`{lp85,m0r0,sp80,vc33}`, and write
`results/experiment_4533_per_level_goal_reinduction.json` with bare top-level
fields for `honest_verdict`, `inference_substrate`, `model_specs`,
`core_efficiency_baseline`, `core_efficiency_best`, `efficiency_delta`,
`deepest_level_reached_per_core_game`, `core_solves_preserved`,
`barrier_refinement`, `target_levels_sweep`, `chosen_submitted_config`,
`positive_control_passed`, `false_negative_risk_checked`,
`offline_reproduced`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`. When `efficiency_delta == 0.0`, the artifact SHALL
also include `null_delta_methodology_note` explaining that baseline equals best
because no lever reached a deeper level, not because of a measurement bug.

Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; e.g. success: reinduction_<game>_reached_L2_core_efficiency_<n>_above_2.0074 OR complete: reinduction_no_deeper_level_barrier_refined_honest_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade search/induction, no headline GGUF load (1s floor). If the LLM induction tier is genuinely invoked, declare live_llm_inference + add the model precondition."
- `model_specs`: principle "names the inducer/model actually invoked (or 'offline_dsl_induction_no_llm') -- the methodology field whose absence false-flagged the .418 A2 as METHODOLOGY_MISSING."
- `core_efficiency_baseline`: principle "2.0074 -- the REAL per-level metric control (NOT median actions, retired)."
- `core_efficiency_best`: principle "the HEADLINE -- did per-level re-induction reach a deeper level and raise core_efficiency."
- `efficiency_delta`: principle "core_efficiency_best - core_efficiency_baseline, emitted explicitly so a null (delta 0.0) is annotated, not a control==best TAUTOLOGY false-positive."
- `deepest_level_reached_per_core_game`: principle "best_level per CORE game per config -- the direct evidence of reaching MORE levels (the score lever)."
- `core_solves_preserved`: principle "HARD empirical gate on {lp85,m0r0,sp80,vc33}; a dropped CORE solve fails the lever regardless."
- `barrier_refinement`: principle "if no deeper level is reached, the concrete actionable refinement of what still blocks the re-induced L2 predicate."
- `target_levels_sweep`: principle "the {target_levels -> (core_efficiency, deepest_level, core_solves_preserved)} table so the decision is auditable."
- `chosen_submitted_config`: principle "what was wired into SUBMITTED_AGENT_CONFIG (or 'unchanged' if null); must keep parity tests consistent."
- `positive_control_passed`: principle "proves the harness can detect a real re-induction through a level-conditioned predicate change."
- `false_negative_risk_checked`: principle "a null is valid only if the positive control passed."
- `offline_reproduced`: principle "any new level reached must offline-reproduce to count."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent corpus/model drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

A success artifact SHALL require some CORE game to reach L2, a strict
`core_efficiency_best > 2.0074`, `core_solves_preserved=true`, and
`offline_reproduced=true` for any newly reached level before changing
`SUBMITTED_AGENT_CONFIG`. Otherwise the submitted configuration SHALL remain
unchanged and the artifact SHALL report the honest null plus barrier refinement.

### REQ-ARC-WMTE-4534: Trust-Energy Next-Level Frontier Routing

Experiment 4534 SHALL use the oracle-distinct world-model trust energy as a
next-level-distance heuristic after the per-level goal re-induction attempt.
When a re-induced `L_{n+1}` predicate is available from experiment 4533, the
frontier SHALL preserve depth as the no-regression primary key and use
lower-is-closer trust energy only as the goal-bias tie/secondary key. When
experiment 4533 produced an honest null and no live `L_{n+1}` predicate is
available, the experiment SHALL still compare the matched cached no-energy and
energy-routed measurements on `lp85` and `sp80`, and SHALL independently
characterize whether the trust-energy signal separates deeper-progress states
from L1-stuck states.

Experiment 4534 SHALL write
`results/experiment_4534_energy_trust_next_level_routing.json` with bare
top-level fields for `honest_verdict`, `inference_substrate`,
`verifier_is_oracle`, `core_efficiency_baseline`,
`core_efficiency_energy_routed`, `no_energy_control`,
`energy_separation_auroc`, `deepest_level_reached_per_core_game`,
`core_solves_preserved`, `positive_control_passed`,
`false_negative_risk_checked`, `random_seed`,
`reproducibility_checksum`, and `preconditions_checked`. Required field
principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: energy_routing_<game>_reached_L2_oracle_distinct OR complete: energy_routing_no_deeper_level_signal_characterized_honest_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade + trust-energy scoring, no headline GGUF load (1s floor)."
- `verifier_is_oracle`: principle "MUST be false -- the trust energy is oracle-DISTINCT (ranks held-out generalization), NOT the executable win-check; a circular win does not count (Circularity / Oracle-Distinctness Discipline)."
- `core_efficiency_baseline`: principle "2.0074 -- the per-level metric control."
- `core_efficiency_energy_routed`: principle "the HEADLINE -- did energy routing toward the re-induced goal reach a deeper level."
- `no_energy_control`: principle "matched control measured the SAME way (energy off) -- the apples-to-apples comparison."
- `energy_separation_auroc`: principle "AUROC of the trust energy separating deeper-progress states from L1-stuck states -- the signal characterization that is the deliverable on a null."
- `deepest_level_reached_per_core_game`: principle "best_level per CORE game with vs without energy routing -- the score-lever evidence."
- `core_solves_preserved`: principle "HARD gate on {lp85,m0r0,sp80,vc33}; a dropped CORE solve FAILS the lever."
- `positive_control_passed`: principle "the energy SHOULD separate deeper-progress on a known-L2 game; guards a silently-broken signal."
- `false_negative_risk_checked`: principle "a null is valid only if the positive control passed."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

A success artifact SHALL require an energy-routed configuration to reach level
2 or deeper on a CORE game where the matched no-energy control does not,
`core_solves_preserved=true`, `positive_control_passed=true`, and
`verifier_is_oracle=false`. Otherwise the artifact SHALL report an honest null,
including `energy_separation_auroc` and the positive-control status, without
claiming a circular win or changing the submitted configuration.

### REQ-ARC-WMTE-4544: Live LLM Proposer Re-Induction Gate

Experiment 4544 SHALL replace the empty offline-DSL post-level proposer with the
frozen local GGUF generator already wired through
`E3AgentPolicy._proposer()`: Qwen3.5-9B-MTP via `LocalGGUFProposer`. On every
level-up re-induction trigger, the policy SHALL ask the generator for separate
GOAL predicate and DYNAMICS/world-model candidates, rank executable candidates
with the oracle-distinct world-model trust energy, plan toward the selected
GOAL predicate, and run a bounded counterexample-guided refinement loop of at
most three rounds. Each refinement round SHALL feed the verifier's first
counterexample or plan-reachability failure back to the proposer, and SHALL stop
as soon as a candidate plan reaches the induced goal in the selected world
model. `CARNOT_ARC_DISABLE_INDUCTION=1` SHALL remain a control-only escape hatch
and SHALL NOT be set for the live LLM proposer condition.

Experiment 4544 SHALL measure the matched offline-DSL baseline and LLM-proposer
condition the same way on `lp85` and `m0r0`, report the CORE-preservation check
over `{lp85,m0r0,sp80,vc33}`, characterize `llm_proposer_value` as the count
and rate of level-ups where the LLM produced a reachable plan the offline DSL
did not, and write
`results/experiment_4544_llm_proposer_reinduction.json` with bare top-level
fields for `honest_verdict`, `inference_substrate`, `model_specs`,
`core_efficiency_baseline`, `core_efficiency_best`, `efficiency_delta`,
`null_delta_methodology_note` when the delta is zero, `llm_proposer_value`,
`deepest_level_reached_per_core_game`, `core_solves_preserved`,
`refinement_rounds_used`, `barrier_refinement`, `chosen_submitted_config`,
`positive_control_passed`, `false_negative_risk_checked`,
`verifier_is_oracle`, `offline_reproduced`, `random_seed`,
`reproducibility_checksum`, and `preconditions_checked`.

Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; e.g. success: llm_proposer_<game>_reached_L2_core_efficiency_<n>_above_2.0074 OR complete: llm_proposer_no_deeper_level_proposer_value_characterized_honest_null."
- `inference_substrate`: principle "live_llm_inference -- the Qwen3.5-9B-MTP generator is genuinely loaded + invoked (60s floor)."
- `model_specs`: principle "names the generator actually invoked (Qwen3.5-9B-MTP GGUF + the resolved .gguf path)."
- `core_efficiency_baseline`: principle "2.0074 -- the REAL per-level metric control, measured the SAME way as best."
- `core_efficiency_best`: principle "the HEADLINE -- did the LLM proposer reach a deeper CORE level and raise core_efficiency."
- `efficiency_delta`: principle "core_efficiency_best - core_efficiency_baseline, emitted explicitly so a null delta is annotated."
- `null_delta_methodology_note`: principle "present when efficiency_delta==0.0 -- states the equality is an honest no-deeper-level null, not a measurement bug."
- `llm_proposer_value`: principle "the count/rate of level-ups where the LLM proposer produced a REACHABLE plan the offline DSL could not."
- `deepest_level_reached_per_core_game`: principle "best_level per CORE game per condition -- direct evidence of reaching MORE levels."
- `core_solves_preserved`: principle "HARD empirical gate on {lp85,m0r0,sp80,vc33}; a dropped CORE solve fails the lever."
- `refinement_rounds_used`: principle "bounded counterexample-guided refinement rounds per level-up -- proves the loop ran and stayed bounded."
- `barrier_refinement`: principle "if no deeper CORE level is reached, the concrete actionable refinement of what the LLM plan still gets wrong."
- `chosen_submitted_config`: principle "what was wired into SUBMITTED_AGENT_CONFIG (or 'unchanged' if null); parity tests must stay consistent."
- `positive_control_passed`: principle "proves the LLM proposer can produce a reachable plan the DSL could not on a known-L2 game."
- `false_negative_risk_checked`: principle "a null is valid only if the positive control passed."
- `verifier_is_oracle`: principle "false -- the world-model trust energy ranks candidates by held-out generalization, oracle-DISTINCT."
- `offline_reproduced`: principle "any new level reached must offline-reproduce to count."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent corpus/model drift on replay."
- `preconditions_checked`: principle "records offline arcade, Qwen3.5-9B-MTP GGUF cache, and llama_cpp checks; pre-empts missing-resource fabrication."

A success artifact SHALL require some CORE game to reach L2 under the LLM
proposer, `core_efficiency_best > 2.0074`, `core_solves_preserved=true`,
`positive_control_passed=true`, `verifier_is_oracle=false`, and
`offline_reproduced=true` for any new level before changing
`SUBMITTED_AGENT_CONFIG` or raising `SUBMITTED_TARGET_LEVELS`. Otherwise the
submitted configuration SHALL remain unchanged and the artifact SHALL report an
honest null with `efficiency_delta=0.0`, `llm_proposer_value` characterized, and
a concrete `barrier_refinement`.

### REQ-ARC-WMTE-4551: Offline/Live Proposer Parity Guard

Experiment 4551 SHALL add a no-model regression guard that compares the local
submission gate's effective proposer/induction configuration against
`SUBMITTED_AGENT_CONFIG`. The guard SHALL identify when
`scripts/kaggle/arc_local_submission_gate.py` sets
`CARNOT_ARC_DISABLE_INDUCTION=1` while the submitted
`make_carnot_agent -> E3AgentPolicy` path ships live proposer induction through
`E3AgentPolicy._proposer()`. On a mismatch, the gate SHALL emit
`proposer_config_mismatch=true` and a specific divergence record instead of
silently presenting the bare-explorer `core_efficiency` as the submitted
agent's live-proposer number. When the effective offline configuration matches
the submitted proposer/induction configuration, the guard SHALL pass cleanly
with `proposer_config_mismatch=false`.

Experiment 4551 SHALL write
`results/experiment_4551_offline_live_proposer_parity.json` with bare top-level
fields for `honest_verdict`, `inference_substrate`,
`parity_guard_mechanism`, `proposer_config_mismatch_detected`,
`tests_added_pass`, and `preconditions_checked`.

Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; shipped: offline_live_proposer_parity_guard_added OR complete: proposer_parity_partial_<reason>."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- runs the gate/parity logic against fixtures, no model load (1s floor)."
- `parity_guard_mechanism`: principle "names the parity check + where it fires -- the fix that stops the offline gate silently understating the submitted agent."
- `proposer_config_mismatch_detected`: principle "whether the .419-state offline gate (induction disabled) is correctly flagged as a mismatch vs the LLM-proposer SUBMITTED config -- the concrete bug this catches."
- `tests_added_pass`: principle "Tests Must Run and Assert -- both the mismatch-fires and the matched-config-clean cases."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4551-PROPOSER-PARITY

**Given** `SUBMITTED_AGENT_CONFIG` declares `policy=E3AgentPolicy` with the live
LLM-proposer induction path available and the local gate fixture disables
induction via `CARNOT_ARC_DISABLE_INDUCTION=1`
**When** the offline/live proposer parity guard runs
**Then** it reports `proposer_config_mismatch=true`, names the disabled
induction divergence, and exposes the mismatch in the local gate JSON payload.

**Given** a local gate fixture whose effective induction/proposer configuration
matches the submitted E3 live-proposer configuration
**When** the same guard runs
**Then** it reports `proposer_config_mismatch=false` and emits no divergence.

### REQ-ARC-WMTE-4557: Family-B Executable World-Model Proposer Gate

Experiment 4557 SHALL rebuild the per-level LLM re-induction proposer as a
Family-B executable Python world-model inducer. The proposer SHALL emit
`engine(grid, action, data)` and `is_level_complete(grid)` candidates, the
selector SHALL verify candidates against held-out transitions that were not
included in the induction prompt, and planning SHALL run only through accepted
GOAL+DYNAMICS candidates whose held-out executable transition score clears the
configured verifier threshold. The bounded counterexample-guided refinement
loop SHALL run for at most three rounds, feeding the first held-out transition
failure or plan reachability counterexample back to the proposer before retry.

Experiment 4557 SHALL run the known reachable executable-proposer positive
control before any CORE efficiency measurement. If the positive control does
not produce a verified executable world model and a reachable plan while the
offline DSL has no reachable fixture plan, the experiment SHALL write
`results/experiment_4557_executable_world_model_proposer.json` with the honest
barrier and SHALL NOT measure or fabricate `core_efficiency_best` or
`efficiency_delta`. Only when the positive control passes SHALL the experiment
measure the executable proposer against the matched offline-DSL baseline on
`lp85` and `m0r0`, report the CORE-preservation games `sp80` and `vc33`, and
characterize `llm_proposer_value` as the count and rate of level-ups where the
executable proposer produced a reachable plan the DSL did not.

Experiment 4557 SHALL write
`results/experiment_4557_executable_world_model_proposer.json` with bare
top-level fields for `honest_verdict`, `inference_substrate`, `model_specs`,
`positive_control_passed`, `false_negative_risk_checked`,
`core_efficiency_baseline`, `core_efficiency_best`, `efficiency_delta`,
`null_delta_methodology_note` when the delta is zero, `llm_proposer_value`,
`deepest_level_reached_per_core_game`, `core_solves_preserved`,
`refinement_rounds_used`, `barrier_refinement`, `verifier_is_oracle`,
`chosen_submitted_config`, `offline_reproduced`, `random_seed`,
`reproducibility_checksum`, and `preconditions_checked`.

Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: executable_proposer_<game>_reached_L2_core_efficiency_<n>_above_2.0074 OR complete: executable_proposer_positive_control_<passed|failed>_<deeper|no_deeper>_barrier_refined."
- `inference_substrate`: principle "live_llm_inference -- the Qwen3.5-9B-MTP generator is genuinely loaded + invoked (60s floor); declare it so adversarial_verify does not false-flag DURATION_TOO_SHORT."
- `model_specs`: principle "names the generator actually invoked (Qwen3.5-9B-MTP GGUF + resolved .gguf path) -- the methodology field whose absence false-flagged the .419 A2."
- `positive_control_passed`: principle "THE GATE -- the executable proposer must produce the known reachable fixture plan FIRST (the .420 break); an efficiency claim is invalid without it."
- `false_negative_risk_checked`: principle "a no-deeper-level null is valid only if positive_control_passed=True (else the proposer is broken, not the idea -- the .420 uninformative-null trap)."
- `core_efficiency_baseline`: principle "2.0074 -- the REAL per-level metric control (.420/.419 baseline, measured the SAME way)."
- `core_efficiency_best`: principle "the HEADLINE -- did the executable proposer reach a deeper CORE level and raise core_efficiency."
- `efficiency_delta`: principle "core_efficiency_best - baseline, emitted explicitly so a null (0.0) is annotated, not a control==best TAUTOLOGY false-positive."
- `null_delta_methodology_note`: principle "present when efficiency_delta==0.0 -- states the equality is an honest no-deeper-level null, not a measurement bug."
- `llm_proposer_value`: principle "count/rate of level-ups where the executable proposer produced a REACHABLE plan the offline DSL could not -- the measured value (vs .420's count=0)."
- `deepest_level_reached_per_core_game`: principle "best_level per CORE game per condition -- the direct score-lever evidence."
- `core_solves_preserved`: principle "HARD empirical gate on {lp85,m0r0,sp80,vc33} -- a dropped CORE solve FAILS the lever regardless."
- `refinement_rounds_used`: principle "the bounded counterexample-guided rounds per level-up -- proves the loop ran + bounds wall-clock."
- `barrier_refinement`: principle "if the positive control fails or no deeper level is reached, the CONCRETE actionable refinement of what the executable proposer still gets wrong -- the deliverable on a null + the retire/.422 input."
- `verifier_is_oracle`: principle "false -- the world-model trust energy RANKS candidate executable models by held-out generalization, oracle-DISTINCT (the LLM generates; the energy verifies)."
- `chosen_submitted_config`: principle "what was wired into SUBMITTED_AGENT_CONFIG (or 'unchanged' if null); keeps test_arc_submitted_agent_parity.py consistent."
- `offline_reproduced`: principle "any new level must offline-reproduce (arc_solver_kit.reproduce) to count."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent corpus/model drift on replay."
- `preconditions_checked`: principle "records resources verified (offline arcade, Qwen3.5-9B-MTP cached, llama_cpp); pre-empts missing-resource fabrication."

A success artifact SHALL require a CORE game to reach L2 under the executable
proposer, `core_efficiency_best > 2.0074`, `core_solves_preserved=true`,
`positive_control_passed=true`, `verifier_is_oracle=false`, and
`offline_reproduced=true` for any new level before changing
`SUBMITTED_AGENT_CONFIG` or raising `SUBMITTED_TARGET_LEVELS`. Otherwise the
submitted configuration SHALL remain unchanged and the artifact SHALL report the
honest barrier. If the positive control fails, `core_efficiency_best` and
`efficiency_delta` SHALL remain unmeasured rather than being set equal to the
baseline.

#### SCENARIO-ARC-WMTE-4557-POSITIVE-CONTROL-FIRST

**Given** the offline arcade, Qwen3.5-9B-MTP GGUF cache, `llama_cpp`, and
REQ-ARC-WMTE-4557 spec preconditions all pass
**When** experiment 4557 runs the executable-proposer positive control
**Then** CORE efficiency measurement is skipped unless the proposer emits a
held-out-verified executable world model and a plan that reaches the fixture
goal while the DSL control remains unreachable.

**Given** a proposed executable world-model candidate reproduces the induction
prefix but fails a held-out transition
**When** the bounded re-induction helper evaluates candidates with the
Family-B verifier threshold enabled
**Then** the candidate is rejected before planning, a
`heldout_transition_verification_failed` counterexample is recorded, and the
proposer receives at most three total refinement rounds.

### REQ-ARC-WMTE-4664: L2 Goal Predicate Satisfiability and Multi-Level Harness Fix

The live E3 L1-to-L2 re-induction path SHALL pass the grid that completed the
previous level into the next-level LLM induction prompt as a WIN-STATE exemplar
even when the active post-boundary transition window has no next-level win
positive. The prompt SHALL label that grid as "a state that COMPLETED the
previous level; the next level's completion likely looks structurally similar"
so the generated `is_level_complete(grid)` is conditioned on a positive
level-complete exemplar without treating it as a verified L2 oracle.

After an executable world-model candidate clears the existing held-out DYNAMICS
gate and before planning, the bounded LLM re-induction helper SHALL evaluate
the induced `is_level_complete` predicate over the reachable grids produced by
the accepted model from the current start grid. If the predicate is never true
over that reachable set, the helper SHALL reject the candidate with a
counterexample of kind `degenerate_goal_predicate`, feed it into the bounded
refinement loop, and SHALL NOT call the planner for that candidate. The
held-out transition verifier remains DYNAMICS-only; this requirement adds the
missing GOAL satisfiability verification.

The live multi-level measurement harness SHALL be able to measure depth `>=2`:
its E3 policies SHALL target at least two levels, the rollout SHALL continue
after the first level-up until the policy is done or the budget expires, and
any local live proposer used for the LLM arm SHALL run on a non-colliding port
with `/props` verification that the served model is Qwen3.5-9B-MTP rather than
the persistent gemma server on port 8919.

Experiment 4664 SHALL write
`results/experiment_4664_l2_goal_predicate_induction_live.json` with
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `verifier_is_oracle`, `solve_provenance`,
`live_path_reachable`, `win_state_exemplar_injected`,
`goal_predicate_satisfiable`, `l2_plan_len`, `l2_plan_reaches_goal`,
`metric_harness_fixed`, `proposer_served_model`,
`generic_agent_reached_level`, `offline_reproduced`, `reproduced_levels`,
`residual_cause_hypothesis`, `bare_control_passed`,
`false_negative_risk_checked`, `parity_test_green`, `random_seed`,
`reproducibility_checksum`, and `preconditions_checked`. If no target reaches
L2 or any goal predicate remains degenerate, the artifact SHALL include
`null_methodology_note` explaining the honest null and residual cause.

Required field principles SHALL include:

- `honest_verdict`: "terminal prefix; success: l2_goal_induction_generic_agent_reached_L2_<games> OR complete: l2_goal_induction_no_deepening_residual_<cause>."
- `inference_substrate`: "live_llm_inference -- the L2 induction loads + runs the Qwen3.5-9B-MTP GGUF (60s duration floor); declared honestly because the induction arm is a real LLM run."
- `verifier_is_oracle`: "MUST be false -- the induced is_level_complete goal predicate is oracle-DISTINCT from the executable reproduction win-check."
- `solve_provenance`: "live_agent_self_discovery -- a generic-agent L2 via the fixed runtime induction is the REAL deliverable, NOT a hand-built GameAdapter (development_proxy) and NOT outer_loop_re."
- `live_path_reachable`: "HARD gate -- the changed modules (arc_competition_agent/arc_llm_reinduction/arc_executable_world_model) are in the E3AgentPolicy import closure; arc_orphan_solver_lint passes."
- `win_state_exemplar_injected`: "the L1 win-grid captured at _begin_level_goal_episode and injected into the L2 induce prompt's WIN-STATE block (the missing positive exemplar -- the precise root-cause fix)."
- `goal_predicate_satisfiable`: "the held-out gate checks DYNAMICS only; a constant-False goal sails through today and yields no_reachable_plan. This field records the induced L2 goal is True on >=1 reachable grid -- the missing verification."
- `l2_plan_len`: "plan_len=0 was the measured failure; non-empty is the fix working."
- `l2_plan_reaches_goal`: "reaches_goal=False (no_reachable_plan) was the measured failure; True is the fix working."
- `metric_harness_fixed`: "the degenerate live_multi_level_solve_rate harness fixed (target_levels>=2 + no early break + non-colliding /props-verified Qwen port) so any lever is measurable -- depth>=2 was impossible by construction before."
- `proposer_served_model`: "the model the proposer /props reported (MUST be Qwen3.5-9B-MTP, NOT gemma) -- the port-8919 confound guard so the measurement is on the declared model."
- `generic_agent_reached_level`: "per-game (lp85, sc25) the deepest level the GENERIC live agent reached via the fixed induction -- the headline (>=2 is the win)."
- `offline_reproduced`: "a generic L2 counts only if offline-reproduced via arc_solver_kit.reproduce (ARC Solve Reproducibility); a live-only trajectory is provisional."
- `reproduced_levels`: "the integer new-level count the generic agent banked offline (>=1 at L2 is the bridge crossed)."
- `residual_cause_hypothesis`: "if the fix nulls, names the residual (single_exemplar_goal_insufficient | l2_dynamics_wrong) -- the .431 target; 'none' if it crossed."
- `null_methodology_note`: "present when a goal stays degenerate / no L2 -- states the null is honest (passing controls), not a measurement bug."
- `bare_control_passed`: "the POSITIVE CONTROL -- lp85/sc25 reach L1 + have L2 reachable per registry (headroom exists); a no-L2 null is valid only then."
- `false_negative_risk_checked`: "true with both games' L1 reach + registry-L2-reachable confirmed -- a 'no deepening' null is valid only then."
- `parity_test_green`: "HARD gate -- test_arc_submitted_agent_parity.py passes; the deployed agent == the measured agent."
- `random_seed`: "determinism precondition for reproducibility."
- `reproducibility_checksum`: "content-addressed hash catches silent harness/corpus drift on replay."
- `preconditions_checked`: "records resources verified (Qwen cached, offline arcade, live modules importable, /props served Qwen on a free port); pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4664-WIN-STATE-EXEMPLAR

**Given** a level-up boundary was observed and the active L2 transition window
contains no L2 win positive
**When** the live E3 policy starts the next-level goal episode and invokes
executable world-model induction
**Then** the induction prompt includes the captured previous-level-complete grid
in a WIN-STATE block with the structural-similarity label required by
REQ-ARC-WMTE-4664.

#### SCENARIO-ARC-WMTE-4664-GOAL-SATISFIABILITY

**Given** an induced executable dynamics model passes the held-out transition
gate but its `is_level_complete` predicate is false on every reachable grid
sampled from the accepted model
**When** bounded LLM re-induction evaluates the candidate
**Then** the candidate is rejected before planning with counterexample kind
`degenerate_goal_predicate`, refinement is attempted within the K<=3 bound, and
a later candidate may plan only after the goal predicate is satisfiable on at
least one reachable grid.

#### SCENARIO-ARC-WMTE-4664-METRIC-HARNESS

**Given** the live multi-level metric harness rolls out E3 policy attempts
**When** a game reaches its first level-up before the action budget is exhausted
**Then** the rollout does not break at that first level-up, the policy targets
at least two levels, and the measurement can record a depth `>=2` attempt when
the generic live agent reaches L2.

### REQ-ARC-WMTE-4604: Change-Weighted Trust Energy Gate

The live hidden-state `E3AgentPolicy` world-model gate SHALL replace the
degenerate full-grid binary exact-match threshold with an oracle-distinct trust
energy. The trust score SHALL split recorded transitions into an observed
prefix and held-out suffix, SHALL score held-out grid-changing transitions only
for the headline consistency metric, and SHALL compute change-weighted
agreement as correctly predicted changed cells divided by true changed cells.
The gate SHALL require at least one correctly predicted real changed cell before
planning so an identity/no-op world model cannot pass on no-op-heavy or
click-heavy games. The legacy binary exact-full-grid gate SHALL remain
available as a matched no-regression control.

Experiment 4604 SHALL write
`results/experiment_4604_world_model_trust_energy.json` with bare top-level
fields for `honest_verdict`, `inference_substrate`, `verifier_is_oracle`,
`solve_provenance`, `world_model_trust_pass_rate_new`,
`world_model_trust_pass_rate_binary`, `trust_pass_rate_delta`,
`first_win_rate_new`, `first_win_delta`, `first_win_ci`,
`median_actions_to_first_levelup_new`, `identity_engine_rejected`,
`binary_gate_control_passed`, `false_negative_risk_checked`,
`solve_rate_preserved`, `chosen_submitted_config`,
`residual_world_model_gaps`, `offline_reproduced`, `random_seed`,
`reproducibility_checksum`, and `preconditions_checked`. If
`trust_pass_rate_delta == 0`, the artifact SHALL include
`null_delta_methodology_note`.

Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: world_model_trust_energy_pass_rate_up_<n>_first_win_up OR complete: world_model_trust_energy_no_value_honest_null_residual_logged."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline trust-gate scoring over cached transitions (1s floor); if the optional LLM-induction arm runs, declare live_llm_inference for THAT arm + the Qwen3.5-9B-MTP iGPU precondition (NEVER the 3090s)."
- `verifier_is_oracle`: principle "MUST be false -- the trust energy ranks induced world-models by HELD-OUT generalization, oracle-DISTINCT from running the executable win-check (a circular trust claim would not count)."
- `solve_provenance`: principle "live_agent_self_discovery if the fixed trust gate lets the SCORED agent's own planner reach a level it previously could not; development_proxy if measured via the offline twin. NOT outer_loop_re."
- `world_model_trust_pass_rate_new`: principle "the HEADLINE -- fraction of world-model games where the new change-weighted+held-out gate PASSES a world-model path AND the planner USES it; > the binary-gate baseline is the 0.08-wall crack."
- `world_model_trust_pass_rate_binary`: principle "the matched binary exact-match-gate baseline on the SAME games (the apples-to-apples control; the current 0/6 failure surface)."
- `trust_pass_rate_delta`: principle "new - binary (positive = the degenerate gate fixed), emitted explicitly so a null (0) is annotated."
- `first_win_rate_new`: principle "first-win-rate on held-out variants WITH the new gate (the downstream effect of using the world-model path)."
- `first_win_delta`: principle "new - binary first-win-rate, emitted explicitly so a null (0.0) is annotated."
- `first_win_ci`: principle "bootstrap CI on the first-win delta; a claim above baseline requires the CI to exclude the binary-gate baseline."
- `median_actions_to_first_levelup_new`: principle "ACTION cost WITH the new gate -- the leaderboard tiebreaker (RHAE rewards efficiency)."
- `identity_engine_rejected`: principle "HARD non-degeneracy assertion -- an IDENTITY world-model (predict no change) MUST be REJECTED by the new gate on a no-op-heavy game (the GAP-WM-TRUST-GATE failure mode the binary gate false-passed)."
- `binary_gate_control_passed`: principle "the POSITIVE CONTROL -- the new gate must beat the binary gate on the SAME games; a null is valid only if this ran (no broken-control trap)."
- `false_negative_risk_checked`: principle "true with the binary-gate control run -- a no-value null is valid only then."
- `null_delta_methodology_note`: principle "present when trust_pass_rate_delta==0 -- states the equality is an honest no-value null, not a measurement bug."
- `solve_rate_preserved`: principle "HARD gate -- the new gate must NOT drop solve-rate on games the binary gate already solved."
- `chosen_submitted_config`: principle "what (if anything) is recommended for SUBMITTED_AGENT_CONFIG (enable the trust-energy gate) -- the A6 input; 'unchanged' if null."
- `residual_world_model_gaps`: principle "which world-model game still 0-passes after the fix -- the Missing-Verifier Gap Logging entry."
- `offline_reproduced`: principle "any newly-solved level must offline-reproduce to count."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent harness/corpus drift on replay."
- `preconditions_checked`: principle "records resources verified (offline arcade, WorldModelVerifier importable); pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4604-IDENTITY-REJECTED

**Given** a no-op-heavy transition set with at least one real grid-changing
transition
**When** the new change-weighted held-out trust gate scores an identity engine
that predicts no changes
**Then** the engine is rejected because it predicts zero real changed cells
correctly, even if the legacy full-grid binary gate rewards no-op transitions.

#### SCENARIO-ARC-WMTE-4604-BINARY-CONTROL

**Given** the same hidden-state world-model games and held-out variants are
measured under the new trust-energy gate and under the legacy binary
exact-match gate
**When** experiment 4604 writes its artifact
**Then** it reports both pass rates, their delta, first-win delta with bootstrap
CI, solve-rate preservation, residual zero-pass games, and a terminal honest
verdict without treating the held-out trust energy as an oracle win-check.

### REQ-ARC-WMTE-4608: Persist Winning Trust Primitive And Measure Untuned Transfer

Experiment 4608 SHALL read
`results/experiment_4604_world_model_trust_energy.json` and
`results/experiment_4605_live_integration_scored_agent.json`, select the
strongest measured reusable primitive, persist it in `arc_solver_kit`, and
record the reusable asset in `ops/arc_solve_registry.yaml` under
`general_gotchas`. Because experiment 4604 raised both
`world_model_trust_pass_rate` and first-win-rate while experiment 4605 reported
an honest first-win null, the expected winner is the A1 trust-energy gate
persisted as `world_model_trust_energy_gate_operator` with registry gotcha
`primitive_world_model_trust_energy_gate_operator`.

The solver kit operator SHALL accept transition rows and world-model candidates,
rank them with the change-weighted held-out trust energy, require
`verifier_is_oracle=false`, reject identity/no-op degeneracy, and report value
against the legacy binary exact-match gate before any solve claim. Experiment
4608 SHALL apply the persisted primitive to two or more games not tuned in A1
or A2, offline-reproduce any newly counted level through
`arc_solver_kit.reproduce`, and write
`results/experiment_4608_primitive_persist_transfer.json`.

The artifact SHALL emit principle-annotated top-level fields for
`honest_verdict`, `inference_substrate`, `verifier_is_oracle`,
`solve_provenance`, `primitive_persisted`, `transfer_games`,
`transfer_value_per_game`, `offline_reproduced`, `registry_updated`,
`random_seed`, `reproducibility_checksum`, and `preconditions_checked`. Required
field principles are:

- `honest_verdict`: principle "terminal prefix; success: primitive_persisted_transfer_<game>_value_added OR complete: primitive_persisted_transfer_null_characterized."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates for the offline transfer; declared so a fast real run is not DURATION_TOO_SHORT/METHODOLOGY false-flagged."
- `verifier_is_oracle`: principle "MUST be false -- the persisted primitive RANKS/ROUTES (trust energy) or WIRES the verifier, oracle-distinct from the win-check."
- `solve_provenance`: principle "development_proxy if a transfer solve is via the offline twin; live_agent_self_discovery if the persisted primitive improves the SCORED agent's own path. NOT outer_loop_re."
- `primitive_persisted`: principle "names the arc_solver_kit operator + registry general_gotcha id added/extended -- the reusable asset (Solver-Reuse Discipline); without it the A1/A2 effort is wasted per the ARC reuse rule."
- `transfer_games`: principle "the games the primitive was applied to (NOT tuned on) -- the generalization test."
- `transfer_value_per_game`: principle "the per-game value-add (trust-pass / first-win lift) -- the cross-game evidence the primitive generalizes."
- `offline_reproduced`: principle "only offline-reproduced new levels count toward reproducible_total_levels."
- `registry_updated`: principle "the primitive + transfer dead-ends persisted so the next milestone reuses, not re-derives."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

If no untuned transfer game shows positive trust-pass or first-win lift, the
artifact SHALL report `complete: primitive_persisted_transfer_null_characterized`,
persist the best-characterized primitive-as-built, record transfer dead-ends in
the registry, and SHALL NOT increment `reproducible_total_levels`.

#### SCENARIO-ARC-WMTE-4608

**Given** the offline arcade precondition passes, A1 and A2 artifacts exist, and
the A1 trust-energy gate has the stronger measured signal
**When** experiment 4608 persists the solver-kit trust operator and measures it
on at least two untuned transfer games
**Then** the artifact records the persisted operator and registry gotcha, the
untuned transfer games, per-game trust-pass or first-win value-add, offline
reproduction accounting for any new level, a stable checksum, and a terminal
success only when at least one transfer game adds value.

### REQ-ARC-WMTE-4609: Submitted Agent Integration Gate Consolidation

Experiment 4609 SHALL consolidate the ARC sprint integration candidates into
the scored submitted agent configuration. The workflow SHALL run the offline
arcade precondition and SHALL read A1, A2, A3, and A4 through
`scripts/summarize_artifact.py` before importing any metric. A lever may be
admitted only when its upstream artifact has a terminal success verdict, is not
stamped or live-rechecked as adversarial, and passes its positive control. Any
flagged-adversarial, positive-control-failed, false-negative-risk, or non-success
artifact SHALL be quarantined and SHALL NOT contribute headline numbers.

Experiment 4609 SHALL write `results/experiment_4609_integration_gate.json`
with bare top-level fields for `honest_verdict`, `inference_substrate`,
`verifier_is_oracle`, `levers_integrated`, `flagged_artifacts_excluded`,
`world_model_trust_pass_rate_integrated`, `first_win_rate_integrated`,
`median_actions_to_first_levelup_integrated`,
`live_submittable_level_count_integrated`, `parity_test_green`,
`submitted_config_raised_metric_clean`, `null_delta_methodology_note`,
`random_seed`, `reproducibility_checksum`, and `preconditions_checked`.
Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: integrated_<metric>_raised_config_shipped OR complete: integration_no_clean_metric_bare_config_kept_honest_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline end-to-end re-measure (1s floor)."
- `verifier_is_oracle`: principle "MUST be false for every aggregated value claim -- the integrated levers are oracle-distinct."
- `levers_integrated`: principle "names the upstream levers (A1/A2/A3) admitted into SUBMITTED_AGENT_CONFIG -- the audit trail."
- `flagged_artifacts_excluded`: principle "names any flagged_adversarial / positive-control-failed upstream artifact NOT aggregated (the fabrication gate + FALSE_NEGATIVE_RISK compliance)."
- `world_model_trust_pass_rate_integrated`: principle "the integrated world_model_trust_pass_rate (A1's headline metric on the shipped config) -- did the 0.08-wall fix make it into the SCORED agent."
- `first_win_rate_integrated`: principle "the integrated held-out first-win-rate on the shipped config (A2's effect)."
- `live_submittable_level_count_integrated`: principle "the integrated live-submittable count (must stay > 33)."
- `parity_test_green`: principle "HARD gate -- test_arc_submitted_agent_parity.py passes; the integrated config is the single source of truth."
- `submitted_config_raised_metric_clean`: principle "True only if a CLEAN (non-flagged, control-passed) lever raised a real metric on the SCORED config; false -> honest null, bare config kept."
- `null_delta_methodology_note`: principle "present where any integrated delta == 0 -- states the equality is an honest no-value null, not a bug."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified (offline arcade, upstream artifacts present); pre-empts missing-resource fabrication."

If no clean lever raises a scored-agent metric, the artifact SHALL report
`complete: integration_no_clean_metric_bare_config_kept_honest_null`, SHALL keep
`levers_integrated=[]`, SHALL leave `SUBMITTED_AGENT_CONFIG` unchanged from the
bare submitted-control snapshot, and SHALL still report the clean A4
live-submittable count without treating it as a raised config metric.

#### SCENARIO-ARC-WMTE-4609

**Given** A1 or A2 is flagged adversarial or fails its positive control, and A3
does not offline-reproduce a new level
**When** experiment 4609 runs the consolidation gate
**Then** those upstream metrics are excluded, the submitted config records no
clean raised metric, the parity test stays green, the live-submittable count is
reported from the clean refreshed package, and the artifact emits an honest null
with a stable checksum.

### REQ-ARC-WMTE-4610: Canonical World-Model Trust Pass-Rate Metric Helper

Experiment 4610 SHALL provide a canonical helper that reads the A1 artifact
`results/experiment_4604_world_model_trust_energy.json` and the ARC solve
registry, without loading a model, and computes `world_model_trust_pass_rate`
as the explicit integer numerator divided by the explicit integer denominator.
The numerator SHALL count only world-model games where the new trust gate passed
and the planner used the world-model path. Degenerate identity/no-op selected
candidates SHALL NOT count as genuine passes, even if a source row marks
`new_trust_pass=true` and `new_planner_used=true`. The denominator SHALL be the
number of world-model game measurement rows tried.

Experiment 4610 SHALL write
`results/experiment_4610_world_model_trust_pass_rate_metric.json` with
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `world_model_trust_pass_rate`,
`trust_pass_numerator`, `trust_pass_denominator`, `coheadline_block`,
`tests_added`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`. If the computed trust-pass rate equals the matched
baseline, the artifact SHALL include `null_delta_methodology_note`.

Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: world_model_trust_pass_rate_metric_helper_shipped_tests_green."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts -- reads the A1 artifact + registry, no model load (100us floor)."
- `world_model_trust_pass_rate`: principle "the canonical co-headline metric value computed from A1 (numerator/denominator explicit, not just the float)."
- `trust_pass_numerator`: principle "the integer count of world-model games passing the new gate AND used by the planner -- explicit so the metric does not collide as a bare fraction (the TAUTOLOGY trap)."
- `trust_pass_denominator`: principle "the integer count of world-model games tried -- explicit denominator."
- `coheadline_block`: principle "world_model_trust_pass_rate reported side-by-side with reproducible_total_levels / live-submittable / first-win-rate / generic_transfer / action-efficiency -- the single canonical metric surface."
- `tests_added`: principle "the asserting tests (every test has >=1 assertion; no skips) -- the metric helper is verified, not asserted."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4610

**Given** an A1 world-model artifact with `k` non-degenerate rows whose
`new_trust_pass` and `new_planner_used` flags are both true across `n`
measurement rows
**When** the 4610 helper computes the metric
**Then** it returns `k/n`, exposes `trust_pass_numerator=k` and
`trust_pass_denominator=n`, excludes identity/no-op selected candidates from
the numerator, and emits a `null_delta_methodology_note` when the computed rate
equals the baseline.

### REQ-ARC-WMTE-4622: Canonical Offline-To-Live Transfer Ratio Metric Helper

Experiment 4622 SHALL provide a canonical helper that reads the upstream
offline verifier signal artifact, the A2 live graduated-value-head artifact,
and the ARC solve registry without loading a model. The helper SHALL compute
`offline_to_live_transfer_ratio` as the positive live lift attributable to the
graduated value head divided by the offline AUROC component. The live lift
SHALL be computed from the graduated arm versus the best matched linear/bare
baseline: positive first-win-rate lift, or positive action-efficiency lift
when first-win is equal. Both the `offline_auroc_component` and the
`live_lift_component` SHALL be reported explicitly so a high-offline/zero-live
case is visible as bridge-not-crossed rather than hidden in one scalar.

Experiment 4622 SHALL report the transfer ratio side-by-side with the other
ARC co-headline metrics in one canonical `coheadline_block`: reproducible total
levels from `ops/arc_solve_registry.yaml`, live-submittable level count,
first-win-rate, and action efficiency. When the live lift equals zero, the
artifact SHALL include `null_delta_methodology_note`.

Experiment 4622 SHALL write
`results/experiment_4622_offline_to_live_transfer_ratio_metric.json` with
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `offline_to_live_transfer_ratio`,
`offline_auroc_component`, `live_lift_component`, `coheadline_block`,
`tests_added`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`.

Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: offline_to_live_transfer_ratio_metric_helper_shipped_tests_green."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts -- reads the A1/A2 artifacts + registry, no model load (100us floor)."
- `offline_to_live_transfer_ratio`: principle "the canonical co-headline metric value (live lift attributable to the value head, with the offline AUROC reported alongside)."
- `offline_auroc_component`: principle "the offline LOO-AUROC (the verifier signal) -- explicit so a high-offline/zero-live case is visibly the bridge-not-crossed state."
- `live_lift_component`: principle "the LIVE first-win/efficiency lift the value head produces -- explicit so the metric does not hide the offline->live gap."
- `coheadline_block`: principle "offline_to_live_transfer_ratio reported side-by-side with reproducible_total_levels / live-submittable / first-win-rate / action-efficiency -- the single canonical metric surface."
- `tests_added`: principle "the asserting tests (every test has >=1 assertion; no skips) -- the metric helper is verified, not asserted."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4622

**Given** an offline verifier artifact with explicit LOO-AUROC and an A2 live
artifact with graduated, linear-baseline, and bare-control first-win/action
metrics
**When** the 4622 helper computes the transfer metric
**Then** it reports `offline_auroc_component`, `live_lift_component`,
`offline_to_live_transfer_ratio`, and a canonical `coheadline_block`; it emits a
near-zero transfer ratio plus `null_delta_methodology_note` for high-offline
AUROC and zero live lift, and emits a positive transfer ratio when the value
head produces positive live first-win or efficiency lift.

### REQ-ARC-WMTE-4623: Adversarial Verify Offline/Live Bridge Hardening

The `scripts/adversarial_verify.py` reader SHALL protect the ARC offline-to-live
bridge thesis from two narrow false reads. First, an ARC artifact whose
`honest_verdict` or headline text claims a live search win, efficiency win, or
solve by the live agent SHALL report a measured live metric field such as
`first_win_rate_*`, `actions_*`, or `live_*`. If the artifact reports only an
offline AUROC or offline discriminator metric for that live-win claim, the
reader SHALL emit a WARN flag of kind `OFFLINE_SUBSTITUTED_FOR_LIVE`. An artifact
that honestly reports an offline AUROC as an offline result, without a live-win
claim, SHALL NOT fire this guard.

Second, the reader SHALL apply a calibrated cheap learned-value substrate floor:
a verifier-scoring artifact that declares a cached-candidate learned value-head,
CNN, or linear forward-pass substrate MAY pass a sub-1s duration only when it
also carries methodology evidence: `model_specs`, `random_seed`, and
`reproducibility_checksum`. A sub-second verifier-scoring run without that
methodology SHALL still emit `DURATION_TOO_SHORT`.

Experiment 4623 SHALL write
`results/experiment_4623_adversarial_verify_hardening.json` with
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `offline_live_overclaim_guard_added`,
`cheap_value_substrate_floor_added`, `honest_offline_result_not_flagged`,
`no_methodology_fast_run_still_fires`, `tests_added`,
`research_conductor_modified`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`.

Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: adversarial_verify_hardened_offline_live_overclaim_guard_plus_cheap_value_substrate_tests_green."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts -- reads the fixtures + edits the linter, no model load (100us floor)."
- `offline_live_overclaim_guard_added`: principle "the guard that a LIVE-win claim must carry a LIVE metric, not only an offline AUROC (the .426 bridge-thesis protection)."
- `cheap_value_substrate_floor_added`: principle "the calibrated substrate floor so a methodology-bearing fast value-head scoring run is not DURATION_TOO_SHORT false-flagged (the .425 A1 regression fixture)."
- `honest_offline_result_not_flagged`: principle "HARD -- an artifact honestly reporting an offline AUROC as an offline result (no live-win claim) does NOT fire guard 1 (narrow, not a hole)."
- `no_methodology_fast_run_still_fires`: principle "HARD -- a no-methodology sub-second verifier-scoring run STILL fires DURATION_TOO_SHORT (guard 2 is narrow, the .425 A1 0.44s fabrication case stays catchable)."
- `tests_added`: principle "the asserting tests (every test >=1 assertion; no skips) -- both guards are verified."
- `research_conductor_modified`: principle "MUST be false -- this edits adversarial_verify.py (the linter), never scripts/research_conductor.py."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified (adversarial_verify.py parses, fixtures present); pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4623-OFFLINE-LIVE-OVERCLAIM

**Given** an ARC artifact claims a live first-win, action-efficiency, solve, or
live-agent search improvement
**When** `adversarial_verify.py` checks the artifact
**Then** the artifact is WARN-flagged when it reports only offline AUROC evidence
and no measured live metric field, while an honest offline-only AUROC result is
not flagged.

#### SCENARIO-ARC-WMTE-4623-CHEAP-VALUE-SUBSTRATE

**Given** a verifier-scoring artifact declares a cached-candidate learned
value-head, CNN, or linear forward pass
**When** its duration is below 1s
**Then** it avoids `DURATION_TOO_SHORT` only if `model_specs`, `random_seed`, and
`reproducibility_checksum` are present; the same fast run without methodology
still fires `DURATION_TOO_SHORT`.

### REQ-ARC-WMTE-4635: Adversarial Verify Intrinsic Reward And CNN Floor Hardening

The `scripts/adversarial_verify.py` reader SHALL protect the .427 ARC
generation thesis from two narrow false reads. First, an ARC artifact whose
`honest_verdict` or headline text claims a curiosity, exploration, or
learning-progress win SHALL carry a measured downstream delta such as
`solve_rate_delta`, `state_coverage_delta`, or `first_win_rate_delta` versus a
control. If the claim is backed only by an intrinsic-reward / curiosity-bonus
magnitude increase and no downstream delta, the reader SHALL emit a WARN flag
of kind `intrinsic-reward-without-downstream-gain`. An artifact that honestly
reports an intrinsic-bonus magnitude as a diagnostic, without a win claim,
SHALL NOT fire this guard. An explicit zero downstream delta paired with an
honest null-delta methodology note and passed control SHALL be treated as
measured downstream evidence, not as a stub-default zero.

Second, the reader SHALL apply a calibrated self-supervised CNN substrate
floor: a verifier-scoring artifact that declares a cached-frame CNN
action-effect / frame-change forward pass MAY pass a sub-1s duration only when
it also carries methodology evidence: `model_specs`, `random_seed`, and
`reproducibility_checksum`, plus a CNN or value substrate marker. A sub-second
verifier-scoring run without that methodology SHALL still emit
`DURATION_TOO_SHORT`.

Experiment 4635 SHALL write
`results/experiment_4635_adversarial_verify_hardening.json` with
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `intrinsic_reward_overclaim_guard_added`,
`cnn_substrate_floor_added`, `honest_diagnostic_not_flagged`,
`no_methodology_fast_run_still_fires`, `tests_added`,
`research_conductor_modified`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`.

Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: adversarial_verify_hardened_intrinsic_reward_guard_plus_cnn_substrate_tests_green."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts -- reads the fixtures + edits the linter, no model load (100us floor)."
- `intrinsic_reward_overclaim_guard_added`: principle "the guard that a curiosity/exploration win must carry a measured downstream metric, not only a rising intrinsic-bonus magnitude (the .427 A1 generation-thesis protection)."
- `cnn_substrate_floor_added`: principle "the calibrated substrate floor so a methodology-bearing fast CNN action-effect scoring run is not DURATION_TOO_SHORT false-flagged (the .427 A2 fixture)."
- `honest_diagnostic_not_flagged`: principle "HARD -- an artifact honestly reporting an intrinsic-bonus magnitude as a diagnostic (no win claim) does NOT fire guard 1 (narrow, not a hole)."
- `no_methodology_fast_run_still_fires`: principle "HARD -- a no-methodology sub-second CNN-scoring run STILL fires DURATION_TOO_SHORT (guard 2 is narrow, the genuine fabrication case stays catchable)."
- `tests_added`: principle "the asserting tests (every test >=1 assertion; no skips) -- both guards are verified."
- `research_conductor_modified`: principle "MUST be false -- this edits adversarial_verify.py (the linter), never scripts/research_conductor.py."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified (adversarial_verify.py parses, fixtures present); pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4635-INTRINSIC-REWARD-DOWNSTREAM

**Given** an ARC artifact claims a curiosity, exploration, or learning-progress
win
**When** `adversarial_verify.py` checks the artifact
**Then** the artifact is WARN-flagged when it reports only intrinsic-bonus
magnitude evidence and no downstream `solve_rate_delta`,
`state_coverage_delta`, or `first_win_rate_delta`; an honest diagnostic-only
bonus report is not flagged.

#### SCENARIO-ARC-WMTE-4635-SELF-SUPERVISED-CNN-SUBSTRATE

**Given** a verifier-scoring artifact declares a cached-frame CNN action-effect
or frame-change forward pass
**When** its duration is below 1s
**Then** it avoids `DURATION_TOO_SHORT` only if `model_specs`, `random_seed`,
`reproducibility_checksum`, and the CNN/value substrate marker are present; the
same fast run without methodology still fires `DURATION_TOO_SHORT`.

### REQ-ARC-WMTE-4647: Adversarial Verify Goal-Energy Ablation Hardening

The `scripts/adversarial_verify.py` reader SHALL protect the .428
energy-DRIVES-generation thesis from a narrow false read. An ARC artifact whose
`honest_verdict` or headline text claims a graded goal-energy, energy heuristic,
or energy-as-fitness generation win SHALL carry uniform-energy ablation evidence,
such as `uniform_energy_ablation_passed` or a uniform/random-energy ablation arm.
If the artifact claims the goal-energy arm raised live solve-rate or first-win
rate over the baseline but reports no uniform-energy ablation evidence, the
reader SHALL emit a WARN flag of kind `goal-energy-without-ablation`.

An artifact that honestly reports `uniform_energy_ablation_passed` or a
uniform/random-energy ablation arm SHALL NOT fire this guard, even when the
ablation failed. An artifact that reports goal-energy magnitude as a diagnostic,
without a generation-win claim, SHALL NOT fire this guard.

Experiment 4647 SHALL write
`results/experiment_4647_adversarial_verify_hardening.json` with
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `goal_energy_ablation_guard_added`,
`honest_ablation_not_flagged`, `diagnostic_not_flagged`, `tests_added`,
`research_conductor_modified`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`.

Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: adversarial_verify_hardened_goal_energy_ablation_guard_tests_green."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts -- reads the fixtures + edits the linter, no model load (100us floor)."
- `goal_energy_ablation_guard_added`: principle "the guard that an energy-driven-generation win must carry a uniform-energy ablation control, not only 'energy-on beat the baseline' (the .428 A1 generation-thesis protection)."
- `honest_ablation_not_flagged`: principle "HARD -- an artifact honestly reporting the uniform-energy ablation (uniform_energy_ablation_passed) does NOT fire the guard (narrow, not a hole)."
- `diagnostic_not_flagged`: principle "HARD -- an artifact reporting an energy magnitude as a diagnostic (no win claim) does NOT fire the guard."
- `tests_added`: principle "the asserting tests (every test >=1 assertion; no skips) -- the guard is verified."
- `research_conductor_modified`: principle "MUST be false -- this edits adversarial_verify.py (the linter), never scripts/research_conductor.py."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified (adversarial_verify.py parses, fixtures present); pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4647-GOAL-ENERGY-ABLATION

**Given** an ARC artifact claims a graded goal-energy, energy heuristic, or
energy-as-fitness live generation win
**When** `adversarial_verify.py` checks the artifact
**Then** the artifact is WARN-flagged when it reports only energy-on beating the
baseline and no uniform/random-energy ablation evidence; an artifact with
`uniform_energy_ablation_passed` or an ablation arm is not flagged.

#### SCENARIO-ARC-WMTE-4647-GOAL-ENERGY-DIAGNOSTIC

**Given** an ARC artifact reports goal-energy or energy magnitude as a diagnostic
without a generation-win claim
**When** `adversarial_verify.py` checks the artifact
**Then** it does not emit `goal-energy-without-ablation`.

### REQ-ARC-WMTE-4625: Offline-To-Live Bridge SOTA Ingestion For .427

Experiment 4625 SHALL synthesize the 2026-06-23 SOTA ingestion focused on the
.426 headline open problem: why a good offline verifier or value head regresses
the live search under distribution shift, calibration error, and compute cost.
The workflow SHALL read the Exp 4613 world-model trust ingestion artifact and
note, `docs/research-notes/arc-representation-not-the-bottleneck-2026-06-23.md`,
`research-studying.md`, and `research-references.md`. It SHALL use only the
reliable channel (`scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`,
low-concurrency WebSearch/WebFetch, and direct arXiv HTTP checks) and SHALL NOT
invoke `/deep-research`. If the Hugging Face network precondition is blocked,
the workflow SHALL report `blocked_network` rather than fabricate citations.

Experiment 4625 SHALL write
`docs/research-notes/offline-live-bridge-literature-2026-06-23.md` and
`results/experiment_4625_sota_ingestion_offline_live_bridge.json`. The artifact
SHALL include `honest_verdict`, `inference_substrate`, `methods_mapped`,
`flagged_for_next_roadmap`, `note_path`, `deep_research_not_used`,
`preconditions_checked`, `citations_verified`, `random_seed`, and
`field_principles`. The success verdict SHALL be
`success: sota_ingestion_offline_live_bridge_mapped`, the substrate SHALL be
`aggregation_from_upstream_artifacts`, and `deep_research_not_used` SHALL be
true. Each mapped method SHALL cite real arXiv IDs and SHALL include the
implementation cost over the current A1 disambiguation / A2 graduated-value-head
stack plus a concrete `fails_when` condition.

Required field principles SHALL include:

- `honest_verdict`: principle `terminal prefix; success: sota_ingestion_offline_live_bridge_mapped.`
- `inference_substrate`: principle `aggregation_from_upstream_artifacts -- literature read + synthesis, no model load (100us floor).`
- `methods_mapped`: principle `the strongest 3-5 SOTA methods with REAL arXiv IDs + per-method implement-cost-over-current-stack + fails_when -- the actionable ingestion (no citation = fabrication).`
- `flagged_for_next_roadmap`: principle `the strongest method(s) flagged as candidate .427 inputs -- closes discover->ingest->plan->experiment.`
- `note_path`: principle `docs/research-notes/offline-live-bridge-literature-2026-06-23.md -- the per-track note (the SOTA-Ingestion Cycle deliverable).`
- `deep_research_not_used`: principle `MUST be true -- /deep-research is banned in the autonomous loop; used sweep helpers + low-concurrency WebSearch/WebFetch.`
- `preconditions_checked`: principle `records network reachability verified; pre-empts fabricated citations.`

### SCENARIO-ARC-WMTE-4625: Offline-To-Live Bridge Methods Map Onto A1/A2 And Flag .427 Inputs

Given the network precondition succeeds and the .425/.426 bridge corpus is
readable
When experiment 4625 ingests focused papers for DAgger-style search-distribution
retraining, post-hoc value calibration, learned-heuristic A*/Q* search,
SLOPE-style pruning, goal-conditioned offline value, and amortized/local
heuristic compute
Then the note and JSON artifact map three to five strongest methods onto the
current A1 disambiguation / A2 graduated-value-head stack, cite real arXiv IDs
including `1011.0686`, `2604.11351`, `1706.04599`, `2102.04518`,
`2406.04935`, `2206.03023`, `2511.10264`, and `2303.09477`, flag one or more
`.427` roadmap candidates, record all network and citation preconditions
checked, and record that `/deep-research` was not used.

### REQ-ARC-WMTE-4637: Intrinsic-Motivation / Action-Effect SOTA Ingestion For .428

Experiment 4637 SHALL synthesize the 2026-06-23 SOTA ingestion focused on the
.427 headline open problem: generate better live exploration via dense online
intrinsic-motivation or learning-progress signals plus action-effect prediction
for action efficiency. The workflow SHALL read
`results/experiment_4625_sota_ingestion_offline_live_bridge.json`,
`docs/research-notes/offline-live-bridge-literature-2026-06-23.md`,
`docs/research-notes/arc-representation-not-the-bottleneck-2026-06-23.md`,
`research-studying.md`, and `research-references.md`. It SHALL use only the
reliable channel (`scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`,
low-concurrency WebSearch/WebFetch, and direct arXiv HTTP checks) and SHALL NOT
invoke `/deep-research`. If the Hugging Face network precondition is blocked,
the workflow SHALL report `blocked_network` rather than fabricate citations.

Experiment 4637 SHALL write
`docs/research-notes/intrinsic-motivation-action-effect-literature-2026-06-23.md`
and `results/experiment_4637_sota_ingestion_intrinsic_motivation.json`. The
artifact SHALL include `honest_verdict`, `inference_substrate`,
`methods_mapped`, `flagged_for_next_roadmap`, `note_path`,
`deep_research_not_used`, `preconditions_checked`, `citations_verified`,
`random_seed`, and `field_principles`. The success verdict SHALL be
`success: sota_ingestion_intrinsic_motivation_action_effect_mapped`, the
substrate SHALL be `aggregation_from_upstream_artifacts`, and
`deep_research_not_used` SHALL be true. Each mapped method SHALL cite real
arXiv IDs and SHALL include the implementation cost over the current A1
dense-curiosity / A2 action-effect stack plus a concrete `fails_when`
condition.

Required field principles SHALL include:

- `honest_verdict`: principle `terminal prefix; success: sota_ingestion_intrinsic_motivation_action_effect_mapped.`
- `inference_substrate`: principle `aggregation_from_upstream_artifacts -- literature read + synthesis, no model load (100us floor).`
- `methods_mapped`: principle `the strongest 3-5 SOTA methods with REAL arXiv IDs + per-method implement-cost-over-current-stack + fails_when -- the actionable ingestion (no citation = fabrication).`
- `flagged_for_next_roadmap`: principle `the strongest method(s) flagged as candidate .428 inputs -- closes discover->ingest->plan->experiment.`
- `note_path`: principle `docs/research-notes/intrinsic-motivation-action-effect-literature-2026-06-23.md -- the per-track note (the SOTA-Ingestion Cycle deliverable).`
- `deep_research_not_used`: principle `MUST be true -- /deep-research is banned in the autonomous loop; used sweep helpers + low-concurrency WebSearch/WebFetch.`
- `preconditions_checked`: principle `records network reachability verified; pre-empts fabricated citations.`

### SCENARIO-ARC-WMTE-4637: Dense Curiosity And Action-Effect Methods Map Onto .428 Inputs

Given the network precondition succeeds and the .426 offline/live bridge corpus
is readable
When experiment 4637 ingests focused papers for Curiosity-Critic cumulative
prediction-error-improvement rewards, Learning Progress Monitoring and
aleatoric-noise curiosity guards, ICM/RND prediction-error baselines, ARC
clickability/action-effect expansion, graph-based exploration, and executable
world-model action-effect planning
Then the note and JSON artifact map three to five strongest methods onto the
current A1 dense-curiosity / A2 action-effect stack, cite real arXiv IDs
including `2604.18701`, `2509.25438`, `2102.04399`, `1705.05363`,
`1810.12894`, `2601.10904`, `2603.24621`, `2512.24156`, and `2605.05138`,
flag one or more `.428` roadmap candidates, record all network and citation
preconditions checked, and record that `/deep-research` was not used.

### REQ-ARC-WMTE-4649: Energy-Fitness Generator SOTA Ingestion For .429

Experiment 4649 SHALL synthesize the 2026-06-23 SOTA ingestion focused on the
.428 headline open problem: turn the current goal-energy heuristic and
action-effect expansion prior into a next generator via energy-as-fitness
quality-diversity evolution, macro-action vocabulary induction, and
hierarchical subgoal search. The workflow SHALL read
`results/experiment_4637_sota_ingestion_intrinsic_motivation.json`,
`docs/research-notes/arc-generation-wall-energy-config-space-2026-06-22.md`,
`ops/verifier_gaps.md`, `research-studying.md`, and
`research-references.md`. It SHALL use only the reliable channel
(`scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`,
low-concurrency WebSearch/WebFetch, and direct arXiv HTTP checks) and SHALL NOT
invoke `/deep-research`. If the Hugging Face network precondition is blocked,
the workflow SHALL report `blocked_network` rather than fabricate citations.

Experiment 4649 SHALL write
`docs/research-notes/energy-fitness-generator-literature-2026-06-23.md` and
`results/experiment_4649_sota_ingestion_energy_fitness_generator.json`. The
artifact SHALL include `honest_verdict`, `inference_substrate`,
`methods_mapped`, `flagged_for_next_roadmap`, `note_path`,
`deep_research_not_used`, `preconditions_checked`, `citations_verified`,
`random_seed`, and `field_principles`. The success verdict SHALL be
`success: sota_ingestion_energy_fitness_generator_mapped`, the substrate SHALL
be `aggregation_from_upstream_artifacts`, and `deep_research_not_used` SHALL be
true. Each mapped method SHALL cite real arXiv IDs and SHALL include the
implementation cost over the current goal-energy / action-effect stack plus a
concrete `fails_when` condition.

Required field principles SHALL include:

- `honest_verdict`: principle `terminal prefix; success: sota_ingestion_energy_fitness_generator_mapped.`
- `inference_substrate`: principle `aggregation_from_upstream_artifacts -- literature read + synthesis, no model load (100us floor).`
- `methods_mapped`: principle `the strongest 3-5 SOTA methods with REAL arXiv IDs + per-method implement-cost-over-current-stack + fails_when -- the actionable ingestion (no citation = fabrication).`
- `flagged_for_next_roadmap`: principle `the strongest method(s) flagged as candidate .429 inputs -- closes discover->ingest->plan->experiment.`
- `note_path`: principle `docs/research-notes/energy-fitness-generator-literature-2026-06-23.md -- the per-track note (the SOTA-Ingestion Cycle deliverable).`
- `deep_research_not_used`: principle `MUST be true -- /deep-research is banned in the autonomous loop; used sweep helpers + low-concurrency WebSearch/WebFetch.`
- `preconditions_checked`: principle `records network reachability verified; pre-empts fabricated citations.`

### SCENARIO-ARC-WMTE-4649: Energy-Fitness, Macro, And Hierarchical Methods Map Onto .429 Inputs

Given the network precondition succeeds and the .427 intrinsic/action-effect
corpus plus the energy-config-space menu are readable
When experiment 4649 ingests focused papers for bidirectional evolutionary
search, quality-diversity under sparse rewards and deceptive fitness,
empowerment/eigenoption macro-action induction, hierarchical latent-world-model
planning, subgoal-guided policy heuristic search, PoE-World factored executable
models, and executable ARC world-model planning
Then the note and JSON artifact map three to five strongest methods onto the
current goal-energy / action-effect stack, cite real arXiv IDs including
`2605.28814`, `2308.05483`, `2504.01915`, `2605.27130`, `2107.07031`,
`2502.02962`, `2302.04693`, `1810.04586`, `1710.11089`, `2604.03208`,
`2506.07255`, `2504.04366`, `2505.10819`, and `2605.05138`, flag one or more
`.429` roadmap candidates, record all network and citation preconditions
checked, and record that `/deep-research` was not used.

### REQ-ARC-WMTE-4661: Generation-Guidance SOTA Ingestion For .430

Experiment 4661 SHALL synthesize the 2026-06-24 SOTA ingestion focused on the
localized `.429` generation-guidance wall after the macro-action
horizon-collapse, off-centroid click-heatmap generator, just-explore
schedule-extraction, and standalone goal-energy heuristic levers were retired
or nulled. The workflow SHALL read
`results/experiment_4649_sota_ingestion_energy_fitness_generator.json`,
`docs/research-notes/energy-fitness-generator-literature-2026-06-23.md`,
`research-studying.md`, `research-references.md`, and the three 2026-06-23
dead-lever notes for macro vocabularies, click heatmaps, and just-explore. It
SHALL also read the current A1 value-routing and A2 energy-fitness QD artifacts
before describing implementation cost over the current stack.

The workflow SHALL use only the reliable channel:
`scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`, low-concurrency
WebSearch/WebFetch, and direct arXiv HTTP checks. It SHALL NOT invoke
`/deep-research`. If `https://huggingface.co/api/models` or
`.venv/bin/python scripts/sweep_clusters.py --help` fails, the workflow SHALL
report `blocked_network` or `blocked_sweep_clusters` rather than fabricate
citations.

Experiment 4661 SHALL write
`docs/research-notes/generation-guidance-sota-ingestion-2026-06-24.md` and
`results/experiment_4661_sota_ingestion_generation_guidance.json`. The
artifact SHALL include `honest_verdict`, `inference_substrate`,
`deep_research_not_used`, `methods_mapped`, `citations_verified`,
`flagged_for_next_roadmap`, `dead_levers_not_reflagged`, `note_path`,
`preconditions_checked`, `random_seed`, and `field_principles`. The success
verdict SHALL be `success: sota_ingestion_generation_guidance_mapped`, the
substrate SHALL be `aggregation_from_upstream_artifacts`, and
`deep_research_not_used` SHALL be true. Each mapped method SHALL cite real
arXiv IDs and SHALL include the implementation cost over the current A1
value-routing, A2 energy-fitness QD, and live E3 stack plus a concrete
`fails_when` condition.

Required field principles SHALL include:

- `honest_verdict`: principle `terminal prefix; success: sota_ingestion_generation_guidance_mapped.`
- `inference_substrate`: principle `aggregation_from_upstream_artifacts -- literature read + synthesis, no model load (100us floor).`
- `deep_research_not_used`: principle `MUST be true -- /deep-research is BANNED in the autonomous loop; used sweep helpers + low-concurrency WebSearch/WebFetch.`
- `methods_mapped`: principle `the strongest 3-5 SOTA methods with REAL arXiv IDs + per-method implement-cost-over-current-stack + fails_when (no citation = fabrication).`
- `citations_verified`: principle `each cited arXiv ID with an HTTP-200 verification -- pre-empts fabricated citations.`
- `flagged_for_next_roadmap`: principle `the strongest method(s) flagged as candidate .430 inputs (flagged_for_v430) -- closes discover->ingest->plan->experiment.`
- `dead_levers_not_reflagged`: principle `names the DEAD levers (macro/click/schedule/goal-energy heuristic) confirmed NOT re-flagged -- honors the week's falsifications.`
- `note_path`: principle `the per-track research-note path (the SOTA-Ingestion Cycle deliverable).`
- `preconditions_checked`: principle `records network reachability verified; pre-empts fabricated citations.`

### SCENARIO-ARC-WMTE-4661: Surviving Guidance Methods Map Onto .430 Inputs

Given the network and sweep-helper preconditions succeed and the .428/.429
generation artifacts plus dead-lever notes are readable
When experiment 4661 ingests focused papers for hierarchical subgoal search,
factored executable world models, and distribution-shift-corrected value
routing
Then the note and JSON artifact map three to five strongest methods onto the
current A1 value-routing / A2 energy-fitness QD / live E3 stack, cite real
arXiv IDs including `2604.03208`, `2506.07255`, `2504.04366`, `2505.10819`,
`2605.05138`, `1011.0686`, `2604.11351`, `1706.04599`, `2102.04518`,
`2605.28814`, `2308.05483`, and `2504.01915`, flag one or more
`flagged_for_v430` roadmap candidates, record the macro/click/schedule/
goal-energy heuristic levers as not re-flagged, record all network and
citation preconditions checked, and record that `/deep-research` was not used.

### REQ-ARC-WMTE-4673: Structural Deepening SOTA Ingestion For .431

Experiment 4673 SHALL deepen the `.429` structural fallback after the precise
`.430` levers both failed to produce a live multi-level lift. The workflow SHALL
read `results/experiment_4661_sota_ingestion_generation_guidance.json`,
`docs/research-notes/generation-guidance-sota-ingestion-2026-06-24.md`,
`results/experiment_4664_l2_goal_predicate_induction_live.json`,
`results/experiment_4665_dagger_distribution_shift_value_routing.json`,
`research-studying.md`, and `research-references.md` before scoping the `.431`
candidate methods. The synthesis SHALL explicitly carry forward the A1 residual
`single_exemplar_goal_insufficient` / unsatisfiable L2 goal predicates and the
A2 residual `missing_verifier_gap_live_frontier_not_separated` / zero live-lift
distribution-corrected value routing.

The workflow SHALL use only the reliable channel:
`scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`, low-concurrency
WebSearch/WebFetch of the top five to eight papers, and direct arXiv HTTP
checks. It SHALL NOT invoke `/deep-research`. If
`https://huggingface.co/api/models` or
`.venv/bin/python scripts/sweep_clusters.py --help` fails, the workflow SHALL
report `blocked_network` or `blocked_sweep_clusters` rather than fabricate
citations.

Experiment 4673 SHALL write
`docs/research-notes/structural-deepening-sota-ingestion-2026-06-24.md` and
`results/experiment_4673_sota_ingestion_structural_deepening.json`. The
artifact SHALL include `honest_verdict`, `inference_substrate`,
`deep_research_not_used`, `methods_mapped`, `citations_verified`,
`flagged_for_next_roadmap`, `note_path`, `preconditions_checked`,
`random_seed`, and `field_principles`. The success verdict SHALL be
`success: sota_ingestion_structural_deepening_mapped`, the substrate SHALL be
`aggregation_from_upstream_artifacts`, and `deep_research_not_used` SHALL be
true. Each mapped method SHALL cite real arXiv IDs and SHALL include the
implementation cost over the current A1 L2-goal-induction, A2
distribution-corrected value-routing, and live E3 stack plus a concrete
`fails_when` condition and residual-scope note. No method claim is valid without
a verified arXiv ID.

Required field principles SHALL include:

- `honest_verdict`: principle `terminal prefix; success: sota_ingestion_structural_deepening_mapped.`
- `inference_substrate`: principle `aggregation_from_upstream_artifacts -- literature read + synthesis, no model load (100us floor).`
- `deep_research_not_used`: principle `MUST be true -- /deep-research is BANNED in the autonomous loop; used sweep helpers + low-concurrency WebSearch/WebFetch.`
- `methods_mapped`: principle `the strongest 3-5 SOTA methods with REAL arXiv IDs + per-method implement-cost-over-current-stack + fails_when (no citation = fabrication).`
- `citations_verified`: principle `each cited arXiv ID with an HTTP-200 verification -- pre-empts fabricated citations.`
- `flagged_for_next_roadmap`: principle `the strongest method(s) flagged as candidate .431 inputs (flagged_for_v431) -- closes discover->ingest->plan->experiment.`
- `note_path`: principle `the per-track research-note path (the SOTA-Ingestion Cycle deliverable).`
- `preconditions_checked`: principle `records network reachability verified; pre-empts fabricated citations.`

### SCENARIO-ARC-WMTE-4673: Structural Fallback Maps Onto .431 Inputs

Given the network and sweep-helper preconditions succeed and the .429 D
ingestion plus A1/A2 `.430` residual artifacts are readable
When experiment 4673 ingests focused papers for hierarchical subgoal search,
factored executable world models, and subgoal-conditioned distribution-shift
value routing
Then the note and JSON artifact map three to five strongest methods onto the
current A1 L2-goal-induction / A2 distribution-corrected value-routing / live
E3 stack, cite real arXiv IDs including `2604.03208`, `2506.07255`,
`2504.04366`, `2505.10819`, `2605.05138`, `2605.12913`, `2604.11351`, and
`1011.0686`, flag one or more `flagged_for_v431` roadmap candidates, record
all network and citation preconditions checked, record the A1/A2 residual scope,
and record that `/deep-research` was not used.

### REQ-ARC-WMTE-4685: Directed-Exploration SOTA Ingestion For .432

Experiment 4685 SHALL map the `.432` fallback after the `.431` A1 hierarchical
subgoal search and A2 PoE-World factored-planner attempts leave the wall at
L1-first-contact. The workflow SHALL read
`results/experiment_4673_sota_ingestion_structural_deepening.json`,
`docs/research-notes/structural-deepening-sota-ingestion-2026-06-24.md`,
`results/experiment_4676_hierarchical_subgoal_search_live.json`,
`results/experiment_4677_poe_world_factored_subgoal_planner.json`,
`research-studying.md`, and `research-references.md` before scoping methods.
The synthesis SHALL explicitly carry forward the A1 residual
`wall_diagnosis=l1_first_contact` / `value_head_still_not_separating` and the
A2 residual `candidate_generation_coverage_factored=0.0` /
`experts_overfit_prefix`, because `.432` targets the action-proposal
distribution that fails to make a winning L1 trajectory appear on 24 of 25
games.

The workflow SHALL use only the reliable channel:
`scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`, low-concurrency
WebSearch/WebFetch of the top five to eight papers, and direct arXiv HTTP
checks. It SHALL NOT invoke `/deep-research`. If
`https://huggingface.co/api/models` or
`.venv/bin/python scripts/sweep_clusters.py --help` fails, the workflow SHALL
report `blocked_network` or `blocked_sweep_clusters` rather than fabricate
citations.

Experiment 4685 SHALL write
`docs/research-notes/directed-exploration-sota-ingestion-2026-06-24.md` and
`results/experiment_4685_sota_ingestion_directed_exploration.json`. The
artifact SHALL include `honest_verdict`, `inference_substrate`,
`deep_research_not_used`, `methods_mapped`, `citations_verified`,
`flagged_for_next_roadmap`, `note_path`, `preconditions_checked`,
`random_seed`, and `field_principles`. The success verdict SHALL be
`success: sota_ingestion_directed_exploration_mapped`, the substrate SHALL be
`aggregation_from_upstream_artifacts`, and `deep_research_not_used` SHALL be
true. Each mapped method SHALL cite real arXiv IDs and SHALL include the
implementation cost over the current live E3 explorer, A1 subgoal search, and
A2 factored planner plus a concrete `fails_when` condition and L1-first-contact
residual-scope note. No method claim is valid without a verified arXiv ID.

Required field principles SHALL include:

- `honest_verdict`: principle `terminal prefix; success: sota_ingestion_directed_exploration_mapped.`
- `inference_substrate`: principle `aggregation_from_upstream_artifacts -- literature read + synthesis, no model load (100us floor).`
- `deep_research_not_used`: principle `MUST be true -- /deep-research is BANNED in the autonomous loop; used sweep helpers + low-concurrency WebSearch/WebFetch.`
- `methods_mapped`: principle `the strongest 3-5 SOTA methods with REAL arXiv IDs + per-method implement-cost-over-current-stack + fails_when (no citation = fabrication).`
- `citations_verified`: principle `each cited arXiv ID with an HTTP-200 verification -- pre-empts fabricated citations.`
- `flagged_for_next_roadmap`: principle `the strongest method(s) flagged as candidate .432 inputs (flagged_for_v432) -- closes discover->ingest->plan->experiment.`
- `note_path`: principle `the per-track research-note path (the SOTA-Ingestion Cycle deliverable).`
- `preconditions_checked`: principle `records network reachability verified; pre-empts fabricated citations.`

### SCENARIO-ARC-WMTE-4685: Directed Exploration Maps Onto .432 Inputs

Given the network and sweep-helper preconditions succeed and the `.431` A1/A2
residual artifacts are readable
When experiment 4685 ingests focused papers for intrinsic-motivation directed
exploration, novelty/empowerment-driven proposal, strategy-guided exploration,
and program-synthesis action-model induction
Then the note and JSON artifact map three to five strongest methods onto the
current live E3 explorer / A1 hierarchical subgoal search / A2 factored planner
stack, cite real arXiv IDs including `1712.06560`, `1810.12894`, `2002.06038`,
`2005.05960`, `2102.11137`, `2502.10077`, `2505.10819`, and `2603.02045`,
flag one or more `flagged_for_v432` roadmap candidates, record all network and
citation preconditions checked, record the A1/A2 L1-first-contact residual
scope, and record that `/deep-research` was not used.

### REQ-ARC-WMTE-4688: Controllable-Novelty Proposal Policy Live Gate

Experiment 4688 SHALL implement the `.432` build-first lever from
REQ-ARC-WMTE-4685 in the live `E3AgentPolicy` import closure. The live
`StepwiseExplorer` SHALL expose an opt-in controllable-novelty proposal policy
that reshapes each state's action-proposal order before value-ranking or
frontier selection consumes the actions. The proposal policy SHALL embed visible
frame-delta features together with action-effect features, maintain an episodic
kNN novelty table for within-episode novelty, maintain an RND-style lifelong
novelty score for cross-episode novelty, and apply an intrinsic proposal bonus
under a small family of exploration temperatures.

The controllability gate is load-bearing. With the gate enabled, novelty SHALL
score only controllable action-effect deltas caused by the agent's own action;
raw frame novelty, cosmetic animation, or uncontrolled frame changes SHALL NOT
earn controllable novelty. Experiment 4688 SHALL include a matched no-novelty
ablation and a cosmetic-novelty ablation that disables the gate and scores raw
frame novelty. Any positive solve claim SHALL be valid only when the
controllable-novelty arm reaches and offline-reproduces a new level while both
ablations fail under the same budget.

Before measurement, experiment 4688 SHALL verify that the Qwen3.5-9B-MTP GGUF
is cached, `arc_solver_kit.offline_arcade()` initializes, the live
`E3AgentPolicy` / `StepwiseExplorer` / value-learner / frame-change modules
import, and `LocalGGUFProposer` on a free non-8919 port reports a Qwen model
through `/props`. If those preconditions fail, it SHALL write a blocked artifact
rather than fabricate a run. The measurement SHALL run on at least one clean
L1-only target, prefer one of `bp35`, `re86`, `s5i5`, `g50t`, `r11l`, or
`lf52`, replay-verify any claimed generic-agent new level with
`arc_solver_kit.reproduce`, and report the target's reachable-L1 positive
control so no-new-level nulls are auditable.

The artifact at
`results/experiment_4688_controllable_novelty_proposal_policy_live.json` SHALL
include `honest_verdict`, `inference_substrate`, `verifier_is_oracle`,
`solve_provenance`, `live_path_reachable`, `controllability_gate_on`,
`generic_first_win_by_config`, `generic_agent_reached_level`,
`offline_reproduced`, `reproduced_levels`,
`no_novelty_ablation_reached_level`,
`cosmetic_novelty_ablation_reached_level`,
`residual_cause_hypothesis`, `bare_control_passed`,
`false_negative_risk_checked`, `proposer_served_model`,
`chosen_submitted_config`, `parity_test_green`, `random_seed`,
`reproducibility_checksum`, `preconditions_checked`, and `field_principles`.
For complete/null verdicts it SHALL also include `null_methodology_note`.

Required field principles SHALL include:

- `honest_verdict`: principle `terminal prefix; success: controllable_novelty_generic_agent_new_level_<game>_L<n> OR complete: controllable_novelty_no_new_level_residual_<cause>.`
- `inference_substrate`: principle `live_llm_inference -- the live E3 explorer's world-model induction (and the strategy-conditioned arm) loads + runs the Qwen3.5-9B-MTP GGUF (60s floor); declared honestly because the live agent runs a real LLM during exploration.`
- `verifier_is_oracle`: principle `MUST be false -- the controllable-novelty bonus is oracle-DISTINCT from the executable reproduction win-check.`
- `solve_provenance`: principle `live_agent_self_discovery -- a generic-agent new level via runtime controllable-novelty exploration is the REAL deliverable, NOT a hand-built GameAdapter (development_proxy) and NOT outer_loop_re.`
- `live_path_reachable`: principle `HARD gate -- the changed modules (StepwiseExplorer etc.) are in the E3AgentPolicy import closure; arc_orphan_solver_lint passes.`
- `controllability_gate_on`: principle `true -- the novelty embedding is over CONTROLLABLE action-effect deltas, not raw frame novelty (the .427 noisy-TV fix); the COSMETIC-NOVELTY ablation sets this false.`
- `generic_first_win_by_config`: principle `per-config (novelty temperature / ablation) generic first-win + multi-level rate -- the honest measurement on the standard harness (the 0.04 baseline locked in by B1).`
- `generic_agent_reached_level`: principle `the deepest level the GENERIC live agent reached via controllable-novelty exploration on the target -- the headline (a NEW level is the win).`
- `offline_reproduced`: principle `a generic new level counts only if offline-reproduced via arc_solver_kit.reproduce (ARC Solve Reproducibility); a live-only trajectory is provisional.`
- `reproduced_levels`: principle `the integer new-level count the generic agent banked offline (>=1 is the bridge crossed for solve).`
- `no_novelty_ablation_reached_level`: principle `the matched no-novelty-bonus (flat exploration) ablation's reached_level -- MUST be lower than the controllable-novelty policy for the win to be attributable to directed exploration (not flat exploration).`
- `cosmetic_novelty_ablation_reached_level`: principle `the cosmetic-novelty (controllability-gate-off, raw-frame novelty) ablation's reached_level -- MUST be lower than the controllable-novelty policy for the win to be attributable to the controllability gate (not just any novelty).`
- `residual_cause_hypothesis`: principle `if it nulls, names the residual (winning_prefix_still_not_proposed | controllability_embedding_rewards_non_winning_controllable_states | novelty_revisits_dead_states) -- the .433 target; 'none' if it crossed.`
- `null_methodology_note`: principle `present when no new level -- states the null is honest (passing ablations + reachable L1 headroom), not a measurement bug.`
- `bare_control_passed`: principle `the POSITIVE CONTROL -- the target has a reachable winning L1 trajectory offline; a no-new-level null is valid only then.`
- `false_negative_risk_checked`: principle `true with both ablations run + reachable-L1-headroom confirmed -- a 'no new level' null is valid only then.`
- `proposer_served_model`: principle `the model the proposer /props reported (MUST be Qwen3.5-9B-MTP, NOT gemma) -- the port-8919 confound guard.`
- `chosen_submitted_config`: principle `the recommended SUBMITTED_AGENT_CONFIG change (controllable-novelty proposal on, novelty weight/temperature) -- the A6 input; 'unchanged' if null.`
- `parity_test_green`: principle `HARD gate -- test_arc_submitted_agent_parity.py passes; the deployed agent == the measured agent.`
- `random_seed`: principle `determinism precondition for reproducibility.`
- `reproducibility_checksum`: principle `content-addressed hash catches silent harness/corpus drift on replay.`
- `preconditions_checked`: principle `records resources verified (Qwen cached, offline arcade, live modules importable, /props served Qwen on a free port); pre-empts missing-resource fabrication.`

### SCENARIO-ARC-WMTE-4688: Controllable Novelty Is Attributed Or Nulls Honestly

Given the Qwen/free-port, offline arcade, live import, OpenSpec, and live-path
lint preconditions pass
When experiment 4688 runs the live `StepwiseExplorer` with controllable novelty,
matched no-novelty, and cosmetic-novelty proposal configurations on a clean
L1-only target
Then the artifact reports per-config first-win/reached-level results, identifies
whether the generic agent reached a new level, counts that level only if
`arc_solver_kit.reproduce` verifies it offline, marks `verifier_is_oracle=false`
because novelty is distinct from the executable win check, and either recommends
the controllable-novelty submitted config after both ablations fail or records
one of `winning_prefix_still_not_proposed`,
`controllability_embedding_rewards_non_winning_controllable_states`, or
`novelty_revisits_dead_states` as the next residual.

### REQ-ARC-WMTE-4689: Program-Synthesis Action-Effect Proposal Filter

Experiment 4689 SHALL implement the `.432` A2 program-synthesis
action-effect proposal filter from REQ-ARC-WMTE-4685 in the live
`E3AgentPolicy` import closure. The live agent SHALL induce small per-game
action-to-effect programs from its own observed transition prefixes using the
Qwen3.5-9B-MTP local proposer on a free non-8919 port, validate every induced
program against held-out transitions that were not shown in the proposal
prefix, and reject every program that fails held-out validation. Only surviving
held-out-validated programs SHALL be allowed to prune primitive live explorer
proposals.

The filter SHALL use surviving programs to keep mechanically relevant primitive
clicks or key actions, where mechanical relevance means at least one trusted
program applies to the candidate action and predicts a visible state effect.
The filter MAY fall back to the unfiltered candidate set when no trusted program
matches any candidate, but it SHALL record that fallback as a no-prune event.
The induced programs and the proposal filter SHALL remain oracle-distinct from
the executable reproduction win-check; a proposal may enter the pool because a
program predicts an effect, but a solve only counts after live execution and
offline reproduction accepts it.

Experiment 4689 SHALL measure candidate-generation coverage against a matched
blind-proposal baseline on the same target games. Coverage is the fraction of
attempts where the known winning action or plan appears in the generated pool.
A generation claim SHALL require
`candidate_generation_coverage_filter >
candidate_generation_coverage_blind_baseline`, so the winning plan appeared in
the filter-pruned pool where blind sweeping did not. A downstream first-win
claim SHALL additionally require a positive held-out first-win-rate delta with
bootstrap CI excluding the blind baseline on at least one game. If coverage
does not rise, the artifact SHALL report an honest null and log exactly one
residual bridge gap from `heldout_transitions_too_sparse` or
`program_cannot_target_winning_action`.

Before measurement, experiment 4689 SHALL verify that the Qwen3.5-9B-MTP GGUF
is cached, `arc_solver_kit.offline_arcade()` initializes, the live
`arc_executable_world_model`, `arc_llm_reinduction`, and
`arc_competition_agent` modules import, and the proposer `/props` endpoint
reports Qwen3.5-9B-MTP rather than Gemma. If any precondition fails, it SHALL
write a blocked artifact with `preconditions_checked` populated rather than
fabricate a run. The live-path lint and submitted-agent parity test are hard
gates.

The artifact at
`results/experiment_4689_program_synthesis_action_effect_proposal_filter.json`
SHALL include `honest_verdict`, `inference_substrate`,
`verifier_is_oracle`, `solve_provenance`, `live_path_reachable`,
`candidate_generation_coverage_filter`,
`candidate_generation_coverage_blind_baseline`, `coverage_delta`,
`heldout_programs_kept`, `heldout_programs_rejected`,
`live_first_win_rate_filter`, `live_baseline_blind_proposal`,
`first_win_rate_delta`, `live_lift_ci`, `bare_control_passed`,
`false_negative_risk_checked`, `null_methodology_note`,
`chosen_submitted_config`, `proposer_served_model`, `parity_test_green`,
`offline_reproduced`, `residual_bridge_gap`, `random_seed`,
`reproducibility_checksum`, and `preconditions_checked`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success: program_synthesis_filter_coverage_up_heldout_firstwin_lift_<game> OR complete: program_synthesis_filter_no_coverage_gain_residual_logged."
- `inference_substrate`: principle "live_llm_inference -- the program-synthesis induction loads + runs the Qwen3.5-9B-MTP GGUF (60s floor)."
- `verifier_is_oracle`: principle "MUST be false -- the induced action->effect programs are oracle-DISTINCT from the executable reproduction win-check."
- `solve_provenance`: principle "live_agent_self_discovery -- the generic agent's OWN runtime program induction; NOT a hand-built adapter (development_proxy), NOT outer_loop_re."
- `live_path_reachable`: principle "HARD gate -- the changed modules are in the E3AgentPolicy import closure; arc_orphan_solver_lint passes."
- `candidate_generation_coverage_filter`: principle "does the winning action/plan APPEAR in the filter-pruned proposal pool -- the make-a-winner-appear signal (the metric that distinguishes generation from selection)."
- `candidate_generation_coverage_blind_baseline`: principle "the matched BLIND-PROPOSAL baseline coverage -- a coverage CLAIM requires the filter coverage to exceed it (the winner proposed where blind sweeping did not)."
- `coverage_delta`: principle "filter - blind coverage (positive = the winner now proposed); emitted explicitly so a null (0) is annotated."
- `heldout_programs_kept`: principle "the count of induced action->effect programs that PASSED held-out transition validation (only these prune proposals -- the experts_overfit_prefix fix)."
- `heldout_programs_rejected`: principle "the count REJECTED for failing held-out transitions -- proves held-out rejection actually ran (no prefix-overfit program survived)."
- `live_first_win_rate_filter`: principle "the held-out first-win-rate WITH the proposal filter on the SCORED agent."
- `live_baseline_blind_proposal`: principle "the matched blind-proposal baseline first-win on the SAME games (the no-regression control)."
- `first_win_rate_delta`: principle "filter - baseline first-win-rate; emitted explicitly so a null is annotated."
- `live_lift_ci`: principle "bootstrap CI on the first-win lift; a claim above baseline requires the CI to exclude it."
- `bare_control_passed`: principle "the POSITIVE CONTROL -- the matched baseline ran on a corpus with reachable L1 headroom; a no-coverage-gain null is valid only then."
- `false_negative_risk_checked`: principle "true with the matched blind baseline + reachable-headroom confirmed -- a 'no coverage gain' null is valid only then."
- `null_methodology_note`: principle "present when coverage_delta==0 -- states the equality is an honest no-value null, not a measurement bug."
- `chosen_submitted_config`: principle "the recommended SUBMITTED_AGENT_CONFIG change (proposal filter on, held-out trust threshold) -- the A6 input; 'unchanged' if null."
- `proposer_served_model`: principle "the model the proposer /props reported (MUST be Qwen3.5-9B-MTP) -- the port-8919 confound guard."
- `parity_test_green`: principle "HARD gate -- test_arc_submitted_agent_parity.py passes."
- `offline_reproduced`: principle "any newly-solved variant must offline-reproduce to count."
- `residual_bridge_gap`: principle "the .433 generation gap logged if coverage does not rise (heldout_transitions_too_sparse | program_cannot_target_winning_action)."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent harness/corpus drift on replay."
- `preconditions_checked`: principle "records resources verified (Qwen cached, offline arcade, live modules importable, /props served Qwen); pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4689-HELDOUT-REJECTION

Given observed transition prefixes, held-out transitions, and proposed small
action-effect programs
When the program-synthesis proposal filter validates the programs
Then every program that fails held-out transitions is rejected, every surviving
program is counted in `heldout_programs_kept`, every rejected program is counted
in `heldout_programs_rejected`, and only surviving programs can prune live
primitive proposals.

#### SCENARIO-ARC-WMTE-4689-PROPOSAL-PRUNING

Given held-out-validated action-effect programs and a live explorer candidate
set
When the live `StepwiseExplorer` generates primitive proposals
Then the proposal filter keeps candidates with trusted predicted effects,
prunes mechanically irrelevant candidates before controllable novelty or value
ranking, records diagnostics, and falls back to the unfiltered set only when no
candidate has a trusted predicted effect.

#### SCENARIO-ARC-WMTE-4689-COVERAGE-CONTROL

Given the proposal filter and a matched blind-proposal baseline on the same
target games
When experiment 4689 measures generated candidate pools
Then the artifact reports filter coverage, blind coverage, explicit coverage
delta, held-out kept/rejected counts, first-win delta, bootstrap CI, and an
honest null note plus residual when the winning plan does not newly appear.

### REQ-ARC-WMTE-4700: Object-Centric Perception Proposal Diagnostic

Experiment 4700 SHALL implement the `.434` perception pivot in the live
`E3AgentPolicy` import closure. The live `StepwiseExplorer` SHALL expose an
opt-in object-centric proposal policy that augments and ranks primitive
candidate actions before value-ranking or frontier selection consumes them. The
deployable representation SHALL derive connected-component object slots from
the visible frame, infer object-correspondence summaries against the previous
frame, include action-context features, and propose relational slot keypoints
such as component centroids and local object-constellation gaps without reading
the executable win predicate.

The experiment SHALL run a sensitivity/ceiling diagnostic before claiming any
deployable fix. On at least one clean L1 target, preferably one of `bp35`,
`re86`, `s5i5`, `g50t`, or `r11l`, it SHALL measure proposal coverage of the
known reachable winning L1 trajectory under three representations:
`order1`, `object_centric`, and `upper_bound_ceiling`. The `order1`
representation SHALL preserve the existing frame-only proposal ordering. The
`object_centric` representation SHALL use only deployable object/relational
frame features. The `upper_bound_ceiling` representation MAY use ground-truth
object correspondence or winning-prefix annotations as a diagnostic ceiling,
but SHALL be labeled as non-deployable and SHALL NOT count as a solve claim.
`perception_is_the_wall` SHALL be true only when the upper-bound ceiling raises
winning-prefix proposal coverage where `order1` does not.

If the diagnostic is perception-sensitive, Experiment 4700 SHALL measure the
deployable object-centric proposal policy in live generic E3 exploration against
a matched order-1 representation ablation with the same budget. Any positive
claim SHALL require the object-centric arm to reach a new level, the offline
`arc_solver_kit.reproduce` gate to reproduce that level, and the order-1
ablation to fail under the same budget. Otherwise the artifact SHALL report an
honest null with one residual cause from
`object_decomposition_correct_but_winning_action_not_proposable`,
`offpath_calibration_insufficient`, or `perception_not_the_wall_search_is`.

Before measurement, Experiment 4700 SHALL verify that the Qwen3.5-9B-MTP GGUF
is cached, `arc_solver_kit.offline_arcade()` initializes, the live
`E3AgentPolicy` / `StepwiseExplorer` / value-learner / frame-change modules
import, and `LocalGGUFProposer` on a free non-8919 port reports Qwen through
`/props`. If those preconditions fail, it SHALL write a blocked artifact with
`preconditions_checked` populated rather than fabricate a run. The live-path
lint and submitted-agent parity test are hard gates.

The artifact at
`results/experiment_4700_object_centric_perception_proposal_live.json` SHALL
include `honest_verdict`, `inference_substrate`, `verifier_is_oracle`,
`solve_provenance`, `live_path_reachable`, `perception_is_the_wall`,
`proposal_coverage_by_representation`, `generic_agent_reached_level`,
`offline_reproduced`, `reproduced_levels`,
`order1_ablation_reached_level`, `offpath_calibrated`,
`residual_cause_hypothesis`, `null_methodology_note`,
`bare_control_passed`, `false_negative_risk_checked`,
`proposer_served_model`, `chosen_submitted_config`,
`parity_test_green`, `random_seed`, `reproducibility_checksum`,
`preconditions_checked`, and `field_principles`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success: object_centric_perception_generic_agent_new_level_<game>_L<n> OR complete: object_centric_perception_no_new_level_residual_<cause>."
- `inference_substrate`: principle "live_llm_inference -- the live E3 explorer's world-model induction loads + runs the Qwen3.5-9B-MTP GGUF (60s floor); declared honestly because the live agent runs a real LLM during exploration. model_specs MUST name the GGUF."
- `verifier_is_oracle`: principle "MUST be false -- the perception representation is oracle-DISTINCT from the executable reproduction win-check."
- `solve_provenance`: principle "live_agent_self_discovery -- a generic-agent new level via runtime object-centric exploration is the REAL deliverable, NOT a hand-built GameAdapter and NOT outer_loop_re."
- `live_path_reachable`: principle "HARD gate -- the changed modules (StepwiseExplorer / arc_value_learner) are in the E3AgentPolicy import closure; arc_orphan_solver_lint passes."
- `perception_is_the_wall`: principle "the DECISIVE diagnostic result -- does an upper-bound representation raise proposal-coverage of the winning L1 trajectory where order-1 does not."
- `proposal_coverage_by_representation`: principle "proposal-coverage of the winning L1 trajectory under {order-1, deployable object-centric, upper-bound ceiling}."
- `generic_agent_reached_level`: principle "the deepest level the GENERIC live agent reached via object-centric proposal conditioning on the target."
- `offline_reproduced`: principle "a generic new level counts only if offline-reproduced via arc_solver_kit.reproduce."
- `reproduced_levels`: principle "the integer new-level count the generic agent banked offline."
- `order1_ablation_reached_level`: principle "the matched ORDER-1-REPRESENTATION ablation's reached_level."
- `offpath_calibrated`: principle "true -- the deployable representation was calibrated on the LIVE off-path search distribution, including dead-ends."
- `residual_cause_hypothesis`: principle "if it nulls, names the residual (object_decomposition_correct_but_winning_action_not_proposable | offpath_calibration_insufficient | perception_not_the_wall_search_is); 'none' if it crossed."
- `null_methodology_note`: principle "present when no new level -- states the null is honest (order-1 ablation + reachable L1 headroom + the diagnostic finding), not a measurement bug."
- `bare_control_passed`: principle "the POSITIVE CONTROL -- the target has a reachable winning L1 trajectory offline."
- `false_negative_risk_checked`: principle "true with the order-1 ablation run + reachable-L1-headroom confirmed."
- `proposer_served_model`: principle "the model the proposer /props reported (MUST be Qwen3.5-9B-MTP, NOT gemma)."
- `chosen_submitted_config`: principle "the recommended SUBMITTED_AGENT_CONFIG change (object-centric proposal conditioning on, representation params); 'unchanged' if null."
- `parity_test_green`: principle "HARD gate -- test_arc_submitted_agent_parity.py passes."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent harness/corpus drift on replay."
- `preconditions_checked`: principle "records resources verified (Qwen cached, offline arcade, live modules importable, /props served Qwen on a free port); pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4700-PROPOSAL-DIAGNOSTIC

Given a clean L1 target with a reachable offline winning trajectory
When Experiment 4700 compares order-1, deployable object-centric, and
upper-bound ceiling proposal representations
Then it reports proposal coverage for each representation, sets
`perception_is_the_wall` from the upper-bound-vs-order1 coverage lift, labels
the ceiling as non-deployable, and makes no solve claim from the ceiling arm.

#### SCENARIO-ARC-WMTE-4700-LIVE-WIRING

Given the diagnostic is perception-sensitive and live-path preconditions pass
When the live `StepwiseExplorer` runs with object-centric proposal conditioning
and a matched order-1 ablation
Then the artifact counts a new level only if live exploration reaches it,
offline reproduction verifies it, and the order-1 ablation fails; otherwise it
records an honest residual for the next milestone.

### REQ-ARC-WMTE-4706: Perception Quality LOO And Off-Path CI-Gates

Experiment 4706 SHALL ship a perception-quality measurement substrate for the
`.434` perception pivot before A1/A2 integration claims depend on it. The
module SHALL expose a helper that computes leave-one-game-out AUROC for a
representation over cached cross-game rows, a CI gate that records the
order-1 chance baseline `0.503` and the richer representation AUROC, and a
floor gate that fails when perception LOO regresses below the A1-established
floor. The LOO gate SHALL fail on at-chance fixtures and on regressions toward
chance even when in-sample discrimination is high.

The module SHALL also expose an off-path discrimination metric for live search
distribution rows. It SHALL report the representation's discrimination on
winning-path rows separately from discrimination on the LIVE off-path/frontier
distribution that `StepwiseExplorer.search_distribution_samples()` exposes,
including dead-end/off-path states, and it SHALL report the
winning-path-vs-off-path AUROC gap. A large gap or near-chance off-path AUROC
SHALL be flagged because it predicts the offline-to-live null.

The artifact at
`results/experiment_4706_perception_quality_cigate.json` SHALL include
`honest_verdict`, `inference_substrate`, `verifier_is_oracle`,
`loo_discrimination_gate_added`, `offpath_discrimination_metric_added`,
`perception_quality_floor_cigate_added`, `tests_added`, `random_seed`,
`reproducibility_checksum`, `preconditions_checked`, and `field_principles`.
The verifier SHALL remain oracle-distinct from the executable win-check and the
inference substrate SHALL be cached-corpus verifier computation without live LLM
inference.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success: perception_quality_loo_plus_offpath_cigate_shipped_tests_green."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- LOO/off-path computation over cached corpora (1s floor); no live_llm_inference."
- `verifier_is_oracle`: principle "MUST be false -- the CI-gates guard the measurement substrate, oracle-distinct from the executable win-check."
- `loo_discrimination_gate_added`: principle "the perception LOO-discrimination gate (records the order-1 chance baseline 0.503 + the richer representation's LOO-AUROC; fails on regression toward chance)."
- `offpath_discrimination_metric_added`: principle "the off-path discrimination metric (held-out discrimination on the LIVE off-path search distribution + the winning-path-vs-off-path gap; the .425-B2 bridge-gap made measurable)."
- `perception_quality_floor_cigate_added`: principle "the floor CI-gate that fails on a perception LOO regression below the A1 floor."
- `tests_added`: principle "the unit tests added for all three guards (Tests Must Run and Assert: flag the at-chance / winning-paths-only / regression fixtures, pass the honest fixtures)."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4706-LOO-DISCRIMINATION

Given cached cross-game perception rows with game ids, labels, and
representation features
When Experiment 4706 computes leave-one-game-out AUROC for the representation
Then it records the order-1 baseline `0.503`, records the richer representation
LOO-AUROC, fails at/near chance fixtures, and passes above-chance fixtures.

#### SCENARIO-ARC-WMTE-4706-OFFPATH-DISCRIMINATION

Given scored winning-path rows and scored live off-path/frontier rows from the
search distribution
When Experiment 4706 measures off-path discrimination
Then it reports winning-path AUROC, off-path AUROC, the AUROC gap, row counts,
and flags a winning-path-only representation while passing an off-path
calibrated representation.

#### SCENARIO-ARC-WMTE-4706-PERCEPTION-QUALITY-FLOOR

Given the A1-established perception LOO floor
When a candidate perception representation reports a lower LOO-AUROC
Then the perception-quality floor CI-gate fails; otherwise it passes and records
the measured LOO, floor, and source.

### REQ-ARC-WMTE-4713: Surface Present Winner With Off-Path Calibrated Ranker

Experiment 4713 SHALL attack the `.434` surfacing layer after Experiment 4700
proved that object-centric perception makes the winning L1 trajectory present
in the proposal pool. The live `StepwiseExplorer` object-centric proposal path
SHALL expose an opt-in off-path-calibrated oracle-distinct verifier ranker that
uses deployable structural candidate features and live off-path transition rows,
including dead-ends, to rerank the object-centric proposal pool before search
consumes it. The ranker SHALL remain distinct from the executable reproduction
win-check: it may learn from visible-state features, action context, and
observed off-path transition labels, but it SHALL NOT call the reproduction
gate to score candidate actions.

The experiment SHALL run on a clean L1 target, preferably one of `bp35`,
`re86`, `s5i5`, `g50t`, or `r11l`, and SHALL reconfirm that the object-centric
pool contains the winning candidate on the chosen target. It SHALL report the
winning candidate's pre-surfacing rank, precision-at-k without surfacing, and
precision-at-k after applying the off-path-calibrated ranker. The matched
no-surfacing ablation SHALL use the same object-centric pool and budget with
the `.433` explorer value head behavior. A positive surfacing claim SHALL
require precision-at-k lift, generic live E3 reaching a new level, offline
reproduction through `arc_solver_kit.reproduce`, and a lower reached level from
the no-surfacing ablation.

Before measurement, Experiment 4713 SHALL verify the persisted A1
`object_centric_representation_builder_operator` is importable, the
Qwen3.5-9B-MTP GGUF is cached, `arc_solver_kit.offline_arcade()` initializes,
and a free non-8919 `LocalGGUFProposer` port reports Qwen through `/props`.
If any precondition fails, it SHALL write a blocked artifact with
`preconditions_checked` populated. If the present winner is not separable from
same-depth distractors, the artifact SHALL record the residual cause and append
a Missing-Verifier gap entry to `ops/verifier_gaps.md`.

The artifact at
`results/experiment_4713_surface_present_winner_verifier_ranker.json` SHALL
include `honest_verdict`, `inference_substrate`, `verifier_is_oracle`,
`solve_provenance`, `live_path_reachable`, `winner_present_coverage`,
`winner_rank_pre_surfacing`, `precision_at_k_with_surfacing`,
`precision_at_k_no_surfacing`, `generic_agent_reached_level`,
`no_surfacing_ablation_reached_level`, `offline_reproduced`,
`reproduced_levels`, `offpath_calibrated`, `bare_control_passed`,
`false_negative_risk_checked`, `null_methodology_note`,
`missing_verifier_gap_logged`, `chosen_submitted_config`,
`proposer_served_model`, `parity_test_green`, `random_seed`,
`reproducibility_checksum`, and `preconditions_checked`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success: surfaced_present_winner_generic_agent_new_level_<game>_L<n> OR complete: surface_present_winner_no_new_level_residual_<cause>."
- `inference_substrate`: principle "live_llm_inference -- the live E3 explorer's world-model induction loads + runs the Qwen3.5-9B-MTP GGUF (60s floor); model_specs MUST name the GGUF."
- `verifier_is_oracle`: principle "MUST be false -- the ranker is a learned/energy verifier oracle-DISTINCT from the executable reproduction win-check (gate-eligible per the Circularity discipline)."
- `solve_provenance`: principle "live_agent_self_discovery -- the generic agent's OWN runtime exploration; NOT a hand-built adapter (development_proxy), NOT outer_loop_re."
- `live_path_reachable`: principle "HARD gate -- the changed StepwiseExplorer ranking path is in the E3AgentPolicy import closure; arc_orphan_solver_lint passes."
- `winner_present_coverage`: principle "the object-centric proposal-coverage of the winning trajectory on the target (the .433 A1 coverage-1.0 claim re-confirmed on THIS game -- a winner must be present to surface)."
- `winner_rank_pre_surfacing`: principle "the winning candidate's rank in the pool BEFORE surfacing (the .433 baseline was rank 59,161) -- the buried-winner the ranker must lift."
- `precision_at_k_with_surfacing`: principle "precision-at-k (e.g. top-8) of the present winner WITH the off-path-calibrated ranker -- the make-the-present-winner-actionable signal."
- `precision_at_k_no_surfacing`: principle "the matched NO-SURFACING ablation (the .433 explorer value head, same pool) -- a surfacing claim requires precision-at-k to exceed it."
- `generic_agent_reached_level`: principle "the deepest level the GENERIC live agent reached via surfacing -- the downstream headline (a NEW level is the bridge crossed)."
- `no_surfacing_ablation_reached_level`: principle "the matched NO-SURFACING ablation reached_level -- MUST be lower for the win to be attributable to the surfacing, not the budget."
- `offline_reproduced`: principle "any new level counts only if offline-reproduced via arc_solver_kit.reproduce; a live-only trajectory is provisional."
- `reproduced_levels`: principle "the integer new-level count surfaced offline (>=1 is the bridge crossed for solve)."
- `offpath_calibrated`: principle "true -- the ranker was calibrated on the LIVE off-path search distribution (incl. dead-ends), the .425-B2 bridge fix; an on-winning-paths-only fit is the known live-null trap."
- `bare_control_passed`: principle "the POSITIVE CONTROL -- coverage-1.0 holds on the target (a winner is present) + reachable L1 headroom; a no-new-level null is valid only then."
- `false_negative_risk_checked`: principle "true with the no-surfacing ablation + winner-present confirmed -- a 'no new level' null is valid only then."
- `null_methodology_note`: principle "present when no new level; states the null is honest (winner present + ablation run + the precision-at-k delta), not a measurement bug."
- `missing_verifier_gap_logged`: principle "if the present winner is not separable, the gap (the discriminator a new verifier would need) is appended to ops/verifier_gaps.md per the Missing-Verifier Gap Logging discipline."
- `chosen_submitted_config`: principle "the recommended SUBMITTED_AGENT_CONFIG change (surfacing ranker on, params) -- the A7 input; 'unchanged' if null."
- `proposer_served_model`: principle "the model the proposer /props reported (MUST be Qwen3.5-9B-MTP) -- the port-8919 confound guard."
- `parity_test_green`: principle "HARD gate -- test_arc_submitted_agent_parity.py passes."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent harness/corpus drift on replay."
- `preconditions_checked`: principle "records resources verified (A1 operator importable, Qwen cached, offline arcade, /props served Qwen); pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4713-PRECISION-AT-K

Given a target whose object-centric proposal pool contains every winning L1
candidate
When Experiment 4713 applies the off-path-calibrated ranker to the same pool
Then it reports the pre-surfacing ranks, no-surfacing precision-at-k, and
surfacing precision-at-k, and a surfacing win requires the latter to improve.

#### SCENARIO-ARC-WMTE-4713-LIVE-ABLATION

Given Qwen, offline arcade, live-path lint, and parity preconditions pass
When live generic E3 runs with surfacing enabled and a matched no-surfacing
ablation over the same object-centric pool and budget
Then a new level counts only when the surfaced arm reaches and offline
reproduces it while the ablation reaches a lower level; otherwise the artifact
emits an honest residual and logs a missing-verifier gap when appropriate.

### REQ-ARC-WMTE-4725: Silent Representation No-Op Audit For Generation/Exploration Nulls

Experiment 4725 SHALL audit the `.428` through `.434` ARC generation and
exploration-lever null artifacts before the planner treats recent negative
results as capability limits. The audit SHALL enumerate every in-scope null
artifact for experiments 4628, 4640, 4653, 4664, 4676, 4688, 4700, 4701,
4710, 4712, 4713, and 4715, read each artifact's exercise-evidence fields, and
inspect the exercised module code when the artifact indicates a suspected
representation/archive/CNN/arm degeneracy.

The audit SHALL classify each null as `trustworthy_null` when the mechanism ran
on non-degenerate data, or `silent_bug_must_reopen` when the artifact shows a
wrong-shape tensor, empty or unchanged generated pool, dead archive, dropped
dict-candidate, byte-identical arms, or no-op transform paired with flat
coverage/delta. The audit SHALL explicitly adjudicate the `.434` A4 all-arms
`0.04` online-driver signature as
`online_driver_arms_degenerate (no-op, must reopen)` or `trustworthy_null`.
The audit SHALL confirm the `arc_go_explore._frame_grid` fix by verifying a
leading-singleton `(1,H,W)` frame collapses to a 2-D `(H,W)` grid.

Experiment 4725 SHALL write
`results/experiment_4725_silent_bug_audit.json` and
`ops/arc_null_silent_bug_audit.md` with no solve claims. The artifact SHALL
include `honest_verdict`, `inference_substrate`, `nulls_audited`,
`silent_bug_nulls`, `a4_tautology_verdict`, `trustworthy_nulls`,
`reopen_recommendations`, `go_explore_fix_confirmed`, `audit_report_path`,
`verifier_is_oracle`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; complete: silent_bug_audit_<N>_nulls_<K>_must_reopen."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts -- reads upstream nulls + module code, no model load (100us floor)."
- `nulls_audited`: principle "the count of .428-.434 generation/exploration-lever nulls inspected -- the audit scope."
- `silent_bug_nulls`: principle "the list classified silent_bug_must_reopen with evidence (degenerate shape / empty pool / dead archive / dropped candidate / byte-identical arms) -- the load-bearing finding."
- `a4_tautology_verdict`: principle "the explicit adjudication of the .434 A4 all-arms-0.04 TAUTOLOGY: online_driver_arms_degenerate (no-op, must reopen) | trustworthy_null -- this GROUNDS whether .435 A1 is a valid reopen."
- `trustworthy_nulls`: principle "the list classified trustworthy_null (the mechanism genuinely ran on non-degenerate data) -- so the loop does NOT re-run a real null."
- `reopen_recommendations`: principle "the prioritized re-run list for the planner (which closed levers reopen) -- the audit's actionable output."
- `go_explore_fix_confirmed`: principle "true if the Go-Explore _frame_grid (1,64,64) fix (2026-06-25) is confirmed landed."
- `audit_report_path`: principle "ops/arc_null_silent_bug_audit.md -- the human-readable per-null table."
- `verifier_is_oracle`: principle "false -- an audit invokes no oracle."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent drift in the audited artifact set."
- `preconditions_checked`: principle "records resources verified (null artifacts present, arc_go_explore importable); pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4725-AUDIT-CLASSIFICATION

Given the `.428` through `.434` generation/exploration null artifacts are
present and `arc_go_explore` imports
When Experiment 4725 audits their exercise evidence
Then it emits one verdict per null, classifies dead archives, cloned arms,
empty generated pools, and the `.434` all-arms `0.04` online-driver tautology as
`silent_bug_must_reopen`, keeps non-degenerate exercised mechanisms as
`trustworthy_null`, and writes a prioritized reopen list without claiming a new
solve.

### REQ-ARC-WMTE-4755: Generation-Lever Silent No-Op Audit Refresh

Experiment 4755 SHALL audit the `.428` through `.433` generation-lever null
artifacts for silent representation no-ops before the planner propagates any
closed negative result. The audit SHALL read the prior
`results/experiment_4725_silent_bug_audit.json` reopen list, inspect the
checked-in generation-lever artifacts for experiments 4640, 4641, 4653, 4676,
4677, 4688, 4689, 4700, 4701, 4710, and 4713 when present, and classify each
lever from upstream artifact evidence only. It SHALL treat byte-identical arms, a scorer
or CNN that never fires, a dead archive, an empty or unchanged candidate pool,
and an array-shape or representation mismatch as silent no-op signatures. It
SHALL classify a lever as `must_reopen` only when the artifact's null is a
dead-code or silent-representation artifact, and as `genuine_null` when the
mechanism has positive exercise evidence on non-degenerate data.

Experiment 4755 SHALL verify the upstream artifacts exist and that the offline
ARC arcade import exits successfully before emitting a complete audit. Missing
upstream artifacts SHALL produce `honest_verdict=blocked_upstream_artifacts`;
an arcade import failure SHALL produce `honest_verdict=blocked_arcade_import`.
The audit SHALL not auto-fix any lever or claim any solve.

Experiment 4755 SHALL write
`results/experiment_4755_silent_bug_audit.json` with principle-annotated
top-level fields for `honest_verdict`, `inference_substrate`,
`preconditions_checked`, `levers_audited`, `silent_no_op_findings`, and
`must_reopen`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; an audit-complete report is complete_."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts; 100us floor."
- `preconditions_checked`: principle "records the upstream-artifact checks."
- `levers_audited`: principle "the list of generation-lever nulls inspected -- the audit coverage."
- `silent_no_op_findings`: principle "per-lever {lever, no_op_signature, classification} -- distinguishes dead-code artifacts from genuine nulls so the planner does not trust a dead-code null."
- `must_reopen`: principle "the levers whose null is a dead-code artifact and must be re-tested validly -- the actionable output."

#### SCENARIO-ARC-WMTE-4755-GENERATION-LEVER-AUDIT

Given the `.428` through `.433` generation-lever artifacts, the prior
Experiment 4725 audit, and an importable offline ARC arcade
When Experiment 4755 inspects each lever for byte-identical arms, non-firing
scorers/CNNs, dead archives, empty or unchanged generated pools, and
array-shape or representation mismatches
Then it emits one `silent_no_op_findings` entry per audited lever, separates
`must_reopen` from `genuine_null`, records preconditions, writes
`results/experiment_4755_silent_bug_audit.json`, and makes no code changes to
the audited levers.

### REQ-ARC-WMTE-4732: Adversarial Verify Lever Exercise-Evidence Guard

The `scripts/adversarial_verify.py` reader SHALL mechanize the Experiment 4725
silent-bug audit as a standing check named
`LEVER_EXERCISE_EVIDENCE_DEGENERATE`. For ARC generation or exploration-lever
artifacts that declare `candidate_generation_coverage`, Go-Explore/archive
diagnostics, proposal/candidate pools, online-driver arms, or an
`inference_substrate` naming a generation/exploration lever, the reader SHALL
flag exercise evidence as degenerate when an archive records zero injections or
zero cells during a non-trivial run, a candidate/proposal pool is empty or
unchanged before and after a transform, a grid tensor reports a leading
singleton shape where a 2-D grid is expected, online-driver arms are
byte-identical to more than five significant figures, or coverage/delta remains
zero with no non-degenerate mechanism evidence.

The guard SHALL default to WARN and SHALL escalate to CRITICAL when the
degenerate artifact also headlines a solve, flips a gate, is submitted, or is
already quarantined as adversarial. The guard SHALL NOT flag a genuine flat
generation/exploration null that emits non-degenerate exercise evidence such as
positive archive injections/cells, a non-empty changed candidate/proposal pool,
a correct 2-D grid shape, distinct arm distributions, positive online training
counters without scorer errors, or explicit `arms_non_degenerate=true`.

Experiment 4732 SHALL write
`results/experiment_4732_adversarial_verify_exercise_evidence_guard.json` with
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `check_added`, `dead_code_exemplars_flagged`,
`trustworthy_null_not_flagged`, `existing_suite_green`, `pinning_test_path`,
`verifier_is_oracle`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success: lever_exercise_evidence_guard_shipped_pinned."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts -- guard dev + a pytest run, no model load (100us floor)."
- `check_added`: principle "LEVER_EXERCISE_EVIDENCE_DEGENERATE -- the new guard name; mechanizes the operator's manual silent-dead-code catch."
- `dead_code_exemplars_flagged`: principle "the three dead-code nulls (Go-Explore (1,64,64); exp4710 CNN dict-candidate; .434 A4 byte-identical arms) are flagged in retrospect -- the guard works."
- `trustworthy_null_not_flagged`: principle "a genuine non-degenerate null is NOT flagged -- the false-positive carve-out works (a real null must survive)."
- `existing_suite_green`: principle "the existing adversarial_verify tests stay green -- no existing check weakened (the never-weaken-the-linter rule)."
- `pinning_test_path`: principle "tests/python/test_adversarial_verify_lever_exercise_evidence.py -- the standing pin."
- `verifier_is_oracle`: principle "false -- a linter check invokes no oracle."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent drift."
- `preconditions_checked`: principle "records resources verified (adversarial_verify importable, exemplars present); pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4732-LEVER-EVIDENCE

Given an ARC generation/exploration artifact declares a lever mechanism and its
exercise evidence is a dead archive, dropped/errored candidate pool, degenerate
grid shape, or byte-identical online-driver arms
When `adversarial_verify.py` checks the artifact
Then it emits `LEVER_EXERCISE_EVIDENCE_DEGENERATE`, catches the historical
Go-Explore `(1,64,64)`, Exp 4710 CNN dict-candidate, and `.434` A4 all-arms
`0.04` signatures, and does not flag a flat null that explicitly reports
non-degenerate exercise evidence such as `arms_non_degenerate=true`.

### REQ-ARC-WMTE-4743: Adversarial Verify Null-Delta Carveout Hardening

The `scripts/adversarial_verify.py` reader SHALL harden the `.436` ARC
generation/exploration valid-test lane without weakening existing fabrication
checks. When an ARC generation/exploration artifact reports
`arms_non_degenerate=true`, a non-empty `null_delta_methodology_note`, and
`positive_control_passed=true`, equal first-win arms at the baseline SHALL
downgrade `TAUTOLOGY` from CRITICAL to WARN rather than quarantining the
artifact. Without those markers, the equal-arm signature SHALL remain CRITICAL.

The same reader SHALL extend `LEVER_EXERCISE_EVIDENCE_DEGENERATE` to catch
declared-but-unrun active-probe mechanisms. If an ARC exploration/probe
artifact declares probe/posterior fields such as `probe_actions_taken`,
`hypothesis_posterior_built`, `posterior_entropy_reduction`, active-probe, or
hypothesis-posterior diagnostics but reports that the mechanism did not run
(`probe_actions_taken==0`, `hypothesis_posterior_built=false`, or
`posterior_entropy_reduction==0.0`), the artifact SHALL be flagged
`LEVER_EXERCISE_EVIDENCE_DEGENERATE`. A flat null with positive exercise
evidence (`probe_actions_taken>0` and `posterior_entropy_reduction>0`) SHALL
NOT be flagged by this extension.

Experiment 4743 SHALL write
`results/experiment_4743_adversarial_verify_carveout_hardening.json` with
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `tautology_carveout_added`,
`exercise_evidence_extension_added`, `a1_exemplar_downgraded_to_warn`,
`a2_exemplar_flagged`, `positive_exercise_null_not_flagged`,
`existing_suite_green`, `pinning_test_path`, `verifier_is_oracle`,
`random_seed`, `reproducibility_checksum`, and `preconditions_checked`.

#### SCENARIO-ARC-WMTE-4743-CARVEOUT-HARDENING

Given Exp 4726 reports non-degenerate online-action-learning arms with a
validated null-delta note, Exp 4727 declares an active probe that never runs,
and a synthetic flat active-probe null reports positive exercise evidence
When `adversarial_verify.py` checks those artifacts
Then Exp 4726's equal first-win arms produce WARN `TAUTOLOGY` flags and no
critical `TAUTOLOGY`, Exp 4727 emits
`LEVER_EXERCISE_EVIDENCE_DEGENERATE`, and the positive-exercise null does not
emit `LEVER_EXERCISE_EVIDENCE_DEGENERATE`.

### REQ-ARC-WMTE-4692: Persist Directed-Exploration Primitive And Transfer Null

Experiment 4692 SHALL persist the strongest reusable `.433` A1/A2
directed-exploration primitive into `arc_solver_kit` and
`ops/arc_solve_registry.yaml`, then measure cross-game transfer on at least
three cached held-out games. If Experiment 4688's controllable-novelty proposal
policy cleared its success gate, Experiment 4692 SHALL persist that
controllable-novelty operator. Else if Experiment 4689's program-synthesis
action-effect proposal filter cleared its success gate, it SHALL persist that
held-out-validated action-effect filter operator. If both are value-null, the
experiment SHALL persist the strongest characterized reusable component instead.
For the cached `.433` artifacts, that component is the controllable-novelty
embedding mechanism from Experiment 4688: it embeds controllable action-effect
deltas, rejects raw-frame/cosmetic novelty when the controllability gate is on,
and exposes an oracle-distinct intrinsic proposal score for live exploration.

The persisted primitive SHALL remain oracle-distinct from the executable
win-check. It may generate, rank, embed, induce, or filter candidate actions,
but a new level SHALL count only when the offline reproduction gate accepts it.
The transfer measurement SHALL apply the persisted operator to cached held-out
games and SHALL report, per game, candidate-coverage delta, live solve-rate
delta, first-win delta, and `offline_reproduced_new_level`. A zero-value
transfer is valid only when recorded with a residual dead-end for the next
attack.

Experiment 4692 SHALL write
`results/experiment_4692_primitive_persist_transfer.json` with
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `verifier_is_oracle`, `primitive_persisted`,
`transfer_games`, `transfer_value_per_game`, `offline_reproduced_new_level`,
`residual_dead_end`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success: primitive_persisted_transfer_<value|null>_characterized OR complete: primitive_persisted_transfer_null_characterized."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline transfer measurement over cached games (1s floor)."
- `verifier_is_oracle`: principle "MUST be false -- the persisted primitive generates/ranks/induces, oracle-distinct from the executable win-check."
- `primitive_persisted`: principle "the operator id + derived_from_artifacts + source -- the reusable scaffolding captured so the live solver reuses it, never re-derives."
- `transfer_games`: principle "the >=3 held-out games the persisted operator was applied to (the cross-game transfer measurement)."
- `transfer_value_per_game`: principle "per-game coverage/first-win/solve-rate delta + offline_reproduced_new_level -- honest transfer value, null characterized if zero."
- `offline_reproduced_new_level`: principle "true only if the transfer banked a strictly NEW offline-reproduced level (else reproducible_total_levels is unchanged -- stated honestly)."
- `residual_dead_end`: principle "the characterized transfer-null residual if value is zero (the next-attack record); per the .431 A5 pattern."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4692-PERSIST-STRONGEST-COMPONENT

**Given** Experiments 4688 and 4689 both report complete value-null verdicts
and `chosen_submitted_config=unchanged`
**When** Experiment 4692 selects the reusable primitive
**Then** it persists `controllable_novelty_embedding_operator` under the
registry gotcha id `primitive_controllable_novelty_embedding_operator`,
records the 4688/4689 source artifacts, and explains that this captures
characterized directed-exploration scaffolding rather than fabricating a banked
level.

#### SCENARIO-ARC-WMTE-4692-TRANSFER-MEASUREMENT

**Given** cached held-out games and the persisted controllable-novelty
embedding operator
**When** Experiment 4692 measures transfer
**Then** each game reports coverage, first-win, solve-rate, and offline
reproduction deltas, and a zero-value run emits
`complete: primitive_persisted_transfer_null_characterized` with a
`residual_dead_end`.

### REQ-ARC-WMTE-4697: Amortized-Exploration SOTA Ingestion For .433

Experiment 4697 SHALL map the `.433` fallback after the `.432` directed
exploration arms fail to make a transferable hidden-game first-contact prior.
The workflow SHALL read
`results/experiment_4685_sota_ingestion_directed_exploration.json`,
`docs/research-notes/directed-exploration-sota-ingestion-2026-06-24.md`,
`results/experiment_4688_controllable_novelty_proposal_policy_live.json`,
`results/experiment_4689_program_synthesis_action_effect_proposal_filter.json`,
`python/carnot/agentic/arc_go_explore.py`, `research-studying.md`, and
`research-references.md` before scoping methods. The synthesis SHALL explicitly
carry forward the `.432` residuals
`winning_prefix_still_not_proposed` and
`heldout_transitions_too_sparse`, plus the scored hidden-game transfer wall:
per-game directed exploration is re-derived from scratch and therefore can
still fail when a first scored hidden-game submission remains at 0.08.

The workflow SHALL use only the reliable channel:
`scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`, low-concurrency
WebSearch/WebFetch of the top five to eight papers, and direct arXiv HTTP
checks. It SHALL NOT invoke `/deep-research`. If
`https://huggingface.co/api/models` or
`.venv/bin/python scripts/sweep_clusters.py --help` fails, the workflow SHALL
report `blocked_network` or `blocked_sweep_clusters` rather than fabricate
citations.

Experiment 4697 SHALL write
`docs/research-notes/amortized-exploration-sota-ingestion-2026-06-24.md` and
`results/experiment_4697_sota_ingestion_amortized_exploration.json`. The
artifact SHALL include `honest_verdict`, `inference_substrate`,
`deep_research_not_used`, `methods_mapped`, `citations_verified`,
`flagged_for_next_roadmap`, `note_path`, `preconditions_checked`,
`random_seed`, and `field_principles`. The success verdict SHALL be
`success: sota_ingestion_amortized_exploration_mapped`, the substrate SHALL be
`aggregation_from_upstream_artifacts`, and `deep_research_not_used` SHALL be
true. Each mapped method SHALL cite real arXiv IDs and SHALL include the
implementation cost over the current live E3 explorer, A1 controllable-novelty
proposal policy, A2 program-synthesis action-effect filter, and existing
`arc_go_explore.py` archive, plus a concrete `fails_when` condition and a
cross-game transfer residual-scope note. No method claim is valid without a
verified arXiv ID.

Required field principles SHALL include:

- `honest_verdict`: principle `terminal prefix; success: sota_ingestion_amortized_exploration_mapped.`
- `inference_substrate`: principle `aggregation_from_upstream_artifacts -- literature read + synthesis, no model load (100us floor).`
- `deep_research_not_used`: principle `MUST be true -- /deep-research is BANNED in the autonomous loop; used sweep helpers + low-concurrency WebSearch/WebFetch.`
- `methods_mapped`: principle `the strongest 3-5 SOTA methods with REAL arXiv IDs + per-method implement-cost-over-current-stack + fails_when (no citation = fabrication).`
- `citations_verified`: principle `each cited arXiv ID with an HTTP-200 verification -- pre-empts fabricated citations.`
- `flagged_for_next_roadmap`: principle `the strongest method(s) flagged as candidate .433 inputs (flagged_for_v433) -- closes discover->ingest->plan->experiment.`
- `note_path`: principle `the per-track research-note path (the SOTA-Ingestion Cycle deliverable).`
- `preconditions_checked`: principle `records network reachability verified; pre-empts fabricated citations.`

### SCENARIO-ARC-WMTE-4697: Amortized Exploration Maps Onto .433 Inputs

Given the network and sweep-helper preconditions succeed and the `.432` A1/A2
residual artifacts are readable
When experiment 4697 ingests focused papers for amortized/meta exploration,
learned exploration priors, in-context exploration, language-agent adaptation,
and Go-Explore return-then-explore archives
Then the note and JSON artifact map three to five strongest methods onto the
current live E3 explorer / A1 controllable-novelty proposal policy / A2
program-synthesis action-effect filter / `arc_go_explore.py` stack, cite real
arXiv IDs including `1802.07245`, `1901.10995`, `2004.12919`, `2008.02790`,
`2210.14215`, `2310.09971`, `2601.19810`, and `2603.03680`, flag one or more
`flagged_for_v433` roadmap candidates, record all network and citation
preconditions checked, record the `.432` null residual plus scored hidden-game
transfer scope, and record that `/deep-research` was not used.

### REQ-ARC-WMTE-4701: Amortized First-Contact Prior Plus Go-Explore Live Wiring

Experiment 4701 SHALL implement the second `.433` candidate-generation lever in
the live `E3AgentPolicy` import closure. The live `StepwiseExplorer` SHALL accept
an opt-in amortized first-contact action prior distilled from successful and
near-miss cross-game first-contact traces. If sequence-model training is not
available within the experiment budget, the prior MAY degrade to a deterministic
frequency prior over reusable first-contact action families, but it SHALL still
derive from logged cross-game trace rows rather than game-specific IDs. The prior
SHALL rank or seed candidate actions before the per-game explorer starts from an
empty prior.

The live explorer SHALL also consume `arc_go_explore.py` as a first-class
return-then-explore archive producer. The archive SHALL store replayable prefixes
for reached cells, select under-visited/frontier cells, replay the selected prefix
from reset, and then explore onward. `arc_go_explore.py` SHALL be reachable from
the transitive import closure of `E3AgentPolicy`, and
`scripts/arc_orphan_solver_lint.py` SHALL classify `go_explore_solve()` as a
solver-like surface so the wiring remains mechanically enforced.

Experiment 4701 SHALL measure candidate-generation coverage against a matched
NO-PRIOR baseline on the same target game or games. Coverage is the fraction of
attempts where the known winning action or plan appears in the generated
pool/archive. A coverage claim SHALL require
`candidate_generation_coverage_with_prior >
candidate_generation_coverage_no_prior_baseline`, so the winning plan appears
with the amortized prior plus archive where the empty-prior explorer did not. A
downstream first-win claim SHALL additionally require a positive held-out
first-win-rate delta with bootstrap CI excluding the no-prior baseline on at
least one game. If coverage does not rise, the artifact SHALL report an honest
null and log exactly one residual bridge gap from
`cross_game_traces_encode_game_ids`, `archive_expands_dead_cells_no_goal_gradient`,
or `replay_cannot_restore_cell`.

Before measurement, Experiment 4701 SHALL verify that the Qwen3.5-9B-MTP GGUF is
cached, `arc_solver_kit.offline_arcade()` initializes, `arc_go_explore` and
`arc_competition_agent` import, and a `LocalGGUFProposer` on a free non-8919 port
reports Qwen3.5-9B-MTP through `/props`. The Qwen proposer SHALL run through the
iGPU/HIP launcher, not a 3090 CUDA override. If any precondition fails, the
experiment SHALL write a blocked artifact with `preconditions_checked` populated
rather than fabricate a run.

The artifact at
`results/experiment_4701_amortized_exploration_prior_go_explore_live.json` SHALL
include `honest_verdict`, `inference_substrate`, `model_specs`,
`verifier_is_oracle`, `solve_provenance`, `live_path_reachable`,
`go_explore_now_live_reachable`,
`candidate_generation_coverage_with_prior`,
`candidate_generation_coverage_no_prior_baseline`, `coverage_delta`,
`live_first_win_rate_with_prior`, `live_baseline_no_prior`,
`first_win_rate_delta`, `live_lift_ci`, `no_prior_ablation_failed`,
`bare_control_passed`, `false_negative_risk_checked`,
`null_methodology_note`, `chosen_submitted_config`, `proposer_served_model`,
`parity_test_green`, `offline_reproduced`, `residual_bridge_gap`,
`random_seed`, `reproducibility_checksum`, `preconditions_checked`, and
`field_principles`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success: amortized_prior_go_explore_coverage_up_heldout_firstwin_lift_<game> OR complete: amortized_prior_go_explore_no_coverage_gain_residual_logged."
- `inference_substrate`: principle "live_llm_inference -- the amortized-prior distillation / LLM-proposed first moves load + run the Qwen3.5-9B-MTP GGUF (60s floor); model_specs MUST name the GGUF."
- `verifier_is_oracle`: principle "MUST be false -- the amortized prior + archive are oracle-DISTINCT from the executable reproduction win-check."
- `solve_provenance`: principle "live_agent_self_discovery -- the generic agent's OWN runtime exploration; NOT a hand-built adapter (development_proxy), NOT outer_loop_re."
- `live_path_reachable`: principle "HARD gate -- arc_go_explore is now in the E3AgentPolicy import closure (no longer orphaned); arc_orphan_solver_lint passes."
- `go_explore_now_live_reachable`: principle "true -- arc_go_explore.py is wired into the live E3 path (was ORPHANED); the orphan-lint confirms it (the residual-effort-not-wasted fix)."
- `candidate_generation_coverage_with_prior`: principle "does the winning action/plan APPEAR in the {amortized-prior + archive} pool -- the make-a-winner-appear signal (distinguishes generation from selection)."
- `candidate_generation_coverage_no_prior_baseline`: principle "the matched NO-PRIOR baseline coverage -- a coverage CLAIM requires the prior+archive coverage to exceed it (the winner generated where the empty-prior explorer did not)."
- `coverage_delta`: principle "with-prior - no-prior coverage (positive = the winner now generated); emitted explicitly so a null (0) is annotated."
- `live_first_win_rate_with_prior`: principle "the held-out first-win-rate WITH the amortized prior + archive on the SCORED agent."
- `live_baseline_no_prior`: principle "the matched no-prior baseline first-win on the SAME games (the no-regression control + the ablation)."
- `first_win_rate_delta`: principle "with-prior - baseline first-win-rate; emitted explicitly so a null is annotated."
- `live_lift_ci`: principle "bootstrap CI on the first-win lift; a claim above baseline requires the CI to exclude it."
- `no_prior_ablation_failed`: principle "true -- the matched NO-PRIOR ablation does NOT win where the prior+archive does; proves the win is attributable to the amortized prior, not the archive coverage alone."
- `bare_control_passed`: principle "the POSITIVE CONTROL -- the matched baseline ran on a corpus with reachable L1 headroom; a no-coverage-gain null is valid only then."
- `false_negative_risk_checked`: principle "true with the no-prior ablation + reachable-headroom confirmed -- a 'no coverage gain' null is valid only then."
- `null_methodology_note`: principle "present when coverage_delta==0 -- states the equality is an honest no-value null, not a measurement bug."
- `chosen_submitted_config`: principle "the recommended SUBMITTED_AGENT_CONFIG change (amortized prior on, archive on, return-then-explore params) -- the A6 input; 'unchanged' if null."
- `proposer_served_model`: principle "the model the proposer /props reported (MUST be Qwen3.5-9B-MTP) -- the port-8919 confound guard."
- `parity_test_green`: principle "HARD gate -- test_arc_submitted_agent_parity.py passes."
- `offline_reproduced`: principle "any newly-solved variant must offline-reproduce to count."
- `residual_bridge_gap`: principle "the .434 generation gap logged if coverage does not rise (cross_game_traces_encode_game_ids | archive_expands_dead_cells_no_goal_gradient | replay_cannot_restore_cell)."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent harness/corpus drift on replay."
- `preconditions_checked`: principle "records resources verified (Qwen cached, offline arcade, live modules importable, /props served Qwen); pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4701-LIVE-WIRING

Given the Qwen/free-port, offline arcade, OpenSpec, and live import
preconditions pass
When Experiment 4701 constructs the live `E3AgentPolicy` with the amortized
first-contact prior and Go-Explore prefix archive enabled
Then `StepwiseExplorer` seeds proposal ordering from the amortized prior,
injects replayable archive prefixes before falling back to empty-prior
exploration, exposes prior/archive diagnostics, and
`scripts/arc_orphan_solver_lint.py` confirms `arc_go_explore.py` is reachable
from the live agent path.

#### SCENARIO-ARC-WMTE-4701-COVERAGE-ABLATION

Given a target with reachable L1 headroom and a matched no-prior baseline
When Experiment 4701 measures proposal/archive coverage and held-out first-win
rate
Then the artifact reports coverage with prior, coverage without prior, the
coverage delta, first-win lift with bootstrap CI, whether the no-prior ablation
failed, and either recommends the submitted config only after the prior+archive
arm wins and offline-reproduces while the no-prior arm fails, or records one of
`cross_game_traces_encode_game_ids`,
`archive_expands_dead_cells_no_goal_gradient`, or
`replay_cannot_restore_cell` as the residual.

### REQ-ARC-WMTE-4831: Amortized In-Context Exploration Prior Live Gate

Experiment 4831 SHALL rerun the amortized exploration-prior class after the
Go-Explore frame-grid fix and distinguish a live generation lift from the
Experiment 4701 dead-archive silent bug. The live `StepwiseExplorer` /
`E3AgentPolicy` proposal path SHALL accept an opt-in amortized in-context
exploration prior distilled from banked first-contact winning trajectories. The
prior SHALL condition proposal scores on the early exploration prefix observed
in the fresh game, SHALL use reusable action-family/context signatures rather
than game IDs or raw game labels, and SHALL remain oracle-distinct from the
executable reproduction win-check.

Experiment 4831 SHALL run an archive-alive check before interpreting any solve
metric. The Go-Explore archive SHALL report `observations>0`, `stored_cells>0`,
and `prefixes_injected>0`; otherwise the experiment SHALL emit
`blocked_dead_go_explore_archive` and no solve claim. A valid measurement SHALL
also prove that the prior materially changes proposal order relative to the
matched lambda-zero/no-prior ablation, so a flat first-win delta cannot be a
silent proposal no-op.

Experiment 4831 SHALL measure held-out generic first-win with the prior against
a matched no-prior ablation on the same games, seeds, and budget. A success
claim SHALL require `first_win_rate_with_prior` above the locked 0.04 generic
baseline and a positive with-prior minus no-prior bootstrap CI95 excluding zero.
The imitation control SHALL separately report games not present in the
distillation set, and the lift SHALL hold on those held-out games to count as
learning to explore rather than copying a demonstrator or memorizing game IDs.
If the archive is alive and the prior is exercised but the CI includes zero or
the with-prior first-win does not rise above 0.04, the artifact SHALL emit the
genuine-null verdict
`complete_amortized_prior_no_first_win_lift_l1_wall_survives`.

The artifact at
`results/experiment_4831_amortized_incontext_exploration_prior_live.json` SHALL
include `honest_verdict`, `go_explore_archive_alive`,
`prior_changed_proposals`, `first_win_rate_with_prior`,
`first_win_rate_no_prior_ablation`, `first_win_delta_ci95`,
`imitation_control_heldout_games`, `live_path_reachable`, `solve_provenance`,
`inference_substrate`, `preconditions_checked`, `random_seed`,
`reproducibility_checksum`, and `field_principles`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; a lift is success_amortized_prior_raises_first_win_above_baseline, a genuine null is complete_amortized_prior_no_first_win_lift_l1_wall_survives, a dead archive is blocked_dead_go_explore_archive."
- `go_explore_archive_alive`: principle "observations>0 AND stored_cells>0 AND prefixes_injected>0 -- the exp4701 silent-bug guard; a 0 means a non-test."
- `prior_changed_proposals`: principle "the prior must materially alter proposals vs no-prior (the S2/S3 no-op lesson) -- else the first-win delta is a silent no-op."
- `first_win_rate_with_prior`: principle "held-out generic first-win WITH the prior -- must rise above the 0.04 baseline."
- `first_win_rate_no_prior_ablation`: principle "the matched lambda=0 control."
- `first_win_delta_ci95`: principle "must EXCLUDE 0 for a PASS -- a genuine lift, not noise."
- `imitation_control_heldout_games`: principle "the lift must hold on games NOT in the distillation set -- exploration, not memorization (the exp4697 imitation trap)."
- `live_path_reachable`: principle "the prior must be in the E3AgentPolicy proposal import closure (arc_orphan_solver_lint)."
- `solve_provenance`: principle "live_agent_self_discovery -- a first-win lift is the agent solving via its OWN (prior-biased) exploration."
- `inference_substrate`: principle "live_llm_inference -- the live E3 agent runs; 60s floor."
- `preconditions_checked`: principle "records arcade/go_explore checks so a missing-resource run emits blocked_, never a fabricated first-win."
- `random_seed`: principle "determinism for the distillation + the with/without runs + bootstrap."
- `reproducibility_checksum`: principle "content hash of (distilled traces, prior, games, seeds) so a replication catches drift."

#### SCENARIO-ARC-WMTE-4831-IN-CONTEXT-PRIOR

Given banked first-contact winning trajectories and a fresh game's early
exploration prefix
When Experiment 4831 distills the amortized in-context prior and ranks live
proposal candidates
Then the scores depend on reusable prefix/action-family context rather than game
IDs, the ranked proposal order differs from the no-prior order when matching
context evidence exists, and diagnostics expose the exercised context buckets.

#### SCENARIO-ARC-WMTE-4831-HELDOUT-GATE

Given the offline arcade and `arc_go_explore` preconditions pass
When Experiment 4831 runs the archive-alive check, no-prior ablation, prior arm,
and held-out imitation control
Then the artifact blocks on a dead archive, succeeds only when the positive
first-win delta CI95 excludes zero and the lift holds on games outside the
distillation set, or retires the amortized-prior generation class with a genuine
null after positive archive and proposal-change evidence.

### REQ-ARC-WMTE-4705: Submitted A1/A2 Integration Gate For Object-Centric And Amortized Exploration

Experiment 4705 SHALL read the object-centric proposal artifact
`results/experiment_4700_object_centric_perception_proposal_live.json` and the
amortized-prior plus Go-Explore artifact
`results/experiment_4701_amortized_exploration_prior_go_explore_live.json`.
The gate SHALL audit whether the A1 object-centric submitted config
(`object_centric_proposal_enabled`, `object_centric_proposal_mode`, and
representation params) or the A2 amortized-prior/archive submitted config
(`amortized_first_contact_prior_enabled`, `amortized_first_contact_prior_mode`,
`go_explore_archive_enabled`, and `go_explore_archive_mode`) cleared its
upstream success gate and is actually folded into `SUBMITTED_AGENT_CONFIG`.
If both upstream artifacts are null or explicitly `unchanged`, the gate SHALL
record `config_integrated` as unchanged with the reason, SHALL set
`config_changed=false`, and SHALL emit both `null_delta_methodology_note` and
`positive_control_passed` so identical pre/post measurements are marked as by
construction rather than presented as a finding.

Experiment 4705 SHALL re-measure the integrated held-out first-win rate and
multi-level deepen rate on the scored `E3AgentPolicy` by invoking the
Experiment 4605 held-out lane with deepening enabled. The artifact at
`results/experiment_4705_integration_gate.json` SHALL include
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `verifier_is_oracle`, `config_integrated`,
`config_changed`, `null_delta_methodology_note`, `positive_control_passed`,
`first_win_rate_integrated`, `multi_level_deepen_rate_integrated`,
`parity_test_green`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`. The artifact SHALL also include the A1/A2 audit,
pre-integration comparison metrics, no-regression status, source artifact
paths/checksums, the scored-lane measurement, the parity-test result,
`submitted_to_leaderboard=false`, and spec refs. The verifier SHALL remain
oracle-distinct from the executable win-check.

Required field principles are:

- `honest_verdict`: principle "terminal prefix; success: integrated_<config>_shipped_parity_green OR complete: integration_unchanged_both_levers_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline end-to-end re-measure on the held-out lane (1s floor)."
- `verifier_is_oracle`: principle "MUST be false -- the integrated object-centric/amortized-prior signals are oracle-distinct from the executable win-check."
- `config_integrated`: principle "the A1/A2 config folded into SUBMITTED_AGENT_CONFIG (or 'unchanged' with the reason) -- the single-source-of-truth update."
- `config_changed`: principle "bool -- true only if a lever cleared its gate; when false, the pre/post numbers are identical BY CONSTRUCTION (not a finding)."
- `null_delta_methodology_note`: principle "present when config_changed=false (pre==post) -- why the no-change is honest; the marker the TAUTOLOGY carve-out reads (the recurring A6 flag fix)."
- `positive_control_passed`: principle "bool(parity_test_green AND no_regression) -- GATES the null-delta exemption so an unvalidated no-change is NOT excused."
- `first_win_rate_integrated`: principle "the integrated SCORED-agent held-out first-win-rate (no regression vs pre-integration) -- the retargeted scored-lane metric."
- `multi_level_deepen_rate_integrated`: principle "the integrated SCORED-agent multi-level deepen-rate (the deeper scored lever) -- measured on the held-out lane."
- `parity_test_green`: principle "HARD gate -- test_arc_submitted_agent_parity.py passes; the deployed agent == the measured agent."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4705

**Given** `SUBMITTED_AGENT_CONFIG` imports with `E3AgentPolicy`, the 4700/4701
A1/A2 artifacts are readable, and the 4605 scored held-out lane is available
**When** Experiment 4705 audits the upstream chosen submitted configs, measures
the scored held-out lane with deepening enabled, and runs the submitted-agent
parity test
**Then** the result artifact reports either the exact A1/A2 config folded into
the single source of truth or an unchanged reason when both levers are null,
reports `config_changed` truthfully, reports held-out first-win rate and
multi-level deepen rate, reports no regression versus the pre-integration
config, sets `verifier_is_oracle=false`, requires `parity_test_green=true`, and
emits `null_delta_methodology_note` plus `positive_control_passed` whenever the
pre/post metrics are identical because no config changed.

### REQ-ARC-WMTE-4704: Persist .434 A1/A2 Primitive And Transfer Characterization

Experiment 4704 SHALL persist the strongest reusable `.434` A1/A2 primitive
into `arc_solver_kit` and `ops/arc_solve_registry.yaml`, then measure
cross-game transfer on at least three cached held-out games. If Experiment 4700
cleared the object-centric perception gate, the persisted primitive SHALL be the
A1 object-centric perception representation operator. Else if Experiment 4701
cleared the amortized prior plus Go-Explore gate, the persisted primitive SHALL
be the A2 amortized-prior plus return-then-explore archive operator. If both are
value-null, the experiment SHALL persist the strongest characterized reusable
component. For the cached `.434` artifacts, that fallback component is the A1
object-centric representation builder: it derives connected-component object
slots, relational gap keypoints, and proposal-side coverage diagnostics without
reading the executable win predicate.

The persisted primitive SHALL remain oracle-distinct from the executable
win-check. It may generate, perceive, induce, or rank candidate actions, but a
new level SHALL count only when the offline reproduction gate accepts it. The
transfer measurement SHALL apply the persisted operator to at least three cached
held-out games and report, per game, candidate-coverage delta, first-win-rate
delta, solve-rate delta, and `offline_reproduced_new_level`. A zero-value
transfer is valid only when the artifact records a residual dead-end for the
next attack.

Experiment 4704 SHALL write
`results/experiment_4704_primitive_persist_transfer.json` with
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `verifier_is_oracle`, `primitive_persisted`,
`transfer_games`, `transfer_value_per_game`, `offline_reproduced_new_level`,
`residual_dead_end`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success: primitive_persisted_transfer_<value|null>_characterized OR complete: primitive_persisted_transfer_null_characterized."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline transfer measurement over cached games (1s floor)."
- `verifier_is_oracle`: principle "MUST be false -- the persisted primitive generates/perceives/induces, oracle-distinct from the executable win-check."
- `primitive_persisted`: principle "the operator id + derived_from_artifacts + source -- the reusable scaffolding captured so the live solver reuses it, never re-derives."
- `transfer_games`: principle "the >=3 held-out games the persisted operator was applied to (the cross-game transfer measurement)."
- `transfer_value_per_game`: principle "per-game coverage/first-win/solve-rate delta + offline_reproduced_new_level -- honest transfer value, null characterized if zero."
- `offline_reproduced_new_level`: principle "true only if the transfer banked a strictly NEW offline-reproduced level (else reproducible_total_levels is unchanged -- stated honestly)."
- `residual_dead_end`: principle "the characterized transfer-null residual if value is zero (the next-attack record); per the .432 A5 pattern."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4704-PERSIST-STRONGEST-COMPONENT

**Given** Experiments 4700 and 4701 both report complete value-null verdicts
and `chosen_submitted_config=unchanged`
**When** Experiment 4704 selects the reusable primitive
**Then** it persists `object_centric_representation_builder_operator` under the
registry gotcha id `primitive_object_centric_representation_builder_operator`,
records the 4700/4701 source artifacts, and explains that this captures
characterized object-centric perception scaffolding rather than fabricating a
banked level.

#### SCENARIO-ARC-WMTE-4704-TRANSFER-MEASUREMENT

**Given** cached held-out games and the persisted object-centric representation
builder
**When** Experiment 4704 measures transfer
**Then** each game reports coverage, first-win, solve-rate, and offline
reproduction deltas, and a zero-value run emits
`complete: primitive_persisted_transfer_null_characterized` with a
`residual_dead_end`.

### REQ-ARC-WMTE-4712: Perception-Grounded Structural Alignment L2 Goal

Experiment 4712 SHALL use the persisted A1 object-centric perception substrate
to derive an oracle-distinct structural L2 goal for `lp85` marker-pair shape
alignment. The detector SHALL operate from a single live frame, identify
moveable corner-marker pieces and same-color goal sprites as detected objects,
and expose `is_level_complete(grid)` as "every detected piece is aligned to its
goal sprite" without requiring an L2-winning exemplar grid or the environment's
own level-up checker.

The live `E3AgentPolicy` level-up reinduction path SHALL pass the structural
goal candidate into `execute_bounded_llm_reinduction` when the current frame
contains marker-pair alignment objects. The `.430` goal-satisfiability check
SHALL evaluate the structural predicate over reachable model states before
planning, and `plan_in_model` SHALL only return a plan when that plan reaches
the structural goal. A successful bank requires `lp85` to reach L2 through the
generic live agent path and reproduce through `arc_solver_kit.reproduce`;
otherwise the artifact SHALL report the measured residual cause without
fabricating a solve.

The detector SHALL report only one matched goal sprite per detected marker
piece. Distractor solid 2x2 sprites in the same frame SHALL NOT inflate
`goal_count`; for a valid structural candidate `goal_count <= piece_count`,
and a complete frame requires `aligned_piece_count == piece_count`.
If the detector-fixed live run still does not bank L2, the follow-on artifact
SHALL run a multi-exemplar fallback that banks at least two independently
replayed L1-completion frames and reports whether the structural candidate fits
all of them without using the environment level-up oracle.

Experiment 4712 SHALL write
`results/experiment_4712_perception_grounded_l2_goal_lp85.json` with
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `goal_predicate_satisfiable`, `goal_expression`,
`l2_plan_reaches_goal`, `reproduced_levels`, `offline_reproduced`,
`solve_provenance`, `verifier_is_oracle`, `live_path_reachable`,
`registry_precheck_generic_l2`, `parity_test_green`,
`residual_cause_hypothesis`, `proposer_served_model`, `random_seed`,
`reproducibility_checksum`, and `preconditions_checked`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success: perception_grounded_l2_goal_lp85_L2_offline_reproduced OR complete: l2_perception_goal_no_deepening_residual_<cause>."
- `inference_substrate`: principle "live_llm_inference -- the reinduction path loads + runs the Qwen3.5-9B-MTP GGUF (60s floor); model_specs MUST name the GGUF."
- `goal_predicate_satisfiable`: principle "the .430 gate checks DYNAMICS only; this records the L2 alignment goal is True on >=1 reachable grid -- the verification exemplar-replay never produced."
- `goal_expression`: principle "MUST be a STRUCTURAL predicate over DETECTED objects (e.g. structural_piece_sprite_alignment_over_detected_objects), not a per-game hardcode or a flat exemplar grid."
- `l2_plan_reaches_goal`: principle "plan_len=0 / no_reachable_plan was the measured failure; True means the plan reaches the satisfiable goal."
- `reproduced_levels`: principle "the integer level reached; >=2 is the lp85 L2 bank (only reproduced levels count)."
- `offline_reproduced`: principle "a reproduced L2 via arc_solver_kit.reproduce is the fix working; a live-only trajectory is provisional."
- `solve_provenance`: principle "live_agent_self_discovery -- a generic-agent L2 via the perception-grounded goal is self-discovery; an adapter L2 is a dev proxy that does NOT prove the live fix."
- `verifier_is_oracle`: principle "MUST be false -- the alignment predicate is oracle-distinct from the env's own level-up check."
- `live_path_reachable`: principle "HARD gate -- the changed modules are in the E3AgentPolicy import closure; arc_orphan_solver_lint passes."
- `registry_precheck_generic_l2`: principle "confirms the generic live path does not ALREADY self-discover lp85 L2 (vs the adaptered registry row) -- a duplicate of an already-banked GENERIC level is a CRITICAL adversarial flag."
- `parity_test_green`: principle "HARD gate -- test_arc_submitted_agent_parity.py passes; the deployed agent == the measured agent."
- `residual_cause_hypothesis`: principle "if it nulls, names the residual (object_detector_cannot_resolve_pieces_sprites | alignment_under_determined | no_reachable_plan) -- the .435 target; 'none' if it banked."
- `null_methodology_note`: principle "present when no L2; states the null is honest (lp85 L1 reachable + object detector ran + goal satisfiability checked), not a measurement bug."
- `proposer_served_model`: principle "the model the proposer /props reported (MUST be Qwen3.5-9B-MTP, NOT gemma) -- the port-8919 confound guard."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent harness/corpus drift on replay."
- `preconditions_checked`: principle "records resources verified (A1 operator importable, Qwen cached, offline arcade + lp85 env, /props served Qwen); pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4712-STRUCTURAL-ALIGNMENT-GOAL

**Given** a live frame containing same-color corner-marker pieces and same-color
solid goal sprites
**When** the A1 object-centric perception substrate builds marker-pair objects
**Then** the structural predicate returns true exactly when every detected
corner-marker piece has a goal sprite in its interior alignment position.
The exposed diagnostics SHALL use one-to-one piece/goal pairing so a real
`lp85` post-L1 frame with many distractor 2x2 tiles does not over-segment into
more goal sprites than detected pieces.

#### SCENARIO-ARC-WMTE-4712-LIVE-REINDUCTION-WIRING

**Given** `E3AgentPolicy` enters `level_up_reinduction` on `lp85`
**When** the current root grid contains a detected structural alignment goal
**Then** `execute_bounded_llm_reinduction` evaluates that structural goal in
the goal-satisfiability check and records the goal expression and detector
diagnostics in the induction attempt.

### REQ-ARC-WMTE-4653: Energy-Fitness QD Generator Live Injection

Experiment 4653 SHALL implement the `.429` energy-as-fitness
quality-diversity generator as a live-path-reachable module under
`python/carnot/agentic`, imported by `arc_graph_explore` and reachable from
`E3AgentPolicy`. The generator SHALL maintain a MAP-Elites-style archive over
multi-action sequences, seed it from already available live search candidates or
frontier fragments, evolve sequences with insert/delete/swap/splice mutation and
shared-state crossover, and score elites with a CPU-only oracle-distinct fitness
made from goal-energy delta, action-effect cell-recall/effect evidence, and
first-win action efficiency. The behavior descriptor SHALL be derived from
visible action-effect predictions so the archive keeps diverse niches rather
than collapsing to one near-miss.

The live integration SHALL inject generated multi-action sequences into the
SCORED `E3AgentPolicy` / `StepwiseExplorer` search pool additively: primitive
actions remain available, correctness is not regressed when QD is disabled, and
QD induction cost is charged to the QD arm. The implementation SHALL include a
matched search-only control and a random-mutation / no-energy-fitness QD
ablation. It SHALL NOT modify `scripts/research_conductor.py`, SHALL NOT plan
over retired composite macro vocabularies, SHALL keep
`verifier_is_oracle=false`, and SHALL pass `scripts/arc_orphan_solver_lint.py`.

Experiment 4653 SHALL write
`results/experiment_4653_energy_fitness_qd_generation_live.json`. The artifact
SHALL measure energy-fitness QD against search-only and random-mutation controls
on cell-recall-reachable hard-game evidence, report `winner_generated`,
`winner_generated_count`, live solve-rate, depth-of-live-solve,
actions-to-win, first-win deltas, and paired bootstrap CI. Any newly generated
winner SHALL count only when offline reproduced by `arc_solver_kit.reproduce` or
an equivalent no-quota replay gate. If QD produces no new winner, the artifact
SHALL report the honest P0.1-shadow null and the residual Missing-Verifier /
bridge gap.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success: energy_fitness_qd_winner_generated_<n> OR complete: energy_fitness_qd_no_winner_generated_honest_null_gap_sharpened."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline-arcade live-search measurement over cached variants (1s floor); the QD scorer (goal-energy + action-effect CNN forward-pass) is CPU, declared so a fast pass is not DURATION_TOO_SHORT false-flagged; no live_llm_inference."
- `verifier_is_oracle`: principle "MUST be false -- goal-energy + action-effect predict the win from VISIBLE state, oracle-DISTINCT from running the executable win-check."
- `solve_provenance`: principle "live_agent_self_discovery -- the QD generator improves the SCORED live agent's OWN candidate generation (arc_graph_explore/E3AgentPolicy); NOT a parallel solver, NOT outer_loop_re."
- `live_path_reachable`: principle "HARD gate -- the QD module is imported by arc_graph_explore AND reachable from E3AgentPolicy; arc_orphan_solver_lint passes (NOT orphaned)."
- `winner_generated`: principle "the HEADLINE primary gate (structural, n-independent) -- the QD-generated winning sequence appears in the pool where best-first did NOT reach it at equal budget; offline-reproduced."
- `winner_generated_count`: principle "the integer count of new winners the QD generator put in the pool that the search-only baseline missed (>=1 is the win)."
- `live_solve_rate_qd`: principle "LIVE multi-level solve-rate WITH the energy-fitness QD generator on the SCORED agent."
- `live_solve_rate_search_baseline`: principle "the matched search-only baseline solve-rate on the SAME games (the no-regression control)."
- `solve_rate_delta`: principle "qd - search_baseline (positive = QD generation crossed the bridge), emitted explicitly so a null (0) is annotated."
- `random_mutation_ablation_passed`: principle "the ENERGY-SPECIFIC control -- the energy-fitness QD win must beat a random-mutation/no-energy-fitness QD (else it is the search/branching, not the energy fitness, doing the work)."
- `qd_lift_ci`: principle "bootstrap CI on winner_generated / solve-rate vs the random-mutation ablation; a claim requires the CI to exclude it."
- `first_win_rate_delta`: principle "qd - search_baseline first-win-rate; emitted explicitly so a regression is caught (generation must not cost first-wins)."
- `bare_control_passed`: principle "the POSITIVE CONTROL -- the games are pre-confirmed cell_recall-reachable (headroom exists); a no-winner null is valid only then."
- `false_negative_risk_checked`: principle "true with the search-only baseline + random-mutation ablation + reachable-headroom confirmed -- a 'no winner' null is valid only then."
- `p01_shadow_note`: principle "records the operator's honest caveat -- #2 lives under the P0.1 de-novo-generation shadow; the bet is population-seeding + diversity escapes it. States whether the bet paid off."
- `null_delta_methodology_note`: principle "present when a delta==0 / winner_generated false -- states the equality is an honest no-value null, not a bug."
- `chosen_submitted_config`: principle "the recommended SUBMITTED_AGENT_CONFIG change (QD generator on, archive/mutation params) -- the A6 input; 'unchanged' if null."
- `parity_test_green`: principle "HARD gate -- test_arc_submitted_agent_parity.py passes."
- `offline_reproduced`: principle "any QD-generated winner must offline-reproduce (arc_solver_kit.reproduce) to count."
- `residual_bridge_gaps`: principle "the Missing-Verifier / bridge gap logged if QD nulls -- the next-attack record."
- `random_seed`: principle "determinism precondition for reproducibility (QD mutation RNG seeded)."
- `reproducibility_checksum`: principle "content-addressed hash catches silent harness/corpus drift on replay."
- `preconditions_checked`: principle "records resources verified (offline arcade, E3AgentPolicy + explorer + goal-induction + action-effect importable, exp4020 asset present); pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4653

**Given** the offline arcade, E3 live-path imports, Exp4020 goal-energy asset,
and action-effect scorer imports are available
**When** Experiment 4653 runs energy-fitness QD, search-only, and random-mutation
controls on matched cell-recall-reachable hard-game evidence
**Then** the QD module is live-path reachable, primitive candidates remain
available, the artifact reports winner-generation, live solve-rate, depth,
actions-to-win, first-win deltas, random-mutation ablation status, bootstrap CI,
offline reproduction status for any generated winner, and an honest null note
when no new winner is generated.

### REQ-ARC-WMTE-4659: Adversarial Verify QD Ablation And Value-Routing Cost Hardening

The `scripts/adversarial_verify.py` reader SHALL protect the .429 A2
energy-as-fitness QD claim and the .429 A1 value-routing cost-fix claim from
over-attribution. An ARC artifact that claims a QD / energy-fitness generation
win through `winner_generated=true`, `winner_generated_count > 0`, or
`solve_rate_delta > 0` SHALL report `random_mutation_ablation_passed=true`.
If the win is claimed while `random_mutation_ablation_passed` is false or
absent, the reader SHALL emit a CRITICAL flag of kind
`qd-without-random-mutation-ablation`. If a QD / energy-fitness live-generation
claim omits `random_mutation_ablation_passed`, the reader SHALL emit a WARN flag
of kind `qd-random-mutation-ablation-omitted`.

An ARC artifact that claims a value-routing live first-win or solve-rate lift
through `first_win_rate_delta > 0` or `solve_rate_delta > 0` SHALL report a
finite `per_node_feature_cost_ms` and `sim_timed_out=false`. If a value-routing
live win is claimed while either cost-control field is absent or
`sim_timed_out` is true, the reader SHALL emit a CRITICAL flag of kind
`value-routing-without-cost-control`. If a value-routing live claim omits either
cost-control field, the reader SHALL emit a WARN flag of kind
`value-routing-cost-control-omitted`.

The honest .429 A1/A2 artifacts,
`results/experiment_4652_value_routing_cost_fix_live.json` and
`results/experiment_4653_energy_fitness_qd_generation_live.json`, SHALL NOT fire
either new guard because they report the relevant controls and do not overclaim
positive live lift.

Experiment 4659 SHALL write
`results/experiment_4659_adversarial_verify_hardening.json` with
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `qd_ablation_guard_added`,
`value_routing_cost_guard_added`, `honest_artifacts_not_flagged`,
`tests_added`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`.

Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: adversarial_verify_hardened_qd_ablation_and_value_routing_cost_guards_tests_green."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts -- reads fixtures + edits the linter, no model load (100us floor)."
- `qd_ablation_guard_added`: principle "the QD-WITHOUT-RANDOM-MUTATION-ABLATION guard (a QD generation win must beat a random-mutation ablation, else flagged)."
- `value_routing_cost_guard_added`: principle "the VALUE-ROUTING-WITHOUT-COST-CONTROL guard (a value-routing win must report per-node cost + no-timeout, else flagged)."
- `honest_artifacts_not_flagged`: principle "the honest A1/A2 artifacts (which report their controls) are NOT flagged -- false-positive guard (like the .428 goal-energy-ablation guard)."
- `tests_added`: principle "the unit tests for both guards (Tests Must Run and Assert: flag the over-claim, pass the honest)."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4659-QD-RANDOM-MUTATION-ABLATION

**Given** an ARC artifact claims an energy-fitness QD generation win
**When** `adversarial_verify.py` checks the artifact
**Then** the artifact is CRITICAL-flagged when
`random_mutation_ablation_passed` is false or absent, WARN-flagged when that
field is omitted, and not flagged when the ablation passes.

#### SCENARIO-ARC-WMTE-4659-VALUE-ROUTING-COST-CONTROL

**Given** an ARC artifact claims a value-routing live first-win or solve-rate
lift
**When** `adversarial_verify.py` checks the artifact
**Then** the artifact is CRITICAL-flagged when it lacks a finite
`per_node_feature_cost_ms` or `sim_timed_out=false`, WARN-flagged when either
field is omitted, and not flagged when both controls are reported.

### REQ-ARC-WMTE-4656: Primitive Persist Transfer For .429 A1/A2

Experiment 4656 SHALL read
`results/experiment_4652_value_routing_cost_fix_live.json` and
`results/experiment_4653_energy_fitness_qd_generation_live.json`, select the
strongest reusable primitive from the .429 A1/A2 milestone, and persist it as
durable solver scaffolding. If A1's value-routing cost fix clears its live gate,
the persisted primitive SHALL be the A1 value-routing cost-fix operator. If A1
does not clear and A2's energy-fitness QD generator clears, the persisted
primitive SHALL be the A2 QD generator. If both are value-null, the workflow
SHALL persist the strongest characterized component, with the expected fallback
being the cheap-feature value-routing cost fix because it removes the prior
per-node feature-cost timeout while remaining oracle-distinct from the
executable win-check.

The persisted primitive SHALL be registered in `arc_solver_kit` as
`cheap_value_routing_cost_fix_operator`, selected by
`select_primitive_operators()` for graph-explore/live-path transfer, and recorded
under `ops/arc_solve_registry.yaml` `general_gotchas` as
`primitive_cheap_value_routing_cost_fix_operator` with the source artifacts and a
clear reuse note. Experiment 4656 SHALL apply the persisted operator to at least
three held-out cached games and report per-game solve-rate delta, first-win
delta, action-efficiency delta, and `offline_reproduced_new_level`. If no game
shows positive transfer value, the artifact SHALL report a characterized null
with a residual dead-end and SHALL NOT increment reproducible total levels.

Experiment 4656 SHALL write
`results/experiment_4656_primitive_persist_transfer.json` with principle-annotated
top-level fields for `honest_verdict`, `inference_substrate`,
`verifier_is_oracle`, `primitive_persisted`, `transfer_games`,
`transfer_value_per_game`, `offline_reproduced_new_level`,
`residual_dead_end`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`.

Required field principles are:

- `honest_verdict`: principle "terminal prefix; success: primitive_persisted_transfer_<value|null>_characterized OR complete: primitive_persisted_transfer_null_characterized."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline transfer measurement over cached games (1s floor)."
- `verifier_is_oracle`: principle "MUST be false -- the persisted primitive ranks/routes/generates, oracle-distinct from the executable win-check."
- `primitive_persisted`: principle "the operator id + derived_from_artifacts + source -- the reusable scaffolding captured so the live solver reuses it, never re-derives."
- `transfer_games`: principle "the >=3 held-out games the persisted operator was applied to (the cross-game transfer measurement)."
- `transfer_value_per_game`: principle "per-game solve-rate/first-win/efficiency delta + offline_reproduced_new_level -- honest transfer value, null characterized if zero."
- `offline_reproduced_new_level`: principle "true only if the transfer banked a strictly NEW offline-reproduced level (else reproducible_total_levels is unchanged -- stated honestly)."
- `residual_dead_end`: principle "the characterized transfer-null residual if value is zero (the next-attack record); per the .428 A5 pattern."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4656

**Given** the offline arcade precondition passes, the A1/A2 artifacts are
readable, the cheap value-routing primitive is registered in `arc_solver_kit`,
and the registry contains the persisted primitive gotcha
**When** Experiment 4656 selects the strongest reusable .429 component and
measures transfer on at least three cached games
**Then** it writes the 4656 result artifact with the persisted operator,
derived-from artifacts, per-game transfer deltas, `verifier_is_oracle=false`,
strict new-level reproduction accounting, a stable checksum, and an honest
residual dead-end when transfer value is null.

### REQ-ARC-WMTE-4668: Primitive Persist Transfer For .430 A1/A2

Experiment 4668 SHALL read
`results/experiment_4664_l2_goal_predicate_induction_live.json` and
`results/experiment_4665_dagger_distribution_shift_value_routing.json`, select
the strongest reusable primitive from the .430 A1/A2 milestone, and persist it
as durable solver scaffolding. If the A1 L2-goal-induction operator clears its
generic-agent L2 gate, the persisted primitive SHALL be the A1 goal-predicate
induction operator. If A1 does not clear and the A2 DAgger-lite
distribution-corrected value-routing operator clears its live-lift gate, the
persisted primitive SHALL be the A2 corrected value-routing operator. If both
are value-null, the workflow SHALL persist the strongest characterized reusable
component, with the expected fallback being the DAgger-lite off-path
data-collection operator because it reduced the measured search-distribution
shift while remaining oracle-distinct from the executable win-check.

The persisted fallback primitive SHALL be registered in `arc_solver_kit` as
`dagger_off_path_data_collection_operator`, selected by
`select_primitive_operators()` for graph-explore/live-path transfer, and recorded
under `ops/arc_solve_registry.yaml` `general_gotchas` as
`primitive_dagger_off_path_data_collection_operator` with the source artifacts
and a clear reuse note. Experiment 4668 SHALL apply the persisted operator to at
least three held-out cached games and report per-game solve-rate delta,
first-win delta, action-efficiency delta, and `offline_reproduced_new_level`.
If no game shows positive transfer value, the artifact SHALL report a
characterized null with a residual dead-end and SHALL NOT increment reproducible
total levels.

Experiment 4668 SHALL write
`results/experiment_4668_primitive_persist_transfer.json` with
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `verifier_is_oracle`, `primitive_persisted`,
`transfer_games`, `transfer_value_per_game`, `offline_reproduced_new_level`,
`residual_dead_end`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`.

Required field principles are:

- `honest_verdict`: principle "terminal prefix; success: primitive_persisted_transfer_<value|null>_characterized OR complete: primitive_persisted_transfer_null_characterized."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline transfer measurement over cached games (1s floor)."
- `verifier_is_oracle`: principle "MUST be false -- the persisted primitive ranks/routes/induces, oracle-distinct from the executable win-check."
- `primitive_persisted`: principle "the operator id + derived_from_artifacts + source -- the reusable scaffolding captured so the live solver reuses it, never re-derives."
- `transfer_games`: principle "the >=3 held-out games the persisted operator was applied to (the cross-game transfer measurement)."
- `transfer_value_per_game`: principle "per-game solve-rate/first-win/efficiency delta + offline_reproduced_new_level -- honest transfer value, null characterized if zero."
- `offline_reproduced_new_level`: principle "true only if the transfer banked a strictly NEW offline-reproduced level (else reproducible_total_levels is unchanged -- stated honestly)."
- `residual_dead_end`: principle "the characterized transfer-null residual if value is zero (the next-attack record); per the .429 A5 pattern."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4668

**Given** the offline arcade precondition passes, the A1/A2 artifacts are
readable, the DAgger off-path primitive is registered in `arc_solver_kit`, and
the registry contains the persisted primitive gotcha
**When** Experiment 4668 selects the strongest reusable .430 component and
measures transfer on at least three cached games
**Then** it writes the 4668 result artifact with the persisted operator,
derived-from artifacts, per-game transfer deltas, `verifier_is_oracle=false`,
strict new-level reproduction accounting, a stable checksum, and an honest
residual dead-end when transfer value is null.

### REQ-ARC-WMTE-4657: Submitted A1/A2 Integration Gate

Experiment 4657 SHALL read
`results/experiment_4652_value_routing_cost_fix_live.json` and
`results/experiment_4653_energy_fitness_qd_generation_live.json`, fold the
admissible A1/A2 submitted-agent configuration into
`SUBMITTED_AGENT_CONFIG`, and re-measure the scored `E3AgentPolicy` config
against the pre-integration submitted config. The gate SHALL treat the A1
cost-fixed value-routing path as integrated only when the submitted config has
the artifact's positive `value_weight_set`, the declared cheap
`cross_game_features_v3:v2_plus_frame_delta` feature subset, and
`verifier_is_oracle=false`. The gate SHALL treat A2 as unchanged when its
`chosen_submitted_config` is `unchanged` or when the submitted config keeps QD
generation disabled.

Experiment 4657 SHALL write
`results/experiment_4657_integration_gate.json` with principle-annotated
top-level fields for `honest_verdict`, `inference_substrate`,
`verifier_is_oracle`, `config_integrated`,
`live_first_win_rate_integrated`,
`live_multi_level_solve_rate_integrated`,
`live_submittable_level_count_integrated`, `parity_test_green`,
`random_seed`, `reproducibility_checksum`, and `preconditions_checked`. The
artifact SHALL also record the pre-integration comparison config, first-win and
multi-level deltas versus that config, the A1/A2 integration audit, the parity
test command result, `submitted_to_leaderboard=false`, and a stable checksum.
If both A1 and A2 remain null or unchanged, the artifact SHALL emit
`complete: integration_unchanged_both_levers_null`; otherwise a successful
parity-green no-regression integration SHALL emit
`success: integrated_<config>_shipped_parity_green`.

Required field principles are:

- `honest_verdict`: principle "terminal prefix; success: integrated_<config>_shipped_parity_green OR complete: integration_unchanged_both_levers_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline end-to-end re-measure (1s floor)."
- `verifier_is_oracle`: principle "MUST be false -- the integrated value-routing/QD signals are oracle-distinct from the executable win-check."
- `config_integrated`: principle "the A1/A2 config folded into SUBMITTED_AGENT_CONFIG (or 'unchanged' with the reason) -- the single-source-of-truth update."
- `live_first_win_rate_integrated`: principle "the integrated SCORED-agent live first-win-rate (no regression vs pre-integration)."
- `live_multi_level_solve_rate_integrated`: principle "the integrated SCORED-agent multi-level (>=2) live solve-rate (the deeper wall)."
- `live_submittable_level_count_integrated`: principle "the integrated live-submittable count (must stay > 33)."
- `parity_test_green`: principle "HARD gate -- test_arc_submitted_agent_parity.py passes; the deployed agent == the measured agent."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4657

**Given** the offline arcade and submitted-agent import preconditions pass, the
A1 and A2 artifacts are readable, and `SUBMITTED_AGENT_CONFIG` is the scored
`E3AgentPolicy` source of truth
**When** Experiment 4657 folds in A1/A2 and runs the submitted-agent parity
test
**Then** the result artifact reports the integrated config, live first-win
rate, live multi-level solve rate, live-submittable count greater than 33,
no-regression deltas versus the pre-integration config,
`verifier_is_oracle=false`, `parity_test_green=true`, and a stable
reproducibility checksum.

### REQ-ARC-WMTE-4669: ARC Sprint A1/A2 Submitted Config Integration Gate

Experiment 4669 SHALL read the previous integration gate
`results/experiment_4657_integration_gate.json`, the L2 goal-predicate
induction artifact
`results/experiment_4664_l2_goal_predicate_induction_live.json`, and the
DAgger-lite value-routing artifact
`results/experiment_4665_dagger_distribution_shift_value_routing.json`. The
gate SHALL audit whether the A1 L2-goal-induction submitted config
(win-grid exemplar injection, goal-predicate satisfiability check, and fixed
multi-level harness) or the A2 distribution-corrected value-head submitted
config (checkpoint, positive `value_weight`, and
`cross_game_features_v3:v2_plus_frame_delta` feature subset) is actually
recommended by the upstream artifact and folded into `SUBMITTED_AGENT_CONFIG`.
If both upstream artifacts are null or explicitly `unchanged`, the gate SHALL
record `config_integrated` as unchanged with the reason instead of fabricating
an integration.

Experiment 4669 SHALL re-measure or reuse the fixed cached live measurements
for the scored `E3AgentPolicy` against the pre-integration config, and SHALL
write `results/experiment_4669_integration_gate.json` with
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `verifier_is_oracle`, `config_integrated`,
`live_first_win_rate_integrated`,
`live_multi_level_solve_rate_integrated`,
`live_submittable_level_count_integrated`, `parity_test_green`,
`random_seed`, `reproducibility_checksum`, and `preconditions_checked`. The
artifact SHALL also include the A1/A2 audit, pre-integration comparison
metrics, no-regression status, source artifact paths/checksums, the parity-test
result, `submitted_to_leaderboard=false`, and spec refs. The verifier SHALL
remain oracle-distinct from the executable win-check.

Required field principles are:

- `honest_verdict`: principle "terminal prefix; success: integrated_<config>_shipped_parity_green OR complete: integration_unchanged_both_levers_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline end-to-end re-measure (1s floor)."
- `verifier_is_oracle`: principle "MUST be false -- the integrated L2-goal-induction / value-routing signals are oracle-distinct from the executable win-check."
- `config_integrated`: principle "the A1/A2 config folded into SUBMITTED_AGENT_CONFIG (or 'unchanged' with the reason) -- the single-source-of-truth update."
- `live_first_win_rate_integrated`: principle "the integrated SCORED-agent live first-win-rate (no regression vs pre-integration)."
- `live_multi_level_solve_rate_integrated`: principle "the integrated SCORED-agent multi-level (>=2) live solve-rate (the deeper wall) -- measured on the FIXED non-degenerate harness."
- `live_submittable_level_count_integrated`: principle "the integrated live-submittable count (must stay > 33)."
- `parity_test_green`: principle "HARD gate -- test_arc_submitted_agent_parity.py passes; the deployed agent == the measured agent."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4669

**Given** `SUBMITTED_AGENT_CONFIG` imports with `E3AgentPolicy`, the 4657
pre-integration comparison artifact is present, and the 4664/4665 A1/A2
artifacts are readable
**When** Experiment 4669 audits the upstream chosen submitted configs and runs
the submitted-agent parity test
**Then** the result artifact reports either the exact A1/A2 config folded into
the single source of truth or an unchanged reason when both levers are null,
reports live first-win rate, fixed-harness multi-level solve rate, live
submittable count greater than 33, no-regression versus the pre-integration
config, `verifier_is_oracle=false`, `parity_test_green=true`, and a stable
reproducibility checksum.

### REQ-ARC-WMTE-4549: Reusable LLM-Proposer Re-Induction Primitive Transfer

The solver kit SHALL persist the live LLM-proposer re-induction loop from
experiment 4544 as a reusable primitive operator named
`llm_proposer_reinduction_operator`. The operator SHALL detect a level-up,
clear stale level-local induction state, ask a proposer for separate GOAL,
DYNAMICS/world-model, and plan candidates when a live proposer is available,
rank candidates by oracle-distinct trust-energy evidence, and run a bounded
counterexample-guided refinement loop of at most three rounds. If the live
proposer is unavailable or the upstream A1 artifact is null, the operator SHALL
fall back to the persisted DSL/verifier predicate inducer while preserving the
same transfer artifact fields and explicitly reporting that no reachable
LLM-produced deeper plan was available.

Experiment 4549 SHALL read
`results/experiment_4544_llm_proposer_reinduction.json`, persist the operator
in `python/carnot/agentic/arc_solver_kit.py`, extend the existing
`primitive_per_level_reinduction_operator` `general_gotchas` registry row
rather than duplicating it, and apply the persisted primitive to at least two
deeper games that were not tuned in A1 from `{tn36,tr87,tu93,sc25}`. For each
transfer game, the artifact SHALL report the deepest offline-reproduced level
available to the transfer replay, whether the primitive produced a reachable
deeper-level plan, whether it re-induced a different representation-correct
`L_{n+1}` predicate, and the dead-end that remains if no deeper level is
banked. Any newly claimed deeper level SHALL pass `arc_solver_kit.reproduce()`
before it counts toward `reproducible_total_levels`; otherwise the registry
total SHALL remain unchanged and the artifact SHALL report a terminal honest
null.

Experiment 4549 SHALL write
`results/experiment_4549_llm_proposer_primitive_persist_transfer.json` with
bare top-level fields for `honest_verdict`, `inference_substrate`,
`primitive_persisted`, `transfer_games`,
`transfer_deepest_level_per_game`, `reachable_plan_produced`,
`representation_transfer`, `offline_reproduced`, `registry_updated`,
`random_seed`, `reproducibility_checksum`, and `preconditions_checked`.
Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: llm_proposer_primitive_persisted_transfer_<game>_L<n> OR complete: llm_proposer_primitive_persisted_transfer_null_characterized."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates for the DSL-fallback transfer; live_llm_inference + model_specs if the LLM proposer is invoked in the transfer runs."
- `primitive_persisted`: principle "names the arc_solver_kit operator + registry general_gotcha id added/extended -- the reusable asset (Solver-Reuse Discipline); without it the A1 effort is wasted per the ARC reuse rule."
- `transfer_games`: principle "the deeper games the primitive was applied to (NOT tuned on) -- the generalization test."
- `transfer_deepest_level_per_game`: principle "best_level reached per transfer game -- the cross-game evidence the primitive generalizes."
- `reachable_plan_produced`: principle "whether the persisted primitive produced a REACHABLE deeper-level plan on a transfer game (the A1-barrier-clearing evidence), distinct from mere representation re-induction."
- `representation_transfer`: principle "whether the primitive re-induced a DIFFERENT (correct) L_{n+1} predicate even without a full solve -- a representation win short of a bank."
- `offline_reproduced`: principle "only offline-reproduced new levels count toward reproducible_total_levels."
- `registry_updated`: principle "the primitive + transfer dead-ends persisted so the next milestone reuses, not re-derives."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

### SCENARIO-ARC-WMTE-4549: LLM-Proposer Primitive Transfers Without Retuning A1

Given the offline arcade precondition passes and the experiment 4544 A1 artifact
exists
When experiment 4549 runs the persisted solver-kit LLM-proposer re-induction
operator on at least two of `tn36`, `tr87`, `tu93`, and `sc25`
Then the artifact records the persisted operator, the transfer games, each
game's deepest offline-reproduced level, any reachable deeper-level plan, any
different representation-correct `L_{n+1}` predicate, and a terminal honest null
unless a newly reproduced deeper level passes the offline reproduction gate.

### REQ-ARC-WMTE-4536: Submitted A1/A2 Integration Gate Refresh

Experiment 4536 SHALL read the A1 per-level goal re-induction artifact from
`results/experiment_4533_per_level_goal_reinduction.json` and the A2
trust-energy routing artifact from
`results/experiment_4534_energy_trust_next_level_routing.json`. The workflow
SHALL select only levers that strictly raise CORE per-level efficiency above the
canonical `2.0074` baseline by reaching a deeper CORE level while preserving
every CORE solve. Median-actions or action-trimming effects SHALL NOT count as
score levers. An upstream artifact with `flagged_adversarial: true` SHALL be
rejected unless its only flag is the control-equals-best null-delta tautology:
it MUST include explicit `efficiency_delta: 0.0`, a non-empty
`null_delta_methodology_note`, and matching baseline/best core efficiency. Such
an artifact MAY be aggregated only as a valid null and SHALL NOT be treated as
an improvement.

Experiment 4536 SHALL re-measure the current submitted-default configuration
end-to-end through `scripts/kaggle/arc_local_submission_gate.py --check` and
write `results/experiment_4536_integration_8game_gate.json` with bare
top-level fields for `honest_verdict`, `inference_substrate`,
`core_efficiency_baseline`, `core_efficiency_integrated`,
`core_solves_preserved`, `levers_integrated`, `additivity_checked`,
`heldout_solve_rate`, `ready_for_operator_submit`,
`false_negative_risk_checked`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`. Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: integrated_core_efficiency_<n>_above_2.0074 OR complete: no_lever_raises_core_efficiency_honest_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade end-to-end via the per-level gate."
- `core_efficiency_baseline`: principle "2.0074 -- the REAL per-level metric control (NOT median actions, retired)."
- `core_efficiency_integrated`: principle "the HEADLINE -- the SUBMITTED-config per-level efficiency after wiring the winners (did it solve MORE levels)."
- `core_solves_preserved`: principle "integration must preserve every CORE solve (set-containment)."
- `levers_integrated`: principle "names which of A1/A2 were wired -- traceable to their measured deltas; [] is an honest null."
- `additivity_checked`: principle "integrated CORE core_efficiency vs the naive sum of isolated A1+A2 deltas -- surfaces a destructive re-induction x energy-routing interaction instead of burying it."
- `heldout_solve_rate`: principle "the real transfer signal; integration should not regress it."
- `ready_for_operator_submit`: principle "True if the integrated config is a CORE-preserved core_efficiency improvement worth a 1/day submission slot; the task NEVER submits (operator-only)."
- `false_negative_risk_checked`: principle "an honest null only valid with the 2.0074 baseline measured the same way."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

A success artifact SHALL require `core_efficiency_integrated > 2.0074`,
`core_solves_preserved=true`, at least one integrated A1/A2 deeper-level lever,
and `ready_for_operator_submit=true`. If neither A1 nor A2 raised CORE
per-level efficiency without a CORE level loss, the artifact SHALL report
`complete: no_lever_raises_core_efficiency_honest_null`, keep
`levers_integrated=[]`, keep the submitted configuration unchanged, and SHALL
NOT submit to the leaderboard.

### REQ-ARC-WMTE-4548: Submitted A1/A4 Integration Gate

Experiment 4548 SHALL read the A1 live LLM-proposer re-induction artifact from
`results/experiment_4544_llm_proposer_reinduction.json` and the A4 cached
CNN frame-change ranker artifact from
`results/experiment_4547_frame_change_predictor.json`. The workflow SHALL
select only levers that strictly raise CORE per-level efficiency above the
canonical `2.0074` baseline by reaching a deeper CORE level, or by reaching a
CORE level within budget that the matched bare explorer missed, while preserving
every CORE solve. Median-actions effects SHALL NOT count as score levers unless
the per-level gate shows a CORE efficiency increase. An upstream artifact with
`flagged_adversarial: true` SHALL be rejected unless its only flag is the
control-equals-best null-delta tautology: it MUST include explicit
`efficiency_delta: 0.0`, a non-empty `null_delta_methodology_note`, matching
baseline/best core efficiency, and no other corrigendum flags. Such an artifact
MAY be recorded only as a valid null and SHALL NOT be treated as an improvement.

Experiment 4548 SHALL re-measure the submitted-default configuration
end-to-end through `scripts/kaggle/arc_local_submission_gate.py --check` and
write `results/experiment_4548_integration_8game_gate.json` with bare
top-level fields for `honest_verdict`, `inference_substrate`,
`core_efficiency_baseline`, `core_efficiency_integrated`,
`core_solves_preserved`, `levers_integrated`, `additivity_checked`,
`heldout_solve_rate`, `ready_for_operator_submit`,
`false_negative_risk_checked`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`. Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: integrated_core_efficiency_<n>_above_2.0074 OR complete: no_lever_raises_core_efficiency_honest_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade end-to-end via the per-level gate (if the integrated config invokes the LLM proposer in the measured run, declare live_llm_inference + add the model precondition)."
- `core_efficiency_baseline`: principle "2.0074 -- the REAL per-level metric control (NOT median actions, retired)."
- `core_efficiency_integrated`: principle "the HEADLINE -- the SUBMITTED-config per-level efficiency after wiring the winners (did it solve MORE/DEEPER levels)."
- `core_solves_preserved`: principle "integration must preserve every CORE solve (set-containment)."
- `levers_integrated`: principle "names which of A1/A4 were wired -- traceable to their measured deltas; [] is an honest null."
- `additivity_checked`: principle "integrated CORE core_efficiency vs the naive sum of isolated A1+A4 deltas -- surfaces a destructive LLM-proposer x CNN-ranker interaction instead of burying it."
- `heldout_solve_rate`: principle "the real transfer signal; integration should not regress it."
- `ready_for_operator_submit`: principle "True if the integrated config is a CORE-preserved core_efficiency improvement worth a 1/day submission slot; the task NEVER submits (operator-only)."
- `false_negative_risk_checked`: principle "an honest null only valid with the 2.0074 baseline measured the same way."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

A success artifact SHALL require `core_efficiency_integrated > 2.0074`,
`core_solves_preserved=true`, at least one integrated A1/A4 core-efficiency
lever, and `ready_for_operator_submit=true`. If neither A1 nor A4 raised CORE
per-level efficiency without a CORE level loss, the artifact SHALL report
`complete: no_lever_raises_core_efficiency_honest_null`, keep
`levers_integrated=[]`, keep the submitted configuration unchanged, and SHALL
NOT submit to the leaderboard.

### REQ-ARC-WMTE-4560: Submitted A1/A2/A4 Two-Metric Integration Gate

Experiment 4560 SHALL read the A1 verifier-router generic-transfer artifact
from `results/experiment_4556_verifier_router_generic_transfer.json`, the A2
executable world-model proposer artifact from
`results/experiment_4557_executable_world_model_proposer.json`, and the A4
hidden-field state probe artifact from
`results/experiment_4559_hidden_field_state_probe.json`. The workflow SHALL
select only levers that raise a real metric: A1 only when
`generic_transfer_rate_with_verifier > 0.04` with a non-negative confidence
interval and no CORE level loss, A2 only when a CORE game reaches L2 while
`core_solves_preserved=true`, and A4 only when a newly offline-reproduced bank
is present. An upstream artifact with `flagged_adversarial: true` SHALL be
rejected unless its only flag is the control-equals-best null-delta tautology
with an explicit zero delta and a non-empty null-delta methodology note; that
exception MAY be recorded only as a valid null and SHALL NOT be treated as an
improvement.

Experiment 4560 SHALL re-measure the SUBMITTED configuration end-to-end through
`scripts/kaggle/arc_local_submission_gate.py --check` for CORE per-level
efficiency and through `measure_generic_transfer_over_variants` for held-out
generic transfer. It SHALL write
`results/experiment_4560_integration_8game_gate.json` with bare top-level
fields for `honest_verdict`, `inference_substrate`,
`core_efficiency_integrated`, `generic_transfer_rate_integrated`,
`core_solves_preserved`, `levers_integrated`, `additivity_checked`,
`heldout_solve_rate`, `ready_for_operator_submit`,
`false_negative_risk_checked`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`. Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: integrated_generic_transfer_<n>_above_0.04_or_core_efficiency_above_2.0074 OR complete: no_lever_raises_a_metric_honest_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade end-to-end (if the integrated config invokes the LLM proposer in the measured run, declare live_llm_inference + add the model precondition)."
- `core_efficiency_integrated`: principle "the SUBMITTED-config per-level efficiency after wiring winners vs baseline 2.0074."
- `generic_transfer_rate_integrated`: principle "the SUBMITTED-config held-out variant transfer after wiring winners vs baseline 0.04 -- the leaderboard-honest HEADLINE."
- `core_solves_preserved`: principle "integration must preserve every CORE solve (set-containment)."
- `levers_integrated`: principle "names which of A1/A2/A4 were wired -- traceable to their measured deltas; [] is an honest null."
- `additivity_checked`: principle "integrated metric vs the naive sum of isolated A1/A2/A4 deltas -- surfaces a destructive interaction instead of burying it."
- `heldout_solve_rate`: principle "the real transfer signal; integration should not regress it."
- `ready_for_operator_submit`: principle "True if the integrated config is a CORE-preserved improvement on either metric worth a 1/day submission slot; the task NEVER submits (operator-only)."
- `false_negative_risk_checked`: principle "an honest null is only valid with both baselines measured the same way."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

A success artifact SHALL require at least one accepted A1/A2/A4 lever,
`core_solves_preserved=true`, and either
`generic_transfer_rate_integrated > 0.04` or
`core_efficiency_integrated > 2.0074`. If no lever raises either metric, the
artifact SHALL report `complete: no_lever_raises_a_metric_honest_null`, keep
`levers_integrated=[]`, keep the submitted configuration unchanged, and SHALL
NOT submit to the leaderboard.

### REQ-ARC-WMTE-4561: Persist Winning Primitive And Measure Cross-Game Transfer

Experiment 4561 SHALL read the A1 verifier-router generic-transfer artifact
from `results/experiment_4556_verifier_router_generic_transfer.json` and the
A2 executable world-model proposer artifact from
`results/experiment_4557_executable_world_model_proposer.json`, select the
best-characterized primitive with the strongest measured signal, persist it as
a reusable solver-kit operator, and record that reusable asset in
`ops/arc_solve_registry.yaml` under `general_gotchas`. If A1 and A2 are both
honest nulls, the workflow SHALL still persist the best-characterized
primitive-as-built, report the transfer null, and state what evidence or
capability would be needed for the primitive to add value.

For the A1 path, the solver kit SHALL expose
`verifier_router_candidate_ranking_operator` as a generic candidate-ranking
operator. The operator SHALL rank candidate actions or plans by a verifier
score while keeping depth/search cost primary outside the compatible candidate
set, preserve deterministic tie-breaking by original order, and report ordering
gain versus the incoming order. The operator SHALL be registered in
`primitive_operator_registry()` and surfaced by `select_primitive_operators()`
and `arc_solve_learning.recommend_approach()` for graph/config/program-editor
routes so live solves can reuse it before game-specific reverse engineering.

Experiment 4561 SHALL apply the persisted primitive to at least two games that
were not tuned in A1/A2 and SHALL write
`results/experiment_4561_primitive_persist_transfer.json` with bare top-level
fields for `honest_verdict`, `inference_substrate`, `primitive_persisted`,
`transfer_games`, `transfer_value_per_game`, `offline_reproduced`,
`registry_updated`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`. Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: primitive_persisted_transfer_<game>_value_added OR complete: primitive_persisted_transfer_null_characterized."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates for the offline-verifier transfer; live_llm_inference + model_specs if the executable proposer is invoked in the transfer runs."
- `primitive_persisted`: principle "names the arc_solver_kit operator + registry general_gotcha id added/extended -- the reusable asset (Solver-Reuse Discipline); without it the A1/A2 effort is wasted per the ARC reuse rule."
- `transfer_games`: principle "the games the primitive was applied to (NOT tuned on) -- the generalization test."
- `transfer_value_per_game`: principle "the per-game value-add (verifier-router ordering gain / executable-proposer reachable plan) -- the cross-game evidence the primitive generalizes."
- `offline_reproduced`: principle "only offline-reproduced new levels count toward reproducible_total_levels."
- `registry_updated`: principle "the primitive + transfer dead-ends persisted so the next milestone reuses, not re-derives."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

If no held-out transfer game shows positive ordering gain or a reachable plan,
the artifact SHALL report
`complete: primitive_persisted_transfer_null_characterized`, keep
`offline_reproduced=false` unless a new level passed `arc_solver_kit.reproduce`,
and record the transfer dead-ends in the registry rather than incrementing
`reproducible_total_levels`.

### REQ-ARC-WMTE-4537: Reusable Per-Level Re-Induction Primitive Transfer

The solver kit SHALL persist the per-level re-induction loop from experiment
4533 as a reusable primitive operator named `per_level_reinduction_operator`.
The operator SHALL detect a
`frame.levels_completed` increment, clear stale level-local induction state,
re-induce the `L_{n+1}` predicate from the post-transition representation, and
return a route that keeps depth as the primary search key while using the
newly induced predicate only as a goal-bias/routing hint. The operator SHALL be
registered in `primitive_operator_registry()` and selected by
`select_primitive_operators()` for deepening and graph/config/program-editor
mechanics so `arc_solve_learning.recommend_approach()` can surface it before
per-game reverse engineering.

Experiment 4537 SHALL read
`results/experiment_4533_per_level_goal_reinduction.json`, persist the
operator in `python/carnot/agentic/arc_solver_kit.py`, add a
`general_gotchas` registry row documenting the primitive, and apply the
persisted primitive to at least two deeper games that were not tuned in A1 from
`{tn36,tr87,tu93,sc25}`. For each transfer game, the artifact SHALL report the
deepest offline-reproduced level reached by the transfer replay, whether the
primitive re-induced a different representation-correct `L_{n+1}` predicate,
and the route/dead-end that remains if no deeper level is banked. Any newly
claimed deeper level SHALL pass `arc_solver_kit.reproduce()` before it counts
toward `reproducible_total_levels`; otherwise the registry total SHALL remain
unchanged and the artifact SHALL report a terminal honest null.

Experiment 4537 SHALL write
`results/experiment_4537_reinduction_primitive_persist_transfer.json` with bare
top-level fields for `honest_verdict`, `inference_substrate`,
`primitive_persisted`, `transfer_games`,
`transfer_deepest_level_per_game`, `representation_transfer`,
`offline_reproduced`, `registry_updated`, `random_seed`,
`reproducibility_checksum`, and `preconditions_checked`. Required field
principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: reinduction_primitive_persisted_transfer_<game>_L<n> OR complete: reinduction_primitive_persisted_transfer_null_characterized."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade primitive application, no headline LLM load."
- `primitive_persisted`: principle "names the arc_solver_kit operator + registry general_gotcha id added -- the reusable asset (Solver-Reuse Discipline); without it the A1 effort is wasted per the ARC reuse rule."
- `transfer_games`: principle "the deeper games the primitive was applied to (NOT tuned on) -- the generalization test."
- `transfer_deepest_level_per_game`: principle "best_level reached per transfer game -- the cross-game evidence the primitive generalizes."
- `representation_transfer`: principle "whether the primitive re-induced a DIFFERENT (correct) L_{n+1} predicate even without a full solve -- a representation win short of a bank."
- `offline_reproduced`: principle "only offline-reproduced new levels count toward reproducible_total_levels."
- `registry_updated`: principle "the primitive + transfer dead-ends persisted so the next milestone reuses, not re-derives."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

### SCENARIO-ARC-WMTE-4537: Primitive Transfers Without Retuning A1

Given the offline arcade precondition passes and the A1 artifact exists
When experiment 4537 runs the persisted solver-kit re-induction operator on
at least two of `tn36`, `tr87`, `tu93`, and `sc25`
Then the artifact records the persisted operator, the transfer games, each
game's deepest offline-reproduced level, any different representation-correct
`L_{n+1}` predicate, and a terminal honest null unless a newly reproduced
deeper level passes the offline reproduction gate.

### REQ-ARC-WMTE-4525: Standing Loop Level-Up Registry Bank

Experiment 4525 SHALL run the offline arcade precondition and use the standing
ARC loop output as the source of truth for a level-up bank. The experiment SHALL
prefer an unadaptered public game for first-contact routing, but when all public
games already have registry rows it MAY deepen a registered adapter target. A
banked level SHALL require `offline_reproduced=true`, a reproduction gate whose
`reached_level` is greater than the target game's prior registry level, and a
persisted update to `ops/arc_solve_registry.yaml` that records the win
condition, action model, solver artifact, gotchas, and dead-ends needed by the
next attempt.

Experiment 4525 SHALL write
`results/experiment_4525_levelup_attempt.json` with bare top-level fields for
`honest_verdict`, `inference_substrate`, `offline_reproduced`,
`reproduced_levels`, `registry_updated`, `random_seed`,
`reproducibility_checksum`, `preconditions_checked`, `target_game`,
`reproduction_gate`, `solution_labels`, `dead_ends`, and `registry_update`.
Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: <game>_L<n>_offline_reproduced OR complete: <game>_routing_recommendation_delta_identified (un-adaptered: honest progress, not a bank)."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade solve, no headline LLM load."
- `offline_reproduced`: principle "a solve not reproducible offline is wasted effort -- only reproduced levels count toward reproducible_total_levels."
- `reproduced_levels`: principle "the integer new-level count banked this task (>=1 to satisfy the level-up guarantee on an attempted bank)."
- `registry_updated`: principle "the per-game win-condition/action-model/gotchas/dead-ends persisted so the next attempt reuses, not re-derives."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

A success artifact SHALL set `honest_verdict` to
`success: <game>_L<n>_offline_reproduced`, `offline_reproduced=true`,
`reproduced_levels >= 1`, and `registry_updated=true`. If the target only
produces a routing recommendation, repeats an already registered level, or
stalls at the prior level, the artifact SHALL use a `complete:` verdict, set
`reproduced_levels=0`, preserve the dead-end evidence, and SHALL NOT increment
the registry count.

### REQ-ARC-WMTE-4535: ARC Sprint Level-Up Attempt Guarantee Bank

Experiment 4535 SHALL run the offline arcade precondition and bank at least one
new reproducible ARC level independent of the A1/A2 headline path. The workflow
SHALL prefer deepening a shallow L1 game from `sp80`, `su15`, or `cn04`, SHALL
derive only the next-level win/action/state delta from the offline environment,
and SHALL use the standing ARC loop output plus `arc_solver_kit.reproduce` as
the source of truth for any bank. A banked level SHALL require
`offline_reproduced=true`, a reproduction gate whose `reached_level` is greater
than the target game's prior registry level, and a persisted update to
`ops/arc_solve_registry.yaml` that records the win condition, action model,
solver module, gotchas, and dead-ends needed by the next attempt.

Experiment 4535 SHALL write
`results/experiment_4535_levelup_attempt.json` with bare top-level fields for
`honest_verdict`, `inference_substrate`, `offline_reproduced`,
`reproduced_levels`, `registry_updated`, `random_seed`,
`reproducibility_checksum`, `preconditions_checked`, `target_game`,
`reproduction_gate`, `solution_labels`, `dead_ends`, and `registry_update`.
Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: <game>_L<n>_offline_reproduced OR complete: <game>_delta_identified_no_bank (honest progress, not a bank)."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade solve, no headline LLM load."
- `offline_reproduced`: principle "a solve not reproducible offline is wasted effort -- only reproduced levels count toward reproducible_total_levels."
- `reproduced_levels`: principle "the integer new-level count banked this task (>=1 to satisfy the level-up guarantee on an attempted bank)."
- `registry_updated`: principle "the per-game win-condition/action-model/gotchas/dead-ends persisted so the next attempt reuses, not re-derives."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

A success artifact SHALL set `honest_verdict` to
`success: <game>_L<n>_offline_reproduced`, `offline_reproduced=true`,
`reproduced_levels >= 1`, and `registry_updated=true`. If the target only
identifies a delta, repeats an already registered level, or stalls at the prior
level, the artifact SHALL use a `complete:` verdict, set
`reproduced_levels=0`, preserve the dead-end evidence, and SHALL NOT increment
the registry count.

### REQ-ARC-WMTE-4546: ARC Sprint Rotation Level-Up Bank

Experiment 4546 SHALL run the offline arcade precondition and bank at least one
new reproducible ARC level independent of the A1 headline path. The workflow
SHALL prefer a shallow game not recently deepened, SHALL select `su15`, SHALL
derive only the L2 win/action/state delta from the offline environment, and
SHALL use the standing ARC loop output plus `arc_solver_kit.reproduce` as the
source of truth for the bank. A banked level SHALL require
`offline_reproduced=true`, a reproduction gate whose `reached_level` is greater
than the target game's prior registry level, and a persisted update to
`ops/arc_solve_registry.yaml` that records the win condition, action model,
solver module, gotchas, and dead-end evidence needed by the next attempt.

Experiment 4546 SHALL write
`results/experiment_4546_levelup_attempt.json` with bare top-level fields for
`honest_verdict`, `inference_substrate`, `offline_reproduced`,
`reproduced_levels`, `registry_updated`, `random_seed`,
`reproducibility_checksum`, `preconditions_checked`, `target_game`,
`reproduction_gate`, `solution_labels`, `dead_ends`, and `registry_update`.
Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: <game>_L<n>_offline_reproduced OR complete: <game>_delta_identified_no_bank (honest progress, not a bank)."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade solve, no headline LLM load."
- `offline_reproduced`: principle "a solve not reproducible offline is wasted effort -- only reproduced levels count toward reproducible_total_levels."
- `reproduced_levels`: principle "the integer new-level count banked this task (>=1 to satisfy the level-up guarantee on an attempted bank)."
- `registry_updated`: principle "the per-game win-condition/action-model/gotchas/dead-ends persisted so the next attempt reuses, not re-derives."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

A success artifact SHALL set `honest_verdict` to
`success: su15_L2_offline_reproduced`, `offline_reproduced=true`,
`reproduced_levels >= 1`, and `registry_updated=true`. If `su15` repeats the
already registered L1, stalls at the prior level, or produces a non-reproducing
delta, the artifact SHALL use a `complete:` verdict, set
`reproduced_levels=0`, preserve the dead-end evidence, and SHALL NOT increment
the registry count.

### REQ-ARC-WMTE-4558: ARC Sprint Rotation Level-Up Attempt Ledger

Experiment 4558 SHALL run the offline arcade precondition and attempt to bank at
least one new reproducible ARC level independent of the A1 headline path. The
workflow SHALL prefer a rotated game not deepened in the .419/.420 sprint,
SHALL use the standing `scripts/arc_loop_solve.py` artifacts plus
`arc_solver_kit.reproduce` as the source of truth, and SHALL compare every
candidate reproduction gate against `ops/arc_solve_registry.yaml` before
claiming a bank. A banked level SHALL require `offline_reproduced=true`, a
reproduction gate whose `reached_level` is greater than the target game's prior
registry level, and a persisted registry update that records the next-level
win condition, action model, solver module, gotchas, and dead-end evidence.

Experiment 4558 SHALL write
`results/experiment_4558_levelup_attempt.json` with bare top-level fields for
`honest_verdict`, `inference_substrate`, `offline_reproduced`,
`reproduced_levels`, `registry_updated`, `random_seed`,
`reproducibility_checksum`, `preconditions_checked`, `target_game`,
`reproduction_gate`, `solution_labels`, `dead_ends`, and `registry_update`.
Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: <game>_L<n>_offline_reproduced OR complete: <game>_delta_identified_no_bank (honest progress, not a bank)."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade solve, no headline LLM load."
- `offline_reproduced`: principle "a solve not reproducible offline is wasted effort -- only reproduced levels count toward reproducible_total_levels."
- `reproduced_levels`: principle "the integer new-level count banked this task (>=1 satisfies the level-up guarantee on an attempted bank)."
- `registry_updated`: principle "the per-game win-condition/action-model/gotchas/dead-ends persisted so the next attempt reuses, not re-derives."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

A success artifact SHALL set `honest_verdict` to
`success: <game>_L<n>_offline_reproduced`, `offline_reproduced=true`,
`reproduced_levels >= 1`, and `registry_updated=true`. If the rotated target
only produces a routing recommendation, repeats an already registered level, or
stalls before a reproducing next-level gate, the artifact SHALL use
`complete: <game>_delta_identified_no_bank`, set `reproduced_levels=0`, preserve
the dead-end evidence, and SHALL NOT increment `reproducible_total_levels`.

### REQ-ARC-WMTE-4570: Rotated cn04 L1 to L2 Offline Bank

Experiment 4570 SHALL satisfy the ARC sprint level-up attempt guarantee by
deepening a rotated graph/world-model-family game that was not the wasted
`.421` `m0r0` attempt. The workflow SHALL choose `cn04` over the stalled
hidden-state `ar25` path, SHALL run the standing `scripts/arc_loop_solve.py`
entrypoint for `cn04`, SHALL derive the L2 win/action/state delta from the
offline environment, and SHALL register a `cn04` `GameAdapter` whose L2 plan is
gated by `arc_solver_kit.reproduce`.

The `cn04` adapter SHALL model the marker-pair shape-alignment mechanic:
ACTION1-4 move the selected visible sprite, ACTION5 rotates a singleton selected
sprite or cycles a stacked variant, ACTION6 selects a visible sprite through
camera-derived display coordinates, and a level is complete only when every
original 8/13 marker cell is paired with exactly one same-colored marker cell
from another visible sprite. The adapter SHALL replay the prior L1 seed and the
derived L2 delta, then the experiment SHALL update `ops/arc_solve_registry.yaml`
only if the reproduction gate advances beyond the prior `cn04` registry level.

Experiment 4570 SHALL write
`results/experiment_4570_levelup_attempt.json` with bare top-level fields for
`honest_verdict`, `inference_substrate`, `offline_reproduced`,
`reproduced_levels`, `target_game`, `registry_updated`, `random_seed`,
`reproducibility_checksum`, `preconditions_checked`, `reproduction_gate`,
`solution_labels`, `dead_ends`, and `registry_update`. Required field
principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: <game>_L<n>_offline_reproduced OR complete: <game>_delta_identified_no_bank (honest progress, not a bank)."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade solve, no headline LLM load (1s floor); declared so a fast real solve is not DURATION_TOO_SHORT false-flagged."
- `offline_reproduced`: principle "a solve not reproducible offline is wasted effort -- only reproduced levels count toward reproducible_total_levels."
- `reproduced_levels`: principle "the integer new-level count banked this task (>=1 satisfies the level-up guarantee on an attempted bank)."
- `target_game`: principle "the rotated game attempted -- traceable to the rotation discipline (not a recently-deepened game)."
- `registry_updated`: principle "the per-game win-condition/action-model/gotchas/dead-ends persisted so the next attempt reuses, not re-derives."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

A success artifact SHALL set `target_game=cn04`,
`honest_verdict=success: cn04_L2_offline_reproduced`,
`offline_reproduced=true`, `reproduced_levels >= 1`,
`registry_updated=true`, and SHALL increment `reproducible_total_levels`
monotonically beyond 52. If the L2 gate does not reproduce, the artifact SHALL
use `complete: cn04_delta_identified_no_bank`, set `reproduced_levels=0`,
record the stalled approach in `dead_ends`, and SHALL NOT increment the
registry count.

### REQ-ARC-WMTE-4559: Hidden-Field State Key Probe

Experiment 4559 SHALL extend the standing ARC offline search state key for
`ka59`, `ar25`, and `ft09` from a visible grid-only key to a registered
`(grid, hidden_fields)` key. The added hidden fields SHALL be read from the
current offline environment internals at each state-key call, not hardcoded
from registry constants: `ka59` SHALL include the live StepCounter HUD counter,
`ar25` SHALL include the hidden ACTION7 undo-stack depth, and `ft09` SHALL
include the local color-cycle register/order and current color-cycle phases.

Experiment 4559 SHALL run the offline arcade precondition, attempt the L2
deepening targets through the standing `OfflineSolver`/adapter route, and gate
any claimed new level with `arc_solver_kit.reproduce()`. A no-bank null is valid
only if a positive control demonstrates that the extended state key
disambiguates at least one pair of states that the grid-only key aliases;
otherwise the artifact SHALL report the specific unreadable register as a
sharpened `GAP-ARCH-GRID-ONLY-STATE` blocker.

Experiment 4559 SHALL write
`results/experiment_4559_hidden_field_state_probe.json` with bare top-level
fields for `honest_verdict`, `inference_substrate`, `hidden_fields_added`,
`state_disambiguation_control_passed`, `false_negative_risk_checked`,
`offline_reproduced`, `reproduced_levels`, `missing_verifier_gaps`,
`registry_updated`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`.

Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: hidden_field_state_<game>_L2_offline_reproduced OR complete: hidden_field_state_gap_sharpened_no_bank_honest_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade solve with the extended state hash, no headline LLM load."
- `hidden_fields_added`: principle "the HUD/hidden registers added to the state_key per game (read from internal state) -- traceable to GAP-ARCH-GRID-ONLY-STATE."
- `state_disambiguation_control_passed`: principle "the POSITIVE CONTROL -- the extended state_key disambiguates a pair the grid-only key aliased; guards a no-op change; a no-bank null is valid only if this passed."
- `false_negative_risk_checked`: principle "a no-bank null is valid only if the disambiguation positive control passed."
- `offline_reproduced`: principle "only offline-reproduced new levels count toward reproducible_total_levels."
- `reproduced_levels`: principle "the integer new-level count banked this task."
- `missing_verifier_gaps`: principle "if no bank, which specific register the search still cannot read -- the sharpened GAP-ARCH-GRID-ONLY-STATE entry."
- `registry_updated`: principle "the per-game hidden-field findings + dead-ends persisted so the next attempt reuses."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

### SCENARIO-ARC-WMTE-4559: Hidden Registers Split Grid-Aliased States

Given the offline arcade precondition passes and the `ka59`, `ar25`, and `ft09`
adapters are registered
When experiment 4559 computes a grid-only key and an extended adapter state key
for same-frame state pairs that differ only in StepCounter, undo-stack depth, or
color-cycle register state
Then the grid-only keys are equal, the extended state keys differ, and a no-bank
artifact sets `state_disambiguation_control_passed=true`,
`false_negative_risk_checked=true`, and records the remaining unreadable
register gap if no L2 reproduction gate passes.

### REQ-ARC-WMTE-4571: ka59 StepCounter State-Key Probe

Experiment 4571 SHALL re-scope the hidden-field state-key probe to `ka59` only.
The `ka59` adapter state key SHALL include the live bottom-row StepCounter HUD
register read from the offline environment internals at each state-key call:
`game.urgssjskot.current_steps` for the current counter and
`game.urgssjskot.koyyeuyzyr` or `current_level.get_data('StepCounter')` for the
limit. These values SHALL NOT be hardcoded from the registry.

Experiment 4571 SHALL run the offline arcade precondition and then execute the
positive control before any L2 deepening attempt. The positive control SHALL
demonstrate that the extended `ka59` adapter key disambiguates at least one pair
of same-grid states that the grid-only key aliases and whose StepCounter
registers differ. If the control fails, the experiment SHALL write the terminal
artifact with `state_disambiguation_control_passed=false`, record the unreadable
register in `missing_verifier_gaps`, and SHALL NOT run or claim a bank. If the
control passes, the experiment SHALL attempt `ka59` L2 through the standing
`OfflineSolver`/adapter route and SHALL accept a new level only when
`arc_solver_kit.reproduce()` offline-reproduces the claimed L2.

Experiment 4571 SHALL write
`results/experiment_4571_hidden_field_state_probe_ka59.json` with bare
top-level fields for `honest_verdict`, `inference_substrate`,
`hidden_fields_added`, `state_disambiguation_control_passed`,
`false_negative_risk_checked`, `offline_reproduced`, `reproduced_levels`,
`missing_verifier_gaps`, `registry_updated`, `random_seed`,
`reproducibility_checksum`, and `preconditions_checked`.

Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: hidden_field_state_ka59_L2_offline_reproduced OR complete: hidden_field_state_ka59_gap_sharpened_no_bank_honest_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade solve with the extended state hash, no headline LLM load (1s floor)."
- `hidden_fields_added`: principle "the ka59 StepCounter register added to the state_key (read from internal state) -- traceable to GAP-ARCH-GRID-ONLY-STATE."
- `state_disambiguation_control_passed`: principle "THE GATE-FIRST POSITIVE CONTROL -- the extended state_key disambiguates a pair the grid-only key aliased; a no-bank null is valid only if this passed (else the change is a no-op)."
- `false_negative_risk_checked`: principle "a no-bank null is valid only if the disambiguation positive control passed."
- `offline_reproduced`: principle "only offline-reproduced new levels count toward reproducible_total_levels."
- `reproduced_levels`: principle "the integer new-level count banked this task."
- `missing_verifier_gaps`: principle "if no bank, which specific register the search still cannot read -- the sharpened GAP-ARCH-GRID-ONLY-STATE entry."
- `registry_updated`: principle "the ka59 hidden-field findings + dead-ends persisted so the next attempt reuses."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

### SCENARIO-ARC-WMTE-4571: ka59 StepCounter Splits Grid-Aliased States

Given the offline arcade precondition passes and the `ka59` adapter is registered
When experiment 4571 compares a grid-only key with the extended adapter state
key for a same-grid `ka59` state pair whose `StepCounter.current_steps` values
differ
Then the grid-only keys are equal, the extended state keys differ,
`state_disambiguation_control_passed=true`, `false_negative_risk_checked=true`,
and any no-bank terminal artifact records the remaining `ka59` L2 register gap
without incrementing `reproducible_total_levels`.

### REQ-ARC-WMTE-4572: .422 Submitted Two-Metric Integration Gate

Experiment 4572 SHALL read
`results/experiment_4568_clickability_action_effect_predictor.json`,
`results/experiment_4569_verifier_guided_expansion.json`, and
`results/experiment_4571_hidden_field_state_probe_ka59.json` before changing the
submitted agent. It SHALL select only levers that raised a real metric without a
CORE-game level loss: A1 only when median actions-to-first-levelup is strictly
lower than the blind baseline with confidence support and a passed positive
control; A2 only when generic transfer is strictly above `0.04` with confidence
support and the random-priority control passes; A4 only when a new level bank is
offline reproduced. Flagged-adversarial upstream artifacts SHALL be excluded
unless their only flag is an explicit null-delta TAUTOLOGY carve-out with a
null-delta methodology note.

If no upstream lever passes, experiment 4572 SHALL leave
`SUBMITTED_AGENT_CONFIG` and `SUBMITTED_TARGET_LEVELS` unchanged, remeasure the
submitted/bare explorer action metric and generic-transfer metric using the same
baseline method, and write an honest null. If at least one lever passes, the
workflow SHALL wire only those winners into the submitted config, rerun both
metrics, preserve every CORE solve by set containment, check additivity against
the naive isolated A1/A2 deltas, and set operator-submit readiness only for a
CORE-preserved integrated metric lift. The workflow SHALL never submit to the
leaderboard.

Experiment 4572 SHALL write `results/experiment_4572_integration_gate.json` with
bare top-level fields for `honest_verdict`, `inference_substrate`,
`median_actions_to_first_levelup_integrated`, `generic_transfer_rate_integrated`,
`levers_integrated`, `additivity_checked`, `core_solves_preserved`,
`heldout_solve_rate`, `ready_for_operator_submit`,
`false_negative_risk_checked`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`.

Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: integrated_actions_to_levelup_below_blind_or_generic_transfer_above_0.04 OR complete: no_lever_raises_a_metric_honest_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade end-to-end (if the integrated run invokes the LLM proposer, declare live_llm_inference + add the Qwen3.5-9B-MTP model precondition); a CNN predictor forward pass is NOT live_llm_inference."
- `median_actions_to_first_levelup_integrated`: principle "the SUBMITTED-config action efficiency after wiring winners vs the blind baseline -- the leaderboard scoring lever."
- `generic_transfer_rate_integrated`: principle "the SUBMITTED-config held-out variant transfer after wiring winners vs baseline 0.04 -- the leaderboard-honest HEADLINE."
- `levers_integrated`: principle "names which of A1/A2/A4 were wired -- traceable to their measured deltas; [] is an honest null."
- `additivity_checked`: principle "integrated metric vs the naive sum of isolated A1/A2 deltas -- surfaces a destructive interaction instead of burying it."
- `core_solves_preserved`: principle "integration must preserve every CORE solve (set-containment); a dropped solve FAILS the lever."
- `heldout_solve_rate`: principle "the real transfer signal; integration should not regress it."
- `ready_for_operator_submit`: principle "True if the integrated config is a CORE-preserved improvement on either metric worth a 1/day submission slot; the task NEVER submits (operator-only)."
- `false_negative_risk_checked`: principle "an honest null is only valid with both baselines measured the same way."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

### SCENARIO-ARC-WMTE-4572: Null Levers Keep Submitted Config Bare

Given A1 reports no strict action-efficiency gain, A2 fails the random-priority
control or fails to raise generic transfer above `0.04`, and A4 reproduces no
new bank
When experiment 4572 selects integration levers and reruns both action and
generic-transfer measurements
Then `levers_integrated=[]`, `ready_for_operator_submit=false`, the submitted
configuration remains unchanged, `additivity_checked` reports the integrated
deltas against the isolated A1/A2 deltas, `core_solves_preserved=true`, and the
terminal artifact starts with `complete: no_lever_raises_a_metric_honest_null`.

### REQ-ARC-WMTE-4573: Persist A1/A2 Winning Primitive With Leave-One-Game Transfer

Experiment 4573 SHALL read
`results/experiment_4568_clickability_action_effect_predictor.json` and
`results/experiment_4569_verifier_guided_expansion.json`, select the strongest
measured primitive signal, persist it as a reusable solver-kit operator, and
record that reusable asset in `ops/arc_solve_registry.yaml` under
`general_gotchas` as `primitive_persistent_action_effect_memory_operator`.
If A1 and A2 are both transfer nulls, the workflow SHALL
persist the best-characterized primitive-as-built, report the transfer null, and
state what evidence or candidate-generation capability would be needed for value.

For the A1 path, the solver kit SHALL expose
`persistent_action_effect_memory_operator` backed by `PersistentAEM`. The
operator SHALL build a cross-game action-effect memory from cached frame/action
rows while excluding each transfer target game from its memory, rank compatible
action candidates by predicted effect with deterministic original-order
tie-breaking, and report actions-to-first-levelup before and after ranking. The
operator SHALL be registered in `primitive_operator_registry()`, selected by
`select_primitive_operators()`, and surfaced by
`arc_solve_learning.recommend_approach()` so live solves can reuse it before
game-specific reverse engineering.

Experiment 4573 SHALL apply the persisted primitive to at least two games that
were not used to build that game's memory and SHALL write
`results/experiment_4573_primitive_persist_transfer.json` with bare top-level
fields for `honest_verdict`, `inference_substrate`, `primitive_persisted`,
`transfer_games`, `transfer_value_per_game`, `offline_reproduced`,
`registry_updated`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`. Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: primitive_persisted_transfer_<game>_value_added OR complete: primitive_persisted_transfer_null_characterized."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates for the offline transfer; a CNN predictor forward pass is NOT live_llm_inference."
- `primitive_persisted`: principle "names the arc_solver_kit operator + registry general_gotcha id added/extended -- the reusable asset (Solver-Reuse Discipline); without it the A1/A2 effort is wasted per the ARC reuse rule."
- `transfer_games`: principle "the games the primitive was applied to (NOT tuned on) -- the generalization test."
- `transfer_value_per_game`: principle "the per-game value-add (predictor actions-reduced / expansion winner-generated) -- the cross-game evidence the primitive generalizes."
- `offline_reproduced`: principle "only offline-reproduced new levels count toward reproducible_total_levels."
- `registry_updated`: principle "the primitive + transfer dead-ends persisted so the next milestone reuses, not re-derives."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

If no held-out transfer game shows positive action reduction or winner
generation, the artifact SHALL report
`complete: primitive_persisted_transfer_null_characterized`, keep
`offline_reproduced=false` unless a new level passed `arc_solver_kit.reproduce`,
and record the transfer dead-ends in the registry rather than incrementing
`reproducible_total_levels`.

### SCENARIO-ARC-WMTE-4573: Action-Effect Memory Transfers Without Target-Game Tuning

Given the offline arcade precondition passes and the experiment 4568/4569
artifacts are present
When experiment 4573 selects the A1 action-effect predictor as the
best-characterized primitive and applies `persistent_action_effect_memory_operator`
to cached transfer games with each target game excluded from the memory
Then the artifact records the persisted operator, transfer games, per-game
actions reduced or null, offline reproduction status, registry update status,
and a terminal honest verdict without incrementing `reproducible_total_levels`
unless a new level passes `arc_solver_kit.reproduce()`.

### REQ-ARC-WMTE-4587: Offline ARC Methodology Descriptor Guard

Experiment 4587 SHALL harden `scripts/adversarial_verify.py` so an offline ARC
artifact is not `METHODOLOGY_MISSING`-warned merely because it has no
`model_specs`. The verifier SHALL recognize an offline ARC methodology
descriptor when an artifact declares
`inference_substrate=verifier_ensemble_against_cached_candidates` or
`inference_substrate=aggregation_from_upstream_artifacts`, carries a
`reproducibility_checksum`, and cites offline methodology evidence such as
`solver_module`, a persisted solver-kit operator (`primitive_persisted`),
`offline_reproduced`, `reproduction_gate` / `reproduce_gate`, or
`verifier_checkpoint`. Such artifacts use the offline
solver/reproduction/checkpoint descriptor as their methodology rather than a
live model specification.

The guard SHALL NOT weaken live-model verification: an artifact declaring
`inference_substrate=live_llm_inference` and missing
`model_specs` / `target_model` SHALL still emit `METHODOLOGY_MISSING`.

Experiment 4587 SHALL write
`results/experiment_4587_offline_arc_methodology_guard.json` with bare
top-level fields for `honest_verdict`, `inference_substrate`,
`guard_mechanism`, `offline_arc_artifact_not_warned`,
`real_llm_still_warned`, `tests_added_pass`, and `preconditions_checked`.
Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; shipped: offline_arc_methodology_guard_added OR complete: offline_arc_methodology_guard_partial_<reason>."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- runs the guard against fixtures, no model load (1s floor)."
- `guard_mechanism`: principle "names the recognized offline-arc methodology descriptor + where it suppresses the warn -- the fix that stops every ARC artifact false-warning."
- `offline_arc_artifact_not_warned`: principle "an offline-arc fixture (substrate + cited solver/checksum, no model_specs) is NOT METHODOLOGY_MISSING-warned -- the recurring-warn fix."
- `real_llm_still_warned`: principle "a live_llm_inference fixture missing model_specs IS still warned -- guards against weakening the real methodology check."
- `tests_added_pass`: principle "Tests Must Run and Assert -- both the not-warned-on-offline-arc and still-warned-on-real-LLM cases."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

### SCENARIO-ARC-WMTE-4587: Offline ARC Descriptor Suppresses Only The False Warn

Given an offline ARC fixture with
`inference_substrate=verifier_ensemble_against_cached_candidates`, a cited
`solver_module`, a `reproducibility_checksum`, and no `model_specs`
When `adversarial_verify` checks methodology
Then the fixture does not receive `METHODOLOGY_MISSING`. Given a
`live_llm_inference` fixture with `random_seed` and `reproducibility_checksum`
but no `model_specs`, the same check still emits `METHODOLOGY_MISSING`.

### REQ-ARC-WMTE-4581: Rotated ar25 Self-Play Level-Up And Verifier Checkpoint

Experiment 4581 SHALL satisfy the ARC sprint level-up attempt guarantee through
the standing ARC self-play loop while avoiding the stalled `ka59` hidden-register
path and the recently deepened `.421-.422` `m0r0`/`cn04` targets. The workflow
SHALL first record the preferred rotated targets `sk48` and `ar25`, SHALL skip
recorded dead-end approaches, and SHALL accept a bank only when
`arc_solver_kit.reproduce()` offline-reproduces a target level above that game's
prior `ops/arc_solve_registry.yaml` row.

For `ar25`, the L2 delta SHALL be derived from the offline environment state:
after the reproduced L1 prefix, the selected vertical mirror moves left twice,
ACTION5 selects the reflected object, and ACTION2 moves the object down eight
steps so the vertical reflection covers the L2 fixed target cells. The
`ar25` `GameAdapter` SHALL expose that deterministic L1+L2 tail to
`scripts/arc_loop_solve.py`, and the experiment SHALL persist a learned-verifier
checkpoint at `models/arc_verifier_ar25.json` trained on the run's positive
steps-to-go trace and negative off-path/dead-end trace evidence.

Experiment 4581 SHALL write
`results/experiment_4581_levelup_selfplay.json` with bare top-level fields for
`honest_verdict`, `inference_substrate`, `offline_reproduced`,
`reproduced_levels`, `target_game`, `verifier_checkpoint_updated`,
`registry_updated`, `random_seed`, `reproducibility_checksum`,
`preconditions_checked`, `reproduction_gate`, `solution_labels`, `dead_ends`,
`registry_update`, and `verifier_delta`. Required field principles SHALL be
included for:

- `honest_verdict`: principle "terminal prefix; success: <game>_L<n>_offline_reproduced OR complete: <game>_delta_identified_no_bank (honest progress, not a bank)."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade solve + verifier training, no headline LLM load (1s floor); declared so a fast real solve is not DURATION_TOO_SHORT false-flagged."
- `offline_reproduced`: principle "a solve not reproducible offline is wasted effort -- only reproduced levels count toward reproducible_total_levels."
- `reproduced_levels`: principle "the integer new-level count banked this task (>=1 satisfies the level-up guarantee)."
- `target_game`: principle "the rotated game attempted -- traceable to the rotation discipline (not a game deepened in .419-.422)."
- `verifier_checkpoint_updated`: principle "the learned-verifier checkpoint trained on this run's pos/neg traces (the self-play self-improvement step the operator mandated every milestone)."
- `registry_updated`: principle "the per-game win-condition/action-model/gotchas/dead-ends persisted so the next attempt reuses, not re-derives."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

A success artifact SHALL set `honest_verdict` to
`success: ar25_L2_offline_reproduced`, `target_game=ar25`,
`offline_reproduced=true`, `reproduced_levels>=1`,
`verifier_checkpoint_updated=true`, and `registry_updated=true`. If the
standing loop repeats an existing level, routes to per-game reverse engineering,
or does not update the learned verifier checkpoint, the artifact SHALL use
`complete: <game>_delta_identified_no_bank`, set `reproduced_levels=0`, preserve
dead-end evidence, and SHALL NOT increment `reproducible_total_levels`.

### REQ-ARC-WMTE-4593: Rotated ft09 Self-Play Level-Up And Verifier Checkpoint

Experiment 4593 SHALL satisfy the ARC sprint level-up attempt guarantee through
the standing ARC self-play loop while avoiding the stalled `ka59` hidden-register
path, the recently deepened `.420-.423` `ar25`/`cn04`/`m0r0` targets, and the
recorded no-grounded-L3-delta `cd82`/`sp80`/`su15` approaches. The workflow SHALL
attempt the preferred `sk48` rotation first and record its routing-only dead-end
if it does not bank, then MAY rotate to another shallow L1 game. A bank SHALL be
accepted only when `arc_solver_kit.reproduce()` offline-reproduces a target level
above that game's prior `ops/arc_solve_registry.yaml` row.

For `ft09`, the L2 delta SHALL be derived from the offline environment's local
constraint state: after the reproduced four-click L1 prefix, the L2 `cwU=[9,12]`
constraint cells SHALL click the seven Hkx cells at display coordinates
`(22,16)`, `(22,24)`, `(38,24)`, `(22,32)`, `(38,32)`, `(30,48)`, and `(22,48)`
so every adjacent Hkx/NTi cell satisfies the `bsT` equal-vs-different predicate
before the level counter advances. The `ft09` `GameAdapter` SHALL expose that
deterministic L1+L2 tail to `scripts/arc_loop_solve.py`, and the experiment SHALL
persist a learned-verifier checkpoint at `models/arc_verifier_ft09.json` trained
on the run's positive steps-to-go trace and negative off-path/dead-end trace
evidence.

Experiment 4593 SHALL write
`results/experiment_4593_levelup_selfplay.json` with bare top-level fields for
`honest_verdict`, `inference_substrate`, `offline_reproduced`,
`reproduced_levels`, `target_game`, `verifier_checkpoint_updated`,
`registry_updated`, `random_seed`, `reproducibility_checksum`,
`preconditions_checked`, `reproduction_gate`, `solution_labels`, `dead_ends`,
`registry_update`, and `verifier_delta`. Required field principles SHALL be
included for:

- `honest_verdict`: principle "terminal prefix; success: <game>_L<n>_offline_reproduced OR complete: <game>_delta_identified_no_bank (honest progress, not a bank)."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade solve + verifier training, no headline LLM load (1s floor); declared so a fast real solve is not DURATION_TOO_SHORT false-flagged."
- `offline_reproduced`: principle "a solve not reproducible offline is wasted effort -- only reproduced levels count toward reproducible_total_levels."
- `reproduced_levels`: principle "the integer new-level count banked this task (>=1 satisfies the level-up guarantee)."
- `target_game`: principle "the rotated game attempted -- traceable to the rotation discipline (not a game deepened in .420-.423)."
- `verifier_checkpoint_updated`: principle "the learned-verifier checkpoint trained on this run's pos/neg traces (the self-play self-improvement step the operator mandated every milestone)."
- `registry_updated`: principle "the per-game win-condition/action-model/gotchas/dead-ends persisted so the next attempt reuses, not re-derives."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

A success artifact SHALL set `honest_verdict` to
`success: ft09_L2_offline_reproduced`, `target_game=ft09`,
`offline_reproduced=true`, `reproduced_levels>=1`,
`verifier_checkpoint_updated=true`, and `registry_updated=true`. If the standing
loop repeats an existing level, routes to per-game reverse engineering, or does
not update the learned verifier checkpoint, the artifact SHALL use
`complete: <game>_delta_identified_no_bank`, set `reproduced_levels=0`, preserve
dead-end evidence, and SHALL NOT increment `reproducible_total_levels`.

### REQ-ARC-WMTE-4526: Submitted Deeper-Level Integration Gate

Experiment 4526 SHALL read the A1 forward-walk artifact and the A2
reach-deeper-levels artifact, select only non-`flagged_adversarial` levers that
raise the CORE per-level efficiency above the canonical `2.0074` baseline by
reaching a deeper CORE level, and reject action-trimming-only levers as retired
score levers. If no A2 lever raises CORE efficiency without a CORE level loss,
the workflow SHALL leave `SUBMITTED_AGENT_CONFIG` unchanged and report an honest
null rather than wiring a median-action or stop-after-level-up surrogate.

Experiment 4526 SHALL write
`results/experiment_4526_integration_8game_gate.json` with bare top-level fields
for `honest_verdict`, `inference_substrate`, `core_efficiency_baseline`,
`core_efficiency_integrated`, `core_solves_preserved`, `levers_integrated`,
`additivity_checked`, `heldout_solve_rate`, `nav_diagnostics`,
`ready_for_operator_submit`, `false_negative_risk_checked`, `random_seed`,
`reproducibility_checksum`, and `preconditions_checked`. Required field
principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: integrated_core_efficiency_<n>_above_2.0074 OR complete: no_lever_raises_core_efficiency_honest_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade end-to-end via the per-level gate (no headline GGUF load unless the induction tier is invoked)."
- `core_efficiency_baseline`: principle "2.0074 -- the REAL per-level metric control (NOT median actions, retired as a score lever)."
- `core_efficiency_integrated`: principle "the HEADLINE -- the SUBMITTED-config per-level efficiency after wiring the winners (did it solve MORE levels)."
- `core_solves_preserved`: principle "integration must preserve every CORE solve (set-containment)."
- `levers_integrated`: principle "names which of A1/A2 were wired -- traceable to their measured deltas; [] is an honest null."
- `additivity_checked`: principle "integrated CORE median vs the naive sum of isolated A1+A2 deltas -- surfaces a destructive batch-expand x stop-after-levelup interaction instead of burying it."
- `heldout_solve_rate`: principle "the real transfer signal; integration should not regress it."
- `nav_diagnostics`: principle "reset_replay_steps end-to-end -- did the integrated config actually cut the replay tax."
- `ready_for_operator_submit`: principle "True if the integrated config is a CORE-preserved improvement worth a 1/day submission slot; the task NEVER submits (operator-only)."
- `false_negative_risk_checked`: principle "an honest null only valid with the 7760 baseline measured the same way."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

A success artifact SHALL require `core_efficiency_integrated > 2.0074`,
`core_solves_preserved=true`, at least one integrated A2 reach-deeper lever, no
CORE game losing a previously reached level, and `ready_for_operator_submit=true`.
Otherwise the artifact SHALL report
`complete: no_lever_raises_core_efficiency_honest_null`, keep
`levers_integrated=[]`, and SHALL NOT submit to the leaderboard.

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
explicit `efficiency_delta`, deepest level reached per CORE game for each lever,
a concrete L1 to L2 barrier diagnosis, and reports success only if a CORE game
reaches L2 without losing a CORE level and offline reproduction confirms the new
level. When no lever raises efficiency, it records the null-delta methodology
note.

### SCENARIO-ARC-WMTE-4533: Level-Up Re-Induces A New Goal Predicate

Given a live E3 policy has induced an L1 predicate and then observes
`frame.levels_completed` increase while `target_levels > 1`
When experiment 4533 enables per-level re-induction and sweeps target levels
Then the policy records a level-boundary induction event, clears stale predicate
state, attempts a fresh post-transition induction for the next level, reports
the level-conditioned predicate-change positive control, and only wires the
submitted config if the CORE per-level efficiency strictly exceeds `2.0074`
without losing a CORE solve.

### SCENARIO-ARC-WMTE-4534: Trust Energy Routes Same-Depth Frontier Toward Next Goal

Given experiment 4533 either supplies a re-induced `L_{n+1}` predicate or
honestly reports that no live predicate is available
When experiment 4534 enables trust-energy next-level routing on the eligible
frontier
Then depth remains the primary frontier key, lower trust energy wins only within
the depth-compatible candidate set, the artifact compares `lp85` and `sp80`
against a matched no-energy control, sets `verifier_is_oracle=false`, and a null
is accepted only when the positive-control energy separation check passes and
`energy_separation_auroc` is reported.

### SCENARIO-ARC-WMTE-4544: LLM Proposer Refines Post-Level Plans

Given a live E3 policy observes a level-up and has post-transition frames for
the next level
When experiment 4544 enables the Qwen3.5-9B-MTP proposer instead of the
offline-DSL-only control
Then the policy invokes the local GGUF generator, records separate GOAL and
DYNAMICS candidate metadata, selects an executable world model with the
oracle-distinct trust energy, retries at most three verifier-guided refinement
rounds, reports whether the LLM produced a reachable plan the offline DSL did
not, and wires the submitted configuration only when a CORE game reaches L2
with strict per-level efficiency improvement and no CORE solve loss.

### SCENARIO-ARC-WMTE-4536: A1/A2 Integration Preserves Honest Null

Given A1 reports a permitted flagged null-delta tautology and A2 is flagged
without that exception while reaching no deeper CORE level
When experiment 4536 selects integration levers and runs the 8-game per-level
submission gate
Then no lever is wired, `core_efficiency_integrated` is measured on the current
submitted default, CORE solve preservation is reported, and the artifact uses
`complete: no_lever_raises_core_efficiency_honest_null` unless the submitted
configuration strictly raises CORE per-level efficiency above `2.0074`.

### SCENARIO-ARC-WMTE-4548: A1/A4 Integration Keeps Honest Null

Given A1's live LLM-proposer artifact is flagged with more than the permitted
null-delta tautology and A4's CNN ranker artifact reports no CORE efficiency
increase
When experiment 4548 selects integration levers and runs the 8-game per-level
submission gate
Then no lever is wired, `core_efficiency_integrated` is measured on the current
submitted default, CORE solve preservation and held-out solve rate are reported,
and the artifact uses `complete: no_lever_raises_core_efficiency_honest_null`
unless the submitted configuration strictly raises CORE per-level efficiency
above `2.0074` with an accepted A1/A4 lever.

### SCENARIO-ARC-WMTE-4560: A1/A2/A4 Integration Reports The Honest Two-Metric Null

Given A1's verifier-router artifact is flagged with zero generic-transfer
delta, A2's executable proposer artifact has no CORE L2 reproduction, and A4's
hidden-field state probe reports no new bank
When experiment 4560 selects integration levers, runs the 8-game per-level gate,
and re-measures generic transfer over variants
Then no lever is wired, both `core_efficiency_integrated` and
`generic_transfer_rate_integrated` are reported against the `2.0074` and `0.04`
baselines, CORE solve preservation and held-out solve rate are reported, and
the artifact uses `complete: no_lever_raises_a_metric_honest_null` unless the
integrated SUBMITTED configuration strictly raises either metric with an
accepted A1/A2/A4 lever.

### SCENARIO-ARC-WMTE-4561: Verifier-Router Primitive Persists And Transfers Honestly

Given A1's verifier-router artifact has the strongest measured signal among
the current A1/A2 artifacts and the offline arcade precondition passes
When experiment 4561 persists
`verifier_router_candidate_ranking_operator` and applies it to at least two
non-tuned games
Then the artifact records the persisted operator and registry
`primitive_verifier_router_candidate_ranking_operator` gotcha, each held-out
game's ordering-gain value, any transfer dead-end, a stable checksum, and a
terminal success only when at least one held-out game has positive value-add.

### SCENARIO-ARC-WMTE-4525: Standing Loop Bank Persists One Reproduced Level

Given the offline arcade precondition passes and the standing loop writes a
target result whose reproduction gate reaches a level above the target's prior
registry level
When experiment 4525 builds the level-up artifact
Then it records the required field principles, computes a stable
`reproducibility_checksum`, updates the registry entry and total level count,
records dead-end attempts, and reports success only when the offline
reproduction gate is true and at least one new level was banked.

### SCENARIO-ARC-WMTE-4535: sp80 L2 Splitter Placement Banks Offline

Given the offline arcade precondition passes, `sp80` has a prior reproduced L1
registry row, and the `sp80` adapter derives the L2 spill-splitter placement
delta from the offline environment state
When experiment 4535 runs the standing loop to target level 2 and builds the
level-up artifact
Then the full L1+L2 label plan is replayed through `arc_solver_kit.reproduce`,
the terminal artifact records the required field principles, computes a stable
`reproducibility_checksum`, updates the `sp80` registry row and total level
count, records dead-end attempts, and reports success only when the offline
reproduction gate reaches level 2 with `offline_reproduced=true`.

### SCENARIO-ARC-WMTE-4546: su15 Adapter Banks L2 Offline

Given the prior reproduced `su15` L1 trajectory and offline L2 state exposing
eight level-0 fruit pieces plus a level-3 target zone
When experiment 4546 registers the `su15` `GameAdapter`, replays the standing
loop to target level 2, and runs `arc_solver_kit.reproduce`
Then the full L1+L2 label plan is replayed through the offline reproduction
gate, the terminal artifact records `target_game=su15`,
`offline_reproduced=true`, `reproduced_levels>=1`,
`registry_updated=true`, and a stable `reproducibility_checksum`, and success
is not reported unless the registry bank advances beyond the prior L1 row.

### SCENARIO-ARC-WMTE-4558: Rotated Attempt Reports Bank Only Past Registry Level

Given the offline arcade precondition passes, the registry records
`reproducible_total_levels: 52`, and rotated standing-loop artifacts include
candidate games whose reproduction gates either route to per-game RE or replay
only an already registered level
When experiment 4558 builds the terminal level-up artifact
Then the artifact records the required field principles, computes a stable
`reproducibility_checksum`, preserves dead-end evidence for the stalled
rotation targets, reports `success: <game>_L<n>_offline_reproduced` only if a
candidate's reproduced `reached_level` exceeds its prior registry level, and
otherwise reports `complete: <game>_delta_identified_no_bank` with
`reproduced_levels=0`, `registry_updated=false`, and no increment to
`reproducible_total_levels`.

### SCENARIO-ARC-WMTE-4570: cn04 Adapter Banks L2 Offline

Given the offline arcade precondition passes, the registry records `cn04` as a
reproduced L1 game, and the rotated target selection skips the stalled `ar25`
hidden-state path and the already banked `m0r0` L2 path
When experiment 4570 registers the `cn04` marker-pair `GameAdapter`, runs the
standing loop, and replays the full L1+L2 label plan through
`arc_solver_kit.reproduce`
Then the artifact records `target_game=cn04`, the required field principles, a
stable `reproducibility_checksum`, `offline_reproduced=true`,
`reproduced_levels>=1`, `registry_updated=true`, the `cn04` registry row is
persisted with the L2 win/action/state delta and dead-end evidence, and
`reproducible_total_levels` increments monotonically beyond 52.

### SCENARIO-ARC-WMTE-4581: ar25 Self-Play Banks L2 And Updates Verifier

Given the offline arcade precondition passes, `sk48` remains routing-only,
`ka59` is excluded by its recorded hidden-register dead end, and the registry
records `ar25` as reproduced only through L1
When experiment 4581 runs the standing loop with the `ar25` L2 mirror/object
delta, replays the full L1+L2 label plan through `arc_solver_kit.reproduce`,
and writes the learned verifier checkpoint
Then the artifact records `target_game=ar25`, the required field principles, a
stable `reproducibility_checksum`, `offline_reproduced=true`,
`reproduced_levels>=1`, `verifier_checkpoint_updated=true`,
`registry_updated=true`, the `ar25` registry row persists the L2 win/action/state
delta plus dead-end evidence, and `reproducible_total_levels` increments
monotonically beyond the prior total.

### SCENARIO-ARC-WMTE-4593: ft09 Self-Play Banks L2 And Updates Verifier

Given the offline arcade precondition passes, `sk48` remains routing-only,
`ka59` is excluded by its recorded hidden-register dead end, and the registry
records `ft09` as reproduced only through L1
When experiment 4593 runs the standing loop with the `ft09` L2 local-constraint
color-cycle delta, replays the full L1+L2 click plan through
`arc_solver_kit.reproduce`, and writes the learned verifier checkpoint
Then the artifact records `target_game=ft09`, the required field principles, a
stable `reproducibility_checksum`, `offline_reproduced=true`,
`reproduced_levels>=1`, `verifier_checkpoint_updated=true`,
`registry_updated=true`, the `ft09` registry row persists the L2
win/action/state delta plus dead-end evidence, and `reproducible_total_levels`
increments monotonically beyond the prior total.

### SCENARIO-ARC-WMTE-4526: Integration Wires Only Deeper-Level Winners

Given the A1 forward-walk artifact, the A2 reach-deeper-levels artifact, and the
canonical per-level baseline are present
When experiment 4526 selects levers and runs the local submission gate
Then it integrates only A2 levers that raise CORE per-level efficiency by
reaching a deeper CORE level, rejects retired action-trimming levers, records
the 8-game gate measurement and held-out solve rate, and reports an honest null
when no lever beats `2.0074`.

### REQ-ARC-WMTE-4613: World-Model Trust Literature Ingestion For .426

Experiment 4613 SHALL synthesize the focused 2026-06-23 SOTA ingestion for the
world-model trust-energy and scored-agent verifier-integration track. The
workflow SHALL read the prior Exp 4601 ingestion, `research-studying.md`,
`research-references.md`, and the 2026-06-11 search-layer note, SHALL use the
reliable channel (`scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`,
and low-concurrency WebSearch/WebFetch), and SHALL NOT invoke `/deep-research`.
If the required network reachability precondition fails, the workflow SHALL
write an honest blocked-network verdict instead of fabricating citations.

Experiment 4613 SHALL write
`docs/research-notes/world-model-trust-literature-2026-06-23.md` and
`results/experiment_4613_sota_ingestion_world_model_trust.json`. The artifact
SHALL include `honest_verdict`, `inference_substrate`, `methods_mapped`,
`flagged_for_next_roadmap`, `note_path`, `deep_research_not_used`,
`preconditions_checked`, `citations_verified`, `random_seed`, and
`field_principles`. The success verdict SHALL be
`success: sota_ingestion_world_model_trust_mapped`, the substrate SHALL be
`aggregation_from_upstream_artifacts`, and `deep_research_not_used` SHALL be
true. Each mapped method SHALL cite real arXiv IDs and SHALL include the
implementation cost over the current A1 trust-energy / A2 live-integration
stack plus a concrete `fails_when` condition.

Required field principles SHALL include:

- `honest_verdict`: principle `terminal prefix; success: sota_ingestion_world_model_trust_mapped.`
- `inference_substrate`: principle `aggregation_from_upstream_artifacts -- literature read + synthesis, no model load (100us floor).`
- `methods_mapped`: principle `the strongest 3-5 SOTA methods with REAL arXiv IDs + per-method implement-cost-over-current-stack + fails_when -- the actionable ingestion (no citation = fabrication).`
- `flagged_for_next_roadmap`: principle `the strongest method(s) flagged as candidate .426 inputs -- closes discover->ingest->plan->experiment.`
- `note_path`: principle `docs/research-notes/world-model-trust-literature-2026-06-23.md -- the per-track note (the SOTA-Ingestion Cycle deliverable).`
- `deep_research_not_used`: principle `MUST be true -- /deep-research is banned in the autonomous loop; used sweep helpers + low-concurrency WebSearch/WebFetch.`
- `preconditions_checked`: principle `records network reachability verified; pre-empts fabricated citations.`

### SCENARIO-ARC-WMTE-4613: SOTA Methods Map Onto A1/A2 And Flag .426 Inputs

Given the network precondition succeeds and the prior .424/.425 corpus is
readable
When experiment 4613 ingests the focused papers for executable world models,
verifier scaling, closed-loop world-model evaluation, world-model policy
optimization, learned-heuristic search, learned pruning, and goal-conditioned
value functions
Then the note and JSON artifact map three to five strongest methods onto the
current A1 trust-energy / A2 scored-agent integration stack, cite real arXiv IDs
including `2605.05138`, `2502.01989`, `2510.18135`, `2511.09515`,
`2102.04518`, `2406.04935`, `2206.03023`, and `2502.20379`, flag one or more
`.426` roadmap candidates, record all network and citation preconditions
checked, and record that `/deep-research` was not used.

### REQ-ARC-WMTE-4616: Offline-To-Live Bridge Cause Disambiguation

Experiment 4616 SHALL diagnose why the learned ARC value head transfers
offline but regresses the live depth-first explorer. The runner SHALL first
check that `arc_solver_kit.offline_arcade()` and the `LearnedVerifier`,
`cross_game_features_v3`, and `collect_trajectory_data` imports are available.
If either precondition fails, it SHALL write a blocked artifact with
`honest_verdict` beginning `blocked_`, include `preconditions_checked`, and
SHALL NOT fabricate bridge evidence.

The experiment SHALL evaluate three controlled arms against the matched bare
BFS control and the value-head-as-is control over the diagnostic corpus where
the value head has an offline routing win: compute-cost, distribution-shift,
and calibration. The compute-cost arm SHALL compare value-head solves or
first-win at equal node budget versus equal wall-clock budget. The
distribution-shift arm SHALL compare the value head's AUROC on winning-path
states against off-path frontier states that the live search expands. The
calibration arm SHALL measure rank-to-true-steps monotonicity and whether
post-hoc isotonic/Platt-style calibration alone changes the live routing. A
diagnosis is valid only when the bare BFS control ran and the offline value-head
win is confirmed.

Experiment 4616 SHALL write
`results/experiment_4616_offline_live_bridge_disambiguation.json` with
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `verifier_is_oracle`, `binding_bridge_cause`,
`compute_cost_evidence`, `distribution_shift_evidence`,
`calibration_evidence`, `indicated_fix`, `offline_win_confirmed`,
`positive_control_passed`, `false_negative_risk_checked`,
`per_game_variance`, `residual_bridge_gaps`, `random_seed`,
`reproducibility_checksum`, and `preconditions_checked`. The artifact SHALL
set `verifier_is_oracle=false` because the SpatialValueNet/value head is a
learned ranking signal, oracle-distinct from the executable win-check.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success: bridge_cause_isolated_<compute|shift|calibration>_fix_identified OR complete: bridge_cause_inseparable_multi_cause_honest_residual_logged."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline value-head scoring + live-search arms over cached transitions (1s floor); if an LLM arm runs, declare live_llm_inference + the Qwen3.5-9B-MTP iGPU precondition."
- `verifier_is_oracle`: principle "MUST be false -- the SpatialValueNet is a learned value ranking states, oracle-DISTINCT from running the executable win-check."
- `binding_bridge_cause`: principle "which of compute_cost, distribution_shift, calibration binds, or inseparable_multi_cause."
- `compute_cost_evidence`: principle "value-head solves/first-win at equal node budget vs equal wall-clock."
- `distribution_shift_evidence`: principle "AUROC on winning-path states versus live off-path frontier states."
- `calibration_evidence`: principle "rank-to-cost monotonicity plus whether recalibration changes routing."
- `indicated_fix`: principle "decision-point-only eval/cached features, DAgger search-distribution retraining/bounded-pruning, or isotonic calibration."
- `offline_win_confirmed`: principle "positive control that the value head wins offline on the diagnostic corpus."
- `positive_control_passed`: principle "bare BFS matched control ran and the offline win exists."
- `false_negative_risk_checked`: principle "true with matched control plus offline-win confirmation."
- `per_game_variance`: principle "LOO-AUROC spread across games, including whether the bridge cause is uniform."
- `residual_bridge_gaps`: principle "Missing-Verifier / bridge gaps logged for any cause not isolated."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent harness/corpus drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4616-BRIDGE-CAUSE

**Given** the offline arcade and value learner imports are available, the
diagnostic corpus confirms an offline value-head routing win, and the matched
bare BFS control runs
**When** experiment 4616 evaluates compute-cost, distribution-shift, and
calibration arms
**Then** it reports the binding bridge cause when exactly one arm crosses its
diagnostic threshold, otherwise reports `inseparable_multi_cause`, records the
corresponding indicated fix, keeps `verifier_is_oracle=false`, and emits a
stable checksum.

### REQ-ARC-WMTE-4630: Rotated Clean-Nav Self-Play Level Bank

Experiment 4630 SHALL run the standing ARC self-play loop on the next eligible
clean-navigation L1 game, with `ls20` as the preferred target. The workflow SHALL
verify `arc_solver_kit.offline_arcade()`, inspect the ARC solve registry for
dead-end skips, avoid the recently failed or hidden-state-bound targets `sk48`,
`dc22`, `ka59`, and `wa30`, and deepen only a game whose registry row is a
reproduced L1. The `ls20` deepen SHALL register a reusable `GameAdapter` that
replays the L1 seed plus the L2 delta through `scripts/arc_loop_solve.py`, runs
the `arc_solver_kit.reproduce()` gate, and trains/checkpoints
`models/arc_verifier_ls20.json` from the positive steps-to-go trace and recorded
off-path/dead-end notes.

Experiment 4630 SHALL write
`results/experiment_4630_levelup_selfplay.json` with bare top-level fields for
`honest_verdict`, `inference_substrate`, `verifier_is_oracle`,
`solve_provenance`, `offline_reproduced`, `reproduced_levels`, `target_game`,
`verifier_checkpoint_updated`, `registry_updated`, `random_seed`,
`reproducibility_checksum`, and `preconditions_checked`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success: <game>_L<n>_offline_reproduced OR complete: <game>_delta_identified_no_bank (honest progress, not a bank)."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade solve + verifier training, no headline LLM load (1s floor); declared so a fast real solve is not DURATION_TOO_SHORT false-flagged."
- `verifier_is_oracle`: principle "MUST be false -- the learned verifier ranks/routes the search, oracle-distinct from the executable reproduction win-check."
- `solve_provenance`: principle "development_proxy -- the offline dev twin (arc_loop_solve + a hand GameAdapter); honest that this is the registry/dev proxy, not live-agent self-discovery on a hidden game."
- `offline_reproduced`: principle "a solve not reproducible offline is wasted effort -- only reproduced levels count toward reproducible_total_levels."
- `reproduced_levels`: principle "the integer new-level count banked this task (>=1 satisfies the level-up guarantee)."
- `target_game`: principle "the rotated game attempted -- traceable to the rotation discipline (a clean-nav game not deepened in .421-.426, not sk48/dc22/ka59/wa30)."
- `verifier_checkpoint_updated`: principle "the learned-verifier checkpoint trained on this run's pos/neg traces (the self-play self-improvement step the operator mandated every milestone)."
- `registry_updated`: principle "the per-game win-condition/action-model/gotchas/dead-ends persisted so the next attempt reuses, not re-derives."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4630

**Given** the offline ARC environment is importable, the registry records `ls20`
as reproduced through L1 with no blocking dead-end, and the skipped targets have
registry or rotation reasons
**When** experiment 4630 runs `scripts/arc_loop_solve.py --game ls20
--target-level 2`
**Then** the loop reaches L2 through offline reproduction, updates
`models/arc_verifier_ls20.json`, records the registry bank from total 55 to 56,
writes the required principle-annotated artifact, and reports
`verifier_is_oracle=false` with `solve_provenance=development_proxy`.

### REQ-ARC-WMTE-4642: Rotated Level-Up Self-Play Bank With Checkpoint

Experiment 4642 SHALL satisfy the ARC sprint level-up attempt guarantee by
running the standing ARC self-play loop, accepting a bank only when
`arc_solver_kit.reproduce()` offline-reproduces a target level above that game's
current `ops/arc_solve_registry.yaml` row and the run persists a learned-verifier
checkpoint. The workflow SHALL verify `arc_solver_kit.offline_arcade()` before
the solve, inspect registry dead-end evidence, skip `ls20`, `sk48`, `dc22`,
`ka59`, and `wa30`, and record why the preferred rotated L1 targets
`r11l`, `g50t`, and `bp35` plus the alternatives `m0r0` and `cn04` did not bank
in this milestone. If all preferred and requested alternatives fail to bank in a
bounded offline attempt, the experiment MAY use the first gateable adaptered
fallback only when the artifact records the rotation exception honestly.

For the fallback bank, `ft09` L3 SHALL extend the local-constraint color-cycle
adapter from the offline environment state. After the reproduced L1+L2 prefix,
the adapter SHALL click the L3 Hkx cells whose binary color assignment is forced
by the four visible `bsT` 3x3 predicates, using display coordinates derived from
the level camera transform. The standing loop SHALL replay this deterministic
L3 tail through `scripts/arc_loop_solve.py --game ft09 --target-level 3`, run
the reproduction gate, and checkpoint `models/arc_verifier_ft09.json` from the
positive steps-to-go trace and the recorded off-path/dead-end evidence.

Experiment 4642 SHALL write
`results/experiment_4642_levelup_selfplay.json` with bare top-level fields for
`honest_verdict`, `inference_substrate`, `verifier_is_oracle`,
`solve_provenance`, `offline_reproduced`, `reproduced_levels`, `target_game`,
`verifier_checkpoint_updated`, `registry_updated`, `random_seed`,
`reproducibility_checksum`, and `preconditions_checked`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success: <game>_L<n>_offline_reproduced OR complete: <game>_delta_identified_no_bank (honest progress, not a bank)."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade solve + verifier training, no headline LLM load (1s floor); declared so a fast real solve is not DURATION_TOO_SHORT false-flagged."
- `verifier_is_oracle`: principle "MUST be false -- the learned verifier ranks/routes the search, oracle-distinct from the executable reproduction win-check."
- `solve_provenance`: principle "development_proxy -- the offline dev twin (arc_loop_solve + a hand GameAdapter); honest that this is the registry/dev proxy, not live-agent self-discovery on a hidden game."
- `offline_reproduced`: principle "a solve not reproducible offline is wasted effort -- only reproduced levels count toward reproducible_total_levels."
- `reproduced_levels`: principle "the integer new-level count banked this task (>=1 satisfies the level-up guarantee)."
- `target_game`: principle "the rotated game attempted -- traceable to the rotation discipline (a clean game not deepened in .422-.427, not ls20/sk48/dc22/ka59/wa30)."
- `verifier_checkpoint_updated`: principle "the learned-verifier checkpoint trained on this run's pos/neg traces (the self-play self-improvement step the operator mandated every milestone)."
- `registry_updated`: principle "the per-game win-condition/action-model/gotchas/dead-ends persisted so the next attempt reuses, not re-derives."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4642

**Given** the offline ARC environment is importable, the registry records `ft09`
as reproduced through L2, the preferred clean L1 targets and requested
alternatives have bounded dead-end evidence for this milestone, and prohibited
targets remain skipped
**When** experiment 4642 runs the standing loop on `ft09` with target level 3
**Then** the loop reaches L3 through offline reproduction, updates
`models/arc_verifier_ft09.json`, records the registry bank from total 56 to 57,
writes the required principle-annotated artifact with an explicit fallback exception,
and reports `verifier_is_oracle=false` with
`solve_provenance=development_proxy`.

### REQ-ARC-WMTE-4654: Rotated Clean Level-Up Self-Play Bank

Experiment 4654 SHALL satisfy the ARC sprint level-up attempt guarantee by
running the standing ARC self-play loop on a rotation target that was not
deepened in attempts `.424` through `.428`, accepting a bank only when
`arc_solver_kit.reproduce()` offline-reproduces a level above the target game's
current `ops/arc_solve_registry.yaml` row and the run persists a learned-verifier
checkpoint. The workflow SHALL verify `arc_solver_kit.offline_arcade()` before
the solve, inspect registry dead-end evidence, skip `ft09`, `ls20`, `sk48`,
`dc22`, `ka59`, `wa30`, and the no-grounded-L3-delta rows `cd82`, `sp80`, and
`su15`, and record why the preferred hard L1 targets `bp35`, `re86`, and `sb26`
were skipped for this milestone.

For the clean bank, `vc33` L2 SHALL extend the support-clearance config adapter
from the offline environment state. After the reproduced L1 prefix
`lower_click * 3`, the adapter SHALL click the L2 support controls at display
coordinate `(1,25)` twice and `(1,45)` five times, using the level camera
display-to-grid transform, and SHALL replay the full path through
`scripts/arc_loop_solve.py --game vc33 --target-level 2`. The standing loop
SHALL run the reproduction gate and checkpoint `models/arc_verifier_vc33.json`
from the positive steps-to-go trace plus the recorded off-path/dead-end evidence.

Experiment 4654 SHALL write
`results/experiment_4654_levelup_selfplay.json` with bare top-level fields for
`honest_verdict`, `inference_substrate`, `verifier_is_oracle`,
`solve_provenance`, `offline_reproduced`, `reproduced_levels`, `target_game`,
`verifier_checkpoint_updated`, `registry_updated`, `random_seed`,
`reproducibility_checksum`, and `preconditions_checked`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success: <game>_L<n>_offline_reproduced OR complete: <game>_delta_identified_no_bank (honest progress, not a bank)."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade solve + verifier training, no headline LLM load (1s floor); declared so a fast real solve is not DURATION_TOO_SHORT false-flagged."
- `verifier_is_oracle`: principle "MUST be false -- the learned verifier ranks/routes the search, oracle-distinct from the executable reproduction win-check."
- `solve_provenance`: principle "development_proxy -- the offline dev twin (arc_loop_solve + a hand GameAdapter); honest that this is the registry/dev proxy, not live-agent self-discovery on a hidden game."
- `offline_reproduced`: principle "a solve not reproducible offline is wasted effort -- only reproduced levels count toward reproducible_total_levels."
- `reproduced_levels`: principle "the integer new-level count banked this task (>=1 satisfies the level-up guarantee)."
- `target_game`: principle "the rotated game attempted -- traceable to the rotation discipline (a clean game not deepened in .424-.428, not ft09/ls20/sk48/dc22/ka59/wa30)."
- `verifier_checkpoint_updated`: principle "the learned-verifier checkpoint trained on this run's pos/neg traces (the self-play self-improvement step the operator mandated every milestone)."
- `registry_updated`: principle "the per-game win-condition/action-model/gotchas/dead-ends persisted so the next attempt reuses, not re-derives."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4654

**Given** the offline ARC environment is importable, the registry records `vc33`
as reproduced through L1, the preferred hard L1 targets lack a grounded
next-level adapter for this milestone, and prohibited recent/dead-end targets
remain skipped
**When** experiment 4654 runs the standing loop on `vc33` with target level 2
**Then** the loop reaches L2 through offline reproduction, updates
`models/arc_verifier_vc33.json`, records the registry bank from total 57 to 58,
writes the required principle-annotated artifact without using an oracle verifier,
and reports `solve_provenance=development_proxy`.

### REQ-ARC-WMTE-4666: Level-Up Self-Play Bank With Verifier Checkpoint

Experiment 4666 SHALL satisfy the ARC sprint level-up attempt guarantee by
writing `results/experiment_4666_levelup_selfplay.json` from standing-loop
evidence that is accepted only when the executable offline reproduction gate
reports a level above the target game's current
`ops/arc_solve_registry.yaml` row and a learned-verifier checkpoint is present.
The workflow SHALL verify `arc_solver_kit.offline_arcade()` before the solve,
inspect registry dead-end evidence, skip stalled preferred approaches, and
record the target-selection ledger for `bp35`, `re86`, `sb26`, `m0r0`, `cn04`,
`lp85`, `tr87`, `tn36`, and `ar25`.

The preferred clean targets for the sprint are `bp35`, `re86`, and `sb26`
L1-to-L2. If those targets lack a grounded next-level adapter and the requested
alternatives fail to exceed their current registry depths, Experiment 4666 MAY
use an explicit fallback exception only when an existing standing-loop artifact
already satisfies the bank gate. The fallback exception SHALL be visible in the
artifact rather than silently rewriting the rotation policy. For this milestone,
the admissible bank evidence is `results/arc_loop_solve_dc22.json`, which SHALL
show `offline_reproduced=true`, `reproduced_levels=2`, a level increase over
the registry's dc22 L1 row, `solve_provenance=development_proxy`, and
`models/arc_verifier_dc22.json`.

Experiment 4666 SHALL write
`results/experiment_4666_levelup_selfplay.json` with bare top-level fields for
`honest_verdict`, `inference_substrate`, `verifier_is_oracle`,
`solve_provenance`, `offline_reproduced`, `reproduced_levels`, `target_game`,
`verifier_checkpoint_updated`, `registry_updated`, `random_seed`,
`reproducibility_checksum`, and `preconditions_checked`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success: <game>_L<n>_offline_reproduced OR complete: <game>_delta_identified_no_bank (honest progress, not a bank)."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade solve + verifier training, no headline LLM load (1s floor); declared so a fast real solve is not DURATION_TOO_SHORT false-flagged."
- `verifier_is_oracle`: principle "MUST be false -- the learned verifier ranks/routes the search, oracle-distinct from the executable reproduction win-check."
- `solve_provenance`: principle "development_proxy -- the offline dev twin (arc_loop_solve + a hand GameAdapter); honest that this is the registry/dev proxy, not live-agent self-discovery on a hidden game."
- `offline_reproduced`: principle "a solve not reproducible offline is wasted effort -- only reproduced levels count toward reproducible_total_levels."
- `reproduced_levels`: principle "the integer new-level count banked this task (>=1 satisfies the level-up guarantee)."
- `target_game`: principle "the rotated game attempted -- traceable to the rotation discipline (a clean game not deepened in .425-.429, not vc33/ft09/ls20/sk48/dc22/ka59/wa30)."
- `verifier_checkpoint_updated`: principle "the learned-verifier checkpoint trained on this run's pos/neg traces (the self-play self-improvement step the operator mandated every milestone)."
- `registry_updated`: principle "the per-game win-condition/action-model/gotchas/dead-ends persisted so the next attempt reuses, not re-derives."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4666

**Given** the offline ARC environment is importable, the registry records `dc22`
as reproduced through L1, preferred clean targets do not produce a grounded L2
bank in this milestone, and `results/arc_loop_solve_dc22.json` contains a
standing-loop L2 reproduction gate plus a learned-verifier checkpoint
**When** experiment 4666 builds its result artifact from that standing-loop
evidence
**Then** it records `success: dc22_L2_offline_reproduced`, marks
`target_selection.fallback_exception=true`, updates the dc22 registry row from
L1 to L2 and total 58 to 59, preserves `verifier_is_oracle=false`, and writes a
stable `reproducibility_checksum`.

#### SCENARIO-ARC-WMTE-4616-BLOCKED-PRECONDITION

**Given** the offline arcade or value-learner imports are unavailable
**When** experiment 4616 runs its precondition check
**Then** it writes a blocked artifact with `preconditions_checked`, a terminal
`blocked_` verdict, no fabricated bridge evidence, and a stable checksum.

### REQ-ARC-WMTE-4617: Graduate Spatial Value Head Into Live Path

Experiment 4617 SHALL graduate the position-preserving `SpatialValueNet` from
the experiment-only value-Q-head scripts into a live-path-reachable
`carnot.agentic` module. The graduated head SHALL keep a coarse 4x4 spatial map
before the scalar MLP head, SHALL expose the same `frame -> float` lower-is-better
contract as the existing `ValueNet`, and SHALL be loadable from live model
checkpoints without importing any `scripts/experiments/*` module.

The live integration SHALL replace the linear `LearnedVerifier` warm-start in
`scripts/arc_loop_solve.py` with the graduated spatial checkpoint loader and
SHALL make the same module reachable from `E3AgentPolicy`. The live scored path
SHALL apply the experiment-4616 compute-cost bridge fix by using cached,
decision-point-only/bounded value evaluation or an inert tie-breaker
configuration; it SHALL NOT ship the known-regressed heavy `value_weight=5`
A* mode. The `scripts/arc_orphan_solver_lint.py` live-path reachability gate
SHALL pass after the integration.

Experiment 4617 SHALL measure the graduated spatial head against the current
linear baseline and a bare BFS control on matched held-out public-game variants.
It SHALL report live first-win-rate, median actions-to-first-level-up, solve-rate,
and a paired bootstrap confidence interval on the first-win delta. A success
claim SHALL require the graduated arm to beat the linear baseline with confidence
or to reduce actions-to-first-level-up while preserving solve-rate and not
dropping below bare BFS. Otherwise it SHALL emit an honest null.

Experiment 4617 SHALL write
`results/experiment_4617_graduate_spatial_value_head_live.json` with
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `verifier_is_oracle`, `solve_provenance`,
`value_head_live_path_reachable`, `bridge_fix_applied`,
`first_win_rate_graduated`, `first_win_rate_linear_baseline`,
`first_win_rate_bare`, `first_win_delta`, `first_win_ci`,
`median_actions_to_first_levelup_graduated`, `actions_delta`,
`value_weight_used`, `parity_test_green`, `bare_and_linear_controls_passed`,
`false_negative_risk_checked`, `solve_rate_preserved`,
`chosen_submitted_config`, `offline_reproduced`, `random_seed`,
`reproducibility_checksum`, and `preconditions_checked`. When
`first_win_delta == 0`, the artifact SHALL include
`null_delta_methodology_note`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success: spatial_value_head_graduated_live_first_win_up_<n> OR complete: spatial_value_head_graduated_no_live_value_honest_null_gap_sharpened."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline-arcade live-search measurement over cached variants (1s floor); any LLM arm on the iGPU declares live_llm_inference."
- `verifier_is_oracle`: principle "MUST be false -- the SpatialValueNet is a learned value, oracle-DISTINCT from the executable win-check."
- `solve_provenance`: principle "live_agent_self_discovery -- this graduates the value head into the SCORED live agent's OWN path (arc_loop_solve warm-start + E3AgentPolicy); NOT a parallel solver, NOT outer_loop_re."
- `value_head_live_path_reachable`: principle "HARD gate -- the graduated module is imported by scripts/arc_loop_solve.py AND reachable from E3AgentPolicy; arc_orphan_solver_lint passes (NOT an orphaned scripts/experiments solver, per the Live-Path Reachability Discipline)."
- `bridge_fix_applied`: principle "the A1-indicated fix actually wired (decision-point-only/cached / DAgger-retrained / isotonic-calibrated) -- documents the targeted, not naive, integration."
- `first_win_rate_graduated`: principle "the HEADLINE -- held-out LIVE first-win-rate WITH the graduated SpatialValueNet (> the linear baseline is the bridge crossed)."
- `first_win_rate_linear_baseline`: principle "the matched LINEAR-verifier baseline on the SAME variants (today's warm-start that 'actively misled' -- the apples-to-apples control)."
- `first_win_rate_bare`: principle "the bare-BFS control (no value head) -- the no-regression floor."
- `first_win_delta`: principle "graduated - linear baseline (positive = the position-preserving head crosses the bridge), emitted explicitly so a null (0) is annotated."
- `first_win_ci`: principle "bootstrap CI on the first-win delta; a claim above the linear baseline requires the CI to exclude it."
- `median_actions_to_first_levelup_graduated`: principle "ACTION cost WITH the graduated head -- the leaderboard tiebreaker (RHAE rewards efficiency)."
- `actions_delta`: principle "linear_actions - graduated (positive = fewer actions); emitted explicitly so a null is annotated."
- `value_weight_used`: principle "MUST be ~0 (tie-breaker) or the A1-indicated bounded mode -- documents that this did NOT repeat the value_weight=5 regression."
- `parity_test_green`: principle "HARD gate -- test_arc_submitted_agent_parity.py passes; the integrated config stays the single source of truth."
- `bare_and_linear_controls_passed`: principle "the POSITIVE CONTROLS -- graduated must beat the linear baseline on the SAME variants AND not drop below bare BFS; a null is valid only if both ran."
- `false_negative_risk_checked`: principle "true with both controls run -- a no-value null is valid only then."
- `null_delta_methodology_note`: principle "present when first_win_delta==0 -- states the equality is an honest no-value null, not a measurement bug."
- `solve_rate_preserved`: principle "HARD gate -- graduating the head must NOT drop solve-rate vs bare BFS."
- `chosen_submitted_config`: principle "the recommended SUBMITTED_AGENT_CONFIG (graduated head on, value mode, weight) -- the A6 input; 'unchanged' if null."
- `offline_reproduced`: principle "any newly-solved variant must offline-reproduce to count."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified (offline arcade, E3AgentPolicy + ValueNet importable, A1 artifact present); pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4617-LIVE-PATH

**Given** the offline arcade, `E3AgentPolicy`, `ValueNet`, and the experiment-4616
A1 artifact are available
**When** experiment 4617 integrates the graduated spatial value head
**Then** `SpatialValueNet` is importable from the live `carnot.agentic` module,
`scripts/arc_loop_solve.py` imports the spatial checkpoint loader instead of the
linear `LearnedVerifier` warm-start, `E3AgentPolicy` can reach the same loader,
the A1 decision-point/cached bridge fix is active, `value_weight` is not the
known-regressed heavy A* weight, and `scripts/arc_orphan_solver_lint.py` passes.

#### SCENARIO-ARC-WMTE-4617-MATCHED-CONTROLS

**Given** the graduated, linear-baseline, and bare-BFS arms run on the same
held-out variants
**When** experiment 4617 builds its result artifact
**Then** the artifact reports first-win-rate, solve-rate, median
actions-to-first-level-up, paired bootstrap CI, explicit `first_win_delta` and
`actions_delta`, sets `verifier_is_oracle=false`, requires offline reproduction
for any newly solved variant, and emits `null_delta_methodology_note` when the
graduated and linear first-win rates are equal.

#### SCENARIO-ARC-WMTE-4617-BLOCKED-PRECONDITION

**Given** the offline arcade, live policy imports, value-net imports, or A1
artifact are unavailable
**When** experiment 4617 runs its precondition check
**Then** it writes a blocked artifact with `preconditions_checked`, a terminal
`blocked_` verdict, no fabricated live-value evidence, and a stable checksum.

### REQ-ARC-WMTE-4620: Primitive Persist Transfer for Offline-To-Live Bridge Fix

Experiment 4620 SHALL read
`results/experiment_4616_offline_live_bridge_disambiguation.json` and
`results/experiment_4617_graduate_spatial_value_head_live.json`, select the
strongest measured reusable primitive, persist it as an `arc_solver_kit`
operator, and record the reusable asset under `general_gotchas` in
`ops/arc_solve_registry.yaml`. Because Experiment 4617 graduated the
SpatialValueNet into the live path but reported no first-win or action lift,
while Experiment 4616 isolated `compute_cost` and identified
decision-point-only/cached value evaluation as the reusable bridge fix, the
expected persisted primitive is `value_head_bridge_fix_operator` with registry
gotcha `primitive_value_head_bridge_fix_operator`.

The persisted bridge-fix operator SHALL compare a baseline candidate order to a
bounded decision-point-only value-head order, SHALL cache repeated state scores,
SHALL report value-head evaluation count and elapsed cost proxies, SHALL keep
`verifier_is_oracle=false`, and SHALL report value only when an untuned transfer
game has a live first-win lift or an efficiency lift without fabricating a new
level. Experiment 4620 SHALL apply the persisted primitive to two or more
transfer games not tuned in Experiment 4616 or 4617, offline-reproduce any newly
counted level through `arc_solver_kit.reproduce`, and write
`results/experiment_4620_primitive_persist_transfer.json`.

The artifact SHALL emit principle-annotated top-level fields for
`honest_verdict`, `inference_substrate`, `verifier_is_oracle`,
`solve_provenance`, `primitive_persisted`, `transfer_games`,
`transfer_value_per_game`, `offline_reproduced`, `registry_updated`,
`random_seed`, `reproducibility_checksum`, and `preconditions_checked`.
Required field principles are:

- `honest_verdict`: principle "terminal prefix; success: primitive_persisted_transfer_<game>_value_added OR complete: primitive_persisted_transfer_null_characterized."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates for the offline transfer; declared so a fast real run is not DURATION_TOO_SHORT/METHODOLOGY false-flagged."
- `verifier_is_oracle`: principle "MUST be false -- the persisted primitive RANKS/ROUTES (the value head) or RE-CALIBRATES, oracle-distinct from the win-check."
- `solve_provenance`: principle "development_proxy if a transfer solve is via the offline twin; live_agent_self_discovery if the persisted primitive improves the SCORED agent's own path. NOT outer_loop_re."
- `primitive_persisted`: principle "names the arc_solver_kit operator + registry general_gotcha id added/extended -- the reusable asset (Solver-Reuse Discipline); without it the A1/A2 effort is wasted per the ARC reuse rule."
- `transfer_games`: principle "the games the primitive was applied to (NOT tuned on) -- the generalization test."
- `transfer_value_per_game`: principle "the per-game value-add (live first-win / efficiency lift) -- the cross-game evidence the primitive generalizes."
- `offline_reproduced`: principle "only offline-reproduced new levels count toward reproducible_total_levels."
- `registry_updated`: principle "the primitive + transfer dead-ends persisted so the next milestone reuses, not re-derives."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

If no untuned transfer game shows positive live first-win or efficiency lift,
the artifact SHALL report `complete: primitive_persisted_transfer_null_characterized`,
persist the best-characterized primitive-as-built, record transfer dead-ends in
the registry, and SHALL NOT increment `reproducible_total_levels`.

#### SCENARIO-ARC-WMTE-4620

**Given** the offline arcade precondition passes, Experiment 4616 and 4617
artifacts exist, and the bridge-fix helper has the stronger measured reusable
signal
**When** experiment 4620 persists the solver-kit bridge-fix operator and
measures it on at least two untuned transfer games
**Then** the artifact records the persisted operator and registry gotcha, the
untuned transfer games, per-game first-win or efficiency value-add, offline
reproduction accounting for any new level, a stable checksum, and a terminal
success only when at least one transfer game adds value.

### REQ-ARC-WMTE-4632: Persist 4629 Action-Effect Ranker And Measure Untuned Transfer

Experiment 4632 SHALL read
`results/experiment_4628_dense_curiosity_progress_loop.json` and
`results/experiment_4629_graduate_action_effect_predictor_live.json`, select
the stronger measured reusable primitive, persist it as reusable solver-kit
scaffolding, and extend the existing
`primitive_persistent_action_effect_memory_operator` general gotcha rather than
duplicating the action-effect primitive. Because Experiment 4628 reported
`complete: dense_curiosity_loop_no_live_lift_honest_null_gap_sharpened` while
Experiment 4629 reported live action-efficiency and first-win lift with
solve-rate preserved, the expected persisted primitive is
`persistent_action_effect_memory_operator` extended with the 4629
`persistent_aem_plus_optional_cnn` action-effect ranking signal.

The persisted action-effect ranker SHALL remain oracle-distinct from the win
check, SHALL rank candidate actions by cross-game frame/action-effect evidence
with deterministic original-order tie-breaking, SHALL support the live
`LiveActionEffectScorer`/PersistentAEM path used by the scored E3 agent, and
SHALL report per-game live action-efficiency or first-win lift only on transfer
games whose target game rows are excluded from the effect memory. Experiment
4632 SHALL apply the persisted primitive to two or more transfer games not tuned
in the target scorer, offline-reproduce any newly counted level through
`arc_solver_kit.reproduce`, and write
`results/experiment_4632_primitive_persist_transfer.json`.

The artifact SHALL emit principle-annotated top-level fields for
`honest_verdict`, `inference_substrate`, `verifier_is_oracle`,
`solve_provenance`, `primitive_persisted`, `transfer_games`,
`transfer_value_per_game`, `offline_reproduced`, `registry_updated`,
`random_seed`, `reproducibility_checksum`, and `preconditions_checked`.
Required field principles are:

- `honest_verdict`: principle "terminal prefix; success: primitive_persisted_transfer_<game>_value_added OR complete: primitive_persisted_transfer_null_characterized."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates for the offline transfer; declared so a fast real run is not DURATION_TOO_SHORT/METHODOLOGY false-flagged."
- `verifier_is_oracle`: principle "MUST be false -- the persisted primitive directs exploration or prunes actions, oracle-distinct from the win-check."
- `solve_provenance`: principle "development_proxy if a transfer solve is via the offline twin; live_agent_self_discovery if the persisted primitive improves the SCORED agent's own path. NOT outer_loop_re."
- `primitive_persisted`: principle "names the arc_solver_kit operator + registry general_gotcha id added/extended -- the reusable asset (Solver-Reuse Discipline); without it the A1/A2 effort is wasted per the ARC reuse rule."
- `transfer_games`: principle "the games the primitive was applied to (NOT tuned on) -- the generalization test."
- `transfer_value_per_game`: principle "the per-game value-add (live solve-rate / action-efficiency lift) -- the cross-game evidence the primitive generalizes."
- `offline_reproduced`: principle "only offline-reproduced new levels count toward reproducible_total_levels."
- `registry_updated`: principle "the primitive + transfer dead-ends persisted so the next milestone reuses, not re-derives."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

If no untuned transfer game shows positive live solve-rate, first-win, or
action-efficiency lift, the artifact SHALL report
`complete: primitive_persisted_transfer_null_characterized`, persist the
best-characterized primitive-as-built, record transfer dead-ends in the
registry, and SHALL NOT increment `reproducible_total_levels`.

#### SCENARIO-ARC-WMTE-4632

**Given** the offline arcade precondition passes, Experiment 4628 and 4629
artifacts exist, and the 4629 action-effect ranker has the stronger measured
signal
**When** experiment 4632 extends the solver-kit action-effect operator and
measures it on at least two target-excluded transfer games
**Then** the artifact records the persisted operator and registry gotcha, the
untuned transfer games, per-game solve-rate/action-efficiency value-add,
offline reproduction accounting for any new level, a stable checksum, and a
terminal success only when at least one transfer game adds value.

### REQ-ARC-WMTE-4633: Consolidate Clean ARC Sprint Wins Into The Scored Agent

Experiment 4633 SHALL consolidate the 2026-06-23 ARC integration sprint into
the scored submitted agent configuration. The workflow SHALL first verify
`arc_solver_kit.offline_arcade()` and the presence of
`results/experiment_4628_dense_curiosity_progress_loop.json`,
`results/experiment_4629_graduate_action_effect_predictor_live.json`,
`results/experiment_4630_levelup_selfplay.json`, and
`results/experiment_4631_refresh_submission_package.json`. It SHALL read A1,
A2, and A3 through `scripts/summarize_artifact.py` before aggregating any
number. A lever may be admitted into `SUBMITTED_AGENT_CONFIG` only when its
upstream artifact has a terminal success verdict, is not stamped or
live-rechecked as adversarial, passes its positive control, and raises a real
scored-agent metric. Any flagged-adversarial, positive-control-failed,
false-negative-risk, or non-success artifact SHALL be quarantined and SHALL NOT
contribute headline numbers.

If Experiment 4628 cleanly raises live solve-rate or coverage, the workflow
SHALL ship its dense curiosity progress loop setting in
`SUBMITTED_AGENT_CONFIG`. If Experiment 4629 cleanly raises live
action-efficiency, the workflow SHALL ship its `persistent_aem_plus_optional_cnn`
action-effect ranking setting in `SUBMITTED_AGENT_CONFIG`. If Experiment 4630
cleanly banks a level, the workflow SHALL rely on the refreshed Experiment 4631
package that already folded it in. If no clean upstream lever raises a
scored-agent metric, the workflow SHALL keep the bare submitted config, rerun
the parity and orphan-lint gates, and emit an honest null. The gate SHALL never
submit to the leaderboard and SHALL write
`results/experiment_4633_integration_gate.json`.

Experiment 4633 SHALL emit bare top-level fields for `honest_verdict`,
`inference_substrate`, `verifier_is_oracle`, `levers_integrated`,
`flagged_artifacts_excluded`, `live_solve_rate_integrated`,
`action_efficiency_integrated`,
`offline_to_live_transfer_ratio_integrated`,
`live_submittable_level_count_integrated`, `parity_test_green`,
`orphan_lint_green`, `submitted_config_raised_metric_clean`,
`null_delta_methodology_note`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`. Required field principles are:

- `honest_verdict`: principle "terminal prefix; success: integrated_<metric>_raised_config_shipped OR complete: integration_no_clean_metric_bare_config_kept_honest_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline end-to-end re-measure (1s floor)."
- `verifier_is_oracle`: principle "MUST be false for every aggregated value claim -- the integrated levers are oracle-distinct."
- `levers_integrated`: principle "names the upstream levers (A1/A2/A3) admitted into SUBMITTED_AGENT_CONFIG -- the audit trail."
- `flagged_artifacts_excluded`: principle "names any flagged_adversarial / positive-control-failed upstream artifact NOT aggregated (the fabrication gate + FALSE_NEGATIVE_RISK compliance)."
- `live_solve_rate_integrated`: principle "the integrated held-out live solve-rate on the shipped config (A1's effect)."
- `action_efficiency_integrated`: principle "the integrated median actions-to-first-levelup + efficiency term on the shipped config (A2's effect)."
- `offline_to_live_transfer_ratio_integrated`: principle "the integrated bridge co-metric on the shipped config -- did the offline verifier signal transfer to the SCORED agent."
- `live_submittable_level_count_integrated`: principle "the integrated live-submittable count (must stay > 33)."
- `parity_test_green`: principle "HARD gate -- test_arc_submitted_agent_parity.py passes; the integrated config is the single source of truth."
- `orphan_lint_green`: principle "HARD gate -- arc_orphan_solver_lint passes; the graduated A1/A2 modules stay live-path-reachable."
- `submitted_config_raised_metric_clean`: principle "True only if a CLEAN (non-flagged, control-passed) lever raised a real metric on the SCORED config; false -> honest null, bare config kept."
- `null_delta_methodology_note`: principle "present where any integrated delta == 0 -- states the equality is an honest no-value null, not a bug."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified (offline arcade, upstream artifacts present); pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4633

**Given** the offline arcade precondition passes, Experiments 4628 through 4631
exist, and A1/A2/A3 have been summarized through `scripts/summarize_artifact.py`
**When** experiment 4633 audits the upstream levers and re-measures the shipped
submitted configuration
**Then** it wires only clean metric-raising levers into
`SUBMITTED_AGENT_CONFIG`, excludes flagged or failed-control artifacts from all
aggregated numbers, runs the submitted-agent parity and orphan-solver lint
gates, writes `results/experiment_4633_integration_gate.json`, and emits an
honest null with the bare config if no clean lever raises a metric.

### REQ-ARC-WMTE-4634: Canonical Live Action-Efficiency Metric Helper

Experiment 4634 SHALL provide a canonical helper that reads the A2 action-effect
predictor artifact `results/experiment_4629_graduate_action_effect_predictor_live.json`,
the A6 integration artifact `results/experiment_4633_integration_gate.json`, the
offline-to-live co-metric artifact
`results/experiment_4622_offline_to_live_transfer_ratio_metric.json`, and the ARC
solve registry without loading a model. The helper SHALL compute
`live_action_efficiency` as the mean over live-solved levels of
`min(human_baseline_actions / agent_actions, 1)^2`. It SHALL expose each
per-level row's `agent_actions`, `baseline_actions`, and `efficiency` so the
leaderboard score term is auditable and not a single opaque aggregate.

Experiment 4634 SHALL report `live_action_efficiency` side-by-side with the other
ARC co-headline metrics in one canonical `coheadline_block`: reproducible total
levels from `ops/arc_solve_registry.yaml`, live-submittable level count,
first-win-rate, and `offline_to_live_transfer_ratio`. If zero live levels are
solved, efficiency is undefined; the helper SHALL report
`live_action_efficiency: 0.0` and include `null_delta_methodology_note` instead
of fabricating a non-null value.

Experiment 4634 SHALL write
`results/experiment_4634_live_action_efficiency_metric.json` with
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `live_action_efficiency`, `per_level_efficiency`,
`coheadline_block`, `tests_added`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`.

Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: live_action_efficiency_metric_helper_shipped_tests_green."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts -- reads the A2 artifact + registry, no model load (100us floor)."
- `live_action_efficiency`: principle "the canonical co-headline metric value (mean min(human/agent,1)^2 over live-solved levels) -- the leaderboard score term."
- `per_level_efficiency`: principle "the per-level efficiency + agent-actions + baseline-actions -- explicit so the metric is auditable, not a single opaque number."
- `coheadline_block`: principle "live_action_efficiency reported side-by-side with reproducible_total_levels / live-submittable / first-win-rate / offline_to_live_transfer_ratio -- the single canonical metric surface."
- `tests_added`: principle "the asserting tests (every test has >=1 assertion; no skips) -- the metric helper is verified, not asserted."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4634

**Given** an A2 live artifact with explicit per-level agent action counts and
baseline action references for solved levels
**When** the 4634 helper computes the live action-efficiency metric
**Then** it reports the mean `min(human_baseline_actions / agent_actions, 1)^2`,
records per-level agent and baseline actions, reports the value in the canonical
coheadline block beside reproducible total levels, live-submittable count,
first-win-rate, and offline-to-live transfer ratio, and emits
`null_delta_methodology_note` with value `0.0` when there are no solved live
levels.

### REQ-ARC-WMTE-4640: Exp4020 Graded Goal-Energy Live Generation

Experiment 4640 SHALL wire Exp4020's induced visible-state `is_goal(state)`
predicate into the live ARC generation path as a graded goal-satisfaction
energy. The energy SHALL score the fraction of target groups satisfied,
returning `0.0` only when the predicate fires and `1.0 - satisfied/total`
otherwise; it SHALL NOT collapse to the binary `unsatisfied_targets == 0`
gate. `graph_explore_solve_v2` SHALL accept this graded energy as a light dense
bias and combine it convexly with the existing navigation heuristic,
`alpha * arc_goal_distance + beta * graded_goal_energy`, while preserving depth
as the no-regression primary search term. Any returned/generated plan SHALL pass
the visible predicate gate before it is emitted.

The scored `E3AgentPolicy` / `StepwiseExplorer` path SHALL import the goal-energy
module, keep the Exp4629 action-effect predictor enabled, and install the same
lower-is-better graded goal-energy bias in the submitted default configuration.
The goal-energy module SHALL be live-path reachable from `arc_graph_explore` or
`arc_competition_agent`, and `scripts/arc_orphan_solver_lint.py` SHALL remain
green.

Experiment 4640 SHALL compare the graded goal-energy arm against the matched
action-effect-only baseline and a uniform-energy ablation on the same held-out
public-game variants. The terminal artifact
`results/experiment_4640_goal_energy_generation_live.json` SHALL include
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `verifier_is_oracle`, `solve_provenance`,
`live_path_reachable`, `goal_energy_source`, `gap_closed`,
`live_solve_rate_goal_energy`, `live_solve_rate_baseline`,
`solve_rate_delta`, `first_win_rate_delta`, `median_actions_to_win_delta`,
`uniform_energy_ablation_passed`, `live_lift_ci`, `bare_control_passed`,
`false_negative_risk_checked`, `null_delta_methodology_note`,
`chosen_submitted_config`, `offline_reproduced`, `residual_bridge_gaps`,
`random_seed`, `reproducibility_checksum`, and `preconditions_checked`.

#### SCENARIO-ARC-WMTE-4640

**Given** Exp4020's artifact is present with a sandboxed
`goal_predicate_code`, the offline Arcade imports, and the E3 live explorer
imports
**When** Experiment 4640 compiles the predicate and runs the matched baseline,
goal-energy, and uniform-energy arms
**Then** a success artifact is permitted only when live solve-rate or
first-win-rate improves over the baseline with a bootstrap CI excluding zero,
beats the uniform-energy ablation, preserves offline reproduction for newly
solved variants, and keeps the live-path lint green; otherwise the artifact
MUST emit an honest null with explicit zero deltas and residual bridge gaps.

### REQ-ARC-WMTE-4645: ARC Sprint Integration Gate Consolidation

Experiment 4645 SHALL consolidate the 2026-06-23 ARC integration sprint's
clean scored-agent levers into the submitted ARC agent configuration. The
workflow SHALL first verify `arc_solver_kit.offline_arcade()` and the presence
of `results/experiment_4640_goal_energy_generation_live.json`,
`results/experiment_4641_action_effect_expansion_prior_live.json`,
`results/experiment_4642_levelup_selfplay.json`, and
`results/experiment_4643_refresh_submission_package.json`. It SHALL read A1,
A2, and A3 through `scripts/summarize_artifact.py` before aggregating any
number. A lever may be admitted into `SUBMITTED_AGENT_CONFIG` only when its
upstream artifact has a terminal success verdict, is not stamped or
live-rechecked as adversarial, passes its positive control, and raises a real
scored-agent metric. A1 SHALL additionally pass the uniform-energy ablation
before it may ship. Any flagged-adversarial, positive-control-failed,
uniform-energy-ablation-failed, false-negative-risk, or non-success artifact
SHALL be quarantined and SHALL NOT contribute headline numbers.

If Experiment 4640 cleanly raises live solve-rate or first-win over both the
matched baseline and the uniform-energy ablation, the workflow SHALL keep the
graded goal-energy setting enabled in `SUBMITTED_AGENT_CONFIG`. If Experiment
4641 cleanly raises the live multi-level solve rate or depth, the workflow
SHALL keep the action-effect expansion prior enabled. If Experiment 4642 banks
a level, the workflow SHALL rely on the Experiment 4643 refreshed package that
already folded it in. If no clean upstream lever raises a scored-agent metric,
the workflow SHALL keep the bare submitted config, rerun the parity and
orphan-lint gates, and emit an honest null. A bare==integrated metric equality
SHALL be reported as an honest null with `null_delta_methodology_note`, never
as a fabricated matched-pair improvement. The gate SHALL never submit to the
leaderboard and SHALL write `results/experiment_4645_integration_gate.json`.

Experiment 4645 SHALL emit bare top-level fields for `honest_verdict`,
`inference_substrate`, `verifier_is_oracle`, `levers_integrated`,
`flagged_artifacts_excluded`, `live_solve_rate_integrated`,
`live_multi_level_solve_rate_integrated`, `action_efficiency_integrated`,
`offline_to_live_transfer_ratio_integrated`,
`live_submittable_level_count_integrated`, `parity_test_green`,
`orphan_lint_green`, `submitted_config_raised_metric_clean`,
`null_delta_methodology_note`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`. Required field principles are:

- `honest_verdict`: principle "terminal prefix; success: integrated_<metric>_raised_config_shipped OR complete: integration_no_clean_metric_bare_config_kept_honest_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline end-to-end re-measure (1s floor)."
- `verifier_is_oracle`: principle "MUST be false for every aggregated value claim -- the integrated levers are oracle-distinct."
- `levers_integrated`: principle "names the upstream levers (A1/A2/A3) admitted into SUBMITTED_AGENT_CONFIG -- the audit trail."
- `flagged_artifacts_excluded`: principle "names any flagged_adversarial / positive-control-failed / uniform-energy-ablation-failed upstream artifact NOT aggregated (the fabrication gate + FALSE_NEGATIVE_RISK compliance)."
- `live_solve_rate_integrated`: principle "the integrated held-out live solve-rate on the shipped config (A1's effect)."
- `live_multi_level_solve_rate_integrated`: principle "the integrated >=2-level live solve-rate (A2's deeper-solve effect) -- the new wall metric."
- `action_efficiency_integrated`: principle "the integrated median actions-to-first-levelup + efficiency term on the shipped config."
- `offline_to_live_transfer_ratio_integrated`: principle "the integrated bridge co-metric on the shipped config -- did the offline signal transfer to the SCORED agent."
- `live_submittable_level_count_integrated`: principle "the integrated live-submittable count (must stay > 33)."
- `parity_test_green`: principle "HARD gate -- test_arc_submitted_agent_parity.py passes; the integrated config is the single source of truth."
- `orphan_lint_green`: principle "HARD gate -- arc_orphan_solver_lint passes; the graduated A1/A2 modules stay live-path-reachable."
- `submitted_config_raised_metric_clean`: principle "True only if a CLEAN (non-flagged, control-passed, uniform-energy-ablation-passed for A1) lever raised a real metric on the SCORED config; false -> honest null, bare config kept."
- `null_delta_methodology_note`: principle "present where any integrated delta == 0 -- states the equality is an honest no-value null, not a bug (the .427 TAUTOLOGY-false-flag fix)."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified (offline arcade, upstream artifacts present); pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4645

**Given** the offline arcade precondition passes, Experiments 4640 through 4643
exist, and A1/A2/A3 have been summarized through `scripts/summarize_artifact.py`
**When** experiment 4645 audits the upstream levers and re-measures the shipped
submitted configuration
**Then** it wires only clean metric-raising levers into
`SUBMITTED_AGENT_CONFIG`, excludes flagged, failed-control, or failed-uniform
artifacts from all aggregated numbers, runs the submitted-agent parity and
orphan-solver lint gates, writes `results/experiment_4645_integration_gate.json`,
and emits an honest null with the bare config if no clean lever raises a
metric, including when bare and integrated metrics are exactly equal.

### REQ-ARC-WMTE-4646: Canonical Live Multi-Level Solve-Rate Metric Helper

Experiment 4646 SHALL provide a canonical helper that reads the A1 goal-energy
artifact `results/experiment_4640_goal_energy_generation_live.json`, the A2
action-effect expansion-prior artifact
`results/experiment_4641_action_effect_expansion_prior_live.json`, the A6
integration artifact `results/experiment_4645_integration_gate.json`, the live
action-efficiency artifact
`results/experiment_4634_live_action_efficiency_metric.json`, the
offline-to-live co-metric artifact
`results/experiment_4622_offline_to_live_transfer_ratio_metric.json`, and the
ARC solve registry without loading a model. The helper SHALL compute
`live_multi_level_solve_rate` as the count of live attempts reaching depth
`>=2` divided by the count of live attempts. It SHALL expose a
`depth_histogram` with `depth_0`, `depth_1`, `depth_2`, and `depth_3_plus`
counts so the new-wall metric is auditable and not a single opaque aggregate.

Experiment 4646 SHALL report `live_multi_level_solve_rate` side-by-side with the
other ARC co-headline metrics in one canonical `coheadline_block`: reproducible
total levels from `ops/arc_solve_registry.yaml`, live-submittable count,
first-win-rate, `live_action_efficiency`, and
`offline_to_live_transfer_ratio`. If no live attempt reaches depth `>=2`, the
helper SHALL report `live_multi_level_solve_rate: 0.0` and include
`null_delta_methodology_note` instead of fabricating a non-null value.

Experiment 4646 SHALL write
`results/experiment_4646_live_multi_level_solve_rate_metric.json` with
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `live_multi_level_solve_rate`, `depth_histogram`,
`coheadline_block`, `tests_added`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`.

Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: live_multi_level_solve_rate_metric_helper_shipped_tests_green."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts -- reads the A1/A2 artifacts + registry, no model load (100us floor)."
- `live_multi_level_solve_rate`: principle "the canonical co-headline metric value (fraction of live attempts solving >=2 levels) -- the NEW WALL the generation levers must close."
- `depth_histogram`: principle "the per-depth count of live attempts (depth 0/1/2/3+) -- explicit so the metric is auditable, not a single opaque number."
- `coheadline_block`: principle "live_multi_level_solve_rate reported side-by-side with reproducible_total_levels / live-submittable / first-win-rate / live_action_efficiency / offline_to_live_transfer_ratio -- the single canonical metric surface."
- `tests_added`: principle "the asserting tests (every test has >=1 assertion; no skips) -- the metric helper is verified, not asserted."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4646

**Given** A1/A2 live artifacts with per-attempt `reached_level` or
`depth_of_live_solve` values
**When** the 4646 helper computes the live multi-level solve-rate metric
**Then** it reports attempts with depth `>=2` divided by all attempted live
rows, records the depth histogram, reports the value in the canonical
coheadline block beside reproducible total levels, live-submittable count,
first-win-rate, live action-efficiency, and offline-to-live transfer ratio, and
emits `null_delta_methodology_note` with value `0.0` when no attempt reaches
depth `>=2`.

### REQ-ARC-WMTE-4670: Multi-Level Harness CI Gate And Proposer Port Hygiene

Experiment 4670 SHALL provide a deterministic CI-gate at
`python/carnot/experiment_4670_multilevel_harness_cigate.py` that locks in the
Exp 4664 multi-level measurement substrate. The gate SHALL fail if the
multi-level rollout configuration targets fewer than two levels, or if source
inspection finds a break-at-first-level-up branch in the rollout body. This
prevents the prior degenerate metric where `live_multi_level_solve_rate` was
structurally forced to `0.0` before a second level-up could be attempted.

The gate SHALL also validate generation-measurement artifacts that claim Qwen:
the artifact must include a `/props`-verified `proposer_served_model` matching
Qwen3.5-9B-MTP. A Qwen-claimed artifact served by gemma on port 8919 SHALL fail
the guard instead of silently contributing wrong-model measurements.

The gate SHALL enforce first-win and multi-level performance floors derived
from the fixed A1 measurement artifact so future harness changes cannot quietly
drop live first-win or depth `>=2` rates below the A1-established floor. The
floor helper SHALL be deterministic and testable on synthetic regression and
honest fixtures without live LLM inference.

Experiment 4670 SHALL write
`results/experiment_4670_multilevel_harness_cigate.json` with
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `verifier_is_oracle`,
`degenerate_metric_cigate_added`, `port_hygiene_guard_added`,
`first_win_floor_cigate_added`, `tests_added`, `random_seed`,
`reproducibility_checksum`, and `preconditions_checked`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success: multilevel_harness_cigate_plus_port_hygiene_shipped_tests_green."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- harness-config + artifact checks + a small offline rollout (1s floor); no live_llm_inference (the CI-gate uses a mock/NoOp proposer)."
- `verifier_is_oracle`: principle "MUST be false -- the CI-gates guard the measurement substrate, oracle-distinct from the executable win-check."
- `degenerate_metric_cigate_added`: principle "the CI-gate that FAILS on target_levels<2 / break-at-first-win (the multi-level metric can never re-degenerate to 0.0-by-construction)."
- `port_hygiene_guard_added`: principle "the guard that a 'Qwen' generation measurement must verify proposer_served_model==Qwen (catches the port-8919 gemma-squat silent-wrong-model confound)."
- `first_win_floor_cigate_added`: principle "the floor CI-gate that fails on a live first-win/multi-level solve-rate regression below the A1 floor."
- `tests_added`: principle "the unit tests added for all three guards (Tests Must Run and Assert: flag the degenerate/wrong-model/regression fixtures, pass the honest fixtures)."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4670-DEGENERATE-METRIC-GATE

**Given** a multi-level rollout fixture with `target_levels=1` or a
break-at-first-level-up branch
**When** the Exp 4670 CI-gate validates the rollout configuration
**Then** it fails with explicit degenerate-metric errors. Given a fixed rollout
with `target_levels>=2` and no first-win break, the same gate passes.

#### SCENARIO-ARC-WMTE-4670-PORT-HYGIENE

**Given** a generation-measurement artifact that claims Qwen but reports
`proposer_served_model` as gemma from port 8919
**When** the Exp 4670 port-hygiene guard runs
**Then** it fails. Given a `/props`-verified Qwen3.5-9B-MTP artifact on a
non-colliding port, it passes.

#### SCENARIO-ARC-WMTE-4670-FIRST-WIN-FLOOR

**Given** live first-win or multi-level solve-rate measurements below the
A1-established floor
**When** the Exp 4670 performance-floor guard runs
**Then** it fails with the regressed metric names. Given measurements at or
above both floors, it passes.

### REQ-ARC-WMTE-4676: Hierarchical Subgoal Search Over Live E3 Frontier

Experiment 4676 SHALL first run the fixed live diagnostic across the generic
E3 agent's first-win and multi-level behavior before selecting the intervention
target. The diagnostic SHALL compare explore and value-routed configurations
over action budgets `200`, `400`, and `800` on the public variant set, SHALL
record per-config first-win and multi-level rates, and SHALL classify the
binding wall as `l1_first_contact` when generic first contact remains sparse or
`l2_deepening` when first contact is broad but deeper levels do not appear. Any
local LLM proposer used by the diagnostic or intervention SHALL be constructed
on a non-colliding port and `/props`-verified as Qwen3.5-9B-MTP, not the gemma
server commonly resident on port 8919.

The live `E3AgentPolicy` level-up induction path SHALL expose a bounded
hierarchical subgoal-search option. The option SHALL mine candidate subgoal
predicates from failed live search states, SHALL allow the .430 A1 LLM
goal-induction tier to act as a subgoal proposer that emits several near-goal
candidate predicates instead of only one terminal predicate, and SHALL call the
existing executable `plan_in_model` low-level planner for each selected leg in
the chain `start -> subgoal(s) -> goal`. The .430 A2 distribution-corrected
value head MAY be used only as a local within-subgoal tie-breaker and SHALL NOT
replace the executable reproduction gate.

Experiment 4676 SHALL measure the diagnostic-selected target against three
matched arms: induced hierarchical subgoals, no-subgoal flat search with the
same budget, and random-subgoal search with the same budget. A success claim
SHALL require the generic live agent to reach a new level through runtime
subgoal decomposition, replay that trajectory through
`arc_solver_kit.reproduce`, and show both matched ablations reaching a lower
level. If no new level is reached, the artifact SHALL report the null as
complete with one of `bounded_search_cannot_reach_subgoal`,
`subgoals_mechanically_irrelevant`, or `value_head_still_not_separating`.

Experiment 4676 SHALL write
`results/experiment_4676_hierarchical_subgoal_search_live.json` with
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `verifier_is_oracle`, `solve_provenance`,
`live_path_reachable`, `wall_diagnosis`, `generic_first_win_by_config`,
`subgoal_decomposition`, `per_subgoal_reachable`,
`generic_agent_reached_level`, `offline_reproduced`, `reproduced_levels`,
`no_subgoal_ablation_reached_level`,
`random_subgoal_ablation_reached_level`, `residual_cause_hypothesis`,
`null_methodology_note`, `bare_control_passed`,
`false_negative_risk_checked`, `proposer_served_model`,
`chosen_submitted_config`, `parity_test_green`, `random_seed`,
`reproducibility_checksum`, and `preconditions_checked`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success: hierarchical_subgoal_generic_agent_new_level_<game>_L<n> OR complete: hierarchical_subgoal_no_new_level_residual_<cause>."
- `inference_substrate`: principle "live_llm_inference -- the subgoal-proposer induction loads + runs the Qwen3.5-9B-MTP GGUF (60s floor); declared honestly because the proposer arm is a real LLM run."
- `verifier_is_oracle`: principle "MUST be false -- the subgoal proposer + value head are oracle-DISTINCT from the executable reproduction win-check."
- `solve_provenance`: principle "live_agent_self_discovery -- a generic-agent new level via runtime subgoal decomposition is the REAL deliverable, NOT a hand-built GameAdapter (development_proxy) and NOT outer_loop_re."
- `live_path_reachable`: principle "HARD gate -- the changed modules are in the E3AgentPolicy import closure; arc_orphan_solver_lint passes."
- `wall_diagnosis`: principle "STEP-1 result -- l1_first_contact | l2_deepening; resolves 0.04-vs-0.59 so subgoal search targets the binding wall."
- `generic_first_win_by_config`: principle "per-config (explore vs value_routed x budget) generic first-win + multi-level rate -- the honest measurement the prior 0.59-vs-0.04 ambiguity demanded."
- `subgoal_decomposition`: principle "the subgoal sequence the proposer emitted for the target (the make-a-winner-appear evidence: a multi-step winner assembled from reachable legs)."
- `per_subgoal_reachable`: principle "per-subgoal whether bounded low-level search reached it -- the mechanism check."
- `generic_agent_reached_level`: principle "the deepest level the GENERIC live agent reached via subgoal search on the target -- the headline (a NEW level is the win)."
- `offline_reproduced`: principle "a generic new level counts only if offline-reproduced via arc_solver_kit.reproduce."
- `reproduced_levels`: principle "the integer new-level count the generic agent banked offline (>=1 is the bridge crossed for solve)."
- `no_subgoal_ablation_reached_level`: principle "the matched flat-search ablation's reached_level -- MUST be lower than subgoal search for an attributable win."
- `random_subgoal_ablation_reached_level`: principle "the random-subgoal ablation's reached_level -- MUST be lower than induced subgoal search for an attributable win."
- `residual_cause_hypothesis`: principle "if it nulls, names the residual (bounded_search_cannot_reach_subgoal | subgoals_mechanically_irrelevant | value_head_still_not_separating) -- the .432 target; 'none' if it crossed."
- `null_methodology_note`: principle "present when no new level -- states the null is honest (passing ablations + reachable headroom), not a measurement bug."
- `bare_control_passed`: principle "the POSITIVE CONTROL -- the target has reachable headroom (reaches L1 by exploration OR L2 is registry-reachable)."
- `false_negative_risk_checked`: principle "true with the ablations run + reachable-headroom confirmed -- a no-new-level null is valid only then."
- `proposer_served_model`: principle "the model the proposer /props reported (MUST be Qwen3.5-9B-MTP, NOT gemma) -- the port-8919 confound guard."
- `chosen_submitted_config`: principle "the recommended SUBMITTED_AGENT_CONFIG change (subgoal search on, subgoal budget) -- 'unchanged' if null."
- `parity_test_green`: principle "HARD gate -- test_arc_submitted_agent_parity.py passes; the deployed agent == the measured agent."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent harness/corpus drift on replay."
- `preconditions_checked`: principle "records resources verified (Qwen cached, offline arcade, live modules importable, /props served Qwen on a free port); pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4676-DIAGNOSTIC

**Given** the fixed live E3 measurement harness and a `/props`-verified Qwen
proposer on a free port
**When** experiment 4676 sweeps explore/value-routed modes over budgets 200,
400, and 800
**Then** it records first-win and multi-level rates for each config and
classifies the binding wall before selecting the subgoal-search target.

#### SCENARIO-ARC-WMTE-4676-HIERARCHICAL-PLAN

**Given** an executable world model, a final goal predicate, and candidate
subgoal predicates mined from live failed-search states or proposed by the A1
induction tier
**When** hierarchical subgoal search runs
**Then** it invokes bounded `plan_in_model` for each selected leg, chains the
reachable legs, records per-subgoal reachability, and uses the A2 value head
only to break ties among subgoals within the same leg.

#### SCENARIO-ARC-WMTE-4676-ABLATIONS

**Given** a target where hierarchical subgoal search reaches a new level
**When** the matched no-subgoal and random-subgoal ablations run with the same
budget
**Then** both ablations must reach a lower level before the artifact can claim
the new level is attributable to induced subgoal decomposition.

### REQ-ARC-WMTE-4677: PoE-World Factored Executable Subgoal Planner

Experiment 4677 SHALL implement the `.431` A2 PoE-World structural fallback as
a live-path-reachable factored executable planner. The implementation SHALL
induce small programmatic object-level experts from live transitions, where each
expert carries an explicit precondition, effect rule, object class, and held-out
transition trust weight. Expert induction SHALL use the live local proposer on a
non-colliding port and SHALL `/props`-verify that the served model is
Qwen3.5-9B-MTP rather than the gemma server commonly resident on port 8919. The
planner SHALL keep only replay-stable factors whose held-out trust meets the
configured threshold, compose those factors into a product executable model, and
use that product model to generate mechanically valid subgoal-conditioned action
sequences.

The product-model planner SHALL accept A1-style subgoal predicates and the final
goal predicate, SHALL use the .430 A2 value head only to score which
product-model states or subgoals deserve live expansion, and SHALL keep the
programmatic experts plus value head oracle-distinct from the executable
reproduction win-check. Any plan emitted by the factored planner SHALL be
executed by the live E3 path and audited by replay; the product model may
generate candidates, but it SHALL NOT count a solve unless the live
execution/reproduction gate accepts it.

Experiment 4677 SHALL measure candidate-generation coverage against the matched
flat-search baseline on the same target games: `candidate_generation_coverage`
is the fraction of attempts where the winning action or plan appears in the
generated candidate pool. A positive generation claim SHALL require factored
coverage to exceed flat coverage and the winning plan to appear in the factored
pool where flat search did not. A downstream live-lift claim SHALL additionally
require a positive first-win-rate or multi-level solve-rate delta with bootstrap
CI excluding the flat baseline on at least one game. If coverage does not rise,
the artifact SHALL report an honest null with one residual bridge gap from
`expert_factors_not_independent`, `product_model_plans_live_invalid`, or
`experts_overfit_prefix`.

Experiment 4677 SHALL write
`results/experiment_4677_poe_world_factored_subgoal_planner.json` with
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `verifier_is_oracle`, `solve_provenance`,
`live_path_reachable`, `candidate_generation_coverage_factored`,
`candidate_generation_coverage_flat_baseline`, `coverage_delta`,
`live_first_win_rate_factored`, `live_solve_rate_factored`,
`live_baseline_flat_search`, `first_win_rate_delta`, `solve_rate_delta`,
`live_lift_ci`, `expert_trust_weights`, `bare_control_passed`,
`false_negative_risk_checked`, `null_methodology_note`,
`chosen_submitted_config`, `proposer_served_model`, `parity_test_green`,
`offline_reproduced`, `residual_bridge_gap`, `random_seed`,
`reproducibility_checksum`, and `preconditions_checked`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success: poe_world_factored_planner_coverage_up_live_<firstwin|solverate>_lift_<game> OR complete: poe_world_factored_planner_no_coverage_gain_residual_logged."
- `inference_substrate`: principle "live_llm_inference -- the programmatic-expert induction loads + runs the Qwen3.5-9B-MTP GGUF (60s floor)."
- `verifier_is_oracle`: principle "MUST be false -- the programmatic experts + value head are oracle-DISTINCT from the executable reproduction win-check."
- `solve_provenance`: principle "live_agent_self_discovery -- the generic agent's OWN runtime factored-model planning; NOT a hand-built adapter (development_proxy), NOT outer_loop_re."
- `live_path_reachable`: principle "HARD gate -- the changed modules are in the E3AgentPolicy import closure; arc_orphan_solver_lint passes."
- `candidate_generation_coverage_factored`: principle "does the winning action/plan APPEAR in the factored-planner-generated pool -- the make-a-winner-appear signal (the metric that distinguishes generation from selection)."
- `candidate_generation_coverage_flat_baseline`: principle "the matched flat-search baseline coverage -- a coverage CLAIM requires the factored coverage to exceed it (the winner generated where flat search did not)."
- `coverage_delta`: principle "factored - flat coverage (positive = the winner now appears); emitted explicitly so a null (0) is annotated."
- `live_first_win_rate_factored`: principle "the live first-win-rate WITH the factored planner on the SCORED agent."
- `live_solve_rate_factored`: principle "the live multi-level (>=2) solve-rate WITH the factored planner."
- `live_baseline_flat_search`: principle "the matched flat-search baseline first-win + solve-rate on the SAME games (the no-regression control)."
- `first_win_rate_delta`: principle "factored - baseline first-win-rate; emitted explicitly so a null is annotated."
- `solve_rate_delta`: principle "factored - baseline multi-level solve-rate; emitted explicitly so a null is annotated."
- `live_lift_ci`: principle "bootstrap CI on the chosen live-lift metric; a claim above baseline requires the CI to exclude it."
- `expert_trust_weights`: principle "the held-out transition trust per induced expert -- only replay-stable factors are composed (no brittle-expert fabrication)."
- `bare_control_passed`: principle "the POSITIVE CONTROL -- the matched baseline ran on a corpus with reachable headroom; a no-coverage-gain null is valid only then."
- `false_negative_risk_checked`: principle "true with the matched flat baseline + reachable-headroom confirmed -- a 'no coverage gain' null is valid only then."
- `null_methodology_note`: principle "present when coverage_delta==0 -- states the equality is an honest no-value null, not a measurement bug."
- `chosen_submitted_config`: principle "the recommended SUBMITTED_AGENT_CONFIG change (factored planner on, trust threshold) -- the A6 input; 'unchanged' if null."
- `proposer_served_model`: principle "the model the proposer /props reported (MUST be Qwen3.5-9B-MTP) -- the port-8919 confound guard."
- `parity_test_green`: principle "HARD gate -- test_arc_submitted_agent_parity.py passes."
- `offline_reproduced`: principle "any newly-solved variant must offline-reproduce to count."
- `residual_bridge_gap`: principle "the .432 generation gap logged if coverage does not rise (expert_factors_not_independent | product_model_plans_live_invalid)."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent harness/corpus drift on replay."
- `preconditions_checked`: principle "records resources verified (Qwen cached, offline arcade, live modules importable, /props served Qwen); pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4677-TRUSTED-FACTORS

**Given** live transition records and a programmatic-expert proposer
**When** Experiment 4677 induces object-level precondition/effect experts
**Then** each expert is scored on a held-out transition suffix, only
replay-stable experts above the trust threshold are composed, and the artifact
records their trust weights.

#### SCENARIO-ARC-WMTE-4677-PRODUCT-PLANNING

**Given** trusted programmatic experts, A1-style subgoal predicates, a final
goal predicate, and an A2 value head
**When** the factored product planner runs
**Then** it composes the experts into an executable product model, plans through
subgoal-conditioned legs, records reachability for each leg, and emits only
candidate plans that can be audited by live E3 execution.

#### SCENARIO-ARC-WMTE-4677-COVERAGE-CONTROL

**Given** the factored planner and a matched flat-search baseline on the same
target games
**When** Experiment 4677 measures generated candidate pools
**Then** the artifact reports factored coverage, flat coverage, explicit
coverage delta, live first-win and solve-rate deltas, bootstrap CI, and an
honest null note when the winning plan does not newly appear.

### REQ-ARC-WMTE-4749: Structured ProductWorldModel Engine Replacement

Experiment 4749 SHALL add a live-path-reachable `structured_load_engine(game)`
adapter for level-up re-induction that induces trusted programmatic object
experts with `induce_programmatic_object_experts`, composes them into a
`ProductWorldModel`, and returns the product model's `engine(grid, action,
data) -> ndarray` dynamics surface with the existing `is_level_complete`
provider or structural goal. The adapter SHALL be gated by
`CARNOT_ARC_STRUCTURED_ENGINE=1` and otherwise preserve the existing
`e3.load_engine` free-form path. The structured path SHALL reject or report a
dead identity engine before any accuracy claim unless at least one real-frame
candidate action changes more than zero cells.

Experiment 4749 SHALL measure held-out transition accuracy for the structured
engine and the free-form engine on the same real game transitions, then write
`results/experiment_4749_structured_engine_vs_freeform.json`. A success claim
SHALL require either a wide structured-vs-freeform accuracy win or a reproduced
new L2 bank; an honest no-improvement or blocked-resource outcome SHALL remain
terminal and non-fabricated.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix complete:/success:/passed:/shipped:; a wide structured-vs-freeform accuracy win OR an L2 bank is success_, an honest no-improvement null is complete_."
- `inference_substrate`: principle "live_llm_inference (the experts are induced via the live proposer + scored on real transitions); 60s duration floor."
- `preconditions_checked`: principle "records the GGUF/arcade/import checks so a silent-missing-resource run cannot fabricate an accuracy number."
- `structured_engine_non_degenerate`: principle "the structured engine must CHANGE >0 cells on >=1 action before any accuracy claim -- guards the dead-engine (identity) failure the .428-.433 audit found."
- `freeform_heldout_accuracy`: principle "the baseline 0.12 the structured engine must beat -- the explicit comparator, not a moving goalpost."
- `structured_heldout_accuracy`: principle "the structured engine's held-out transition accuracy -- the load-bearing measurement (target >=0.5)."
- `l2_proposer_failed`: principle "did the L2 reinduction still proposer_fail with the structured engine -- the direct test of whether the engine wall is cleared."
- `offline_reproduced`: principle "true only if a NEW level is independently re-derived by arc_solver_kit.reproduce -- the only real level-up signal."
- `solve_provenance`: principle "live_agent_self_discovery if the live agent advanced via its own attempts; development_proxy for the offline twin -- never over-credit."
- `verifier_is_oracle`: principle "false -- the held-out accuracy is execution-grounded against observed transitions, not a learned-verifier moat claim."

#### SCENARIO-ARC-WMTE-4749-STRUCTURED-ENGINE-ADAPTER

**Given** real transition records and a programmatic-expert proposer
**When** `structured_load_engine` is built for a game
**Then** it composes trusted experts into `ProductWorldModel.engine`, returns a
load-engine-shaped `(engine, goal)` pair, and records that the engine changes at
least one real frame on at least one action before reporting held-out accuracy.

#### SCENARIO-ARC-WMTE-4749-LIVE-WIRING

**Given** a level-up re-induction episode in the live `E3AgentPolicy` path
**When** `CARNOT_ARC_STRUCTURED_ENGINE=1`
**Then** the call to `execute_bounded_llm_reinduction` uses the structured
load-engine adapter instead of `e3.load_engine` while the default environment
continues to use the free-form loader.

#### SCENARIO-ARC-WMTE-4749-ACCURACY-ARTIFACT

**Given** structured and free-form engines evaluated on the same held-out
transitions
**When** Experiment 4749 writes its artifact
**Then** it reports both accuracies, non-degeneracy, L2 proposer failure status,
offline reproduction status, live-path reachability, required field principles,
and a terminal honest verdict without treating the verifier as an oracle.

### REQ-ARC-WMTE-4753: Persist .437 Structured/Detector Levers And Transfer Characterization

Experiment 4753 SHALL read the `.437` A1 structured-engine artifact
`results/experiment_4749_structured_engine_vs_freeform.json` and A2 fixed
structural-alignment detector artifact
`results/experiment_4750_structural_alignment_detector_fix.json`, verify the
literal `import arcade` precondition, and inspect the live
`arc_competition_agent` path for the env-gated structured-engine hook and the
fixed detector structural goal provider. If the upstream artifacts validated
their gates, the experiment SHALL confirm the corresponding live-path lever is
persisted and SHALL run `scripts/arc_orphan_solver_lint.py` before claiming live
reachability. If a required precondition is missing, the artifact SHALL emit a
terminal `blocked_<resource>` verdict and SHALL not fabricate transfer deltas.

When preconditions pass, Experiment 4753 SHALL characterize action-efficiency
and first-effect transfer value on at least two games without making a new-level
claim. The transfer measurement SHALL compare a baseline route against the
persisted .437 levers, report per-game `action_efficiency_delta` and
`first_effect_delta`, and keep `offline_reproduced_new_level=false`.

The terminal artifact `results/experiment_4753_persist_transfer.json` SHALL
include principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `preconditions_checked`, `transfer_games`,
`offline_reproduced_new_level`, `solve_provenance`, and `verifier_is_oracle`,
plus `live_path_persistence`, `upstream_validation`, `transfer_value_per_game`,
`arc_orphan_solver_lint`, `random_seed`, `reproducibility_checksum`, and
`duration_s`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; characterization-complete is complete_."
- `inference_substrate`: principle "live_llm_inference."
- `preconditions_checked`: principle "records arcade + upstream-artifact checks."
- `transfer_games`: principle "the >=2 games the levers were characterized on -- the transfer evidence, not a single-game artifact."
- `offline_reproduced_new_level`: principle "false expected -- this is characterization, NOT a level-bank (no over-claim)."
- `solve_provenance`: principle "development_proxy -- characterizes the levers on the offline dev twin, asserting no new level (offline_reproduced_new_level=false); declared (paired with verifier_is_oracle) because A5 makes an offline_reproduced_new_level claim, per the ARC Live-Path Reachability discipline."
- `verifier_is_oracle`: principle "false -- the transfer value is an execution-grounded efficiency measurement, not a learned-verifier moat claim."

#### SCENARIO-ARC-WMTE-4753-BLOCKED-PRECONDITION

**Given** the A1/A2 artifacts or literal `import arcade` precondition are
missing
**When** Experiment 4753 runs
**Then** it writes `results/experiment_4753_persist_transfer.json` with a
terminal `blocked_<resource>` verdict, records the failed precondition, keeps
`offline_reproduced_new_level=false`, and leaves transfer deltas unclaimed.

#### SCENARIO-ARC-WMTE-4753-LIVE-PERSISTENCE

**Given** the A1/A2 artifacts are present and any validated lever is eligible
for persistence
**When** Experiment 4753 inspects the live agent path
**Then** it records whether `CARNOT_ARC_STRUCTURED_ENGINE` reaches
`arc_structured_world_model`, whether the fixed structural-alignment goal
provider reaches `execute_bounded_llm_reinduction`, and whether
`arc_orphan_solver_lint.py` passes.

#### SCENARIO-ARC-WMTE-4753-TRANSFER-CHARACTERIZATION

**Given** all preconditions pass
**When** Experiment 4753 measures the persisted levers on two or more transfer
games
**Then** it reports per-game action-efficiency and first-effect deltas,
summarizes positive or null transfer honestly, sets
`offline_reproduced_new_level=false`, and uses
`solve_provenance=development_proxy` with `verifier_is_oracle=false`.

### REQ-ARC-WMTE-4754: Submitted .437 Lever Config Integration Confirmation

Experiment 4754 SHALL confirm that the `.437` structured-engine and fixed
detector levers are reflected in the scored submitted agent configuration only
when their A1/A2 validation artifacts authorize them. Before any confirmation
claim, it SHALL verify the cached Qwen3.5-9B-MTP GGUF, offline arcade
initialization, and `make_carnot_agent` import preconditions. Missing required
resources SHALL produce a terminal `blocked_<resource>` artifact without
fabricating integration or readiness.

The experiment SHALL inspect `arc_competition_agent.py` for the
`CARNOT_ARC_STRUCTURED_ENGINE` live hook and the fixed detector/trust-metric
hook, derive the submitted env-gate state from the `.437` A1/A2 artifacts,
construct the submitted `make_carnot_agent` entrypoint under that env-gate
state, run one offline smoke step, confirm the frozen Qwen3.5-9B-MTP generator
declaration remains intact, and run `scripts/arc_orphan_solver_lint.py`.

The terminal artifact
`results/experiment_4754_submitted_agent_config.json` SHALL include
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `preconditions_checked`,
`agent_constructs_and_smoke_runs`, and `submission_package_ready`, plus
`env_gate_state`, `levers_integrated`, `frozen_generator_intact`,
`arc_orphan_solver_lint`, `submitted_agent_config`,
`submitted_to_leaderboard=false`, `random_seed`, `reproducibility_checksum`,
and `duration_s`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; an integration-confirmed run is success_/complete_."
- `inference_substrate`: principle "live_llm_inference."
- `preconditions_checked`: principle "records GGUF/arcade checks."
- `agent_constructs_and_smoke_runs`: principle "the agent entrypoint builds + a smoke step runs offline -- the integration gate."
- `submission_package_ready`: principle "True only if OPERATOR-ready; this task NEVER submits (operator-only external publication)."

#### SCENARIO-ARC-WMTE-4754

**Given** the Qwen3.5-9B-MTP GGUF cache, offline arcade initialization, and
`make_carnot_agent` import preconditions pass
**When** Experiment 4754 derives the `.437` env-gate state, constructs the
submitted agent entrypoint, runs one offline smoke step, and runs
`arc_orphan_solver_lint.py`
**Then** it writes `results/experiment_4754_submitted_agent_config.json` with a
terminal success/complete verdict, records whether the `.437` levers are on or
unchanged, confirms the frozen generator declaration, sets
`submitted_to_leaderboard=false`, and sets `submission_package_ready=true` only
when the smoke, lint, package-path, and frozen-generator checks are green.

### REQ-ARC-WMTE-4680: Persist Milestone Generation Primitive And Transfer Null

Experiment 4680 SHALL persist the strongest reusable `.432` A1/A2 generation
component into `arc_solver_kit` and `ops/arc_solve_registry.yaml`, then measure
cross-game transfer on at least three cached held-out games. If Experiment 4676
or 4677 cleared its success gate, the cleared hierarchical-subgoal or
PoE-World factored-planner operator SHALL be persisted. If both are value-null,
the experiment SHALL persist the strongest characterized component instead. For
the `.432` artifacts this component is the programmatic-expert trust-weighting
gate from Experiment 4677: it ranks generated object-level experts by held-out
transition trust, keeps only replay-stable experts, and rejects overfit prefix
experts before product-model planning.

The persisted primitive SHALL be oracle-distinct from the executable
win-check. It may generate, rank, induce, or filter candidate experts, but a
new level SHALL count only when the offline reproduction gate accepts it. The
transfer measurement SHALL apply the persisted operator to cached held-out games
and SHALL report, per game, candidate-coverage delta, live solve-rate delta,
first-win delta, and `offline_reproduced_new_level`. A zero-value transfer is
valid only when recorded with a residual dead-end for the next attack.

Experiment 4680 SHALL write
`results/experiment_4680_primitive_persist_transfer.json` with
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `verifier_is_oracle`, `primitive_persisted`,
`transfer_games`, `transfer_value_per_game`, `offline_reproduced_new_level`,
`residual_dead_end`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success: primitive_persisted_transfer_<value|null>_characterized OR complete: primitive_persisted_transfer_null_characterized."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline transfer measurement over cached games (1s floor)."
- `verifier_is_oracle`: principle "MUST be false -- the persisted primitive generates/ranks/induces, oracle-distinct from the executable win-check."
- `primitive_persisted`: principle "the operator id + derived_from_artifacts + source -- the reusable scaffolding captured so the live solver reuses it, never re-derives."
- `transfer_games`: principle "the >=3 held-out games the persisted operator was applied to (the cross-game transfer measurement)."
- `transfer_value_per_game`: principle "per-game coverage/solve-rate/first-win delta + offline_reproduced_new_level -- honest transfer value, null characterized if zero."
- `offline_reproduced_new_level`: principle "true only if the transfer banked a strictly NEW offline-reproduced level (else reproducible_total_levels is unchanged -- stated honestly)."
- `residual_dead_end`: principle "the characterized transfer-null residual if value is zero (the next-attack record); per the .430 A5 pattern."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4680-PERSIST-STRONGEST-COMPONENT

**Given** Experiment 4676 and 4677 both report complete value-null verdicts
**When** Experiment 4680 selects the reusable primitive
**Then** it persists `programmatic_expert_trust_weighting_operator` under the
registry gotcha id `primitive_programmatic_expert_trust_weighting_operator`,
records the 4676/4677 source artifacts, and explains that this captures
characterized generation scaffolding rather than fabricating a banked level.

#### SCENARIO-ARC-WMTE-4680-TRANSFER-MEASUREMENT

**Given** cached held-out games and the persisted expert-trust operator
**When** Experiment 4680 measures transfer
**Then** each game reports coverage, solve-rate, first-win, and offline
reproduction deltas, and a zero-value run emits
`complete: primitive_persisted_transfer_null_characterized` with a
`residual_dead_end`.

### REQ-ARC-WMTE-4681: Submitted A1/A2 Integration Gate For Structural Deepening

Experiment 4681 SHALL read the previous submitted integration gate
`results/experiment_4669_integration_gate.json`, the hierarchical-subgoal
artifact `results/experiment_4676_hierarchical_subgoal_search_live.json`, and
the PoE-World factored-planner artifact
`results/experiment_4677_poe_world_factored_subgoal_planner.json`. The gate
SHALL audit whether the A1 hierarchical subgoal-search submitted config
(`hierarchical_subgoal_search_enabled`, `hierarchical_subgoal_budget`) or the A2
factored-planner submitted config (`factored_planner_enabled`,
`factored_trust_threshold`) cleared its upstream success gate and is actually
folded into `SUBMITTED_AGENT_CONFIG`. If both upstream artifacts are null or
explicitly `unchanged`, the gate SHALL record `config_integrated` as unchanged
with the reason and SHALL set `config_changed=false`; in that case identical
pre/post measurements are by construction and SHALL NOT be presented as a
metric finding.

Experiment 4681 SHALL re-measure or reuse the fixed cached live measurements
for the scored `E3AgentPolicy` against the pre-integration config, and SHALL
write `results/experiment_4681_integration_gate.json` with
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `verifier_is_oracle`, `config_integrated`,
`config_changed`, `live_first_win_rate_integrated`,
`live_multi_level_solve_rate_integrated`,
`live_submittable_level_count_integrated`, `parity_test_green`, `random_seed`,
`reproducibility_checksum`, and `preconditions_checked`. The artifact SHALL
also include the A1/A2 audit, pre-integration comparison metrics,
no-regression status, source artifact paths/checksums, the parity-test result,
`submitted_to_leaderboard=false`, and spec refs. The verifier SHALL remain
oracle-distinct from the executable win-check.

Required field principles are:

- `honest_verdict`: principle "terminal prefix; success: integrated_<config>_shipped_parity_green OR complete: integration_unchanged_both_levers_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline end-to-end re-measure (1s floor)."
- `verifier_is_oracle`: principle "MUST be false -- the integrated subgoal-search / factored-planner signals are oracle-distinct from the executable win-check."
- `config_integrated`: principle "the A1/A2 config folded into SUBMITTED_AGENT_CONFIG (or 'unchanged' with the reason) -- the single-source-of-truth update."
- `config_changed`: principle "bool -- true only if a lever cleared its gate; when false, the pre/post numbers are identical BY CONSTRUCTION (not a finding) -- pre-empts the .430 A6 TAUTOLOGY flag."
- `live_first_win_rate_integrated`: principle "the integrated SCORED-agent live first-win-rate (no regression vs pre-integration)."
- `live_multi_level_solve_rate_integrated`: principle "the integrated SCORED-agent multi-level (>=2) live solve-rate (the deeper wall) -- measured on the FIXED non-degenerate harness."
- `live_submittable_level_count_integrated`: principle "the integrated live-submittable count (must stay > 33)."
- `parity_test_green`: principle "HARD gate -- test_arc_submitted_agent_parity.py passes; the deployed agent == the measured agent."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4681

**Given** `SUBMITTED_AGENT_CONFIG` imports with `E3AgentPolicy`, the 4669
pre-integration comparison artifact is present, and the 4676/4677 A1/A2
artifacts are readable
**When** Experiment 4681 audits the upstream chosen submitted configs and runs
the submitted-agent parity test
**Then** the result artifact reports either the exact A1/A2 config folded into
the single source of truth or an unchanged reason when both levers are null,
reports `config_changed` truthfully, reports live first-win rate,
fixed-harness multi-level solve rate, live submittable count greater than 33,
no-regression versus the pre-integration config, `verifier_is_oracle=false`,
`parity_test_green=true`, and a stable reproducibility checksum.

### REQ-ARC-WMTE-4693: Submitted A1/A2 Integration Gate For Directed Exploration

Experiment 4693 SHALL read the controllable-novelty proposal artifact
`results/experiment_4688_controllable_novelty_proposal_policy_live.json` and
the program-synthesis action-effect proposal-filter artifact
`results/experiment_4689_program_synthesis_action_effect_proposal_filter.json`.
The gate SHALL audit whether the A1 controllable-novelty submitted config
(`controllable_novelty_proposal_enabled`, novelty weight, and novelty
temperature) or the A2 program-synthesis proposal-filter submitted config
(`program_synthesis_proposal_filter_enabled`,
`program_synthesis_proposal_filter_trust_threshold`) cleared its upstream
success gate and is actually folded into `SUBMITTED_AGENT_CONFIG`. If both
upstream artifacts are null or explicitly `unchanged`, the gate SHALL record
`config_integrated` as unchanged with the reason, SHALL set
`config_changed=false`, and SHALL state that identical pre/post measurements are
by construction rather than a finding.

Experiment 4693 SHALL re-measure the integrated held-out first-win rate and
multi-level deepen rate on the scored `E3AgentPolicy` by invoking the
Experiment 4605 held-out lane with deepening enabled. The artifact at
`results/experiment_4693_integration_gate.json` SHALL include
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `verifier_is_oracle`, `config_integrated`,
`config_changed`, `first_win_rate_integrated`,
`multi_level_deepen_rate_integrated`, `parity_test_green`, `random_seed`,
`reproducibility_checksum`, and `preconditions_checked`. The artifact SHALL
also include the A1/A2 audit, pre-integration comparison metrics,
no-regression status, source artifact paths/checksums, the scored-lane
measurement, the parity-test result, `submitted_to_leaderboard=false`, and spec
refs. The verifier SHALL remain oracle-distinct from the executable win-check.

Required field principles are:

- `honest_verdict`: principle "terminal prefix; success: integrated_<config>_shipped_parity_green OR complete: integration_unchanged_both_levers_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline end-to-end re-measure on the held-out lane (1s floor)."
- `verifier_is_oracle`: principle "MUST be false -- the integrated controllable-novelty / proposal-filter signals are oracle-distinct from the executable win-check."
- `config_integrated`: principle "the A1/A2 config folded into SUBMITTED_AGENT_CONFIG (or 'unchanged' with the reason) -- the single-source-of-truth update."
- `config_changed`: principle "bool -- true only if a lever cleared its gate; when false, the pre/post numbers are identical BY CONSTRUCTION (not a finding) -- pre-empts the .430 A6 TAUTOLOGY flag."
- `first_win_rate_integrated`: principle "the integrated SCORED-agent held-out first-win-rate (no regression vs pre-integration) -- the retargeted scored-lane metric."
- `multi_level_deepen_rate_integrated`: principle "the integrated SCORED-agent multi-level deepen-rate (the deeper scored lever) -- measured on the held-out lane."
- `parity_test_green`: principle "HARD gate -- test_arc_submitted_agent_parity.py passes; the deployed agent == the measured agent."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4693

**Given** `SUBMITTED_AGENT_CONFIG` imports with `E3AgentPolicy`, the 4688/4689
A1/A2 artifacts are readable, and the 4605 scored held-out lane is available
**When** Experiment 4693 audits the upstream chosen submitted configs, measures
the scored held-out lane with deepening enabled, and runs the submitted-agent
parity test
**Then** the result artifact reports either the exact A1/A2 config folded into
the single source of truth or an unchanged reason when both levers are null,
reports `config_changed` truthfully, reports held-out first-win rate and
multi-level deepen rate, reports no regression versus the pre-integration
config, sets `verifier_is_oracle=false`, requires `parity_test_green=true`, and
emits a stable reproducibility checksum.

### REQ-ARC-WMTE-4682: Candidate-Generation Coverage CI Gate And Honest First-Win Floor

Experiment 4682 SHALL provide a deterministic CI-gate at
`python/carnot/experiment_4682_generation_coverage_cigate.py` that locks in the
candidate-generation measurement substrate for `.431` A1/A2 work. The gate
SHALL measure `candidate_generation_coverage`: whether the known winning
action or winning plan appears in the generated candidate pool before any value
head ranks it. This metric SHALL be computed from cached search traces and
known winners, not from live LLM inference and not from the executable
win-check.

The CI-gate SHALL compare a method's candidate-generation coverage against a
matched flat-search baseline over the same variant signatures. A generation
improvement claim SHALL pass only when method coverage is strictly higher than
the flat baseline, so ranking-only changes cannot be misreported as generation
progress when the winning action or plan is absent from the pool.

The gate SHALL also lock in the honest generic-first-win measurement substrate:
the standard configuration is the 25 public color01 variant signatures from
Experiment 4665/4676 with a finite action budget of 200, yielding the honest
generic first-win floor of `1/25 = 0.04`. A first-win measurement SHALL fail if
it uses a permissive harness, including a degenerate easy variant subset or an
unbounded/non-standard action budget. A standard-config measurement at or above
the 0.04 first-win floor SHALL pass.

The gate SHALL enforce a candidate-generation coverage floor so future A1/A2
changes cannot silently regress coverage below the established floor. The floor
helper SHALL be deterministic and testable on synthetic regression and honest
fixtures without live LLM inference.

Experiment 4682 SHALL write
`results/experiment_4682_generation_coverage_cigate.json` with
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `verifier_is_oracle`, `coverage_metric_added`,
`honest_firstwin_floor_added`, `coverage_floor_cigate_added`, `tests_added`,
`random_seed`, `reproducibility_checksum`, and `preconditions_checked`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success: generation_coverage_cigate_plus_honest_firstwin_floor_shipped_tests_green."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- coverage computation over cached traces + a small offline rollout (1s floor); no live_llm_inference (the CI-gate uses cached traces / a NoOp proposer)."
- `verifier_is_oracle`: principle "MUST be false -- the CI-gates guard the measurement substrate, oracle-distinct from the executable win-check."
- `coverage_metric_added`: principle "the candidate-generation-coverage metric/gate (is the winner present in the generated pool; coverage-up-vs-flat-baseline) -- the generation analog of the .430 multi-level-harness gate."
- `honest_firstwin_floor_added`: principle "the gate that a generic-first-win measurement must use the STANDARD config (variant set + action budget) -- catches a permissive harness silently inflating the 0.04 reality."
- `coverage_floor_cigate_added`: principle "the floor CI-gate that fails on a generation-coverage regression below the A1 floor."
- `tests_added`: principle "the unit tests added for all three guards (Tests Must Run and Assert: flag the winner-absent / permissive-harness / regression fixtures, pass the honest fixtures)."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4682-COVERAGE-METRIC

**Given** a cached search trace and a known winning action or plan
**When** the Exp 4682 helper scans the generated candidate pool
**Then** it reports coverage `1.0` when the winner is present and `0.0` when
the winner is absent. Given matched method and flat-baseline trace sets, the
coverage-up CI-gate passes only when method coverage exceeds flat coverage.

#### SCENARIO-ARC-WMTE-4682-HONEST-FIRSTWIN

**Given** a generic-first-win measurement
**When** the Exp 4682 honest-first-win guard validates its harness
**Then** the standard 25-variant color01 set with budget 200 and first-win rate
`0.04` passes, while a degenerate easy variant subset or an unbounded budget is
flagged as a permissive harness.

#### SCENARIO-ARC-WMTE-4682-COVERAGE-FLOOR

**Given** a candidate-generation coverage measurement and the established A1
coverage floor
**When** the Exp 4682 floor guard runs
**Then** a measurement below the floor fails with a named regression error, and
an honest measurement at or above the floor passes.

### REQ-ARC-WMTE-4694: Proposal-Coverage CI Gate And Honest First-Win Floor

Experiment 4694 SHALL provide a deterministic CI-gate at
`python/carnot/experiment_4694_proposal_coverage_cigate.py` that locks in the
L1 first-contact action-proposal measurement substrate for `.432` A1/A2 work.
The gate SHALL measure `proposal_coverage`: whether the explorer's
action-proposal distribution reaches the known winning L1 trajectory by
proposing every next action along that trajectory at the matching winning
prefix. This metric SHALL be computed from cached exploration traces and known
winning L1 trajectories, not from live LLM inference and not from the
executable win-check.

The CI-gate SHALL compare a method's proposal coverage against a matched
flat-exploration baseline over the same variant signatures. A directed
exploration improvement claim SHALL pass only when method proposal coverage is
strictly higher than the flat baseline, so search depth, value ranking, or
candidate selection changes cannot be misreported as proposal-distribution
progress when the winning L1 trajectory was never proposed.

The gate SHALL re-affirm the honest generic-first-win measurement substrate
from Experiment 4676: the standard configuration is the 25 public color01
variant signatures with a finite action budget of 200, yielding the honest
generic first-win floor of `1/25 = 0.04`. A first-win measurement SHALL fail if
it uses a permissive harness, including a degenerate easy variant subset or an
unbounded/non-standard action budget. A standard-config measurement at or
above the 0.04 first-win floor SHALL pass.

The gate SHALL enforce a proposal-coverage floor so future A1/A2 changes cannot
silently regress proposal coverage below the established floor. The floor
helper SHALL be deterministic and testable on synthetic regression and honest
fixtures without live LLM inference.

Experiment 4694 SHALL write
`results/experiment_4694_proposal_coverage_cigate.json` with
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `verifier_is_oracle`,
`proposal_coverage_metric_added`, `honest_firstwin_floor_added`,
`proposal_coverage_floor_cigate_added`, `tests_added`, `random_seed`,
`reproducibility_checksum`, and `preconditions_checked`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success: proposal_coverage_cigate_plus_honest_firstwin_floor_shipped_tests_green."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- proposal-coverage computation over cached traces + a small offline rollout (1s floor); no live_llm_inference (the CI-gate uses cached traces / a NoOp explorer)."
- `verifier_is_oracle`: principle "MUST be false -- the CI-gates guard the measurement substrate, oracle-distinct from the executable win-check."
- `proposal_coverage_metric_added`: principle "the L1-first-contact proposal-coverage metric/gate (does the explorer's proposal distribution reach the winning L1 trajectory; coverage-up-vs-flat-baseline) -- the proposal-stage analog of the .431 generation-coverage gate."
- `honest_firstwin_floor_added`: principle "the gate that a generic-first-win measurement must use the STANDARD config (variant set + action budget) -- catches a permissive harness silently inflating the 0.04 reality."
- `proposal_coverage_floor_cigate_added`: principle "the floor CI-gate that fails on a proposal-coverage regression below the A1 floor."
- `tests_added`: principle "the unit tests added for all three guards (Tests Must Run and Assert: flag the trajectory-not-proposed / permissive-harness / regression fixtures, pass the honest fixtures)."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4694-PROPOSAL-COVERAGE

**Given** a cached exploration trace and a known winning L1 trajectory
**When** the Exp 4694 helper scans the action-proposal distribution at each
winning-prefix step
**Then** it reports proposal coverage `1.0` only when every next winning action
was proposed, and `0.0` when any winning action on the trajectory was not
proposed. Given matched method and flat-baseline trace sets, the coverage-up
CI-gate passes only when method proposal coverage exceeds flat coverage.

#### SCENARIO-ARC-WMTE-4694-HONEST-FIRSTWIN

**Given** a generic-first-win measurement
**When** the Exp 4694 honest-first-win guard validates its harness
**Then** the standard 25-variant color01 set with budget 200 and first-win rate
`0.04` passes, while a degenerate easy variant subset or an unbounded budget is
flagged as a permissive harness.

#### SCENARIO-ARC-WMTE-4694-PROPOSAL-COVERAGE-FLOOR

**Given** a proposal-coverage measurement and the established A1 floor
**When** the Exp 4694 floor guard runs
**Then** a measurement below the floor fails with a named regression error, and
an honest measurement at or above the floor passes.

### REQ-ARC-WMTE-4678: Level-Up Self-Play Bank With Verifier Checkpoint

Experiment 4678 SHALL bank one new reproducible ARC public-game level through
the standing self-play loop and SHALL train plus persist the learned verifier on
the run's positive and negative traces. The selected target SHALL honor the
rotation discipline by preferring a clean L1-only game not deepened in `.426`
through `.430` and SHALL skip known stalled targets including `dc22`, `vc33`,
`ft09`, `ls20`, `sk48`, `ka59`, and `wa30`. The implemented target for this
bank is `sb26` L1->L2: it reuses the grounded color-match slot operator, reverse
engineers only the L2 nested-frame delta from the offline environment adapter,
and counts the level only when `arc_solver_kit.reproduce()` returns
`reproduced=true` and `reached_level >= 2`.

Experiment 4678 SHALL invoke the standing entrypoint
`scripts/arc_loop_solve.py --game sb26 --target-level 2`, SHALL persist
`models/arc_verifier_sb26.json`, SHALL update the `sb26` registry row with the
banked L2 win condition/action model/gotchas/dead-ends/checkpoint metadata, and
SHALL bump `reproducible_total_levels` from 59 to 60 only after the offline
reproduction gate accepts the solution. The learned verifier SHALL be
oracle-distinct: it may route or rank candidate states, but the executable
offline reproduction gate is the only source of truth for a bank.

Experiment 4678 SHALL write
`results/experiment_4678_levelup_selfplay.json` with principle-annotated
top-level fields for `honest_verdict`, `inference_substrate`,
`verifier_is_oracle`, `solve_provenance`, `offline_reproduced`,
`reproduced_levels`, `target_game`, `verifier_checkpoint_updated`,
`registry_updated`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success: <game>_L<n>_offline_reproduced OR complete: <game>_delta_identified_no_bank (honest progress, not a bank)."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade solve + verifier training, no headline LLM load (1s floor); declared so a fast real solve is not DURATION_TOO_SHORT false-flagged."
- `verifier_is_oracle`: principle "MUST be false -- the learned verifier ranks/routes the search, oracle-distinct from the executable reproduction win-check."
- `solve_provenance`: principle "development_proxy -- the offline dev twin (arc_loop_solve + a hand GameAdapter); honest that this is the registry/dev proxy, not live-agent self-discovery on a hidden game."
- `offline_reproduced`: principle "a solve not reproducible offline is wasted effort -- only reproduced levels count toward reproducible_total_levels."
- `reproduced_levels`: principle "the integer new-level count banked this task (>=1 satisfies the level-up guarantee)."
- `target_game`: principle "the rotated game attempted -- traceable to the rotation discipline (a clean game not deepened in .426-.430, not dc22/vc33/ft09/ls20/sk48/ka59/wa30)."
- `verifier_checkpoint_updated`: principle "the learned-verifier checkpoint trained on this run's pos/neg traces (the self-play self-improvement step the operator mandated every milestone)."
- `registry_updated`: principle "the per-game win-condition/action-model/gotchas/dead-ends persisted so the next attempt reuses, not re-derives."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4678-ROTATED-TARGET

**Given** the registry contains clean L1-only games, recent deepened games, and
recorded dead-end approaches
**When** Experiment 4678 selects a target
**Then** it chooses `sb26` L1->L2, records why `r11l` and `lf52` prefix-rooted
graph searches stalled, and skips `dc22`, `vc33`, `ft09`, `ls20`, `sk48`,
`ka59`, and `wa30`.

#### SCENARIO-ARC-WMTE-4678-BANK-AND-CHECKPOINT

**Given** the `sb26` L2 nested-frame color-match delta and an offline arcade
environment
**When** the standing self-play loop runs for `sb26` target level 2
**Then** the loop reaches L2, the offline reproduction gate reproduces L2, the
learned verifier checkpoint is written to `models/arc_verifier_sb26.json`, and
the artifact reports exactly one newly reproduced level.

#### SCENARIO-ARC-WMTE-4678-REGISTRY-GATE

**Given** the registry has `sb26.levels_reproduced == 1` and
`reproducible_total_levels == 59`
**When** Experiment 4678 applies a successful bank artifact
**Then** it updates only after `offline_reproduced=true`, writes the sb26 L2
win-condition/action-model/gotcha/dead-end/checkpoint metadata, and bumps the
total to 60.

### REQ-ARC-WMTE-4679: Refresh Operator-Resubmit Package From Current Registry

Experiment 4679 SHALL rebuild the ARC operator-resubmit package from the
current `ops/arc_solve_registry.yaml`, using
`results/experiment_4667_refresh_submission_package.json` and
`results/experiment_4667_submission_package_operator_resubmit.json` as the .430
A4 baseline. It SHALL fold in the current milestone's A3 bank from
`results/experiment_4678_levelup_selfplay.json` and any newly offline-reproduced
A1/A2 variant solve found in
`results/experiment_4676_hierarchical_subgoal_search_live.json` or
`results/experiment_4677_poe_world_factored_subgoal_planner.json`. The rebuilt
package SHALL enforce claimed-caps discipline: no per-game submitted claim may
exceed the current registry reproduced depth, the reproduced depth reported by
the offline gate, or the replay/adaptive path depth.

Experiment 4679 SHALL re-validate every claimed submittable level on a fresh
offline environment by requiring either a replayable banked trajectory or the
persisted `primitive_env_adaptive_resolve_operator` resolver for version-drifted
games. Any newly version-drifted game that lacks a replayable trajectory SHALL
be represented through `env_adaptive_recovery` only when the registry exposes
that primitive transfer; otherwise it SHALL be excluded rather than over-claimed.

Experiment 4679 SHALL write
`results/experiment_4679_refresh_submission_package.json` and
`results/experiment_4679_submission_package_operator_resubmit.json` with
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `verifier_is_oracle`,
`live_submittable_level_count`, `live_submittable_count_prev`, `count_delta`,
`levels_folded_in`, `refreshed_package_path`, `per_game_submittable`,
`ready_for_operator_submit`, `offline_reproduced`, `random_seed`,
`reproducibility_checksum`, and `preconditions_checked`. The
`live_submittable_count_prev` SHALL be 59, `count_delta` SHALL equal
`live_submittable_level_count - 59`, a zero delta SHALL include a
`null_delta_methodology_note`, and `ready_for_operator_submit` SHALL be true only
when the count is strictly greater than 33 and every counted row is
offline-reproduction gated. The task SHALL NOT submit to the leaderboard.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success: package_refreshed_live_submittable_<n>_above_33 OR complete: package_refreshed_unchanged_depth."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline reproduction-gated packaging, no LLM load (1s floor)."
- `verifier_is_oracle`: principle "MUST be false -- packaging + env-adaptive replay bank already-earned solves (no circular moat claim)."
- `live_submittable_level_count`: principle "the offline-reproduction-gated count of levels with a replayable trajectory or env-adaptive re-solver (the honest leaderboard score, must stay > 33)."
- `live_submittable_count_prev`: principle "59 -- the .430 A4 count, the apples-to-apples comparison."
- `count_delta`: principle "live_submittable_level_count - 59 (>=0; positive = more submittable levels folded in), emitted explicitly so a null is annotated."
- `levels_folded_in`: principle "names the games whose new banks (A3 + A1/A2 variant solves) were folded into the refreshed package this milestone."
- `refreshed_package_path`: principle "the path to the refreshed validated package the operator's live-submit driver will load -- the deliverable the operator resubmits."
- `per_game_submittable`: principle "per-game offline-reproduced level + has-trajectory + drift-robust flags -- the audit trail that no level is over-claimed above its offline depth."
- `ready_for_operator_submit`: principle "True if the refreshed package's live-submittable count beats 33 and every claim is offline-reproduction-gated; the task NEVER submits (operator-only)."
- `offline_reproduced`: principle "every claimed-submittable level must offline-reproduce to count (no frozen-trajectory over-claim)."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent registry/trajectory drift on replay."
- `preconditions_checked`: principle "records resources verified (offline arcade, registry loadable); pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4679-REGISTRY-REFRESH

**Given** the .430 A4 package is live-submittable at 59 and the current registry
contains the Exp4678 `sb26` L2 bank
**When** Experiment 4679 rebuilds from the current registry
**Then** it folds `sb26` into the refreshed package only up to the offline
reproduced depth, emits `count_delta == live_submittable_level_count - 59`, and
marks the package ready for operator submit without performing submission.

#### SCENARIO-ARC-WMTE-4679-REPLAY-ADAPTIVE-GATE

**Given** a game has a current reproduced depth but no fresh replayable
trajectory
**When** Experiment 4679 validates the package rows
**Then** the game counts only if it has a persisted env-adaptive resolver, every
counted row records the trajectory/adaptive path audit, and rows without either
path are excluded from the live-submittable count.

### REQ-ARC-WMTE-4671: Adversarial Verify L2 Goal And Multi-Level Metric Hardening

The `scripts/adversarial_verify.py` reader SHALL protect the .430 A1 L2
goal-induction claim and the multi-level solve-rate metric from two degenerate
overclaims. An ARC artifact that claims a generic agent reached L2 through
runtime goal induction, including `generic_agent_reached_level >= 2` or an L2
text claim paired with positive `reproduced_levels`, SHALL report
`goal_predicate_satisfiable=true` and `l2_plan_reaches_goal=true`. If such a
win is claimed while either field is absent or false, the reader SHALL emit a
CRITICAL flag of kind
`l2-goal-induction-without-satisfiability-check`. If such a claim omits either
field, the reader SHALL also emit a WARN flag of kind
`l2-goal-satisfiability-check-omitted`.

An ARC artifact that reports a positive `live_multi_level_solve_rate` or
`multi_level_solve_rate` SHALL also report a nondegenerate metric harness:
`metric_harness_fixed` with `target_levels >= 2` and
`break_at_first_win=false`, or equivalent top-level harness fields. If the
positive multi-level solve-rate is reported while this harness evidence is
absent or invalid, the reader SHALL emit a CRITICAL flag of kind
`multi-level-without-nondegenerate-metric`. If the harness evidence is omitted,
the reader SHALL also emit a WARN flag of kind
`multi-level-nondegenerate-metric-omitted`.

The honest .430 A1 artifact,
`results/experiment_4664_l2_goal_predicate_induction_live.json`, SHALL NOT fire
either new guard because it does not overclaim L2 success and it reports the
satisfiability and fixed-harness controls.

Experiment 4671 SHALL write
`results/experiment_4671_adversarial_verify_hardening.json` with
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `l2_goal_satisfiability_guard_added`,
`multilevel_metric_guard_added`, `honest_artifacts_not_flagged`,
`tests_added`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`.

Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: adversarial_verify_hardened_l2_goal_and_multilevel_metric_guards_tests_green."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts -- reads fixtures + edits the linter, no model load (100us floor)."
- `l2_goal_satisfiability_guard_added`: principle "the L2-GOAL-INDUCTION-WITHOUT-SATISFIABILITY-CHECK guard (an L2-via-induction win must report a satisfiable goal + reachable plan, else flagged as a possible degenerate-goal vacuous pass)."
- `multilevel_metric_guard_added`: principle "the MULTI-LEVEL-WITHOUT-NONDEGENERATE-METRIC guard (a multi-level solve-rate claim must report the fixed target_levels>=2/no-break harness, else flagged as the degenerate 0.0-by-construction metric)."
- `honest_artifacts_not_flagged`: principle "the honest A1 artifact (which reports its satisfiability + harness controls) is NOT flagged -- false-positive guard (like the .429 QD-ablation guard)."
- `tests_added`: principle "the unit tests for both guards (Tests Must Run and Assert: flag the over-claim, pass the honest)."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4671-L2-GOAL-SATISFIABILITY

**Given** an ARC artifact claims a generic-agent L2 win through runtime goal
induction
**When** `adversarial_verify.py` checks the artifact
**Then** the artifact is CRITICAL-flagged when it lacks
`goal_predicate_satisfiable=true` or `l2_plan_reaches_goal=true`,
WARN-flagged when either field is omitted, and not flagged when both controls
are reported true.

#### SCENARIO-ARC-WMTE-4671-MULTI-LEVEL-NONDEGENERATE-METRIC

**Given** an ARC artifact reports a positive multi-level solve-rate
**When** `adversarial_verify.py` checks the artifact
**Then** the artifact is CRITICAL-flagged when it lacks a fixed
`target_levels>=2` and no-break harness, WARN-flagged when that harness is
omitted, and not flagged when the fixed harness is reported.

### REQ-ARC-WMTE-4683: Adversarial Verify Subgoal And Coverage-Baseline Hardening

The `scripts/adversarial_verify.py` reader SHALL protect the `.431` A1
hierarchical-subgoal-search lever and A2 candidate-generation-coverage metric
from two degenerate overclaims. An ARC artifact that claims a generic live
agent reached a new level via hierarchical subgoal search, including positive
`reproduced_levels` or a `generic_agent_reached_level` success claim in a
subgoal-search context, SHALL report `subgoal_decomposition`,
`per_subgoal_reachable`, `no_subgoal_ablation_reached_level`,
`random_subgoal_ablation_reached_level`, and `offline_reproduced=true`.
The reported decomposition SHALL be nontrivial, per-subgoal reachability SHALL
be true, and both matched ablation levels SHALL be strictly lower than the
subgoal-search reached level. If such a win is claimed while any evidence is
absent, false, degenerate, or not strictly lower, the reader SHALL emit a
CRITICAL flag of kind `subgoal-search-without-decomposition-evidence`. If such
a claim omits any required evidence field, the reader SHALL also emit a WARN
flag of kind `subgoal-decomposition-evidence-omitted`.

An ARC artifact that claims candidate-generation coverage rose, including a
positive `candidate_generation_coverage_*` improvement claim or positive
`coverage_delta` in a generation-coverage context, SHALL report the matched
flat-search baseline field `candidate_generation_coverage_flat_baseline`. If
the coverage-up claim omits that matched baseline, the reader SHALL emit a
CRITICAL flag of kind `generation-coverage-without-baseline` and a WARN flag
of kind `generation-coverage-baseline-omitted`.

The honest `.431` artifacts,
`results/experiment_4676_hierarchical_subgoal_search_live.json` and
`results/experiment_4677_poe_world_factored_subgoal_planner.json`, SHALL NOT
fire either new guard. The former is an honest no-new-level null that reports
the subgoal/ablation fields, and the latter reports the matched flat-search
coverage baseline.

Experiment 4683 SHALL write
`results/experiment_4683_adversarial_verify_hardening.json` with
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `subgoal_decomposition_guard_added`,
`coverage_baseline_guard_added`, `honest_artifacts_not_flagged`,
`tests_added`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`.

Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: adversarial_verify_hardened_subgoal_decomposition_and_coverage_baseline_guards_tests_green."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts -- reads fixtures + edits the linter, no model load (100us floor)."
- `subgoal_decomposition_guard_added`: principle "the SUBGOAL-SEARCH-WITHOUT-DECOMPOSITION-EVIDENCE guard (a subgoal-search win must report the decomposition + per-subgoal reachability + passing no-subgoal AND random-subgoal ablations + offline_reproduced, else flagged as a possible flat-search win mislabeled)."
- `coverage_baseline_guard_added`: principle "the GENERATION-COVERAGE-WITHOUT-BASELINE guard (a coverage-up claim must report the matched flat-search baseline coverage, else flagged as unfalsifiable)."
- `honest_artifacts_not_flagged`: principle "the honest A1/A2 artifacts (which report their decomposition/ablations + flat-baseline coverage) are NOT flagged -- false-positive guard (like the .430 multi-level-metric guard)."
- `tests_added`: principle "the unit tests for both guards (Tests Must Run and Assert: flag the over-claim, pass the honest)."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4683-SUBGOAL-DECOMPOSITION

**Given** an ARC artifact claims a generic-agent new-level win through
hierarchical subgoal search
**When** `adversarial_verify.py` checks the artifact
**Then** the artifact is CRITICAL-flagged when it lacks a nontrivial
`subgoal_decomposition`, true `per_subgoal_reachable`, strictly lower
`no_subgoal_ablation_reached_level` and
`random_subgoal_ablation_reached_level`, or `offline_reproduced=true`;
WARN-flagged when any required evidence field is omitted; and not flagged when
all required evidence is reported and both ablations are lower.

#### SCENARIO-ARC-WMTE-4683-COVERAGE-BASELINE

**Given** an ARC artifact claims candidate-generation coverage rose
**When** `adversarial_verify.py` checks the artifact
**Then** the artifact is CRITICAL-flagged and WARN-flagged when it omits
`candidate_generation_coverage_flat_baseline`, and not flagged when the
matched flat-search baseline coverage is reported.

### REQ-ARC-WMTE-4695: Adversarial Verify Directed-Exploration Hardening

The `scripts/adversarial_verify.py` reader SHALL protect the `.432` A1
controllable-novelty proposal lever and A2 program-synthesis proposal-filter
lever from two degenerate overclaims. An ARC artifact that claims a generic
live agent reached a new level via controllable novelty, including positive
`reproduced_levels` or a `generic_agent_reached_level` success claim in a
controllable-novelty context, SHALL report
`no_novelty_ablation_reached_level`,
`cosmetic_novelty_ablation_reached_level`, and
`offline_reproduced=true`. Both ablation reached levels SHALL be strictly
lower than the controllable-novelty policy's `generic_agent_reached_level`.
If such a win is claimed while any evidence is absent, false, or not strictly
lower, the reader SHALL emit a CRITICAL flag of kind
`novelty-proposal-without-ablation`. If such a claim omits any required
evidence field, the reader SHALL also emit a WARN flag of kind
`novelty-proposal-ablation-omitted`.

An ARC artifact that claims program-synthesis proposal-filter
candidate-generation coverage rose, including a positive `coverage_delta`, a
positive `candidate_generation_coverage_filter` coverage-up success claim, or
filter coverage exceeding the blind baseline in a program-synthesis context,
SHALL report finite `heldout_programs_rejected` and finite
`candidate_generation_coverage_blind_baseline`. If such a coverage-up claim
does not report either required field, the reader SHALL emit a CRITICAL flag
of kind `proposal-filter-without-heldout-rejection`. If such a claim omits any
required evidence field, the reader SHALL also emit a WARN flag of kind
`proposal-filter-heldout-rejection-omitted`.

The honest `.432` artifacts,
`results/experiment_4688_controllable_novelty_proposal_policy_live.json` and
`results/experiment_4689_program_synthesis_action_effect_proposal_filter.json`,
SHALL NOT fire either new guard. The former reports the no-novelty and
cosmetic-novelty ablations plus `offline_reproduced`, and the latter reports
held-out rejected-program counts plus the matched blind-proposal coverage
baseline.

Experiment 4695 SHALL write
`results/experiment_4695_adversarial_verify_hardening.json` with
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `novelty_ablation_guard_added`,
`proposal_filter_heldout_guard_added`, `honest_artifacts_not_flagged`,
`tests_added`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`.

Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: adversarial_verify_hardened_novelty_ablation_and_proposal_filter_heldout_guards_tests_green."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts -- reads fixtures + edits the linter, no model load (100us floor)."
- `novelty_ablation_guard_added`: principle "the NOVELTY-PROPOSAL-WITHOUT-ABLATION guard (a controllable-novelty win must report the no-novelty-bonus AND cosmetic-novelty ablations strictly lower + offline_reproduced, else flagged as a possible flat-exploration win mislabeled / controllability-gate-adds-nothing)."
- `proposal_filter_heldout_guard_added`: principle "the PROPOSAL-FILTER-WITHOUT-HELDOUT-REJECTION guard (a coverage-up claim must report the held-out rejected-program count AND the matched blind-proposal baseline, else flagged as unfalsifiable / experts_overfit_prefix-unguarded)."
- `honest_artifacts_not_flagged`: principle "the honest A1/A2 artifacts (which report their ablations + held-out rejection + blind-baseline coverage) are NOT flagged -- false-positive guard (like the .431 coverage-baseline guard)."
- `tests_added`: principle "the unit tests for both guards (Tests Must Run and Assert: flag the over-claim, pass the honest)."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4695-NOVELTY-ABLATION

**Given** an ARC artifact claims a generic-agent new-level win through
controllable novelty
**When** `adversarial_verify.py` checks the artifact
**Then** the artifact is CRITICAL-flagged when it lacks strictly lower
`no_novelty_ablation_reached_level` and
`cosmetic_novelty_ablation_reached_level` values or
`offline_reproduced=true`; WARN-flagged when any required evidence field is
omitted; and not flagged when all required evidence is reported and both
ablations are lower.

#### SCENARIO-ARC-WMTE-4695-PROPOSAL-FILTER-HELDOUT

**Given** an ARC artifact claims program-synthesis proposal-filter
candidate-generation coverage rose
**When** `adversarial_verify.py` checks the artifact
**Then** the artifact is CRITICAL-flagged when it lacks finite
`heldout_programs_rejected` or
`candidate_generation_coverage_blind_baseline`; WARN-flagged when either
required evidence field is omitted; and not flagged when both the held-out
rejected-program count and matched blind-proposal baseline coverage are
reported.

### REQ-ARC-WMTE-4707: Adversarial Verify First-Win Null And Perception Hardening

The `scripts/adversarial_verify.py` reader SHALL protect the `.433`
held-out-first-win readiness lane and object-centric perception lane from two
degenerate classifications. A top-level equality between
`first_win_baseline` and `first_win_rate_integrated` SHALL be treated as an
honest null-delta only when the artifact also reports a non-empty
`null_delta_methodology_note`, `positive_control_passed=true`, and an explicit
zero `first_win_delta_vs_baseline`. In that case the TAUTOLOGY flag SHALL be
downgraded from CRITICAL to WARN. If either null marker is absent, false, or
the zero delta is absent, the equality SHALL remain CRITICAL.

An ARC artifact that claims an object-centric or relational representation
lifted first-win, reached a new level, or reports `reproduced_levels >= 1` in
an object-centric perception context SHALL report
`order1_ablation_reached_level` strictly lower than
`generic_agent_reached_level` and `offline_reproduced=true`. If such a
perception-attributable win is claimed while either field is omitted, the
reader SHALL emit a WARN flag of kind `perception-overclaim-omitted`. If the
order-1 ablation is absent, not strictly lower, or the offline reproduction
gate is absent/false, the reader SHALL emit a CRITICAL flag of kind
`perception-overclaim`.

The honest `.433` artifacts SHALL NOT fire the new guarded criticals:
`results/experiment_4691_held_out_first_win_readiness.json` when it reports
the first-win null-delta markers, and
`results/experiment_4700_object_centric_perception_proposal_live.json` when it
reports its order-1 ablation evidence and does not make a reproduced
perception win claim.

Experiment 4707 SHALL write
`results/experiment_4707_adversarial_verify_hardening.json` with
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `firstwin_nulldelta_carveout_added`,
`perception_overclaim_guard_added`, `honest_artifacts_not_flagged`,
`tests_added`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`.

Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: adversarial_verify_hardened_firstwin_nulldelta_and_perception_overclaim_guards_tests_green."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts -- reads fixtures + edits the linter, no model load (100us floor)."
- `firstwin_nulldelta_carveout_added`: principle "the HELD-OUT-FIRST-WIN-lane null-delta carve-out (a flat first-win equality -> WARN only with null_delta_methodology_note + positive_control_passed; unvalidated -> stays CRITICAL) -- the exp4691 false-positive fix."
- `perception_overclaim_guard_added`: principle "the PERCEPTION-OVERCLAIM guard (a perception-attributable win must report the order-1-representation ablation strictly lower + offline_reproduced, else flagged as a possible search-budget win mislabeled)."
- `honest_artifacts_not_flagged`: principle "the honest A1/A4 artifacts (A1 order-1 ablation + offline_reproduced; A4 null-delta markers) are NOT flagged -- false-positive guard (like the .432 coverage-baseline guard)."
- `tests_added`: principle "the unit tests for both guards (Tests Must Run and Assert: flag the over-claim/unvalidated, pass the honest)."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4707-FIRSTWIN-NULLDELTA

**Given** a held-out-first-win artifact reports a flat
`first_win_baseline == first_win_rate_integrated`
**When** `adversarial_verify.py` checks the artifact
**Then** the equality is WARN-flagged when a non-empty
`null_delta_methodology_note`, `positive_control_passed=true`, and
`first_win_delta_vs_baseline==0` are present; otherwise it remains
CRITICAL-flagged.

#### SCENARIO-ARC-WMTE-4707-PERCEPTION-OVERCLAIM

**Given** an ARC artifact claims an object-centric or relational
representation produced a first-win or new-level result
**When** `adversarial_verify.py` checks the artifact
**Then** the claim is CRITICAL-flagged unless
`order1_ablation_reached_level` is strictly lower than
`generic_agent_reached_level` and `offline_reproduced=true`; omitted evidence
also emits a WARN; and honest null artifacts with order-1 evidence do not fire
the new guard.

### REQ-ARC-WMTE-4644: Persist Graded Goal-Energy Primitive And Measure Untuned Transfer

Experiment 4644 SHALL consolidate the milestone's best-characterized A1/A2
primitive into reusable solver scaffolding. If A1's Exp4020 graded
goal-satisfaction energy has the strongest clean signal, or if all upstreams
are value-null and A1 is the best-characterized primitive-as-built, the workflow
SHALL persist `graded_goal_energy_search_heuristic_operator` in
`arc_solver_kit` and add or extend the matching
`primitive_graded_goal_energy_search_heuristic_operator` `general_gotchas`
registry row without duplicating existing primitive entries. If A2's
action-effect expansion prior is the stronger clean signal, the workflow SHALL
record why the existing action-effect primitive was extended instead. The
persisted primitive SHALL be oracle-distinct from the win-check: it may
rank/generated frontier candidates by convex
`alpha * navigation + beta * graded_goal_energy`, but it SHALL NOT call the
executable verifier to score candidates.

Experiment 4644 SHALL apply the persisted primitive to at least two games that
were not used to tune the selected upstream primitive, using cached live/offline
candidate measurements when live quota is unavailable. For every transfer game
it SHALL report solve-rate delta, first-win delta, and action-efficiency lift
where those quantities are measurable. Any newly claimed level SHALL pass
`arc_solver_kit.reproduce`; only reproduced new levels count toward
`reproducible_total_levels`. If no transfer game improves, the artifact SHALL
persist the primitive-as-built, report
`complete: primitive_persisted_transfer_null_characterized`, and record the
transfer dead-ends in the registry.

The terminal artifact
`results/experiment_4644_primitive_persist_transfer.json` SHALL include
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `verifier_is_oracle`, `solve_provenance`,
`primitive_persisted`, `transfer_games`, `transfer_value_per_game`,
`offline_reproduced`, `registry_updated`, `random_seed`,
`reproducibility_checksum`, and `preconditions_checked`.

Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: primitive_persisted_transfer_<game>_value_added OR complete: primitive_persisted_transfer_null_characterized."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates for the offline transfer; declared so a fast real run is not DURATION_TOO_SHORT/METHODOLOGY false-flagged."
- `verifier_is_oracle`: principle "MUST be false -- the persisted primitive directs goal-energy generation or prunes/expands actions, oracle-distinct from the win-check."
- `solve_provenance`: principle "development_proxy if a transfer solve is via the offline twin; live_agent_self_discovery if the persisted primitive improves the SCORED agent's own path. NOT outer_loop_re."
- `primitive_persisted`: principle "names the arc_solver_kit operator + registry general_gotcha id added/extended -- the reusable asset (Solver-Reuse Discipline); without it the A1/A2 effort is wasted per the ARC reuse rule."
- `transfer_games`: principle "the games the primitive was applied to (NOT tuned on) -- the generalization test."
- `transfer_value_per_game`: principle "the per-game value-add (live solve-rate / first-win / action-efficiency lift) -- the cross-game evidence the primitive generalizes."
- `offline_reproduced`: principle "only offline-reproduced new levels count toward reproducible_total_levels."
- `registry_updated`: principle "the primitive + transfer dead-ends persisted so the next milestone reuses, not re-derives."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4644

**Given** the A1 and A2 live artifacts are present, `offline_arcade()` imports,
the `arc-world-model-trust-energy` spec declares `REQ-ARC-WMTE-4644`, and the
registry contains the persisted primitive gotcha
**When** Experiment 4644 selects the strongest clean upstream primitive, runs
the reusable operator on untuned transfer games, and validates any newly claimed
level through `arc_solver_kit.reproduce`
**Then** the artifact reports either a value-added transfer success for the
first improving game or an honest characterized null, keeps
`verifier_is_oracle=false`, names the operator and registry gotcha, records
per-game value/dead-end details, and writes a stable checksum.

### REQ-ARC-WMTE-4690: Rotated Level-Up Self-Play Bank and Checkpoint

Experiment 4690 SHALL run the standing ARC self-play loop on a rotated clean
public target that was not deepened in experiments .427-.431 and is not one of
`sb26`, `dc22`, `vc33`, `ft09`, `ls20`, `sk48`, `ka59`, or `wa30`. The target
selection SHALL prefer `bp35`, `re86`, `s5i5`, `g50t`, `r11l`, or `lf52` and
SHALL record skipped stalled approaches from the registry before selecting the
first target with a grounded next-level delta. For this sprint the selected
target is `lf52`, deepened from L1 to L2 through a hand GameAdapter that
captures only the next-level rail-carried peg-jump delta derived from the
offline environment.

The experiment SHALL verify `arc_solver_kit.offline_arcade()` before running,
invoke `scripts/arc_loop_solve.py --game lf52 --target-level 2 --no-hazard-prune`
unless a bankable cached loop artifact already exists, and SHALL count progress
only when `arc_solver_kit.reproduce` reports `reproduced=true` and the reached
level exceeds the prior registry level. On success it SHALL update
`ops/arc_solve_registry.yaml` for `lf52`, bump `reproducible_total_levels` from 60 to 61, persist the learned verifier checkpoint at
`models/arc_verifier_lf52.json`, and write
`results/experiment_4690_levelup_selfplay.json`.

The terminal artifact
`results/experiment_4690_levelup_selfplay.json` SHALL include
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `verifier_is_oracle`, `solve_provenance`,
`offline_reproduced`, `reproduced_levels`, `target_game`,
`verifier_checkpoint_updated`, `registry_updated`, `random_seed`,
`reproducibility_checksum`, and `preconditions_checked`.

Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: <game>_L<n>_offline_reproduced OR complete: <game>_delta_identified_no_bank (honest progress, not a bank)."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade solve + verifier training, no headline LLM load (1s floor); declared so a fast real solve is not DURATION_TOO_SHORT false-flagged."
- `verifier_is_oracle`: principle "MUST be false -- the learned verifier ranks/routes the search, oracle-distinct from the executable reproduction win-check."
- `solve_provenance`: principle "development_proxy -- the offline dev twin (arc_loop_solve + a hand GameAdapter); honest that this is the registry/dev proxy, not live-agent self-discovery on a hidden game."
- `offline_reproduced`: principle "a solve not reproducible offline is wasted effort -- only reproduced levels count toward reproducible_total_levels."
- `reproduced_levels`: principle "the integer new-level count banked this task (>=1 satisfies the level-up guarantee)."
- `target_game`: principle "the rotated game attempted -- traceable to the rotation discipline (a clean game not deepened in .427-.431, not sb26/dc22/vc33/ft09/ls20/sk48/ka59/wa30)."
- `verifier_checkpoint_updated`: principle "the learned-verifier checkpoint trained on this run's pos/neg traces (the self-play self-improvement step the operator mandated every milestone)."
- `registry_updated`: principle "the per-game win-condition/action-model/gotchas/dead-ends persisted so the next attempt reuses, not re-derives."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4690-ROTATED-TARGET

**Given** the registry has `lf52` reproduced only to L1, `sb26/dc22/vc33/ft09/ls20`
were recently deepened or skipped by the operator, and `sk48/ka59/wa30` are
forbidden for this rotation
**When** Experiment 4690 selects a target
**Then** it selects `lf52`, records the skipped stalled or forbidden targets,
and marks the solve provenance as the development proxy.

#### SCENARIO-ARC-WMTE-4690-BANK-AND-CHECKPOINT

**Given** the `lf52` GameAdapter exposes the known L1 seed and the L2
rail-carried peg-jump tail
**When** the standing loop replays the adapter through the offline reproduction
gate
**Then** it reaches L2, reports `offline_reproduced=true`, banks exactly one new
level, and updates `models/arc_verifier_lf52.json` from positive and negative
self-play traces while keeping `verifier_is_oracle=false`.

#### SCENARIO-ARC-WMTE-4690-REGISTRY-GATE

**Given** the prior registry total is 60 and the `lf52` row is L1
**When** the reproduction gate reaches L2 and the learned verifier checkpoint is
written
**Then** the registry persists the L2 win condition, action model, gotchas, and
dead-ends, bumps `reproducible_total_levels` to 61, and the artifact schema
validates with a stable checksum.

### REQ-ARC-WMTE-4702: Level-Up Self-Play Bank and Learned-Verifier Checkpoint

Experiment 4702 SHALL run the standing ARC self-play loop on a rotated clean
public target that was not deepened in experiments .428-.432 and is not one of
`lf52`, `sb26`, `dc22`, `vc33`, `ft09`, `ls20`, `sk48`, `ka59`, or `wa30`.
The target selection SHALL prefer `bp35`, `re86`, `s5i5`, `g50t`, or `r11l`,
SHALL check registry dead-ends before selecting, and SHALL record any stalled
approaches tried during this run. For this sprint the selected target is `re86`,
deepened from L1 to L2 by reusing the generic sprite-overlay resize verifier on
the post-L1 environment state and registering only the next-level action delta
inside the standing `GameAdapter`.

The experiment SHALL verify `arc_solver_kit.offline_arcade()` before running,
invoke
`scripts/arc_loop_solve.py --game re86 --target-level 2 --no-hazard-prune`
unless a bankable cached loop artifact already exists, and SHALL count progress
only when `arc_solver_kit.reproduce` reports `reproduced=true` and the reached
level exceeds the prior registry level. On success it SHALL update
`ops/arc_solve_registry.yaml` for `re86`, bump `reproducible_total_levels` from
61 to 62, persist the learned verifier checkpoint at
`models/arc_verifier_re86.json`, and write
`results/experiment_4702_levelup_selfplay.json`.

The terminal artifact
`results/experiment_4702_levelup_selfplay.json` SHALL include
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `verifier_is_oracle`, `solve_provenance`,
`offline_reproduced`, `reproduced_levels`, `target_game`,
`verifier_checkpoint_updated`, `registry_updated`, `random_seed`,
`reproducibility_checksum`, and `preconditions_checked`.

Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: <game>_L<n>_offline_reproduced OR complete: <game>_delta_identified_no_bank (honest progress, not a bank)."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline arcade solve + verifier training, no headline LLM load (1s floor); declared so a fast real solve is not DURATION_TOO_SHORT false-flagged."
- `verifier_is_oracle`: principle "MUST be false -- the learned verifier ranks/routes the search, oracle-distinct from the executable reproduction win-check."
- `solve_provenance`: principle "development_proxy -- the offline dev twin (arc_loop_solve + a hand GameAdapter); honest that this is the registry/dev proxy, not live-agent self-discovery on a hidden game."
- `offline_reproduced`: principle "a solve not reproducible offline is wasted effort -- only reproduced levels count toward reproducible_total_levels."
- `reproduced_levels`: principle "the integer new-level count banked this task (>=1 satisfies the level-up guarantee)."
- `target_game`: principle "the rotated game attempted -- traceable to the rotation discipline (a clean game not deepened in .428-.432, not lf52/sb26/dc22/vc33/ft09/ls20/sk48/ka59/wa30)."
- `verifier_checkpoint_updated`: principle "the learned-verifier checkpoint trained on this run's pos/neg traces (the self-play self-improvement step the operator mandated every milestone)."
- `registry_updated`: principle "the per-game win-condition/action-model/gotchas/dead-ends persisted so the next attempt reuses, not re-derives."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified; pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4702-ROTATED-TARGET

**Given** the registry has `re86` reproduced only to L1, `lf52/sb26/dc22/vc33/ft09/ls20`
were recently deepened or skipped by the operator, `sk48/ka59/wa30` are forbidden
for this rotation, and `bp35` has no grounded next-level adapter
**When** Experiment 4702 selects a target
**Then** it selects `re86`, records the skipped stalled or forbidden targets,
and marks the solve provenance as the development proxy.

#### SCENARIO-ARC-WMTE-4702-BANK-AND-CHECKPOINT

**Given** the `re86` GameAdapter exposes the known L1 sprite-overlay seed and
the verifier-derived L2 overlay tail
**When** the standing loop replays the adapter through the offline reproduction
gate
**Then** it reaches L2, reports `offline_reproduced=true`, banks exactly one new
level, and updates `models/arc_verifier_re86.json` from positive and negative
self-play traces while keeping `verifier_is_oracle=false`.

#### SCENARIO-ARC-WMTE-4702-REGISTRY-GATE

**Given** the prior registry total is 61 and the `re86` row is L1
**When** the reproduction gate reaches L2 and the learned verifier checkpoint is
written
**Then** the registry persists the L2 win condition, action model, gotchas, and
dead-ends, bumps `reproducible_total_levels` to 62, and the artifact schema
validates with a stable checksum.

### REQ-ARC-WMTE-4714: BP35 Level-Up Self-Play Bank and Learned-Verifier Checkpoint

Experiment 4714 SHALL run the standing ARC self-play loop on a rotated clean
public target that was not deepened in experiments .429-.433 and is not one of
`lf52`, `sb26`, `dc22`, `vc33`, the `.433` A3 target, or failed `sk48`. The
target selection SHALL check `ops/arc_solve_registry.yaml` dead-ends before
selecting, SHALL reject duplicate registry levels, and SHALL prefer the hard
clean L1 targets `bp35`, `re86`, `s5i5`, `g50t`, or `r11l`. For this sprint the
selected target is `bp35`, deepened from L1 to L2 by a BP35 GameAdapter delta
that replays the existing L1 goal-directed seed, clears same-row blockers on the
L2 platform route, and reaches the L2 goal through the offline reproduction gate.

The experiment SHALL verify `arc_solver_kit.offline_arcade()` before running,
verify `scripts/arc_loop_solve.py` is runnable, invoke
`scripts/arc_loop_solve.py --game bp35 --target-level 2 --no-hazard-prune`
unless a bankable cached loop artifact already exists, and SHALL count progress
only when `arc_solver_kit.reproduce` reports `reproduced=true` and the reached
level exceeds the prior registry level. On success it SHALL update
`ops/arc_solve_registry.yaml` for `bp35`, bump `reproducible_total_levels` from
62 to 63, persist the learned verifier checkpoint at
`models/arc_verifier_bp35.json`, and write
`results/experiment_4714_levelup_selfplay.json`.

The terminal artifact
`results/experiment_4714_levelup_selfplay.json` SHALL include
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `offline_reproduced`, `reproduced_levels`,
`new_levels_banked`, `reproducible_total_levels`, `verifier_checkpoint`,
`verifier_is_oracle`, `solve_provenance`, `registry_precheck_passed`,
`random_seed`, `reproducibility_checksum`, and `preconditions_checked`.

Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: <game>_L<n>_offline_reproduced OR complete: <game>_delta_identified_no_bank."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- arc_loop_solve scores against the offline sim (1s floor), no live LLM load."
- `offline_reproduced`: principle "only offline-reproduced levels count (ARC Solve Reproducibility); a live-only trajectory is provisional."
- `reproduced_levels`: principle "the integer level reached on the target game."
- `new_levels_banked`: principle ">=1 is the Level-Up Guarantee met; 0 rotates the target next milestone."
- `reproducible_total_levels`: principle "the registry header after this run (62->63+ if banked) -- the monotonic north-star metric."
- `verifier_checkpoint`: principle "models/arc_verifier_<game>.json -- the self-play loop trains+checkpoints the learned verifier (the self-improvement-every-milestone mandate)."
- `verifier_is_oracle`: principle "MUST be false -- the learned verifier routes/ranks; the executable reproduction gate is the oracle-distinct authority."
- `solve_provenance`: principle "live_agent_self_discovery for a generic bank, OR development_proxy honestly if a GameAdapter delta was required."
- `registry_precheck_passed`: principle "confirms the target level is NOT already in the registry (a duplicate is a CRITICAL adversarial flag)."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent harness/corpus drift on replay."
- `preconditions_checked`: principle "records resources verified (offline arcade, arc_loop_solve runnable); pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4714-ROTATED-TARGET

**Given** the registry records `bp35` reproduced only to L1, the operator skip
list excludes the recently deepened targets, and the `bp35` row contains prior
dead-end notes about missing grounded next-level adapters
**When** Experiment 4714 selects a target
**Then** it selects `bp35`, records the prior dead-ends as prechecked rather
than duplicated, and marks `registry_precheck_passed=true`.

#### SCENARIO-ARC-WMTE-4714-BANK-AND-CHECKPOINT

**Given** the `bp35` GameAdapter exposes the known L1 seed and the grounded L2
block-clear navigation tail
**When** the standing loop replays the adapter through the offline reproduction
gate
**Then** it reaches L2, reports `offline_reproduced=true`, banks exactly one new
level, writes `models/arc_verifier_bp35.json`, and keeps
`verifier_is_oracle=false`.

#### SCENARIO-ARC-WMTE-4714-REGISTRY-GATE

**Given** the prior registry total is 62 and the `bp35` row is L1
**When** the reproduction gate reaches L2 and the learned verifier checkpoint is
written
**Then** the registry persists the L2 win condition, action model, gotchas, and
dead-ends, bumps `reproducible_total_levels` to 63, and the artifact schema
validates with a stable checksum.

#### SCENARIO-ARC-WMTE-4714-NO-DUPLICATE

**Given** the registry already records `bp35` at L2 or above
**When** Experiment 4714 performs the registry precheck
**Then** it refuses to re-bank the same level and reports the duplicate as a
critical registry-precheck failure rather than producing a success artifact.

### REQ-ARC-WMTE-4728: Rotated Level-Up Self-Play Bank and Learned-Verifier Checkpoint

Experiment 4728 SHALL run the standing ARC self-play loop on a rotated target
that is not one of the .429-.434 recently deepened or failed targets and is not
already recorded at the requested level in `ops/arc_solve_registry.yaml`. The
target selection SHALL precheck `dead_ends` for `re86`, `s5i5`, `g50t`, and
`r11l`, SHALL record `g50t`'s distance-12 L2 stall as a no-bank dead-end when
that path does not reproduce a new level, and MAY rotate to `ar25` L2->L3 when
the preferred hard clean L1 targets are already banked or stalled.

The experiment SHALL verify `arc_solver_kit.offline_arcade()` before running,
verify `scripts/arc_loop_solve.py` is runnable, invoke
`scripts/arc_loop_solve.py --game ar25 --target-level 3 --no-hazard-prune`
unless a bankable cached loop artifact already exists, and SHALL count progress
only when `arc_solver_kit.reproduce` reports `reproduced=true` and the reached
level exceeds the prior registry level. On success it SHALL update
`ops/arc_solve_registry.yaml` for `ar25`, bump `reproducible_total_levels` from
63 to 64, persist the learned verifier checkpoint at
`models/arc_verifier_ar25.json`, and write
`results/experiment_4728_levelup_selfplay.json`.

The `ar25` L3 tail SHALL be a grounded continuation of the existing L1+L2
reflection adapter: move the horizontal reflection axis from `y=16` to `y=9`,
move the L-shaped object to `(11,14)`, and move the two-row object to `(3,14)`
so the executable `vplrhaovhr()` reflection coverage predicate advances to L3.

The terminal artifact
`results/experiment_4728_levelup_selfplay.json` SHALL include
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `offline_reproduced`, `reproduced_levels`,
`new_levels_banked`, `reproducible_total_levels`, `verifier_checkpoint`,
`verifier_is_oracle`, `solve_provenance`, `registry_precheck_passed`,
`random_seed`, `reproducibility_checksum`, and `preconditions_checked`.

Required field principles SHALL be included for:

- `honest_verdict`: principle "terminal prefix; success: <game>_L<n>_offline_reproduced OR complete: <game>_delta_identified_no_bank."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- arc_loop_solve scores against the offline sim (1s floor), no live LLM load."
- `offline_reproduced`: principle "only offline-reproduced levels count (ARC Solve Reproducibility); a live-only trajectory is provisional."
- `reproduced_levels`: principle "the integer level reached on the target game."
- `new_levels_banked`: principle ">=1 is the Level-Up Guarantee met; 0 rotates the target next milestone."
- `reproducible_total_levels`: principle "the registry header after this run (63->64+ if banked) -- the monotonic north-star metric."
- `verifier_checkpoint`: principle "models/arc_verifier_<game>.json -- the self-play loop trains+checkpoints the learned verifier (the self-improvement-every-milestone mandate)."
- `verifier_is_oracle`: principle "MUST be false -- the learned verifier routes/ranks; the executable reproduction gate is the oracle-distinct authority."
- `solve_provenance`: principle "live_agent_self_discovery for a generic bank, OR development_proxy honestly if a GameAdapter delta was required."
- `registry_precheck_passed`: principle "confirms the target level is NOT already in the registry (a duplicate is a CRITICAL adversarial flag)."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent harness/corpus drift on replay."
- `preconditions_checked`: principle "records resources verified (offline arcade, arc_loop_solve runnable); pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4728-ROTATED-TARGET

**Given** `re86` is already L2, `s5i5` and `g50t` carry stalled L2 dead-end
notes, `r11l` is still prefix-rooted at L1, and `ar25` is L2
**When** Experiment 4728 selects a target
**Then** it records those prechecks, selects `ar25` for L3, and marks
`registry_precheck_passed=true`.

#### SCENARIO-ARC-WMTE-4728-BANK-AND-CHECKPOINT

**Given** the `ar25` GameAdapter exposes the L1 seed, L2 mirror/object delta,
and L3 horizontal-axis plus object-placement tail
**When** the standing loop replays the adapter through the offline reproduction
gate
**Then** it reaches L3, reports `offline_reproduced=true`, banks exactly one new
level, writes `models/arc_verifier_ar25.json`, and keeps
`verifier_is_oracle=false`.

#### SCENARIO-ARC-WMTE-4728-REGISTRY-GATE

**Given** the prior registry total is 63 and the `ar25` row is L2
**When** the reproduction gate reaches L3 and the learned verifier checkpoint is
written
**Then** the registry persists the L3 win condition, action model, gotchas, and
dead-ends, bumps `reproducible_total_levels` to 64, and the artifact schema
validates with a stable checksum.

### REQ-ARC-WMTE-4709: Structured-World-Model SOTA Ingestion For .434

Experiment 4709 SHALL map the `.434` fallback after the `.433` A1 object-centric
perception arm and A2 amortized-prior plus Go-Explore arm fail to make a
winner appear at L1. The workflow SHALL read
`results/experiment_4697_sota_ingestion_amortized_exploration.json`,
`docs/research-notes/amortized-exploration-sota-ingestion-2026-06-24.md`,
`results/experiment_4700_object_centric_perception_proposal_live.json`,
`results/experiment_4701_amortized_exploration_prior_go_explore_live.json`,
`python/carnot/agentic/arc_executable_world_model.py`,
`research-studying.md`, and `research-references.md` before scoping methods.
The synthesis SHALL explicitly carry forward the `.433` residuals
`object_centric_perception_no_new_level_residual_offpath_calibration_insufficient`
and `amortized_prior_go_explore_no_coverage_gain_residual_logged`, plus the
deeper wall: if perception and amortized exploration do not surface the winning
prefix, the explorer needs an induced structured, executable object-relational transition model
that it can plan in and actively probe.

The workflow SHALL use only the reliable channel:
`scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`, low-concurrency
WebSearch/WebFetch of the top five to eight papers, and direct arXiv HTTP
checks. It SHALL NOT invoke `/deep-research`. If
`https://huggingface.co/api/models` or
`.venv/bin/python scripts/sweep_clusters.py --help` fails, the workflow SHALL
report `blocked_network` or `blocked_sweep_clusters` rather than fabricate
citations.

Experiment 4709 SHALL write
`docs/research-notes/structured-world-model-active-probing-sota-ingestion-2026-06-25.md`
and `results/experiment_4709_sota_ingestion_structured_world_model.json`. The
artifact SHALL include `honest_verdict`, `inference_substrate`,
`deep_research_not_used`, `methods_mapped`, `citations_verified`,
`flagged_for_next_roadmap`, `note_path`, `preconditions_checked`,
`random_seed`, and `field_principles`. The success verdict SHALL be
`success: sota_ingestion_structured_world_model_mapped`, the substrate SHALL be
`aggregation_from_upstream_artifacts`, and `deep_research_not_used` SHALL be
true. Each mapped method SHALL cite real arXiv IDs and SHALL include the
implementation cost over the current live E3 explorer,
`arc_executable_world_model`, A1 object-centric perception, and A2 amortized
prior plus Go-Explore stack, plus a concrete `fails_when` condition and a
residual-scope note. No method claim is valid without a verified arXiv ID.

Required field principles SHALL include:

- `honest_verdict`: principle `terminal prefix; success: sota_ingestion_structured_world_model_mapped.`
- `inference_substrate`: principle `aggregation_from_upstream_artifacts -- literature read + synthesis, no model load (100us floor).`
- `deep_research_not_used`: principle `MUST be true -- /deep-research is BANNED in the autonomous loop; used sweep helpers + low-concurrency WebSearch/WebFetch.`
- `methods_mapped`: principle `the strongest 3-5 SOTA methods with REAL arXiv IDs + per-method implement-cost-over-current-stack + fails_when (no citation = fabrication).`
- `citations_verified`: principle `each cited arXiv ID with an HTTP-200 verification -- pre-empts fabricated citations.`
- `flagged_for_next_roadmap`: principle `the strongest method(s) flagged as candidate .434 inputs (flagged_for_v434) -- closes discover->ingest->plan->experiment.`
- `note_path`: principle `the per-track research-note path (the SOTA-Ingestion Cycle deliverable).`
- `preconditions_checked`: principle `records network reachability verified; pre-empts fabricated citations.`

### SCENARIO-ARC-WMTE-4709: Structured World Models And Active Probing Map Onto .434 Inputs

Given the network and sweep-helper preconditions succeed and the `.433` A1/A2
residual artifacts are readable
When experiment 4709 ingests focused papers for object-centric structured world
models, program-induced executable transition models, MCTS planning in learned
object models, epistemic uncertainty planning, and active hypothesis testing
Then the note and JSON artifact map three to five strongest methods onto the
current live E3 explorer / `arc_executable_world_model` / A1 object-centric
perception / A2 amortized-prior plus Go-Explore stack, cite real arXiv IDs
including `2410.08822`, `2511.02225`, `2601.06604`, `2511.06136`,
`2307.02427`, `2210.13455`, `2506.01876`, and `2309.08477`, flag one or more
`flagged_for_v434` roadmap candidates, record all network and citation
preconditions checked, record the `.433` null residuals and structured
world-model / active-probing next-wall scope, and record that `/deep-research`
was not used.

### REQ-ARC-WMTE-4722: Active-Probe World-Model SOTA Ingestion For .435

Experiment 4722 SHALL map the next `.435` frontier after `.434` perception
and surfacing work: active-probe / hypothesis-driven world-model induction.
The workflow SHALL read
`results/experiment_4709_sota_ingestion_structured_world_model.json`,
`research-references.md`,
`python/carnot/agentic/arc_competition_agent.py`, and
`python/carnot/agentic/arc_executable_world_model.py` before scoping methods.
It SHALL treat the `.433`/`.434` structured-world-model work as upstream input
and SHALL NOT duplicate `flagged_for_v434` as the roadmap result. The synthesis
SHALL map the strongest active-probe methods onto the current
`E3AgentPolicy` and `arc_executable_world_model` stack, including what each
method would take to implement and where each fails.

The workflow SHALL use the reliable low-concurrency channel: a network
reachability check for `https://arxiv.org`, focused WebSearch/WebFetch over
the top five to eight papers, and direct arXiv URL verification. It SHALL NOT
invoke `/deep-research`. If arXiv is unreachable, the workflow SHALL report
`blocked_network` rather than fabricate citations or method claims.

Experiment 4722 SHALL write
`docs/research-notes/active-probe-world-model-sota-ingestion-20260625.md`
and `results/experiment_4722_sota_ingestion_active_probe_world_model.json`.
The artifact SHALL include `honest_verdict`, `inference_substrate`,
`methods_mapped`, `citations`, `flagged_for_next_roadmap`, `note_path`,
`verifier_is_oracle`, `random_seed`, `reproducibility_checksum`,
`preconditions_checked`, and `field_principles`. The success verdict SHALL be
`success: sota_ingestion_active_probe_world_model_mapped`, the substrate SHALL
be `aggregation_from_upstream_artifacts`, and `verifier_is_oracle` SHALL be
false. Each mapped method SHALL cite real arXiv IDs/URLs and SHALL include
`maps_to_current_stack`, `implement_cost_over_current_stack`, and
`fails_when`. No solve claim, training claim, model-load claim, or leaderboard
claim is valid in this literature-ingestion artifact.

Required field principles SHALL include:

- `honest_verdict`: principle `terminal prefix; success: sota_ingestion_active_probe_world_model_mapped.`
- `inference_substrate`: principle `aggregation_from_upstream_artifacts -- literature synthesis + WebFetch, no model load (100us floor).`
- `methods_mapped`: principle `the 3-5 strongest methods, each with maps_to_current_stack + implement_cost_over_current_stack + fails_when -- the actionable ingestion (discover -> ingest -> plan).`
- `citations`: principle `real arXiv IDs/URLs for every method claim -- an ingestion with no verifiable citations is fabrication per adversarial_verify discipline.`
- `flagged_for_next_roadmap`: principle `the strongest method(s) flagged_for_v435 -- closes the discover->ingest->plan loop so SOTA flows into the next milestone's experiments.`
- `note_path`: principle `docs/research-notes/active-probe-world-model-sota-ingestion-20260625.md -- the human-readable per-track synthesis.`
- `verifier_is_oracle`: principle `false -- a literature synthesis invokes no oracle.`
- `random_seed`: principle `determinism precondition (the search/synthesis seed).`
- `reproducibility_checksum`: principle `content-addressed hash of the ingested source set.`
- `preconditions_checked`: principle `records the network-reachability check; pre-empts missing-resource fabrication.`

### SCENARIO-ARC-WMTE-4722: Active Probes Map The Next World-Model Frontier

Given arXiv is reachable and the `.433` structured-world-model ingestion plus
the `.434` research-reference pre-sweep are readable
When experiment 4722 ingests focused papers for active sequential hypothesis
testing, epistemic planning, active world-model learning, object-interaction
world models, causality-aware object-world-model control, object-model MCTS,
and object-world-model policy breakage
Then the note and JSON artifact map three to five methods onto
`E3AgentPolicy` and `arc_executable_world_model`, cite real arXiv IDs including
`2007.07853`, `2210.13455`, `2309.08477`, `2506.01876`, `2511.02225`,
`2511.06136`, `2511.14262`, and `2601.06604`, flag one or more
`flagged_for_v435` candidates, record that `/deep-research` was not invoked,
record `verifier_is_oracle=false`, and emit no ARC solve claim.

### REQ-ARC-WMTE-4727: Live Hypothesis-Posterior Active Probe Controller

The live `E3AgentPolicy` induction path SHALL support an active-probe
controller that maintains a small posterior over candidate goal/dynamics
hypotheses, asks `arc_executable_world_model` for each hypothesis' predicted
transition under candidate live actions, scores candidate probes by expected
posterior split / information gain, executes the most discriminating probe
before committing to a solve plan, and updates the posterior from the observed
transition. The energy verifier SHALL rank probes by transition
discrimination only and SHALL NOT call the environment win-check, so
`verifier_is_oracle=false` remains gate-eligible.

The same scored-agent path SHALL expose a matched no-probe ablation that keeps
the passive induce-then-plan behavior under the same budget. A success claim
for Experiment 4727 is valid only when the generic live agent reaches a new
level with active probing, offline reproduction confirms it via
`arc_solver_kit.reproduce`, and the no-probe ablation reaches a lower level.
If no new level is reached, Experiment 4727 SHALL still emit an honest null
after confirming probes ran, posterior entropy dropped, reachable L1 headroom
was checked, and the no-probe ablation ran.

Experiment 4727 SHALL write
`results/experiment_4727_active_probe_disambiguation.json` with
`honest_verdict`, `inference_substrate`, `verifier_is_oracle`,
`solve_provenance`, `live_path_reachable`, `hypothesis_posterior_built`,
`probe_actions_taken`, `posterior_entropy_reduction`,
`generic_agent_reached_level`, `no_probe_ablation_reached_level`,
`offline_reproduced`, `reproduced_levels`, `bare_control_passed`,
`false_negative_risk_checked`, `null_methodology_note`,
`missing_verifier_gap_logged`, `chosen_submitted_config`,
`proposer_served_model`, `parity_test_green`, `random_seed`,
`reproducibility_checksum`, and `preconditions_checked`. The artifact SHALL
include field principles for those fields. If any hard precondition is missing,
the artifact SHALL report a terminal blocked verdict and record the exact
failed resource in `preconditions_checked` instead of fabricating a live run.

### SCENARIO-ARC-WMTE-4727-ACTIVE-PROBE-SPLITS-POSTERIOR

Given two candidate dynamics hypotheses that make different predictions for at
least one live action
When the active-probe controller scores candidate actions
Then the highest-ranked probe is the action whose predicted transition buckets
split the posterior with the largest expected information gain, the posterior
update increases mass on hypotheses matching the observed transition, and the
diagnostics report a positive posterior entropy reduction.

### SCENARIO-ARC-WMTE-4727-ARTIFACT-CONTRACT

Given the preconditions pass and Experiment 4727 runs the generic E3 path with
active probing and the matched no-probe ablation
When the workflow writes
`results/experiment_4727_active_probe_disambiguation.json`
Then it records Qwen3.5-9B-MTP `/props` provenance, `verifier_is_oracle=false`,
live-path reachability via `arc_orphan_solver_lint`, the probe-vs-no-probe
level delta, offline reproduction evidence for any new level, a deterministic
checksum, and an honest null methodology note plus verifier gap when active
probing cannot disambiguate.

### REQ-ARC-WMTE-4734: Epistemic-MCTS / Causal-Probe SOTA Ingestion For .436

Experiment 4734 SHALL map the next `.436` frontier after `.435` built the
hypothesis-posterior active-probe controller. The workflow SHALL read
`results/experiment_4722_sota_ingestion_active_probe_world_model.json`,
`research-references.md`,
`python/carnot/agentic/arc_competition_agent.py`, and
`python/carnot/agentic/arc_executable_world_model.py` before scoping methods.
It SHALL map the two `.435`-flagged methods that were not built by Experiment
4727: the epistemic object-model MCTS probe planner and the factored
interaction / causal probe bank. It SHALL explicitly avoid re-flagging the
hypothesis-posterior active-probe controller as the `.436` roadmap result.

The workflow SHALL use the reliable low-concurrency channel: a network
reachability check for `https://arxiv.org`, focused WebSearch/WebFetch over
the top five to eight papers, and direct arXiv URL verification. It SHALL NOT
invoke `/deep-research`. If arXiv is unreachable, the workflow SHALL report
`blocked_network` rather than fabricate citations or method claims.

Experiment 4734 SHALL write
`docs/research-notes/epistemic-mcts-causal-probe-sota-ingestion-20260625.md`
and `results/experiment_4734_sota_ingestion_epistemic_mcts_causal_probe.json`.
The artifact SHALL include `honest_verdict`, `inference_substrate`,
`methods_mapped`, `citations`, `flagged_for_next_roadmap`, `note_path`,
`verifier_is_oracle`, `random_seed`, `reproducibility_checksum`,
`preconditions_checked`, and `field_principles`. The success verdict SHALL be
`success: sota_ingestion_epistemic_mcts_causal_probe_mapped`, the substrate
SHALL be `aggregation_from_upstream_artifacts`, and `verifier_is_oracle` SHALL
be false. Each mapped method SHALL cite real arXiv IDs/URLs and SHALL include
`maps_to_current_stack`, `implement_cost_over_current_stack`, and
`fails_when`. No solve claim, training claim, model-load claim, or leaderboard
claim is valid in this literature-ingestion artifact.

Required field principles SHALL include:

- `honest_verdict`: principle `terminal prefix; success: sota_ingestion_epistemic_mcts_causal_probe_mapped.`
- `inference_substrate`: principle `aggregation_from_upstream_artifacts -- literature synthesis + WebFetch, no model load (100us floor).`
- `methods_mapped`: principle `the 3-5 strongest methods, each with maps_to_current_stack + implement_cost_over_current_stack + fails_when -- the actionable ingestion (discover -> ingest -> plan).`
- `citations`: principle `real arXiv IDs/URLs for every method claim -- an ingestion with no verifiable citations is fabrication per adversarial_verify discipline.`
- `flagged_for_next_roadmap`: principle `the strongest method(s) flagged_for_v436 -- closes the discover->ingest->plan loop so SOTA flows into the next milestone's experiments.`
- `note_path`: principle `docs/research-notes/epistemic-mcts-causal-probe-sota-ingestion-20260625.md -- the human-readable per-track synthesis.`
- `verifier_is_oracle`: principle `false -- a literature synthesis invokes no oracle.`
- `random_seed`: principle `determinism precondition (the search/synthesis seed).`
- `reproducibility_checksum`: principle `content-addressed hash of the ingested source set.`
- `preconditions_checked`: principle `records the network-reachability check; pre-empts missing-resource fabrication.`

### SCENARIO-ARC-WMTE-4734: Next Frontier Maps To V436 Candidates

Given arXiv is reachable and the `.434` active-probe world-model ingestion plus
the `.435` research-reference pre-sweep are readable
When experiment 4734 ingests focused papers for epistemic MCTS,
object-centric MCTS, causal object-centric MCTS, factored interactive
world-models, causality-aware object-world-model control, and
object-world-model policy breakage
Then the note and JSON artifact map three to five methods onto
`E3AgentPolicy` and `arc_executable_world_model`, cite real arXiv IDs including
`2210.13455`, `2309.08477`, `2506.01876`, `2511.02225`, `2511.06136`,
`2511.14262`, `2601.06604`, and `2606.14418`, flag one or more
`flagged_for_v436` candidates, do not flag the already-built
`hypothesis_posterior_active_probe_controller`, record that `/deep-research`
was not invoked, record `verifier_is_oracle=false`, and emit no ARC solve
claim.

### REQ-ARC-WMTE-4730: Persist .435 Online-Warm Action-Effect Controller

Experiment 4730 SHALL read the `.435` A1/A2 artifacts
`results/experiment_4726_online_action_learning_driver_valid_test.json` and
`results/experiment_4727_active_probe_disambiguation.json`, choose the strongest
characterized reusable primitive, persist it in `arc_solver_kit`, and register
it in `ops/arc_solve_registry.yaml` as
`primitive_online_warm_action_effect_controller_operator`. If A1's
online-action-learning arms are non-degenerate, the workflow SHALL persist the
A1 online-warm action-effect controller even when its first-win lift is a
characterized null. If A1 is degenerate and A2 actually built a posterior and
executed probes, the workflow MAY select the A2 active-probe controller instead.
The persisted primitive SHALL rank or route candidate actions from cached
action-effect evidence and SHALL NOT call the executable win-check, so
`verifier_is_oracle=false`.

Experiment 4730 SHALL measure honest leave-one-game transfer on at least three
cached held-out games by excluding the target game from the action-effect memory,
scoring cached held-out candidate rows, and reporting per-game solve-rate,
first-win, coverage, and action-efficiency deltas. A null transfer SHALL be
reported as a characterized result, not hidden. New levels SHALL count only when
`arc_solver_kit.reproduce` confirms them; for this persistence measurement the
usual value of `offline_reproduced_new_level` is false.

Experiment 4730 SHALL write
`results/experiment_4730_primitive_persist_transfer.json` with
`honest_verdict`, `inference_substrate`, `persisted_operator`,
`transfer_value_per_game`, `offline_reproduced_new_level`,
`verifier_is_oracle`, `random_seed`, `reproducibility_checksum`,
`preconditions_checked`, and `field_principles`. If the offline arcade or either
`.435` artifact is unavailable, the experiment SHALL emit a terminal
`blocked_<resource>` verdict and SHALL NOT fabricate transfer values.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; complete: <operator>_persisted_transfer_<characterized|null>."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- scores cached held-out rows (1s floor), no live LLM load."
- `persisted_operator`: principle "the reusable arc_solver_kit operator name + the registry entry -- the self-learning capture so the live agent reuses it on hidden games."
- `transfer_value_per_game`: principle "per-game leave-one-game transfer deltas, reported HONESTLY; a transfer null is a valid characterized result, not a failure to hide."
- `offline_reproduced_new_level`: principle "true only if persisting banked a strictly new offline-reproduced level (usually false for a persist task)."
- `verifier_is_oracle`: principle "false -- the persisted operator ranks/routes/perceives; the reproduction gate is the oracle-distinct authority."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent harness/corpus drift."
- `preconditions_checked`: principle "records resources verified (offline arcade, .435 artifacts present); pre-empts missing-resource fabrication."

### SCENARIO-ARC-WMTE-4730-PERSIST-STRONGEST-435-PRIMITIVE

Given A1 has non-degenerate online-warm arms and A2 did not execute probe
actions
When Experiment 4730 selects a primitive
Then it persists `online_warm_action_effect_controller_operator`, records the
A1/A2 upstream signal comparison, and exposes the operator through
`arc_solver_kit.primitive_operator_registry()` and the ARC solve registry.

### SCENARIO-ARC-WMTE-4730-LEAVE-ONE-GAME-TRANSFER

Given cached action-effect rows exist for at least three held-out games
When Experiment 4730 measures leave-one-game transfer
Then each game excludes its own rows from the memory, reports solve-rate,
first-win, coverage, and action-efficiency deltas, records transfer dead-ends
for null games, keeps `verifier_is_oracle=false`, and writes a stable
`results/experiment_4730_primitive_persist_transfer.json` checksum.

### REQ-ARC-WMTE-4737: Goal-Energy Candidate-Generation Guidance Valid Test

Experiment 4737 SHALL reopen the Exp4640 goal-energy generation null by adding
an additive goal-energy proposal-guidance hook to the live `E3AgentPolicy` /
`StepwiseExplorer` candidate-generation path. The hook SHALL score candidate
states produced by a transition predictor for the current frame and candidate
action, combine lower-is-better navigation/proposal rank with graded
goal-distance or structural-progress energy, and reorder the existing proposal
pool without removing the baseline proposal path. The scorer SHALL be
oracle-distinct: it may score predicted candidate states, but SHALL NOT call an
executable win-check or environment verifier when ranking proposals.

Before measuring lift, Experiment 4737 SHALL prove the goal-energy generation
arm is non-degenerate. The non-degeneracy gate SHALL require real candidate
state scoring evidence, non-zero goal-energy score variance across the
candidate pool, and a measurable proposal ranking or pool delta versus the
baseline ordering. If this gate fails because the arm cloned cached attempts,
scored a constant neutral frame, or otherwise no-opped, the experiment SHALL
emit `complete: goal_energy_generation_arms_degenerate_confirmed_harness_bug`
and SHALL NOT treat the result as a capability null.

After `arms_non_degenerate=true`, Experiment 4737 SHALL run the matched
`experiment_4605_live_integration_scored_agent` held-out harness for baseline
and goal-energy-guided arms, measure CPU goal-energy scoring milliseconds per
candidate, run `scripts/arc_orphan_solver_lint.py`, and run
`tests/python/test_arc_submitted_agent_parity.py`. A success requires either a
held-out first-win lift of at least `+0.05` over baseline or a multi-level probe
that reaches L2 and passes `arc_solver_kit.reproduce`. If the non-degenerate
arms give flat first-win, the artifact SHALL include a non-empty
`null_delta_methodology_note` and `positive_control_passed =
bool(parity_test_green AND arms_non_degenerate AND bare_control_passed)` so the
honest equality is not quarantined as a tautology.

The terminal artifact
`results/experiment_4737_goal_energy_candidate_generation_valid_test.json`
SHALL include principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `arms_non_degenerate`,
`candidate_pool_differs_from_baseline`, `goal_energy_score_variance`,
`goal_energy_first_win`, `baseline_first_win`,
`goal_energy_vs_baseline_delta`, `cpu_scoring_ms_per_candidate`,
`goal_free_l2_reached`, `offline_reproduced`, `reproduced_levels`,
`solve_provenance`, `verifier_is_oracle`, `live_path_reachable`,
`bare_control_passed`, `false_negative_risk_checked`,
`null_delta_methodology_note`, `positive_control_passed`,
`chosen_submitted_config`, `proposer_served_model`, `parity_test_green`,
`random_seed`, `reproducibility_checksum`, and `preconditions_checked`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success: goal_energy_generation_first_win_lift_<delta>_or_l2_<game> OR complete: goal_energy_generation_arms_degenerate_confirmed_harness_bug OR complete: goal_energy_generation_no_first_win_lift_residual_<cause>."
- `inference_substrate`: principle "live_llm_inference (the live candidate generation loads the Qwen GGUF, 60s floor); model_specs MUST name the GGUF."
- `arms_non_degenerate`: principle "THE FIRST GATE -- the goal-energy arm scores REAL candidate states with non-zero score variance + a candidate pool that DIFFERS from baseline (NOT cloned/neutral-on-cached-frame); a false here means the prior null was dead code (a BUG to fix, not a capability null)."
- `candidate_pool_differs_from_baseline`: principle "the explicit evidence the goal-energy pool/ranking differs from baseline (rank/coverage delta > 0) -- the no-op catch that exp4640 failed (it cloned baseline)."
- `goal_energy_score_variance`: principle "non-zero variance of the graded goal-energy across candidate states -- proves it scored REAL states, not a constant neutral value on the cached frame."
- `goal_energy_first_win`: principle "the held-out first-win of the goal-energy-guided arm -- measured ONLY after arms_non_degenerate=True."
- `baseline_first_win`: principle "the unbiased-proposal baseline (the current submitted behavior) -- the no-guidance control."
- `goal_energy_vs_baseline_delta`: principle "goal_energy_first_win - baseline_first_win; >=+0.05 is the gate; emitted explicitly so a null (0) is annotated and not auto-quarantined as TAUTOLOGY (pair with null_delta_methodology_note + positive_control_passed when flat)."
- `cpu_scoring_ms_per_candidate`: principle "the Kaggle path is CPU under a 12h/600-RPM cap; goal-energy scoring too slow per candidate makes the lever infeasible regardless of offline gains."
- `goal_free_l2_reached`: principle "a goal-energy-guided L2 deepening proves the wall is crossed by GUIDING generation toward the goal."
- `offline_reproduced`: principle "a new level counts only if offline-reproduced via arc_solver_kit.reproduce."
- `reproduced_levels`: principle "the integer level the goal-energy-guided agent reached on the multi-level probe."
- `solve_provenance`: principle "live_agent_self_discovery for a generic goal-energy-guided L2; development_proxy if an adapter was needed."
- `verifier_is_oracle`: principle "MUST be false -- the graded goal-energy scores candidate states (oracle-distinct; it does not run the win-check); gate-eligible."
- `live_path_reachable`: principle "HARD gate -- the changed E3AgentPolicy candidate-generation path is in the scored agent's import closure; arc_orphan_solver_lint passes."
- `bare_control_passed`: principle "the POSITIVE CONTROL -- the held-out harness has reachable first-win headroom; a flat null is valid only then."
- `false_negative_risk_checked`: principle "true with non-degenerate arms + reachable headroom -- a 'no lift' null is valid only then."
- `null_delta_methodology_note`: principle "present when goal_energy_vs_baseline_delta ~0 on NON-degenerate arms; states the equality is an honest no-lift null (the TAUTOLOGY carve-out reads it), not a measurement bug -- the .435-A1 escape fix."
- `positive_control_passed`: principle "bool(parity_test_green AND arms_non_degenerate AND bare_control_passed) -- GATES the TAUTOLOGY null-delta exemption; an unvalidated flat result is NOT excused."
- `chosen_submitted_config`: principle "the recommended SUBMITTED_AGENT_CONFIG change (goal-energy guidance on, weight) -- the A6 input; 'unchanged' if null."
- `proposer_served_model`: principle "the model the proposer /props reported (MUST be Qwen3.5-9B-MTP) -- the port-8919 confound guard."
- `parity_test_green`: principle "HARD gate -- test_arc_submitted_agent_parity.py passes."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent harness/corpus drift on replay."
- `preconditions_checked`: principle "records resources verified (CUDA if training, Qwen cached, offline arcade, /props served Qwen); pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4737-NON-DEGENERATE-GOAL-GUIDANCE

Given the Qwen3.5-9B-MTP GGUF is cached and `/props` verifies Qwen on a free
non-8919 port, the offline arcade imports, and `arc-world-model-trust-energy`
declares `REQ-ARC-WMTE-4737`
When Experiment 4737 constructs baseline and goal-energy-guided live
`E3AgentPolicy` proposal paths
Then the goal-energy arm first proves `arms_non_degenerate=true` by scoring
predicted candidate states with non-zero score variance and a proposal ranking
delta, or exits with the degenerate-harness-bug verdict before any lift claim.

#### SCENARIO-ARC-WMTE-4737-HELDOUT-NULL-OR-LIFT

Given the non-degeneracy gate has passed
When Experiment 4737 runs the matched held-out first-win harness and the
lp85/sc25 multi-level probe
Then it reports `goal_energy_first_win`, `baseline_first_win`, explicit delta,
CPU scoring cost, parity and live-path gates, and either a success from
`delta>=+0.05` or L2 reproduction, or an honest no-lift residual with
`null_delta_methodology_note` and `positive_control_passed`.

### REQ-ARC-WMTE-4738: Valid Energy-Fitness QD Candidate Generation Test

Experiment 4738 SHALL reopen the invalid Experiment 4653 energy-fitness QD
null by testing the live `E3AgentPolicy` candidate-generation path with a
non-degeneracy gate before any lift claim. The live QD generator SHALL expose an
additive candidate-pool evolution hook that is reachable from
`StepwiseExplorer._candidates()`, keeps primitive naive-search candidates
available, mutates candidate configuration fields such as click placement and
toggle/color actions, scores generated candidates with lower-is-better
oracle-distinct energy, and keeps diverse elites in a MAP-Elites-style archive
keyed by visible behavior descriptors such as action kind, spatial bucket, and
local color/object context.

Before measuring lift, Experiment 4738 SHALL prove the three arms
`naive-search`, `random-mutation`, and `energy-QD` produce distinct candidate
pools. The gate SHALL compute pairwise candidate-pool Jaccard values and require
each arm pair to have Jaccard `< 1.0`, and it SHALL require the energy-QD pool
to contain a non-zero count of candidates that the naive enumerator did not
produce. If these arms are byte-identical or the QD arm generates no novel
candidates, the experiment SHALL emit
`complete: energy_qd_generation_arms_degenerate_confirmed_harness_bug` and SHALL
NOT report a capability null.

After `arms_non_degenerate=true`, Experiment 4738 SHALL run matched
`experiment_4605_live_integration_scored_agent` held-out first-win measurements
for `naive-search`, `random-mutation`, and `energy-QD`, measure CPU generation
milliseconds per generation, run `scripts/arc_orphan_solver_lint.py`, and run
`tests/python/test_arc_submitted_agent_parity.py`. A success requires either an
energy-QD held-out first-win lift of at least `+0.05` over naive search or an
energy-QD multi-level probe on `lp85`/`sc25` that reaches L2 and passes
`arc_solver_kit.reproduce`. If non-degenerate arms give flat first-win, the
artifact SHALL include a non-empty `null_delta_methodology_note` and
`positive_control_passed = bool(parity_test_green AND arms_non_degenerate AND
bare_control_passed)`.

The terminal artifact
`results/experiment_4738_energy_fitness_qd_generation_valid_test.json` SHALL
include principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `arms_non_degenerate`, `arm_pool_jaccard`,
`novel_candidates_generated`, `energy_qd_first_win`, `naive_search_first_win`,
`energy_qd_vs_naive_delta`, `cpu_generation_ms`, `goal_free_l2_reached`,
`offline_reproduced`, `reproduced_levels`, `solve_provenance`,
`verifier_is_oracle`, `live_path_reachable`, `bare_control_passed`,
`false_negative_risk_checked`, `null_delta_methodology_note`,
`positive_control_passed`, `chosen_submitted_config`, `proposer_served_model`,
`parity_test_green`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success: energy_qd_generation_first_win_lift_<delta>_or_l2_<game> OR complete: energy_qd_generation_arms_degenerate_confirmed_harness_bug OR complete: energy_qd_generation_no_first_win_lift_residual_<cause>."
- `inference_substrate`: principle "live_llm_inference (the live candidate generation loads the Qwen GGUF, 60s floor); model_specs MUST name the GGUF."
- `arms_non_degenerate`: principle "THE FIRST GATE -- the naive-search / random-mutation / energy-QD arms produce DISTINCT candidate pools (NOT byte-identical) + a non-zero novel-candidate count; a false here means the prior null (exp4653) was dead code (a BUG, not a capability null)."
- `arm_pool_jaccard`: principle "the pairwise candidate-pool Jaccard between arms (< 1.0 proves distinct pools) -- the no-op catch that exp4653 failed (byte-identical arms)."
- `novel_candidates_generated`: principle "the count of QD candidates NOT in the naive-enumerator pool -- proves the QD generator actually generated novel candidates (winner_generated_count was 0 in exp4653)."
- `energy_qd_first_win`: principle "the held-out first-win of the energy-QD arm -- measured ONLY after arms_non_degenerate=True."
- `naive_search_first_win`: principle "the naive-enumeration baseline -- the no-QD control."
- `energy_qd_vs_naive_delta`: principle "energy_qd_first_win - naive_search_first_win; >=+0.05 is the gate; emitted explicitly so a null (0) is annotated (pair with null_delta_methodology_note + positive_control_passed when flat)."
- `cpu_generation_ms`: principle "the Kaggle path is CPU under a 12h/600-RPM cap; QD generation too slow per turn makes the lever infeasible regardless of offline gains."
- `goal_free_l2_reached`: principle "an energy-QD-generated L2 proves the wall is crossed by GENERATING the winner, not selecting."
- `offline_reproduced`: principle "a new level counts only if offline-reproduced via arc_solver_kit.reproduce."
- `reproduced_levels`: principle "the integer level the energy-QD agent reached on the multi-level probe."
- `solve_provenance`: principle "live_agent_self_discovery for a generic energy-QD-generated L2; development_proxy if an adapter was needed."
- `verifier_is_oracle`: principle "MUST be false -- the energy fitness scores candidate configurations (oracle-distinct; it does not run the win-check); gate-eligible."
- `live_path_reachable`: principle "HARD gate -- the changed candidate-generation path is in the scored agent's import closure; arc_orphan_solver_lint passes."
- `bare_control_passed`: principle "the POSITIVE CONTROL -- the held-out harness has reachable first-win headroom; a flat null is valid only then."
- `false_negative_risk_checked`: principle "true with non-degenerate arms + reachable headroom -- a 'no lift' null is valid only then."
- `null_delta_methodology_note`: principle "present when energy_qd_vs_naive_delta ~0 on NON-degenerate arms; states the equality is an honest no-lift null (the TAUTOLOGY carve-out reads it) -- the .435-A1 escape fix."
- `positive_control_passed`: principle "bool(parity_test_green AND arms_non_degenerate AND bare_control_passed) -- GATES the TAUTOLOGY null-delta exemption."
- `chosen_submitted_config`: principle "the recommended SUBMITTED_AGENT_CONFIG change (energy-QD generator on, params) -- the A6 input; 'unchanged' if null."
- `proposer_served_model`: principle "the model the proposer /props reported (MUST be Qwen3.5-9B-MTP) -- the port-8919 confound guard."
- `parity_test_green`: principle "HARD gate -- test_arc_submitted_agent_parity.py passes."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent harness/corpus drift on replay."
- `preconditions_checked`: principle "records resources verified (Qwen cached, offline arcade, /props served Qwen); pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4738-NON-DEGENERATE-QD-ARMS

Given the Qwen3.5-9B-MTP GGUF is cached, `/props` verifies Qwen on a non-8919
port, the offline arcade imports, and `arc-world-model-trust-energy` declares
`REQ-ARC-WMTE-4738`
When Experiment 4738 constructs naive-search, random-mutation, and energy-QD
live candidate pools from `StepwiseExplorer._candidates()`
Then the arms first prove `arms_non_degenerate=true` with pairwise Jaccard
values below 1.0 and a non-zero QD novel-candidate count, or exit with the
degenerate-harness-bug verdict before any lift claim.

#### SCENARIO-ARC-WMTE-4738-HELDOUT-NULL-OR-LIFT

Given Experiment 4738 has proven non-degenerate QD candidate pools and the
parity/orphan-lint gates are green
When it runs the held-out first-win harness and lp85/sc25 multi-level probe
Then it either reports an energy-QD first-win lift of at least +0.05 or an
offline reproduced L2, or emits a complete no-lift residual with
`null_delta_methodology_note` and `positive_control_passed` when the matched
first-win delta is flat.

### REQ-ARC-WMTE-4741: Persist .436 Primitive And Leave-One-Game Transfer

Experiment 4741 SHALL persist the strongest characterized `.436` A1/A2
generation primitive into `arc_solver_kit` and `ops/arc_solve_registry.yaml` so
the live ARC agent can reuse it on hidden games. The preferred A2 persistence
surface is `energy_fitness_qd_generator_operator` with registry gotcha id
`primitive_energy_fitness_qd_generator_operator`. If Experiment 4737's
goal-energy guidance arms are degenerate, Experiment 4741 SHALL prefer
Experiment 4738's energy-fitness QD generator; if Experiment 4738's QD arms are
degenerate, it SHALL prefer Experiment 4737's goal-energy candidate-generation
guidance. When both are non-degenerate, it SHALL pick the better-characterized
reusable primitive using artifact evidence such as novel generated candidates,
offline-reproduced probe evidence, and measured non-degeneracy diagnostics.

The persisted operator SHALL be oracle-distinct: it may generate, score, rank,
or route cached candidate rows, but SHALL NOT call the executable win-check
while assigning candidate fitness. New levels SHALL count only when a proposed
solution passes `arc_solver_kit.reproduce`; transfer-only cached scoring SHALL
set `verifier_is_oracle=false` and SHALL NOT claim a newly banked level.

Experiment 4741 SHALL measure leave-one-game transfer on at least three cached
held-out games using `verifier_ensemble_against_cached_candidates` without live
LLM load. For every held-out game it SHALL exclude that game's rows from the
operator's cross-game characterization, apply the persisted operator to cached
rows, and report solve-rate, first-win, candidate-generation coverage, and
action-efficiency deltas honestly. A transfer null is valid when recorded as a
characterized null with a `transfer_dead_ends` note.

The terminal artifact
`results/experiment_4741_primitive_persist_transfer.json` SHALL include
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `persisted_operator`, `transfer_value_per_game`,
`offline_reproduced_new_level`, `verifier_is_oracle`, `random_seed`,
`reproducibility_checksum`, and `preconditions_checked`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; complete: <operator>_persisted_transfer_<characterized|null>."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- scores cached held-out rows (1s floor), no live LLM load."
- `persisted_operator`: principle "the reusable arc_solver_kit operator name + the registry entry -- the self-learning capture so the live agent reuses it on hidden games."
- `transfer_value_per_game`: principle "per-game leave-one-game transfer deltas, reported HONESTLY; a transfer null is a valid characterized result, not a failure to hide."
- `offline_reproduced_new_level`: principle "true only if persisting banked a strictly new offline-reproduced level (usually false for a persist task)."
- `verifier_is_oracle`: principle "false -- the persisted operator generates/scores/routes; the reproduction gate is the oracle-distinct authority."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent harness/corpus drift."
- `preconditions_checked`: principle "records resources verified (offline arcade, .436 artifacts present); pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4741-PERSIST-STRONGEST-436-PRIMITIVE

Given the `.436` A1 and A2 artifacts are present, the offline arcade imports,
and `arc-world-model-trust-energy` declares `REQ-ARC-WMTE-4741`
When Experiment 4741 selects the primitive to persist
Then `arc_solver_kit.primitive_operator_registry()` and
`ops/arc_solve_registry.yaml` expose the chosen operator with `derived_from`,
`note`, and `transfer_dead_ends`, and the artifact records the selection
rationale without claiming verifier-oracle authority.

#### SCENARIO-ARC-WMTE-4741-LEAVE-ONE-GAME-TRANSFER

Given the selected `.436` primitive is persisted
When Experiment 4741 applies it to three cached held-out games with each target
game excluded from cross-game characterization
Then the artifact reports per-game solve-rate, first-win, coverage, and
action-efficiency deltas, sets `offline_reproduced_new_level` only for
`arc_solver_kit.reproduce`-banked levels, and records a characterized
`transfer_dead_ends` note for value-null transfer.

### REQ-ARC-WMTE-4742: .436 Submitted-Agent Integration Gate

Experiment 4742 SHALL read the `.436` A1/A2 artifacts
`results/experiment_4737_goal_energy_candidate_generation_valid_test.json` and
`results/experiment_4738_energy_fitness_qd_generation_valid_test.json`, choose
the strongest non-unchanged `chosen_submitted_config` that has a positive
held-out first-win lift, and integrate that change into `SUBMITTED_AGENT_CONFIG`
only when the submitted config exactly matches the selected change, the
submitted-agent parity test passes, and the measured held-out first-win does
not regress versus the pre-integration baseline. If both upstream artifacts
select `unchanged` or have no positive held-out lift, the integration SHALL be
a no-op with `integrated_change="none"` and SHALL report the flat delta
honestly rather than fabricating a win.

Experiment 4742 SHALL verify the offline arcade is importable and both `.436`
A1/A2 artifacts are present before measuring. Missing resources SHALL produce a
terminal `blocked_<resource>` verdict and record the failed resource in
`preconditions_checked`. The experiment SHALL run
`tests/python/test_arc_submitted_agent_parity.py` as a hard gate, set
`verifier_is_oracle=false`, write
`results/experiment_4742_integration_gate.json`, and never submit to the
leaderboard.

The artifact SHALL include `honest_verdict`, `inference_substrate`,
`integrated_change`, `live_first_win_rate_integrated`,
`live_first_win_rate_pre_integration`, `parity_test_green`,
`null_delta_methodology_note`, `positive_control_passed`,
`verifier_is_oracle`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`. When
`live_first_win_rate_integrated == live_first_win_rate_pre_integration`, the
artifact SHALL emit a non-empty `null_delta_methodology_note` and
`positive_control_passed` SHALL equal
`bool(parity_test_green AND no_regression_vs_pre_integration)`. The artifact
SHALL NOT rely on a `tautology_guard` prose field for the null-delta carve-out.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success: integrated_<change>_first_win_<delta> OR complete: integration_no_change_all_levers_unchanged."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- scores the integrated config over cached variants (1s floor)."
- `integrated_change`: principle "the change integrated into SUBMITTED_AGENT_CONFIG (or 'none' if all A1/A2 selected unchanged)."
- `live_first_win_rate_integrated`: principle "the held-out first-win of the integrated config -- the scored-lane proxy."
- `live_first_win_rate_pre_integration`: principle "the pre-integration baseline -- the no-regression control."
- `parity_test_green`: principle "HARD gate -- the integrated agent is byte-for-byte the SUBMITTED_AGENT_CONFIG; a parity miss invalidates the integration."
- `null_delta_methodology_note`: principle "present when both deltas ~0; the TAUTOLOGY carve-out reads it (honest no-change, not a measurement bug)."
- `positive_control_passed`: principle "bool(parity_test_green AND no_regression) -- GATES the TAUTOLOGY exemption; an unvalidated integration is NOT excused."
- `verifier_is_oracle`: principle "false -- the integration gate measures the agent; no oracle is invoked."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent harness/corpus drift."
- `preconditions_checked`: principle "records resources verified (offline arcade, A1/A2 artifacts present); pre-empts missing-resource fabrication."

### SCENARIO-ARC-WMTE-4742-HONEST-NULL-INTEGRATION

Given the `.436` A1 and A2 artifacts both select
`chosen_submitted_config="unchanged"`
When Experiment 4742 runs the integration gate and the submitted-agent parity
test passes
Then it keeps `SUBMITTED_AGENT_CONFIG` unchanged, records
`integrated_change="none"`, reports equal integrated and pre-integration
held-out first-win rates, emits `null_delta_methodology_note`,
sets `positive_control_passed` from parity plus no-regression, sets
`verifier_is_oracle=false`, and writes a stable
`results/experiment_4742_integration_gate.json` checksum.

### REQ-ARC-WMTE-4746: Epistemic-MCTS / Causal-Probe / MATM SOTA Ingestion For .437

Experiment 4746 SHALL map the `.437` ARC frontier after `.436` valid-tested
the guidance-class generation levers instead of building the `.435`-flagged
epistemic-MCTS and causal-probe tracks. The workflow SHALL read
`results/experiment_4734_sota_ingestion_epistemic_mcts_causal_probe.json`,
`research-references.md`,
`docs/research-notes/matm-transactive-memory-ingestion-2026-06-23.md`,
`python/carnot/agentic/arc_competition_agent.py`, and
`python/carnot/agentic/arc_executable_world_model.py` before scoping methods.
It SHALL map the epistemic object-model MCTS probe planner, the factored
interaction / causal probe bank, the object-world-model breakage falsifier, and
the within-game similarity-keyed partial-trajectory retrieval efficiency lever
onto the current `E3AgentPolicy`, `StepwiseExplorer.adj`, and
`arc_executable_world_model` stack. It SHALL explicitly bound MATM as an
action-efficiency candidate, not a level-bank or cross-game retrieval claim.

The workflow SHALL use the reliable low-concurrency channel: a network
reachability check for `https://arxiv.org`, focused WebSearch/WebFetch over the
top five to eight papers, and direct arXiv URL verification. It SHALL NOT
invoke `/deep-research`. If arXiv is unreachable, the workflow SHALL report
`blocked_network` rather than fabricate citations or method claims.

Experiment 4746 SHALL write
`docs/research-notes/epistemic-mcts-causal-probe-sota-ingestion-20260625.md`
and
`results/experiment_4746_sota_ingestion_epistemic_mcts_causal_probe.json`. The
artifact SHALL include `honest_verdict`, `inference_substrate`,
`methods_mapped`, `citations`, `flagged_for_next_roadmap`, `note_path`,
`verifier_is_oracle`, `random_seed`, `reproducibility_checksum`,
`preconditions_checked`, and `field_principles`. The success verdict SHALL be
`success: sota_ingestion_epistemic_mcts_causal_probe_matm_mapped`, the
substrate SHALL be `aggregation_from_upstream_artifacts`, and
`verifier_is_oracle` SHALL be false. Each mapped method SHALL cite real arXiv
IDs/URLs and SHALL include `maps_to_current_stack`,
`implement_cost_over_current_stack`, and `fails_when`. No solve claim, training
claim, model-load claim, cross-game MATM claim, or leaderboard claim is valid
in this literature-ingestion artifact.

Required field principles SHALL include:

- `honest_verdict`: principle `terminal prefix; success: sota_ingestion_epistemic_mcts_causal_probe_matm_mapped.`
- `inference_substrate`: principle `aggregation_from_upstream_artifacts -- literature synthesis + WebFetch, no model load (100us floor).`
- `methods_mapped`: principle `the 3-5 strongest methods, each with maps_to_current_stack + implement_cost_over_current_stack + fails_when -- the actionable ingestion (discover -> ingest -> plan).`
- `citations`: principle `real arXiv IDs/URLs for every method claim -- an ingestion with no verifiable citations is fabrication per adversarial_verify discipline.`
- `flagged_for_next_roadmap`: principle `the strongest method(s) flagged_for_v437 -- closes the discover->ingest->plan loop so SOTA flows into the next milestone's experiments.`
- `note_path`: principle `docs/research-notes/epistemic-mcts-causal-probe-sota-ingestion-20260625.md -- the human-readable per-track synthesis.`
- `verifier_is_oracle`: principle `false -- a literature synthesis invokes no oracle.`
- `random_seed`: principle `determinism precondition (the search/synthesis seed).`
- `reproducibility_checksum`: principle `content-addressed hash of the ingested source set.`
- `preconditions_checked`: principle `records the network-reachability check; pre-empts missing-resource fabrication.`

### SCENARIO-ARC-WMTE-4746: Next Frontier Maps To V437 Candidates

Given arXiv is reachable and the `.435` epistemic-MCTS / causal-probe
ingestion plus the `.436` research-reference pre-sweep are readable
When experiment 4746 ingests focused papers for epistemic MCTS,
object-centric MCTS, causal object-centric MCTS, factored interactive
world-models, causality-aware object-world-model control, object-world-model
policy breakage, and MATM-style similarity-keyed trajectory retrieval
Then the note and JSON artifact map three to five methods onto
`E3AgentPolicy`, `StepwiseExplorer.adj`, and `arc_executable_world_model`, cite
real arXiv IDs including `2210.13455`, `2511.02225`, `2511.06136`,
`2511.14262`, `2601.06604`, `2606.14418`, and `2606.19911`, flag one or more
`flagged_for_v437` candidates, record that `/deep-research` was not invoked,
record `verifier_is_oracle=false`, emit no ARC solve claim, and identify MATM
as a within-game action-efficiency candidate rather than a level-bank.

### REQ-ARC-WMTE-4739: Rotated ARC Level-Up Self-Play Attempt

Experiment 4739 SHALL use the standing ARC loop to attempt one rotated public
game deepening without duplicating an already banked registry level. It SHALL
precheck `ops/arc_solve_registry.yaml`, skip the recent operator skip list
(`ar25`, `bp35`, `lf52`, `sb26`, `dc22`, `vc33`, `sk48`), prefer the clean hard
targets `re86`, `s5i5`, `g50t`, and `r11l`, and use `m0r0` or `cn04` only as
allowed fallback alternatives. A target level SHALL pass registry precheck only
when the registry's `levels_reproduced` for that game is lower than the
attempted target level.

Before writing a terminal artifact, Experiment 4739 SHALL verify
`arc_solver_kit.offline_arcade()` and `scripts/arc_loop_solve.py --help`, run
`scripts/arc_loop_solve.py --game <target> --target-level <n>
--no-hazard-prune`, and report the result through the offline reproduction gate
from `arc_solver_kit.reproduce`. New levels SHALL count only when the standing
loop advances beyond the prechecked registry level and the offline reproduction
gate confirms the reached level. The learned verifier checkpoint SHALL be
reported as `models/arc_verifier_<game>.json` when the run produces or refreshes
one, but the checkpoint SHALL remain oracle-distinct from the executable
reproduction authority.

The terminal artifact
`results/experiment_4739_levelup_selfplay.json` SHALL include
principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `offline_reproduced`, `reproduced_levels`,
`new_levels_banked`, `reproducible_total_levels`, `verifier_checkpoint`,
`verifier_is_oracle`, `solve_provenance`, `registry_precheck_passed`,
`random_seed`, `reproducibility_checksum`, and `preconditions_checked`. If no
new level banks, the artifact SHALL emit a terminal `complete:` verdict,
preserve `reproducible_total_levels` unchanged, and record per-game dead-end
deltas for the attempted rotated targets instead of fabricating a bank.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success: <game>_L<n>_offline_reproduced OR complete: <game>_delta_identified_no_bank."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- arc_loop_solve scores against the offline sim (1s floor), no live LLM load."
- `offline_reproduced`: principle "only offline-reproduced levels count (ARC Solve Reproducibility); a live-only trajectory is provisional."
- `reproduced_levels`: principle "the integer level reached on the target game."
- `new_levels_banked`: principle ">=1 is the Level-Up Guarantee met; 0 rotates the target next milestone."
- `reproducible_total_levels`: principle "the registry header after this run (64->65+ if banked) -- the monotonic north-star metric."
- `verifier_checkpoint`: principle "models/arc_verifier_<game>.json -- the self-play loop trains+checkpoints the learned verifier (the self-improvement-every-milestone mandate)."
- `verifier_is_oracle`: principle "MUST be false -- the learned verifier routes/ranks; the executable reproduction gate is the oracle-distinct authority."
- `solve_provenance`: principle "live_agent_self_discovery for a generic bank, OR development_proxy honestly if a GameAdapter delta was required."
- `registry_precheck_passed`: principle "confirms the target level is NOT already in the registry (a duplicate is a CRITICAL adversarial flag)."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent harness/corpus drift on replay."
- `preconditions_checked`: principle "records resources verified (offline arcade, arc_loop_solve runnable); pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4739-ROTATED-PRECHECK

Given the ARC registry records `re86` at L2 and the hard preferred L1-to-L2
targets have stalled or routing-only notes
When Experiment 4739 selects a target
Then it does not re-bank an already recorded level, records the skipped and
stalled target notes, and attempts only a target level greater than the
registry's reproduced level for that game.

#### SCENARIO-ARC-WMTE-4739-REPRODUCTION-GATED-SELFPLAY

Given the offline arcade and standing loop are importable
When Experiment 4739 runs `scripts/arc_loop_solve.py --game re86 --target-level 3 --no-hazard-prune`
Then it records the loop artifact, the offline reproduction gate result, the
learned verifier checkpoint path, `verifier_is_oracle=false`, and
`solve_provenance=development_proxy`.

#### SCENARIO-ARC-WMTE-4739-NO-BANK-RESIDUAL

Given the standing loop reproduces only the registry's prior level for the
selected target
When Experiment 4739 writes the artifact
Then it emits `complete: <game>_delta_identified_no_bank`, keeps
`new_levels_banked=0`, leaves `reproducible_total_levels=64`, and records the
attempted target deltas as dead-ends for the next rotation.

### REQ-ARC-WMTE-4758: .437 SOTA Ingestion for Structured World-Model Induction and Perception-Grounded Goals

Experiment 4758 SHALL run the reserved `.437` SOTA-ingestion slot for the
headline open problem: structured world-model induction plus
perception-grounded goals for ARC. The workflow SHALL use only the reliable
channel: read `research-studying.md`, `research-references.md`,
`scripts/sweep_clusters.py`, and `scripts/sweep_semscholar.py`; verify network
reachability with `curl -sf -o /dev/null https://huggingface.co/api/models`;
emit the ARC cluster URLs; run a focused Semantic Scholar pass; and WebFetch
the top five to eight arXiv papers. It SHALL NOT invoke `/deep-research`, load
models, train, submit to the leaderboard, or make a solve claim.

On a successful ingestion-synthesis run, Experiment 4758 SHALL write
`results/experiment_4758_sota_ingestion.json` and update `research-studying.md`
with an idempotent Exp 4758 section. The artifact SHALL include the required
fields `honest_verdict`, `inference_substrate`, `preconditions_checked`,
`methods_mapped`, `flagged_for_438`, `citations`, `fresh_sweep`, `note_path`,
`random_seed`, `reproducibility_checksum`, and `field_principles`.
The complete-path `honest_verdict` SHALL start with `complete_` and equal
`complete_sota_ingestion_structured_world_model_goal_frontier_mapped`;
`inference_substrate` SHALL equal `aggregation_from_upstream_artifacts`; and
`flagged_for_438` SHALL name the strongest method as a `.438` roadmap
candidate. If the Hugging Face network precondition fails, the workflow SHALL
write a blocked artifact with `honest_verdict=blocked_network`, no method
claims, no citations, and a failed network check recorded in
`preconditions_checked`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; an ingestion-synthesized run is complete_."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts; 100us floor."
- `preconditions_checked`: principle "records the network check."
- `methods_mapped`: principle "the strongest 3-5 SOTA methods with REAL arXiv IDs -- no fabricated citations (adversarial_verify bar)."
- `flagged_for_438`: principle "the strongest method flagged as a candidate input for the .438 roadmap -- closes discover->ingest->plan."

The method map SHALL contain three to five methods, and each method SHALL cite
only real verified arXiv IDs from the fetched source set:
`2605.05138`, `2402.12275`, `2606.14418`, `2605.14937`, `2606.08775`,
`2503.23145`, `2601.06604`, and `2511.02225`. Each method SHALL state what it
takes over from the current Carnot stack and where it fails. The note SHALL
surface the strongest candidate for `.438` as a verifier-refined executable
world-model induction loop with perception-grounded goal-conditioned planning,
using `E3AgentPolicy`, `arc_executable_world_model`, `ProductWorldModel`, and
the structural-alignment goal pipeline as the target integration points.

#### SCENARIO-ARC-WMTE-4758

Given the network precondition passes, the discovered corpus is readable, the
reliable sweep helpers are available, and the top arXiv sources are HTTP-200
verified
When Experiment 4758 runs
Then it writes `results/experiment_4758_sota_ingestion.json`, records that
`/deep-research` was not invoked, maps three to five SOTA methods with real
arXiv IDs, updates `research-studying.md` with the Exp 4758 mapping note, and
flags exactly one strongest method for `.438`.

#### SCENARIO-ARC-WMTE-4758-BLOCKED-NETWORK

Given the Hugging Face network precondition fails
When Experiment 4758 runs
Then it writes a `blocked_network` artifact, records the failed network check,
leaves `methods_mapped`, `citations`, and `flagged_for_438` empty, and makes no
paper, solve, model-load, training, or leaderboard claim.

### REQ-ARC-WMTE-4761: Structural Energy S0 Transition-Correctness Core-Bet Probe

Experiment 4761 SHALL run the `.438` S0 core-bet probe for oracle-distinct
structural energy. The workflow SHALL first verify that `arc_solver_kit`
initializes the offline arcade and that `cross_game_features_v3` imports. If
either precondition fails, the artifact SHALL stop with
`honest_verdict=blocked_offline_arcade_missing` or
`honest_verdict=blocked_structural_features_missing` and SHALL NOT report a
fabricated AUROC.

The dataset SHALL be built from the banked ARC games only. For each usable game,
the workflow SHALL replay the banked recorded transitions, fit an
`InducedWorldModel` on a prefix of those recorded transitions, and score held-out
off-path transitions that were not fit on. The transition-correctness label
SHALL be oracle-free: a positive row is the model's predicted next grid when it
exactly matches the held-out observed next grid, and a negative row is the
model's predicted next grid when it mispredicts. Negatives SHALL be real
induced-engine near misses, never synthetic corruptions of the observed
ground-truth next grid.

The primary head SHALL use only the `object_relational` and `frame_delta`
families from `cross_game_features_v3`, with the v1/v2 frame marginals,
`action_conditioned`, and `predicate_distance` excluded. It SHALL train a
logistic head under leave-one-GAME-out folds and report structural LOO AUROC,
bootstrap CI95, in-sample AUROC, a majority-class chance control, a v2
frame-marginal logistic control on identical folds and seed, the
structural-minus-marginal delta CI95, per-family LOO AUROC for the structural
families, and an origin-probe leak audit that classifies induced-vs-real rows on
the same feature view.

The artifact `results/experiment_4761_structural_energy_s0_core_bet_probe.json`
SHALL set `verifier_is_oracle=false` and include principle-annotated fields for
`honest_verdict`, `verifier_is_oracle`, `inference_substrate`,
`preconditions_checked`, `loo_auroc_structural`, `loo_auroc_ci95`,
`loo_auroc_marginal_control`, `structural_minus_marginal_delta_ci95`,
`per_family_loo`, `origin_probe_auroc`, `near_miss_negative_fraction`,
`in_sample_auroc`, `random_seed`, and `reproducibility_checksum`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; a clean cross-game above-chance result is success_, an honest null is complete_; a null with CI-incl-0.5 RETIRES the direction."
- `verifier_is_oracle`: principle "MUST be false -- the energy never runs the env win-check; required for check_circular_moat_overclaim."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- scores structural features over cached transitions, no LLM; 1s floor."
- `preconditions_checked`: principle "records the arcade/feature-import checks so a silent-missing-resource run cannot fabricate an AUROC."
- `loo_auroc_structural`: principle "the load-bearing measurement -- cross-game leave-one-GAME-out AUROC on held-out transition-correctness with the structural features."
- `loo_auroc_ci95`: principle "bootstrap CI95; the lower bound vs 0.5 is the decisive non-circular signal."
- `loo_auroc_marginal_control`: principle "the v2 frame-marginal head on identical folds -- proves whether STRUCTURE adds transfer beyond marginals."
- `structural_minus_marginal_delta_ci95`: principle "must exclude 0 -- otherwise structure carries no signal beyond marginals and the direction retires."
- `per_family_loo`: principle ">=1 non-frame_delta structural family must independently clear 0.55 -- kills the frame_delta-single-lever risk."
- `origin_probe_auroc`: principle "induced-vs-real leak audit; must be < 0.6 -- else the head is a provenance discriminator."
- `near_miss_negative_fraction`: principle "fraction of negatives that are REAL near-misses (<=5% cells changed); GAP-3 aced synthetic corruptions but failed real near-misses."
- `in_sample_auroc`: principle "positive control > 0.60 -- else the harness is broken and the LOO null is uninformative."
- `random_seed`: principle "determinism is the precondition for reproducibility; a third party must re-run the LOO."
- `reproducibility_checksum`: principle "content hash of the (corpus, folds, features) so a future replication catches drift."

The S0 gate SHALL pass only if structural LOO AUROC has CI95 lower bound greater
than true chance 0.5, the point estimate is greater than 0.60, the
structural-minus-marginal delta CI95 excludes 0, at least one genuinely
structural family clears 0.55 LOO, the origin probe scores AUROC less than 0.6,
and the in-sample positive control exceeds 0.60. If the CI95 includes 0.5, the
structural-minus-marginal delta CI95 includes 0, or the origin probe leaks, the
artifact SHALL retire the energy-guided direction with an honest null verdict.

#### SCENARIO-ARC-WMTE-4761-S0-GATE

Given the offline arcade and structural feature spine are importable and at
least sixteen banked games produce held-out transition-correctness rows
When Experiment 4761 runs the structural S0 probe
Then it writes
`results/experiment_4761_structural_energy_s0_core_bet_probe.json`, keeps
`verifier_is_oracle=false`, reports the matched structural, marginal, majority,
per-family, leak-audit, near-miss, and positive-control measurements, and marks
the gate as passed only when every numeric S0 criterion is satisfied.

#### SCENARIO-ARC-WMTE-4761-BLOCKED-PRECONDITION

Given the offline arcade smoke check or structural feature import check fails
When Experiment 4761 runs
Then it writes a blocked artifact with the failed precondition recorded, no
claimed AUROC, `verifier_is_oracle=false`, and a reproducibility checksum.

### REQ-ARC-WMTE-4778: SOTA Ingestion For S0-Prime Structural-Energy Leak-Robust S1-S4 Planning

Experiment 4778 SHALL run the reserved `.440` SOTA-ingestion slot for the
oracle-distinct S0' structural-energy headline and the leak-robustness problem
surfaced by S0/S0'. The workflow SHALL verify that `research-studying.md` and
`research-references.md` are present, read the discovered structural-energy
corpus, emit focused `scripts/sweep_clusters.py` URLs for energy/object/world
model tracks, run `scripts/sweep_semscholar.py` on focused structural-energy
and shortcut-robust evaluation queries, and use low-concurrency
WebSearch/WebFetch plus direct arXiv checks to verify the top five to eight
source papers. The workflow SHALL NOT invoke `/deep-research`, load models,
train, submit to a leaderboard, or make a solve claim.

On a successful ingestion-synthesis run, Experiment 4778 SHALL write
`results/experiment_4778_sota_ingestion_structural_energy.json` and update
`research-studying.md` with an idempotent Exp 4778 section marked INGESTED. The
artifact SHALL include the required fields `honest_verdict`, `methods_mapped`,
`arxiv_ids_cited`, `flagged_for_v440`, `inference_substrate`,
`preconditions_checked`, `citations`, `fresh_sweep`, `s0prime_context`,
`leak_robust_evaluation_note`, `note_path`, `random_seed`, `duration_s`,
`reproducibility_checksum`, and `field_principles`.

The complete-path `honest_verdict` SHALL start with `success_` and equal
`success_sota_ingestion_structural_energy_mapped`; `inference_substrate` SHALL
equal `aggregation_from_upstream_artifacts`; and `flagged_for_v440` SHALL name
the strongest method or methods for the `.440` planner. The mapping SHALL cover
S1 through S4 of the structural-energy program and SHALL include a specific
leak-robust evaluation note for origin/provenance controls,
shortcut-confound probes, and counterfactual/invariance probes before any
S1-S4 continuation claim.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; mapping emitted is success_sota_ingestion_structural_energy_mapped."
- `methods_mapped`: principle "the strongest 3-5 methods mapped onto S1-S4 + leak-robust eval, each with a real arXiv ID."
- `arxiv_ids_cited`: principle "every method claim must cite a verifiable arXiv ID -- an ingestion with no citations is fabrication."
- `flagged_for_v440`: principle "the strongest method(s) flagged so the .440 planner reads the mapping."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts; 0.0001s floor."

The source set SHALL contain only verified arXiv IDs from the focused pass:
`1907.02893`, `1911.12247`, `2006.15055`, `2301.08243`, `2505.10819`,
`2505.13910`, `2510.04542`, and `2605.05138`. The method map SHALL contain
three to five methods, and every method claim SHALL cite only IDs from
`arxiv_ids_cited`. The note SHALL identify Slot Attention/object-centric slots,
C-SWM relational contrastive transition energy, PoE-World/product-of-experts
factors, JEPA-class predictive representation energy, executable/code
world-model induction, and ShortcutProbe/IRM-style shortcut and invariance
evaluation as the ingested literature informing the S1-S4 build.

#### SCENARIO-ARC-WMTE-4778

Given the research files are present, the reliable sweep helpers are available,
Semantic Scholar is attempted through the reliable script, and the top arXiv
sources are HTTP-200 verified
When Experiment 4778 runs
Then it writes
`results/experiment_4778_sota_ingestion_structural_energy.json`, records that
`/deep-research` was not invoked, maps three to five SOTA methods onto S1-S4
and leak-robust evaluation with real arXiv IDs, updates `research-studying.md`
with the Exp 4778 mapping note, and flags the strongest candidate inputs for
`.440`.

#### SCENARIO-ARC-WMTE-4778-LEAK-ROBUST

Given S0' reports a structural signal after origin matching but still carries
an adversarial warning
When Experiment 4778 synthesizes the SOTA mapping
Then its artifact records the S0' context, requires origin/provenance leak
controls, shuffled-label or shortcut probes, and counterfactual/invariance
stress tests, and refuses a mapping whose methods or leak-evaluation notes cite
unverified or missing arXiv IDs.

### REQ-ARC-WMTE-4771: S0 Prime Origin-Matched Structural Energy Re-Test

Experiment 4771 SHALL run the `.439` S0' origin-matched re-test for the
structural-energy direction. The workflow SHALL first verify that
`arc_solver_kit.offline_arcade()` exits successfully and that
`cross_game_features_v3` imports. If either precondition fails, the artifact
SHALL stop with `honest_verdict=blocked_offline_arcade_missing` or
`honest_verdict=blocked_structural_features_missing` and SHALL NOT report a
fabricated AUROC.

The dataset SHALL remove the Experiment 4761 origin confound. For every usable
banked game, both positive and negative rows SHALL be induced-engine predictions
on held-out transitions. A positive row is an induced prediction `s_hat` that
exactly matches the held-out observed next grid `s'`; a negative row is an
induced prediction `s_hat` that does not match `s'`. The classifier features
SHALL be computed over `(s, s_hat)` so every scored sample has induced origin.
The artifact SHALL record per-game correct/wrong counts and SHALL mark folds
with only one test class as skipped instead of fabricating a fold AUROC.

The primary head SHALL use only the `object_relational` and `frame_delta`
families from `cross_game_features_v3`. It SHALL report leave-one-GAME-out
structural AUROC with bootstrap CI95, majority-class and v2 frame-marginal
controls on identical folds, structural-minus-marginal delta CI95, individual
family LOO AUROC, an origin-probe audit expected to fall below `0.6`, a
shuffled-label LOO leak control expected to score no higher than `0.55`, and an
in-sample positive control expected to exceed `0.60`.

Experiment 4771 SHALL write
`results/experiment_4771_structural_energy_s0prime_origin_matched.json`, set
`verifier_is_oracle=false`, and include principle-annotated fields for
`honest_verdict`, `verifier_is_oracle`, `inference_substrate`,
`preconditions_checked`, `loo_auroc_structural`, `loo_auroc_ci95`,
`origin_probe_auroc`, `shuffled_label_control_auroc`,
`structural_minus_marginal_delta_ci95`, `per_family_loo`,
`per_game_class_balance`, `in_sample_auroc`, `random_seed`, and
`reproducibility_checksum`.

The decisive re-open gate SHALL pass only if structural LOO CI95 lower bound is
greater than `0.5`, structural LOO point estimate is greater than `0.60`,
structural-minus-marginal delta CI95 excludes `0`, at least one structural
family independently clears `0.55`, origin-probe AUROC is less than `0.6`,
shuffled-label control AUROC is no greater than `0.55`, and in-sample AUROC is
greater than `0.60`. A collapse to chance after origin matching SHALL retire
the structural-energy direction as `complete_structural_energy_s0prime_retired_was_origin_leak`.

#### SCENARIO-ARC-WMTE-4771-ORIGIN-MATCHED-GATE

Given the offline arcade and structural feature spine are importable and banked
games produce induced held-out predictions
When Experiment 4771 runs the origin-matched S0' re-test
Then it writes
`results/experiment_4771_structural_energy_s0prime_origin_matched.json`, keeps
`verifier_is_oracle=false`, reports structural, marginal, majority, per-family,
origin-probe, shuffled-label, per-game balance, and positive-control
measurements, and emits a terminal `success_` verdict only when every S0' gate
criterion is satisfied.

#### SCENARIO-ARC-WMTE-4771-FIELD-PRINCIPLES

The required field principles are: `honest_verdict` = "terminal prefix; a
surviving origin-matched signal is
success_structural_energy_s0prime_reopens_s1, a collapse is
complete_structural_energy_s0prime_retired_was_origin_leak.";
`verifier_is_oracle` = "MUST be false -- the energy never runs the env
win-check; required for check_circular_moat_overclaim.";
`inference_substrate` = "verifier_ensemble_against_cached_candidates -- scores
structural features over cached transitions, no LLM; 1s floor.";
`preconditions_checked` = "records the arcade/feature-import checks so a
silent-missing-resource run cannot fabricate an AUROC.";
`loo_auroc_structural` = "the load-bearing measurement -- cross-game LOO AUROC
on the ORIGIN-MATCHED correct-vs-wrong induced-prediction label.";
`loo_auroc_ci95` = "bootstrap CI95; the lower bound vs 0.5 decides
reopen-vs-retire."; `origin_probe_auroc` = "with origin matched (all induced)
this MUST drop below 0.6 (ideally ~0.5) -- the direct test that the S0 0.733
leak is removed; if it is still high, the matching failed.";
`shuffled_label_control_auroc` = "permuted-label LOO must be <=
0.55 -- the second leak control proving the structural signal is genuinely
label-correlated, not a nuisance path.";
`structural_minus_marginal_delta_ci95` = "must exclude 0 -- structure must
still beat frame-marginals on the clean (origin-matched) label.";
`per_family_loo` = ">=1 family must independently clear 0.55 on the clean label
-- kills the single-lever risk."; `per_game_class_balance` = "records each game's correct/wrong
induced-prediction counts -- a fold needs both classes; an all-one-class game
cannot contribute to LOO."; `in_sample_auroc` = "positive control > 0.60 --
else the harness is broken and the result is uninformative."; `random_seed` =
"determinism is the precondition for reproducibility.";
`reproducibility_checksum` = "content hash of (corpus, folds, induced-engine
fits, features) so a replication catches drift."

### REQ-ARC-WMTE-4788: Energy-Guided Search SOTA Ingestion For S2/S3

Experiment 4788 SHALL run the reserved `.441` SOTA-ingestion slot for
energy-guided search and EBM planning after S1 builds a usable energy
landscape. The workflow SHALL verify that `research-studying.md` and
`research-references.md` are present, read the already-discovered corpus
filtered to energy-guided search, emit focused `scripts/sweep_clusters.py`
cluster URLs for energy-based models and neural/value-guided search, run
`scripts/sweep_semscholar.py` on focused energy-guided search queries, and use
low-concurrency WebSearch/WebFetch plus direct arXiv checks to verify the top
five to eight source papers. The workflow SHALL NOT invoke `/deep-research`,
load models, train, submit to a leaderboard, or make a solve claim.

On a successful ingestion-synthesis run, Experiment 4788 SHALL write
`results/experiment_4788_sota_ingestion_energy_guided_search.json` and update
`research-studying.md` with an idempotent Exp 4788 section marked INGESTED.
The artifact SHALL include the required fields `honest_verdict`,
`methods_mapped`, `arxiv_ids_cited`, `flagged_for_v441`,
`inference_substrate`, `preconditions_checked`, `citations`, `fresh_sweep`,
`s1_context`, `note_path`, `random_seed`, `duration_s`,
`reproducibility_checksum`, and `field_principles`.

The complete-path `honest_verdict` SHALL start with `success_` and equal
`success_sota_ingestion_energy_guided_search_mapped`; `inference_substrate`
SHALL equal `aggregation_from_upstream_artifacts`; and `flagged_for_v441`
SHALL name the strongest method or methods for the `.441` planner. The mapping
SHALL cover S2 and S3 of the live induce->verify->plan loop: energy/value-guided
MCTS and best-first expansion, gradient-guided discrete/local search,
EBM-as-planner trajectory refinement, and product-of-experts planning. The
mapping SHALL state how each method grafts the S1 energy landscape onto the
current loop, what it takes over from the current stack, and where it fails.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; mapping emitted is success_sota_ingestion_energy_guided_search_mapped."
- `methods_mapped`: principle "the strongest 3-5 methods mapped onto S2/S3, each with a real arXiv ID."
- `arxiv_ids_cited`: principle "every method claim must cite a verifiable arXiv ID -- an ingestion with no citations is fabrication."
- `flagged_for_v441`: principle "the strongest method(s) flagged so the .441 planner reads the mapping."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts (0.0001s floor)."

The source set SHALL contain only verified arXiv IDs from the focused pass:
`1909.06878`, `2103.11505`, `2202.11705`, `2206.09914`, `2304.14391`,
`2309.15028`, `2502.07202`, and `2505.10819`. The method map SHALL contain
three to five methods, and every method claim SHALL cite only IDs from
`arxiv_ids_cited`. The note SHALL identify Model Based Planning with EBMs,
Policy-Guided Heuristic Search, COLD Decoding, discrete Langevin proposals,
zero-shot compositional EBM planning, value-guided MCTS decoding, Monte Carlo
Tree Diffusion, and PoE-World/product-of-experts world modeling as the ingested
literature informing S2/S3 search and generation guidance.

#### SCENARIO-ARC-WMTE-4788

Given the research files are present, the reliable sweep helpers are available,
Semantic Scholar is attempted through the reliable script, and the top arXiv
sources are HTTP-200 verified
When Experiment 4788 runs
Then it writes
`results/experiment_4788_sota_ingestion_energy_guided_search.json`, records
that `/deep-research` was not invoked, maps three to five SOTA methods onto
S2/S3 with real arXiv IDs, updates `research-studying.md` with the Exp 4788
mapping note, and flags the strongest candidate inputs for `.441`.

#### SCENARIO-ARC-WMTE-4788-NO-FABRICATION

Given a candidate artifact omits citations, cites an unverified arXiv ID, uses
a stale `.440` roadmap flag, maps only S1/S4 without S2/S3, or claims a
model-load, training, leaderboard, or solve result
When the Experiment 4788 artifact validator runs
Then it rejects the artifact before the result can be written.

### REQ-ARC-WMTE-4798: Energy-Guided Generation SOTA Ingestion For S3

Experiment 4798 SHALL run the reserved `.442` SOTA-ingestion slot for
energy/value-guided generation after S2 grafts the S1 energy as a live trust
gate. The workflow SHALL verify that `research-studying.md` and
`research-references.md` are present, read the already-discovered corpus
filtered to energy-guided generation, emit focused `scripts/sweep_clusters.py`
cluster URLs for energy-based models and neural/value-guided search, run
`scripts/sweep_semscholar.py` on a focused energy-guided generation query, and
use low-concurrency WebSearch/WebFetch plus direct arXiv checks to verify the
top five to eight source papers. The workflow SHALL NOT invoke
`/deep-research`, load models, train, submit to a leaderboard, or make a solve
claim.

On a successful ingestion-synthesis run, Experiment 4798 SHALL write
`results/experiment_4798_sota_ingestion_energy_guided_generation.json` and
update `research-studying.md` with an idempotent Exp 4798 section marked
INGESTED. The artifact SHALL include the required fields `honest_verdict`,
`methods_mapped`, `arxiv_ids_cited`, `flagged_for_v442`,
`inference_substrate`, `preconditions_checked`, `citations`, `fresh_sweep`,
`s1_s2_context`, `note_path`, `random_seed`, `duration_s`,
`reproducibility_checksum`, and `field_principles`.

The complete-path `honest_verdict` SHALL start with `success_` and equal
`success_sota_ingestion_energy_guided_generation_mapped`;
`inference_substrate` SHALL equal `aggregation_from_upstream_artifacts`; and
`flagged_for_v442` SHALL name the strongest method or methods for the `.442`
planner. The mapping SHALL cover S3 of the live explorer by explaining how
each method uses the S1/S2 lower-is-better energy to put a winner into the
candidate pool instead of merely reranking a frozen pool. The mapping SHALL
cover energy-guided sampling, classifier/score guidance, energy/value-guided
tree generation, energy-as-fitness evolutionary search, and plan-with-energy
trajectory generation.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; mapping emitted is success_sota_ingestion_energy_guided_generation_mapped."
- `methods_mapped`: principle "the strongest 3-5 methods mapped onto S3, each with a real arXiv ID."
- `arxiv_ids_cited`: principle "every method claim must cite a verifiable arXiv ID -- an ingestion with no citations is fabrication."
- `flagged_for_v442`: principle "the strongest method(s) flagged so the .442 planner reads the mapping."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts (0.0001s floor)."

The source set SHALL contain only verified arXiv IDs from the focused pass:
`1806.10230`, `1909.06878`, `2012.04322`, `2105.05233`, `2202.11705`,
`2207.12598`, `2309.15028`, and `2502.07202`. The method map SHALL contain
three to five methods, and every method claim SHALL cite only IDs from
`arxiv_ids_cited`. The note SHALL identify COLD Decoding, classifier guidance,
classifier-free score guidance, value-guided MCTS decoding, Monte Carlo Tree
Diffusion, model-based planning with EBMs, Guided Evolutionary Strategies, and
Quality-Diversity Optimization as the ingested literature informing S3
generation guidance.

#### SCENARIO-ARC-WMTE-4798

Given the research files are present, the reliable sweep helpers are available,
Semantic Scholar is attempted through the reliable script, and the top arXiv
sources are HTTP-200 verified
When Experiment 4798 runs
Then it writes
`results/experiment_4798_sota_ingestion_energy_guided_generation.json`,
records that `/deep-research` was not invoked, maps three to five SOTA methods
onto S3 with real arXiv IDs, updates `research-studying.md` with the Exp 4798
mapping note, and flags the strongest candidate inputs for `.442`.

#### SCENARIO-ARC-WMTE-4798-NO-FABRICATION

Given a candidate artifact omits citations, cites an unverified arXiv ID, uses
a stale `.441` roadmap flag, maps only S2 without S3, omits the "put a winner
into the pool" generation handoff, or claims a model-load, training,
leaderboard, or solve result
When the Experiment 4798 artifact validator runs
Then it rejects the artifact before the result can be written.

### REQ-ARC-WMTE-4808: Fresh Energy-Guided Generation SOTA Ingestion For S3

Experiment 4808 SHALL run the reserved `.443` SOTA-ingestion slot for
energy/value-guided generation. The workflow SHALL verify that
`research-studying.md` and `research-references.md` are present, read the
discovered corpus filtered to energy-guided generation, emit focused
`scripts/sweep_clusters.py` cluster URLs for energy-based models and
neural/value-guided search, run `scripts/sweep_semscholar.py` on focused
energy-guided generation queries, and use low-concurrency WebSearch/WebFetch
plus direct arXiv checks to verify the top five to eight source papers. The
workflow SHALL NOT invoke `/deep-research`, load models, train, submit to a
leaderboard, or make a solve claim.

On a successful ingestion-synthesis run, Experiment 4808 SHALL write
`results/experiment_4808_sota_ingestion_energy_guided_generation.json` and
update `research-studying.md` with an idempotent Exp 4808 section marked
INGESTED. The artifact SHALL include the required fields `honest_verdict`,
`methods_mapped`, `arxiv_ids_cited`, `flagged_for_v443`,
`inference_substrate`, `preconditions_checked`, `citations`, `fresh_sweep`,
`s3_context`, `note_path`, `random_seed`, `duration_s`,
`reproducibility_checksum`, and `field_principles`.

The complete-path `honest_verdict` SHALL start with `success_` and equal
`success_sota_ingestion_energy_guided_generation_mapped`;
`inference_substrate` SHALL equal `aggregation_from_upstream_artifacts`; and
`flagged_for_v443` SHALL name the strongest method or methods for the `.443`
planner. The mapping SHALL cover S3 of the live explorer by explaining how
each method uses lower-is-better energy or value feedback to put a winner into the
candidate pool instead of merely reranking a frozen pool. The mapping
SHALL cover energy-guided sampling, classifier/score guidance,
energy/value-guided tree generation, energy-as-fitness evolutionary search,
and plan-with-energy trajectory generation.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; mapping emitted is success_sota_ingestion_energy_guided_generation_mapped."
- `methods_mapped`: principle "the strongest 3-5 methods mapped onto S3, each with a real arXiv ID."
- `arxiv_ids_cited`: principle "every method claim must cite a verifiable arXiv ID."
- `flagged_for_v443`: principle "the strongest method(s) flagged so the .443 planner reads the mapping."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts (0.0001s floor)."

The source set SHALL contain only verified arXiv IDs from the focused pass:
`1806.10230`, `1909.06878`, `2202.11705`, `2207.12598`, `2305.12018`,
`2309.15028`, `2502.07202`, and `2605.28814`. The method map SHALL contain
three to five methods, and every method claim SHALL cite only IDs from
`arxiv_ids_cited`. The note SHALL identify COLD Decoding, BOLT, classifier-free
guidance, value-guided MCTS decoding, Monte Carlo Tree Diffusion,
model-based planning with EBMs, Guided Evolutionary Strategies, and
Bidirectional Evolutionary Search as the ingested literature informing S3
generation guidance.

#### SCENARIO-ARC-WMTE-4808

Given the research files are present, the reliable sweep helpers are available,
Semantic Scholar is attempted through the reliable script, and the top arXiv
sources are HTTP-200 verified
When Experiment 4808 runs
Then it writes
`results/experiment_4808_sota_ingestion_energy_guided_generation.json`,
records that `/deep-research` was not invoked, maps three to five SOTA methods
onto S3 with real arXiv IDs, updates `research-studying.md` with the Exp 4808
mapping note, and flags the strongest candidate inputs for `.443`.

#### SCENARIO-ARC-WMTE-4808-NO-FABRICATION

Given a candidate artifact omits citations, cites an unverified arXiv ID, uses
a stale `.442` roadmap flag, maps only S2 without S3, omits the "put a winner
into the pool" generation handoff, or claims a model-load, training,
leaderboard, or solve result
When the Experiment 4808 artifact validator runs
Then it rejects the artifact before the result can be written.

### REQ-ARC-WMTE-4818: Reserved Energy-Guided Generation SOTA Ingestion For S3

Experiment 4818 SHALL run the reserved `.444` SOTA-ingestion slot for
energy/value-guided generation. The workflow SHALL verify that
`research-studying.md` and `research-references.md` are present, read the
discovered corpus filtered to energy-guided generation, emit focused
`scripts/sweep_clusters.py` cluster URLs for energy-based models and
neural/value-guided search, run `scripts/sweep_semscholar.py` on focused
energy-guided generation queries, and use low-concurrency WebSearch/WebFetch
plus direct arXiv checks to verify the top five to eight source papers. The
workflow SHALL NOT invoke `/deep-research`, load models, train, submit to a
leaderboard, or make a solve claim.

On a successful ingestion-synthesis run, Experiment 4818 SHALL write
`results/experiment_4818_sota_ingestion_energy_guided_generation.json` and
update `research-studying.md` with an idempotent Exp 4818 section marked
INGESTED. The artifact SHALL include the required fields `honest_verdict`,
`methods_mapped`, `arxiv_ids_cited`, `flagged_for_v444`,
`inference_substrate`, `preconditions_checked`, `citations`, `fresh_sweep`,
`s3_context`, `note_path`, `random_seed`, `duration_s`,
`reproducibility_checksum`, and `field_principles`.

The complete-path `honest_verdict` SHALL start with `success_` and equal
`success_sota_ingestion_energy_guided_generation_mapped`;
`inference_substrate` SHALL equal `aggregation_from_upstream_artifacts`; and
`flagged_for_v444` SHALL name the strongest method or methods for the `.444`
planner. The mapping SHALL cover S3 of the live explorer by explaining how
each method uses lower-is-better energy or value feedback to put a winner into
the candidate pool instead of merely reranking a frozen pool. The mapping
SHALL cover energy-guided sampling, classifier/score guidance,
energy/value-guided tree generation, energy-as-fitness evolutionary search,
and plan-with-energy trajectory generation.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; mapping emitted is success_sota_ingestion_energy_guided_generation_mapped."
- `methods_mapped`: principle "the strongest 3-5 methods mapped onto S3, each with a real arXiv ID."
- `arxiv_ids_cited`: principle "every method claim must cite a verifiable arXiv ID."
- `flagged_for_v444`: principle "the strongest method(s) flagged so the .444 planner reads the mapping."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts (0.0001s floor)."

The source set SHALL contain only verified arXiv IDs from the focused pass:
`1806.10230`, `1909.06878`, `2202.11705`, `2207.12598`, `2305.12018`,
`2309.15028`, `2502.07202`, and `2605.28814`. The method map SHALL contain
three to five methods, and every method claim SHALL cite only IDs from
`arxiv_ids_cited`. The note SHALL identify COLD Decoding, BOLT,
classifier-free guidance, value-guided MCTS decoding, Monte Carlo Tree
Diffusion, model-based planning with EBMs, Guided Evolutionary Strategies, and
Bidirectional Evolutionary Search as the ingested literature informing S3
generation guidance.

#### SCENARIO-ARC-WMTE-4818

Given the research files are present, the reliable sweep helpers are available,
Semantic Scholar is attempted through the reliable script, and the top arXiv
sources are HTTP-200 verified
When Experiment 4818 runs
Then it writes
`results/experiment_4818_sota_ingestion_energy_guided_generation.json`,
records that `/deep-research` was not invoked, maps three to five SOTA methods
onto S3 with real arXiv IDs, updates `research-studying.md` with the Exp 4818
mapping note, and flags the strongest candidate inputs for `.444`.

#### SCENARIO-ARC-WMTE-4818-NO-FABRICATION

Given a candidate artifact omits citations, cites an unverified arXiv ID, uses
a stale `.443` roadmap flag, maps only S2 without S3, omits the "put a winner
into the pool" generation handoff, or claims a model-load, training,
leaderboard, or solve result
When the Experiment 4818 artifact validator runs
Then it rejects the artifact before the result can be written.

### REQ-ARC-WMTE-4828: Cross-Family Transfer SOTA Ingestion For S4

Experiment 4828 SHALL run the reserved `.445` SOTA-ingestion slot for
cross-domain and cross-family transfer of learned verifiers, reward models,
and energies. The workflow SHALL verify that `research-studying.md` and
`research-references.md` are present, read the discovered corpus filtered to
cross-family transfer and held-out robustness, emit focused
`scripts/sweep_clusters.py` cluster URLs for verifier/reward, energy-based,
and neural-guided search literature, run `scripts/sweep_semscholar.py` on
focused cross-family transfer queries, and use low-concurrency
WebSearch/WebFetch plus direct arXiv checks to verify the top five to eight
source papers. The workflow SHALL NOT invoke `/deep-research`, load models,
train, submit to a leaderboard, or make a solve claim.

On a successful ingestion-synthesis run, Experiment 4828 SHALL write
`results/experiment_4828_sota_ingestion_cross_family_transfer.json` and
update `research-studying.md` with an idempotent Exp 4828 section marked
INGESTED. The artifact SHALL include the required fields `honest_verdict`,
`methods_mapped`, `arxiv_ids_cited`, `flagged_for_v445`,
`inference_substrate`, `preconditions_checked`, `citations`, `fresh_sweep`,
`s4_context`, `note_path`, `random_seed`, `duration_s`,
`reproducibility_checksum`, and `field_principles`.

The complete-path `honest_verdict` SHALL start with `success_` and equal
`success_sota_ingestion_cross_family_transfer_mapped`;
`inference_substrate` SHALL equal `aggregation_from_upstream_artifacts`; and
`flagged_for_v445` SHALL name the strongest method or methods for the `.445`
planner. The mapping SHALL cover S4 by explaining how each method stress-tests
or trains the energy under held-out family shift instead of accepting a pooled
source-family average. The mapping SHALL cover reward/verifier OOD evaluation,
representation anchoring, worst-family robust optimization, risk extrapolation
or invariant-risk regularization, and transferable-reward stress tests.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; mapping emitted is success_sota_ingestion_cross_family_transfer_mapped."
- `methods_mapped`: principle "the strongest 3-5 methods mapped onto S4, each with a real arXiv ID."
- `arxiv_ids_cited`: principle "every method claim must cite a verifiable arXiv ID."
- `flagged_for_v445`: principle "the strongest method(s) flagged so the .445 planner reads the mapping."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts (0.0001s floor)."

The source set SHALL contain only verified arXiv IDs from the focused pass:
`1911.08731`, `2003.00688`, `2007.01434`, `2012.07421`, `2311.14743`,
`2403.13787`, `2602.08489`, and `2605.25629`. The method map SHALL contain
three to five methods, and every method claim SHALL cite only IDs from
`arxiv_ids_cited`. The note SHALL identify Group DRO, REx, DomainBed, WILDS,
reward-model distribution-shift analysis, RewardBench, transferable reward,
and representation anchoring as the ingested literature informing S4
cross-family transfer.

#### SCENARIO-ARC-WMTE-4828

Given the research files are present, the reliable sweep helpers are available,
Semantic Scholar is attempted through the reliable script, and the top arXiv
sources are HTTP-200 verified
When Experiment 4828 runs
Then it writes
`results/experiment_4828_sota_ingestion_cross_family_transfer.json`, records
that `/deep-research` was not invoked, maps three to five SOTA methods onto S4
with real arXiv IDs, updates `research-studying.md` with the Exp 4828 mapping
note, and flags the strongest candidate inputs for `.445`.

#### SCENARIO-ARC-WMTE-4828-NO-FABRICATION

Given a candidate artifact omits citations, cites an unverified arXiv ID, uses
a stale `.444` roadmap flag, maps only S3 without S4, omits the held-out family
handoff, or claims a model-load, training, leaderboard, or solve result
When the Experiment 4828 artifact validator runs
Then it rejects the artifact before the result can be written.

### REQ-ARC-WMTE-4821: S3 Structural-Energy Generation Lift

Experiment 4821 SHALL run the S3 generation-lift test authorized by the
S2-v3 corpus-wide bounded selection result. The live `E3AgentPolicy`
induce->verify->plan path SHALL pass a lower-is-better goal-energy callable
into `arc_executable_world_model.plan_in_model` when `goal_guidance_lambda > 0`
and SHALL pass no goal-energy guidance when `goal_guidance_lambda == 0`, making
lambda zero the matched bare-explorer control. The guidance SHALL be reachable
from the submitted `E3AgentPolicy` import closure and `scripts/arc_orphan_solver_lint.py`
SHALL pass before the artifact can claim `live_path_reachable=true`.

The S3 measurement SHALL select held-out games where the matched lambda-zero
bare explorer banks zero levels and a positive-control source marks the winner
reachable in the offline environment. If fewer than five such headroom games are
available, the artifact SHALL stop as
`complete_structural_energy_s3_inconclusive_no_generation_headroom`. Otherwise
it SHALL compare the E-guided arm against the matched lambda-zero arm and count
only winners newly entering the pool under E-guidance that the bare control did
not produce. Any banked level SHALL be gated by `arc_solver_kit.reproduce` with
`offline_reproduced=true`; a banked level that was already in the bare pool
SHALL be recorded as re-ranking, not generation.

Experiment 4821 SHALL write
`results/experiment_4821_structural_energy_s3_generation_lift.json` with
principle-annotated fields for `honest_verdict`, `verifier_is_oracle`,
`live_path_reachable`, `inference_substrate`, `n_headroom_games`,
`winners_newly_entering_pool_delta_ci95`, `new_levels_not_in_bare_pool`,
`solve_provenance`, `game_results`, `positive_control_passed`, `random_seed`,
and `reproducibility_checksum`. The artifact SHALL set `verifier_is_oracle=false`
and `solve_provenance=live_agent_self_discovery`.

The PASS verdict SHALL be
`success_structural_energy_s3_generation_authorizes_s4` only if at least five
headroom games are tested, the E-guided arm banks at least one new
offline-reproduced level not present in the bare pool, and the bootstrap CI95
for the winners-newly-entering-pool delta excludes zero. If the CI95 includes
zero, or every banked level was already in the bare pool, the verdict SHALL be
`complete_structural_energy_s3_bounded_no_generation_lift`. The artifact SHALL
set `retire_if_same_verdict=true`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; a generation win is success_structural_energy_s3_generation_authorizes_s4, a null is complete_structural_energy_s3_bounded_no_generation_lift, insufficient headroom is complete_structural_energy_s3_inconclusive_no_generation_headroom."
- `verifier_is_oracle`: principle "MUST be false -- the energy guides generation, oracle-distinct; required for check_circular_moat_overclaim."
- `live_path_reachable`: principle "the goal_energy guidance must be in the E3AgentPolicy plan_in_model import closure (arc_orphan_solver_lint) -- a guidance the live agent cannot reach adds no live value."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates (the energy guides over cached candidates; if the live LLM induces fresh, declare live_llm_inference)."
- `n_headroom_games`: principle ">=5 held-out games where the bare explorer banks 0 AND the winner is reachable -- the ONLY games that test generation (positive control: generation COULD help)."
- `winners_newly_entering_pool_delta_ci95`: principle "the decisive metric -- E-guided minus lambda=0 fraction of winners NEWLY entering the pool; must EXCLUDE 0 for a PASS."
- `new_levels_not_in_bare_pool`: principle "any banked level must NOT have been already in the bare explorer's pool -- else it is re-ranking, not generation (the second killer)."
- `solve_provenance`: principle "live_agent_self_discovery -- a generation lift is the agent solving via its OWN E-guided attempts; offline_reproduced gated."
- `game_results`: principle "per-game: winner-rank, banked-by-E, banked-by-bare, was-already-in-bare-pool -- the per-game generation log."
- `positive_control_passed`: principle "the headroom games' winners ARE reachable -- so a null means 'energy could have generated the winner but didn't', not 'no winner existed'."
- `random_seed`: principle "determinism for the guided/unguided runs + bootstrap."
- `reproducibility_checksum`: principle "content hash of (energy, games, lambda, seeds) so a replication catches drift."

#### SCENARIO-ARC-WMTE-4821-GENERATION-LIFT

Given the offline arcade is available, `E3AgentPolicy` imports, the S2-v3
structural-energy artifact is present, and at least five positive-control
headroom games are available
When Experiment 4821 compares E-guided planning against the lambda-zero matched
control
Then it writes
`results/experiment_4821_structural_energy_s3_generation_lift.json`, records
per-game winner-rank, E-banked, bare-banked, and already-in-bare-pool fields,
keeps `verifier_is_oracle=false`, and emits a success verdict only when the
new-winner delta CI95 excludes zero.

#### SCENARIO-ARC-WMTE-4821-LIVE-PLAN-WIRING

Given an `E3AgentPolicy` has an induced goal predicate and
`goal_guidance_lambda > 0`
When the live induce->verify->plan path calls `plan_in_model`
Then the call includes a `goal_energy` guidance callable; when
`goal_guidance_lambda == 0`, the same path calls `plan_in_model` without
goal-energy guidance.

### REQ-ARC-WMTE-4825: .444 ARC Null Silent-Bug Audit

Experiment 4825 SHALL audit the `.444` ARC null artifacts before their negative
verdicts are treated as capability limits. The audit SHALL read
`results/experiment_4821_structural_energy_s3_generation_lift.json`,
`results/experiment_4822_levelup_attempt.json`, and
`results/experiment_4824_heldout_first_win_readiness.json`. Missing `.444`
source artifacts SHALL produce a blocked artifact rather than a fabricated
audit. Experiment 4823 is a self-play checkpoint slot and SHALL be excluded
from the null count unless it emits a terminal ARC null artifact.

For S3/Experiment 4821, the audit SHALL verify the matched lambda-zero control:
the guided and lambda-zero arms use the same games, seeds, and budget; the
artifact reports `lambda_guidance > 0` and a `lambda0_control` with
`lambda=0.0` and `matched_control=true`; the per-game headroom set has at least
the required minimum; all headroom games have `positive_control_reachable=true`;
and any E-banked level counted as new is not already in the bare explorer pool.
The audit SHALL independently confirm live-path reachability from the artifact's
`live_path_reachable=true` and a passing `arc_orphan_solver_lint` precondition.

The audit SHALL additionally verify that S3 guidance was non-trivially
exercised. `s3_guidance_exercised` SHALL be true only when the S3 artifact
contains positive evidence that the lambda-one run produced different candidate
proposals, search states, explored nodes, or equivalent search fingerprints
than the matched lambda-zero control on at least one headroom game. If the
matched controls pass but no such differing proposal/search evidence exists,
the audit SHALL classify S3 as `inconclusive_guidance_no_op` rather than a
trustworthy generation null.

For the other ARC nulls, the audit SHALL verify that the lever was genuinely
exercised. The level-up no-bank null SHALL have non-empty attempted-game rows,
target/depth accounting, same-depth or terminal residual evidence, reproduction
gates for reproduced depths, and same-depth versus new-depth accounting. The
held-out first-win null SHALL have the held-out attempt floor, positive-control
parity, parity-test success, proxy aggregation evidence, and a non-empty flat
`0.04` methodology note.

Experiment 4825 SHALL write `results/experiment_4825_silent_bug_audit.json`
and append a human-readable section to `ops/arc_null_silent_bug_audit.md`. The
artifact SHALL include principle-annotated top-level fields for
`honest_verdict`, `s3_controls_verified`, `s3_guidance_exercised`,
`nulls_audited`, and `inference_substrate`, plus per-null verdicts,
silent-bug findings, trusted nulls, S3 control checks, preconditions, checksums
for audited artifacts, duration, random seed, and a stable reproducibility
checksum. The audit SHALL make no solve, leaderboard, training, ops-status, or
fresh live-inference claim.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; audit complete is complete_/success_."
- `s3_controls_verified`: principle "the load-bearing check -- matched lambda=0, NEW-levels-not-re-ranking, reachable-winner positive control; else S3's verdict is uninterpretable."
- `s3_guidance_exercised`: principle "true only if lambda=1 produced different candidate proposals/search than lambda=0 on >=1 headroom game -- else the S3 0.0 delta is a no-op, not a genuine generation null."
- `nulls_audited`: principle "count of nulls re-examined."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts (0.0001s floor)."

#### SCENARIO-ARC-WMTE-4825-SILENT-BUG-AUDIT

Given the `.444` S3 generation-lift, level-up no-bank, and held-out first-win
null artifacts are present
When Experiment 4825 cross-checks exercised-evidence fields and S3 matched
control evidence
Then it writes `results/experiment_4825_silent_bug_audit.json`, sets
`s3_controls_verified=true` only when the lambda-zero control is matched,
headroom winners are reachable, NEW levels are not re-ranking, and live-path
reachability is confirmed, sets `s3_guidance_exercised=true` only when
proposal/search evidence differs between guided and bare arms on at least one
headroom game, and marks S3 `inconclusive_guidance_no_op` when the guidance was
not exercised.

#### SCENARIO-ARC-WMTE-4825-BLOCKED-PRECONDITION

Given any required `.444` source artifact is missing
When Experiment 4825 runs
Then it writes a blocked artifact with the missing source in
`preconditions_checked`, no silent-bug fabrication, no markdown append, and a
stable reproducibility checksum.

### REQ-ARC-WMTE-4781: S1 Contrastive Structural Energy Landscape

Experiment 4781 SHALL promote the S0' origin-matched structural signal into a
contrastive lower-is-better transition energy over induced held-out
predictions. The workflow SHALL first verify that
`arc_solver_kit.offline_arcade()` exits successfully and that
`cross_game_features_v3` imports. If either precondition fails, the artifact
SHALL stop with `honest_verdict=blocked_offline_arcade_missing` or
`honest_verdict=blocked_structural_features_missing` and SHALL NOT report a
fabricated AUROC.

The dataset SHALL reuse the Experiment 4771 S0' origin-matched construction:
positive rows are induced predictions that match the held-out observed next
grid, negative rows are induced predictions that do not match that held-out
grid, and both classes have induced origin. The energy SHALL use only the
`object_relational` and `frame_delta` slices from `cross_game_features_v3` for
the primary structural model. It SHALL train a contrastive margin or NCE-style
linear energy across at least ten deterministic random seeds so that lower
energy ranks correct transitions above wrong near misses, then report
leave-one-GAME-out AUROC ranking by -E.

Experiment 4781 SHALL write
`results/experiment_4781_structural_energy_s1_contrastive_landscape.json`, set
`verifier_is_oracle=false`, and include principle-annotated top-level fields
for `honest_verdict`, `verifier_is_oracle`, `inference_substrate`,
`preconditions_checked`, `energy_ranking_loo_auroc_mean`,
`energy_ranking_loo_auroc_ci95`, `n_seeds`,
`denoising_direction_agreement`, `origin_probe_auroc`,
`shuffled_label_control_auroc`, `per_family_loo`, `in_sample_auroc`,
`random_seeds_used`, and `reproducibility_checksum`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; a usable energy landscape is success_structural_energy_s1_landscape_authorizes_s2, a bounded result is complete_structural_energy_s1_discriminates_no_usable_landscape."
- `verifier_is_oracle`: principle "MUST be false -- the energy never runs the env win-check; required for check_circular_moat_overclaim."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates (scores structural features over cached transitions, no LLM; 1s floor)."
- `preconditions_checked`: principle "records the arcade/feature-import checks so a silent-missing-resource run cannot fabricate an AUROC."
- `energy_ranking_loo_auroc_mean`: principle "the load-bearing measurement -- mean cross-game LOO AUROC ranking by -E across >=10 seeds."
- `energy_ranking_loo_auroc_ci95`: principle ">=0.70 with CI95 excluding chance across seeds is the bar -- multi-seed is what S0' (single-seed) did not establish."
- `n_seeds`: principle ">=10 -- the single-seed S0' result must be shown robust before the energy is trusted as a landscape."
- `denoising_direction_agreement`: principle "the energy-vs-classifier distinction -- -deltaE descent must point toward correctness, a property a point classifier lacks."
- `origin_probe_auroc`: principle "carry the S0' leak control -- must stay < 0.6 (origin matched); a regression means the contrastive training reintroduced an origin shortcut."
- `shuffled_label_control_auroc`: principle "<= 0.55 -- the second leak control must hold under contrastive training."
- `per_family_loo`: principle ">=2 independent structural families each >= 0.60 -- kills the frame_delta-single-lever risk for the energy."
- `in_sample_auroc`: principle "positive control > 0.60 -- else the harness is broken."
- `random_seeds_used`: principle "the list of >=10 seeds -- determinism + reproducibility for the multi-seed claim."
- `reproducibility_checksum`: principle "content hash of (corpus, folds, energy training config) so a replication catches drift."

The S1 gate SHALL pass only when cross-game energy-ranking LOO AUROC is at
least `0.70` with CI95 excluding chance across at least ten seeds, at least two
independent structural families each clear `0.60` LOO, the denoising-direction
agreement is materially above chance, origin-probe AUROC remains below `0.6`,
shuffled-label control AUROC is no greater than `0.55`, and the in-sample
positive control exceeds `0.60`. If any gate fails, the artifact SHALL report
`complete_structural_energy_s1_discriminates_no_usable_landscape` instead of
authorizing S2.

#### SCENARIO-ARC-WMTE-4781-CONTRASTIVE-LANDSCAPE-GATE

Given the S0' origin-matched dataset can be rebuilt and the structural feature
spine is importable
When Experiment 4781 trains the multi-seed contrastive energy landscape
Then it writes
`results/experiment_4781_structural_energy_s1_contrastive_landscape.json`,
keeps `verifier_is_oracle=false`, reports multi-seed LOO ranking,
frame-marginal control, denoising-direction, origin-probe, shuffled-label,
per-family, and positive-control measurements, and emits
`success_structural_energy_s1_landscape_authorizes_s2` only when every S1 gate
criterion is satisfied.

### REQ-ARC-WMTE-4791: S2 Off-Path Structural-Energy Trust Gate

Experiment 4791 SHALL graft the Experiment 4781 S1 structural transition energy
into the live ARC world-model trust path by making
`WorldModelVerifier`/`E3AgentPolicy` rank candidate induced engines with
off-path predicted-transition energy before planning. The energy ranking SHALL
score only candidate predictions on held-out off-path transitions, without
calling the executable oracle, win predicate, or observed next grid. The matched
control SHALL use the incumbent binary `WorldModelVerifier` exact-accuracy
`<0.5` cutoff on the identical candidate engine sets.

The workflow SHALL first verify that `arc_solver_kit.offline_arcade()` exits
successfully and that `WorldModelVerifier` imports. If either precondition
fails, the artifact SHALL stop with
`honest_verdict=blocked_offline_arcade_missing` or
`honest_verdict=blocked_world_model_verifier_missing` and SHALL NOT fabricate a
held-out off-path `cell_recall`.

Experiment 4791 SHALL write
`results/experiment_4791_structural_energy_s2_offpath_trust_gate.json`, set the
energy arm `verifier_is_oracle=false`, confirm `live_path_reachable=true` by
passing `scripts/arc_orphan_solver_lint.py`, and include principle-annotated
top-level fields for `honest_verdict`, `verifier_is_oracle`,
`live_path_reachable`, `inference_substrate`, `preconditions_checked`,
`energy_selected_offpath_cell_recall`,
`accuracy_gate_selected_offpath_cell_recall`,
`energy_minus_accuracy_delta_ci95`, `n_heldout_games`, `random_seed`, and
`reproducibility_checksum`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; a live trust-gate win is success_structural_energy_s2_trust_gate_authorizes_s3, a bound is complete_structural_energy_s2_no_live_trust_value."
- `verifier_is_oracle`: principle "MUST be false for the energy ranking (oracle-distinct); the binary-accuracy CONTROL is execution-grounded (true) -- S2 measures whether oracle-distinct energy beats execution-grounded selection."
- `live_path_reachable`: principle "the energy gate must be in the E3AgentPolicy import closure (arc_orphan_solver_lint passes) -- a gate the live agent cannot reach adds no live value (ARC Live-Path Reachability Discipline)."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates (scores the energy over cached candidate induced engines off-path predictions, no fresh LLM; 1s floor)."
- `preconditions_checked`: principle "records the arcade / WorldModelVerifier import checks so a silent-missing-resource run cannot fabricate a cell_recall."
- `energy_selected_offpath_cell_recall`: principle "the held-out off-path cell_recall of the engine the structural energy selected -- the load-bearing measurement."
- `accuracy_gate_selected_offpath_cell_recall`: principle "the held-out off-path cell_recall of the engine the incumbent binary accuracy<0.5 gate selected -- the control the energy must beat."
- `energy_minus_accuracy_delta_ci95`: principle "must EXCLUDE 0 (energy > accuracy-gate) for S2 to pass -- the oracle-distinct value-add over cheap execution-grounded selection."
- `n_heldout_games`: principle ">=N held-out games -- the delta must hold across games, not a single-game artifact."
- `random_seed`: principle "determinism for reproducibility of the selection + bootstrap."
- `reproducibility_checksum`: principle "content hash of (engine candidates, folds, energy config) so a replication catches drift."

The S2 gate SHALL pass only when the energy-selected engines' mean held-out
off-path `cell_recall` minus the binary-accuracy-gate-selected engines' mean
held-out off-path `cell_recall` is greater than zero with a bootstrap CI95 that
excludes zero across at least five held-out hidden-state games. If the CI95
includes zero, the artifact SHALL report
`complete_structural_energy_s2_no_live_trust_value` with
`retire_if_same_verdict=true`.

#### SCENARIO-ARC-WMTE-4791-OFFPATH-TRUST-GATE

Given the S1 structural energy artifact exists, the offline arcade and
`WorldModelVerifier` import preconditions pass, and cached candidate induced
engines are available for held-out hidden-state ARC games
When Experiment 4791 ranks each game's identical candidate engine set by
off-path structural energy and by the incumbent binary exact-accuracy gate
Then it writes
`results/experiment_4791_structural_energy_s2_offpath_trust_gate.json`, keeps
the energy arm `verifier_is_oracle=false`, records the binary-control oracle
declaration separately, confirms the gate is live-path reachable through
`arc_orphan_solver_lint.py`, reports both selected engines' held-out off-path
`cell_recall`, and emits
`success_structural_energy_s2_trust_gate_authorizes_s3` only when the
energy-minus-accuracy bootstrap CI95 excludes zero above zero.

### REQ-ARC-WMTE-4801: S2-v2 Diverse-Pool Structural-Energy Trust Gate

Experiment 4801 SHALL rerun the S2 off-path trust gate with behaviorally
diverse candidate pools instead of the degenerate Experiment 4791 cached-pair
pools. For each held-out hidden-state game, the workflow SHALL assemble at
least three genuinely induced candidate engines when available, using varied
refinement rounds, cached induction seeds, and deterministic programmatic
expert-induction snapshots derived from prefix transitions. It SHALL NOT inject
a deliberately broken zero-recall engine to manufacture diversity.

The workflow SHALL first verify that `arc_solver_kit.offline_arcade()` exits
successfully and that `WorldModelVerifier` imports. If either precondition
fails, the artifact SHALL stop with
`honest_verdict=blocked_offline_arcade_missing` or
`honest_verdict=blocked_world_model_verifier_missing` and SHALL NOT fabricate a
held-out off-path `cell_recall`.

A game is EFFECTIVE only when its candidate rows contain at least three
candidate engines and the exact `candidate_rows[*].heldout_cell_recall` field
spans more than `1e-3` across the pool. Experiment 4801 SHALL compute
`n_effective_games` from that exact field, SHALL set `min_heldout_games` to
`n_effective_games`, SHALL require `n_effective_games >= 5` before drawing a
bounded or passing selection conclusion, and SHALL report
`complete_structural_energy_s2v2_inconclusive_insufficient_candidate_diversity`
if fewer than five effective games can be assembled.

Experiment 4801 SHALL write
`results/experiment_4801_structural_energy_s2v2_diverse_trust_gate.json`, set
the energy arm `verifier_is_oracle=false`, confirm `live_path_reachable=true`
by passing `scripts/arc_orphan_solver_lint.py`, and include principle-annotated
top-level fields for `honest_verdict`, `verifier_is_oracle`,
`n_effective_games`, `min_heldout_games`, `game_results`,
`candidate_pool_diversity`, `energy_minus_accuracy_delta_ci95`,
`live_path_reachable`, `preconditions_checked`, `positive_control_passed`,
`false_negative_risk_checked`, `candidates_genuinely_induced`, `random_seed`,
and `reproducibility_checksum`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; a live trust win is success_structural_energy_s2v2_trust_gate_authorizes_s3, a genuine null on a diverse pool is complete_structural_energy_s2v2_bounded_diverse_pool, an under-diverse non-test is complete_structural_energy_s2v2_inconclusive_insufficient_candidate_diversity."
- `verifier_is_oracle`: principle "MUST be false for the energy ranking (oracle-distinct); the binary-accuracy CONTROL is execution-grounded."
- `n_effective_games`: principle "games whose candidate pool has >=2 distinct held-out off-path cell_recall -- the ONLY games that genuinely test selection; the exp4791 failure was n_effective=2 of 5. MUST equal the linter-computed effective count (cell_recall spread > 1e-3); a mismatch is a CRITICAL fabrication."
- `min_heldout_games`: principle "set to n_effective; check_engine_selection_candidate_diversity requires effective>=this, so a degenerate pool cannot pass as a null."
- `game_results`: principle "per-game candidate_rows + energy-selected vs accuracy-selected candidate + each off-path cell_recall -- the per-game logging exp4791 lacked (it only reported the aggregate)."
- `candidate_pool_diversity`: principle "records distinct off-path cell_recall count per game -- proves the pool genuinely spans generalization, not near-duplicate induction snapshots."
- `energy_minus_accuracy_delta_ci95`: principle "must EXCLUDE 0 (energy>accuracy) for PASS, on >=5 EFFECTIVE games -- a 0-delta is only meaningful on a diverse pool."
- `live_path_reachable`: principle "the energy gate must be in the E3AgentPolicy import closure (arc_orphan_solver_lint)."
- `preconditions_checked`: principle "records the arcade / WorldModelVerifier import checks so a silent-missing-resource run cannot fabricate a cell_recall."
- `positive_control_passed`: principle "REQUIRED for a BOUNDED verdict -- a documented oracle/headroom upper bound proving the diverse pool CONTAINS a candidate the accuracy gate misses, so BOUNDED means 'energy could have won but didn't', not a degenerate non-test."
- `false_negative_risk_checked`: principle "engages check_false_negative_risk on the bounded verdict -- a null is only trustworthy with a passing positive control."
- `candidates_genuinely_induced`: principle "the diversity spread must come from genuinely-induced engines (varied rounds/seeds), NOT an injected broken 0.0-recall sabotage candidate -- a PASS off a sabotage pool is invalid."
- `random_seed`: principle "determinism for the candidate generation + selection + bootstrap."
- `reproducibility_checksum`: principle "content hash of (candidate engines, folds, energy config) so a replication catches drift."

The S2-v2 gate SHALL pass only when, across at least five effective games, the
energy-selected engines' mean held-out off-path `cell_recall` minus the
binary-accuracy-gate-selected engines' mean held-out off-path `cell_recall` is
greater than zero with a bootstrap CI95 that excludes zero. If at least five
effective games exist and the CI95 includes zero, the artifact SHALL report
`complete_structural_energy_s2v2_bounded_diverse_pool` only when
`positive_control_passed=true` and `false_negative_risk_checked=true`; otherwise
it SHALL remain inconclusive.

#### SCENARIO-ARC-WMTE-4801-DIVERSE-TRUST-GATE

Given the S1 structural energy artifact exists, the offline arcade and
`WorldModelVerifier` import preconditions pass, and cached or deterministic
programmatic induced engines are available for held-out hidden-state ARC games
When Experiment 4801 ranks only effective games by off-path structural energy
and by the incumbent binary exact-accuracy gate
Then it writes
`results/experiment_4801_structural_energy_s2v2_diverse_trust_gate.json`, keeps
the energy arm `verifier_is_oracle=false`, logs every effective game's
`candidate_rows[*].heldout_cell_recall`, records whether the two selectors
differed, records both selectors' off-path `cell_recall`, keeps skipped or
under-diverse attempts in `candidate_pool_diversity`, and emits a pass only when
the energy-minus-accuracy bootstrap CI95 excludes zero above zero.

### REQ-ARC-WMTE-4811: S2-v3 Corpus-Wide Structural-Energy Trust Gate

Experiment 4811 SHALL settle the S2 engine-selection question corpus-wide
instead of repeating Experiment 4801's five-game narrow sample. The workflow
SHALL read the complete offline corpus from `environment_files/`, record
`n_available_games` as that corpus size, attempt every available game, and
record `n_games_attempted` as the number of games for which candidate generation
and scoring was attempted. `n_games_attempted` SHALL match `n_available_games`
unless an explicit blocked precondition stops the run before scoring.

For each attempted game, the workflow SHALL actively assemble at least three
genuinely induced candidate engines from varied deterministic induction rounds,
refinement snapshots, seeds, or temperatures. It SHALL NOT rely only on
near-duplicate `world_model.py` / `world_model.best.py` cached pairs and SHALL
NOT inject a deliberately broken zero-recall engine to manufacture diversity.
A game is EFFECTIVE only when its `candidate_rows[*].heldout_cell_recall` spans
more than `1e-3` across the candidate pool. The required effective count SHALL
be `max(10, ceil(0.6 * max(n_games_attempted, n_available_games)))`; a 25-game
corpus therefore requires at least 15 effective games before a PASS or BOUNDED
selection conclusion is valid.

Experiment 4811 SHALL write
`results/experiment_4811_structural_energy_s2v3_corpus_wide_trust_gate.json`,
set the energy arm `verifier_is_oracle=false`, keep the binary-accuracy control
execution-grounded, confirm `live_path_reachable=true` by passing
`scripts/arc_orphan_solver_lint.py`, and include principle-annotated top-level
fields for `honest_verdict`, `verifier_is_oracle`, `n_available_games`,
`n_games_attempted`, `n_effective_games`, `game_results`,
`energy_minus_accuracy_delta_ci95`, `positive_control_passed`,
`candidates_genuinely_induced`, `live_path_reachable`, `preconditions_checked`,
`random_seed`, and `reproducibility_checksum`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; a corpus-wide trust win is success_structural_energy_s2v3_trust_gate_authorizes_s3, a genuine corpus-wide null is complete_structural_energy_s2v3_bounded_corpus_wide, an under-covered result is complete_structural_energy_s2v3_inconclusive_insufficient_corpus_diversity."
- `verifier_is_oracle`: principle "MUST be false for the energy ranking (oracle-distinct); the binary-accuracy CONTROL is execution-grounded."
- `n_available_games`: principle "the full offline corpus size (ls environment_files/) -- the .443 B1 audit verifies this against the real corpus; the gate requires effective >= max(10, 0.6*this)."
- `n_games_attempted`: principle "must ~= n_available_games -- S2-v2's failure was testing 5 of 25; a corpus-wide test attempts ALL games."
- `n_effective_games`: principle "games whose candidate pool spans > 1e-3 off-path cell_recall -- must be >= max(10, ceil(0.6*n_available)) for a PASS/BOUNDED verdict; else INCONCLUSIVE."
- `game_results`: principle "per-game candidate_rows (heldout_cell_recall + offpath_structural_energy + accuracy) + energy-selected vs accuracy-selected candidate -- the corpus-wide per-game log."
- `energy_minus_accuracy_delta_ci95`: principle "must EXCLUDE 0 for PASS, on >= the required effective games -- a corpus-wide delta, not n=5."
- `positive_control_passed`: principle "REQUIRED for a BOUNDED verdict -- a documented headroom upper bound proving the diverse pool contains a candidate the accuracy gate misses, so BOUNDED means 'energy could have won but didn't'."
- `candidates_genuinely_induced`: principle "the diversity must come from genuinely-induced engines (varied rounds/seeds), NOT an injected broken 0.0-recall sabotage candidate."
- `live_path_reachable`: principle "the energy gate must be in the E3AgentPolicy import closure (arc_orphan_solver_lint)."
- `preconditions_checked`: principle "records the arcade / corpus checks so a silent-missing-resource run cannot fabricate a cell_recall."
- `random_seed`: principle "determinism for the candidate generation + selection + bootstrap."
- `reproducibility_checksum`: principle "content hash of (candidate engines, folds, energy config) so a replication catches drift."

The S2-v3 gate SHALL pass only when `n_effective_games` meets the corpus-wide
required effective count and the energy-selected engines' mean held-out
off-path `cell_recall` minus the binary-accuracy-gate-selected engines' mean
held-out off-path `cell_recall` is greater than zero with a bootstrap CI95 that
excludes zero. If that many effective games exist and the CI95 includes zero,
the artifact SHALL report
`complete_structural_energy_s2v3_bounded_corpus_wide` only when
`positive_control_passed=true`. If fewer effective games can be formed after
attempting the full corpus, the artifact SHALL report
`complete_structural_energy_s2v3_inconclusive_insufficient_corpus_diversity`
and SHALL retire the off-path-trust-gate framing via `retire_if_same_verdict`.

#### SCENARIO-ARC-WMTE-4811-CORPUS-WIDE-TRUST-GATE

Given the offline arcade precondition passes and the offline corpus contains at
least 20 games
When Experiment 4811 actively generates varied induced candidate pools for all
available games and ranks effective games by off-path structural energy versus
the incumbent binary exact-accuracy gate
Then it writes
`results/experiment_4811_structural_energy_s2v3_corpus_wide_trust_gate.json`,
records `n_available_games`, `n_games_attempted`, `n_effective_games`, and
every attempted game's `candidate_rows[*].heldout_cell_recall`, keeps the
energy arm `verifier_is_oracle=false`, and emits a PASS or BOUNDED verdict only
when the effective-game count clears the corpus-wide floor.

### REQ-ARC-WMTE-4815: .443 ARC Null Silent-Bug Audit

Experiment 4815 SHALL audit the `.443` ARC null artifacts before their negative
verdicts are treated as capability limits. The audit SHALL read
`results/experiment_4811_structural_energy_s2v3_corpus_wide_trust_gate.json`,
`results/experiment_4812_levelup_attempt.json`, and
`results/experiment_4814_heldout_first_win_readiness.json`. Missing `.443`
source artifacts SHALL produce a blocked artifact rather than a fabricated
audit. Experiment 4813 is a successful self-play verifier checkpoint refresh
and SHALL be excluded from the null count.

For S2-v3/Experiment 4811, the audit SHALL verify that `n_available_games`
matches the real corpus size from `environment_files/`, that
`n_games_attempted` equals `n_available_games`, that the effective-game floor is
`max(10, ceil(0.6 * max(n_games_attempted, n_available_games)))`, and that
`adversarial_verify.check_engine_selection_candidate_diversity` does not emit
`DEGENERATE_CANDIDATE_POOL`. The audit SHALL classify S2-v3 as a silent bug if
the artifact under-declares the corpus, attempts only a subset, misses the
effective-game floor, lacks per-game candidate-selection logs, or trips
`DEGENERATE_CANDIDATE_POOL`.

For the other ARC nulls, the audit SHALL verify that the lever was genuinely
exercised. The level-up no-bank null SHALL have non-empty attempted-game rows,
target/depth accounting, same-depth or terminal residual evidence, reproduction
gates for reproduced depths, and same-depth versus new-depth accounting. The
held-out first-win null SHALL have the held-out attempt floor, positive-control
parity, parity-test success, proxy aggregation evidence, and a non-empty flat
`0.04` methodology note.

Experiment 4815 SHALL write `results/experiment_4815_silent_bug_audit.json`
with principle-annotated top-level fields for `honest_verdict`,
`s2v3_corpus_coverage_verified`, `nulls_audited`, and `inference_substrate`,
plus per-null verdicts, silent-bug findings, trusted nulls, the S2-v3 corpus
coverage check, preconditions, checksums for audited artifacts, duration,
random seed, and a stable reproducibility checksum. The audit SHALL make no
solve, leaderboard, training, ops-document, or fresh live-inference claim.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; audit complete is complete_/success_."
- `s2v3_corpus_coverage_verified`: principle "the load-bearing check -- n_available_games must match the real corpus AND DEGENERATE_CANDIDATE_POOL must not fire, else S2-v3's verdict is again under-covered."
- `nulls_audited`: principle "count of nulls re-examined."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts (0.0001s floor)."

#### SCENARIO-ARC-WMTE-4815-SILENT-BUG-AUDIT

Given the `.443` S2-v3, level-up no-bank, and held-out first-win null artifacts
are present
When Experiment 4815 cross-checks exercised-evidence fields and runs
`adversarial_verify.check_engine_selection_candidate_diversity`
Then it writes `results/experiment_4815_silent_bug_audit.json`, sets
`s2v3_corpus_coverage_verified=true` only when the declared corpus count equals
`environment_files/`, `n_games_attempted == n_available_games`, the effective
count clears the corpus-wide floor, and `DEGENERATE_CANDIDATE_POOL` does not
fire, keeps the other ARC nulls trusted only when their exercised-evidence
fields are non-degenerate, and reports the complete audit with
`inference_substrate=aggregation_from_upstream_artifacts`.

#### SCENARIO-ARC-WMTE-4815-BLOCKED-PRECONDITION

Given any required `.443` source artifact is missing
When Experiment 4815 runs
Then it writes a blocked artifact with the missing source in
`preconditions_checked`, no silent-bug fabrication, and a stable
reproducibility checksum.

### REQ-ARC-WMTE-4768: Structural-Energy SOTA Ingestion For S1-S4

Experiment 4768 SHALL run the reserved `.438` SOTA-ingestion slot for the
structural-energy build after the S0 transition-correctness probe. The workflow
SHALL verify that `research-studying.md` and `research-references.md` are
present, read the already-discovered corpus for structural-energy and
object-centric world-model references, emit focused `scripts/sweep_clusters.py`
cluster URLs for energy-based models and world-model induction, run
`scripts/sweep_semscholar.py` on focused structural-energy queries, and use
low-concurrency WebSearch/WebFetch plus direct arXiv checks to verify the top
five to eight source papers. The workflow SHALL NOT invoke `/deep-research`,
load models, train, submit to a leaderboard, or make a solve claim.

On a successful ingestion-synthesis run, Experiment 4768 SHALL write
`results/experiment_4768_sota_ingestion_structural_energy.json` and update
`research-studying.md` with an idempotent Exp 4768 section marked INGESTED. The
artifact SHALL include the required fields `honest_verdict`, `methods_mapped`,
`arxiv_ids_cited`, `flagged_for_v439`, `inference_substrate`,
`preconditions_checked`, `citations`, `fresh_sweep`, `s0_context`,
`note_path`, `random_seed`, `duration_s`, `reproducibility_checksum`, and
`field_principles`.

The complete-path `honest_verdict` SHALL start with `success_` and equal
`success_sota_ingestion_structural_energy_mapped`; `inference_substrate` SHALL
equal `aggregation_from_upstream_artifacts`; and `flagged_for_v439` SHALL name
the strongest method or methods for the `.439` planner. The mapping SHALL cover
S1 through S4 of the structural-energy program: contrastive structural energy,
off-path trust gating, generation lift, and cross-game/family transfer
survival. The mapping SHALL state what each method takes over from the current
stack and where it fails.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; mapping emitted is success_sota_ingestion_structural_energy_mapped."
- `methods_mapped`: principle "the strongest 3-5 methods mapped onto S1-S4 -- the actionable output, each with a real arXiv ID."
- `arxiv_ids_cited`: principle "every method claim must cite a verifiable arXiv ID -- an ingestion with no citations is fabrication."
- `flagged_for_v439`: principle "the strongest method(s) flagged so the .439 planner reads the mapping (discover->ingest->plan->experiment)."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts; 0.0001s floor."

The source set SHALL contain only verified arXiv IDs from the focused pass:
`2006.15055`, `2301.08243`, `2307.01668`, `2505.10819`, `2507.04920`,
`2510.04542`, `2602.02900`, and `2605.05138`. The method map SHALL contain
three to five methods, and every method claim SHALL cite only IDs from
`arxiv_ids_cited`. The note SHALL identify Slot Attention/object-centric slots,
PoE-World/product-of-experts factors, MC-ETM/diffusion-contrastive transition
energies, JEPA-class representation prediction, executable/code world-model
induction, and object-centric denoising diffusion priors as the ingested
literature informing the S1-S4 build.

#### SCENARIO-ARC-WMTE-4768

Given the research files are present, the reliable sweep helpers are available,
Semantic Scholar is attempted through the reliable script, and the top arXiv
sources are HTTP-200 verified
When Experiment 4768 runs
Then it writes
`results/experiment_4768_sota_ingestion_structural_energy.json`, records that
`/deep-research` was not invoked, maps three to five SOTA methods onto S1-S4
with real arXiv IDs, updates `research-studying.md` with the Exp 4768 mapping
note, and flags the strongest candidate inputs for `.439`.

#### SCENARIO-ARC-WMTE-4768-NO-FABRICATION

Given a candidate artifact omits citations, cites an unverified arXiv ID, uses
a stale `.438` roadmap flag, or claims a model-load, training, leaderboard, or
solve result
When the Experiment 4768 artifact validator runs
Then it rejects the artifact before the result can be written.

### REQ-ARC-WMTE-4765: .438 ARC Null Silent-Bug Audit

Experiment 4765 SHALL audit the `.438` ARC null and no-bank residual artifacts
before their negative results are trusted as mechanism evidence. The audit SHALL
read `results/experiment_4761_structural_energy_s0_core_bet_probe.json`,
`results/experiment_4762_levelup_attempt.json`,
`results/experiment_4764_heldout_first_win_readiness.json`, and the prior
silent-bug audits `results/experiment_4725_silent_bug_audit.json` and
`results/experiment_4755_silent_bug_audit.json`. Missing `.438` source
artifacts SHALL produce a blocked artifact rather than a fabricated audit.

The audit SHALL cross-check each source artifact's exercised-evidence fields:
for S0, the candidate row counts, near-miss negatives, in-sample positive
control, and origin-probe rows/AUROC; for the level-up no-bank residual, the
attempted-game ledger, reproduction gates, solution labels, and timed no-gate
residuals; and for the held-out first-win null, the held-out attempt count,
positive-control parity, flat `0.04` methodology note, and parity test. It
SHALL flag silent bugs for dead or identity engines that change zero cells,
byte-identical A/B arms, unannotated `0.04` first-win tautologies, dead archive
or `_frame_grid (1,64,64)` representation no-ops, and S0 leak signatures where
the origin probe was not run or `origin_probe_auroc >= 0.6`.

Experiment 4765 SHALL write
`results/experiment_4765_silent_bug_audit.json` and append a human-readable
section to `ops/arc_null_silent_bug_audit.md`. The artifact SHALL include
principle-annotated top-level fields for `honest_verdict`, `nulls_audited`,
`silent_bugs_found`, and `inference_substrate`, plus per-null verdicts,
preconditions, checksums for audited artifacts, duration, random seed, and a
stable reproducibility checksum. The audit SHALL make no solve, leaderboard,
training, or live-inference claim.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; audit complete is complete_/success_."
- `nulls_audited`: principle "count of nulls re-examined -- a null that tested dead code is not a trustworthy null."
- `silent_bugs_found`: principle "the load-bearing output -- which nulls must be reopened vs trusted."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts; 0.0001s floor."

#### SCENARIO-ARC-WMTE-4765-SILENT-BUG-AUDIT

Given the `.438` S0, level-up no-bank, and held-out first-win artifacts are
present, along with the prior silent-bug audit artifacts
When Experiment 4765 cross-checks exercised-evidence fields
Then it writes `results/experiment_4765_silent_bug_audit.json`, appends to
`ops/arc_null_silent_bug_audit.md`, classifies the S0 leak signature as a
silent bug when `origin_probe_auroc >= 0.6` or the origin probe was not run,
keeps genuinely exercised no-bank and flat first-win nulls trusted only when
their evidence fields are non-degenerate, and reports the complete audit with
`inference_substrate=aggregation_from_upstream_artifacts`.

#### SCENARIO-ARC-WMTE-4765-BLOCKED-PRECONDITION

Given any required `.438` or prior-audit source artifact is missing
When Experiment 4765 runs
Then it writes a blocked artifact with the missing source in
`preconditions_checked`, no silent-bug fabrication, no markdown append, and a
stable reproducibility checksum.

### REQ-ARC-WMTE-4751: SK48 Level-Up Self-Play Bank and Learned-Verifier Checkpoint

Experiment 4751 SHALL bank exactly one new offline-reproduced public ARC level
by deepening `sk48` from the registry's reproduced L1 row to L2 through the
standing ARC self-play loop. Before counting the result, it SHALL precheck
`ops/arc_solve_registry.yaml` and reject the run if `sk48` already records
`levels_reproduced >= 2`. It SHALL verify that the Qwen3.5-9B-MTP GGUF cache is
present and that `arc_solver_kit.offline_arcade()` initializes, but the
adaptered `sk48` bank SHALL keep `inference_substrate` honest as the offline
verifier-routed loop unless a live LLM proposer is genuinely invoked.

Experiment 4751 SHALL use the live-path standing mechanism, not a parallel
solver: `scripts/arc_loop_solve.py --game sk48 --target-level 2 --no-hazard-prune`.
A new level SHALL count only when `arc_solver_kit.reproduce` confirms the
reached level offline, the reached level is strictly greater than the
prechecked registry level, and the learned verifier checkpoint is recorded at
`models/arc_verifier_sk48.json`.

The terminal artifact `results/experiment_4751_levelup_selfplay.json` SHALL
include principle-annotated top-level fields for `honest_verdict`,
`inference_substrate`, `preconditions_checked`, `offline_reproduced`,
`reproduced_levels`, `solve_provenance`, and `verifier_is_oracle`, plus
`new_levels_banked`, `reproducible_total_levels`, `verifier_checkpoint`,
`registry_precheck_passed`, `random_seed`, `reproducibility_checksum`,
`model_specs`, and the standing loop reproduction gate. On success it SHALL
update the `sk48` registry row and advance `reproducible_total_levels` from 64
to 65.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success: <game>_L<n>_offline_reproduced when a level is banked, complete: <game>_delta_identified_no_bank otherwise."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates for this adaptered offline self-play bank; live_llm_inference; 60s floor applies only when the live proposer is actually invoked."
- `preconditions_checked`: principle "records GGUF/arcade checks."
- `offline_reproduced`: principle "true only if arc_solver_kit.reproduce re-derives the claimed level offline -- the ONLY counted-level signal (a live-recorded trajectory is provisional)."
- `reproduced_levels`: principle "the count of offline-reproduced levels this run -- feeds reproducible_total_levels growth."
- `solve_provenance`: principle "live_agent_self_discovery if the live agent advanced via its OWN attempts + runtime RE; development_proxy for the offline dev twin -- never outer_loop_re."
- `verifier_is_oracle`: principle "false -- a learned/energy verifier routes/prunes; if the verifier IS the executable oracle the win is execution_grounded, not a moat."
- `new_levels_banked`: principle "exactly +1 for the Level-Up Attempt Guarantee; zero is a complete no-bank attempt."
- `reproducible_total_levels`: principle "authoritative registry total after a reproduced bank; Experiment 4751 advances 64->65 only on sk48 L2."
- `verifier_checkpoint`: principle "models/arc_verifier_sk48.json records the trained learned verifier checkpoint from the standing loop."
- `registry_precheck_passed`: principle "confirms sk48 L2 was not already reachable in the authoritative registry before counting."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash of the reproduction-gated artifact fields."
- `model_specs`: principle "records the cached Qwen3.5-9B-MTP GGUF precondition and whether it was invoked."

#### SCENARIO-ARC-WMTE-4751-REGISTRY-PRECHECK

Given the ARC registry records `sk48` at L1 and
`reproducible_total_levels=64`
When Experiment 4751 selects its target
Then it selects `sk48` target L2, records that the target level is not already
banked by the registry, and rejects any duplicate registry that already records
`sk48` L2 or deeper.

#### SCENARIO-ARC-WMTE-4751-REPRODUCTION-GATED-BANK

Given the Qwen3.5-9B-MTP GGUF cache and offline arcade preconditions pass
When Experiment 4751 runs `scripts/arc_loop_solve.py --game sk48 --target-level 2 --no-hazard-prune`
Then `results/arc_loop_solve_sk48.json` reports `offline_reproduced=true`,
`reproduced_levels=2`, a reproduction gate with `reproduced=true`,
`solve_provenance=development_proxy`, and `verifier_is_oracle=false` in the
terminal artifact.

#### SCENARIO-ARC-WMTE-4751-REGISTRY-GATE

Given the standing loop reproduces `sk48` L2 and trains
`models/arc_verifier_sk48.json`
When Experiment 4751 writes `results/experiment_4751_levelup_selfplay.json`
Then it emits `success: sk48_L2_offline_reproduced`, records
`new_levels_banked=1`, updates the registry row to `levels_reproduced: 2`, and
sets `reproducible_total_levels=65`.

### REQ-ARC-WMTE-4762: ARC Rotation Level-Up Attempt Guarantee Ledger

Experiment 4762 SHALL run the standing ARC solve loop on the current rotation
target set. It SHALL precheck `arc_solver_kit.offline_arcade()`, read
`ops/arc_solve_registry.yaml`, prefer first-contact public games from `re86`,
`sb26`, `bp35`, and `lf52` when any of those games are not already reproduced,
and otherwise attempt a +1 deepen on the shallowest already-reproduced game
that has a standing-loop route. Every attempted game SHALL be compared against
the registry's prior `levels_reproduced`; a reproduced level counts only when
the offline reproduction gate reaches a strictly higher level than that prior
registry depth.

Experiment 4762 SHALL write
`results/experiment_4762_levelup_attempt.json` with principle-annotated
top-level fields for `honest_verdict`, `solve_provenance`,
`offline_reproduced`, `reproduced_levels`, `inference_substrate`,
`verifier_is_oracle`, `preconditions_checked`, `target_game`,
`attempted_games`, `dead_ends`, `registry_update`,
`reproducibility_checksum`, and `schema_errors`. If no attempted game banks a
new level, the artifact SHALL use a terminal `complete_<game>_no_new_level`
verdict, set `offline_reproduced=false`, set `reproduced_levels=0`, preserve
the registry total, and record the residual delta for each retired target
rather than fabricating a bank.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; a banked level is success_, a no-bank is complete_<game>_no_new_level_residual_<cause>."
- `solve_provenance`: principle "live_agent_self_discovery -- the agent advanced via its OWN attempts + runtime RE; NOT outer_loop_re and NOT a bare development_proxy adapter for a headline."
- `offline_reproduced`: principle "only reproduced levels count toward reproducible_total_levels -- a live-recorded trajectory alone is provisional."
- `reproduced_levels`: principle "the new reproducible depth; the monotonic ARC progress metric."
- `inference_substrate`: principle "live_llm_inference if induction runs; otherwise adapter_search_only_no_induction for standing-loop adapter/search-only runs."
- `verifier_is_oracle`: principle "the live solver's reproduction gate is execution-grounded (true); this is a SOLVE task, not a moat claim."
- `preconditions_checked`: principle "records arcade/env/generator checks so a missing-resource run emits blocked_, never a fabricated solve."

#### SCENARIO-ARC-WMTE-4762-ROTATION-PRECHECK

Given the ARC registry already records `re86`, `sb26`, `bp35`, and `lf52` at
L2
When Experiment 4762 selects rotation attempts
Then it treats those games as deepen targets instead of first-contact banks,
records their prior levels, and retires each same-depth verdict without
retrying the same game.

#### SCENARIO-ARC-WMTE-4762-REPRODUCTION-GATE

Given a standing-loop result reports `offline_reproduced=true` but reaches only
the registry's prior level
When Experiment 4762 builds its terminal artifact
Then `new_levels_banked=0`, `offline_reproduced=false`, `reproduced_levels=0`,
and the dead-end entry explains that the result reproduced an existing depth.

#### SCENARIO-ARC-WMTE-4762-STABLE-ARTIFACT

Given the rotation attempts have terminal loop artifacts or timed no-gate
residuals
When Experiment 4762 writes `results/experiment_4762_levelup_attempt.json`
Then the checksum is deterministic, `schema_errors=[]`, and the artifact
contains all required principle-annotated fields.

### REQ-ARC-WMTE-4772: ARC Rotation Level-Up Attempt Guarantee

Experiment 4772 SHALL run the standing ARC solve loop on exactly one rotation
target. It SHALL precheck `arc_solver_kit.offline_arcade()`, read
`ops/arc_solve_registry.yaml`, prefer an unreproduced public game from `re86`,
`sb26`, `bp35`, or `lf52`, and otherwise select a +1 deepen on the shallowest
already-reproduced game that has a standing-loop route. The selected loop result
SHALL be compared against the registry's prior `levels_reproduced`; a reproduced
level counts only when the offline reproduction gate reaches a strictly higher
level than that prior registry depth.

Experiment 4772 SHALL write
`results/experiment_4772_levelup_attempt.json` with principle-annotated
top-level fields for `honest_verdict`, `solve_provenance`,
`offline_reproduced`, `reproduced_levels`, `inference_substrate`,
`verifier_is_oracle`, `preconditions_checked`, `target_game`,
`rotation_selection`, `attempted_games`, `dead_ends`, `registry_update`,
`reproducibility_checksum`, and `schema_errors`. If the selected target does not
bank a new level, the artifact SHALL use a terminal
`complete_<game>_no_new_level_residual_<cause>` verdict, set
`offline_reproduced=false`, set `reproduced_levels=0`, preserve the registry
total, and record the residual so the next milestone rotates rather than
reattempting the same angle.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; a banked level is success_, a no-bank is complete_<game>_no_new_level_residual_<cause>."
- `solve_provenance`: principle "live_agent_self_discovery -- the agent advanced via its OWN attempts + runtime RE; NOT outer_loop_re (CRITICAL)."
- `offline_reproduced`: principle "only reproduced levels count -- a live-recorded trajectory alone is provisional."
- `reproduced_levels`: principle "the new reproducible depth; the monotonic ARC progress metric."
- `inference_substrate`: principle "live_llm_inference if induction runs; 60s floor."
- `verifier_is_oracle`: principle "the reproduction gate is execution-grounded (true); this is a SOLVE task, not a moat claim."
- `preconditions_checked`: principle "records arcade/env/generator checks so a missing-resource run emits blocked_, never a fabricated solve."

#### SCENARIO-ARC-WMTE-4772-ROTATION-TARGET

Given the ARC registry already records `re86`, `sb26`, `bp35`, and `lf52` as
reproduced
When Experiment 4772 selects a rotation target
Then it selects the shallowest already-reproduced standing-loop deepen target,
records the prior level and target level, and records which public games were
not first-contact targets.

#### SCENARIO-ARC-WMTE-4772-REPRODUCTION-GATE

Given the standing loop result reports `offline_reproduced=true` but reaches
only the registry's prior level
When Experiment 4772 builds its terminal artifact
Then `new_levels_banked=0`, `offline_reproduced=false`, `reproduced_levels=0`,
and the dead-end entry explains that the result reproduced an existing depth.

#### SCENARIO-ARC-WMTE-4772-STABLE-ARTIFACT

Given the selected rotation target has a terminal loop artifact
When Experiment 4772 writes `results/experiment_4772_levelup_attempt.json`
Then the checksum is deterministic, `schema_errors=[]`, and the artifact
contains all required principle-annotated fields.

### REQ-ARC-WMTE-4782: ARC Rotated Level-Up Attempt Guarantee

Experiment 4782 SHALL run the standing ARC solve loop on exactly one rotated
target. It SHALL precheck `arc_solver_kit.offline_arcade()`, read
`ops/arc_solve_registry.yaml`, skip the recently retired `ka59` residual and
the recently banked `re86` self-play target, prefer the public rotation set
`lf52`, `sb26`, and `bp35`, and otherwise select a +1 deepen on the shallowest
already-reproduced standing-loop target. If the registry already records every
public rotation target at a reproduced depth, Experiment 4782 SHALL treat the
selected public game as a deepen attempt and compare the terminal loop result
against the registry's prior `levels_reproduced`.

Experiment 4782 SHALL write
`results/experiment_4782_levelup_attempt.json` with principle-annotated
top-level fields for `honest_verdict`, `solve_provenance`,
`offline_reproduced`, `reproduced_levels`, `inference_substrate`,
`verifier_is_oracle`, `preconditions_checked`, `target_game`,
`rotation_selection`, `attempted_games`, `dead_ends`, `registry_update`,
`retire_if_same_verdict`, `reproducibility_checksum`, and `schema_errors`. A
level counts only when the offline reproduction gate reaches a strictly higher
level than the registry prior. If the selected target does not bank a new level,
the artifact SHALL use a terminal
`complete_<game>_no_new_level_residual_<cause>` verdict, set
`offline_reproduced=false`, set `reproduced_levels=0`, preserve the registry
total, set `retire_if_same_verdict=true`, and record the residual so the next
milestone rotates rather than reattempting the same verdict.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; a banked level is success_, a no-bank is complete_<game>_no_new_level_residual_<cause>."
- `solve_provenance`: principle "live_agent_self_discovery -- the agent advanced via its OWN attempts + runtime RE; NOT outer_loop_re (CRITICAL)."
- `offline_reproduced`: principle "only reproduced levels count -- a live-recorded trajectory alone is provisional."
- `reproduced_levels`: principle "the new reproducible depth; the monotonic ARC progress metric."
- `inference_substrate`: principle "live_llm_inference if induction runs (60s floor)."
- `verifier_is_oracle`: principle "the reproduction gate is execution-grounded (true); this is a SOLVE task, not a moat claim."
- `preconditions_checked`: principle "records arcade/env/generator checks so a missing-resource run emits blocked_, never a fabricated solve."

#### SCENARIO-ARC-WMTE-4782-ROTATION-TARGET

Given the ARC registry already records `lf52`, `sb26`, and `bp35` as reproduced
and the milestone retires `ka59` and `re86` from the target pool
When Experiment 4782 selects a rotation target
Then it selects a public rotation deepen target, records the registry prior and
target level, and records why the retired targets were excluded.

#### SCENARIO-ARC-WMTE-4782-REPRODUCTION-GATE

Given the standing loop result reports `offline_reproduced=true` but reaches
only the registry's prior level
When Experiment 4782 builds its terminal artifact
Then `new_levels_banked=0`, `offline_reproduced=false`, `reproduced_levels=0`,
and the dead-end entry explains that the result reproduced an existing depth.

#### SCENARIO-ARC-WMTE-4782-STABLE-ARTIFACT

Given the selected rotation target has a terminal loop artifact
When Experiment 4782 writes `results/experiment_4782_levelup_attempt.json`
Then the checksum is deterministic, `schema_errors=[]`, and the artifact
contains all required principle-annotated fields.

### REQ-ARC-WMTE-4792: ARC Rotated Level-Up Attempt Guarantee

Experiment 4792 SHALL run the standing ARC solve loop on the rotated target
set required by the milestone note. It SHALL precheck
`arc_solver_kit.offline_arcade()`, read `ops/arc_solve_registry.yaml`, retire
`ka59` and `lf52` as recent no-bank residuals, retire `re86` as recently covered
by self-play, prefer `sb26` followed by `bp35`, and otherwise select a +1
deepen on the shallowest already-reproduced standing-loop target. The selected
target SHALL be passed through `recommend_approach(game)` and the resulting
recommendation SHALL be recorded in the terminal artifact. If the first
preferred target does not bank a strictly new reproduced level, Experiment 4792
SHALL rotate to the next preferred target before falling back to the shallowest
already-solved standing-loop game.

Experiment 4792 SHALL write
`results/experiment_4792_levelup_attempt.json` with principle-annotated
top-level fields for `honest_verdict`, `solve_provenance`,
`offline_reproduced`, `reproduced_levels`, `inference_substrate`,
`verifier_is_oracle`, `preconditions_checked`, `target_game`,
`rotation_selection`, `approach_recommendation`, `attempted_games`,
`dead_ends`, `registry_update`, `retire_if_same_verdict`,
`reproducibility_checksum`, and `schema_errors`. A level counts only when the
offline reproduction gate reaches a strictly higher level than the registry
prior. If no selected target banks a new level, the artifact SHALL use a
terminal `complete_<game>_no_new_level_residual_<cause>` verdict, set
`offline_reproduced=false`, set `reproduced_levels=0`, preserve the registry
total, set `retire_if_same_verdict=true`, and record the residual so the next
milestone rotates rather than reattempting the same verdict. A missing target
offline environment SHALL produce `blocked_<game>_offline_env_missing` and SHALL
not fabricate reproduced levels.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; a banked level is success_, a no-bank is complete_<game>_no_new_level_residual_<cause>."
- `solve_provenance`: principle "live_agent_self_discovery -- the agent advanced via its OWN attempts; NOT outer_loop_re (CRITICAL)."
- `offline_reproduced`: principle "only reproduced levels count -- a live-recorded trajectory alone is provisional."
- `reproduced_levels`: principle "the new reproducible depth; the monotonic ARC progress metric."
- `inference_substrate`: principle "live_llm_inference if induction runs (60s floor)."
- `verifier_is_oracle`: principle "the reproduction gate is execution-grounded (true); this is a SOLVE task."
- `preconditions_checked`: principle "records arcade/env/generator checks so a missing-resource run emits blocked_, never a fabricated solve."

#### SCENARIO-ARC-WMTE-4792-ROTATION-TARGET

Given the ARC registry already records `sb26` and `bp35` as reproduced and the
milestone retires `ka59`, `lf52`, and `re86` from the target pool
When Experiment 4792 selects rotation targets
Then it selects `sb26` first, records `bp35` as the next rotate-on-no-bank
target, records each registry prior and target level, records why the retired
targets were excluded, and records the recommendation for the selected target.

#### SCENARIO-ARC-WMTE-4792-REPRODUCTION-GATE

Given the standing loop result reports `offline_reproduced=true` but reaches
only the registry's prior level
When Experiment 4792 builds its terminal artifact
Then `new_levels_banked=0`, `offline_reproduced=false`, `reproduced_levels=0`,
and the dead-end entry explains that the result reproduced an existing depth
before the workflow rotates to the next target.

#### SCENARIO-ARC-WMTE-4792-STABLE-ARTIFACT

Given the selected rotation target has a terminal loop artifact
When Experiment 4792 writes `results/experiment_4792_levelup_attempt.json`
Then the checksum is deterministic, `schema_errors=[]`, and the artifact
contains all required principle-annotated fields.

### REQ-ARC-WMTE-4802: ARC Rotated Level-Up Attempt Guarantee

Experiment 4802 SHALL run the standing ARC solve loop on the rotated target set
required by the 2026-06-26 conductor note. It SHALL precheck
`arc_solver_kit.offline_arcade()`, read `ops/arc_solve_registry.yaml`, retire
`ka59`, `lf52`, `re86`, and `sc25` as recently covered targets, prefer `bp35`
followed by `sb26`, and otherwise rotate to a +1 deepen attempt on the
shallowest already-reproduced game. The selected target SHALL be passed through
`recommend_approach(game)` and the recommendation SHALL be recorded in the
terminal artifact. If a target does not bank a strictly new reproduced level,
Experiment 4802 SHALL rotate to the next target and record the no-bank residual
instead of reusing the same verdict.

Experiment 4802 SHALL write
`results/experiment_4802_levelup_attempt.json` with principle-annotated
top-level fields for `honest_verdict`, `solve_provenance`,
`offline_reproduced`, `reproduced_levels`, `inference_substrate`,
`verifier_is_oracle`, `preconditions_checked`, `target_game`,
`rotation_selection`, `approach_recommendation`, `attempted_games`,
`dead_ends`, `registry_update`, `retire_if_same_verdict`,
`reproducibility_checksum`, and `schema_errors`. A level counts only when the
offline reproduction gate reaches a strictly higher level than the registry
prior. Timed adapter-free fallbacks SHALL be recorded as no-gate dead ends, not
as reproduced levels. A missing target offline environment SHALL produce
`blocked_<game>_offline_env_missing` and SHALL not fabricate reproduced levels.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; a banked level is success_, a no-bank is complete_<game>_no_new_level_residual_<cause>."
- `solve_provenance`: principle "live_agent_self_discovery -- the agent advanced via its OWN attempts; NOT outer_loop_re (CRITICAL)."
- `offline_reproduced`: principle "only reproduced levels count -- a live-recorded trajectory alone is provisional."
- `reproduced_levels`: principle "the new reproducible depth; the monotonic ARC progress metric."
- `inference_substrate`: principle "live_llm_inference if induction runs (60s floor)."
- `verifier_is_oracle`: principle "the reproduction gate is execution-grounded (true); this is a SOLVE task."
- `preconditions_checked`: principle "records arcade/env/generator checks so a missing-resource run emits blocked_, never a fabricated solve."

#### SCENARIO-ARC-WMTE-4802-ROTATION-TARGET

Given the ARC registry already records `bp35` and `sb26` as reproduced and the
milestone retires `ka59`, `lf52`, `re86`, and `sc25` from the target pool
When Experiment 4802 selects rotation targets
Then it selects `bp35` first, records `sb26` as the next rotate-on-no-bank
target, records each registry prior and target level, records why the retired
targets were excluded, and records the recommendation for the selected target.

#### SCENARIO-ARC-WMTE-4802-REPRODUCTION-GATE

Given the standing loop result reports `offline_reproduced=true` but reaches
only the registry's prior level
When Experiment 4802 builds its terminal artifact
Then `new_levels_banked=0`, `offline_reproduced=false`, `reproduced_levels=0`,
and the dead-end entry explains that the result reproduced an existing depth
before the workflow rotates to the next target.

#### SCENARIO-ARC-WMTE-4802-ADAPTER-FREE-SEEDING

Given an adapter-free first-contact trajectory reaches a level and needs to
seed a learned verifier checkpoint
When `scripts/arc_loop_solve.py` executes `solve_via_explore`
Then the learned verifier dependency is imported before use and the script does
not fail with an unbound `LearnedVerifier` name.

#### SCENARIO-ARC-WMTE-4802-STABLE-ARTIFACT

Given the selected rotation target has a terminal loop artifact
When Experiment 4802 writes `results/experiment_4802_levelup_attempt.json`
Then the checksum is deterministic, `schema_errors=[]`, and the artifact
contains all required principle-annotated fields.

### REQ-ARC-WMTE-4812: ARC Rotated Level-Up Attempt Guarantee

Experiment 4812 SHALL run the standing ARC solve loop on the target selected by
the 2026-06-26 conductor rotation note. It SHALL precheck
`arc_solver_kit.offline_arcade()`, read `ops/arc_solve_registry.yaml`, prefer
`bp35` followed by `sb26` only when either is still a first-contact target, and
otherwise select a +1 deepen attempt on the shallowest already-reproduced
standing-loop game. The selected target SHALL be passed through
`recommend_approach(game)` and the recommendation SHALL be recorded in the
terminal artifact. A target result SHALL bank progress only when the offline
reproduction gate reaches a strictly higher level than the registry prior; a
same-depth result SHALL be retired and the next candidate SHALL be attempted or
recorded as the rotate-on-no-bank residual.

Experiment 4812 SHALL write
`results/experiment_4812_levelup_attempt.json` with principle-annotated
top-level fields for `honest_verdict`, `solve_provenance`,
`offline_reproduced`, `reproduced_levels`, `inference_substrate`,
`verifier_is_oracle`, `preconditions_checked`, `target_game`,
`rotation_selection`, `approach_recommendation`, `attempted_games`,
`dead_ends`, `registry_update`, `retire_if_same_verdict`,
`reproducibility_checksum`, and `schema_errors`. A missing target offline
environment SHALL produce `blocked_<game>_offline_env_missing` and SHALL not
fabricate reproduced levels.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; a banked level is success_, a no-bank is complete_<game>_no_new_level_residual_<cause>."
- `solve_provenance`: principle "live_agent_self_discovery -- the agent advanced via its OWN attempts; NOT outer_loop_re (CRITICAL)."
- `offline_reproduced`: principle "only reproduced levels count."
- `reproduced_levels`: principle "the new reproducible depth; the monotonic ARC progress metric."
- `inference_substrate`: principle "live_llm_inference if induction runs (60s floor)."
- `verifier_is_oracle`: principle "the reproduction gate is execution-grounded (true); this is a SOLVE task."
- `preconditions_checked`: principle "records arcade/env/generator checks so a missing-resource run emits blocked_, never a fabricated solve."

#### SCENARIO-ARC-WMTE-4812-ROTATION-TARGET

Given the ARC registry records `bp35` and `sb26` as already reproduced and
contains shallower reproduced standing-loop games
When Experiment 4812 selects its rotation target
Then it selects the shallowest already-reproduced standing-loop game for a +1
deepen, records `bp35` and `sb26` as non-first-contact public candidates, and
records the recommendation for the selected target.

#### SCENARIO-ARC-WMTE-4812-REPRODUCTION-GATE

Given the standing loop result reports `offline_reproduced=true` but reaches
only the registry's prior level
When Experiment 4812 builds its terminal artifact
Then `new_levels_banked=0`, `offline_reproduced=false`, `reproduced_levels=0`,
and the dead-end entry explains that the result reproduced an existing depth
before the workflow rotates or retires the same verdict.

#### SCENARIO-ARC-WMTE-4812-STABLE-ARTIFACT

Given the selected rotation target has a terminal loop artifact
When Experiment 4812 writes `results/experiment_4812_levelup_attempt.json`
Then the checksum is deterministic, `schema_errors=[]`, and the artifact
contains all required principle-annotated fields.

### REQ-ARC-WMTE-4822: ARC Rotated Level-Up Attempt Guarantee

Experiment 4822 SHALL run the standing ARC solve loop on the target selected by
the 2026-06-26 conductor rotation note. It SHALL precheck
`arc_solver_kit.offline_arcade()`, read `ops/arc_solve_registry.yaml`, prefer
`bp35`, `sb26`, then `lf52` when any of those games is still a first-contact
target, and otherwise select a +1 deepen attempt on the shallowest
already-reproduced standing-loop game. The selected target SHALL be passed
through `recommend_approach(game)` and the recommendation SHALL be recorded in
the terminal artifact. A target result SHALL bank progress only when the offline
reproduction gate reaches a strictly higher level than the registry prior; a
same-depth result SHALL be retired and the next candidate SHALL be attempted or
recorded as the rotate-on-no-bank residual.

Experiment 4822 SHALL write
`results/experiment_4822_levelup_attempt.json` with principle-annotated
top-level fields for `honest_verdict`, `solve_provenance`,
`offline_reproduced`, `reproduced_levels`, `inference_substrate`,
`verifier_is_oracle`, `preconditions_checked`, `target_game`,
`rotation_selection`, `approach_recommendation`, `attempted_games`,
`dead_ends`, `registry_update`, `retire_if_same_verdict`,
`reproducibility_checksum`, and `schema_errors`. A missing target offline
environment SHALL produce `blocked_<game>_offline_env_missing` and SHALL not
fabricate reproduced levels.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; a banked level is success_, a no-bank is complete_<game>_no_new_level_residual_<cause>."
- `solve_provenance`: principle "live_agent_self_discovery -- the agent advanced via its OWN attempts; NOT outer_loop_re (CRITICAL)."
- `offline_reproduced`: principle "only reproduced levels count."
- `reproduced_levels`: principle "the new reproducible depth; the monotonic ARC progress metric."
- `inference_substrate`: principle "live_llm_inference if induction runs (60s floor)."
- `verifier_is_oracle`: principle "the reproduction gate is execution-grounded (true); this is a SOLVE task."
- `preconditions_checked`: principle "records arcade/env/generator checks so a missing-resource run emits blocked_, never a fabricated solve."

#### SCENARIO-ARC-WMTE-4822-ROTATION-TARGET

Given the ARC registry records `bp35`, `sb26`, and `lf52` as already reproduced
and contains shallower reproduced standing-loop games
When Experiment 4822 selects its rotation target
Then it selects the shallowest already-reproduced standing-loop game for a +1
deepen, records `bp35`, `sb26`, and `lf52` as non-first-contact public
candidates, and records the recommendation for the selected target.

#### SCENARIO-ARC-WMTE-4822-REPRODUCTION-GATE

Given the standing loop result reports `offline_reproduced=true` but reaches
only the registry's prior level
When Experiment 4822 builds its terminal artifact
Then `new_levels_banked=0`, `offline_reproduced=false`, `reproduced_levels=0`,
and the dead-end entry explains that the result reproduced an existing depth
before the workflow rotates or retires the same verdict.

#### SCENARIO-ARC-WMTE-4822-STABLE-ARTIFACT

Given the selected rotation target has a terminal loop artifact
When Experiment 4822 writes `results/experiment_4822_levelup_attempt.json`
Then the checksum is deterministic, `schema_errors=[]`, and the artifact
contains all required principle-annotated fields.

### REQ-ARC-WMTE-4842: ARC Rotated Level-Up Attempt Guarantee

Experiment 4842 SHALL run the standing ARC solve loop on the target selected by
the 2026-06-27 operator-c scored-lever rotation note. It SHALL precheck
`arc_solver_kit.offline_arcade()`, read `ops/arc_solve_registry.yaml`, prefer
`sb26`, then `lf52`, then `bp35` when any of those games is still a
first-contact target, and otherwise select a +1 deepen attempt on the
shallowest already-reproduced standing-loop game. The selected target SHALL be
passed through `recommend_approach(game)` and the recommendation SHALL be
recorded in the terminal artifact. A target result SHALL bank progress only
when the offline reproduction gate reaches a strictly higher level than the
registry prior; a same-depth result SHALL be retired and the next candidate
SHALL be attempted or recorded as the rotate-on-no-bank residual.

Experiment 4842 SHALL write
`results/experiment_4842_levelup_attempt.json` with principle-annotated
top-level fields for `honest_verdict`, `solve_provenance`,
`offline_reproduced`, `reproduced_levels`, `inference_substrate`,
`verifier_is_oracle`, `preconditions_checked`, `target_game`,
`rotation_selection`, `approach_recommendation`, `attempted_games`,
`dead_ends`, `registry_update`, `retire_if_same_verdict`,
`reproducibility_checksum`, and `schema_errors`. A missing target offline
environment SHALL produce `blocked_<game>_offline_env_missing` and SHALL not
fabricate reproduced levels.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; banked is success_, no-bank is complete_<game>_no_new_level_residual_<cause>."
- `solve_provenance`: principle "live_agent_self_discovery; NOT outer_loop_re (CRITICAL)."
- `offline_reproduced`: principle "only reproduced levels count."
- `reproduced_levels`: principle "the new reproducible depth; the monotonic ARC metric."
- `inference_substrate`: principle "live_llm_inference if induction runs (60s floor)."
- `verifier_is_oracle`: principle "the reproduction gate is execution-grounded (true); this is a SOLVE task."
- `preconditions_checked`: principle "records arcade/env/generator checks; a missing resource emits blocked_, never a fabricated solve."

#### SCENARIO-ARC-WMTE-4842-ROTATION-TARGET

Given the ARC registry records `sb26`, `lf52`, and `bp35` as already reproduced
and contains shallower reproduced standing-loop games
When Experiment 4842 selects its rotation target
Then it selects the shallowest already-reproduced standing-loop game for a +1
deepen, records `sb26`, `lf52`, and `bp35` as non-first-contact public
candidates in that order, and records the recommendation for the selected
target.

#### SCENARIO-ARC-WMTE-4842-REPRODUCTION-GATE

Given the standing loop result reports `offline_reproduced=true` but reaches
only the registry's prior level
When Experiment 4842 builds its terminal artifact
Then `new_levels_banked=0`, `offline_reproduced=false`, `reproduced_levels=0`,
and the dead-end entry explains that the result reproduced an existing depth
before the workflow rotates or retires the same verdict.

#### SCENARIO-ARC-WMTE-4842-STABLE-ARTIFACT

Given the selected rotation target has a terminal loop artifact
When Experiment 4842 writes `results/experiment_4842_levelup_attempt.json`
Then the checksum is deterministic, `schema_errors=[]`, and the artifact
contains all required principle-annotated fields.

### REQ-ARC-WMTE-4832: ARC Rotated Level-Up Attempt Guarantee

Experiment 4832 SHALL run the standing ARC solve loop on the target selected by
the 2026-06-26 conductor rotation note. It SHALL precheck
`arc_solver_kit.offline_arcade()`, read `ops/arc_solve_registry.yaml`, prefer
`bp35`, `sb26`, then `lf52` when any of those games is still a first-contact
target, and otherwise select a +1 deepen attempt on the shallowest
already-reproduced standing-loop game. The selected target SHALL be passed
through `recommend_approach(game)` and the recommendation SHALL be recorded in
the terminal artifact. A target result SHALL bank progress only when the offline
reproduction gate reaches a strictly higher level than the registry prior; a
same-depth result SHALL be retired and the next candidate SHALL be attempted or
recorded as the rotate-on-no-bank residual.

Experiment 4832 SHALL write
`results/experiment_4832_levelup_attempt.json` with principle-annotated
top-level fields for `honest_verdict`, `solve_provenance`,
`offline_reproduced`, `reproduced_levels`, `inference_substrate`,
`verifier_is_oracle`, `preconditions_checked`, `target_game`,
`rotation_selection`, `approach_recommendation`, `attempted_games`,
`dead_ends`, `registry_update`, `retire_if_same_verdict`,
`reproducibility_checksum`, and `schema_errors`. A missing target offline
environment SHALL produce `blocked_<game>_offline_env_missing` and SHALL not
fabricate reproduced levels.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; banked is success_, no-bank is complete_<game>_no_new_level_residual_<cause>."
- `solve_provenance`: principle "live_agent_self_discovery; NOT outer_loop_re (CRITICAL)."
- `offline_reproduced`: principle "only reproduced levels count."
- `reproduced_levels`: principle "the new reproducible depth; the monotonic ARC metric."
- `inference_substrate`: principle "live_llm_inference if induction runs (60s floor)."
- `verifier_is_oracle`: principle "the reproduction gate is execution-grounded (true); this is a SOLVE task."
- `preconditions_checked`: principle "records arcade/env/generator checks; a missing resource emits blocked_, never a fabricated solve."

#### SCENARIO-ARC-WMTE-4832-ROTATION-TARGET

Given the ARC registry records `bp35`, `sb26`, and `lf52` as already reproduced
and contains shallower reproduced standing-loop games
When Experiment 4832 selects its rotation target
Then it selects the shallowest already-reproduced standing-loop game for a +1
deepen, records `bp35`, `sb26`, and `lf52` as non-first-contact public
candidates, and records the recommendation for the selected target.

#### SCENARIO-ARC-WMTE-4832-REPRODUCTION-GATE

Given the standing loop result reports `offline_reproduced=true` but reaches
only the registry's prior level
When Experiment 4832 builds its terminal artifact
Then `new_levels_banked=0`, `offline_reproduced=false`, `reproduced_levels=0`,
and the dead-end entry explains that the result reproduced an existing depth
before the workflow rotates or retires the same verdict.

#### SCENARIO-ARC-WMTE-4832-STABLE-ARTIFACT

Given the selected rotation target has a terminal loop artifact
When Experiment 4832 writes `results/experiment_4832_levelup_attempt.json`
Then the checksum is deterministic, `schema_errors=[]`, and the artifact
contains all required principle-annotated fields.

### REQ-ARC-WMTE-4763: ARC Self-Play Verifier Checkpoint Refresh

Experiment 4763 SHALL run the standing ARC self-play loop on a banked game and
write `results/experiment_4763_self_play_verifier_checkpoint.json`. The
workflow SHALL precheck that `arc_solver_kit.offline_arcade()` initializes,
that `ops/arc_solve_registry.yaml` is loadable, and that the selected target
game already has `levels_reproduced >= 1` in the registry. A missing arcade,
registry, target, or banked level SHALL produce a terminal `blocked_` artifact
that records the failed precondition and SHALL NOT claim a refreshed
checkpoint.

The self-play run SHALL execute `scripts/arc_loop_solve.py` for the selected
banked target, warm-start from `models/arc_verifier_<game>.json` when present,
run the verifier-routed search and reproduction gate, and train plus persist
the learned verifier when trajectory labels are available. The terminal
artifact SHALL count success only when the learned verifier checkpoint's mtime
advances past the recorded pre-run mtime and the loop reproduction gate passes
on at least one level. A failed self-play run SHALL preserve the residual cause,
state count, gate output, and checkpoint metadata rather than fabricating a
green checkpoint.

The artifact SHALL include principle-annotated top-level fields for
`honest_verdict`, `verifier_checkpoint_refreshed`, `inference_substrate`,
`solve_provenance`, and `preconditions_checked`, plus `target_game`,
`checkpoint_path`, `checkpoint_mtime_before_ns`, `checkpoint_mtime_after_ns`,
`checkpoint_mtime_delta_ns`, `states_expanded`, `offline_reproduced`, `reproduced_levels`,
`reproduction_gate`, `self_play_residual`, `loop_result_path`, `random_seed`,
`reproducibility_checksum`, and `schema_errors`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; checkpoint refreshed + gate green is success_."
- `verifier_checkpoint_refreshed`: principle "the self-improvement signal -- the learned verifier improved from its own play."
- `inference_substrate`: principle "live_llm_inference; 60s floor."
- `solve_provenance`: principle "live_agent_self_discovery -- self-play is the agent improving on its own attempts."
- `preconditions_checked`: principle "records arcade/registry checks; a missing target emits blocked_, never a fabricated checkpoint."

#### SCENARIO-ARC-WMTE-4763-CHECKPOINT-REFRESHED

Given a selected banked target has a recorded pre-run verifier checkpoint mtime
When the standing self-play loop reports `offline_reproduced=true`,
`reproduced_levels>=1`, a positive `states_expanded` count, and a learned
verifier checkpoint whose mtime advanced
Then Experiment 4763 writes a `success_` artifact with
`verifier_checkpoint_refreshed=true`, records the search-state count, preserves
the reproduction gate, and emits `schema_errors=[]`.

#### SCENARIO-ARC-WMTE-4763-RESIDUAL-NO-FABRICATION

Given the loop result is missing, the reproduction gate fails, the reproduced
level is below one, or the verifier checkpoint mtime does not advance
When Experiment 4763 builds its terminal artifact
Then it writes a `complete_` residual verdict, keeps
`verifier_checkpoint_refreshed=false`, records the failing residual cause, and
does not invent a checkpoint path or refreshed mtime.

#### SCENARIO-ARC-WMTE-4763-BLOCKED-PRECONDITION

Given the offline arcade, registry, target selection, or banked-level
precondition fails
When Experiment 4763 runs
Then it writes a `blocked_` artifact with the failed precondition recorded,
`verifier_checkpoint_refreshed=false`, and a deterministic reproducibility
checksum.

### REQ-ARC-WMTE-4773: ARC Self-Play Verifier Checkpoint Refresh

Experiment 4773 SHALL run the standing ARC self-play loop on a banked game and
write `results/experiment_4773_self_play_verifier_checkpoint.json`. The
workflow SHALL precheck that `arc_solver_kit.offline_arcade()` initializes,
that `ops/arc_solve_registry.yaml` is loadable, and that the selected target
game already has `levels_reproduced >= 1` in the registry. A missing arcade,
registry, target, or banked level SHALL produce a terminal `blocked_` artifact
that records the failed precondition and SHALL NOT claim a refreshed
checkpoint.

The self-play run SHALL execute `scripts/arc_loop_solve.py` for the selected
banked target, warm-start from `models/arc_verifier_<game>.json` when present,
run the verifier-routed search and reproduction gate, and train plus persist
the learned verifier when trajectory labels are available. The terminal
artifact SHALL count success only when the learned verifier checkpoint's mtime
advances past the recorded pre-run mtime and the loop reproduction gate passes
on at least one level. A failed self-play run SHALL preserve the residual cause,
state count, gate output, and checkpoint metadata rather than fabricating a
green checkpoint.

The artifact SHALL include principle-annotated top-level fields for
`honest_verdict`, `verifier_checkpoint_refreshed`, `inference_substrate`,
`solve_provenance`, and `preconditions_checked`, plus `target_game`,
`checkpoint_path`, `checkpoint_mtime_before_ns`, `checkpoint_mtime_after_ns`,
`states_expanded`, `offline_reproduced`, `reproduced_levels`,
`reproduction_gate`, `self_play_residual`, `loop_result_path`, `random_seed`,
`reproducibility_checksum`, and `schema_errors`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; checkpoint refreshed + gate green is success_."
- `verifier_checkpoint_refreshed`: principle "the self-improvement signal -- the learned verifier improved from its own play."
- `inference_substrate`: principle "live_llm_inference; 60s floor."
- `solve_provenance`: principle "live_agent_self_discovery -- self-play is the agent improving on its own attempts."
- `preconditions_checked`: principle "records arcade/registry checks; a missing target emits blocked_, never a fabricated checkpoint."

#### SCENARIO-ARC-WMTE-4773-CHECKPOINT-REFRESHED

Given a selected banked target has a recorded pre-run verifier checkpoint mtime
When the standing self-play loop reports `offline_reproduced=true`,
`reproduced_levels>=1`, a positive `states_expanded` count, and a learned
verifier checkpoint whose mtime advanced
Then Experiment 4773 writes a `success_` artifact with
`verifier_checkpoint_refreshed=true`, records the search-state count, preserves
the reproduction gate, and emits `schema_errors=[]`.

#### SCENARIO-ARC-WMTE-4773-RESIDUAL-NO-FABRICATION

Given the loop result is missing, the reproduction gate fails, the reproduced
level is below one, or the verifier checkpoint mtime does not advance
When Experiment 4773 builds its terminal artifact
Then it writes a `complete_` residual verdict, keeps
`verifier_checkpoint_refreshed=false`, records the failing residual cause, and
does not invent a checkpoint path or refreshed mtime.

#### SCENARIO-ARC-WMTE-4773-BLOCKED-PRECONDITION

Given the offline arcade, registry, target selection, or banked-level
precondition fails
When Experiment 4773 runs
Then it writes a `blocked_` artifact with the failed precondition recorded,
`verifier_checkpoint_refreshed=false`, and a deterministic reproducibility
checksum.

### REQ-ARC-WMTE-4783: ARC Self-Play Verifier Checkpoint Refresh

Experiment 4783 SHALL run the standing ARC self-play loop on a banked game and
write `results/experiment_4783_self_play_verifier_checkpoint.json`. The
workflow SHALL precheck that `arc_solver_kit.offline_arcade()` initializes,
that `ops/arc_solve_registry.yaml` is loadable, and that the selected target
game already has `levels_reproduced >= 1` in the registry. A missing arcade,
registry, target, or banked level SHALL produce a terminal `blocked_` artifact
that records the failed precondition and SHALL NOT claim a refreshed
checkpoint.

The self-play run SHALL execute `scripts/arc_loop_solve.py` for the selected
banked target, warm-start from `models/arc_verifier_<game>.json` when present,
run the verifier-routed search and reproduction gate, and train plus persist
the learned verifier when trajectory labels are available. The terminal
artifact SHALL count success only when the learned verifier checkpoint's mtime
advances past the recorded pre-run mtime and the loop reproduction gate passes
on at least one level. A failed self-play run SHALL preserve the residual cause,
state count, gate output, and checkpoint metadata rather than fabricating a
green checkpoint.

The artifact SHALL include principle-annotated top-level fields for
`honest_verdict`, `verifier_checkpoint_refreshed`, `inference_substrate`,
`solve_provenance`, and `preconditions_checked`, plus `target_game`,
`checkpoint_path`, `checkpoint_mtime_before_ns`, `checkpoint_mtime_after_ns`,
`states_expanded`, `offline_reproduced`, `reproduced_levels`,
`reproduction_gate`, `self_play_residual`, `loop_result_path`, `random_seed`,
`reproducibility_checksum`, and `schema_errors`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; checkpoint refreshed + gate green is success_."
- `verifier_checkpoint_refreshed`: principle "the self-improvement signal -- the learned verifier improved from its own play."
- `inference_substrate`: principle "live_llm_inference; 60s floor."
- `solve_provenance`: principle "live_agent_self_discovery -- self-play is the agent improving on its own attempts."
- `preconditions_checked`: principle "records arcade/registry checks; a missing target emits blocked_, never a fabricated checkpoint."

#### SCENARIO-ARC-WMTE-4783-CHECKPOINT-REFRESHED

Given a selected banked target has a recorded pre-run verifier checkpoint mtime
When the standing self-play loop reports `offline_reproduced=true`,
`reproduced_levels>=1`, a positive `states_expanded` count, and a learned
verifier checkpoint whose mtime advanced
Then Experiment 4783 writes a `success_` artifact with
`verifier_checkpoint_refreshed=true`, records the search-state count, preserves
the reproduction gate, and emits `schema_errors=[]`.

#### SCENARIO-ARC-WMTE-4783-RESIDUAL-NO-FABRICATION

Given the loop result is missing, the reproduction gate fails, the reproduced
level is below one, or the verifier checkpoint mtime does not advance
When Experiment 4783 builds its terminal artifact
Then it writes a `complete_` residual verdict, keeps
`verifier_checkpoint_refreshed=false`, records the failing residual cause, and
does not invent a checkpoint path or refreshed mtime.

#### SCENARIO-ARC-WMTE-4783-BLOCKED-PRECONDITION

Given the offline arcade, registry, target selection, or banked-level
precondition fails
When Experiment 4783 runs
Then it writes a `blocked_` artifact with the failed precondition recorded,
`verifier_checkpoint_refreshed=false`, and a deterministic reproducibility
checksum.

### REQ-ARC-WMTE-4793: ARC Self-Play Verifier Checkpoint Refresh

Experiment 4793 SHALL run the standing ARC self-play loop on a banked game and
write `results/experiment_4793_self_play_verifier_checkpoint.json`. The
workflow SHALL precheck that `arc_solver_kit.offline_arcade()` initializes,
that `ops/arc_solve_registry.yaml` is loadable, and that the selected target
game already has `levels_reproduced >= 1` in the registry. A missing arcade,
registry, target, or banked level SHALL produce a terminal `blocked_` artifact
that records the failed precondition and SHALL NOT claim a refreshed
checkpoint.

The self-play run SHALL execute `scripts/arc_loop_solve.py` for the selected
banked target, warm-start from `models/arc_verifier_<game>.json` when present,
run the verifier-routed search and reproduction gate, and train plus persist
the learned verifier when trajectory labels are available. The terminal
artifact SHALL count success only when the learned verifier checkpoint's mtime
advances past the recorded pre-run mtime and the loop reproduction gate passes
on at least one level. A failed self-play run SHALL preserve the residual cause,
search-state count, gate output, and checkpoint metadata rather than
fabricating a green checkpoint.

The artifact SHALL include principle-annotated top-level fields for
`honest_verdict`, `verifier_checkpoint_refreshed`, `inference_substrate`,
`solve_provenance`, and `preconditions_checked`, plus `target_game`,
`checkpoint_path`, `checkpoint_mtime_before_ns`, `checkpoint_mtime_after_ns`,
`checkpoint_mtime_delta_ns`, `states_expanded`, `search_state_count`,
`offline_reproduced`, `reproduced_levels`, `reproduction_gate`,
`self_play_residual`, `loop_result_path`, `random_seed`,
`reproducibility_checksum`, and `schema_errors`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; checkpoint refreshed + gate green is success_."
- `verifier_checkpoint_refreshed`: principle "the self-improvement signal."
- `inference_substrate`: principle "live_llm_inference (60s floor)."
- `solve_provenance`: principle "live_agent_self_discovery -- self-play is the agent improving on its own attempts."
- `preconditions_checked`: principle "records arcade/registry checks; a missing target emits blocked_, never a fabricated checkpoint."

#### SCENARIO-ARC-WMTE-4793-CHECKPOINT-REFRESHED

Given a selected banked target has a recorded pre-run verifier checkpoint mtime
When the standing self-play loop reports `offline_reproduced=true`,
`reproduced_levels>=1`, a positive search-state count, and a learned verifier
checkpoint whose mtime advanced
Then Experiment 4793 writes a `success_` artifact with
`verifier_checkpoint_refreshed=true`, records both `states_expanded` and
`search_state_count`, preserves the reproduction gate, and emits
`schema_errors=[]`.

#### SCENARIO-ARC-WMTE-4793-RESIDUAL-NO-FABRICATION

Given the loop result is missing, the reproduction gate fails, the reproduced
level is below one, or the verifier checkpoint mtime does not advance
When Experiment 4793 builds its terminal artifact
Then it writes a `complete_` residual verdict, keeps
`verifier_checkpoint_refreshed=false`, records the failing residual cause, and
does not invent a checkpoint path or refreshed mtime.

#### SCENARIO-ARC-WMTE-4793-BLOCKED-PRECONDITION

Given the offline arcade, registry, target selection, or banked-level
precondition fails
When Experiment 4793 runs
Then it writes a `blocked_` artifact with the failed precondition recorded,
`verifier_checkpoint_refreshed=false`, and a deterministic reproducibility
checksum.

### REQ-ARC-WMTE-4803: ARC Self-Play Verifier Checkpoint Refresh

Experiment 4803 SHALL run the standing ARC self-play loop on a banked game and
write `results/experiment_4803_self_play_verifier_checkpoint.json`. The
workflow SHALL precheck that `arc_solver_kit.offline_arcade()` initializes,
that `ops/arc_solve_registry.yaml` is loadable, and that the selected target
game already has `levels_reproduced >= 1` in the registry. A missing arcade,
registry, target, or banked level SHALL produce a terminal `blocked_` artifact
that records the failed precondition and SHALL NOT claim a refreshed
checkpoint.

The self-play run SHALL execute `scripts/arc_loop_solve.py` for the selected
banked target, warm-start from `models/arc_verifier_<game>.json` when present,
run the verifier-routed search and reproduction gate, and train plus persist
the learned verifier when trajectory labels are available. The terminal
artifact SHALL count success only when the learned verifier checkpoint's mtime
advances past the recorded pre-run mtime and the loop reproduction gate passes
on at least one level. A failed self-play run SHALL preserve the residual cause,
search-state count, gate output, and checkpoint metadata rather than
fabricating a green checkpoint.

The artifact SHALL include principle-annotated top-level fields for
`honest_verdict`, `verifier_checkpoint_refreshed`, `inference_substrate`,
`solve_provenance`, and `preconditions_checked`, plus `target_game`,
`checkpoint_path`, `checkpoint_mtime_before_ns`, `checkpoint_mtime_after_ns`,
`checkpoint_mtime_delta_ns`, `states_expanded`, `search_state_count`,
`offline_reproduced`, `reproduced_levels`, `reproduction_gate`,
`self_play_residual`, `loop_result_path`, `random_seed`,
`reproducibility_checksum`, and `schema_errors`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; checkpoint refreshed + gate green is success_."
- `verifier_checkpoint_refreshed`: principle "the self-improvement signal."
- `inference_substrate`: principle "live_llm_inference (60s floor)."
- `solve_provenance`: principle "live_agent_self_discovery -- self-play is the agent improving on its own attempts."
- `preconditions_checked`: principle "records arcade/registry checks; a missing target emits blocked_, never a fabricated checkpoint."

#### SCENARIO-ARC-WMTE-4803-CHECKPOINT-REFRESHED

Given a selected banked target has a recorded pre-run verifier checkpoint mtime
When the standing self-play loop reports `offline_reproduced=true`,
`reproduced_levels>=1`, a positive search-state count, and a learned verifier
checkpoint whose mtime advanced
Then Experiment 4803 writes a `success_` artifact with
`verifier_checkpoint_refreshed=true`, records both `states_expanded` and
`search_state_count`, preserves the reproduction gate, and emits
`schema_errors=[]`.

#### SCENARIO-ARC-WMTE-4803-RESIDUAL-NO-FABRICATION

Given the loop result is missing, the reproduction gate fails, the reproduced
level is below one, or the verifier checkpoint mtime does not advance
When Experiment 4803 builds its terminal artifact
Then it writes a `complete_` residual verdict, keeps
`verifier_checkpoint_refreshed=false`, records the failing residual cause, and
does not invent a checkpoint path or refreshed mtime.

#### SCENARIO-ARC-WMTE-4803-BLOCKED-PRECONDITION

Given the offline arcade, registry, target selection, or banked-level
precondition fails
When Experiment 4803 runs
Then it writes a `blocked_` artifact with the failed precondition recorded,
`verifier_checkpoint_refreshed=false`, and a deterministic reproducibility
checksum.

### REQ-ARC-WMTE-4813: ARC Self-Play Verifier Checkpoint Refresh

Experiment 4813 SHALL run the standing ARC self-play loop on a banked game and
write `results/experiment_4813_self_play_verifier_checkpoint.json`. The
workflow SHALL precheck that `arc_solver_kit.offline_arcade()` initializes,
that `ops/arc_solve_registry.yaml` is loadable, and that the selected target
game already has `levels_reproduced >= 1` in the registry. A missing arcade,
registry, target, or banked level SHALL produce a terminal `blocked_` artifact
that records the failed precondition and SHALL NOT claim a refreshed
checkpoint.

The self-play run SHALL execute `scripts/arc_loop_solve.py` for the selected
banked target, warm-start from `models/arc_verifier_<game>.json` when present,
run the verifier-routed search and reproduction gate, and train plus persist
the learned verifier when trajectory labels are available. The terminal
artifact SHALL count success only when the learned verifier checkpoint's mtime
advances past the recorded pre-run mtime and the loop reproduction gate passes
on at least one level. A failed self-play run SHALL preserve the residual cause,
search-state count, gate output, and checkpoint metadata rather than
fabricating a green checkpoint.

The artifact SHALL include principle-annotated top-level fields for
`honest_verdict`, `verifier_checkpoint_refreshed`, `inference_substrate`,
`solve_provenance`, and `preconditions_checked`, plus `target_game`,
`checkpoint_path`, `checkpoint_mtime_before_ns`, `checkpoint_mtime_after_ns`,
`checkpoint_mtime_delta_ns`, `states_expanded`, `search_state_count`,
`offline_reproduced`, `reproduced_levels`, `reproduction_gate`,
`self_play_residual`, `loop_result_path`, `random_seed`,
`reproducibility_checksum`, and `schema_errors`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; checkpoint refreshed + gate green is success_."
- `verifier_checkpoint_refreshed`: principle "the self-improvement signal."
- `inference_substrate`: principle "live_llm_inference (60s floor)."
- `solve_provenance`: principle "live_agent_self_discovery -- self-play is the agent improving on its own attempts."
- `preconditions_checked`: principle "records arcade/registry checks; a missing target emits blocked_, never a fabricated checkpoint."

#### SCENARIO-ARC-WMTE-4813-CHECKPOINT-REFRESHED

Given a selected banked target has a recorded pre-run verifier checkpoint mtime
When the standing self-play loop reports `offline_reproduced=true`,
`reproduced_levels>=1`, a positive search-state count, and a learned verifier
checkpoint whose mtime advanced
Then Experiment 4813 writes a `success_` artifact with
`verifier_checkpoint_refreshed=true`, records both `states_expanded` and
`search_state_count`, preserves the reproduction gate, and emits
`schema_errors=[]`.

#### SCENARIO-ARC-WMTE-4813-RESIDUAL-NO-FABRICATION

Given the loop result is missing, the reproduction gate fails, the reproduced
level is below one, or the verifier checkpoint mtime does not advance
When Experiment 4813 builds its terminal artifact
Then it writes a `complete_` residual verdict, keeps
`verifier_checkpoint_refreshed=false`, records the failing residual cause, and
does not invent a checkpoint path or refreshed mtime.

#### SCENARIO-ARC-WMTE-4813-BLOCKED-PRECONDITION

Given the offline arcade, registry, target selection, or banked-level
precondition fails
When Experiment 4813 runs
Then it writes a `blocked_` artifact with the failed precondition recorded,
`verifier_checkpoint_refreshed=false`, and a deterministic reproducibility
checksum.

### REQ-ARC-WMTE-4823: ARC Self-Play Verifier Checkpoint Refresh

Experiment 4823 SHALL run the standing ARC self-play loop on a banked game and
write `results/experiment_4823_self_play_verifier_checkpoint.json`. The
workflow SHALL precheck that `arc_solver_kit.offline_arcade()` initializes,
that `ops/arc_solve_registry.yaml` is loadable, and that the selected target
game already has `levels_reproduced >= 1` in the registry. A missing arcade,
registry, target, or banked level SHALL produce a terminal `blocked_` artifact
that records the failed precondition and SHALL NOT claim a refreshed
checkpoint.

The self-play run SHALL execute `scripts/arc_loop_solve.py` for the selected
banked target, warm-start from `models/arc_verifier_<game>.json` when present,
run the verifier-routed search and reproduction gate, and train plus persist
the learned verifier when trajectory labels are available. The terminal
artifact SHALL count success only when the learned verifier checkpoint's mtime
advances past the recorded pre-run mtime and the loop reproduction gate passes
on at least one level. A failed self-play run SHALL preserve the residual cause,
search-state count, gate output, and checkpoint metadata rather than
fabricating a green checkpoint.

The artifact SHALL include principle-annotated top-level fields for
`honest_verdict`, `verifier_checkpoint_refreshed`, `inference_substrate`,
`solve_provenance`, and `preconditions_checked`, plus `target_game`,
`checkpoint_path`, `checkpoint_mtime_before_ns`, `checkpoint_mtime_after_ns`,
`checkpoint_mtime_delta_ns`, `states_expanded`, `search_state_count`,
`offline_reproduced`, `reproduced_levels`, `reproduction_gate`,
`self_play_residual`, `loop_result_path`, `random_seed`,
`reproducibility_checksum`, and `schema_errors`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; checkpoint refreshed + gate green is success_."
- `verifier_checkpoint_refreshed`: principle "the self-improvement signal."
- `inference_substrate`: principle "live_llm_inference (60s floor)."
- `solve_provenance`: principle "live_agent_self_discovery -- self-play is the agent improving on its own attempts."
- `preconditions_checked`: principle "records arcade/registry checks; a missing target emits blocked_, never a fabricated checkpoint."

#### SCENARIO-ARC-WMTE-4823-CHECKPOINT-REFRESHED

Given a selected banked target has a recorded pre-run verifier checkpoint mtime
When the standing self-play loop reports `offline_reproduced=true`,
`reproduced_levels>=1`, a positive search-state count, and a learned verifier
checkpoint whose mtime advanced
Then Experiment 4823 writes a `success_` artifact with
`verifier_checkpoint_refreshed=true`, records both `states_expanded` and
`search_state_count`, preserves the reproduction gate, and emits
`schema_errors=[]`.

#### SCENARIO-ARC-WMTE-4823-RESIDUAL-NO-FABRICATION

Given the loop result is missing, the reproduction gate fails, the reproduced
level is below one, or the verifier checkpoint mtime does not advance
When Experiment 4823 builds its terminal artifact
Then it writes a `complete_` residual verdict, keeps
`verifier_checkpoint_refreshed=false`, records the failing residual cause, and
does not invent a checkpoint path or refreshed mtime.

#### SCENARIO-ARC-WMTE-4823-BLOCKED-PRECONDITION

Given the offline arcade, registry, target selection, or banked-level
precondition fails
When Experiment 4823 runs
Then it writes a `blocked_` artifact with the failed precondition recorded,
`verifier_checkpoint_refreshed=false`, and a deterministic reproducibility
checksum.

### REQ-ARC-WMTE-4833: ARC Self-Play Verifier Checkpoint Refresh

Experiment 4833 SHALL run the standing ARC self-play loop on a banked game and
write `results/experiment_4833_self_play_verifier_checkpoint.json`. The
workflow SHALL precheck that `arc_solver_kit.offline_arcade()` initializes,
that `ops/arc_solve_registry.yaml` is loadable, and that the selected target
game already has `levels_reproduced >= 1` in the registry. A missing arcade,
registry, target, or banked level SHALL produce a terminal `blocked_` artifact
that records the failed precondition and SHALL NOT claim a refreshed
checkpoint.

The self-play run SHALL execute `scripts/arc_loop_solve.py` for the selected
banked target, warm-start from `models/arc_verifier_<game>.json` when present,
run the verifier-routed search and reproduction gate, and train plus persist
the learned verifier when trajectory labels are available. The terminal
artifact SHALL count success only when the learned verifier checkpoint's mtime
advances past the recorded pre-run mtime and the loop reproduction gate passes
on at least one level. A failed self-play run SHALL preserve the residual cause,
search-state count, gate output, and checkpoint metadata rather than
fabricating a green checkpoint.

The artifact SHALL include principle-annotated top-level fields for
`honest_verdict`, `verifier_checkpoint_refreshed`, `inference_substrate`,
`solve_provenance`, and `preconditions_checked`, plus `target_game`,
`checkpoint_path`, `checkpoint_mtime_before_ns`, `checkpoint_mtime_after_ns`,
`checkpoint_mtime_delta_ns`, `states_expanded`, `search_state_count`,
`offline_reproduced`, `reproduced_levels`, `reproduction_gate`,
`self_play_residual`, `loop_result_path`, `random_seed`,
`reproducibility_checksum`, and `schema_errors`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; refreshed + gate green is success_."
- `verifier_checkpoint_refreshed`: principle "the self-improvement signal."
- `inference_substrate`: principle "live_llm_inference (60s floor)."
- `solve_provenance`: principle "live_agent_self_discovery -- self-play is the agent improving on its own attempts."
- `preconditions_checked`: principle "records arcade/registry checks; a missing target emits blocked_, never a fabricated checkpoint."

#### SCENARIO-ARC-WMTE-4833-CHECKPOINT-REFRESHED

Given a selected banked target has a recorded pre-run verifier checkpoint mtime
When the standing self-play loop reports `offline_reproduced=true`,
`reproduced_levels>=1`, a positive search-state count, and a learned verifier
checkpoint whose mtime advanced
Then Experiment 4833 writes a `success_` artifact with
`verifier_checkpoint_refreshed=true`, records both `states_expanded` and
`search_state_count`, preserves the reproduction gate, and emits
`schema_errors=[]`.

#### SCENARIO-ARC-WMTE-4833-RESIDUAL-NO-FABRICATION

Given the loop result is missing, the reproduction gate fails, the reproduced
level is below one, or the verifier checkpoint mtime does not advance
When Experiment 4833 builds its terminal artifact
Then it writes a `complete_` residual verdict, keeps
`verifier_checkpoint_refreshed=false`, records the failing residual cause, and
does not invent a checkpoint path or refreshed mtime.

#### SCENARIO-ARC-WMTE-4833-BLOCKED-PRECONDITION

Given the offline arcade, registry, target selection, or banked-level
precondition fails
When Experiment 4833 runs
Then it writes a `blocked_` artifact with the failed precondition recorded,
`verifier_checkpoint_refreshed=false`, and a deterministic reproducibility
checksum.

### REQ-ARC-WMTE-4843: ARC Self-Play Verifier Checkpoint Refresh

Experiment 4843 SHALL run the standing ARC self-play loop on a banked game and
write `results/experiment_4843_self_play_verifier_checkpoint.json`. The
workflow SHALL precheck that `arc_solver_kit.offline_arcade()` initializes,
that `ops/arc_solve_registry.yaml` is loadable, and that the selected target
game already has `levels_reproduced >= 1` in the registry. A missing arcade,
registry, target, or banked level SHALL produce a terminal `blocked_` artifact
that records the failed precondition and SHALL NOT claim a refreshed
checkpoint.

The self-play run SHALL execute `scripts/arc_loop_solve.py` for the selected
banked target, warm-start from `models/arc_verifier_<game>.json` when present,
run the verifier-routed search and reproduction gate, and train plus persist
the learned verifier when trajectory labels are available. The terminal
artifact SHALL count success only when the learned verifier checkpoint's mtime
advances past the recorded pre-run mtime and the loop reproduction gate passes
on at least one level. A failed self-play run SHALL preserve the residual cause,
search-state count, gate output, and checkpoint metadata rather than
fabricating a green checkpoint.

The artifact SHALL include principle-annotated top-level fields for
`honest_verdict`, `verifier_checkpoint_refreshed`, `inference_substrate`,
`solve_provenance`, and `preconditions_checked`, plus `target_game`,
`checkpoint_path`, `checkpoint_mtime_before_ns`, `checkpoint_mtime_after_ns`,
`checkpoint_mtime_delta_ns`, `states_expanded`, `search_state_count`,
`offline_reproduced`, `reproduced_levels`, `reproduction_gate`,
`self_play_residual`, `loop_result_path`, `random_seed`,
`reproducibility_checksum`, and `schema_errors`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; refreshed + gate green is success_."
- `verifier_checkpoint_refreshed`: principle "the self-improvement signal."
- `inference_substrate`: principle "live_llm_inference (60s floor)."
- `solve_provenance`: principle "live_agent_self_discovery -- self-play is the agent improving on its own attempts."
- `preconditions_checked`: principle "records arcade/registry checks; a missing target emits blocked_, never a fabricated checkpoint."

#### SCENARIO-ARC-WMTE-4843-CHECKPOINT-REFRESHED

Given a selected banked target has a recorded pre-run verifier checkpoint mtime
When the standing self-play loop reports `offline_reproduced=true`,
`reproduced_levels>=1`, a positive search-state count, and a learned verifier
checkpoint whose mtime advanced
Then Experiment 4843 writes a `success_` artifact with
`verifier_checkpoint_refreshed=true`, records both `states_expanded` and
`search_state_count`, preserves the reproduction gate, and emits
`schema_errors=[]`.

#### SCENARIO-ARC-WMTE-4843-RESIDUAL-NO-FABRICATION

Given the loop result is missing, the reproduction gate fails, the reproduced
level is below one, or the verifier checkpoint mtime does not advance
When Experiment 4843 builds its terminal artifact
Then it writes a `complete_` residual verdict, keeps
`verifier_checkpoint_refreshed=false`, records the failing residual cause, and
does not invent a checkpoint path or refreshed mtime.

#### SCENARIO-ARC-WMTE-4843-BLOCKED-PRECONDITION

Given the offline arcade, registry, target selection, or banked-level
precondition fails
When Experiment 4843 runs
Then it writes a `blocked_` artifact with the failed precondition recorded,
`verifier_checkpoint_refreshed=false`, and a deterministic reproducibility
checksum.

### REQ-ARC-WMTE-4835: .445 ARC Null Silent-Bug Audit

Experiment 4835 SHALL audit the `.445` ARC null artifacts
`results/experiment_4831_amortized_incontext_exploration_prior_live.json`,
`results/experiment_4832_levelup_attempt.json`, and
`results/experiment_4834_heldout_first_win_readiness.json` for silent no-op
bugs. It SHALL write `results/experiment_4835_silent_bug_audit.json` and append
a human-readable section to `ops/arc_null_silent_bug_audit.md`. The audit SHALL
use only upstream artifacts and SHALL declare
`inference_substrate=aggregation_from_upstream_artifacts` with the `0.0001s`
duration floor.

The A1 Exp 4831 audit SHALL treat the first-win verdict as a non-test unless
`go_explore_archive_alive` reports `observations>0`, `stored_cells>0`, and
`prefixes_injected>0`, and `prior_changed_proposals=true` with diagnostics
showing materially changed proposal order versus the no-prior ablation. It
SHALL also verify that the imitation control uses held-out games outside the
distillation set before trusting an A1 null. For every other audited ARC null,
the audit SHALL verify that the relevant lever was genuinely exercised before
classifying the null as trustworthy: level-up artifacts need attempted-game,
solution-label, reproduction-gate, and same-depth accounting evidence, while
held-out first-win artifacts need the held-out attempt floor, positive-control
parity, parity-test success, aggregation/proxy evidence, and the flat-null
methodology note.

The artifact SHALL include principle-annotated top-level fields for
`honest_verdict`, `a1_archive_alive_and_prior_exercised`, `nulls_audited`, and
`inference_substrate`, plus per-null verdicts, silent-bug findings, trusted
nulls, the A1 control check, preconditions, audited-artifact checksums,
duration, random seed, and a stable reproducibility checksum. The audit SHALL
make no solve, leaderboard, training, ops-status, or fresh live-inference claim.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; audit complete is complete_/success_."
- `a1_archive_alive_and_prior_exercised`: principle "the load-bearing check -- archive alive (positive cells) AND prior changed proposals, else A1 is a non-test (exp4701 recurrence or no-op)."
- `nulls_audited`: principle "count of nulls re-examined."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts (0.0001s floor)."

#### SCENARIO-ARC-WMTE-4835-SILENT-BUG-AUDIT

Given the `.445` A1 amortized-prior, level-up no-bank, and held-out first-win
null artifacts are present
When Experiment 4835 cross-checks exercised-evidence fields
Then it writes `results/experiment_4835_silent_bug_audit.json`, sets
`a1_archive_alive_and_prior_exercised=true` only when Exp 4831 has positive
archive observations, stored cells, injected prefixes, and changed proposals,
classifies A1 as `silent_bug_must_reopen` when the archive is dead or the prior
is a no-op, verifies the imitation-control held-out split, and classifies the
level-up and held-out readiness nulls only after their exercised-lever evidence
is present.

#### SCENARIO-ARC-WMTE-4835-BLOCKED-PRECONDITION

Given any required `.445` source artifact is missing
When Experiment 4835 runs
Then it writes a blocked artifact with the missing source in
`preconditions_checked`, no silent-bug fabrication, no markdown append, and a
stable reproducibility checksum.

### REQ-ARC-WMTE-4848: Object-Centric World-Model Planning SOTA Ingestion For .447

Experiment 4848 SHALL run the reserved `.447` SOTA-ingestion slot for
object-centric world models and structured or relational planning that consume
the Exp 4838 A1 object-identity perception layer. The workflow SHALL verify
that `research-studying.md`, `research-references.md`, and
`results/experiment_4838_sota_ingestion_perception_representation.json` are
present, read the discovered corpus filtered to object-centric world-models,
object-relational dynamics, graph/slot planners, model-predictive control, and
MCTS over object-relational state, emit focused `scripts/sweep_clusters.py`
cluster URLs for world-model, affordance/action-effect, and neural-guided
world-model literature, run `scripts/sweep_semscholar.py` on focused
object-world-model and relational-planning queries, and use low-concurrency
WebSearch/WebFetch plus direct arXiv checks to verify the top five to eight
source papers. The workflow SHALL NOT invoke `/deep-research`, re-ingest the
nulled exploration-strategy class, load models, train, submit to a leaderboard,
or make a solve claim.

On a successful ingestion-synthesis run, Experiment 4848 SHALL write
`results/experiment_4848_sota_ingestion_object_world_model.json` and update
`research-studying.md` with an idempotent Exp 4848 section marked INGESTED.
The artifact SHALL include the required fields `honest_verdict`,
`methods_mapped`, `arxiv_ids_cited`, `flagged_for_v447`,
`inference_substrate`, `preconditions_checked`, `citations`, `fresh_sweep`,
`a1_perception_layer_input`, `object_world_model_mapping_note`, `note_path`,
`random_seed`, `duration_s`, `reproducibility_checksum`, and
`field_principles`.

The complete-path `honest_verdict` SHALL start with `success_` and equal
`success_sota_ingestion_object_world_model_mapped`; `inference_substrate`
SHALL equal `aggregation_from_upstream_artifacts`; and `flagged_for_v447`
SHALL name the strongest method or methods for the `.447` planner. The mapping
SHALL explain how each method consumes the A1 object layer from Exp 4838
(`object_ids`, slots, relation edges, persistence tracks, object/action
bindings, and causal shortcut guards) to produce a proposable winner on a novel
game rather than merely ranking, re-weighting, or exploring an unchanged pool.
The mapping SHALL cover object-relational transition graphs, slot-structured
rollout world models, object-centric MCTS or MuZero-style planning,
goal-conditioned object-level MPC, modular interaction-primitive policies, and
ARC composable slot-transition loops.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; mapping emitted is success_sota_ingestion_object_world_model_mapped."
- `methods_mapped`: principle "the strongest 3-5 object-centric world-model/planning methods mapped onto consuming the A1 perception layer, each with a real arXiv ID."
- `arxiv_ids_cited`: principle "every method claim must cite a verifiable arXiv ID."
- `flagged_for_v447`: principle "the strongest method(s) flagged so the .447 planner reads the mapping."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts (0.0001s floor)."

The source set SHALL contain only verified arXiv IDs from the focused pass:
`1911.12247`, `2402.03326`, `2507.03298`, `2511.02225`, `2601.06604`,
`2605.14937`, `2606.12316`, and `2606.14418`. The method map
SHALL contain three to five methods, and every method claim SHALL cite only IDs
from `arxiv_ids_cited`. The note SHALL identify Contrastively-trained
Structured World Models, Slot Structured World Models, Dyn-O, FIOC-WM,
ObjectZero, Slot-MPC, Loop-OWM, and COMET as the ingested literature informing
`.447` object-world-model planning.

#### SCENARIO-ARC-WMTE-4848

Given the research files and Exp 4838 perception artifact are present, the
reliable sweep helpers are available, Semantic Scholar is attempted through the
reliable script, and the top arXiv sources are HTTP-200 verified
When Experiment 4848 runs
Then it writes
`results/experiment_4848_sota_ingestion_object_world_model.json`, records that
`/deep-research` was not invoked, maps three to five SOTA methods onto the
object-centric world-model/planning handoff with real arXiv IDs, updates
`research-studying.md` with the Exp 4848 mapping note, and flags the strongest
candidate inputs for `.447`.

#### SCENARIO-ARC-WMTE-4848-NO-FABRICATION

Given a candidate artifact omits citations, cites an unverified arXiv ID, uses
a stale `.446` roadmap flag, maps generic exploration re-weighting instead of
object-world-model planning, omits the A1-object-layer consumption or
proposable-winner handoff, omits the Exp 4838 upstream carry-forward, or claims
a model-load, training, leaderboard, or solve result
When the Experiment 4848 artifact validator runs
Then it rejects the artifact before the result can be written.

### REQ-ARC-WMTE-4838: Perception/Representation SOTA Ingestion For The L1-First-Contact Wall

Experiment 4838 SHALL run the reserved `.446` SOTA-ingestion slot for
object-centric, slot, relational, and structured-state perception. The workflow
SHALL verify that `research-studying.md` and `research-references.md` are
present, read the discovered corpus filtered to object-centric and relational
representation learning, emit focused `scripts/sweep_clusters.py` cluster URLs
for world-model, affordance/action-effect, and neural-guided world-model
literature, run `scripts/sweep_semscholar.py` on focused perception and
representation queries, and use low-concurrency WebSearch/WebFetch plus direct
arXiv checks to verify the top five to eight source papers. The workflow SHALL
NOT invoke `/deep-research`, re-ingest the nulled exploration-strategy class,
load models, train, submit to a leaderboard, or make a solve claim.

On a successful ingestion-synthesis run, Experiment 4838 SHALL write
`results/experiment_4838_sota_ingestion_perception_representation.json` and
update `research-studying.md` with an idempotent Exp 4838 section marked
INGESTED. The artifact SHALL include the required fields `honest_verdict`,
`methods_mapped`, `arxiv_ids_cited`, `flagged_for_v446`,
`inference_substrate`, `preconditions_checked`, `citations`, `fresh_sweep`,
`l1_wall_context`, `perception_representation_mapping_note`, `note_path`,
`random_seed`, `duration_s`, `reproducibility_checksum`, and
`field_principles`.

The complete-path `honest_verdict` SHALL start with `success_` and equal
`success_sota_ingestion_perception_representation_mapped`;
`inference_substrate` SHALL equal `aggregation_from_upstream_artifacts`; and
`flagged_for_v446` SHALL name the strongest method or methods for the `.446`
planner. The mapping SHALL cover the L1-first-contact wall by explaining how
each method makes the winning prefix representable and proposable on a novel
game instead of merely re-weighting an unchanged candidate pool. The mapping
SHALL cover slot/object state construction, relational transition graphs,
object-centric world-model planning, causal/object-level latent prediction, and
ARC-specific composable slot-transition loops.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; mapping emitted is success_sota_ingestion_perception_representation_mapped."
- `methods_mapped`: principle "the strongest 3-5 perception/representation methods mapped onto the L1-wall root cause, each with a real arXiv ID."
- `arxiv_ids_cited`: principle "every method claim must cite a verifiable arXiv ID."
- `flagged_for_v446`: principle "the strongest method(s) flagged so the .446 planner reads the mapping (perception, not more exploration)."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts (0.0001s floor)."

The source set SHALL contain only verified arXiv IDs from the focused pass:
`1802.04687`, `1911.12247`, `2006.15055`, `2402.03326`, `2507.03298`,
`2601.06604`, `2602.11389`, and `2606.12316`. The method map SHALL contain
three to five methods, and every method claim SHALL cite only IDs from
`arxiv_ids_cited`. The note SHALL identify Neural Relational Inference,
Contrastively-trained Structured World Models, Slot Attention, Slot Structured
World Models, Dyn-O, Object-Centric World Models with MCTS, Causal-JEPA, and
Loop-OWM as the ingested literature informing `.446` perception and
representation.

#### SCENARIO-ARC-WMTE-4838

Given the research files are present, the reliable sweep helpers are available,
Semantic Scholar is attempted through the reliable script, and the top arXiv
sources are HTTP-200 verified
When Experiment 4838 runs
Then it writes
`results/experiment_4838_sota_ingestion_perception_representation.json`,
records that `/deep-research` was not invoked, maps three to five SOTA methods
onto the L1-first-contact perception/representation root cause with real arXiv
IDs, updates `research-studying.md` with the Exp 4838 mapping note, and flags
the strongest candidate inputs for `.446`.

#### SCENARIO-ARC-WMTE-4838-NO-FABRICATION

Given a candidate artifact omits citations, cites an unverified arXiv ID, uses
a stale `.445` roadmap flag, maps exploration re-weighting instead of
perception/representation, omits the winner-representable/proposable handoff,
or claims a model-load, training, leaderboard, or solve result
When the Experiment 4838 artifact validator runs
Then it rejects the artifact before the result can be written.

### REQ-ARC-WMTE-4775: .439 ARC Null Silent-Bug Audit

Experiment 4775 SHALL audit the `.439` ARC null artifacts before their negative
or reopen/retire results are trusted as mechanism evidence. The audit SHALL read
`results/experiment_4771_structural_energy_s0prime_origin_matched.json`,
`results/experiment_4772_levelup_attempt.json`, and
`results/experiment_4774_heldout_first_win_readiness.json`. Missing `.439`
source artifacts SHALL produce a blocked artifact rather than a fabricated
audit.

The audit SHALL cross-check each source artifact's exercised-evidence fields:
for S0' it SHALL verify that origin matching is real, that contributing
per-game class-balance rows contain both correct and wrong induced predictions,
that single-class games are skipped rather than folded into LOO, that the
origin-probe leak control was recomputed on the origin-matched data rather than
copied from S0, and that the shuffled-label control actually permuted labels
and reran LOO rather than emitting a hardcoded `0.5`; for the level-up no-bank
residual it SHALL verify a real attempted-game ledger, solution labels, and
reproduction gate; and for the held-out first-win null it SHALL verify the
held-out attempt floor, positive-control parity, and the flat `0.04`
methodology note. It SHALL flag silent bugs for dead or identity engines,
byte-identical A/B arms, unannotated `0.04` first-win tautologies,
representation no-ops, and any S0' leak control that did not genuinely fire.

Experiment 4775 SHALL write
`results/experiment_4775_silent_bug_audit.json` and append a human-readable
section to `ops/arc_null_silent_bug_audit.md`. The artifact SHALL include
principle-annotated top-level fields for `honest_verdict`, `nulls_audited`,
`s0prime_leak_controls_fired`, and `inference_substrate`, plus per-null
verdicts, silent-bug findings, trusted nulls, preconditions, checksums for
audited artifacts, duration, random seed, and a stable reproducibility checksum.
The audit SHALL make no solve, leaderboard, training, or live-inference claim.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; audit complete is complete_/success_."
- `nulls_audited`: principle "count of nulls re-examined -- a null that tested dead code is not trustworthy."
- `s0prime_leak_controls_fired`: principle "the load-bearing check -- the origin-probe + shuffled-label controls must have genuinely executed, else S0''s reopen/retire verdict is uninterpretable."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts; 0.0001s floor."

#### SCENARIO-ARC-WMTE-4775-SILENT-BUG-AUDIT

Given the `.439` S0', level-up no-bank, and held-out first-win artifacts are
present
When Experiment 4775 cross-checks exercised-evidence fields
Then it writes `results/experiment_4775_silent_bug_audit.json`, appends to
`ops/arc_null_silent_bug_audit.md`, classifies S0' as a silent bug when either
leak control did not genuinely execute, keeps genuinely exercised no-bank and
flat first-win nulls trusted only when their evidence fields are non-degenerate,
and reports the complete audit with
`inference_substrate=aggregation_from_upstream_artifacts`.

#### SCENARIO-ARC-WMTE-4775-BLOCKED-PRECONDITION

Given any required `.439` source artifact is missing
When Experiment 4775 runs
Then it writes a blocked artifact with the missing source in
`preconditions_checked`, no silent-bug fabrication, no markdown append, and a
stable reproducibility checksum.

### REQ-ARC-WMTE-4785: .440 ARC Null Silent-Bug Audit

Experiment 4785 SHALL audit the `.440` ARC null and control-bearing artifacts
before their negative or pass/bound verdicts are trusted as mechanism evidence.
The audit SHALL read
`results/experiment_4781_structural_energy_s1_contrastive_landscape.json`,
`results/experiment_4782_levelup_attempt.json`, and
`results/experiment_4784_heldout_first_win_readiness.json`. Missing `.440`
source artifacts SHALL produce a blocked artifact rather than a fabricated
audit.

The audit SHALL cross-check each source artifact's exercised-evidence fields.
For S1 it SHALL verify that the origin-probe leak control was recomputed or
refit on the origin-matched S1 data rather than copied from S0/S0', that the
shuffled-label control actually permuted labels and reran LOO energy metrics,
that the denoising-direction agreement came from executable energy-direction
code rather than a hardcoded artifact value, and that at least ten distinct
random seeds contributed to the multi-seed result. For the level-up no-bank
residual it SHALL verify a real attempted-game ledger, solution labels, and
reproduction gate. For the held-out first-win null it SHALL verify the held-out
attempt floor, positive-control parity, and the flat `0.04` methodology note.
It SHALL flag silent bugs for dead or identity engines, byte-identical A/B arms,
representation no-ops, and any S1 leak or denoising control that did not
genuinely fire.

Experiment 4785 SHALL write
`results/experiment_4785_silent_bug_audit.json` and append a human-readable
section to `ops/arc_null_silent_bug_audit.md`. The artifact SHALL include
principle-annotated top-level fields for `honest_verdict`, `nulls_audited`,
`s1_controls_fired`, and `inference_substrate`, plus per-null verdicts,
silent-bug findings, trusted nulls, S1 control checks, preconditions, checksums
for audited artifacts, duration, random seed, and a stable reproducibility
checksum. The audit SHALL make no solve, leaderboard, training, or live-
inference claim.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; audit complete is complete_/success_."
- `nulls_audited`: principle "count of nulls re-examined -- a null that tested dead code is not trustworthy."
- `s1_controls_fired`: principle "the load-bearing check -- the leak + denoising controls must have genuinely executed, else S1's pass/bound verdict is uninterpretable."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts (0.0001s floor)."

#### SCENARIO-ARC-WMTE-4785-SILENT-BUG-AUDIT

Given the `.440` S1, level-up no-bank, and held-out first-win artifacts are
present
When Experiment 4785 cross-checks exercised-evidence fields
Then it writes `results/experiment_4785_silent_bug_audit.json`, appends to
`ops/arc_null_silent_bug_audit.md`, classifies S1 as a silent bug when any leak
or denoising control did not genuinely execute, keeps genuinely exercised
no-bank and flat first-win nulls trusted only when their evidence fields are
non-degenerate, and reports the complete audit with
`inference_substrate=aggregation_from_upstream_artifacts`.

#### SCENARIO-ARC-WMTE-4785-BLOCKED-PRECONDITION

Given any required `.440` source artifact is missing
When Experiment 4785 runs
Then it writes a blocked artifact with the missing source in
`preconditions_checked`, no silent-bug fabrication, no markdown append, and a
stable reproducibility checksum.

### REQ-ARC-WMTE-4795: .441 ARC Null Silent-Bug Audit

Experiment 4795 SHALL audit the `.441` ARC null artifacts before their negative
verdicts are treated as capability limits. The audit SHALL read
`results/experiment_4791_structural_energy_s2_offpath_trust_gate.json`,
`results/experiment_4792_levelup_attempt.json`, and
`results/experiment_4794_heldout_first_win_readiness.json`. Missing `.441`
source artifacts SHALL produce a blocked artifact rather than a fabricated
audit. Experiment 4793 is a success checkpoint refresh and SHALL be excluded
from the null count unless it emits a terminal null/residual artifact.

The audit SHALL verify that each null's lever was genuinely exercised. For
S2/Experiment 4791 it SHALL run `scripts/arc_orphan_solver_lint.py`, confirm
that the structural-energy gate is reachable from the live `E3AgentPolicy`
import path, and reconstruct the control as the real incumbent binary
`WorldModelVerifier` exact-accuracy `>=0.5` selection over the identical
candidate engine sets. The audit SHALL flag S2 as a silent bug if the energy
gate is not live-path reachable, if the control is not the real binary gate, if
the energy and binary arms do not use identical candidate sets, or if the
binary selected candidate cannot be reconstructed as the first candidate
passing the exact-accuracy threshold or the first candidate when none pass. For
the level-up no-bank null it SHALL verify non-empty attempted-game rows,
solution labels, reproduction gates, and same-depth/new-depth accounting. For
the held-out first-win null it SHALL verify the held-out attempt floor,
positive-control parity, parity-test success, and the flat `0.04` methodology
note.

Experiment 4795 SHALL write
`results/experiment_4795_silent_bug_audit.json` and append a human-readable
section to `ops/arc_null_silent_bug_audit.md`. The artifact SHALL include
principle-annotated top-level fields for `honest_verdict`, `nulls_audited`,
`s2_live_path_reachable_confirmed`, and `inference_substrate`, plus per-null
verdicts, silent-bug findings, trusted nulls, S2 control checks, the
`arc_orphan_solver_lint.py` result, preconditions, checksums for audited
artifacts, duration, random seed, and a stable reproducibility checksum. The
audit SHALL make no solve, leaderboard, training, or fresh live-inference
claim.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; audit complete is complete_/success_."
- `nulls_audited`: principle "count of nulls re-examined."
- `s2_live_path_reachable_confirmed`: principle "the load-bearing check -- S2's energy gate must be reachable from E3AgentPolicy, else its 'live trust gate' claim is hollow."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts (0.0001s floor)."

#### SCENARIO-ARC-WMTE-4795-SILENT-BUG-AUDIT

Given the `.441` S2, level-up no-bank, and held-out first-win null artifacts
are present
When Experiment 4795 cross-checks exercised-evidence fields and runs
`scripts/arc_orphan_solver_lint.py`
Then it writes `results/experiment_4795_silent_bug_audit.json`, appends to
`ops/arc_null_silent_bug_audit.md`, confirms
`s2_live_path_reachable_confirmed=true` only when the lint passes and the S2
artifact also recorded a live reachable gate, keeps S2 trusted only when the
real binary gate is reconstructed over identical candidate sets, and reports
the complete audit with
`inference_substrate=aggregation_from_upstream_artifacts`.

#### SCENARIO-ARC-WMTE-4795-BLOCKED-PRECONDITION

Given any required `.441` source artifact is missing
When Experiment 4795 runs
Then it writes a blocked artifact with the missing source in
`preconditions_checked`, no silent-bug fabrication, no markdown append, and a
stable reproducibility checksum.

### REQ-ARC-WMTE-4805: .442 ARC Null Silent-Bug Audit

Experiment 4805 SHALL audit the `.442` ARC null artifacts before their negative
verdicts are treated as capability limits. The audit SHALL read
`results/experiment_4801_structural_energy_s2v2_diverse_trust_gate.json`,
`results/experiment_4802_levelup_attempt.json`, and
`results/experiment_4804_heldout_first_win_readiness.json`. Missing `.442`
source artifacts SHALL produce a blocked artifact rather than a fabricated
audit. Experiment 4803 is a success checkpoint refresh and SHALL be excluded
from the null count unless it emits a terminal null/residual artifact.

The audit SHALL verify that each null's lever was genuinely exercised. For
S2-v2/Experiment 4801 it SHALL run
`adversarial_verify.check_engine_selection_candidate_diversity` on the source
artifact and SHALL flag S2-v2 as a silent bug if
`DEGENERATE_CANDIDATE_POOL` fires. The S2-v2 audit SHALL also confirm
`n_effective_games >= min_heldout_games`, confirm that every effective game
logs `game_results[*].candidate_rows`, the energy-selected candidate, the
accuracy-gate-selected candidate, and both selectors' held-out off-path
`cell_recall`, and run `scripts/arc_orphan_solver_lint.py` to keep the graft
live-path-reachable. For the level-up no-bank null it SHALL verify non-empty
attempted-game rows, target/depth accounting, solution-label or terminal
residual evidence, reproduction gates for reproduced depths, and same-depth
vs. new-depth accounting. For the held-out first-win null it SHALL verify the
held-out attempt floor, positive-control parity, parity-test success, proxy
aggregation evidence, and the flat `0.04` methodology note.

Experiment 4805 SHALL write
`results/experiment_4805_silent_bug_audit.json` and append a human-readable
section to `ops/arc_null_silent_bug_audit.md`. The artifact SHALL include
principle-annotated top-level fields for `honest_verdict`,
`s2v2_candidate_pool_diverse`, `nulls_audited`, and `inference_substrate`,
plus per-null verdicts, silent-bug findings, trusted nulls, the S2-v2
diversity check, the `arc_orphan_solver_lint.py` result, preconditions,
checksums for audited artifacts, duration, random seed, and a stable
reproducibility checksum. The audit SHALL make no solve, leaderboard,
training, or fresh live-inference claim.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; audit complete is complete_/success_."
- `s2v2_candidate_pool_diverse`: principle "the load-bearing check -- DEGENERATE_CANDIDATE_POOL must NOT fire on S2-v2, else its verdict is again a non-test."
- `nulls_audited`: principle "count of nulls re-examined."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts (0.0001s floor)."

#### SCENARIO-ARC-WMTE-4805-SILENT-BUG-AUDIT

Given the `.442` S2-v2, level-up no-bank, and held-out first-win null artifacts
are present
When Experiment 4805 cross-checks exercised-evidence fields, runs
`adversarial_verify.check_engine_selection_candidate_diversity`, and runs
`scripts/arc_orphan_solver_lint.py`
Then it writes `results/experiment_4805_silent_bug_audit.json`, appends to
`ops/arc_null_silent_bug_audit.md`, sets
`s2v2_candidate_pool_diverse=true` only when `DEGENERATE_CANDIDATE_POOL` does
not fire and `n_effective_games >= min_heldout_games`, keeps S2-v2 trusted only
when per-game energy-vs-accuracy selections are logged and the live-path lint
passes, keeps the other ARC nulls trusted only when their exercised-evidence
fields are non-degenerate, and reports the complete audit with
`inference_substrate=aggregation_from_upstream_artifacts`.

#### SCENARIO-ARC-WMTE-4805-BLOCKED-PRECONDITION

Given any required `.442` source artifact is missing
When Experiment 4805 runs
Then it writes a blocked artifact with the missing source in
`preconditions_checked`, no silent-bug fabrication, no markdown append, and a
stable reproducibility checksum.

### REQ-ARC-WMTE-4731: .435 Submitted-Agent Integration Gate

Experiment 4731 SHALL read the `.435` A1/A2 artifacts
`results/experiment_4726_online_action_learning_driver_valid_test.json` and
`results/experiment_4727_active_probe_disambiguation.json`, choose the
strongest non-unchanged `chosen_submitted_config` that has a positive held-out
first-win lift, and integrate that change into `SUBMITTED_AGENT_CONFIG` only
when the submitted config exactly matches the selected change, the
submitted-agent parity test passes, and the measured held-out first-win does
not regress versus the pre-integration baseline. If both upstream artifacts
select `unchanged` or have no positive held-out lift, the integration SHALL be
a no-op with `integrated_change="none"` and SHALL report the flat delta
honestly rather than fabricating a win.

Experiment 4731 SHALL verify the offline arcade is importable and both `.435`
A1/A2 artifacts are present before measuring. Missing resources SHALL produce a
terminal `blocked_<resource>` verdict and record the failed resource in
`preconditions_checked`. The experiment SHALL run the submitted-agent parity
test as a hard gate, set `verifier_is_oracle=false`, write
`results/experiment_4731_integration_gate.json`, and never submit to the
leaderboard.

The artifact SHALL include `honest_verdict`, `inference_substrate`,
`integrated_change`, `live_first_win_rate_integrated`,
`live_first_win_rate_pre_integration`, `parity_test_green`,
`null_delta_methodology_note`, `positive_control_passed`,
`verifier_is_oracle`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`. When
`live_first_win_rate_integrated == live_first_win_rate_pre_integration`, the
artifact SHALL emit a non-empty `null_delta_methodology_note` and
`positive_control_passed` SHALL equal
`bool(parity_test_green AND no_regression_vs_pre_integration)`. The artifact
SHALL NOT rely on a `tautology_guard` prose field for the null-delta carve-out.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success: integrated_<change>_first_win_<delta> OR complete: integration_no_change_all_levers_unchanged."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- scores the integrated config over cached variants (1s floor)."
- `integrated_change`: principle "the change integrated into SUBMITTED_AGENT_CONFIG (or 'none' if all A1/A2 selected unchanged)."
- `live_first_win_rate_integrated`: principle "the held-out first-win of the integrated config -- the scored-lane proxy."
- `live_first_win_rate_pre_integration`: principle "the pre-integration baseline -- the no-regression control."
- `parity_test_green`: principle "HARD gate -- the integrated agent is byte-for-byte the SUBMITTED_AGENT_CONFIG; a parity miss invalidates the integration."
- `null_delta_methodology_note`: principle "present when both deltas ~0; the TAUTOLOGY carve-out reads it (honest no-change, not a measurement bug)."
- `positive_control_passed`: principle "bool(parity_test_green AND no_regression) -- GATES the TAUTOLOGY exemption; an unvalidated integration is NOT excused."
- `verifier_is_oracle`: principle "false -- the integration gate measures the agent; no oracle is invoked."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent harness/corpus drift."
- `preconditions_checked`: principle "records resources verified (offline arcade, A1/A2 artifacts present); pre-empts missing-resource fabrication."

### SCENARIO-ARC-WMTE-4731-HONEST-NULL-INTEGRATION

Given the `.435` A1 and A2 artifacts both select
`chosen_submitted_config="unchanged"`
When Experiment 4731 runs the integration gate and the submitted-agent parity
test passes
Then it keeps `SUBMITTED_AGENT_CONFIG` unchanged, records
`integrated_change="none"`, reports equal integrated and pre-integration
held-out first-win rates, emits `null_delta_methodology_note`,
sets `positive_control_passed` from parity plus no-regression, and writes a
stable `results/experiment_4731_integration_gate.json` checksum.

### REQ-ARC-WMTE-4621: ARC Sprint Integration Gate for the Scored Agent

Experiment 4621 SHALL consolidate the ARC sprint's measured wins into the scored
submitted agent configuration. The workflow SHALL first verify
`arc_solver_kit.offline_arcade()` and the presence of
`results/experiment_4617_graduate_spatial_value_head_live.json`,
`results/experiment_4618_levelup_selfplay.json`, and
`results/experiment_4619_refresh_submission_package.json`. It SHALL read A2
and A3 through `scripts/summarize_artifact.py` before aggregating any number.
A lever may be admitted into `SUBMITTED_AGENT_CONFIG` only when its upstream
artifact has a terminal success verdict, is not stamped or live-rechecked as
adversarial, passes its positive control, and raises a real scored-agent metric.
Any flagged-adversarial, positive-control-failed, false-negative-risk, or
non-success artifact SHALL be quarantined and SHALL NOT contribute headline
numbers.

If Experiment 4617 cleanly raises live first-win rate or action efficiency, the
workflow SHALL ship its graduated SpatialValueNet setting in
`SUBMITTED_AGENT_CONFIG` with the measured value mode and value weight. If
Experiment 4618 cleanly banks a level, the workflow SHALL rely on the refreshed
Experiment 4619 package that already folded it in. If no clean upstream lever
raises a scored-agent metric, the workflow SHALL keep the bare submitted config,
rerun the parity and orphan-lint gates, and emit an honest null. The gate SHALL
never submit to the leaderboard and SHALL write
`results/experiment_4621_integration_gate.json`.

Experiment 4621 SHALL emit bare top-level fields for `honest_verdict`,
`inference_substrate`, `verifier_is_oracle`, `levers_integrated`,
`flagged_artifacts_excluded`, `offline_to_live_transfer_ratio_integrated`,
`first_win_rate_integrated`, `median_actions_to_first_levelup_integrated`,
`live_submittable_level_count_integrated`, `parity_test_green`,
`orphan_lint_green`, `submitted_config_raised_metric_clean`,
`null_delta_methodology_note`, `random_seed`, `reproducibility_checksum`, and
`preconditions_checked`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success: integrated_<metric>_raised_config_shipped OR complete: integration_no_clean_metric_bare_config_kept_honest_null."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates -- offline end-to-end re-measure (1s floor)."
- `verifier_is_oracle`: principle "MUST be false for every aggregated value claim -- the integrated levers are oracle-distinct."
- `levers_integrated`: principle "names the upstream levers (A2/A3) admitted into SUBMITTED_AGENT_CONFIG -- the audit trail."
- `flagged_artifacts_excluded`: principle "names any flagged_adversarial / positive-control-failed upstream artifact NOT aggregated (the fabrication gate + FALSE_NEGATIVE_RISK compliance)."
- `offline_to_live_transfer_ratio_integrated`: principle "the integrated bridge co-metric (B1's headline on the shipped config) -- did the offline->live bridge fix make it into the SCORED agent."
- `first_win_rate_integrated`: principle "the integrated held-out first-win-rate on the shipped config (A2's effect)."
- `live_submittable_level_count_integrated`: principle "the integrated live-submittable count (must stay > 33)."
- `parity_test_green`: principle "HARD gate -- test_arc_submitted_agent_parity.py passes; the integrated config is the single source of truth."
- `orphan_lint_green`: principle "HARD gate -- arc_orphan_solver_lint passes; the graduated value head stays live-path-reachable."
- `submitted_config_raised_metric_clean`: principle "True only if a CLEAN (non-flagged, control-passed) lever raised a real metric on the SCORED config; false -> honest null, bare config kept."
- `null_delta_methodology_note`: principle "present where any integrated delta == 0 -- states the equality is an honest no-value null, not a bug."
- `random_seed`: principle "determinism precondition for reproducibility."
- `reproducibility_checksum`: principle "content-addressed hash catches silent drift on replay."
- `preconditions_checked`: principle "records resources verified (offline arcade, upstream artifacts present); pre-empts missing-resource fabrication."

#### SCENARIO-ARC-WMTE-4621

**Given** A2 is flagged adversarial or fails its positive control, A3 does not
offline-reproduce a new level, and A4 reports a clean refreshed package
**When** experiment 4621 runs the integration gate
**Then** it excludes the flagged or non-success upstream metrics, leaves
`SUBMITTED_AGENT_CONFIG` unchanged, keeps `test_arc_submitted_agent_parity.py`
and `scripts/arc_orphan_solver_lint.py` green, reports the live-submittable count
from the clean refreshed package, and writes an honest-null artifact with a
stable checksum.

### REQ-ARC-WMTE-4841: Grid-Grounded Object Identity Perception Probe

Experiment 4841 SHALL build a prototype object-identity perception module under
`python/carnot/agentic/` that segments rendered ARC frames into connected
components and tracks object identity across frame transitions by shape,
connectivity, overlap, and displacement rather than by color identity. The
module SHALL expose deterministic per-object tracks and object-relational
features suitable for later goal-predicate grounding, while making no live
solver integration or first-win claim in this milestone.

The probe SHALL measure real rendered frames for `lp85`, `r11l`, and `tu93`
from the offline environment or banked frame sources. Missing frames for one
game SHALL be recorded and skipped; if no target game has frames, the experiment
SHALL write a terminal `blocked_no_offline_frames` artifact instead of
fabricating correspondence scores. Before frame measurement the workflow SHALL
verify `.venv/bin/python -c "from carnot.agentic import arc_solver_kit as k; k.offline_arcade()"`
or the equivalent import smoke; failure SHALL produce
`blocked_offline_arcade_missing`.

Experiment 4841 SHALL compare the shape/motion tracker against a color-centroid
baseline on the same frames and SHALL report a quantitative per-game
correspondence score rather than a binary self-grade. The positive control on
`tu93` SHALL require stable player and goal identity tracks on visible-goal
solving transitions. The headline gate SHALL pass only when the `tu93` positive
control passes and at least two of `lp85`, `r11l`, and `tu93` have a
shape/motion correspondence score materially above the color-centroid baseline.
If the shape/motion tracker also fails on at least two games, the artifact SHALL
report the deeper rendered-grid object-identity null honestly.

Experiment 4841 SHALL write
`results/experiment_4841_object_identity_perception_probe.json` with
principle-annotated top-level fields for `honest_verdict`,
`measured_on_real_frames`, `per_game_correspondence`,
`positive_control_tu93_passed`, `games_with_recovery`, `verifier_is_oracle`,
`live_path_reachable`, `solve_provenance`, `inference_substrate`,
`preconditions_checked`, `random_seed`, and `reproducibility_checksum`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; recovery is success_object_identity_perception_recovers_goal_grounding, a null is complete_object_identity_unrecoverable_from_rendered_grid_deeper_finding."
- `measured_on_real_frames`: principle "true iff the correspondence scores are measured on REAL lp85/r11l/tu93 frames, NOT synthetic -- the mechanic-template synthetic-only-pass trap; a false here is a NON-TEST."
- `per_game_correspondence`: principle "per-game mapping game -> (shape_motion_score, color_centroid_baseline_score, n_frames) -- the quantitative recovery measure vs the color-centroid comparator on the same frames."
- `positive_control_tu93_passed`: principle "the tu93 visible-goal control must yield stable player+goal identity tracks -- a Phase-Prototype positive control so a global null is not a harness artifact."
- `games_with_recovery`: principle "count of the three test games (lp85, r11l, tu93) where the shape/motion tracker materially beats color-centroid -- >=2 for a PASS."
- `verifier_is_oracle`: principle "true -- execution-grounded structural perception, NOT an oracle-distinct moat (circularity discipline); this probe is about grounding."
- `live_path_reachable`: principle "the module is importable by the live agent (arc_orphan_solver_lint passes) -- a perception layer the live agent cannot reach is wasted effort."
- `solve_provenance`: principle "development_proxy -- an offline perception probe on banked frames, NOT a live first-win; declared honestly (not live_agent_self_discovery)."
- `inference_substrate`: principle "live_llm_inference if any LLM induction runs, else verifier_ensemble_against_cached_candidates -- declare what actually ran (the probe is mostly CPU segmentation; do not claim live inference if none ran)."
- `preconditions_checked`: principle "records arcade/frame-availability checks so a missing-resource run emits blocked_, never a fabricated correspondence score."
- `random_seed`: principle "determinism for any stochastic segmentation/tracking step + the baseline."
- `reproducibility_checksum`: principle "content hash of (frames, tracker params, baseline) so a replication catches drift."

#### SCENARIO-ARC-WMTE-4841-REAL-FRAME-CORRESPONDENCE

Given real offline/banked frames are available for at least one of `lp85`,
`r11l`, or `tu93`
When Experiment 4841 runs the object-identity tracker and the color-centroid
baseline on the same transition sequence
Then the artifact records `measured_on_real_frames=true`, one
`per_game_correspondence` row per available target game, frame counts, stable
shape/motion scores, color-baseline scores, `games_with_recovery`, and a stable
checksum over frames and tracker parameters.

#### SCENARIO-ARC-WMTE-4841-TU93-POSITIVE-CONTROL

Given `tu93` real solving-transition frames are available
When the prototype tracker runs on that sequence
Then `positive_control_tu93_passed=true` only if at least one player-like moving
track and one goal-like persistent track remain stable enough to ground the
visible-goal control.

#### SCENARIO-ARC-WMTE-4841-LIVE-PATH-REACHABLE

Given the object-identity module is committed under `python/carnot/agentic/`
When `scripts/arc_orphan_solver_lint.py` runs
Then the lint passes and the artifact records `live_path_reachable=true` only
when the module is importable from the live agent path or consciously
allow-listed with a reason.

### REQ-ARC-WMTE-4845: A1 Perception Probe Adversarial Audit

Experiment 4845 SHALL adversarially audit the Exp 4841 object-identity
perception probe before the .446 A1 result is trusted. The workflow SHALL read
`results/experiment_4841_object_identity_perception_probe.json` through
`scripts/summarize_artifact.py`, run `scripts/adversarial_verify.py` on that
artifact, run the four hostile checks below, and append the audit outcome to
`ops/arc_null_silent_bug_audit.md`.

The audit SHALL verify all of the following:

- `measured_on_real_frames=true` and every `per_game_correspondence` row used
  for the decision comes from real `lp85`/`r11l`/`tu93` banked/offline frames,
  not synthetic alignment cases.
- The shape/motion tracker materially differs from the color-centroid baseline:
  per-game shape-minus-baseline deltas are measured and are not all zero.
- The `tu93` positive control genuinely passed, and any >=2-game recovery
  claim matches the `per_game_correspondence` rows rather than celebrating over
  failing numbers.
- `scripts/arc_orphan_solver_lint.py` passes, `live_path_reachable=true`, and
  `solve_provenance=development_proxy` rather than `live_agent_self_discovery`.

Experiment 4845 SHALL write
`results/experiment_4845_perception_probe_audit.json` with
principle-annotated top-level fields for `honest_verdict`,
`a1_genuinely_exercised`, and `inference_substrate`. It SHALL also record the
source artifact path and checksum, summarizer result, adversarial verification
result, per-game correspondence deltas, the four hostile check payloads,
preconditions checked, random seed, duration, and a reproducibility checksum.
A clean audit SHALL use a terminal `complete_` verdict and set
`a1_genuinely_exercised=true`; a synthetic-only, baseline-degenerate,
positive-control-failed, mismatched-verdict, unreachable-live-path, or dishonest
provenance result SHALL be recorded as a non-test with
`a1_genuinely_exercised=false`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; audit complete is complete_/success_."
- `a1_genuinely_exercised`: principle "the load-bearing check -- measured_on_real_frames AND tracker!=baseline no-op AND positive control real AND verdict matches numbers; else A1 is a non-test (synthetic-only or a silent degenerate-to-baseline)."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts (0.0001s floor)."

#### SCENARIO-ARC-WMTE-4845-A1-HOSTILE-AUDIT

Given the Exp 4841 perception artifact is present
When Experiment 4845 runs the summarizer, adversarial verifier, real-frame,
tracker-vs-baseline, positive-control, recovery-claim, live-path, and
solve-provenance checks
Then it writes `results/experiment_4845_perception_probe_audit.json`, appends a
stable section to `ops/arc_null_silent_bug_audit.md`, records
`inference_substrate=aggregation_from_upstream_artifacts`, and marks
`a1_genuinely_exercised=true` only when the hostile checks all pass.

#### SCENARIO-ARC-WMTE-4845-NON-TEST-CLASSIFICATION

Given an Exp 4841-like artifact that is synthetic-only, has shape/motion scores
equal to the color-centroid baseline on every game, fails the `tu93` positive
control, claims success with fewer than two recovered games, is unreachable from
the live path, or declares `solve_provenance=live_agent_self_discovery`
When Experiment 4845 audits that artifact
Then the audit records the corresponding failed check, emits a terminal
`complete_a1_perception_probe_non_test_*` verdict, and sets
`a1_genuinely_exercised=false`.

### REQ-ARC-WMTE-4851: L1 First-Contact Generation Coverage Diagnostic

Experiment 4851 SHALL measure the L1-first-contact candidate-generation wall as
a Phase-Prototype diagnostic, not as another solve lever. The workflow SHALL use
the live generic first-contact proposer path (`StepwiseExplorer._candidates`,
with optional `OfflineSolver` adaptered proposal path only for the positive
control) from a cold offline reset, shall keep the proposer blind to banked
answers, and shall load banked L1 winning prefixes only after proposal
enumeration for executable-oracle classification. It SHALL not inject a
`GameAdapter` into the held-out generic proposer measurement.

The diagnostic SHALL first verify that `arc_solver_kit.offline_arcade()` is
available, that no LLM generator is required unless the configured proposer
actually invokes one, and that at least three games have banked L1 ground truth
present in both `ops/arc_solve_registry.yaml` and the offline environment. A
missing resource SHALL produce a terminal `blocked_*` artifact rather than a
fabricated coverage bucket.

For each measured held-out game the workflow SHALL run the cold live proposer
under a bounded action budget, capture each enumerated candidate pool, and
classify the game into exactly one of `COVERED`, `ENUMERATED_BUT_LOST`, or
`NEVER_ENUMERATED`. `COVERED` means the live run reaches an L1 win within
budget. `ENUMERATED_BUT_LOST` means the banked winning prefix was proposed at
each corresponding prefix state but the run did not reach the win within the
budget. `NEVER_ENUMERATED` means at least one winning-prefix action primitive
was absent from the recorded proposal pools at the corresponding prefix state.

The workflow SHALL include a positive control on an adaptered game such as
`tu93`; the positive control must classify as `COVERED` or the diagnostic is
retired as a harness artifact. A passing diagnostic SHALL report a named
dominant bucket over at least three held-out games and write
`results/experiment_4851_generation_coverage_diagnostic.json`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; a measured decomposition is complete_generation_wall_<dominant_bucket>_dominant (e.g. complete_generation_wall_never_enumerated_dominant)."
- `per_game_coverage`: principle "per-game mapping game -> {bucket in COVERED|ENUMERATED_BUT_LOST|NEVER_ENUMERATED, winning_prefix_len, pool_size, reached_l1_win, budget_actions} -- the quantitative measurement."
- `dominant_bucket`: principle "the bucket the majority of held-out games fall in -- the headline that redirects .448 (never_enumerated -> generation expressibility; enumerated_but_lost -> budget/pruning; covered -> ranking)."
- `positive_control_game`: principle "the adaptered game used as positive control; it MUST be COVERED or the measurement is a harness artifact (a Phase-Prototype positive control)."
- `positive_control_covered`: principle "true iff the positive control came out COVERED -- the load-bearing not-a-harness-artifact check."
- `proposer_blind_to_banked_answer`: principle "true -- the banked winning prefix was NOT injected into the proposer (the tautology trap B1 audits)."
- `n_games_measured`: principle ">=3 held-out games for a non-degenerate distribution."
- `verifier_is_oracle`: principle "true -- the reproduction gate defining the winner is the executable oracle (circularity discipline)."
- `live_path_reachable`: principle "the instrumentation hooks the live proposer (arc_orphan_solver_lint passes) -- a diagnostic the live agent cannot reach is wasted effort."
- `solve_provenance`: principle "development_proxy -- an offline coverage measurement, NOT a live first-win; declared honestly."
- `inference_substrate`: principle "live_llm_inference if the proposer invokes the LLM (60s floor), else verifier_ensemble_against_cached_candidates -- declare what actually ran."
- `preconditions_checked`: principle "records arcade/generator/ground-truth checks; a missing resource emits blocked_, never a fabricated bucket."
- `random_seed`: principle "determinism for the proposer's stochastic search."
- `reproducibility_checksum`: principle "content hash of (games, proposer config, budget) so a replication catches drift."

#### SCENARIO-ARC-WMTE-4851-COVERAGE-BUCKETS

Given banked L1 winning prefixes and a bounded cold `StepwiseExplorer` run
When Experiment 4851 compares recorded proposal pools against the executable
oracle prefixes after the run
Then every measured held-out game is assigned exactly one coverage bucket,
records `winning_prefix_len`, `pool_size`, `reached_l1_win`, and
`budget_actions`, and the artifact reports the majority `dominant_bucket`.

#### SCENARIO-ARC-WMTE-4851-POSITIVE-CONTROL

Given the adaptered `tu93` positive-control path is available
When Experiment 4851 runs the adaptered proposal control without injecting the
banked answer
Then `positive_control_game="tu93"` and `positive_control_covered=true`; if the
control is not covered, the diagnostic emits a retired/non-test verdict rather
than claiming a generation-wall decomposition.

#### SCENARIO-ARC-WMTE-4851-BLOCKED-PRECONDITION

Given the offline arcade is unavailable, generator availability is required but
missing, or fewer than three banked L1 prefixes are present in the registry and
offline environment
When Experiment 4851 starts
Then it writes
`results/experiment_4851_generation_coverage_diagnostic.json` with a terminal
`blocked_*` verdict, empty or partial coverage fields, and explicit
`preconditions_checked` details.

### REQ-ARC-WMTE-4858: Generation Expressibility SOTA Ingestion For .448

Experiment 4858 SHALL run the reserved `.448` SOTA-ingestion slot for methods
that widen what the first-contact proposer can express after Experiment 4851
reported `dominant_bucket="NEVER_ENUMERATED"`. The workflow SHALL read
`results/experiment_4848_sota_ingestion_object_world_model.json`,
`results/experiment_4851_generation_coverage_diagnostic.json`,
`research-studying.md`, and `research-references.md`; it SHALL filter the
corpus to program synthesis, library learning over action primitives,
object-relational MCTS proposers, executable world-model program search, and
neural-guided program search for ARC/grid games. The workflow SHALL use the
reliable channel: `scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`,
low-concurrency WebSearch/WebFetch, and direct arXiv HTTP checks. It SHALL NOT
invoke `/deep-research`, re-ingest the nulled exploration-strategy class, re-open
the concluded energy-stage class, load models, train, submit to a leaderboard,
or make a solve claim.

On a successful ingestion-synthesis run, Experiment 4858 SHALL write
`results/experiment_4858_sota_ingestion_generation_expressibility.json`, update
`research-studying.md` with an idempotent Exp 4858 section marked INGESTED, and
update `research-references.md` with an idempotent Exp 4858 source section. The
artifact SHALL include the required fields `honest_verdict`, `methods_mapped`,
`arxiv_ids_cited`, `aimed_at_dominant_bucket`, `flagged_for_v448`,
`inference_substrate`, `preconditions_checked`, `citations`, `fresh_sweep`,
`upstream_artifacts`, `generation_expressibility_mapping_note`, `note_path`,
`references_path`, `random_seed`, `duration_s`, `reproducibility_checksum`, and
`field_principles`.

The complete-path `honest_verdict` SHALL start with `success_` and equal
`success_sota_ingestion_generation_expressibility_mapped`;
`aimed_at_dominant_bucket` SHALL equal `NEVER_ENUMERATED`;
`inference_substrate` SHALL equal `aggregation_from_upstream_artifacts`; and
`flagged_for_v448` SHALL name the strongest method or methods for the `.448`
planner. The mapping SHALL explain how each method changes the proposer's
action-primitive vocabulary so a missing winning prefix enters the pool before
any ranker scores it. The mapping SHALL consume only partial or noisy object
signals from the Exp 4848 object-world-model handoff, not exact object identity
as a precondition.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; mapping emitted is success_sota_ingestion_generation_expressibility_mapped."
- `methods_mapped`: principle "the strongest 3-5 generation-expressibility methods, each mapped onto putting the winning prefix into the pool, each with a real arXiv ID."
- `arxiv_ids_cited`: principle "every method claim must cite a verifiable arXiv ID (no fabrication -- adversarial_verify bar)."
- `aimed_at_dominant_bucket`: principle "the A1 dominant_bucket the ingestion targets (never_enumerated -> expressibility; covered -> ranking)."
- `flagged_for_v448`: principle "the strongest method(s) flagged so the .448 planner reads the mapping."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts (0.0001s floor)."

The source set SHALL contain only verified arXiv IDs from the focused pass:
`2006.08381`, `2310.19791`, `2411.17708`, `2507.14172`, `2507.15877`,
`2601.06604`, `2605.05138`, and `2606.14418`. The method map SHALL contain
three to five methods, and every method claim SHALL cite only IDs from
`arxiv_ids_cited`. The note SHALL identify DreamCoder, LILO, ARC neurally guided
program induction, SOAR, execution-guided neural program synthesis,
ObjectZero/COMET object-centric MCTS, and executable ARC-AGI-3 world models as
the ingested literature informing `.448` generation expressibility.

#### SCENARIO-ARC-WMTE-4858

Given the research files, Exp 4848 handoff artifact, and Exp 4851 A1 diagnostic
artifact are present, and Exp 4851 reports `dominant_bucket="NEVER_ENUMERATED"`
When Experiment 4858 runs through the reliable channel
Then it writes
`results/experiment_4858_sota_ingestion_generation_expressibility.json`,
records that `/deep-research` was not invoked, maps three to five SOTA methods
onto the generation-expressibility handoff with real arXiv IDs, updates
`research-studying.md` and `research-references.md` idempotently, and flags the
strongest candidate inputs for `.448`.

#### SCENARIO-ARC-WMTE-4858-NO-FABRICATION

Given a candidate artifact omits citations, cites an unverified arXiv ID, targets
ranking despite `NEVER_ENUMERATED`, requires exact object identity instead of
partial/noisy object signals, maps generic exploration or concluded energy-stage
reranking instead of proposer expressibility, uses a stale `.447` flag, or
claims a model-load, training, leaderboard, or solve result
When the Experiment 4858 artifact validator runs
Then it rejects the artifact before the result can be written.

### REQ-ARC-WMTE-4855: Hostile A1 Generation-Coverage Diagnostic Audit

Experiment 4855 SHALL adversarially audit the Exp 4851 A1 generation-coverage
diagnostic before any capstone trusts the dominant-bucket finding. The audit
SHALL read `results/experiment_4851_generation_coverage_diagnostic.json`,
`python/carnot/experiment_4851_generation_coverage_diagnostic.py`,
`scripts/summarize_artifact.py`, and `scripts/adversarial_verify.py`.

The audit SHALL verify four load-bearing gates: the held-out proposer was blind
to the banked winning answer and used banked prefixes only after proposal
enumeration for classification; the positive control is a real adaptered game
and is covered; the per-game coverage buckets support the claimed
`dominant_bucket` over at least three measured games; and the live path remains
reachable while `solve_provenance=development_proxy`. If any gate fails,
`a1_genuinely_diagnostic` SHALL be false and the dominant-bucket finding SHALL
be labeled untrustworthy/non-test. A missing A1 artifact or script SHALL write a
terminal `blocked_a1_artifact_missing` artifact.

Experiment 4855 SHALL run the disciplined artifact summarizer, run
`scripts/adversarial_verify.py` against the A1 artifact, run
`scripts/arc_orphan_solver_lint.py`, write
`results/experiment_4855_generation_diagnostic_audit.json`, and append a stable
section to `ops/arc_null_silent_bug_audit.md`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; audit complete is complete_a1_generation_diagnostic_audited."
- `a1_genuinely_diagnostic`: principle "the load-bearing check -- proposer-blind AND positive-control-covered AND buckets-match-claim AND live-path-reachable; else A1 is a tautology/harness non-test and its dominant-bucket finding is void."
- `proposer_blind_confirmed`: principle "true iff the banked winner was used only to classify, never to seed the proposer (the tautology trap)."
- `positive_control_confirmed`: principle "true iff the positive control really came out COVERED on a real adaptered game."
- `buckets_match_claim`: principle "true iff the per-game buckets support the claimed dominant_bucket."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts (0.0001s floor)."

#### SCENARIO-ARC-WMTE-4855-A1-HOSTILE-AUDIT

Given the Exp 4851 generation-coverage diagnostic artifact and script are
present
When Experiment 4855 audits the proposal path, positive control, bucket
distribution, live lint result, solve provenance, summarizer result, and
adversarial verifier result
Then it writes `results/experiment_4855_generation_diagnostic_audit.json`,
appends a stable section to `ops/arc_null_silent_bug_audit.md`, records
`inference_substrate=aggregation_from_upstream_artifacts`, and sets
`a1_genuinely_diagnostic=true` only when all four gates pass.

#### SCENARIO-ARC-WMTE-4855-NON-TEST-CLASSIFICATION

Given an Exp 4851-like artifact or script that injects the banked answer into
the proposer, lacks a covered adaptered positive control, has per-game buckets
that contradict the claimed dominant bucket, measures fewer than three games,
has an unreachable live path, or declares any provenance other than
`development_proxy`
When Experiment 4855 audits it
Then the audit records the corresponding failed check, emits a terminal
`complete_a1_generation_diagnostic_non_test_*` verdict, and sets
`a1_genuinely_diagnostic=false`.

### REQ-ARC-WMTE-4861: Generation Wall Induce-Plan Fork Probe

Experiment 4861 SHALL diagnose the Exp 4851 `NEVER_ENUMERATED` L1
first-contact wall as a Phase-Prototype+Validation fork. The workflow SHALL use
the fixed held-out game list `cd82`, `cn04`, `ls20`, `m0r0`, `r11l`, `sk48`,
`sp80`, `su15`, and `wa30`, plus the `tu93` positive control. The held-out
planner SHALL be blind to banked winning prefixes; banked prefixes may be read
only after planning to classify the planned candidate pool against executable
oracle ground truth.

The probe SHALL first check, before measuring any game, that
`arc_solver_kit.offline_arcade()` is available, the Qwen3.5-9B-MTP local GGUF
generator is available on the non-CUDA iGPU path (never a 3090 fallback), and at
least three Exp 4851 `NEVER_ENUMERATED` games plus `tu93` have registry and
offline-environment ground truth. Missing resources SHALL produce terminal
`blocked_offline_arcade_missing`, `blocked_generator_unavailable`, or
`blocked_no_heldout_games` artifacts rather than a fabricated fork verdict.

For each measured held-out game the workflow SHALL collect cold-start
transitions without a `GameAdapter`, invoke the live
`arc_competition_agent.E3AgentPolicy._induce_and_plan` path, load the induced
engine through `arc_executable_world_model.load_engine`, plan through
`arc_executable_world_model.plan_in_model`, and then classify the planned
candidate pool as `COVERED`, `ENUMERATED_BUT_LOST`, or `NEVER_ENUMERATED`.
The artifact SHALL also report held-out off-path transition accuracy for the
induced engine. Each per-game row SHALL be checkpointed atomically, and a soft
elapsed budget stop SHALL write a schema-valid `partial=true` artifact with
checkpoint evidence and exit successfully.

The positive control `tu93` SHALL come out high-accuracy and `COVERED`; if it
does not, the probe is retired as a harness artifact. Otherwise the joint fork
verdict SHALL be `GUIDANCE_WALL` when median held-out engine accuracy is high
and at least one held-out game migrates from Exp 4851 `NEVER_ENUMERATED` to
planned `COVERED`, `PLANNER_GAP` when median accuracy is high and no game
migrates, and `INDUCER_CEILING` when accuracy is low and no game migrates.
Experiment 4861 SHALL write
`results/experiment_4861_generation_wall_fork_probe.json`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; a measured fork is complete_generation_wall_<fork>_<detail> (e.g. complete_generation_wall_inducer_ceiling_low_accuracy_no_migration)."
- `fork_verdict`: principle "one of GUIDANCE_WALL | PLANNER_GAP | INDUCER_CEILING -- the headline that redirects .449."
- `per_game_fork`: principle "per-game mapping game -> {engine_heldout_accuracy, planned_bucket in COVERED|ENUMERATED_BUT_LOST|NEVER_ENUMERATED, migrated (bare NEVER->planned COVERED), winning_prefix_len} -- the quantitative joint measurement."
- `coverage_migration_count`: principle "how many NEVER_ENUMERATED games migrated to COVERED under induce->plan -- the GUIDANCE signal."
- `median_engine_heldout_accuracy`: principle "the induced world-model quality across games -- distinguishes INDUCER ceiling (low) from planner gap (high)."
- `positive_control_game`: principle "tu93 -- MUST be HIGH accuracy + COVERED or the measurement is a harness artifact."
- `positive_control_migrated`: principle "true iff tu93 came out HIGH accuracy + COVERED -- the load-bearing not-a-harness-artifact check."
- `planner_blind_to_banked_answer`: principle "true -- the banked winning prefix was NOT injected into induction or planning (the tautology trap B1 audits)."
- `n_games_measured`: principle ">=3 NEVER_ENUMERATED held-out games for a non-degenerate joint table."
- `verifier_is_oracle`: principle "true -- the reproduction gate defining the winner is the executable oracle (circularity discipline)."
- `live_path_reachable`: principle "the probe calls the live e3.load_engine/plan_in_model path (arc_orphan_solver_lint passes) -- a diagnostic the live agent cannot reach is wasted effort."
- `solve_provenance`: principle "development_proxy -- an offline fork measurement, NOT a live first-win; declared honestly."
- `checkpoint_emitted`: principle "a capped run still emits a usable partial (the 2026-06-25 wall-clock fix); per-game checkpointing."
- `inference_substrate`: principle "live_llm_inference (60s floor) -- induce->plan invokes the LLM."
- `preconditions_checked`: principle "records arcade/generator/held-out-games checks; a missing resource emits blocked_, never a fabricated fork."
- `random_seed`: principle "determinism for induction + planning stochastic search."
- `reproducibility_checksum`: principle "content hash of (games, induce/plan config, budget) so a replication catches drift."

#### SCENARIO-ARC-WMTE-4861-BLOCKED-PRECONDITION

Given the offline arcade is unavailable, the iGPU Qwen3.5-9B-MTP generator is
unavailable, or fewer than three Exp 4851 `NEVER_ENUMERATED` held-out games plus
`tu93` are present
When Experiment 4861 starts
Then it writes
`results/experiment_4861_generation_wall_fork_probe.json` with a terminal
`blocked_*` verdict, empty or partial fork rows, explicit
`preconditions_checked`, `planner_blind_to_banked_answer=true`, and no
fabricated migration count.

#### SCENARIO-ARC-WMTE-4861-JOINT-FORK

Given a covered `tu93` positive control and at least three measured
`NEVER_ENUMERATED` held-out games
When Experiment 4861 combines held-out engine accuracy with planned-bucket
migration
Then it emits one of `GUIDANCE_WALL`, `PLANNER_GAP`, or `INDUCER_CEILING`,
records `coverage_migration_count`, `median_engine_heldout_accuracy`, and
`per_game_fork`, and derives the terminal `honest_verdict` from that joint fork.

#### SCENARIO-ARC-WMTE-4861-PARTIAL-CHECKPOINT

Given the soft elapsed budget is reached after at least one game checkpoint
When Experiment 4861 stops the current run
Then it writes a schema-valid artifact with `partial=true`,
`checkpoint_emitted=true`, the measured per-game rows preserved, and exit code
0 so the next run can resume from the checkpoints.

### REQ-ARC-WMTE-4871: GPU-Fixed Generation Wall Induce-Plan Fork Probe

Experiment 4871 SHALL rerun the Exp 4861 generation-wall induce-plan fork with
the 2026-06-27 offline-induction GPU allocation fixed. The workflow SHALL keep
the Exp 4861 live path, game list, banked-prefix blindness, held-out
world-model accuracy measurement, checkpointing, positive control, and joint
fork verdict rules, but SHALL replace the old iGPU-only precondition with an
offline generator precondition that accepts either the iGPU HIP llama-server or
the conductor-owned GPU-0 CUDA llama-server selected by
`CARNOT_ARC_GENERATOR_CUDA_GPU=0` when the CUDA card has the configured free
VRAM headroom. The precondition SHALL health-check the selected generator via
`LocalGGUFProposer._ensure_server()` and SHALL NOT reject a healthy GPU-0 CUDA
server merely because `CUDA_VISIBLE_DEVICES` is set.

The probe SHALL first check, before measuring any game, that
`arc_solver_kit.offline_arcade()` is available, the Qwen3.5-9B-MTP local GGUF
generator is healthy on `gpu0_cuda` or `igpu_hip`, and at least three Exp 4851
`NEVER_ENUMERATED` games plus `tu93` have registry and offline-environment
ground truth. Missing resources SHALL produce terminal
`blocked_offline_arcade_missing`, `blocked_generator_unavailable`, or
`blocked_no_heldout_games` artifacts rather than a fabricated fork verdict.

Experiment 4871 SHALL write
`results/experiment_4871_generation_wall_fork_probe_gpu_fixed.json` with the
same bare top-level fields as Exp 4861 plus `generator_backend` and
`model_specs`. A passing measurement SHALL require `generator_backend` to be
`gpu0_cuda` or `igpu_hip`, `positive_control_migrated=true` for the `tu93`
positive control, at least three measured held-out games, and one of
`GUIDANCE_WALL`, `PLANNER_GAP`, or `INDUCER_CEILING` as the named joint fork
verdict. If the run stops on the elapsed budget after a game checkpoint, the
artifact SHALL be schema-valid with `partial=true` and exit code 0 so the next
run can resume.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; a measured fork is complete_generation_wall_<fork>_<detail> (e.g. complete_generation_wall_inducer_ceiling_low_accuracy_no_migration)."
- `fork_verdict`: principle "one of GUIDANCE_WALL | PLANNER_GAP | INDUCER_CEILING -- the headline that redirects .450."
- `median_engine_heldout_accuracy`: principle "EMIT BARE (not a {value,principle} dict) -- A1b's gated_on reads this raw value. The induced world-model quality across games; distinguishes INDUCER ceiling (low) from planner gap (high)."
- `n_games_measured`: principle "EMIT BARE -- B1 reads it. >=3 NEVER_ENUMERATED held-out games for a non-degenerate joint table."
- `per_game_fork`: principle "per-game mapping game -> {engine_heldout_accuracy, planned_bucket in COVERED|ENUMERATED_BUT_LOST|NEVER_ENUMERATED, migrated, winning_prefix_len} -- the quantitative joint measurement."
- `coverage_migration_count`: principle "how many NEVER_ENUMERATED games migrated to COVERED under induce->plan -- the GUIDANCE signal."
- `positive_control_game`: principle "tu93 -- MUST be HIGH accuracy + COVERED or the measurement is a harness artifact."
- `positive_control_migrated`: principle "true iff tu93 came out HIGH accuracy + COVERED -- the load-bearing not-a-harness-artifact check."
- `generator_backend`: principle "which server actually served (gpu0_cuda | igpu_hip) -- proves the GPU fix worked and the .448 cuda_3090_generator_disallowed veto is gone."
- `planner_blind_to_banked_answer`: principle "true -- the banked winning prefix was NOT injected into induction or planning (the tautology trap B1 audits)."
- `verifier_is_oracle`: principle "true -- the reproduction gate defining the winner is the executable oracle (circularity discipline)."
- `live_path_reachable`: principle "the probe calls the live e3.load_engine/plan_in_model path (arc_orphan_solver_lint passes)."
- `solve_provenance`: principle "development_proxy -- an offline fork measurement, NOT a live first-win; declared honestly."
- `checkpoint_emitted`: principle "a capped run still emits a usable partial (the 2026-06-25 wall-clock fix); per-game checkpointing."
- `inference_substrate`: principle "live_llm_inference (60s floor) -- induce->plan invokes the LLM on the GPU-0 generator."
- `preconditions_checked`: principle "records arcade/generator/held-out-games checks; a missing resource emits blocked_, never a fabricated fork."
- `model_specs`: principle "names the actual generator invoked (Qwen3.5-9B-MTP via the GPU-0 CUDA llama-server) -- methodology for adversarial_verify."
- `random_seed`: principle "determinism for induction + planning stochastic search."
- `reproducibility_checksum`: principle "content hash of (games, induce/plan config, budget) so a replication catches drift."

#### SCENARIO-ARC-WMTE-4871-GPU-FIXED-PRECONDITION

Given `_generator_server_and_env()` selects the conductor-owned CUDA
llama-server with `CUDA_VISIBLE_DEVICES=0`
When Experiment 4871 checks generator availability
Then it health-checks the server, records `generator_backend="gpu0_cuda"`, and
does not emit `cuda_3090_generator_disallowed`.

#### SCENARIO-ARC-WMTE-4871-JOINT-FORK

Given a covered `tu93` positive control and at least three measured
`NEVER_ENUMERATED` held-out games
When Experiment 4871 combines held-out engine accuracy with planned-bucket
migration
Then it emits one of `GUIDANCE_WALL`, `PLANNER_GAP`, or `INDUCER_CEILING`,
records `generator_backend`, `model_specs`, `coverage_migration_count`,
`median_engine_heldout_accuracy`, and `per_game_fork`, and derives the
terminal `honest_verdict` from that joint fork.

#### SCENARIO-ARC-WMTE-4871-PARTIAL-CHECKPOINT

Given the soft elapsed budget is reached after at least one game checkpoint
When Experiment 4871 stops the current run
Then it writes a schema-valid artifact with `partial=true`,
`checkpoint_emitted=true`, the measured per-game rows preserved, and exit code
0 so the next run can resume from the checkpoints.

### REQ-ARC-WMTE-4872: CEGIS World-Model Refinement Accuracy Lift

Experiment 4872 SHALL run only after the Exp 4871 A1 artifact exists and reports
`median_engine_heldout_accuracy < 0.5`. It SHALL reuse the Exp 4871 GPU-fixed
generator precondition, held-out game list, live executable world-model path, and
banked-answer blindness, but SHALL evaluate a bounded counterexample-guided
repair loop over the induced executable `world_model.py` program rather than a
parallel solver. If the offline arcade, generator, live path, or A1 low-accuracy
baseline is unavailable, the workflow SHALL write a terminal `blocked_*`
artifact instead of fabricating a CEGIS verdict.

For each measured A1 held-out game, the workflow SHALL induce or load the
initial executable engine, collect real off-path transition mispredictions, split
those failures into repair counterexamples and a disjoint remeasurement split,
and invoke the live Qwen3.5-9B-MTP proposer to repair the executable program.
A repair SHALL be accepted only when it fixes at least one repair
counterexample and does not regress observed-prefix replay. The loop SHALL be
bounded, checkpointed per game and round, and SHALL stop on the soft elapsed
budget with a schema-valid partial artifact.

The final score SHALL compare the A1 baseline held-out accuracy with the refined
engine accuracy on a split disjoint from the counterexamples used for repair.
Experiment 4872 SHALL write
`results/experiment_4872_cegis_world_model_refinement.json` with the median
per-game held-out accuracy delta, a bootstrap CI95 for that delta, the per-game
accuracy table, `delta_on_truly_heldout_split=true`, `verifier_is_oracle=false`,
`live_path_reachable=true` only when the E3 induction/repair import path passes
orphan-solver lint, and `solve_provenance=development_proxy`.

The falsifiable gate SHALL pass only when the median delta is positive, the CI95
excludes 0, at least three games are measured, the split is truly disjoint, and
the positive control records at least one known-correctable misprediction fixed.
If the CI95 includes 0, the artifact SHALL retire this CEGIS lever with a
terminal `complete_cegis_no_heldout_accuracy_lift_residual_*` verdict.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; a real lift is success_cegis_engine_accuracy_lift_<delta>; a null is complete_cegis_no_heldout_accuracy_lift_residual_<cause>."
- `cegis_heldout_accuracy_delta_median`: principle "median (refined - baseline) held-out transition accuracy -- did CEGIS MOVE the inducer wall?"
- `cegis_heldout_accuracy_delta_ci95`: principle "bootstrap CI95 of the delta; PASS requires it to exclude 0 (not a noise lift)."
- `per_game_accuracy_delta`: principle "per-game {baseline, refined, delta, counterexamples_fixed, cegis_rounds} -- the quantitative intervention table."
- `delta_on_truly_heldout_split`: principle "true -- the re-measure split is DISJOINT from the repair counterexamples (B1 audits; else it is a tautology)."
- `positive_control_passed`: principle "a known-correctable misprediction was fixed -> a flat null is a real ceiling, not a harness no-op."
- `verifier_is_oracle`: principle "false -- the held-out transition score is oracle-distinct from the env's level-up check (circularity discipline)."
- `live_path_reachable`: principle "the refinement improves the live e3 induction/repair path (arc_orphan_solver_lint passes), not a parallel solver."
- `solve_provenance`: principle "development_proxy -- an inducer-accuracy measurement, NOT a banked level."
- `checkpoint_emitted`: principle "a capped run still emits a usable partial (per-game + per-round checkpointing)."
- `inference_substrate`: principle "live_llm_inference (60s floor) -- CEGIS repair invokes the LLM on the GPU-0 generator."
- `model_specs`: principle "names the actual generator (Qwen3.5-9B-MTP via the GPU-0 CUDA llama-server)."
- `preconditions_checked`: principle "records arcade/generator/A1-baseline checks; a missing resource emits blocked_."
- `random_seed`: principle "determinism for the CEGIS stochastic repair search."
- `reproducibility_checksum`: principle "content hash of (games, A1 baseline, CEGIS config, held-out split) so a replication catches drift."

#### SCENARIO-ARC-WMTE-4872-A1-LOW-ACCURACY-GATE

Given the Exp 4871 artifact is missing, malformed, or has
`median_engine_heldout_accuracy >= 0.5`
When Experiment 4872 starts
Then it writes a terminal `blocked_a1_baseline_missing` or
`blocked_a1_not_inducer_ceiling` artifact and does not invoke CEGIS repair.

#### SCENARIO-ARC-WMTE-4872-REPAIR-ACCEPTANCE

Given a real held-out transition misprediction and observed-prefix replay rows
When a proposed executable repair is scored
Then the repair is accepted only if it fixes at least one repair
counterexample, preserves observed-prefix replay accuracy, and records the
accepted round in the per-game checkpoint.

#### SCENARIO-ARC-WMTE-4872-TRULY-HELDOUT-DELTA

Given repair counterexamples and a disjoint remeasurement split
When Experiment 4872 builds the final artifact
Then `delta_on_truly_heldout_split=true`, the per-game deltas are computed on
the remeasurement split rather than the repair examples, and the median delta
CI95 determines the terminal success or retirement verdict.

### REQ-ARC-WMTE-4876: A1 GPU-Fixed Fork Probe And A1b CEGIS Adversarial Audit

Experiment 4876 SHALL adversarially audit the `.449` A1 fork probe artifact
`results/experiment_4871_generation_wall_fork_probe_gpu_fixed.json`, its source
`python/carnot/experiment_4871_generation_wall_fork_probe_gpu_fixed.py`, and
the optional A1b CEGIS refinement artifact
`results/experiment_4872_cegis_world_model_refinement.json` before any capstone
trusts the fork verdict or inducer delta. If the A1 artifact or source is
missing, the workflow SHALL write
`results/experiment_4876_fork_probe_inducer_audit.json` with
`honest_verdict=blocked_a1_artifact_missing`, preserve the required fields with
false gates, append the audit report, and stop without fabricating A1/A1b
passes.

The A1 audit SHALL run `scripts/summarize_artifact.py` and
`scripts/adversarial_verify.py` on the A1 artifact, inspect the source
induce-plan path, and run `scripts/arc_orphan_solver_lint.py`. The falsifiable
gate `a1_genuinely_diagnostic` SHALL be true iff A1 ran live on an accepted
GPU-0 CUDA or iGPU HIP generator, cleared the 60-second live-inference duration
floor, was not flagged adversarial, kept induction/planning blind to the banked
winner until post-planning classification, passed the real `tu93` positive
control with high accuracy and `COVERED`, had per-game accuracy/bucket numbers
supporting the claimed fork verdict with `n_games_measured>=3`, and remained
live-path-reachable.

If the A1b artifact ran, Experiment 4876 SHALL run
`scripts/adversarial_verify.py` on it, verify every repair counterexample split
is disjoint from its remeasurement split, verify the reported median delta and
CI95 match the per-game deltas, record whether its positive control passed,
confirm `verifier_is_oracle=false`, and confirm the live E3 repair path is
reachable. The falsifiable gate `a1b_delta_trustworthy` SHALL be true iff A1b's
delta is on a truly held-out disjoint split and oracle-distinct, or iff A1b was
gate-skipped because A1 was high-accuracy or blocked. A1b positive-control
failure SHALL be recorded as residual evidence even when the held-out/oracle
mechanics are trustworthy.

Experiment 4876 SHALL append an idempotent section to
`ops/arc_null_silent_bug_audit.md` and write
`results/experiment_4876_fork_probe_inducer_audit.json`. The artifact SHALL
include the required principle-annotated fields:

- `honest_verdict`: principle "terminal prefix; audit complete is complete_a1_a1b_audited."
- `a1_genuinely_diagnostic`: principle "the load-bearing check -- ran-live-on-GPU0 AND planner-blind AND positive-control-migrated AND numbers-match-fork AND live-path-reachable; else A1's fork verdict is void."
- `a1_ran_live_on_gpu0`: principle "true iff generator_backend is gpu0_cuda/igpu_hip AND duration clears the 60s floor AND not flagged_adversarial -- proves the .448 cuda_3090_generator_disallowed block is gone."
- `planner_blind_confirmed`: principle "true iff the banked winner was used only to classify, never to seed induction/planning."
- `positive_control_confirmed`: principle "true iff tu93 really came out HIGH accuracy + COVERED."
- `numbers_match_fork`: principle "true iff the per-game accuracy x bucket numbers support the claimed fork_verdict."
- `a1b_delta_trustworthy`: principle "true iff A1b's accuracy delta is on a held-out split DISJOINT from the repair set and oracle-distinct (or A1b was gate-skipped)."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts (0.0001s floor)."

#### SCENARIO-ARC-WMTE-4876-A1-A1B-AUDIT

Given the Exp 4871 artifact and source are present, and the Exp 4872 artifact
is present or was gate-skipped
When Experiment 4876 audits A1 and A1b
Then it writes `results/experiment_4876_fork_probe_inducer_audit.json`, appends
the audit report, sets `a1_genuinely_diagnostic=true` only when every A1
load-bearing check passes, sets `a1b_delta_trustworthy=true` only for a
disjoint held-out oracle-distinct A1b delta or an explicit gate skip, and
records failed positive-control or numeric checks as reasons that void the
corresponding A1 fork verdict or A1b lift.

#### SCENARIO-ARC-WMTE-4876-BLOCKED-A1-ARTIFACT

Given the Exp 4871 A1 artifact or source is missing
When Experiment 4876 starts
Then it writes a terminal `blocked_a1_artifact_missing` artifact with the
required principle fields, false diagnostic gates, precondition evidence, and no
fabricated A1b trust result.

### REQ-ARC-WMTE-4865: Hostile A1 Induce-Plan Fork Probe Audit

Experiment 4865 SHALL adversarially audit
`results/experiment_4861_generation_wall_fork_probe.json` and
`python/carnot/experiment_4861_generation_wall_fork_probe.py` before any
capstone trusts the A1 fork verdict. If either input is missing, it SHALL write
`results/experiment_4865_fork_probe_audit.json` with
`honest_verdict=blocked_a1_artifact_missing` and no fabricated pass checks.

The audit SHALL run `scripts/summarize_artifact.py` and
`scripts/adversarial_verify.py` on the A1 artifact, inspect the Exp 4861
induce-plan source path, and append an idempotent section to
`ops/arc_null_silent_bug_audit.md`. The falsifiable gate
`a1_genuinely_diagnostic` SHALL be true iff all four load-bearing checks pass:
planner blindness to the banked answer, a real `tu93` positive control with
high held-out accuracy and `COVERED`, per-game accuracy/bucket numbers that
support `fork_verdict` with `n_games_measured>=3`, and live-path reachability
plus `solve_provenance=development_proxy`. A blocked or retired Exp 4861
artifact SHALL therefore be reported as a non-test rather than a trusted fork.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; audit complete is complete_a1_fork_probe_audited."
- `a1_genuinely_diagnostic`: principle "the load-bearing check -- planner-blind AND positive-control-migrated AND numbers-match-fork AND live-path-reachable; else A1 is a tautology/harness non-test and its fork verdict is void."
- `planner_blind_confirmed`: principle "true iff the banked winner was used only to classify, never to seed induction/planning (the tautology trap)."
- `positive_control_confirmed`: principle "true iff tu93 really came out HIGH accuracy + COVERED."
- `numbers_match_fork`: principle "true iff the per-game accuracy x bucket numbers support the claimed fork_verdict."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts (0.0001s floor)."

#### SCENARIO-ARC-WMTE-4865-A1-FORK-AUDIT

Given the Exp 4861 fork-probe artifact and script are present
When Experiment 4865 audits the source path, positive control, numeric fork
table, live-path reachability, and verifier outputs
Then it writes `results/experiment_4865_fork_probe_audit.json`, appends the
audit report, and sets `a1_genuinely_diagnostic=true` only when every
load-bearing check passes.

#### SCENARIO-ARC-WMTE-4865-NON-TEST-CLASSIFICATION

Given an Exp 4861-like artifact or script that injects the banked answer before
classification, lacks a passing `tu93` positive control, has fork verdicts that
do not match its per-game numbers, measures fewer than three games, lacks the
live induce-plan path, or declares non-development solve provenance
When Experiment 4865 audits it
Then the audit records the corresponding failed check, emits a terminal
`complete_a1_fork_probe_non_test_*` verdict, and sets
`a1_genuinely_diagnostic=false`.

### REQ-ARC-WMTE-4868: V449 Frontier SOTA Ingestion For The Generation-Wall Fork

Experiment 4868 SHALL run the reserved `.449` SOTA-ingestion slot aimed at the
Exp 4861 generation-wall fork verdict. The workflow SHALL read
`results/experiment_4861_generation_wall_fork_probe.json`,
`results/experiment_4858_sota_ingestion_generation_expressibility.json`,
`research-studying.md`, `research-references.md`, `scripts/sweep_clusters.py`,
and `scripts/sweep_semscholar.py`. It SHALL use the reliable channel:
`scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`, low-concurrency
WebSearch/WebFetch, and direct arXiv HTTP checks. It SHALL NOT invoke
`/deep-research`, re-ingest nulled macro-vocabulary/click-heatmap coverage,
exploration-strategy, or energy classes, load models, train, submit to a
leaderboard, modify `scripts/research_conductor.py`, or make a solve claim.

If Exp 4861 supplies `fork_verdict=INDUCER_CEILING`, the mapping SHALL target
world-model inducer quality: stronger small/local coding-model inducers,
test-time dynamics/world-model adaptation, counterexample-guided refinement, and
the sovereignty/decentralization tension between the SWE-strong cloud
Family-B-style executable-world-model inducer and a local open inducer. If Exp
4861 supplies `GUIDANCE_WALL` or `PLANNER_GAP`, the mapping SHALL instead target
neural-guided planning, MCTS, and search-verifier loops over the executable
world model. If Exp 4861 is blocked and its `fork_verdict` is null, the workflow
SHALL record that fact honestly and may follow the operator-reserved likely
`INDUCER_CEILING` branch only as `requested_fallback_reason`, without claiming
that Exp 4861 measured the fork.

On a successful ingestion-synthesis run, Experiment 4868 SHALL write
`results/experiment_4868_sota_ingestion_v449_frontier.json`, update
`research-studying.md` with an idempotent Exp 4868 section marked INGESTED, and
update `research-references.md` with an idempotent Exp 4868 source section. The
artifact SHALL include the required fields `honest_verdict`, `methods_mapped`,
`arxiv_ids_cited`, `aimed_at_fork_verdict`, `flagged_for_v449`,
`inference_substrate`, `preconditions_checked`, `citations`, `fresh_sweep`,
`upstream_artifacts`, `sota_to_experiment_mapping_note`, `note_path`,
`references_path`, `random_seed`, `duration_s`, `reproducibility_checksum`, and
`field_principles`.

The complete-path `honest_verdict` SHALL equal
`success_sota_ingestion_v449_frontier_mapped`; `inference_substrate` SHALL equal
`aggregation_from_upstream_artifacts`; `methods_mapped` SHALL contain three to
five methods; and every method claim SHALL cite only verified arXiv IDs from
`arxiv_ids_cited`. The source set SHALL contain only direct arXiv HTTP-200
checks from the focused pass: `2203.13474`, `2502.07786`, `2506.02918`,
`2507.03160`, `2507.15877`, `2509.03956`, `2605.05138`, and `2606.11521`.
For the reserved INDUCER_CEILING branch, the strongest methods SHALL include
the executable-world-model inducer quality ladder, test-time world-model/dynamics
adaptation, counterexample-guided refinement, and local open-code-model
induction.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; mapping emitted is success_sota_ingestion_v449_frontier_mapped."
- `methods_mapped`: principle "the strongest 3-5 methods aimed at A1's fork verdict (inducer-quality OR guided-planning), each with a real arXiv ID."
- `arxiv_ids_cited`: principle "every method claim must cite a verifiable arXiv ID (no fabrication -- adversarial_verify bar)."
- `aimed_at_fork_verdict`: principle "the A1 fork_verdict the ingestion targets (INDUCER_CEILING -> inducer quality; GUIDANCE/PLANNER -> planning/search)."
- `flagged_for_v449`: principle "the strongest method(s) flagged so the .449 planner reads the mapping."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts (0.0001s floor)."

#### SCENARIO-ARC-WMTE-4868-V449-FRONTIER-MAPPED

Given the research files, Exp 4861 fork-probe artifact, and Exp 4858 handoff
artifact are present
When Experiment 4868 runs through the reliable channel
Then it writes `results/experiment_4868_sota_ingestion_v449_frontier.json`,
records that `/deep-research` was not invoked, maps three to five SOTA methods
onto the `.449` frontier branch with real arXiv IDs, updates
`research-studying.md` and `research-references.md` idempotently, records any
blocked/null Exp 4861 fork verdict honestly, and flags the strongest candidate
inputs for `.449`.

#### SCENARIO-ARC-WMTE-4868-NO-FABRICATION

Given a candidate artifact omits citations, cites an unverified arXiv ID,
targets a fork verdict inconsistent with its branch, uses a stale `.448` flag,
re-ingests retired coverage/exploration/energy classes, invokes
`/deep-research`, claims a model-load, training, leaderboard, or solve result,
or modifies `scripts/research_conductor.py`
When the Experiment 4868 artifact validator runs
Then it rejects the artifact before the result can be written.

### REQ-ARC-WMTE-4879: V450 Frontier SOTA Ingestion Follows The Measured A1/A1b Fork

Experiment 4879 SHALL run the reserved `.450` SOTA-ingestion slot after reading
the measured A1 fork probe and A1b CEGIS-refinement artifacts:
`results/experiment_4871_generation_wall_fork_probe_gpu_fixed.json`,
`results/experiment_4872_cegis_world_model_refinement.json`, and the `.448` to
`.449` handoff at `results/experiment_4868_sota_ingestion_v449_frontier.json`.
It SHALL also read `research-studying.md`, `research-references.md`,
`scripts/sweep_clusters.py`, and `scripts/sweep_semscholar.py`. The workflow
SHALL use only the reliable channel: `scripts/sweep_clusters.py`,
`scripts/sweep_semscholar.py`, low-concurrency WebSearch/WebFetch, and direct
arXiv HTTP checks. It SHALL NOT invoke `/deep-research`, SHALL NOT re-ingest
retired macro-vocabulary/click-heatmap coverage, exploration-strategy, or energy
classes, SHALL NOT load a model, train, submit to a leaderboard, make a solve
claim, or modify `scripts/research_conductor.py`, `ops/changelog.md`,
`ops/status.md`, or `_bmad/traceability.md`.

The workflow SHALL preserve the actual A1/A1b facts in the artifact. If A1
reports `fork_verdict=INDUCER_CEILING` and A1b reports a held-out CEGIS accuracy
delta whose CI95 excludes zero, the mapping SHALL target scaling the CEGIS and
engine-refinement loop. If A1 reports `fork_verdict=INDUCER_CEILING` and A1b
reports a null held-out CEGIS delta, the mapping SHALL target the next inducer
candidate: test-time dynamics/world-model adaptation first, then a stronger
local-open-code inducer or a Family-B reference-versus-local inducer A/B. If A1
reports `GUIDANCE_WALL` or `PLANNER_GAP` with high engine accuracy, the mapping
SHALL target neural-guided planning, MCTS, and search-verifier loops over the
accurate executable world model. If the checked-in A1 `fork_verdict` is null
because the positive control failed, the workflow SHALL record the null verdict
honestly and may follow the computed inducer-ceiling residual branch only while
explicitly preserving the positive-control caveat.

On a successful ingestion-synthesis run, Experiment 4879 SHALL write
`results/experiment_4879_sota_ingestion_v450_frontier.json`, update
`research-studying.md` with an idempotent Exp 4879 INGESTED section, and update
`research-references.md` with an idempotent Exp 4879 source section. The
artifact SHALL include the required user-facing fields `honest_verdict`,
`methods_mapped`, `arxiv_ids_cited`, `aimed_at_fork_verdict`,
`flagged_for_v450`, and `inference_substrate`, plus provenance fields sufficient
to audit the reliable sweep and upstream artifacts.

The complete-path `honest_verdict` SHALL equal
`success_sota_ingestion_v450_frontier_mapped`; `inference_substrate` SHALL equal
`aggregation_from_upstream_artifacts`; `methods_mapped` SHALL contain three to
five methods; every method claim SHALL cite only verified arXiv IDs from
`arxiv_ids_cited`; and `duration_s` SHALL use the aggregation-only `0.0001`
floor. For the checked-in A1/A1b residual branch, the source set SHALL include
direct arXiv HTTP-200 checks for `2203.13474`, `2506.02918`, `2507.03160`,
`2507.15877`, `2509.03956`, `2605.05138`, `2606.25421`, and `2606.26217`.
The strongest `.450` flags SHALL prioritize test-time dynamics adaptation and
the Family-B reference-versus-local open-code inducer A/B; the current CEGIS
loop SHALL be recorded as nulled rather than promoted.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; mapping emitted is success_sota_ingestion_v450_frontier_mapped."
- `methods_mapped`: principle "the strongest 3-5 methods aimed at A1's ACTUAL fork verdict + A1b's CEGIS result, each with a real arXiv ID."
- `arxiv_ids_cited`: principle "every method claim must cite a verifiable arXiv ID (no fabrication -- adversarial_verify bar)."
- `aimed_at_fork_verdict`: principle "the A1 fork_verdict the ingestion targets (INDUCER_CEILING -> inducer/CEGIS scale; GUIDANCE/PLANNER -> planning/search)."
- `flagged_for_v450`: principle "the strongest method(s) flagged so the .450 planner reads the mapping."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts (0.0001s floor)."

#### SCENARIO-ARC-WMTE-4879-V450-FRONTIER-MAPPED

Given the research files, Exp 4871 A1 artifact, Exp 4872 A1b artifact, and Exp
4868 handoff artifact are present
When Experiment 4879 runs through the reliable channel
Then it writes `results/experiment_4879_sota_ingestion_v450_frontier.json`,
records that `/deep-research` was not invoked, maps three to five SOTA methods
onto the `.450` frontier with real arXiv IDs, records the actual A1 null
positive-control caveat and A1b zero CEGIS delta, updates `research-studying.md`
and `research-references.md` idempotently, and flags the strongest `.450`
candidate inputs.

#### SCENARIO-ARC-WMTE-4879-NO-FABRICATION

Given a candidate artifact omits citations, cites an unverified arXiv ID,
targets a fork verdict inconsistent with A1/A1b, promotes the nulled CEGIS loop
as the strongest `.450` flag, uses a stale `.449` flag, re-ingests retired
coverage/exploration/energy classes, invokes `/deep-research`, claims a
model-load, training, leaderboard, or solve result, or modifies
`scripts/research_conductor.py`
When the Experiment 4879 artifact validator runs
Then it rejects the artifact before the result can be written.

### REQ-ARC-WMTE-4882: TTT Dynamics Value-Gap Fork Probe

Experiment 4882 SHALL rerun the generation-wall fork on a corrigendum-corrected
graded metric. The workflow SHALL first check the offline ARC arcade, then the
GPU-fixed generator precondition from Experiment 4871 accepting either GPU-0
CUDA or iGPU HIP, and then the held-out NEVER_ENUMERATED games
`cd82`, `cn04`, `ls20`, `m0r0`, `r11l`, `sk48`, `sp80`, `su15`, and `wa30`
plus the `tu93` positive control. Missing preconditions SHALL emit
`blocked_offline_arcade_missing`, `blocked_generator_unavailable`, or
`blocked_no_heldout_games` artifacts rather than a fabricated fork.

For each measured game, Experiment 4882 SHALL measure held-out
`cell_recall` as the fraction of actually changed cells whose location the
engine predicts as changed, and `changed_cell_value_accuracy` as the fraction
of cells that were both actually changed and predicted changed whose predicted
value equals the true next grid. The baseline SHALL use the warm-started live
dynamics path. The intervention SHALL fit a compact test-time dynamics adapter
from the agent's own cold-start transitions and SHALL remeasure value accuracy
on a disjoint held-out split. The banked winning answer SHALL be used only as
classification ground truth after planning, not as induction, adaptation, or
planning input. The positive control `tu93` SHALL be non-degenerate
(`cell_recall > 0`) or the probe family SHALL retire with
`complete_ttt_dynamics_positive_control_degenerate_retired`.

Experiment 4882 SHALL write
`results/experiment_4882_ttt_dynamics_value_gap.json` with the required fields
`honest_verdict`, `fork_verdict`,
`tta_changed_cell_value_accuracy_delta_median`,
`tta_value_accuracy_delta_ci95`, `per_game_value_gap`,
`engine_cell_recall_median`, `coverage_migration_count`,
`positive_control_game`, `positive_control_non_degenerate`,
`delta_on_truly_heldout_split`, `planner_blind_to_banked_answer`,
`verifier_is_oracle`, `live_path_reachable`, `generator_backend`,
`solve_provenance`, `checkpoint_emitted`, `inference_substrate`,
`preconditions_checked`, `model_specs`, `random_seed`, and
`reproducibility_checksum`.

If the value-accuracy delta CI95 excludes zero and at least one
NEVER_ENUMERATED game migrates to `COVERED`, the artifact SHALL report
`fork_verdict=INDUCER_CEILING_BEATABLE`. If the CI95 excludes zero and no game
migrates, it SHALL report `fork_verdict=PLANNER_GAP`. If the CI95 includes zero,
it SHALL report `fork_verdict=INDUCER_CEILING_HARD`. The median value-accuracy
delta SHALL be emitted as a bare numeric
`tta_changed_cell_value_accuracy_delta_median`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; a real value lift is success_ttt_dynamics_value_gap_closed_<delta>; a null is complete_ttt_dynamics_no_value_lift_<fork>; a degenerate control is complete_ttt_dynamics_positive_control_degenerate_retired."
- `fork_verdict`: principle "one of INDUCER_CEILING_BEATABLE | PLANNER_GAP | INDUCER_CEILING_HARD -- the headline that redirects .451."
- `tta_changed_cell_value_accuracy_delta_median`: principle "EMIT BARE (not a {value,principle} dict) -- A1b's gated_on reads this raw value. Median (adapted - baseline) changed-cell VALUE accuracy across games; did TTA close the corrigendum value gap?"
- `tta_value_accuracy_delta_ci95`: principle "bootstrap CI95 of the value-accuracy delta; PASS requires it to exclude 0 for a real lift."
- `per_game_value_gap`: principle "per-game {cell_recall_baseline, value_acc_baseline, value_acc_adapted, value_delta, planned_bucket in COVERED|ENUMERATED_BUT_LOST|NEVER_ENUMERATED, migrated} -- the quantitative table."
- `engine_cell_recall_median`: principle "the corrigendum change-LOCATION floor; proves the graded metric is non-degenerate (where exact-match was 0)."
- `coverage_migration_count`: principle "how many NEVER_ENUMERATED games migrated to COVERED under the adapted engine + plan_in_model."
- `positive_control_game`: principle "tu93 -- MUST be non-degenerate (cell_recall > 0) on the graded metric or the measurement is a harness artifact."
- `positive_control_non_degenerate`: principle "true iff tu93 came out with HIGH cell_recall -- the load-bearing fix for the .449 degenerate-metric failure."
- `delta_on_truly_heldout_split`: principle "true -- the adapted re-measure split is DISJOINT from the TTA adapter's fit transitions (B1 audits; else tautology)."
- `planner_blind_to_banked_answer`: principle "true -- the banked winning prefix was NOT injected into induction, adaptation, or planning."
- `verifier_is_oracle`: principle "false -- the held-out transition accuracy is oracle-distinct from the env's level-up check (circularity discipline)."
- `live_path_reachable`: principle "the adaptation improves the live e3.load_engine/plan_in_model path (arc_orphan_solver_lint passes)."
- `generator_backend`: principle "which server served (gpu0_cuda | igpu_hip) -- proves the GPU fix and a genuine live run."
- `solve_provenance`: principle "development_proxy -- an offline inducer-accuracy measurement, NOT a live first-win; declared honestly."
- `checkpoint_emitted`: principle "a capped run still emits a usable partial (the 2026-06-25 wall-clock fix); per-game checkpointing."
- `inference_substrate`: principle "live_llm_inference (60s floor) -- induce/adapt/plan invokes the LLM on the GPU-0 generator."
- `preconditions_checked`: principle "records arcade/generator/held-out-games checks; a missing resource emits blocked_, never a fabricated fork."
- `model_specs`: principle "names the actual generator invoked (Qwen3.5-9B-MTP via the GPU-0 CUDA llama-server) -- methodology for adversarial_verify."
- `random_seed`: principle "determinism for induction + adaptation + planning stochastic search."
- `reproducibility_checksum`: principle "content hash of (games, induce/adapt/plan config, held-out split, budget) so a replication catches drift."

#### SCENARIO-ARC-WMTE-4882-GRADED-METRIC

Given held-out off-path transitions and an engine prediction
When Experiment 4882 scores the engine
Then it reports change-location `cell_recall` and changed-cell
`value_accuracy` without using exact-full-grid-match as the primary metric.

#### SCENARIO-ARC-WMTE-4882-DISJOINT-TTA-DELTA

Given baseline and adapted dynamics measurements
When Experiment 4882 builds the value-gap table
Then every adapter fit transition id is disjoint from every adapted held-out
remeasure transition id, and the bare median changed-cell value-accuracy delta
plus bootstrap CI95 drive the fork verdict.

#### SCENARIO-ARC-WMTE-4882-PARTIAL-CHECKPOINT

Given the run reaches its elapsed budget after at least one game
When Experiment 4882 stops the current process
Then it writes per-game checkpoints and a schema-valid partial artifact so the
next run can resume without losing the primary value-accuracy measurement.

### REQ-ARC-WMTE-4883: Inducer-Ceiling Reference Versus Local A/B

Experiment 4883 SHALL run only when Experiment 4882 produced the low-value-delta
regime: `results/experiment_4882_ttt_dynamics_value_gap.json` exists,
`tta_changed_cell_value_accuracy_delta_median < 0.1`, and the A1 artifact
contains per-game baselines plus the held-out split. If the offline arcade,
GPU-fixed generator, or low-delta A1 baseline preconditions fail, Experiment
4883 SHALL emit `blocked_offline_arcade_missing`,
`blocked_generator_unavailable`, or `blocked_a1_baseline_missing` rather than a
fabricated A/B result.

For the same A1 NEVER_ENUMERATED held-out games and the same A1 held-out split,
Experiment 4883 SHALL measure two induction lanes through the live
`arc_executable_world_model.load_engine` interface: a Family-B reference
executable-world-model inducer lane and the local open-code
Qwen3.5-9B-MTP lane. Both lanes SHALL score held-out off-path transitions with
the Experiment 4882 graded metric (`cell_recall` and
`changed_cell_value_accuracy`), SHALL report per-lane per-game deltas against
A1's `value_acc_baseline`, and SHALL use per-game/per-lane checkpointing so a
capped run still emits a usable partial. The Family-B lane is a ceiling
measurement only; the deployment path remains the local Qwen3.5-9B-MTP lane.

Experiment 4883 SHALL write
`results/experiment_4883_inducer_ceiling_ab.json` with the required fields
`honest_verdict`, `inducer_ceiling_attribution`,
`reference_lane_value_accuracy_delta`, `local_lane_value_accuracy_delta`,
`per_lane_per_game`, `delta_on_truly_heldout_split`, `verifier_is_oracle`,
`live_path_reachable`, `reference_lane_is_ceiling_only`,
`solve_provenance`, `checkpoint_emitted`, `inference_substrate`,
`model_specs`, `preconditions_checked`, `random_seed`, and
`reproducibility_checksum`.

If the reference lane raises held-out changed-cell value accuracy above A1's
baseline while the local lane does not, the attribution SHALL be
`LOCAL_MODEL_IS_CEILING`. If neither lane raises value accuracy above A1's
baseline and both lane delta CI95 intervals include zero, the attribution SHALL
be `METHOD_IS_CEILING` and the artifact SHALL set `retire_if_same_verdict=true`.
If the local lane also raises value accuracy, the attribution SHALL be
`LOCAL_ALREADY_SUFFICIENT`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; a real lane lift is success_inducer_ceiling_<attribution>; a flat null is complete_inducer_ceiling_neither_lane_lifts_method_is_ceiling."
- `inducer_ceiling_attribution`: principle "one of LOCAL_MODEL_IS_CEILING | METHOD_IS_CEILING | LOCAL_ALREADY_SUFFICIENT -- redirects .451."
- `reference_lane_value_accuracy_delta`: principle "Family-B reference lane changed-cell value-accuracy delta vs A1 baseline (the capability ceiling)."
- `local_lane_value_accuracy_delta`: principle "local open-code lane delta vs A1 baseline (the deployment lane -- what actually ships)."
- `per_lane_per_game`: principle "per-lane per-game {value_acc, cell_recall, delta_vs_baseline, ci95} -- the quantitative A/B table."
- `delta_on_truly_heldout_split`: principle "true -- both lanes scored on the SAME held-out split as A1, disjoint from any fit set (B1 audits)."
- `verifier_is_oracle`: principle "false -- the held-out transition score is oracle-distinct from the env's level-up check (circularity discipline)."
- `live_path_reachable`: principle "both lanes use the live e3 load_engine interface (arc_orphan_solver_lint passes), not a parallel solver."
- `reference_lane_is_ceiling_only`: principle "true -- the Family-B reference lane is a ceiling measurement; the deployment lane stays local (decentralization Rule 2)."
- `solve_provenance`: principle "development_proxy -- an inducer-ceiling measurement, NOT a banked level."
- `checkpoint_emitted`: principle "a capped run still emits a usable partial (per-game + per-lane checkpointing)."
- `inference_substrate`: principle "live_llm_inference (60s floor) -- both induction lanes invoke an LLM."
- `model_specs`: principle "names both inducers (Family-B reference + local Qwen3.5-9B-MTP via the GPU-0 CUDA llama-server)."
- `preconditions_checked`: principle "records arcade/generator/A1-baseline checks; a missing resource emits blocked_."
- `random_seed`: principle "determinism for both induction lanes' stochastic search."
- `reproducibility_checksum`: principle "content hash of (games, A1 baseline, both lane configs, held-out split) so a replication catches drift."

#### SCENARIO-ARC-WMTE-4883-A1-LOW-VALUE-GATE

Given the offline arcade and GPU-fixed generator preconditions
When Experiment 4883 reads the A1 value-gap artifact
Then it blocks if the artifact is missing, malformed, lacks per-game baselines
or held-out split ids, or has
`tta_changed_cell_value_accuracy_delta_median >= 0.1`, and otherwise proceeds
using A1's same held-out games and split.

#### SCENARIO-ARC-WMTE-4883-SAME-SPLIT-AB

Given reference and local lane inducers
When Experiment 4883 measures each lane on a game
Then both lanes emit an executable engine, load it through
`arc_executable_world_model.load_engine`, score the same held-out transition
ids with the graded metric, and report per-lane per-game value accuracy,
cell recall, delta versus A1 baseline, and CI95.

#### SCENARIO-ARC-WMTE-4883-ATTRIBUTION

Given at least three completed games with both lanes measured
When Experiment 4883 aggregates lane deltas
Then it emits one of `LOCAL_MODEL_IS_CEILING`, `METHOD_IS_CEILING`, or
`LOCAL_ALREADY_SUFFICIENT`, keeps `verifier_is_oracle=false`, marks the
reference lane as ceiling-only, and retires the inducer direction when neither
lane's delta CI95 excludes zero.

### REQ-ARC-WMTE-4887: Value-Gap And Inducer-Ceiling Adversarial Audit

Experiment 4887 SHALL audit the Experiment 4882 A1 TTT dynamics value-gap
probe and, when present, the Experiment 4883 A1b inducer-ceiling A/B before any
capstone trusts the fork verdict or inducer attribution. If the A1 artifact or
source script is missing, Experiment 4887 SHALL write
`results/experiment_4887_value_gap_inducer_audit.json` with
`honest_verdict=blocked_a1_artifact_missing` and SHALL not fabricate any A1 or
A1b pass.

The audit SHALL run `scripts/summarize_artifact.py` and
`scripts/adversarial_verify.py` on the A1 artifact, SHALL run
`scripts/adversarial_verify.py` on A1b when A1b exists, SHALL inspect the A1
source path so the banked winning answer is used only after planning for
classification, and SHALL require `arc_orphan_solver_lint` to pass before
declaring the live path reachable. A1 is genuinely diagnostic only when the
graded `tu93` positive control is non-degenerate (`cell_recall > 0`), the
reported value delta is remeasured on a split disjoint from the TTA adapter fit
set, the planner is blind to the banked answer, per-game
`cell_recall`/value-accuracy/delta numbers support the reported
`fork_verdict`, the live path is reachable, and the A1 run used GPU-0 CUDA or
iGPU HIP for at least the live-model duration floor without an adversarial
flag. A1b is trustworthy only when it was gate-skipped because A1 did not enter
the low-value inducer-ceiling branch, or when both lanes scored the same A1
held-out games and split, remained oracle-distinct, used a reachable live path,
and labeled the reference lane as ceiling-only.

Experiment 4887 SHALL write
`results/experiment_4887_value_gap_inducer_audit.json` with required top-level
fields `honest_verdict`, `a1_genuinely_diagnostic`,
`a1_positive_control_non_degenerate_confirmed`,
`a1_delta_on_heldout_disjoint_confirmed`, `planner_blind_confirmed`,
`numbers_match_fork`, `a1b_ab_trustworthy`, `inference_substrate`,
`checks`, `a1_failure_reasons`, `a1b_failure_reasons`,
`a1_summarizer_result`, `a1_adversarial_result`,
`a1b_adversarial_result`, `live_lint_result`, `preconditions_checked`,
`field_principles`, `duration_s`, and `reproducibility_checksum`. The workflow
SHALL append an idempotent Experiment 4887 section to
`ops/arc_null_silent_bug_audit.md`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; audit complete is complete_a1_a1b_audited."
- `a1_genuinely_diagnostic`: principle "the load-bearing check -- non-degenerate-graded-control AND held-out-disjoint-delta AND planner-blind AND numbers-match-fork AND live-path-reachable AND ran-live-on-GPU0; else A1's fork verdict is void."
- `a1_positive_control_non_degenerate_confirmed`: principle "true iff the GRADED tu93 positive control really had cell_recall > 0 -- proves the .449 degenerate-metric failure is FIXED."
- `a1_delta_on_heldout_disjoint_confirmed`: principle "true iff A1's value-accuracy delta was re-measured on a split DISJOINT from the TTA adapter's fit set (not a tautology)."
- `planner_blind_confirmed`: principle "true iff the banked winner was used only to classify, never to seed induction/adaptation/planning."
- `numbers_match_fork`: principle "true iff the per-game (cell_recall, value-accuracy, delta) numbers support the claimed fork_verdict."
- `a1b_ab_trustworthy`: principle "true iff both A1b lanes scored the SAME held-out split as A1 and oracle-distinct (or A1b was gate-skipped)."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts (0.0001s floor)."

#### SCENARIO-ARC-WMTE-4887-A1-A1B-AUDIT

Given checked-in A1 and optional A1b artifacts plus the A1 source script
When Experiment 4887 runs the hostile audit
Then it writes a schema-valid `complete_a1_a1b_audited` artifact whose A1 and
A1b trust booleans are true only when all load-bearing checks pass, and whose
failure reasons make any degenerate positive control, TTA tautology,
banked-answer injection, fork-number mismatch, unreachable live path,
wrong/too-short substrate, circular verifier, or unfair A1b split explicit.

#### SCENARIO-ARC-WMTE-4887-BLOCKED-A1-ARTIFACT

Given the A1 artifact or A1 source script is missing
When Experiment 4887 runs
Then it writes `blocked_a1_artifact_missing`, marks every trust boolean false,
records the missing precondition, and does not claim A1 or A1b audit success.

### REQ-ARC-WMTE-4890: V451 Frontier SOTA Ingestion Follows The Actual Value-Gap Fork

Experiment 4890 SHALL run the reserved `.451` SOTA-ingestion slot after reading
the actual A1 value-gap fork artifact
`results/experiment_4882_ttt_dynamics_value_gap.json`, the optional A1b
inducer-ceiling attribution artifact
`results/experiment_4883_inducer_ceiling_ab.json`, and the `.450` handoff
`results/experiment_4879_sota_ingestion_v450_frontier.json`. It SHALL also
read `research-studying.md`, `research-references.md`,
`scripts/sweep_clusters.py`, and `scripts/sweep_semscholar.py`. If the A1
artifact is missing, Experiment 4890 SHALL write a blocked deliverable with
`honest_verdict=blocked_a1_artifact_missing` and SHALL NOT fabricate a mapping.

The workflow SHALL use only the reliable channel:
`scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`, low-concurrency
WebSearch/WebFetch, and direct arXiv HTTP checks. It SHALL NOT invoke
`/deep-research`, SHALL NOT re-ingest the retired energy-as-ARC-lever,
coverage/vocabulary, exploration-strategy, selection/ranking, or
perception-from-grid classes, SHALL NOT load a model, train, submit to a
leaderboard, make a solve claim, or modify `scripts/research_conductor.py`,
`ops/changelog.md`, `ops/status.md`, or `_bmad/traceability.md`.

The mapping SHALL branch on A1's actual `fork_verdict` and A1b's
`inducer_ceiling_attribution`. If A1 reports `INDUCER_CEILING_BEATABLE`, the
mapping SHALL target test-time-adaptation scaling plus first-win conversion. If
A1 reports `PLANNER_GAP`, the mapping SHALL target neural-guided planning,
MCTS, and search-verifier loops over the accurate engine. If A1 reports
`INDUCER_CEILING_HARD` and A1b reports `LOCAL_MODEL_IS_CEILING`, the mapping
SHALL target stronger local open-code inducers. If A1 reports
`INDUCER_CEILING_HARD` and A1b reports `METHOD_IS_CEILING`, the mapping SHALL
target alternative world-model representations beyond executable code,
including decision-oriented or agent-authored targets and action-prefix latent
adapters. For the checked-in Exp 4882/4883 branch, the required target is
`INDUCER_CEILING_HARD` with `METHOD_IS_CEILING`.

On a successful ingestion-synthesis run, Experiment 4890 SHALL write
`results/experiment_4890_sota_ingestion_v451_frontier.json`, update
`research-studying.md` with an idempotent Exp 4890 INGESTED section, and update
`research-references.md` with an idempotent Exp 4890 source section. The
artifact SHALL include the required user-facing fields `honest_verdict`,
`aimed_at_fork_verdict`, `methods_mapped`, `arxiv_ids_cited`,
`flagged_for_v451`, `banned_channels_excluded`, `inference_substrate`, and
`preconditions_checked`, plus provenance fields sufficient to audit the
reliable sweep and upstream artifacts.

The complete-path `honest_verdict` SHALL equal
`success_sota_ingestion_v451_frontier_mapped`; `inference_substrate` SHALL
equal `aggregation_from_upstream_artifacts`; `methods_mapped` SHALL contain
three to five methods; every method claim SHALL cite only verified arXiv IDs
from `arxiv_ids_cited`; each method SHALL include `experiment_graft`,
`validation_gate`, and `fails_when`; `banned_channels_excluded` SHALL be true;
and `duration_s` SHALL use the aggregation-only `0.0001` floor. For the
checked-in branch, the source set SHALL include direct arXiv HTTP-200 checks
for `2503.18938`, `2505.08073`, `2602.23997`, `2603.19312`, `2606.25421`, and
`2606.26217`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; mapping emitted is success_sota_ingestion_v451_frontier_mapped."
- `aimed_at_fork_verdict`: principle "the A1 fork verdict the ingestion targets (BEATABLE->scale TTA+convert; PLANNER_GAP->planning/search; CEILING_HARD->stronger local inducer / alt representation)."
- `methods_mapped`: principle "the strongest 3-5 methods, each with a real arXiv ID + experiment_graft + validation_gate + fails_when."
- `arxiv_ids_cited`: principle "every method claim cites a verifiable HTTP-200 arXiv ID (no fabrication -- adversarial_verify bar)."
- `flagged_for_v451`: principle "the strongest method(s) flagged so the .451 planner reads the mapping (discover->ingest->plan->experiment)."
- `banned_channels_excluded`: principle "true -- /deep-research NOT invoked; energy/coverage/exploration/selection classes NOT re-ingested."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts (0.0001s floor)."
- `preconditions_checked`: principle "records reliable-channel checks + the A1 fork-verdict read; banned channels explicitly excluded."

#### SCENARIO-ARC-WMTE-4890-V451-FRONTIER-MAPPED

Given the research files, Exp 4882 A1 artifact, Exp 4883 A1b artifact, and Exp
4879 handoff artifact are present
When Experiment 4890 runs through the reliable channel
Then it writes `results/experiment_4890_sota_ingestion_v451_frontier.json`,
records that `/deep-research` was not invoked, maps three to five alternative
world-model representation methods onto the `.451` frontier with real HTTP-200
arXiv IDs, records the actual A1 `INDUCER_CEILING_HARD` verdict and A1b
`METHOD_IS_CEILING` attribution, updates `research-studying.md` and
`research-references.md` idempotently, and flags the strongest `.451`
candidate inputs.

#### SCENARIO-ARC-WMTE-4890-BLOCKED-A1

Given the Exp 4882 A1 artifact is missing
When Experiment 4890 checks preconditions
Then it writes `blocked_a1_artifact_missing`, records the missing A1
precondition, keeps `methods_mapped=[]`, keeps `banned_channels_excluded=true`,
and does not fabricate an A1 fork verdict.

#### SCENARIO-ARC-WMTE-4890-NO-FABRICATION

Given a candidate artifact omits citations, cites an unverified arXiv ID,
targets a fork verdict inconsistent with A1/A1b, maps a retired
energy/coverage/exploration/selection/perception class, invokes
`/deep-research`, claims a model-load, training, leaderboard, or solve result,
or modifies `scripts/research_conductor.py`
When the Experiment 4890 artifact validator runs
Then it rejects the artifact before the result can be written.

### REQ-ARC-WMTE-4892: Decision-Need Target Value-Gap Fork Probe

Experiment 4892 SHALL test the `.451` decision-need representation branch
against the exact Experiment 4882 A1 graded value-gap setup. The workflow SHALL
first check `arc_solver_kit.offline_arcade()`, the GPU-fixed Qwen3.5-9B-MTP
generator precondition accepting either GPU-0 CUDA or iGPU HIP without rejecting
ambient `CUDA_VISIBLE_DEVICES`, and
`results/experiment_4882_ttt_dynamics_value_gap.json`. Missing preconditions
SHALL emit `blocked_offline_arcade_missing`,
`blocked_generator_unavailable`, or `blocked_a1_baseline_missing` rather than a
fabricated fork.

For the same A1 held-out games `cd82`, `cn04`, `ls20`, `m0r0`, `r11l`, `sk48`,
`sp80`, `su15`, and `wa30`, and the `tu93` positive control, Experiment 4892
SHALL read the Exp 4882 executable-code baseline `cell_recall` and
`changed_cell_value_accuracy`, author a non-code decision-need target table
from the agent's own cold-start transitions, and score that table on the same
deterministic held-out transition ids used by A1. The target table SHALL encode
decision-relevant dynamics facts such as per-action changed-cell values,
click-relative action effects, object/cell persistence, and hidden-register
deltas as data rows rather than generated executable code. The representation
fit ids SHALL be disjoint from the held-out score ids, and the banked winning
answer SHALL be used only after planning to classify coverage migration.

Experiment 4892 SHALL write
`results/experiment_4892_decision_need_targets_value_gap.json` with required
fields `honest_verdict`, `fork_verdict`,
`decision_need_value_accuracy_delta_median`,
`decision_need_value_accuracy_delta_ci95`, `per_game_value_gap`,
`engine_cell_recall_median`, `coverage_migration_count`,
`positive_control_game`, `positive_control_non_degenerate`,
`delta_on_truly_heldout_split`, `planner_blind_to_banked_answer`,
`verifier_is_oracle`, `live_path_reachable`, `generator_backend`,
`solve_provenance`, `checkpoint_emitted`, `inference_substrate`,
`model_specs`, `random_seed`, and `reproducibility_checksum`. The median
decision-need value-accuracy delta SHALL be emitted as a bare numeric field.

If the value-accuracy delta CI95 excludes zero and at least one
NEVER_ENUMERATED game migrates to `COVERED`, the artifact SHALL report
`fork_verdict=REPRESENTATION_UNLOCKS_VALUE`. If the CI95 excludes zero and no
game migrates, it SHALL report `fork_verdict=PLANNER_GAP`. If the CI95 includes
zero, it SHALL report `fork_verdict=VALUE_GAP_REPRESENTATION_INVARIANT`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; a real value lift is success_decision_need_value_gap_closed_<delta>; a null is complete_decision_need_no_value_lift_<fork>; a degenerate control is complete_decision_need_positive_control_degenerate_retired."
- `fork_verdict`: principle "one of REPRESENTATION_UNLOCKS_VALUE | PLANNER_GAP | VALUE_GAP_REPRESENTATION_INVARIANT -- the headline that redirects .452."
- `decision_need_value_accuracy_delta_median`: principle "EMIT BARE (not a {value,principle} dict) -- A1b's gated_on reads this raw value. Median (decision-need - code-engine baseline) changed-cell VALUE accuracy across games; did the non-code representation close the value gap?"
- `decision_need_value_accuracy_delta_ci95`: principle "bootstrap CI95 of the value-accuracy delta; PASS requires it to exclude 0 for a real lift."
- `per_game_value_gap`: principle "per-game {cell_recall, value_acc_code_baseline, value_acc_decision_need, value_delta, planned_bucket in COVERED|ENUMERATED_BUT_LOST|NEVER_ENUMERATED, migrated} -- the quantitative table."
- `engine_cell_recall_median`: principle "the corrigendum change-LOCATION floor; proves the graded metric is non-degenerate (where exact-match was 0)."
- `coverage_migration_count`: principle "how many NEVER_ENUMERATED games migrated to COVERED under the decision-need representation + plan_in_model."
- `positive_control_game`: principle "tu93 -- MUST be non-degenerate (cell_recall > 0) on the graded metric or the measurement is a harness artifact."
- `positive_control_non_degenerate`: principle "true iff tu93 came out with HIGH cell_recall -- carries forward the .450 degenerate-metric fix."
- `delta_on_truly_heldout_split`: principle "true -- the representation is scored on a split DISJOINT from the transitions used to author the targets (B1 audits; else tautology)."
- `planner_blind_to_banked_answer`: principle "true -- the banked winning prefix was NOT injected into authoring, fitting, or planning."
- `verifier_is_oracle`: principle "false -- the held-out transition accuracy is oracle-distinct from the env's level-up check (circularity discipline)."
- `live_path_reachable`: principle "the representation improves the live e3.load_engine/plan_in_model path (arc_orphan_solver_lint passes)."
- `generator_backend`: principle "which server served (gpu0_cuda | igpu_hip) -- proves the GPU fix and a genuine live run."
- `solve_provenance`: principle "development_proxy -- an offline inducer-accuracy measurement, NOT a live first-win; declared honestly."
- `checkpoint_emitted`: principle "a capped run still emits a usable partial (the 2026-06-25 wall-clock fix); per-game checkpointing."
- `inference_substrate`: principle "live_llm_inference (60s floor) -- authoring/fitting/planning invokes the LLM on the GPU-0 generator."
- `model_specs`: principle "names the actual generator invoked (Qwen3.5-9B-MTP via the GPU-0 CUDA llama-server) -- methodology for adversarial_verify."
- `random_seed`: principle "determinism for target authoring + fitting + planning stochastic search."
- `reproducibility_checksum`: principle "content hash of (games, authoring/fit/plan config, held-out split, budget) so a replication catches drift."

#### SCENARIO-ARC-WMTE-4892-DECISION-NEED-TABLE

Given the agent's own cold-start transitions
When Experiment 4892 authors decision-need targets
Then it produces a non-code target table whose executable prediction wrapper
can improve changed-cell value accuracy without using held-out transitions or
the banked answer as supervision.

#### SCENARIO-ARC-WMTE-4892-SAME-SPLIT-DELTA

Given the Exp 4882 A1 baseline artifact and held-out split
When Experiment 4892 builds the value-gap table
Then every target authoring transition id is disjoint from every held-out
score transition id, and the bare median changed-cell value-accuracy delta
plus bootstrap CI95 drive the fork verdict.

#### SCENARIO-ARC-WMTE-4892-FORK-VERDICT

Given at least three NEVER_ENUMERATED games and a non-degenerate `tu93`
positive control
When Experiment 4892 aggregates value deltas and migration classifications
Then it emits one of `REPRESENTATION_UNLOCKS_VALUE`, `PLANNER_GAP`, or
`VALUE_GAP_REPRESENTATION_INVARIANT`, keeps `verifier_is_oracle=false`, and
sets `retire_if_same_verdict=true` when the delta CI95 includes zero.

#### SCENARIO-ARC-WMTE-4892-PARTIAL-CHECKPOINT

Given the run reaches its elapsed budget after measuring at least one game
When Experiment 4892 stops the current process
Then it writes per-game checkpoints and a schema-valid partial artifact so the
next run can resume without losing the primary value-accuracy measurement.

### REQ-ARC-WMTE-4893: Action-Prefix Latent Adapter Value-Gap Fork Probe

Experiment 4893 SHALL test the `.451` action-prefix latent representation
branch only when Experiment 4892 produced the low-value regime:
`results/experiment_4892_decision_need_targets_value_gap.json` exists,
contains `decision_need_value_accuracy_delta_median < 0.1`, and exposes
per-game code-engine baselines plus the held-out transition split. The workflow
SHALL first check `arc_solver_kit.offline_arcade()`, the GPU-fixed
Qwen3.5-9B-MTP generator precondition accepting either GPU-0 CUDA or iGPU HIP
without rejecting ambient `CUDA_VISIBLE_DEVICES`, and the Exp 4892 artifact.
Missing or gated-off preconditions SHALL emit a `blocked_` or `skipped_`
artifact rather than a fabricated fork.

For the same Exp 4892 held-out games and `tu93` positive control, Experiment
4893 SHALL induce a non-code action-prefix latent adapter through the live E3
interface (`arc_executable_world_model.load_engine`) and score the exact
held-out transition ids with the same graded metric shape: `cell_recall` and
`changed_cell_value_accuracy`. The latent adapter SHALL encode observed action
prefixes into compact verifier-checkable future-state deltas, decode only
accepted deltas into engine facts, keep adapter fit ids disjoint from held-out
score ids, and keep the banked winning answer out of induction, fitting,
scoring, and planning.

Experiment 4893 SHALL write
`results/experiment_4893_action_prefix_latent_adapter.json` with required
fields `honest_verdict`, `fork_verdict`,
`action_prefix_value_accuracy_delta_median`,
`action_prefix_value_accuracy_delta_ci95`, `per_game_value_gap`,
`ran_genuinely_live`, `delta_on_truly_heldout_split`, `verifier_is_oracle`,
`live_path_reachable`, `generator_backend`, `solve_provenance`,
`checkpoint_emitted`, `inference_substrate`, `model_specs`,
`preconditions_checked`, `random_seed`, and `reproducibility_checksum`. The
median action-prefix value-accuracy delta SHALL be emitted as a bare numeric
field.

If the value-accuracy delta CI95 excludes zero, the artifact SHALL report
`fork_verdict=REPRESENTATION_MATTERS`. If the CI95 includes zero, it SHALL
report `fork_verdict=VALUE_GAP_REPRESENTATION_INVARIANT_HARD` and
`retire_if_same_verdict=true`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; a real lift is success_action_prefix_latent_value_gap_closed_<delta>; a flat null is complete_action_prefix_latent_no_value_lift_representation_invariant_hard."
- `fork_verdict`: principle "one of REPRESENTATION_MATTERS | VALUE_GAP_REPRESENTATION_INVARIANT_HARD -- redirects .452."
- `action_prefix_value_accuracy_delta_median`: principle "median (latent-representation - A1 code-engine baseline) changed-cell value accuracy; did the latent representation close the value gap?"
- `action_prefix_value_accuracy_delta_ci95`: principle "bootstrap CI95 of the latent-representation value-accuracy delta; PASS requires it to exclude 0 for a real lift."
- `per_game_value_gap`: principle "per-game {value_acc_code_baseline, value_acc_latent, cell_recall, delta, ci95} -- the quantitative table."
- `ran_genuinely_live`: principle "true iff duration_s > 60 and the latent induction/scoring genuinely ran (the explicit .450 A1b 13.7s non-test fix)."
- `delta_on_truly_heldout_split`: principle "true -- scored on the SAME held-out split as A1, disjoint from any fit set (B1 audits)."
- `verifier_is_oracle`: principle "false -- the held-out transition score is oracle-distinct from the env's level-up check (circularity discipline)."
- `live_path_reachable`: principle "the latent adapter uses the live e3 load_engine interface (arc_orphan_solver_lint passes), not a parallel solver."
- `generator_backend`: principle "which server served (gpu0_cuda | igpu_hip) -- proves the GPU fix and a genuine live run."
- `solve_provenance`: principle "development_proxy -- a representation-accuracy measurement, NOT a banked level."
- `checkpoint_emitted`: principle "a capped run still emits a usable partial (per-game checkpointing)."
- `inference_substrate`: principle "live_llm_inference (60s floor) -- the latent induction invokes an LLM on the GPU-0 generator."
- `model_specs`: principle "names the inducer (Qwen3.5-9B-MTP via the GPU-0 CUDA llama-server) -- methodology for adversarial_verify."
- `preconditions_checked`: principle "records arcade/generator/A1-baseline checks; a missing resource emits blocked_."
- `random_seed`: principle "determinism for the latent-adapter induction stochastic search."
- `reproducibility_checksum`: principle "content hash of (games, A1 baseline, latent-adapter config, held-out split) so a replication catches drift."

#### SCENARIO-ARC-WMTE-4893-A1-LOW-VALUE-GATE

Given the Exp 4892 decision-need artifact
When Experiment 4893 checks preconditions
Then it proceeds only when the bare decision-need value delta is below `0.1`
and the artifact contains per-game code baselines plus held-out transition ids;
otherwise it writes a gated `blocked_` or `skipped_` artifact.

#### SCENARIO-ARC-WMTE-4893-ACTION-PREFIX-LATENT

Given the agent's own cold-start transitions
When Experiment 4893 induces the action-prefix latent adapter
Then it creates non-code latent future-state deltas keyed by compact action
prefixes, decodes accepted deltas into verifier-checkable engine facts, and
scores the same held-out split without using banked answers.

#### SCENARIO-ARC-WMTE-4893-GENUINELY-LIVE

Given at least three held-out games are available
When Experiment 4893 measures the latent adapter
Then a passing artifact records `ran_genuinely_live=true`, `duration_s > 60`,
`inference_substrate=live_llm_inference`, and the live path includes
`arc_executable_world_model.load_engine`.

#### SCENARIO-ARC-WMTE-4893-FORK-VERDICT

Given the per-game latent value-accuracy deltas against the Exp 4892 code
baseline
When Experiment 4893 computes the median delta and bootstrap CI95
Then it emits `REPRESENTATION_MATTERS` only when the CI95 excludes zero, and
otherwise emits `VALUE_GAP_REPRESENTATION_INVARIANT_HARD`.

### REQ-ARC-WMTE-4900: V452 Frontier SOTA Ingestion Follows The Representation Fork

Experiment 4900 SHALL run the `.452` SOTA-ingestion slot after reading the
actual A1 decision-need representation fork artifact
`results/experiment_4892_decision_need_targets_value_gap.json`, the optional
A1b action-prefix latent representation artifact
`results/experiment_4893_action_prefix_latent_adapter.json`, and the `.451`
handoff `results/experiment_4890_sota_ingestion_v451_frontier.json`. It SHALL
also read `research-studying.md`, `research-references.md`,
`scripts/sweep_clusters.py`, and `scripts/sweep_semscholar.py`. If the A1
artifact is missing, Experiment 4900 SHALL write a blocked deliverable with
`honest_verdict=blocked_a1_artifact_missing`, `methods_mapped=[]`,
`banned_channels_excluded=true`, and SHALL NOT fabricate a mapping.

The workflow SHALL use only the reliable channel:
`scripts/sweep_clusters.py`, `scripts/sweep_semscholar.py`, low-concurrency
WebSearch/WebFetch, and direct arXiv HTTP checks. It SHALL NOT invoke
`/deep-research`, SHALL NOT re-ingest the retired energy-as-ARC-lever,
TTA-on-code-engine, stronger-local-code-inducer, coverage/vocabulary,
exploration-strategy, selection/ranking, or perception-from-grid classes,
SHALL NOT load a model, train, submit to a leaderboard, make a solve claim, or
modify `scripts/research_conductor.py`, `ops/changelog.md`, `ops/status.md`,
or `_bmad/traceability.md`.

The mapping SHALL branch on A1's actual `fork_verdict` and A1b's
`fork_verdict` when A1b ran. If A1 reports `REPRESENTATION_UNLOCKS_VALUE` or
A1b reports `REPRESENTATION_MATTERS`, the mapping SHALL target first-win
conversion by turning the value-accurate representation into banked levels via
neural-guided `plan_in_model`, MCTS, or search over the value-accurate engine.
If A1 reports `PLANNER_GAP`, the mapping SHALL target neural-guided planning,
MCTS, and search-verifier loops over the value-accurate representation. If A1
reports `VALUE_GAP_REPRESENTATION_INVARIANT` and A1b reports
`VALUE_GAP_REPRESENTATION_INVARIANT_HARD`, the mapping SHALL target a third
representation class or an explicit operator-escalation note that the
change-VALUE gap is a deep representation-invariant ceiling under the deadline.
For the checked-in Exp 4892/4893 branch, the required target is
`VALUE_GAP_REPRESENTATION_INVARIANT` plus
`VALUE_GAP_REPRESENTATION_INVARIANT_HARD`.

On a successful ingestion-synthesis run, Experiment 4900 SHALL write
`results/experiment_4900_sota_ingestion_v452_frontier.json`, update
`research-studying.md` with an idempotent Exp 4900 INGESTED section, and update
`research-references.md` with an idempotent Exp 4900 source section. The
artifact SHALL include the required user-facing fields `honest_verdict`,
`aimed_at_fork_verdict`, `methods_mapped`, `arxiv_ids_cited`,
`flagged_for_v452`, `banned_channels_excluded`, `inference_substrate`, and
`preconditions_checked`, plus provenance fields sufficient to audit the
reliable sweep and upstream artifacts.

The complete-path `honest_verdict` SHALL equal
`success_sota_ingestion_v452_frontier_mapped`; `aimed_at_fork_verdict` SHALL
equal the actual A1 fork verdict; `inference_substrate` SHALL equal
`aggregation_from_upstream_artifacts`; `methods_mapped` SHALL contain three to
five methods; every method claim SHALL cite only verified arXiv IDs from
`arxiv_ids_cited`; each method SHALL include `experiment_graft`,
`validation_gate`, and `fails_when`; `banned_channels_excluded` SHALL be true;
and `duration_s` SHALL use the aggregation-only `0.0001` floor. For the
checked-in branch, the source set SHALL include direct arXiv HTTP-200 checks
for `2503.18938`, `2505.08073`, `2602.23997`, `2512.10016`, `2602.16229`,
`2606.25421`, `2606.26217`, and `2603.19312`, while the mapped hard-invariant
methods SHALL prioritize the remaining `.450` representation candidates
`2503.18938`, `2505.08073`, and `2602.23997`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; mapping emitted is success_sota_ingestion_v452_frontier_mapped."
- `aimed_at_fork_verdict`: principle "the A1 fork verdict the ingestion targets (UNLOCKS_VALUE/MATTERS->convert to first-win; PLANNER_GAP->planning/search; INVARIANT->third representation class or escalate)."
- `methods_mapped`: principle "the strongest 3-5 methods, each with a real arXiv ID + experiment_graft + validation_gate + fails_when."
- `arxiv_ids_cited`: principle "every method claim cites a verifiable HTTP-200 arXiv ID (no fabrication -- adversarial_verify bar)."
- `flagged_for_v452`: principle "the strongest method(s) flagged so the .452 planner reads the mapping (discover->ingest->plan->experiment)."
- `banned_channels_excluded`: principle "true -- /deep-research NOT invoked; energy/TTA-on-code-engine/stronger-local-code-inducer/coverage/exploration/selection classes NOT re-ingested."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts (0.0001s floor)."
- `preconditions_checked`: principle "records reliable-channel checks + the A1 fork-verdict read; banned channels explicitly excluded."

#### SCENARIO-ARC-WMTE-4900-V452-FRONTIER-MAPPED

Given the research files, Exp 4892 A1 artifact, Exp 4893 A1b artifact, and Exp
4890 handoff artifact are present
When Experiment 4900 runs through the reliable channel
Then it writes `results/experiment_4900_sota_ingestion_v452_frontier.json`,
records that `/deep-research` was not invoked, maps three to five third
representation methods onto the `.452` frontier with real HTTP-200 arXiv IDs,
records the actual A1 `VALUE_GAP_REPRESENTATION_INVARIANT` verdict and A1b
`VALUE_GAP_REPRESENTATION_INVARIANT_HARD` result, updates
`research-studying.md` and `research-references.md` idempotently, and flags the
strongest `.452` candidate inputs.

#### SCENARIO-ARC-WMTE-4900-BLOCKED-A1

Given the Exp 4892 A1 artifact is missing
When Experiment 4900 checks preconditions
Then it writes `blocked_a1_artifact_missing`, records the missing A1
precondition, keeps `methods_mapped=[]`, keeps `banned_channels_excluded=true`,
and does not fabricate an A1 fork verdict.

#### SCENARIO-ARC-WMTE-4900-NO-FABRICATION

Given a candidate artifact omits citations, cites an unverified arXiv ID,
targets a fork verdict inconsistent with A1/A1b, maps a retired
energy/TTA/stronger-local-code-inducer/coverage/exploration/selection/perception
class, invokes `/deep-research`, claims a model-load, training, leaderboard, or
solve result, or modifies `scripts/research_conductor.py`
When the Experiment 4900 artifact validator runs
Then it rejects the artifact before the result can be written.

### REQ-ARC-WMTE-4903: Env-Grounded Location-Pruned First-Win Search

Experiment 4903 SHALL test the `.452` env-grounded search intervention on the
same A1 held-out games `cd82`, `cn04`, `ls20`, `m0r0`, `r11l`, `sk48`, `sp80`,
`su15`, and `wa30`, with `tu93` as the positive control. The workflow SHALL
first check `arc_solver_kit.offline_arcade()`, the GPU-fixed Qwen3.5-9B-MTP
generator precondition accepting either GPU-0 CUDA or iGPU HIP without
rejecting ambient `CUDA_VISIBLE_DEVICES`, and
`results/experiment_4892_decision_need_targets_value_gap.json`. Missing
preconditions SHALL emit `blocked_offline_arcade_missing`,
`blocked_generator_unavailable`, or `blocked_a1_baseline_missing` rather than a
fabricated search lift.

The intervention SHALL use the induced executable engine only as a
change-LOCATION action prior: for each observed real state it SHALL rank legal
actions by predicted changed-cell saliency, execute top-ranked candidates in
the real offline environment, read the true next-state values from that
environment, and score progress with a learned verifier that is oracle-distinct
from the level-up check. It SHALL NOT ask the engine to predict change-VALUE
for frontier states, and the banked winning prefix SHALL be used only after the
run to classify coverage migration. The search SHALL stay live-path-reachable
through the existing `StepwiseExplorer` action-prior expansion hook and the
`arc_executable_world_model.load_engine` / `plan_in_model` induction path; it
SHALL NOT become a parallel solver.

Experiment 4903 SHALL write
`results/experiment_4903_env_grounded_location_pruned_search.json` with required
fields `honest_verdict`, `fork_verdict`,
`value_grounded_first_win_delta_median`,
`value_grounded_first_win_delta_ci95`, `median_actions_to_first_win`,
`per_game_first_win`, `change_location_prior_used_not_value`,
`coverage_migration_count`, `positive_control_game`,
`positive_control_non_degenerate`, `planner_blind_to_banked_answer`,
`verifier_is_oracle`, `live_path_reachable`, `generator_backend`,
`solve_provenance`, `checkpoint_emitted`, `inference_substrate`, `model_specs`,
`random_seed`, and `reproducibility_checksum`. The median first-win delta SHALL
be emitted as a bare numeric field named
`value_grounded_first_win_delta_median`.

If the first-win delta CI95 excludes zero and at least one `NEVER_ENUMERATED`
game migrates to `COVERED` at bounded action cost, the artifact SHALL report
`fork_verdict=ENV_GROUNDED_SEARCH_UNLOCKS_FIRST_WIN`. If the CI95 excludes zero
but solved games exceed the configured bounded action-cost threshold, it SHALL
report `fork_verdict=SEARCH_BUDGET_BOUND`. If the CI95 includes zero, it SHALL
report `fork_verdict=WALL_DEEPER_THAN_VALUE_PREDICTION` and
`retire_if_same_verdict=true`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; a real lift is success_env_grounded_search_first_win_unlocked_<delta>; a null is complete_env_grounded_search_no_first_win_lift_<fork>; a degenerate control is complete_env_grounded_positive_control_degenerate_retired."
- `fork_verdict`: principle "one of ENV_GROUNDED_SEARCH_UNLOCKS_FIRST_WIN | SEARCH_BUDGET_BOUND | WALL_DEEPER_THAN_VALUE_PREDICTION -- the headline that redirects .453."
- `value_grounded_first_win_delta_median`: principle "EMIT BARE (not a {value,principle} dict) -- A1b's gated_on reads this raw value. Median (env-grounded-search - baseline) first-win rate across held-out games; did reading change-VALUE from the env instead of predicting it unlock first-wins?"
- `value_grounded_first_win_delta_ci95`: principle "bootstrap CI95 of the first-win delta; PASS requires it to exclude 0 for a real lift."
- `median_actions_to_first_win`: principle "the EFFICIENCY axis -- the real-env action cost of grounding value; an unbounded cost is SEARCH_BUDGET_BOUND."
- `per_game_first_win`: principle "per-game {first_win_baseline, first_win_env_grounded, delta, actions_to_first_win, states_expanded, bucket in COVERED|ENUMERATED_BUT_LOST|NEVER_ENUMERATED, migrated} -- the quantitative table."
- `change_location_prior_used_not_value`: principle "true -- the induced model supplied ONLY the change-LOCATION action-ranking; change-VALUE was read from the env, never predicted (the .451-invariant-finding sidestep)."
- `coverage_migration_count`: principle "how many NEVER_ENUMERATED games migrated to COVERED under env-grounded search at a bounded action cost."
- `positive_control_game`: principle "tu93 -- its change-LOCATION ranker MUST be non-degenerate (ranks the truly-changing action highly) or the measurement is a harness artifact."
- `positive_control_non_degenerate`: principle "true iff tu93's change-LOCATION prior is non-degenerate -- carries forward the .450/.451 degenerate-metric fix."
- `planner_blind_to_banked_answer`: principle "true -- the banked winning prefix was NOT injected into ranking, search, or progress scoring."
- `verifier_is_oracle`: principle "false -- the change-LOCATION model + learned verifier are oracle-distinct from the env's level-up check; this is a SEARCH-STRATEGY result, not a verifier-moat claim (circularity discipline)."
- `live_path_reachable`: principle "the search improves the live StepwiseExplorer/plan_in_model/_induce_and_plan path (arc_orphan_solver_lint passes)."
- `generator_backend`: principle "which server served (gpu0_cuda | igpu_hip) -- proves the GPU fix and a genuine live run."
- `solve_provenance`: principle "development_proxy -- a live-path mechanism measurement on the dev twin, NOT a registry bank (A2 banks)."
- `checkpoint_emitted`: principle "a capped run still emits a usable partial (the 2026-06-25 wall-clock fix); per-game checkpointing."
- `inference_substrate`: principle "live_llm_inference (60s floor) -- the change-LOCATION induction invokes the LLM on the GPU-0 generator."
- `model_specs`: principle "names the actual generator invoked (Qwen3.5-9B-MTP via the GPU-0 CUDA llama-server) -- methodology for adversarial_verify."
- `random_seed`: principle "determinism for the action-prior induction + the best-first search tie-breaking."
- `reproducibility_checksum`: principle "content hash of (games, search/prior config, held-out split, action budget) so a replication catches drift."

#### SCENARIO-ARC-WMTE-4903-LOCATION-PRIOR-NOT-VALUE

Given an induced engine, legal action candidates, and real environment states
When Experiment 4903 ranks actions for expansion
Then it orders actions by predicted changed-cell locations only, records
`change_location_prior_used_not_value=true`, and reads all resulting next-grid
values from the real environment transition.

#### SCENARIO-ARC-WMTE-4903-FORK-VERDICT

Given at least three `NEVER_ENUMERATED` games and a non-degenerate `tu93`
positive control
When Experiment 4903 aggregates per-game first-win deltas, coverage migration,
and action cost
Then it emits one of `ENV_GROUNDED_SEARCH_UNLOCKS_FIRST_WIN`,
`SEARCH_BUDGET_BOUND`, or `WALL_DEEPER_THAN_VALUE_PREDICTION`, with
`verifier_is_oracle=false` and `retire_if_same_verdict=true` for the null branch.

#### SCENARIO-ARC-WMTE-4903-PARTIAL-CHECKPOINT

Given the run reaches its elapsed budget after measuring at least one game
When Experiment 4903 stops the current process
Then it writes per-game checkpoints and a schema-valid partial artifact so the
next run can resume without losing the first-win measurement.

### REQ-ARC-WMTE-4904: Latent-Action Interface Value-Gap Fork Probe

Experiment 4904 SHALL run only when Experiment 4903 remains in the low first-win
regime:
`results/experiment_4903_env_grounded_location_pruned_search.json` exists,
contains a bare `value_grounded_first_win_delta_median < 0.1`, and points to
the Exp 4892 held-out split plus code-engine value-accuracy baseline. The
workflow SHALL first check `arc_solver_kit.offline_arcade()`, the GPU-fixed
Qwen3.5-9B-MTP generator precondition accepting either GPU-0 CUDA or iGPU HIP
without rejecting ambient `CUDA_VISIBLE_DEVICES`, and the A1/Exp 4892 baseline
artifacts. Missing or gated-off preconditions SHALL emit a `blocked_` or
`skipped_` artifact rather than a fabricated fork.

For the same Exp 4903 held-out games and `tu93` positive control, Experiment
4904 SHALL infer self-supervised latent action tokens from cold ARC
transitions, align E3 legal controls to those latent tokens, and score the same
held-out transition ids through a latent-action state adapter. The adapter
SHALL use the live E3 induction interface
`arc_executable_world_model.load_engine`, SHALL NOT convert the representation
through the decision-need or action-prefix-delta table formats, SHALL keep
latent fit ids disjoint from held-out score ids, and SHALL keep the banked
winning answer out of induction, fitting, scoring, and planning.

Experiment 4904 SHALL write
`results/experiment_4904_latent_action_interface.json` with required fields
`honest_verdict`, `fork_verdict`,
`latent_action_value_accuracy_delta_median`,
`latent_action_value_accuracy_delta_ci95`, `per_game_value_gap`,
`ran_genuinely_live`, `delta_on_truly_heldout_split`,
`verifier_is_oracle`, `live_path_reachable`, `generator_backend`,
`solve_provenance`, `checkpoint_emitted`, `inference_substrate`,
`model_specs`, `preconditions_checked`, `random_seed`, and
`reproducibility_checksum`. The median latent-action value-accuracy delta SHALL
be emitted as a bare numeric field.

If the value-accuracy delta CI95 excludes zero, the artifact SHALL report
`fork_verdict=REPRESENTATION_MATTERS`. If the CI95 includes zero, it SHALL
report `fork_verdict=VALUE_GAP_REPRESENTATION_INVARIANT_4_CLASSES` and
`retire_if_same_verdict=true`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; a real lift is success_latent_action_value_gap_closed_<delta>; a flat null is complete_latent_action_no_value_lift_representation_invariant_4_classes."
- `fork_verdict`: principle "one of REPRESENTATION_MATTERS | VALUE_GAP_REPRESENTATION_INVARIANT_4_CLASSES -- redirects .453."
- `latent_action_value_accuracy_delta_median`: principle "median (latent-action - A1 code-engine baseline) changed-cell value accuracy; did a latent-action interface close the value gap?"
- `latent_action_value_accuracy_delta_ci95`: principle "bootstrap CI95 of the latent-action value-accuracy delta; PASS requires it to exclude 0 for a real lift."
- `per_game_value_gap`: principle "per-game {value_acc_code_baseline, value_acc_latent_action, cell_recall, delta, ci95} -- the quantitative table."
- `ran_genuinely_live`: principle "true iff duration_s > 60 and the latent-action induction/scoring genuinely ran (the explicit .450 A1b 13.7s non-test fix)."
- `delta_on_truly_heldout_split`: principle "true -- scored on the SAME held-out split as A1, disjoint from any fit set (B1 audits)."
- `verifier_is_oracle`: principle "false -- the held-out transition score is oracle-distinct from the env's level-up check (circularity discipline)."
- `live_path_reachable`: principle "the latent-action interface uses the live e3 load_engine interface (arc_orphan_solver_lint passes), not a parallel solver."
- `generator_backend`: principle "which server served (gpu0_cuda | igpu_hip) -- proves the GPU fix and a genuine live run."
- `solve_provenance`: principle "development_proxy -- a representation-accuracy measurement, NOT a banked level."
- `checkpoint_emitted`: principle "a capped run still emits a usable partial (per-game checkpointing)."
- `inference_substrate`: principle "live_llm_inference (60s floor) -- the latent-action induction invokes an LLM on the GPU-0 generator."
- `model_specs`: principle "names the inducer (Qwen3.5-9B-MTP via the GPU-0 CUDA llama-server) -- methodology for adversarial_verify."
- `preconditions_checked`: principle "records arcade/generator/A1-baseline checks; a missing resource emits blocked_."
- `random_seed`: principle "determinism for the latent-action induction stochastic search."
- `reproducibility_checksum`: principle "content hash of (games, A1 baseline, latent-action config, held-out split) so a replication catches drift."

#### SCENARIO-ARC-WMTE-4904-A1-LOW-FIRST-WIN-GATE

Given the Exp 4903 env-grounded search artifact
When Experiment 4904 checks preconditions
Then it proceeds only when the bare first-win delta is below `0.1` and the Exp
4892 artifact contains per-game code baselines plus held-out transition ids;
otherwise it writes a gated `blocked_` or `skipped_` artifact.

#### SCENARIO-ARC-WMTE-4904-LATENT-ACTION-INTERFACE

Given the agent's own cold-start transitions
When Experiment 4904 induces latent action tokens
Then it creates non-code latent action embeddings from observed transitions,
aligns E3 controls to those tokens, and scores the same held-out split without
using banked answers or the decision-need/action-prefix table formats.

#### SCENARIO-ARC-WMTE-4904-GENUINELY-LIVE

Given at least three held-out games are available
When Experiment 4904 measures the latent-action interface
Then a passing artifact records `ran_genuinely_live=true`, `duration_s > 60`,
`inference_substrate=live_llm_inference`, and the live path includes
`arc_executable_world_model.load_engine`.

#### SCENARIO-ARC-WMTE-4904-FORK-VERDICT

Given the per-game latent-action value-accuracy deltas against the Exp 4892
code-engine baseline
When Experiment 4904 computes the median delta and bootstrap CI95
Then it emits `REPRESENTATION_MATTERS` only when the CI95 excludes zero, and
otherwise emits `VALUE_GAP_REPRESENTATION_INVARIANT_4_CLASSES`.

### REQ-ARC-WMTE-4908: Env-Grounded Search A1/A1b Adversarial Audit

Experiment 4908 SHALL audit the `.452` Exp 4903 A1 env-grounded
location-pruned first-win headline and the gated Exp 4904 A1b latent-action
interface before any capstone trusts either fork. If the A1 artifact, A1 source
script, or required A1b artifact when the A1 first-win delta is below `0.1` is
missing, the workflow SHALL write
`results/experiment_4908_env_grounded_search_audit.json` with
`honest_verdict=blocked_upstream_artifact_missing` and SHALL not fabricate any
audit pass.

The A1 audit SHALL confirm that change-VALUE came from executed real
environment transitions rather than model-predicted values, that the banked
winning prefix was not injected into ranking/search/scoring, that the `tu93`
change-LOCATION positive control is non-degenerate, that the per-game
first-win/action-cost/migration numbers recompute the named fork verdict, and
that the live path is reachable rather than an orphan solver. The A1
trustworthy capstone boolean SHALL be exactly
`a1_value_from_real_env AND a1_planner_blind AND
a1_positive_control_non_degenerate AND a1_numbers_match_fork`.

The A1b audit SHALL confirm that Exp 4904 either correctly gate-skipped because
A1 lifted first-win by at least `0.1`, or genuinely ran live on the same
held-out split with `duration_s > 60`, `ran_genuinely_live=true`,
`inference_substrate=live_llm_inference`, and oracle-distinct
`verifier_is_oracle=false`. The workflow SHALL run
`scripts/summarize_artifact.py` and `scripts/adversarial_verify.py` on the
audited artifacts, SHALL include a per-claim audit table with boolean pass/fail
and an evidence line, SHALL set `adversarial_flags_found=true` when either
adversarial verification report contains flags, SHALL declare
`inference_substrate=aggregation_from_upstream_artifacts`, and SHALL NOT modify
`scripts/research_conductor.py`, `ops/changelog.md`, `ops/status.md`, or
`_bmad/traceability.md`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; complete_a1_a1b_audited."
- `a1_value_from_real_env`: principle "true iff the search read change-VALUE from executed env transitions, not the model's prediction -- the whole .452 premise."
- `a1_planner_blind`: principle "true iff the banked winning prefix was NOT injected into ranking/search/scoring (else the first-win is a tautology)."
- `a1_positive_control_non_degenerate`: principle "true iff tu93's change-LOCATION prior is non-degenerate (ranks the truly-changing action highly)."
- `a1_numbers_match_fork`: principle "true iff the per-game (first-win x action-cost x migration) numbers support the named fork verdict."
- `a1_trustworthy`: principle "real-env AND planner-blind AND positive-control AND numbers-match -- the capstone trusts A1 only if true."
- `a1b_ran_genuinely_live`: principle "true iff A1b duration_s>60 and genuinely ran (the .450 A1b 13.7s non-test fix); or a1b_gate_skipped=true."
- `a1b_gate_skipped`: principle "true iff A1 lifted first-win >= 0.1 so the gate correctly skipped the last representation swing."
- `adversarial_flags_found`: principle "any adversarial_verify flag on the A1/A1b artifacts (fabrication gate)."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts (reads upstream artifacts + the A1 script; no LLM)."

#### SCENARIO-ARC-WMTE-4908-A1-AUDIT

Given the Exp 4903 artifact and source script are present
When Experiment 4908 audits A1
Then it records a per-claim table for real-env value grounding,
planner-blindness, positive-control non-degeneracy, numbers-match-fork, and
live-path reachability, with each row carrying a boolean pass and evidence
line, and computes `a1_trustworthy` from the four load-bearing A1 booleans.

#### SCENARIO-ARC-WMTE-4908-A1B-LIVE-OR-GATE-SKIPPED

Given the Exp 4903 first-win delta and optional Exp 4904 artifact
When Experiment 4908 audits A1b
Then it records `a1b_gate_skipped=true` only when the A1 first-win delta is at
least `0.1`, otherwise it requires the Exp 4904 artifact to have genuinely run
live for more than 60 seconds on the same held-out split and with an
oracle-distinct verifier.

#### SCENARIO-ARC-WMTE-4908-BLOCKED-UPSTREAM

Given the required A1 artifact/script is missing, or the A1b artifact is
missing while A1 remains below the `0.1` first-win gate
When Experiment 4908 starts
Then it writes `blocked_upstream_artifact_missing`, records the missing
precondition, sets all pass booleans false except any explicitly proven gate
skip, and exits without fabricating the audit.

### REQ-ARC-WMTE-4911: V453 Frontier SOTA Ingestion After Env-Grounded Wall

Experiment 4911 SHALL run the `.453` SOTA-ingestion slot after reading A1's
env-grounded search fork artifact
`results/experiment_4903_env_grounded_location_pruned_search.json`, A1b's
last-representation artifact
`results/experiment_4904_latent_action_interface.json` when A1 did not clear
the first-win gate, `ops/north-star.md`, `research-studying.md`,
`research-references.md`, `scripts/sweep_clusters.py`, and
`scripts/sweep_semscholar.py`. If either required upstream artifact is missing
for the checked-in wall branch, Experiment 4911 SHALL write
`results/experiment_4911_sota_ingestion_v453_frontier.json` with
`honest_verdict=blocked_upstream_artifact_missing`, `methods_mapped=[]`,
`post_sprint_pivot_methods=[]`, `banned_channels_excluded=true`, and SHALL NOT
fabricate an A1 or A1b fork verdict.

The workflow SHALL branch on A1's actual `fork_verdict`. If A1 reports
`ENV_GROUNDED_SEARCH_UNLOCKS_FIRST_WIN`, the `.453` map SHALL target
neural-guided/value-guided online search, learned action priors, and
sample-efficient interactive search. If A1 reports `SEARCH_BUDGET_BOUND`, the
map SHALL target cheaper action-priors, amortized search, and planning
efficiency. If A1 reports `WALL_DEEPER_THAN_VALUE_PREDICTION` and A1b reports
`VALUE_GAP_REPRESENTATION_INVARIANT_4_CLASSES`, the map SHALL record that the
change-VALUE wall survives four representation classes plus env-grounding and
SHALL only promote final-sprint causal/world-model diagnostic candidates, not
another replay of retired ARC levers.

Regardless of the A1 branch, Experiment 4911 SHALL also map the post-sprint
verifier-moat / oracle-distinct / FoVer-paper pivot from `ops/north-star.md`
sections 1 and 5. This pivot SHALL focus on learned or energy verifiers in
non-saturated domains where self-consistency is not near ceiling and an
oracle-distinct verifier can add value. It SHALL NOT claim a solve, load a
model, train, submit to a leaderboard, invoke `/deep-research`, re-ingest the
nulled energy-as-ARC-lever, TTA-on-code-engine, stronger-local-code-inducer,
decision-need-targets, action-prefix-latents, coverage/vocabulary,
exploration, selection/ranking, or perception-from-grid classes, or modify
`scripts/research_conductor.py`, `ops/changelog.md`, `ops/status.md`, or
`_bmad/traceability.md`.

On a successful ingestion-synthesis run, Experiment 4911 SHALL write
`results/experiment_4911_sota_ingestion_v453_frontier.json`, update
`research-studying.md` with an idempotent Exp 4911 INGESTED section, and update
`research-references.md` with an idempotent Exp 4911 source section. The
artifact SHALL include `honest_verdict`, `aimed_at_fork_verdict`,
`methods_mapped`, `post_sprint_pivot_methods`, `arxiv_ids_cited`, `citations`,
`banned_channels_excluded`, `flagged_for_v453`, `inference_substrate`,
`random_seed`, and `reproducibility_checksum`, plus provenance fields that
record upstream artifacts, the reliable sweep channel, the branch selected, and
the no-fabrication guards.

The complete-path `honest_verdict` SHALL equal
`success_sota_ingestion_v453_frontier_mapped`; `aimed_at_fork_verdict` SHALL
equal A1's actual `WALL_DEEPER_THAN_VALUE_PREDICTION` branch for the checked-in
artifacts; `inference_substrate` SHALL equal
`aggregation_from_upstream_artifacts`; `methods_mapped` SHALL contain three to
five final-sprint wall-diagnostic methods; `post_sprint_pivot_methods` SHALL
contain at least three verifier-moat candidates; every method claim SHALL cite
only verified arXiv IDs from `arxiv_ids_cited`; `banned_channels_excluded`
SHALL be true; `duration_s` SHALL use the aggregation-only `0.0001` floor; and
the source set SHALL include direct arXiv HTTP-200 checks for `2401.12497`,
`2505.02074`, `2602.11389`, `2504.07257`, `2605.18871`, `2505.14999`,
`2606.04579`, and `2604.24198`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; success_sota_ingestion_v453_frontier_mapped."
- `aimed_at_fork_verdict`: principle "the A1 fork verdict the ingestion targets (selects the .453 branch)."
- `methods_mapped`: principle "the strongest 3-5 methods, each with a real arXiv ID + experiment_graft + fails_when."
- `post_sprint_pivot_methods`: principle "the verifier-moat / oracle-distinct candidates for when the ARC sprint retires 6/30 (north-star §1/§5)."
- `arxiv_ids_cited`: principle "every method claim cites a verifiable HTTP-200 arXiv ID (no fabrication)."
- `citations`: principle "HTTP-200 arXiv source metadata backing every method claim."
- `banned_channels_excluded`: principle "true -- /deep-research NOT invoked; nulled classes NOT re-ingested."
- `flagged_for_v453`: principle "the strongest method(s) flagged so the .453 planner reads the mapping (discover->ingest->plan->experiment)."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts (reads upstream verdicts + sweeps; 0.0001s floor)."
- `random_seed`: principle "determinism for the sweep query ordering."
- `reproducibility_checksum`: principle "content hash of (branch, sweep queries, mapped methods) so a replication catches drift."

#### SCENARIO-ARC-WMTE-4911-V453-WALL-AND-PIVOT-MAPPED

Given the research files, north-star note, Exp 4903 A1 artifact, and Exp 4904
A1b artifact are present
When Experiment 4911 runs through the reliable channel
Then it writes `results/experiment_4911_sota_ingestion_v453_frontier.json`,
records that `/deep-research` was not invoked, records A1's
`WALL_DEEPER_THAN_VALUE_PREDICTION` and A1b's
`VALUE_GAP_REPRESENTATION_INVARIANT_4_CLASSES`, maps three to five final ARC
wall-diagnostic methods plus the post-sprint verifier-moat candidates with real
HTTP-200 arXiv IDs, updates `research-studying.md` and
`research-references.md` idempotently, and flags the strongest `.453` planner
inputs.

#### SCENARIO-ARC-WMTE-4911-BLOCKED-UPSTREAM

Given the Exp 4903 A1 artifact or required Exp 4904 A1b artifact is missing
When Experiment 4911 checks preconditions
Then it writes `blocked_upstream_artifact_missing`, records the missing
precondition, keeps `methods_mapped=[]`, keeps
`post_sprint_pivot_methods=[]`, keeps `banned_channels_excluded=true`, and does
not fabricate either fork verdict.

#### SCENARIO-ARC-WMTE-4911-NO-FABRICATION

Given a candidate artifact omits citations, cites an unverified arXiv ID,
targets a fork verdict inconsistent with A1/A1b, maps a nulled class, invokes
`/deep-research`, claims a model-load, training, leaderboard, or solve result,
or modifies `scripts/research_conductor.py`
When the Experiment 4911 artifact validator runs
Then it rejects the artifact before the result can be written.

### REQ-ARC-WMTE-4914: Causal Abstraction Wall Diagnostic

Experiment 4914 SHALL run the `.453` causal state-abstraction diagnostic
selected by Experiment 4911. It SHALL first check `arc_solver_kit.offline_arcade()`,
the GPU-fixed Qwen3.5-9B-MTP generator precondition accepting either GPU-0
CUDA or iGPU HIP without rejecting ambient `CUDA_VISIBLE_DEVICES`, and
`results/experiment_4903_env_grounded_location_pruned_search.json`. Missing
preconditions SHALL write
`results/experiment_4914_causal_abstraction_wall_diagnostic.json` with
`blocked_offline_arcade_missing`, `blocked_generator_unavailable`, or
`blocked_a1_baseline_missing` rather than fabricating a wall classification.

For at least three Exp 4903 `NEVER_ENUMERATED` failed games, plus the positive
controls `tu93` and one already-solved L2-or-deeper game, Experiment 4914 SHALL
load the live E3 engine through `arc_executable_world_model.load_engine`,
collect real offline-env transitions, derive the minimal causal variables
needed to classify changed-cell value and progress-to-goal labels, and classify
each required variable as observable only when an extractor proves it is
readable from the ARC frame, action, or env state the agent already sees.
Latent counters, off-screen registers, banked winning-prefix indices, and
interaction-dependent variables without such an extractor SHALL be classified
hidden. The diagnostic SHALL NOT become a decision-need target table, SHALL NOT
claim a solve, SHALL NOT report a static-ranking lift, and SHALL use banked
winning prefixes only as classification ground truth after induction.

Experiment 4914 SHALL write
`results/experiment_4914_causal_abstraction_wall_diagnostic.json` with required
fields `honest_verdict`, `fork_verdict`, `per_game_causal_abstraction`,
`minimal_abstraction_is_observable_subset`, `positive_control_games`,
`positive_control_classifies_observable`, `is_decision_need_table_in_disguise`,
`planner_blind_to_banked_answer`, `verifier_is_oracle`, `live_path_reachable`,
`generator_backend`, `solve_provenance`, `checkpoint_emitted`,
`inference_substrate`, `model_specs`, `preconditions_checked`, `random_seed`,
and `reproducibility_checksum`.

If every failed-game minimal abstraction is a subset of observable variables
and every positive control classifies observable, the artifact SHALL report
`fork_verdict=WALL_IS_OBSERVABLE_VARIABLE_GAP`. If any failed-game minimal
abstraction requires a hidden variable and every positive control classifies
observable, it SHALL report `fork_verdict=WALL_IS_HIDDEN_STATE`. If any
positive control classifies hidden, it SHALL report
`fork_verdict=DIAGNOSTIC_DEGENERATE_RETIRED` and retire the diagnostic lens.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; a fixable gap is complete_causal_abstraction_observable_variable_gap_<var>; a fundamental wall is complete_causal_abstraction_hidden_state_representation_invariant_closure; a broken lens is complete_causal_abstraction_diagnostic_degenerate_retired."
- `fork_verdict`: principle "one of WALL_IS_OBSERVABLE_VARIABLE_GAP | WALL_IS_HIDDEN_STATE | DIAGNOSTIC_DEGENERATE_RETIRED -- the mechanistic closure verdict for the FoVer paper's ARC section + the .454 handoff."
- `per_game_causal_abstraction`: principle "per-game {required_variables: list, observable_from_interface: {var: bool}, classification in OBSERVABLE_GAP|HIDDEN_STATE, evidence} -- the quantitative classification table."
- `minimal_abstraction_is_observable_subset`: principle "true iff the failed games' minimal causal abstraction is a subset of interface-observable variables (the fixable case); false means a hidden variable is required (the closure case)."
- `positive_control_games`: principle "tu93 + a solved L2 game -- on solved games the abstraction MUST classify observable, else the diagnostic is broken."
- `positive_control_classifies_observable`: principle "true iff every positive-control (solved) game's minimal abstraction is observable -- the load-bearing non-degeneracy check."
- `is_decision_need_table_in_disguise`: principle "false -- A1 produces a CLASSIFICATION report, NOT a change-VALUE-predicting table (exp4911 fails_when forbids it)."
- `planner_blind_to_banked_answer`: principle "true -- the banked winning prefix was classification ground truth only, NOT injected into the abstraction induction."
- `verifier_is_oracle`: principle "false -- the causal-abstraction classifier is oracle-distinct from the env's level-up check; a DIAGNOSTIC, not a moat claim (circularity discipline)."
- `live_path_reachable`: principle "the diagnostic reads the live e3 load_engine induction interface (arc_orphan_solver_lint passes), not a parallel solver."
- `generator_backend`: principle "which server served (gpu0_cuda | igpu_hip) -- proves the GPU fix and a genuine live run."
- `solve_provenance`: principle "development_proxy -- a diagnostic over the dev twin, NOT a registry bank."
- `checkpoint_emitted`: principle "a capped run still emits a usable partial (per-game checkpointing)."
- `inference_substrate`: principle "live_llm_inference (60s floor) -- the causal-abstraction induction invokes the LLM on the GPU-0 generator."
- `model_specs`: principle "names the actual generator invoked (Qwen3.5-9B-MTP via the GPU-0 CUDA llama-server) -- methodology for adversarial_verify."
- `preconditions_checked`: principle "records arcade/generator/A1-baseline checks; a missing resource emits blocked_."
- `random_seed`: principle "determinism for the causal-abstraction induction stochastic search."
- `reproducibility_checksum`: principle "content hash of (games, failed transitions, abstraction config, held-out split) so a replication catches drift."

#### SCENARIO-ARC-WMTE-4914-CLASSIFICATION-REPORT

Given the Exp 4903 failed-game rows, live E3 engines, and real offline-env
transition samples
When Experiment 4914 induces the causal abstraction
Then it emits a per-game table of required variables, per-variable
observability proof, and `OBSERVABLE_GAP` or `HIDDEN_STATE` classification
without emitting a change-VALUE target table, solve claim, or static-ranking
lift.

#### SCENARIO-ARC-WMTE-4914-POSITIVE-CONTROL

Given `tu93` and a solved L2-or-deeper game are available
When Experiment 4914 applies the same classifier
Then every positive control must classify observable, otherwise the artifact
reports `DIAGNOSTIC_DEGENERATE_RETIRED`.

#### SCENARIO-ARC-WMTE-4914-FORK-VERDICT

Given at least three failed `NEVER_ENUMERATED` games and passing positive
controls
When Experiment 4914 aggregates the per-game classifications
Then it emits exactly one of `WALL_IS_OBSERVABLE_VARIABLE_GAP` or
`WALL_IS_HIDDEN_STATE`, and otherwise emits
`DIAGNOSTIC_DEGENERATE_RETIRED` for a broken positive-control lens.

#### SCENARIO-ARC-WMTE-4914-PARTIAL-CHECKPOINT

Given the run reaches its elapsed budget after measuring at least one game
When Experiment 4914 stops the current process
Then it writes per-game checkpoints and a schema-valid partial artifact so the
next run can resume without losing the causal-abstraction ledger.

### REQ-ARC-WMTE-4918: Causal Abstraction Diagnostic Audit

Experiment 4918 SHALL adversarially audit the Experiment 4914 causal
abstraction wall diagnostic before any capstone trusts the `.453` closure
verdict. It SHALL read
`results/experiment_4914_causal_abstraction_wall_diagnostic.json`,
`python/carnot/experiment_4914_causal_abstraction_wall_diagnostic.py`, and
`results/experiment_4903_env_grounded_location_pruned_search.json`. If the
Experiment 4914 artifact or source, or the Experiment 4903 artifact, is
missing, it SHALL write
`results/experiment_4918_causal_abstraction_audit.json` with
`honest_verdict=blocked_a1_artifact_missing`, record the missing precondition,
and SHALL not fabricate trust.

Experiment 4918 SHALL evaluate six load-bearing checks: (1) `real_transitions`
is true only when every failed-game row cited by Experiment 4914 exists as an
actual failed `NEVER_ENUMERATED` row in Experiment 4903, used real env reads,
used zero change-value predictions, and the Experiment 4914 source's default
classifier collects observed transitions rather than placeholder rows; (2)
`not_value_table` is true only when Experiment 4914 is a classification report,
not an exp4911-style change-value decision-need table; (3)
`observable_claims_verified` is true only when at least two variables that A1
classified observable have proof extractors readable from the visible frame,
candidate action, or env state; (4) `positive_control_observable` is true only
when all positive-control rows classify observable; (5)
`oracle_distinct_planner_blind` is true only when `verifier_is_oracle=false`
and `planner_blind_to_banked_answer=true`; and (6) `numbers_match_fork` is true
only when the per-game classification table supports the named fork verdict.

Experiment 4918 SHALL write
`results/experiment_4918_causal_abstraction_audit.json` with required fields
`honest_verdict`, `a1_diagnostic_trustworthy`, `checks`,
`a1_failure_reasons`, `observable_claims_spot_checked`,
`inference_substrate`, `preconditions_checked`, `source_a1_artifact`,
`source_a1_script`, `source_exp4903_artifact`, `transition_cross_checks`,
`not_value_table_evidence`, `positive_control_evidence`,
`oracle_distinct_planner_blind_evidence`, `numbers_match_fork_evidence`,
`field_principles`, `duration_s`, and `reproducibility_checksum`. The
`a1_diagnostic_trustworthy` field SHALL equal the boolean AND of exactly those
six checks, and each failed check SHALL append a named reason to
`a1_failure_reasons`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; complete_a1_causal_abstraction_audited (trusted or with named failures)."
- `a1_diagnostic_trustworthy`: principle "AND of the 6 checks -- the capstone trusts A1's closure verdict ONLY if this is true."
- `checks`: principle "per-check booleans {real_transitions, not_value_table, observable_claims_verified, positive_control_observable, oracle_distinct_planner_blind, numbers_match_fork}."
- `a1_failure_reasons`: principle "list of named failures (empty if trusted) -- the audit reports honestly, no rubber-stamp."
- `observable_claims_spot_checked`: principle ">=2 claimed-observable variables verified readable from frame/env state (the load-bearing honesty check)."
- `inference_substrate`: principle "verifier_ensemble_against_cached_candidates (reads cached artifacts; 1s floor)."
- `preconditions_checked`: principle "records exp4914/exp4903 presence; a missing input emits blocked_."

#### SCENARIO-ARC-WMTE-4918-A1-DIAGNOSTIC-AUDIT

Given the checked-in Experiment 4914 artifact and source and the Experiment
4903 artifact are present
When Experiment 4918 runs the audit
Then it writes a schema-valid `complete_a1_causal_abstraction_audited`
artifact, records all six check booleans with evidence, verifies at least two
observable claims from extractor proofs, cross-checks cited failed games
against Experiment 4903, and sets `a1_diagnostic_trustworthy=true` only when
all checks pass.

#### SCENARIO-ARC-WMTE-4918-NAMED-FAILURES

Given any of the six audit checks fails
When Experiment 4918 builds the audit artifact
Then `a1_diagnostic_trustworthy=false`, the matching check boolean is false,
and `a1_failure_reasons` names the failing condition rather than silently
rubber-stamping A1.

#### SCENARIO-ARC-WMTE-4918-BLOCKED-A1-ARTIFACT

Given the Experiment 4914 artifact, Experiment 4914 source script, or
Experiment 4903 artifact is missing
When Experiment 4918 checks preconditions
Then it writes `blocked_a1_artifact_missing`, marks every trust check false,
records the missing path, and emits an untrusted artifact without inventing
diagnostic evidence.

### REQ-ARC-WMTE-4933: MATM Similarity-Keyed Within-Game Retrieval

Experiment 4933 SHALL add a flag-gated coarse state-descriptor index to the
live `StepwiseExplorer` navigation graph. The submitted default SHALL remain
exact-frame-hash navigation. When the flag is enabled, `StepwiseExplorer` SHALL
quantize `cross_game_features_v2(frame)` into a deterministic bucket, index only
states observed within the current hidden-game rollout, and allow
`_shortest_path` to return an observed action sub-sequence from a similar
bucketed state only after routing the candidate prefix through the existing
oracle-distinct value/goal/world-model-verifier checks. The retrieved prefix
SHALL be treated as a navigation candidate, not as an environment oracle or a
cross-game memory.

Experiment 4933 SHALL write
`results/experiment_4933_matm_similarity_retrieval_efficiency.json` with
principle-annotated fields for `honest_verdict`, `verifier_is_oracle`,
`forward_walk_hit_rate_delta`, `actions_to_first_levelup_delta`,
`reached_level_regression`, `submitted_parity_test_green`,
`flag_eligible_to_default_on`, `live_path_reachable`,
`moves_reproducible_total_levels`, `post_sprint_pivot_gate_noted`,
`arxiv_ids_cited`, `inference_substrate`, `preconditions_checked`,
`random_seed`, and `reproducibility_checksum`. The artifact SHALL compare the
flagged similarity retrieval configuration against the submitted exact-hash
baseline on `tu93`, `lp85`, `sp80`, `cn04`, and `m0r0`. The gate SHALL pass only
when forward-walk hit rate is strictly higher, actions-to-first-level-up drops
by at least one on at least two games, reached levels do not regress, the
submitted parity test is green, and lazy value routing remains in budget.
Otherwise the artifact SHALL retire the lever with an honest null.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; a landed efficiency lever is success_matm_similarity_retrieval_action_efficiency_up; a null is complete_matm_similarity_retrieval_no_efficiency_gain_retired."
- `verifier_is_oracle`: principle "false -- the retrieved prefix is scored by the verifier router, NOT the env oracle (oracle-distinct; passes check_circular_moat_overclaim)."
- `forward_walk_hit_rate_delta`: principle "per-game forward_walk_hit_rate vs the SUBMITTED exact-hash baseline -- the retrieval-quality signal (strictly up to PASS)."
- `actions_to_first_levelup_delta`: principle "per-game actions-to-first-levelup vs baseline -- the squared-scored efficiency metric (down >=1 on >=2 games to PASS)."
- `reached_level_regression`: principle "must be ZERO -- an efficiency lever may not regress any reproduced level."
- `submitted_parity_test_green`: principle "test_arc_submitted_agent_parity.py green -- the flag does not break the submitted agent."
- `flag_eligible_to_default_on`: principle "true only if the full gate passes; otherwise the lever RETIRES (retire_if_same_verdict)."
- `live_path_reachable`: principle "true -- the index lives in the live StepwiseExplorer (arc_orphan_solver_lint passes), reachable by E3AgentPolicy."
- `moves_reproducible_total_levels`: principle "false (expected) -- this is action-efficiency; it moves reproducible_total_levels only if a retrieved sub-sequence banks a strictly new level."
- `post_sprint_pivot_gate_noted`: principle "the distributional-energy-verifier validation gate (arXiv:2605.18871) noted for the post-6/30 handoff -- NOT started here."
- `arxiv_ids_cited`: principle "2606.19911 (MATM) + 2603.10600 (Trajectory-Informed Memory) + 2605.18871 (post-sprint pivot) -- no fabrication."
- `inference_substrate`: principle "live_llm_inference if induction runs (60s floor); else the honest replay/scorecard substrate."
- `preconditions_checked`: principle "records arcade/metaharness/fixture/generator checks; a missing resource emits blocked_."
- `random_seed`: principle "determinism for the LSH bucketing + the A/B replay."
- `reproducibility_checksum`: principle "content hash of (games, baseline config, similarity-index config) so a replication catches drift."

#### SCENARIO-ARC-WMTE-4933-LIVE-WIRING

Given `StepwiseExplorer` has exact forward edges and a current state whose
coarse `cross_game_features_v2` bucket matches a prior observed state
When similarity retrieval is enabled and exact `_shortest_path` fails
Then `_shortest_path` may return the prior state's observed action sub-sequence
only if the value/goal/world-model-verifier router accepts it, and
`navigation_diagnostics()` reports similarity hits separately from exact
hash hits.

#### SCENARIO-ARC-WMTE-4933-FLAG-OFF-PARITY

Given the submitted agent is constructed without the experiment flag
When `test_arc_submitted_agent_parity.py` inspects the live E3 defaults
Then similarity retrieval remains disabled, exact-hash navigation behavior is
unchanged, and the submitted config is not eligible to default the flag on
unless the full Experiment 4933 efficiency gate passes.

#### SCENARIO-ARC-WMTE-4933-ARTIFACT-GATE

Given the reproduced-game replay/scorecard resources are available
When Experiment 4933 builds its artifact
Then it records per-game forward-walk hit-rate deltas,
actions-to-first-level-up deltas, reached-level regression status, arXiv IDs
`2606.19911`, `2603.10600`, and `2605.18871`, and the post-6/30
distributional-energy-verifier validation gate without starting that pivot.

## Implementation Status (REQ-ARC-WMTE-4908)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-ARC-WMTE-4908 | Implemented (`python/carnot/experiment_4908_env_grounded_search_audit.py`) | Implemented (`tests/python/test_experiment_4908_env_grounded_search_audit.py`) |

### REQ-ARC-WMTE-4897: Value-Gap Representation Adversarial Audit

Experiment 4897 SHALL audit the Experiment 4892 A1 decision-need value-gap
probe and the optional Experiment 4893 A1b action-prefix latent adapter before
any capstone trusts the `.451` representation fork. If the A1 artifact or
source script is missing, Experiment 4897 SHALL write
`results/experiment_4897_value_gap_representation_audit.json` with
`honest_verdict=blocked_a1_artifact_missing` and SHALL not fabricate any A1 or
A1b pass.

The audit SHALL run `scripts/summarize_artifact.py` and
`scripts/adversarial_verify.py` on the A1 artifact, SHALL run
`scripts/adversarial_verify.py` on A1b when A1b exists, SHALL inspect the A1
source path so the banked winning answer is used only after planning for
classification, and SHALL require `arc_orphan_solver_lint` to pass before
declaring the live path reachable. A1 is genuinely diagnostic only when the
graded `tu93` positive control is non-degenerate (`cell_recall > 0`), the
reported value delta is scored on a split disjoint from the decision-need
authoring set, the planner is blind to the banked answer, per-game
`cell_recall`/value-accuracy/delta numbers support the reported
`fork_verdict`, the live path is reachable, and the A1 run used GPU-0 CUDA or
iGPU HIP for at least the live-model duration floor without an adversarial
flag. A1b is genuinely live only when it was gate-skipped because A1 closed the
value gap or blocked, or when it cleared the 60s live floor without a
`DURATION_TOO_SHORT` flag, scored the same held-out games and split as A1,
remained oracle-distinct, and used a reachable live path.

Experiment 4897 SHALL write
`results/experiment_4897_value_gap_representation_audit.json` with required
top-level fields `honest_verdict`, `a1_genuinely_diagnostic`,
`a1_positive_control_non_degenerate_confirmed`,
`a1_delta_on_heldout_disjoint_confirmed`, `planner_blind_confirmed`,
`numbers_match_fork`, `a1b_ran_genuinely_live`, `inference_substrate`,
`checks`, `a1_failure_reasons`, `a1b_failure_reasons`,
`a1_summarizer_result`, `a1_adversarial_result`,
`a1b_adversarial_result`, `live_lint_result`, `preconditions_checked`,
`field_principles`, `duration_s`, and `reproducibility_checksum`. The workflow
SHALL append an idempotent Experiment 4897 section to
`ops/arc_null_silent_bug_audit.md`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; audit complete is complete_a1_a1b_audited."
- `a1_genuinely_diagnostic`: principle "the load-bearing check -- non-degenerate-graded-control AND held-out-disjoint-delta AND planner-blind AND numbers-match-fork AND live-path-reachable AND ran-live-on-GPU0; else A1's fork verdict is void."
- `a1_positive_control_non_degenerate_confirmed`: principle "true iff the GRADED tu93 positive control really had cell_recall > 0 -- carries forward the .450 degenerate-metric fix."
- `a1_delta_on_heldout_disjoint_confirmed`: principle "true iff A1's value-accuracy delta was scored on a split DISJOINT from the representation's authoring set (not a tautology)."
- `planner_blind_confirmed`: principle "true iff the banked winner was used only to classify, never to seed authoring/fitting/planning."
- `numbers_match_fork`: principle "true iff the per-game (cell_recall, value-accuracy, delta) numbers support the claimed fork_verdict."
- `a1b_ran_genuinely_live`: principle "true iff A1b cleared the 60s live floor (duration_s>60, NOT DURATION_TOO_SHORT -- the explicit .450 A1b non-test fix) on the same held-out split (or A1b was gate-skipped)."
- `inference_substrate`: principle "aggregation_from_upstream_artifacts (0.0001s floor)."

#### SCENARIO-ARC-WMTE-4897-A1-A1B-AUDIT

Given checked-in A1 and optional A1b artifacts plus the A1 source script
When Experiment 4897 runs the hostile audit
Then it writes a schema-valid `complete_a1_a1b_audited` artifact whose A1 and
A1b booleans are true only when all load-bearing checks pass, and whose failure
reasons make any degenerate graded control, held-out split tautology,
banked-answer injection, fork-number mismatch, unreachable live path,
wrong/too-short substrate, circular verifier, or unfair A1b split explicit.

#### SCENARIO-ARC-WMTE-4897-A1B-LIVE-OR-GATE-SKIPPED

Given Experiment 4893 exists or A1 makes it unnecessary
When Experiment 4897 evaluates A1b
Then `a1b_ran_genuinely_live=true` only when A1b is gate-skipped because A1
closed the value gap or blocked, or when A1b reports `duration_s > 60`,
`ran_genuinely_live=true`, no `DURATION_TOO_SHORT` adversarial flag, the same
A1 held-out game split, oracle-distinct scoring, and a reachable live path.

#### SCENARIO-ARC-WMTE-4897-BLOCKED-A1-ARTIFACT

Given the A1 artifact or A1 source script is missing
When Experiment 4897 runs
Then it writes `blocked_a1_artifact_missing`, marks every trust boolean false,
records the missing precondition, and does not claim A1 or A1b audit success.

### REQ-ARC-WMTE-4852: Rotated ARC Level-Up Attempt Guarantee

Experiment 4852 SHALL run the standing ARC solve loop on a rotated target that
is not `.446`'s `ka59`. It SHALL precheck `arc_solver_kit.offline_arcade()`,
read `ops/arc_solve_registry.yaml`, inspect the shallow L1-only candidates
`g50t`, `s5i5`, `wa30`, and `r11l`, and use registry dead-end notes to avoid
known no-grounded-delta walls. The preferred target SHALL be the first
candidate with a grounded next-level delta and no stronger skip reason; for the
2026-06-27 registry this is `s5i5`, whose marker-coverage L2 delta remains
adapter-required while `g50t`, `wa30`, and `r11l` carry prior no-bank or
hidden-state/stalled notes. The selected target SHALL be passed through
`recommend_approach(game)`, then `scripts/arc_loop_solve.py --game <target>`,
and the terminal artifact SHALL record either a banked new reproduced level or
the residual dead-end.

Experiment 4852 SHALL write `results/experiment_4852_levelup_attempt.json` with
principle-annotated top-level fields for `honest_verdict`,
`solve_provenance`, `target_game`, `offline_reproduced`,
`reproduced_levels`, `new_levels_banked`, `inference_substrate`, and
`preconditions_checked`. It SHALL also include `rotation_selection`,
`approach_recommendation`, `attempted_games`, `dead_ends`,
`registry_update`, `retire_if_same_verdict`, `reproducibility_checksum`, and
`schema_errors`. A missing target offline environment SHALL produce
`blocked_<game>_offline_env_missing` and SHALL not fabricate reproduced levels.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; banked is success_<game>_levelup_banked, no-bank is complete_<game>_no_new_level_residual_<cause>."
- `solve_provenance`: principle "live_agent_self_discovery -- the agent solved via its own attempts/RE; NOT outer_loop_re (CRITICAL)."
- `target_game`: principle "the rotated target (must differ from .446 ka59) so coverage sweeps the corpus."
- `offline_reproduced`: principle "only reproduced levels count toward reproducible_total_levels."
- `reproduced_levels`: principle "the new reproducible depth; the monotonic ARC progress metric."
- `new_levels_banked`: principle ">=1 for a PASS; 0 records the rotation dead-end for the next planner."
- `inference_substrate`: principle "live_llm_inference if induction runs (60s floor)."
- `preconditions_checked`: principle "records arcade/env/generator checks; a missing resource emits blocked_, never a fabricated solve."

#### SCENARIO-ARC-WMTE-4852-ROTATED-TARGET

Given the ARC registry records `g50t`, `s5i5`, `wa30`, and `r11l` as L1-only
and records `.446`'s `ka59` as the previous target
When Experiment 4852 selects a target
Then it excludes `ka59`, audits the shallow candidates' dead-end notes, selects
`s5i5` for the grounded marker-coverage adapter-required delta, and records
the `recommend_approach(s5i5)` payload.

#### SCENARIO-ARC-WMTE-4852-REPRODUCTION-GATE

Given the standing loop result for the selected target does not report an
offline-reproduced depth greater than the registry prior
When Experiment 4852 builds its terminal artifact
Then `new_levels_banked=0`, `offline_reproduced=false`,
`reproduced_levels=0`, `retire_if_same_verdict=true`, and the dead-end entry
records the `needs_per_game_RE` or no-bank residual instead of incrementing the
registry total.

#### SCENARIO-ARC-WMTE-4852-STABLE-ARTIFACT

Given a selected target, preconditions, recommendation, and standing-loop
result
When Experiment 4852 writes `results/experiment_4852_levelup_attempt.json`
Then the checksum is deterministic, `schema_errors=[]`, and every required
principle-annotated field is present.

### REQ-ARC-WMTE-4862: Rotated ARC Level-Up Attempt Guarantee

Experiment 4862 SHALL run the standing ARC solve loop on a rotated target that
is not `.447`'s `s5i5` and not `.446`'s `ka59`. It SHALL precheck
`arc_solver_kit.offline_arcade()`, read `ops/arc_solve_registry.yaml`, inspect
the shallow L1-only candidates `g50t`, `wa30`, and `r11l`, and use
`recommend_approach(game)` plus registry dead-end notes to avoid known
no-grounded-delta walls. The preferred target for the 2026-06-27 registry SHALL
be `r11l`: `g50t` is skipped for prior target-offset/clone-replay no-bank
notes, `wa30` is skipped for hidden-state-bound notes, and `r11l` has a
grounded click-template handle-average L2 delta derived from the offline
environment and replay seed.

Experiment 4862 SHALL register any needed `GameAdapter` delta, run
`.venv/bin/python scripts/arc_loop_solve.py --game r11l`, reproduction-gate the
result through `arc_solver_kit.reproduce`, update
`ops/arc_solve_registry.yaml` only when the gate reports a depth above the prior
registry level, and write `results/experiment_4862_levelup_attempt.json`.

The artifact SHALL include principle-annotated top-level fields for
`honest_verdict`, `solve_provenance`, `target_game`, `offline_reproduced`,
`reproduced_levels`, `new_levels_banked`, `inference_substrate`, and
`preconditions_checked`. It SHALL also include `rotation_selection`,
`approach_recommendation`, `attempted_games`, `dead_ends`, `registry_update`,
`retire_if_same_verdict`, `reproducibility_checksum`, and `schema_errors`. A
missing offline arcade or target environment SHALL produce
`blocked_<game>_offline_env_missing` and SHALL not fabricate reproduced levels.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; banked is success_<game>_levelup_banked, no-bank is complete_<game>_no_new_level_residual_<cause>."
- `solve_provenance`: principle "live_agent_self_discovery -- the agent solved via its own attempts/RE; NOT outer_loop_re (CRITICAL)."
- `target_game`: principle "the rotated target (must differ from .447 s5i5 and .446 ka59) so coverage sweeps the corpus."
- `offline_reproduced`: principle "only reproduced levels count toward reproducible_total_levels."
- `reproduced_levels`: principle "the new reproducible depth; the monotonic ARC progress metric."
- `new_levels_banked`: principle ">=1 for a PASS; 0 records the rotation dead-end for the next planner."
- `inference_substrate`: principle "live_llm_inference if induction runs (60s floor)."
- `preconditions_checked`: principle "records arcade/env/generator checks; a missing resource emits blocked_, never a fabricated solve."

#### SCENARIO-ARC-WMTE-4862-ROTATED-TARGET

Given the ARC registry records `g50t`, `wa30`, and `r11l` as L1-only and
records `.447`'s `s5i5` and `.446`'s `ka59` as excluded recent targets
When Experiment 4862 selects a target
Then it excludes `s5i5` and `ka59`, audits shallow candidates' dead-end notes,
selects `r11l` for the grounded click-template L2 handle-average delta, and
records the `recommend_approach(r11l)` payload.

#### SCENARIO-ARC-WMTE-4862-REPRODUCTION-GATE

Given the standing loop result for `r11l` reports an offline-reproduced depth
greater than the registry prior
When Experiment 4862 builds its terminal artifact
Then `new_levels_banked>=1`, `offline_reproduced=true`,
`reproduced_levels>=prior+1`, the registry total increases only by the new
banked levels, and `solve_provenance=live_agent_self_discovery` is preserved in
the artifact.

#### SCENARIO-ARC-WMTE-4862-STABLE-ARTIFACT

Given a selected target, preconditions, recommendation, and standing-loop
result
When Experiment 4862 writes `results/experiment_4862_levelup_attempt.json`
Then the checksum is deterministic, `schema_errors=[]`, every required
principle-annotated field is present, and same-depth or failed reproduction
attempts emit a terminal no-bank verdict without incrementing the registry
total.

### REQ-ARC-WMTE-4969: Fresh Deep ARC Level-Up Attempt Ledger

Experiment 4969 SHALL run the standing ARC solve loop on one rotated fresh deep
target, preferring `tu93` from the registry's reproduced L5 row and otherwise
falling back to `tn36` L7 or `cn04` L3 only when the preferred target is blocked.
The workflow SHALL read `ops/arc_solve_registry.yaml`, call
`recommend_approach(game)` before any search, exclude recently attempted or
hidden-state-bound lanes, and compute the standing-loop target level as
`prior_reproduced_levels + 1` so it cannot re-solve an already banked level.
If the selected registry row has no grounded next-level delta or the standing
loop fails to offline-reproduce a strictly deeper level, the experiment SHALL
write an honest no-bank artifact and record the residual dead-end without
incrementing the registry total.

Experiment 4969 SHALL write `results/experiment_4969_levelup_attempt.json` with
principle-annotated top-level fields for `honest_verdict`,
`solve_provenance`, `target_game`, `offline_reproduced`,
`reproduced_levels`, `new_levels_banked`, `live_path_reachable`,
`verifier_is_oracle`, `inference_substrate`, `preconditions_checked`,
`random_seed`, and `reproducibility_checksum`. It SHALL also include
`rotation_selection`, `approach_recommendation`, `attempted_games`,
`dead_ends`, `registry_update`, `retire_if_same_verdict`, `loop_command`,
`loop_artifact`, and `schema_errors`.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; banked is success_<game>_levelup_banked, no-bank is complete_<game>_no_new_level_residual_<cause>."
- `solve_provenance`: principle "live_agent_self_discovery -- the agent advanced via its own attempts/runtime RE; NOT outer_loop_re (CRITICAL) and NOT a re-solve of an already-banked level (duplicate CRITICAL)."
- `target_game`: principle "the rotated FRESH grounded DEEP deepen target (differs from .457 tr87/s5i5 / .456 ar25/vc33 AND from A2's and A3's targets)."
- `offline_reproduced`: principle "only reproduced levels count toward reproducible_total_levels."
- `reproduced_levels`: principle "the new reproducible depth; the monotonic ARC progress metric."
- `new_levels_banked`: principle ">=1 for a PASS; 0 records the honest rotation dead-end for the next planner (the deepen well is dry across all regimes)."
- `live_path_reachable`: principle "true -- the deepen runs through arc_loop_solve + a live GameAdapter (arc_orphan_solver_lint passes), not a parallel solver."
- `verifier_is_oracle`: principle "the reproduction gate is the executable oracle (circularity discipline)."
- `inference_substrate`: principle "live_llm_inference if induction runs (60s floor); else verifier_ensemble_against_cached_candidates / the honest offline arcade substrate."
- `preconditions_checked`: principle "records arcade/env/generator checks; a missing resource emits blocked_, never a fabricated solve."
- `random_seed`: principle "determinism for the offline search."
- `reproducibility_checksum`: principle "content hash of (game, plan, claimed level) so a replication catches drift."

#### SCENARIO-ARC-WMTE-4969-FRESH-DEEP-TARGET

Given the registry contains `tu93` at L5, `tn36` at L7, and `cn04` at L3
When Experiment 4969 selects its target
Then it selects `tu93`, records the `recommend_approach(tu93)` payload, excludes
the recent and hidden-state-bound lanes, and sets the standing-loop target level
to 6 rather than the script's lower default.

#### SCENARIO-ARC-WMTE-4969-REPRODUCTION-GATE

Given the standing loop result reports a reproduced depth greater than the
registry prior
When Experiment 4969 builds its terminal artifact
Then `new_levels_banked>=1`, `offline_reproduced=true`,
`reproduced_levels>=prior+1`, and `solve_provenance=live_agent_self_discovery`.
Otherwise it emits `complete_<game>_no_new_level_residual_<cause>`,
`new_levels_banked=0`, `retire_if_same_verdict=true`, and a dead-end record
without incrementing the registry total.

#### SCENARIO-ARC-WMTE-4969-STABLE-ARTIFACT

Given a selected target, preconditions, recommendation, standing-loop result,
and registry prior
When Experiment 4969 writes `results/experiment_4969_levelup_attempt.json`
Then the checksum is deterministic, `schema_errors=[]`, every required
principle-annotated field is present, and missing offline environment
preconditions produce `blocked_<game>_offline_env_missing` without fabricated
reproduced levels.

### REQ-ARC-WMTE-4853: ARC Self-Play Verifier Checkpoint Refresh Artifact

Experiment 4853 SHALL run the standing ARC self-play loop on a banked target
with an existing learned verifier checkpoint, warm-start from that checkpoint,
reproduction-gate at least one offline level, train on this run's self-play
traces, refresh the checkpoint, and write
`results/experiment_4853_self_play_verifier_checkpoint.json`.

The success gate SHALL be falsifiable: `verifier_checkpoint_refreshed=true`
only when the checkpoint mtime advances, `offline_reproduced=true`, and
`reproduced_levels >= 1`. A failed loop, missing checkpoint, stale mtime, or
failed reproduction gate SHALL write a terminal `complete_*` or `blocked_*`
artifact with the residual instead of fabricating checkpoint progress.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; refreshed + gate green is success_self_play_checkpoint_refreshed."
- `verifier_checkpoint_refreshed`: principle "the self-improvement signal -- the learned verifier trained on this run's traces (FR-11)."
- `checkpoint_path`: principle "the models/arc_verifier_<game>.json path (mirror-ready per decentralization Rule 3)."
- `offline_reproduced`: principle "reproduction gate must pass on >=1 level for the checkpoint to be a real self-play artifact."
- `reproduced_levels`: principle "the depth confirmed this run."
- `inference_substrate`: principle "live_llm_inference (60s floor)."
- `solve_provenance`: principle "live_agent_self_discovery -- self-play is the agent improving on its own attempts."
- `preconditions_checked`: principle "records arcade/registry/checkpoint checks; a missing target emits blocked_, never a fabricated checkpoint."

#### SCENARIO-ARC-WMTE-4853-CHECKPOINT-REFRESHED

Given `arc_solver_kit.offline_arcade()` is available, the selected target is
banked in `ops/arc_solve_registry.yaml`, and
`models/arc_verifier_<game>.json` exists before the standing loop runs
When `scripts/arc_loop_solve.py --game <banked-target>` reports an
offline-reproduced level and writes the learned verifier checkpoint
Then Experiment 4853 records `verifier_checkpoint_refreshed=true`,
`checkpoint_path`, advanced checkpoint mtimes, `offline_reproduced=true`,
`reproduced_levels >= 1`, `solve_provenance=live_agent_self_discovery`,
and a terminal success verdict.

#### SCENARIO-ARC-WMTE-4853-RESIDUAL-NO-FABRICATION

Given the loop result is missing, the reproduction gate fails, the checkpoint
path is absent, or the checkpoint mtime does not advance
When Experiment 4853 builds its artifact
Then it records `verifier_checkpoint_refreshed=false`, preserves the residual
cause, keeps `offline_reproduced` and `reproduced_levels` grounded in the
loop result, and does not emit a success verdict.

#### SCENARIO-ARC-WMTE-4853-BLOCKED-PRECONDITION

Given the offline arcade is unavailable, the registry is missing, the selected
target is not banked, or the checkpoint file is absent before the run
When Experiment 4853 checks preconditions
Then it writes `results/experiment_4853_self_play_verifier_checkpoint.json`
with a terminal `blocked_*` verdict, explicit `preconditions_checked`, and no
fabricated checkpoint refresh.

### REQ-ARC-WMTE-4863: ARC Self-Play Verifier Checkpoint Refresh Artifact

Experiment 4863 SHALL run the standing ARC self-play loop on a banked target
with an existing learned verifier checkpoint, warm-start from that checkpoint,
reproduction-gate at least one offline level, train on this run's self-play
traces, refresh the checkpoint, and write
`results/experiment_4863_self_play_verifier_checkpoint.json`.

The success gate SHALL be falsifiable: `verifier_checkpoint_refreshed=true`
only when the checkpoint mtime advances, `offline_reproduced=true`, and
`reproduced_levels >= 1`. A failed loop, missing checkpoint, stale mtime, or
failed reproduction gate SHALL write a terminal `complete_*` or `blocked_*`
artifact with the residual instead of fabricating checkpoint progress.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; refreshed + gate green is success_self_play_checkpoint_refreshed."
- `verifier_checkpoint_refreshed`: principle "the self-improvement signal -- the learned verifier trained on this run's traces (FR-11)."
- `checkpoint_path`: principle "the models/arc_verifier_<game>.json path (mirror-ready per decentralization Rule 3)."
- `offline_reproduced`: principle "reproduction gate must pass on >=1 level for the checkpoint to be a real self-play artifact."
- `reproduced_levels`: principle "the depth confirmed this run."
- `inference_substrate`: principle "live_llm_inference (60s floor)."
- `solve_provenance`: principle "live_agent_self_discovery -- self-play is the agent improving on its own attempts."
- `preconditions_checked`: principle "records arcade/registry/checkpoint checks; a missing target emits blocked_, never a fabricated checkpoint."

#### SCENARIO-ARC-WMTE-4863-CHECKPOINT-REFRESHED

Given `arc_solver_kit.offline_arcade()` is available, the selected target is
banked in `ops/arc_solve_registry.yaml`, and
`models/arc_verifier_<game>.json` exists before the standing loop runs
When `scripts/arc_loop_solve.py --game <banked-target>` reports an
offline-reproduced level and writes the learned verifier checkpoint
Then Experiment 4863 records `verifier_checkpoint_refreshed=true`,
`checkpoint_path`, advanced checkpoint mtimes, `offline_reproduced=true`,
`reproduced_levels >= 1`, `solve_provenance=live_agent_self_discovery`,
and a terminal success verdict.

#### SCENARIO-ARC-WMTE-4863-RESIDUAL-NO-FABRICATION

Given the loop result is missing, the reproduction gate fails, the checkpoint
path is absent, or the checkpoint mtime does not advance
When Experiment 4863 builds its artifact
Then it records `verifier_checkpoint_refreshed=false`, preserves the residual
cause, keeps `offline_reproduced` and `reproduced_levels` grounded in the
loop result, and does not emit a success verdict.

#### SCENARIO-ARC-WMTE-4863-BLOCKED-PRECONDITION

Given the offline arcade is unavailable, the registry is missing, the selected
target is not banked, or the checkpoint file is absent before the run
When Experiment 4863 checks preconditions
Then it writes `results/experiment_4863_self_play_verifier_checkpoint.json`
with a terminal `blocked_*` verdict, explicit `preconditions_checked`, and no
fabricated checkpoint refresh.

### REQ-ARC-WMTE-4874: ARC Continuous Self-Play Verifier Checkpoint Refresh Artifact

Experiment 4874 SHALL run the standing ARC self-play loop on a banked target
with an existing learned verifier checkpoint, warm-start from that checkpoint,
reproduction-gate at least one offline level, train on this run's self-play
traces, refresh the checkpoint, and write
`results/experiment_4874_self_play_verifier_checkpoint.json`.

The preconditions SHALL verify that `arc_solver_kit.offline_arcade()` exits 0,
that `ops/arc_solve_registry.yaml` is loadable, that the selected target has a
banked level, and that `models/arc_verifier_<game>.json` exists before the run.
If any required resource is missing, Experiment 4874 SHALL write a terminal
`blocked_*` artifact that records the failed precondition and SHALL NOT claim a
refreshed checkpoint.

The success gate SHALL be falsifiable: `verifier_checkpoint_refreshed=true`
only when the checkpoint mtime advances, `offline_reproduced=true`, and
`reproduced_levels >= 1`. A failed loop, missing checkpoint, stale mtime, or
failed reproduction gate SHALL write a terminal `complete_*` or `blocked_*`
artifact with the residual instead of fabricating checkpoint progress.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; refreshed + gate green is success_self_play_checkpoint_refreshed."
- `verifier_checkpoint_refreshed`: principle "the self-improvement signal -- the learned verifier trained on this run's traces (FR-11)."
- `checkpoint_path`: principle "the models/arc_verifier_<game>.json path (mirror-ready per decentralization Rule 3)."
- `offline_reproduced`: principle "reproduction gate must pass on >=1 level for the checkpoint to be a real self-play artifact."
- `reproduced_levels`: principle "the depth confirmed this run."
- `inference_substrate`: principle "live_llm_inference (60s floor)."
- `solve_provenance`: principle "live_agent_self_discovery -- self-play is the agent improving on its own attempts."
- `preconditions_checked`: principle "records arcade/registry/checkpoint checks; a missing target emits blocked_, never a fabricated checkpoint."

#### SCENARIO-ARC-WMTE-4874-CHECKPOINT-REFRESHED

Given `arc_solver_kit.offline_arcade()` is available, the selected target is
banked in `ops/arc_solve_registry.yaml`, and
`models/arc_verifier_<game>.json` exists before the standing loop runs
When `scripts/arc_loop_solve.py --game <banked-target>` reports an
offline-reproduced level and writes the learned verifier checkpoint
Then Experiment 4874 records `verifier_checkpoint_refreshed=true`,
`checkpoint_path`, advanced checkpoint mtimes, `offline_reproduced=true`,
`reproduced_levels >= 1`, `solve_provenance=live_agent_self_discovery`,
and `honest_verdict=success_self_play_checkpoint_refreshed`.

#### SCENARIO-ARC-WMTE-4874-RESIDUAL-NO-FABRICATION

Given the loop result is missing, the reproduction gate fails, the checkpoint
path is absent, or the checkpoint mtime does not advance
When Experiment 4874 builds its artifact
Then it records `verifier_checkpoint_refreshed=false`, preserves the residual
cause, keeps `offline_reproduced` and `reproduced_levels` grounded in the
loop result, and does not emit a success verdict.

#### SCENARIO-ARC-WMTE-4874-BLOCKED-PRECONDITION

Given the offline arcade is unavailable, the registry is missing, the selected
target is not banked, or the checkpoint file is absent before the run
When Experiment 4874 checks preconditions
Then it writes `results/experiment_4874_self_play_verifier_checkpoint.json`
with a terminal `blocked_*` verdict, explicit `preconditions_checked`, and no
fabricated checkpoint refresh.

### REQ-ARC-WMTE-4885: Rotated ARC Continuous Self-Play Verifier Checkpoint Refresh

Experiment 4885 SHALL run the standing ARC self-play loop on a rotated banked
target with an existing learned verifier checkpoint, preferring `ls20` so the
continuous self-learning checkpoint task rotates off the prior `re86` target.
The loop SHALL warm-start from `models/arc_verifier_ls20.json`, reproduction-gate
at least one offline level, train on this run's self-play traces, refresh the
checkpoint, and write
`results/experiment_4885_self_play_verifier_checkpoint.json`.

The preconditions SHALL verify that `arc_solver_kit.offline_arcade()` exits 0,
that `ops/arc_solve_registry.yaml` is loadable, that the selected target has a
banked level, and that `models/arc_verifier_<game>.json` exists before the run.
If any required resource is missing, Experiment 4885 SHALL write a terminal
`blocked_*` artifact that records the failed precondition and SHALL NOT claim a
refreshed checkpoint.

The success gate SHALL be falsifiable: `verifier_checkpoint_refreshed=true`
only when the checkpoint mtime advances, `offline_reproduced=true`, and
`reproduced_levels >= 1`. A failed loop, missing checkpoint, stale mtime, or
failed reproduction gate SHALL write a terminal `complete_*` or `blocked_*`
artifact with the residual instead of fabricating checkpoint progress.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; refreshed + gate green is success_self_play_checkpoint_refreshed."
- `verifier_checkpoint_refreshed`: principle "the self-improvement signal -- the learned verifier trained on this run's traces (FR-11)."
- `checkpoint_path`: principle "the models/arc_verifier_<game>.json path (mirror-ready per decentralization Rule 3)."
- `offline_reproduced`: principle "reproduction gate must pass on >=1 level for the checkpoint to be a real self-play artifact."
- `reproduced_levels`: principle "the depth confirmed this run."
- `target_game`: principle "the rotated self-play target (differs from .449 re86)."
- `inference_substrate`: principle "live_llm_inference (60s floor)."
- `solve_provenance`: principle "live_agent_self_discovery -- self-play is the agent improving on its own attempts."
- `preconditions_checked`: principle "records arcade/registry/checkpoint checks; a missing target emits blocked_, never a fabricated checkpoint."

#### SCENARIO-ARC-WMTE-4885-CHECKPOINT-REFRESHED

Given `arc_solver_kit.offline_arcade()` is available, `ls20` is banked in
`ops/arc_solve_registry.yaml`, and `models/arc_verifier_ls20.json` exists before
the standing loop runs
When `scripts/arc_loop_solve.py --game ls20` reports an offline-reproduced
level and writes the learned verifier checkpoint
Then Experiment 4885 records `verifier_checkpoint_refreshed=true`,
`checkpoint_path=models/arc_verifier_ls20.json`, advanced checkpoint mtimes,
`offline_reproduced=true`, `reproduced_levels >= 1`, `target_game=ls20`,
`solve_provenance=live_agent_self_discovery`, and
`honest_verdict=success_self_play_checkpoint_refreshed`.

#### SCENARIO-ARC-WMTE-4885-RESIDUAL-NO-FABRICATION

Given the loop result is missing, the reproduction gate fails, the checkpoint
path is absent, or the checkpoint mtime does not advance
When Experiment 4885 builds its artifact
Then it records `verifier_checkpoint_refreshed=false`, preserves the residual
cause, keeps `offline_reproduced` and `reproduced_levels` grounded in the
loop result, and does not emit a success verdict.

#### SCENARIO-ARC-WMTE-4885-BLOCKED-PRECONDITION

Given the offline arcade is unavailable, the registry is missing, the selected
target is not banked, the checkpoint file is absent before the run, or the
selection is the prior `re86` target
When Experiment 4885 checks preconditions
Then it writes `results/experiment_4885_self_play_verifier_checkpoint.json`
with a terminal `blocked_*` verdict, explicit `preconditions_checked`, and no
fabricated checkpoint refresh.

### REQ-ARC-WMTE-4895: Rotated ARC Continuous Self-Play Verifier Checkpoint Refresh

Experiment 4895 SHALL run the standing ARC self-play loop on a rotated banked
target with an existing learned verifier checkpoint, selecting a target that is
neither `.450`'s `ls20` nor `.449`'s `re86`. The loop SHALL warm-start from
`models/arc_verifier_<game>.json`, reproduction-gate at least one offline
level, train on this run's self-play traces, refresh the checkpoint, and write
`results/experiment_4895_self_play_verifier_checkpoint.json`.

The preconditions SHALL verify that `arc_solver_kit.offline_arcade()` exits 0,
that `ops/arc_solve_registry.yaml` is loadable, that the selected target has a
banked level, and that `models/arc_verifier_<game>.json` exists before the run.
If any required resource is missing, Experiment 4895 SHALL write a terminal
`blocked_*` artifact that records the failed precondition and SHALL NOT claim a
refreshed checkpoint.

The success gate SHALL be falsifiable: `verifier_checkpoint_refreshed=true`
only when the checkpoint mtime advances, `offline_reproduced=true`, and
`reproduced_levels >= 1`. A failed loop, missing checkpoint, stale mtime,
failed reproduction gate, or rotation back to `ls20` or `re86` SHALL write a
terminal `complete_*` or `blocked_*` artifact with the residual instead of
fabricating checkpoint progress.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; refreshed + gate green is success_self_play_checkpoint_refreshed."
- `verifier_checkpoint_refreshed`: principle "the self-improvement signal -- the learned verifier trained on this run's traces (FR-11)."
- `checkpoint_path`: principle "the models/arc_verifier_<game>.json path (mirror-ready per decentralization Rule 3)."
- `offline_reproduced`: principle "reproduction gate must pass on >=1 level for the checkpoint to be a real self-play artifact."
- `reproduced_levels`: principle "the depth confirmed this run."
- `target_game`: principle "the rotated self-play target (differs from .450 ls20 / .449 re86)."
- `inference_substrate`: principle "live_llm_inference (60s floor)."
- `solve_provenance`: principle "live_agent_self_discovery -- self-play is the agent improving on its own attempts."
- `preconditions_checked`: principle "records arcade/registry/checkpoint checks; a missing target emits blocked_, never a fabricated checkpoint."

#### SCENARIO-ARC-WMTE-4895-CHECKPOINT-REFRESHED

Given `arc_solver_kit.offline_arcade()` is available, a non-`ls20`/non-`re86`
target is banked in `ops/arc_solve_registry.yaml`, and
`models/arc_verifier_<game>.json` exists before the standing loop runs
When `scripts/arc_loop_solve.py --game <banked-target>` reports an
offline-reproduced level and writes the learned verifier checkpoint
Then Experiment 4895 records `verifier_checkpoint_refreshed=true`,
`checkpoint_path=models/arc_verifier_<game>.json`, advanced checkpoint mtimes,
`offline_reproduced=true`, `reproduced_levels >= 1`, the rotated
`target_game`, `solve_provenance=live_agent_self_discovery`, and
`honest_verdict=success_self_play_checkpoint_refreshed`.

#### SCENARIO-ARC-WMTE-4895-RESIDUAL-NO-FABRICATION

Given the loop result is missing, the reproduction gate fails, the checkpoint
path is absent, or the checkpoint mtime does not advance
When Experiment 4895 builds its artifact
Then it records `verifier_checkpoint_refreshed=false`, preserves the residual
cause, keeps `offline_reproduced` and `reproduced_levels` grounded in the
loop result, and does not emit a success verdict.

#### SCENARIO-ARC-WMTE-4895-BLOCKED-PRECONDITION

Given the offline arcade is unavailable, the registry is missing, the selected
target is not banked, the checkpoint file is absent before the run, or the
selection is either recent self-play target `ls20` or `re86`
When Experiment 4895 checks preconditions
Then it writes `results/experiment_4895_self_play_verifier_checkpoint.json`
with a terminal `blocked_*` verdict, explicit `preconditions_checked`, and no
fabricated checkpoint refresh.

### REQ-ARC-WMTE-4906: Rotated ARC Continuous Self-Play Verifier Checkpoint Refresh

Experiment 4906 SHALL run the standing ARC self-play loop on a rotated banked
target with an existing learned verifier checkpoint, selecting a target that is
not `.451`'s `sk48`, `.450`'s `ls20`, or `.449`'s `re86`. The loop SHALL
warm-start from `models/arc_verifier_<game>.json`, reproduction-gate at least
one offline level, train on this run's self-play traces, refresh the checkpoint,
and write `results/experiment_4906_self_play_verifier_checkpoint.json`.

The preconditions SHALL verify that `arc_solver_kit.offline_arcade()` exits 0,
that `ops/arc_solve_registry.yaml` is loadable, that the selected target has a
banked level, and that `models/arc_verifier_<game>.json` exists before the run.
If any required resource is missing, Experiment 4906 SHALL write a terminal
`blocked_*` artifact that records the failed precondition and SHALL NOT claim a
refreshed checkpoint.

The success gate SHALL be falsifiable: `verifier_checkpoint_refreshed=true`
only when the checkpoint mtime advances, `offline_reproduced=true`, and
`reproduced_levels >= 1`. A failed loop, missing checkpoint, stale mtime,
failed reproduction gate, or rotation back to `sk48`, `ls20`, or `re86` SHALL
write a terminal `complete_*` or `blocked_*` artifact with the residual instead
of fabricating checkpoint progress.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; refreshed + gate green is success_self_play_checkpoint_refreshed."
- `verifier_checkpoint_refreshed`: principle "the self-improvement signal -- the learned verifier trained on this run's traces (FR-11)."
- `checkpoint_path`: principle "the models/arc_verifier_<game>.json path (mirror-ready per decentralization Rule 3)."
- `offline_reproduced`: principle "reproduction gate must pass on >=1 level for the checkpoint to be a real self-play artifact."
- `reproduced_levels`: principle "the depth confirmed this run."
- `target_game`: principle "the rotated self-play target (differs from .451 sk48 / .450 ls20 / .449 re86)."
- `inference_substrate`: principle "live_llm_inference (60s floor)."
- `solve_provenance`: principle "live_agent_self_discovery -- self-play is the agent improving on its own attempts."
- `preconditions_checked`: principle "records arcade/registry/checkpoint checks; a missing target emits blocked_, never a fabricated checkpoint."

#### SCENARIO-ARC-WMTE-4906-CHECKPOINT-REFRESHED

Given `arc_solver_kit.offline_arcade()` is available, a non-`sk48`/non-`ls20`/non-`re86`
target is banked in `ops/arc_solve_registry.yaml`, and
`models/arc_verifier_<game>.json` exists before the standing loop runs
When `scripts/arc_loop_solve.py --game <banked-target>` reports an
offline-reproduced level and writes the learned verifier checkpoint
Then Experiment 4906 records `verifier_checkpoint_refreshed=true`,
`checkpoint_path=models/arc_verifier_<game>.json`, advanced checkpoint mtimes,
`offline_reproduced=true`, `reproduced_levels >= 1`, the rotated
`target_game`, `solve_provenance=live_agent_self_discovery`, and
`honest_verdict=success_self_play_checkpoint_refreshed`.

#### SCENARIO-ARC-WMTE-4906-RESIDUAL-NO-FABRICATION

Given the loop result is missing, the reproduction gate fails, the checkpoint
path is absent, or the checkpoint mtime does not advance
When Experiment 4906 builds its artifact
Then it records `verifier_checkpoint_refreshed=false`, preserves the residual
cause, keeps `offline_reproduced` and `reproduced_levels` grounded in the
loop result, and does not emit a success verdict.

#### SCENARIO-ARC-WMTE-4906-BLOCKED-PRECONDITION

Given the offline arcade is unavailable, the registry is missing, the selected
target is not banked, the checkpoint file is absent before the run, or the
selection is one of the recent self-play targets `sk48`, `ls20`, or `re86`
When Experiment 4906 checks preconditions
Then it writes `results/experiment_4906_self_play_verifier_checkpoint.json`
with a terminal `blocked_*` verdict, explicit `preconditions_checked`, and no
fabricated checkpoint refresh.

### REQ-ARC-WMTE-4916: Rotated ARC Continuous Self-Play Verifier Checkpoint Refresh

Experiment 4916 SHALL run the standing ARC self-play loop on a rotated banked
target with an existing learned verifier checkpoint, selecting a target that is
not `.452`'s `vc33`, `.451`'s `sk48`, `.450`'s `ls20`, `.449`'s `re86`, and
not A2's level-up target `cn04`. The loop SHALL warm-start from
`models/arc_verifier_<game>.json`, reproduction-gate at least one offline
level, train on this run's self-play traces, refresh the checkpoint, and write
`results/experiment_4916_self_play_verifier_checkpoint.json`.

The preconditions SHALL verify that `arc_solver_kit.offline_arcade()` exits 0,
that `ops/arc_solve_registry.yaml` is loadable, that the selected target has a
banked level, and that `models/arc_verifier_<game>.json` exists before the run.
If any required resource is missing, Experiment 4916 SHALL write a terminal
`blocked_*` artifact that records the failed precondition and SHALL NOT claim a
refreshed checkpoint.

The success gate SHALL be falsifiable: `verifier_checkpoint_refreshed=true`
only when the checkpoint mtime advances, `offline_reproduced=true`, and
`reproduced_levels >= 1`. A failed loop, missing checkpoint, stale mtime,
failed reproduction gate, or rotation back to `vc33`, `sk48`, `ls20`, `re86`,
or `cn04` SHALL write a terminal `complete_*` or `blocked_*` artifact with the
residual instead of fabricating checkpoint progress.

Required field principles SHALL include:

- `honest_verdict`: principle "terminal prefix; refreshed + gate green is success_self_play_checkpoint_refreshed."
- `verifier_checkpoint_refreshed`: principle "the self-improvement signal -- the learned verifier trained on this run's traces (FR-11)."
- `checkpoint_path`: principle "the models/arc_verifier_<game>.json path (mirror-ready per decentralization Rule 3)."
- `offline_reproduced`: principle "reproduction gate must pass on >=1 level for the checkpoint to be a real self-play artifact."
- `reproduced_levels`: principle "the depth confirmed this run."
- `target_game`: principle "the rotated self-play target (differs from .452 vc33 / .451 sk48 / .450 ls20 / .449 re86 AND from A2's target)."
- `inference_substrate`: principle "live_llm_inference (60s floor)."
- `solve_provenance`: principle "live_agent_self_discovery -- self-play is the agent improving on its own attempts."
- `preconditions_checked`: principle "records arcade/registry/checkpoint checks; a missing target emits blocked_, never a fabricated checkpoint."

#### SCENARIO-ARC-WMTE-4916-CHECKPOINT-REFRESHED

Given `arc_solver_kit.offline_arcade()` is available, a rotated target outside
`vc33`/`sk48`/`ls20`/`re86`/`cn04` is banked in
`ops/arc_solve_registry.yaml`, and `models/arc_verifier_<game>.json` exists
before the standing loop runs
When `scripts/arc_loop_solve.py --game <banked-target>` reports an
offline-reproduced level and writes the learned verifier checkpoint
Then Experiment 4916 records `verifier_checkpoint_refreshed=true`,
`checkpoint_path=models/arc_verifier_<game>.json`, advanced checkpoint mtimes,
`offline_reproduced=true`, `reproduced_levels >= 1`, the rotated
`target_game`, `solve_provenance=live_agent_self_discovery`, and
`honest_verdict=success_self_play_checkpoint_refreshed`.

#### SCENARIO-ARC-WMTE-4916-RESIDUAL-NO-FABRICATION

Given the loop result is missing, the reproduction gate fails, the checkpoint
path is absent, or the checkpoint mtime does not advance
When Experiment 4916 builds its artifact
Then it records `verifier_checkpoint_refreshed=false`, preserves the residual
cause, keeps `offline_reproduced` and `reproduced_levels` grounded in the
loop result, and does not emit a success verdict.

#### SCENARIO-ARC-WMTE-4916-BLOCKED-PRECONDITION

Given the offline arcade is unavailable, the registry is missing, the selected
target is not banked, the checkpoint file is absent before the run, or the
selection is one of the recent or conflicting targets `vc33`, `sk48`, `ls20`,
`re86`, or `cn04`
When Experiment 4916 checks preconditions
Then it writes `results/experiment_4916_self_play_verifier_checkpoint.json`
with a terminal `blocked_*` verdict, explicit `preconditions_checked`, and no
fabricated checkpoint refresh.
