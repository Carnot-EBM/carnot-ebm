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
