# ARC-AGI Live Agent Capability Specification

**Capability:** arc-agi
**Version:** 0.1.0
**Status:** Draft

## Overview

Defines ARC-AGI-3 live-agent requirements that are not level-solve claims. The
agent must use its own visible runtime events. It must not read game source,
hidden state, offline ground-truth search, or hand adapters on the scored path.

## Requirements

### REQ-ARC-ARM-6387: Active Reward-Machine Discriminator

Experiment 6387 SHALL add a bounded, default-off reward-machine hypothesis
frontier over visible live events. Each hypothesis SHALL be a small automaton
with bounded states, visible event symbols, deterministic transitions, and
source-linked transition evidence. The mechanism SHALL use legal actions from
the live frame action set. It SHALL not mutate the action set.

The frontier SHALL choose a probe only when at least two active hypotheses make
different outcome predictions for one legal action. It SHALL score legal actions
by expected hypothesis elimination. If no legal disagreement action exists, if
evidence is late, if all predictions are unknown, or if the bounded capacity is
exhausted without a safe split, the policy SHALL abstain and defer to the
unchanged base policy.

The policy SHALL freeze the chosen action, legal action set, active hypothesis
IDs, and predictions before the environment transition is read. The resulting
transition SHALL be used only as evaluation evidence for the next step. The
frontier SHALL reject duplicate evidence, refuse contradictory all-mismatch
evidence without eliminating all hypotheses, evict deterministically at capacity,
and time out stale pending probes.

Observed transitions SHALL feed the Exp6386 two-sided goal-evidence contract.
Reward-machine evidence can guide future probes, but it SHALL NOT terminate
search, update `ops/arc_solve_registry.yaml`, or claim a game or level solve.
The feature SHALL be reachable from
`make_carnot_agent -> E3AgentPolicy`, SHALL default off in
`SUBMITTED_AGENT_CONFIG`, and the shipped default SHALL not change actions.

Experiment 6387 SHALL write
`results/experiment_6387_arc_active_reward_machine_discriminator.json` with the
required top-level fields named by the task. The artifact SHALL set
`arc_solve_claim=false`, SHALL omit `solve_provenance`, SHALL set
`verifier_is_oracle=false`, and SHALL set
`arc_active_reward_machine_ready_score=1.0` only when treatment reachability and
evidence integrity pass with zero forbidden access and zero registry writes.

### SCENARIO-ARC-ARM-6387-LEGAL-DISAGREEMENT

**Given** two to five game-blind reward-machine hypotheses over visible events
and a runtime legal action set
**When** one legal action has unique disagreement and a non-legal action would
also split hypotheses
**Then** the frontier selects only the legal disagreement action, freezes the
action and active predictions before the outcome, and records expected
hypothesis elimination without reading source, adapters, offline BFS, or hidden
state.

### SCENARIO-ARC-ARM-6387-ABSTAIN-AND-BOUNDS

**Given** no-disagreement actions, delayed evidence, repeated frames,
contradictory evidence, duplicate evidence, capacity overflow, and stale pending
probe deadlines
**When** the frontier ranks probes and ingests outcomes
**Then** it abstains when no safe split exists, deduplicates repeated evidence,
records contradictions without wrong elimination, evicts deterministically, times
out stale probes, and defers to the base policy.

### SCENARIO-ARC-ARM-6387-TWO-SIDED-EVIDENCE

**Given** a frozen reward-machine probe and the next visible environment
transition
**When** the transition is ingested after the action freeze
**Then** each active hypothesis receives a source-linked two-sided event, firing
witnesses and non-firing contrasts are evaluated by the Exp6386 contract, and
unverified or rejected hypotheses do not terminate search or earn solve credit.

### SCENARIO-ARC-ARM-6387-LIVE-DEFAULT-OFF

**Given** the normal live entrypoint
`make_carnot_agent -> E3AgentPolicy`
**When** the submitted default constructs the policy
**Then** the reward-machine feature is off, base-policy fallback is unchanged,
and enabling `CARNOT_ARC_ACTIVE_REWARD_MACHINE=1` proves reachability without a
registry write or a solve claim.

### SCENARIO-ARC-ARM-6387-ARTIFACT-NO-SOLVE

**Given** Exp6386 passed and the registry hash is captured before the run
**When** Exp6387 writes its artifact
**Then** all required fields are present, protected files are unchanged,
forbidden access counts are zero, `arc_solve_claim` is false,
`verifier_is_oracle` is false, `solve_provenance` is absent, and the registry
hash is unchanged.

### REQ-ARC-ARM-6388: Goal-Evidence Response Calibration

Experiment 6388 SHALL calibrate goal-evidence response on matched visible ARC
trajectory prefixes after the Exp6387 gate passes. The harness SHALL compare
the current gate, a frozen-prior control, passive two-sided evidence, and active
reward-machine evidence. Each arm SHALL receive matched model calls, token
capacity, trajectory exposure, deadlines, and evaluation opportunities.

The model set SHALL include `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`, resolved through the cached SOTA GGUF path.
The harness SHALL use the GGUF-embedded tokenizer path. It SHALL report zero
AutoTokenizer usage.

For each model and prefix, the harness SHALL freeze the goal hypothesis,
confidence or abstention, evidence references, and next legal probe before it
reads later transitions. Later transitions SHALL label calibration only. No arm
SHALL terminate a level, update solve credit, write the solve registry, read
hidden source, use offline search, use a GameAdapter, use an external scorer, or
read hidden state.

Experiment 6388 SHALL write
`results/experiment_6388_arc_goal_evidence_response_calibration.json` with the
required top-level fields named by the task. The artifact SHALL set
`arc_solve_claim=false`, SHALL omit `solve_provenance`, SHALL set
`verifier_is_oracle=false`, and SHALL set
`arc_evidence_calibration_ready_score=1.0` only when all models and arms have
complete receipts, all controls pass, the active treatment fires, forbidden
access counts are zero, and the registry hash is unchanged.

### SCENARIO-ARC-ARM-6388-MATCHED-PREFIXES

**Given** sealed visible frames, actions, legal sets, evidence identities,
prefix boundaries, and evaluation labels
**When** Exp6388 preregisters the current-gate, frozen-prior, passive
two-sided, and active reward-machine arms
**Then** every arm receives matched token capacity, trajectory exposure,
deadlines, and evaluation opportunities without duplicate solve targets.

### SCENARIO-ARC-ARM-6388-FROZEN-PREDICTIONS

**Given** a model, arm, and live trajectory prefix
**When** the harness records a goal hypothesis, confidence or abstention,
evidence references, and next legal probe
**Then** the prediction receipt is sealed before the later transition label is
read, and the later transition is used only for calibration.

### SCENARIO-ARC-ARM-6388-METRICS-AND-CONTROLS

**Given** accepted, rejected, unverifiable, false accept, false reject, true
accept, and true reject outcomes
**When** Exp6388 compares active evidence with the current gate
**Then** it reports precision, coverage, calibration error, monotonicity,
hypothesis elimination, response to added evidence, unrounded deltas, and the
shuffled-evidence, duplicate-evidence, surface-relabeled, no-win-window,
model-identity-blind, action-order, deadline, and result-before-prediction
controls.

### SCENARIO-ARC-ARM-6388-ARTIFACT-NO-SOLVE

**Given** the Exp6387 gate passes and the registry hash is captured before the
run
**When** Exp6388 writes its artifact
**Then** all required fields are present, `solve_provenance` is absent,
protected files are unchanged, forbidden access counts are zero,
`arc_solve_claim` is false, `verifier_is_oracle` is false, and the registry hash
is unchanged.

### REQ-ARC-ARM-6393: Scalar ARC Gate-Metric Contract

Experiment 6393 SHALL replay immutable Exp6388 row-level evidence into bare
numeric gate fields. It SHALL recompute pooled admission precision, admission
precision delta, false-accept count, and false-accept delta from frozen rows.
It SHALL NOT trust the nested Exp6388 aggregate as the source of truth.

The producer SHALL emit `delta_admission_precision_scalar` and
`delta_false_accept_count_scalar` as finite bare numbers. It SHALL keep
by-model values in separate detail fields. Mapping, list, string, bool, NaN,
infinity, rounded sign-change, missing-row, duplicate-row, stale-hash, and
model-order attacks SHALL fail closed.

Experiment 6393 SHALL write
`results/experiment_6393_arc_scalar_gate_metric_contract.json` with the required
top-level fields named by the task. The artifact SHALL set
`arc_gate_metric_contract_ready_score=1.0` only when the row replay reproduces
the V549 Exp6388 metrics, every scalar gate field is finite, all coercion
attacks fail closed, Exp6388 and Exp6389 remain unchanged, and no live route or
solve claim is made.

### SCENARIO-ARC-ARM-6393-ROW-REPLAY

**Given** immutable Exp6388 frozen prediction rows
**When** Exp6393 recomputes the active and current gate counts
**Then** active pooled admission precision is 1.0, delta admission precision is
0.75, active false accepts are 0, delta false accepts are -9, and by-model
detail rows are preserved outside the conductor scalar fields.

### SCENARIO-ARC-ARM-6393-ATTACKS-FAIL-CLOSED

**Given** malformed scalar values, non-finite floats, rounded sign changes,
missing model rows, duplicate rows, stale hashes, and model-order swaps
**When** Exp6393 validates the producer contract
**Then** each case is rejected before a ready score can be set.

### SCENARIO-ARC-ARM-6393-GATE-REPLAY

**Given** the planned Exp6400 gate predicates over Exp6393
**When** the conductor comparison function evaluates the new artifact fields
**Then** each comparison receives a finite bare number and the exact operands,
operator, result, and reason are recorded.

### REQ-ARC-ARM-6400: Default-Off Active Goal Shadow

Experiment 6400 SHALL replay the deferred Exp6389 shadow after the Exp6393
scalar gate passes. It SHALL run the active-goal mechanism as a default-off
shadow on the normal live ARC path. This default-off shadow SHALL use the live
policy's own visible frames, legal actions, executed actions, transition
records, and runtime reverse-engineering evidence. It SHALL not read game
source, hidden state, offline ground-truth search, per-game adapters, or oracle
outcomes before the action is frozen.

The model set SHALL include `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`, resolved through `cached_sota_pair()`. The
producer SHALL use only the GGUF-embedded tokenizer path and SHALL report zero
`AutoTokenizer` usage. The producer SHALL record model file hashes, revisions,
quantization, embedded tokenizer receipts, GPU offload receipts, live entrypoint
hashes, Exp6393 gate receipts, the two-sided goal contract receipt, the active
reward-machine route receipt, and ARC registry and claims precheck hashes.

The producer SHALL freeze at least six fresh live attempt windows and at least
36 visible transitions before evaluation. For each model and prefix, it SHALL
freeze the goal hypothesis, accepted / rejected / unverifiable disposition,
legal disagreement probe, and counterfactual action ranking before it reads the
next transition. Route-off and active-goal shadow cells SHALL match models,
windows, action budgets, prompt budgets, evidence prefixes, and evaluation
calls. The shadow SHALL never alter the executed action.

The producer SHALL report treatment reachability, treatment firing, goal
admission precision, false accepts, false rejects, abstention, action-ranking
differences, exact progress proxies, latency, verification cost, provenance
counts, attack probes, and protected-file status by model and window. It SHALL
emit `active_shadow_treatment_fired_count` and
`delta_shadow_false_accept_count` as bare integers. It SHALL set
`arc_active_goal_shadow_ready_score=1.0` only when the live route is reachable,
the treatment fires, matched work passes, evidence provenance is clean, false
accepts do not increase, executed actions do not change, and no solve or
registry claim occurs.

Experiment 6400 SHALL write
`results/experiment_6400_arc_default_off_active_goal_shadow.json` with the
required top-level fields named by the task. It SHALL set
`verifier_is_oracle=true` only for post-action transition checks. Goal
hypotheses, model text, and shadow action ranks SHALL not be treated as
oracles. It SHALL not update `ops/arc_solve_registry.yaml` or claim a game or
level solve.

The artifact SHALL include `status`, `exp6393_gate_receipts`, `MODEL_SPECS`,
`models_used`, `cached_sota_pair_receipts`,
`model_file_hashes_revisions_quantizations_and_tokenizers`,
`embedded_gguf_tokenizer_receipts`, `autotokenizer_usage_count`,
`cuda_offload_and_runtime_receipts_by_model`,
`live_entrypoint_policy_and_reward_machine_hashes`,
`arc_registry_and_claims_precheck_hashes`,
`fresh_live_window_manifest_path_hash_and_counts`, `live_attempt_provenance`,
`preregistered_route_off_and_shadow_contract`, `matched_work_receipts`,
`frozen_goal_probe_and_counterfactual_action_records`,
`per_model_window_admission_abstention_action_influence_progress_and_cost_results`,
`active_shadow_treatment_fired_count`, `delta_shadow_admission_precision`,
`delta_shadow_false_accept_count`, `delta_shadow_exact_progress_proxy`,
`model_row_prefix_state_goal_duplicate_budget_and_action_leakage_attack_matrix`,
`hidden_source_access_count`, `offline_ground_truth_search_count`,
`per_game_adapter_count`, `oracle_before_action_count`,
`executed_action_change_count`, `solve_claim_count`, `solve_registry_modified`,
`arc_active_goal_shadow_ready_score`, `harm_underpowered_missing_and_flagged_cells`,
`protected_files_unchanged`, `preconditions_checked`, `inference_substrate`,
`verifier_is_oracle`, `field_principles`, `field_provenance`, `random_seed`,
`duration_s`, `tests_run`, `reproducibility_checksum`, and `honest_verdict`.

### SCENARIO-ARC-ARM-6400-GATE-REPLAY

**Given** Exp6393 passed with scalar gate fields
**When** Exp6400 revalidates the deferred Exp6389 gate
**Then** every gate comparison uses a finite bare scalar, records operands and
reasons, and refuses the old nested-delta shape before live-window evaluation.

### SCENARIO-ARC-ARM-6400-MATCHED-SHADOW

**Given** at least six fresh live attempt windows and 36 visible transitions
**When** route-off and active-goal shadow cells are evaluated
**Then** both arms receive matched model ids, action budgets, prompt budgets,
evidence prefixes, and evaluation calls, while the shadow records treatment
reachability and firing without changing the executed action.

### SCENARIO-ARC-ARM-6400-FROZEN-PROBES

**Given** a model, live window, and evidence prefix
**When** the shadow records a goal hypothesis and legal disagreement probe
**Then** the hypothesis, disposition, probe, and counterfactual action rank are
sealed before the next transition label is read.

### SCENARIO-ARC-ARM-6400-ATTACKS-FAIL-CLOSED

**Given** model-row swaps, prefix truncation, stale goal state, constant-false
goals, duplicate transitions, route budget mismatch, and shadow-to-action
leakage
**When** the producer validates shadow evidence
**Then** each attack fails closed before readiness can be set.

### SCENARIO-ARC-ARM-6400-ARTIFACT-NO-SOLVE

**Given** the solve registry and claims ledger are hashed before the run
**When** Exp6400 writes its artifact
**Then** all required fields are present, forbidden access counts are zero,
`active_shadow_treatment_fired_count` and
`delta_shadow_false_accept_count` are bare integers, protected files are
unchanged, `solve_claim_count` is zero, the solve registry hash is unchanged,
and no solve claim is made.

### REQ-ARC-ARM-6401: Held Active-Goal Causal Route Test

Experiment 6401 SHALL run only after Exp6400 proves the active-goal shadow is
reachable, fires, stays default-off, and does not increase false accepts. It
SHALL compare passive two-sided evidence against active legal disagreement
probes on held live ARC attempt windows. Environment outcomes SHALL remain
hidden until each candidate goal, evidence disposition, selected probe or
passive rank, and action is frozen.

The model set SHALL include `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`, resolved through `cached_sota_pair()`. The
producer SHALL use only GGUF-embedded tokenizers and report zero
`AutoTokenizer` usage. It SHALL revalidate model files, GPU offload receipts,
live entrypoint hashes, policy hashes, reward-machine hashes, evaluator hashes,
route-disable defaults, and ARC registry and claims hashes before evaluation.

The producer SHALL seal at least eight fresh held live attempt windows and at
least 48 visible transitions. It SHALL exclude every Exp6400 window and hash a
disjointness proof before arm evaluation. Passive and active arms SHALL match
model ids, windows, seeds, action budgets, prompt budgets, evidence prefix
lengths, legal action sets, and post-action exact checks.

For every model-window cell, the producer SHALL report goal admission
precision, false accepts, false rejects, unverifiable rate, action influence,
exact progress proxies, regressions, latency, verification cost, treatment
firing, missing treatment cells, paired tests, confidence intervals, and
effective sample sizes. It SHALL not pool missing cells or abstentions as
successes.

The producer SHALL attack window reuse, action-order changes, oracle timing,
model-row swaps, goal-state carryover, unequal legal sets, unequal budgets,
duplicate transitions, and solve-label leakage. Each attack SHALL fail closed
before readiness can be set.

Experiment 6401 SHALL write
`results/experiment_6401_arc_active_goal_causal_holdout.json` with the required
top-level fields named by the task. It SHALL set `verifier_is_oracle=true` only
for post-action environment transition checks. It SHALL not read hidden source,
use offline ground-truth search, use per-game adapters, read oracle outcomes
before action freeze, update `ops/arc_solve_registry.yaml`, or claim a game or
level solve.

The artifact SHALL include `status`, `exp6400_gate_receipts`, `MODEL_SPECS`,
`models_used`, `cached_sota_pair_receipts`,
`model_file_hashes_revisions_quantizations_and_tokenizers`,
`embedded_gguf_tokenizer_receipts`, `autotokenizer_usage_count`,
`cuda_offload_and_runtime_receipts_by_model`,
`live_entrypoint_policy_reward_machine_and_evaluator_hashes`,
`arc_registry_and_claims_hashes`,
`held_live_window_manifest_path_hash_counts_and_exp6400_disjointness`,
`live_attempt_provenance`,
`preregistered_passive_and_active_arm_contract`,
`matched_work_and_legal_action_receipts`,
`pre_action_goal_probe_and_action_freeze_records`,
`oracle_timing_receipts`,
`per_arm_model_window_admission_abstention_action_influence_progress_harm_and_cost_results`,
`treatment_fired_counts`, `delta_admission_precision`,
`delta_false_accept_count`, `delta_exact_progress_proxy`,
`paired_tests_confidence_intervals_and_effective_sample_sizes`,
`window_action_oracle_model_state_legal_set_budget_duplicate_and_label_attack_matrix`,
`hidden_source_access_count`, `offline_ground_truth_search_count`,
`per_game_adapter_count`, `oracle_before_action_count`, `solve_claim_count`,
`solve_registry_modified`, `arc_active_goal_causal_ready_score`,
`route_promotion_eligible`, `harm_underpowered_missing_and_flagged_cells`,
`protected_files_unchanged`, `preconditions_checked`, `inference_substrate`,
`verifier_is_oracle`, `field_principles`, `field_provenance`, `random_seed`,
`duration_s`, `tests_run`, `reproducibility_checksum`, and `honest_verdict`.

`arc_active_goal_causal_ready_score` SHALL equal 1.0 only when matched work
executes, the active treatment fires, all actions are frozen before outcomes,
provenance is clean, false accepts do not increase, and no solve or registry
claim occurs. `delta_false_accept_count` and `delta_exact_progress_proxy` SHALL
be bare numbers. `route_promotion_eligible` SHALL be true only when the causal
contract is ready and `delta_exact_progress_proxy` is positive.

### SCENARIO-ARC-ARM-6401-GATE-AND-HOLDOUTS

**Given** Exp6400 has terminal ready-score, treatment-fire, and false-accept
gate fields
**When** Exp6401 starts
**Then** it replays those gates, pins model, runtime, evaluator, policy,
registry, and claims hashes, seals at least eight held windows and 48 visible
transitions, and proves no held window reuses an Exp6400 window.

### SCENARIO-ARC-ARM-6401-MATCHED-CAUSAL-ARMS

**Given** passive two-sided and active-disagreement arms
**When** model-window cells are evaluated
**Then** both arms use the same model ids, windows, seeds, action budgets,
prompt budgets, evidence prefix lengths, legal action sets, and post-action
exact checks.

### SCENARIO-ARC-ARM-6401-FROZEN-ACTIONS

**Given** a model, held window, and live transition prefix
**When** a passive rank or active legal probe is selected
**Then** candidate goals, evidence disposition, selected rank or probe, and
action are sealed before the next environment transition is read.

### SCENARIO-ARC-ARM-6401-PAIRED-METRICS

**Given** paired passive and active rows
**When** Exp6401 computes admission, harm, action influence, and progress
metrics
**Then** it reports paired deltas, confidence intervals, effective sample
sizes, missing treatment-fire cells, and promotion eligibility without treating
missing cells or abstentions as successes.

### SCENARIO-ARC-ARM-6401-ATTACKS-FAIL-CLOSED

**Given** window reuse, action-order changes, oracle timing, model-row swaps,
goal-state carryover, unequal legal sets, unequal budgets, duplicate
transitions, and solve-label leakage
**When** the producer validates causal evidence
**Then** each attack fails closed before `arc_active_goal_causal_ready_score`
can be set.

### SCENARIO-ARC-ARM-6401-ARTIFACT-NO-SOLVE

**Given** the solve registry and claims ledger are hashed before the run
**When** Exp6401 writes its artifact
**Then** all required fields are present, forbidden access counts are zero,
`delta_false_accept_count` and `delta_exact_progress_proxy` are bare numbers,
protected files are unchanged, `solve_claim_count` is zero, the solve registry
hash is unchanged, and no solve claim is made.
