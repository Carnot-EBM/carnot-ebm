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

### REQ-ARC-ARM-6402: Active-Goal Safety Audit

Experiment 6402 SHALL independently audit the V550 active-goal chain from
registered artifacts, sidecars, sources, live entrypoints, model ids, policy
hashes, reward-machine hashes, window manifests, and registry hashes. It SHALL
register expected paths, existence states, and hashes before reading artifact
conclusions. It SHALL preserve absent, blocked, skipped, null, flagged,
retired, and clean states without filling missing windows or recreating ARC
calls.

The audit SHALL recompute scalar gates and scientific readiness from terminal
fields in Exp6393, Exp6400, and Exp6401. It SHALL treat post-action environment
transitions as evaluation-oracle evidence only after verified action freeze.
Models, goal hypotheses, reward machines, routes, and shadow ranks SHALL remain
non-oracles. The audit SHALL not invoke an LLM, run new ARC attempts, search
hidden game source, run exhaustive offline ground-truth BFS, use per-game
adapters, promote a policy, update `ops/arc_solve_registry.yaml`, write a
claims ledger, or change solve records.

The audit SHALL attack hidden game-source access, offline ground-truth search,
exhaustive BFS, per-game adapter use, development-proxy substitution,
outer-loop reverse engineering, oracle-before-action access, timestamp reorder,
freeze-receipt forgery, window reuse, duplicate transitions, model-row swaps,
stale goal state, legal-action mismatch, unequal work, treatment non-firing,
constant-false goal acceptance, false-accept aggregation, abstention pooling,
missing-cell pooling, progress-proxy relabeling, route enablement, solve
wording, solve-registry writes, and claims-ledger writes.

The audit SHALL verify MODEL_SPECS, cached SOTA receipts, embedded tokenizer
receipts, zero AutoTokenizer use, declared inference substrate, task-linked GPU
evidence, and absence of a legacy headline cell for present LLM tasks. It SHALL
compare all registered result and source hashes with current hashes and with
embedded upstream receipts where available. It SHALL verify that active-goal
code remains default-off and cannot alter normal executed actions.

Experiment 6402 SHALL write
`results/experiment_6402_arc_active_goal_safety_audit.json` with `status`,
`audit_registration_path_hash_and_expected_scope`,
`present_absent_blocked_skipped_null_flagged_and_retired_artifact_matrix`,
`recomputed_scalar_gates_and_readiness`,
`source_entrypoint_policy_reward_machine_model_window_and_registry_hash_matrix`,
`live_attempt_provenance_checks`,
`hidden_source_search_bfs_adapter_proxy_and_outer_loop_attack_results`,
`oracle_timing_freeze_window_duplicate_model_state_legal_work_and_firing_attack_results`,
`goal_false_accept_abstention_missing_progress_enablement_solve_and_registry_attack_results`,
`model_policy_and_inference_substrate_checks`,
`default_off_reachability_and_executed_action_integrity_checks`,
`critical_major_and_minor_findings`, `route_promotion_count`,
`solve_claim_count`, `solve_registry_modified`, `claims_ledger_modified`,
`public_arc_claim_eligibility`, `upstream_artifacts_modified`,
`protected_files_unchanged`, `preconditions_checked`, `inference_substrate`,
`verifier_is_oracle`, `field_principles`, `field_provenance`, `random_seed`,
`duration_s`, `tests_run`, `reproducibility_checksum`, and `honest_verdict`.

`public_arc_claim_eligibility` SHALL be false unless all present evidence is
clean, causal work matches, false accepts do not increase, route enablement is
zero, solve claims are zero, solve-registry writes are zero, and claims-ledger
writes are zero. `verifier_is_oracle` SHALL be false for the audit. The field
principles SHALL map every recomputed readiness, route-promotion, solve,
registry, and public-claim field to its fail-closed purpose.

### SCENARIO-ARC-ARM-6402-REGISTRATION-FIRST

**Given** the expected active-goal artifact chain, sidecars, source files, live
entrypoints, model ids, and registry files
**When** Exp6402 starts
**Then** it records existence states and hashes before loading artifact
conclusions, preserves absent paths as absent, and later compares every current
hash with the registration entry.

### SCENARIO-ARC-ARM-6402-READINESS-RECOMPUTE

**Given** Exp6393, Exp6400, and Exp6401 terminal fields
**When** Exp6402 recomputes readiness
**Then** readiness uses only bare terminal fields, fail-closes malformed or
missing fields, records route promotion separately from public-claim
eligibility, and never treats a route-value finding as a solve.

### SCENARIO-ARC-ARM-6402-ATTACKS-FAIL-CLOSED

**Given** forbidden provenance, oracle timing, freeze, window, duplicate, model,
goal, legal-action, work, treatment-fire, aggregation, progress, route,
wording, registry, and claims-ledger attacks
**When** Exp6402 validates the V550 active-goal chain
**Then** each attack records a passed fail-closed control or a finding, and any
unclean control keeps `public_arc_claim_eligibility` false.

### SCENARIO-ARC-ARM-6402-MODEL-POLICY-SUBSTRATE

**Given** present Exp6400 and Exp6401 model, tokenizer, GPU, policy, reward
machine, and inference-substrate receipts
**When** Exp6402 audits those receipts
**Then** all mandated model ids, cached SOTA receipts, model hashes, tokenizer
receipts, zero AutoTokenizer use, task-linked GPU evidence, default-off route
flags, and source hashes are checked without loading a model or invoking an
LLM.

### SCENARIO-ARC-ARM-6402-ARTIFACT-NO-PROMOTION

**Given** the audit is not an ARC solve, policy promotion, registry update, or
claims-ledger update
**When** Exp6402 writes its artifact
**Then** `route_promotion_count`, `solve_claim_count`, registry writes, and
claims-ledger writes are zero, protected files are unchanged, upstream
artifacts are unchanged, `public_arc_claim_eligibility` is false, and the
honest verdict states that the audit is complete and no public ARC claim is
eligible.

### REQ-ARC-ARM-6421: Explicit Opt-In Active-Goal Executed-Policy A/B

Experiment 6421 SHALL run a fresh route-off versus explicit-opt-in causal A/B
on the canonical live ARC policy path. The route SHALL remain shipped default
off before and after the run. The explicit-opt-in arm SHALL be reversible by
constructing a fresh policy with the flag unset.

Before any policy window is scored, Exp6421 SHALL revalidate the Exp6413
authenticated GGUF receipt gate, the Exp6400 shadow gate, the Exp6401 causal
gate, the Exp6402 safety audit, the solve registry, current shipped defaults,
canonical live entrypoint, generator model and tokenizer hashes, GPU receipts,
exact game interface, game roster, seeds, budgets, and protected held family.
The solve registry precheck SHALL cover every registered game. The task SHALL
not target a level, extend solve credit, update a solve registry, or write a
public ARC claim.

The `MODEL_SPECS` field SHALL include the shipped canonical live generator,
`unsloth/gemma-4-31B-it-qat-GGUF`, and at least one mandated SOTA model
resolved through `cached_sota_pair()`, including
`unsloth/gemma-4-31B-it-GGUF`. Tokenizer receipts SHALL come from the embedded
GGUF tokenizer. The producer SHALL not call `AutoTokenizer`.
Authenticated process and raw-output receipts MAY be inherited from Exp6413
only when the Exp6413 gate is revalidated and every inherited receipt is
content-addressed in the new artifact.

The A/B SHALL preregister matched route-off and explicit-opt-in arms over fresh
agent-visible windows. Each pair SHALL match games, seeds, observations, action
budgets, generator calls, prompts, token budgets, legal action set, and initial
agent state. The only behavioral difference SHALL be explicit enablement of the
active-goal route. The route-on arm MAY change only to a legal candidate action
that appears in the live policy's own candidate receipt. Executed actions,
observations, budgets, and terminal reasons SHALL be preserved per window.

Exp6421 SHALL measure route firing, changed legal executed actions, legal-action
rate, exact observation consistency, progress proxy, action count, latency, GPU
cost, deadline misses, and harmful regressions. Exact legal-action and observed
transition checks MAY be oracle-scoped. Routes, model output, progress proxies,
and policy scores SHALL not be treated as oracles.

Exp6421 SHALL attack route-label swaps, action substitution, observation reuse,
budget mismatch, off-path fixtures, model receipt reuse, game duplication,
source access, hidden adapter use, and solve-credit leakage. Each attack SHALL
fail closed before `arc_executed_policy_influence_ready_score` can be set to
1.0.

Experiment 6421 SHALL write
`results/experiment_6421_arc_opt_in_executed_policy_ab.json` with `status`,
`exp6413_gate_receipt`, `solve_registry_precheck_path_hash_and_results`,
`MODEL_SPECS`, `models_used`, `cached_sota_pair_receipts`,
`canonical_generator_model_file_and_embedded_tokenizer_hashes`,
`autotokenizer_usage_count`,
`canonical_live_entrypoint_route_policy_game_interface_and_config_hashes`,
`shipped_default_before_and_after`,
`preregistered_off_and_opt_in_arm_contract`,
`matched_games_seeds_observations_actions_model_calls_prompts_tokens_and_initial_state_receipts`,
`authenticated_model_process_and_raw_output_receipts`,
`per_window_route_candidate_executed_action_observation_budget_and_terminal_receipts`,
`per_arm_route_firing_policy_change_legal_action_observation_progress_actions_latency_gpu_deadline_and_harm_results`,
`causal_policy_delta`, `attack_matrix`, `source_access_count`,
`per_game_adapter_count`, `outer_loop_re_used`, `level_solve_claimed`,
`solve_registry_modified`, `route_default_promoted`,
`public_arc_claim_eligibility`, `arc_executed_policy_influence_ready_score`,
`harm_underpowered_missing_and_flagged_cells`, `protected_files_unchanged`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`field_principles`, `field_provenance`, `random_seed`, `duration_s`,
`tests_run`, `reproducibility_checksum`, and `honest_verdict`.

`arc_executed_policy_influence_ready_score` SHALL equal 1.0 only when the route
fires, causes a reproducible legal executed-policy change, produces no harmful
regression, authentic receipts pass, the shipped default remains off, and no
solve or registry mutation occurs. `public_arc_claim_eligibility` SHALL be
false unless a later audit permits a narrow internal policy claim. The honest
verdict SHALL start with a terminal prefix.

### SCENARIO-ARC-ARM-6421-PRECONDITIONS

**Given** Exp6413, Exp6400, Exp6401, Exp6402, the solve registry, and the live
policy source
**When** Exp6421 starts
**Then** it records gate receipts, registry hashes and every-game precheck
results, current default-off route values, model and tokenizer hashes, GPU
receipts, game roster, seeds, budgets, exact game interface hashes, and
protected held-family receipts before arm evaluation.

### SCENARIO-ARC-ARM-6421-MATCHED-OPT-IN-ARMS

**Given** a fresh agent-visible window
**When** route-off and explicit-opt-in arms are built
**Then** games, seeds, observations, action budgets, generator calls, prompts,
token budgets, legal action sets, and initial agent state match, and the only
planned difference is explicit active-goal enablement.

### SCENARIO-ARC-ARM-6421-EXECUTED-POLICY-CHANGE

**Given** a matched window where the active-goal route fires
**When** both arms execute policy actions
**Then** the opt-in arm changes to a legal candidate action, preserves exact
observation consistency, records route candidates and executed actions, and
keeps the change bounded to policy behavior rather than solve credit.

### SCENARIO-ARC-ARM-6421-ATTACKS-FAIL-CLOSED

**Given** route-label swaps, action substitution, observation reuse, budget
mismatch, off-path fixtures, model receipt reuse, game duplication, source
access, hidden adapter use, and solve-credit leakage
**When** Exp6421 validates the A/B evidence
**Then** every attack fails closed before the ready score can be set.

### SCENARIO-ARC-ARM-6421-NO-SOLVE-OR-PROMOTION

**Given** the completed Exp6421 artifact
**When** it is validated
**Then** no level solve is claimed, the solve registry is unchanged, the route
default is not promoted, public claim eligibility is false, verifier oracle
scope is limited to legal-action and exact observed-transition checks, and all
required fields have principles.

### REQ-ARC-ARM-6422: Held-Family Policy Safety Audit

Experiment 6422 SHALL independently audit Exp6421 and, when preconditions
permit, replay the frozen route-off versus explicit-opt-in policy comparison on
a pre-sealed held ARC live-window family through the canonical
`make_carnot_agent -> E3AgentPolicy` path. The audit SHALL not repair Exp6421,
tune from held outcomes, target a level, claim a solve, mutate
`ops/arc_solve_registry.yaml`, or promote the active-goal route.

The audit SHALL hash every available Exp6421 artifact, sidecar, source, route
configuration, model receipt, solve registry, held manifest, checker, and
determination record. Missing sidecars and blocked, null, underpowered, or
flagged cells SHALL remain explicit findings rather than being dropped.

The held-family precheck SHALL registry-precheck every held game or synthetic
held window id, prove the held manifest was sealed before Exp6421 outcomes,
exclude duplicate windows, and record whether any target was already credited
at the tested level. If the held family is missing, opened too late, duplicated,
or solve-credit contaminated, the audit SHALL still write a terminal artifact
with readiness zero.

`MODEL_SPECS` SHALL carry the same shipped canonical live generator evidence as
Exp6421 and the mandated dense GGUF model
`unsloth/gemma-4-31B-it-GGUF` resolved through `cached_sota_pair()`. Tokenizer
receipts SHALL come from embedded GGUF tokenizers, and the producer SHALL not
call `AutoTokenizer`.

Experiment 6422 SHALL recompute route firing, legal executed-policy change,
exact observations, progress proxy, actions, latency, deadline misses, and harm
from raw policy rows. It SHALL compare reported Exp6421 deltas against held
recomputed deltas, and SHALL attack route-label swaps, action substitution,
observation reuse, budget mismatch, off-path fixtures, model substitution,
source access, exhaustive search, per-game adapter use, duplicate games, hidden
retuning, and solve-credit leakage.

Experiment 6422 SHALL write
`results/experiment_6422_arc_held_family_policy_safety_audit.json` with
`status`, `expected_and_available_exp6421_inputs`,
`upstream_artifact_sidecar_source_route_model_checker_and_determination_hashes`,
`missing_input_findings`, `solve_registry_precheck_path_hash_and_results`,
`held_manifest_path_hash_counts_seal_time_disjointness_and_duplicate_checks`,
`frozen_route_config_hash`, `MODEL_SPECS`, `models_used`,
`cached_sota_pair_receipts`, `embedded_gguf_tokenizer_receipts`,
`autotokenizer_usage_count`, `authenticated_model_and_live_policy_receipts`,
`matched_held_off_and_opt_in_work_receipts`,
`recomputed_route_firing_policy_change_legal_action_observation_progress_actions_latency_deadline_and_harm_results`,
`reported_vs_recomputed_deltas`, `attack_matrix`, `source_access_count`,
`exhaustive_search_count`, `per_game_adapter_count`, `hidden_retuning_count`,
`outer_loop_re_used`, `level_solve_claimed`, `solve_registry_modified`,
`shipped_default_preserved`, `public_arc_claim_eligibility`,
`arc_held_policy_safety_audit_ready_score`,
`adversarial_and_determination_preservation_findings`,
`harm_underpowered_missing_and_flagged_cells`, `protected_files_unchanged`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`field_principles`, `field_provenance`, `random_seed`, `duration_s`,
`tests_run`, `reproducibility_checksum`, and `honest_verdict`.

`arc_held_policy_safety_audit_ready_score` SHALL equal 1.0 only when eligible
held policy influence reproduces without harm, all model and policy receipts
are authentic, all critical attacks fail closed, shipped default-off behavior
is preserved, and no solve or registry claim occurs. The top-level
`verifier_is_oracle` SHALL be false. Exact legal-action and observed-transition
checks MAY remain scoped semantic oracles. The honest verdict SHALL start with
a terminal prefix.

### SCENARIO-ARC-ARM-6422-HASH-AND-MISSING-INPUTS

**Given** Exp6421, Exp6402, route sources, model receipts, held manifests,
checkers, determination records, and sidecars if present
**When** Exp6422 starts
**Then** it records path hashes and explicit missing-input findings before it
uses any upstream conclusion.

### SCENARIO-ARC-ARM-6422-HELD-REPLAY

**Given** a held live-window manifest sealed before Exp6421 outcomes
**When** Exp6422 replays the frozen route on matched route-off and opt-in arms
**Then** games, seeds, observations, legal actions, budgets, prompts, token
budgets, model calls, and initial policy state match, and only the explicit
route enablement can change the executed legal action.

### SCENARIO-ARC-ARM-6422-RECOMPUTE-AND-ATTACKS

**Given** raw held policy rows and Exp6421 reported deltas
**When** Exp6422 recomputes metrics and runs adversarial controls
**Then** route firing, legal policy change, observations, progress, latency,
deadline, and harm are recomputed from rows, reported deltas are compared, and
all critical attacks fail closed before readiness can be one.

### SCENARIO-ARC-ARM-6422-NO-SOLVE-OR-REGISTRY

**Given** the completed Exp6422 artifact
**When** it is validated
**Then** source access, exhaustive search, per-game adapters, hidden retuning,
outer-loop RE, level solve claims, solve registry writes, public claim
eligibility, and route promotion are all absent, and every required field has a
principle.

## REQ-ARC-BENCH-6267: One held-out ARC number that can move

The ARC loop MUST maintain a single comparable measurement of live-agent
capability, computed the way the competition computes it.

**Why.** `reproducible_total_levels` reached 183 of 183 on 2026-07-17. Every
public game is cleared and hand-adaptered, so the metric that steered this work
for months is pinned and can never move again. Nothing replaced it. Measured
2026-08-13 across the last 10 milestones: 16 ARC tasks, 13 ending
`ready_no_solve_claim` or `default_off`. Three tasks named "holdout" -- Exp6295,
Exp6308, Exp6401 -- emitted metric sets sharing zero keys. Milestone .542 and
milestone .550 cannot be compared.

`scripts/arc_bench.py` runs the adapter-free path, the same first-contact
mechanism the live agent uses on a game it has never seen, and reports levels
cleared against actions spent under one fixed schema.

### SCENARIO-ARC-BENCH-6267-HELD-OUT-PATH

**Given** a public game that has a registered `GameAdapter`
**When** the benchmark runs that game
**Then** the adapter is bypassed, the run uses `graph_explore_solve_v2`, and the
row records `adapter_used: false`.

### SCENARIO-ARC-BENCH-6267-SCORED-SHAPE

**Given** a completed benchmark run
**When** the report is written
**Then** each row carries `levels_cleared` and `actions_spent`, where actions are
counted by wrapping `env.step` rather than taken from the returned solution
length, because the two differ by three orders of magnitude on ls20 (13 against
17,197) and the competition scores what was spent.

### SCENARIO-ARC-BENCH-6267-ERRORS-VISIBLE

**Given** a game that raises during the run
**When** the report is aggregated
**Then** the game appears as an error row and is counted in `games_errored`, and
is excluded from `clear_rate` rather than counted as a clean zero, so a sweep
that drops its hard cases cannot report a rising average while getting worse.

### SCENARIO-ARC-BENCH-6267-ROSTER-CHANGE-RESETS-ROTATION

**Given** a persisted rotation offset taken against an earlier roster
**When** the roster changes
**Then** the offset resolves to 0, so a newly added game is benchmarked on the
next run instead of being skipped for as many runs as the offset is ahead.

### SCENARIO-ARC-BENCH-6267-HELD-OUT-CAVEAT-CARRIED

**Given** any benchmark report or console output
**When** it is produced
**Then** it states that these are the 25 public games with their adapter
bypassed, that disabling the adapter removes the hand-written route and not the
knowledge that produced it, and that the number is not a hidden-game result.

## REQ-ARC-FLAG-LEDGER-6268: Measured ARC improvements promote themselves

An ARC capability shipped behind a flag MUST be able to reach default-on by
measured evidence, and MUST NOT reach it any other way.

**Why.** The agent carries 101 distinct `CARNOT_ARC_*` flags. Nothing recorded
which were on, why, or on what evidence. A loop that generates options and never
chooses between them is not improving; 101 unchosen options is a search space
nobody searches.

### SCENARIO-ARC-FLAG-LEDGER-6268-REGRESSION-REFUSES

**Given** an arm that clears new levels on three games and loses a level on one
game it previously cleared, so the aggregate improves
**When** the promotion rule evaluates it
**Then** promotion is REFUSED and the lost game is named, because an aggregate
win that costs a game is the ARC engine-store failure that destroyed ka59 from
1.0 to 0.0.

### SCENARIO-ARC-FLAG-LEDGER-6268-ONE-GAME-IS-NOT-EVIDENCE

**Given** an arm that improves exactly one game and regresses none
**When** the promotion rule evaluates it
**Then** the verdict is HOLD, because the search is deterministic and a single
improved game is a coincidence rather than a capability.

### SCENARIO-ARC-FLAG-LEDGER-6268-EFFICIENCY-COUNTS

**Given** an arm that clears the same levels using strictly fewer actions on
every cleared game and more on none
**When** the promotion rule evaluates it
**Then** promotion is granted, because the competition metric squares efficiency.

### SCENARIO-ARC-FLAG-LEDGER-6268-NO-IMPLICIT-MEASUREMENT

**Given** a flag with no recorded evidence, or whose last recorded verdict
refused promotion
**When** promotion is requested
**Then** the request is refused, the recorded reason is repeated, and no
measurement is run implicitly, because a promotion that measures itself is a
promotion nobody reviewed.

### SCENARIO-ARC-FLAG-LEDGER-6268-DISCOVERY-FROM-SOURCE

**Given** a capability shipped behind a new `CARNOT_ARC_*` flag
**When** the ledger discovers flags
**Then** the flag set is read from the agent source rather than a maintained
list, so a new flag is tracked from the milestone it lands in.

### SCENARIO-ARC-FLAG-LEDGER-6268-CORRUPT-LEDGER-FAILS-LOUD

**Given** an unreadable `ops/arc_flag_ledger.yaml`
**When** the ledger is loaded
**Then** the process exits with an error rather than starting a fresh ledger,
because silently replacing it would erase every promotion and its evidence.

## REQ-ARC-BENCH-6269: The benchmark must be able to reach the flag it measures

A flag measurement MUST run on an engine whose import closure contains the flag,
or MUST be refused.

**Why.** `arc_bench.py`'s original engine drives `graph_explore_solve_v2`. Only 48
of the 95 tracked `CARNOT_ARC_*` flags live inside that closure. Setting one of
the other 47 and running the sweep produces a byte-identical result, because the
code that reads it never executes. The promotion rule reads that as HOLD -- "no
level gained and no clear efficiency gain" -- which files a real capability as
worthless, with evidence attached. Wrong for 47 flags, in the most damaging
direction available.

The `scored` engine drives `E3AgentPolicy` through
`arc_leaderboard_eval.run_game`. That is the policy `make_carnot_agent(Agent)`
builds, so a flag measured there is measured on the agent the competition runs.
Coverage: explore 48, scored 89, neither 6.

### SCENARIO-ARC-BENCH-6269-REFUSE-UNREACHABLE

**Given** a flag outside the selected engine's transitive import closure
**When** a measurement is requested without `--force`
**Then** the request is refused, no sweep is run, and the message states that the
code reading the flag never executes.

### SCENARIO-ARC-BENCH-6269-ROUTE-TO-A-CAPABLE-ENGINE

**Given** a flag refused on one engine that IS reachable on another
**When** the refusal is printed
**Then** it names the engine that can measure it, because a refusal without a next
step is a dead end.

### SCENARIO-ARC-BENCH-6269-FORCED-NULL-IS-STAMPED

**Given** `--force` is passed for an unreachable flag
**When** the result is recorded
**Then** the entry carries `benchmark_reachable: false`, so a later reader cannot
mistake the null for evidence that the capability does nothing.

### SCENARIO-ARC-BENCH-6269-CHARGED-ACTIONS

**Given** a scored-engine run whose driver reports both `actions` and
`charged_actions`
**When** the row is written
**Then** `actions_spent` takes `charged_actions`, because the live gateway bills
resets and the two differ (vc33: 387 against 400); reporting the smaller number
would show an efficiency gain the competition would not pay out.

### SCENARIO-ARC-BENCH-6269-ENV-RESTORED

**Given** the scored engine sets `CARNOT_ARC_DISABLE_INDUCTION` for one cell
**When** that cell finishes, by success or by exception
**Then** the variable is restored to its prior value, so one cell cannot change
the meaning of every later cell in the same sweep.

### SCENARIO-ARC-BENCH-6269-ENGINE-ENTRY-DECLARED

**Given** an engine offered by `arc_bench`
**When** reachability is computed
**Then** that engine has a declared entry module, and a test fails if the two
lists disagree, because an undeclared engine falls back to another engine's
closure and silently under-reports its own reach.
