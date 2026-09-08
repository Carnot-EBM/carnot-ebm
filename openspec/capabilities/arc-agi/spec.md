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

### REQ-ARC-ARM-6434: Collision-Certified State-Key Suffix

Experiment 6434 SHALL add a generic, default-off state-key suffix route to the
adapter-bypassed live ARC graph explorer. The route MAY extend a state key only
after the current base key has aliased two distinct live observation histories.
The certificate SHALL record the base key, the distinct observation-history
hashes, alias evidence, and the minimal action-suffix length that separates the
known histories. The route SHALL use only visible observations, legal actions,
and the agent's own action history. It SHALL not use a game id, game-specific
feature, source-derived rule, hand `GameAdapter`, hidden state, offline
ground-truth search, or solve registry evidence.

The route SHALL remain shipped default off before and after the run. The old
manual suffix knob MAY remain for explicit diagnostics, but Exp6434 SHALL use
only the collision-certified opt-in arm for the treatment. Identical histories,
non-aliasing HUD changes, resets, process restarts, hash instability, and hash
substitution SHALL not produce a certificate.

Experiment 6434 SHALL run matched baseline and explicit opt-in arms over the
full 25-game public roster with adapters bypassed. It SHALL use at least three
seeds with identical expansion and action budgets, identical initial state
receipts, and the canonical live graph-explore game interface. It SHALL report
frontier exhaustion, unique states, alias certificates, environment steps, legal
actions, exact observations, terminal reasons, cleared-state observations,
errors, wall time, and action cost by game and seed. Cleared-state observations
are diagnostic only. They SHALL not create solve credit.

Experiment 6434 SHALL write
`results/experiment_6434_arc_state_key_reachability_ab.json` with the fields
named by the task. It SHALL set `verifier_is_oracle=true` only for legal-action,
exact observed-transition, state-hash, and collision-certificate checks.
Reachability metrics SHALL not be solve oracles. It SHALL not update
`ops/arc_solve_registry.yaml`, claim a game or level solve, promote a default,
or create public ARC claim eligibility.

`arc_state_key_reachability_ready_score` SHALL equal 1.0 only when certified
premature frontier collapse decreases, no baseline-cleared game regresses, no
new error appears, all critical attacks fail closed, the default remains off,
and no solve or registry mutation occurs. Otherwise the score SHALL be zero and
the blocked reason SHALL name the failed gate.

### SCENARIO-ARC-ARM-6434-CERTIFICATE

**Given** two live paths reach the same base state key with distinct visible
observation histories
**When** the certified suffix route is enabled
**Then** the second key is extended by the minimal separating action suffix, a
certificate records both history hashes and alias evidence, and no game id,
adapter, source rule, hidden state, or solve registry value is used.

### SCENARIO-ARC-ARM-6434-NO-CERTIFICATE

**Given** non-aliasing base keys, an identical history replay, a monotone HUD
change, a reset, a process restart, hash instability, and hash substitution
**When** the certifier receives those cases
**Then** it records no accepted collision certificate and fails closed on the
hash attacks.

### SCENARIO-ARC-ARM-6434-MATCHED-AB

**Given** the canonical adapter-bypassed public ARC roster
**When** Exp6434 runs baseline and explicit opt-in arms
**Then** games, seeds, expansion budgets, action budgets, exact game interface,
initial states, legal action receipts, and shipped defaults match, with the
certified suffix route as the only treatment difference.

### SCENARIO-ARC-ARM-6434-ATTACKS-FAIL-CLOSED

**Given** game-id branching, hidden adapter use, source access, offline BFS,
false certificates, history truncation, hash substitution, budget mismatch,
seed mismatch, state leakage, and solve-credit leakage
**When** Exp6434 validates readiness
**Then** each attack fails closed before readiness can be one.

### SCENARIO-ARC-ARM-6434-NO-SOLVE-OR-PROMOTION

**Given** the completed Exp6434 artifact
**When** it is validated
**Then** source access, exhaustive search, per-game adapters, outer-loop RE,
level solve claims, solve registry writes, default promotion, and public claim
eligibility are all absent, and every no-solve, regression, attack, and
readiness field has a principle.

### REQ-ARC-ARM-6458: Representation-Objective Generalization A/B

Experiment 6458 SHALL recover the failed Exp6434 state-key reachability work
with bounded shards, atomic checkpoints, and resume. It SHALL measure held live
policy decisions only. It SHALL not attempt, claim, or credit a public game or
level solve.

The producer SHALL freeze deterministic tuning, safety, and held game rosters
from readable immutable observation/action transition traces. Tuning and held
rosters SHALL be disjoint and hash-recorded. The producer MAY tune only generic
state suffix thresholds and objective weights on the tuning roster. It SHALL
evaluate held rows after those values are frozen.

The producer SHALL compare four matched arms:
`current_state_key_current_objective`,
`collision_suffix_current_objective`,
`collision_suffix_reachability_objective`, and
`collision_suffix_shuffled_objective_placebo`. The only representation
treatment SHALL be the generic collision-certified suffix. The only objective
treatment SHALL be a reachability-aware objective computed from visible prior
actions and tuning evidence. The placebo SHALL use the same suffix and a
seeded shuffled objective.

The producer SHALL use only observations, action IDs, action coordinates, and
post-action next-state observations present in the immutable trace corpus. It
SHALL not read game source, run offline ground-truth BFS, inject hidden state,
create or consume a per-game `GameAdapter`, use outer-loop reverse engineering,
or read a recorded next state before the action is frozen.

The producer SHALL run bounded CPU shards by roster, game, prefix, seed, and
arm. It SHALL cap each cell, checkpoint atomically after every cell, print
progress, resume completed cells without repeating them, and always write a
terminal partial or complete artifact. Each held row SHALL include the game,
trace prefix, seed, arm, state collision, legal-action set, chosen action,
recorded next-state reachability, policy influence, state count, action cost,
timeout, and checkpoint receipt.

Experiment 6458 SHALL write
`results/experiment_6458_arc_representation_objective_generalization_ab.json`
with `status`, `registry_precheck_and_hash`,
`no_game_or_level_solve_claim`, `solve_registry_unchanged`,
`game_source_access_count`, `offline_ground_truth_bfs_count`,
`per_game_adapter_count`, `canonical_live_path_receipts`,
`tuning_and_held_roster_manifest_and_disjointness`,
`arm_objective_and_suffix_precommitment`,
`shard_budgets_and_checkpoint_manifest`,
`resume_and_terminal_partial_receipts`, `per_unit_rows`,
`collision_rates_by_arm`, `legal_action_coverage_by_arm`,
`held_next_state_reachability_by_arm`, `policy_influence_by_arm`,
`action_cost_timeout_and_regression_results`,
`paired_effects_and_uncertainty`, `aggregate_row_recomputation`,
`attack_matrix`, `current_adversarial_findings`,
`arc_objective_generalization_ready_score`, `protected_files_unchanged`,
`blocked_reason`, `gate_check_summary`, `preconditions_checked`,
`inference_substrate`, `verifier_is_oracle`, `field_principles`,
`field_provenance`, `random_seed`, `duration_s`, `tests_run`,
`reproducibility_checksum`, and `honest_verdict`.

The artifact SHALL set `verifier_is_oracle=false`. Recorded next-state
transitions are post-action evaluation evidence. They SHALL not be used as a
pre-action oracle. The artifact SHALL set `no_game_or_level_solve_claim=true`,
`solve_registry_unchanged=true`, `game_source_access_count=0`,
`offline_ground_truth_bfs_count=0`, and `per_game_adapter_count=0`. It SHALL
not add `solve_provenance`.

`arc_objective_generalization_ready_score` SHALL equal 1.0 only when the
combined generic suffix and reachability objective reduces collisions and
improves held recorded next-state reachability or legal policy choice over both
single-change arms, does not regress the frozen safety roster, all claims
recompute from held rows, provenance boundaries pass, a nonzero held sample
completes, and critical findings are zero. Otherwise the score SHALL be zero
and `gate_check_summary` SHALL name the failed gate.

### SCENARIO-ARC-ARM-6458-PRECONDITIONS

**Given** the solve registry, live ARC entrypoints, immutable traces, and a
writable results directory
**When** Exp6458 starts
**Then** it records the registry hash, confirms no solve claim, imports the
canonical adapter-bypassed live path, confirms readable observation/action
traces, confirms no game-source access and no adapter use, checks a monotonic
clock, checks atomic checkpoint writes, and records explicit shard budgets.

### SCENARIO-ARC-ARM-6458-DISJOINT-TUNING-HELD

**Given** all readable trace games
**When** Exp6458 freezes rosters
**Then** tuning, safety, and held rosters are deterministic, hash-recorded,
tuning and held games are disjoint, and held evaluation uses only the
precommitted suffix threshold and objective weights.

### SCENARIO-ARC-ARM-6458-MATCHED-ARMS

**Given** a held game, prefix, and seed
**When** Exp6458 evaluates policy choices
**Then** all four arms use the same observation prefix, legal-action set, seed,
and action budget. Only the representation and objective names differ.

### SCENARIO-ARC-ARM-6458-CHECKPOINT-RESUME

**Given** a partially completed checkpoint
**When** Exp6458 resumes
**Then** completed cells are not repeated, each new cell is atomically
checkpointed, progress is printed, and a terminal partial or complete artifact
is written.

### SCENARIO-ARC-ARM-6458-ROWS-RECOMPUTE

**Given** held per-unit rows
**When** aggregate metrics are recomputed
**Then** collision rates, legal-action coverage, held next-state reachability,
policy choice changes, action cost, regressions, timeout rate, and paired
effects are reproduced from those rows without aggregate-only evidence.

### SCENARIO-ARC-ARM-6458-ATTACKS-FAIL-CLOSED

**Given** tuning-held leakage, source access, adapter use, oracle next-state
access before action, registry mutation, completed-cell repetition, checkpoint
truncation, placebo bias, timeout exclusion, and aggregate-row mismatch attacks
**When** Exp6458 validates readiness
**Then** each critical attack fails closed before readiness can be one.

### SCENARIO-ARC-ARM-6458-NO-SOLVE-OR-PROMOTION

**Given** the completed Exp6458 artifact
**When** it is validated
**Then** source access, offline BFS, per-game adapters, solve claims, solve
provenance, registry writes, and public solve credit are absent, and every
field and readiness condition has a principle.

### REQ-ARC-ARM-6471: Generic Safety Shield for Frozen ARC Objective

Experiment 6471 SHALL freeze the best Exp6458 representation, objective,
runtime trace roster, safety roster, and row reducer. It SHALL add a generic
safety veto with a conservative fallback. It SHALL make no game or level solve
claim.

The producer SHALL compare five matched arms on leave-one-game-out runtime
trace rows: `baseline_current_policy`, `objective_only_frozen_exp6458`,
`shielded_objective_generic_fallback`, `ablated_shield_objective_no_veto`, and
`shuffled_shield_objective_control`. Each arm SHALL use the same game, trace
prefix, seed, legal-action set, and action budget. The shield SHALL derive each
decision from live-reachable runtime features only. It SHALL not use game
identity, per-game thresholds, hidden state, game source, offline
ground-truth search, or a per-game adapter.

The producer SHALL run bounded CPU shards with atomic checkpoints and resume.
It SHALL write a terminal partial or complete artifact. Each row SHALL include
the game, trace, arm, decision, shield reason, legal action, reachability
metric, safety result, timing, and provenance counts.

Experiment 6471 SHALL write
`results/experiment_6471_arc_generic_safety_shield_objective_ab.json` with
`status`, `registry_precheck`, `no_solve_claim`,
`frozen_representation_objective_and_roster_hashes`,
`leave_one_game_out_manifest`, `generic_shield_and_fallback_hash`,
`canonical_reducer_hash`, `checkpoint_and_resume_receipts`, `per_unit_rows`,
`reachability_by_arm`, `legal_action_results_by_arm`,
`safety_roster_results_by_arm`, `g50t_safety_result`,
`aggregate_row_recomputation`, `source_and_adapter_access_receipts`,
`attack_matrix`, `current_adversarial_findings`,
`arc_safety_shield_ready_score`, `protected_files_unchanged`,
`blocked_reason`, `gate_check_summary`, `preconditions_checked`,
`inference_substrate`, `verifier_is_oracle`, `field_principles`,
`field_provenance`, `random_seed`, `duration_s`, `tests_run`,
`reproducibility_checksum`, and `honest_verdict`.

The artifact SHALL set `no_solve_claim=true`. It SHALL omit
`solve_provenance`. It SHALL set `verifier_is_oracle=false`. The ready score
SHALL equal 1.0 only when shielded held reachability preserves or improves the
objective-only arm, the full frozen safety roster does not regress, `g50t`
does not regress, stored aggregates exactly equal the canonical row reducer,
source and adapter receipts are clean, protected files are unchanged, all
critical attacks fail closed, and current critical findings are zero.

### SCENARIO-ARC-ARM-6471-PRECHECK-AND-FREEZE

**Given** the solve registry, Exp6458 artifact, immutable runtime traces, and
a writable results directory
**When** Exp6471 starts
**Then** it records the registry hash, confirms no solve claim, freezes the
Exp6458 representation, objective, safety roster, leave-one-game-out splits,
shield, fallback, and canonical reducer hashes, and records that the registry
will not be updated.

### SCENARIO-ARC-ARM-6471-GENERIC-SHIELD

**Given** a live runtime feature row with legal actions, prior action history,
and frozen objective and baseline actions
**When** the generic shield evaluates the objective action
**Then** it vetoes only from game-blind runtime features, falls back to a legal
baseline action, records the shield reason, and does not read source, adapters,
hidden state, or recorded next state before the action is frozen.

### SCENARIO-ARC-ARM-6471-MATCHED-ROWS

**Given** a leave-one-game-out fold, trace prefix, and seed
**When** Exp6471 evaluates all arms
**Then** baseline, objective-only, shielded, ablated-shield, and shuffled-shield
rows use the same observation prefix, legal-action set, seed, and action cost.

### SCENARIO-ARC-ARM-6471-CHECKPOINT-RESUME

**Given** a partially completed checkpoint
**When** Exp6471 resumes
**Then** completed cells are skipped, new cells are atomically checkpointed, and
a terminal partial or complete artifact is written.

### SCENARIO-ARC-ARM-6471-ROWS-RECOMPUTE

**Given** Exp6471 per-unit rows
**When** the canonical reducer recomputes aggregates
**Then** reachability, legal-action results, safety roster results, `g50t`,
paired deltas, and row checksums exactly equal the stored aggregate fields.

### SCENARIO-ARC-ARM-6471-ATTACKS-FAIL-CLOSED

**Given** source access, game identity leakage, per-game thresholds, adapter
use, unreachable solver routing, duplicate solve claim, safety suppression, and
aggregate mismatch attacks
**When** Exp6471 validates readiness
**Then** each critical attack fails closed before readiness can be one.

### SCENARIO-ARC-ARM-6471-NO-SOLVE-OR-PROMOTION

**Given** the completed Exp6471 artifact
**When** it is validated
**Then** source access, offline BFS, per-game adapters, solve claims, solve
provenance, registry writes, and public solve credit are absent.

### REQ-ARC-ARM-6499: Conservative Prefix-Energy Progress Alignment

Experiment 6499 SHALL replay frozen public ARC prefixes from the live agent's
own recorded attempts. It SHALL test whether conservative prefix energy aligns
with later recorded live-agent progress beyond simple controls. It SHALL not
change policy, inspect game source, build a per-game adapter, run offline
ground-truth BFS, or claim a new game or level solve.

The producer SHALL evaluate the Exp6488 upstream gate before it reads rows. The
receipt SHALL record the path, hash, field, expected value, observed value, type,
and pass state. The producer SHALL run a solve-registry precheck before prefix
selection. The precheck SHALL record the registry path, hash, and already
reproduced games and levels.

The producer SHALL freeze games, levels, seeds, prefix checkpoints, horizon,
energy version, progress metric, controls, statistical tests, and exclusion rules
before replay. It SHALL use only prefixes with live-path receipts from the scored
agent or the offline live twin. It SHALL reject source-derived traces,
exhaustive offline search, development proxy adapters, duplicate prefixes, and
duplicate credited solves.

The producer SHALL compute energy from the frozen prefix only. It SHALL not
change later actions. It SHALL compare energy with step count, action count,
valid-action fraction, state size, novelty, and shuffled-energy controls. It
SHALL report per game, level, prefix, seed, horizon, and control rows. It SHALL
also report roster coverage, headroom, confidence intervals, calibration, safety
regressions, and leave-one-game-out directional stability. No regressing game
may be removed from the aggregate.

Experiment 6499 SHALL write
`results/experiment_6499_arc_energy_progress_alignment.json` with `status`,
`upstream_gate_receipt`, `arc_registry_precheck`, `frozen_roster_manifest`,
`live_path_receipts`, `solve_provenance`, `rows`, `roster_coverage_rows`,
`incremental_alignment_rows`, `leave_one_game_out_rows`,
`confidence_intervals`, `safety_regression_rows`, `arc_attack_matrix`,
`no_policy_change_receipt`, `no_new_solve_claim`,
`arc_alignment_execution_complete_score`,
`arc_energy_alignment_ready_score`, `per_unit_rows`,
`aggregate_row_recomputation`, `gate_check_summary`, `preconditions_checked`,
`protected_files_unchanged`, `inference_substrate`, `verifier_is_oracle`,
`field_principles`, `field_provenance`, `random_seed`, `duration_s`,
`tests_run`, `reproducibility_checksum`, and `honest_verdict`. It MAY also
write `calibration_rows` when they are row-derived.

The field principles SHALL be:
`status`: Terminal ARC alignment diagnostic state.
`upstream_gate_receipt`: Exp6488 path, hash, field, expected, and observed
value.
`arc_registry_precheck`: Registry path, hash, and already reproduced
games/levels.
`frozen_roster_manifest`: Games, levels, seeds, prefixes, horizons, energy,
metrics, controls, and exclusions.
`live_path_receipts`: Proof that each prefix came from the reachable live agent
path.
`solve_provenance`: live_agent_self_discovery for live prefixes; this task
makes no new solve claim.
`rows`: Per game, level, prefix, seed, horizon, energy, progress, and control
metrics.
`roster_coverage_rows`: Coverage and headroom by game and level.
`incremental_alignment_rows`: Energy contribution beyond simple controls.
`leave_one_game_out_rows`: Held directional stability.
`confidence_intervals`: Predeclared row-derived uncertainty.
`safety_regression_rows`: Invalidity and any game-level regression signals.
`arc_attack_matrix`: Leakage, source, adapter, filtering, duplicate, redefine,
threshold, and mutation attacks.
`no_policy_change_receipt`: Proof that replay did not alter live actions.
`no_new_solve_claim`: True.
`arc_alignment_execution_complete_score`: Execution-completeness field.
`arc_energy_alignment_ready_score`: Same-roadmap policy gate field.
`per_unit_rows`: Required game/level/prefix/seed/horizon/control rows.
`aggregate_row_recomputation`: Every alignment and readiness headline
recomputed from rows.
`gate_check_summary`: Exact gate evaluation or blocked_* reason and observed
value.
`preconditions_checked`: Lineage lock, registry, live path, roster, and energy
version.
`protected_files_unchanged`: Active roadmap and conductor unchanged.
`inference_substrate`: frozen_live_arc_prefix_replay_no_new_llm.
`verifier_is_oracle`: False for energy; exact environment feedback is
authoritative for recorded progress.
`field_principles`: Reason for each provenance, alignment, and safety field.
`field_provenance`: Trace hashes, registry, environment receipts, and reducers.
`random_seed`: Frozen roster and interval seeds.
`duration_s`: Measured replay and task wall time.
`tests_run`: Commands and exit codes.
`reproducibility_checksum`: Hash over gate, registry, roster, traces, and rows.
`honest_verdict`: complete_positive, complete_null, disqualified, or blocked_*
with gate_check_summary.
`calibration_rows`: Calibration bins keep the alignment signal inspectable.

The artifact SHALL set
`inference_substrate=frozen_live_arc_prefix_replay_no_new_llm`,
`verifier_is_oracle=false`, and `no_new_solve_claim=true`.
`solve_provenance` SHALL state `live_agent_self_discovery` for accepted live
prefixes and SHALL state that the task makes no new solve claim.
`arc_alignment_execution_complete_score` SHALL equal 1.0 only when every frozen
row and attack is accounted for. `arc_energy_alignment_ready_score` SHALL equal
1.0 only when energy has positive held incremental alignment beyond controls,
leave-one-game-out direction is stable, roster coverage and headroom are
adequate, and no safety-regression signal exists. Otherwise it SHALL be zero.

### SCENARIO-ARC-ARM-6499-LIVE-PREFIX-PROVENANCE

**Given** frozen prefix candidates and live ARC entrypoint receipts
**When** Exp6499 selects rows
**Then** every accepted prefix has a live-agent receipt, a trace hash, no source
access, no offline BFS, no adapter use, and no duplicate prefix or solve claim.

### SCENARIO-ARC-ARM-6499-FROZEN-ROSTER-AND-PRECHECK

**Given** the Exp6488 lineage gate and solve registry
**When** Exp6499 starts
**Then** it records the exact upstream gate value, registry hash, already
reproduced games and levels, frozen games, levels, seeds, horizons, energy
version, progress metric, controls, tests, and exclusions before replay.

### SCENARIO-ARC-ARM-6499-DIRECT-PROGRESS-ALIGNMENT

**Given** frozen live prefix rows
**When** Exp6499 computes conservative prefix energy
**Then** later progress is measured from the existing frozen trace without
changing subsequent actions, and every row records game, level, prefix, seed,
horizon, energy, progress, and control metrics.

### SCENARIO-ARC-ARM-6499-CONFOUND-CONTROLS

**Given** per-prefix energy and later progress rows
**When** Exp6499 recomputes alignment
**Then** it compares energy against step count, action count, valid-action
fraction, state size, novelty, and shuffled-energy controls, and it reports
incremental alignment, confidence intervals, calibration, and leave-one-game-out
direction from rows.

### SCENARIO-ARC-ARM-6499-ATTACKS-FAIL-CLOSED

**Given** future-outcome leakage, source access, per-game features, roster
filtering, duplicate prefix, solved-level duplication, progress redefinition,
post-hoc thresholding, and policy mutation attacks
**When** Exp6499 validates readiness
**Then** every critical attack fails closed before readiness can be one.

### SCENARIO-ARC-ARM-6499-NO-SOLVE-BOUNDARY

**Given** the completed Exp6499 artifact
**When** it is validated
**Then** source access, offline BFS, per-game adapters, policy changes, registry
writes, solve claims, and new public solve credit are absent.

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

## REQ-ARC-6843: Live ARC Evidence Stratum Freeze

Experiment 6843 SHALL freeze an immutable inventory of terminal live-agent ARC
evidence available at execution time. The inventory SHALL separate model,
policy, game, budget, run identity, tool-loop state, supervisor state, and
receipt completeness. It SHALL treat in-flight processes as observations only.
It SHALL not wait for, stop, signal, lease, or otherwise alter a live process.

The producer SHALL require `v598_evidence_root_ready_score=1`, readable ARC
registry and status files, and a readable canonical live-path source identity
before any complete inventory verdict. If a gate fails, it SHALL still write a
terminal `complete_blocked_live_arc_inventory` artifact with
`gate_check_summary` naming the failed check, expected value, and observed
value. The current live run does not need to finish.

The producer SHALL inventory terminal supervisor, tool-gap, trajectory, and
leaderboard artifacts by file bytes, source hash, schema/status, and producer
configuration. It SHALL not import producer claims as ground truth. It SHALL
reject incomplete artifacts from eligible rows and list them in
`incomplete_artifact_manifest`.

The producer SHALL declare `inference_substrate=read_only_live_artifact_inventory`.
That substrate is a no-new-LLM provenance audit over existing files and process
metadata. GGUF, CUDA, model, and live-run strings quoted from source artifacts
or observed process commands SHALL NOT be interpreted as model invocation by the
inventory itself.

The artifact SHALL write
`results/experiment_6843_live_arc_evidence_stratum_freeze.json` with these
top-level fields: `field_principles`, `preconditions_checked`,
`inference_substrate`, `duration_s`, `source_artifact_hashes`,
`process_observations`, `reproducibility_checksum`, `rows`,
`terminal_artifact_manifest`, `incomplete_artifact_manifest`,
`configuration_strata`, `supervisor_eligible_cells`, `tool_gap_eligible_cells`,
`unmatched_cell_reasons`, `arc_inventory_complete_score`,
`supervisor_cells_ready_score`, `tool_gap_cells_ready_score`, `solve_claim`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, and
`honest_verdict`.

`arc_inventory_complete_score` SHALL be 1 only when all preconditions pass, at
least one terminal row exists, duplicate run-game-stratum identities are absent,
all row hashes are present, and all unmatched cells are explicitly explained.
`supervisor_cells_ready_score` and `tool_gap_cells_ready_score` SHALL be derived
from exact eligible cell counts. These readiness fields SHALL not claim an ARC
solve or mechanism effect. `solve_claim` SHALL be false and
`verifier_is_oracle` SHALL be false. A complete inventory with no solve or
mechanism-effect claim SHALL use `verdict_class=null`; gate failures SHALL use
`verdict_class=blocked`.

### SCENARIO-ARC-6843-TERMINAL-DETECTION

**Given** complete, partial, blocked, and missing ARC artifacts
**When** Exp6843 classifies them
**Then** only terminal complete artifacts can produce eligible rows, blocked
artifacts remain terminal but ineligible, and partial or missing artifacts go to
`incomplete_artifact_manifest`.

### SCENARIO-ARC-6843-STRATUM-SEPARATION

**Given** two terminal rows for the same game with different model, policy,
budget, supervisor mode, or tool-loop state
**When** Exp6843 builds rows
**Then** the rows remain separate and duplicate identities in the same complete
stratum fail closed.

### SCENARIO-ARC-6843-PROCESS-OBSERVATION

**Given** a live ARC process and its model worker
**When** Exp6843 samples process state
**Then** it records PID, command, start time, state, and observed configuration
without sending signals, changing leases, deleting locks, or waiting for
completion.

### SCENARIO-ARC-6843-CHECKSUMS

**Given** the same terminal artifacts and process observations
**When** Exp6843 rebuilds the artifact
**Then** each source artifact hash and row hash is stable, and the top-level
`reproducibility_checksum` ignores only timing fields.

### SCENARIO-ARC-6843-NO-SOLVE-CLAIM

**Given** terminal leaderboard rows and supervisor/tool-gap receipts
**When** Exp6843 emits readiness scores
**Then** `solve_claim=false`, `verdict_class` is drawn from the closed verdict
set, complete no-effect inventories use `verdict_class=null`, and
`honest_verdict` starts with `complete_` while making no game-level solve claim.

## REQ-ARC-6844: Supervisor Action Outcome Credit Audit

Experiment 6844 SHALL audit supervisor action credit only from frozen receipt
artifacts. It SHALL not launch a live ARC run. It SHALL recompute rows from
exact action and outcome receipts. It SHALL not import prior benefit,
adoption, or readiness aggregates as outcome truth.

The producer SHALL require `arc_inventory_complete_score=1`, at least one
eligible supervisor cell, immutable attempts, exact later outcomes, action
identities, temporal order, matched cells, and nonzero matched headroom before
it can mark supervisor effect eligibility. If a gate fails, it SHALL still
write a terminal `complete_blocked_supervisor_outcome_credit_audit` artifact.
The blocked artifact SHALL include `gate_check_summary` with the failed check,
expected value, and observed value.

The reducer SHALL emit each eligible redirect and each matched control with
game, run, model, policy, budget, supervisor mode, action, receipt, next state,
later exact outcome, dose, and headroom. Dose SHALL attach to the specific
applied action identity. Direction SHALL come from the exact later trajectory
outcome. A row with no progress and no regression SHALL be recorded as
abstention, not as a benefit.

The analysis SHALL compare only matched cells. It SHALL keep configurations
with different games, tool-loop states, supervisor modes, policies, models, or
budgets in separate strata. It SHALL report unmatched configurations as
diagnostics rather than pairing them. It SHALL report transition progress,
regression, abstention, invalid-action rate, and uncertainty by stratum.

The artifact SHALL write
`results/experiment_6844_supervisor_action_outcome_credit_audit.json` with
these top-level fields: `field_principles`, `preconditions_checked`,
`inference_substrate`, `duration_s`, `source_artifact_hashes`,
`reproducibility_checksum`, `per_game_results`, `configuration_strata`,
`exact_outcome_join_results`, `action_credit_results`, `headroom_results`,
`unmatched_cell_results`, `transition_progress_results`,
`invalid_action_results`, `supervisor_causal_audit_complete_score`,
`supervisor_effect_eligible_score`, `solve_claim`, `gate_check_summary`,
`verifier_is_oracle`, `verdict_class`, and `honest_verdict`.

`inference_substrate` SHALL identify deterministic CPU live-receipt audit work.
`supervisor_causal_audit_complete_score` SHALL be derived from artifact
completeness. `supervisor_effect_eligible_score` SHALL be 1 only when matching,
timing, headroom, and exact outcome gates pass. That score SHALL mean
eligibility only. It SHALL not mean the effect is positive. `solve_claim` SHALL
be false. `verifier_is_oracle` SHALL be false because exact later outcomes are
external receipts. A complete no-solve, no-effect-eligibility audit SHALL use
`verdict_class=blocked` when any required gate fails, otherwise `verdict_class`
SHALL be `null`.

### SCENARIO-ARC-6844-ACTION-OUTCOME-JOIN

**Given** exact live outcome rows with proposal, application, environment step,
and outcome identities
**When** Exp6844 reduces action credit rows
**Then** every redirect and matched control keeps the same identity chain,
records the applied action, links dose to that action, and assigns direction
from the later outcome.

### SCENARIO-ARC-6844-GATES-FAIL-CLOSED

**Given** missing receipts, duplicate identities, invalid temporal order,
partial verification failure, or zero matched headroom
**When** Exp6844 evaluates preconditions
**Then** it writes `complete_blocked_supervisor_outcome_credit_audit`, sets
`supervisor_effect_eligible_score=0`, and records the failed check and observed
value in `gate_check_summary`.

### SCENARIO-ARC-6844-STRATA-NOT-POOLED

**Given** rows from different games, models, policies, budgets, tool-loop
states, or supervisor modes
**When** Exp6844 compares redirects with controls
**Then** only rows in the same stratum are compared, and unmatched cells stay in
`unmatched_cell_results`.

### SCENARIO-ARC-6844-HASHES-NO-SOLVE

**Given** the same frozen source artifacts
**When** Exp6844 rebuilds the audit
**Then** source hashes, row hashes, and the reproducibility checksum are stable;
`solve_claim=false`; `verifier_is_oracle=false`; and `honest_verdict` starts
with `complete_`.

## REQ-ARC-6845: Tool-Gap Causal Support Audit

Experiment 6845 SHALL audit tool-gap obligations from frozen terminal ARC
artifacts. It SHALL not launch a live ARC run. It SHALL keep request transport,
tool receipt, agent-visible response, next action, and later outcome as separate
links.

The producer SHALL require `arc_inventory_complete_score=1`, at least one
terminal tool-gap cell, immutable raw transcript hashes, request and response
identities, agent-visible receipt text, next-action receipts, exact later
outcomes, duplicate-free identities, temporal order, same-configuration
matching, and nonzero headroom before effect eligibility can be 1. If a gate
fails, it SHALL still write a terminal
`complete_blocked_tool_gap_causal_support_audit` artifact. The blocked artifact
SHALL include `gate_check_summary` with the failed check, expected value, and
observed value.

The reducer SHALL build a fresh obligation ledger. Each obligation row SHALL
record the missing fact, requested tool, actual call, exact response,
agent-visible text, next action, and later exact outcome. A transport receipt
SHALL not imply utility. A visible response with no changed next action SHALL
count as transport success but not as use.

The analysis SHALL keep each run configuration in its own stratum. Loop-off
`r11l`/`lp85` and `ls20`/`wa30` evidence SHALL remain separate from loop-on
`sp80`/`su15`, `tu93`/`cn04`, and `m0r0`/`sk48` evidence unless execution-time
receipts prove exact parity. Unmatched and no-headroom rows SHALL be reported as
diagnostics rather than pooled.

The artifact SHALL write
`results/experiment_6845_tool_gap_causal_support_audit.json` with these
top-level fields: `field_principles`, `preconditions_checked`,
`inference_substrate`, `duration_s`, `source_artifact_hashes`,
`reproducibility_checksum`, `per_game_results`, `configuration_strata`,
`obligation_ledger`, `request_receipt_joins`, `agent_visibility_results`,
`next_action_results`, `later_outcome_results`, `transport_results`,
`utility_results`, `headroom_results`, `unmatched_cell_results`,
`tool_gap_audit_complete_score`, `tool_gap_effect_eligible_score`,
`solve_claim`, `gate_check_summary`, `verifier_is_oracle`, `verdict_class`, and
`honest_verdict`.

`inference_substrate` SHALL identify deterministic CPU transcript audit work.
`tool_gap_audit_complete_score` SHALL be derived from audit completeness.
`tool_gap_effect_eligible_score` SHALL be 1 only when valid joins, timing,
matching, exact outcomes, and headroom gates pass. That score SHALL mean
eligibility only. It SHALL not mean the effect is positive. `solve_claim` SHALL
be false. `verifier_is_oracle` SHALL be false. A complete no-solve,
no-effect-eligibility audit SHALL use `verdict_class=blocked` when any required
gate fails; otherwise `verdict_class` SHALL be `null`.

### SCENARIO-ARC-6845-REQUEST-RECEIPT-JOIN

**Given** a tool-gap obligation with request, call, response, and transcript
identities
**When** Exp6845 reduces the row
**Then** the request-to-receipt join is explicit, duplicate-free, and hash-bound.

### SCENARIO-ARC-6845-VISIBILITY-NEXT-ACTION

**Given** a tool response that was rendered back to the agent
**When** Exp6845 measures utility
**Then** transport success, response use, and next-action change are reported as
separate metrics.

### SCENARIO-ARC-6845-GATES-FAIL-CLOSED

**Given** missing outcomes, missing identities, invalid timing, duplicate
identities, no terminal tool-gap cells, or zero headroom
**When** Exp6845 evaluates gates
**Then** it writes `complete_blocked_tool_gap_causal_support_audit`, sets
`tool_gap_effect_eligible_score=0`, and records the failed check and observed
value in `gate_check_summary`.

### SCENARIO-ARC-6845-STRATA-NOT-POOLED

**Given** loop-off and loop-on rows for the same model, policy, and budget
**When** Exp6845 builds strata
**Then** rows remain separated by game, run, model, policy, budget, tool-loop
state, supervisor state, and requested tool.

### SCENARIO-ARC-6845-HASHES-NO-SOLVE

**Given** the same frozen source artifacts
**When** Exp6845 rebuilds the audit
**Then** source hashes, raw transcript hashes, row hashes, and the
reproducibility checksum are stable; `solve_claim=false`;
`verifier_is_oracle=false`; and `honest_verdict` starts with `complete_`.

## REQ-ARC-6846: Typed ARC Shadow Monitor

Experiment 6846 SHALL wire the Exp6836 typed obligation program into the
canonical ARC supervisor and tool-gap action seam as a default-off shadow
monitor. The monitor SHALL be reachable from
`make_carnot_agent -> E3AgentPolicy`, SHALL use one shared generic atom mapping
to the Exp6836 typed program, and SHALL never add a per-game adapter, game
recipe, source-derived model, offline search path, live run, or game-level solve
claim.

The producer SHALL require `typed_obligation_program_ready_score=1`,
`arc_inventory_complete_score=1`, canonical source identity, terminal replay
rows, exact external labels, and default-off configuration before readiness can
be 1. If any gate fails, it SHALL still write terminal
`complete_blocked_typed_arc_shadow_monitor` output with `gate_check_summary`
naming the failed check, expected value, and observed value.

The replay SHALL read only frozen Exp6836 and Exp6843 evidence. For every frozen
terminal row it SHALL emit the shadow guard decision, typed-program energy,
per-atom diagnostic, exact external label, false intervention, missed violation,
agreement, error type, latency, and byte-identity result for the unmutated
action. Disabled mode SHALL preserve byte-identical action output.

The artifact SHALL write
`results/experiment_6846_typed_arc_shadow_monitor.json` with these top-level
fields: `field_principles`, `preconditions_checked`, `inference_substrate`,
`duration_s`, `source_artifact_hashes`, `reproducibility_checksum`,
`default_off_receipt`, `canonical_reachability_receipt`,
`atom_mapping_manifest`, `per_game_results`, `exact_agreement_results`,
`false_intervention_results`, `missed_violation_results`, `latency_results`,
`action_byte_identity_results`, `typed_arc_shadow_monitor_ready_score`,
`solve_claim`, `gate_check_summary`, `verifier_is_oracle`, `verdict_class`, and
`honest_verdict`.

`inference_substrate` SHALL be `deterministic CPU canonical-path shadow replay`.
`typed_arc_shadow_monitor_ready_score` SHALL be derived from canonical
reachability, default-off safety, replay determinism, complete diagnostics, and
bounded latency. `solve_claim` SHALL be false. `verifier_is_oracle` SHALL be
false because external trajectory facts define truth. A terminal artifact SHALL
use a closed `verdict_class` and an `honest_verdict` that starts with
`complete_`.

### SCENARIO-ARC-6846-DEFAULT-OFF-NO-ACTION-MUTATION

**Given** the submitted ARC policy with
`CARNOT_ARC_TYPED_OBLIGATION_SHADOW_MONITOR` unset
**When** an action exits the canonical decision seam
**Then** no monitor is constructed by default and the returned action bytes are
identical to the proposed action bytes.

### SCENARIO-ARC-6846-CANONICAL-REACHABILITY

**Given** the normal live entrypoint
`make_carnot_agent -> E3AgentPolicy`
**When** Exp6846 inspects the canonical agent source
**Then** it proves the shadow monitor hook is reachable at the trajectory
supervisor observation seam and the tool-gap-aware action seam.

### SCENARIO-ARC-6846-ATOM-MAPPING

**Given** Exp6836 typed obligation atoms and Exp6843 frozen ARC rows
**When** the shadow replay maps live row facts to obligations
**Then** the manifest maps generic live atoms to shared Exp6836 typed-program
atom fields without a per-game adapter, game recipe, source-derived model, or
offline search path.

### SCENARIO-ARC-6846-FAIL-CLOSED-DIAGNOSTICS

**Given** a missing ready score, stale source identity, absent terminal row,
missing exact label, or non-default configuration
**When** Exp6846 evaluates gates
**Then** it writes `complete_blocked_typed_arc_shadow_monitor`, sets
`typed_arc_shadow_monitor_ready_score=0`, and records the failed check and
observed value in `gate_check_summary`.

### SCENARIO-ARC-6846-REPLAY-DETERMINISM

**Given** the same frozen Exp6836 and Exp6843 inputs
**When** Exp6846 rebuilds the replay artifact
**Then** row hashes, action identity hashes, source hashes, and the
reproducibility checksum are stable except for wall-clock duration.

### SCENARIO-ARC-6846-LATENCY

**Given** every frozen terminal row
**When** the shadow monitor evaluates its typed guard
**Then** each row records a nonnegative bounded latency and readiness is 1 only
if the latency bound passes.

### SCENARIO-ARC-6846-MISSING-FIELDS

**Given** a replay row with missing required receipt fields
**When** Exp6846 validates the inventory
**Then** the artifact fails closed before readiness and reports the missing
field names in the gate summary.

### SCENARIO-ARC-6846-ARTIFACT-NO-SOLVE

**Given** the same frozen source artifacts
**When** Exp6846 writes its artifact
**Then** it reports no policy benefit, `solve_claim=false`,
`verifier_is_oracle=false`, `verdict_class` in the closed set, and an
`honest_verdict` that starts with `complete_`.

## REQ-ARC-6857: Dynamic Live ARC Receipt Router

Experiment 6857 SHALL discover terminal ARC receipts at execution time. It
SHALL not launch a game. It SHALL not wait for or interrupt a live process.
It SHALL not use game source code. It SHALL not select evidence from one
hard-coded experiment path or from file modification time alone.

The router SHALL require `v599_evidence_contract_ready_score=1`. It SHALL
require readable `ops/arc_solve_registry.yaml` and `ops/arc_flag_ledger.yaml`.
It SHALL require an empty solve scope. A failed precondition SHALL produce
`complete_blocked_dynamic_live_arc_receipt_router`. The blocked artifact SHALL
name the failed check and its observed value in `gate_check_summary`.

The router SHALL define explicit schemas for supervisor, lever-harness,
tool-loop, shadow, and canonical-agent receipts. It SHALL discover candidates
under declared repository-relative roots. It SHALL validate terminal status,
artifact completeness, stored hashes, experiment identity, generator
provenance, live reachability, exact row identity, and exact later outcomes.
It SHALL ignore stale stored paths. It SHALL quarantine ambiguous newest
candidates, changed generators, mixed policies, partial artifacts, stale
hashes, missing exact outcomes, missing agent-visible tool receipts,
development proxies, outer-loop reverse engineering, source-reading rows,
pooled policy changes, flagged verification failures, and duplicate row
identities.

Each normalized row SHALL record its source path and hash. It SHALL also
record experiment ID, terminal status, game, model, generator provenance,
policy hash, budget, supervisor mode, tool mode, attempt identity, live seam,
row kind, row identity, and exact outcome when present. Supervisor action and
tool event rows SHALL join only to the exact next action, transition, and later
level outcome from the same attempt. An unmatched row SHALL remain in
`unmatched_receipt_rows`.

The router SHALL keep changed configurations in separate strata. It SHALL
compute supervisor headroom only for provenance-qualified matched
opportunities. `supervisor_headroom_ready_score` SHALL be 1 only when at least
one such opportunity has nonzero exact outcome headroom. It SHALL count
first-party tool-gap chains separately. Effect eligibility SHALL require both
live-agent reachability and generator provenance.

The artifact SHALL write
`results/experiment_6857_dynamic_live_arc_receipt_router.json`. It SHALL have
these top-level fields: `field_principles`, `preconditions_checked`,
`inference_substrate`, `duration_s`, `source_artifact_hashes`,
`discovery_roots`, `accepted_schema_manifest`, `rejected_source_manifest`,
`rows`, `provenance_qualified_manifest`, `generator_provenance_rows`,
`live_reachability_rows`, `configuration_strata`,
`supervisor_headroom_rows`, `first_party_tool_gap_rows`,
`unmatched_receipt_rows`, `arc_receipt_router_complete_score`,
`supervisor_headroom_ready_score`,
`tool_gap_first_party_receipts_ready_score`, `solve_claim`,
`game_level_solve_count`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, and `honest_verdict`.

`inference_substrate` SHALL equal
`read_only_terminal_live_artifact_discovery`. `solve_claim` SHALL be false.
`game_level_solve_count` SHALL be zero. `verifier_is_oracle` SHALL be false.
`verdict_class` SHALL be one of `positive`, `circular_positive`, `null`,
`blocked`, `disqualified`, or `partial`. `honest_verdict` SHALL start with
`complete_`.

### SCENARIO-ARC-6857-STALE-PATH-AND-AMBIGUITY

**Given** a stale stored path or two terminal candidates with the same exact
attempt identity and conflicting hashes
**When** Exp6857 discovers evidence
**Then** it ignores the stale path, does not use modification time as a tie
breaker, and quarantines the ambiguous candidates.

### SCENARIO-ARC-6857-PARTIAL-HASH-AND-GENERATOR

**Given** a partial artifact, a stale declared hash, or changed generator
provenance
**When** Exp6857 validates the candidate
**Then** the candidate is ineligible and its exact reason appears in
`rejected_source_manifest`.

### SCENARIO-ARC-6857-EXACT-JOINS

**Given** supervisor or tool rows with exact attempt and event identities
**When** Exp6857 joins receipts
**Then** it links only the exact next action, transition, and later level
outcome; missing exact outcomes or missing agent-visible tool receipts remain
unmatched and cannot become effect eligible.

### SCENARIO-ARC-6857-CONFIGURATION-SEPARATION

**Given** rows with different policy hashes, budgets, supervisor modes, tool
modes, models, or games
**When** Exp6857 builds strata
**Then** it keeps the rows separate and quarantines any pooled policy change.

### SCENARIO-ARC-6857-DUPLICATE-ROW-IDENTITY

**Given** two rows with the same exact row identity
**When** Exp6857 freezes the manifest
**Then** both rows are ineligible, readiness fails closed, and the duplicate
identity appears in `gate_check_summary`.

### SCENARIO-ARC-6857-HEADROOM-AND-TOOL-CHAIN

**Given** provenance-qualified exact supervisor joins and first-party tool-gap
chains
**When** Exp6857 computes readiness
**Then** supervisor readiness requires nonzero exact outcome headroom, while
first-party tool-gap readiness is counted separately.

### SCENARIO-ARC-6857-NO-SOLVE-AND-PROCESS-OBSERVATION

**Given** terminal artifacts and any currently live ARC process
**When** Exp6857 writes the frozen manifest
**Then** it records stable hashes and read-only process observations, sends no
signal, waits for no process, claims no solve, and emits a terminal
`complete_` verdict.

## REQ-ARC-6859: First-Party Tool-Gap Receipt Wiring

Experiment 6859 SHALL add a first-party receipt contract at the canonical live
agent seam. The contract SHALL be default-off, SHALL NOT change the default
agent policy, SHALL NOT enable a new tool action, SHALL NOT read game source,
and SHALL NOT launch a live game. It SHALL require
`arc_receipt_router_complete_score=1`, reachable canonical dispatch, delivery,
and action/outcome seams, and the stable
`carnot.arc.first_party_tool_gap_receipt.v1` schema. A failed precondition SHALL
write `complete_blocked_first_party_tool_gap_receipt_wiring` and name the failed
check and observed value in `gate_check_summary`.

One immutable receipt identity SHALL bind attempt, decision point, gap,
request, response, agent-visible delivery, next action, and exact later outcome.
Every hop SHALL store a timestamp, source hash, payload hash, hop identity, and
prior-hop identity. Persistence SHALL survive restart. Byte-identical duplicate
hops SHALL deduplicate, while conflicting duplicate identities SHALL be
quarantined and excluded from complete joins.

The artifact SHALL keep receipt transport, agent visibility, response use,
action change, progress, and causal eligibility as separate facts. Fixture,
terminal replay, authentic-live, reconstructed, and development-proxy rows
SHALL remain separately labelled. Reconstructed and development-proxy rows
SHALL be quarantined from live utility claims. The contract-ready score SHALL
depend only on schema stability and reachable transport completeness. The
live-effect eligibility score SHALL require an authentic-live, first-party,
agent-visible, response-used, exact later-outcome join with valid headroom;
fixture transport SHALL NOT open it.

The artifact SHALL write
`results/experiment_6859_first_party_tool_gap_receipt_wiring.json`. It SHALL
contain `field_principles`, `preconditions_checked`, `inference_substrate`,
`duration_s`, `source_artifact_hashes`, `reproducibility_checksum`, `rows`,
`receipt_schema`, `canonical_seam_manifest`, `gap_detection_rows`,
`request_rows`, `tool_response_rows`, `agent_delivery_rows`,
`next_action_rows`, `exact_outcome_rows`, `join_completeness_rows`,
`provenance_class_rows`, `restart_results`, `deduplication_results`,
`default_off_verified`, `tool_gap_receipt_contract_ready_score`,
`tool_gap_live_effect_claim_eligible_score`, `solve_claimed`,
`game_level_solve_count`, `gate_check_summary`, `verifier_is_oracle`,
`verdict_class`, and `honest_verdict`.

`inference_substrate` SHALL equal
`canonical_live_seam_default_off_fixture_and_terminal_replay`.
`solve_claimed` SHALL be false, `game_level_solve_count` SHALL be zero, and
`verifier_is_oracle` SHALL be false. `verdict_class` SHALL be one of
`positive`, `circular_positive`, `null`, `blocked`, `disqualified`, or
`partial`; `honest_verdict` SHALL start with `complete_`.

### SCENARIO-ARC-6859-NO-GAP-AND-DEFAULT-OFF

**Given** no detected gap or an unset receipt flag
**When** the canonical policy selects an action
**Then** no receipt row or file is created and the returned action identity is
unchanged.

### SCENARIO-ARC-6859-GAP-REJECTION-AND-TOOL-ERROR

**Given** an unknown request, malformed arguments, or a tool exception
**When** canonical dispatch returns its existing error response
**Then** gap, request, and response hops share one immutable receipt identity
without altering dispatch behavior.

### SCENARIO-ARC-6859-VISIBILITY-USE-AND-NEXT-ACTION

**Given** transported responses that may be hidden, delivered, used, or unused
**When** the real induction loop and `E3AgentPolicy.next_move` seams run
**Then** visibility, response use, action change, and next-action identity are
recorded separately.

### SCENARIO-ARC-6859-EXACT-OUTCOME-AND-CAUSAL-ELIGIBILITY

**Given** a next action and its exact later observation
**When** the receipt is completed
**Then** progress is joined without equating it to causation, and effect
eligibility requires authentic-live provenance, explicit use, and headroom.

### SCENARIO-ARC-6859-PERSISTENCE-RESTART-AND-DEDUPLICATION

**Given** complete and pending persisted chains
**When** the transport restarts or sees a duplicate hop
**Then** identities persist, identical bytes deduplicate, and conflicts fail
closed.

### SCENARIO-ARC-6859-FIXTURE-REPLAY-AND-PROVENANCE

**Given** deterministic fixtures and provenance-qualified terminal receipts
**When** Exp6859 runs them through the real seam and replays terminal rows
**Then** fixture, replay, and authentic-live evidence stay separate and proxy
or reconstructed evidence cannot support a live utility claim.

### SCENARIO-ARC-6859-BLOCKED-GATE-AND-NO-SOLVE

**Given** a failed gate or any completed receipt audit
**When** Exp6859 writes its terminal artifact
**Then** blocked gates name their failed check and observed value, no solve is
claimed, the game-level solve count is zero, and the verdict is row-supported.

## REQ-ARC-6861: Generalization-Floor ARC-Scope Detection Uses Word-Boundary Matching

`scripts/arc_levelup_guarantee_lint.py:_is_generalization_attempt` SHALL decide
whether a task prompt is ARC-scoped with token-aware matching, never a bare
substring test. The prior test `"arc" in prompt` matched inside "research".
Almost every task prompt in this repo names research-roadmap,
research_conductor, or research-program. So the scope gate passed for every
task, and one generic ML phrase such as "held-out" completed a match. Milestone
2026.09.609 shipped 12 tasks with zero ARC work, and the lint reported
"OK (soft): 3 generalization-testing-floor task(s) detected".

A prompt SHALL count as ARC-scoped only when it contains at least one of:

- a public survey game id (the existing `_GAMES` word-boundary regex);
- the word `arc` at word boundaries (covers "ARC", "ARC-AGI-3");
- an `arc_*` / `arc-*` / `arc<digit>*` identifier (`arc_solver_kit`,
  `arc3_replay_scorecard`);
- a live-agent entrypoint name (`E3AgentPolicy`, `make_carnot_agent`);
- the offline `arcade` harness, or `arcprize`.

Words that merely contain the letters a-r-c ("research", "archive",
"architecture", "march") SHALL NOT scope a prompt as ARC. The scope pattern
stays deliberately generous in the ARC direction: a genuine ARC task that
stops counting would make the floor warn on compliant roadmaps and teach
readers to ignore the warning, which is worse than a rare over-match. The
check stays WARN-only per the CLAUDE.md "ARC-AGI-3 Generalization-Testing
Floor" rule.

### SCENARIO-ARC-6861-RESEARCH-IS-NOT-ARC-SCOPE

**Given** a task prompt that names research-program / research_conductor and a
generalization signal such as "held-out", with no ARC content
**When** `_is_generalization_attempt` evaluates it
**Then** the prompt does not count toward the generalization-testing floor.

### SCENARIO-ARC-6861-609-ROADMAP-COUNTS-ZERO

**Given** the pinned milestone 2026.09.609 roadmap prompts
(`tests/python/fixtures/roadmap_2026_09_609_prompts.yaml`), which contain zero
ARC tasks
**When** `count_generalization_attempts` runs over them
**Then** the count is 0, where the pre-fix substring test reported 3.

### SCENARIO-ARC-6861-GENUINE-ARC-TASK-STILL-COUNTS

**Given** a prompt naming the word ARC, an arc_* identifier, an arc3_*
identifier, a live entrypoint, or a game id, together with a generalization
signal
**When** `_is_generalization_attempt` evaluates it
**Then** the prompt counts toward the floor (no under-match regression).

## REQ-ARC-FLAG-LEDGER-6862: A measured null is a finding, not a coverage gap

The flag ledger SHALL distinguish "nobody has measured this flag" from "this
flag was measured and did not help". Before this requirement the only
transition out of `unevaluated` was promotion to `on`
(REQ-ARC-FLAG-LEDGER-6268), so a flag that was measured and refused promotion
recorded its evidence, set `promotable: False`, and stayed `unevaluated`
forever. As of 2026-09-03 the ledger held 136 flags, all `unevaluated`, 15 of
them already carrying evidence -- and every consumer read the 15 findings as
untested work. The two facts demand opposite responses: a coverage gap says go
measure; a measured null says stop spending.

The states are: `unevaluated` (never measured), `off_measured` (measured, did
not earn promotion -- the terminal measured-null state), and `on` (promoted on
evidence). `scripts/arc_flag_ledger.py:state_after_measurement` decides the
transition:

- A REFUSED or HOLD verdict moves `unevaluated` to `off_measured`.
- An `UNINTERPRETABLE_*` verdict moves nothing: the arm timed out or the lever
  never took effect, so nothing was measured, and filing that as a null is the
  exact conflation `verdict()` forbids. The human path for a
  FIRED_NO_EFFECT judged a real null (after reading fire_counters) is
  `--record-null`.
- `on` is never demoted by measurement bookkeeping. Demoting a shipped
  default is operator judgment.
- Existing ledger entries are NOT migrated. The new state applies to
  measurements made after this requirement landed; reclassifying past entries
  is an operator call.

`--record-null FLAG` SHALL record a measured null from a run made outside
`--measure`/`--sweep` (the r11l tools A/B shape: a leaderboard eval pair the
sweep never sees). It requires a non-empty `--note` and at least one existing
`--evidence-path`, records each path with its sha256, sets
`promotable: False` and `state: off_measured`, and refuses: an empty note, no
paths, a missing path, an untracked flag, and a flag whose state is `on`.

`scripts/outer_loop_dashboard.py:flag_lines` SHALL count only `unevaluated`
flags as shipped-but-untested and SHALL report `off_measured` flags
separately, never folded into the untested count.

### SCENARIO-ARC-FLAG-LEDGER-6862-MEASURED-NULL-IS-NOT-UNEVALUATED

**Given** an unevaluated flag whose measurement verdict is HOLD or REFUSED
**When** the measurement is recorded (via `--measure` or the sweep)
**Then** the flag's state becomes `off_measured`, distinguishable from
`unevaluated` by any consumer.

### SCENARIO-ARC-FLAG-LEDGER-6862-UNINTERPRETABLE-CLAIMS-NO-NULL

**Given** a measurement whose verdict is `UNINTERPRETABLE_*` (timed out, or
the lever never took effect)
**When** the measurement is recorded
**Then** the state does not move, because nothing was measured.

### SCENARIO-ARC-FLAG-LEDGER-6862-ON-IS-NEVER-DEMOTED-BY-BOOKKEEPING

**Given** a flag whose state is `on`
**When** a later measurement refuses promotion, or `--record-null` is invoked
**Then** the state stays `on` (record-null refuses outright), because flipping
a shipped default off is operator judgment.

### SCENARIO-ARC-FLAG-LEDGER-6862-EXTERNAL-NULL-IS-CHECKABLE

**Given** an A/B run made outside the sweep whose outcome is a clean null
**When** `--record-null` records it with a note and evidence paths
**Then** the entry carries the note, each path with its sha256, `promotable:
False`, and `state: off_measured`; and the command refuses an empty note,
zero paths, a missing path, or an untracked flag.

### SCENARIO-ARC-FLAG-LEDGER-6862-DASHBOARD-SEPARATES-THE-TWO-FACTS

**Given** a flag set containing unevaluated, off_measured, and on states
**When** the dashboard renders its flags block
**Then** the shipped-but-untested count includes only `unevaluated` flags and
`off_measured` flags are listed separately as measured-null.

## REQ-ARC-FLAG-SWEEP-6271: Sweep values preserve flag semantics

The ARC flag ledger SHALL classify flags from how their environment values are
used before admitting them to an automated `=1` sweep. Boolean capability
toggles MAY be swept. Numeric knobs, filesystem paths, inverse/disable flags,
guards, permissions, and unknown values SHALL be excluded so that a sweep
cannot turn a timeout into one second, replace a path with `1`, remove a
capability, or weaken a guard and then record the resulting damage as evidence
about the capability.

Classification SHALL follow a value assigned from `os.environ.get(...)` to a
local `int(...)`, `float(...)`, or `Path(...)` conversion in the same lexical
scope. It SHALL retain that classification when the environment read and the
conversion are on separate lines or separated by validation and error handling.

### SCENARIO-ARC-FLAG-SWEEP-6271-MULTILINE-NUMERIC

**Given** `CARNOT_ARC_INDUCE_TIMEOUT` is read into a local variable and that
variable is converted with `int(...)` later in the same function
**When** the flag ledger classifies the flag for an automated sweep
**Then** it classifies the flag as numeric and excludes it from the `=1` sweep.

## Implementation Status (REQ-ARC-FLAG-SWEEP-6271)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-ARC-FLAG-SWEEP-6271 and SCENARIO-ARC-FLAG-SWEEP-6271-MULTILINE-NUMERIC | Implemented (`scripts/arc_flag_ledger.py`: lexical assignment-to-conversion tracing) | Implemented (`tests/python/test_arc_bench_and_flag_ledger.py::test_a_numeric_knob_is_never_swept`; conductor-equivalent shard and changed-line coverage) |

## REQ-ARC-7010: Forward evaluation rows carry complete, fail-closed provenance

Every newly produced ARC evaluation row SHALL carry one top-level
`arc_eval_provenance` record built and validated by the same versioned schema.
The record SHALL explicitly bind hardware identity, model identity, context,
server execution, lease authority, completion accounting, policy and factory
identity, source commit, and `solve_provenance`. Consumers SHALL validate that
record before granting headline eligibility and SHALL reject missing, null,
malformed, aliased, extra, or contradictory fields. They SHALL NOT infer
missing provenance from filenames, ports, process reuse, or historical result
files, and existing historical artifacts SHALL NOT be rewritten or backfilled.

A live-LLM record SHALL require GPU UUID and model, CUDA device, model
repository, filename, and content hash, `n_ctx`, server binary and command
hashes, endpoint and port, a current lease identity and time window, request,
completion, and error counters, policy and factory hashes, git commit, and one
of `live_agent_self_discovery`, `development_proxy`, or `outer_loop_re` as its
`solve_provenance`. Counter totals, endpoint/port identity, lease time bounds,
and row/record solve provenance SHALL agree. A partial request or a stale lease
is ineligible.

A path that invoked no LLM MAY remain a legal evaluation row only when it uses
the canonical no-LLM substrate, supplies explicit `not_applicable` values for
every hardware/model/context/server/lease field, and reports zero request,
completion, and error counters. Such a row SHALL never claim GGUF or CUDA use.

### SCENARIO-ARC-7010-LIVE-ROUND-TRIP

**Given** a complete deterministic CUDA-shaped evaluation record
**When** the producer builds it and the consumer validates it
**Then** it round-trips under one shared required-key schema with a stable
record hash and is eligible for headline use.

### SCENARIO-ARC-7010-REJECTION-MATRIX

**Given** one fixture for each required field and variants that are absent,
null, malformed, aliased, extra, or contradictory
**When** the consumer validates the fixtures
**Then** every invalid fixture fails closed, including partial counters,
endpoint/port reuse mismatch, stale leases, and row/record solve disagreement.

### SCENARIO-ARC-7010-NO-LLM-IS-EXPLICIT

**Given** an evaluation path that invoked neither a model nor CUDA
**When** its provenance is built
**Then** the canonical no-LLM substrate and explicit `not_applicable` values
are accepted, while any GGUF, GPU, CUDA, or non-zero completion claim rejects.

### SCENARIO-ARC-7010-SOLVE-PROVENANCE-ENUM

**Given** otherwise complete evaluation rows for live self-discovery,
development proxy, and outer-loop re-evaluation
**When** each row is validated
**Then** all three named values are accepted and any absent, aliased, or
unknown value rejects.

### SCENARIO-ARC-7010-HISTORICAL-ROWS-STAY-HISTORICAL

**Given** a historical artifact without the forward contract
**When** a headline consumer encounters it
**Then** the consumer preserves the artifact bytes but refuses headline
eligibility instead of guessing or backfilling provenance.

## Implementation Status (REQ-ARC-7010)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-ARC-7010 and all SCENARIO-ARC-7010 variants | Implemented (`python/carnot/agentic/arc_eval_provenance.py`, `python/carnot/experiment_7010_arc_eval_provenance_contract.py`, `scripts/arc_leaderboard_eval.py`, `scripts/outer_loop_dashboard.py`) | Implemented (`tests/python/test_arc_eval_provenance_contract_20260905.py`, `tests/python/test_eval_generator_provenance.py`; 100% scoped statement coverage for the two provenance contract modules) |

## REQ-ARC-7025: One Live Belief Shadow Trace Carries Complete Transport Evidence

Experiment 7025 SHALL run one bounded transport trace through
`make_carnot_agent` and `E3AgentPolicy`. It SHALL evaluate the exact bare
Exp7017 `task_compute_receipt_ready_score=1` and Exp7024
`belief_selector_live_path_ready_score=1` gates before live work. It SHALL
resolve `MODEL_SPECS` through `cached_sota_pair()` and use the exact cached
`unsloth/Qwen3.6-35B-A3B-GGUF` file. A legacy model MAY appear only in a
separate smoke-test label and SHALL NOT satisfy this requirement.

The live preflight SHALL require a CUDA-enabled `llama-server`, one supported
idle RTX 3090 with enough free VRAM, writable checkpoint and result paths, an
eligible official live episode, an owned GPU lease, an owned server and port,
and authority to stop only those owned resources. Failure SHALL write a
schema-complete artifact with `verdict_class=blocked`, an `honest_verdict`
starting with `blocked_belief_shadow_live_trace`, and the exact failed check,
expected value, and observed value. It SHALL NOT use CPU, HIP, a substitute
model, a fixture episode, or an unowned server as evidence.

The trace SHALL compare a no-belief control cell and a belief-shadow cell with
the same seed, action budget, prompt, model settings, visible observation, and
candidate set. The shadow selector SHALL execute the bounded belief query and
MAY record a counterfactual ranking change. It SHALL return the unchanged
control candidate order, so each emitted shadow action equals its matched
control action. The run SHALL not inspect game source, use offline
ground-truth search, load a per-game adapter, target a reproduced level, bank a
new level, or modify the solve registry.

Each completed cell SHALL be checkpointed. A restart SHALL resume completed
cells without issuing their model request or live action again. The artifact
SHALL count requests, completed responses, errors, actions, belief queries,
abstentions, and ranking changes. It SHALL carry one valid Exp7010 provenance
record for each evaluation cell plus task-linked setup, model-load, inference,
output-write, and cleanup receipts. The receipts SHALL bind the model file and
hash, observed server `n_ctx`, CUDA offload argument, GPU identity and samples,
model process, server process and port, lease, sequential-runner decision,
counters, checkpoint, and owned-resource teardown.

`belief_shadow_trace_ready_score` SHALL equal bare integer one only when the
mandated SOTA file executed with CUDA offload, at least one belief query fired,
all shadow actions exactly matched control actions, every Exp7010 provenance
record and task-compute receipt validated, the registry remained unchanged,
the checkpoint was resumable, and cleanup confirmed that each owned process,
lease, and port was released. This score is transport readiness only. It SHALL
make no value or solve claim.

### SCENARIO-ARC-7025-MODEL-CUDA-IDENTITY

**Given** the Qwen model was selected through `cached_sota_pair()`
**When** the server starts and inference completes
**Then** the file name and SHA-256 match the selected spec, `-ngl` proves CUDA
offload, `/props` supplies the observed `n_ctx`, and the GPU UUID, process,
server, port, and lease identities agree.

### SCENARIO-ARC-7025-SHADOW-PARITY-AND-QUERY

**Given** matched control and belief-shadow cells over one visible live frame
**When** both policies choose an action
**Then** the belief query count increases, ranking influence is recorded, the
candidate set stays identical, and the shadow action equals the control action.

### SCENARIO-ARC-7025-PROVENANCE-AND-COUNTERS

**Given** a completed live request for each cell
**When** the artifact is reduced
**Then** Exp7010 provenance validates and request, completion, error, action,
query, abstention, and ranking-change counters equal the underlying rows.

### SCENARIO-ARC-7025-MISSING-RECEIPT-FAILS-CLOSED

**Given** an otherwise complete artifact with a model, CUDA, server, lease,
counter, phase, runner, checkpoint, or teardown receipt removed or changed
**When** the artifact validator runs
**Then** readiness becomes zero and the missing or contradictory receipt is
named without reconstruction from another field.

### SCENARIO-ARC-7025-CHECKPOINT-RESUME

**Given** the control cell was checkpointed before an interruption
**When** the run resumes
**Then** it reuses the sealed completed cell, runs only the remaining shadow
cell, and records that no completed-cell request or action was duplicated.

### SCENARIO-ARC-7025-OWNED-TEARDOWN

**Given** the experiment owns one GPU lease and one model server process and
port
**When** cleanup runs on success or failure
**Then** only those owned resources are stopped, the process is reaped, the
lease is released, the port is free, and the cleanup phase is terminal.

## REQ-ARC-7030: ARC model identity joins snapshot intent to server observation

The shared ARC evaluation provenance schema SHALL keep the requested Hugging
Face snapshot path separate from the path that the server reports. The current
schema SHALL preserve `requested_model_path`, `requested_model_filename`,
`requested_hf_id`, `requested_revision`, `observed_server_model_path`,
`resolved_model_path`, and `model_file_hash` as distinct fields. It SHALL also
retain the existing hardware, server, lease, counter, and solve-provenance
checks. The validator SHALL read complete legacy schema rows through an
explicit version branch. It SHALL not rewrite historical artifacts.

The current producer SHALL accept one live model identity only when all of the
following facts agree:

- The requested path is an existing file in the selected Hugging Face
  snapshot.
- The requested basename is one `.gguf` filename.
- The snapshot hub ID and revision equal the selected model specification.
- The requested path resolves to the observed canonical server path, or the
  two canonical files have the same SHA-256 content hash.
- The resolved file is the content-addressed blob that the selected snapshot
  reaches.
- The recorded file hash equals the bytes of both accepted paths.

The validator SHALL reject a wrong hash, hub ID, revision, missing file,
broken symlink, directory, misleading `.gguf` basename, or an observed blob
outside the selected snapshot relation. It SHALL not infer identity from a
basename or file size. The canonical evaluation producer and the live belief
shadow runner SHALL use the same identity builder and validator.

### SCENARIO-ARC-7030-EXP7025-SNAPSHOT-BLOB-JOIN

**Given** an Exp7025-shaped snapshot `.gguf` symlink whose target is an
extensionless Hugging Face blob
**When** the server reports that canonical blob path
**Then** the shared current-schema receipt preserves both paths and joins them
by exact content hash, hub ID, revision, and requested filename.

### SCENARIO-ARC-7030-IDENTITY-NEGATIVE-MATRIX

**Given** one mutation for each identity field or filesystem relation
**When** the shared builder or validator checks the receipt
**Then** wrong hash, hub, revision, missing file, broken symlink, directory,
misleading suffix, and unreachable observed blob all fail closed.

### SCENARIO-ARC-7030-LEGACY-VERSION-READ

**Given** a complete valid provenance row from the prior schema
**When** the current consumer reads the row
**Then** the explicit legacy branch accepts the unchanged row
**And** current-schema fields are not guessed or backfilled.

### SCENARIO-ARC-7030-SHARED-PRODUCER-WIRING

**Given** the canonical submitted evaluator and the live belief shadow runner
**When** either path records a live model request
**Then** both paths call the shared current-schema identity bridge
**And** neither path constructs identity from a server basename or alias.

## REQ-ARC-7031: A cold process audits the model identity boundary

Before a live ARC model can receive credit, an independent process SHALL test
the shared model identity bridge with new filesystem fixtures. The process
SHALL use isolated Python startup, a private working directory, and a minimal
environment. It SHALL import the production validator from
`carnot.agentic.arc_eval_provenance`. It SHALL not import fixture data from
Exp7030.

The audit SHALL first require a readable and valid Exp7030 artifact with
`arc_model_identity_bridge_ready_score` equal to one. It SHALL recompute the
artifact hash and every source hash that Exp7030 cites. Any missing source,
changed hash, import failure, or unwritable path SHALL produce a terminal
`blocked_arc_model_identity_cold_audit` artifact. The gate summary SHALL record
the exact failed check, expected value, and observed value.

The cold process SHALL accept a snapshot `.gguf` symlink when it resolves to
the extensionless content-addressed blob reported by the server. The complete
current provenance receipt SHALL survive JSON serialization and shared
validation. A direct regular `.gguf` file in the selected snapshot SHALL also
remain valid when the server reports that exact canonical file. A complete
legacy provenance row SHALL remain valid without added current-schema fields.

The audit SHALL change one identity fact per negative row. It SHALL reject a
changed content hash, repository, revision, requested filename, observed path,
broken link, invalid path type, stale server identity, same-size different
bytes, and ambiguous hard link. Each rejected row SHALL record the exact shared
validator reason. A file size or filesystem alias SHALL not establish model
identity.

The audit SHALL prove that the live belief consumer imports the audited shared
builder and validator path. A local copied validator SHALL fail this check.
`arc_model_identity_audit_ready_score` SHALL equal one only when both positive
rows pass, all negative rows fail closed, both regression rows pass, all source
hashes agree, the fresh process terminates successfully, and consumer
reachability is proven.

### SCENARIO-ARC-7031-COLD-SNAPSHOT-ROUND-TRIP

**Given** a new snapshot link and extensionless blob in a private directory
**When** an isolated process builds and serializes the full current receipt
**Then** the shared validator accepts the unchanged record after deserialization.

### SCENARIO-ARC-7031-ONE-FACTOR-MUTATIONS

**Given** ten independent fixtures that each change one identity fact
**When** the isolated process calls the shared identity bridge
**Then** every mutation fails closed with its exact rejection reason.

### SCENARIO-ARC-7031-DIRECT-AND-LEGACY-REGRESSIONS

**Given** a direct regular snapshot `.gguf` file and a complete legacy row
**When** the shared validator checks each fixture
**Then** both remain accepted without aliases or inferred current fields.

### SCENARIO-ARC-7031-UPSTREAM-DRIFT-BLOCKS

**Given** a missing or changed Exp7030 artifact or cited source
**When** the parent process checks the upstream evidence
**Then** it writes a schema-complete blocked artifact and does not start the cold audit.

### SCENARIO-ARC-7031-LIVE-CONSUMER-REACHABILITY

**Given** the live belief consumer and the shared provenance module
**When** the cold process compares imported function identity and source paths
**Then** the consumer reaches the audited shared validator and has no local copy.

## REQ-ARC-7039: A live capture separates model reports from resolved identity

One owned CUDA llama.cpp process SHALL load the cached
`unsloth/Qwen3.6-35B-A3B-GGUF` file selected through `cached_sota_pair()`.
The process SHALL run on an idle supported RTX 3090 under owned GPU and port
leases. It SHALL issue exactly one fixed one-token diagnostic request. It SHALL
not open an ARC game, take an ARC action, or change the production ARC model
identity validator.

The experiment SHALL preserve the exact launch `-m` argument, the independent
process command line, and the complete `/props` JSON object. It SHALL preserve
the raw `model_path`, `model`, and `model_alias` values as three separate
observations. It SHALL not replace a raw value with an absolute, resolved, or
normalized value.

The experiment SHALL resolve only non-empty absolute path candidates. Each
derived row SHALL retain its source field and raw value. It SHALL record the
resolved path, reachability, file type, alias state, SHA-256, and comparison
with the selected model hash. Missing, blank, non-string, relative, broken, and
unreadable values SHALL remain `unknown`. A reachable file with the wrong hash,
or two reachable candidates with different canonical paths, SHALL be
`contradicted`.

The report shape SHALL be one of `snapshot_alias`, `resolved_blob`,
`direct_file`, `conflicting`, or `unknown`. A symlink that resolves to the
selected Hugging Face blob is `snapshot_alias`. An exact content-addressed blob
path is `resolved_blob`. Another exact regular model file is `direct_file`.
Contradicted candidates take precedence over these positive shapes. No usable
absolute file candidate produces `unknown`.

The artifact SHALL record the selected hub ID, revision, snapshot filename,
quantization, exact file hash, GPU UUID, and server build. It SHALL include all
task-required raw, derived, process, lease, probe, cleanup, counter, principle,
and terminal-verdict fields. `arc_report_channel_forensics_ready_score` SHALL
equal one only when the capture is complete, owned, reproducible, and safely
cleaned up and its measured duration meets the repository's 60-second
live-inference evidence floor. This score SHALL describe evidence completeness.
It SHALL not claim that the production validator is correct or that ARC belief
has value.

If a precondition or live step fails, the experiment SHALL write a complete
terminal blocked artifact. Its `gate_check_summary` SHALL name the first failed
check with the exact expected and observed values. The runner SHALL not use a
CPU server or a smaller model as a fallback.

### SCENARIO-ARC-7039-RAW-AND-RESOLVED-STAY-DISTINCT

**Given** `/props.model_path` contains a snapshot symlink
**When** the report is reduced
**Then** the raw row retains the symlink text, the derived row contains the
resolved blob path, and the raw `/props` object remains unchanged.

### SCENARIO-ARC-7039-REPORT-SHAPE-MATRIX

**Given** missing, relative, snapshot-alias, canonical-blob, direct-file, and
conflicting report fixtures
**When** the report classifier runs
**Then** each fixture produces the required closed classification
**And** unknown or contradicted evidence never becomes an inferred pass.

### SCENARIO-ARC-7039-OWNED-LIVE-CAPTURE

**Given** every RTX 3090, VRAM, cache, CUDA, ownership, access, evidence, and
writable-path precondition passes
**When** the experiment starts one owned llama.cpp server
**Then** it records one full report and one completed one-token request
**And** cleanup reaps only the owned process and releases both leases.

### SCENARIO-ARC-7039-BLOCKED-CAPTURE

**Given** any required precondition or live operation fails
**When** the runner reaches its terminal path
**Then** it writes a schema-complete blocked artifact with readiness zero
**And** the exact failed check, expected value, and observed value remain in the
gate summary.

## REQ-ARC-7051: Official live model-report evidence is requalified after terminal hashing

One owned CUDA llama.cpp process SHALL load the cached
`unsloth/Qwen3.6-35B-A3B-GGUF` snapshot selected through
`cached_sota_pair()`. The process SHALL run on one idle supported RTX 3090.
The experiment SHALL acquire its own GPU and port leases before launch. It
SHALL block without stopping any unattributed llama.cpp server. It SHALL not
use CPU inference, a legacy model, or a smaller fallback.

The experiment SHALL send fixed diagnostic requests that each produce at
least one genuine token. The interval from owned-server admission through
owned-process shutdown SHALL be at least 75 seconds. It SHALL record monotonic
timestamps for admission, first token, last token, and shutdown. It SHALL not
open an ARC game, take an ARC action, or change the shared production identity
validator.

The artifact SHALL preserve the complete raw `/props` JSON object. It SHALL
preserve raw `model_path`, `model`, `model_alias`, launch `-m`, and process
command observations in separate fields. Only derived rows MAY resolve paths.
Every reachable model-file observation SHALL have a SHA-256 row that compares
with the selected snapshot hash. Missing evidence SHALL remain `unknown`.
Reachable conflicting evidence SHALL remain `contradicted`.

The artifact SHALL be complete before its reproducibility checksum is
computed. It SHALL use `carnot.terminal_artifacts.payload_sha256` over the
declared projection. The projection SHALL exclude only
`reproducibility_checksum` and `checksum_recomputation_rows`. The artifact
SHALL declare these exclusions in `checksum_contract`. A clean JSON reader
SHALL recompute the checksum before readiness becomes one. A later mutation to
any included field SHALL invalidate the artifact.

The artifact SHALL contain every field required by the active Exp7051 task,
including model, raw report, derived identity, file hash, process, lease, GPU,
phase-clock, cleanup, checksum, counter, principle, and terminal verdict
fields. `model_report_evidence_ready_score` SHALL equal bare integer one only
when preconditions, ownership, CUDA offload, diagnostic generation, identity
capture, 75-second owned duration, terminal checksum recomputation, and cleanup
all pass. This score describes evidence validity only. It SHALL not claim model
quality, belief value, or an ARC solve.

Any precondition or live failure SHALL produce a schema-complete terminal
`blocked` artifact. Its `gate_check_summary` SHALL identify the first failed
check with exact expected and observed values. Cleanup SHALL signal and reap
only the owned process. It SHALL release the owned GPU and port leases.

### SCENARIO-ARC-7051-PRIOR-FAILURES: Short duration and stale checksum fail closed

**Given** the Exp7039 duration of 46.968 seconds or an included-field mutation after hashing
**When** the Exp7051 validator recomputes terminal evidence
**Then** readiness is zero and the exact duration or checksum check fails
**And** neither failure can retain a positive verdict.

### SCENARIO-ARC-7051-TERMINAL-CHECKSUM: A clean reader confirms the terminal projection

**Given** a complete terminal payload with the two declared excluded fields
**When** the canonical helper hashes it and a new JSON reader recomputes it
**Then** both labeled SHA-256 values agree
**And** the recomputation receipt is preserved without creating a self-reference.

### SCENARIO-ARC-7051-RAW-EVIDENCE: Raw observations remain separate from derived paths

**Given** a server report with absolute, relative, missing, or conflicting identity candidates
**When** the report is reduced
**Then** every raw value remains unchanged and separately attributable
**And** only reachable absolute files receive resolved paths and hashes.

### SCENARIO-ARC-7051-OWNED-LIVE-INTERVAL: Tokens and shutdown prove owned duration

**Given** all hardware, cache, binary, lease, evidence, and path preconditions pass
**When** the experiment runs its fixed diagnostics and stops its server
**Then** first-token and last-token events fall inside one owned interval of at least 75 seconds
**And** CUDA GPU samples and process evidence identify that owned server.

### SCENARIO-ARC-7051-CLEANUP: Cleanup affects only owned resources

**Given** success or a failure after resource acquisition
**When** cleanup runs
**Then** only the recorded owned process can receive a signal
**And** process reaping, GPU lease release, and port release are terminal evidence.

### SCENARIO-ARC-7051-BLOCKED-PREFLIGHT: Missing or unattributed resources block before launch

**Given** a busy GPU, insufficient VRAM, missing CUDA build, missing snapshot,
unreadable prior evidence, unwritable output, or an unattributed server
**When** the experiment checks preconditions
**Then** it writes a complete blocked artifact with readiness zero
**And** it does not launch or stop a model process.

## Implementation Status (REQ-ARC-7051)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-ARC-7051 and SCENARIO-ARC-7051-* | Implemented (`python/carnot/experiment_7051_v618_model_report_requalification.py`; `scripts/experiments/experiment_7051_v618_model_report_requalification.py`) | 16 tests (`tests/python/test_experiment_7051_v618_model_report_requalification.py`) |

## REQ-ARC-7052: Typed model identity keeps raw reports and proof obligations separate

The shared ARC model identity receipt SHALL keep these facts in separate fields:
`requested_model_path`, `requested_model_filename`, `requested_hf_id`,
`requested_revision`, `launch_model_argument`, `observed_server_model_path`,
`observed_server_resolved_path`, `resolved_model_path`, and
`model_file_hash`. Raw server values SHALL remain unchanged. Only the derived
fields MAY contain canonical path resolutions.

The receipt SHALL emit one typed row for each required obligation: absolute
raw report, path resolution, selected snapshot relation, launch/report
agreement, content hash, hub, revision, requested filename, unique file
identity, and source provenance. Each row SHALL name its evidence source and
use exactly one status from `supported`, `contradicted`, or `unknown`. Every
required obligation SHALL be `supported` before the receipt can validate.

The shared builder SHALL accept three explicit path forms. A snapshot alias is
valid only when it resolves to the selected content-addressed blob. A canonical
blob is valid only when its hash and selected snapshot relation agree. A direct
regular GGUF file is valid only when its requested, launch, reported, and
resolved identities agree. All forms SHALL preserve the hub, revision, file
name, content hash, source provenance, and single-link identity checks.

The builder and validator SHALL fail closed on a relative alias, broken link,
wrong snapshot, same-size different content, changed hub, changed revision,
conflicting `/props` identity fields, hard-link ambiguity, symlink swap,
missing evidence, or changed source-artifact checksum. The submitted evaluator,
running-server reuse check, and belief-shadow consumer SHALL call the same
builder and validator. A named legacy reader SHALL accept only complete older
rows and SHALL not infer any current field or rewrite an old artifact.

Exp7052 SHALL first validate the Exp7051 artifact. It SHALL require the bare
integer `model_report_evidence_ready_score=1`. It SHALL independently recompute
the artifact's declared terminal checksum and its source file hash. It SHALL
also require importable production modules and writable code, test, temporary,
and artifact paths. An upstream or precondition failure SHALL produce a
terminal `blocked` artifact, never `partial`. Its `gate_check_summary` SHALL
name the failed check and retain the exact expected and observed values.

After preconditions pass, Exp7052 SHALL reproduce the frozen Exp7051 report
shape with temporary files. Test and worker execution SHALL not read mutable
Exp7051 report values. A fresh isolated Python process SHALL rebuild the
positive receipt and run one-factor attacks. Its positive obligation bytes
SHALL equal the parent process bytes. Every negative case SHALL fail closed.

The result SHALL contain every field required by the active Exp7052 task.
`typed_identity_attack_audit_ready_score` SHALL equal bare integer one only
when all positive paths pass, every attack fails closed, the fresh process
agrees, legacy handling is explicit, and all current consumers use the shared
code. The verifier is not an ARC correctness oracle.

### SCENARIO-ARC-7052-FROZEN-RAW-REPORT: Raw observations reproduce without artifact reads

**Given** deterministic temporary files with the exact Exp7051 `model_path`,
missing `model`, and relative `model_alias` shape
**When** the shared builder constructs the identity receipt
**Then** all three raw observations remain byte-for-byte unchanged
**And** their canonical resolutions remain separate derived facts.

### SCENARIO-ARC-7052-THREE-POSITIVE-PATHS: Explicit file forms retain all checks

**Given** a selected snapshot alias, its canonical blob, or a direct regular
GGUF file
**When** every required obligation has independent support
**Then** the shared validator accepts the receipt
**And** no path form weakens hub, revision, hash, file name, link, or source checks.

### SCENARIO-ARC-7052-ONE-FACTOR-ATTACKS: Each identity mutation fails closed

**Given** one valid frozen fixture
**When** one raw fact, selected fact, filesystem relation, content byte set, or
source checksum changes
**Then** at least one required obligation is contradicted or unknown
**And** the validator rejects the receipt.

### SCENARIO-ARC-7052-COLD-BYTE-AGREEMENT: A new interpreter reproduces the audit

**Given** no imported parent module state and a private temporary directory
**When** the worker rebuilds the positive fixture and every one-factor attack
**Then** its positive obligation bytes equal the parent bytes
**And** it reports a fail-closed result for every attack.

### SCENARIO-ARC-7052-LEGACY-AND-WIRING: Compatibility and consumers stay explicit

**Given** a complete version-one or version-two provenance row and the current
evaluator, reuse check, and belief-shadow paths
**When** compatibility and reachability are audited
**Then** the named legacy reader accepts the complete old row without inference
**And** each current consumer reaches the shared current-schema builder and validator.

### SCENARIO-ARC-7052-BLOCKED-PRECONDITION: Invalid upstream evidence stops the audit

**Given** a bad Exp7051 score, checksum, file hash, import, or writable path
**When** Exp7052 checks its preconditions
**Then** it writes one schema-complete terminal blocked artifact
**And** readiness remains zero with the exact failed check in the gate summary.

## Implementation Status (REQ-ARC-7052)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-ARC-7052 and SCENARIO-ARC-7052-* | Planned | RED tests pending |

## REQ-ARC-7072: Claim-grade live compaction A/B releases or retires the existing flag

Exp7072 SHALL test the existing `CARNOT_ARC_INDUCE_TOOL_COMPACT` behavior
without changing compaction. It SHALL execute only through
`make_carnot_agent` and `E3AgentPolicy`, with the production local GGUF runner.
The main comparison SHALL change only that flag. It SHALL leave the growth and
carried-state budget variables at their current defaults.

Before a live cell starts, the experiment SHALL require a valid Exp7052 with
the bare integer `typed_identity_attack_audit_ready_score=1`. It SHALL
independently hash Exp7052. It SHALL also require an unchanged solve registry,
an owned idle RTX 3090 lease, an owned port lease, a CUDA-linked runner, clean
stop authority, writable checkpoints, the pinned
`unsloth/Qwen3.8-27B-GGUF`, and the cached
`unsloth/gemma-4-26B-A4B-it-GGUF` returned through `cached_sota_pair()`. An
unattributed process or resource SHALL block the run. Cleanup SHALL never stop
an unattributed process. No CPU, remote-model, or legacy-small fallback is
permitted.

The frozen manifest SHALL contain at least 30 new paired Qwen cells and at
least eight paired Gemma cells across multiple hidden-game or rotation source
groups. Every pair SHALL match model, prompt, seed, action budget, token budget,
context budget, time budget, and normalized start state. Arm order SHALL be
counterbalanced. Every selected unit SHALL pass a registry precheck that
excludes reproduced public levels, source reading, exhaustive offline ground
truth, per-game adapters, and development-proxy solves.

Every completed cell SHALL record exact progress, levels reached, solves,
actions, tokens, peak context, p95 context, compactions, refetches, parser
failures, wall time, crashes, duplicate submissions, configuration bytes,
imported production symbols, model identity, lease identity, and
`solve_provenance=live_agent_self_discovery`. Checkpoint resume SHALL accept
only rows from the same frozen manifest and SHALL not execute a completed cell
again.

Value analysis SHALL require `tool_loop_reachable_score=1` and
`compaction_treatment_activated_score=1`. Treatment activation is one only when
compaction fires in at least 80 percent of eligible treatment cells and in zero
control cells. A non-firing treatment or an unreached tool loop SHALL produce a
`disqualified` harness verdict, not a scientific null.

A positive Qwen release SHALL require one-sided exact-quality noninferiority at
the declared margin, at least 10 percent lower p95 context, no worse parser
failure rate, and no more than 10 percent wall-time regression. Gemma SHALL not
show a directionally opposite exact-quality result. Every aggregate and paired
interval SHALL be recomputable from retained cell rows. Any other complete,
qualified result SHALL be `null`. If that result matches Exp6473, the existing
compaction scope SHALL be marked for mechanical retirement. A failed
precondition SHALL produce a schema-complete `blocked` artifact with the failed
check, expected value, and observed value.

### SCENARIO-ARC-7072-FLAG-ISOLATION: The main A/B changes one flag

**Given** two arms for one frozen paired cell
**When** their canonical configuration bytes are compared
**Then** only `CARNOT_ARC_INDUCE_TOOL_COMPACT` differs
**And** growth, state budget, prompt, seed, budgets, and start state match.

### SCENARIO-ARC-7072-ACTIVATION: Non-fire cannot become a scientific null

**Given** completed control and treatment rows
**When** fewer than 80 percent of eligible treatment cells compact, any control
cell compacts, or the tool loop is unreachable
**Then** value readiness remains zero
**And** the verdict class is `disqualified`.

### SCENARIO-ARC-7072-PARSER-FAILURE: Parser failures remain charged

**Given** a live response that does not parse as a valid tool call
**When** the cell reaches a terminal state
**Then** the parser failure is retained in the cell and aggregate rows
**And** it cannot be omitted from the safety gate or paired analysis.

### SCENARIO-ARC-7072-PAIRING: Order, seed, identity, and population stay frozen

**Given** the preregistered manifest
**When** the experiment executes or resumes
**Then** Qwen has at least 30 pairs and Gemma has at least eight pairs
**And** arm order is counterbalanced while every within-pair seed and model
identity matches.

### SCENARIO-ARC-7072-CHECKPOINT: Resume does not duplicate live work

**Given** a checkpoint with a completed row from the active manifest
**When** the experiment resumes
**Then** that cell is not submitted again
**And** rows from any other manifest are rejected.

### SCENARIO-ARC-7072-CLEANUP: Only owned resources are released

**Given** success, a cell failure, or a post-acquisition exception
**When** cleanup runs
**Then** only recorded owned processes, GPU leases, and port leases are released
**And** exit and release outcomes remain in cleanup rows.

### SCENARIO-ARC-7072-PROVENANCE: Only live self-discovery can receive credit

**Given** a selected hidden or rotation unit
**When** its registry and production-path evidence is checked
**Then** source reading, reproduced public levels, exhaustive truth, adapters,
and development proxies are absent
**And** any solve credit uses `live_agent_self_discovery`.

### SCENARIO-ARC-7072-AGGREGATION: Rows determine release or retirement

**Given** complete paired rows for both models
**When** activation, exact quality, context, parser, wall, and replication gates
are recomputed
**Then** every headline equals its row-derived value
**And** the terminal verdict is positive, null, blocked, disqualified, or
partial with a class-consistent prefix.

## Implementation Status (REQ-ARC-7072)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-ARC-7072 and SCENARIO-ARC-7072-* | Implemented | `test_experiment_7072_v619_live_arc_compaction_ab.py` |

## REQ-ARC-7099: Adapter-withheld E3 preflight proves one live action cell

Exp7099 SHALL test whether the scored E3 mechanism can generate and execute
an action while public-game knowledge is withheld. It SHALL make no solve
claim. It SHALL set `solve_provenance=development_proxy` and
`arc_registry_delta=0`.

Before game selection, the experiment SHALL require two idle RTX 3090 leases,
a CUDA-enabled llama.cpp runner, the exact cached GGUF files, offline ARC, a
readable solve registry, and writable raw and aggregate paths. `MODEL_SPECS`
SHALL contain `unsloth/Qwen3.8-27B-GGUF` as the live-generator arm and
`unsloth/Qwen3.6-35B-A3B-GGUF` as the headline control. Each arm SHALL execute.
No download, CPU fallback, remote model, legacy model, or model substitution is
permitted. A failed precondition SHALL write a terminal `blocked` artifact with
`inference_substrate_class=blocked_no_run`. Its gate summary SHALL name the
failed check, expected value, and observed value.

The experiment SHALL inspect the registry before model outcomes are visible.
It SHALL exclude each game and level with a credited receipt from this same
adapter-withheld live mechanism. It SHALL then freeze at least four public
games. The frozen set SHALL span at least two mechanic classes and at least two
registry depths.

Each action cell SHALL run in a fresh isolated worker. The policy SHALL not
import or read game adapters, game source, registry trajectories, known action
recipes, per-game checkpoints, or hand-built solvers. Adapter lookup SHALL be
unavailable. The worker SHALL receive only ordinary observations, the generic
action schema, the public runtime API, and the live E3 route. It SHALL construct
the policy through `make_carnot_agent`, and the constructed policy SHALL be an
`E3AgentPolicy`. The advice-only branch in `arc_loop_solve.py` SHALL not execute.

Each required model SHALL run at least one bounded action cell on at least two
frozen games. Every decision SHALL retain the observation hash, candidate
actions, simulation invocation, returned forecast, interpretation,
forecast-consuming selected action, exact environment transition, levels before
and after, latency, model identity, GPU identity, seed, and budget. Raw attempt
traces SHALL be written outside `results/` before aggregation. Their hashes
SHALL be checked during aggregation.

`adapter_withheld_live_path_ready_score` SHALL equal bare integer one only when
both model arms emit a valid action, cause an exact environment transition,
pass every forbidden-path receipt, and retain a complete forecast-to-action
ledger. A level advance is not required. Advice-only text, invalid actions,
unreachable solvers, substituted model identities, modified trace bytes, or an
incomplete ledger SHALL score zero.

The result SHALL contain every field required by the active Exp7099 task.
`offline_reproduced` SHALL be true only when each counted transition replays
exactly in a fresh environment. The verifier SHALL declare that it is not an
ARC correctness oracle. The verdict class SHALL be one of `positive`,
`circular_positive`, `null`, `blocked`, `disqualified`, or `partial`. The honest
verdict SHALL use an approved terminal prefix and agree with the verdict class.

### SCENARIO-ARC-7099-BLOCKED: A missing live prerequisite stops all cells

**Given** one missing lease, runner capability, exact model file, offline ARC
module, registry, or writable path
**When** Exp7099 runs its preconditions
**Then** it writes one schema-complete blocked artifact
**And** no model or environment action cell executes.

### SCENARIO-ARC-7099-FREEZE: Selection precedes model outcomes

**Given** registry rows and prior adapter-withheld receipts
**When** Exp7099 selects its public development games
**Then** it excludes every previously credited game and level
**And** freezes at least four games across mechanic classes and registry depths.

### SCENARIO-ARC-7099-ISOLATION: Policy-visible code cannot reach withheld knowledge

**Given** a fresh action worker
**When** the worker imports modules or reads files
**Then** adapters, source, registry trajectories, recipes, checkpoints, and
hand-built solvers remain unavailable
**And** adapter lookup and the advice-only branch fail closed.

### SCENARIO-ARC-7099-E3-ACTION: Both pinned models act through scored policy

**Given** both exact cached models and at least two frozen games
**When** bounded action cells execute
**Then** `make_carnot_agent` constructs `E3AgentPolicy` for every cell
**And** each model emits and executes at least one valid runtime action.

### SCENARIO-ARC-7099-LEDGER: Forecast evidence causes the selected action

**Given** candidate actions for one ordinary observation
**When** the live route invokes its simulation mechanism
**Then** the retained forecast is interpreted before action selection
**And** the selected valid action identifies the forecast it consumed.

### SCENARIO-ARC-7099-TRANSITION: An action changes the fresh environment exactly

**Given** one selected valid action and its pre-action observation
**When** the worker calls the public runtime API
**Then** it records the exact returned observation and level transition
**And** fresh-environment replay reproduces the same transition bytes.

### SCENARIO-ARC-7099-ADVERSARIAL: Invalid output and changed evidence cannot pass

**Given** advice text, an invalid action, a substituted model ID, a forbidden
read, a forbidden import, or modified raw trace bytes
**When** the artifact validator recomputes readiness
**Then** readiness is zero
**And** the failed row remains visible in the gate summary.

### SCENARIO-ARC-7099-NONCLAIM: Mechanism readiness is not a solve

**Given** complete valid transition rows with no level advance
**When** Exp7099 aggregates the action cells
**Then** the result can be positive for adapter-withheld path readiness
**And** solve provenance remains `development_proxy` with registry delta zero.

## Implementation Status (REQ-ARC-7099)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-ARC-7099 and SCENARIO-ARC-7099-* | Planned | RED tests pending |

## REQ-ARC-7127: One bounded adapter-withheld E3 cell SHALL execute both arms

Exp7127 SHALL write
`results/experiment_7127_v626_adapter_withheld_arc_loo.json`. It SHALL write a
schema-complete terminal blocked artifact before model setup. It SHALL then
check the registry rank, exact adapter treatment, cached model, two idle RTX
3090 leases, CUDA llama.cpp health, writable raw storage, and subprocess
support. It SHALL not depend on Exp7126 or any upstream readiness score.

The producer SHALL select registry eligibility rank one before it reads arm
outcomes. It SHALL hash the registry and selected public-game fixture at this
boundary. The selected game and target level SHALL already exist in the public
registry. The producer SHALL not claim a new game or level solve. It SHALL not
write `ops/arc_solve_registry.yaml`.

`MODEL_SPECS` SHALL declare `unsloth/Qwen3.6-35B-A3B-GGUF` as the headline
model with Q4_K_M quantization. The producer SHALL resolve the local path by
calling `cached_sota_pair()`. It SHALL use the GGUF's llama.cpp chat template.
It SHALL not download or substitute a model.

Setup SHALL run in a subprocess with a 300-second cap. The adapter-withheld arm
SHALL run in a fresh subprocess with a 1500-second cap. The adapter-visible
control SHALL run in a different fresh subprocess with the same cap. Validation
and publication SHALL have a reserved 300-second cap. A timeout SHALL terminate
the owned process group and record the exact stop reason.

Both arms SHALL construct the real scored policy with `make_carnot_agent` and
SHALL verify that its policy is an `E3AgentPolicy`. The withheld arm SHALL
remove only the selected game's adapter. The control SHALL retain it. Prompts,
budgets, seed, tools, model bytes, and executable environment SHALL otherwise
match. A synthetic entrypoint or an arm with zero executed attempts SHALL not
count as a completed arm.

The producer SHALL persist phase, process, request, token, proposal, verifier,
action, transition, reward, and level receipts after each event. Bulky raw
traces SHALL stay outside `results/`. Each manifest row SHALL bind its path,
byte count, and SHA-256. Level and action metrics SHALL use executed
transitions only.

A complete pair with zero levels in both arms SHALL be terminal `null`. Missing
runtime evidence SHALL be `blocked`. Adapter leakage, a fake entrypoint, stale
process reuse, or registry mutation SHALL be `disqualified`. An external or
stable block SHALL not use `partial`. Every level row SHALL set
`solve_provenance=development_proxy`, `solve_claim_made=false`, and
`offline_reproduced=false`.

The artifact SHALL contain `field_principles`, `preconditions_checked`,
`run_date`, `MODEL_SPECS`, `models_used`, `model_repository`, `model_path`,
`model_hash`, `model_quantization`, `inference_substrate`,
`inference_substrate_class`, `execution_venue`, `gpu_telemetry_rows`,
`token_rows`, `duration_s`, `source_artifact_hashes`, `raw_trace_manifest`,
`rows`, `per_game_results`, `phase_receipt_rows`, `process_rows`,
`request_rows`, `proposal_rows`, `verifier_rows`, `action_rows`,
`transition_rows`, `arm_rows`, `selected_game`,
`registry_rank_before_outcomes`, `adapter_withheld_exactly`,
`real_e3_entrypoint_used`, `fresh_process_per_arm`, `setup_cap_s`,
`withheld_arm_cap_s`, `control_arm_cap_s`, `finalization_cap_s`,
`withheld_levels`, `control_levels`, `level_delta`, `solve_provenance`,
`solve_claim_made`, `offline_reproduced`, `registry_mutated`,
`arc_loo_cell_complete_score`, `random_seed`, `reproducibility_checksum`,
`gate_check_summary`, `verifier_is_oracle`, `verdict_class`, and
`honest_verdict`. `field_principles` SHALL explain every listed field.

### SCENARIO-ARC-7127-PREFLIGHT: The terminal artifact exists before setup

**Given** a missing model, lease, CUDA runner, registry, raw store, or subprocess
**When** Exp7127 checks its local prerequisites
**Then** the schema-complete blocked artifact already exists
**And** no setup or arm process runs.

### SCENARIO-ARC-7127-PAIR: Only the selected adapter differs

**Given** registry rank one and two fresh arm processes
**When** the paired cell executes
**Then** only the selected game's adapter is absent in the withheld arm
**And** common prompts, budgets, seed, tools, model, and environment match.

### SCENARIO-ARC-7127-RUNTIME: Real E3 actions produce complete receipts

**Given** the real scored factory and an executable public environment
**When** each arm proposes and executes actions
**Then** each arm records at least one request, proposal, action, and transition
**And** all required phases and process identities have durable receipts.

### SCENARIO-ARC-7127-NULL: A complete zero pair is terminal evidence

**Given** both arms ran at least one executed attempt within their caps
**When** neither arm advances a level
**Then** `arc_loo_cell_complete_score` is one and `verdict_class` is `null`
**And** the result is not blocked or partial.

### SCENARIO-ARC-7127-ADVERSARIAL: Invalid evidence fails closed

**Given** adapter leakage, a no-op fake arm, process reuse, zero attempts,
missing phase receipts, a cap overrun, or registry mutation
**When** the artifact validator recomputes the paired cell
**Then** no positive or null measurement passes
**And** the verdict becomes blocked or disqualified by the stated rule.

### SCENARIO-ARC-7127-NONCLAIM: Public development evidence earns no solve credit

**Given** any completed level or transition row
**When** provenance is validated
**Then** every level row remains a `development_proxy` and not reproduced
**And** `solve_claim_made` and `registry_mutated` remain false.

## Implementation Status (REQ-ARC-7127)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-ARC-7127 and SCENARIO-ARC-7127-* | `python/carnot/experiment_7127_v626_adapter_withheld_arc_loo.py` and `scripts/experiments/experiment_7127_v626_adapter_withheld_arc_loo.py` | `tests/python/test_experiment_7127_v626_adapter_withheld_arc_loo.py` |
