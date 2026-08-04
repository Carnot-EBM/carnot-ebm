# Agentic Harness Capability Specification

**Capability:** agentic-harness
**Version:** 0.1.0
**Status:** Draft

## Overview

Defines machine-checkable harness contracts for live agent execution preflights.
These contracts are deliberately narrower than ARC solving: they establish
authority, scope, isolation, teardown, and denial behavior before a scored public
or hidden game can be entered.

## Requirements

### REQ-ARC-LRCL-5915: Live Runner Capability Lease Preflight

Experiment 5915 SHALL define and qualify the live-runner capability lease that
Exp5916 needs before any live `E3AgentPolicy` call. The lease SHALL be
machine-checkable and SHALL include an authority source, grantee task ID, exact
runner identity, exact environment identity, allowed command pattern, allowed
episode class, issue time, expiry time, nonce, signature or deterministic hash,
adapter-disabled requirement, resource bounds, and revocation state. The
experiment SHALL write
`results/experiment_5915_arc_live_runner_capability_lease.json`.

The environment/conductor binding SHALL validate the lease before any live call.
A local boolean set only by the experiment SHALL NOT self-authorize the runner.
Validation SHALL compare the lease to the bound task, runner, environment,
command pattern, episode class, adapter-disabled state, resource bounds, expiry,
nonce replay ledger, revocation state, and signature/hash. Missing, expired,
wrong-task, wrong-environment, widened-command, revoked, adapter-enabled, and
replayed-nonce leases SHALL be denied before execution.

Before the preflight, Exp5915 SHALL run a registry precheck and SHALL hash
Exp5901, Exp5902, the E3 entrypoint, the typed-memory provenance guard, runner
configuration, environment bindings, output path, disk/RAM resource receipts, and
protected files. It SHALL assert no public level target, no scored public
execution, no model loader, no source/BFS/adapter/prior-game/hidden-state access,
no registry update, and no edits to protected files.

Exp5915 SHALL run only a bounded non-scored synthetic dry run. The dry run SHALL
prove E3 import, state isolation, allowed-path execution, teardown, and absence
of persistent cross-cell state. It SHALL NOT enter a scored public game, load
GGUF weights, update the registry, inspect hidden state, run offline BFS, or use
a per-game adapter.

Experiment 5915 SHALL write bare top-level fields `status`,
`preconditions_checked`, `registry_precheck`,
`public_level_target_selected`, `upstream_memory_hash_receipts`,
`capability_lease_schema`,
`authority_source_and_environment_binding`,
`issue_expiry_nonce_and_revocation_receipts`,
`command_episode_and_resource_scope`, `adapter_disabled_e3_receipt`,
`bounded_non_scored_dry_run`,
`state_isolation_and_teardown_receipts`, `denial_path_matrix`,
`scored_public_execution_count`, `model_load_count`,
`source_bfs_adapter_prior_game_and_hidden_state_access_count`,
`registry_unchanged`, `protected_files_unchanged`,
`live_runner_capability_ready_score`, `duration_s`,
`inference_substrate`, `verifier_is_oracle`, `field_provenance`,
`test_commands`, `test_exit_codes`, `reproducibility_checksum`, and
`honest_verdict`.

Required field provenance principles SHALL include:

- `authority_source_and_environment_binding`: principle "the experiment cannot grant itself permission with an unbound local flag."
- `scored_public_execution_count`: principle "must be bare zero in this preflight."
- `source_bfs_adapter_prior_game_and_hidden_state_access_count`: principle "must be bare zero."
- `live_runner_capability_ready_score`: principle "emit bare 1.0 only for externally bound scoped permission, clean dry run/isolation/teardown, and complete denial-path enforcement."
- `inference_substrate`: principle "use live_runner_capability_preflight_no_llm_no_scored_game."
- `verifier_is_oracle`: principle "false; this task checks runner authority and isolation only."
- `honest_verdict`: principle "use complete_ready:, retired:, or blocked_precondition:."

### SCENARIO-ARC-LRCL-5915-BOUND-LEASE-DRY-RUN

**Given** a conductor-bound lease for Exp5916 with the exact E3 runner identity,
synthetic non-scored environment identity, allowed dry-run command, adapter-
disabled requirement, bounded resource scope, unexpired issue/expiry times,
fresh nonce, non-revoked state, and valid signature/hash
**When** the harness validates the lease and runs the bounded synthetic dry run
**Then** validation occurs before execution, E3 imports without loading a model,
the allowed path runs exactly the synthetic non-scored episode class, teardown
clears cell-local state, a second cell starts without persistent state, and all
public/scored/model/source/BFS/adapter/prior-game/hidden-state counts remain
bare zero.

### SCENARIO-ARC-LRCL-5915-DENIAL-MATRIX

**Given** lease variants that are missing, expired, wrong-task, wrong-
environment, command-widened, revoked, adapter-enabled, or nonce-replayed
**When** each variant is submitted to the live-runner binding
**Then** the runner denies the call before execution, records a deterministic
denial reason, does not mutate the dry-run environment, and does not consume any
model, scored public game, registry, source, BFS, adapter, prior-game, or hidden-
state resource.

### SCENARIO-ARC-LRCL-5915-STABLE-ARTIFACT

**Given** the registry and protected files before the preflight
**When** Exp5915 writes
`results/experiment_5915_arc_live_runner_capability_lease.json`
**Then** all required fields are present, the checksum is reproducible, the
registry hash is unchanged, protected file hashes are unchanged, the
`inference_substrate` is
`live_runner_capability_preflight_no_llm_no_scored_game`,
`verifier_is_oracle` is false, `live_runner_capability_ready_score` is bare
`1.0` only if the lease binding, dry run, isolation, teardown, and denial matrix
all pass, and `honest_verdict` begins with `complete_ready:`, `retired:`, or
`blocked_precondition:`.

### REQ-ARC-LRHL-5916: Held Live E3 Structured Memory A/B

Experiment 5916 SHALL run the exact Exp5915-qualified live-runner capability
gate before any model load, then execute only a preregistered, budget-matched,
adapter-disabled held live `E3AgentPolicy` A/B for no memory, raw agent-owned
tape, and structured evidence indexing. The experiment SHALL target held
generalization only. Because all 25 public games are cleared, no public level
target may be selected, no public solve may be headlined, and
`ops/arc_solve_registry.yaml` SHALL NOT be updated by this task.

Before live inference, Exp5916 SHALL run the registry precheck and replay/hash
Exp5915's capability lease. It SHALL define `MODEL_SPECS` with at least
`unsloth/Qwen3.6-35B-A3B-GGUF` and
`unsloth/gemma-4-26B-A4B-it-GGUF`; it MAY include
`unsloth/gemma-4-31B-it-GGUF` only when cache and budget permit, never as a
replacement for the required pair. The required pair SHALL be resolved through
`cached_sota_pair()`, concrete GGUF file hashes SHALL be recorded, tokenizer
preflight SHALL load the embedded GGUF tokenizer, and HuggingFace
`AutoTokenizer` SHALL NOT be used for GGUFs. The gate SHALL verify public
llama-cpp CUDA support, two healthy RTX 3090s, RAM/disk/VRAM headroom, real
offload/utilization receipts, output path writability, protected-workload
safety, lease scope/expiry/environment receipts, and teardown readiness. Any
failed gate SHALL produce a terminal `blocked_precondition:` artifact before
model load or live inference.

The preregistration SHALL freeze held measurement episode groups, arms, seeds,
counterbalanced order, action/token/wall-clock/query/byte budgets, proposal
prompts, decoding, primary held accuracy/progress/efficiency metrics, safety
metrics, and confidence thresholds before outcomes. Every cell SHALL instantiate
fresh adapter-disabled `E3AgentPolicy`, environment, proposer, and memory
state. Game adapters, game source access, offline BFS, prior-game logs,
registry trajectories, hidden state, and per-game constants SHALL be disabled
and counted as bare zero.

The raw-tape and structured-index arms SHALL consume identical agent-owned event
bytes, identical proposal prompts/decoding, and identical budgets; memory access
structure SHALL be the only treatment. Exp5916 SHALL measure environment
score/progress, held objective accuracy, actions, invalid/no-op/repeated
actions, tokens, latency, GPU receipts, query counts, bytes, evidence
utilization, and per-model/game/episode intervals. It SHALL run structured-index
shuffle and relevant-evidence deletion controls on a bounded confirmatory subset
to connect any live effect to Exp5901's causal mechanism rather than extra bytes
or calls.

Experiment 5916 SHALL write
`results/experiment_5916_arc_structured_memory_live_held_ab.json` with bare
top-level fields `status`, `preconditions_checked`,
`upstream_capability_gate_and_hashes`, `registry_precheck`,
`public_level_solve_claimed`,
`preregistered_held_episode_group_and_arm_design`, `model_specs`,
`model_file_hashes`,
`embedded_tokenizer_loader_cuda_gpu_utilization_and_vram_receipts`,
`capability_scope_expiry_and_environment_receipts`,
`submitted_e3_and_adapter_disabled_receipts`,
`identical_event_byte_and_budget_parity`,
`no_memory_raw_and_structured_live_metrics`,
`held_accuracy_progress_efficiency_and_safety_metrics`,
`per_model_game_episode_lower_bounds`,
`shuffled_and_deletion_confirmatory_controls`,
`evidence_utilization_receipts`,
`source_bfs_adapter_prior_game_and_hidden_state_access_count`,
`incidental_completion_receipts`, `registry_unchanged`,
`state_isolation_and_teardown_receipts`, `protected_files_unchanged`,
`structured_memory_live_ready_score`, `duration_s`,
`inference_substrate`, `verifier_is_oracle`, `field_provenance`,
`test_commands`, `test_exit_codes`, `reproducibility_checksum`, and
`honest_verdict`.

Required field provenance principles SHALL include:

- `public_level_solve_claimed`: principle "must be bare false because the experiment targets held generalization, not a cleared level."
- `identical_event_byte_and_budget_parity`: principle "memory access structure is the only treatment."
- `source_bfs_adapter_prior_game_and_hidden_state_access_count`: principle "must be bare zero."
- `structured_memory_live_ready_score`: principle "emit bare 1.0 only for positive preregistered structured-over-raw and structured-over-none lower bounds with no safety, budget, capability, or authority regression."
- `inference_substrate`: principle "use live_llm_inference."
- `verifier_is_oracle`: principle "false; policy memory consumes only visible agent-owned events."
- `honest_verdict`: principle "use complete_positive:, complete_null:, unsafe:, blocked_precondition:, or blocked:."

### SCENARIO-ARC-LRHL-5916-PRECONDITION-BLOCK

**Given** the registry, Exp5915 capability replay, GGUF model cache, embedded
tokenizer, llama-cpp CUDA, dual RTX 3090, resource, output-path,
protected-workload, submitted-E3, isolation, and teardown checks
**When** any required precondition fails before model load
**Then** Exp5916 writes a complete blocked artifact with all required fields,
zero live inference rows, no public solve claim, unchanged registry/protected
files, source/BFS/adapter/prior-game/hidden-state access count equal to bare
zero, and an honest verdict beginning with `blocked_precondition:`.

### SCENARIO-ARC-LRHL-5916-MATCHED-HELD-LIVE-AB

**Given** all preconditions pass and the submitted `E3AgentPolicy` is created
fresh with adapters disabled for every model/game/episode/arm cell
**When** no-memory, raw-tape, and structured-index arms run in the
preregistered counterbalanced order
**Then** prompts, decoding, action/token/wall-clock/query/byte budgets, and
agent-owned event bytes are matched between raw and structured arms; the
artifact reports held accuracy, progress, efficiency, safety, GPU, query-byte,
evidence-utilization, and per-model/game/episode lower-bound metrics without
updating the registry.

### SCENARIO-ARC-LRHL-5916-CAUSAL-CONTROLS

**Given** the structured-index treatment has live evidence utilization on the
bounded confirmatory subset
**When** relevant evidence is deleted and structured links are shuffled under
the same budgets
**Then** relevant deletion reduces structured utility, shuffling does not
preserve the live effect, safety and budget regressions are reported, and
`structured_memory_live_ready_score` remains `0.0` unless both preregistered
structured lower bounds are positive with clean safety, budget, capability, and
authority receipts.

### SCENARIO-ARC-LRHL-5916-NO-SOLVE-CREDIT

**Given** a held live run records incidental progress or a completion
**When** Exp5916 writes the artifact
**Then** `public_level_solve_claimed` is bare `false`, incidental completions
are telemetry only, no registry credit is requested, `registry_unchanged` is
true, and the task does not modify `ops/arc_solve_registry.yaml`.

### REQ-ARC-LREB-5928: Live Runner Execution Binding

Experiment 5928 SHALL independently qualify the execution prerequisite that
Exp5916 lacked: a parent/controller-issued, process-bound capability consumed
by the actual adapter-disabled live runner before any environment or model
action. This requirement establishes execution authority only. It SHALL NOT run
a model comparison, solve a game, select a scoring target, attempt a public
level, update `ops/arc_solve_registry.yaml`, or treat a self-issued token as
authority. The experiment SHALL write
`results/experiment_5928_arc_live_runner_execution_binding.json`.

The capability issuer SHALL be separate from the child runner. The parent
controller SHALL issue the capability with signing material unavailable to the
child. The child SHALL verify with public material only and SHALL NOT be able to
issue, broaden, refresh, or self-approve a capability. A self-issued or
child-issued capability SHALL be denied before execution.

The capability SHALL bind the issuer identity, child OS process identity,
executable hash, exact argv hash, environment allowlist hash, adapter-disabled
flag, exact scope, issue and monotonic expiry, nonce, run ID, and exact output
path. The actual child runner SHALL consume the capability before any synthetic
environment action. Missing, self-issued, expired, replayed, wrong-process,
wrong-command, wrong-environment, wrong-executable, adapter-enabled,
scope-broadened, or output-mismatched capabilities SHALL be refused before
environment/model action.

Before the dry run, Exp5928 SHALL run a registry precheck and hash the live
runner, capability code, Exp5915/Exp5916/Exp5902 receipts, environment schema,
output path, registry, and protected files. It SHALL assert no model load, no
scoring target, no public solve target, atomic output readiness, teardown
readiness, and immutable registry/protected files.

Exp5928 SHALL run only a bounded non-scoring adapter-disabled dry run through
the actual parent and child process boundary. The dry run SHALL capture parent
issue, child consume, OS process and executable receipts, exact command and
environment receipts, monotonic expiry, adapter-disabled binding, teardown,
output binding, denial receipts, replay/nonce invalidation, orphan-process
check, no credential/secret persistence, and before/after registry hash equality.

Experiment 5928 SHALL write bare top-level fields `status`,
`preconditions_checked`, `registry_precheck_receipt`,
`no_model_inference_or_level_attempt`,
`issuer_child_and_os_process_receipts`,
`capability_schema_scope_expiry_nonce_and_run_id`,
`executable_argv_environment_and_output_binding`,
`adapter_disabled_binding`,
`actual_live_entrypoint_consumption_receipt`,
`absent_self_issued_expired_replayed_wrong_process_command_environment_scope_and_output_denials`,
`non_scoring_dry_run_receipt`,
`teardown_nonce_invalidation_and_orphan_check`, `registry_unchanged`,
`protected_files_unchanged`, `live_runner_execution_binding_ready_score`,
`duration_s`, `inference_substrate`, `field_provenance`, `test_commands`,
`test_exit_codes`, `reproducibility_checksum`, and `honest_verdict`.

Required field provenance principles SHALL include:

- `actual_live_entrypoint_consumption_receipt`: principle "only a capability consumed by the actual child runner before environment action counts; fixture-only validation is insufficient."
- `registry_unchanged`: principle "must include exact before/after hash equality."
- `live_runner_execution_binding_ready_score`: principle "emit bare 1.0 only for external issuer separation, actual child consumption, all denial paths, clean teardown, and immutable registry."
- `inference_substrate`: principle "use actual_live_runner_capability_preflight_no_llm."
- `honest_verdict`: principle "use complete_ready:, retired:, or blocked_precondition:."

### SCENARIO-ARC-LREB-5928-PARENT-CHILD-CONSUME

**Given** a parent controller starts an adapter-disabled child runner with an
exact command, exact environment allowlist, exact output path, and teardown
ledger
**When** the parent issues a process-bound capability to that child
**Then** the actual `arc_competition_agent` live entrypoint consumes it before
the synthetic environment action, records the child PID/PPID/executable/argv/env
binding, performs no model load or level attempt, writes only the bound output,
and exits cleanly.

### SCENARIO-ARC-LREB-5928-DENIAL-MATRIX

**Given** absent, self-issued, expired, replayed, wrong-process,
wrong-command, wrong-environment, wrong-executable, adapter-enabled,
scope-broadened, and output-mismatched capability variants
**When** each variant is submitted to the live-runner binding
**Then** the runner denies the call before environment/model action, records a
deterministic denial reason, does not mutate synthetic state, and does not
consume model, scoring, registry, source, BFS, adapter, prior-game, or hidden
state resources.

### SCENARIO-ARC-LREB-5928-TEARDOWN-IMMUTABILITY

**Given** the dry run has completed and the nonce has been consumed
**When** Exp5928 performs teardown and immutable-file checks
**Then** the nonce cannot be replayed before teardown, the nonce ledger is
removed during teardown, no child process remains, no issuer secret is written
to the environment or output, the registry before/after hashes are exactly
equal, protected file hashes are unchanged, and
`live_runner_execution_binding_ready_score` is bare `1.0` only if every binding,
denial, teardown, and immutability gate passed.

### REQ-ARC-LRBH-5929: Capability-Bound Held Live Structured Memory A/B

Experiment 5929 SHALL rerun the preregistered no-memory, raw-tape, and
structured-index structured ARC memory comparison only through the actual
Exp5928-qualified, parent-issued, process-bound, adapter-disabled E3 execution
path. It SHALL write
`results/experiment_5929_arc_structured_memory_bound_live_ab.json`.

The experiment SHALL satisfy these subrequirements before any model load or
live held inference: REQ-ARC-LRBH-5929-CAPABILITY-REPLAY (registry precheck and
Exp5928 capability receipt replay/validation before model load),
REQ-ARC-LRBH-5929-HELD-CELL (selected cells are frozen held generalization
episodes and not public solve targets), REQ-ARC-LRBH-5929-ARM-ISOLATION (fresh
adapter-disabled E3 policy, environment, proposer, and memory state per cell),
REQ-ARC-LRBH-5929-BYTE-BUDGET-PARITY (raw-tape and structured-index arms share
identical event bytes, prompts, seeds, decoding, model allocation, context,
token, action, query, byte, latency, and stopping budgets),
REQ-ARC-LRBH-5929-ADAPTER-DISABLED (no per-game adapter, source, BFS,
prior-game log, hidden-state, or public solve target access),
REQ-ARC-LRBH-5929-LIVE-PROVENANCE (only live-agent rows can populate utility
metrics; Exp5901 offline causal evidence can be hashed but cannot fill missing
rows), REQ-ARC-LRBH-5929-TEARDOWN (capability expiry, nonce invalidation,
orphan-process, output, protected workload, checkpoint/resume, RAM/disk/VRAM,
nonzero offload, public llama.cpp CUDA, embedded GGUF tokenizer, and dual RTX
health receipts are recorded), and REQ-ARC-LRBH-5929-REGISTRY-IMMUTABILITY
(`ops/arc_solve_registry.yaml` remains byte-hash unchanged).

Experiment 5929 SHALL define `MODEL_SPECS` with
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. The required pair SHALL resolve through
`cached_sota_pair()` and the cached third family SHALL resolve through the
local GGUF cache. Exact model hashes SHALL be recorded. Embedded GGUF
tokenizers SHALL be loaded through llama.cpp/vocab-only or equivalent GGUF
metadata checks, and HuggingFace `AutoTokenizer` SHALL NOT be used on GGUFs.
Any failed gate SHALL produce a complete `blocked_precondition:` artifact
before model load or held inference.

Before inference, Exp5929 SHALL freeze held episode cells, event-byte streams,
memory transforms, prompts, seeds, model allocation, context/token/action/query/
byte/wall-clock budgets, stopping rules, primary utility metric, intervals, and
the no-target/no-registry-credit rule. The only treatment difference between
raw-tape and structured-index arms SHALL be the memory representation.

Experiment 5929 SHALL report retrieval relevance, action legality, verified
progress events, efficiency, abstention, context/token/latency/GPU/memory use,
arm crossover, capability expiry/teardown, and per-model/episode rows. If an
incidental level outcome appears, `solve_provenance` SHALL be
`live_agent_self_discovery`, the registry SHALL remain unchanged, and the
already-cleared public level SHALL NOT be headlined, resubmitted, or credited.
Any proxy, adapted, offline, or off-path row SHALL void the live claim and keep
`structured_memory_live_ready_score` at `0.0`.

Experiment 5929 SHALL write bare top-level fields `status`,
`gate_and_capability_replay_receipt`, `preconditions_checked`,
`registry_precheck_and_selected_held_cells`, `model_specs`,
`model_file_hashes`,
`embedded_tokenizer_loader_cuda_gpu_and_vram_receipts`,
`actual_bound_e3_entrypoint_receipt`, `adapter_disabled`,
`no_per_game_adapter_or_public_solve_target`, `solve_provenance`,
`identical_event_bytes_and_arm_budget_parity`,
`sealed_prompts_seeds_models_arms_and_stopping_rules`,
`per_model_episode_retrieval_progress_legality_efficiency_and_abstention`,
`primary_live_utility_comparison_and_intervals`,
`token_context_latency_gpu_and_memory_accounting`,
`capability_expiry_teardown_and_orphan_receipts`, `registry_unchanged`,
`protected_files_unchanged`, `structured_memory_live_ready_score`,
`duration_s`, `inference_substrate`, `verifier_is_oracle`,
`field_provenance`, `test_commands`, `test_exit_codes`,
`reproducibility_checksum`, and `honest_verdict`.

Required field provenance principles SHALL include:

- `solve_provenance`: principle "use live_agent_self_discovery for any observed level outcome; no development proxy or outer-loop result can satisfy this task."
- `identical_event_bytes_and_arm_budget_parity`: principle "arms may differ only in memory representation."
- `structured_memory_live_ready_score`: principle "emit bare 1.0 only for complete bound live rows, adapter-disabled execution, interval-separated held utility over both controls, clean teardown, and immutable registry."
- `inference_substrate`: principle "use actual_capability_bound_adapter_disabled_e3_local_mandated_gguf_public_llama_cpp_cuda."
- `verifier_is_oracle`: principle "true only for environment legality, progress, exact event replay, capability checks, and registry hashes."
- `honest_verdict`: principle "use complete_positive:, complete_null:, retired:, or blocked_precondition:."

### SCENARIO-ARC-LRBH-5929-PRECONDITION-BLOCK

**Given** the registry, Exp5928 capability replay, held-cell isolation, GGUF
cache, embedded tokenizer, public llama.cpp CUDA, dual RTX health, resources,
output path, protected workload, adapter-disabled E3 entrypoint, checkpoint/
resume, and teardown checks
**When** any required precondition fails before model load
**Then** Exp5929 writes a complete blocked artifact with all required fields,
zero live-agent rows, immutable registry/protected files, no public solve target,
no proxy provenance, `structured_memory_live_ready_score` equal to bare `0.0`,
and an honest verdict beginning with `blocked_precondition:`.

### SCENARIO-ARC-LRBH-5929-BOUND-MATCHED-HELD-LIVE-AB

**Given** every precondition passes and the actual Exp5928-bound child runner
consumes its capability before creating fresh adapter-disabled E3 cells
**When** no-memory, raw-tape, and structured-index arms run in the frozen order
on held E3 episodes
**Then** raw-tape and structured-index cells have identical underlying event
bytes and matched model/context/action/token/query/byte/latency budgets; only
memory representation differs; per-model/episode rows report retrieval,
progress, action legality, efficiency, abstention, token/context/latency/GPU
and memory accounting; and no registry update or public solve claim occurs.

### SCENARIO-ARC-LRBH-5929-NO-SOLVE-CREDIT

**Given** a bound held live row incidentally reaches a level outcome
**When** Exp5929 writes the artifact
**Then** `solve_provenance` is `live_agent_self_discovery`, the level outcome is
telemetry only, `ops/arc_solve_registry.yaml` is hash-unchanged, no public
target is selected, and no headline/resubmission/registry-credit claim is made.

### REQ-ARC-CPTB-5970: Strip-Swap HUD Convention Sentinel

Experiment 5970 SHALL provide deterministic row-strip and column-strip swap
perturbations for the convention-perturbation transfer battery. Each transform
SHALL exchange an edge-adjacent strip of width `t` with the immediately interior
strip of width `t`, for top, bottom, left, or right placement, with
`t >= EDGE_BAR_EDGE_TOLERANCE`. The transform SHALL reject non-integer widths,
degenerate widths, invalid dimensions, invalid axis/edge combinations,
overlapping slices, and any permutation that loses or duplicates cells. Applying
the same transform twice SHALL be the inverse, and every transform SHALL preserve
the grid multiset exactly.

Exp5970 SHALL declare deterministic condition metadata for the strip direction,
width, edge placement, condition id, and declared target predicate without
changing existing CPTB roll or salience semantics. The metadata SHALL record the
HUD edge-adjacency predicate targeted by each condition, the frontier predicate
dose, moved HUD mask pixels, detector mask changes, grid-difference
localization, and byte-identity of playfield pixels outside the swapped bands.

Exp5970 SHALL run static sentinels for representative top, bottom, left, right,
no-HUD, and frontier-only masks. The static matrix SHALL prove that the intended
HUD predicate changes for the targeted edge, non-target sentinels remain
preserved outside the swapped bands, and collateral dose is quantified rather
than assumed inert.

Exp5970 SHALL run a bounded adapter-free, no-LLM live-path sentinel through
`make_carnot_agent` and `E3AgentPolicy` on the known HUD-moving anchors and
matched control games. The agent SHALL receive transformed observations through
the normal choose-action path. Game source, `GameAdapter` routes, offline BFS,
prior-game trajectories, hidden state, and solve registry updates SHALL remain
disabled or unused. Readiness requires at least one strip condition to violate
the HUD convention while retaining non-empty valid live action/progress support.
If that support is absent, the artifact SHALL emit an honest null or retirement
and SHALL NOT authorize Exp5971.

Exp5970 SHALL write
`results/experiment_5970_arc_strip_swap_sentinel.json` and SHALL NOT modify
`ops/arc_solve_registry.yaml`, shipped policy flags, or
`scripts/research_conductor.py`. All incidental level outcomes SHALL be
telemetry only and SHALL NOT be reported as a public solve, solve credit, or
registry update.

Experiment 5970 SHALL write bare top-level fields `status`,
`preconditions_checked`, `registry_precheck_and_hash`,
`transform_schema_parameters_and_hash`,
`row_column_inverse_and_multiset_receipts`,
`static_target_and_non_target_dose_matrix`,
`detector_mask_and_predicate_change_matrix`,
`collateral_playfield_change_bounds`,
`live_agent_path_and_disabled_escape_hatches`,
`sentinel_game_arm_seed_and_budget_manifest`,
`anchor_support_and_behavioral_validity`,
`shipped_flag_and_registry_immutability`, `no_solve_credit_receipt`,
`protected_files_unchanged`, `strip_swap_sentinel_ready_score`, `duration_s`,
`inference_substrate`, `verifier_is_oracle`, `missing_verifier_gaps`,
`field_provenance`, `test_commands`, `test_exit_codes`,
`reproducibility_checksum`, and `honest_verdict`.

Required field provenance principles SHALL include:

- `preconditions_checked`: principle "the transform and live path must be authentic and bounded before measurement."
- `registry_precheck_and_hash`: principle "all public levels are already cleared; this task does not target or register a solve."
- `transform_schema_parameters_and_hash`: principle "strip direction, width, placement, and condition IDs are deterministic and versioned."
- `row_column_inverse_and_multiset_receipts`: principle "transforms are lossless permutations with exact round trips."
- `static_target_and_non_target_dose_matrix`: principle "the intended HUD convention is violated and unrelated levers are quantified, not assumed inert."
- `detector_mask_and_predicate_change_matrix`: principle "report exact pixel/mask/predicate changes for every sentinel."
- `collateral_playfield_change_bounds`: principle "content outside the swapped bands remains byte-identical."
- `live_agent_path_and_disabled_escape_hatches`: principle "use make_carnot_agent/E3AgentPolicy with source, BFS, adapters, priors, and hidden state disabled."
- `sentinel_game_arm_seed_and_budget_manifest`: principle "games, arms, seeds, conditions, and budgets are sealed before outcomes."
- `anchor_support_and_behavioral_validity`: principle "readiness requires convention violation with non-empty valid live support."
- `shipped_flag_and_registry_immutability`: principle "both remain byte-identical."
- `no_solve_credit_receipt`: principle "any incidental level outcome is not a new result or registry mutation."
- `protected_files_unchanged`: principle "emit readiness only for immutable protected state."
- `strip_swap_sentinel_ready_score`: principle "emit bare 1.0 only for authentic targeted dose, viable support, and immutable protected state."
- `duration_s`: principle "record measured adapter-free ARC runtime."
- `inference_substrate`: principle "use offline_arcade_live_agent_runtime_self_discovery_no_llm."
- `verifier_is_oracle`: principle "false for the HUD convention hypothesis."
- `missing_verifier_gaps`: principle "list limited anchor support and public-game generalization gaps."
- `honest_verdict`: principle "use complete_ready:, complete_null:, retired:, or blocked:."

### SCENARIO-ARC-CPTB-5970-LOSSLESS-STRIP-SWAPS

**Given** valid top, bottom, left, and right strip-swap parameters with
`t >= EDGE_BAR_EDGE_TOLERANCE`
**When** each transform is applied to a grid and then applied again as its
inverse
**Then** the original grid is restored exactly, the cell multiset is unchanged,
outside-band playfield pixels are byte-identical, and invalid parameters are
rejected before a transform is produced.

### SCENARIO-ARC-CPTB-5970-STATIC-DOSE-MATRIX

**Given** representative top, bottom, left, right, no-HUD, and frontier-only
sentinels
**When** Exp5970 applies every strip-swap condition
**Then** the matching HUD edge predicate changes, moved HUD mask pixels are
reported, non-target predicates and outside-band playfield pixels are preserved
or quantified, detector-mask deltas are reported, and the static matrix carries
bounded collateral dose for every sentinel.

### SCENARIO-ARC-CPTB-5970-BOUNDED-LIVE-PATH

**Given** the known HUD-moving anchors, matched controls, sealed seed and action
budget, unchanged registry and shipped flags, and no source/BFS/adapter/prior-
game/hidden-state route
**When** transformed observations are passed through `make_carnot_agent` and
`E3AgentPolicy`
**Then** at least one strip condition violates the HUD convention while
retaining non-empty valid action/progress support, or the artifact emits an
honest null/retirement; the registry and shipped flags remain unchanged and no
solve credit is claimed.

### REQ-ARC-CPTB-5971: Strip-Swap Battery Gate Replay and Game-Unit Inference

Experiment 5971 SHALL replay the exact Exp5970 strip-swap sentinel gate before
running the full battery. The replay SHALL verify the exact Exp5970 artifact
path, artifact hash, `strip_swap_sentinel_ready_score == 1.0`, transform schema
hash, condition definitions, 25-public-game manifest, four preregistered CPTB
arms, five preregistered seeds, action and wall-time budgets, SDK/cache/resource
availability, registry hash, shipped policy flags, and disabled source, BFS,
adapter, registry-trajectory, hidden-prior, and per-game-calibration routes.
If any precondition fails, Exp5971 SHALL emit a terminal
`blocked_precondition:` artifact before full execution.

Exp5971 SHALL freeze the game x arm x seed x condition matrix before outcomes.
The matrix SHALL cover all 25 public games, the four arms `CTRL`, `FRONT`,
`HUDO`, and `SHIP`, five seeds, and original plus strip-swap observation
conditions. Every executed cell SHALL be driven through
`make_carnot_agent` and `E3AgentPolicy` with fresh policy, environment, and
cell-local state. GameAdapter, game source reads, offline BFS, per-game
calibration/model lookup, registry trajectory replay, hidden priors, and
denominator repair SHALL be forbidden. Missing, errored, generator-invalid, and
completed cells SHALL remain explicit terminal states.

Exp5971 SHALL record per-cell actions, observations, progress, levels, elapsed
time, errors, generator validity, transform dose, HUD/frontier masks, policy
decisions, and cell health. The artifact SHALL also expose per-game,
per-seed, per-arm outcomes so no aggregate can hide game or seed reversals.

Exp5971 SHALL analyze the shipped, HUD-removed, frontier-removed, and combined
control arms under original and strip conditions as preregistered. Inference
SHALL use the game as the replication unit. The artifact SHALL report per-game
paired deltas, exact one-sided sign tests, game jackknife intervals, seed
stability, and support p-floors. Seeds SHALL NOT be counted as new game
support.

Convention-dependence SHALL be interpretable only when the transform changes
the HUD predicate, the original anchor is won by at least one matched arm, the
transformed anchor retains valid support, and enough games discriminate the
arms. If any pass region is empty, destroyed, one/two-game-only, or otherwise
underpowered, Exp5971 SHALL refuse a forced positive or flag recommendation and
SHALL emit `complete_null:` or `complete_underpowered:` with the exact empty or
underpowered region. Exp5971 SHALL NOT recommend a shipped flag change, claim
hidden-game transfer, credit a public solve, or mutate
`ops/arc_solve_registry.yaml`.

Experiment 5971 SHALL write
`results/experiment_5971_arc_strip_swap_battery.json` with bare top-level
fields `status`, `preconditions_checked`, `gate_replay_receipt`,
`registry_precheck_and_hash`,
`transform_condition_arm_game_seed_and_budget_seal`,
`expected_completed_missing_errored_and_generator_invalid_cells`,
`live_agent_path_and_disabled_escape_hatches`,
`per_cell_actions_progress_levels_time_and_health`,
`per_game_per_seed_per_arm_outcomes`,
`static_and_behavioral_transform_dose`,
`anchor_survival_and_discriminating_game_support`,
`game_unit_sign_jackknife_intervals_and_p_floors`,
`convention_dependence_decision`,
`overall_hud_value_not_identified_receipt`,
`shipped_flag_and_registry_immutability`, `no_solve_credit_receipt`,
`protected_files_unchanged`, `duration_s`, `inference_substrate`,
`verifier_is_oracle`, `missing_verifier_gaps`, `field_provenance`,
`test_commands`, `test_exit_codes`, `reproducibility_checksum`, and
`honest_verdict`.

Required field provenance principles SHALL include:

- `preconditions_checked`: principle "full execution starts only after the authentic sentinel gate and complete matrix/resources are verified."
- `gate_replay_receipt`: principle "Exp5970 exact path/hash/value must satisfy `strip_swap_sentinel_ready_score == 1.0`."
- `registry_precheck_and_hash`: principle "this is generalization measurement over already-cleared games, not a solve task."
- `transform_condition_arm_game_seed_and_budget_seal`: principle "the entire factorial design is immutable before outcomes."
- `expected_completed_missing_errored_and_generator_invalid_cells`: principle "every planned cell has one honest terminal state."
- `live_agent_path_and_disabled_escape_hatches`: principle "only the reachable adapter-free live mechanism receives credit."
- `per_cell_actions_progress_levels_time_and_health`: principle "accuracy, efficiency, and execution validity remain jointly visible."
- `per_game_per_seed_per_arm_outcomes`: principle "no aggregate can hide game/seed direction reversals."
- `static_and_behavioral_transform_dose`: principle "intended convention dose and actual policy effect are both measured."
- `anchor_survival_and_discriminating_game_support`: principle "no verdict is allowed from a destroyed or empty pass region."
- `game_unit_sign_jackknife_intervals_and_p_floors`: principle "games are the replication unit and exact attainable significance is explicit."
- `convention_dependence_decision`: principle "state supported dependence, invariance, underpower, or uninterpretable null without forced inference."
- `overall_hud_value_not_identified_receipt`: principle "this battery cannot establish overall lever value from inadequate game support."
- `shipped_flag_and_registry_immutability`: principle "both remain byte-identical."
- `no_solve_credit_receipt`: principle "incidental levels are measurements only and never registry credit."
- `protected_files_unchanged`: principle "active roadmap, conductor, exclusions, history, and unrelated changes remain immutable."
- `duration_s`: principle "use measured `offline_arcade_live_agent_runtime_self_discovery_no_llm`."
- `inference_substrate`: principle "use measured `offline_arcade_live_agent_runtime_self_discovery_no_llm`."
- `verifier_is_oracle`: principle "false; public-game convention evidence does not prove hidden transfer."
- `missing_verifier_gaps`: principle "public-game convention evidence does not prove hidden transfer."
- `field_provenance`: principle "artifact fields carry principle annotations tied to the preregistered safeguards."
- `test_commands`: principle "record focused, coverage, full-suite, spec, E2E, adversarial, protected-file, and clutter checks."
- `test_exit_codes`: principle "record the actual exit code for each verification command."
- `reproducibility_checksum`: principle "hash measured rows and immutable precondition receipts, excluding wall-clock duration."
- `honest_verdict`: principle "use `complete_positive:`, `complete_null:`, `complete_underpowered:`, or `blocked:`."

### SCENARIO-ARC-CPTB-5971-GATE-REPLAY-MATRIX-SEAL

**Given** Exp5970's ready artifact, the current registry, shipped flags, CPTB
arm definitions, all 25 public games, and the five seeds
**When** Exp5971 prepares execution
**Then** it verifies the exact Exp5970 path/hash/value and freezes the full
original-plus-strip matrix before any outcome row is read or written.

### SCENARIO-ARC-CPTB-5971-LIVE-PATH-CELL-HEALTH

**Given** a sealed cell for any game, arm, seed, and condition
**When** the cell is executed
**Then** the action loop goes through `make_carnot_agent` and `E3AgentPolicy`,
records actions/progress/levels/time/HUD/frontier health, disables source,
BFS, adapters, per-game priors, and hidden routes, and records one terminal
state without fabricating replacement rows.

### SCENARIO-ARC-CPTB-5971-GAME-UNIT-FORCED-VERDICT-REFUSAL

**Given** complete or partially complete original and strip rows
**When** Exp5971 analyzes convention transfer
**Then** the game is the replication unit, exact sign-test p-floors and
jackknife support are reported, anchor-destroyed or one/two-game-only regions
force `complete_null:` or `complete_underpowered:`, and no flag flip, hidden
transfer, public solve credit, or registry mutation is claimed.
