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
