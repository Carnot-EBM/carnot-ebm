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
