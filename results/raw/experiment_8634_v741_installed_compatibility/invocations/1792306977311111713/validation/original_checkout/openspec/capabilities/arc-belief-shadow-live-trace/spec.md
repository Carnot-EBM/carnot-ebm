# ARC Repaired Belief Shadow Live Trace

## REQ-ARC-7032: The repaired shadow cell preserves actions and complete provenance

Experiment 7032 SHALL run one bounded official-live ARC trace through
`make_carnot_agent` and `E3AgentPolicy`. It SHALL compare an explicit flag-off
control with a belief-shadow decision from the same pre-action observation.
The shadow selector MAY query belief evidence. It SHALL return the control
candidate order and the emitted action SHALL remain byte-for-byte identical.

Before model start, the experiment SHALL validate the unchanged Exp7030 and
Exp7031 artifacts with their production validators. It SHALL recompute every
cited source hash and upstream artifact hash. It SHALL also require readable
Exp7017 and Exp7024 gates, a registry-prechecked official episode, one idle
supported RTX 3090, sufficient free VRAM, owned GPU and port leases, the CUDA
`llama-server` binary and library, writable output paths, and owned stop
authority. Any failed gate SHALL produce a complete blocked artifact. The run
SHALL NOT use a substitute model, CPU fallback, an unowned server, or inferred
evidence.

`MODEL_SPECS` SHALL resolve through `cached_sota_pair()` and execute
`unsloth/Qwen3.6-35B-A3B-GGUF`. The requested snapshot `.gguf` path, filename,
hub ID, revision, file hash, quantization, embedded-tokenizer receipt, server
command, PID, port, context, GPU UUID, and observed canonical model path SHALL
be recorded separately. The shared Exp7030 identity bridge SHALL accept the
server's extensionless content-addressed blob only when it is the exact target
of the requested snapshot file. The experiment SHALL NOT call
`AutoTokenizer`.

The action budget and episode SHALL be frozen before either decision. Each
decision SHALL record its prompt hash, belief evidence IDs, query result, base
and counterfactual rankings, selected action, observation receipt, progress,
model request and token counts, and provenance record. The run SHALL record
task-linked setup, model-load, inference, output-write, and cleanup phases,
GPU samples, process and lease identities, context, registry hashes, and owned
resource cleanup.

`belief_shadow_trace_ready_score` SHALL be the bare integer one only when all
upstream, identity, CUDA, official-live, parity, nonempty-query, provenance,
task-compute, registry, and cleanup gates pass. A repeated snapshot-to-blob
identity failure SHALL reuse the exact Exp7025 verdict
`blocked_belief_shadow_live_trace:live_trace_execution`. The artifact SHALL
make no game-level solve claim. An incidental completion MAY be recorded only
as `live_agent_self_discovery` and SHALL stay outside the headline.

### SCENARIO-ARC-7032-PINNED-UPSTREAMS

**Given** clean Exp7030 and Exp7031 artifacts with cited source hashes
**When** preflight recomputes their validators, artifact hashes, and source hashes
**Then** model start is allowed only when every recorded value is unchanged.

### SCENARIO-ARC-7032-EXTENSIONLESS-BLOB-IDENTITY

**Given** a requested snapshot `.gguf` symlink to an extensionless cache blob
**When** the owned server reports that canonical blob path
**Then** the shared identity bridge preserves both paths and accepts their exact join.

### SCENARIO-ARC-7032-SHADOW-PARITY-AND-QUERY

**Given** matched flag-off and belief-shadow policies at one pre-action state
**When** both policies rank the same candidate set
**Then** belief telemetry contains a nonempty query and the shadow action equals control.

### SCENARIO-ARC-7032-COMPUTE-AND-CLEANUP-RECEIPTS

**Given** one owned CUDA server and one official-live action
**When** the terminal artifact is reduced
**Then** every model, context, phase, GPU, process, lease, provenance, and cleanup receipt validates.

### SCENARIO-ARC-7032-FAILS-CLOSED

**Given** a missing or contradictory required field or receipt
**When** the artifact validator runs
**Then** readiness cannot equal one and the exact damaged evidence is named.

## Implementation Status (REQ-ARC-7032)

| Requirement | Implementation | Verification |
|---|---|---|
| REQ-ARC-7032 and all SCENARIO-ARC-7032 variants | `python/carnot/experiment_7032_repaired_belief_shadow_live_trace.py`; `scripts/experiments/experiment_7032_repaired_belief_shadow_live_trace.py` | `tests/python/test_experiment_7032_repaired_belief_shadow_live_trace.py` |
