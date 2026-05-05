# Carnot Conductor Runtime Charter

**Experiment:** 1280 NLAH conductor charter
**Status:** Complete
**Scope:** Shared runtime semantics for Carnot conductor tasks

## Purpose

Carnot roadmap entries are executable natural-language research contracts. This
charter makes the shared harness explicit so that conductor tasks can be planned,
implemented, verified, restarted, and audited without relying on hidden
controller assumptions.

Experiment-specific logic remains in the roadmap task prompt. This charter
defines the invariant contract around terminal artifacts, gates, state, roles,
failure handling, and acceptance-object alignment.

## Contracts

Each task must declare the following before execution:

- Required terminal deliverable path, usually `results/*.json`.
- Required terminal fields and schema name.
- Applicable OpenSpec requirements and scenarios.
- Gate prerequisites, if any.
- Required verification commands or documented non-applicability.
- Expected docs and ops reconciliation targets.

Valid terminal statuses are:

- `complete`: the deliverable exists, schema fields are present, verification is
  recorded, and the acceptance object is satisfied.
- `blocked`: the task could not execute because a gate, tool, model, permission,
  or prerequisite is missing, and the artifact records the exact blocker.
- `failed`: the task executed and produced a terminal negative result.
- `retired`: the task is intentionally no longer pursued and records the reason.

Invalid terminal states are:

- `in_progress`
- bootstrap-only skeletons
- missing deliverable
- stale skeleton copied from an older run
- malformed JSON artifact
- artifact without `honest_verdict`
- artifact whose local verifier success conflicts with the final acceptance
  object

## Roles

### Conductor/Runtime Parent

The conductor owns task dispatch, gate evaluation, state packet creation,
timeout policy, and terminal audit. It must not infer completion from commit
presence, log rows, or roadmap entries alone.

### Planner

The planner converts milestone goals into task contracts, upstream gates, prior
failure carry-forwards, and acceptance objects. It must avoid adding paper claims
unless measured artifact fields already support them.

### Implementer

The implementer changes code, specs, docs, or artifacts required by the task. It
must preserve unrelated user or conductor changes and must write a terminal
blocked artifact when prerequisites prevent execution.

### Verifier/Auditor

The verifier runs targeted tests, lint, spec coverage, artifact validation, and
gate checks. It must report failed global checks honestly and distinguish
pre-existing repository failures from task-local failures.

### Retro/Backfill Auditor

The retro auditor reads authoritative result artifacts and conductor logs to
evaluate milestone criteria. It must count missing or stale artifacts as unmet
criteria.

### Paper-Claim Auditor

The paper-claim auditor maps public claims to measured artifact fields. It must
flag unsupported claims and may not upgrade a claim from speculative to measured
without terminal evidence.

## Stage Templates

### Standard Implementation

```text
PLAN -> IMPLEMENT -> VERIFY -> WRITE_ARTIFACT -> AUDIT_TERMINAL
```

Acceptance object: the required result artifact plus targeted verification and
spec/ops reconciliation.

### Retrospective Or Backfill

```text
LOAD_PRIOR_ARTIFACTS -> ANALYZE -> WRITE_BACKFILL -> VERIFY_TERMINAL
```

Acceptance object: an honest retrospective or backfill artifact derived from
authoritative source artifacts and logs.

### Gated Task

```text
GATE_CHECK -> RUN_OR_BLOCK -> WRITE_TERMINAL_ARTIFACT
```

Acceptance object: either a complete task artifact when the gate is open, or a
blocked artifact that records the failed upstream path, field, operator,
expected value, and actual value.

## Deterministic Adapter Hooks

Harnesses should name exact hooks when they apply:

- Targeted tests: `.venv/bin/pytest <paths> -q --no-cov`
- Lint: `ruff check <paths>`
- Format check: `ruff format --check <paths>`
- Type check: `.venv/bin/mypy <paths>`
- Spec coverage: `.venv/bin/python scripts/check_spec_coverage.py <test_paths>`
- JSON validation: `jq . <artifact>`
- Reconciliation: `bash scripts/validate-reconciliation.sh`
- Diff hygiene: `git diff --check`
- Gate audit: `python3 scripts/audit_roadmap_gates.py research-roadmap-next.yaml`
- Prior failure validation:
  `python3 scripts/validate_prior_failures.py research-roadmap-next.yaml`

When a hook is not applicable, the terminal artifact or ops note should say why.

## File-Backed State Packet

Each conductor task should preserve restartable state in this shape:

```text
runs/<exp-id>/
  TASK.md
  HARNESS.md
  state/task_history.jsonl
  state/gates.json
  artifacts/result.json
  artifacts/evidence.md
  RESPONSE.md
```

The final milestone-facing artifact may still be mirrored to `results/*.json`.
The run packet is the richer trace store for restart, retro, and future
meta-harness search.

## Failure Taxonomy

The shared taxonomy is:

- `missing_deliverable`: required terminal artifact path does not exist.
- `bootstrap_only_artifact`: artifact exists but is only a pre-run skeleton.
- `stale_skeleton`: artifact was not updated by the current run.
- `gate_blocked`: upstream path, field, operator, or value did not satisfy the
  gate.
- `blocked_no_sota_gguf`: a required local SOTA GGUF model is absent or
  unusable.
- `blocked_missing_tool`: a required command, package, hardware target, or
  external tool is unavailable.
- `artifact_schema_invalid`: JSON exists but required fields or types are wrong.
- `local_verifier_mismatch`: a local verifier succeeded but the final artifact
  or benchmark gate failed.
- `timeout_with_progress`: task timed out after producing useful partial
  evidence.
- `timeout_without_progress`: task timed out without producing useful evidence.
- `no_file_changes_produced`: implementation task ended without required file
  changes or an honest blocked artifact.
- `malformed_json_artifact`: JSON cannot be parsed.

Blocked, failed, and retired artifacts should use one of these values or record
why a new value is needed.

## Gate Semantics

Every gate must record:

- upstream artifact path
- field path
- operator
- expected value
- actual value, when available
- missing-field behavior
- terminal blocked artifact path

If a task is invoked while the gate is closed, the correct outcome is a terminal
blocked artifact, not a missing artifact and not repeated doomed reruns.

## Acceptance-Object Alignment

Extra verifier, search, or orchestration layers are allowed only when their
local success criteria are aligned with the final artifact or benchmark
acceptance object.

Every verifier module should declare:

- local acceptance object
- final acceptance object
- known mismatch risks
- what happens when local and final acceptance disagree

Local success may speed up implementation, but it cannot override a failed final
artifact schema, benchmark metric, gate check, or paper-claim audit.

## Terminal Artifact Rule

Before a task is counted complete, the conductor must verify:

- required deliverable exists
- status is terminal
- `honest_verdict` is present
- required fields are present
- gates are either satisfied or honestly blocked
- verification commands are recorded
- specs, traceability, status, and changelog are reconciled when changed
