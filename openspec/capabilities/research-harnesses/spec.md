# Research Harnesses Capability Specification

**Capability:** research-harnesses
**Version:** 0.1.0
**Status:** Draft
**Traces to:** FR-11, GitHub issue #9

## Overview

Defines the shared runtime semantics for Carnot research tasks as executable
natural-language harnesses. Experiment-specific logic remains in roadmap task
prompts, while this capability defines the common contracts, state model,
failure taxonomy, gate semantics, and acceptance-object alignment rules that
the conductor and downstream auditors must enforce.

## Requirements

### REQ-HARNESS-001: NLAH Charter Artifacts

The repository shall maintain a Natural-Language Agent Harness charter for the
Carnot conductor with these deliverables:

- `ops/conductor-runtime-charter.md`
- `openspec/capabilities/research-harnesses/spec.md`
- `results/experiment_1280_nlah_conductor_charter.json`

The terminal result artifact shall include `status`, `charter_written`,
`openspec_written`, `failure_taxonomy_count`,
`terminal_artifact_rules_defined`, `gate_semantics_defined`,
`file_backed_state_packet_defined`,
`acceptance_object_alignment_rule_defined`, and `honest_verdict`.

### REQ-HARNESS-002: Terminal Artifact Contract

Each conductor task shall declare a required deliverable path and a required
terminal schema before execution begins. Terminal statuses are `complete`,
`blocked`, `failed`, and `retired`.

The conductor shall not count `in_progress`, bootstrap-only skeletons, missing
deliverables, malformed JSON artifacts, or stale artifacts as complete.

### REQ-HARNESS-003: Role Boundaries

The conductor harness shall define non-overlapping responsibilities for these
roles:

- conductor/runtime parent
- planner
- implementer
- verifier/auditor
- retro/backfill auditor
- paper-claim auditor

Each role shall declare what it may mutate, what it must verify, and what it is
not allowed to infer from commit presence alone.

### REQ-HARNESS-004: Stage Templates

The conductor harness shall define reusable stage templates for at least:

- `PLAN -> IMPLEMENT -> VERIFY -> WRITE_ARTIFACT -> AUDIT_TERMINAL`
- `LOAD_PRIOR_ARTIFACTS -> ANALYZE -> WRITE_BACKFILL -> VERIFY_TERMINAL`
- `GATE_CHECK -> RUN_OR_BLOCK -> WRITE_TERMINAL_ARTIFACT`

Each stage template shall name the acceptance object that determines whether the
stage is complete.

### REQ-HARNESS-005: Deterministic Adapter Hooks

The conductor harness shall document deterministic adapter hooks for targeted
tests, lint or formatting checks, JSON validation, result artifact validation,
gate evaluation, paper-claim audits, and repository file checks.

Adapter hooks shall be invoked by explicit command names or by documented
equivalent checks. A task may skip a hook only when the terminal artifact records
why the hook is not applicable.

### REQ-HARNESS-006: File-Backed State Packet

Each conductor task shall support a file-backed state packet shape that can
preserve task prompts, harness instructions, gate state, artifact evidence, and
restart history.

The canonical packet shape is:

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

The benchmark-facing result may still be mirrored to `results/*.json`, but the
state packet is the restartable evidence record.

### REQ-HARNESS-007: Failure Taxonomy

The conductor harness shall define a shared failure taxonomy that includes at
least:

- `missing_deliverable`
- `bootstrap_only_artifact`
- `stale_skeleton`
- `gate_blocked`
- `blocked_no_sota_gguf`
- `blocked_missing_tool`
- `artifact_schema_invalid`
- `local_verifier_mismatch`
- `timeout_with_progress`
- `timeout_without_progress`
- `no_file_changes_produced`
- `malformed_json_artifact`

Terminal artifacts shall use the taxonomy when recording blocked, failed, or
retired outcomes.

### REQ-HARNESS-008: Gate Semantics

Every gated task shall name the upstream artifact path, field, operator, and
value. If a gated task is invoked while the gate is closed, it shall write a
terminal blocked artifact with the exact gate that failed.

Downstream tasks shall not infer completion from commit presence, log presence,
or roadmap presence alone.

### REQ-HARNESS-009: Acceptance-Object Alignment

Extra verifier, search, or orchestration layers are allowed only when their
local success criteria are aligned with the final artifact or benchmark
acceptance object.

Every verifier module shall declare its acceptance object and known mismatch
risks. A local verifier success shall not mark the task complete when the final
artifact schema or benchmark gate disagrees.

### REQ-HARNESS-010: Meta-Harness Conductor Search

The repository shall provide a deterministic meta-harness search workflow that
evaluates at least five conductor-policy candidates over at least eight cheap
conductor-harness eval cases.

The workflow shall write:

- `scripts/meta_harness_conductor_search.py`
- `ops/meta-harness-conductor-skill.md`
- `ops/conductor-harness-eval-suite.md`
- `results/experiment_1281_meta_harness_conductor_search.json`
- `meta_harness_runs/`

The terminal result artifact shall include `candidate_harnesses_evaluated`,
`eval_cases_defined`, `baseline_score`, `best_score`,
`improvement_over_baseline`, `best_candidate_id`,
`pareto_frontier_written`, `trace_store_written`, `trace_store_path`,
`recommended_policy_changes`, `hardcoded_leakage_audit_passed`, and
`honest_verdict`.

### REQ-HARNESS-011: Full Trace Store

The meta-harness workflow shall preserve candidate policy source, scores,
execution traces, verifier outputs, gate evaluation, artifact timeline, and
final candidate artifacts in a filesystem trace store. The proposer-facing
history shall be navigable with normal filesystem tools rather than compressed
into scalar scores only.

### REQ-HARNESS-012: Candidate Leakage Audit

The meta-harness workflow shall audit candidate policies for hard-coded
experiment-id leakage or task-specific string leakage. A terminal `complete`
artifact shall set `hardcoded_leakage_audit_passed=true` only when all candidate
policies pass the leakage audit.

### REQ-HARNESS-013: Conductor Supervisor Health Guard

The repository shall provide a deterministic external supervisor for the
research conductor. The supervisor shall:

- maintain a PID file for duplicate-process detection and cleanup;
- inspect the conductor heartbeat file and write `heartbeat_missing` or
  `heartbeat_stale` alerts when the heartbeat is absent or older than the
  configured threshold;
- leave fresh heartbeats alert-free;
- identify and terminate orphan conductor processes that are not the legitimate
  conductor PID recorded in state; and
- archive and truncate the working conductor log when the log handle must be
  reset, recording a `log_handle_reset` alert.

These checks shall be testable with temporary filesystem paths and mocked
process operations so the unit suite does not kill real conductor processes.

### REQ-HARNESS-014: Generated Test Import Guard

The repository shall provide a lightweight audit for planner-generated Python
tests. The audit shall parse test files without importing them, inspect local
`carnot.*` and `scripts.*` import targets, and fail when a test imports a local
module path that is neither present in the repository nor declared by the
audited roadmap as an implementation deliverable.

The audit shall report how many roadmaps, tests, and local import targets were
checked, how many missing imports were found, and how many missing imports were
allowed because the roadmap explicitly declared the implementation deliverable.

### REQ-HARNESS-015: Upstream-Only PRD Gap And Failure Taxonomy Tables

The repository shall provide a deterministic aggregation helper for PRD gap and
agent-failure taxonomy tables. The helper shall read only the upstream result
artifact files that exist at execution time, record absent upstream files as
missing, and avoid fabricating field values from roadmap expectations or hidden
live model inference. Each lane classification shall cite artifact paths and
exact supporting field names, classify lanes as `closed`, `partial`, `blocked`,
`honest_null`, or `missing`, and count the applicable failure-taxonomy tags.

For Exp5439 specifically, the terminal artifact shall write
`results/experiment_5439_prd_gap_agent_failure_table_v494.json` with
`upstream_artifacts_read`, `upstream_artifacts_missing`, `closed_lanes`,
`partial_lanes`, `blocked_lanes`, `honest_null_lanes`, `missing_lanes`,
`failure_taxonomy_counts`, `prd_gap_table_ready`, `inference_substrate`, and a
terminal-prefixed `honest_verdict`.

For Exp5452 specifically, the terminal artifact shall write
`results/experiment_5452_prd_gap_agent_failure_table_v495.json` with the
required fields `milestone`, `artifacts_expected`, `artifacts_found`,
`closed_count`, `partial_count`, `blocked_count`, `honest_null_count`,
`missing_count`, `prd_gap_table`, `agent_failure_table`,
`unsupported_claims_detected`, `inference_substrate`, and `honest_verdict`.
The helper shall read the Exp5441 through Exp5451 result artifacts, classify
missing expected artifacts as `missing`, map every non-missing result to the PRD
goal lanes for verifiable reasoning, continuous self-learning, hardware
acceleration readiness, ARC progress, model locality, and safety/traceability,
and preserve partial, blocked, honest-null, no-speedup, tautology-risk,
measurement-unavailable, and unsupported-claim evidence without upgrading it
from roadmap intent.

### REQ-HARNESS-5920: Task-Owned Clean Boundary With Global Failure Delta

The research harness shall permit a task to qualify a fresh deterministic
consumer boundary without requiring the unrelated global suite to be clean when
a prior exact node-id baseline is recorded. For Exp5920, the task-owned clean
boundary SHALL include focused unit tests, coverage for new code, schema
validation, stream replay, fresh-process replay, tamper matrix, immutable hash
checks, adversarial verification, spec coverage, applicable E2E checks,
protected-file checks, root-clutter checks, and the required global command
`.venv/bin/pytest tests/python -q`.

The global command SHALL be recorded separately as a node-id delta check:
readiness may use `global_suite_failure_delta<=0` only when every nonzero node
is present in the pre-task baseline and no new node id appears. This rule SHALL
NOT suppress, deselect, relabel, or rewrite unrelated failures, and it SHALL
NOT reopen a retired experiment scope.

### REQ-HARNESS-5940: Gateway-Charged Action Accounting For ARC Score Claims

The offline ARC harness (`scripts/arc_leaderboard_eval.py:run_game`) charges an
action ONLY on a non-RESET move, while the live competition gateway charges a
RESET an action as well (`arc_agi/scorecard.py:701-704` `inc_reset_count`
increments both `resets` and `actions`, reached from `update_scorecard`).
Because the scorer's per-level cost is a DIFFERENCE of cumulative CHARGED
counts (`:479`) and the per-level score is
`min((baseline_actions / level_actions)**2 * 100, 115)`, a reset taken BEFORE a
level-up lands inside that level's denominator and is squared.

Any harness or analyser that reports an ARC efficiency or per-game score SHALL:

1. Distinguish three units explicitly and never conflate them: OFFLINE ACTIONS
   (excludes resets), FRAMES (loop iterations, includes resets), and
   GATEWAY-CHARGED (non-RESET moves plus resets — the only unit the competition
   score is a function of). The identity
   `gateway_charged == frames == offline_actions + n_resets` SHALL hold.
2. Record PER-LEVEL reset attribution (`resets_before_levelups`,
   `level_up_charged`), not merely a whole-run `n_resets`. A whole-run count is
   insufficient to recover the correction, because the scorer differences
   cumulative counts.
3. Compute any score through the INSTALLED scorer objects
   (`arc_agi.scorecard.EnvironmentScoreCalculator`, or a `Scorecard`/`Card`
   driven through its real mutators and scored via
   `EnvironmentScorecard.from_scorecard`) and NEVER through a paraphrase of the
   scoring formula.
4. Read per-level human baselines through `env.info` (the
   `_baseline_actions` helper), and assert them non-zero. A dead baseline
   channel makes both charge models sum to 0.0, which reads as a clean null.
5. Report a bound honestly as uninformative when per-level attribution is
   absent, rather than presenting a bound whose best case is zero by
   construction as though it answered the magnitude question.

### REQ-HARNESS-SAMPLER-NO-SPECANN: Phase 3 SpecAnn Ban

Phase 3 substrate sampler MUST NOT use Spectral Annealing (Deep Think
DT-COMPOSITION 2026-05-08).

## Scenarios

### SCENARIO-HARNESS-5940-1: A Reset Before A Level-Up Costs Score

**Given** a level completed in 10 charged actions against a human baseline of 10
**And** 5 RESETs were taken before that level-up
**When** the score is computed through the installed scorer chain
**Then** the level is charged 15, not 10
**And** its score falls from `(10/10)^2*100 = 100` to `(10/15)^2*100 = 44.44`

### SCENARIO-HARNESS-5940-2: The Post-Solve Tail Is Free

**Given** a run that completes a level and then spends 530 further frames,
30 of them RESETs, without completing another level
**When** the score is computed through the installed scorer chain
**Then** the score is identical to the same run truncated at the level-up,
because an incomplete level scores 0.0 regardless of what it is charged

### SCENARIO-HARNESS-5940-3: A Whole-Run Reset Count Cannot Determine The Correction

**Given** a recorded row carrying only `n_resets` and `level_up_actions`
**When** the gateway-charged score is bounded from that row alone
**Then** the bound's best case equals the offline score exactly, because every
reset may legally have landed in the free post-solve tail
**And** the analyser MUST report the bound as uninformative about magnitude
rather than presenting either endpoint as the answer

### SCENARIO-HARNESS-5940-4: Greedy Allocation Understates The Worst Case At The Cap

**Given** a level solved SUPERHUMANLY (27 charged actions against a human
baseline of 39), whose raw score `(39/27)^2*100 = 208.6` is capped at 115
**When** a worst-case reset allocation is searched by greedy marginal gain
**Then** greedy allocates nothing, because the cap's flat region has zero
marginal, and returns the UNCORRECTED score
**And** an exact allocation search MUST be used instead

### SCENARIO-HARNESS-5940-5: A Zeroed Baseline Channel Is Refused, Not Scored

**Given** a row whose per-level `human_actions` are all zero, because the
baselines were read off the wrong attribute rather than through `env.info`
**When** the row is re-scored
**Then** the row is reported UNRESCORABLE with an explicit reason
**And** it is NOT scored as 0.0 under both charge models, which would read as
"the two charge models agree"

### SCENARIO-HARNESS-001: Bootstrap Skeleton Is Not Complete

**Given** a task wrote a result artifact with `status="in_progress"`
**When** the conductor audits terminal completion
**Then** the task is not counted as complete
**And** the failure is classified as `bootstrap_only_artifact` or
`stale_skeleton`.

### SCENARIO-HARNESS-002: Closed Gate Writes Blocked Artifact

**Given** a task is gated on `results/upstream.json.field >= 0.8`
**And** the upstream field is missing or below threshold
**When** the task is invoked
**Then** the task writes a terminal blocked artifact
**And** the artifact records the upstream path, field, operator, expected value,
actual value, and `failure_type="gate_blocked"`.

### SCENARIO-HARNESS-003: Local Verifier Cannot Override Artifact Gate

**Given** a local verifier reports success
**And** the required result artifact is missing or schema-invalid
**When** the conductor audits terminal completion
**Then** the task is not counted as complete
**And** the failure is classified as `local_verifier_mismatch` or
`artifact_schema_invalid`.

### SCENARIO-HARNESS-004: File-Backed State Supports Restart

**Given** a task is interrupted after implementation but before terminal audit
**When** the conductor restarts from the file-backed state packet
**Then** it can recover the task prompt, harness instructions, gate state,
artifact evidence, and previous verification outputs without relying on a
compressed summary only.

### SCENARIO-HARNESS-005: Meta-Harness Writes Candidate Trace Store

**Given** the deterministic conductor eval suite defines at least eight cases
**When** the meta-harness search workflow evaluates candidate policies
**Then** it writes at least five `meta_harness_runs/candidate_*` directories
**And** each candidate directory includes `policy.md`, `score.json`,
`traces/verifier_outputs.jsonl`, and `results/final_artifact.json`.

### SCENARIO-HARNESS-006: Meta-Harness Reports Honest Frontier

**Given** a baseline conductor policy and at least four candidate improvements
**When** the meta-harness search workflow completes
**Then** the result artifact reports `baseline_score`, `best_score`,
`improvement_over_baseline`, `best_candidate_id`, and
`pareto_frontier_written=true`
**And** if no candidate improves the baseline, `honest_verdict` records the
negative result rather than fabricating improvement.

### SCENARIO-HARNESS-007: Hard-Coded Leakage Blocks Clean Audit

**Given** a candidate policy text contains a hard-coded experiment id such as
`exp1281`
**When** the leakage audit runs
**Then** the candidate is marked as leaking
**And** the terminal artifact cannot set `hardcoded_leakage_audit_passed=true`.

### SCENARIO-HARNESS-008: Supervisor Alerts And Recovers Deterministically

**Given** supervisor file paths are redirected to a temporary workspace
**When** the heartbeat file is stale or missing, an extra conductor PID is
present, or the conductor log handle is reset
**Then** the supervisor writes the matching structured alert
**And** only the orphan PID is terminated
**And** the legitimate conductor PID and unrelated host state are left intact.

### SCENARIO-HARNESS-009: Orphan Generated Test Import Is Blocked

**Given** a generated pytest imports
`carnot.reporting.milestone_117_activation_manifest`
**And** the audited project has no matching Python module file
**And** the audited roadmap does not declare that module as an implementation
deliverable
**When** the generated-test import guard runs
**Then** the audit fails before pytest collection
**And** the orphan import is reported with the test file and module path.

### SCENARIO-HARNESS-010: PRD Gap Table Uses Existing Artifact Fields Only

**Given** a PRD gap table task names a fixed set of upstream result artifacts
**When** at least one upstream file exists and at least one upstream file is
absent
**Then** the helper records the existing paths in `upstream_artifacts_read`
**And** records the absent paths in `upstream_artifacts_missing`
**And** every non-missing lane cites exact `supporting_fields.field_name`
entries from existing artifacts
**And** the terminal artifact declares
`inference_substrate="aggregation_from_upstream_artifacts"`.

### SCENARIO-HARNESS-011: V495 PRD Gap Table Preserves Bounded And Null Evidence

**Given** the Exp5452 task names the expected Exp5441 through Exp5451 result
artifacts
**And** those artifacts report a mixture of complete, bounded, blocked, and
honest-null evidence
**When** the Exp5452 aggregation helper runs
**Then** it writes `results/experiment_5452_prd_gap_agent_failure_table_v495.json`
with the required artifact fields
**And** every `prd_gap_table` row cites existing artifact paths and supporting
fields
**And** `agent_failure_table` records no-bank, measurement-unavailable,
tautology-risk, unsupported-claim, precondition-block, gate-block, and
implementation-failure pattern status
**And** `unsupported_claims_detected` lists rejected unsupported claims instead
of promoting them into PRD progress.

### SCENARIO-HARNESS-5920: Clean Task Boundary Allows Preserved Global Debt

**Given** a task-owned focused boundary exits zero
**And** a pre-task global-suite baseline records exact unrelated node ids
**When** the same global command exits nonzero after the task
**Then** the task can still report a ready boundary only if the failure delta is
at most zero by exact node id
**And** every unrelated failure remains visible in the artifact.

### SCENARIO-HARNESS-SAMPLER-1: Direct HUBO Evaluation At Production Scale

**Given** a Phase 3 substrate sampler evaluates an unreduced HUBO energy
**And** the production-scale problem has n≥128 variables
**When** the sampler performs inference-time argmin
**Then** HUBO direct evaluation MUST succeed without QUBO reduction at
production scale (n≥128).

### SCENARIO-HARNESS-SAMPLER-2: Future SpecAnn Proposals Rebut Rejection

**Given** a future Phase 3 planning proposal recommends SpecAnn or Spectral
Annealing
**When** it enters the research harness
**Then** the proposal MUST document why the rejection rationale no longer
applies.

## Implementation Status

| Requirement | Documentation | Artifact |
|-------------|---------------|----------|
| REQ-HARNESS-001 | Implemented (`ops/conductor-runtime-charter.md`) | Implemented (`results/experiment_1280_nlah_conductor_charter.json`) |
| REQ-HARNESS-002 | Implemented (`ops/conductor-runtime-charter.md`) | Implemented |
| REQ-HARNESS-003 | Implemented (`ops/conductor-runtime-charter.md`) | Implemented |
| REQ-HARNESS-004 | Implemented (`ops/conductor-runtime-charter.md`) | Implemented |
| REQ-HARNESS-005 | Implemented (`ops/conductor-runtime-charter.md`) | Implemented |
| REQ-HARNESS-006 | Implemented (`ops/conductor-runtime-charter.md`) | Implemented |
| REQ-HARNESS-007 | Implemented (`ops/conductor-runtime-charter.md`) | Implemented |
| REQ-HARNESS-008 | Implemented (`ops/conductor-runtime-charter.md`) | Implemented |
| REQ-HARNESS-009 | Implemented (`ops/conductor-runtime-charter.md`) | Implemented |
| REQ-HARNESS-010 | Implemented (`ops/meta-harness-conductor-skill.md`, `ops/conductor-harness-eval-suite.md`) | Implemented (`results/experiment_1281_meta_harness_conductor_search.json`) |
| REQ-HARNESS-011 | Implemented (`scripts/meta_harness_conductor_search.py`) | Implemented (`meta_harness_runs/`) |
| REQ-HARNESS-012 | Implemented (`scripts/meta_harness_conductor_search.py`) | Implemented |
| REQ-HARNESS-013 | Implemented (`scripts/conductor_supervisor.py`) | Implemented (`results/experiment_1027_conductor_supervisor.json`) |
| REQ-HARNESS-014 | Implemented (`scripts/audit_orphan_test_imports.py`) | Implemented (`tests/python/test_audit_orphan_test_imports.py`) |
| REQ-HARNESS-015 | Implemented (`python/carnot/reporting/prd_gap_agent_failure_table_v494_5439.py`; planned `python/carnot/reporting/prd_gap_agent_failure_table_v495_5452.py`) | Implemented (`results/experiment_5439_prd_gap_agent_failure_table_v494.json`); planned (`results/experiment_5452_prd_gap_agent_failure_table_v495.json`) |
| REQ-HARNESS-5920 | Implemented (`python/carnot/experiment_5920_prospective_event_stream_admission.py`) | Implemented (`tests/python/test_experiment_5920_prospective_event_stream_admission.py`) |
| REQ-HARNESS-5940 | Implemented (`scripts/arc_gateway_rescore.py`, `scripts/arc_gateway_exact_attribution.py`, `scripts/arc_leaderboard_eval.py` per-level reset instrumentation) | Implemented (`tests/python/test_arc_gateway_rescore.py`, `results/outer_loop_arc_gateway_rescore_20260726.json`) |
| REQ-HARNESS-SAMPLER-NO-SPECANN | Implemented (`_bmad/architecture.md`, `ops/exclusion_manifest.yaml`) | Implemented (`results/experiment_1563_specann_rejection_architecture_record.json`) |
