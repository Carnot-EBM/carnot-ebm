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

### REQ-HARNESS-6393: Numeric Gate Producers Must Expose Bare Finite Scalars

Any producer that feeds a conductor numeric comparison gate SHALL expose the
gated field as a finite bare number. It SHALL place maps, lists, per-model
details, and diagnostic receipts in separate non-gated fields.

The producer SHALL validate the comparison surface before it reports readiness.
Mapping, list, string, bool, NaN, infinity, rounded sign-change, missing-row,
duplicate-row, stale-hash, and order-swap inputs SHALL fail closed. The replay
receipt SHALL record the exact operands and conductor comparison result.

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

### REQ-HARNESS-6040: Unattended Writers Of Never-Prune Records Must Merge

Any unattended writer (systemd timer, cron, conductor task) that updates a
durable never-prune status record MUST merge its fields over the existing
record rather than replacing the file wholesale.

Origin: on 2026-07-29 the daily-prep timer's bare
`write_text(json.dumps({...six keys}))` on `ops/arc-daily-prep-status.json`
deleted the seven submission-trail keys the file also carried, including
`submission_ref` and `prior_submission_scores` — the leaderboard score-by-date
history. Because the writer is unattended the loss was silent, and the
conductor's routine `git add -A` would have published the deletion.

A writer subject to this requirement MUST additionally:

- Retire, never delete, any field that describes a superseded artifact
  version. Such fields move to an append-only history list so the record stays
  never-prune-compliant while no longer describing the current version.
- Keep cumulative fields (histories that are not per-version facts) live rather
  than retiring them alongside per-version fields.
- Tolerate a corrupt or unreadable prior record without blocking its primary
  job.

The merge MUST be implemented as a pure function so it is testable without the
writer's network side effects, and its tests MUST be written against the
actual record that was destroyed rather than a synthetic example.

### REQ-HARNESS-6050: A Matched-Pair A/B Grid MUST Pass A Treatment-Activation Pre-Flight

Before a matched-pair A/B grid is run on the live agent, a SHORT probe MUST be run
per cell in both arms, the two arms' ACTION TRACES MUST be diffed, and the grid MUST
be REFUSED unless the treatment perturbs enough cells for the planned test to be
able to reach significance.

Origin: on 2026-07-29 three A/B measurements on the live ARC agent each returned a
null after hours of wall-clock. The post-mortem diffed the arms' action traces and
found 8 of 12 cells byte-IDENTICAL, 3 truncation-only, and 1 genuinely perturbed. A
byte-identical pair cannot differ on ANY downstream endpoint, so the ceiling on
discordant pairs was 1 and the smallest p-value the grid could ever have reported
was 1.0. The null was guaranteed before the first cell started and was knowable in
minutes.

The pre-flight MUST:

- Classify every matched pair as IDENTICAL, TRUNCATION_ONLY, PERTURBED or MISSING.
  TRUNCATION_ONLY (one arm cut short by a wall-clock cap, hard timeout or crash,
  with both arms agreeing on every action they both took) is a MISSING OBSERVATION
  and MUST NOT be scored as a zero, because a treatment that does more work is
  systematically slower and coding its truncations as zeros biases against it.
- Distinguish a truncation caused by a RESOURCE CAP from an early termination the
  arm chose. The latter is a real behavioural difference and MUST be classified
  PERTURBED.
- DERIVE its threshold from the planned test rather than picking a number. For a
  two-sided sign test over matched pairs, all discordant pairs one-way gives
  `p = 2 * (1/2)**d`, so `d = 6` is the smallest count reaching `alpha = 0.05`
  (d=5 gives 0.0625, d=6 gives 0.03125).
- Take the ceiling on discordant pairs from PERTURBED cells only. A charitable
  ceiling that also credits truncations MAY be reported for argument but MUST NOT
  decide the verdict.
- Measure an A/A NOISE FLOOR (one arm against a byte-identical replicate of itself)
  and attribute a perturbation to the treatment only where the harness is
  demonstrably deterministic on that cell. Where no noise floor was measured the
  verdict MUST say so, because the ARC generator samples at a temperature and sends
  no seed, so raw perturbation cannot be attributed to the treatment.
- State on every verdict, PASS included, that PASS does not mean the experiment is
  powered: perturbation is necessary but not sufficient for discordance on the
  endpoint.

A REFUSAL IS A SUCCESSFUL OUTCOME of the pre-flight. It converts "we measured it
and found nothing" — which future planners read as evidence against the treatment —
into "the treatment never activated, so we have learned nothing about it yet."

The classifier and the verdict MUST be pure functions, testable with no GPU, and
they MUST be validated RETROSPECTIVELY against a grid already known to have been
underpowered. A pre-flight that cannot refuse a known-dead grid does not work.

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

### SCENARIO-HARNESS-6040-1: A New Prep Must Not Destroy The Submission Trail

**Given** `ops/arc-daily-prep-status.json` records `submission_ref` 54768046,
`submitted_at`, and `prior_submission_scores` for kernel version 9
**When** the unattended daily-prep timer writes a fresh status for version 10
**Then** no prior key is dropped without being archived
**And** `prior_submission_scores` remains live, because it is cumulative rather
than a per-version fact

### SCENARIO-HARNESS-6040-2: A Fresh Version Must Not Look Already-Submitted

**Given** the prior record carries `submitted: true` for kernel version 9
**When** a status for version 10 is written
**Then** the per-version submission fields are moved into an append-only
`submission_history` entry naming version 9
**And** they are absent from the live record, so the unsubmitted version 10 is
not readable as already submitted

### SCENARIO-HARNESS-6040-3: Re-Prepping The Same Version Keeps Its Record

**Given** the prior record describes kernel version 9 and is submitted
**When** version 9 is re-prepped
**Then** its own submission fields stay live and no history entry is created

### SCENARIO-HARNESS-6050-1: A Byte-Identical Arm Pair Is Not A Tie

**Given** two arms of a matched-pair A/B whose action traces are byte-identical on a cell
**When** the pre-flight classifies that pair
**Then** it MUST be classified IDENTICAL and MUST NOT contribute to the ceiling on
discordant pairs, because no endpoint computed downstream of those actions can differ.

### SCENARIO-HARNESS-6050-2: A Wall-Clock Truncation Is A Missing Observation

**Given** one arm was cut short by a wall-clock cap and the two arms agreed on every
action they both took
**When** the pre-flight classifies that pair
**Then** it MUST be classified TRUNCATION_ONLY and MUST NOT be scored as a zero for the
truncated arm.

**And Given** the shorter arm instead terminated on its own terms
**Then** the pair MUST be classified PERTURBED, because an early stop the agent chose is
a real behavioural difference.

### SCENARIO-HARNESS-6050-3: The Grid Is Refused When No Outcome Could Reach Alpha

**Given** the probe finds fewer PERTURBED cells than the derived threshold of 6
**When** the pre-flight renders a verdict
**Then** it MUST REFUSE the grid, and MUST report the best reachable p-value and the
number of matched pairs the observed perturbation rate would require.

### SCENARIO-HARNESS-6050-4: The Pre-Flight Refuses A Grid Already Known To Be Dead

**Given** the 12 committed matched pairs of the 2026-07-29 engine-retention A/B
(`results/arc_engine_retention_20260729/cells/`), independently known to be 8 identical
/ 3 truncation-only / 1 perturbed
**When** the pre-flight is run against them
**Then** it MUST REFUSE, reporting a strict ceiling of 1 (best reachable p = 1.0), a
charitable ceiling of 4 (best reachable p = 0.125), and 72 matched pairs needed at the
observed perturbation rate.

### SCENARIO-HARNESS-6050-5: A Nondeterministic Harness Cannot Buy A Pass

**Given** every cell perturbs under A/B but every cell also perturbs under A/A
**When** the pre-flight renders a verdict with the A/A noise floor supplied
**Then** attributable perturbation MUST be 0 and the grid MUST be REFUSED, because a
difference on a cell where the harness does not repeat itself cannot be attributed to
the treatment.

**And Given** no A/A noise floor was measured at all
**Then** the verdict MUST carry an explicit warning that a PASS computed without one is
uninterpretable.

### SCENARIO-HARNESS-SAMPLER-2: Future SpecAnn Proposals Rebut Rejection

**Given** a future Phase 3 planning proposal recommends SpecAnn or Spectral
Annealing
**When** it enters the research harness
**Then** the proposal MUST document why the rejection rationale no longer
applies.


### SCENARIO-HARNESS-6050-6: Equal traces with both arms truncated are not a measurement
GIVEN two arms that agree on every action and NEITHER finished
WHEN the pair is classified
THEN the class is `BOTH_TRUNCATED`, not `IDENTICAL`. It is a missing observation: neither arm was
allowed to reach an endpoint, so the pair can neither show the treatment is inert nor serve as a
determinism witness.

### SCENARIO-HARNESS-6050-7: A truncated A/A pair cannot license attribution
GIVEN six A/B-perturbed cells whose A/A replicates were each cut short
WHEN the verdict is computed
THEN it is REFUSE with `n_perturbed_attributable == 0`. A truncated A/A measures nothing, so it
must never certify the harness as deterministic on that cell. Enforced twice — once structurally,
and once at the point of use, because `noise_pairs` is plain data a caller can hand-build. With a
genuine complete A/A floor the same A/B data PASSES, proving the refusal is caused by the
truncation and not by an unrelated tightening.

### SCENARIO-HARNESS-6050-8: Probing a subset decides against the PLANNED grid size
GIVEN 4 of 12 probed cells perturb attributably (a 1-in-3 rate)
WHEN `planned_n_cells` exceeds the probed count and a noise floor was measured
THEN the decision uses the projected attributable count at the planned size: REFUSE at 12 and 15,
PASS at 24. Without a measured noise floor the projection is refused outright, because forecasting
an unattributed rate only predicts how much noise the larger grid will contain.

### SCENARIO-HARNESS-6050-9: An unfinished grid is INCONCLUSIVE, never REFUSE
GIVEN a grid stopped with cells still outstanding that could carry the ceiling to `required`
WHEN the verdict is computed
THEN it is `INCONCLUSIVE`, the report says the grid is INCOMPLETE and "This is NOT a refusal", and
the interpretation states it must never be cited as evidence the treatment is inert. A PASS is
never downgraded by outstanding cells (more cells cannot un-perturb measured ones), and outstanding
cells that could NOT close the gap still yield a genuine REFUSE.

### SCENARIO-HARNESS-6050-10: Truncated cells are excluded from the rate, not scored as zeros
GIVEN a grid containing MISSING and TRUNCATION_ONLY cells
WHEN the perturbation rate is computed
THEN the denominator is USABLE observations only, and the comparable-cell rate is still emitted
under a name that says it counts truncations as zeros — so a refusal that holds under both is
visibly stronger than one that needs the favourable denominator.

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
| REQ-HARNESS-6040 | Implemented (`scripts/kaggle/prep_daily_submission.py:_merge_prep_status`) | Implemented (`tests/python/test_prep_daily_status_merge.py`, `ops/arc-daily-prep-status.json`) |
| REQ-HARNESS-6050 | Implemented (`python/carnot/analysis/treatment_activation_preflight.py` — `classify_trace_pair`, `preflight_verdict`, `min_one_way_discordant_pairs`, `two_sided_sign_test_p`, `format_report`; pure, zero-GPU, threshold DERIVED from the sign test rather than hardcoded; A/A noise-floor attribution). **FOUR classes** — a 2026-07-30 review added `BOTH_TRUNCATED` because `classify_trace_pair` tested `a == b` BEFORE consulting completeness, so a pair in which NEITHER arm got past action 27 of a 400-action budget was labelled IDENTICAL and its recorded reason asserted "no downstream endpoint can differ" — unsupportable when neither arm reached an endpoint (live on cn04 and r11l, so the published retention partition of "8 identical / 3 truncated / 1 perturbed" over-counted IDENTICAL by two; the honest partition is 6 / 3 / 2 / 1). **THREE verdicts** — `INCONCLUSIVE` was added because every unrun cell classifies as MISSING and MISSING is not PERTURBED, so a partially-run probe returned REFUSE: the module's own "a missing observation is never a zero" principle violated at the verdict level, and it would have laundered a process decision (a probe stopped early because review found a bug in the arm under test) into a finding that the treatment is inert. Attribution now requires BOTH A/A arms complete, enforced twice (structurally via the class fix, and again at the point of use because `noise_pairs` is plain data a caller can hand-build). `planned_n_cells` now DECIDES: it was inert — only `ceiling_strict >= required` mattered, so a 33%-perturbation probe of 12 cells was refused at planned_n_cells = 12, 24, 48 AND 1000 alike, though 1000 at that rate expects ~333 against the 6 needed; projection is gated on a measured noise floor, because forecasting an unattributed rate just predicts how much noise the bigger grid will contain. The perturbation rate is now over USABLE observations: truncation-affected cells were in the denominator and never the numerator, i.e. scored as zeros, which this module's own class documentation forbids — and the bias has a direction, since the arm that gets truncated is systematically the slower one and a treatment that does more work is systematically slower. On the retention grid that is 1/7 = 0.1429 (42 pairs needed), not 1/12 = 0.0833 (72).) | Implemented (`tests/python/test_treatment_activation_preflight.py` — **36 tests**, including the RETROSPECTIVE validation against the 12 committed matched pairs of the 2026-07-29 engine-retention A/B, which the pre-flight correctly REFUSES: strict ceiling 1, best reachable p = 1.0. Read-only on `results/`.) **HONEST NOTE:** with 5 truncation-affected cells rather than 3, that refusal's CHARITABLE ceiling is now 6, which just reaches alpha=0.05 — so it no longer holds "even under the most generous possible coding of the missing observations" and rests on the strict reading. That reading is the correct one (coding a missing observation as favourable is the error this module exists to prevent), but the weaker margin is stated rather than buried. **RETRACTION (2026-07-30):** the previously-reported "free A/A noise floor" — built by pairing the retention grid's `ret1` cells against a DIFFERENT experiment's `31b` cells on the grounds that both were "retention ON at seed 1 on the same GGUF" — was NOT an A/A. The held-out cells ran before commit `11cd3c3a9` introduced engine retention at all and carry none of its diagnostic fields, so ≥6 agentic commits separate the sets and the comparison was an A/B of the treatment under test. The claims it supported ("2 of 5 cells diverge under IDENTICAL code", hence "vc33's attributable perturbation is ZERO") are WITHDRAWN from the artifact, the spec, ops docs and the tests. Replaced by a real same-commit A/A committed at `results/arc_goalspec_f9a458e87_preflight_20260730/cells` (`post` vs `postb`, both at `f9a458e87`, equality of ten witness fields asserted from the cells themselves rather than argued in prose): **1 of 2 pairs diverges EVEN WITH `CARNOT_ARC_GENERATOR_SEED` set** — ft09 at action 26, and the two runs disagreed on whether a plan was found at all (1 vs 0). So seeding the sampler is NECESSARY BUT NOT SUFFICIENT and a noise floor must be MEASURED per grid. n=2 supports that qualitative claim and no rate is asserted. The retention grid's one perturbed cell is now reported as UNATTRIBUTABLE, not proven-noise; the REFUSE verdict is unchanged either way, so the retraction costs a claim, not a decision. The invalid pairing is pinned as a test so it cannot quietly return, keyed on the fields the two cells RECORD. Three further review findings are each pinned: the CLI refuses to pick silently between duplicate `arm__cell__*` records (it took `sorted(matches)[0]`, so a second seed or re-run replicate would have been neither read, nor reported, nor counted as MISSING — `--suffix` is the explicit resolution and multiple matches now exit 2); the verdict carries per-cell record provenance so a reader can see WHICH file each number came from; and the module docstring's advice to build a floor from "cells from a DIFFERENT experiment that happen to share treatment, seed, model and game" — which is exactly the retracted practice — was corrected, not softened. **The probe is NOT cheap on this path:** one 31B induction is 200–1200 s, so a 24-cell probe costs roughly what the grid it screens costs; the leverage came entirely from the FREE retrospective use. |
| REQ-HARNESS-SAMPLER-NO-SPECANN | Implemented (`_bmad/architecture.md`, `ops/exclusion_manifest.yaml`) | Implemented (`results/experiment_1563_specann_rejection_architecture_record.json`) |

## REQ-HARNESS-6051: A Verified-Inert Code Drift MAY Be Acknowledged, Never Silently Bumped

`artifact_freshness_lint` refuses a commit that leaves a registered artifact stale w.r.t. the code
that built it. Three registered artifacts declare a code dependency but NO `rebuild_command`, so when
that dependency legitimately changes the author cannot rebuild — and the only remaining moves were to
edit the recorded sha256 silently or to pass `--no-verify`. The lint's own docstring names the second
as the failure mode to avoid; the first is worse, because it launders an unverified change into a
verified-looking provenance block.

### SCENARIO-HARNESS-6051-1: A complete acknowledgement clears the drift
GIVEN a `provenance.freshness_acknowledgements` entry naming `path`, `sha256_was`, the EXACT
`sha256_now`, a non-empty `reason` and a non-empty `evidence`
WHEN the lint checks the artifact
THEN the status is `fresh` and the detail line says `drift ACKNOWLEDGED as verified-inert`.

### SCENARIO-HARNESS-6051-2: An acknowledgement without a reason or evidence is IGNORED
GIVEN an entry with an empty or whitespace `reason` or `evidence`
WHEN the lint checks the artifact
THEN it remains `stale`. An acknowledgement that does not say why is indistinguishable from a silent
hash bump, so it must not clear anything.

### SCENARIO-HARNESS-6051-3: The acknowledgement is pinned to one exact hash
GIVEN an acknowledgement for hash H
WHEN the dependency is edited again, producing H'
THEN the acknowledgement no longer applies and the artifact goes stale again. This is the safety
property: an acknowledgement can never become a standing exemption for a file.

### SCENARIO-HARNESS-6051-4: A malformed block is empty, not fatal
GIVEN a missing, non-list, or garbage `freshness_acknowledgements`
WHEN the lint runs
THEN it yields no acknowledgements and does not raise. A crash in the freshness layer blocks every
commit while reporting nothing about staleness, which is strictly worse than a miss.

## Implementation Status (REQ-HARNESS-6051)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-HARNESS-6051 | `scripts/artifact_freshness_lint.py:_acknowledged_inert_drift` + the per-dependency branch in `check_artifact`, which reports acknowledged drift explicitly in the `fresh` detail rather than hiding it. | `tests/python/test_artifact_freshness_acknowledgement_2026_07_30.py` (7 tests) including an end-to-end pass through `check_artifact` (stale → acknowledged-fresh → stale again for the wrong hash) and a test asserting the three real committed acknowledgements are well-formed and name checkable evidence. **Applied 2026-07-30 to the three artifacts with no `rebuild_command`, and ONLY after the drift was verified inert two ways:** structurally (`_guard_engine_write`'s first statement is an early return when `PYTEST_CURRENT_TEST` is unset, and the module diff is purely additive — zero removed or modified lines) and EMPIRICALLY (the four registered artifacts that DO have a rebuild command were rebuilt in place and deep-diffed: **77,000 leaf values compared, zero research numbers moved** — only `elapsed_s`/`measurement_wall_s` timing jitter and, in exp6021, its own recorded code-provenance hashes). Preferring the rebuild where one exists is the documented rule: it proves inertness by construction rather than by argument. |

## REQ-HARNESS-6052: An A/A Noise Floor SHALL Be Measured On Every Arm The A/B Spans

`preflight_verdict` accepted ONE `noise_pairs` mapping — one arm's A/A replicate — and treated it as
licensing attribution for a comparison that spans TWO arms. So a grid whose CONTROL arm was the
nondeterministic one would credit the control arm's self-perturbation to the treatment, and the
module would stamp every such cell attributable.

This is not a theoretical hole. The 2026-07-30 composite treatment-activation grid measured a
head-vs-headb floor of 0/6 while every A/B pair held exactly ONE unreplicated base run, and the
treatment under test changed a search DEDUP KEY — precisely the kind of change that can stabilise
iteration order, so the two arms' determinism could not be assumed equal. The module's own docstring
already says a control whose validity rests on an argument is not a control; the signature committed
that error.

### SCENARIO-HARNESS-6052-1: A second arm that perturbs removes attribution
GIVEN every cell PERTURBED under A/B, arm A's A/A all IDENTICAL, and arm B's A/A all PERTURBED
WHEN `preflight_verdict` is called with both `noise_pairs` and `noise_pairs_b`
THEN `n_perturbed_attributable` is 0 and the verdict is REFUSE, where the single-arm call returned
the full count and PASS.

### SCENARIO-HARNESS-6052-2: A cell absent from the second arm is UNWITNESSED, never a pass
GIVEN a partial second-arm replicate covering a strict subset of the probed cells
WHEN the verdict is computed
THEN only the covered cells are attributable, the uncovered ones appear in
`cells_perturbed_but_lacking_a_second_arm_noise_witness`, and they are reported as unattributable
rather than silently dropped. A second arm is expensive, so partial coverage is the realistic case
and it must degrade to a missing observation.

### SCENARIO-HARNESS-6052-3: A truncated second-arm replicate does not certify
GIVEN a second-arm A/A pair that is TRUNCATION_ONLY, or IDENTICAL with an incomplete arm
WHEN the verdict is computed
THEN the cell is not attributable. Agreement over a prefix is not a determinism witness, and the
error direction matters: a false PASS spends hours and yields a number nobody can attribute, while a
false REFUSE costs one experiment.

### SCENARIO-HARNESS-6052-4: A single-arm call still works, and says that it is single-armed
GIVEN only `noise_pairs`
WHEN the verdict is computed
THEN the attributable count is unchanged from before this requirement (backward compatible), AND the
result carries `single_arm_noise_floor_warning`, which `format_report` prints. The realistic misuse
is a caller who reads `noise_floor_measured: True` and stops there.

### SCENARIO-HARNESS-6052-5: "Never spoke" is distinguished from "said no"
GIVEN a cell PERTURBED under A/B, IDENTICAL in arm A's A/A, and absent from arm B's mapping
WHEN the verdict is computed
THEN its unattributable reason is MISSING, not PERTURBED. Conflating them is the missing-vs-present
error this module exists to prevent, committed inside its own reason strings.

## Implementation Status (REQ-HARNESS-6052)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-HARNESS-6052 | `python/carnot/analysis/treatment_activation_preflight.py:preflight_verdict` — new `noise_pairs_b` parameter; attribution loops over EVERY supplied noise mapping (`all()` semantics) rather than checking one, so with a single mapping the behaviour is bit-identical to before and with two neither arm can speak for the other. New output fields `per_cell_noise_b`, `n_noise_arms_measured`, `cells_perturbed_but_lacking_a_second_arm_noise_witness`, `single_arm_noise_floor_warning`; `format_report` prints the warning as a WARNING line. | `tests/python/test_treatment_activation_preflight_two_arm_noise_floor.py` — 8 tests, one per scenario plus a both-arms-clean control asserting the stricter rule costs no attribution and a guard that the pre-existing no-floor warning is not shadowed. All **36** pre-existing tests in `test_treatment_activation_preflight.py` pass unchanged, which is the backward-compatibility claim. **MEASURED, not assumed:** the hypothesis that motivated this (base is the noisy arm, collapsing the 2026-07-30 grid's attributable count 4 → 1) was tested by running a real `baseb` arm at 8441055c0 on the four attributable cells — base vs baseb IDENTICAL on all four (tu93 28/28, tr87 27/27, tn36 31/31, vc33 14/14), base-arm floor **0/4**. The hypothesis is REFUTED and the count stands at 4; the requirement stands regardless, because the point is that it had to be measured rather than argued. **Honest scope:** that is a result about those four cells at budget=60, NOT a proof the nondeterminism channel is closed — `probe_cell.py` pins no `PYTHONHASHSEED`, and the base worktree's own `results/proto_just_explore_budget_scan.json` documents the graph explorer's frontier as hash-order-nondeterministic across fresh processes ("the SAME seed in three FRESH processes gave atfl=362, then 153, then no-solve"). Future grids should pin it on both arms. |

## REQ-HARNESS-6053: An ABSENT `duration_s` On A Compute-Bound Artifact SHALL Be A Methodology Gap

`check_duration_vs_claim` returns immediately when `duration_s` is absent or non-finite, so the
entire DURATION_TOO_SHORT family never runs on an artifact that simply OMITS the field. The
2026-07-30 composite treatment-activation artifact declared `inference_substrate:
live_llm_inference` — a 60s floor — with `duration_s: null`, and `adversarial_verify` returned
"0 flagged". That clean result was then cited as evidence the artifact was sound. It was not
evidence about the duration at all: the field the check is DEFINED on was missing.

Verified by injection on that artifact: `1.0` and `0.0001` both fire CRITICAL DURATION_TOO_SHORT,
`null` passes silently. The asymmetry is strictly worse than a short duration — an artifact that
omits the field is INVISIBLE to the check built to catch fabrication, while an honest one recording
a real 35s gets flagged. CLAUDE.md's Adversarial Artifact Verification rule names `duration_s` as
the load-bearing fabrication-detection signal precisely because real compute takes wall-clock time,
so its ABSENCE on a compute-bound artifact is a methodology gap in the same sense as a missing seed.

### SCENARIO-HARNESS-6053-1: A compute-bound artifact with no `duration_s` is flagged
GIVEN an artifact declaring `live_llm_inference` (or carrying a compute-bound marker) with
`model_specs`, `random_seed` and `reproducibility_checksum` all present but `duration_s` absent
WHEN `check_methodology_present` runs
THEN it emits METHODOLOGY_MISSING naming `duration_s` and nothing else.

### SCENARIO-HARNESS-6053-2: Any value the duration check would skip is caught here
GIVEN `duration_s` set to `null`, `""`, the STRING `"1200"`, NaN, inf, `[]` or `{}`
WHEN `check_methodology_present` runs
THEN each is flagged. The bug is not "null is missing" but "the duration check skips anything it
cannot treat as a finite number", so a fabricator need only write a string to stay invisible.

### SCENARIO-HARNESS-6053-3: Absence and implausibility stay separate findings
GIVEN a live-LLM artifact with `duration_s: 1.0`
WHEN both checks run
THEN `check_methodology_present` is silent and `check_duration_vs_claim` emits DURATION_TOO_SHORT.
The field was measured; it is a plausibility problem, not a methodology gap, and double-reporting
would corrupt the downstream counts.

### SCENARIO-HARNESS-6053-4: Every existing exemption is inherited
GIVEN an aggregation-only artifact, or one with no compute-bound marker and no live substrate
WHEN `check_methodology_present` runs
THEN no METHODOLOGY_MISSING is emitted. The addition lives inside the function that already carries
the aggregation, ARC-no-LLM, deterministic-verifier and precondition-blocked carve-outs, so it
inherits them rather than re-deriving a second divergent set.

## Implementation Status (REQ-HARNESS-6053)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-HARNESS-6053 | `scripts/adversarial_verify.py:check_methodology_present` — `duration_s` joins `model_specs`/`random_seed`/`reproducibility_checksum` in the `missing` list, gated on `_is_finite_number`. Reported at `warn`, not `critical`: the corpus predates the requirement and quarantining historical artifacts for a newly-added field would be a retroactive gate, not a fabrication finding. | `tests/python/test_adversarial_verify_absent_duration_2026_07_30.py` — 12 tests, one per scenario plus a present-and-finite control and a non-compute-bound no-op guard. The fixture deliberately carries every OTHER methodology field so a passing "the flag fired" assertion cannot pass for the wrong reason. **Corpus impact measured over all 6311 result artifacts BEFORE shipping: 656 gain the string `duration_s` inside a METHODOLOGY_MISSING warn they ALREADY carried for another reason, and ZERO artifacts acquire a warn they did not previously have.** So the change adds detail to existing findings and quarantines nothing retroactively. **How it was found:** by probing the linter with injected durations while verifying this session's own artifact, not by a unit test — the pre-existing tests all supplied a duration, so they exercised the populated path while the absent path was unguarded. Same mode as REQ-ARC-WMTE-6050 above ("tests test what the author thought to test") and the fourth instance in one session of a comparator treating "no value" as "a value". |

## REQ-INFRA-6197: Terminal Artifact State SHALL Be Classified By A Shared Fail-Closed Contract

Carnot shall provide a reusable terminal-artifact classifier outside
`scripts/research_conductor.py`. The classifier SHALL read only the artifact
path and artifact payload, normalize `status` and `honest_verdict`, and classify
artifacts as terminal only for the explicit outcomes `complete`, `ready`,
`positive`, `null`, `blocked`, `skipped`, `retired`, and `flagged`.

The classifier SHALL reject missing files, unreadable or malformed JSON,
non-object JSON, `running`, `running_bootstrap`, bootstrap-only, unknown,
partial, and contradictory status/verdict combinations as nonterminal. A
conductor completion receipt SHALL NOT override the artifact path state. A
flagged artifact SHALL be terminal only when the artifact itself carries
`flagged_adversarial`, a non-empty `corrigendum_pending`, or an explicit
flagged status/verdict.

Exp6197 SHALL replay immutable hashed fixtures for good complete/ready/positive,
honest blocked, skipped/gated, retired, adversarial-flagged, missing,
malformed, running, running_bootstrap, bootstrap-only, Exp6183, Exp6194,
Exp6195, and Exp6196 artifacts, then write
`results/experiment_6197_v537_terminal_artifact_contract.json` atomically. The
artifact SHALL include principle-annotated field provenance for every
load-bearing required field, SHALL record `conductor_receipt_override_count: 0`
and `protected_artifact_mutation_count: 0` as bare numbers, and SHALL preserve
historical artifacts byte-for-byte.

The Exp6197 artifact SHALL include these required fields: `status`,
`fixture_paths_and_hashes`, `accepted_terminal_prefixes`,
`rejected_nonterminal_prefixes`, `status_verdict_cross_product`,
`exp6183_classification`, `exp6196_classification`,
`valid_fixture_classifications`, `conductor_receipt_override_count`,
`protected_artifact_mutation_count`, `classifier_module_and_hash`,
`focused_test_commands`, `focused_test_exit_codes`,
`full_suite_command_and_classified_exit_code`, `inference_substrate`,
`verifier_is_oracle`, `field_provenance`, `field_principles`, `duration_s`,
`reproducibility_checksum`, and `honest_verdict`.

### SCENARIO-INFRA-6197-1: Artifact path state outranks conductor receipts
GIVEN a bootstrap-only, missing, malformed, or running artifact path
WHEN a conductor receipt claims `OK` or `complete`
THEN the classifier returns a nonterminal class and records that the receipt did
not override the artifact state.

### SCENARIO-INFRA-6197-2: Status and verdict must agree or fail closed
GIVEN every cross-product of terminal, nonterminal, unknown, and contradictory
`status`/`honest_verdict` prefixes
WHEN the shared classifier normalizes the pair
THEN agreed terminal pairs return their terminal class, agreed nonterminal pairs
return nonterminal classes, and mixed or incompatible pairs return
`contradictory` or `unknown` rather than a terminal success.

### SCENARIO-INFRA-6197-3: Immutable historical fixtures are replayed by hash
GIVEN committed result artifacts including Exp6183, Exp6194, Exp6195, Exp6196,
an honest blocked gate artifact, a retired artifact, an adversarial-flagged
artifact, and a malformed artifact
WHEN Exp6197 builds its fixture report
THEN every present fixture records path, sha256, status, verdict, terminal flag,
and class without mutating any historical artifact.

### SCENARIO-INFRA-6197-4: Exp6183 and Exp6196 remain nonterminal
GIVEN `results/experiment_6183_transition_v536.json` and
`results/experiment_6196_v536_capstone_reconciliation.json` both have
`status: running_bootstrap`
WHEN they are classified with simulated completion receipts
THEN both classifications remain nonterminal bootstrap states.

### SCENARIO-INFRA-6197-5: Exp6197 writes one terminal contract artifact
GIVEN the shared classifier module and immutable fixture classifications
WHEN `python -m carnot.experiment_6197_v537_terminal_artifact_contract --date
20260807` runs
THEN it atomically writes the required result artifact with all required fields,
bare zero mutation/override counts, focused and full-suite command receipts,
`inference_substrate`, `verifier_is_oracle`, field principles, a stable
reproducibility checksum, and a terminal-prefixed `honest_verdict`.

## Implementation Status (REQ-INFRA-6197)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6197 | Shared classifier: `python/carnot/terminal_artifacts.py`; artifact writer: `python/carnot/experiment_6197_v537_terminal_artifact_contract.py`. | Focused tests: `tests/python/test_terminal_artifact_contract_6197.py` and `tests/python/test_experiment_6197_v537_terminal_artifact_contract.py`. |

## REQ-INFRA-6198: V537 Post-Marker Source Delta And Scope Audit SHALL Be Dated, Null-Safe, And Roadmap-Mechanical

Carnot SHALL build Exp6198 as a deterministic source-delta and staged-scope
audit for the V537 roadmap. The audit SHALL hash the exact
`<!-- V537-PLANNER-REFRESH-20260807-END -->` marker, reject every candidate
without reproducible evidence strictly after the marker timestamp, and treat
planning-time WybeCoder/RepoZero findings as sealed planner context rather
than runtime deltas.

Exp6198 SHALL audit `research-roadmap-next.yaml` with `Roadmap`,
`scripts/exclusion_manifest_lint.py`, and `scripts/audit_roadmap_gates.py`.
When the staged next-roadmap file is absent, it SHALL record the fallback and
audit `research-roadmap.yaml` without inventing staged content. The audit SHALL
mechanically verify all fourteen task prompts, IDs, deliverables,
dependencies, prior-failure records, model routing, structured gates,
allocation rules, the single GateMate continuity task, the single ARC slot,
and at least one prospective continuous self-learning task.

The Exp6198 artifact SHALL be written atomically to
`results/experiment_6198_v537_post_marker_source_scope_audit.json`. Its
`inference_substrate` SHALL be exactly
`post_marker_source_ingestion_and_roadmap_scope_audit`, `verifier_is_oracle`
SHALL be false, and accepted source deltas SHALL append to
`research-references.md` only after the V537 marker. If `accepted_count` is 0,
`research-references.md` SHALL remain byte-identical and `honest_verdict`
SHALL start with `complete_null:`.

The Exp6198 artifact SHALL include these required fields: `status`,
`planner_marker_and_hash`, `query_window`, `source_channel_receipts`,
`discovered_candidates`, `accepted_findings`,
`rejected_or_duplicate_findings`, `accepted_count`,
`references_append_receipt`, `roadmap_path_and_hash`,
`roadmap_schema_result`, `exclusion_manifest_lint_result`,
`retired_scope_match_count`, `prior_failure_contract_result`,
`gate_structure_result`, `model_specs_rule_result`, `task_count`,
`infra_slot_count`, `phase_d_slot_count`, `arc_slot_count`,
`continuous_self_learning_slot_count`, `hardware_continuity_result`,
`prompt_section_and_ending_result`, `protected_files_unchanged`,
`inference_substrate`, `verifier_is_oracle`, `field_provenance`,
`field_principles`, `test_commands`, `test_exit_codes`, `duration_s`,
`reproducibility_checksum`, and `honest_verdict`.

### SCENARIO-INFRA-6198-1: Marker Bounds Runtime Evidence
GIVEN the sealed V537 planner refresh marker in `research-references.md`
WHEN Exp6198 ingests external source receipts
THEN it records the marker text, byte/hash receipt, exclusive timestamp window,
and rejects every candidate dated at or before the marker.

### SCENARIO-INFRA-6198-2: Zero Source Delta Preserves References
GIVEN every discovered source is pre-marker, secondary-only, duplicate,
self-referential, endpoint-failed, or missing a strict post-marker timestamp
WHEN Exp6198 writes its artifact
THEN `accepted_count` is 0, `accepted_findings` is empty,
`research-references.md` is byte-identical before and after, and
`honest_verdict` starts with `complete_null:`.

### SCENARIO-INFRA-6198-3: Accepted Findings Require Strict Date And Scope Safety
GIVEN a candidate has a source date equal to the marker timestamp, a bare
same-day date, a duplicate ID or content hash, a retired-scope conflict, or no
new method/gate applicability
WHEN Exp6198 classifies candidates
THEN each candidate is rejected or guarded before any references append. Only a
primary or first-party candidate dated strictly after the marker with no
retirement conflict may be accepted.

### SCENARIO-INFRA-6198-4: Roadmap Scope Audit Is Mechanical
GIVEN the V537 roadmap has fourteen tasks
WHEN Exp6198 audits schema, exclusions, prior failures, gates, prompts,
deliverables, dependencies, allocation, hardware continuity, and model rules
THEN it reports schema/lint/gate success, at least two infrastructure tasks,
six Phase-D tasks, exactly one ARC task, one GateMate task, and at least one
prospective continuous self-learning task.

### SCENARIO-INFRA-6198-5: Exp6198 Artifact Schema Is Machine-Checkable
GIVEN the source receipts, roadmap audit, field principles, command receipts,
and protected-file hashes
WHEN Exp6198 validates the artifact
THEN every required field is present, every required field has provenance and a
principle, `verifier_is_oracle=false`, the inference substrate is exactly
`post_marker_source_ingestion_and_roadmap_scope_audit`, and the checksum
matches the normalized payload.

## Implementation Status (REQ-INFRA-6198)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6198 | `python/carnot/experiment_6198_v537_post_marker_source_scope_audit.py`; terminal artifact `results/experiment_6198_v537_post_marker_source_scope_audit.json`. | `tests/python/test_experiment_6198_v537_post_marker_source_scope_audit.py`. |

## REQ-INFRA-6211: V538 Source Delta And ARC Causal Preregistration SHALL Be Strictly Post-Marker And Nonmutating

Carnot SHALL build Exp6211 as a deterministic V538 source-delta and ARC
causal-scope preregistration audit. The audit SHALL hash the exact
`<!-- V538-PLANNER-REFRESH-20260807-END -->` marker, search only a window
strictly after that marker, and reject planning-time ARCANA, Cost-Effective
Agent Harnesses, Hyper-SET, audited skill-graph, Extropic, Kona, KAN,
Hugging Face, GitHub, EBT, and ARM-EBM findings as runtime deltas unless a
new dated reproducible record changes a V538 method or gate.

Exp6211 SHALL record receipts for all named source channels even when a
channel is unavailable, returns zero records, or returns only duplicate or
pre-marker evidence. It SHALL deduplicate candidates against both discovered
rows and existing `research-references.md` text before any append. If
`accepted_count` is 0, `research-references.md` SHALL remain byte-identical
and `honest_verdict` SHALL start with `complete_null:`.

Exp6211 SHALL audit `research-roadmap.yaml` without changing it. It SHALL use
`Roadmap`, `scripts/exclusion_manifest_lint.py`, and
`scripts/audit_roadmap_gates.py`, then mechanically verify model routing,
prompt endings, prior-failure records, retired dependency absence, active
hardware boundaries, phase counts, the ARC task count, and the continuous
self-learning slot count. `retired_scope_match_count` SHALL be the bare
integer 0 when no hard retired-scope exposure is present.

Exp6211 SHALL freeze the ARC A/B causal contracts before live measurement.
The machine-readable contract SHALL define treatment activation, A/A noise
floor, matched budget, matched game/seed controls, per-game losses, forbidden
source/BFS/adapter/hidden-state/registry access, no solve claim, registry hash
nonmutation, and allowed outcome vocabulary. A missing or permissive causal
contract SHALL make the artifact invalid.

The Exp6211 artifact SHALL be written atomically to
`results/experiment_6211_v538_post_marker_source_scope_prereg.json`. Its
`inference_substrate` SHALL be exactly
`post_marker_source_ingestion_and_arc_causal_preregistration`,
`verifier_is_oracle` SHALL be false, and every required field SHALL have a
field-principle entry.

The Exp6211 artifact SHALL include these required fields: `status`,
`planner_marker_and_hash`, `query_window`, `source_channel_receipts`,
`discovered_candidates`, `accepted_findings`,
`rejected_or_duplicate_findings`, `accepted_count`,
`references_append_receipt`, `roadmap_path_and_hash`,
`roadmap_schema_result`, `exclusion_manifest_lint_result`,
`retired_scope_match_count`, `prior_failure_contract_result`,
`gate_structure_result`, `model_specs_rule_result`, `task_count`,
`phase_counts`, `arc_task_count`, `continuous_self_learning_slot_count`,
`hardware_boundary_result`, `arc_outcome_vocabulary`,
`matched_control_contract`, `no_solve_and_registry_nonmutation_contract`,
`protected_files_unchanged`, `inference_substrate`, `verifier_is_oracle`,
`field_provenance`, `field_principles`, `test_commands`,
`test_exit_codes`, `duration_s`, `reproducibility_checksum`, and
`honest_verdict`.

### SCENARIO-INFRA-6211-1: V538 Marker Bounds Runtime Evidence
GIVEN the sealed V538 planner refresh marker in `research-references.md`
WHEN Exp6211 ingests source receipts and candidates
THEN it records the marker text, marker hash, reference hash, exclusive start
timestamp, and rejects every candidate dated at or before the marker.

### SCENARIO-INFRA-6211-2: Duplicate Or Null Source Search Preserves References
GIVEN every discovered source is pre-marker, duplicate, secondary-only,
endpoint-failed, or already present in `research-references.md`
WHEN Exp6211 writes its artifact
THEN `accepted_count` is the bare integer 0, `accepted_findings` is empty,
`research-references.md` is byte-identical before and after, and every named
source channel still has a receipt.

### SCENARIO-INFRA-6211-3: Accepted Findings Require Strict Date, Novelty, And Scope Safety
GIVEN a candidate has a source date equal to the marker timestamp, a bare
same-day date, a duplicate ID or content hash, existing reference text, a
retired-scope conflict, or no method/gate applicability
WHEN Exp6211 classifies candidates
THEN it rejects or guards the candidate before any references append. Only a
primary or first-party candidate dated strictly after the V538 marker with no
retirement conflict and new V538 applicability may be accepted.

### SCENARIO-INFRA-6211-4: Roadmap Counts And Contracts Are Mechanical
GIVEN the V538 active roadmap has fourteen tasks
WHEN Exp6211 audits schema, exclusions, prior failures, gates, prompts, model
rules, phase counts, ARC tasks, CSL slots, and hardware boundaries
THEN it reports clean schema, gate, exclusion, model, and prior-failure
results, exact phase counts, four ARC A/B tasks, one continuous self-learning
slot, and no unauthorized hardware promotion.

### SCENARIO-INFRA-6211-5: ARC Causal Contracts Fail Closed
GIVEN the four operator-authorized ARC A/Bs have not run live measurement
WHEN Exp6211 validates the preregistration artifact
THEN it requires the frozen outcome vocabulary, treatment-fire gate, A/A noise
floor, matched game/seed/budget controls, per-game-loss reporting, no-solve
boundary, forbidden-access counts, and registry hash before/after
nonmutation contract.

### SCENARIO-INFRA-6211-6: Exp6211 Artifact Schema Is Machine-Checkable
GIVEN source receipts, roadmap audits, causal contracts, command receipts, and
protected-file hashes
WHEN Exp6211 validates the artifact
THEN every required field is present, every required field has provenance and
a principle, `verifier_is_oracle=false`, the inference substrate is exactly
`post_marker_source_ingestion_and_arc_causal_preregistration`, and the
checksum matches the normalized payload.

## Implementation Status (REQ-INFRA-6211)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6211 | `python/carnot/experiment_6211_v538_post_marker_source_scope_prereg.py`; terminal artifact `results/experiment_6211_v538_post_marker_source_scope_prereg.json`. | `tests/python/test_experiment_6211_v538_post_marker_source_scope_prereg.py`. |

## REQ-INFRA-6225: V538-To-V539 Terminal Transition SHALL Preserve Mixed Truth Before Forward Work

Carnot SHALL build Exp6225 as a deterministic terminal-boundary handoff from
milestone `2026.08.538` into milestone `2026.08.539`. The handoff SHALL read
the V538 capstone and operational retro through `scripts/summarize_artifact.py`
before it copies any conclusion. It SHALL classify every V538 task by its exact
declared deliverable path and the current terminal-artifact classifier. It
SHALL preserve blocked, skipped, partial, flagged, ready, missing, and complete
states without upgrading them from conductor receipts or nearby sidecar files.

Exp6225 SHALL validate the V539 roadmap without activating or editing it. The
validation SHALL require exactly fourteen task ids in `exp6225` through
`exp6238` order, no duplicate ids, no retired dependencies, valid deliverables,
well-formed structured gates, complete `prior_failures` records, valid prompt
endings, and protected-file rules. Every LLM task SHALL name at least one
mandated SOTA GGUF. Every ARC task SHALL carry the live-agent provenance
contract.

Exp6225 SHALL record that `research-complete.yaml` contains duplicate
milestone records as an input caveat. It SHALL not deduplicate or rewrite that
file. Exp6225 SHALL record that `_bmad/architecture.md` is stale under the
repository 30-day rule. It SHALL not reconcile or rewrite architecture in this
transition task.

The Exp6225 artifact SHALL be written atomically to
`results/experiment_6225_v539_terminal_transition.json`. Its
`inference_substrate` SHALL be exactly
`deterministic_v538_v539_terminal_transition_audit`, `verifier_is_oracle`
SHALL be false, and `retired_dependency_count` and `id_collision_count` SHALL
be bare integer `0` for a passing handoff.

The Exp6225 artifact SHALL include these required fields: `status`,
`v538_milestone_and_roadmap_hash`, `v538_task_terminal_matrix`,
`v538_capstone_path_hash_and_summary`,
`operational_retro_path_hash_and_summary`,
`blocked_skipped_partial_flagged_and_ready_counts`,
`research_complete_duplicate_record_note`, `v539_roadmap_path_and_hash`,
`v539_task_ids_and_deliverables`, `task_count`, `phase_counts`,
`dependency_validation`, `gated_on_validation`, `prior_failure_validation`,
`retired_dependency_count`, `id_collision_count`, `model_policy_validation`,
`prompt_contract_validation`, `architecture_staleness_receipt`,
`protected_files_unchanged`, `preconditions_checked`, `inference_substrate`,
`verifier_is_oracle`, `field_provenance`, `field_principles`,
`test_commands`, `test_exit_codes`, `duration_s`,
`reproducibility_checksum`, and `honest_verdict`.

### SCENARIO-INFRA-6225-1: V538 Exact Deliverables Classify Fail Closed
GIVEN the V538 completed milestone record, capstone, and exact deliverable
paths
WHEN Exp6225 builds the V538 terminal matrix
THEN every V538 task from Exp6211 through Exp6224 is classified from its exact
path, missing artifacts remain missing, sidecars are ignored, and current-rule
flag counts stay visible.

### SCENARIO-INFRA-6225-2: Missing Artifacts Do Not Become Terminal
GIVEN a declared V538 deliverable path that does not exist
WHEN the terminal matrix classifier runs
THEN the row reports `present=false`, `terminal=false`, and
`terminal_class=missing`.

### SCENARIO-INFRA-6225-3: V539 Identity And Dependency Audit Is Exact
GIVEN the V539 roadmap
WHEN Exp6225 validates ids, deliverables, dependencies, gates, prior failures,
and retired-dependency references
THEN the task id list is exactly Exp6225 through Exp6238 in order,
`id_collision_count=0`, `retired_dependency_count=0`, and each failure list is
empty.

### SCENARIO-INFRA-6225-4: LLM And ARC Prompt Contracts Are Mechanical
GIVEN V539 tasks that use local LLMs or ARC live-agent work
WHEN Exp6225 audits prompt text
THEN each LLM task names a mandated GGUF, each ARC task declares
`solve_provenance must be live_agent_self_discovery`, and every prompt ends
with the required run command and conductor-protection sentence.

### SCENARIO-INFRA-6225-5: Protected Files Stay Byte-Identical
GIVEN before-state hashes for the active roadmap, conductor, ops ledgers,
traceability, architecture, and historical inputs
WHEN Exp6225 writes its own result artifact
THEN those protected paths have identical before and after hashes. Only the
Exp6225 spec, implementation, tests, and result artifact may change.

### SCENARIO-INFRA-6225-6: Artifact Schema Is Principle Annotated
GIVEN the transition report
WHEN Exp6225 validates the payload before writing
THEN every required field is present, every required field has field
provenance and a field-principle entry, the checksum matches the normalized
payload, and `honest_verdict` starts with a terminal prefix.

## Implementation Status (REQ-INFRA-6225)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6225 | `python/carnot/experiment_6225_v539_terminal_transition.py`; terminal artifact `results/experiment_6225_v539_terminal_transition.json`. | `tests/python/test_experiment_6225_v539_terminal_transition.py`. |

## REQ-INFRA-6226: V539 Post-Marker Source Scope Freeze SHALL Be Strict, Null-Safe, And Contract-Complete

Carnot SHALL build Exp6226 as a deterministic V539 source-delta and scope
freeze audit. The audit SHALL hash the exact
`<!-- V539-PLANNER-REFRESH-20260809-END -->` marker and SHALL search only
source evidence dated strictly after that marker. Evidence dated at or before
the marker, same-day evidence without a time after the marker, duplicates,
secondary-only rows, endpoint failures, and retired-scope conflicts SHALL NOT
append to `research-references.md`.

Exp6226 SHALL record receipts for arXiv, OpenReview, Extropic, Semantic
Scholar EBT citations, Semantic Scholar ARM-EBM citations, Hugging Face
Papers, targeted GitHub discovery, and Logical Intelligence. Each receipt
SHALL record authority, source role, query, URL, access outcome, date evidence,
candidate ids, and receipt hash even when the channel returns null evidence or
is unavailable.

Exp6226 SHALL atomically write a minimal bootstrap artifact before optional
network work. The final artifact SHALL preserve a receipt for that bootstrap
write and SHALL hash the marker, staged roadmap, protected files, and source
inputs in `preconditions_checked`. The final artifact SHALL be written to
`results/experiment_6226_v539_post_marker_source_scope_freeze.json`.

Exp6226 SHALL freeze six machine-readable contracts before later V539 tasks
run. The frozen runtime contract SHALL require task-owned process provenance
and bounded wait ownership. The ARC provenance contract SHALL require
`solve_provenance=live_agent_self_discovery` and zero hidden-game source,
offline BFS, adapter truth, hidden state, or registry trajectory access. The
code content-margin contract SHALL separate parse, compile, public-test, and
hidden-test content margins. The continual self-learning contract SHALL accept
only fresh events with post-outcome verifier-approved commits, frozen model
weights, rollback, and provenance. The sampler contract SHALL require treatment
activation before outcome interpretation. The hardware boundary SHALL forbid
Extropic, TSU, Z1, Kona, GateMate, KV260, PolarFire, power, energy, latency, or
speedup claims without a new authenticated route or physical-state receipt.

Exp6226 SHALL audit `research-roadmap.yaml` and `research-roadmap-next.yaml`
without changing either file. It SHALL run roadmap schema, exclusion, prior
failure, gate, model, retired-dependency, and prompt-ending checks. A null
`accepted_count` SHALL be valid and SHALL NOT block later tasks when no
accepted finding exists and the freeze contracts validate. A repository-wide
full-suite command SHALL remain recorded. It MAY be nonblocking for Exp6226
only when focused Exp6226 tests, new-code coverage, roadmap, exclusion, prior
failure, gate, model, retired-dependency, prompt-ending, and adversarial checks
pass and the broad failure is classified as unrelated existing suite failure.

The Exp6226 artifact SHALL include these required fields: `status`,
`planner_marker_and_hash`, `bootstrap_artifact_write_receipt`, `query_window`,
`source_channel_receipts`, `discovered_candidates`, `accepted_findings`,
`rejected_or_duplicate_findings`, `accepted_count`,
`references_append_receipt`, `frozen_runtime_contract`,
`frozen_arc_provenance_contract`, `frozen_code_content_margin_contract`,
`frozen_csl_contract`, `frozen_sampler_activation_contract`,
`frozen_hardware_boundary`, `roadmap_path_and_hash`,
`roadmap_schema_result`, `exclusion_manifest_lint_result`,
`prior_failure_contract_result`, `protected_files_unchanged`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`field_provenance`, `field_principles`, `test_commands`,
`test_exit_codes`, `duration_s`, `reproducibility_checksum`, and
`honest_verdict`.

### SCENARIO-INFRA-6226-1: V539 Marker Bounds Runtime Evidence
GIVEN the sealed V539 planner refresh marker in `research-references.md`
WHEN Exp6226 ingests source receipts and candidates
THEN it records the marker text, marker hash, reference hash, and exclusive
start timestamp, and rejects every candidate dated at or before the marker.

### SCENARIO-INFRA-6226-2: Bootstrap Artifact Survives Optional Network Work
GIVEN Exp6226 starts before source endpoints are queried
WHEN it writes the final artifact
THEN the final artifact preserves the bootstrap path, hash, status, and
precondition hashes from the minimal atomic write.

### SCENARIO-INFRA-6226-3: Null Or Duplicate Source Search Preserves References
GIVEN every discovered source is pre-marker, duplicate, secondary-only,
endpoint-failed, already present, or missing a strict post-marker timestamp
WHEN Exp6226 writes its artifact
THEN `accepted_count` may be `null` or `0`, `accepted_findings` is empty,
`research-references.md` is byte-identical before and after, and every named
source channel still has a receipt.

### SCENARIO-INFRA-6226-4: Accepted Findings Require Strict Date, Novelty, And Scope Safety
GIVEN a candidate has a source date equal to the marker timestamp, a bare
same-day date, a duplicate ID or content hash, existing reference text, a
retired-scope conflict, or no method/gate applicability
WHEN Exp6226 classifies candidates
THEN it rejects or guards the candidate before any references append. Only a
primary or first-party candidate dated strictly after the V539 marker with no
retirement conflict and new V539 applicability may be accepted.

### SCENARIO-INFRA-6226-5: Frozen Contracts Fail Closed
GIVEN later V539 tasks depend on Exp6226 contracts
WHEN Exp6226 validates the artifact
THEN missing provenance, inactive treatment, merged parse/content margins,
decision-time memory writes, mutable model weights, solve claims, or hardware
claims without authenticated receipts make the artifact invalid.

### SCENARIO-INFRA-6226-6: Artifact Schema Is Principle Annotated
GIVEN source receipts, roadmap audits, freeze contracts, command receipts, and
protected-file hashes
WHEN Exp6226 validates the artifact
THEN every required field is present, every required field has field
provenance and a principle, `verifier_is_oracle=false`, the inference substrate
is exactly `post_marker_source_ingestion_and_v539_scope_freeze`, and the
checksum matches the normalized payload. A nonzero repository-wide full-suite
receipt does not invalidate the source freeze when it is classified as an
unrelated existing suite failure.

## Implementation Status (REQ-INFRA-6226)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6226 | `python/carnot/experiment_6226_v539_post_marker_source_scope_freeze.py`; terminal artifact `results/experiment_6226_v539_post_marker_source_scope_freeze.json`. | `tests/python/test_experiment_6226_v539_post_marker_source_scope_freeze.py`. |

## REQ-INFRA-6227: Llama-Server Signal Sender Diagnostic SHALL Be Bounded, Owned, And Honest

Carnot SHALL build Exp6227 as a runtime diagnostic for the Gemma-4-31B native
`llama-server` reaper recurrence and caller-hang failure mode. The diagnostic
SHALL resolve the exact cached
`unsloth/gemma-4-31B-it-GGUF` `Q4_K_M` GGUF using the local resolver, SHALL use
the GGUF-embedded chat template, and SHALL run the native llama.cpp server. No
legacy model, Hugging Face tokenizer side channel, or substitute GGUF may be
used.

Exp6227 SHALL reconstruct the two persisted long-run incidents from available
logs before launching a new server. The reconstruction SHALL separate server
death evidence, connection failure evidence, retry evidence, and caller wait or
hang evidence. Absence of privileged sender evidence SHALL be recorded as an
honest unlocalized taxonomy rather than inferred.

Exp6227 SHALL run one short server lifecycle under a task-owned process with a
hard outer deadline. Before launch, during health/token calls, and after
cleanup it SHALL capture GPU owners, process ancestry, process identity,
session, process group, cgroup, stderr, exit status, health and token timing,
caller wait state, signal-tool availability, and sender evidence when the host
exposes it. It SHALL use auditd, eBPF, bpftrace, strace, or equivalent tracing
only when already available and permitted without privilege escalation, and
SHALL record unavailable or denied tools explicitly.

Exp6227 SHALL specify and test a finite wait, retry, and owned-process cleanup
contract. Cleanup SHALL target only the recorded PID/start-time identity, SHALL
refuse PID reuse and unrelated owners, and SHALL never kill unrelated
processes. The contract SHALL classify connection refusal, dead-server retry,
deadline expiry, timeout, externally signaled exits, and unknown sender cases
exactly. A controlled owned-child death test SHALL prove bounded wait and
owned-only cleanup. The diagnostic SHALL NOT add speculative checkpointing and
SHALL NOT mutate GGUF files.

The Exp6227 artifact SHALL be written to
`results/experiment_6227_llama_server_signal_sender_diagnostic.json` and SHALL
include these required fields: `status`,
`prior_incident_paths_hashes_and_timeline`, `preconditions_checked`,
`gpu_owner_receipts_before_during_after`, `model_specs`,
`exact_gguf_and_llama_cpp_receipts`, `owned_process_tree_snapshots`,
`launch_command_session_and_cgroup_receipts`, `health_and_token_timeline`,
`signal_trace_tool_availability`, `signal_events_and_sender_receipts`,
`server_exit_and_stderr_receipts`,
`caller_thread_and_wait_state_receipts`, `bounded_reproduction_deadline`,
`root_cause_taxonomy`, `sender_identified_score`, `unlocalized_reason`,
`finite_wait_retry_cleanup_contract`, `controlled_owned_child_death_test`,
`unrelated_process_kill_count`, `gguf_mutation_count`,
`runtime_diagnostic_ready_score`, `protected_files_unchanged`,
`inference_substrate`, `verifier_is_oracle`, `field_provenance`,
`field_principles`, `test_commands`, `test_exit_codes`, `duration_s`,
`reproducibility_checksum`, and `honest_verdict`. The
`unrelated_process_kill_count` and `gguf_mutation_count` fields SHALL be bare
integer zeroes. `sender_identified_score=0` MAY still yield
`runtime_diagnostic_ready_score=1` when unlocalized evidence and the finite
contract are complete.

### SCENARIO-INFRA-6227-1: Prior Incidents Are Reconstructed Without Collapsing Failure Phases
GIVEN persisted Exp6212, expanded-roster run logs, and server logs
WHEN Exp6227 builds the prior-incident timeline
THEN it records path hashes and separates server death, connection failure,
retry, caller wait or hang, and timeout evidence instead of reporting a single
ambiguous crash bucket.

### SCENARIO-INFRA-6227-2: Owned Gemma Server Lifecycle Captures Runtime Evidence
GIVEN no disallowed GPU owner and the exact cached Gemma-4-31B Q4_K_M GGUF
WHEN Exp6227 launches one native llama.cpp server under its own process
identity
THEN it records preconditions, exact GGUF and llama.cpp receipts, process tree,
session, process group, cgroup, GPU owners, health, token, stderr, exit status,
caller wait state, and every bounded timeout.

### SCENARIO-INFRA-6227-3: Sender Evidence Is Honest When Privileges Are Insufficient
GIVEN signal tracing tools are absent, denied, or do not expose sender PID
WHEN a server exits from a signal or prior logs show interrupt handling
THEN Exp6227 records tool availability and denial receipts, sets
`sender_identified_score=0`, and explains the unlocalized reason without
guessing the sender.

### SCENARIO-INFRA-6227-4: Finite Retry And Cleanup Refuse Unowned Or Reused Processes
GIVEN connection refusal, dead server retry, deadline expiry, PID reuse, and
unrelated-owner cases
WHEN Exp6227 applies the recovery contract
THEN every wait is bounded, retries are finite, cleanup targets only the
recorded PID/start-time identity, and unrelated-process kill count remains the
bare integer `0`.

### SCENARIO-INFRA-6227-5: Artifact Schema Is Principle Annotated And Mutates No GGUF
GIVEN the runtime diagnostic artifact
WHEN Exp6227 validates it
THEN every required field is present, every required field has provenance and a
principle, `verifier_is_oracle=false`, the inference substrate is exactly
`native_llama_cpp_owned_signal_sender_diagnostic`, the checksum matches the
normalized payload, protected files are unchanged, and `gguf_mutation_count` is
the bare integer `0`.

## Implementation Status (REQ-INFRA-6227)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6227 | `python/carnot/experiment_6227_llama_server_signal_sender_diagnostic.py`; terminal artifact `results/experiment_6227_llama_server_signal_sender_diagnostic.json`. | `tests/python/test_experiment_6227_llama_server_signal_sender_diagnostic.py`. |

## REQ-INFRA-6228: Three-Family Native Llama-Server Supervisor SHALL Prove Endurance

Carnot SHALL provide Exp6228 as the first reusable native `llama-server`
supervisor outside the conductor. The supervisor SHALL own each launched server
by PID, process start time, command hash, process group, and parent process
identity. Cleanup SHALL refuse PID reuse, command drift, parent drift, and
unrelated owners. All health waits, token waits, retry counts, endurance
windows, and cleanup waits SHALL be bounded.

Exp6228 SHALL qualify exactly these cached GGUF families with preferred
quantization `Q4_K_M`: `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. It SHALL use exact cached files, cache
revisions, embedded GGUF chat templates, and the native CUDA llama.cpp server
build. It SHALL never call `AutoTokenizer` on a GGUF repository. It SHALL not
modify the conductor or the GGUF cache.

Before model load, Exp6228 SHALL enumerate GPU owners, free VRAM, exact GGUF
files, native build identity, output writability, reserved ports, and bounded
windows. It SHALL persist a `preconditions_checked` receipt before any server
loads a model. Unsafe GPU ownership or missing files SHALL block the run
without starting or killing any server.

For each family, Exp6228 SHALL launch a task-owned native `llama-server`, parse
CUDA layer or tensor placement from the actual server log, and confirm owned
VRAM placement from GPU interval samples. It SHALL collect repeated
deterministic raw output bytes across an endurance interval. It SHALL inject
one controlled owned-child death, prove bounded cleanup, restart the owned
server within the retry budget, collect a recovery token, and finish with a
process and VRAM leak-free teardown.

Each readiness score SHALL be conjunctive and principle annotated.
`qwen_runtime_ready_score`, `gemma_4_31b_runtime_ready_score`, and
`gemma_4_26b_runtime_ready_score` SHALL be `1` only when that family has
owned-process receipts, log-parsed CUDA evidence, GPU interval CUDA evidence,
repeated deterministic raw tokens, endurance samples, controlled owned-child
failure, successful recovery, and leak-free cleanup. The dense score SHALL be
independent. `two_family_runtime_ready_score` SHALL require at least two ready
families. `three_family_runtime_ready_score` SHALL require all three ready
families.

The Exp6228 artifact SHALL be written to
`results/experiment_6228_supervised_three_family_runtime_endurance.json` and
SHALL include these required fields: `status`,
`upstream_diagnostic_path_and_hash`, `preconditions_checked`,
`supervisor_contract_and_paths_hashes`, `model_specs`,
`exact_gguf_paths_sizes_hashes_revisions_quantizations`,
`embedded_chat_template_receipts`, `llama_cpp_build_and_cuda_receipts`,
`gpu_owner_intervals_by_family`,
`server_command_pid_starttime_process_group_and_lifetime_by_family`,
`parsed_cuda_layer_or_tensor_placement_by_family`,
`repeated_raw_token_hashes_and_latencies_by_family`,
`endurance_window_and_health_samples_by_family`,
`controlled_owned_child_failure_and_recovery_by_family`,
`retry_and_wait_bounds`, `final_process_and_vram_leak_check`,
`qwen_runtime_ready_score`, `gemma_4_31b_runtime_ready_score`,
`gemma_4_26b_runtime_ready_score`, `two_family_runtime_ready_score`,
`three_family_runtime_ready_score`, `unrelated_process_kill_count`,
`gguf_mutation_count`, `protected_files_unchanged`, `inference_substrate`,
`verifier_is_oracle`, `field_provenance`, `field_principles`,
`test_commands`, `test_exit_codes`, `duration_s`, `reproducibility_checksum`,
and `honest_verdict`. The `unrelated_process_kill_count` and
`gguf_mutation_count` fields SHALL be bare integer zeroes.
The `inference_substrate` field SHALL be exactly
`local_three_family_native_llama_server_supervised_cuda_endurance`.

### SCENARIO-INFRA-6228-DEAD-PORT-SLOW-LOAD-AND-EARLY-EXIT-ARE-BOUNDED
GIVEN a dead port, slow model load, early server exit, and retry exhaustion
WHEN the supervisor waits for health and applies its retry budget
THEN every wait returns a classified finite receipt and the retry count never
exceeds the contract.

### SCENARIO-INFRA-6228-OWNERSHIP-REFUSES-PID-REUSE-AND-UNRELATED-OWNERS
GIVEN a recorded server PID identity, a PID reuse case, an unrelated owner,
and a cleanup timeout
WHEN cleanup runs
THEN it signals only the matching owned PID or process group, refuses identity
or owner drift, records leak status, and keeps `unrelated_process_kill_count`
as bare integer `0`.

### SCENARIO-INFRA-6228-CUDA-READINESS-USES-LOGS-AND-GPU-INTERVALS
GIVEN server flags, server logs, and GPU owner interval samples
WHEN Exp6228 parses CUDA placement
THEN readiness requires actual log evidence and owned GPU interval evidence.
Flags alone SHALL NOT make a family ready.

### SCENARIO-INFRA-6228-ENDURANCE-AND-RECOVERY-QUALIFY-EACH-FAMILY
GIVEN all three exact cached GGUFs and a native CUDA server
WHEN Exp6228 qualifies each family
THEN each family records repeated deterministic raw token hashes, endurance
health samples, one controlled owned-child failure, successful bounded
recovery, and final leak-free teardown.

### SCENARIO-INFRA-6228-ARTIFACT-SCORES-ARE-CONJUNCTIVE
GIVEN a terminal Exp6228 artifact
WHEN validation recomputes readiness
THEN dense readiness is separate, two-family readiness needs at least two
ready families, three-family readiness needs all three, every field has
provenance and a principle, `verifier_is_oracle=false`, protected files are
unchanged, the checksum matches, and GGUF mutation count is the bare integer
`0`.

## Implementation Status (REQ-INFRA-6228)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6228 | `python/carnot/inference/llama_server_supervisor.py`; `python/carnot/experiment_6228_supervised_three_family_runtime_endurance.py`; terminal artifact `results/experiment_6228_supervised_three_family_runtime_endurance.json`. | `tests/python/test_experiment_6228_supervised_three_family_runtime_endurance.py`; `tests/python/test_llama_server_supervisor.py`. |

## REQ-INFRA-6262: Declared Terminal Artifacts SHALL Be Readiness-Gated By Exact Artifact State

Carnot SHALL add a fail-closed readiness boundary outside
`scripts/research_conductor.py` for declared experiment artifacts. The boundary
SHALL reuse `python/carnot/terminal_artifacts.py` as the only verdict
vocabulary. It SHALL NOT trust conductor receipt text, completion logs, or
roadmap intent over the exact declared artifact path.

The adversarial verifier SHALL emit one named CRITICAL finding for a declared
experiment artifact whose exact path is missing, malformed, running,
bootstrap-only, partial, contradictory, or unknown. Missing paths and malformed
JSON SHALL keep distinct details. Honest `blocked` artifacts and gate-skipped
terminal artifacts SHALL remain terminal controls.

Generic artifact sweeps SHALL auto-enroll only experiment artifacts that carry
the readiness contract marker `status`. Exact declared paths that are missing,
malformed, or not loadable SHALL still be checkable by an explicit verifier
handoff. This keeps legacy terminal-verdict-only fixtures out of the new
readiness contract while preserving fail-closed checks for declared task
deliverables.

Gate-field eligibility SHALL require both conditions:

- The exact artifact path classifies as terminal.
- The exact top-level field exists as a bare field on that artifact.

Principle-wrapped values, nested values, conductor receipt fields, similar field
names, and absent fields SHALL NOT be gate eligible. A receipt SHALL record an
override attempt, but it SHALL NOT make a nonterminal artifact terminal.

Exp6262 SHALL write
`results/experiment_6262_terminal_artifact_readiness_contract.json`. The
artifact SHALL include these required fields: `status`,
`exp6228_path_hash_and_exact_classification`,
`classifier_source_hash_before_after`,
`adversarial_verifier_source_hash_before_after`,
`supported_terminal_classes`, `rejected_nonterminal_classes`,
`exact_path_over_receipt_precedence`, `gate_field_eligibility_contract`,
`exp6228_regression_flag_code_and_severity`,
`honest_blocked_control_result`, `gate_skip_control_result`,
`receipt_override_negative_control`, `readiness_missing_negative_control`,
`false_positive_fixture_results`, `focused_test_results`,
`qa_layer_audit_results`, `protected_files_unchanged`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`field_provenance`, `field_principles`, `test_commands`, `test_exit_codes`,
`terminal_artifact_contract_ready_score`, `duration_s`,
`reproducibility_checksum`, and `honest_verdict`.

The artifact SHALL declare
`inference_substrate=deterministic_terminal_artifact_readiness_replay_no_model`.

`terminal_artifact_contract_ready_score` SHALL be the bare integer `1` only
when every required control passes and every focused command exits zero.

### SCENARIO-INFRA-6262-1: Exp6228 Preconditions Receipt Is Critical
GIVEN `results/experiment_6228_supervised_three_family_runtime_endurance.json`
contains a valid precondition receipt without terminal `status` and
`honest_verdict`
WHEN adversarial verification checks the declared artifact boundary
THEN it emits `NONTERMINAL_DECLARED_ARTIFACT` with severity `critical`
AND the detail records the exact path classification as `unknown`.

### SCENARIO-INFRA-6262-2: Nonterminal Classes Fail Closed
GIVEN declared artifact fixtures that are running, bootstrap-only, partial,
contradictory, missing, malformed, or unknown
WHEN the readiness boundary classifies each fixture
THEN each fixture receives a CRITICAL adversarial finding
AND missing and malformed details remain distinct.

### SCENARIO-INFRA-6262-3: Receipts Cannot Override Exact Paths
GIVEN a nonterminal or missing declared artifact
AND a conductor receipt that says `OK`
WHEN the readiness boundary classifies the exact path
THEN the result records the receipt override attempt
AND the terminal class remains nonterminal.

### SCENARIO-INFRA-6262-4: Gates Need Terminal Bare Fields
GIVEN an artifact with a top-level field, a nested field, a
principle-wrapped field, and a receipt field
WHEN gate-field eligibility is evaluated
THEN only a terminal artifact with an exact bare top-level field is eligible.

### SCENARIO-INFRA-6262-5: Honest Terminal Controls Stay Clean
GIVEN complete, null, honest blocked, and gate-skipped terminal artifacts
WHEN the adversarial verifier checks them
THEN no `NONTERMINAL_DECLARED_ARTIFACT` finding is emitted.

### SCENARIO-INFRA-6262-6: Exp6262 Writes The Readiness Contract Artifact
GIVEN the focused tests, source hashes, Exp6228 regression classification, QA
checks, and protected-file hashes
WHEN `python -m carnot.experiment_6262_terminal_artifact_readiness_contract
--date 20260810` runs
THEN the result artifact contains all required fields, a matching checksum, a
bare readiness score of `1`, and a terminal-prefixed honest verdict.

## REQ-INFRA-6238: V539 Exact-Path Capstone SHALL Preserve Terminal States And Reject Unsupported Claims

Carnot SHALL build Exp6238 as the V539 branch-independent capstone. The
capstone SHALL load the active V539 roadmap, derive the exact task id to
declared deliverable matrix for Exp6225 through Exp6238, and SHALL NOT
substitute same-number sidecars, conductor receipts, or corrected historical
artifacts for a missing exact path.

Exp6238 SHALL classify every exact deliverable with the shared terminal
artifact classifier. The classifier result SHALL outrank conductor receipts.
Missing, malformed, bootstrap-only, unknown, partial, blocked, skipped,
flagged, null, retired, and ready outcomes SHALL remain distinct in the
artifact. A skipped branch SHALL be terminal operational evidence and SHALL
contribute no scientific success.

Exp6238 SHALL read every present upstream artifact through
`scripts/summarize_artifact.py`, SHALL run current `scripts/adversarial_verify.py`
logic, and SHALL record a `scripts/determination_preservation_lint.py --all`
receipt. A live critical adversarial flag, a stamped flag, a failed structured
gate, a missing required upstream field, a missing exact artifact, or an
inactive treatment SHALL exclude the affected branch from any positive claim.

Exp6238 SHALL recompute every structured roadmap gate from the exact upstream
artifact field. Gate evaluation SHALL record upstream task, field, operator,
expected value, actual value, pass/fail state, conductor receipt comparison,
and a principle. A missing upstream artifact or missing upstream field SHALL
fail closed.

Exp6238 SHALL reconcile runtime durability, ARC depth, executable-code
format/content, fresh-event continuous learning, shadow reachability, sampler
activation/equivalence/default state, and hardware status independently. It
SHALL keep runtime canaries separate from durable runtime readiness, ARC
registry depth separate from live depth promotion, parse recovery separate from
hidden-test content margin, deterministic memory replay separate from fresh CSL,
shadow reachability separate from CSL promotion, and treatment activation
separate from a null sampler result. Hardware claim count SHALL be the bare
integer `0` unless a new independently admissible hardware receipt exists in
the declared V539 task set.

The Exp6238 artifact SHALL be written atomically to
`results/experiment_6238_v539_adversarial_capstone.json` with
`inference_substrate=aggregation_from_upstream_artifacts` and
`verifier_is_oracle=false`. The artifact SHALL include these required fields:
`status`, `roadmap_path_hash_and_task_ids`, `exact_task_artifact_matrix`,
`conductor_receipt_matrix`, `terminal_classifier_results`,
`adversarial_results_by_task`, `determination_preservation_results`,
`gate_cascade_receipts`,
`missing_blocked_skipped_partial_null_flagged_retired_and_ready_counts`,
`prior_failure_retirement_actions`, `runtime_final_status_and_family_scores`,
`arc_provenance_registry_hash_level_depth_and_promotion_summary`,
`code_parse_recovery_and_content_margin_summary`,
`fresh_stream_and_continuous_learning_summary`, `shadow_consumer_summary`,
`sampler_activation_quality_equivalence_and_default_summary`,
`hardware_boundary_and_claim_count`, `protected_files_unchanged`,
`spec_traceability_status_changelog_known_issues_updates`,
`research_complete_reconciliation`, `preconditions_checked`,
`inference_substrate`, `verifier_is_oracle`, `field_provenance`,
`field_principles`, `test_commands`, `test_exit_codes`, `duration_s`,
`reproducibility_checksum`, and `honest_verdict`.

### SCENARIO-INFRA-6238-1: Exact Declared Paths Refuse Substitutes
GIVEN the V539 roadmap declares Exp6225 through Exp6238 deliverable paths
WHEN Exp6238 builds its artifact matrix
THEN every task row records the exact declared path, hash, existence state,
terminal class, and ignored same-number aliases. Missing exact paths stay
missing.

### SCENARIO-INFRA-6238-2: Current Adversarial And Preservation Checks Gate Claims
GIVEN present V539 artifacts
WHEN Exp6238 runs `summarize_artifact.py`, `adversarial_verify.py`, and
`determination_preservation_lint.py --all`
THEN live critical flags and stamped flags are recorded per task, and no flagged
artifact contributes to branch success or a hardware claim.

### SCENARIO-INFRA-6238-3: Structured Gates Fail Closed From Exact Fields
GIVEN roadmap tasks with `gated_on` entries
WHEN the upstream artifact is missing or the upstream field is absent, null, or
not equal to the expected value
THEN the gate row fails, records the exact actual value, and preserves any
downstream gate-block as terminal operational evidence only.

### SCENARIO-INFRA-6238-4: Branch Summaries Stay Independent
GIVEN runtime, ARC, code, CSL, shadow, sampler, and hardware branches
WHEN Exp6238 reconciles V539
THEN each branch reports only its own admissible upstream fields. Runtime
readiness does not promote ARC, parse recovery does not become content gain,
fresh-stream absence blocks CSL, sampler activation is separate from its
equivalence/null decision, and hardware claim count remains the bare integer
`0` without a new receipt.

### SCENARIO-INFRA-6238-5: Retire-If-Same-Verdict Is Recorded Without Guessing
GIVEN a V539 task carries `prior_failures` with `retire_if_same_verdict=true`
WHEN the exact current artifact does not reproduce the same failed verdict
scope, or the exact artifact is missing
THEN Exp6238 records no exclusion-manifest update. It records a candidate only
when the exact same failed verdict recurs.

### SCENARIO-INFRA-6238-6: Artifact Schema Is Principle Annotated
GIVEN the capstone report
WHEN Exp6238 validates the payload before writing
THEN every required field has provenance and a field-principle entry, count and
gate records carry principles, `verifier_is_oracle=false`, hardware claim count
is the bare integer `0`, the checksum matches the normalized payload, and
`honest_verdict` starts with a terminal prefix.

## Implementation Status (REQ-INFRA-6238)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6238 | `python/carnot/experiment_6238_v539_adversarial_capstone.py`; terminal artifact `results/experiment_6238_v539_adversarial_capstone.json`. | `tests/python/test_experiment_6238_v539_adversarial_capstone.py`. |

## REQ-INFRA-6260: V540 Terminal Transition SHALL Preserve V539 Exact-Path States

Carnot SHALL build Exp6260 as the V539-to-V540 handoff. It SHALL classify each
V539 task from the exact declared deliverable path recorded by the V539
capstone. It SHALL NOT promote a task from conductor status, same-number
aliases, sidecars, or roadmap presence.

Exp6260 SHALL use `python/carnot/terminal_artifacts.py` for all artifact
classification. Missing, malformed, bootstrap-only, running, partial, and
unknown artifacts SHALL stay nonterminal. Exp6228 SHALL remain nonterminal when
its exact artifact contains only preconditions, `status=preconditions_recorded`,
null readiness fields, and no terminal `honest_verdict`.

Exp6260 SHALL validate the V540 roadmap without activating a next roadmap or
editing the active roadmap. The task set SHALL contain exactly Exp6260 through
Exp6271 in order. Every task SHALL declare a `results/*.json` deliverable,
valid dependencies, valid structured gates, complete `prior_failures`, and
`agent_type=codex` with `model=gpt-5.5`. Gates SHALL reference fields named in
the upstream prompt's `REQUIRED ARTIFACT FIELDS` block. Retired dependencies
SHALL be rejected.

Exp6260 SHALL scan tracked and untracked experiment paths before writing its
artifact. The scan SHALL prove that concurrent Exp6240 and Exp6244 through
Exp6246 files do not collide with the reserved Exp6260 through Exp6271 task
range. It SHALL record protected-file hashes before and after the artifact
write.

The Exp6260 artifact SHALL be written atomically to
`results/experiment_6260_v540_terminal_transition.json` with
`inference_substrate=deterministic_v539_v540_terminal_transition_audit` and
`verifier_is_oracle=false`. The artifact SHALL include these required fields:
`status`, `v539_milestone_roadmap_and_hash`, `v539_task_terminal_matrix`,
`exp6228_nonterminal_classification`, `v539_capstone_path_hash_and_summary`,
`operational_retro_path_hash_and_summary`,
`missing_nonterminal_blocked_skipped_null_flagged_retired_and_ready_counts`,
`concurrent_exp6240_6244_6245_6246_collision_receipts`,
`v540_roadmap_path_and_hash`, `v540_task_ids_and_deliverables`, `task_count`,
`phase_counts`, `dependency_validation`, `gated_on_validation`,
`prior_failure_validation`, `retired_dependency_count`, `id_collision_count`,
`model_policy_validation`, `prompt_contract_validation`,
`protected_files_unchanged`, `preconditions_checked`, `inference_substrate`,
`verifier_is_oracle`, `field_provenance`, `field_principles`, `test_commands`,
`test_exit_codes`, `duration_s`, `reproducibility_checksum`, and
`honest_verdict`. The `task_count` SHALL be 12. The
`retired_dependency_count` and `id_collision_count` SHALL be bare integer `0`.

### SCENARIO-INFRA-6260-1: Exact Declared Path Outranks Aliases
GIVEN a V539 task declares one result artifact path
WHEN another same-number artifact exists
THEN Exp6260 classifies only the declared path and records ignored aliases.

### SCENARIO-INFRA-6260-2: Preconditions-Only Artifacts Stay Nonterminal
GIVEN Exp6228 has `status=preconditions_recorded` and no terminal
`honest_verdict`
WHEN Exp6260 classifies the exact Exp6228 artifact
THEN the classification is nonterminal and no conductor receipt can promote it.

### SCENARIO-INFRA-6260-3: Reserved Id Collision Scan Is Fail-Closed
GIVEN files may exist for concurrent Exp6240 and Exp6244 through Exp6246 work
WHEN Exp6260 scans tracked and untracked experiment paths
THEN those concurrent ids are recorded separately and any unexpected Exp6260
through Exp6271 pre-existing path is a collision.

### SCENARIO-INFRA-6260-4: V540 Roadmap Contracts Are Mechanical
GIVEN the V540 roadmap tasks
WHEN Exp6260 validates task ids, deliverables, dependencies, gates, priors,
agent routing, model routing, and prompt endings
THEN exactly 12 tasks in Exp6260 through Exp6271 order pass, with no retired
dependency and no duplicate id.

### SCENARIO-INFRA-6260-5: Protected Hashes Prove Non-Mutation
GIVEN protected files are hashed before artifact generation
WHEN Exp6260 writes its result
THEN each protected path records before and after hashes and the result fails
closed if any protected file changed.

### SCENARIO-INFRA-6260-6: Artifact Schema Is Principle Annotated
GIVEN the transition report is built
WHEN Exp6260 validates it before writing
THEN every required field has provenance and a field-principle entry,
`verifier_is_oracle=false`, the checksum matches, and `honest_verdict` starts
with a terminal prefix.

## Implementation Status (REQ-INFRA-6260)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6260 | `python/carnot/experiment_6260_v540_terminal_transition.py`; terminal artifact `results/experiment_6260_v540_terminal_transition.json`. | `tests/python/test_experiment_6260_v540_terminal_transition.py`. |

## REQ-INFRA-6261: V540 Post-Marker Source Scope Freeze SHALL Be Strictly Later, Null-Safe, And Contract-Complete

Carnot SHALL build Exp6261 as a deterministic V540 post-marker source sweep
and scope freeze. The sweep SHALL hash the exact
`<!-- V540-PLANNER-REFRESH-20260809-END -->` marker and SHALL accept only
stable, reproducible evidence dated strictly after the marker commit time.
Evidence dated at the marker time, before the marker time, or with only a bare
same-day date SHALL be rejected before any reference append.

Exp6261 SHALL record receipts for arXiv, OpenReview, Extropic, Semantic
Scholar EBT and ARM-EBM citation routes, Hugging Face Papers, targeted GitHub,
and Logical Intelligence. It SHALL deduplicate candidates against
`research-references.md` and prior candidate identifiers before append. If the
accepted count is zero, `research-references.md` SHALL remain byte-identical,
`accepted_count` SHALL be the bare integer `0`, and `honest_verdict` SHALL
start with `complete_null:`.

Exp6261 SHALL freeze machine-readable contracts for terminal artifacts, clean
cached-SOTA replay, energy familiarity, chronological continuous self-learning,
mode-jump generality, model provenance, and the hardware boundary. The frozen
contracts SHALL state that cached replay is not on-policy proof. They SHALL
also state that no current board or Extropic TSU route supports Carnot
execution, speed, power, or availability claims.

The Exp6261 artifact SHALL be written atomically to
`results/experiment_6261_v540_post_marker_source_scope_freeze.json` with
`inference_substrate=post_marker_source_ingestion_and_v540_scope_freeze` and
`verifier_is_oracle=false`. It SHALL audit the V540 roadmap without editing
`research-roadmap.yaml`.

The Exp6261 artifact SHALL include these required fields: `status`,
`planner_marker_and_hash`, `query_window`, `source_channel_receipts`,
`discovered_candidates`, `accepted_findings`,
`rejected_duplicate_or_watch_only_findings`, `accepted_count`,
`references_append_receipt`, `frozen_terminal_artifact_contract`,
`frozen_cached_sota_replay_contract`, `frozen_energy_familiarity_contract`,
`frozen_chronological_csl_contract`, `frozen_sampler_generality_contract`,
`frozen_model_provenance_contract`, `frozen_hardware_boundary`,
`roadmap_path_and_hash`, `roadmap_schema_result`,
`exclusion_manifest_lint_result`, `prior_failure_contract_result`,
`protected_files_unchanged`, `preconditions_checked`, `inference_substrate`,
`verifier_is_oracle`, `field_provenance`, `field_principles`,
`test_commands`, `test_exit_codes`, `duration_s`,
`reproducibility_checksum`, and `honest_verdict`.

### SCENARIO-INFRA-6261-1: Marker Bound Is Exclusive
GIVEN the sealed V540 planner refresh marker in `research-references.md`
WHEN Exp6261 classifies a candidate at or before the marker commit time
THEN the candidate is rejected, and a bare same-day date is rejected unless a
strictly later timestamp is present.

### SCENARIO-INFRA-6261-2: Null Search Preserves References
GIVEN every source channel returns no strictly later stable evidence, duplicate
evidence, pre-marker evidence, or watch-only evidence
WHEN Exp6261 writes its artifact
THEN `accepted_count` is the bare integer `0`, `accepted_findings` is empty,
`research-references.md` is byte-identical, and `honest_verdict` starts with
`complete_null:`.

### SCENARIO-INFRA-6261-3: Stable URLs And Duplicates Fail Closed
GIVEN a candidate has no stable URL, repeats an earlier source id, repeats an
existing reference block, or lacks a scope-changing reproducible dependency
WHEN Exp6261 deduplicates the candidate ledger
THEN the row is recorded only in
`rejected_duplicate_or_watch_only_findings`.

### SCENARIO-INFRA-6261-4: Frozen Contracts Preserve V540 Boundaries
GIVEN the V540 roadmap narrows onto terminal artifacts, cached replay, energy
familiarity, chronological CSL, and sampler generality
WHEN Exp6261 serializes the frozen contracts
THEN each contract has a stable version, required boundary fields, and explicit
claim limits for cached replay, model provenance, and hardware execution.

### SCENARIO-INFRA-6261-5: Artifact Schema Is Principle Annotated
GIVEN source receipts, roadmap receipts, protected hashes, field principles,
and command receipts
WHEN Exp6261 validates the report before writing
THEN every required field is present, every field has provenance and a
principle, `verifier_is_oracle=false`, the checksum matches the normalized
payload, and the honest verdict has a terminal prefix.

## Implementation Status (REQ-INFRA-6261)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6261 | Implemented by `python/carnot/experiment_6261_v540_post_marker_source_scope_freeze.py`; terminal artifact `results/experiment_6261_v540_post_marker_source_scope_freeze.json`. | Covered by `tests/python/test_experiment_6261_v540_post_marker_source_scope_freeze.py`. |

## REQ-INFRA-6271: V540 Capstone SHALL Preserve Branch-Independent Exact-Path Evidence

Carnot SHALL build Exp6271 as the ungated capstone for milestone
`2026.08.540`. The capstone SHALL load the active roadmap, freeze the roadmap
path and hash, and use the exact declared deliverable path for each Exp6260
through Exp6271 task. It SHALL NOT replace a missing exact artifact with an
alias, sidecar, conductor receipt, or roadmap intent.

Exp6271 SHALL classify every declared deliverable with
`python/carnot/terminal_artifacts.py`. Conductor receipts SHALL be recorded as
context only. A conductor `OK`, `GATE_BLOCK`, or other terminal-looking receipt
SHALL NOT override missing, malformed, running, bootstrap-only, partial,
contradictory, or unknown artifact state. A gate tombstone SHALL stay skipped.

Exp6271 SHALL rerun the current adversarial rules for every present artifact.
It SHALL preserve stamped artifact flags separately from current-rule flags.
Any missing artifact, nonterminal artifact, current critical flag, failed
structured gate, or absent exact readiness field SHALL block promotion for the
affected branch. The terminal-artifact, continuous-learning, and sampler
branches SHALL have separate ledgers. One branch SHALL NOT promote another.

Exp6271 SHALL recompute structured gates from exact bare upstream fields with
the shared gate-field eligibility semantics. Missing upstream artifacts,
missing fields, principle-wrapped fields, and nonterminal upstream artifacts
SHALL fail closed. The capstone SHALL record these failures without
synthesizing default values.

The Exp6271 artifact SHALL be written atomically to
`results/experiment_6271_v540_adversarial_capstone.json` with
`inference_substrate=aggregation_from_upstream_artifacts` and
`verifier_is_oracle=false`. It SHALL make no live-LLM, runtime, ARC solve,
hardware, speed, power, registry, source-mutation, or model-weight claim. Each
of those claim or mutation counters SHALL be the bare integer `0`.

The Exp6271 artifact SHALL include these required fields: `status`,
`milestone_roadmap_path_and_hash`, `exact_declared_deliverable_matrix`,
`conductor_receipt_matrix`, `exact_path_over_receipt_precedence`,
`current_rule_adversarial_results_by_task`,
`terminal_nonterminal_blocked_skipped_null_flagged_retired_and_ready_counts`,
`gate_cascade_receipts`, `terminal_artifact_contract_state`,
`clean_sota_replay_state`, `familiarity_gate_state`,
`continuous_learning_state`, `family_task_transfer_state`,
`shadow_consumer_state`, `sampler_fixture_state`,
`mode_jump_safety_and_value_state`, `sampler_router_state`,
`branch_independent_promotion_ledger`, `prior_failure_retirement_actions`,
`source_mutation_count`, `weight_mutation_count`, `live_llm_call_count`,
`arc_solve_claim_count`, `registry_update_count`, `hardware_claim_count`,
`speed_or_power_claim_count`, `protected_files_unchanged`,
`spec_traceability_status_changelog_reconciliation`, `prd_gap_table`,
`next_milestone_recommendations`, `preconditions_checked`,
`inference_substrate`, `verifier_is_oracle`, `field_provenance`,
`field_principles`, `test_commands`, `test_exit_codes`, `duration_s`,
`reproducibility_checksum`, and `honest_verdict`.

### SCENARIO-INFRA-6271-1: Exact Paths Outrank Receipts

GIVEN a declared deliverable path is missing or nonterminal
AND the conductor log contains a terminal-looking receipt for that task
WHEN Exp6271 builds its deliverable matrix
THEN the matrix keeps the exact-path classifier result
AND records that the receipt attempted no promotion.

### SCENARIO-INFRA-6271-2: Gate Cascades Read Exact Bare Fields

GIVEN a downstream task declares a `gated_on` upstream field
WHEN the upstream artifact is missing, nonterminal, lacks the exact field, or
wraps the field in a principle object
THEN Exp6271 records the gate as failed
AND it does not synthesize a default value.

### SCENARIO-INFRA-6271-3: Current Flags Do Not Hide Stamped Flags

GIVEN a present artifact carries a stamped adversarial flag
AND the current verifier returns a separate set of flags
WHEN Exp6271 records adversarial results
THEN the stamped and current-rule flag states stay distinct
AND any current critical flag blocks branch promotion.

### SCENARIO-INFRA-6271-4: Branch Ledgers Are Independent

GIVEN the continuous-learning branch has a closed familiarity gate
AND the sampler branch has a ready fixture suite but no workload value
WHEN Exp6271 computes promotion ledgers
THEN each branch reports its own ready, blocked, skipped, null, or missing
state
AND neither branch may launder the other into promotion.

### SCENARIO-INFRA-6271-5: Forbidden Claim Counters Are Bare Zero

GIVEN the capstone is an aggregation over checked-in evidence
WHEN Exp6271 validates its report
THEN source mutation, weight mutation, live-LLM call, ARC solve, registry
update, hardware, speed, and power counters are all bare `0`
AND the normalized checksum matches the payload.

## Implementation Status (REQ-INFRA-6271)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6271 | Implemented by `python/carnot/experiment_6271_v540_adversarial_capstone.py`; terminal artifact `results/experiment_6271_v540_adversarial_capstone.json`. | Covered by `tests/python/test_experiment_6271_v540_adversarial_capstone.py`. |

## REQ-INFRA-6272: V541 Transition SHALL Preserve V540 Exact Evidence And Validate The Reserved Roadmap

Carnot SHALL build Exp6272 as the exact handoff from terminal milestone
`2026.08.540` into `2026.08.541`. It SHALL classify every V540 task from the
declared deliverable path recorded by the V540 capstone or archived milestone
ledger. A conductor receipt SHALL be evidence only. It SHALL NOT promote a
missing, nonterminal, blocked, skipped, null, flagged, retired, or ready state
into another class.

Exp6272 SHALL preserve focused task checks and broad-suite checks as separate
receipts. A broad-suite failure SHALL NOT erase a focused pass, and a focused
pass SHALL NOT hide the broad-suite failure. Gate skips, missing files, null
results, blocked results, ready artifacts, and unsupported-backend findings
SHALL remain separate rows and separate counts.

Exp6272 SHALL validate the V541 roadmap without activating a staged roadmap
and without editing `research-roadmap.yaml`. It SHALL verify exactly twelve
task ids in Exp6272 through Exp6283 order, their deliverable paths, phases,
dependencies, structured gates, prior-failure blocks, task-specific routing,
model policy, prompt run-command contracts, prompt endings, protected files,
and tracked plus untracked reserved-id collisions. Structured gates SHALL be
checked against the upstream task's `REQUIRED ARTIFACT FIELDS` block. Any
retired dependency or incomplete `prior_failures` entry SHALL fail closed.

The Exp6272 artifact SHALL be written atomically to
`results/experiment_6272_v541_terminal_transition.json` with
`inference_substrate=aggregation_from_upstream_artifacts` and
`verifier_is_oracle=false`. `task_count` SHALL equal `12`.
`retired_dependency_count` and `id_collision_count` SHALL be bare integer `0`.
The artifact SHALL include these required fields: `status`,
`v540_milestone_roadmap_and_hash`, `v540_task_terminal_matrix`,
`v540_capstone_path_hash_and_summary`,
`operational_retro_path_hash_and_summary`,
`focused_and_broad_validation_receipts_by_task`,
`missing_nonterminal_blocked_skipped_null_flagged_retired_and_ready_counts`,
`v541_roadmap_path_and_hash`, `v541_task_ids_and_deliverables`, `task_count`,
`phase_counts`, `dependency_validation`, `gated_on_validation`,
`prior_failure_validation`, `retired_dependency_count`, `id_collision_count`,
`agent_routing_validation`, `model_policy_validation`,
`prompt_contract_validation`, `protected_files_unchanged`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`field_provenance`, `field_principles`, `test_commands`, `test_exit_codes`,
`duration_s`, `reproducibility_checksum`, and `honest_verdict`.

### SCENARIO-INFRA-6272-1: V540 Exact Paths Outrank Receipts

GIVEN the V540 capstone or archive names declared deliverables
AND the conductor ledger contains terminal-looking receipts
WHEN Exp6272 builds the V540 terminal matrix
THEN it reclassifies each exact path with `python/carnot/terminal_artifacts.py`
AND records receipt override attempts without changing the exact-path class.

### SCENARIO-INFRA-6272-2: Focused And Broad Validation Receipts Stay Separate

GIVEN a task has focused commands and a broad-suite command
WHEN Exp6272 summarizes validation evidence
THEN focused passes, broad-suite failures, timeouts, and missing receipts are
reported separately by task.

### SCENARIO-INFRA-6272-3: V541 Roadmap Shape And Collisions Fail Closed

GIVEN the V541 roadmap reserves Exp6272 through Exp6283
WHEN Exp6272 validates ids and deliverables
THEN exactly twelve tasks in order are required
AND tracked plus untracked collisions outside the task-owned Exp6272 files
produce a nonzero collision count.

### SCENARIO-INFRA-6272-4: Gates And Prior Failures Are Structured

GIVEN a V541 task declares `requires`, `gated_on`, or `prior_failures`
WHEN Exp6272 validates the roadmap
THEN every dependency must exist and not be retired
AND every gate must reference an upstream required artifact field
AND every prior-failure entry must include experiment id, verdict,
addressed-by text, and `retire_if_same_verdict: true`.

### SCENARIO-INFRA-6272-5: Routing And Prompt Contracts Match V541 Policy

GIVEN V541 uses Codex GPT-5.5 for formulaic code tasks and Opus for declared
judgment or cross-file tasks
WHEN Exp6272 validates task routing
THEN each task's `agent_type` and `model` match the direct V541 policy
AND every prompt includes its matching run command and protected-file ending.

### SCENARIO-INFRA-6272-6: Artifact Is Checksummed And Non-Mutating

GIVEN precondition hashes, roadmap checks, command receipts, and field
principles are assembled
WHEN Exp6272 writes the handoff artifact
THEN every required field has provenance and a principle
AND the checksum matches the normalized payload
AND protected files remain byte-identical except the task-owned result.

## Implementation Status (REQ-INFRA-6272)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6272 | Pending implementation: `python/carnot/experiment_6272_v541_terminal_transition.py`; terminal artifact `results/experiment_6272_v541_terminal_transition.json`. | Pending focused tests: `tests/python/test_experiment_6272_v541_terminal_transition.py`. |

## REQ-INFRA-6273: V541 Post-Marker Source Scope Freeze SHALL Be Strict, Null-Safe, And Contract-Complete

Carnot SHALL build Exp6273 as a deterministic V541 post-marker source sweep
and scope freeze. The sweep SHALL hash the exact
`<!-- V541-PLANNER-REFRESH-20260810-END -->` marker and SHALL accept only
stable, reproducible evidence dated strictly after the marker commit time.
Evidence at the marker time, before the marker time, or with only a bare same
day date SHALL be rejected before any reference append.

Exp6273 SHALL record receipts for arXiv, OpenReview, Extropic, Semantic
Scholar EBT and ARM-EBM citation routes, Hugging Face Papers, targeted GitHub,
and Logical Intelligence. It SHALL deduplicate candidates against all earlier
`research-references.md` blocks and prior candidate identifiers. A zero-source
delta SHALL be terminal. In that case `research-references.md` SHALL remain
byte-identical, `accepted_count` SHALL be the bare integer `0`, and
`honest_verdict` SHALL start with `complete_null:`.

Exp6273 SHALL freeze machine-readable contracts for ASP semantics, the
flagship GGUF benchmark, certified cache admission, chronological
self-learning, variable-cardinality sampling, ARC live provenance, and the
hardware boundary. The ASP contract SHALL state that an exact ASP solver is an
oracle. The hardware boundary SHALL state that no current board or TSU route
supports Carnot execution, speed, power, energy-efficiency, or availability
claims.

The Exp6273 artifact SHALL be written atomically to
`results/experiment_6273_v541_post_marker_source_scope_freeze.json` with
`inference_substrate=post_marker_source_ingestion_and_v541_scope_freeze` and
`verifier_is_oracle=false`. It SHALL audit the V541 roadmap without editing
`research-roadmap.yaml`.

The Exp6273 artifact SHALL include these required fields: `status`,
`planner_marker_and_hash`, `query_window`, `source_channel_receipts`,
`discovered_candidates`, `accepted_findings`,
`rejected_duplicate_or_watch_only_findings`, `accepted_count`,
`references_append_receipt`, `frozen_asp_semantics_contract`,
`frozen_flagship_benchmark_contract`, `frozen_certified_cache_contract`,
`frozen_chronological_csl_contract`,
`frozen_variable_cardinality_sampler_contract`,
`frozen_arc_live_provenance_contract`, `frozen_hardware_boundary`,
`roadmap_path_and_hash`, `roadmap_schema_result`,
`exclusion_manifest_lint_result`, `prior_failure_contract_result`,
`protected_files_unchanged`, `preconditions_checked`, `inference_substrate`,
`verifier_is_oracle`, `field_provenance`, `field_principles`,
`test_commands`, `test_exit_codes`, `duration_s`,
`reproducibility_checksum`, and `honest_verdict`.

### SCENARIO-INFRA-6273-1: Marker Bound Is Exclusive

GIVEN the sealed V541 planner refresh marker in `research-references.md`
WHEN Exp6273 classifies a candidate at or before the marker commit time
THEN the candidate is rejected, and a bare same-day date is rejected unless a
strictly later timestamp is present.

### SCENARIO-INFRA-6273-2: Zero Findings Preserve References

GIVEN every source channel returns no strictly later stable evidence, duplicate
evidence, pre-marker evidence, or watch-only evidence
WHEN Exp6273 writes its artifact
THEN `accepted_count` is the bare integer `0`, `accepted_findings` is empty,
`research-references.md` is byte-identical, and `honest_verdict` starts with
`complete_null:`.

### SCENARIO-INFRA-6273-3: Stable URLs And Duplicates Fail Closed

GIVEN a candidate has no stable URL, repeats an earlier source id, repeats an
existing reference block, or lacks a scope-changing reproducible dependency
WHEN Exp6273 deduplicates the candidate ledger
THEN the row is recorded only in
`rejected_duplicate_or_watch_only_findings`.

### SCENARIO-INFRA-6273-4: Frozen Contracts Preserve V541 Boundaries

GIVEN the V541 roadmap depends on ASP semantics, flagship GGUF verification,
certified cache admission, chronological CSL, variable-cardinality sampling,
ARC live provenance, and no hardware claim
WHEN Exp6273 serializes the frozen contracts
THEN each contract has a stable version, required boundary fields, an ASP
oracle statement, immutable GGUF weights, live ARC provenance, and explicit
claim limits for boards and TSU routes.

### SCENARIO-INFRA-6273-5: Artifact Schema Is Principle Annotated

GIVEN source receipts, roadmap receipts, protected hashes, field principles,
and command receipts
WHEN Exp6273 validates the report before writing
THEN every required field is present, every field has provenance and a
principle, `verifier_is_oracle=false`, the checksum matches the normalized
payload, and the honest verdict has a terminal prefix.

## Implementation Status (REQ-INFRA-6273)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6273 | Pending implementation: `python/carnot/experiment_6273_v541_post_marker_source_scope_freeze.py`; terminal artifact `results/experiment_6273_v541_post_marker_source_scope_freeze.json`. | Pending focused tests: `tests/python/test_experiment_6273_v541_post_marker_source_scope_freeze.py`. |

## REQ-INFRA-6283: V541 Capstone SHALL Preserve Exact Paths And Branch-Independent Evidence

Carnot SHALL build Exp6283 as an ungated adversarial capstone for milestone
`2026.08.541`. It SHALL load the active roadmap, the activated milestone
document, conductor receipts, the ARC solve registry, the current adversarial
rules, and the exact declared deliverables for Exp6272 through Exp6282. It SHALL
classify each task from its declared path. Conductor receipts SHALL be recorded
as evidence only. A conductor OK, FLAGGED, GATE_BLOCK, or skip receipt SHALL NOT
promote a missing, nonterminal, blocked, skipped, null, flagged, retired, or
ready exact artifact.

Exp6283 SHALL re-run the current adversarial verifier on every present declared
artifact and keep stamped artifact flags separate from current-rule flags. It
SHALL recompute every structured gate from exact bare upstream fields by using
the terminal artifact classifier. Wrapped, missing, nonterminal, or absent
fields SHALL fail closed without defaults. A gate tombstone SHALL remain
skipped.

Exp6283 SHALL publish independent ledgers for ASP verification, continuous
self-learning, variable-cardinality sampling, and ARC live-path mechanic
routing. A positive or ready field in one branch SHALL NOT promote another
branch. Missing, nonterminal, flagged, or gate-skipped artifacts SHALL never be
promoted. Prior-failure `retire_if_same_verdict` actions SHALL be applied by
exact current verdict comparison and recorded without changing the ARC registry
or making a solve claim.

The Exp6283 artifact SHALL be written atomically to
`results/experiment_6283_v541_adversarial_capstone.json` with
`inference_substrate=aggregation_from_upstream_artifacts` and
`verifier_is_oracle=false`. It SHALL include these required fields: `status`,
`milestone_roadmap_path_and_hash`, `exact_declared_deliverable_matrix`,
`conductor_receipt_matrix`, `exact_path_over_receipt_precedence`,
`current_rule_adversarial_results_by_task`,
`terminal_nonterminal_blocked_skipped_null_flagged_retired_and_ready_counts`,
`gate_cascade_receipts`, `asp_semantic_compiler_state`,
`flagship_asp_benchmark_state`, `certified_admission_state`,
`chronological_continuous_learning_state`, `heldout_transfer_state`,
`shadow_consumer_state`, `variable_cardinality_backend_state`,
`mode_jump_safety_and_value_state`, `arc_mechanic_router_state`,
`arc_provenance_and_registry_receipts`,
`branch_independent_promotion_ledger`, `prior_failure_retirement_actions`,
`publication_gate_g1_g2_g3_g4_and_unmet_gates`, `source_mutation_count`,
`weight_mutation_count`, `unauthorized_external_call_count`,
`hidden_game_source_access_count`, `outer_loop_ground_truth_search_count`,
`arc_level_solve_claim_count`, `registry_update_count`, `hardware_claim_count`,
`speed_power_or_energy_claim_count`, `protected_files_unchanged`,
`spec_traceability_status_changelog_reconciliation`, `prd_gap_table`,
`next_milestone_recommendations`, `preconditions_checked`,
`inference_substrate`, `verifier_is_oracle`, `field_provenance`,
`field_principles`, `test_commands`, `test_exit_codes`, `duration_s`,
`reproducibility_checksum`, and `honest_verdict`.

`source_mutation_count`, `weight_mutation_count`,
`unauthorized_external_call_count`, `hidden_game_source_access_count`,
`outer_loop_ground_truth_search_count`, `arc_level_solve_claim_count`,
`registry_update_count`, `hardware_claim_count`, and
`speed_power_or_energy_claim_count` SHALL be bare integer `0`. Every required
field SHALL have a one-line field principle and provenance entry. The
publication gate SHALL come from `scripts/publication_gate.py`.

### SCENARIO-INFRA-6283-1: Exact Paths Outrank Receipts

GIVEN V541 tasks have declared deliverable paths
AND conductor receipts contain OK, FLAGGED, GATE_BLOCK, or skipped rows
WHEN Exp6283 builds the declared deliverable matrix
THEN each row is classified from the exact JSON path
AND any receipt override attempt is recorded without changing the class.

### SCENARIO-INFRA-6283-2: Gates Read Only Terminal Bare Fields

GIVEN a V541 task declares a structured `gated_on` field
WHEN Exp6283 recomputes the gate
THEN the gate reads only the upstream exact artifact's bare field
AND missing, wrapped, nonterminal, or missing-artifact fields fail closed.

### SCENARIO-INFRA-6283-3: Current And Stamped Flags Stay Separate

GIVEN an artifact has a stamped `flagged_adversarial` or `corrigendum_pending`
field
WHEN Exp6283 re-runs the current adversarial verifier
THEN stamped flags and current-rule critical or warning flags are reported in
separate fields.

### SCENARIO-INFRA-6283-4: Branch Ledgers Cannot Launder Evidence

GIVEN ASP, self-learning, sampler, and ARC branches have different terminal
states
WHEN Exp6283 builds promotion ledgers
THEN each branch promotes only from its own exact terminal, unflagged, ready
evidence
AND missing, skipped, null, blocked, or flagged artifacts block only their own
branch.

### SCENARIO-INFRA-6283-5: Zero Claim Counters Are Bare Integers

GIVEN Exp6283 is an aggregation over checked-in evidence
WHEN the artifact is validated
THEN all mutation, unauthorized-call, hidden-source, outer-loop search, solve,
registry, hardware, speed, power, and energy counters are bare integer `0`
AND the normalized checksum matches the payload.

### SCENARIO-INFRA-6283-6: Publication Gate And Reconciliation Are Recorded

GIVEN the capstone has classified exact V541 evidence
WHEN Exp6283 runs publication-gate and reconciliation checks
THEN it records G1 through G4 and unmet gates from
`scripts/publication_gate.py`
AND it records OpenSpec, traceability, status, and changelog hashes without
editing operator-curated public documents.

## Implementation Status (REQ-INFRA-6283)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6283 | Pending implementation: `python/carnot/experiment_6283_v541_adversarial_capstone.py`; terminal artifact `results/experiment_6283_v541_adversarial_capstone.json`. | Pending focused tests: `tests/python/test_experiment_6283_v541_adversarial_capstone.py`. |

## REQ-INFRA-6284: V542 Transition SHALL Preserve V541 Exact Evidence And Validate The Reserved Roadmap

Carnot SHALL build Exp6284 as the exact V541-to-V542 handoff. It SHALL read the
V541 capstone, the V541 operational retro, the exact V541 declared artifacts,
the staged V542 roadmap, the exclusion manifest, and the roadmap validation
helpers. It SHALL classify every V541 task from the task's declared artifact
path under the current terminal-artifact rules. Conductor receipts and raw
evidence receipts SHALL be recorded separately. They SHALL NOT override the
exact artifact path classification or artifact-level eligibility.

Exp6284 SHALL validate exactly 13 V542 tasks in Exp6284 through Exp6296 order.
It SHALL validate task ids, deliverables, dependencies, structured gates,
prior-failure blocks, reserved file collisions, agent routing, model policy,
prompt run commands, prompt endings, protected files, and local SOTA GGUF
contracts. Structured gates SHALL reference fields that appear in the upstream
task's `REQUIRED ARTIFACT FIELDS` block. Dependencies SHALL NOT point to retired
experiment ids. Each non-empty `prior_failures` row SHALL include a prior id,
verdict, changed-mechanism explanation, and `retire_if_same_verdict=true`.

The Exp6284 artifact SHALL be written atomically to
`results/experiment_6284_v542_terminal_transition.json` with
`inference_substrate=aggregation_from_upstream_artifacts` and
`verifier_is_oracle=false`. It SHALL include these required fields: `status`,
`v541_milestone_roadmap_and_hash`, `v541_task_terminal_matrix`,
`v541_capstone_path_hash_and_summary`,
`operational_retro_path_hash_and_summary`,
`focused_and_broad_validation_receipts_by_task`,
`missing_nonterminal_blocked_skipped_null_flagged_retired_and_ready_counts`,
`v542_roadmap_path_and_hash`, `v542_task_ids_and_deliverables`, `task_count`,
`phase_counts`, `dependency_validation`, `gated_on_validation`,
`prior_failure_validation`, `retired_dependency_count`, `id_collision_count`,
`agent_routing_validation`, `model_policy_validation`,
`prompt_contract_validation`, `raw_evidence_eligibility_policy`,
`protected_files_unchanged`, `preconditions_checked`, `inference_substrate`,
`verifier_is_oracle`, `field_provenance`, `field_principles`,
`test_commands`, `test_exit_codes`, `duration_s`,
`reproducibility_checksum`, and `honest_verdict`.

`task_count` SHALL be the bare integer `13`. `retired_dependency_count` and
`id_collision_count` SHALL be bare integer `0`. Every required field SHALL have
a one-line field principle and provenance entry. The honest verdict SHALL start
with `complete:`, `complete_ready:`, `complete_null:`, `blocked:`,
`blocked_safety:`, or `skipped:`.

### SCENARIO-INFRA-6284-1: V541 Exact Paths Outrank Receipts

GIVEN the V541 capstone records declared deliverable paths and conductor
receipts
WHEN Exp6284 builds the V541 terminal matrix
THEN every row is classified from the exact JSON path
AND any terminal-looking receipt is recorded without changing the class.

### SCENARIO-INFRA-6284-2: Raw Evidence Cannot Promote Flagged Artifacts

GIVEN a V541 artifact has raw receipts but is flagged, null, blocked, skipped,
missing, retired, or nonterminal
WHEN Exp6284 records raw evidence eligibility
THEN raw receipts stay separate from artifact-level eligibility
AND only exact terminal unflagged artifact state can feed a roadmap gate.

### SCENARIO-INFRA-6284-3: V542 Roadmap Contracts Fail Closed

GIVEN the V542 roadmap declares tasks, deliverables, dependencies, gates, and
prior failures
WHEN Exp6284 validates the roadmap
THEN the task ids match Exp6284 through Exp6296 in order
AND bad gates, retired dependencies, duplicate ids, incomplete priors, bad
routes, bad model policy, and bad prompt endings are reported as failures.

### SCENARIO-INFRA-6284-4: SOTA GGUF Prompt Contracts Are Enforced

GIVEN a live LLM V542 task requires local GGUF model execution
WHEN Exp6284 validates the model policy
THEN the task prompt names mandated SOTA GGUF ids in `MODEL_SPECS`
AND legacy or missing headline model ids fail validation.

### SCENARIO-INFRA-6284-5: Artifact Schema Is Principle Annotated

GIVEN precondition hashes, protected hashes, command receipts, roadmap checks,
and V541 terminal classifications
WHEN Exp6284 validates the report before writing
THEN every required field is present, every field has provenance and a
principle, `task_count=13`, zero counts are bare integers, and the checksum
matches the normalized payload.

## Implementation Status (REQ-INFRA-6284)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6284 | Pending implementation: `python/carnot/experiment_6284_v542_terminal_transition.py`; terminal artifact `results/experiment_6284_v542_terminal_transition.json`. | Pending focused tests: `tests/python/test_experiment_6284_v542_terminal_transition.py`. |

## REQ-INFRA-6285: V542 Post-Marker Source Freeze SHALL Be Strict, Null-Safe, And Contract-Complete

Carnot SHALL build Exp6285 as a deterministic V542 post-marker source sweep
and scope freeze. The sweep SHALL hash the exact
`<!-- V542-PLANNER-REFRESH-20260810-END -->` marker and SHALL accept only
stable, reproducible evidence dated strictly after the marker commit time.
Evidence at the marker time, before the marker time, or with only a bare same
day date SHALL be rejected before any reference append.

Exp6285 SHALL record receipts for arXiv, OpenReview, Extropic, Semantic
Scholar EBT and ARM-EBM citation routes, Hugging Face Papers, targeted GitHub,
and Logical Intelligence. It SHALL deduplicate candidates against all earlier
`research-references.md` blocks and prior candidate identifiers. A zero-source
delta SHALL be terminal. In that case `research-references.md` SHALL remain
byte-identical, `accepted_count` SHALL be the bare integer `0`, and
`honest_verdict` SHALL start with `complete_null:`.

Exp6285 SHALL freeze machine-readable contracts for partial atom evidence,
exact-vertex continuous relaxation, flagship live refinement, revocable atomic
memory, crystallization, ARC causal validation, and the hardware boundary. The
contract set SHALL state that the exact ASP solver is an oracle for exact
validity. The hardware boundary SHALL state that no current board or TSU route
supports Carnot execution, speed, power, energy-efficiency, or availability
claims.

The Exp6285 artifact SHALL be written atomically to
`results/experiment_6285_v542_post_marker_source_scope_freeze.json` with
`inference_substrate=web_and_bibliographic_search_only` and
`verifier_is_oracle=false`. It SHALL audit the V542 roadmap without editing
`research-roadmap.yaml`.

The Exp6285 artifact SHALL include these required fields: `status`,
`planner_marker_and_hash`, `query_window`, `source_channel_receipts`,
`discovered_candidates`, `accepted_findings`,
`rejected_duplicate_or_watch_only_findings`, `accepted_count`,
`references_append_receipt`, `frozen_partial_atom_contract`,
`frozen_continuous_relaxation_contract`,
`frozen_flagship_refinement_contract`, `frozen_revocable_memory_contract`,
`frozen_crystallization_contract`, `frozen_arc_causal_contract`,
`frozen_hardware_boundary`, `roadmap_path_and_hash`, `roadmap_schema_result`,
`exclusion_manifest_lint_result`, `prior_failure_contract_result`,
`protected_files_unchanged`, `preconditions_checked`, `inference_substrate`,
`verifier_is_oracle`, `field_provenance`, `field_principles`,
`test_commands`, `test_exit_codes`, `duration_s`,
`reproducibility_checksum`, and `honest_verdict`.

### SCENARIO-INFRA-6285-1: Marker Bound Is Exclusive

GIVEN the sealed V542 planner refresh marker in `research-references.md`
WHEN Exp6285 classifies a candidate at or before the marker commit time
THEN the candidate is rejected, and a bare same-day date is rejected unless a
strictly later timestamp is present.

### SCENARIO-INFRA-6285-2: Zero Findings Preserve References

GIVEN every source channel returns no strictly later stable evidence, duplicate
evidence, pre-marker evidence, or watch-only evidence
WHEN Exp6285 writes its artifact
THEN `accepted_count` is the bare integer `0`, `accepted_findings` is empty,
`research-references.md` is byte-identical, and `honest_verdict` starts with
`complete_null:`.

### SCENARIO-INFRA-6285-3: Stable URLs And Duplicates Fail Closed

GIVEN a candidate has no stable URL, repeats an earlier source id, repeats an
existing reference block, or lacks a scope-changing reproducible dependency
WHEN Exp6285 deduplicates the candidate ledger
THEN the row is recorded only in
`rejected_duplicate_or_watch_only_findings`.

### SCENARIO-INFRA-6285-4: Frozen Contracts Preserve V542 Boundaries

GIVEN V542 depends on fail-closed partial atom evidence, exact-vertex
continuous relaxation, flagship live refinement, revocable atomic memory,
crystallization partitions, matched ARC causal validation, and no hardware
claim
WHEN Exp6285 serializes the frozen contracts
THEN each contract has a stable version, required boundary fields, an exact ASP
oracle statement, frozen model weights, live ARC provenance, and explicit claim
limits for boards and TSU routes.

### SCENARIO-INFRA-6285-5: Artifact Schema Is Principle Annotated

GIVEN source receipts, roadmap receipts, protected hashes, field principles,
and command receipts
WHEN Exp6285 validates the report before writing
THEN every required field is present, every field has provenance and a
principle, `verifier_is_oracle=false`, the checksum matches the normalized
payload, and the honest verdict has a terminal prefix.

## Implementation Status (REQ-INFRA-6285)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6285 | Pending implementation: `python/carnot/experiment_6285_v542_post_marker_source_scope_freeze.py`; terminal artifact `results/experiment_6285_v542_post_marker_source_scope_freeze.json`. | Pending focused tests: `tests/python/test_experiment_6285_v542_post_marker_source_scope_freeze.py`. |

## REQ-INFRA-6286: V541 Evidence Eligibility Ledger SHALL Validate Raw Receipts Without Laundering Flagged Artifacts

Carnot SHALL build Exp6286 as a deterministic V541 evidence eligibility
ledger for V542. The ledger SHALL read the V541 capstone, operational retro,
the Exp6274, Exp6275, Exp6276, Exp6280, Exp6281, and Exp6282 artifacts, the
Exp6275 event corpus, all Exp6275 raw output sidecars, the Exp6275 sealed
manifest, the Exp6275 formal sidecar, the current adversarial verifier, the
terminal artifact classifier, the anomaly escalation log, and the exclusion
manifest. It SHALL hash these inputs before deriving any eligibility row.

Exp6286 SHALL re-run current adversarial verification for each present V541
artifact and keep stamped flags separate from current-rule flags. A stamped or
current flagged artifact SHALL remain ineligible for artifact-level promotion.
Raw Exp6275 rows MAY be validated as receipts only. Raw-row validation SHALL
check model output, prompt, seed, token, sidecar, and outcome provenance. It
SHALL NOT rescore parse, semantic, or scientific outcomes. A raw row missing
any required provenance SHALL be quarantined. Eligible and quarantined row
counts SHALL sum to the exact Exp6275 event-corpus row count.

Exp6286 SHALL write immutable eligible-row and quarantine manifests for
Exp6275 rows. These manifests SHALL NOT edit the Exp6275 artifact, event
corpus, raw sidecars, sealed manifest, or formal sidecar. Valid raw rows SHALL
NOT reopen the Exp6275 artifact-level readiness, because the headline artifact
is stamped flagged. The ledger SHALL mark Exp6274 and Exp6280 source substrate
reusable. It SHALL mark unchanged Exp6276 and Exp6281 treatments ineligible
for V542 extension. It SHALL mark Exp6282 source reusable but the result
unpromoted.

The Exp6286 artifact SHALL be written atomically to
`results/experiment_6286_v541_evidence_eligibility_ledger.json` with
`inference_substrate=aggregation_from_upstream_artifacts` and
`verifier_is_oracle=false`. It SHALL include these required fields: `status`,
`v541_capstone_path_hash_and_terminal_class`,
`current_rule_adversarial_results_by_v541_task`, `asp_compiler_eligibility`,
`flagship_artifact_eligibility`,
`flagship_raw_manifest_paths_and_hashes`,
`flagship_raw_row_validation_rules`, `eligible_flagship_raw_row_count`,
`quarantined_flagship_raw_row_count`,
`flagship_raw_row_eligibility_manifest_path_and_hash`,
`dual_cache_treatment_eligibility`,
`global_threshold_control_eligibility`, `typed_backend_eligibility`,
`mode_jump_treatment_eligibility`, `arc_router_source_eligibility`,
`arc_result_eligibility`, `branch_stop_ledger`,
`no_claim_laundering_receipt`, `source_mutation_count`,
`protected_files_unchanged`, `preconditions_checked`,
`inference_substrate`, `verifier_is_oracle`, `field_provenance`,
`field_principles`, `test_commands`, `test_exit_codes`, `duration_s`,
`reproducibility_checksum`, and `honest_verdict`.

`source_mutation_count` SHALL be the bare integer `0`. Every required field
SHALL have one short field principle and a provenance entry. The honest verdict
SHALL start with a terminal prefix.

### SCENARIO-INFRA-6286-1: Claim Laundering Stays Blocked

GIVEN Exp6275 or Exp6282 is stamped flagged
WHEN Exp6286 validates source or raw receipts from that experiment
THEN the source receipts may be recorded
AND artifact-level promotion remains closed.

### SCENARIO-INFRA-6286-2: Raw Row Counts Are Conserved

GIVEN the Exp6275 event corpus has a fixed row count
WHEN Exp6286 validates raw-row provenance
THEN eligible and quarantined row counts sum to the exact source row count.

### SCENARIO-INFRA-6286-3: Missing Provenance Quarantines A Row

GIVEN an Exp6275 event row lacks model output, prompt, seed, token, sidecar, or
outcome provenance
WHEN Exp6286 validates the row by provenance only
THEN the row appears in the quarantine manifest with the missing fields named.

### SCENARIO-INFRA-6286-4: Hash Drift Fails Closed

GIVEN a source artifact, raw sidecar, event corpus, rule file, or protected
file hash changes after preconditions are frozen
WHEN Exp6286 validates the report
THEN the protected-file or manifest hash receipt records the drift and the
report is invalid for a clean verdict.

### SCENARIO-INFRA-6286-5: Source Mutation Count Is Bare Zero

GIVEN Exp6286 is an aggregation over checked-in evidence
WHEN the artifact is validated
THEN `source_mutation_count` is the bare integer `0`, required field
principles exist, and the normalized checksum matches the payload.

## Implementation Status (REQ-INFRA-6286)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6286 | Pending implementation: `python/carnot/experiment_6286_v541_evidence_eligibility_ledger.py`; terminal artifact `results/experiment_6286_v541_evidence_eligibility_ledger.json`. | Pending focused tests: `tests/python/test_experiment_6286_v541_evidence_eligibility_ledger.py`. |

## REQ-INFRA-6296: V542 Capstone SHALL Preserve Exact Branch Evidence Without Claim Laundering

Carnot SHALL build Exp6296 as an ungated branch-independent capstone for
milestone `2026.08.542`. The capstone SHALL load the active V542 roadmap,
the exact declared deliverables for Exp6284 through Exp6295, conductor
receipts, the ARC registry, the current adversarial verifier, the terminal
artifact classifier, and protected file hashes before deriving any branch
state. It SHALL write
`results/experiment_6296_v542_adversarial_capstone.json`.

Exp6296 SHALL classify each task by its exact declared artifact path. A
conductor OK receipt SHALL NOT override a missing, malformed, running,
partial, unknown, flagged, blocked, skipped, retired, null, or nonterminal
artifact. A gate tombstone SHALL remain skipped. Same-number aliases and
sidecars SHALL be recorded only as ignored aliases.

Exp6296 SHALL re-run current adversarial verification for every present
upstream artifact. It SHALL preserve stamped artifact flags separately from
current-rule flags. Missing artifacts and nonterminal artifacts SHALL NOT feed
branch promotion. Structured gates SHALL be recomputed from exact bare fields
only. Missing fields SHALL stay missing. Principle-wrapped fields SHALL NOT
be treated as gate inputs.

Exp6296 SHALL publish separate promotion ledgers for exact-state energy
refinement, revocable continuous self-learning, and ARC mechanic-route causal
validation. Oracle repair alone SHALL NOT promote the energy-refinement
branch. Replay-only gains SHALL NOT promote continuous learning. A no-solve
ARC canary SHALL NOT become an ARC solve claim. Prior failures with
`retire_if_same_verdict=true` SHALL be evaluated by exact verdict string and
recorded without editing the exclusion manifest.

The Exp6296 artifact SHALL include these required fields: `status`,
`milestone_roadmap_path_and_hash`, `exact_declared_deliverable_matrix`,
`conductor_receipt_matrix`, `exact_path_over_receipt_precedence`,
`current_rule_adversarial_results_by_task`,
`terminal_nonterminal_blocked_skipped_null_flagged_retired_and_ready_counts`,
`gate_cascade_receipts`, `v541_evidence_eligibility_state`,
`asp_continuous_relaxation_state`, `partial_atom_adapter_state`,
`flagship_refinement_benchmark_state`, `oracle_value_boundary_receipt`,
`revocable_memory_state`, `chronological_crystallization_state`,
`nonreplay_transfer_receipt`, `heldout_memory_transfer_state`,
`shadow_consumer_state`, `arc_causal_canary_state`, `arc_holdout_state`,
`arc_provenance_and_registry_receipts`,
`branch_independent_promotion_ledger`, `prior_failure_retirement_actions`,
`publication_gate_g1_g2_g3_g4_and_unmet_gates`, `source_mutation_count`,
`weight_mutation_count`, `unauthorized_external_call_count`,
`hidden_game_source_access_count`, `outer_loop_ground_truth_search_count`,
`arc_level_solve_claim_count`, `registry_update_count`,
`hardware_claim_count`, `speed_power_or_energy_claim_count`,
`protected_files_unchanged`,
`spec_traceability_status_changelog_reconciliation`, `prd_gap_table`,
`next_milestone_recommendations`, `preconditions_checked`,
`inference_substrate`, `verifier_is_oracle`, `field_provenance`,
`field_principles`, `test_commands`, `test_exit_codes`, `duration_s`,
`reproducibility_checksum`, and `honest_verdict`.

`source_mutation_count`, `weight_mutation_count`,
`unauthorized_external_call_count`, `hidden_game_source_access_count`,
`outer_loop_ground_truth_search_count`, `arc_level_solve_claim_count`,
`registry_update_count`, `hardware_claim_count`, and
`speed_power_or_energy_claim_count` SHALL be bare integer `0`. Every required
field SHALL have one short field principle and a provenance entry. The honest
verdict SHALL start with a terminal prefix.

### SCENARIO-INFRA-6296-1: Exact Paths Outrank Receipts

GIVEN a V542 task has an exact declared deliverable that is missing or
nonterminal and a conductor receipt that says OK
WHEN Exp6296 builds the deliverable matrix
THEN the artifact keeps the missing or nonterminal class, records the override
attempt, and does not use same-number aliases.

### SCENARIO-INFRA-6296-2: Gates Read Only Terminal Bare Fields

GIVEN a structured gate points at a nonterminal artifact, a missing artifact,
a missing field, or a principle-wrapped field
WHEN Exp6296 recomputes the gate cascade
THEN the gate fails closed and records why the exact bare upstream field was
ineligible.

### SCENARIO-INFRA-6296-3: Flags Stay Separate And Exclude Promotion

GIVEN an upstream V542 artifact has a stamped adversarial flag or a current
critical verifier flag
WHEN Exp6296 computes the task state and branch ledger
THEN stamped and current flags remain separate, and the affected branch is not
promoted.

### SCENARIO-INFRA-6296-4: Branches Cannot Launder Claims

GIVEN exact-state energy refinement, continuous learning, and ARC have
different terminal states
WHEN Exp6296 builds its promotion ledger
THEN oracle repair alone is not model value, replay-only gains are not
transfer, and an ARC no-solve canary is not a solve claim.

### SCENARIO-INFRA-6296-5: Retire-If-Same-Verdict Is Mechanical

GIVEN a prior failure has `retire_if_same_verdict=true`
WHEN the current exact artifact verdict equals the prior verdict
THEN Exp6296 records a retirement action and leaves the exclusion manifest
unchanged.

### SCENARIO-INFRA-6296-6: Artifact Schema Is Principle Annotated

GIVEN the capstone report, field principles, provenance, protected hashes, and
command receipts
WHEN Exp6296 validates the report before writing
THEN every required field is present, all forbidden claim counters are bare
integer zero, the checksum matches the normalized payload, and the honest
verdict has a terminal prefix.

## Implementation Status (REQ-INFRA-6296)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6296 | Implemented: `python/carnot/experiment_6296_v542_adversarial_capstone.py`; terminal artifact `results/experiment_6296_v542_adversarial_capstone.json`. | Implemented: `tests/python/test_experiment_6296_v542_adversarial_capstone.py`. |

## REQ-INFRA-6297: V542-To-V543 Handoff SHALL Preserve Terminal Classes And Validate The Reserved Roadmap

Carnot SHALL build Exp6297 as a deterministic terminal-boundary handoff from
milestone `2026.08.542` into milestone `2026.08.543`. The handoff SHALL read
the V542 capstone and operational retro as input evidence. It SHALL classify
every V542 declared deliverable by exact path with the shared terminal-artifact
classifier. It SHALL preserve missing, nonterminal, blocked, skipped, null,
flagged, retired, and ready states without promotion from conductor receipts or
nearby aliases.

Exp6297 SHALL validate the V543 roadmap without activating
`research-roadmap-next.yaml` and without editing `research-roadmap.yaml`. The
validation SHALL require exactly 13 tasks in Exp6297 through Exp6309 order, no
duplicate ids, no reserved-id or deliverable collisions, no retired
dependencies, valid structured gates, complete `prior_failures` entries, Codex
direct routing through `model: gpt-5.5`, and prompt endings that preserve the
conductor. Each live LLM task SHALL name a mandated local GGUF family in
`MODEL_SPECS`.

Exp6297 SHALL validate every structured gate against the upstream task's
`REQUIRED ARTIFACT FIELDS` block. A gate SHALL be valid only when the upstream
task exists, the operator is supported, and the named field is declared by the
upstream artifact contract. Retired dependencies and incomplete prior-failure
records SHALL make the handoff blocked.

The Exp6297 artifact SHALL be written atomically to
`results/experiment_6297_v543_terminal_transition.json`. Its
`inference_substrate` SHALL be exactly `aggregation_from_upstream_artifacts`,
`verifier_is_oracle` SHALL be false, and `retired_dependency_count` and
`id_collision_count` SHALL be bare integer `0` for a passing handoff.

The Exp6297 artifact SHALL include these required fields: `status`,
`v542_roadmap_path_and_hash`, `v542_task_terminal_matrix`,
`v542_capstone_path_hash_and_summary`,
`operational_retro_path_hash_and_summary`,
`focused_and_broad_validation_receipts_by_task`,
`missing_nonterminal_blocked_skipped_null_flagged_retired_and_ready_counts`,
`v543_roadmap_path_and_hash`, `v543_task_ids_and_deliverables`,
`task_count`, `phase_counts`, `dependency_validation`,
`gated_on_validation`, `prior_failure_validation`,
`retired_dependency_count`, `id_collision_count`,
`agent_routing_validation`, `model_policy_validation`,
`prompt_contract_validation`, `protected_files_unchanged`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`field_provenance`, `field_principles`, `test_commands`,
`test_exit_codes`, `duration_s`, `reproducibility_checksum`, and
`honest_verdict`.

### SCENARIO-INFRA-6297-1: V542 Exact Deliverables Classify Fail Closed

GIVEN the V542 capstone matrix and exact deliverable paths
WHEN Exp6297 builds the V542 terminal matrix
THEN every V542 task from Exp6284 through Exp6296 is classified from its exact
path, missing artifacts remain missing, aliases are ignored, and flagged
artifacts remain flagged.

### SCENARIO-INFRA-6297-2: V543 Identity And Collision Audit Is Exact

GIVEN the V543 roadmap
WHEN Exp6297 validates ids, deliverables, reserved ranges, dependencies, and
retired references
THEN the task id list is exactly Exp6297 through Exp6309 in order,
`task_count=13`, `id_collision_count=0`, and
`retired_dependency_count=0`.

### SCENARIO-INFRA-6297-3: Gates And Prior Failures Are Structured

GIVEN a gated V543 task and a task with prior failures
WHEN Exp6297 audits the roadmap
THEN each gate names an existing upstream task and a field in that upstream
task's required artifact fields, and each prior-failure entry names the prior
experiment, verdict, changed mechanism, and `retire_if_same_verdict=true`.

### SCENARIO-INFRA-6297-4: Routing, Model Policy, And Prompt Endings Are Mechanical

GIVEN V543 tasks routed directly to Codex and live LLM tasks
WHEN Exp6297 audits route and prompt contracts
THEN each Codex task uses `model: gpt-5.5`, no Gemini route is accepted, each
live LLM task names a mandated GGUF family in `MODEL_SPECS`, and each prompt
ends with the required run command and conductor-protection sentence.

### SCENARIO-INFRA-6297-5: Protected Files Stay Byte-Identical

GIVEN before-state hashes for the active roadmap, conductor, ops ledgers,
traceability, completed record, exclusion manifest, V542 inputs, and V543
roadmap files
WHEN Exp6297 writes its own result artifact
THEN those protected paths have identical before and after hashes.

### SCENARIO-INFRA-6297-6: Artifact Schema Is Principle Annotated

GIVEN the handoff report
WHEN Exp6297 validates the payload before writing
THEN every required field is present, every required field has provenance and a
field-principle entry, the checksum matches the normalized payload, and
`honest_verdict` starts with a terminal prefix.

## Implementation Status (REQ-INFRA-6297)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6297 | Pending implementation: `python/carnot/experiment_6297_v543_terminal_transition.py`; terminal artifact `results/experiment_6297_v543_terminal_transition.json`. | Pending focused tests: `tests/python/test_experiment_6297_v543_terminal_transition.py`. |

## REQ-INFRA-6310: V543-To-V544 Handoff SHALL Preserve Terminal And Adversarial Classes

Carnot SHALL build Exp6310 as a deterministic terminal-boundary handoff from
milestone `2026.08.543` into milestone `2026.08.544`. The handoff SHALL read
Exp6309, the V543 operational retro, the exclusion manifest, the active and
staged roadmap paths, and all V543 declared artifacts. It SHALL classify every
V543 declared artifact by exact path with `python/carnot/terminal_artifacts.py`.
It SHALL preserve missing, nonterminal, blocked, skipped, null, flagged,
retired, ready, positive, safety-only, and shadow-only states without promoting
one branch from another branch's evidence.

Exp6310 SHALL validate the V544 roadmap without activating
`research-roadmap-next.yaml` and without editing `research-roadmap.yaml`. The
validation SHALL require exactly 13 tasks in Exp6310 through Exp6322 order, no
duplicate ids, no reserved-id or deliverable collisions, no retired
dependencies, valid structured gates, complete `prior_failures` entries, Codex
direct routing through `model: gpt-5.5`, and prompt endings that preserve the
conductor. Each live LLM task SHALL name the three mandated local GGUF model
identities when it needs all three model families.

Exp6310 SHALL prove that the shared activation bus, shared-state initializer,
licensed cross-family transfer, external text scorer, KAN revival, and
unchanged hardware probe scopes are not scheduled by the V544 roadmap. A
negated mention in a task prompt is not an activation. A dependency on a retired
experiment SHALL make the handoff blocked.

The Exp6310 artifact SHALL be written atomically to
`results/experiment_6310_v544_terminal_transition.json`. Its
`inference_substrate` SHALL be exactly `aggregation_from_upstream_artifacts`,
`verifier_is_oracle` SHALL be false, `task_count` SHALL be bare integer `13`,
and `retired_dependency_count` and `id_collision_count` SHALL be bare integer
`0` for a passing handoff.

The Exp6310 artifact SHALL include these required fields: `status`,
`v543_roadmap_path_and_hash`, `v543_task_terminal_matrix`,
`v543_capstone_path_hash_and_summary`,
`operational_retro_path_hash_and_summary`,
`missing_nonterminal_blocked_skipped_null_flagged_retired_ready_and_positive_counts`,
`v544_roadmap_path_and_hash`, `v544_task_ids_and_deliverables`,
`task_count`, `phase_counts`, `dependency_validation`,
`gated_on_validation`, `prior_failure_validation`,
`retired_dependency_count`, `id_collision_count`,
`agent_routing_validation`, `model_policy_validation`,
`prompt_contract_validation`, `protected_files_unchanged`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`field_provenance`, `field_principles`, `test_commands`,
`test_exit_codes`, `duration_s`, `reproducibility_checksum`, and
`honest_verdict`.

### SCENARIO-INFRA-6310-1: V543 Exact Deliverables Preserve Current Classes

GIVEN the Exp6309 exact declared task matrix and V543 deliverable paths
WHEN Exp6310 builds the V543 terminal matrix
THEN every V543 task from Exp6297 through Exp6309 is classified from its exact
path, Exp6303 remains missing, Exp6300 and Exp6301 remain flagged, Exp6302 and
Exp6305 remain skipped, and Exp6304 and Exp6306 remain positive.

### SCENARIO-INFRA-6310-2: V544 Identity And Collision Audit Is Exact

GIVEN the V544 roadmap
WHEN Exp6310 validates ids, deliverables, reserved ranges, dependencies, and
retired references
THEN the task id list is exactly Exp6310 through Exp6322 in order,
`task_count=13`, `id_collision_count=0`, and
`retired_dependency_count=0`.

### SCENARIO-INFRA-6310-3: Gates And Prior Failures Are Structured

GIVEN a gated V544 task and a task with prior failures
WHEN Exp6310 audits the roadmap
THEN each gate names an existing upstream task and a field in that upstream
task's required artifact fields, and each prior-failure entry names the prior
experiment, verdict, changed mechanism, and `retire_if_same_verdict=true`.

### SCENARIO-INFRA-6310-4: Retired Branches Are Not Reactivated

GIVEN the V543 retired initializer and licensed-transfer gate repeats
WHEN Exp6310 audits V544 title, prompt, dependency, and gate text
THEN no task schedules the shared bus, blocked initializer retry, licensed
cross-family transfer retry, external text scorer, KAN revival, or unchanged
hardware probe scope.

### SCENARIO-INFRA-6310-5: Routing, Model Policy, And Prompt Endings Are Mechanical

GIVEN V544 tasks routed directly to Codex and live LLM tasks
WHEN Exp6310 audits route and prompt contracts
THEN each task uses `agent_type: codex` with `model: gpt-5.5`, no Gemini route
is accepted, live three-family prompts name the three mandated GGUF identities,
and each prompt ends with the required run command and conductor-protection
sentence.

### SCENARIO-INFRA-6310-6: Artifact Schema Is Principle Annotated

GIVEN the handoff report
WHEN Exp6310 validates the payload before writing
THEN every required field is present, every required field has provenance and a
field-principle entry, the checksum matches the normalized payload, and
`honest_verdict` starts with a terminal prefix.

## Implementation Status (REQ-INFRA-6310)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6310 | Pending implementation: `python/carnot/experiment_6310_v544_terminal_transition.py`; terminal artifact `results/experiment_6310_v544_terminal_transition.json`. | Pending focused tests: `tests/python/test_experiment_6310_v544_terminal_transition.py`. |

## REQ-INFRA-6311: V544 Post-Marker Source Freeze SHALL Be Strict, Null-Safe, And Contract-Complete

Carnot SHALL build Exp6311 as a deterministic V544 post-marker source sweep
and execution-scope freeze. The sweep SHALL hash the exact
`<!-- V544-PLANNER-REFRESH-20260811-END -->` marker in
`research-references.md`. It SHALL record the marker line. It SHALL use the
marker commit time as the exclusive lower bound for novelty.

Exp6311 SHALL search arXiv first. It SHALL then search OpenReview, Hugging
Face Papers, Semantic Scholar citation routes for EBT and ARM-EBM, Extropic,
Logical Intelligence, and GitHub. Each channel SHALL record direct URLs, query
timestamps, dates, raw endpoint status, and a disposition. HTTP failures,
browser challenges, empty endpoints, and inaccessible pages SHALL be recorded
as receipts, not promoted findings.

Exp6311 SHALL deduplicate candidates against all earlier
`research-references.md` planner markers and against repeated source ids,
URLs, titles, and content hashes in the current sweep. Exp6311 SHALL accept
only stable, non-duplicate, reproducible, primary or first-party evidence that
is strictly later than the V544 marker and that changes a local executable
contract. A zero-source delta SHALL be terminal. In that case
`accepted_count` SHALL be the bare integer `0`, `accepted_findings` SHALL be
empty, `research-references.md` SHALL remain byte-identical, and
`honest_verdict` SHALL start with `complete_null:`.

Exp6311 SHALL freeze contracts for the V544 model-local state surface, exact
code-pair fixture, per-model energy, versioned same-domain learner, protected
validation, ARC shadow no-solve path, and no-hardware-claim boundary. The
contracts SHALL explicitly exclude the retired V543 shared activation bus,
shared-state initializer, licensed cross-family transfer, external text
scorer, KAN replacement, generated-answer transport, TSU execution, and
unchanged physical-board probe.

The Exp6311 artifact SHALL be written atomically to
`results/experiment_6311_v544_post_marker_source_scope_freeze.json` with
`inference_substrate=web_and_bibliographic_search_only` and
`verifier_is_oracle=false`. It SHALL not modify `scripts/research_conductor.py`.

The Exp6311 artifact SHALL include these required fields: `status`,
`v544_marker_text_and_line`, `search_window_start_utc`,
`search_completed_utc`, `source_queries_by_channel`, `source_receipts`,
`accepted_findings`, `accepted_count`, `duplicate_findings`,
`watch_only_findings`, `inaccessible_sources`,
`excluded_findings_and_reasons`,
`semantic_scholar_ebt_and_arm_ebm_receipts`, `extropic_status`,
`logical_intelligence_status`, `github_status`,
`frozen_model_local_surface_contract`, `frozen_exact_pair_fixture_contract`,
`frozen_model_local_energy_contract`,
`frozen_versioned_learning_contract`,
`frozen_protected_validation_contract`,
`frozen_arc_shadow_no_solve_contract`, `frozen_hardware_exclusions`,
`roadmap_scope_delta`, `protected_files_unchanged`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`field_provenance`, `field_principles`, `test_commands`,
`test_exit_codes`, `duration_s`, `reproducibility_checksum`, and
`honest_verdict`.

### SCENARIO-INFRA-6311-1: Marker Bound Is Exclusive

GIVEN the sealed V544 planner refresh marker in `research-references.md`
WHEN Exp6311 classifies a candidate at or before the marker commit time
THEN the candidate is rejected, and a bare same-day date is rejected unless a
strictly later timestamp is present.

### SCENARIO-INFRA-6311-2: Zero Findings Preserve References

GIVEN every source channel returns no strictly later stable evidence,
duplicate evidence, inaccessible evidence, pre-marker evidence, or watch-only
evidence
WHEN Exp6311 writes its artifact
THEN `accepted_count` is the bare integer `0`, `accepted_findings` is empty,
`research-references.md` is byte-identical, and `honest_verdict` starts with
`complete_null:`.

### SCENARIO-INFRA-6311-3: Dedupe And Scope Hashes Fail Closed

GIVEN a candidate has no stable URL, repeats an earlier source id, repeats an
existing reference block, repeats a current-sweep content hash, lacks a local
executable consequence, or changes protected input hashes
WHEN Exp6311 deduplicates and validates the sweep
THEN the row is not accepted and the protected hash ledger records the exact
before and after state.

### SCENARIO-INFRA-6311-4: Frozen Contracts Preserve V544 Boundaries

GIVEN V544 depends on model-local surfaces, exact code pairs, per-model energy
heads, versioned same-domain learning, protected validation, ARC shadow
evidence, and no hardware claim
WHEN Exp6311 serializes frozen contracts
THEN each contract has a stable version, required boundary fields, explicit
retired-mechanism exclusions, and no shared-bus, cross-family, generated-text,
TSU, or physical-board promotion path.

### SCENARIO-INFRA-6311-5: Artifact Schema Is Principle Annotated

GIVEN source receipts, protected hashes, field principles, and command
receipts
WHEN Exp6311 validates the report before writing
THEN every required field is present, every field has provenance and a
principle, `verifier_is_oracle=false`, the checksum matches the normalized
payload, and the honest verdict has a terminal prefix.

## Implementation Status (REQ-INFRA-6311)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6311 | Pending implementation: `python/carnot/experiment_6311_v544_post_marker_source_scope_freeze.py`; terminal artifact `results/experiment_6311_v544_post_marker_source_scope_freeze.json`. | Pending focused tests: `tests/python/test_experiment_6311_v544_post_marker_source_scope_freeze.py`. |

## REQ-INFRA-6298: Terminal Evidence Preflight SHALL Fail Closed Before Gate Consumption

Carnot SHALL provide a reusable standalone terminal-evidence preflight for
experiment result artifacts and staged gate fields. The preflight SHALL live
outside `scripts/research_conductor.py`. It SHALL expose a library function
under `python/carnot/` and a thin module CLI runnable as
`.venv/bin/python -m carnot.experiment_6298_terminal_evidence_preflight_linter --date YYYYMMDD`.

The preflight SHALL check required-field presence, a terminal-prefixed
`honest_verdict`, field-principle coverage, field provenance, inference
substrate declaration, substrate duration floors, methodology receipts,
test-command existence, recorded exit-code parity, staged gate-field type,
reproducibility fields, determination preservation, and protected-file hashes.
It SHALL compare only already-recorded test commands and exit codes unless a
fixture explicitly opts into a bounded project-owned command. It SHALL NOT
execute arbitrary command strings embedded in a result artifact.

The preflight SHALL use the current substrate duration table from
`scripts/adversarial_verify.py`. It SHALL fail closed on unknown
compute-bound substrates, missing methodology receipts, missing duration
measurements, and impossible timing below the selected substrate floor.
Structured gate fields SHALL be eligible only when the exact artifact is
terminal, the field is present at the top level, the value is not
principle-wrapped, and the bare value has the expected type.

Exp6298 SHALL replay the V542 failure shapes without rewriting prior
artifacts: Exp6288's implausibly short and methodologically incomplete
compute-bound evidence, and Exp6289 plus Exp6290's recorded verification
receipt failures. Exp6298 SHALL also evaluate synthetic clean, missing-field,
bad-prefix, bad-gate-type, and determination-drop fixtures.

Exp6298 SHALL write
`results/experiment_6298_terminal_evidence_preflight_linter.json`. It SHALL
also write a synthetic fixture manifest and record that manifest's path and
hash. Its `inference_substrate` SHALL be exactly `artifact_qa_lint_tests`.
The terminal artifact SHALL include these required fields: `status`,
`failure_taxonomy`, `source_paths_and_hashes`,
`v542_fixture_paths_hashes_and_expected_classes`,
`synthetic_fixture_manifest_path_and_hash`, `required_field_checks`,
`terminal_prefix_checks`, `field_principle_coverage_checks`,
`substrate_duration_and_methodology_checks`,
`test_command_and_exit_code_checks`, `gate_field_type_checks`,
`determination_preservation_checks`, `clean_fixture_accept_count`,
`bad_fixture_reject_count`, `false_accept_count`, `false_reject_count`,
`cli_contract`, `protected_files_unchanged`, `preconditions_checked`,
`inference_substrate`, `verifier_is_oracle`, `field_provenance`,
`field_principles`, `test_commands`, `test_exit_codes`, `duration_s`,
`random_seed`, `reproducibility_checksum`,
`terminal_evidence_preflight_ready_score`, and `honest_verdict`.

For a readiness artifact, `false_accept_count` and `false_reject_count` SHALL
be bare integer `0`, `verifier_is_oracle` SHALL be false, every required field
SHALL have one field principle, and `honest_verdict` SHALL start with a
terminal prefix. `terminal_evidence_preflight_ready_score` SHALL be `1.0`
only when the clean fixture is accepted, every bad fixture is rejected, both
false-counts are zero, and protected files are byte-identical.

### SCENARIO-INFRA-6298-1: V542 Failure Shapes Replay Without Prior Artifact Mutation

GIVEN the checked-in Exp6288, Exp6289, and Exp6290 result artifacts
WHEN the terminal-evidence preflight replays them as immutable fixtures
THEN Exp6288 is rejected for substrate duration or methodology evidence, and
Exp6289 plus Exp6290 are rejected for recorded test-command receipt parity or
nonzero exit-code evidence.

### SCENARIO-INFRA-6298-2: Synthetic Fixture Matrix Has Zero False Counts

GIVEN clean, missing-field, bad-prefix, bad-gate-type, and determination-drop
synthetic fixtures
WHEN the preflight evaluates the manifest
THEN the clean fixture is accepted, every bad fixture is rejected, and both
false accept and false reject counts are bare integer zero.

### SCENARIO-INFRA-6298-3: Substrate And Methodology Checks Use The Canonical Floor Table

GIVEN an artifact with compute-bound markers, an unknown substrate, missing
methodology receipts, or `duration_s` below the selected floor
WHEN the preflight evaluates substrate evidence
THEN it rejects the artifact with explicit failure classes and records the
selected floor descriptor.

### SCENARIO-INFRA-6298-4: Test Commands Are Compared, Not Executed

GIVEN an artifact records `test_commands` and `test_exit_codes`
WHEN the preflight checks verification receipts
THEN every declared command must have one recorded exit code, no extra exit
code may appear, every recorded code must be integer zero, and the preflight
does not execute artifact-supplied command strings.

### SCENARIO-INFRA-6298-5: Gate Fields Must Be Terminal Exact Bare Typed Values

GIVEN a staged gate points at a missing, nonterminal, principle-wrapped, or
wrong-typed field
WHEN the preflight checks gate-field eligibility
THEN it rejects the staged gate before conductor gate evaluation can consume
the value.

### SCENARIO-INFRA-6298-6: Artifact Schema, Protected Hashes, And CLI Contract Are Validated

GIVEN an Exp6298 report, field principles, provenance, protected hashes,
fixture manifest, and command receipts
WHEN the CLI validates and writes the report
THEN every required field is present, the checksum matches the normalized
payload, protected files remain byte-identical, `verifier_is_oracle=false`,
and the verdict has a terminal prefix.

## Implementation Status (REQ-INFRA-6298)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6298 | Implemented: `python/carnot/terminal_evidence_preflight.py`, `python/carnot/experiment_6298_terminal_evidence_preflight_linter.py`; terminal artifact `results/experiment_6298_terminal_evidence_preflight_linter.json`. | Implemented: `tests/python/test_experiment_6298_terminal_evidence_preflight_linter.py`. |

## REQ-INFRA-6299: V543 Post-Marker Source Freeze SHALL Be Strict, Null-Safe, And Contract-Complete

Carnot SHALL build Exp6299 as a deterministic V543 post-marker source sweep
and scope freeze. The sweep SHALL hash the exact
`<!-- V543-PLANNER-REFRESH-20260810-END -->` marker. It SHALL use the marker
commit time as an exclusive lower bound. Evidence at or before that bound, or
evidence with only an ambiguous same-day date, SHALL be rejected before any
reference append.

Exp6299 SHALL record receipts for arXiv, OpenReview, Extropic, Semantic
Scholar EBT and ARM-EBM citation routes, Hugging Face Papers, targeted GitHub,
and Logical Intelligence. HTTP 429, timeouts, inaccessible pages, and browser
challenges SHALL be recorded as receipts, not findings. Exp6299 SHALL
deduplicate candidates against all prior `research-references.md` blocks and
against repeated source ids, URLs, titles, and content hashes in the current
sweep.

Exp6299 SHALL accept only stable, non-duplicate, reproducible, primary or
first-party evidence that is strictly later than the V543 marker and that
changes a V543 contract or adds a reproducible dependency. A zero-source delta
SHALL be terminal. In that case `research-references.md` SHALL remain
byte-identical, `accepted_count` SHALL be the bare integer `0`, and
`honest_verdict` SHALL start with `complete_null:`.

Exp6299 SHALL freeze machine-readable contracts for the activation bus,
independent integrity audit, activation-to-state initializer, live benchmark,
reference-anchored online learning, evidence-licensed transfer, ARC target
validation, and no-hardware-claim boundary. The contract set SHALL state that
exact ASP or Clingo is an oracle. The hardware boundary SHALL state that no
current board or TSU route supports a hardware, speed, power, energy, or
availability claim.

The Exp6299 artifact SHALL be written atomically to
`results/experiment_6299_v543_post_marker_source_scope_freeze.json` with
`inference_substrate=web_and_bibliographic_search_only` and
`verifier_is_oracle=false`. It SHALL audit the staged roadmap schema, roadmap
gates, exclusion manifest, prior-failure contracts, model policy,
prompt-ending rules, and id collisions without editing `research-roadmap.yaml`.
Because the frozen downstream contracts name future GGUF model families, the
artifact SHALL include top-level methodology descriptors that state no model
was invoked and no random sampling was used.

The Exp6299 artifact SHALL include these required fields: `status`,
`planner_marker_and_hash`, `query_window`, `source_channel_receipts`,
`discovered_candidates`, `accepted_findings`,
`rejected_duplicate_or_watch_only_findings`, `accepted_count`,
`references_append_receipt`, `frozen_activation_bus_contract`,
`frozen_integrity_audit_contract`, `frozen_state_initializer_contract`,
`frozen_live_benchmark_contract`, `frozen_online_learning_contract`,
`frozen_transfer_license_contract`, `frozen_arc_target_validation_contract`,
`frozen_hardware_boundary`, `roadmap_path_and_hash`,
`roadmap_schema_result`, `exclusion_manifest_lint_result`,
`prior_failure_contract_result`, `protected_files_unchanged`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`field_provenance`, `field_principles`, `test_commands`,
`test_exit_codes`, `duration_s`, `reproducibility_checksum`, and
`honest_verdict`.

### SCENARIO-INFRA-6299-1: Marker Bound Is Exclusive

GIVEN the sealed V543 planner refresh marker in `research-references.md`
WHEN Exp6299 classifies a candidate at or before the marker commit time
THEN the candidate is rejected, and a bare same-day date is rejected unless a
strictly later timestamp is present.

### SCENARIO-INFRA-6299-2: Zero Findings Preserve References

GIVEN every source channel returns no strictly later stable evidence, duplicate
evidence, inaccessible evidence, pre-marker evidence, or watch-only evidence
WHEN Exp6299 writes its artifact
THEN `accepted_count` is the bare integer `0`, `accepted_findings` is empty,
`research-references.md` is byte-identical, and `honest_verdict` starts with
`complete_null:`.

### SCENARIO-INFRA-6299-3: Stable URLs And Duplicates Fail Closed

GIVEN a candidate has no stable URL, repeats an earlier source id, repeats an
existing reference block, repeats a current-sweep content hash, or lacks a
scope-changing reproducible dependency
WHEN Exp6299 deduplicates the candidate ledger
THEN the row is recorded only in
`rejected_duplicate_or_watch_only_findings`.

### SCENARIO-INFRA-6299-4: Frozen Contracts Preserve V543 Boundaries

GIVEN V543 depends on a shared activation bus, an independent shortcut audit,
an activation-to-state initializer, a live three-family benchmark,
reference-anchored online learning, evidence-licensed transfer, ARC target
validation, and no hardware claim
WHEN Exp6299 serializes the frozen contracts
THEN each contract has a stable version, required boundary fields, an exact
ASP or Clingo oracle statement, frozen model weights where required, target
validation, and explicit claim limits for boards and TSU routes.

### SCENARIO-INFRA-6299-5: Artifact Schema Is Principle Annotated

GIVEN source receipts, roadmap receipts, protected hashes, field principles,
and command receipts
WHEN Exp6299 validates the report before writing
THEN every required field is present, every field has provenance and a
principle, `verifier_is_oracle=false`, the checksum matches the normalized
payload, and the honest verdict has a terminal prefix.

## Implementation Status (REQ-INFRA-6299)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6299 | Implemented: `python/carnot/experiment_6299_v543_post_marker_source_scope_freeze.py`; terminal artifact `results/experiment_6299_v543_post_marker_source_scope_freeze.json`. | Implemented: `tests/python/test_experiment_6299_v543_post_marker_source_scope_freeze.py`. |

## REQ-INTEG-6309: V543 Adversarial Capstone SHALL Preserve Exact Branch Evidence And Bound Claims

Carnot SHALL build Exp6309 as the V543 adversarial capstone for milestone
`2026.08.543`. The capstone SHALL classify the exact declared deliverable for
each task from Exp6297 through Exp6309. It SHALL NOT substitute aliases,
source modules, conductor receipts, sidecars, or similarly named files for a
missing exact deliverable.

Exp6309 SHALL independently rerun current adversarial artifact checks on the
declared artifact paths and SHALL keep stamped determinations separate from
current-rule determinations. It SHALL also replay the Exp6298 terminal-evidence
preflight fixture logic without rewriting the Exp6298 artifact. Missing,
nonterminal, blocked, skipped, null, flagged, retired, ready, oracle-only,
replay-only, safety-only, and unlicensed-transfer states SHALL remain visible.
The conserved terminal-class denominator SHALL cover exactly 13 tasks.

Exp6309 SHALL adjudicate four branches independently: infrastructure and source
integrity, shared model state to exact energy, continuous online learning plus
licensed transfer, and ARC target-validated routing. The capstone SHALL NOT
count exact-oracle completion as verifier value, replay as transfer, safety as
utility, retrieval as licensed transfer, or ARC proxy metrics as a solve. The
top-level `verifier_is_oracle` field SHALL be the exact string
`mixed_with_explicit_per_branch_boundary`, with each branch carrying its own
oracle boundary.

Exp6309 SHALL replay structured roadmap gates from exact bare upstream fields.
It SHALL apply `retire_if_same_verdict` mechanically. When the current verdict
matches a prior verdict covered by that rule, the report SHALL record an
explicit retirement action and SHALL list any exclusion-manifest update. When
the rule does not fire, the report SHALL state that no manifest update occurred.

The Exp6309 artifact SHALL be written atomically to
`results/experiment_6309_v543_adversarial_capstone.json`. It SHALL include
these required fields: `status`, `milestone_roadmap_path_and_hash`,
`exact_declared_task_artifact_matrix`,
`upstream_terminal_classification_by_task`,
`current_rule_adversarial_results_by_task`,
`missing_nonterminal_blocked_skipped_null_flagged_retired_ready_oracle_only_replay_only_safety_only_and_unlicensed_counts`,
`terminal_evidence_preflight_summary`,
`branch_independent_promotion_ledger`, `shared_activation_bus_verdict`,
`shared_state_initializer_verdict`, `live_three_family_value_verdict`,
`continuous_self_learning_verdict`, `online_learning_safety_verdict`,
`evidence_licensed_transfer_verdict`, `arc_target_validation_verdict`,
`oracle_claim_boundary`, `replay_is_not_transfer_boundary`,
`safety_cannot_promote_utility_boundary`, `arc_no_solve_claim_boundary`,
`prd_gap_verdicts`, `prior_failure_retirement_actions`,
`exclusion_manifest_updates`, `publication_gate_replay`,
`architecture_reconciliation_receipt`,
`openspec_traceability_status_changelog_and_reference_reconciliation_receipts`,
`protected_files_unchanged`, `preconditions_checked`, `inference_substrate`,
`verifier_is_oracle`, `field_provenance`, `field_principles`,
`test_commands`, `test_exit_codes`, `duration_s`,
`reproducibility_checksum`, and `honest_verdict`.

### SCENARIO-INTEG-6309-1: Exact Declared Artifacts Conserve The Denominator

GIVEN the active V543 roadmap and declared deliverables for Exp6297 through
Exp6309
WHEN Exp6309 builds its artifact matrix
THEN exactly 13 task rows are present, every row is classified from the exact
declared path, missing Exp6303 remains missing, and no alias or conductor
receipt replaces it.

### SCENARIO-INTEG-6309-2: Current Rules And Stamps Stay Separate

GIVEN stamped flags and the current `scripts/adversarial_verify.py` rules
WHEN Exp6309 reviews declared artifacts
THEN stamped `flagged_adversarial` state, current critical flags, and
current warnings are recorded separately for each task.

### SCENARIO-INTEG-6309-3: Structured Gates Are Replayed From Bare Fields

GIVEN V543 roadmap gates for Exp6302, Exp6303, Exp6305, and Exp6308
WHEN Exp6309 replays the gates
THEN only terminal exact artifacts with exact bare fields feed gate decisions,
and skipped or missing upstream evidence cannot pass a downstream gate.

### SCENARIO-INTEG-6309-4: Branch Promotion Is Independent

GIVEN the infrastructure, shared-state, online-learning, licensed-transfer,
safety, and ARC route artifacts
WHEN Exp6309 builds the promotion ledger
THEN each branch receives its own promotion verdict and blocker list, with no
pooled score hiding a failed task or fold.

### SCENARIO-INTEG-6309-5: Claim Boundaries Are Explicit

GIVEN oracle, replay, safety, retrieval, and ARC proxy evidence
WHEN Exp6309 writes boundary receipts
THEN exact oracle evidence is not verifier value, replay is not transfer,
safety is not utility, retrieval is not licensed transfer, and ARC route
metrics are not a solve claim.

### SCENARIO-INTEG-6309-6: Schema And Reconciliation Receipts Are Principle Annotated

GIVEN the capstone report
WHEN Exp6309 validates it before writing
THEN every required field is present, every required field has provenance and a
field-principle entry, `verifier_is_oracle` is
`mixed_with_explicit_per_branch_boundary`, the checksum matches the normalized
payload, and `honest_verdict` starts with a terminal prefix.

## Implementation Status (REQ-INTEG-6309)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INTEG-6309 | Planned: `python/carnot/experiment_6309_v543_adversarial_capstone.py`; terminal artifact `results/experiment_6309_v543_adversarial_capstone.json`. | Planned: `tests/python/test_experiment_6309_v543_adversarial_capstone.py`. |

## REQ-INFRA-6322: V544 Capstone SHALL Preserve Exact Evidence And Block Claim Laundering

Carnot SHALL build Exp6322 as the V544 adversarial capstone for milestone
`2026.08.544`. The capstone SHALL classify the exact declared deliverable for
each task from Exp6310 through Exp6322. It SHALL NOT substitute aliases,
sidecars, conductor receipts, summaries, source modules, or similarly named
files for a missing exact deliverable.

Exp6322 SHALL classify missing, nonterminal, flagged, null, blocked, skipped,
oracle-only, safety-only, shadow-only, ready, and positive states before it
reads headline metrics. A missing or flagged artifact SHALL NOT become null,
ready, or positive because another artifact or document says the branch is
healthy. Raw blocked gate status SHALL remain visible even when the shared
terminal classifier records the row as a gate skip.

Exp6322 SHALL promote the model-local representation branch, model-local probe
integrity branch, live model-local verifier branch, versioned factor-local
learning branch, feedback-directed search branch, online self-evolution safety
branch, and ARC live-shadow branch independently. No aggregate score SHALL
rescue a failed model, fold, attack, provenance cell, protected-validation
cell, or ARC zero-credit boundary.

Exp6322 SHALL set `shared_bus_promotion_allowed`,
`cross_family_transfer_promotion_allowed`,
`exact_oracle_as_learned_verifier_allowed`,
`protected_validation_as_progress_allowed`, and `arc_solve_claim_allowed` to
bare boolean `false`. It SHALL preserve ARC shadow-only evidence with
`solve_claimed=false`, `levels_credited=0`, and `registry_update_count=0`.
It SHALL record no hardware speed, power, board, TSU, or availability claim
unless an exact authenticated upstream artifact supports that claim.

Exp6322 SHALL replay `retire_if_same_verdict` mechanics from the staged V544
prior-failure contract. It SHALL record manifest updates only when the staged
contract fires. It SHALL not modify `research-roadmap.yaml` or
`scripts/research_conductor.py`.

The Exp6322 artifact SHALL be written atomically to
`results/experiment_6322_v544_adversarial_capstone.json`. It SHALL include
these required fields: `status`, `roadmap_path_and_hash`,
`declared_task_ids_and_deliverables`, `task_terminal_matrix`,
`missing_nonterminal_flagged_null_blocked_skipped_oracle_only_safety_only_shadow_only_ready_and_positive_counts`,
`source_and_scope_freeze_summary`, `infrastructure_readiness`,
`model_local_representation_verdict`,
`model_local_probe_integrity_verdict`,
`live_model_local_verifier_verdict`,
`versioned_factor_local_learning_verdict`,
`feedback_directed_search_verdict`,
`online_self_evolution_safety_verdict`, `arc_live_shadow_verdict`,
`shared_bus_promotion_allowed`,
`cross_family_transfer_promotion_allowed`,
`exact_oracle_as_learned_verifier_allowed`,
`protected_validation_as_progress_allowed`, `arc_solve_claim_allowed`,
`branch_promotion_matrix`, `exclusion_manifest_updates`,
`failed_experiment_rerun_retirements`, `prd_gap_delta`,
`hardware_claim_boundary`, `reconciled_document_paths_and_hashes`,
`operational_retro_path_and_hash`, `protected_files_unchanged`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`field_provenance`, `field_principles`, `test_commands`,
`test_exit_codes`, `duration_s`, `reproducibility_checksum`, and
`honest_verdict`.

### SCENARIO-INFRA-6322-1: Exact Declared Artifacts Conserve The V544 Denominator

GIVEN the active V544 roadmap and declared deliverables for Exp6310 through
Exp6322
WHEN Exp6322 builds its task matrix
THEN exactly 13 rows are present, every row is classified from its exact
declared path, missing Exp6315 and Exp6317 remain missing, and no alias or
receipt replaces them.

### SCENARIO-INFRA-6322-2: Terminal States Are Classified Before Metrics

GIVEN a null preflight, a blocked corpus, a missing probe artifact, a flagged
integrity audit, and a zero-credit ARC shadow
WHEN Exp6322 computes counts and branch verdicts
THEN null, blocked, skipped, missing, flagged, safety-only, shadow-only, ready,
and positive states remain distinct and no later metric changes the terminal
state.

### SCENARIO-INFRA-6322-3: Branch Promotion Is Independent

GIVEN model-local, continuous-learning, feedback-search, safety, and ARC
shadow artifacts
WHEN Exp6322 builds the promotion matrix
THEN model-local verification can close null or flagged while learning utility,
feedback-search utility, safety, and ARC shadow reachability keep their own
promotion verdicts.

### SCENARIO-INFRA-6322-4: Laundering Guards Are Bare False

GIVEN exact-oracle sidecars, protected validation, safety-only evidence,
cross-family exclusions, and ARC shadow proposals
WHEN Exp6322 writes its artifact
THEN the five laundering-allowed fields are bare boolean `false`, ARC solve
credit remains zero, and protected validation does not become adaptive
progress.

### SCENARIO-INFRA-6322-5: Retirements And Exclusions Are Mechanical

GIVEN the V544 capstone prior-failure contract
WHEN Exp6322 compares the current verdict with the staged prior verdict
THEN `retire_if_same_verdict` fires only on exact verdict equality and
exclusion-manifest updates are recorded only for fired rules.

### SCENARIO-INFRA-6322-6: Schema, Provenance, And Reconciliation Receipts Are Complete

GIVEN the capstone report
WHEN Exp6322 validates it before writing
THEN every required field is present, every required field has provenance and a
field-principle entry, the checksum matches the normalized payload, protected
files are compared by hash, reconciliation paths are exact, and
`honest_verdict` starts with a terminal prefix.

## REQ-INFRA-6323: V545 Transition SHALL Preserve V544 Terminal Evidence And Validate The Next Contract

Carnot SHALL build Exp6323 as the exact V544-to-V545 terminal transition.
The transition SHALL consume the Exp6322 capstone and the exact declared V544
deliverables. It SHALL classify each V544 artifact with
`python/carnot/terminal_artifacts.py`. It SHALL preserve missing,
nonterminal, blocked, skipped, null, flagged, retired, ready, positive,
safety-only, shadow-only, and failed-command states without promotion.

Exp6323 SHALL validate the V545 roadmap contract without activating
`research-roadmap-next.yaml` and without editing `research-roadmap.yaml`.
The validation SHALL confirm exactly 14 tasks in Exp6323 through Exp6336
order, JSON deliverables under `results/`, same-milestone dependencies,
structured gates whose fields appear in upstream required-artifact blocks,
all four `prior_failures` subfields, `agent_type=codex`, and
`model=gpt-5.5`.

Exp6323 SHALL reject scheduled hidden/model-local state retries, external
generated-text scoring, KAN experiments, cross-family transfer, public ARC
re-solves, and unapproved hardware work. LLM prompts SHALL name the mandated
GGUF identities where the prompt requires all local models. All prompt
contracts SHALL end with the required protected-conductor warning.

The Exp6323 artifact SHALL be written atomically to
`results/experiment_6323_v545_terminal_transition.json`. It SHALL include
these required fields: `status`, `v544_roadmap_path_and_hash`,
`v544_task_terminal_matrix`, `v544_capstone_path_hash_and_summary`,
`v544_validation_failure_receipts`,
`missing_nonterminal_blocked_skipped_null_flagged_retired_ready_and_positive_counts`,
`v545_roadmap_path_and_hash`, `v545_task_ids_and_deliverables`,
`task_count`, `phase_counts`, `dependency_validation`,
`gated_on_validation`, `prior_failure_validation`,
`retired_dependency_count`, `id_collision_count`,
`agent_routing_validation`, `model_policy_validation`,
`prompt_contract_validation`, `protected_files_unchanged`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`field_provenance`, `field_principles`, `test_commands`,
`test_exit_codes`, `duration_s`, `reproducibility_checksum`, and
`honest_verdict`. `task_count` SHALL be bare integer `14`.
`retired_dependency_count` and `id_collision_count` SHALL be bare integer
`0`. `honest_verdict` SHALL begin with a terminal prefix.
`field_principles` SHALL cover every required field.

### SCENARIO-INFRA-6323-1: V544 Exact Artifact Classes Stay Visible

GIVEN the Exp6322 capstone declares the V544 task deliverables
WHEN Exp6323 builds its V544 task matrix
THEN every row is classified from its exact path, and missing, skipped, null,
flagged, blocked, safety-only, shadow-only, ready, and positive states remain
distinct.

### SCENARIO-INFRA-6323-2: Failed Required Commands Are Preserved

GIVEN Exp6322 recorded `.venv/bin/pytest tests/python -q` exit `3` and
`scripts/determination_preservation_lint.py --all` exit `1`
WHEN Exp6323 summarizes V544 validation
THEN both nonzero command receipts remain visible and do not become passing
validation.

### SCENARIO-INFRA-6323-3: V545 Roadmap Shape Is Exact

GIVEN the V545 roadmap contract reserves Exp6323 through Exp6336
WHEN Exp6323 validates the roadmap
THEN the roadmap has exactly 14 tasks in order, no duplicate task ids, no
duplicate deliverables, and every deliverable is a JSON path under `results/`.

### SCENARIO-INFRA-6323-4: Dependencies And Gates Stay In Milestone

GIVEN V545 tasks declare `requires` and `gated_on` metadata
WHEN Exp6323 validates the graph
THEN each dependency names a V545 task, no dependency points to a retired
experiment, and each gate field appears in the upstream prompt's required
artifact field block.

### SCENARIO-INFRA-6323-5: Routing, Model Policy, And Exclusions Are Enforced

GIVEN V545 tasks include Codex routing, model ids, prompt text, and hardware
constraints
WHEN Exp6323 validates each task
THEN every task has `agent_type=codex` and `model=gpt-5.5`, each LLM task
names the required GGUF identities, and retired model-local, KAN,
cross-family, public-ARC-resolve, external-scorer, and unapproved-hardware
work is not scheduled.

### SCENARIO-INFRA-6323-6: Output Is Atomic, Checksummed, And Non-Activating

GIVEN Exp6323 has classified V544 and validated V545 in memory
WHEN it writes the result artifact
THEN every required field is present, `field_principles` and
`field_provenance` cover all required fields, protected files are compared by
hash, the checksum matches the normalized payload, and the staged and active
roadmap files are not rewritten.

## Implementation Status (REQ-INFRA-6323)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6323 | Implemented: `python/carnot/experiment_6323_v545_terminal_transition.py`; terminal artifact `results/experiment_6323_v545_terminal_transition.json`. | Implemented: `tests/python/test_experiment_6323_v545_terminal_transition.py`. |

## REQ-INFRA-6324: V545 Post-Marker Source Freeze SHALL Be Strict, Null-Safe, And Contract-Complete

Carnot SHALL build Exp6324 as a deterministic V545 post-marker source sweep
and execution-scope freeze. The sweep SHALL hash the exact
`<!-- V545-PLANNER-REFRESH-20260812-END -->` marker in
`research-references.md`. It SHALL record the marker line. It SHALL use the
marker commit time as the exclusive lower bound for novelty.

Exp6324 SHALL search arXiv first. It SHALL then search OpenReview, Hugging
Face Papers, Semantic Scholar citation routes for EBT and ARM-EBM, Extropic,
Logical Intelligence, and GitHub. Each channel SHALL record direct URLs, query
timestamps, publication or update dates, raw endpoint status, and a
disposition. HTTP failures, browser challenges, empty endpoints, rate limits,
and inaccessible pages SHALL be recorded as receipts, not promoted findings.

Exp6324 SHALL deduplicate candidates against all earlier
`research-references.md` planner markers and against repeated source ids,
URLs, titles, and content hashes in the current sweep. Exp6324 SHALL accept
only stable, non-duplicate, reproducible, primary or first-party evidence that
is strictly later than the V545 marker and that changes a local executable
contract. A zero-source delta SHALL be terminal. In that case
`accepted_count` SHALL be the bare integer `0`, `accepted_findings` SHALL be
empty, `research-references.md` SHALL remain byte-identical, and
`honest_verdict` SHALL start with `complete_null:`.

Exp6324 SHALL freeze contracts for the restricted policy DSL, exact factor
guard, verified fallback, blind checker, anytime-valid certificate, exact
counterexample update, protected validation, ARC influence no-solve path, and
hardware boundary. The contracts SHALL explicitly exclude hidden states,
activations, embeddings, prefix trajectories, pooled representation rescue,
external generated-text scorers, masked-model energy, best-of-N text judges,
KAN experiments, cross-family transfer, GGUF weight updates, public ARC
re-solves, TSU execution, KV260 tasks, PolarFire dependencies, flash,
synthesis, place and route, timing, and any GateMate command beyond one
non-destructive detect.

The Exp6324 artifact SHALL be written atomically to
`results/experiment_6324_v545_post_marker_source_scope_freeze.json` with
`inference_substrate=web_and_bibliographic_search_only` and
`verifier_is_oracle=false`. It SHALL not modify `scripts/research_conductor.py`.

The Exp6324 artifact SHALL include these required fields: `status`,
`v545_marker_text_and_line`, `search_window_start_utc`,
`search_completed_utc`, `source_queries_by_channel`, `source_receipts`,
`accepted_findings`, `accepted_count`, `duplicate_findings`,
`watch_only_findings`, `inaccessible_sources`,
`excluded_findings_and_reasons`,
`semantic_scholar_ebt_and_arm_ebm_receipts`, `extropic_status`,
`logical_intelligence_status`, `github_status`,
`frozen_restricted_policy_contract`,
`frozen_exact_factor_guard_contract`,
`frozen_verified_fallback_contract`, `frozen_blind_checker_contract`,
`frozen_anytime_certificate_contract`,
`frozen_counterexample_update_contract`,
`frozen_protected_validation_contract`,
`frozen_arc_influence_no_solve_contract`,
`frozen_hardware_contract`, `roadmap_scope_delta`,
`protected_files_unchanged`, `preconditions_checked`,
`inference_substrate`, `verifier_is_oracle`, `field_provenance`,
`field_principles`, `test_commands`, `test_exit_codes`, `duration_s`,
`reproducibility_checksum`, and `honest_verdict`.

### SCENARIO-INFRA-6324-1: Marker Bound Is Exclusive

GIVEN the sealed V545 planner refresh marker in `research-references.md`
WHEN Exp6324 classifies a candidate at or before the marker commit time
THEN the candidate is rejected, and a bare same-day date is rejected unless a
strictly later timestamp is present.

### SCENARIO-INFRA-6324-2: Zero Findings Preserve References

GIVEN every source channel returns no strictly later stable evidence,
duplicate evidence, inaccessible evidence, pre-marker evidence, or watch-only
evidence
WHEN Exp6324 writes its artifact
THEN `accepted_count` is the bare integer `0`, `accepted_findings` is empty,
`research-references.md` is byte-identical, and `honest_verdict` starts with
`complete_null:`.

### SCENARIO-INFRA-6324-3: Dedupe And Scope Hashes Fail Closed

GIVEN a candidate has no stable URL, repeats an earlier source id, repeats an
existing reference block, repeats a current-sweep content hash, lacks a local
executable consequence, or changes protected input hashes
WHEN Exp6324 deduplicates and validates the sweep
THEN the row is not accepted and the protected hash ledger records the exact
before and after state.

### SCENARIO-INFRA-6324-4: Frozen Contracts Preserve V545 Boundaries

GIVEN V545 depends on observable restricted programs, exact factors, fallback,
blind checking, anytime certificates, exact counterexamples, protected
validation, ARC influence without solve credit, and one GateMate detect
WHEN Exp6324 serializes frozen contracts
THEN each contract has a stable version, required boundary fields, explicit
retired-mechanism exclusions, and no hidden-state, external-scorer, KAN,
cross-family, ARC-solve, TSU, KV260, PolarFire, flash, synthesis, timing, or
extra-board-command promotion path.

### SCENARIO-INFRA-6324-5: Artifact Schema Is Principle Annotated

GIVEN source receipts, protected hashes, field principles, and command
receipts
WHEN Exp6324 validates the report before writing
THEN every required field is present, every field has provenance and a
principle, `verifier_is_oracle=false`, the checksum matches the normalized
payload, and the honest verdict has a terminal prefix.

## Implementation Status (REQ-INFRA-6324)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6324 | Pending implementation: `python/carnot/experiment_6324_v545_post_marker_source_scope_freeze.py`; terminal artifact `results/experiment_6324_v545_post_marker_source_scope_freeze.json`. | Pending focused tests: `tests/python/test_experiment_6324_v545_post_marker_source_scope_freeze.py`. |

## REQ-INFRA-6337: V546 Bounded Terminal Handoff SHALL Preserve Missing V545 Evidence

Carnot SHALL build Exp6337 as a bounded terminal handoff from milestone
`2026.08.545` into milestone `2026.08.546`. The handoff SHALL classify only
the seven queued V545 identities, Exp6323 through Exp6329. It SHALL use exact
declared result paths as artifact evidence. It SHALL record a missing Exp6323
artifact as missing. It SHALL not infer success from a conductor receipt.

Exp6337 SHALL extract all three Exp6323 wall-clock failure receipts from
`ops/conductor-log.md`. The receipts SHALL include the timestamp, task title,
conductor status, message text, and hard-cap seconds. Exp6337 SHALL record no
invented Exp6323 `honest_verdict` because no artifact exists.

Exp6337 SHALL prove that Exp6330 through Exp6336 were proposal-only. It SHALL
search the conductor log for queued or executed conductor rows for those ids.
It SHALL also search the active V546 roadmap. The receipt SHALL record that
the identifiers appear only in the change proposal or old Exp6323 transition
contract, not as V545 conductor tasks or active V546 task ids.

Exp6337 SHALL validate the thirteen V546 tasks in the active roadmap. It SHALL
run the repository roadmap schema, prior-failure linter, gate audit, exclusion
manifest lint, and deterministic custom checks. The custom checks SHALL reject
duplicate ids, duplicate deliverables, invalid dependencies, bad structured
gates, missing `MODEL_SPECS` obligations for live LLM tasks, incomplete
`prior_failures` entries, non-Codex routing, non-`gpt-5.5` Codex model
routing, and prompts that do not end with
`Do NOT push. Do NOT modify scripts/research_conductor.py.`

The Exp6337 artifact SHALL be written atomically to
`results/experiment_6337_v546_bounded_terminal_handoff.json`. It SHALL include
these required fields: `status`, `v545_milestone_and_queue_hash`,
`queued_v545_task_ids`, `terminal_v545_artifacts_by_task`,
`missing_artifacts_by_task`, `exp6323_failure_receipts`,
`proposal_only_exp6330_through_exp6336_receipt`,
`v545_scientific_terminal_states`, `v545_hardware_terminal_state`,
`v546_milestone_and_doc_hash`, `v546_task_ids`,
`v546_id_collision_check`, `v546_deliverable_checks`,
`v546_dependency_checks`, `v546_structured_gate_checks`,
`v546_prior_failure_checks`, `v546_llm_model_policy_checks`,
`prompt_contract_checks`, `protected_files_unchanged`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`llm_call_count`, `field_provenance`, `field_principles`,
`test_commands`, `test_exit_codes`, `duration_s`,
`reproducibility_checksum`, and `honest_verdict`.
`llm_call_count` SHALL be the bare integer `0`. `verifier_is_oracle` SHALL be
the bare boolean `false`. `field_principles` SHALL cover every required field.

### SCENARIO-INFRA-6337-1: Missing Exp6323 Stays Missing

GIVEN the V545 queue includes Exp6323 through Exp6329
WHEN Exp6337 classifies exact result paths
THEN Exp6323 appears in `missing_artifacts_by_task`, no substitute path is
used, and Exp6324 through Exp6329 keep their exact terminal classifications.

### SCENARIO-INFRA-6337-2: Wall-Clock Receipts Stay Separate

GIVEN the conductor log has three Exp6323 hard wall-clock failures
WHEN Exp6337 parses conductor receipts
THEN all three rows are recorded with timestamps and hard-cap seconds, and no
Exp6323 artifact verdict is fabricated.

### SCENARIO-INFRA-6337-3: Proposal-Only Ghost IDs Do Not Enter The Queue

GIVEN Exp6330 through Exp6336 occur in proposal text but not the V545 queue
WHEN Exp6337 searches conductor and active-roadmap evidence
THEN the receipt reports zero queued or executed conductor tasks for those ids
and no active V546 task-id reuse.

### SCENARIO-INFRA-6337-4: V546 Identity And Graph Checks Fail Closed

GIVEN the active V546 roadmap declares thirteen tasks
WHEN Exp6337 validates ids, deliverables, dependencies, and structured gates
THEN duplicates, missing dependencies, self-dependencies, retired dependencies,
unknown gate upstreams, and gate fields absent from upstream required-field
blocks are reported as failures.

### SCENARIO-INFRA-6337-5: Model, Prior-Failure, And Prompt Contracts Are Enforced

GIVEN V546 tasks include model routing, live LLM obligations, prior failure
entries, and prompt endings
WHEN Exp6337 validates the roadmap
THEN every Codex task uses `gpt-5.5`, live LLM tasks name the required
`MODEL_SPECS` obligations, prior failures have all required subfields, and
malformed endings fail the prompt contract check.

### SCENARIO-INFRA-6337-6: Output Is Atomic, Checksummed, And Non-Mutating

GIVEN the V545 evidence and V546 validation receipts are computed in memory
WHEN Exp6337 writes its artifact
THEN every required field is present, field provenance and field principles
cover all required fields, the checksum matches the normalized payload,
protected hashes remain unchanged, and the active roadmap and conductor are
not edited.

## Implementation Status (REQ-INFRA-6337)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6337 | Implemented: `python/carnot/experiment_6337_v546_bounded_terminal_handoff.py`; terminal artifact `results/experiment_6337_v546_bounded_terminal_handoff.json`. | Implemented: `tests/python/test_experiment_6337_v546_bounded_terminal_handoff.py`. |

## REQ-INFRA-6338: V546 Source Freeze SHALL Validate Planner Receipts And Keep Scope Closed

Carnot SHALL build Exp6338 as a deterministic V546 post-marker source sweep
and scope freeze. The sweep SHALL hash the exact
`<!-- V546-PLANNER-REFRESH-20260812-END -->` marker in
`research-references.md`. It SHALL record the marker line. It SHALL use the
marker commit time as the exclusive lower bound for post-marker novelty.

Exp6338 SHALL validate the five V546 planner-promoted source families with
direct receipts, first-publication dates, access times, and local executable
consequences. The source families are parser-state bias correction, LeJIT
prefix enforcement, NxN e-values, catastrophic remembering, and verification
cost. The receipts SHALL remain separate from any post-marker acceptance count.

Exp6338 SHALL repeat arXiv, OpenReview, Hugging Face Papers, Semantic Scholar
EBT and ARM-EBM routes, Extropic, Logical Intelligence, and GitHub searches
only for the post-marker window. HTTP failures, browser challenges, empty
endpoints, rate limits, and inaccessible pages SHALL be recorded as receipts.
They SHALL not become promoted findings.

Exp6338 SHALL deduplicate by paper identity, repository identity, mechanism,
content hash, URL, title, and already-retired Carnot scope. It SHALL accept
only stable, non-duplicate, reproducible, primary or first-party evidence that
is strictly later than the V546 marker and that changes a local executable
contract. A zero-source delta SHALL be terminal. In that case `accepted_count`
SHALL be the bare integer `0`, `promoted_findings` SHALL preserve only the
planner-promoted receipts, and `honest_verdict` SHALL start with
`complete_null:`.

Exp6338 SHALL freeze exactly three V546 lanes: prefix-constrained policy
generation, certified factor evolution, and ARC action influence. It SHALL also
freeze the exact-oracle boundary, mandatory local GGUF policy, fail-fast gates,
and no-hardware rule. The task SHALL not execute GateMate, KV260, TSU, Kona, or
board commands. The task SHALL not modify `scripts/research_conductor.py`.

The Exp6338 artifact SHALL be written atomically to
`results/experiment_6338_v546_post_marker_source_scope_freeze.json` with
`inference_substrate=web_and_bibliographic_search_only`,
`verifier_is_oracle=false`, and `llm_call_count=0`.

The Exp6338 artifact SHALL include these required fields: `status`,
`v546_marker_text_line_and_hash`, `search_window_start_utc`,
`search_completed_utc`, `source_queries_by_channel`, `source_receipts`,
`promoted_findings`, `accepted_count`, `duplicate_findings`,
`watch_only_findings`, `inaccessible_sources`,
`excluded_findings_and_reasons`, `parser_bias_receipt`, `lejit_receipt`,
`nxn_evalue_receipt`, `catastrophic_remembering_receipt`,
`verification_cost_receipt`,
`semantic_scholar_ebt_and_arm_ebm_receipts`,
`openreview_and_huggingface_status`, `github_status`, `extropic_status`,
`logical_intelligence_status`, `frozen_prefix_generation_contract`,
`frozen_certified_learning_contract`, `frozen_arc_influence_contract`,
`frozen_model_policy`, `frozen_hardware_nonuse_contract`,
`roadmap_scope_delta`, `protected_files_unchanged`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`llm_call_count`, `field_provenance`, `field_principles`, `test_commands`,
`test_exit_codes`, `duration_s`, `reproducibility_checksum`, and
`honest_verdict`.

### SCENARIO-INFRA-6338-1: Marker Bound Is Exclusive

GIVEN the sealed V546 planner refresh marker in `research-references.md`
WHEN Exp6338 classifies a candidate at or before the marker commit time
THEN the candidate is rejected, and a bare same-day date is rejected unless a
strictly later timestamp is present.

### SCENARIO-INFRA-6338-2: Promoted Source Dates Are Direct Receipts

GIVEN the V546 marker promoted parser-state correction, LeJIT, NxN e-values,
catastrophic remembering, and verification-cost reporting
WHEN Exp6338 builds its artifact
THEN each promoted source has a direct URL, first-publication date, access
time, and local executable consequence.

### SCENARIO-INFRA-6338-3: Dedupe And Source Dispositions Fail Closed

GIVEN a candidate repeats an older paper identity, repository identity,
mechanism, content hash, URL, title, or retired Carnot scope
WHEN Exp6338 partitions the sweep
THEN the row is not accepted, and its duplicate, watch-only, inaccessible, or
excluded disposition records the exact reason.

### SCENARIO-INFRA-6338-4: Frozen V546 Contracts Preserve Boundaries

GIVEN V546 has three scientific lanes plus exact-oracle, local-GGUF,
fail-fast, and no-hardware boundaries
WHEN Exp6338 serializes frozen contracts
THEN the artifact admits only prefix-constrained generation, certified factor
evolution, and ARC action influence. It SHALL reject solve credit, hidden
state probes, external scorers, GGUF weight updates, and hardware execution.

### SCENARIO-INFRA-6338-5: Output Is Principle Annotated And Non-Mutating

GIVEN source receipts, protected hashes, field principles, and command
receipts
WHEN Exp6338 validates the report before writing
THEN every required field is present, every field has provenance and a
principle, `accepted_count` and `llm_call_count` are bare integers, the
checksum matches the normalized payload, protected hashes remain unchanged,
and the honest verdict has a terminal prefix.

## Implementation Status (REQ-INFRA-6338)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6338 | Implemented: `python/carnot/experiment_6338_v546_post_marker_source_scope_freeze.py`; terminal artifact `results/experiment_6338_v546_post_marker_source_scope_freeze.json`. | Implemented: `tests/python/test_experiment_6338_v546_post_marker_source_scope_freeze.py`. |

## REQ-INFRA-6349: V546 Capstone SHALL Recompute Terminal Evidence Without Claim Laundering

Carnot SHALL build Exp6349 as a deterministic V546 adversarial capstone over
the 13 declared tasks in milestone `2026.08.546`. The capstone SHALL parse the
active roadmap task specs. It SHALL match each declared deliverable to its exact
artifact path or to an explicit conductor failure or structured gate skip. It
SHALL accept complete, null, blocked, failed, flagged, retired, and skipped
states as evidence classes. It SHALL not rewrite those states.

Exp6349 SHALL recompute every dependency and structured gate from upstream
artifact fields. A failed gate SHALL preserve the skipped downstream artifact.
It SHALL not promote a skipped task through conductor receipts. It SHALL not
turn exact-oracle checks into learned verifiers. It SHALL not count safety-only
evidence as utility. It SHALL not count ARC route reachability as action
influence. It SHALL not count action influence as an ARC solve.

Exp6349 SHALL audit mandatory local GGUF use, embedded llama.cpp tokenizer
receipts, GPU memory release receipts, source-model weight immutability,
generated-label counts, hidden-state access counts, verification-cost
accounting, certified-learning chronology, e-process validity, factor lifecycle
bounds, rollback identity, ARC solve provenance, ARC registry immutability, and
hardware non-use. The capstone itself SHALL use
`inference_substrate=aggregation_from_upstream_artifacts_no_llm`,
`verifier_is_oracle=false`, and bare integer `llm_call_count=0`.

The Exp6349 artifact SHALL be written atomically to
`results/experiment_6349_v546_adversarial_capstone.json` and SHALL include
these required fields: `status`, `milestone_and_roadmap_hash`,
`declared_task_ids_and_deliverables`,
`conductor_terminal_receipts_by_task`,
`artifact_existence_hash_schema_status_and_honest_verdict_by_task`,
`dependency_recomputation`, `structured_gate_recomputation`,
`skipped_task_handling`, `prior_failure_and_retirement_audit`,
`exclusion_manifest_audit`, `prompt_contract_audit`,
`required_field_and_field_principle_audit`,
`model_policy_and_MODEL_SPECS_audit`,
`llama_cpp_embedded_tokenizer_audit`,
`gpu_offload_and_memory_release_audit`,
`source_model_weight_mutation_audit`,
`generated_label_and_hidden_state_audit`,
`exact_oracle_and_learned_claim_boundary_audit`,
`prefix_generation_determination`,
`certified_continuous_learning_determination`,
`eprocess_and_factor_lifecycle_determination`,
`safety_audit_determination`, `arc_action_influence_determination`,
`solve_provenance_audit`, `arc_registry_immutability_audit`,
`hardware_nonuse_and_inference_substrate_audit`,
`verification_cost_accounting_audit`, `three_gap_closure_matrix`,
`prd_requirement_mapping`, `protected_files_changed_with_reasons`,
`docs_and_archive_reconciliation`, `preconditions_checked`,
`inference_substrate`, `verifier_is_oracle`, `llm_call_count`,
`field_provenance`, `field_principles`, `test_commands`, `test_exit_codes`,
`duration_s`, `reproducibility_checksum`, and `honest_verdict`.

### SCENARIO-INFRA-6349-1: Exact Artifacts And Receipts Define The Terminal Matrix

GIVEN the active V546 roadmap declares Exp6337 through Exp6349
WHEN Exp6349 builds the terminal matrix
THEN each task row records the exact deliverable path, hash state, schema
state, status, honest verdict, terminal class, and conductor receipts. The
capstone row records that its own output hash is self-referential and excluded.

### SCENARIO-INFRA-6349-2: Structured Gates Are Recomputed From Raw Fields

GIVEN the roadmap declares gates for Exp6340, Exp6341, Exp6345, and Exp6348
WHEN Exp6349 evaluates those gates
THEN each gate row records the upstream task, artifact field, expected value,
actual value, pass state, and skip effect. Exp6341 remains a structured skip
when Exp6340 `semantic_diversity_gain_score` is `0.0`.

### SCENARIO-INFRA-6349-3: Model, Oracle, And Mutation Boundaries Stay Separate

GIVEN local-GGUF tasks and exact-oracle tasks exist upstream
WHEN Exp6349 audits the evidence
THEN mandatory GGUF ids, embedded tokenizer receipts, GPU release receipts,
zero source-weight mutation, zero generated labels, zero hidden-state access,
and upstream oracle identities are reported without setting the capstone's
`verifier_is_oracle` to true.

### SCENARIO-INFRA-6349-4: Three Gap Closure Matrix Does Not Overclaim

GIVEN V546 has a prefix-generation branch, certified-learning branch, and ARC
action-influence branch
WHEN Exp6349 writes the closure matrix
THEN prefix generation remains null or skipped after the failed canary gate,
certified learning closes only inside the exact-release boundary, ARC action
influence closes only as no-solve action influence, and every hardware
boundary remains non-use.

### SCENARIO-INFRA-6349-5: Output Is Principle Annotated And Failure Aware

GIVEN command receipts and protected-file hashes are available
WHEN Exp6349 validates the report
THEN every required field has provenance and a principle, `llm_call_count` is
bare `0`, `verifier_is_oracle` is false, nonzero command exits force a blocked
verdict, and the checksum matches the normalized payload.

## Implementation Status (REQ-INFRA-6349)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6349 | Pending implementation: `python/carnot/experiment_6349_v546_adversarial_capstone.py`; terminal artifact `results/experiment_6349_v546_adversarial_capstone.json`. | Pending focused tests: `tests/python/test_experiment_6349_v546_adversarial_capstone.py`. |

## REQ-INFRA-6350: V547 Handoff SHALL Preserve V546 Evidence Boundaries

Carnot SHALL build Exp6350 as a bounded V546-to-V547 evidence handoff over
exactly Exp6337 through Exp6349. The handoff SHALL parse the active V547
roadmap, the V547 proposal document, the conductor log, and the exact V546
artifact paths. It SHALL not edit `research-roadmap.yaml`,
`openspec/change-proposals/research-roadmap-vNEXT.md`, or
`scripts/research_conductor.py`.

Exp6350 SHALL preserve the Exp6337 adversarial flag as a flag. It SHALL copy
the conductor and artifact flag text into the result. It SHALL not promote that
artifact to a clean terminal record. It SHALL keep explicit gate blocks,
including Exp6341, separate from missing artifacts.

Exp6350 SHALL classify each V546 task by inference substrate. The classes SHALL
separate live autoregressive generation, tokenizer-only model access,
deterministic replay, synthetic replay, exact-oracle deterministic checking,
web or bibliographic search, and artifact aggregation. Exp6344 and Exp6345
SHALL be marked as no-live-autoregressive-generation evidence.

Exp6350 SHALL record the closed parser/JIT lane, the qualified certified
factor-learning evidence, the open live-generation and consumer gaps, and the
ARC no-solve boundary. It SHALL validate all 13 V547 task identities,
deliverables, dependencies, structured gates, prior-failure entries, model
obligations, and prompt endings.

The Exp6350 artifact SHALL be written atomically to
`results/experiment_6350_v547_bounded_terminal_handoff.json` and SHALL include
these required fields: `status`, `v546_milestone_and_queue_hash`,
`queued_v546_task_ids`, `terminal_v546_artifacts_by_task`,
`blocked_v546_tasks`, `flagged_v546_artifacts_and_reasons`,
`inference_substrate_classification_by_task`,
`live_autoregressive_generation_by_task`, `v546_scientific_terminal_states`,
`closed_parser_jit_receipt`, `qualified_certified_learning_receipt`,
`open_live_generation_and_consumer_gaps`, `arc_no_solve_receipt`,
`v547_milestone_and_doc_hash`, `v547_task_ids`,
`v547_id_collision_check`, `v547_deliverable_checks`,
`v547_dependency_checks`, `v547_structured_gate_checks`,
`v547_prior_failure_checks`, `v547_llm_model_policy_checks`,
`prompt_contract_checks`, `protected_files_unchanged`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`llm_call_count`, `field_provenance`, `field_principles`,
`test_commands`, `test_exit_codes`, `duration_s`, `random_seeds`,
`reproducibility_checksum`, and `honest_verdict`.

### SCENARIO-INFRA-6350-1: V546 Denominator Is Exact

GIVEN the active V546 evidence set is Exp6337 through Exp6349
WHEN Exp6350 builds its terminal matrix
THEN each task maps to one exact artifact or an explicit gate block. Missing or
extra experiment numbers are reported as failures.

### SCENARIO-INFRA-6350-2: Exp6337 Flag Is Never Promoted Clean

GIVEN Exp6337 has a terminal artifact and adversarial flag text
WHEN Exp6350 writes flagged evidence
THEN the flag text is preserved, `clean_promotion_attempted` is false, and the
task terminal class remains flagged.

### SCENARIO-INFRA-6350-3: Substrate Classes Separate Model Access

GIVEN V546 artifacts contain live inference, tokenizer-only, deterministic
replay, synthetic replay, and aggregation receipts
WHEN Exp6350 classifies substrates
THEN Exp6344 and Exp6345 record
`live_autoregressive_generation_invoked=false`, and wrong classifications fail
validation.

### SCENARIO-INFRA-6350-4: V547 Roadmap Checks Fail Closed

GIVEN the active V547 roadmap declares 13 tasks
WHEN Exp6350 validates identities, deliverables, dependencies, gates,
prior-failure entries, model obligations, and prompt endings
THEN duplicate ids, bad gates, missing model obligations, and malformed prompt
endings are reported as failed checks.

### SCENARIO-INFRA-6350-5: Boundary Receipts Do Not Overclaim

GIVEN Exp6340, Exp6341, Exp6342 through Exp6346, and Exp6348 define V546
scientific states
WHEN Exp6350 summarizes boundaries
THEN parser/JIT is closed, factor-learning evidence is qualified by replay and
tokenizer-only limits, live generation and consumer value remain open, and ARC
has no solve claim.

### SCENARIO-INFRA-6350-6: Output Is Principle Annotated And Non-Mutating

GIVEN field principles, command receipts, protected hashes, and random seeds
WHEN Exp6350 validates its report
THEN every required field has provenance and a one-line principle,
`llm_call_count` is bare `0`, `verifier_is_oracle` is false, the checksum
matches, protected files remain unchanged, and the honest verdict has a
terminal prefix.

## Implementation Status (REQ-INFRA-6350)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6350 | Planned: `python/carnot/experiment_6350_v547_bounded_terminal_handoff.py`; terminal artifact `results/experiment_6350_v547_bounded_terminal_handoff.json`. | Planned: `tests/python/test_experiment_6350_v547_bounded_terminal_handoff.py`. |

## REQ-INFRA-6363: V548 Handoff SHALL Preserve V547 Terminal Boundaries And Fail Closed On Queue Drift

Carnot SHALL build Exp6363 as a bounded V547-to-V548 evidence handoff over
exactly Exp6350 through Exp6356. The handoff SHALL run read-only
preconditions before importing artifact fields. It SHALL hash the active
roadmap, the optional next roadmap, the conductor log, and the exclusion
manifest. It SHALL classify each expected V547 artifact path as present,
absent, blocked, flagged, retired-upstream, or null.

Exp6363 SHALL read each present V547 artifact through
`scripts/summarize_artifact.py` before importing any field from that artifact.
It SHALL preserve live adversarial findings separately from stamped fields.
It SHALL not convert flagged, blocked, missing, retired-upstream, or null
evidence into clean evidence. It SHALL not count the proposal-only Exp6357
through Exp6362 identities as executed V547 tasks.

Exp6363 SHALL reconcile conductor rows for Exp6350 through Exp6356. It SHALL
record Exp6350 as flagged, Exp6353 as three gate failures, Exp6354 as three
pre-emptive upstream-retirement skips with no artifact, Exp6355 as three gate
failures on the missing Exp6354 deliverable, and Exp6356 as terminal
`complete_null`.

Exp6363 SHALL compare Exp6352 source constants with terminal process receipts.
It SHALL record the source/artifact `n_ctx` mismatch, nonzero child exit codes,
zero raw bytes, zero prompt/completion tokens, empty `models_used`, missing
stderr, missing top-level `random_seed`, and prose-versus-boolean generation
contradiction. It SHALL not infer a root cause.

Exp6363 SHALL validate the V548 queue without changing it. It SHALL require
Exp6363 through Exp6376, fourteen unique task IDs, unique result JSON
deliverables, ordered dependencies, structured gates whose fields appear in the
upstream required artifact fields, complete prior-failure entries with
`retire_if_same_verdict: true`, coherent agent/model pairs, mandatory GGUF
policy, and prompt contracts that include the project root, date, run command,
and exact final prohibition line. If `research-roadmap-next.yaml` is absent, it
SHALL record that absence and audit the active roadmap without inventing staged
content.

The Exp6363 artifact SHALL be written atomically to
`results/experiment_6363_v548_terminal_handoff_and_queue_preflight.json` with
`inference_substrate=deterministic_repository_evidence_handoff_no_llm` and
`verifier_is_oracle=false`. The artifact SHALL include these required fields:
`status`, `v547_active_roadmap_path_and_hash`, `v547_active_task_ids`,
`v547_terminal_artifacts_by_task`,
`v547_conductor_outcomes_and_attempt_counts`,
`v547_flagged_blocked_missing_and_null_states`,
`exp6352_generation_failure_receipt`,
`exp6352_source_artifact_drift_receipt`,
`proposal_only_v547_ids_not_executed`,
`v548_milestone_doc_and_queue_hashes`, `v548_task_ids`,
`v548_id_collision_check`, `v548_deliverable_checks`,
`v548_dependency_and_structured_gate_checks`,
`v548_gate_field_cross_reference_checks`, `v548_prior_failure_checks`,
`v548_agent_model_and_llm_policy_checks`, `prompt_contract_checks`,
`active_roadmap_modified`, `conductor_modified`, `protected_files_unchanged`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`field_principles`, `field_provenance`, `random_seed`, `duration_s`,
`tests_run`, `reproducibility_checksum`, and `honest_verdict`.

### SCENARIO-INFRA-6363-1: V547 Denominator And States Stay Exact

GIVEN V547 has seven active tasks
WHEN Exp6363 builds the terminal matrix
THEN only Exp6350 through Exp6356 are counted as executed
AND flagged, blocked, missing, retired-upstream, and null states remain
separate.

### SCENARIO-INFRA-6363-2: Exp6352 Drift Is Recorded Without Diagnosis

GIVEN Exp6352 has model receipts but no generated bytes
WHEN Exp6363 compares source and artifact evidence
THEN the `n_ctx` mismatch, empty outputs, missing stderr, missing top-level
seed, and generation contradiction are recorded without a root-cause claim.

### SCENARIO-INFRA-6363-3: V548 Queue Requires Fourteen Tasks

GIVEN the V548 proposal reserves Exp6363 through Exp6376
WHEN Exp6363 validates the staged or active queue
THEN any queue with fewer than fourteen exact task IDs records missing IDs and
uses a terminal blocked verdict.

### SCENARIO-INFRA-6363-4: Gates, Priors, Routing, And Prompts Fail Closed

GIVEN a V548 task declares gates, prior failures, models, or prompt contracts
WHEN Exp6363 validates the queue
THEN malformed gates, missing upstream fields, incomplete prior entries,
bad agent/model pairs, missing mandated GGUF IDs, and prompt contract gaps are
reported as failed checks.

### SCENARIO-INFRA-6363-5: Artifact Is Annotated, Checksummed, And Non-Mutating

GIVEN precondition hashes, summary receipts, queue checks, and command receipts
are assembled
WHEN Exp6363 writes the artifact
THEN every required field and structured gate expression has a principle,
every required field has provenance, protected files remain byte-identical,
the checksum matches, and `verifier_is_oracle` is false.

## Implementation Status (REQ-INFRA-6363)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6363 | Implemented: `python/carnot/experiment_6363_v548_terminal_handoff_and_queue_preflight.py`; terminal artifact `results/experiment_6363_v548_terminal_handoff_and_queue_preflight.json`. | Implemented: `tests/python/test_experiment_6363_v548_terminal_handoff_and_queue_preflight.py`. |

## REQ-INFRA-6377: V549 Handoff SHALL Preserve The Four-Task V548 Boundary And Validate The Full V549 Queue

Carnot SHALL build Exp6377 as a bounded V548-to-V549 evidence handoff over
exactly Exp6363 through Exp6366. The handoff SHALL hash the active roadmap,
optional next roadmap, milestone document, conductor source, conductor log,
exclusion manifest, and the four expected V548 artifacts before importing
artifact fields. It SHALL read each present V548 artifact through
`scripts/summarize_artifact.py` and keep live adversarial findings separate
from stamped artifact fields.

Exp6377 SHALL preserve the V548 evidence classes without promotion. It SHALL
record Exp6363 as blocked, Exp6364 and Exp6365 as clean terminal evidence, and
Exp6366 as a null transport failure with three nonempty raw outputs, zero
parse-valid factor edits, and zero exact-checker calls. It SHALL record
Exp6367 through Exp6376 as proposal-only V548 identities. These identities
SHALL not count as executed, blocked, or missing active V548 tasks.

Exp6377 SHALL validate the V549 queue without changing it. It SHALL require
Exp6377 through Exp6390, fourteen unique task IDs, unique result JSON
deliverables, ascending execution order, ordered dependencies, structured
gates, gate fields that appear in upstream required artifact fields, complete
prior-failure entries with `retire_if_same_verdict: true`, and no dependency
or gate reference to a retired upstream task ID. It SHALL run or mirror the
roadmap schema, prior-failure, gate-audit, and exclusion-manifest checks.

Exp6377 SHALL render queue prompts with the same `{project_root}` and `{date}`
format contract used by the conductor before checking prompt contents. Each
rendered prompt SHALL include `CONTEXT`, `EXISTING CODE TO READ FIRST`,
`TASK`, `CONCRETE STEPS`, the project root, the planning date, a `Run command:`
line, and the exact final prohibition line
`Do NOT push. Do NOT modify scripts/research_conductor.py.`. Each LLM task
SHALL declare `MODEL_SPECS`, at least one mandated local GGUF ID, embedded
GGUF tokenizer use, and a no-`AutoTokenizer` rule. Agent/model pairs SHALL be
coherent.

The Exp6377 artifact SHALL be written atomically to
`results/experiment_6377_v549_terminal_handoff_and_queue_preflight.json` with
`inference_substrate=aggregation_from_upstream_artifacts` and
`verifier_is_oracle=false`. The artifact SHALL include these required fields:
`status`, `v548_active_roadmap_path_and_hash`, `v548_active_task_ids`,
`v548_terminal_artifacts_by_task`, `v548_conductor_outcomes`,
`v548_blocked_null_clean_and_proposal_only_states`,
`exp6365_runtime_boundary`, `exp6366_transport_failure_boundary`,
`v549_milestone_doc_and_queue_hashes`, `v549_task_ids`,
`v549_id_and_deliverable_checks`, `v549_dependency_and_gate_checks`,
`v549_gate_field_cross_reference_checks`, `v549_prior_failure_checks`,
`v549_exclusion_manifest_checks`, `v549_agent_model_and_llm_policy_checks`,
`prompt_contract_checks`, `active_roadmap_modified`, `conductor_modified`,
`protected_files_unchanged`, `preconditions_checked`, `inference_substrate`,
`verifier_is_oracle`, `field_principles`, `field_provenance`, `random_seed`,
`duration_s`, `tests_run`, `reproducibility_checksum`, and
`honest_verdict`.

### SCENARIO-INFRA-6377-1: V548 Active Boundary Is Four Tasks

GIVEN V548 executed Exp6363 through Exp6366
WHEN Exp6377 builds the terminal matrix
THEN only those four tasks are counted as active V548 tasks
AND Exp6367 through Exp6376 are recorded as proposal-only.

### SCENARIO-INFRA-6377-2: Runtime And Transport Boundaries Stay Separate

GIVEN Exp6365 is runtime-ready and Exp6366 is transport-null
WHEN Exp6377 reconciles the two artifacts
THEN complete GGUF execution is recorded separately from parse-valid output,
source binding, and exact-checker calls.

### SCENARIO-INFRA-6377-3: V549 Queue Contains Fourteen Ordered Tasks

GIVEN the V549 roadmap declares Exp6377 through Exp6390
WHEN Exp6377 validates IDs, deliverables, order, dependencies, and gates
THEN any duplicate, missing, extra, unordered, malformed, or retired-upstream
reference fails the corresponding structured check.

### SCENARIO-INFRA-6377-4: Prompt And LLM Policy Contracts Are Rendered

GIVEN roadmap prompts use `{project_root}` and `{date}` placeholders
WHEN Exp6377 validates prompt contracts
THEN it renders prompts with the conductor-compatible values and checks
sections, run command, final prohibition, model IDs, embedded tokenizer rules,
no-`AutoTokenizer` rules, and agent/model pairs.

### SCENARIO-INFRA-6377-5: Artifact Is Annotated, Checksummed, And Non-Mutating

GIVEN precondition hashes, summary receipts, queue checks, and command receipts
are assembled
WHEN Exp6377 writes the artifact
THEN every required field and structured gate expression has a principle,
every required field has provenance, protected files remain byte-identical,
the checksum matches, `random_seed` is null, and `verifier_is_oracle` is false.

## Implementation Status (REQ-INFRA-6377)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6377 | Implemented: `python/carnot/experiment_6377_v549_terminal_handoff_and_queue_preflight.py`; terminal artifact `results/experiment_6377_v549_terminal_handoff_and_queue_preflight.json`. | Implemented: `tests/python/test_experiment_6377_v549_terminal_handoff_and_queue_preflight.py`. |

## REQ-INFRA-6391: V550 Handoff SHALL Preserve V549 Terminal Evidence And Preflight The Active V550 Queue

Carnot SHALL build Exp6391 as a bounded V549-to-V550 evidence handoff over
Exp6377 through Exp6390. The handoff SHALL hash the active roadmap, optional
next roadmap, milestone document, conductor log, conductor source, exclusion
manifest, known issues, and every expected V549 terminal artifact before it
imports artifact fields. It SHALL invoke `scripts/summarize_artifact.py` for
Exp6377 through Exp6390 and preserve absent, complete, positive, null, blocked,
flagged, skipped, and retired states without filling gaps.

Exp6391 SHALL keep artifact verdicts, conductor outcomes, adversarial flags,
and per-task durations as separate facts. It SHALL derive each duration from
the matching terminal artifact receipt. It SHALL not copy one aggregate
milestone duration onto tasks. If a repeated unresolved artifact-versus-
conductor mismatch is present, the handoff SHALL record the retirement trigger
instead of resolving it silently.

Exp6391 SHALL record the factor boundary exactly: Exp6379 contract-ready,
Exp6380 global-null, two qualified Gemma family observations, Qwen invalid,
Exp6381 blocked, Exp6382 absent or blocked, Exp6383 positive rollback control,
and Exp6384 blocked. It SHALL record the ARC boundary exactly: Exp6386 through
Exp6388 ready, Exp6388's scalar detail still nested for
`delta_admission_precision`, and Exp6389 `blocked_gate_check_failed`.

Exp6391 SHALL validate the active V550 queue without changing it. It SHALL
require exactly thirteen unique task IDs from Exp6391 through Exp6403, unique
result JSON deliverables, ascending execution order, structured gates,
required upstream gate fields, complete prior-failure entries, no retired-ID
reuse, and no retired `requires` chain. It SHALL validate the capability
license gates and ARC metric gates before later tasks run.

Exp6391 SHALL render each prompt with the conductor-compatible `{project_root}`
and `{date}` values before checking it. Every rendered prompt SHALL include
`CONTEXT`, `EXISTING CODE TO READ FIRST`, `TASK`, `CONCRETE STEPS`, the project
root, planning date `20260813`, a `Run command:` line, and the final line
`Do NOT push. Do NOT modify scripts/research_conductor.py.`. Every task SHALL
use `agent_type=codex` and `model=gpt-5.5`. Every LLM task SHALL name
`MODEL_SPECS`, `cached_sota_pair()`, at least one mandated GGUF ID, embedded
GGUF tokenizers, no `AutoTokenizer`, and no legacy headline cell.

The Exp6391 artifact SHALL be written atomically to
`results/experiment_6391_v550_terminal_handoff_and_queue_preflight.json` with
`inference_substrate=aggregation_from_upstream_artifacts` and
`verifier_is_oracle=false`. The artifact SHALL include these required fields:
`status`, `v549_active_roadmap_path_and_hash`, `v549_task_ids`,
`v549_terminal_artifacts_by_task`, `v549_artifact_verdicts`,
`v549_conductor_outcomes`, `v549_adversarial_flags`,
`v549_duration_receipts_by_task`, `v549_factor_boundary`,
`v549_arc_boundary`, `v550_milestone_doc_and_queue_hashes`,
`v550_task_ids`, `v550_id_and_deliverable_checks`,
`v550_dependency_and_gate_checks`,
`v550_gate_field_cross_reference_checks`, `v550_prior_failure_checks`,
`v550_exclusion_manifest_checks`,
`v550_agent_model_and_llm_policy_checks`, `prompt_contract_checks`,
`active_roadmap_modified`, `conductor_modified`, `protected_files_unchanged`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`field_principles`, `field_provenance`, `random_seed`, `duration_s`,
`tests_run`, `reproducibility_checksum`, and `honest_verdict`.

### SCENARIO-INFRA-6391-1: V549 Evidence Classes Stay Separate

GIVEN V549 produced complete, positive, null, blocked, absent, and flagged
evidence
WHEN Exp6391 builds its terminal matrix
THEN artifact verdicts, conductor outcomes, adversarial flags, and per-task
durations are recorded in separate fields and no missing artifact is invented.

### SCENARIO-INFRA-6391-2: Factor Boundary Preserves Partial Capability

GIVEN Exp6379 is ready and Exp6380 is globally null
WHEN Exp6391 summarizes factor evidence
THEN both Gemma qualified observations, the invalid Qwen row, the blocked
Exp6381 and Exp6384 gates, the absent or blocked Exp6382 state, and the
positive Exp6383 rollback control remain explicit.

### SCENARIO-INFRA-6391-3: ARC Boundary Preserves Nested Metric Failure

GIVEN Exp6386 through Exp6388 are ready and Exp6389 is gate-blocked
WHEN Exp6391 summarizes ARC evidence
THEN Exp6388's nested `delta_admission_precision` detail is not converted into
a bare gate scalar and Exp6389 remains `blocked_gate_check_failed`.

### SCENARIO-INFRA-6391-4: V550 Queue And Gate Contracts Validate

GIVEN the active V550 roadmap declares Exp6391 through Exp6403
WHEN Exp6391 validates IDs, deliverables, order, gates, prior failures, and
retired references
THEN any duplicate, missing, extra, unordered, malformed, unstructured,
unknown-field, or retired-upstream reference fails the relevant structured
check.

### SCENARIO-INFRA-6391-5: Prompt And LLM Policy Contracts Validate

GIVEN V550 prompts use `{project_root}` and `{date}` placeholders
WHEN Exp6391 validates rendered prompt contracts
THEN it checks all required prompt sections, run commands, final prohibition,
agent/model pairs, model specs, cached SOTA access, mandated GGUF IDs,
embedded-tokenizer rules, no-`AutoTokenizer` rules, and no legacy headline
cell.

### SCENARIO-INFRA-6391-6: Artifact Is Annotated, Checksummed, And Non-Mutating

GIVEN precondition hashes, summary receipts, queue checks, and command receipts
are assembled
WHEN Exp6391 writes its artifact
THEN every required field and structured gate has a principle, every required
field has provenance classified as measured, derived, constant, or upstream,
protected files remain byte-identical, the checksum matches, `random_seed` is
null, and `verifier_is_oracle` is false.

## Implementation Status (REQ-INFRA-6391)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6391 | Planned: `python/carnot/experiment_6391_v550_terminal_handoff_and_queue_preflight.py`; terminal artifact `results/experiment_6391_v550_terminal_handoff_and_queue_preflight.json`. | Planned: `tests/python/test_experiment_6391_v550_terminal_handoff_and_queue_preflight.py`. |

## REQ-INFRA-6410: V551 Evidence Handoff And V552 Queue Preflight

Carnot SHALL build Exp6410 as the exact V551-to-V552 handoff for planning date
20260814. The experiment SHALL preserve V551 artifact verdicts, source
behavior, conductor outcomes, adversarial findings, and scientific claim
eligibility as separate fields. It SHALL not edit historical artifacts,
historical determinations, the active roadmap, the conductor, the ARC solve
registry, or any claim ledger.

Exp6410 SHALL hash the active roadmap, any next-roadmap file, the milestone
document, conductor source and log, exclusion manifest, known issues, north
star, solve registry, claim ledgers, V551 terminal artifacts, and V551
sidecars. Missing protected inputs SHALL be recorded as absent, not invented.

Exp6410 SHALL summarize Exp6404 through Exp6409 with
`scripts/summarize_artifact.py` before importing headline fields. It SHALL run
current adversarial verification for each artifact. It SHALL record Exp6407's
stamped and current adversarial flag as open. It SHALL inspect the Exp6408 and
Exp6409 implementation paths and record that Exp6408 derives runtime duration
and peak-memory receipts in source without an authenticated generation process,
and that Exp6409 inherits the Exp6408 receipt surface. These source findings
SHALL make powered GGUF and prospective continuous-learning positives
audit-required hypotheses, while preserving the historical artifact verdicts.

Exp6410 SHALL validate the complete fourteen-task V552 queue. It SHALL require
unique ordered IDs, result JSON deliverables, structured gates with valid field
types, required upstream gate fields, complete prior-failure entries, no retired
ID reuse, no retired dependency chain, valid exclusion-manifest lint, and valid
roadmap schema checks. It SHALL accept the active `research-roadmap.yaml` as the
V552 queue when `research-roadmap-next.yaml` is absent after activation.

Exp6410 SHALL render every V552 prompt with `{project_root}` and `{date}`.
Every rendered prompt SHALL include `CONTEXT`, `EXISTING CODE TO READ FIRST`,
`TASK`, `CONCRETE STEPS`, the project root, planning date `20260814`, a
`Run command:` line, and the final line
`Do NOT push. Do NOT modify scripts/research_conductor.py.`. Every LLM/GPU task
SHALL name `MODEL_SPECS`, `cached_sota_pair()`, at least one mandated GGUF ID,
embedded GGUF tokenizers, no `AutoTokenizer`, authentic execution receipts, and
no legacy headline model. Non-GPU audit and aggregation tasks MAY use Claude
routes when the roadmap declares them.

Exp6410 SHALL check ARC tasks without claiming a game or level solve. ARC
prompts SHALL not update the solve registry, SHALL not use an outer-loop solver
or per-game adapter as the scored mechanism, and SHALL name the canonical live
agent path.

The Exp6410 artifact SHALL be written atomically to
`results/experiment_6410_v552_terminal_handoff_and_queue_preflight.json` with
`inference_substrate=aggregation_from_upstream_artifacts` and
`verifier_is_oracle=false`. The artifact SHALL include these required fields:
`status`, `v551_active_roadmap_path_and_hash`, `v551_task_ids`,
`v551_terminal_artifacts_and_sidecars_by_task`, `v551_artifact_verdicts`,
`v551_source_execution_findings`, `v551_conductor_outcomes`,
`v551_adversarial_findings`,
`v551_scientific_claim_eligibility_by_task`,
`exp6407_6408_6409_claim_correction`,
`v552_milestone_doc_and_queue_hashes`, `v552_task_ids`,
`v552_id_and_deliverable_checks`, `v552_dependency_and_gate_checks`,
`v552_gate_field_cross_reference_checks`, `v552_prior_failure_checks`,
`v552_exclusion_manifest_checks`,
`v552_agent_model_and_llm_policy_checks`, `v552_arc_no_solve_checks`,
`prompt_contract_checks`, `active_roadmap_modified`, `conductor_modified`,
`solve_registry_modified`, `claims_ledger_modified`,
`protected_files_unchanged`, `preconditions_checked`, `inference_substrate`,
`verifier_is_oracle`, `field_principles`, `field_provenance`, `random_seed`,
`duration_s`, `tests_run`, `reproducibility_checksum`, and `honest_verdict`.
`field_principles` SHALL cover every required field and every structured gate.
`field_provenance` SHALL classify every required field as measured, derived,
constant, or upstream. `random_seed` SHALL be null.

### SCENARIO-INFRA-6410-1: V551 Evidence Classes Stay Separate

GIVEN V551 artifacts include clean, blocked, flagged, unproven, missing, and
null evidence classes
WHEN Exp6410 builds the terminal handoff
THEN artifact verdicts, conductor outcomes, adversarial findings, source
execution findings, and scientific eligibility are recorded in separate fields.

### SCENARIO-INFRA-6410-2: Powered Positives Become Audit-Required Hypotheses

GIVEN Exp6407 remains adversarial-flagged and Exp6408 derives runtime receipts
without authenticated generation
WHEN Exp6410 records the Exp6407 through Exp6409 correction
THEN Exp6407 remains open-flagged, Exp6408 powered GGUF eligibility is false,
Exp6409 prospective CSL eligibility is false, and no historical verdict is
rewritten.

### SCENARIO-INFRA-6410-3: V552 Queue Contains Fourteen Tasks

GIVEN the V552 queue is active in `research-roadmap.yaml`
WHEN Exp6410 validates task identity and deliverables
THEN it records exactly Exp6410 through Exp6423 in ascending order with unique
JSON deliverables.

### SCENARIO-INFRA-6410-4: V552 Gates And Prior Failures Are Valid

GIVEN V552 tasks include structured gates and prior-failure records
WHEN Exp6410 validates dependencies, gate fields, prior failures, and retired
IDs
THEN every gate references an earlier task and declared required artifact field,
every prior failure is complete, and no retired chain is admitted.

### SCENARIO-INFRA-6410-5: Prompt, LLM, And ARC Contracts Validate

GIVEN V552 prompts mix audit, aggregation, GPU, and ARC work
WHEN Exp6410 renders and checks the prompts
THEN all prompt sections, run commands, final prohibitions, GGUF execution
rules, authentic-receipt rules, and ARC no-solve rules are checked before
queue execution.

### SCENARIO-INFRA-6410-6: Artifact Is Annotated, Checksummed, And Non-Mutating

GIVEN protected hashes, summaries, queue checks, and command receipts are
assembled
WHEN Exp6410 writes its artifact
THEN every required field and structured gate has a principle, every required
field has provenance, protected files remain byte-identical, the checksum
matches, `random_seed` is null, and `verifier_is_oracle` is false.

## Implementation Status (REQ-INFRA-6410)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6410 | Implemented: `python/carnot/experiment_6410_v552_terminal_handoff_and_queue_preflight.py`; terminal artifact `results/experiment_6410_v552_terminal_handoff_and_queue_preflight.json`. | Implemented: `tests/python/test_experiment_6410_v552_terminal_handoff_and_queue_preflight.py`. |

## REQ-INFRA-6424: V552 Evidence Handoff And V553 Queue Preflight

Carnot SHALL build Exp6424 as the exact V552-to-V553 handoff for planning date
20260814. The experiment SHALL preserve V552 artifact verdicts, conductor
outcomes, current adversarial findings, and scientific claim eligibility as
separate fields. It SHALL not rewrite historical determinations. It SHALL not
edit the active roadmap, conductor, ARC solve registry, or protected V552
evidence.

Exp6424 SHALL hash the active roadmap, any next-roadmap file, the V553
milestone document, conductor source and log, exclusion manifest, known issues,
north star, solve registry, claim records, every V552 terminal artifact, and
every known V552 sidecar. Missing protected inputs SHALL be recorded as absent,
not invented. Exp6424 SHALL record CPU, RAM, disk, GPU, and model-cache state
without starting research compute.

Exp6424 SHALL summarize Exp6410 through Exp6423 with
`scripts/summarize_artifact.py` before importing headline fields. It SHALL
record each task's honest verdict, conductor outcome, current flag state,
duration, inference substrate, and claim eligibility. Exp6424 SHALL preserve
the Exp6414 and Exp6417 duration flags, the Exp6420 CSL null, and the
Exp6421/Exp6422 no-solve boundary. It SHALL not infer claim eligibility from
an upstream positive summary when a later current audit blocks that claim.

Exp6424 SHALL validate the complete twelve-task V553 queue. It SHALL require
unique ordered IDs, result JSON deliverables, valid agent and model routes,
structured gates with valid field types, upstream gate fields declared in the
required artifact block, complete prior-failure entries, no retired ID reuse,
no retired dependency chain, valid exclusion-manifest lint, and valid roadmap
schema checks. It SHALL accept the active `research-roadmap.yaml` as the V553
queue when `research-roadmap-next.yaml` is absent after activation.

Exp6424 SHALL render every V553 prompt with `{project_root}` and `{date}`.
Every rendered prompt SHALL include `CONTEXT`, `EXISTING CODE TO READ FIRST`,
`TASK`, `CONCRETE STEPS`, the project root, planning date `20260814`, a
`Run command:` line, and the final line
`Do NOT push. Do NOT modify scripts/research_conductor.py.`. Every LLM task
SHALL name `MODEL_SPECS`, `cached_sota_pair()`, at least one mandated GGUF ID,
embedded GGUF tokenizers, no `AutoTokenizer`, fresh raw-output requirements,
and no legacy headline model.

Exp6424 SHALL check the ARC V553 task without claiming a game or level solve.
The ARC prompt SHALL not update the solve registry. It SHALL use the canonical
live path and SHALL forbid game source access, exhaustive ground-truth search,
and per-game adapters.

The Exp6424 artifact SHALL be written atomically to
`results/experiment_6424_v553_terminal_handoff_and_queue_preflight.json` with
`inference_substrate=aggregation_from_upstream_artifacts` and
`verifier_is_oracle=false`. The artifact SHALL include these required fields:
`status`, `v552_active_roadmap_path_and_hash`, `v552_task_ids`,
`v552_terminal_artifacts_and_sidecars_by_task`, `v552_artifact_verdicts`,
`v552_conductor_outcomes`, `v552_current_adversarial_findings`,
`v552_scientific_claim_eligibility_by_task`,
`exp6414_6417_6420_6421_6422_boundary`,
`v553_milestone_doc_and_queue_hashes`, `v553_task_ids`,
`v553_id_and_deliverable_checks`, `v553_dependency_and_gate_checks`,
`v553_gate_field_cross_reference_checks`, `v553_prior_failure_checks`,
`v553_exclusion_manifest_checks`,
`v553_agent_model_and_llm_policy_checks`, `v553_arc_no_solve_checks`,
`prompt_contract_checks`, `active_roadmap_modified`, `conductor_modified`,
`solve_registry_modified`, `protected_files_unchanged`, `blocked_reason`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`field_principles`, `field_provenance`, `random_seed`, `duration_s`,
`tests_run`, `reproducibility_checksum`, and `honest_verdict`.
`field_principles` SHALL cover every required field and every structured gate.
`field_provenance` SHALL classify every required field as measured, derived,
constant, or upstream. `random_seed` SHALL be null.

### SCENARIO-INFRA-6424-1: V552 Evidence Classes Stay Separate

GIVEN V552 artifacts include clean, flagged, null, internal-only, and aggregate
evidence classes
WHEN Exp6424 builds the terminal handoff
THEN artifact verdicts, conductor outcomes, current adversarial findings, and
scientific eligibility are recorded in separate fields.

### SCENARIO-INFRA-6424-2: V552 Boundaries Are Preserved

GIVEN Exp6414 and Exp6417 have current duration flags, Exp6420 has a CSL null,
and Exp6421 and Exp6422 have no-solve ARC results
WHEN Exp6424 records claim eligibility
THEN factor and prospective CSL public eligibility stay false, ARC solve
eligibility stays false, and the historical artifact verdicts remain intact.

### SCENARIO-INFRA-6424-3: V553 Queue Contains Twelve Tasks

GIVEN the V553 queue is active in `research-roadmap.yaml`
WHEN Exp6424 validates task identity and deliverables
THEN it records exactly Exp6424 through Exp6435 in ascending order with unique
JSON deliverables.

### SCENARIO-INFRA-6424-4: V553 Gates And Prior Failures Are Valid

GIVEN V553 tasks include structured gates and prior-failure records
WHEN Exp6424 validates dependencies, gate fields, prior failures, and retired
IDs
THEN every gate references an earlier task and declared required artifact
field, every prior failure is complete, and no retired chain is admitted.

### SCENARIO-INFRA-6424-5: Prompt, LLM, And ARC Contracts Validate

GIVEN V553 prompts mix audit, aggregation, local GGUF, and ARC work
WHEN Exp6424 renders and checks the prompts
THEN all prompt sections, run commands, final prohibitions, GGUF execution
rules, fresh raw-output rules, and ARC no-solve rules are checked before
research execution.

### SCENARIO-INFRA-6424-6: Artifact Is Annotated, Checksummed, And Non-Mutating

GIVEN protected hashes, summaries, queue checks, environment receipts, and
command receipts are assembled
WHEN Exp6424 writes its artifact
THEN every required field and structured gate has a principle, every required
field has provenance, protected files remain byte-identical, the checksum
matches, `random_seed` is null, `blocked_reason` is null on pass, and
`verifier_is_oracle` is false.

## Implementation Status (REQ-INFRA-6424)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6424 | Planned: `python/carnot/experiment_6424_v553_terminal_handoff_and_queue_preflight.py`; terminal artifact `results/experiment_6424_v553_terminal_handoff_and_queue_preflight.json`. | Planned: `tests/python/test_experiment_6424_v553_terminal_handoff_and_queue_preflight.py`. |

## REQ-INFRA-6436: V553 Terminal Handoff And V554 Queue Preflight

Carnot SHALL build Exp6436 as the V553-to-V554 terminal handoff for planning
date 20260815. The handoff SHALL read V553 terminal artifacts exactly as they
exist on disk. It SHALL record missing and zero-byte artifacts without repair.
It SHALL not rerun V553 experiments. It SHALL not edit `research-roadmap.yaml`
or `scripts/research_conductor.py`.

Exp6436 SHALL freeze one row per V553 task from Exp6427 through Exp6435. Each
row SHALL record task ID, deliverable path, byte count, artifact status,
honest verdict, current adversarial findings, claim eligibility, and the V553
capstone determination. Exp6434 SHALL be recorded as missing scientific
evidence when its artifact is zero bytes. The capstone determination SHALL
preserve the narrow factor eligibility and the blocked verification-cost,
prospective CSL, ARC reachability, public ARC, and hardware claims.

Exp6436 SHALL validate the activated V554 queue from `research-roadmap.yaml`.
It SHALL run the Pydantic roadmap schema, prior-failure linter,
exclusion-manifest linter, structured gate checker, duplicate ID check,
duplicate deliverable check, artifact-convention check,
determination-preservation check, prompt terminal-line check, and
root-clutter check. It SHALL require 12 unique task IDs in conductor order,
milestone `2026.08.554` on every task, one concrete JSON deliverable per task,
no retired ID, no retired dependency, and no gate that names a task outside the
active roadmap. If any check fails, Exp6436 SHALL fail closed.

Exp6436 SHALL bind every structured gate to an upstream task and an identical
required artifact field. For each gate it SHALL record upstream task, artifact
field, operator, expected value, producer task, producer deliverable, and
whether the producer prompt declares the field in `REQUIRED ARTIFACT FIELDS`.

Exp6436 SHALL validate V554 model and row contracts. Exp6439 through Exp6443
SHALL include explicit `MODEL_SPECS`, at least one mandated local GGUF from
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, or
`unsloth/gemma-4-26B-A4B-it-GGUF`, `cached_sota_pair()`, embedded GGUF
tokenizers, and no `AutoTokenizer` headline path. Comparative prompts SHALL
require `per_unit_rows`. Blocked-verdict prompts SHALL require
`gate_check_summary`. Required fields and acceptance gates SHALL be covered by
`field_principles`.

Exp6436 SHALL validate the complete four-field prior-failure contract for
Exp6438, Exp6440, Exp6441, Exp6443, Exp6444, and Exp6445. Each entry SHALL
contain `experiment_id`, `verdict`, `addressed_by`, and
`retire_if_same_verdict=true`. If a required rerun task is absent from the
active roadmap, the validation SHALL fail closed and name the missing task.

The Exp6436 artifact SHALL be written atomically to
`results/experiment_6436_v554_terminal_handoff_and_queue_preflight.json` with
`inference_substrate=aggregation_from_upstream_artifacts` and
`verifier_is_oracle=false`. The artifact SHALL include these required fields:
`status`, `v553_terminal_rows`, `v553_artifact_count`,
`v553_missing_or_zero_byte_artifacts`, `v553_flagged_artifacts`,
`v553_underpowered_artifacts`, `v553_terminal_claim_determinations`,
`active_roadmap_hash`, `task_count`, `task_ids_in_order`,
`unique_id_and_deliverable_check`, `milestone_consistency_check`,
`schema_validation`, `prior_failure_validation`,
`exclusion_manifest_validation`, `structured_gate_validation`,
`gate_producer_contract_rows`, `model_policy_validation`,
`per_unit_row_contract_validation`, `prompt_terminal_line_validation`,
`protected_files_unchanged`, `v554_queue_ready_score`, `blocked_reason`,
`gate_check_summary`, `preconditions_checked`, `inference_substrate`,
`verifier_is_oracle`, `field_principles`, `field_provenance`, `random_seed`,
`duration_s`, `tests_run`, `reproducibility_checksum`, and `honest_verdict`.
`field_principles` SHALL cover every required field and the
`v554_queue_ready_score` acceptance gate. `field_provenance` SHALL classify
every required field as measured, derived, constant, or upstream.

### SCENARIO-INFRA-6436-1: V553 Terminal Rows Preserve Final Determinations

GIVEN V553 artifacts include clean, flagged, underpowered, null, capstone, and
zero-byte outcomes
WHEN Exp6436 builds terminal handoff rows
THEN it records one row per V553 task, preserves the V553 capstone claim
determinations, and treats Exp6434 as missing scientific evidence.

### SCENARIO-INFRA-6436-2: V554 Queue Identity Fails Closed On Mismatch

GIVEN the activated V554 queue is read from `research-roadmap.yaml`
WHEN Exp6436 validates task identity, deliverables, milestones, retired IDs,
and gate upstreams
THEN it requires exactly 12 ordered tasks and sets
`v554_queue_ready_score=0.0` with a diagnostic when the active queue is short
or otherwise invalid.

### SCENARIO-INFRA-6436-3: Gate Producers Declare Exact Fields

GIVEN a V554 task has a structured `gated_on` entry
WHEN Exp6436 validates the producer contract
THEN the upstream prompt's `REQUIRED ARTIFACT FIELDS` block must contain the
exact `artifact_field` string and the gate row records the check result.

### SCENARIO-INFRA-6436-4: Prompt Model And Row Contracts Are Enforced

GIVEN V554 prompts include local-GGUF and comparative tasks
WHEN Exp6436 checks prompt contracts
THEN Exp6439 through Exp6443 satisfy the local GGUF policy, comparative prompts
require `per_unit_rows`, blocked verdicts require `gate_check_summary`, and
field principles cover each required field and acceptance gate.

### SCENARIO-INFRA-6436-5: Rerun Prior Failures Are Complete

GIVEN V554 contains changed reruns of V553 scopes
WHEN Exp6436 checks prior failures
THEN Exp6438, Exp6440, Exp6441, Exp6443, Exp6444, and Exp6445 each carry
`experiment_id`, `verdict`, `addressed_by`, and
`retire_if_same_verdict=true`, or the queue fails closed.

### SCENARIO-INFRA-6436-6: Artifact Is Annotated And Non-Mutating

GIVEN protected hashes, validation receipts, command receipts, and terminal
rows are assembled
WHEN Exp6436 writes its artifact
THEN every required field has a principle and provenance, protected files stay
byte-identical, the checksum matches, `random_seed` is null, and
`verifier_is_oracle` is false.

## Implementation Status (REQ-INFRA-6436)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6436 | Planned: `python/carnot/experiment_6436_v554_terminal_handoff_and_queue_preflight.py`; terminal artifact `results/experiment_6436_v554_terminal_handoff_and_queue_preflight.json`. | Planned: `tests/python/test_experiment_6436_v554_terminal_handoff_and_queue_preflight.py`. |

## REQ-INFRA-6404: V551 Handoff SHALL Preserve V550 Evidence And Fail Closed On Queue Mismatch

Carnot SHALL build Exp6404 as a bounded V550-to-V551 evidence handoff over
Exp6391 through Exp6403. The handoff SHALL hash the active roadmap, optional
next roadmap, milestone document, conductor log, conductor source, exclusion
manifest, known issues, ARC solve registry, claims ledger, and every expected
V550 terminal artifact before it imports artifact fields. It SHALL invoke
`scripts/summarize_artifact.py` for Exp6391 through Exp6403 and preserve clean,
partial, null, blocked, skipped, absent, flagged, and retired states without
filling gaps.

Exp6404 SHALL keep artifact verdicts, conductor outcomes, adversarial findings,
public-claim decisions, and per-task durations as separate facts. It SHALL
derive each duration from the matching terminal artifact receipt. It SHALL not
copy one aggregate milestone duration onto tasks. If a repeated unresolved
handoff mismatch is present, the handoff SHALL record the retirement trigger
instead of resolving it silently.

Exp6404 SHALL record the factor boundary exactly: four Exp6395 licenses, Qwen
abstention, two rejected Gemma cells, Exp6396 through Exp6398 positive internal
factor results, Exp6399 public-factor block, and no universal support. It SHALL
record the ARC boundary exactly: Exp6400 shadow readiness, Exp6401 positive
causal progress, Exp6402 clean provenance audit with public ineligibility, no
solve, and zero actual route promotion.

Exp6404 SHALL validate the V551 queue without changing the active roadmap or
the conductor. It SHALL require twelve unique V551 task IDs, unique result JSON
deliverables, ascending execution order, structured gates, required upstream
gate fields, complete prior-failure entries, no retired-ID reuse, no retired
dependency chain, valid exclusion-manifest state, valid memory gates, and valid
ARC gates before activation. If the audited queue contains fewer than the
twelve proposal tasks, Exp6404 SHALL fail closed in its status and verdict
while preserving all evidence facts.

Exp6404 SHALL render each prompt with the conductor-compatible `{project_root}`
and `{date}` values before checking it. Every rendered prompt SHALL include
`CONTEXT`, `EXISTING CODE TO READ FIRST`, `TASK`, `CONCRETE STEPS`, the project
root, planning date `20260813`, a `Run command:` line, and the final line
`Do NOT push. Do NOT modify scripts/research_conductor.py.`. Every task SHALL
use `agent_type=codex` and `model=gpt-5.5`. Every LLM task SHALL name
`MODEL_SPECS`, `cached_sota_pair()`, at least one mandated GGUF ID, embedded
GGUF tokenizers, no `AutoTokenizer`, real GPU receipts, and no legacy headline
cell.

The Exp6404 artifact SHALL be written atomically to
`results/experiment_6404_v551_terminal_handoff_and_queue_preflight.json` with
`inference_substrate=aggregation_from_upstream_artifacts` and
`verifier_is_oracle=false`. The artifact SHALL include these required fields:
`status`, `v550_active_roadmap_path_and_hash`, `v550_task_ids`,
`v550_terminal_artifacts_by_task`, `v550_artifact_verdicts`,
`v550_conductor_outcomes`, `v550_adversarial_findings`,
`v550_duration_receipts_by_task`, `v550_factor_boundary`,
`v550_arc_boundary`, `v551_milestone_doc_and_queue_hashes`,
`v551_task_ids`, `v551_id_and_deliverable_checks`,
`v551_dependency_and_gate_checks`,
`v551_gate_field_cross_reference_checks`, `v551_prior_failure_checks`,
`v551_exclusion_manifest_checks`,
`v551_agent_model_and_llm_policy_checks`, `prompt_contract_checks`,
`active_roadmap_modified`, `conductor_modified`, `solve_registry_modified`,
`claims_ledger_modified`, `protected_files_unchanged`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`field_principles`, `field_provenance`, `random_seed`, `duration_s`,
`tests_run`, `reproducibility_checksum`, and `honest_verdict`.

### SCENARIO-INFRA-6404-1: V550 Evidence Classes Stay Separate

GIVEN V550 produced positive, null, complete, clean, flagged, skipped, blocked,
absent, retired, and partial-capability evidence classes
WHEN Exp6404 builds its terminal matrix
THEN artifact verdicts, conductor outcomes, adversarial findings, public-claim
decisions, and per-task durations are recorded in separate fields and no
missing artifact is invented.

### SCENARIO-INFRA-6404-2: Factor Boundary Preserves Partial Capability

GIVEN Exp6395 issued four narrow licenses while Qwen abstained
WHEN Exp6404 summarizes factor evidence
THEN the four licensed cells, the two rejected Gemma cells, the three Qwen
abstentions, Exp6396 through Exp6398 positive internal results, Exp6399 public
block, and the absence of universal support remain explicit.

### SCENARIO-INFRA-6404-3: ARC Boundary Preserves Internal Progress Only

GIVEN Exp6400 through Exp6402 produced internal ARC readiness and causal
progress evidence
WHEN Exp6404 summarizes ARC evidence
THEN shadow readiness, positive causal progress, clean provenance, public
ineligibility, no solve, and zero actual route promotion remain explicit.

### SCENARIO-INFRA-6404-4: V551 Queue Mismatch Fails Closed

GIVEN the V551 proposal contains twelve tasks
WHEN the audited roadmap contains fewer than twelve V551 task IDs
THEN Exp6404 records the missing IDs, leaves the roadmap and conductor
unchanged, and emits a terminal fail-closed handoff verdict.

### SCENARIO-INFRA-6404-5: Prompt And LLM Policy Contracts Validate

GIVEN V551 prompts use `{project_root}` and `{date}` placeholders
WHEN Exp6404 validates rendered prompt contracts
THEN it checks all required prompt sections, run commands, final prohibition,
agent/model pairs, model specs, cached SOTA access, mandated GGUF IDs,
embedded-tokenizer rules, no-`AutoTokenizer` rules, real GPU receipts, and no
legacy headline cell.

### SCENARIO-INFRA-6404-6: Artifact Is Annotated, Checksummed, And Non-Mutating

GIVEN precondition hashes, summary receipts, queue checks, and command receipts
are assembled
WHEN Exp6404 writes its artifact
THEN every required field and structured gate has a principle, every required
field has provenance classified as measured, derived, constant, or upstream,
the solve registry and claims ledger remain byte-identical, protected files
remain byte-identical, the checksum matches, `random_seed` is null, and
`verifier_is_oracle` is false.

## Implementation Status (REQ-INFRA-6404)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6404 | Planned: `python/carnot/experiment_6404_v551_terminal_handoff_and_queue_preflight.py`; terminal artifact `results/experiment_6404_v551_terminal_handoff_and_queue_preflight.json`. | Planned: `tests/python/test_experiment_6404_v551_terminal_handoff_and_queue_preflight.py`. |

## REQ-INFRA-6379: Canonical Factor-Edit Transport Contract SHALL Use One Source

Carnot SHALL build Exp6379 as deterministic transport infrastructure for the
Exp6366 factor-edit failure. The experiment SHALL freeze and hash the Exp6366
terminal artifact, prompt payload sidecars, bounded schema sidecar, and all
three raw stdout sidecars before it labels any failure. Each frozen raw failure
SHALL preserve overlapping labels from this set: `thinking_leakage`,
`repetition_collapse`, `truncation`, `syntax_failure`, `structural_failure`,
`semantic_failure`, and `unknown`.

Exp6379 SHALL define one canonical bounded factor-edit object. That object
SHALL generate the schema description, prompt instruction fragment, compact
output example, validator field list, source-binding checks, and output-token
lower bound. Hand-written duplicate surfaces, stale examples, field reorder,
missing fixed fields, prompt-schema conflict, and hash drift SHALL fail closed.
The canonical object SHALL include one preregistered bounded
`evidence_summary` variant that may cite visible source evidence. It SHALL not
ask for hidden chain of thought.

Exp6379 SHALL include `MODEL_SPECS` rows for exactly
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. Token capacity checks SHALL use each local
GGUF file's embedded tokenizer in vocab-only mode. The experiment SHALL not call
`AutoTokenizer`, invoke autoregressive generation, add grammar decoding, retry a
parser, or repair output after the fact. It SHALL compute the minimal
serialized-output token count, old-budget margin, required completion lower
bound, fixed headroom, context-window margin, and truncation risk separately for
each model.

Exp6379 SHALL define a bounded repetition policy before any later live
execution. A repeated-token threshold breach SHALL become abstention. A larger
token budget alone SHALL not qualify the transport contract. The artifact SHALL
write `canonical_factor_transport_contract_ready_score=1.0` only when all
deterministic generated surfaces agree, all drift attacks fail closed, all three
embedded-tokenizer capacity receipts exist, and no retired decoding mechanism
is present.

The Exp6379 artifact SHALL be written atomically to
`results/experiment_6379_canonical_factor_edit_transport_contract.json` with
`inference_substrate=deterministic_gguf_vocab_only_transport_contract` and
`verifier_is_oracle=false`. The artifact SHALL include these required fields:
`status`, `upstream_exp6366_path_hash_and_terminal_class`,
`frozen_raw_failure_paths_hashes_and_labels`, `MODEL_SPECS`,
`embedded_gguf_tokenizer_receipts`, `autotokenizer_usage_count`,
`live_autoregressive_generation_invoked`,
`canonical_schema_path_hash_and_version`, `canonical_schema_generated_surfaces`,
`prompt_schema_drift_checks`, `bounded_evidence_summary_variant`,
`per_model_minimum_output_tokens_and_capacity_margins`,
`repetition_policy_and_failure_thresholds`,
`deterministic_transport_mutation_matrix`,
`syntax_structure_source_binding_and_semantic_boundaries`,
`retired_decoding_mechanism_usage_count`,
`canonical_factor_transport_contract_ready_score`,
`no_model_quality_or_utility_claim`, `protected_files_unchanged`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`field_principles`, `field_provenance`, `random_seed`, `duration_s`,
`tests_run`, `reproducibility_checksum`, and `honest_verdict`.

### Required Artifact Fields And Principles

- `status`: Terminal status distinguishes positive and null deterministic transport evidence.
- `upstream_exp6366_path_hash_and_terminal_class`: The Exp6366 terminal artifact is frozen before failure labels are assigned.
- `frozen_raw_failure_paths_hashes_and_labels`: Raw stdout, prompt payload, and schema hashes bind each failure label to bytes.
- `MODEL_SPECS`: The three mandated GGUF model ids are present for tokenizer capacity checks.
- `embedded_gguf_tokenizer_receipts`: Each token receipt uses the embedded GGUF tokenizer in vocab-only mode.
- `autotokenizer_usage_count`: A bare zero records that no external tokenizer path was used.
- `live_autoregressive_generation_invoked`: A bare false records that this run is deterministic transport infrastructure.
- `canonical_schema_path_hash_and_version`: The single canonical object is written and hash-bound as the schema source.
- `canonical_schema_generated_surfaces`: Prompt, schema, validator, example, and source checks share one canonical hash.
- `prompt_schema_drift_checks`: Drift between generated surfaces fails closed before any later live call.
- `bounded_evidence_summary_variant`: The JSON object may include a short visible-evidence summary.
- `per_model_minimum_output_tokens_and_capacity_margins`: Output-token lower bounds and context margins are measured per model.
- `repetition_policy_and_failure_thresholds`: Repeated-token collapse is a preregistered abstention, not a transport pass.
- `deterministic_transport_mutation_matrix`: Known drift, syntax, structure, source, and semantic attacks are rejected.
- `syntax_structure_source_binding_and_semantic_boundaries`: The validator checks transport boundaries but not exact task utility.
- `retired_decoding_mechanism_usage_count`: A bare zero records that retired decoding controls were not used.
- `canonical_factor_transport_contract_ready_score`: Readiness is one only when generators, drift checks, token receipts, and bans agree.
- `no_model_quality_or_utility_claim`: The artifact makes no model-quality, factor-success, or utility claim.
- `protected_files_unchanged`: Conductor, ops, traceability, and Exp6366 inputs stay byte-identical.
- `preconditions_checked`: Preconditions bind upstream evidence, model identities, token receipts, bans, and protected hashes.
- `inference_substrate`: The substrate is deterministic local GGUF tokenizer measurement.
- `verifier_is_oracle`: The validator is not the later exact semantic oracle.
- `field_principles`: Every required field states its guard.
- `field_provenance`: Every required field maps to specs, frozen inputs, generated surfaces, tests, or constants.
- `random_seed`: The seed pins deterministic ordering even though no random sampling occurs.
- `duration_s`: Wall time is measured without padding.
- `tests_run`: Verification commands and exit codes are recorded.
- `reproducibility_checksum`: A normalized checksum detects artifact drift.
- `honest_verdict`: The verdict uses a terminal prefix and states the transport-only claim.

### SCENARIO-INFRA-6379-1: Exp6366 Failures Freeze Before Labels

GIVEN Exp6366 wrote three nonempty raw stdout sidecars and zero parse-valid
objects
WHEN Exp6379 classifies those outputs
THEN the Exp6366 artifact, prompt payloads, schema sidecar, and raw outputs are
hashed first
AND each model keeps all applicable deterministic failure labels.

### SCENARIO-INFRA-6379-2: Canonical Object Generates All Surfaces

GIVEN the canonical bounded factor-edit object
WHEN Exp6379 builds prompt, schema, validator, example, source-binding, and
token-bound surfaces
THEN each surface derives from the same canonical hash
AND duplicate or stale hand-written surfaces fail closed.

### SCENARIO-INFRA-6379-3: Drift And Malformed Output Fail Closed

GIVEN prompt-schema conflict, stale examples, missing fixed fields, reordered
fields, thinking prefixes, markdown fences, repeated tokens, mid-object
truncation, unsupported source spans, or parse-valid semantic corruption
WHEN the deterministic transport validator checks them
THEN each mutation is rejected without parser retry, grammar decoding, or
post-hoc repair.

### SCENARIO-INFRA-6379-4: Token Capacity Uses Embedded GGUF Tokenizers Only

GIVEN the three mandated local GGUF model rows
WHEN Exp6379 computes output capacity receipts
THEN each receipt uses the embedded GGUF tokenizer in vocab-only mode
AND it records minimal output tokens, old-budget margin, lower bound, headroom,
context margin, and truncation risk per model.

### SCENARIO-INFRA-6379-5: Readiness Is Deterministic Transport Only

GIVEN all generators agree, all drift attacks fail closed, all tokenizer
receipts exist, and retired decoding mechanisms are absent
WHEN Exp6379 writes its artifact
THEN `canonical_factor_transport_contract_ready_score` is `1.0`
AND no model quality, exact utility, or semantic-oracle claim is made.

## Implementation Status (REQ-INFRA-6379)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6379 | Implemented: `python/carnot/experiment_6379_canonical_factor_edit_transport_contract.py`; terminal artifact `results/experiment_6379_canonical_factor_edit_transport_contract.json`. | Implemented: `tests/python/test_experiment_6379_canonical_factor_edit_transport_contract.py`. |

## REQ-INFRA-6365: GGUF Child Failure Forensics SHALL Preserve Runtime Diagnostics

Carnot SHALL build Exp6365 as a reusable observable child-process contract for
local llama.cpp GGUF generation. The contract SHALL diagnose Exp6352 without
claiming a factor proposal, model accuracy, or utility result. It SHALL
reconstruct Exp6352 command, source, prompt, sampling, environment, exit, raw
byte, and terminal-class receipts from the committed source, the committed
artifact, and git history before any new full three-model rerun.

Exp6365 SHALL use `cached_sota_pair()` helper calls and exactly these measured
GGUF model ids: `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. It SHALL use only the GGUF-embedded
llama.cpp tokenizer in vocab-only mode. It SHALL never call
`AutoTokenizer`. Prompt token counting SHALL happen before model load. A call
SHALL fail closed when `prompt_tokens + max_tokens > n_ctx`.

Each child call SHALL record sanitized argv, PID, dispatcher, allowed
environment hash, source hash, command hash, prompt hash, prompt token count,
requested output tokens, `n_ctx`, capacity margin, stdout and stderr sidecar
paths and hashes, bounded stdout and stderr excerpts, return code, signal,
timeout state, usage receipt, phase timestamps, and raw byte count. Full
stdout and stderr SHALL be stored as sidecars addressed by their hashes.

Exp6365 SHALL record task-linked GPU samples before load, after load, during
generation, after unload, and after cleanup. Each sample SHALL include GPU
index, memory, utilization, process identity, timestamp, model id, and phase.
Readiness SHALL require authenticated GPU offload and a proved VRAM rise and
release for each completed model row. CUDA readiness SHALL not be accepted from
`nvidia-smi` visibility alone.

Exp6365 SHALL inject deterministic failures for nonzero exit, timeout, empty
stdout, malformed usage receipt, context overflow, source drift, and missing
GPU sample. Each injection SHALL fail closed and preserve diagnostics. A live
row SHALL count as successful only when return code is zero, timeout is false,
raw bytes are nonempty before parsing, prompt and completion token counts are
positive, GPU offload is authenticated, context capacity is sufficient, and no
required GPU sample is missing.

The Exp6365 artifact SHALL be written atomically to
`results/experiment_6365_gguf_child_failure_forensics_and_runtime_contract.json`
with `inference_substrate=local_llama_cpp_gguf_observable_child_process_contract`
and `verifier_is_oracle=false`. It SHALL include these required fields:
`status`, `upstream_exp6352_path_hash_and_terminal_class`,
`reconstructed_exp6352_command_and_source_receipt`,
`exp6352_source_artifact_sampling_drift`, `MODEL_SPECS`, `models_used`,
`cached_sota_pair_receipts`,
`model_file_hashes_revisions_quantizations_and_tokenizers`,
`embedded_gguf_tokenizer_receipts`, `autotokenizer_usage_count`,
`llama_cpp_gpu_offload_support_receipt`,
`task_linked_gpu_samples_by_model_and_phase`,
`dispatcher_and_process_identity_receipts`,
`source_command_prompt_and_environment_hashes_by_call`,
`prompt_token_context_capacity_receipts_by_model`,
`stdout_stderr_sidecar_paths_hashes_and_bounded_excerpts`,
`child_exit_signal_timeout_and_usage_receipts_by_model`,
`load_prompt_generate_unload_cleanup_timings_by_model`,
`raw_output_paths_hashes_and_byte_counts`,
`live_autoregressive_generation_invoked_by_model`,
`failure_injection_matrix`, `vram_rise_and_release_receipts_by_model`,
`gguf_runtime_observability_ready_score`,
`no_proposal_quality_or_utility_claim`, `protected_files_unchanged`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`field_principles`, `field_provenance`, `random_seed`, `duration_s`,
`tests_run`, `reproducibility_checksum`, and `honest_verdict`.

### SCENARIO-INFRA-6365-1: Exp6352 Is Reconstructed Before Rerun

GIVEN the committed Exp6352 source and artifact
WHEN Exp6365 starts
THEN it records the committed source hash, artifact hash, reconstructed child
command, prompt hashes, sampling values, child return codes, empty raw bytes,
missing stderr diagnostics, terminal class, and source-versus-artifact
sampling drift before any new full three-model rerun.

### SCENARIO-INFRA-6365-2: Embedded Tokenizer Enforces Context Capacity

GIVEN one exact prompt and one exact GGUF file
WHEN Exp6365 counts tokens
THEN it uses the embedded GGUF tokenizer in vocab-only mode, records the
prompt count and capacity margin, keeps `autotokenizer_usage_count=0`, and
blocks before model load if prompt plus requested output exceeds `n_ctx`.

### SCENARIO-INFRA-6365-3: Observable Child Runner Preserves Diagnostics

GIVEN a child exits nonzero, times out, emits empty stdout, or emits malformed
usage
WHEN the observable child runner normalizes the receipt
THEN the row fails closed, preserves full stdout and stderr sidecars by hash,
records bounded excerpts, records PID, command, environment, source, signal,
timeout, token, context, and phase receipts, and does not infer a root cause.

### SCENARIO-INFRA-6365-4: GPU Samples Are Task Linked And Required

GIVEN a child claims live llama.cpp generation
WHEN readiness is computed
THEN samples for before load, after load, during generation, after unload, and
after cleanup must be present for that model. Missing samples, missing process
identity, missing GPU offload, or missing VRAM rise and release force the row
to fail closed.

### SCENARIO-INFRA-6365-5: Runtime Score Does Not Claim Proposal Quality

GIVEN three mandatory model rows and deterministic failure injections
WHEN Exp6365 validates the artifact
THEN `gguf_runtime_observability_ready_score` is `1.0` only when all three live
rows meet the child-process contract, every injection fails closed, protected
files are unchanged, field principles and provenance cover every required
field, `verifier_is_oracle=false`, and the artifact makes no proposal,
accuracy, or utility claim.

## Implementation Status (REQ-INFRA-6365)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6365 | Planned: `python/carnot/experiment_6365_gguf_child_failure_forensics_and_runtime_contract.py`; terminal artifact `results/experiment_6365_gguf_child_failure_forensics_and_runtime_contract.json`. | Planned: `tests/python/test_experiment_6365_gguf_child_failure_forensics_and_runtime_contract.py`. |

## REQ-INFRA-6413: Local GGUF Execution Receipts SHALL Bind Process, Model Bytes, Raw Output, Clocks, GPU Telemetry, And Exit State

Carnot SHALL build Exp6413 as the reusable authenticated execution receipt
contract for local llama.cpp GGUF generation. A model name, cached path,
tokenizer call, inherited receipt, or derived GPU-looking value SHALL NOT count
as authentic execution.

Exp6413 SHALL use `cached_sota_pair()` helper calls to resolve exactly these
three mandated GGUF model ids: `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. It SHALL use only embedded GGUF tokenizers.
It SHALL never call `AutoTokenizer`. Legacy small models MAY be CPU smoke
fixtures only. They SHALL NOT satisfy readiness.

Before loading any model, Exp6413 SHALL preflight both RTX 3090 GPUs, free VRAM,
CUDA visibility, llama.cpp GPU offload support, cached GGUF files, full model
file hashes, embedded tokenizer metadata, storage, exact commands, declared
sequential GPU scheduling, and protected training processes. If any required
condition fails, Exp6413 SHALL write the terminal blocked artifact before model
load.

Each authenticated per-model receipt SHALL include process PID, parent PID,
executable, command hash, config hash, model hub id, revision, quantization,
model file path and hash, tokenizer source and hash, device UUID, start, load,
first-token, completion, and end monotonic clocks, PID-bound GPU memory and
utilization samples, prompt hash, raw output byte path, raw output hash and
length, prompt and completion token counts, exit status, stderr hash, and
cleanup result. Raw generated bytes SHALL be stored before parsing.

Exp6413 SHALL run one bounded non-headline canary per mandated family through
the real generation call. It SHALL schedule them sequentially on declared GPU
assignments to avoid oversubscription. Readiness SHALL require distinct,
internally consistent receipts for all three families, CUDA offload, positive
token counts, nonempty raw bytes, process lifetime consistent with monotonic
clocks, model file access matching the immutable bytes, PID-bound GPU telemetry,
zero exit status, and successful cleanup.

The contract SHALL reject forged PID, reused raw hash, substituted model file,
missing first-token clock, constant memory, telemetry from another process,
tokenizer substitution, early process exit, and inherited upstream receipt. The
Exp6413 artifact SHALL set `authenticated_receipt_contract_ready_score=1.0`
only when all three mandated families authenticate and every mutation attack
fails closed. It SHALL set `verifier_is_oracle=false`, because the receipt
proves execution and not semantic correctness.

The Exp6413 artifact SHALL be written atomically to
`results/experiment_6413_authenticated_sota_gguf_execution_receipts.json` with
`inference_substrate=live_llm_inference_local_gguf_sota`. It SHALL include
these required fields: `status`, `MODEL_SPECS`, `models_used`,
`cached_sota_pair_receipts`,
`model_hub_ids_revisions_quantizations_paths_and_hashes`,
`embedded_gguf_tokenizer_receipts`, `autotokenizer_usage_count`,
`gpu_precondition_receipts`, `cuda_and_llamacpp_offload_receipts`,
`receipt_schema_path_hash_and_required_fields`,
`per_model_process_pid_parent_executable_command_and_config_receipts`,
`per_model_device_uuid_and_pid_bound_gpu_sample_receipts`,
`per_model_start_load_first_token_completion_end_monotonic_clocks`,
`per_model_prompt_raw_output_token_exit_stderr_and_cleanup_receipts`,
`per_model_raw_output_paths_and_hashes`,
`constant_or_inherited_receipt_count`, `legacy_headline_cell_count`,
`mutation_attack_matrix`, `authentic_family_count`,
`authenticated_receipt_contract_ready_score`, `protected_files_unchanged`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`field_principles`, `field_provenance`, `random_seed`, `duration_s`,
`tests_run`, `reproducibility_checksum`, and `honest_verdict`.

### SCENARIO-INFRA-6413-1: Model Resolution Uses Cached SOTA GGUF Rows

GIVEN the cached SOTA helper resolves the default pair and the dense pair
WHEN Exp6413 builds `MODEL_SPECS`
THEN it includes the three mandated GGUF ids, hashes each model file, records
revision and quantization, uses embedded tokenizer receipts, and keeps
`autotokenizer_usage_count=0`.

### SCENARIO-INFRA-6413-2: Blocked Preconditions Stop Before Load

GIVEN a required GPU, storage, tokenizer, model file, offload, command, or
protected-process precondition fails
WHEN Exp6413 starts
THEN it writes a terminal blocked artifact and runs no model-generation call.

### SCENARIO-INFRA-6413-3: A Receipt Binds One Process To One Model And One Raw Output

GIVEN a child process runs a bounded canary
WHEN the receipt is validated
THEN the PID, parent PID, executable, command and config hashes, model file
hash, tokenizer hash, raw output hash, clocks, token counts, PID-bound GPU
samples, exit status, stderr hash, and cleanup result must all agree.

### SCENARIO-INFRA-6413-4: Mutation Attacks Fail Closed

GIVEN forged PID, reused raw hash, substituted model file, missing first-token
clock, constant memory, wrong-process telemetry, tokenizer substitution, early
exit, or inherited upstream receipt
WHEN the receipt validator evaluates the mutated row
THEN every attack is rejected and recorded in the mutation matrix.

### SCENARIO-INFRA-6413-5: Readiness Requires Three Distinct CUDA Receipts

GIVEN one bounded canary receipt per mandated model family
WHEN Exp6413 computes readiness
THEN `authenticated_receipt_contract_ready_score` is `1.0` only when all three
families authenticate with CUDA offload, all raw output hashes are distinct,
protected files are unchanged, field principles cover required fields, and all
verification commands exit zero.

## Implementation Status (REQ-INFRA-6413)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6413 | Planned: `python/carnot/experiment_6413_authenticated_sota_gguf_execution_receipts.py`; terminal artifact `results/experiment_6413_authenticated_sota_gguf_execution_receipts.json`. | Planned: `tests/python/test_experiment_6413_authenticated_sota_gguf_execution_receipts.py`. |

## REQ-INFRA-6414: Fresh Three-Family Factor-Event Corpus SHALL Bind Rows To Authenticated GGUF Receipts And Exact Labels

Carnot SHALL build Exp6414 as a fresh V552 factor-event corpus. The corpus SHALL
be independent of V550 and V551 fixtures. The planning date SHALL be
`20260814`.

Exp6414 SHALL revalidate the Exp6413 authenticated execution gate before it
freezes rows. It SHALL use `cached_sota_pair()` helper calls to resolve exactly
these GGUF model ids: `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. It SHALL use only embedded GGUF tokenizers.
It SHALL never call `AutoTokenizer`.

Exp6414 SHALL pre-register at least 72 events before generation. The event set
SHALL be balanced across the three model families, supported and unsupported
constraint families, clean, contradicted, implicit, stale, duplicate, and
superseded classes, and acquisition, retention, and future partitions.

Exp6414 SHALL seal prompts, model configs, event order, partitions, and exact
checker versions before raw output parsing. Every row SHALL bind the process
receipt, model file hash, tokenizer hash, prompt hash, raw output bytes, source
span, proposed typed effect, model-family license state, and exact checker
outcome.

Exp6414 SHALL record model-family and constraint-family cells independently.
Unsupported, abstaining, malformed, or invalid cells SHALL not block or inherit
another cell. They SHALL abstain without fallback.

Exp6414 SHALL prove byte and hash disjointness from V550 and V551 fixtures. It
SHALL prevent future-label visibility before row freeze. It SHALL run attacks
for model-row swaps, output substitution, receipt reuse, cross-family fallback,
license inheritance, checker drift, partition leakage, and post-label row edits.

Exp6414 SHALL write
`results/experiment_6414_fresh_three_family_factor_event_corpus.json` with
these required fields: `status`, `exp6413_gate_receipt`, `MODEL_SPECS`,
`models_used`, `cached_sota_pair_receipts`,
`model_file_and_tokenizer_hashes`, `embedded_gguf_tokenizer_receipts`,
`autotokenizer_usage_count`, `license_and_frozen_harness_bindings`,
`manifest_path_hash_counts_balance_classes_and_partition_seals`,
`prompt_config_event_order_and_checker_freeze_receipts`,
`corpus_disjointness_receipts`,
`per_row_authenticated_process_and_raw_output_bindings`,
`per_row_source_effect_license_and_exact_outcome_bindings`,
`per_cell_transport_evaluability_correctness_abstention_malformed_truncation_duplicate_and_cost_results`,
`unlicensed_cell_abstention_records`, `silent_fallback_count`,
`universal_support_claimed`, `protected_leakage_count`,
`model_output_substitution_count`, `attack_matrix`, `authentic_family_count`,
`fresh_factor_event_corpus_ready_score`, `protected_files_unchanged`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`field_principles`, `field_provenance`, `random_seed`, `duration_s`,
`tests_run`, `reproducibility_checksum`, and `honest_verdict`.

`fresh_factor_event_corpus_ready_score` SHALL be `1.0` only when all three
GGUF families authenticate through Exp6413, all rows are raw-byte and checker
bound, all partitions are sealed, protected leakage is zero, every unsupported
cell abstains without fallback, all mutation attacks fail closed, and all
recorded verification commands exit zero. `verifier_is_oracle` SHALL be true
only for deterministic factor-event checkers. Model text, transport, licenses,
and receipts SHALL not be semantic oracles.

### SCENARIO-INFRA-6414-1: Manifest Is Fresh, Balanced, And Sealed

GIVEN Exp6414 builds the V552 factor-event corpus
WHEN the manifest is written
THEN it contains at least 72 rows balanced across the required model families,
constraint support states, outcome classes, and partitions
AND it contains no byte or hash match to V550 or V551 fixture artifacts.

### SCENARIO-INFRA-6414-2: Rows Bind Process, Bytes, Source, License, And Exact Outcome

GIVEN a pre-registered row
WHEN Exp6414 stores raw output and then runs exact checking
THEN the row binds an accepted Exp6413 process receipt, model file hash,
embedded tokenizer hash, prompt hash, raw output hash, source span, proposed
typed effect, license state, and exact checker outcome.

### SCENARIO-INFRA-6414-3: Cells Stay Independent

GIVEN one model-family and constraint-family cell is unsupported, abstaining,
malformed, duplicate, truncated, or exact-failing
WHEN Exp6414 computes cell results
THEN that cell records its own terminal state and does not block, promote, or
inherit any other cell.

### SCENARIO-INFRA-6414-4: Mutation Attacks Fail Closed

GIVEN model-row swaps, output substitution, receipt reuse, cross-family
fallback, license inheritance, checker drift, partition leakage, and post-label
row edits
WHEN Exp6414 validates the corpus
THEN every attack fails closed and `model_output_substitution_count` remains
zero.

### SCENARIO-INFRA-6414-5: Readiness Is Conjunctive And Narrow

GIVEN all three GGUF families authenticated, rows are sealed, unsupported cells
abstain without fallback, disjointness holds, and exact checkers own semantic
labels
WHEN Exp6414 computes readiness
THEN `fresh_factor_event_corpus_ready_score` is `1.0`
AND `universal_support_claimed` is false.

## Implementation Status (REQ-INFRA-6414)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6414 | Planned: `python/carnot/experiment_6414_fresh_three_family_factor_event_corpus.py`; terminal artifact `results/experiment_6414_fresh_three_family_factor_event_corpus.json`. | Planned: `tests/python/test_experiment_6414_fresh_three_family_factor_event_corpus.py`. |

## REQ-INFRA-6426: Task-Scoped Runtime Receipts SHALL Attribute Phases, GPU Evidence, Runner Choice, And Exit State To One Task

Carnot SHALL build Exp6426 as the reusable task-scoped runtime receipt
contract for V553 powered claims. The planning date SHALL be `20260814`.
The contract SHALL attribute queue wait, model load, generation, exact
verification, and artifact-write time to one task without trusting a model-name
string, inherited wall-clock value, or aggregate GPU sample.

The receipt schema SHALL be versioned and hash-bound. Each receipt row SHALL
include task id, control id, phase, monotonic start and end, wall-clock start
and end, parent PID, child PIDs, command hash, config hash, model hash, runner
selection, device ids, concurrency group, raw-output hash, exit status, and
attribution confidence. Rows SHALL be written through a reusable helper outside
`scripts/research_conductor.py`. The helper SHALL write atomically and SHALL
preserve partial receipts after interruption.

Exp6426 SHALL run four controls: deterministic CPU success, explicit preflight
block, interrupted child process, and one mandated local GGUF CUDA smoke. The
powered control SHALL use `cached_sota_pair()` and SHALL include
`unsloth/gemma-4-26B-A4B-it-GGUF` in `MODEL_SPECS`. It SHALL use only the
embedded GGUF tokenizer and SHALL never call `AutoTokenizer`. If the powered
smoke cannot run, Exp6426 SHALL fail closed with `blocked_reason` while still
representing the CPU, blocked, and interrupted controls.

Before the powered smoke, Exp6426 SHALL preflight both RTX 3090 devices, free
VRAM, model cache, llama.cpp CUDA support, runner binary, tokenizer metadata,
disk, CPU, RAM, and monotonic clock. The powered row SHALL record model file
hashes, embedded tokenizer hashes, raw bytes, first-token or completion
evidence, runner binary and selection receipts, child exit receipts, and
PID-linked `nvidia-smi` samples.

Duration SHALL be recomputed from phase intervals. The recomputation SHALL
reject negative intervals, unexplained overlaps, missing intervals, synthesized
runtime fields, and wall-clock-only intervals. Queue wait, model load,
generation, exact verification, and artifact write SHALL remain separate phase
rows.

The contract SHALL reject forged PID, stale `nvidia-smi` sample, model-name-only
substitution, raw-output reuse, runner swap, clock rollback, truncated receipt,
concurrency collision, CPU fallback, and child-exit omission. Exp6426 SHALL set
`runtime_receipt_contract_ready_score=1.0` only when all four control paths are
represented, the powered path is authentic, every phase recomputes, and all
critical attacks fail closed. It SHALL set `verifier_is_oracle=false`, because
process and hash checks authenticate execution but do not prove semantic
correctness.

Exp6426 SHALL write
`results/experiment_6426_task_scoped_runtime_receipt_contract.json` with these
required fields: `status`, `receipt_schema_version_and_hash`,
`helper_source_and_test_hashes`, `MODEL_SPECS`, `models_used`,
`cached_sota_pair_receipts`, `model_file_and_embedded_tokenizer_hashes`,
`autotokenizer_usage_count`, `runner_binary_and_selection_receipts`,
`device_inventory_and_preflight_receipts`, `per_unit_rows`,
`cpu_blocked_interrupted_and_powered_control_rows`,
`per_phase_monotonic_and_wall_clock_receipts`,
`parent_child_pid_and_exit_receipts`, `pid_linked_gpu_samples`,
`concurrency_group_receipts`, `command_config_model_and_raw_output_hashes`,
`synthesized_runtime_field_count`, `cpu_fallback_count`,
`attribution_failure_count`, `recomputed_duration_s`,
`reported_vs_recomputed_duration_delta`, `attack_matrix`,
`runtime_receipt_contract_ready_score`, `current_adversarial_findings`,
`protected_files_unchanged`, `blocked_reason`, `preconditions_checked`,
`inference_substrate`, `verifier_is_oracle`, `field_principles`,
`field_provenance`, `random_seed`, `duration_s`, `tests_run`,
`reproducibility_checksum`, and `honest_verdict`.

### SCENARIO-INFRA-6426-1: Versioned Helper Writes Atomic And Partial Receipts

GIVEN a task-scoped receipt helper
WHEN a control records phases and then an interruption happens
THEN the helper writes completed rows atomically, preserves the partial receipt,
and records the interrupted child PID and exit status.

### SCENARIO-INFRA-6426-2: Four Controls Represent The Runtime Contract

GIVEN deterministic CPU success, explicit preflight block, interrupted child,
and mandated local GGUF powered smoke controls
WHEN Exp6426 builds the terminal artifact
THEN `per_unit_rows` contains one row per control and phase
AND `runtime_receipt_contract_ready_score` can be `1.0` only when all four
control paths are present and the powered path authenticates.

### SCENARIO-INFRA-6426-3: Powered Smoke Uses Cached Gemma26 GGUF And Embedded Tokenizer

GIVEN `cached_sota_pair()` resolves mandated local GGUF models
WHEN Exp6426 selects the powered smoke model
THEN `MODEL_SPECS` includes `unsloth/gemma-4-26B-A4B-it-GGUF`
AND tokenizer receipts come from the embedded GGUF tokenizer with
`autotokenizer_usage_count=0`.

### SCENARIO-INFRA-6426-4: Duration Recomputes From Monotonic Phase Intervals

GIVEN queue wait, model load, generation, exact verification, and artifact
write phase rows
WHEN Exp6426 recomputes duration
THEN negative, overlapping-unexplained, missing, synthesized, or wall-clock-only
intervals fail closed.

### SCENARIO-INFRA-6426-5: Attribution Attacks Fail Closed

GIVEN forged PID, stale GPU sample, model-name-only substitution, raw-output
reuse, runner swap, clock rollback, truncated receipt, concurrency collision,
CPU fallback, or child-exit omission
WHEN the contract validator evaluates the mutated rows
THEN every critical attack is rejected and recorded in `attack_matrix`.

## Implementation Status (REQ-INFRA-6426)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6426 | Planned: `python/carnot/task_runtime_receipts.py` and `python/carnot/experiment_6426_task_scoped_runtime_receipt_contract.py`; terminal artifact `results/experiment_6426_task_scoped_runtime_receipt_contract.json`. | Planned: `tests/python/test_experiment_6426_task_scoped_runtime_receipt_contract.py`. |

## REQ-INFRA-6427: Fresh Constraint-Saturation Factor Corpus SHALL Bind Row-First Exact Outcomes To Task-Scoped Runtime Receipts

Carnot SHALL provide Exp6427 at
`python/carnot/experiment_6427_fresh_constraint_saturation_factor_corpus.py`.
The command
`.venv/bin/python -m carnot.experiment_6427_fresh_constraint_saturation_factor_corpus --date 20260814`
SHALL write
`results/experiment_6427_fresh_constraint_saturation_factor_corpus.json`.

Exp6427 SHALL revalidate the Exp6426 runtime-receipt gate, CUDA devices, free
VRAM, model bytes, embedded GGUF tokenizers, runner selection, task-scoped
receipt helper, exact checkers, license matrix, disk space, source manifests,
and absence of the fresh raw-output directory before generation. It SHALL use
`cached_sota_pair()` helper calls to resolve
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. It SHALL use embedded GGUF tokenizers only
and SHALL never call `AutoTokenizer`.

Exp6427 SHALL preregister at least 144 rows before generation. The matrix SHALL
be balanced across the three model families, three factor families,
simultaneous constraint-count buckets `1-2`, `3-4`, `5-6`, and `7-8`,
independent and interacting constraint classes, and fixed seeds. Acquisition,
calibration, and untouched future partitions SHALL be sealed before raw output
bytes are parsed.

Every event SHALL be generated through the task-scoped receipt helper. Each row
SHALL bind a unique prompt hash, raw-output byte hash, model hash, PID, GPU
sample binding, event time, source identity, source license, checker identity,
and partition. Raw outputs SHALL not reuse Exp6414 bytes and SHALL not be reused
across event IDs.

Exp6427 SHALL parse only the factor proposal surface. It SHALL not be a
finite-ID generated-answer experiment, grammar experiment, parser-retry
experiment, hidden-state experiment, or external-text-scoring experiment.
Deterministic exact rules SHALL score each constraint and joint satisfaction.
The artifact SHALL record evaluable, correct, abstained, malformed, truncated,
duplicate, unsupported, and unlicensed outcomes without model pooling.

Exp6427 SHALL write `per_unit_rows` before computing any aggregate. It SHALL
recompute per-constraint success, joint success, exact yield, abstention, and
cost by model, family, constraint count, and interaction class from those rows.
Reported aggregate deltas SHALL be zero when readiness is positive.

The attack matrix SHALL cover model substitution, raw-output reuse, prompt
leakage, event reordering, source fabrication, checker swap, duplicated
effects, pooled identities, CPU fallback, clock truncation, future-label
leakage, and adversarial duration under-reporting. Each attack SHALL fail
closed.

The terminal artifact SHALL include `status`, `exp6426_gate_receipt`,
`MODEL_SPECS`, `models_used`, `cached_sota_pair_receipts`,
`model_file_and_embedded_tokenizer_hashes`, `autotokenizer_usage_count`,
`runner_and_task_scoped_runtime_receipts`,
`manifest_path_hash_counts_balance_and_partition_seals`,
`preregistered_model_family_constraint_count_interaction_and_seed_matrix`,
`per_unit_rows`,
`per_row_prompt_raw_output_model_pid_gpu_source_license_checker_event_time_and_partition_bindings`,
`per_row_constraint_results_and_joint_exact_outcome`,
`per_model_family_constraint_count_and_interaction_results`,
`per_constraint_success`, `joint_success`, `exact_yield`, `abstention_rate`,
`malformed_count`, `truncation_count`, `duplicate_count`,
`raw_output_reuse_count`, `cpu_fallback_count`, `protected_leakage_count`,
`aggregate_recomputation_receipts`, `reported_vs_recomputed_deltas`,
`task_phase_duration_receipts`, `attack_matrix`,
`fresh_row_recomputable_factor_corpus_ready_score`,
`current_adversarial_flag_count`, `harm_underpowered_missing_and_flagged_cells`,
`protected_files_unchanged`, `blocked_reason`, `preconditions_checked`,
`inference_substrate`, `verifier_is_oracle`, `field_principles`,
`field_provenance`, `random_seed`, `duration_s`, `tests_run`,
`reproducibility_checksum`, and `honest_verdict`.

`verifier_is_oracle` SHALL be true only for deterministic per-constraint and
joint exact checks. Model output and factor proposals SHALL not be oracles.
`fresh_row_recomputable_factor_corpus_ready_score` SHALL be bare `1.0` only
when all three model families have authentic cells, every aggregate recomputes
from rows, protected leakage and raw-output reuse are zero, all attacks fail
closed, and `current_adversarial_flag_count` is zero. Otherwise it SHALL be
bare `0.0`.

Field principles SHALL be:

- `status`: Names whether the row-recomputable corpus is complete, blocked, or null.
- `exp6426_gate_receipt`: Pins task-scoped receipt readiness before new rows rely on it.
- `MODEL_SPECS`: Lists only the three mandated local GGUF model identities.
- `models_used`: Counts only authenticated rows from the three mandated model families.
- `cached_sota_pair_receipts`: Shows every model came through the helper path.
- `model_file_and_embedded_tokenizer_hashes`: Binds local model bytes and embedded tokenizer hashes.
- `autotokenizer_usage_count`: Must stay zero because GGUF tokenizers are embedded.
- `runner_and_task_scoped_runtime_receipts`: Binds the runner and helper schema used by each event.
- `manifest_path_hash_counts_balance_and_partition_seals`: Shows the fresh row matrix and sealed partitions.
- `preregistered_model_family_constraint_count_interaction_and_seed_matrix`: Freezes strata and seeds before generation.
- `per_unit_rows`: Provides the immutable rows from which every comparative claim recomputes.
- `per_row_prompt_raw_output_model_pid_gpu_source_license_checker_event_time_and_partition_bindings`: Binds each row to prompt, bytes, model, process, GPU, source, license, checker, time, and split.
- `per_row_constraint_results_and_joint_exact_outcome`: Stores deterministic per-constraint and joint outcomes.
- `per_model_family_constraint_count_and_interaction_results`: Reports row-derived strata without pooling model identities.
- `per_constraint_success`: Reports per-constraint success recomputed from rows.
- `joint_success`: Reports joint exact success recomputed from rows.
- `exact_yield`: Reports evaluable exact yield recomputed from rows.
- `abstention_rate`: Reports abstention from rows, including unsupported and unlicensed rows.
- `malformed_count`: Counts malformed raw parses from rows.
- `truncation_count`: Counts truncated outputs from rows.
- `duplicate_count`: Counts duplicate event or effect surfaces from rows.
- `raw_output_reuse_count`: Must stay zero because reused raw bytes invalidate row independence.
- `cpu_fallback_count`: Must stay zero for authenticated local GGUF rows.
- `protected_leakage_count`: Must stay zero because future labels and exact answers stay sealed.
- `aggregate_recomputation_receipts`: Shows aggregate formulas and row hashes used for recomputation.
- `reported_vs_recomputed_deltas`: Shows reported metrics equal recomputed metrics.
- `task_phase_duration_receipts`: Reports measured monotonic phase intervals.
- `attack_matrix`: Proves known substitution, leakage, reuse, pooling, fallback, and duration attacks fail closed.
- `fresh_row_recomputable_factor_corpus_ready_score`: Bare gate for downstream use.
- `current_adversarial_flag_count`: Must be zero for clean evidence.
- `harm_underpowered_missing_and_flagged_cells`: Names missing or flagged cells instead of hiding them.
- `protected_files_unchanged`: Shows protected upstream and ops files stayed byte-stable.
- `blocked_reason`: Names any precondition blocker.
- `preconditions_checked`: Lists host, model, receipt, raw-dir, license, source, and checker gates.
- `inference_substrate`: Declares fresh task-scoped local GGUF factor-corpus generation.
- `verifier_is_oracle`: Marks only deterministic exact checks as oracles.
- `field_principles`: Documents why each required field exists.
- `field_provenance`: States how each field was produced.
- `random_seed`: Pins the row matrix and deterministic outcomes.
- `duration_s`: Records measured wall time without padding.
- `tests_run`: Records focused, coverage, E2E, adversarial, spec, global, and root checks.
- `reproducibility_checksum`: Content-addresses the payload with volatile fields normalized.
- `honest_verdict`: Gives a terminal-prefix verdict and the narrow evidence boundary.
- `gate:exp6426`: Exp6426 is a gate for receipt mechanics, not a semantic oracle.
- `stratum:model_family`: Model-family rows are disaggregated before any summary.
- `stratum:factor_family`: Factor-family rows are disaggregated before any summary.
- `stratum:constraint_count_bucket`: Constraint-count buckets test joint saturation effects.
- `stratum:interaction_class`: Independent and interacting rows test different exact-satisfaction surfaces.

### SCENARIO-INFRA-6427-1: Preregistration Is Balanced And Sealed

GIVEN the three mandated GGUF model families
WHEN Exp6427 builds its manifest
THEN it preregisters at least 144 rows balanced by model family, factor family,
constraint-count bucket, interaction class, and seed, with acquisition,
calibration, and future partitions sealed before generation.

**Spec traces:** REQ-INFRA-6427

### SCENARIO-INFRA-6427-2: Rows Bind Runtime Receipts And Unique Raw Bytes

GIVEN a preregistered event row
WHEN Exp6427 generates and stores the raw factor proposal bytes
THEN the row binds prompt hash, raw-output hash, model hash, PID, GPU sample,
event time, source identity, source license, checker identity, and partition,
and raw-output reuse remains zero.

**Spec traces:** REQ-INFRA-6427

### SCENARIO-INFRA-6427-3: Exact Outcomes And Aggregates Recompute From Rows

GIVEN immutable `per_unit_rows`
WHEN Exp6427 reports per-constraint success, joint success, exact yield,
abstention, malformed, truncation, duplicate, and cost metrics
THEN each aggregate recomputes from rows with zero reported-vs-recomputed delta.

**Spec traces:** REQ-INFRA-6427

### SCENARIO-INFRA-6427-4: Model Pooling And Leakage Fail Closed

GIVEN attacks for model substitution, pooled identities, prompt leakage,
future-label leakage, source fabrication, checker swap, event reordering, and
duplicated effects
WHEN Exp6427 validates the artifact
THEN every attack fails closed and readiness remains zero for any accepted
attack.

**Spec traces:** REQ-INFRA-6427

### SCENARIO-INFRA-6427-5: Readiness Requires Clean Current Evidence

GIVEN all three model families have authentic cells
WHEN Exp6427 computes
`fresh_row_recomputable_factor_corpus_ready_score`
THEN the score is `1.0` only when all aggregates are row-recomputed,
raw-output reuse, CPU fallback, protected leakage, and adversarial flag counts
are zero, protected files are unchanged, and all tests exit zero.

**Spec traces:** REQ-INFRA-6427

## Implementation Status (REQ-INFRA-6427)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6427 | Planned: `python/carnot/experiment_6427_fresh_constraint_saturation_factor_corpus.py`; terminal artifact `results/experiment_6427_fresh_constraint_saturation_factor_corpus.json`. | Planned: `tests/python/test_experiment_6427_fresh_constraint_saturation_factor_corpus.py`. |

## REQ-INFRA-6450: SOTA Fixed-Policy Candidate Corpus SHALL Bind Typed Tool-Use Plans To Fresh GGUF Bytes And Exact Simulation Labels

Carnot SHALL provide Exp6450 at
`python/carnot/experiment_6450_sota_fixed_policy_candidate_corpus.py`.
The command
`.venv/bin/python -m carnot.experiment_6450_sota_fixed_policy_candidate_corpus --date 20260815`
SHALL write
`results/experiment_6450_sota_fixed_policy_candidate_corpus.json`.

Exp6450 SHALL build a fresh V555 corpus of fixed-policy tool-use problems and
matched candidate action plans. It SHALL avoid the retired finite-ID GGUF
generated-answer transport lane. It SHALL use executable typed actions, a fixed
parser, and an exact local simulator. It SHALL not make a model-ranking claim.

Before inference, Exp6450 SHALL fail closed unless both RTX 3090 GPUs are
visible, mandatory cached GGUF files exist, free VRAM is sufficient, embedded
GGUF tokenizers load, exact simulator imports succeed, disk and clock checks
pass, protected files are hashed, fresh raw-output paths do not exist, and a
sealed problem and partition manifest is written. It SHALL use
`cached_sota_pair()` or the same cache resolver for
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. It SHALL never call `AutoTokenizer` for a
GGUF.

Exp6450 SHALL seal 36 independent policy problems before generation. Each
problem SHALL include fixed entities, observable facts, rule clauses, protected
clauses, a typed tool-action schema, and a deterministic final-state checker.
The sealed manifest SHALL freeze 12 development problems, 12 allocation-held
problems, and 12 selection-held problems by hash before any raw output is read.
Partition membership SHALL not change after sealing. Held exact labels SHALL not
enter prompts or parser inputs.

For every problem and mandated model family, Exp6450 SHALL generate at least
three candidate action plans with frozen prompts, decoding settings, budgets,
and seeds. Raw generated bytes SHALL be written before parsing. One fixed parser
SHALL parse all candidates. Parse failures SHALL remain visible outcomes. The
workflow SHALL not reprompt, perform grammar retries, repair parser failures, or
convert structural validity into semantic success.

Every parsed candidate SHALL run through the exact simulator. The row SHALL
store legality, protected-clause result, goal result, exact success, checker
work, path-stage hashes, raw hash, task-scoped CUDA receipt, timing, partition,
model id, candidate id, and seed. Aggregates SHALL recompute only from
`per_unit_rows`.

Exp6450 SHALL test attacks for output reuse, model-name substitution, hidden CPU
fallback, parser repair, held-label leakage, partition reassignment, duplicate
candidates, exact-veto bypass, and aggregate-row mismatch. Each critical attack
SHALL fail closed. `sota_corpus_ready_score` SHALL be `1.0` only when all three
model families have authenticated eligible rows, raw hashes are unique,
partitions stayed sealed, exact labels recompute, each partition has mixed exact
outcomes and candidate-selection headroom, duration passes live-model checks,
protected files are unchanged, and critical findings are zero.

The terminal artifact SHALL include `status`, `MODEL_SPECS`, `models_used`,
`cached_sota_pair_receipts`,
`model_file_and_embedded_tokenizer_hashes`, `autotokenizer_usage_count`,
`device_and_runner_receipts`, `sealed_problem_and_partition_manifest`,
`preexistence_and_freshness_receipts`,
`fixed_action_schema_and_parser_hash`,
`exact_simulator_and_checker_hashes`, `raw_output_manifest`,
`per_unit_rows`, `eligible_rows_by_model_and_partition`,
`parse_failures_by_model`, `exact_outcomes_by_model_and_partition`,
`candidate_headroom_by_partition`, `raw_output_uniqueness_and_reuse_count`,
`cpu_fallback_count`, `aggregate_row_recomputation`, `attack_matrix`,
`current_adversarial_findings`, `sota_corpus_ready_score`,
`protected_files_unchanged`, `blocked_reason`, `gate_check_summary`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`field_principles`, `field_provenance`, `random_seed`, `duration_s`,
`tests_run`, `reproducibility_checksum`, and `honest_verdict`.

`verifier_is_oracle` SHALL be true only for the deterministic local simulator
and final-state checker. The model and parser SHALL not be oracles.

### SCENARIO-INFRA-6450-1: Preconditions Seal Before Inference

GIVEN the three mandated GGUF families and a fresh output location
WHEN Exp6450 starts
THEN it writes a sealed development, allocation-held, and selection-held
manifest before inference
AND it blocks before generation if GPU, VRAM, model, tokenizer, simulator,
fresh-path, disk, or protected-file checks fail.

**Spec traces:** REQ-INFRA-6450

### SCENARIO-INFRA-6450-2: Candidate Rows Preserve Raw Bytes And Fixed Parsing

GIVEN sealed problems and frozen prompts
WHEN Exp6450 generates candidate plans
THEN every problem, model, candidate, and seed has a `per_unit_rows` entry
with stored raw bytes, one fixed parser result, no parser retry, and no
finite-ID generated-answer lane.

**Spec traces:** REQ-INFRA-6450

### SCENARIO-INFRA-6450-3: Exact Simulator Owns The Labels

GIVEN parsed typed action plans
WHEN Exp6450 runs the local simulator
THEN legality, protected-clause status, goal status, exact success, checker
work, and path-stage hashes recompute from the row without trusting model text
or parser validity as semantic success.

**Spec traces:** REQ-INFRA-6450

### SCENARIO-INFRA-6450-4: Aggregates And Headroom Recompute From Rows

GIVEN immutable `per_unit_rows`
WHEN Exp6450 reports eligible rows, parse failures, exact outcomes, raw-output
reuse, and candidate headroom
THEN every aggregate recomputes from rows and each sealed partition shows mixed
exact outcomes plus real candidate-selection headroom before readiness can be
positive.

**Spec traces:** REQ-INFRA-6450

### SCENARIO-INFRA-6450-5: Attacks Fail Closed

GIVEN attacks for output reuse, model-name substitution, CPU fallback, parser
repair, held-label leakage, partition reassignment, duplicate candidates,
exact-veto bypass, and aggregate mismatch
WHEN Exp6450 validates the artifact
THEN every attack is detected, readiness is zero for any accepted attack, and
the artifact makes no model-ranking claim.

**Spec traces:** REQ-INFRA-6450

## Implementation Status (REQ-INFRA-6450)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6450 | Planned: `python/carnot/experiment_6450_sota_fixed_policy_candidate_corpus.py`; terminal artifact `results/experiment_6450_sota_fixed_policy_candidate_corpus.json`. | Planned: `tests/python/test_experiment_6450_sota_fixed_policy_candidate_corpus.py`. |

## REQ-INFRA-6462: SOTA Raw Persistence Canary SHALL Bind One Event To One Durable Raw Path Before Parsing

Carnot SHALL provide Exp6462 at
`python/carnot/experiment_6462_sota_raw_persistence_uniqueness_canary.py`.
The command
`.venv/bin/python -m carnot.experiment_6462_sota_raw_persistence_uniqueness_canary --date 20260819`
SHALL write
`results/experiment_6462_sota_raw_persistence_uniqueness_canary.json`.

Exp6462 SHALL run a small live matrix across the three mandated GGUF model ids:
`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. It SHALL use `cached_sota_pair()` or the
same cache resolver. It SHALL use embedded GGUF tokenizers only. It SHALL never
call `AutoTokenizer` for a GGUF repository. Legacy models MAY appear only in
blocked or smoke-test records and SHALL NOT supply canary rows.

Before inference, Exp6462 SHALL fail closed unless both RTX 3090 GPUs are
visible, each mandated model file is cached and hashed, each embedded tokenizer
loads, free VRAM and disk pass, a monotonic clock is available, the result path
and raw-output tree are fresh, and protected files are hashed. If any gate
fails, Exp6462 SHALL write a blocked artifact with `blocked_reason`,
`gate_check_summary`, and `preconditions_checked`. It SHALL run no generation
after a failed precondition.

Exp6462 SHALL seal at least four fixed units before generation. For each unit
and each mandated model, it SHALL generate at least two independent outputs
with frozen prompts, decoding settings, and seeds. Before each generation it
SHALL allocate the final raw path and event id. The raw bytes SHALL be written
to a same-directory temporary file, flushed and fsynced where supported,
atomically renamed, and verified by byte count and SHA-256 before parsing.

Every normal generation row SHALL bind event id, unit id, model file hash,
embedded tokenizer hash, device samples, prompt hash, raw hash, parse hash,
checker hash, verdict, raw path, and atomic-write receipt through the Exp6449
path receipt stages. Text equality across two live outputs SHALL be diagnostic
only. It SHALL NOT be used as event uniqueness. Event uniqueness SHALL come
from one-to-one event, path, and durable hash binding.

Exp6462 SHALL emit `per_unit_rows` for every normal generation and every
injected attack. The attack matrix SHALL cover zero-byte rename, stale
preexisting path, reused event id, same raw path under two rows, cloned
candidate row, model substitution, CPU fallback, and receipt reordering. Each
critical attack SHALL fail closed. `raw_persistence_canary_ready_score` SHALL be
`1.0` only when every normal generation has nonzero durable bytes, every normal
row has a one-event/one-path/one-hash binding, every path receipt validates
before parse, CPU fallback count is zero, all reported aggregates recompute
from rows, protected files are unchanged, and all critical attacks fail closed.

The terminal artifact SHALL include `status`, `MODEL_SPECS`, `models_used`,
`cached_sota_pair_receipts`,
`model_file_and_embedded_tokenizer_hashes`, `autotokenizer_usage_count`,
`device_and_runner_receipts`, `sealed_unit_manifest`,
`event_path_allocation_receipts`, `atomic_write_receipts`,
`raw_output_manifest`, `per_unit_rows`,
`one_event_one_path_one_hash_check`, `nonzero_durable_byte_check`,
`raw_text_equality_diagnostic`, `cpu_fallback_count`, `attack_matrix`,
`aggregate_row_recomputation`, `current_adversarial_findings`,
`raw_persistence_canary_ready_score`, `protected_files_unchanged`,
`blocked_reason`, `gate_check_summary`, `preconditions_checked`,
`inference_substrate`, `verifier_is_oracle`, `field_principles`,
`field_provenance`, `random_seed`, `duration_s`, `tests_run`,
`reproducibility_checksum`, and `honest_verdict`.

`verifier_is_oracle` SHALL be true only for byte count checks, SHA-256 checks,
receipt-chain validation, and exact checker arithmetic over the row bindings.
It SHALL NOT make model-output semantics an oracle.

### SCENARIO-INFRA-6462-1: Preconditions Fail Closed Before Inference

GIVEN the three mandated GGUF families and a fresh output location
WHEN Exp6462 starts
THEN it blocks before inference if any GPU, VRAM, cache, tokenizer, disk,
clock, or fresh-path precondition fails
AND it records each failed gate in `preconditions_checked`,
`blocked_reason`, and `gate_check_summary`.

**Spec traces:** REQ-INFRA-6462

### SCENARIO-INFRA-6462-2: Raw Bytes Persist Atomically Before Parse

GIVEN a sealed unit, model, seed, and allocated event id
WHEN Exp6462 stores the generated raw output
THEN the final path is allocated before generation, a temporary file is renamed
atomically, durable byte count and SHA-256 are verified before parse, and the
row records the allocation and write receipts.

**Spec traces:** REQ-INFRA-6462

### SCENARIO-INFRA-6462-3: Path Receipts Bind Event Identity

GIVEN normal canary rows
WHEN Exp6462 builds the Exp6449-style path chain
THEN every row binds event id, unit id, model hash, tokenizer hash, device
sample hash, prompt hash, raw hash, parse hash, checker hash, and verdict
without using raw text equality as event identity.

**Spec traces:** REQ-INFRA-6462

### SCENARIO-INFRA-6462-4: Persistence And Identity Attacks Fail Closed

GIVEN zero-byte rename, stale preexisting path, reused event id, same raw path,
cloned candidate row, model substitution, CPU fallback, and receipt reordering
attacks
WHEN Exp6462 validates the canary rows
THEN every attack is rejected, `per_unit_rows` retains the injected attack row,
and readiness stays below `1.0` for any accepted attack.

**Spec traces:** REQ-INFRA-6462

## Implementation Status (REQ-INFRA-6462)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6462 | Planned: `python/carnot/experiment_6462_sota_raw_persistence_uniqueness_canary.py`; terminal artifact `results/experiment_6462_sota_raw_persistence_uniqueness_canary.json`. | Planned: `tests/python/test_experiment_6462_sota_raw_persistence_uniqueness_canary.py`. |

## REQ-INFRA-6463: SOTA Fixed-Policy Candidate Corpus V2 SHALL Use Sealed Four-Way Partitions And One Raw Event File Per Candidate

Carnot SHALL provide Exp6463 at
`python/carnot/experiment_6463_sota_fixed_policy_candidate_corpus_v2.py`.
The command
`.venv/bin/python -m carnot.experiment_6463_sota_fixed_policy_candidate_corpus_v2 --date 20260819`
SHALL write
`results/experiment_6463_sota_fixed_policy_candidate_corpus_v2.json`.

Exp6463 SHALL first read the Exp6462 canary artifact. It SHALL stop before
model resolution or inference unless `raw_persistence_canary_ready_score` is
`1.0`. A failed pre-gate SHALL emit `blocked_gate_check_failed`,
`gate_check_summary`, and all required empty evidence fields.

Exp6463 SHALL use only these local GGUF model ids through cached local
resolution and embedded tokenizer checks: `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, and
`unsloth/gemma-4-26B-A4B-it-GGUF`. It SHALL never call `AutoTokenizer` for a
GGUF. It SHALL not make a model-ranking claim.

Exp6463 SHALL seal at least 48 fixed-policy units before inference across
development, allocation-held, selection-held, and audit-held partitions: 12
units in each partition.
The sealed manifest SHALL hash each problem, each partition membership, and
each precommitted candidate label before generation. Held labels and partition
membership SHALL not enter prompts.

For each sealed unit and each mandated model, Exp6463 SHALL generate at least
three candidate plans. Each candidate SHALL receive a fresh event id and a final
raw path before generation. The raw bytes SHALL be written with a same-directory
temporary file, fsync where supported, atomic rename, and post-rename SHA-256
and byte-count verification before parsing. A checkpoint SHALL be written after
each event. Resume SHALL skip completed events and SHALL not repeat generation
for any completed event.

Exp6463 SHALL parse each raw candidate once with the fixed parser. It SHALL
record parse failures without repair, reprompt, grammar retry, or parser retry.
Parsed candidates SHALL run through the deterministic simulator and exact
checker. Every `per_unit_rows` row SHALL include the unit, partition, model,
candidate, event id, raw path and hash, parse result, exact result, device
receipt, and timing.

Exp6463 SHALL report eligible rows, parse failures, exact outcomes, candidate
headroom, event identity, and aggregate checks only from `per_unit_rows`. Each
held partition SHALL have mixed exact outcomes and positive candidate-selection
headroom before readiness can be positive. Exact successes and parse failures
SHALL be grouped by model without ranking models.

Exp6463 SHALL attack zero-byte files, event reuse, candidate cloning, held
exposure, membership reassignment, parser repair, CPU fallback, exact-veto
bypass, and aggregate mismatch. Each critical attack SHALL fail closed.
`sota_corpus_ready_score` SHALL be `1.0` only when all mandated models have
eligible rows, every normal event is provenance-complete, partitions stayed
sealed, labels recompute, every held split has headroom, CPU fallback is zero,
protected files are unchanged, and critical findings are zero.

The terminal artifact SHALL include `status`, `MODEL_SPECS`, `models_used`,
`cached_sota_pair_receipts`,
`model_file_and_embedded_tokenizer_hashes`, `autotokenizer_usage_count`,
`device_and_runner_receipts`, `sealed_problem_and_partition_manifest`,
`exposure_ledger`, `checkpoint_and_resume_receipts`, `raw_output_manifest`,
`event_identity_manifest`, `fixed_parser_and_checker_hashes`,
`per_unit_rows`, `eligible_rows_by_model_and_partition`,
`parse_failures_by_model`, `exact_outcomes_by_model_and_partition`,
`candidate_headroom_by_partition`, `one_event_one_path_one_hash_check`,
`cpu_fallback_count`, `aggregate_row_recomputation`, `attack_matrix`,
`current_adversarial_findings`, `sota_corpus_ready_score`,
`protected_files_unchanged`, `blocked_reason`, `gate_check_summary`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`field_principles`, `field_provenance`, `random_seed`, `duration_s`,
`tests_run`, `reproducibility_checksum`, and `honest_verdict`.

`verifier_is_oracle` SHALL be true only for the deterministic simulator,
checker, and row arithmetic. The model and parser SHALL not be oracles.

### SCENARIO-INFRA-6463-1: Pre-Gate And Four-Way Seal

GIVEN the Exp6462 canary artifact and the three mandated GGUF families
WHEN Exp6463 starts
THEN it blocks before model generation if the canary ready score is not `1.0`
AND otherwise writes a sealed 12/12/12/12 development, allocation-held,
selection-held, and audit-held manifest before inference.

**Spec traces:** REQ-INFRA-6463

### SCENARIO-INFRA-6463-2: Event Raw Bytes Persist Before Fixed Parse

GIVEN sealed units, models, and candidate ids
WHEN Exp6463 generates candidates
THEN each event has one fresh event id, one allocated raw path, one nonzero raw
file hash, one fixed parse attempt, no parser repair, and one checkpoint write
after the event.

**Spec traces:** REQ-INFRA-6463

### SCENARIO-INFRA-6463-3: Exact Labels And Held Headroom Recompute

GIVEN immutable `per_unit_rows`
WHEN Exp6463 reports exact outcomes and candidate headroom
THEN exact labels recompute from the deterministic simulator and each held
partition has mixed exact outcomes plus positive candidate-selection headroom.

**Spec traces:** REQ-INFRA-6463

### SCENARIO-INFRA-6463-4: Event Identity And Adversarial Controls Fail Closed

GIVEN attacks for zero-byte files, event reuse, candidate cloning, held
exposure, membership reassignment, parser repair, CPU fallback, exact-veto
bypass, and aggregate mismatch
WHEN Exp6463 validates the artifact
THEN every attack is detected, readiness is zero for any accepted attack, and
the artifact makes no model-ranking claim.

**Spec traces:** REQ-INFRA-6463

## Implementation Status (REQ-INFRA-6463)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6463 | Planned: `python/carnot/experiment_6463_sota_fixed_policy_candidate_corpus_v2.py`; terminal artifact `results/experiment_6463_sota_fixed_policy_candidate_corpus_v2.json`. | Planned: `tests/python/test_experiment_6463_sota_fixed_policy_candidate_corpus_v2.py`. |

## REQ-INFRA-6480: V557 Terminal Evidence Freeze SHALL Preserve V558 Branch Boundaries Without Queue Activation

Carnot SHALL provide Exp6480 at
`python/carnot/experiment_6480_v557_terminal_evidence_and_v558_preflight.py`.
The command
`.venv/bin/python -m carnot.experiment_6480_v557_terminal_evidence_and_v558_preflight --date 20260821`
SHALL write
`results/experiment_6480_v557_terminal_evidence_and_v558_preflight.json`.

Exp6480 SHALL freeze the seven active V557 task artifacts, Exp6473 through
Exp6479. It SHALL record one terminal row per task. Each row SHALL include the
task id, artifact path, existence, byte size, SHA-256 hash, terminal artifact
state, `status`, `honest_verdict`, readiness fields, normalized gate
diagnostics, and adversarial status. Missing, zero-byte, malformed, blocked,
null, and complete artifacts SHALL remain separate states.

Exp6480 SHALL recompute branch boundaries only from upstream artifacts. It
SHALL recompute the Exp6463 lineage retirement from Exp6476. The boundary
SHALL state that V558 may create a new prospective lineage. It SHALL state that
V558 may not reuse Exp6463 held evidence.

Exp6480 SHALL recompute `v557_factor_cache_ready_score` only from Exp6479
`factor_cache_shadow_adapter_ready_score` and exact default-off,
write-admission, persistence, rollback, and test gates. It SHALL not claim a
continuous-learning benefit or release authority.

Exp6480 SHALL recompute `v557_arc_shield_ready_score` only from Exp6471
`arc_safety_shield_ready_score`, current adversarial status, and the no-solve
boundary. It SHALL not claim policy improvement or an ARC solve.

Exp6480 SHALL recompute exact-energy evidence status from Exp6477 and Exp6478.
It SHALL preserve the Exp6478 result as finite no-LLM unit-seed evidence. It
SHALL not extend that result to local-SOTA model outputs.

Exp6480 SHALL set `staged_queue_validation_performed=false`,
`roadmap_activation_performed=false`, and `unrelated_branch_gate_count=0`.
It SHALL not validate or activate a roadmap. It SHALL not repair or rerun any
V557 science task.

The terminal artifact SHALL include `status`, `v557_terminal_rows`,
`artifact_hash_manifest`, `retirement_boundary_rows`,
`exact_energy_evidence_boundary`, `v557_factor_cache_ready_score`,
`v557_arc_shield_ready_score`, `staged_queue_validation_performed`,
`roadmap_activation_performed`, `unrelated_branch_gate_count`,
`per_unit_rows`, `aggregate_row_recomputation`, `protected_files_unchanged`,
`gate_check_summary`, `preconditions_checked`, `inference_substrate`,
`verifier_is_oracle`, `field_principles`, `field_provenance`, `random_seed`,
`duration_s`, `tests_run`, `reproducibility_checksum`, and
`honest_verdict`. `inference_substrate` SHALL equal
`aggregation_from_upstream_artifacts`. `verifier_is_oracle` SHALL be true only
for hashes and deterministic row arithmetic.

Field principles SHALL use this map:

| Field | Principle |
|---|---|
| `status` | A terminal status distinguishes a completed evidence freeze from an interrupted handoff. |
| `v557_terminal_rows` | One row per active V557 task prevents missing or blocked evidence from disappearing in a summary. |
| `artifact_hash_manifest` | Hashes bind each terminal determination to the exact evidence bytes. |
| `retirement_boundary_rows` | Explicit rows keep the invalid Exp6463 lineage from returning under a new name. |
| `exact_energy_evidence_boundary` | The boundary preserves Exp6478 as a finite no-LLM result and prevents a false local-SOTA claim. |
| `v557_factor_cache_ready_score` | A narrow score gives later CSL work a same-roadmap gate without granting benefit or release authority. |
| `v557_arc_shield_ready_score` | A narrow score confirms the generic no-solve shield without claiming policy improvement. |
| `staged_queue_validation_performed` | A false value proves this task did not repeat the retired queue-transition scope. |
| `roadmap_activation_performed` | A false value keeps evidence aggregation separate from conductor state changes. |
| `unrelated_branch_gate_count` | Zero unrelated gates prevents an infrastructure result from suppressing independent science. |
| `per_unit_rows` | Task rows make every aggregate and branch boundary independently checkable. |
| `aggregate_row_recomputation` | Row-derived aggregates catch summaries that disagree with terminal evidence. |
| `protected_files_unchanged` | The task must not alter the active roadmap, conductor, registry, or public results. |
| `gate_check_summary` | Any blocked verdict names the failed check, expected value, observed value, and evidence path. |
| `preconditions_checked` | Precondition receipts prove the expected artifacts and repository state existed before aggregation. |
| `inference_substrate` | Declaring aggregation_from_upstream_artifacts prevents a no-model audit from being read as live inference. |
| `verifier_is_oracle` | Only deterministic hashes and row arithmetic are authoritative in this task. |
| `field_principles` | A field-to-principle map preserves the reason for every evidence field. |
| `field_provenance` | Exact source paths and hashes make each value traceable. |
| `random_seed` | A fixed seed makes attack ordering reproducible. |
| `duration_s` | Measured wall time detects a bootstrap-only artifact. |
| `tests_run` | Recorded commands distinguish executed checks from intended checks. |
| `reproducibility_checksum` | A stable checksum detects later drift in inputs or the terminal artifact. |
| `honest_verdict` | The verdict states completion and each branch boundary without promoting a science claim. |

### SCENARIO-INFRA-6480-1: Seven V557 Terminal Rows Are Frozen

GIVEN completed V557 artifacts Exp6473 through Exp6479
WHEN Exp6480 builds terminal rows
THEN each active V557 task has one row with path, size, hash, status,
`honest_verdict`, readiness fields, gate diagnostics, and adversarial status.

**Spec traces:** REQ-INFRA-6480

### SCENARIO-INFRA-6480-2: Artifact States Stay Distinct

GIVEN missing, zero-byte, malformed, blocked, null, and complete artifacts
WHEN Exp6480 classifies artifact state
THEN each state remains explicit and cannot be merged into a summary-only
terminal result.

**Spec traces:** REQ-INFRA-6480

### SCENARIO-INFRA-6480-3: Exp6463 Lineage Retirement Is Recomputed

GIVEN Exp6476 reports missing immutable pre-inference held label and membership
proof
WHEN Exp6480 writes `retirement_boundary_rows`
THEN Exp6463 is retired for held evidence reuse, while a new prospective V558
lineage remains allowed.

**Spec traces:** REQ-INFRA-6480

### SCENARIO-INFRA-6480-4: Narrow Readiness Scores Use Only Allowed Inputs

GIVEN Exp6479 and Exp6471 readiness fields
WHEN Exp6480 computes `v557_factor_cache_ready_score` and
`v557_arc_shield_ready_score`
THEN the factor score uses only default-off, exact write-admission,
persistence, rollback, and test gates
AND the ARC score uses only shield readiness, adversarial status, and
`no_solve_claim`.

**Spec traces:** REQ-INFRA-6480

### SCENARIO-INFRA-6480-5: Exact-Energy Boundary Does Not Become Local-SOTA

GIVEN Exp6477 exact-record readiness and Exp6478 finite no-LLM selection rows
WHEN Exp6480 computes `exact_energy_evidence_boundary`
THEN it records finite unit-seed support and sets local-SOTA extension false.

**Spec traces:** REQ-INFRA-6480

### SCENARIO-INFRA-6480-6: Artifact Is Annotated And Non-Activating

GIVEN hashes, rows, gates, protected files, and command receipts
WHEN Exp6480 validates the artifact
THEN every required field has a principle and provenance, the checksum
matches, protected files are unchanged, `staged_queue_validation_performed` is
false, `roadmap_activation_performed` is false, and
`unrelated_branch_gate_count` is zero.

**Spec traces:** REQ-INFRA-6480

## Implementation Status (REQ-INFRA-6480)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6480 | Planned: `python/carnot/experiment_6480_v557_terminal_evidence_and_v558_preflight.py`; terminal artifact `results/experiment_6480_v557_terminal_evidence_and_v558_preflight.json`. | Planned: `tests/python/test_experiment_6480_v557_terminal_evidence_and_v558_preflight.py`. |

## REQ-INFRA-6481: Monotonic Phase And Concurrency Receipts SHALL Bind Task Attempts To Processes, Resources, Dependencies, And Outputs

Carnot SHALL provide an optional experiment-local receipt API at
`python/carnot/phase_concurrency_receipts.py` and Exp6481 at
`python/carnot/experiment_6481_monotonic_phase_concurrency_receipt_contract.py`.
The command
`.venv/bin/python -m carnot.experiment_6481_monotonic_phase_concurrency_receipt_contract --date 20260821`
SHALL write
`results/experiment_6481_monotonic_phase_concurrency_receipt_contract.json`.

The receipt schema SHALL be versioned and hash-bound. It SHALL record the
phases `queue_wait`, `dependency_resolution`, `resource_acquisition`,
`model_or_fixture_load`, `execution`, `exact_verification`, `artifact_write`,
and `resource_release`. Each phase row SHALL include task id, attempt id, PID,
process-start identity, monotonic start and end, wall-clock context, and an
exit state. Dependency rows SHALL include dependency path and SHA-256 value.
Resource rows SHALL include resource key, exclusivity, owner PID, owner
process-start identity, monotonic interval, acquisition, and release state.
Output rows SHALL include output path, output SHA-256 value, and write time.

The validator SHALL reject negative intervals, phase inversions, missing
release, PID reuse, dependency hash changes, overlapping exclusive resource
claims, output writes before execution, copied receipts from another task,
borrowed global activity, duplicated attempt ids, forged clocks,
cross-task output paths, and parent-child PID confusion. Independent CPU
resource overlap MAY pass only when the shared CPU resource is non-exclusive.
Exclusive GPU ownership SHALL pass only when intervals are serialized.

Exp6481 SHALL emit one row per phase, dependency, resource interval, output,
concurrency decision, process identity, and attack. Readiness SHALL be
recomputed from rows. The API SHALL not alter conductor dispatch, locks,
scheduling, or active roadmap semantics. `scripts/research_conductor.py` and
`research-roadmap.yaml` SHALL remain byte-identical.

The terminal artifact SHALL include `status`, `receipt_schema_and_hash`,
`phase_rows`, `dependency_hash_rows`, `resource_ownership_rows`,
`concurrency_decision_rows`, `process_identity_rows`, `attack_matrix`,
`phase_concurrency_receipt_ready_score`, `per_unit_rows`,
`aggregate_row_recomputation`, `protected_files_unchanged`,
`gate_check_summary`, `preconditions_checked`, `inference_substrate`,
`verifier_is_oracle`, `field_principles`, `field_provenance`, `random_seed`,
`duration_s`, `tests_run`, `reproducibility_checksum`, and `honest_verdict`.
`inference_substrate` SHALL equal
`deterministic_runtime_receipt_validation_no_llm`. `verifier_is_oracle` SHALL
be true only for deterministic schema, hash, process, and monotonic interval
validation.

Field principles SHALL use this map:

| Field | Principle |
|---|---|
| `status` | A terminal status distinguishes a complete contract build from partial instrumentation. |
| `receipt_schema_and_hash` | A versioned schema prevents later experiments from silently changing receipt meaning. |
| `phase_rows` | Monotonic phase rows separate queue, load, execution, exact verification, and write time. |
| `dependency_hash_rows` | Dependency hashes prove the task consumed the stated upstream bytes. |
| `resource_ownership_rows` | Resource intervals attribute CPUs, GPUs, files, and locks to one task attempt. |
| `concurrency_decision_rows` | Explicit decisions distinguish safe overlap from conflicting exclusive ownership. |
| `process_identity_rows` | PID and process-start identity prevent stale or borrowed activity from being credited. |
| `attack_matrix` | Constructive attacks test the known global-activity and time-order attribution failures. |
| `phase_concurrency_receipt_ready_score` | A conjunctive score blocks reuse until all ownership and ordering attacks fail closed. |
| `per_unit_rows` | Phase, dependency, resource, and attack rows make the contract independently auditable. |
| `aggregate_row_recomputation` | Row-derived readiness catches summaries that omit a failing receipt. |
| `protected_files_unchanged` | The receipt task must not alter conductor or active roadmap behavior. |
| `gate_check_summary` | A blocked verdict identifies the exact contract or test check that failed. |
| `preconditions_checked` | Preconditions prove required clocks, process metadata, and fixture paths were available. |
| `inference_substrate` | Declaring deterministic_runtime_receipt_validation_no_llm prevents fixture activity from becoming a compute claim. |
| `verifier_is_oracle` | Only schema, hash, process, and monotonic interval validation is authoritative. |
| `field_principles` | A field-to-principle map carries the evidence design into later tasks. |
| `field_provenance` | Per-field code and fixture paths make each value traceable. |
| `random_seed` | A fixed seed reproduces attack and overlap scheduling. |
| `duration_s` | Wall time catches a task that emitted without exercising concurrency fixtures. |
| `tests_run` | Recorded commands prove the API and its attacks executed. |
| `reproducibility_checksum` | The checksum binds schema, fixtures, implementation, and result. |
| `honest_verdict` | The verdict states contract readiness without claiming conductor concurrency. |

### SCENARIO-INFRA-6481-MONOTONIC-PHASES: Ordered Phase Rows Fail Closed

GIVEN the eight required phase rows for one task attempt
WHEN the validator recomputes phase order from monotonic clocks
THEN every required phase is present, negative intervals are rejected, phase
inversions are rejected, and wall-clock inversions are diagnostic failures.

**Spec traces:** REQ-INFRA-6481

### SCENARIO-INFRA-6481-DEPENDENCY-BINDING: Dependency Hashes Bind Upstream Bytes

GIVEN dependency rows with paths and SHA-256 values
WHEN a dependency file changes or a row is copied from another task attempt
THEN validation rejects the receipt and records the exact failed dependency or
task binding reason.

**Spec traces:** REQ-INFRA-6481

### SCENARIO-INFRA-6481-RESOURCE-OWNERSHIP: Exclusive Resources Must Be Owned And Released

GIVEN CPU, GPU, file, and lock resource intervals
WHEN validation compares owner PID, process-start identity, acquisition, and
release rows
THEN independent CPU overlap is allowed, serialized exclusive GPU ownership is
allowed, missing release is rejected, and overlapping exclusive ownership is
rejected.

**Spec traces:** REQ-INFRA-6481

### SCENARIO-INFRA-6481-CONCURRENCY-OVERLAP: Decisions Explain Safe And Unsafe Overlap

GIVEN resource intervals from multiple task attempts
WHEN intervals overlap
THEN non-exclusive CPU overlap emits a safe decision row, serialized GPU access
emits a serialized decision row, and impossible exclusive overlap is rejected.

**Spec traces:** REQ-INFRA-6481

### SCENARIO-INFRA-6481-FAIL-CLOSED-VALIDATION: Attribution Attacks Are Rejected

GIVEN attacks for borrowed `nvidia-smi` activity, stale dependency artifacts,
duplicated attempt ids, forged clocks, cross-task output paths, parent-child PID
confusion, PID reuse, and output writes before execution
WHEN the deterministic validator evaluates mutated rows
THEN every attack fails closed and appears in `attack_matrix`.

**Spec traces:** REQ-INFRA-6481

### SCENARIO-INFRA-6481-ARTIFACT: Terminal Artifact Is Row-Recomputed And Nonmutating

GIVEN schema hashes, clock sources, process metadata, fixture paths, protected
file hashes, and command receipts
WHEN Exp6481 writes the terminal artifact
THEN every required field has a principle and provenance, the checksum matches,
readiness is `1.0`, protected conductor and roadmap files are unchanged, and no
conductor behavior is changed.

**Spec traces:** REQ-INFRA-6481

## Implementation Status (REQ-INFRA-6481)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6481 | Planned: `python/carnot/phase_concurrency_receipts.py` and `python/carnot/experiment_6481_monotonic_phase_concurrency_receipt_contract.py`; terminal artifact `results/experiment_6481_monotonic_phase_concurrency_receipt_contract.json`. | Planned: `tests/python/test_experiment_6481_monotonic_phase_concurrency_receipt_contract.py`. |

## REQ-INFRA-6483: V559 Latent-Energy SOTA Ingestion SHALL Map Sources To Bounded Experiments Without Execution

Carnot SHALL provide Exp6483 at
`python/carnot/experiment_6483_v559_latent_energy_sota_ingestion.py`. The
command
`.venv/bin/python -m carnot.experiment_6483_v559_latent_energy_sota_ingestion --date 20260821`
SHALL write
`results/experiment_6483_v559_latent_energy_sota_ingestion.json`.

Exp6483 SHALL read the current research references, study ledger, V559 roadmap
proposal, SOTA note, search helpers, exclusion manifest, and e2e test plan
before source synthesis. It SHALL record the repository state, source cutoff
time, query terms, helper commands, and network method. It SHALL confirm that
the task has no product-execution oracle.

Exp6483 SHALL use low-concurrency arXiv and Semantic Scholar helper routes and
sequential web reads for the top primary sources. It SHALL cover EBM
verification, neural constraints, Ising or probabilistic hardware,
hallucination detection, KAN, energy-guided decisions, and continual
constraint learning. It SHALL recheck EBT `2507.02092`, ARM-EBM `2512.15605`,
OpenReview, Hugging Face Papers, Extropic, GitHub, and Logical Intelligence.

Every primary source row SHALL record source identity, URL, date, query route,
claim boundary, relevance area, and citation validity. Requested secondary or
product surfaces SHALL be separate secondary rows. The workflow SHALL NOT
fabricate a citation, citation count, product capability, hardware result,
model result, ARC result, or execution result.

Exp6483 SHALL select three to five source-backed methods. Each mapping row
SHALL record the exact source, current Carnot surface, expected falsifiable
test, failure boundary, retired-scope risk, and candidate next task. It SHALL
emit retired-scope collision rows against `ops/exclusion_manifest.yaml`.
It SHALL write the research note
`docs/research-notes/v559-latent-energy-sota-ingestion.md` and update the study
ledger without changing `research-references.md` unless a genuine source delta
is found.

The terminal artifact SHALL include `status`, `source_cutoff_utc`,
`query_receipts`, `primary_source_rows`, `secondary_source_rows`,
`method_mapping_rows`, `retired_scope_collision_rows`, `research_note_path`,
`study_ledger_updates`, `no_execution_claim`, `per_unit_rows`,
`aggregate_row_recomputation`, `gate_check_summary`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`field_principles`, `field_provenance`, `random_seed`, `duration_s`,
`tests_run`, `reproducibility_checksum`, and `honest_verdict`.
`inference_substrate` SHALL equal
`primary_source_ingestion_no_product_execution`. `verifier_is_oracle` SHALL be
false. `no_execution_claim` SHALL be true.

Field principles SHALL use this map:

| Field | Principle |
|---|---|
| `status` | A terminal status distinguishes completed source ingestion from a blocked source pass. |
| `source_cutoff_utc` | The cutoff bounds which sources could affect the mapping. |
| `query_receipts` | Query terms and helper commands make the source route reproducible. |
| `primary_source_rows` | One row per checked primary record prevents summary-only citation claims. |
| `secondary_source_rows` | Secondary and product rows stay separate from primary paper evidence. |
| `method_mapping_rows` | Mapping rows connect each source to a current Carnot surface and falsifiable test. |
| `retired_scope_collision_rows` | Collision rows prevent a citation from reopening excluded work by implication. |
| `research_note_path` | The note path points to the human-readable method synthesis. |
| `study_ledger_updates` | Ledger rows record what was ingested or deferred. |
| `no_execution_claim` | A true value prevents paper review from becoming a model, product, hardware, or ARC claim. |
| `per_unit_rows` | Source and mapping rows make the synthesis checkable. |
| `aggregate_row_recomputation` | Row-derived counts catch missing or inflated summaries. |
| `gate_check_summary` | Blocked verdicts name the missing source or mapping gate. |
| `preconditions_checked` | Preconditions prove required files and helpers were read before synthesis. |
| `inference_substrate` | Declaring primary_source_ingestion_no_product_execution states the evidence substrate. |
| `verifier_is_oracle` | Paper claims are not oracles for Carnot behavior. |
| `field_principles` | A field-to-principle map preserves the reason for every evidence field. |
| `field_provenance` | URLs, file paths, and reducer sources make each value traceable. |
| `random_seed` | A fixed seed reproduces ordering decisions. |
| `duration_s` | Wall time shows the ingestion pass was measured. |
| `tests_run` | Command receipts distinguish executed validation from intended checks. |
| `reproducibility_checksum` | The checksum binds source identities and mapping rows. |
| `honest_verdict` | The verdict states whether cited actionable mapping completed without execution. |

### SCENARIO-INFRA-6483-SOURCE-IDENTITY: Source Rows Preserve Identity

GIVEN Exp6483 checks primary paper records
WHEN it emits `primary_source_rows`
THEN at least five rows have resolvable URLs, dates, query routes, citation
validity, relevance areas, and claim boundaries.

**Spec traces:** REQ-INFRA-6483

### SCENARIO-INFRA-6483-CITATION-VALIDITY: Citation Checks Do Not Invent Counts

GIVEN Exp6483 rechecks EBT, ARM-EBM, OpenReview, Hugging Face Papers,
Extropic, GitHub, and Logical Intelligence
WHEN an endpoint omits a count or exposes product-only evidence
THEN the row records the observed state without inventing citation counts or
execution capability.

**Spec traces:** REQ-INFRA-6483

### SCENARIO-INFRA-6483-METHOD-MAPPING: Methods Map To Falsifiable Current Tests

GIVEN three to five selected SOTA methods
WHEN Exp6483 writes `method_mapping_rows`
THEN each row names the source URL, current Carnot surface, expected test,
failure boundary, retired-scope risk, and candidate next task.

**Spec traces:** REQ-INFRA-6483

### SCENARIO-INFRA-6483-NO-EXECUTION: Source Ingestion Has No Execution Oracle

GIVEN Exp6483 reads papers and product pages only
WHEN it validates the artifact
THEN `no_execution_claim` is true, `inference_substrate` is
`primary_source_ingestion_no_product_execution`,
`verifier_is_oracle` is false, and no product, model, hardware, or ARC result
is claimed.

**Spec traces:** REQ-INFRA-6483

### SCENARIO-INFRA-6483-ROWS: Aggregates Recompute From Rows

GIVEN source rows, mapping rows, secondary rows, and retired-scope rows
WHEN Exp6483 computes `aggregate_row_recomputation`
THEN counts derive only from rows, acceptance gates require at least five
primary rows and at least three mappings, and blocked verdicts include
`gate_check_summary`.

**Spec traces:** REQ-INFRA-6483

## Implementation Status (REQ-INFRA-6483)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6483 | Planned: `python/carnot/experiment_6483_v559_latent_energy_sota_ingestion.py`; terminal artifact `results/experiment_6483_v559_latent_energy_sota_ingestion.json`. | Planned: `tests/python/test_experiment_6483_v559_latent_energy_sota_ingestion.py`. |

## REQ-INFRA-6484: Non-Generation Representation Receipts SHALL Bind Fixed Candidates, Raw Vectors, Families, And Transforms

Carnot SHALL provide Exp6484 at
`python/carnot/experiment_6484_non_generation_representation_receipt_contract.py`.
The command
`.venv/bin/python -m carnot.experiment_6484_non_generation_representation_receipt_contract --date 20260821`
SHALL write
`results/experiment_6484_non_generation_representation_receipt_contract.json`.

The contract SHALL validate the preserved paired-representation surface without
loading a large model. It SHALL confirm that the current GGUF generated-answer
transport lane and finite-ID retry patterns are retired by
`ops/exclusion_manifest.yaml`. It SHALL confirm that the paired embedding and
final-token or final-layer embedding surfaces remain preserved.

The receipt schema SHALL be versioned and hash-bound. Each representation
receipt SHALL include prompt hash, candidate hash, pre-model commitment time,
model ID, model hash, family, native dimension, vector hash, write count,
phase intervals, and a no-generation witness. Candidate bytes SHALL be frozen
before model access. Raw vectors SHALL have exactly one durable record before
any transform. Derived features SHALL bind to raw vector hashes and one frozen
transform manifest. Families SHALL keep native dimensions separate and SHALL
not pool or concatenate raw dimensions across families.

Exp6484 SHALL build deterministic fixture vectors for at least three model
families with distinct native dimensions. It SHALL emit one row per fixture,
family, phase, and attack. It SHALL include attacks for generation API calls,
post-load candidate edits, duplicate vector writes, label reads before
persistence, pooled family vectors, dimension-identity shortcuts, norm-only
shortcuts, length-only shortcuts, pair permutations, and claim flips. Each
attack SHALL fail closed.

Readiness SHALL be recomputed from rows. Exp6484 SHALL set
`non_generation_surface_contract_ready_score=1.0` only when every invariant
passes and every attack fails closed. It SHALL set
`inference_substrate="deterministic_representation_contract_no_llm"`. It SHALL
set `verifier_is_oracle=true` only for deterministic receipt validation. It
SHALL NOT modify `scripts/research_conductor.py` or `research-roadmap.yaml`.

The terminal artifact SHALL include `status`, `receipt_schema`,
`fixture_manifest`, `candidate_commitment_rows`,
`raw_vector_persistence_rows`, `no_generation_receipts`,
`family_separation_receipts`, `transform_manifest`, `attack_matrix`,
`non_generation_surface_contract_ready_score`, `per_unit_rows`,
`aggregate_row_recomputation`, `protected_files_unchanged`,
`gate_check_summary`, `preconditions_checked`, `inference_substrate`,
`verifier_is_oracle`, `field_principles`, `field_provenance`, `random_seed`,
`duration_s`, `tests_run`, `reproducibility_checksum`, and
`honest_verdict`.

Field principles SHALL use this map:

| Field | Principle |
|---|---|
| `status` | Terminal contract state. |
| `receipt_schema` | Versioned non-generation representation receipt schema. |
| `fixture_manifest` | Deterministic multi-dimension fixtures. |
| `candidate_commitment_rows` | Proof that candidate bytes predate model access. |
| `raw_vector_persistence_rows` | One durable write per raw vector. |
| `no_generation_receipts` | Proof that no generation API was called. |
| `family_separation_receipts` | Proof that native dimensions were not pooled. |
| `transform_manifest` | Frozen transforms bound to raw hashes. |
| `attack_matrix` | All shortcut and lifecycle attacks. |
| `non_generation_surface_contract_ready_score` | Same-roadmap downstream gate field. |
| `per_unit_rows` | Fixture, phase, and attack rows. |
| `aggregate_row_recomputation` | Ready score recomputed from rows. |
| `protected_files_unchanged` | Active roadmap and conductor unchanged. |
| `gate_check_summary` | Required for any blocked_* verdict. |
| `preconditions_checked` | Retirement and fixture prechecks. |
| `inference_substrate` | `deterministic_representation_contract_no_llm` states that no LLM was loaded. |
| `verifier_is_oracle` | True only for deterministic receipt validation. |
| `field_principles` | Reason for each field. |
| `field_provenance` | Source paths, hashes, and reducers. |
| `random_seed` | Fixed attack ordering seed. |
| `duration_s` | Measured wall time. |
| `tests_run` | Executed commands and exit codes. |
| `reproducibility_checksum` | Hash over schema, fixtures, and attacks. |
| `honest_verdict` | States contract readiness without a model-quality claim. |

### SCENARIO-INFRA-6484-COMMITMENT: Candidates Predate Model Access

GIVEN deterministic prompt and candidate fixtures
WHEN the contract validates candidate commitment rows
THEN prompt hashes, candidate hashes, and pre-model commitment times match the
fixture manifest and predate model access.

**Spec traces:** REQ-INFRA-6484

### SCENARIO-INFRA-6484-PERSISTENCE: Raw Vectors Are Written Once Before Transforms

GIVEN deterministic vectors for three native dimensions
WHEN the contract validates raw vector rows and transform rows
THEN each raw vector has one durable write, label access follows persistence,
and each derived feature binds to the raw hash and frozen transform manifest.

**Spec traces:** REQ-INFRA-6484

### SCENARIO-INFRA-6484-NO-GENERATION: Generation APIs Are Excluded

GIVEN no-generation witness rows
WHEN a generation, completion, chat, or decode method appears
THEN validation rejects the row set and records the generation attack as
failed closed.

**Spec traces:** REQ-INFRA-6484

### SCENARIO-INFRA-6484-FAMILY-SEPARATION: Native Dimensions Stay Separate

GIVEN three family receipts with distinct native dimensions
WHEN validation inspects raw and derived rows
THEN no raw vector is pooled or concatenated across families, and dimension,
norm, or length shortcuts are rejected.

**Spec traces:** REQ-INFRA-6484

### SCENARIO-INFRA-6484-ATTACKS: Shortcut And Lifecycle Attacks Fail Closed

GIVEN attacks for generation API call, post-load candidate edit, duplicate
vector write, label read before persistence, pooled family vectors, dimension
identity, norm-only signal, length-only signal, pair permutation, and claim
flip
WHEN the deterministic validator evaluates the mutated rows
THEN every attack fails closed and appears in `attack_matrix`.

**Spec traces:** REQ-INFRA-6484

### SCENARIO-INFRA-6484-ARTIFACT: Terminal Artifact Is Row-Recomputed And Nonmutating

GIVEN the schema, fixtures, attack rows, precondition checks, protected file
hashes, and command receipts
WHEN Exp6484 writes the terminal artifact
THEN every required field has a principle and provenance, the checksum matches,
readiness is `1.0`, protected files are unchanged, and the verdict makes no
model-quality claim.

**Spec traces:** REQ-INFRA-6484

## Implementation Status (REQ-INFRA-6484)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6484 | Planned: `python/carnot/experiment_6484_non_generation_representation_receipt_contract.py`; terminal artifact `results/experiment_6484_non_generation_representation_receipt_contract.json`. | Planned: `tests/python/test_experiment_6484_non_generation_representation_receipt_contract.py`. |

## REQ-INFRA-6485: Online Cache Transition E-Process Contracts SHALL Bind Chronological Events, Durable Actions, And Adaptive Evidence

Carnot SHALL provide Exp6485 at
`python/carnot/experiment_6485_online_cache_transition_eprocess_contract.py`.
The command
`.venv/bin/python -m carnot.experiment_6485_online_cache_transition_eprocess_contract --date 20260821`
SHALL write
`results/experiment_6485_online_cache_transition_eprocess_contract.json`.

The contract SHALL be deterministic and SHALL NOT claim a learning benefit.
It SHALL record repository state and Exp6479 readiness before fixture replay.
It SHALL confirm that it does not reuse the retired Exp5895 exact-slot
requalification. It SHALL use the default-off Exp6479 factor-cache adapter as
upstream readiness evidence only.

The event schema SHALL be versioned and hash-bound. It SHALL define immutable
event IDs and monotonic receipts for `observe`, `verify`, `propose`,
`quarantine`, `admit`, `promote`, `evict`, `tombstone`, `rollback`, `restart`,
and `no_write`. Each event row SHALL include chronology index, event type,
parent state hash, event payload hash, authority, fixture label, and row hash.
Event IDs SHALL be derived from the immutable event content. Event IDs SHALL
not include mutable summaries or later outcomes.

The action receipt schema SHALL distinguish actual durable actions from stated
intent. Every event SHALL have either one durable action row or one explicit
no-action row. Durable writes SHALL require exact admission authority and a
matching event ID. Action rows SHALL include monotonic receipt index, action
type, event ID, action hash, pre-state hash, post-state hash, admission hash,
durability flag, and no-write reason when no durable action occurs.

The evidence process SHALL define one fixed null before adaptive events. It
SHALL define an update rule, promotion boundary, stopping boundary, and a
geometric mixture over the allowed factor hypotheses. Every adaptive peek SHALL
charge one evidence row. Fixed-horizon comparisons SHALL stay separate from
adaptive decisions. Held events SHALL NOT tune the null, mixture weights,
promotion boundary, or stopping boundary.

Exp6485 SHALL emit one row per event, durable action or no-action, evidence
update, state transition, restart, and attack. It SHALL include attacks for
duplicate events, backdated writes, stated write without action, action without
exact admission, threshold editing, repeated peeking, missing null, rollback
omission, tombstone resurrection, and restart drift. Each attack SHALL fail
closed.

Readiness SHALL be recomputed from rows. Exp6485 SHALL set
`online_transition_contract_ready_score=1.0` only when all event, action,
evidence, lifecycle, restart, protected-file, and attack invariants pass.
It SHALL set `inference_substrate="deterministic_transition_contract_no_llm"`.
It SHALL set `verifier_is_oracle=true` only for exact fixture and receipt
validation. It SHALL NOT modify `scripts/research_conductor.py` or
`research-roadmap.yaml`.

The terminal artifact SHALL include `status`, `event_schema`,
`action_receipt_schema`, `evidence_process_spec`, `frozen_null_receipt`,
`event_rows`, `action_rows`, `evidence_process_rows`, `lifecycle_rows`,
`attack_matrix`, `online_transition_contract_ready_score`, `per_unit_rows`,
`aggregate_row_recomputation`, `protected_files_unchanged`,
`gate_check_summary`, `preconditions_checked`, `inference_substrate`,
`verifier_is_oracle`, `field_principles`, `field_provenance`, `random_seed`,
`duration_s`, `tests_run`, `reproducibility_checksum`, and `honest_verdict`.

Field principles SHALL use this map:

| Field | Principle |
|---|---|
| `status` | Terminal transition-contract state. |
| `event_schema` | Immutable chronological event schema. |
| `action_receipt_schema` | Actual durable action schema. |
| `evidence_process_spec` | Null, update, mixture, promotion, and stopping rules. |
| `frozen_null_receipt` | Proof that the null predates adaptive events. |
| `event_rows` | One row per fixture event. |
| `action_rows` | One row per durable action or explicit no-action. |
| `evidence_process_rows` | One row per sequential update and peek charge. |
| `lifecycle_rows` | Admission, eviction, tombstone, rollback, and restart states. |
| `attack_matrix` | Duplicate, peeking, authority, and resurrection attacks. |
| `online_transition_contract_ready_score` | Same-roadmap downstream gate field. |
| `per_unit_rows` | Event, action, update, and attack rows. |
| `aggregate_row_recomputation` | Ready score recomputed from rows. |
| `protected_files_unchanged` | Active roadmap and conductor unchanged. |
| `gate_check_summary` | Required for any blocked_* verdict. |
| `preconditions_checked` | Adapter and fixture prechecks. |
| `inference_substrate` | deterministic_transition_contract_no_llm. |
| `verifier_is_oracle` | True for exact fixture and receipt validation only. |
| `field_principles` | Reason for every field. |
| `field_provenance` | Source paths, hashes, and reducers. |
| `random_seed` | Fixed fixture and attack seed. |
| `duration_s` | Measured wall time. |
| `tests_run` | Executed checks and exit codes. |
| `reproducibility_checksum` | Hash over schemas, null, rows, and attacks. |
| `honest_verdict` | States contract readiness without claiming a learning gain. |

### SCENARIO-INFRA-6485-EVENTS: Immutable Chronological Events Fail Closed

GIVEN fixture events for observe, verify, propose, quarantine, admit, promote,
evict, tombstone, rollback, restart, and no-write
WHEN the validator recomputes event IDs and monotonic order
THEN IDs match immutable content, chronology is increasing, and duplicate or
backdated events fail closed.

**Spec traces:** REQ-INFRA-6485

### SCENARIO-INFRA-6485-ACTIONS: Durable Actions Require Exact Admission

GIVEN an event stream with admitted and rejected writes
WHEN action receipts are validated
THEN every event has exactly one action or no-action row, admitted writes bind
to exact admission, and stated-write-without-action or action-without-admission
attacks fail closed.

**Spec traces:** REQ-INFRA-6485

### SCENARIO-INFRA-6485-EPROCESS: Adaptive Peeks Are Charged

GIVEN a frozen null, a geometric factor mixture, and sequential observations
WHEN fixed-horizon and adaptive decisions are evaluated
THEN each adaptive peek emits a charge row, fixed-horizon comparisons do not
change thresholds, and threshold editing, repeated peeking, or missing-null
attacks fail closed.

**Spec traces:** REQ-INFRA-6485

### SCENARIO-INFRA-6485-LIFECYCLE: Tombstones And Rollbacks Survive Restart

GIVEN admission, eviction, tombstone, rollback, and restart lifecycle rows
WHEN the state is replayed from rows
THEN tombstoned events cannot resurrect, rollback omission is detected, and
restart state hashes match the replayed state.

**Spec traces:** REQ-INFRA-6485

### SCENARIO-INFRA-6485-ATTACKS: Event, Authority, Peeking, And Resurrection Attacks Fail Closed

GIVEN attacks for duplicate events, backdated writes, stated write without
action, action without exact admission, threshold editing, repeated peeking,
missing null, rollback omission, tombstone resurrection, and restart drift
WHEN the deterministic validator evaluates mutated rows
THEN every attack fails closed and appears in `attack_matrix`.

**Spec traces:** REQ-INFRA-6485

### SCENARIO-INFRA-6485-ARTIFACT: Terminal Artifact Is Row-Recomputed And Nonmutating

GIVEN schemas, frozen null, fixture rows, attack rows, protected-file hashes,
precondition checks, and command receipts
WHEN Exp6485 writes the terminal artifact
THEN every required field has a principle and provenance, the checksum matches,
readiness is `1.0`, protected files are unchanged, and no learning gain is
claimed.

**Spec traces:** REQ-INFRA-6485

## Implementation Status (REQ-INFRA-6485)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6485 | Planned: `python/carnot/experiment_6485_online_cache_transition_eprocess_contract.py`; terminal artifact `results/experiment_6485_online_cache_transition_eprocess_contract.json`. | Planned: `tests/python/test_experiment_6485_online_cache_transition_eprocess_contract.py`. |

## REQ-INFRA-6351: V547 Source Freeze SHALL Validate Planner Receipts And Keep Scope Closed

Carnot SHALL build Exp6351 as a deterministic V547 post-marker source sweep
and scope freeze. The sweep SHALL hash the exact
`<!-- V547-PLANNER-REFRESH-20260812-END -->` marker in
`research-references.md`. It SHALL record the marker line. It SHALL use the
marker commit time as the exclusive lower bound for post-marker novelty.

Exp6351 SHALL validate the six V547 planner-promoted source families with
direct primary URLs, first-publication dates, access times, claim boundaries,
and Carnot executable consequences. The source families are active reward
machine inference, zero-shot goal recognition, read-only memory, solver
hardness controls, distributional EBMs with abstention, and verification
horizon controls. The receipts SHALL remain separate from any post-marker
acceptance count.

Exp6351 SHALL repeat arXiv, OpenReview, Hugging Face Papers, Semantic Scholar
EBT and ARM-EBM routes, Extropic, Logical Intelligence, and GitHub searches
only for the post-marker window. Empty endpoints, secondary mirrors, rate
limits, product pages, and inaccessible pages SHALL be recorded as receipts.
They SHALL not become promoted findings.

Exp6351 SHALL deduplicate by work identity, repository identity, mechanism,
content hash, URL, title, and already-retired Carnot scope. It SHALL accept
only stable, non-duplicate, reproducible, primary or first-party evidence that
is strictly later than the V547 marker and that changes a local executable
contract. A zero-source delta SHALL be terminal. In that case `accepted_count`
SHALL be the bare integer `0`, `llm_call_count` SHALL be the bare integer `0`,
and `honest_verdict` SHALL start with `complete_null:`.

Exp6351 SHALL freeze exactly two V547 scientific lanes: prospective certified
factor learning and falsifiable live ARC goal discovery. It SHALL also freeze
exact-oracle boundaries, required local GGUF models, active ARC provenance
rules, fail-fast gates, the closed parser/JIT lane, and the no-hardware rule.
The task SHALL not execute GateMate, KV260, PolarFire, NPU, TSU, Kona, or
board commands. The task SHALL not modify `scripts/research_conductor.py`.

The Exp6351 artifact SHALL be written atomically to
`results/experiment_6351_v547_post_marker_source_scope_freeze.json` with
`inference_substrate=web_and_bibliographic_search_only`,
`verifier_is_oracle=false`, and `llm_call_count=0`.

The Exp6351 artifact SHALL include these required fields: `status`,
`v547_marker_text_line_and_hash`, `search_window_start_utc`,
`search_completed_utc`, `source_queries_by_channel`, `source_receipts`,
`promoted_findings`, `accepted_count`, `duplicate_findings`,
`watch_only_findings`, `inaccessible_sources`,
`excluded_findings_and_reasons`, `active_reward_machine_receipt`,
`zero_shot_goal_recognition_receipt`, `memoir_receipt`,
`solver_hardness_control_receipt`, `distributional_ebm_receipt`,
`verification_horizon_receipt`,
`semantic_scholar_ebt_and_arm_ebm_receipts`,
`openreview_and_huggingface_status`, `github_status`, `extropic_status`,
`logical_intelligence_status`, `frozen_live_factor_learning_contract`,
`frozen_arc_goal_contract`, `frozen_model_policy`,
`frozen_closed_parser_jit_contract`, `frozen_hardware_nonuse_contract`,
`roadmap_scope_delta`, `protected_files_unchanged`,
`preconditions_checked`, `inference_substrate`, `verifier_is_oracle`,
`llm_call_count`, `field_provenance`, `field_principles`,
`test_commands`, `test_exit_codes`, `duration_s`, `random_seeds`,
`reproducibility_checksum`, and `honest_verdict`.

### SCENARIO-INFRA-6351-1: Marker Bound Is Exclusive

GIVEN the sealed V547 planner refresh marker in `research-references.md`
WHEN Exp6351 classifies a candidate at or before the marker commit time
THEN the candidate is rejected, and a bare same-day date is rejected unless a
strictly later timestamp is present.

### SCENARIO-INFRA-6351-2: Promoted Source Dates Are Direct Receipts

GIVEN the V547 marker promoted active reward-machine inference, zero-shot goal
recognition, Memoir, solver-hardness control, distributional EBM, and
verification-horizon work
WHEN Exp6351 builds its artifact
THEN each promoted source has a direct primary URL, first-publication date,
access time, bounded source claim, and local executable consequence.

### SCENARIO-INFRA-6351-3: Dedupe And Source Dispositions Fail Closed

GIVEN a candidate repeats an older work identity, repository identity,
mechanism, content hash, URL, title, or retired Carnot scope
WHEN Exp6351 partitions the sweep
THEN the row is not accepted, and its duplicate, watch-only, inaccessible, or
excluded disposition records the exact reason.

### SCENARIO-INFRA-6351-4: Frozen V547 Contracts Preserve Boundaries

GIVEN V547 has two scientific lanes plus exact-oracle, local-GGUF,
active-ARC, fail-fast, closed parser/JIT, and no-hardware boundaries
WHEN Exp6351 serializes frozen contracts
THEN the artifact admits only prospective certified factor learning and
falsifiable live ARC goal discovery. It SHALL reject solver-only difficulty
proxies, learned approval, hidden-source ARC evidence, parser/JIT reopening,
GGUF weight updates, and hardware execution.

### SCENARIO-INFRA-6351-5: Output Is Principle Annotated And Non-Mutating

GIVEN source receipts, protected hashes, field principles, and command
receipts
WHEN Exp6351 validates the report before writing
THEN every required field is present, every field has provenance and a
principle, `accepted_count` and `llm_call_count` are bare integers, the
checksum matches the normalized payload, protected hashes remain unchanged,
and the honest verdict has a terminal prefix.

## Implementation Status (REQ-INFRA-6351)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6351 | Planned: `python/carnot/experiment_6351_v547_post_marker_source_scope_freeze.py`; terminal artifact `results/experiment_6351_v547_post_marker_source_scope_freeze.json`. | Planned: `tests/python/test_experiment_6351_v547_post_marker_source_scope_freeze.py`. |

## Implementation Status (REQ-INFRA-6322)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6322 | Implemented: `python/carnot/experiment_6322_v544_adversarial_capstone.py`; terminal artifact `results/experiment_6322_v544_adversarial_capstone.json`. | Implemented: `tests/python/test_experiment_6322_v544_adversarial_capstone.py`. |

## REQ-INFRA-6210: V537 Capstone SHALL Reconcile Exact Declared Deliverables With Fail-Closed Terminality

Carnot SHALL build Exp6210 as the branch-independent capstone for milestone
`2026.08.537`. The capstone SHALL load the active V537 roadmap, derive the
exact task-ID to declared-deliverable manifest for Exp6197 through Exp6209,
and SHALL NOT substitute sidecars, aliases, or glob matches when an exact path
is missing.

Exp6210 SHALL classify every exact deliverable path with the shared Exp6197
terminal-artifact classifier; this Exp6197 terminal-artifact classifier is the
only terminality promotion mechanism. Conductor receipts SHALL be recorded but SHALL
NOT promote missing, malformed, running, running_bootstrap, bootstrap-only,
contradictory, unknown, or partial artifacts. Structured roadmap gates SHALL be
recomputed from upstream artifact fields and compared with conductor receipts
without launching or fabricating skipped downstream artifacts.

Exp6210 SHALL run adversarial verification for every present result artifact
and record the receipts. A CRITICAL adversarial flag, nonterminal artifact
class, missing required field, unclassified nonzero command, or failed
promotion gate SHALL exclude the affected branch from headline eligibility.
Headline eligibility SHALL be reported separately for Phase D, continuous
self-learning, sampler integration, ARC generalization, and hardware
continuity. Negative classes, quarantines, structured skips, source-delta
nulls, stale architecture warnings, and repeated prior-failure retirement
actions SHALL be preserved without rewriting historical result artifacts.

The Exp6210 artifact SHALL be written atomically to
`results/experiment_6210_v537_adversarial_capstone.json` and SHALL include
these required fields: `status`, `milestone_and_declared_task_graph_hash`,
`exact_deliverable_manifest`, `conductor_receipts`,
`terminal_classifier_path_hash_and_results`, `task_terminal_classes`,
`missing_nonterminal_blocked_skipped_null_retired_flagged_counts`,
`structured_gate_recomputation`, `adversarial_verify_receipts_by_artifact`,
`protected_historical_artifact_mutation_count`,
`phase_d_headline_eligibility_and_reason`,
`continuous_self_learning_headline_eligibility_and_reason`,
`sampler_integration_headline_eligibility_and_reason`,
`arc_generalization_headline_eligibility_and_reason`,
`hardware_continuity_state`, `source_delta_state`,
`prior_failure_retirement_actions`,
`spec_traceability_status_changelog_reconciliation_receipts`,
`architecture_freshness_warning`, `forbidden_claim_counts`,
`inference_substrate`, `verifier_is_oracle`, `field_provenance`,
`field_principles`, `test_commands`, `test_exit_codes`, `duration_s`,
`reproducibility_checksum`, and `honest_verdict`. The mutation count and every
forbidden claim count SHALL be bare numeric zero.

### SCENARIO-INFRA-6210-1: Exact manifest refuses substitutes
GIVEN the active V537 roadmap declares Exp6197 through Exp6209 deliverable
paths
WHEN Exp6210 builds its manifest
THEN every task row records the exact declared path, hash and existence state,
and missing exact paths remain missing even when same-number sidecars exist.

### SCENARIO-INFRA-6210-2: Exp6197 classifier outranks conductor receipts
GIVEN an upstream artifact is missing, malformed, running_bootstrap,
bootstrap-only, contradictory, unknown, or partial
WHEN the conductor receipt says `OK`, `COMPLETE`, or another terminal status
THEN Exp6210 records the receipt but preserves the nonterminal class and blocks
headline eligibility for any dependent branch.

### SCENARIO-INFRA-6210-3: Structured gates are recomputed from immutable fields
GIVEN a roadmap task declares `gated_on` upstream field/operator/value entries
WHEN Exp6210 evaluates the gate
THEN the gate row records upstream task, field, operator, expected value,
actual value, pass/fail state, conductor receipt comparison, and structured
skip preservation without writing a downstream artifact.

### SCENARIO-INFRA-6210-4: Adversarial flags and missing fields exclude headlines
GIVEN a present upstream artifact has a CRITICAL adversarial flag, nonterminal
classification, missing required field, unclassified nonzero command, or failed
promotion gate
WHEN Exp6210 computes branch headline eligibility
THEN the branch eligibility row is false and its reason cites the exact task,
artifact path, and exclusion class.

### SCENARIO-INFRA-6210-5: Exp6210 output is atomic, checksummed, and non-mutating
GIVEN the manifest, classifier rows, gate recomputation, adversarial receipts,
field principles, and command receipts are computed in memory
WHEN Exp6210 validates and writes its artifact
THEN it writes one terminal-prefixed artifact with a stable checksum,
`inference_substrate=aggregation_from_upstream_artifacts`,
`verifier_is_oracle=false`, `protected_historical_artifact_mutation_count=0`,
bare zero forbidden claim counts, and no historical result byte rewritten.

## Implementation Status (REQ-INFRA-6210)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-INFRA-6210 | Pending implementation: `python/carnot/experiment_6210_v537_adversarial_capstone.py`; terminal artifact `results/experiment_6210_v537_adversarial_capstone.json`. | Pending focused tests: `tests/python/test_experiment_6210_v537_adversarial_capstone.py`. |

### REQ-OPS-DETERMINATION-RESTORE-6260: The Conductor's Determination Restore Covers Downgrade, Not Only Deletion

**Origin:** 2026-08-12 incident, root-caused 2026-08-13. Five artifacts reached the git index
with `flagged_adversarial` changed `True -> None`, lifting their quarantine.
`_restore_dropped_determinations` already existed and already ran before every conductor
`git add -A`, and restored nothing, because its test was `k not in cur` -- False when the key
is PRESENT and merely nulled. The guard ran and was SILENT NON-FIRING on the exact case it
was written for. Caught only because `determination-preservation-lint` refused a separate
human commit; that lint never runs on conductor commits, which use `--no-verify` deliberately
for the anti-stash-loss reasons documented in `git_commit_and_push`.

`research_conductor.determination_damage(head, cur)` SHALL return every determination-token
key that is present-and-TRUTHY in HEAD and is falsy or absent in the working tree. A
determination is meaningful only when truthy, because `False` and `None` both re-admit a
quarantined artifact to headline aggregation.

The repair SHALL remain strictly additive: it only ever copies a lost determination back, so
it cannot discard a legitimate edit or a re-run's fresh numbers.

A DELIBERATE clear SHALL be preserved. `determination_preservation_lint` documents clearing
as setting the value falsy AND adding a `*_cleared_note` recording what was re-verified; when
that note is present the key SHALL be left alone, so an auditable decision stays expressible.

#### SCENARIO-OPS-DETERMINATION-RESTORE-6260-DOWNGRADE-TO-NULL-IS-DAMAGE

Given HEAD holds `flagged_adversarial: True` and the working tree holds
`flagged_adversarial: None`, the key SHALL be reported as damage and restored.

#### SCENARIO-OPS-DETERMINATION-RESTORE-6260-DELIBERATE-CLEAR-SURVIVES

Given the working tree holds a falsy determination AND a matching `*_cleared_note`, the key
SHALL NOT be restored.

## Implementation Status (REQ-OPS-DETERMINATION-RESTORE-6260)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-OPS-DETERMINATION-RESTORE-6260 | `scripts/research_conductor.py:determination_damage` (pure decision rule) consumed by `_restore_dropped_determinations`. | `tests/python/test_conductor_determination_restore_downgrade_20260813.py` (8/8: both damage shapes, deliberate-clear carve-out, falsy-in-HEAD, non-determination keys, fresh measurements alongside a lost determination). |

### REQ-OPS-VERDICT-ROW-6261: A Verdict Must Survive Contact With Its Own Rows

**Origin:** 2026-08-13 operator question, "how do we make this process self improving?" Three
artifacts in the preceding two days carried headlines their own per-row data contradicted, and
each was caught only by a human reading rows instead of the verdict string. The conductor plans
from `honest_verdict`, so a wrong verdict propagates into the retro, the planner and the
exclusion decisions rather than staying local.

`scripts/verdict_row_consistency_lint.py` SHALL check an artifact's headline against its own
per-unit rows with checks derived from real incidents: ALL_ROWS_NULL (exp6254),
DEGENERATE_CONTROL (exp6252), NO_HEADROOM_MAJORITY and WINS_NOT_EXCEEDING_LOSSES (exp6251), and
COVERAGE_SHORTFALL.

It SHALL be advisory by default. Only ALL_ROWS_NULL SHALL exit non-zero, because "every row
empty" is unambiguous; `--strict` escalates the rest. A hard block on a fuzzy match over
free-form artifacts is how a guard begins punishing one author for another's data.

An artifact with no recognised row container SHALL be reported as `skipped`, never as clean, and
the tool SHALL ALWAYS print its coverage counts — a bare "OK" while silently skipping most
inputs is the guard-is-green-while-blind state this discipline exists to prevent.

Field-name exclusions SHALL be suffix-anchored, never bare substring.

#### SCENARIO-OPS-VERDICT-ROW-6261-DEGENERATE-CONTROL

Given a control arm whose values equal the baseline arm's on at least 80% of rows — including
when both are NESTED under a per-arm dict — the artifact SHALL be flagged, because a control
that cannot differ cannot fail and any gate requiring it to be beaten is vacuous.

#### SCENARIO-OPS-VERDICT-ROW-6261-NO-ROWS-IS-NOT-CLEAN

Given an artifact with no per-unit row container, the result SHALL be `skipped` and the
coverage line SHALL report it, so "could not check" is distinguishable from "checked and clean".

## Implementation Status (REQ-OPS-VERDICT-ROW-6261)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-OPS-VERDICT-ROW-6261 | `scripts/verdict_row_consistency_lint.py`. | `tests/python/test_verdict_row_consistency_lint.py` (11/11), plus verification against the original exp6251/exp6252 artifacts and a synthetic exp6254 shape. **Coverage caveat, measured:** over the 60 most recent artifacts it checks 3 and skips 57, because most carry no per-unit rows. |

### REQ-OPS-MILESTONE-LEDGER-6262: The Substantive-Work Share Is Computed, Not Assumed

**Origin:** 2026-08-13 operator directive "let's make that more visible", following the
observation that seven milestones produced 193 commits and 93 artifacts with zero headline
movement. `ops/north-star.md` §1 already defines churn; nothing computed it.

`scripts/milestone_progress_ledger.py` SHALL classify each completed milestone's task artifacts
by verdict shape into substantive / scaffolding / blocked / null, report the substantive share,
and print the current headline metrics so "did anything move" is answerable in one line.

It SHALL NOT be a gate. Gating on the share would reward relabelling verdicts, which is the
failure `verdict_row_consistency_lint.py` exists to catch.

The classifier SHALL be ungenerous: an unrecognised verdict counts as UNCLASSIFIED, never as
substantive, and ambiguous verdicts resolve toward scaffolding.

#### SCENARIO-OPS-MILESTONE-LEDGER-6262-AMBIGUOUS-COUNTS-AS-SCAFFOLDING

Given a milestone-transition verdict that reads like a result ("exact states and roadmap
contracts validated") but measures nothing, it SHALL classify as scaffolding, not substantive.

#### SCENARIO-OPS-MILESTONE-LEDGER-6262-UNKNOWN-IS-NOT-A-WIN

Given a verdict matching no known shape, it SHALL classify as unclassified rather than
substantive, so the reported share can never drift generous.

## Implementation Status (REQ-OPS-MILESTONE-LEDGER-6262)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-OPS-MILESTONE-LEDGER-6262 | `scripts/milestone_progress_ledger.py`. | `tests/python/test_milestone_progress_ledger.py` (7/7). First measurement, .540-.549: 34 blocked, 31 scaffolding, 17 null, **16 substantive of 108 (15%)**, headline unchanged at 183 levels / 25 games. |

### REQ-OPS-RECURRING-BLOCKER-6263: A Blocker That Recurs Is A Task Nobody Is Doing

**Origin:** 2026-08-13 operator question, "how do we make the unattended operation more self
improving?" Unattended operation fails differently: a failure nobody reads never gets fixed, so
it recurs indefinitely. Measured over 14 milestones: 58 blocked tasks, **31 carrying the
identical verdict `blocked_gate_check_failed`**, nothing escalating. Blocked is the single
largest category of all work (31% of 108 tasks, REQ-OPS-MILESTONE-LEDGER-6262). The same shape
appeared in the publish path the same day: a stale marker aborted every dataset publish for six
days while blaming the wrong cause.

`scripts/recurring_blocker_ledger.py` SHALL group blocked verdicts across completed milestones
by NORMALISED message — stripping experiment ids, milestone versions and embedded counts, so
per-task identifiers cannot hide a recurrence — and report any blocker at or above a threshold.

`--emit-known-issue` SHALL APPEND one dated MANDATORY-NEXT-MILESTONE entry to
`ops/known-issues.md`, reusing the Overdue-Priority Forcing Function rather than adding a new
enforcement path. It SHALL append, never rewrite (never-prune), and SHALL write a single entry
covering all over-threshold blockers so a mechanical process cannot flood the priorities list.

It SHALL NOT block anything. A recurring blocker may be correct; what is unacceptable is that
nothing looked.

It SHALL report DIAGNOSTIC COVERAGE — how many blocked artifacts record any reason at all.
First measurement: **37 of 58 record nothing**, and all 31 `blocked_gate_check_failed`
artifacts carry no `gate_reason` or equivalent. A blocked verdict with no diagnostic cannot be
investigated without re-running the task, which guarantees repetition.

#### SCENARIO-OPS-RECURRING-BLOCKER-6263-PER-TASK-IDS-DO-NOT-HIDE-RECURRENCE

Given two blocked verdicts differing only by experiment id or milestone version, they SHALL
normalise to the same blocker and count as one recurrence, while genuinely different blocker
messages SHALL remain separate.

## Implementation Status (REQ-OPS-RECURRING-BLOCKER-6263)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-OPS-RECURRING-BLOCKER-6263 | `scripts/recurring_blocker_ledger.py`. | `tests/python/test_recurring_blocker_ledger.py` (6/6, pinning both under- and over-collapse of the normaliser). First run escalated `blocked_gate_check_failed` x31 into `ops/known-issues.md`. |

### REQ-OPS-ARTIFACT-CONVENTION-6264: Recording Conventions Are Enforced Adversarially

**Origin:** 2026-08-13 operator directive, "it sounds like we need to enforce conventions with
our adversarial agent." Four adversarial audits already ran at milestone close and all four
reviewed CODE or DOCS; none reviewed the ARTIFACTS.

Two conventions SHALL be stated in the planner's REQUIRED ARTIFACT FIELDS guidance
(prevention): `per_unit_rows` for any comparative claim, and `blocked_reason` for any
`blocked_*` verdict.

`scripts/artifact_convention_audit.py` SHALL review recent artifacts adversarially and return
one of `CHECKABLE | AGGREGATE_ONLY | BLOCKED_WITHOUT_DIAGNOSTIC | CANNOT_DETERMINE`, asking
whether a reader who was not present could CHECK the claim from what is recorded.

It SHALL apply the audit-integrity guard: a flagged verdict whose quoted evidence does not
appear in the artifact SHALL be downgraded to `CANNOT_DETERMINE`. It SHALL never edit an
artifact and SHALL never block. It SHALL be bounded (`--recent`) so a milestone close stays
cheap.

#### SCENARIO-OPS-ARTIFACT-CONVENTION-6264-AGGREGATE-ONLY-IS-FLAGGED

Given an artifact asserting a comparative or readiness claim backed only by counts, with no
per-unit record of which units were checked or what each produced, the verdict SHALL be
`AGGREGATE_ONLY`.

#### SCENARIO-OPS-ARTIFACT-CONVENTION-6264-NO-CLAIM-IS-NOT-A-PROBLEM

Given an artifact that makes no comparative claim and is not blocked, the verdict SHALL be
`CHECKABLE`; the audit SHALL NOT invent a problem.

## Implementation Status (REQ-OPS-ARTIFACT-CONVENTION-6264)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-OPS-ARTIFACT-CONVENTION-6264 | `scripts/artifact_convention_audit.py`, wired as the fifth milestone-close audit in `research_conductor.py`; conventions added to the planner's REQUIRED ARTIFACT FIELDS guidance. | First live run flagged `experiment_3392_archive_v311_activate_v312.json` AGGREGATE_ONLY with a correct missing-check statement, and correctly passed two others as CHECKABLE. |

### REQ-OPS-RECURRING-GATE-6425: Recurring Gate Blocks Carry Replayable Diagnostics

Exp6425 SHALL freeze the `blocked_gate_check_failed` population for milestones
2026.08.536 through 2026.08.549 before classification. It SHALL replay the
structured gate records with `scripts/conductor_gates.py` in read-only mode.
It SHALL NOT infer a gate from title prose when a structured gate exists.

Each blocker occurrence SHALL bind the milestone, task id, upstream id, gate
field, operator, expected value and type, observed value and type, upstream
artifact path, upstream artifact hash, and terminal artifact path. Each
occurrence SHALL classify as one of `correct_expected_refusal`,
`missing_upstream`, `wrong_field_name`, `wrong_field_type`, `stale_artifact`,
`retired_dependency`, `diagnostic_loss`, or `other_with_evidence`.

Future conductor pre-gate blocked artifacts SHALL expose a stable diagnostic
contract in addition to the legacy `gate_check_summary`. The contract SHALL
include `blocked_reason`, failed upstream, failed field, operator, expected
value, observed value, observed type, and evidence path. Existing historical
artifacts SHALL remain unchanged.

Malformed gate inputs SHALL fail closed with a specific diagnostic. This
includes missing fields, strings in numeric gates, NaN values, stale hashes,
retired upstream ids, and contradictory status fields.

#### SCENARIO-OPS-RECURRING-GATE-6425-DIAGNOSTIC-CONTRACT

Given a failing structured gate, `write_blocked_artifact` SHALL write
`blocked_reason` and the first failed gate's upstream, field, operator,
expected value, observed value, observed type, and evidence path.

#### SCENARIO-OPS-RECURRING-GATE-6425-MUTATIONS-FAIL-CLOSED

Given missing fields, string values in numeric gates, NaN values, stale hashes,
retired upstream ids, or contradictory status fields, the Exp6425 diagnostic
matrix SHALL mark each attack as killed with a specific diagnostic and SHALL
not bypass the scientific gate.

## Implementation Status (REQ-OPS-RECURRING-GATE-6425)

| REQ | Implementation | Tests |
|---|---|---|
| REQ-OPS-RECURRING-GATE-6425 | Implemented (`python/carnot/experiment_6425_recurring_gate_block_root_cause.py`, `scripts/conductor_gates.py`, `results/experiment_6425_recurring_gate_block_root_cause.json`). | Implemented (`tests/python/test_experiment_6425_recurring_gate_block_root_cause.py`, `tests/python/test_conductor_gates.py`). |
