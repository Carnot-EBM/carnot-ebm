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
