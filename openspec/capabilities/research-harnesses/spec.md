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
