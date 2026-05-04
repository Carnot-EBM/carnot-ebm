# Research Reporting Capability Specification

**Capability:** research-reporting
**Version:** 0.1.0
**Status:** Draft
**Traces to:** NFR-03

## Overview

Defines how Carnot records experiment-result provenance and how public-facing
documentation must distinguish validated live artifacts from simulated or
otherwise unverified results. The goal is to preserve the research record
without presenting exploratory or simulated artifacts as validated real-world
performance.

## Requirements

### REQ-REPORT-001: Result Provenance Audit

The repository shall provide a cleanup workflow that scans
`results/experiment_*_results.json` and determines each artifact's inference
provenance from:

- the top-level `inference_mode`, if present
- otherwise `metadata.inference_mode`
- otherwise `statistics.inference_mode`
- otherwise known nested experiment metadata fields documented by the cleanup
  implementation

The workflow shall normalize the detected provenance into a top-level summary
without deleting any historical result data.

### REQ-REPORT-002: Result Headers

The cleanup workflow shall annotate each scanned result artifact with:

- a human-readable header stating whether the artifact is validated live,
  simulated, or missing explicit live provenance
- a machine-readable provenance summary describing the normalized mode, the
  source field used, and the resulting status

Artifacts with `live_gpu` provenance shall be marked as validated. Artifacts
with `simulated`, `simulation`, or missing provenance shall receive a warning
header rather than being removed.

### REQ-REPORT-003: README Provenance Disclosure

`README.md` shall present key benchmark claims with explicit provenance labels.
Headline result tables shall distinguish:

- validated live results
- simulated results kept for historical record
- results whose inference provenance is missing or otherwise unverified

The README shall add honest caveats where prior headline improvements were not
validated with explicit live inference provenance.

### REQ-REPORT-004: Report and Landing-Page Disclosure

`docs/technical-report.md` and `docs/index.html` shall include an explicit
"Simulation vs Reality" disclosure that:

- summarizes the audit counts for validated live, simulated, and unverified
  artifacts
- marks headline benchmark claims as live, simulated, or unverified
- revises top-level improvement claims so they no longer imply that simulated
  or unverified benchmarks were validated live results

### REQ-REPORT-005: Curated Research Scan Artifact

The repository shall provide a workflow that writes
`results/experiment_210_results.json` for the focused literature scan on
constraint extraction for instruction-tuned models. The artifact shall include:

- the query themes searched
- a ranked list of papers or benchmark assets
- why each item matters to Carnot's current constraint-extraction gap
- concrete proposed experiments for the next milestone

### REQ-REPORT-006: Research References Update

The research-scan workflow shall update `research-references.md` with a dated,
idempotent section containing:

- direct papers on constraint extraction or verification for instruction
  following and chain-of-thought reasoning
- benchmark assets for evaluating fine-grained constraint extraction
- risk evidence on chain-of-thought monitorability when that evidence changes
  the recommended Carnot direction

### REQ-REPORT-007: Research Studying Queue Update

The research-scan workflow shall update `research-studying.md` with a dated,
idempotent section that:

- ranks the new findings by relevance to Carnot's current gap
- records the main takeaways for experiment planning
- proposes the concrete follow-on experiments targeted for 2026-04-15

### REQ-REPORT-008: Idempotent Research Refresh

Re-running the research-scan workflow shall refresh the Exp 210 sections in
`research-references.md`, `research-studying.md`, and
`results/experiment_210_results.json` without duplicating the section body in
the markdown documents.

### REQ-REPORT-009: Milestone Retrospective Artifact

Milestone retrospective workflows shall read the authoritative experiment
result JSONs for the milestone, evaluate every planned success criterion from
those source fields, and write a machine-readable result artifact that includes:

- `milestone`
- `criteria_results`
- `criteria_met`
- `criteria_total`
- `notable_successes`
- `failures_or_partials`
- `bottlenecks_identified`
- `slowest_experiments`
- `wall_time_minutes`
- `wall_time_improvement_vs_prior_minutes`
- `retro_complete`
- `honest_verdict`

Missing gated experiment artifacts shall count as unmet criteria and shall be
reported explicitly rather than fabricated as successes.

### REQ-REPORT-010: Milestone .90 Operational Retrospective

The Exp 1164 milestone .90 retrospective workflow shall read the authoritative
Exp 1152 through Exp 1163 result JSON artifacts, plus the conductor log and the
prior Exp 1151 retrospective, and write
`results/experiment_1164_milestone_retro_90.json` with:

- `milestone` set to `2026.04.90`
- 13 criteria results, including the self-referential retrospective criterion
- top-level honest verdicts recorded for each Exp 1152 through Exp 1163 source
  artifact
- `exp906_appeared_in_slowest5`, derived from .90 conductor-log spans only
- `arxiv_submission_status`, derived from `exp1153.arxiv_submitted`
- `phase34_mandatory_tasks_complete`, true only when Exp 1154, Exp 1155, and
  Exp 1156 all exist and have honest verdicts
- `kv260_v6_kl_below_threshold`, derived from
  `exp1161.kv260_v6_kl_below_threshold_sequential_gibbs`
- .90 wall time and wall-time delta versus the Exp 1151 .89 baseline of
  257 minutes
- exactly three bottlenecks for .91 planning
- `honest_verdict` formatted as `N_of_13_criteria_met`

Missing source artifacts shall count as unmet criteria and shall be reported in
`failures_or_partials`.

### REQ-REPORT-011: Milestone .91 Success-Criteria Retrospective

The Exp 1177 milestone .91 retrospective workflow shall read the authoritative
Exp 1165 through Exp 1176 result JSON artifacts and write
`results/experiment_1177_milestone_retro_91.json` with:

- `milestone` set to `2026.04.91`
- `criteria_total` set to `13`
- `criteria_met`, derived from the source fields for all 12 prior experiments
  plus the self-referential retrospective criterion
- `criteria_status`, mapping each planned criterion to `MET`, `NOT_MET`, or
  `GATE_BLOCKED`
- `phase4_hold_lift_ready`, true only when
  `exp1167.paper_ready_for_arxiv_hold_lift == true`
- `top_3_successes`, `top_3_gaps`, and `open_items_for_92`
- `honest_verdict` formatted as `N_of_13_criteria_met`

The workflow shall not infer publication readiness from structural paper
completion alone: the Phase 4 hold-lift flag must follow the current Exp 1167
artifact after any operator override. Missing artifacts shall count as
`NOT_MET`, and explicitly blocked gated results shall be surfaced as
`GATE_BLOCKED` rather than as successful criteria.

### REQ-REPORT-012: Milestone .92 Paper-Integrity Retrospective

The Exp 1190 milestone .92 retrospective workflow shall read the authoritative
Exp 1178 through Exp 1189 result JSON artifacts and write
`results/experiment_1190_milestone_retro_92.json` with:

- `milestone` set to `2026.04.92`
- `criteria_total` set to `13`
- `criteria_met` and `criteria_score_pct`, derived from all 13 roadmap
  success criteria
- `paper_integrity_issues_resolved`, derived from Exp 1180 through Exp 1182
- `publication_hold_lifted`, true only when all 18 integrity issues are
  resolved, both audit scripts are active, the full paper audit passes, and
  operator approval is present
- `grpo_v5_result`, `k6_viable`, `k6_retired`, `dot_retired`,
  `latent_grpo_delta_pp`, `hex_operational`, and
  `phase4_stronger_baseline_result`, all derived from source artifacts
- `slowest_task_id`, derived from milestone conductor-log spans when available
- `open_items_for_93`
- `honest_verdict` set to `milestone_complete`, `milestone_partial`, or
  `milestone_failed`

Missing gated experiment artifacts shall count as unmet criteria and shall be
reported explicitly rather than inferred from downstream blocked artifacts.

### REQ-REPORT-013: Milestone .93 Success-Criteria Retrospective

The Exp 1202 milestone .93 retrospective workflow shall read the authoritative
Exp 1191 through Exp 1201 result JSON artifacts, plus the conductor log and the
current publication-hold notes, and write
`results/experiment_1202_milestone_retro_93.json` with:

- `milestone` set to `2026.04.93`
- `criteria_total` set to `12`
- `criteria_met` and `criteria_score_pct`, derived from all 12 roadmap success
  criteria
- `criteria_results`, mapping the 12 planned criteria to boolean pass/fail
  values
- `slowest_tasks`, derived from milestone conductor-log spans when available
- `publication_hold_status` set to `active` while operator approval is still
  required
- `dualgpu_utilization`, derived from Exp 1195 when it ran and otherwise
  reported as unavailable
- `grpo_trajectory`, including v3 `+2.86pp`, v4 `+10pp`, and the Exp 1195 v5
  result or missing/blocked status
- exactly three `significant_findings`
- exactly five `open_items_for_94`
- `retro_complete == true`
- `honest_verdict` set to `milestone_complete`, `milestone_partial`, or
  `milestone_failed`

Missing gated experiment artifacts shall count as unmet criteria. Blocked
artifacts shall only satisfy criteria that explicitly require an honest verdict
rather than a successful metric.

### REQ-REPORT-015: llama.cpp GPU Offload Verification Artifact

The Exp 1207 GPU-offload verification workflow shall produce
`results/experiment_1207_llama_cpp_gpu_offload_fix_v3.json` recording whether
the locally installed `llama-cpp-python` build has CUDA support compiled in
and reaches a usable inference throughput. The artifact shall include:

- `llama_cpp_version`
- `cuda_version_detected`
- `cuda_support_compiled`
- `install_method` (one of `pre-built-wheel`, `source-cmake-cuda`,
  `already-installed`)
- `llama_supports_gpu_offload`
- `throughput_tokens_per_sec`
- `llama_cpp_gpu_offload_verified`
- `honest_verdict` (one of `gpu_offload_verified`,
  `partial_offload_cpu_fallback`, `gpu_offload_failed`)

`llama_cpp_gpu_offload_verified` shall be `true` exactly when
`cuda_support_compiled` is `true` and `throughput_tokens_per_sec` meets or
exceeds 50.0. The honest verdict shall map directly from these inputs:
`gpu_offload_verified` when both conditions hold, `partial_offload_cpu_fallback`
when CUDA support is compiled but throughput falls below the threshold, and
`gpu_offload_failed` when CUDA support is not compiled.

### REQ-REPORT-016: Milestone .95 Success-Criteria Retrospective

The Exp 1228 milestone .95 retrospective workflow shall read the authoritative
Exp 1216 through Exp 1227 result JSON artifacts, plus the conductor log and the
current known-issues publication-hold notes, and write
`results/experiment_1228_milestone_retro_95.json` with:

- `milestone` set to `2026.04.95`
- `criteria_total` set to `13`
- `criteria_met` and `criteria_score_pct`, derived from all 13 planned
  milestone success criteria
- `criteria_results`, mapping the 13 planned criteria to boolean pass/fail
  values
- `slowest_tasks`, derived from milestone conductor-log spans when available
- `publication_hold_status`, reflecting the current publication hold in
  `ops/known-issues.md`
- `grpo_trajectory`, including v4 `+10pp`, v5 `-35pp`, VPS evaluation `+24pp`,
  VPS training result, and v6 FSPO result
- `phase5_derisking_status`, summarizing Phase 5 A/B/C outcomes
- exactly three `significant_findings`
- exactly five `open_items_for_96`
- `retro_complete == true`
- `honest_verdict` set to `milestone_complete`, `milestone_partial`, or
  `milestone_failed`

Missing source artifacts shall count as unmet criteria except for criteria that
explicitly require the artifact to exist with any honest verdict. Partial or
in-progress artifacts shall only satisfy criteria when their named source field
is true.

### REQ-REPORT-014: Retro Boundary Fix Documentation

The Exp 1204 retro-template fix workflow shall document the .94 resolution for
the `Retro Task Boundary Too Tight` known-issues entry without removing the
historical issue text. The workflow shall write
`results/experiment_1204_retro_template_step0_fix.json` with:

- `retro_boundary_issue_found`
- `resolution_note_added`
- `known_issues_file_updated`
- `retro_template_updated`
- `honest_verdict`

The resolution note shall state that Exp 1215 uses the STEP 0 skeleton pattern
and opus/100 turns. `retro_template_updated` shall be true exactly when the
resolution note was added during the workflow run. If the known-issues entry is
already resolved, the workflow shall not duplicate the note and shall report
`honest_verdict == "already_resolved"`.

### REQ-REPORT-017: Milestone .95 Retrospective Retry Artifact

The Exp 1229 milestone .95 retrospective retry workflow shall read the
authoritative Exp 1216 through Exp 1227 result JSON artifacts and write
`results/experiment_1229_milestone_retro_95.json` with:

- `milestone` set to `2026.04.95`
- `criteria_total` set to `13`
- `criteria_results`, mapping the 13 planned criteria to boolean pass/fail
  values with each source experiment recorded
- `criteria_met`, derived from the boolean criteria result count
- `findings_summary`, a brief account of what .95 proved and what failed
- `key_carry_forwards`, listing the highest-priority .96 follow-ups from the
  source artifacts and known-issues priorities
- `retro_complete == true`
- `honest_verdict` formatted as `milestone_N_of_13_criteria_met`

Missing artifacts or false source fields shall count as unmet criteria.
The retrospective self-criterion shall count as met only in the final artifact
that sets `retro_complete == true`.

### REQ-REPORT-018: Milestone .96 Success-Criteria Retrospective

The Exp 1241 milestone .96 retrospective workflow shall read the authoritative
Exp 1229 through Exp 1240 result JSON artifacts and write
`results/experiment_1241_milestone_retro_96.json` with:

- `milestone` set to `2026.04.96`
- `criteria_total` set to `13`
- `criteria_results`, mapping the 13 planned criteria to boolean pass/fail
  values
- `criteria_met`, derived from the boolean criteria result count
- `findings_summary`, a 3-to-5 sentence account of what .96 proved and what
  failed
- `key_carry_forwards`, listing the highest-priority .97 follow-ups
- `retro_complete == true`
- `honest_verdict` formatted as `milestone_N_of_13_criteria_met`

The workflow shall count missing artifacts, absent source fields, and false
source fields as unmet criteria. The verifier-redesign criterion may be
satisfied either by Exp 1233 `k_eff_after_redesign >= 4` or by an equivalent
merged artifact field proving the same threshold. The retrospective
self-criterion shall count as met only in the final artifact that sets
`retro_complete == true`.

### REQ-REPORT-019: Combined .95/.96 Retrospective Closure Artifact

The Exp 1242 combined retrospective workflow shall read the authoritative
Exp 1216 through Exp 1240 result JSON artifacts, plus the bootstrap Exp 1229
and Exp 1241 retrospective artifacts, and write
`results/experiment_1242_combined_retro_95_96.json` with:

- `milestone_95` set to `2026.04.95`
- `milestone_96` set to `2026.04.96`
- `criteria_95_total` and `criteria_96_total` set to `13`
- `criteria_95_results` and `criteria_96_results`, each mapping all 13 planned
  criteria to boolean pass/fail values
- `criteria_95_met` and `criteria_96_met`, derived from the boolean result
  counts
- `findings_summary`, briefly explaining what .96 proved, what failed, and the
  top lesson
- `key_carry_forwards`, listing the highest-priority .97 follow-ups
- `retro_complete == true`
- `honest_verdict` formatted as
  `milestone_96_N_of_13_criteria_met`

Missing artifacts, absent source fields, false source fields, and bootstrap-only
retrospectives shall count as unmet criteria unless a criterion explicitly
requires an honest partial verdict.

### REQ-REPORT-020: Milestone .97 Success-Criteria Retrospective

The Exp 1254 milestone .97 retrospective workflow shall read the authoritative
Exp 1242 through Exp 1253 result JSON artifacts and write
`results/experiment_1254_milestone_retro_97.json` with:

- `milestone` set to `2026.04.97`
- `criteria_total` set to `13`
- `criteria_results`, mapping the 13 planned criteria to boolean pass/fail
  values
- `criteria_met`, derived from the boolean criteria result count
- `findings_summary`, a 2-to-3 sentence account of what .97 proved
- `key_carry_forwards`, listing the highest-priority .98 follow-ups
- `retro_complete == true`
- `honest_verdict` formatted as `milestone_N_of_13_criteria_met`

Missing artifacts, absent source fields, false source fields, and numeric
threshold misses shall count as unmet criteria. The retrospective
self-criterion shall count as met only in the final artifact that sets
`retro_complete == true`.

### REQ-REPORT-021: Combined .95/.96/.97 Retrospective Closure Artifact

The Exp 1255 combined retrospective workflow shall read the stale Exp 1242
combined .95/.96 artifact, the stale Exp 1254 .97 artifact, and the
authoritative Exp 1216 through Exp 1254 result JSON artifacts, and write
`results/experiment_1255_combined_retro_95_96_97.json` with:

- `schema` set to `milestone_retro_combined_v2`
- `criteria_97_total`, `criteria_96_total`, and `criteria_95_total` set to
  `13`
- `criteria_97_results`, `criteria_96_results`, and `criteria_95_results`,
  each mapping all 13 planned criteria to boolean pass/fail values
- `criteria_97_met`, `criteria_96_met`, and `criteria_95_met`, each derived
  from the corresponding boolean result count
- `findings_summary`, a 2-to-3 sentence account of what .97 proved and failed
- `key_carry_forwards`, listing the highest-priority .98 follow-ups
- `retro_complete == true`
- `honest_verdict` formatted as
  `milestone_97_N_of_13_criteria_met`

Missing artifacts, stale bootstrap retrospectives, absent source fields, false
source fields, and numeric threshold misses shall count as unmet criteria
unless the criterion is the current Exp 1255 self-referential completion
criterion.

### REQ-REPORT-022: Milestone .98 Success-Criteria Retrospective

The Exp 1267 milestone .98 retrospective workflow shall read the authoritative
Exp 1255 through Exp 1266 result JSON artifacts and write
`results/experiment_1267_milestone_retro_98.json` with:

- `schema` set to `milestone_retro_v3`
- `milestone` set to `2026.04.98`
- `criteria_total` set to `13`
- `criteria_results`, mapping the 13 planned criteria to boolean pass/fail
  values
- `criteria_met`, derived from the boolean criteria result count
- `findings_summary`, a 2-to-3 sentence account of what .98 achieved and where
  it remained incomplete
- `key_carry_forwards`, listing the highest-priority .99 follow-ups
- `top_successes`, listing the top completed outcomes
- `top_gaps`, listing the top unmet outcomes
- `retro_complete == true`
- `honest_verdict` formatted as `milestone_98_N_of_13_criteria_met`

Missing artifacts, absent source fields, false source fields, numeric threshold
misses, and in-progress source verdicts shall count as unmet criteria. The
retrospective self-criterion shall count as met only in the final artifact that
sets `retro_complete == true`.

## Scenarios

### SCENARIO-REPORT-001: Nested Live Provenance Is Promoted

**Given** a result artifact with no top-level `inference_mode`
**And** `metadata.inference_mode` is `live_gpu`
**When** the cleanup workflow runs
**Then** the artifact is marked as a validated live result
**And** the top-level provenance summary records `live_gpu`
**And** the header states that the artifact is validated

### SCENARIO-REPORT-002: Simulated Artifact Receives Warning

**Given** a result artifact whose detected provenance is `simulated` or
`simulation`
**When** the cleanup workflow runs
**Then** the artifact is preserved
**And** a warning header is added
**And** public documentation labels the referenced benchmark as simulated

### SCENARIO-REPORT-003: Missing Provenance Is Disclosed

**Given** a result artifact with no detectable `inference_mode`
**When** the cleanup workflow runs
**Then** the artifact is preserved
**And** a warning header is added
**And** public documentation labels any referenced benchmark as unverified or
missing provenance rather than as a validated live result

### SCENARIO-REPORT-004: Exp 210 Scan Writes the Curated Artifact

**Given** the research docs exist
**When** the Exp 210 research-scan workflow runs
**Then** `results/experiment_210_results.json` is written
**And** it includes ranked papers, benchmark assets, and proposed experiments
**And** the research docs gain a dated Exp 210 section

### SCENARIO-REPORT-005: Exp 210 Rerun Refreshes In Place

**Given** the Exp 210 sections already exist in the research docs
**When** the research-scan workflow runs again
**Then** the Exp 210 section bodies are replaced in place
**And** the docs do not accumulate duplicate Exp 210 blocks

### SCENARIO-REPORT-006: Missing Gated Artifact Counts As Unmet

**Given** a milestone retrospective expects a gated result artifact
**And** that artifact is missing because the upstream experiment was blocked
**When** the retrospective evaluates the milestone criteria
**Then** the gated criterion is false
**And** the retrospective artifact reports the missing source in
`failures_or_partials`

### SCENARIO-REPORT-007: Milestone .90 Gates Are Derived From Source Fields

**Given** Exp 1152 through Exp 1163 source artifacts exist
**And** Exp 1153 has `arxiv_submitted == false`
**And** Exp 1161 has
`kv260_v6_kl_below_threshold_sequential_gibbs == true`
**When** the Exp 1164 retrospective workflow runs
**Then** the artifact reports `arxiv_submission_status == "upload_pending"`
**And** it reports `kv260_v6_kl_below_threshold == true`
**And** its honest verdict matches the number of true criteria out of 13

### SCENARIO-REPORT-008: Milestone .91 Publication Hold Follows Exp 1167

**Given** Exp 1165 through Exp 1176 source artifacts exist
**And** Exp 1167 has `paper_ready_for_arxiv_hold_lift == false` after an
operator figure-integrity override
**And** Exp 1173 has `status == "blocked"` with
`grpo_v5_honest_result == false`
**When** the Exp 1177 retrospective workflow runs
**Then** `phase4_hold_lift_ready == false`
**And** `paper_v4_phase4_section_integrated == NOT_MET`
**And** `grpo_v5_honest_result == GATE_BLOCKED`
**And** `honest_verdict` matches the number of met criteria out of 13

### SCENARIO-REPORT-009: Milestone .92 Hold Remains Until Full Audit Passes

**Given** Exp 1178 through Exp 1189 source artifacts exist
**And** Exp 1179 and Exp 1180 are missing
**And** Exp 1183 reports `4_test_full_pass == false`
**When** the Exp 1190 retrospective workflow runs
**Then** the missing source artifacts count as unmet criteria
**And** `publication_hold_lifted == false`
**And** `open_items_for_93` includes the missing GPU-offload and critical
paper-integrity tasks
**And** `honest_verdict == "milestone_partial"`

### SCENARIO-REPORT-010: Milestone .93 Missing Gates Remain Unmet

**Given** Exp 1191 through Exp 1201 source artifacts are expected
**And** Exp 1192, Exp 1193, Exp 1194, Exp 1195, and Exp 1197 are missing
**And** Exp 1196, Exp 1198, Exp 1200, and Exp 1201 report blocked artifacts
**When** the Exp 1202 retrospective workflow runs
**Then** missing metric criteria count as false
**And** honest-verdict criteria count as true only when the criterion asks for
honest reporting
**And** `publication_hold_status == "active"`
**And** `honest_verdict == "milestone_failed"`

### SCENARIO-REPORT-011: Exp 1204 Adds Retro Boundary Resolution Note

**Given** `ops/known-issues.md` contains the `Retro Task Boundary Too Tight`
entry
**And** the Exp 1215 milestone-retro roadmap task uses STEP 0 and
`max_turns: 100`
**When** the Exp 1204 retro-template fix workflow runs
**Then** `ops/known-issues.md` gains the .94 resolution note without pruning
the original entry
**And** `results/experiment_1204_retro_template_step0_fix.json` reports
`retro_boundary_issue_found == true`
**And** `resolution_note_added == true`
**And** `known_issues_file_updated == true`
**And** `retro_template_updated == true`
**And** `honest_verdict == "template_updated"`

### SCENARIO-REPORT-012: Exp 1207 Records GPU Offload Verification

**Given** the local `llama-cpp-python` build reports
`llama_supports_gpu_offload == True`
**And** a smoke-test sustains at least 50 tokens per second
**When** the Exp 1207 GPU-offload verification workflow runs
**Then** `results/experiment_1207_llama_cpp_gpu_offload_fix_v3.json` reports
`cuda_support_compiled == true`
**And** `llama_cpp_gpu_offload_verified == true`
**And** `honest_verdict == "gpu_offload_verified"`

### SCENARIO-REPORT-013: Milestone .95 Partial Results Stay Honest

**Given** Exp 1216 through Exp 1227 source artifacts exist
**And** Exp 1217 is blocked without `autofill_script_exists == true`
**And** Exp 1221 has `grpo_v6_fspo_delta_measured == false`
**And** Exp 1225 has `gaming_defense_measured == false`
**When** the Exp 1228 retrospective workflow runs
**Then** those three criteria count as false
**And** the retrospective criterion counts as true
**And** the artifact reports `criteria_met == 10`
**And** `publication_hold_status == "active"`
**And** `honest_verdict == "milestone_partial"`

### SCENARIO-REPORT-014: Exp 1229 Retry Counts Source Criteria

**Given** Exp 1216 through Exp 1227 source artifacts exist
**And** Exp 1217 lacks `autofill_script_exists == true`
**And** Exp 1221 has only a partial wall-budget-exhausted verdict
**And** Exp 1225 lacks `gaming_defense_measured == true`
**When** the Exp 1229 retry retrospective workflow runs
**Then** those three source criteria count as false
**And** the retrospective criterion counts as true in the final artifact
**And** the artifact reports `criteria_met == 10`
**And** `honest_verdict == "milestone_10_of_13_criteria_met"`

### SCENARIO-REPORT-015: Exp 1241 Counts .96 Source Criteria

**Given** Exp 1229 through Exp 1240 source artifacts are expected
**And** Exp 1233 and Exp 1234 are missing
**And** Exp 1230 has `autofill_script_exists == true`
**And** the remaining source artifacts are in progress, blocked, or missing the
required true fields
**When** the Exp 1241 retrospective workflow runs
**Then** only the autofill criterion and the retrospective self-criterion count
as met
**And** the artifact reports `criteria_met == 2`
**And** `honest_verdict == "milestone_2_of_13_criteria_met"`

### SCENARIO-REPORT-016: Exp 1242 Closes Bootstrap .95/.96 Retrospectives

**Given** the Exp 1229 and Exp 1241 retrospective artifacts are still
bootstrap-only
**And** Exp 1216 through Exp 1240 source artifacts contain mixed completed,
partial, blocked, and missing criterion evidence
**When** the Exp 1242 combined retrospective workflow runs
**Then** the bootstrap retrospectives count as unmet source criteria
**And** the self-referential .96 criterion counts as met in the final Exp 1242
artifact
**And** the artifact reports separate .95 and .96 criteria maps and counts
**And** `honest_verdict` is formatted as
`milestone_96_N_of_13_criteria_met`

### SCENARIO-REPORT-017: Exp 1254 Counts .97 Source Criteria

**Given** Exp 1242 through Exp 1253 source artifacts are expected
**And** Exp 1246 is missing
**And** only Exp 1248 and Exp 1251 contain the required true source evidence
**When** the Exp 1254 retrospective workflow runs
**Then** missing and false source criteria count as unmet
**And** the retrospective self-criterion counts as met in the final artifact
**And** the artifact reports `criteria_met == 3`
**And** `honest_verdict == "milestone_3_of_13_criteria_met"`

### SCENARIO-REPORT-018: Exp 1255 Closes .95, .96, and .97 Retrospectives

**Given** Exp 1242 and Exp 1254 are stale bootstrap retrospective artifacts
**And** Exp 1248 reports `post_cd_auroc >= 0.80`
**And** Exp 1251 reports `nonmonotonicity_characterized == true`
**When** the Exp 1255 combined retrospective workflow runs
**Then** the stale prior retrospective criteria count as unmet where referenced
as source evidence
**And** the artifact reports `.97 == 4/13`, `.96 == 2/13`, and `.95 == 10/13`
**And** `retro_complete == true`
**And** `honest_verdict == "milestone_97_4_of_13_criteria_met"`

### SCENARIO-REPORT-019: Exp 1267 Counts .98 Source Criteria

**Given** Exp 1255 through Exp 1266 source artifacts contain the current .98
criterion evidence
**And** Exp 1256 reports `orthogonality_matrix_computed == true`
**And** Exp 1264 reports `tss_instrumented == true`
**And** Exp 1265 reports `diffutruth_comparison_measured == true`
**And** Exp 1266 reports a non-null `quantkan_3bit_auroc`
**When** the Exp 1267 retrospective workflow runs
**Then** missing artifacts, in-progress verdicts, false fields, and threshold
misses count as unmet
**And** the retrospective self-criterion counts as met in the final artifact
**And** the artifact reports `criteria_met == 5`
**And** `honest_verdict == "milestone_98_5_of_13_criteria_met"`


### REQ-PUBLISH-003: HuggingFace README Accuracy Audit

All HuggingFace model READMEs for Phase 1 per-token activation EBMs shall
be audited and updated to:

- Include a Phase 1 disclaimer stating that the models detect model confidence,
  not factual correctness.
- Point users to `pip install carnot` for production use.
- Include a link to the latest full-scale benchmark results (Exp 316) where
  available.

The update operation shall be idempotent: re-running against a README that
already contains the Phase 1 patch shall skip that repo without error.

The carnot-joint-constraint-v1 model card shall include an honest
"RESEARCH PROTOTYPE — weights not published" label when the trained weights
artifact is absent.

### SCENARIO-PUBLISH-005: Phase 1 Patch Is Idempotent

**Given** a model README that already contains the Phase 1 disclaimer section
**When** `model_card_update(repo_id, patch)` is called
**Then** the README is not modified
**And** the repo is recorded in `models_skipped` (not `models_updated`)
**And** no HuggingFace API upload is made for that repo

### SCENARIO-PUBLISH-006: Blocked When Credentials Absent

**Given** HuggingFace credentials are not available (CLI and Python API both fail)
**When** `run_experiment_317()` is called
**Then** the function returns an artifact with `blocked == True`
**And** the artifact includes `exp_317_next_action` with login instructions
**And** `models_updated` is empty
**And** no HuggingFace API calls are made

### REQ-PUBLISH-004: Live HF Publish with Live-GPU Benchmarks Embedded

All HuggingFace model READMEs updated under REQ-PUBLISH-003 shall additionally
embed live-GPU benchmark results from Exp 328 when available.

- The publish wrapper (Exp 330) shall load `results/experiment_328_live_fullscale_results.json`
  and adapt its `first_live_run_evidence` and `baseline_deviation` fields into a
  `per_variant_results`-compatible structure for embedding.
- If Exp 328 results are absent, the wrapper shall fall back to Exp 316 simulated
  results, labeling them clearly as simulated.
- The wrapper artifact shall record `live_benchmark_embedded: true/false` to
  distinguish live from simulated benchmark embedding.
- The operation shall be idempotent: if the Phase 1 sentinel is already present,
  the repo is skipped without re-upload.
- If HF credentials are absent, the artifact shall have `status="blocked"` with
  `next_action="huggingface-cli login"`.

### SCENARIO-PUBLISH-007: Exp 328 Live Results Embedded in Model Cards

**Given** `results/experiment_328_live_fullscale_results.json` exists with `status="success"`
**And** HuggingFace credentials are available
**When** `run_experiment_330()` is called
**Then** the Phase 1 README patch embeds live-GPU accuracy numbers from Exp 328
**And** the patch labels the numbers as `inference_mode=live_gpu`
**And** `live_benchmark_embedded == True` in the artifact

### SCENARIO-PUBLISH-008: Blocked When HF Credentials Absent (Exp 330)

**Given** HuggingFace credentials are not available
**When** `run_experiment_330()` is called
**Then** the artifact has `status="blocked"`
**And** `next_action` contains `"huggingface-cli login"`
**And** no HuggingFace API calls are made

## Implementation Status

| Requirement | Implementation | Tests | Status |
|------------|----------------|-------|--------|
| REQ-REPORT-001 | `scripts/experiment_209_cleanup.py` | `tests/python/test_experiment_209_cleanup.py` | Implemented |
| REQ-REPORT-002 | `scripts/experiment_209_cleanup.py` | `tests/python/test_experiment_209_cleanup.py` | Implemented |
| REQ-REPORT-003 | `scripts/experiment_209_cleanup.py`, `README.md` | `tests/python/test_experiment_209_cleanup.py` | Implemented |
| REQ-REPORT-004 | `scripts/experiment_209_cleanup.py`, `docs/technical-report.md`, `docs/index.html` | `tests/python/test_experiment_209_cleanup.py` | Implemented |
| REQ-REPORT-005 | `scripts/experiment_210_research_scan.py`, `results/experiment_210_results.json` | `tests/python/test_experiment_210_research_scan.py` | Implemented |
| REQ-REPORT-006 | `scripts/experiment_210_research_scan.py`, `research-references.md` | `tests/python/test_experiment_210_research_scan.py` | Implemented |
| REQ-REPORT-007 | `scripts/experiment_210_research_scan.py`, `research-studying.md` | `tests/python/test_experiment_210_research_scan.py` | Implemented |
| REQ-REPORT-008 | `scripts/experiment_210_research_scan.py` | `tests/python/test_experiment_210_research_scan.py` | Implemented |
| REQ-REPORT-009 | `scripts/experiment_1138_milestone_retro_88.py` | `tests/python/test_experiment_1138_milestone_retro_88.py` | Implemented |
| REQ-REPORT-010 | `scripts/experiment_1164_milestone_retro_90.py`, `results/experiment_1164_milestone_retro_90.json` | `tests/python/test_experiment_1164_milestone_retro_90.py` | Implemented |
| REQ-REPORT-011 | `scripts/experiment_1177_milestone_retro_91.py`, `results/experiment_1177_milestone_retro_91.json` | `tests/python/test_experiment_1177_milestone_retro_91.py` | Implemented |
| REQ-REPORT-012 | `scripts/experiment_1190_milestone_retro_92.py`, `results/experiment_1190_milestone_retro_92.json` | `tests/python/test_experiment_1190_milestone_retro_92.py` | Implemented |
| REQ-REPORT-013 | `scripts/experiment_1202_milestone_retro_93.py`, `results/experiment_1202_milestone_retro_93.json` | `tests/python/test_experiment_1202_milestone_retro_93.py` | Implemented |
| REQ-REPORT-014 | `python/carnot/reporting/retro_template_step0_fix.py`, `results/experiment_1204_retro_template_step0_fix.json` | `tests/python/test_retro_template_step0_fix.py` | Implemented |
| REQ-REPORT-015 | `python/carnot/reporting/llama_cpp_gpu_offload_fix.py`, `results/experiment_1207_llama_cpp_gpu_offload_fix_v3.json` | `tests/python/test_llama_cpp_gpu_offload_fix.py` | Implemented |
| REQ-REPORT-016 | `scripts/experiment_1228_milestone_retro_95.py`, `results/experiment_1228_milestone_retro_95.json` | `tests/python/test_experiment_1228_milestone_retro_95.py` | Implemented |
| REQ-REPORT-017 | `python/carnot/reporting/milestone_retro_95_retry.py`, `results/experiment_1229_milestone_retro_95.json` | `tests/python/test_milestone_retro_95_retry.py` | Implemented |
| REQ-REPORT-018 | `python/carnot/reporting/milestone_retro_96.py`, `results/experiment_1241_milestone_retro_96.json` | `tests/python/test_milestone_retro_96.py` | Implemented |
| REQ-REPORT-019 | `python/carnot/reporting/combined_retro_95_96.py`, `results/experiment_1242_combined_retro_95_96.json` | `tests/python/test_combined_retro_95_96.py` | Implemented |
| REQ-REPORT-020 | `python/carnot/reporting/milestone_retro_97.py`, `results/experiment_1254_milestone_retro_97.json` | `tests/python/test_milestone_retro_97.py` | Implemented |
| REQ-REPORT-021 | `python/carnot/reporting/combined_retro_95_96_97.py`, `results/experiment_1255_combined_retro_95_96_97.json` | `tests/python/test_combined_retro_95_96_97.py` | Implemented |
| REQ-REPORT-022 | `python/carnot/reporting/milestone_retro_98.py`, `results/experiment_1267_milestone_retro_98.json` | `tests/python/test_milestone_retro_98.py` | Implemented |
| REQ-PUBLISH-003 | `scripts/experiment_317_hf_publish.py` | `tests/python/test_experiment_317_hf_publish.py` | Implemented |
| REQ-PUBLISH-004 | `scripts/experiment_330_hf_live_publish.py` | `tests/python/test_experiment_330_hf_live_publish.py` | Implemented |
