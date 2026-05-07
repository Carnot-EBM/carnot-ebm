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

### REQ-REPORT-023: Milestone .99 Success-Criteria Retrospective

The Exp 1281 milestone .99 retrospective workflow shall read the authoritative
Exp 1268 through Exp 1280 result JSON artifacts and write
`results/experiment_1281_milestone_retro_99.json` with:

- `schema` set to `milestone_retro_v4`
- `milestone` set to `2026.04.99`
- `criteria_total` set to `14`
- `criteria_results`, mapping the 14 planned criteria to one of `MET`,
  `NOT_MET`, `GATED`, `BLOCKED`, or `MISSING`
- `criteria_met`, derived from the count of `MET` criteria
- `top_successes`, listing the strongest completed outcomes
- `top_gaps`, listing the highest-priority unmet, gated, blocked, or missing
  outcomes
- `self_learning_result`, summarising available self-learning deltas from Exp
  1273 and Exp 1274
- `sota_model_usage_summary`, summarising whether SOTA GGUF models were used
  for headline-eligible LLM work
- `stale_artifacts`, listing missing, in-progress, bootstrap, gated, or blocked
  source artifacts that need carry-forward attention
- `key_carry_forwards`, listing the highest-priority .100 follow-ups
- `retro_complete == true`
- `honest_verdict` formatted as `milestone_99_N_of_14_criteria_met`

Missing artifacts shall count as `MISSING` unless an explicit unmet upstream
gate makes the planned task `GATED`. Blocked terminal artifacts shall count as
`BLOCKED`. In-progress or bootstrap-only artifacts shall count as `NOT_MET`.
The retrospective self-criterion shall count as `MET` only in the final
artifact that sets `retro_complete == true`.

### REQ-REPORT-025: Milestone .100 Success-Criteria Retrospective

The Exp 1295 milestone .100 retrospective workflow shall read the authoritative
Exp 1282 through Exp 1294 result JSON artifacts and write
`results/experiment_1295_milestone_retro_100.json` with:

- `schema` set to `milestone_retro_v5`
- `milestone` set to `2026.04.100`
- `criteria_total` set to `14`
- `criteria_results`, mapping the 14 planned criteria to one of `MET`,
  `NOT_MET`, `GATED`, `BLOCKED`, or `MISSING`
- `criteria_met`, derived from the count of `MET` criteria
- `top_successes`, listing the strongest completed outcomes
- `top_gaps`, listing the highest-priority unmet, gated, blocked, or missing
  outcomes
- `self_learning_result`, summarising DVI/replay, GRPO, and skill-graph
  outcomes from Exp 1288 through Exp 1290
- `sota_model_usage_summary`, summarising SOTA GGUF cache readiness and
  headline-eligible model usage from Exp 1282 and the SOTA-gated certificate
  tasks
- `continuous_repair_summary`, summarising HardNet++, DSP feasibility-channel,
  and energy-bridge outcomes from Exp 1291 through Exp 1293
- `publication_state`, reporting whether Exp 1294 produced an arXiv receipt or
  the exact blocker
- `stale_artifacts`, listing missing, in-progress, bootstrap, gated, or blocked
  source artifacts that need carry-forward attention
- `key_carry_forwards`, listing the highest-priority .101 follow-ups
- `status == "complete"`
- `retro_complete == true`
- `honest_verdict` formatted as `milestone_100_N_of_14_criteria_met`

Missing artifacts shall count as `MISSING` unless an explicit unmet upstream
gate makes the planned task `GATED`. Blocked terminal artifacts shall count as
`BLOCKED`, except conductor pre-gate artifacts whose failed gate is an unmet
milestone dependency shall be classified as `GATED`. In-progress or
bootstrap-only artifacts shall count as `NOT_MET`. The retrospective
self-criterion shall count as `MET` only in the final artifact that sets
`retro_complete == true`.

### REQ-REPORT-026: EBT/ARM/EBM-CoT Energy Bridge Audit v2

The Exp 1306 energy bridge audit workflow shall read the local research notes
and prior milestone artifacts without requiring network access, then write
`results/experiment_1306_ebt_arm_ebm_cot_energy_bridge_audit_v2.json` with:

- `status`
- `energy_bridge_completed`
- `ebt_citation_count_checked`
- `arm_ebm_alignment_note`
- `ebm_cot_sequence_energy_note`
- `extropic_kona_status_checked`
- `hardware_sampler_context_recorded`
- `honest_verdict`

The artifact shall use run date `20260505`, record project root
`/home/ianblenke/github.com/ianblenke/carnot`, carry the Exp 1293 blocked
prior-failure context, and explicitly distinguish verifier-energy work already
implemented locally from EBT, ARM-EBM, EBM-CoT, Extropic TSU, p-bit, and Kona
items that remain strategic or future sampler context.

### REQ-REPORT-027: Milestone .101 Success-Criteria Retrospective

The Exp 1308 milestone .101 retrospective workflow shall read the authoritative
Exp 1296 through Exp 1307 result JSON artifacts, plus the current .101 roadmap
criteria, and write `results/experiment_1308_milestone_retro_101.json` with:

- `schema` set to `milestone_retro_v6`
- `milestone` set to `2026.04.101`
- `criteria_total` set to `13`
- `criteria_results`, mapping all 13 planned criteria to one of `MET`,
  `BLOCKED`, `GATED`, `MISSING`, or `FAILED`
- `criteria_met`, derived from the count of `MET` criteria
- `carry_forward_tasks`, with exact `prior_failures` entries suitable for the
  next milestone planner
- `activation_failures`, separate from gated/skipped tasks and scientific
  negative results
- `gated_or_skipped_tasks`, listing criteria skipped because prerequisite gates
  did not open
- `scientific_negative_results`, listing terminal experiments whose science was
  honestly negative or limited rather than merely activation-blocked
- `docs_reconciled`
- `status == "complete"`
- `retro_complete == true`
- `honest_verdict` formatted as `milestone_101_N_of_13_criteria_met`

Missing artifacts shall count as `MISSING` unless an unmet upstream gate makes
the planned work `GATED`. Conductor pre-gate artifacts whose failed gate is an
unmet milestone dependency shall be classified as `GATED`, not `BLOCKED`.
Terminal artifacts that satisfy a criterion by writing an exact blocker shall
count as `MET` only when that criterion explicitly allows an exact blocker or
hold state. The retrospective self-criterion shall count as `MET` only in the
final artifact that sets `retro_complete == true`.

### REQ-REPORT-028: Milestone .102 Success-Criteria Retrospective

The Exp 1322 milestone .102 retrospective workflow shall read the authoritative
Exp 1309 through Exp 1321 result JSON artifacts, plus the available .102 roadmap
planning documents, and write `results/experiment_1322_milestone_retro_102.json`
with:

- `schema` set to `milestone_retro_v7`
- `milestone` set to `2026.04.102`
- `criteria_total` set to `14`
- `criteria_results`, mapping all 14 planned criteria to one of `MET`,
  `GATED`, `MISSING`, `BLOCKED`, or `FAILED`
- `criteria_met`, derived from the count of `MET` criteria
- `sota_runtime_recovered`, true only when Exp 1309 and Exp 1310 prove a
  headline-capable two-model local SOTA pair
- `certificate_path_headline_ready`, true only when the certificate parse,
  semantic-validator, and safe-prefix gates all meet their planned thresholds
- `continuous_self_learning_advanced`, true only when Exp 1315 reports
  non-forgetting evidence, no memory regressions, and a positive controlled
  self-learning delta
- `repair_generalization_advanced`, derived from Exp 1318 held-out learned
  stop-policy evidence without upgrading replay-distribution generalization
  into a broad repair-generalization claim
- `hardware_claims_honest`, true only when Exp 1319 and Exp 1320 stay scoped to
  audit/design-packet evidence and do not claim unsupported hardware execution
- `publication_state`, derived from Exp 1321
- `carry_forward_tasks`, explaining every unmet gated, missing, blocked, or
  failed criterion and any met-but-partial finding that blocks headline use
- `status == "complete"`
- `retro_complete == true`
- `honest_verdict` formatted as `milestone_102_N_of_14_criteria_met`

Missing artifacts shall count as unmet criteria. A missing artifact may be
classified as `GATED` when an upstream .102 gate demonstrably did not open, but
it still shall not increment `criteria_met`. Blocked conductor pre-gate
artifacts shall classify as `GATED` when the failed gate is an unmet milestone
dependency. Completed measurement artifacts may count as `MET` while still
adding carry-forward work when their measured values leave a downstream gate or
headline path closed. The retrospective self-criterion shall count as `MET` only
in the final artifact that sets `retro_complete == true`.

### REQ-REPORT-029: Milestone .104 Carry-Forward Retrospective

The Exp 1350 milestone .104 retrospective workflow shall read the authoritative
Exp 1337 through Exp 1349 result JSON artifacts, plus the available .104
roadmap planning documents, and write
`results/experiment_1350_milestone_104_retro_carryforward.json` with:

- `status` set to `complete`
- `criteria_total`, derived from the .104 roadmap success criteria
- `criteria_met`, derived only from observed source artifacts and terminal
  blockers that satisfy the planned success criterion
- `experiment_statuses`, summarizing Exp 1337 through Exp 1349 and explicitly
  recording missing artifacts
- `certificate_branch_verdict`, keeping certificate-tail, semantic-validator,
  and scheduler claims inside observed gates
- `self_learning_verdict`, separating replay-only progress from gated DVI/GRPO
  headline readiness
- `hardware_verdict`, distinguishing simulation/accounting evidence from
  unverified hardware execution
- `publication_hold_state`, preserving the active publication hold unless the
  source artifacts show a valid hold lift
- `carry_forward_tasks`, specific enough to seed the next roadmap
- `prior_failure_hygiene_notes`, explaining whether .103 stale skeleton and
  pre-test issues were closed cleanly
- `honest_verdict`, formatted as a conservative milestone carry-forward verdict

Missing roadmap files or experiment artifacts shall count as unmet evidence and
shall be reported explicitly rather than inferred away. Gated DVI/GRPO tasks
shall count as met only when they run or when their planned criterion is exactly
that they remain closed behind structured gates. The retrospective artifact
shall include run metadata using run date `20260505` and project root
`/home/ianblenke/github.com/ianblenke/carnot`.

### REQ-REPORT-030: Milestone .104 Carry-Forward Artifact Integrity Audit

The Exp 1351 `.104` carry-forward artifact integrity audit workflow shall read
the `.104` roadmap/planning context, conductor log, ops changelog/status, and
Exp 1337 through Exp 1350 result artifacts without rerunning experiments or
rewriting `.104` source artifacts. It shall write
`results/experiment_1351_104_carryforward_artifact_integrity_audit.json` with:

- `status`
- `artifact_paths_checked`
- `missing_artifacts`
- `stale_or_blocked_artifacts`
- `gates_open`
- `gates_closed`
- `prior_failure_requirements`
- `docs_reconciliation_needed`
- `terminal_certificate_required`
- `honest_verdict`

The workflow shall mark `terminal_certificate_required == true` and keep
semantic-validator, scheduler, DVI, and GRPO gates closed unless a terminal
Exp 1340 replacement artifact and the relevant semantic-validator upstream
evidence are present. It shall include run metadata using run date `20260505`
and project root `/home/ianblenke/github.com/ianblenke/carnot`.

### REQ-REPORT-031: Milestone .105 Retrospective and .106 Carry-Forward Plan

The Exp 1363 `.105` milestone retrospective workflow shall read the `.105`
roadmap/planning context and Exp 1351 through Exp 1362 result artifacts without
rerunning experiments or rewriting source artifacts. It shall write
`results/experiment_1363_milestone_105_retro_carryforward.json` with:

- `status`
- `criteria_total`
- `criteria_met`
- `experiment_statuses`
- `certificate_branch_verdict`
- `semantic_repair_verdict`
- `self_learning_verdict`
- `hardware_verdict`
- `publication_hold_state`
- `carry_forward_tasks`
- `prior_failure_hygiene_notes`
- `honest_verdict`

The workflow shall distinguish terminal SOTA certificate evidence from a
successful certificate branch: Exp 1353 may count as terminal evidence while
still recording that parse, truthfulness, trigger-token, and UNKNOWN-preserving
rates failed the semantic gate. Gated tasks that only write blocked artifacts
shall not be reported as semantic, DVI, GRPO, or policy-update successes.
Missing gated artifacts shall be reported explicitly. Self-learning verdicts
shall separate replay-only evidence from headline evidence, and hardware and
publication verdicts shall preserve no-hardware-claim and publication-hold
boundaries. The artifact shall include run metadata using run date `20260505`
and project root `/home/ianblenke/github.com/ianblenke/carnot`.

### REQ-REPORT-032: Milestone .109 Retrospective and Carry-Forward Plan

The Exp 1424 `.109` milestone retrospective workflow shall read the `.109`
roadmap planning context, conductor log, and Exp 1412 through Exp 1423 result
artifacts without rerunning the milestone experiments. It shall write
`results/experiment_1424_milestone_109_retro.json` with:

- `status` set to `complete` in the final artifact
- `criteria_total` set to `13`
- `criteria_met`, derived from criteria whose status is exactly `MET`
- `success_criteria_results`, mapping all 13 roadmap criteria to one of
  `MET`, `NOT_MET`, `GATE_BLOCKED`, or `BLOCKED`
- `retired_experiments`, listing exact reruns that should not be proposed again
  without a materially different root-cause fix
- `carry_forward_tasks`, each with a concrete root-cause plan
- `prior_failures_required_next`, containing a `prior_failures` entry for every
  carry-forward task
- `gpu_utilization_summary`, using observed artifact or log telemetry when
  present and reporting unavailability honestly otherwise
- `honest_verdict`, formatted as a concise milestone outcome

Blocked source artifacts shall not count as met criteria even when they contain
a partial positive metric. Missing gated artifacts shall count as
`GATE_BLOCKED` only when an upstream gate closure is visible in the roadmap,
source artifacts, or conductor log; otherwise missing artifacts are `NOT_MET`.
The retrospective self-criterion shall count as `MET` only in the final
artifact when writing the retro brings the milestone to the planned
`criteria_met >= 10/13` threshold. Carry-forward tasks shall include
`prior_failures` entries that name the failed experiment, verdict, root-cause
response, and whether the next attempt should be retired on the same verdict.

### REQ-REPORT-033: Milestone .110 Carry-Forward Activation Manifest

The Exp 1425 `.110` carry-forward activation audit workflow shall read the
terminal Exp 1424 `.109` retrospective, the unresolved upstream source
artifacts named by the retro, and
`openspec/change-proposals/research-roadmap-vNEXT.md` without rerunning or
editing prior experiments. It shall write
`ops/milestone_110_carryforward_manifest.md` and
`results/experiment_1425_109_carryforward_activation_audit.json`.

The markdown manifest shall include a table with columns for:

- track
- prior evidence
- `.110` task
- gate rule
- retire-if-same-verdict rule

Every Exp 1424 carry-forward task shall be mapped to one or more concrete
`.110` experiments or explicitly retired. The exact Exp 1419 200-case
full-scale pipeline rerun shall be forbidden unless a prior micro-gate proves
nonzero accepted repair evidence. The workflow shall confirm that
`scripts/research_conductor.py` and `research-roadmap.yaml` need no changes for
this activation audit.

The final JSON artifact shall include:

- `status`
- `prior_milestone`
- `carryforward_manifest_path`
- `carryforward_manifest_complete`
- `carryforward_task_count`
- `same_verdict_retirement_rules`
- `forbidden_exact_reruns`
- `honest_verdict`

`carryforward_task_count` shall equal the number of Exp 1424 unresolved
carry-forward tracks mapped in the manifest. `same_verdict_retirement_rules`
shall preserve the exact prior verdicts from the source artifacts. A complete
artifact shall not claim activation success if any carry-forward task lacks a
`.110` mapping or retirement rule.

### REQ-REPORT-034: Exp1426 Test Suite Remaining Debt Cluster Map

The Exp 1426 test-suite debt mapping workflow shall read Exp 1421's terminal
artifact, confirm current Python collection and spec-coverage state with narrow
commands, and write
`results/experiment_1426_test_suite_remaining_debt_cluster_map.json` without
rerunning the 30+ minute full Python suite unless cheaper signals are
unavailable.

The final JSON artifact shall include:

- `status`
- `failure_cluster_map_complete`
- `collection_clean_confirmed`
- `failure_clusters_identified`
- `next_cluster_recommended`
- `spec_coverage_debt_count`
- `commands_run`
- `honest_verdict`

The workflow shall group the remaining debt into actionable clusters with
representative test paths, likely ownership, evidence source, and a bounded fix
hint. It shall exclude the Exp 1421 embedding-store cluster from future
recommendations unless fresh evidence shows regression. The workflow shall
recommend exactly one next cluster and shall not claim full-suite health unless
the full-suite command actually passes.

### SCENARIO-REPORT-034: Collection Clean, Spec Coverage Red, Full Suite Not Rerun

Given Exp 1421 reports a clean collection, a fixed embedding-store cluster, a
red full-suite execution with remaining categories, and pre-existing
spec-coverage debt, when Exp 1426 confirms current collection with
`pytest --collect-only` and current spec coverage with
`scripts/check_spec_coverage.py`, then the artifact records
`collection_clean_confirmed == true`, `spec_coverage_debt_count` equal to the
checker count, multiple named failure clusters, exactly one
`next_cluster_recommended`, and an honest verdict that the full suite was not
rerun and remains unproven.

### REQ-REPORT-035: Milestone .110 Terminal Retrospective

The Exp 1438 `.110` milestone retrospective workflow shall read
`openspec/change-proposals/research-roadmap-vNEXT.md`,
`research-roadmap.yaml`, the conductor log, and every available Exp 1425
through Exp 1437 result artifact without rerunning the source experiments. It
shall write `results/experiment_1438_milestone_110_retro.json` with:

- `status` set to `complete` in the final artifact
- `milestone` set to `2026.04.110`
- `criteria_total` set to `14`
- `criteria_met`, derived only from success criteria whose status is `met`
- `success_criteria_results`, mapping all 14 roadmap criteria to `met`,
  `not_met`, `blocked`, or `not_run`
- `repair_v2_verdict`
- `dvi_fr11_verdict`
- `prm_verdict`
- `hardware_verdict`
- `carry_forward_tasks`, each with prior failure verdicts and
  retire-if-same-verdict rules
- `retired_exact_scopes`
- `honest_verdict`

The workflow shall count the retrospective self-criterion as met only in the
final artifact. It shall preserve honest negative evidence: a non-headline
FR-11 run with no positive promoted growth shall not count as continuous
self-learning success, and an RTL artifact blocked by missing source shall not
count as lint/simulation evidence even when tool probes are available.

### SCENARIO-REPORT-035: .110 Retro Scores Positive and Negative Evidence

Given Exp 1425 through Exp 1437 artifacts include repair v2, candidate search,
pipeline micro-validation, DVI v3 deployment, completed PRM labels, DPO
reranker relabeling, and anchored latent repair evidence, but also include an
FR-11 non-headline no-growth result and a blocked missing-RTL-source hardware
artifact, when the Exp 1438 workflow writes the terminal retrospective, then
`criteria_total == 14`, `criteria_met == 12`, the FR-11 criterion is `not_met`,
the RTL evidence criterion is `blocked`, and carry-forward rules include the
same-verdict retirement decisions for repair, FR-11 growth, test debt, and
hardware source implementation.

### REQ-REPORT-036: Milestone .111 Carry-Forward Activation Manifest

The Exp 1439 `.111` carry-forward activation workflow shall write
`results/experiment_1439_110_carryforward_activation_manifest.json` with
`status="in_progress"` before reading source evidence. It shall then read the
terminal Exp 1438 `.110` retrospective, the unresolved upstream source
artifacts named by the retro, and
`openspec/change-proposals/research-roadmap-vNEXT.md` without rerunning or
editing prior experiments. It shall write
`ops/milestone_111_carryforward_manifest.md` and a terminal Exp 1439 artifact.

The markdown manifest shall include a table with columns for:

- track
- prior evidence
- `.111` task
- gate rule
- retire-if-same-verdict rule

Every unresolved `.110` carry-forward track shall map to one or more concrete
`.111` experiments or an explicit non-headline retirement rule. The workflow
shall forbid exact reruns of prototype-only repair scale-up, FR-11 zero-growth,
PRM v1 no-improvement selector, and missing-source RTL lint/simulation paths.
It shall confirm that `scripts/research_conductor.py` and
`research-roadmap.yaml` need no changes for this activation manifest.

The final JSON artifact shall include:

- `status`
- `prior_milestone`
- `carryforward_manifest_path`
- `carryforward_manifest_complete`
- `carryforward_task_count`
- `same_verdict_retirement_rules`
- `forbidden_exact_reruns`
- `honest_verdict`

`carryforward_task_count` shall equal the number of unresolved `.110` tracks
mapped in the manifest. `same_verdict_retirement_rules` shall preserve the
exact prior verdicts from the source artifacts when they exist, otherwise from
the Exp 1438 retrospective. A complete artifact shall not claim activation
success if any unresolved track lacks a `.111` mapping, gate rule, or
same-verdict retirement rule.

### SCENARIO-REPORT-036: Exp 1439 Activates .110 Carry-Forward Work

Given Exp 1438 reports `.110` as `12/14` with carry-forward tracks for
live-SOTA repair provenance, FR-11 positive growth, spec-coverage metadata
debt, DPO provenance limits, PRM selector no-improvement, and missing Discrete
SB RTL source, when Exp 1439 runs for run date `20260506`, then it writes all
required REQ-REPORT-036 fields, maps every unresolved track to a `.111`
experiment or explicit non-headline retirement rule, lists prototype repair,
FR-11 zero-growth, PRM v1 no-improvement, and missing-source RTL lint/sim in
`forbidden_exact_reruns`, and reports no activation-manifest changes needed for
`scripts/research_conductor.py` or `research-roadmap.yaml`.

### REQ-REPORT-037: Exp1440 Spec-Coverage Traceability Metadata Fix

The Exp 1440 workflow shall fix the bounded
`spec_coverage_traceability_metadata` cluster identified by Exp 1426 without
changing runtime implementation behavior unless existing valid OpenSpec
metadata cannot otherwise be recognized by the spec-coverage checker.

The workflow shall write
`results/experiment_1440_spec_coverage_traceability_metadata_fix.json` with
`status="in_progress"` before editing traceability metadata, then record:

- `status`
- `spec_coverage_metadata_cluster_fixed`
- `initial_spec_coverage_debt_count`
- `final_spec_coverage_debt_count`
- `files_changed`
- `commands_run`
- `residual_blockers`
- `honest_verdict`

The checker shall recognize canonical OpenSpec identifiers that contain
multiple uppercase name segments before the numeric suffix, such as
`REQ-INFER-SOTA-004`, `REQ-VER-MATH-001`,
`SCENARIO-INFER-SOTA-004-001`, and `SCENARIO-VER-MATH-001`. Metadata fixes
shall cite requirements or scenarios that are present in
`openspec/capabilities/*/spec.md`.

### SCENARIO-REPORT-037: Exp1440 Closes Current Metadata Debt Cluster

Given Exp 1426 reports 71 spec-coverage metadata misses across the bounded
cluster, when Exp 1440 runs for run date `20260506`, then the artifact records
the initial debt count, applies only traceability/checker fixes needed for
existing OpenSpec anchors, reruns spec coverage, records the final debt count,
and reports any remaining blocker exactly instead of claiming full-suite health
without evidence.

### REQ-REPORT-038: Milestone .111 Terminal Retrospective

The Exp 1452 `.111` milestone retrospective workflow shall write
`results/experiment_1452_milestone_111_retro.json` with `status="in_progress"`
before scoring source evidence. It shall then read the `.111` roadmap criteria
from `openspec/change-proposals/research-roadmap-vNEXT.md`, the active
`research-roadmap.yaml`, every available Exp 1439 through Exp 1451 source
artifact, and the named-but-missing `research-roadmap-next.yaml` input without
rerunning source experiments.

The final artifact shall include:

- `status`
- `milestone`
- `criteria_total`
- `criteria_met`
- `successful_tasks`
- `blocked_tasks`
- `retired_variants`
- `carry_forward_tracks`
- `ops_docs_updated`
- `honest_verdict`

It shall also include a per-criterion evidence map that scores each of the 14
roadmap success criteria as `met`, `unmet`, or
`gate_blocked_with_evidence`. Only `met` criteria shall contribute to
`criteria_met`. Missing artifacts shall be listed explicitly, and missing
gated artifacts shall only be classified as `gate_blocked_with_evidence` when
an upstream source artifact records the failed gate or exact blocker. The
workflow shall not count a runtime gate block, a missing gated downstream
artifact, or a blocked reranker artifact as a success.

Carry-forward rules shall preserve exact prior verdicts for unresolved `.111`
tracks. Same-verdict variants shall be retired when they repeat prototype-only
repair scale-up, no-live-SOTA runtime, FR-11 zero-growth, PRM v1/v3
no-improvement selector, unsupported DPO headline wording, or missing-source
hardware lint/simulation behavior without a changed prerequisite.

### SCENARIO-REPORT-038: .111 Retro Separates Successes from Gate Blocks

Given Exp 1439 through Exp 1451 include completed carry-forward, spec-coverage,
Discrete SB source, FR-11 diagnosis, FR-11 positive growth, PRM no-improvement,
LTLZinc, EBT, and RTL lint/simulation evidence, but Exp 1442 records
`local_sota_runtime_ready == false`, Exp 1443 and Exp 1445 artifacts are
missing, and Exp 1444 records `blocked_gate_check_failed`, when Exp 1452 writes
the terminal retrospective for run date `20260507`, then
`criteria_total == 14`, runtime-dependent repair criteria are reported as
`gate_blocked_with_evidence`, missing artifacts are listed explicitly,
`criteria_met` counts only true successes, and the artifact records exact
carry-forward rules for `.112` planning.

### REQ-REPORT-039: Milestone .112 Scope-Reduction Activation Manifest

The Exp 1453 `.112` scope-reduction activation workflow shall write
`results/experiment_1453_112_scope_reduction_activation_manifest.json` with
`status="in_progress"` before final manifest completion. It shall then read the
terminal Exp 1452 `.111` retrospective, the active `ops/known-issues.md`
scope-reduction directive, `openspec/change-proposals/research-roadmap-vNEXT.md`,
`research-roadmap-next.yaml` when present, and `ops/exclusion_manifest.yaml`
without modifying `scripts/research_conductor.py` or `research-roadmap.yaml`.

The workflow shall write `ops/milestone_112_scope_reduction_manifest.md` with a
table containing:

- requirement
- mapped task id
- deliverable path
- acceptance field
- retire/block rule

Every mandatory scope-reduction requirement from the known-issues directive
shall map to a concrete `.112` task. The final artifact shall include:

- `status`
- `milestone`
- `prior_milestone`
- `scope_reduction_required`
- `required_scope_reduction_task_count`
- `planned_scope_reduction_task_count`
- `scope_reduction_manifest_path`
- `planned_scope_task_ids`
- `carryforward_from_111`
- `forbidden_exact_expansions`
- `honest_verdict`

`required_scope_reduction_task_count` shall be at least 8, and
`planned_scope_reduction_task_count` shall count only tasks whose primary
deliverable reduces active scope, retires or consolidates a noisy lineage,
narrows claims, or blocks future variant expansion. The workflow shall record
the `.111` live-SOTA runtime carry-forward rules before any repair-v3,
energy-reranker, or 100-case scale-up rerun is allowed. It shall explicitly
forbid exact noise-line expansion during `.112`, including new GRPO v15 work,
new WOPR puzzle cartridges, new HardNet++/DSP variants, and broad new
comparator or hardware branches before the relevant narrowing/audit task lands.

### SCENARIO-REPORT-039: Exp 1453 Activates .112 Scope Reduction

Given Exp 1452 reports `.111` as 10 of 14 criteria met, names the live-SOTA
runtime blocker, and `ops/known-issues.md` requires at least 8 scope-reduction
tasks, when Exp 1453 runs for run date `20260507`, then it writes all required
REQ-REPORT-039 fields, maps every mandatory scope item to `.112` tasks, records
at least 10 planned scope-reduction task ids, carries forward the live-SOTA
repair runtime blocker, forbids exact noise-line expansions, writes the
operator markdown manifest, and reports `scripts/research_conductor.py` and
`research-roadmap.yaml` as unchanged by the activation workflow.

### REQ-REPORT-040: Experiment Artifact Signal/Noise Classifier

The Exp 1454 artifact classifier shall write
`results/experiment_1454_experiment_artifact_signal_noise_classifier.json` with
`status="in_progress"` before scanning source artifacts. It shall then scan
every `results/experiment_*.json` file, including the Exp 1454 artifact itself,
and write a deterministic CSV ledger at
`ops/experiment_signal_noise_classification.csv` with one row per scanned
artifact.

Each ledger row shall include the experiment id, source path, title when
available, status, honest verdict, headline-related fields, gate or blocker
fields, retirement fields, key metric fields, a conservative
`SIGNAL` / `NOISE` / `AMBIGUOUS` classification, and a transparent reason.

The classifier shall treat explicit headline eligibility, live verified
improvement, and completed positive milestone evidence as `SIGNAL`. It shall
treat explicit retirements, no-improvement verdicts, negative regressions,
not-viable findings, and failed merit gates as `NOISE`. It shall keep missing
tools, live-runtime blockers, missing upstream gated artifacts, environmental
preconditions, malformed artifacts, in-progress artifacts, and otherwise
uncertain records as `AMBIGUOUS` unless a separate retirement field provides a
specific reason for noise classification.

The workflow shall also write `ops/experiment_signal_noise_summary.md` with
counts, top 50 noise candidates, top signal candidates, and ambiguous
operator-decision items. The terminal Exp 1454 artifact shall include:

- `status`
- `artifacts_scanned`
- `classification_table_path`
- `summary_path`
- `signal_count`
- `noise_count`
- `ambiguous_count`
- `top_50_noise_candidates`
- `heuristic_version`
- `honest_verdict`

### SCENARIO-REPORT-040: Exp 1454 Produces Conservative Scope Ledger

Given result artifacts include headline-eligible successes, explicit
retirements, no-improvement verdicts, live-runtime blockers, missing-tool
blockers, and in-progress artifacts, when Exp 1454 runs for run date
`20260507`, then it writes all required REQ-REPORT-040 fields, scans every
`results/experiment_*.json` file, writes the CSV and markdown summary paths,
counts SIGNAL / NOISE / AMBIGUOUS rows from the CSV, lists no more than 50 top
noise candidates, and keeps environmental blockers AMBIGUOUS rather than
calling them scientific NOISE.

### REQ-REPORT-041: Known-Issues Mandatory Priority Audit

The Exp 1455 known-issues mandatory priority audit shall write
`results/experiment_1455_known_issues_mandatory_priority_audit.json` with
`status="in_progress"` before terminal audit completion. It shall then read
`ops/known-issues.md`, the `.112` scope-reduction manifest, the Exp 1454
signal/noise summary, and the `.112` roadmap proposal without modifying
`scripts/research_conductor.py`.

The audit shall parse the active `MANDATORY-NEXT-MILESTONE PRIORITIES` block,
write `ops/mandatory_priority_audit.md` with one row per active priority, and
assign each row exactly one status from `keep`, `consolidate`, `superseded`,
`parked`, or `retire`. The workflow shall preserve historical known-issues
text, adding a current-active-priorities index instead of deleting the older
audit record.

The terminal artifact shall include:

- `status`
- `initial_priority_count`
- `active_priority_count`
- `trim_fraction`
- `priority_audit_path`
- `known_issues_updated`
- `active_priorities_index_path`
- `retired_or_consolidated_priorities`
- `honest_verdict`

`active_priority_count` shall be no greater than 10, and `trim_fraction` shall
be at least 0.40. The `retired_or_consolidated_priorities` value shall list all
rows whose status is `consolidate`, `superseded`, or `retire`.

### SCENARIO-REPORT-041: Exp 1455 Trims Active Mandatory Priorities

Given the active known-issues mandatory block contains more than 10 current
priorities and includes superseded, parked, and consolidation candidates, when
Exp 1455 runs for run date `20260507`, then it writes the audit table, writes a
current active priority index with no more than 10 items, preserves historical
entries, updates known-issues with a pointer to the audit/index, records all
required REQ-REPORT-041 artifact fields, and reports a trim fraction of at least
0.40.

### REQ-REPORT-042: GRPO/VPRM Lineage Consolidation and Retirement

The Exp 1456 GRPO/VPRM lineage-retirement workflow shall write
`results/experiment_1456_grpo_vprm_lineage_consolidation_retirement.json` with
`status="in_progress"` before terminal consolidation. It shall then review the
GRPO/VPRM-related records from roughly Exp 1063 through Exp 1393 using
`research-complete.yaml`, `ops/experiment_signal_noise_classification.csv`, the
terminal milestone retrospectives, and available `results/experiment_*.json`
artifacts.

The workflow shall write
`ops/lineage-retirements/grpo_vprm_lineage_retired.md` with the reviewed
experiment ids, verdicts, measured positives, repeated blockers, retained
lessons, reopen conditions, and the final retirement decision. It shall update
the exclusion manifest or active exclusion mechanism so future GRPO v15 or
VPRM v15 variant proposals are blocked unless an operator explicitly reopens
the scope with a new root cause and falsifiable acceptance gate.

The terminal artifact shall include:

- `status`
- `lineage_name`
- `experiments_reviewed`
- `consolidation_note_path`
- `grpo_lineage_retired`
- `exclusion_manifest_updated`
- `lessons_retained`
- `future_reopen_conditions`
- `honest_verdict`

The workflow shall preserve useful lessons such as false-negative correction,
candidate-pool saturation, and step-level process supervision instead of
deleting them. It shall retire the lineage as active research scope when the
review shows repeated no-improvement, gate-blocked, smoke-only, missing, or
non-headline outcomes after the earlier positive v1-v4/VPS results.

### SCENARIO-REPORT-042: Exp 1456 Retires GRPO/VPRM Scope

Given the .112 scope-reduction directive identifies GRPO/VPRM v1-v14 as a noisy
lineage, and the evidence record contains early positive GRPO/VPS results,
TinyV regression evidence, repeated SOTA/DVI gate blocks, smoke-only non-headline
results, and final JURY-RL/NGRPO zero-improvement evidence, when Exp 1456 runs
for run date `20260507`, then it writes all required REQ-REPORT-042 fields,
lists at least 14 reviewed experiments, writes the markdown consolidation note,
records useful retained lessons, updates the exclusion manifest with a GRPO v15
/ VPRM v15 block, sets `grpo_lineage_retired == true`, and reports an honest
retirement verdict.

### REQ-REPORT-024: Local Agent Usage Snapshot

The repository shall provide a local operator workflow that inspects the
machine-readable session logs for Codex and Claude and emits a combined usage
snapshot without requiring web scraping.

The workflow shall:

- read the latest Codex `token_count` event from `~/.codex/sessions/**/*.jsonl`
  and surface:
  - `plan_type`
  - primary and secondary rate-limit windows, usage percentages, and reset
    epochs when present
  - total and last token-usage summaries when present
- read Claude session logs from `~/.claude/projects/**/*.jsonl` and aggregate:
  - `input_tokens`
  - `output_tokens`
  - `cache_creation_input_tokens`
  - `cache_read_input_tokens`
  - without double-counting repeated log entries for the same assistant message
- read Claude subscription metadata from `~/.claude/.credentials.json` and
  surface only non-secret plan fields such as `subscriptionType` and
  `rateLimitTier`
- when the operator explicitly requests live Claude usage, issue an
  authenticated `GET https://api.anthropic.com/api/oauth/usage` call using the
  local Claude OAuth access token and surface structured windows such as
  `five_hour`, `seven_day`, `seven_day_sonnet`, and `extra_usage`
- when live Claude usage is available, set the top-level Claude
  `used_percent` and `reset_at` fields from the exact `seven_day` window
- when the Claude logs contain a structured, command-scoped quota percentage
  tied to `/usage` or an equivalent local usage event, the workflow may surface
  the latest reported value as `used_percent`
- free-form Claude assistant prose shall not be interpreted as quota telemetry
- report exact unavailability honestly: if a provider does not expose a local
  percentage-used or reset field, the workflow shall emit `null` plus an
  `unavailable` note instead of guessing
- support both machine-readable JSON output and a human-readable table output

The workflow shall never echo access tokens, refresh tokens, raw credential
payloads, or other secret fields.

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

### SCENARIO-REPORT-020: Exp 1281 Counts .99 Source Criteria

**Given** Exp 1268 through Exp 1280 source artifacts contain the current .99
criterion evidence
**And** Exp 1271 is a blocked terminal artifact without SOTA model IDs or a
certificate parse rate
**And** Exp 1277 is absent because its Exp 1271 parse-rate gate was unmet
**When** the Exp 1281 retrospective workflow runs
**Then** blocked, gated, missing, false, stale, and threshold-miss source
criteria do not increment `criteria_met`
**And** the retrospective self-criterion counts as met in the final artifact
**And** the artifact reports `criteria_met == 12`
**And** `honest_verdict == "milestone_99_12_of_14_criteria_met"`

### SCENARIO-REPORT-025: Exp 1295 Counts .100 Source Criteria

**Given** Exp 1282 through Exp 1294 source artifacts contain the current .100
criterion evidence
**And** Exp 1282 is a blocked conductor pre-gate artifact without SOTA cache
readiness fields
**And** Exp 1284, Exp 1285, Exp 1287, and Exp 1289 are absent because upstream
SOTA certificate gates were not met
**And** Exp 1290 is absent even though Exp 1288 wrote `memory_update_written`
**When** the Exp 1295 retrospective workflow runs
**Then** blocked, gated, missing, false, stale, and threshold-miss source
criteria do not increment `criteria_met`
**And** the retrospective self-criterion counts as met in the final artifact
**And** the artifact reports `criteria_met == 5`
**And** `honest_verdict == "milestone_100_5_of_14_criteria_met"`

### SCENARIO-REPORT-026: Exp 1306 Completes Energy Bridge Audit v2

**Given** Exp 1293 is blocked by missing prior-failure metadata
**And** the local research references include EBT citation signals, ARM-EBM,
EBM-CoT, FALCON, Extropic TSU, p-bit update-dynamics, and Kona context
**When** the Exp 1306 bridge-audit workflow runs for run date `20260505`
**Then** it writes the required REQ-REPORT-026 fields
**And** `energy_bridge_completed == true`
**And** Extropic, p-bit, and Kona are recorded as future sampler or strategy
context rather than local implementation dependencies
**And** `honest_verdict` distinguishes local verifier-energy alignment from
strategic architecture context.

### SCENARIO-REPORT-027: Exp 1308 Counts .101 Gates Separately

**Given** Exp 1296 through Exp 1307 source artifacts contain the current .101
criterion evidence
**And** Exp 1297 records `cached_sota_ready == false` with exact missing model
blockers
**And** Exp 1298 and Exp 1300 are conductor pre-gate artifacts blocked by unmet
milestone dependencies
**And** Exp 1299, Exp 1301, and Exp 1304 are absent because the SOTA certificate
gates did not open
**When** the Exp 1308 retrospective workflow runs
**Then** cache/provenance blocker reporting counts as `MET`
**And** closed downstream SOTA tasks count as `GATED`, not failed science
**And** continuous self-learning, repair policy, bridge audit, publication hold,
and the retrospective self-criterion count from their terminal source fields
**And** the artifact reports `criteria_met == 8`
**And** `honest_verdict == "milestone_101_8_of_13_criteria_met"`

### SCENARIO-REPORT-028: Exp 1322 Counts .102 Gates And Partials Separately

**Given** Exp 1309 through Exp 1321 source artifacts contain the current .102
criterion evidence
**And** Exp 1312 reports `certificate_parse_rate == 0.71223`, below the
semantic-validator and DVI gate threshold of `0.75`
**And** Exp 1313 and Exp 1316 are conductor pre-gate blocked by that parse-rate
miss
**And** Exp 1314 is absent because the certificate-validator gates did not open
**When** the Exp 1322 retrospective workflow runs
**Then** the runtime, answer-stability, continuous self-learning, repair,
hardware, publication, and self-retro criteria that have source evidence count
as `MET`
**And** the gated semantic-validator, safe-prefix, and DVI criteria do not
increment `criteria_met`
**And** `certificate_path_headline_ready == false`
**And** `hardware_claims_honest == true`
**And** the artifact reports `criteria_met == 11`
**And** `honest_verdict == "milestone_102_11_of_14_criteria_met"`

### SCENARIO-REPORT-029: Exp 1350 Reconciles .104 Carry-Forward State

**Given** the .104 roadmap criteria and Exp 1337 through Exp 1349 source
artifacts are available, with gated or missing artifacts recorded honestly
**When** the Exp 1350 retrospective workflow runs for run date `20260505`
**Then** it writes the .104 carry-forward artifact with all required
REQ-REPORT-029 fields
**And** `criteria_total` matches the roadmap success-criteria count
**And** `criteria_met` counts only observed terminal evidence or criteria that
explicitly required a gate to stay closed
**And** the certificate, self-learning, hardware, publication-hold, and prior
failure hygiene verdicts stay inside the source evidence
**And** missing roadmap files or experiment artifacts are reported explicitly
instead of being inferred as successes

### SCENARIO-REPORT-030: Exp 1351 Carries Missing Exp 1340 Forward

**Given** Exp 1337 through Exp 1350 source artifacts are checked
**And** no terminal Exp 1340 certificate artifact exists
**When** the Exp 1351 carry-forward integrity audit runs for run date `20260505`
**Then** it writes all required REQ-REPORT-030 fields
**And** `missing_artifacts` explicitly includes Exp 1340
**And** `terminal_certificate_required == true`
**And** semantic-validator, scheduler, DVI, and GRPO gates remain closed
**And** `prior_failure_requirements` names the prior failure context that `.105`
tasks must cite before retrying gated work.

### SCENARIO-REPORT-031: Exp 1363 Closes .105 Without Overclaiming

**Given** Exp 1351 through Exp 1362 source artifacts are checked
**And** Exp 1353 produced terminal local SOTA certificate rows with
`certificate_parse_rate == 0.0`
**And** semantic repair, DVI, and GRPO work is blocked or missing behind gates
**When** the Exp 1363 retrospective workflow runs for run date `20260505`
**Then** it writes all required REQ-REPORT-031 fields
**And** `criteria_total` matches the `.105` success-criteria count
**And** `criteria_met` counts only observed terminal evidence, mandatory
self-learning evidence, no-hardware mapping evidence, publication-boundary
evidence, and gate-discipline criteria
**And** blocked semantic, DVI, and GRPO artifacts are not reported as successful
semantic repair or policy-update evidence
**And** missing Exp 1356 and Exp 1359 artifacts are listed explicitly
**And** `carry_forward_tasks` names `.106` work with prior-failure hygiene.

### SCENARIO-REPORT-033: Exp 1425 Activates .109 Carry-Forward Work

**Given** Exp 1424 reports `.109` as `10/13` with carry-forward tasks for
repair executor v2, DVI v3 nonforgetting, FR-11 v6, DPO provenance, test debt,
and PRM label completion
**And** the source artifacts preserve exact prior verdicts for Exp 1414, Exp
1415, Exp 1419, Exp 1420, Exp 1421, and Exp 1423
**When** the Exp 1425 activation audit runs for run date `20260506`
**Then** it writes all required REQ-REPORT-033 fields
**And** `ops/milestone_110_carryforward_manifest.md` maps every unresolved
track to a `.110` experiment or an explicit retirement condition
**And** the exact Exp 1419 200-case rerun without nonzero repair evidence is
listed in `forbidden_exact_reruns`
**And** `scripts/research_conductor.py` and `research-roadmap.yaml` are
reported as requiring no activation-audit changes.

### SCENARIO-REPORT-021: Codex Latest Rate-Limit Event Is Surfaced

**Given** a local Codex session tree contains multiple `token_count` events
**And** a newer event includes `plan_type`, primary and secondary rate limits,
and token totals
**When** the local agent-usage workflow runs
**Then** the Codex section reports the newest event only
**And** it exposes the primary and secondary `used_percent`, `window_minutes`,
and `resets_at` fields
**And** it carries forward the latest token totals without fabricating missing
fields

### SCENARIO-REPORT-022: Claude Token Totals Are Aggregated Without Secret Leakage

**Given** local Claude project logs contain multiple assistant messages with
usage payloads
**And** at least one assistant message is repeated in the logs with the same
`sessionId` and `message.id`
**And** `~/.claude/.credentials.json` contains access tokens plus
`subscriptionType` and `rateLimitTier`
**When** the local agent-usage workflow runs
**Then** the Claude section aggregates the token totals across the logs
**And** repeated entries for the same assistant message count once
**And** it reports only `subscription_type` and `rate_limit_tier` from the
credentials file
**And** the output omits access tokens, refresh tokens, and raw credential
objects

### SCENARIO-REPORT-023: Missing Claude Percent Usage Stays Unavailable

**Given** the local Claude logs contain token-usage payloads but no
machine-readable percentage-used field
**When** the local agent-usage workflow runs
**Then** the Claude section reports `used_percent == null`
**And** the human-readable table prints `unavailable` for the Claude plan-usage
cell rather than guessing a percentage

### SCENARIO-REPORT-024: Free-Form Claude Quota Prose Is Ignored

**Given** the local Claude logs contain assistant text mentioning quota usage
in ordinary prose
**And** that text is not attached to a structured `/usage` or equivalent local
usage event
**When** the local agent-usage workflow runs
**Then** the Claude section does not treat that prose as `used_percent`
**And** it reports `used_percent == null` unless a structured usage field is
available elsewhere

### SCENARIO-REPORT-025: Live Claude OAuth Usage Overrides Local Guesswork

**Given** the operator enables the live Claude usage mode
**And** the local Claude OAuth credentials can authenticate
`GET https://api.anthropic.com/api/oauth/usage`
**And** the endpoint returns structured `five_hour` and `seven_day` usage
windows
**When** the local agent-usage workflow runs
**Then** the Claude section reports `used_percent` from the exact
`seven_day.utilization` field
**And** it reports `reset_at` from `seven_day.resets_at`
**And** it surfaces the live usage windows without echoing the OAuth access
token or refresh token


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
| REQ-REPORT-023 | `python/carnot/reporting/milestone_retro_99.py`, `results/experiment_1281_milestone_retro_99.json` | `tests/python/test_milestone_retro_99.py` | Implemented |
| REQ-REPORT-025 | `python/carnot/reporting/milestone_retro_100.py`, `results/experiment_1295_milestone_retro_100.json` | `tests/python/test_milestone_retro_100.py` | Implemented |
| REQ-REPORT-026 | `python/carnot/reporting/energy_bridge_audit_v2.py`, `results/experiment_1306_ebt_arm_ebm_cot_energy_bridge_audit_v2.json` | `tests/python/test_energy_bridge_audit_v2.py` | Planned |
| REQ-REPORT-027 | `python/carnot/reporting/milestone_retro_101.py`, `results/experiment_1308_milestone_retro_101.json` | `tests/python/test_milestone_retro_101.py` | Implemented |
| REQ-REPORT-028 | `python/carnot/reporting/milestone_retro_102.py`, `results/experiment_1322_milestone_retro_102.json` | `tests/python/test_milestone_retro_102.py` | Implemented |
| REQ-REPORT-029 | `python/carnot/reporting/milestone_retro_104.py`, `results/experiment_1350_milestone_104_retro_carryforward.json` | `tests/python/test_milestone_retro_104.py` | Implemented |
| REQ-REPORT-030 | `python/carnot/reporting/carryforward_integrity_audit_104.py`, `results/experiment_1351_104_carryforward_artifact_integrity_audit.json` | `tests/python/test_carryforward_integrity_audit_104.py` | Implemented |
| REQ-REPORT-031 | `python/carnot/reporting/milestone_retro_105.py`, `results/experiment_1363_milestone_105_retro_carryforward.json` | `tests/python/test_milestone_retro_105.py` | Implemented |
| REQ-REPORT-032 | `python/carnot/reporting/milestone_retro_109.py`, `results/experiment_1424_milestone_109_retro.json` | `tests/python/test_milestone_retro_109.py` | Implemented |
| REQ-REPORT-033 | `python/carnot/reporting/milestone_110_carryforward_activation_audit.py`, `results/experiment_1425_109_carryforward_activation_audit.json`, `ops/milestone_110_carryforward_manifest.md` | `tests/python/test_milestone_110_carryforward_activation_audit.py` | Implemented |
| REQ-REPORT-034 | `python/carnot/reporting/test_suite_remaining_debt_cluster_map.py`, `results/experiment_1426_test_suite_remaining_debt_cluster_map.json` | `tests/python/test_test_suite_remaining_debt_cluster_map.py` | Implemented |
| REQ-REPORT-035 | `python/carnot/reporting/milestone_retro_110.py`, `results/experiment_1438_milestone_110_retro.json` | `tests/python/test_milestone_retro_110.py` | Implemented |
| REQ-REPORT-036 | `python/carnot/reporting/milestone_111_carryforward_activation_manifest.py`, `results/experiment_1439_110_carryforward_activation_manifest.json`, `ops/milestone_111_carryforward_manifest.md` | `tests/python/test_milestone_111_carryforward_activation_manifest.py` | Implemented |
| REQ-REPORT-037 | `scripts/check_spec_coverage.py`, `results/experiment_1440_spec_coverage_traceability_metadata_fix.json` | `tests/python/test_spec_coverage_checker.py` | Implemented |
| REQ-REPORT-038 | `python/carnot/reporting/milestone_retro_111.py`, `results/experiment_1452_milestone_111_retro.json` | `tests/python/test_milestone_retro_111.py` | Implemented |
| REQ-REPORT-039 | `python/carnot/reporting/milestone_112_scope_reduction_activation_manifest.py`, `results/experiment_1453_112_scope_reduction_activation_manifest.json`, `ops/milestone_112_scope_reduction_manifest.md` | `tests/python/test_milestone_112_scope_reduction_activation_manifest.py` | Implemented |
| REQ-REPORT-040 | `python/carnot/reporting/experiment_artifact_signal_noise_classifier.py`, `results/experiment_1454_experiment_artifact_signal_noise_classifier.json`, `ops/experiment_signal_noise_classification.csv`, `ops/experiment_signal_noise_summary.md` | `tests/python/test_experiment_artifact_signal_noise_classifier.py` | Implemented |
| REQ-REPORT-041 | `python/carnot/reporting/known_issues_mandatory_priority_audit.py`, `results/experiment_1455_known_issues_mandatory_priority_audit.json`, `ops/mandatory_priority_audit.md`, `ops/active-priorities.md` | `tests/python/test_known_issues_mandatory_priority_audit.py` | Implemented |
| REQ-REPORT-042 | `python/carnot/reporting/grpo_vprm_lineage_retirement.py`, `results/experiment_1456_grpo_vprm_lineage_consolidation_retirement.json`, `ops/lineage-retirements/grpo_vprm_lineage_retired.md`, `ops/exclusion_manifest.yaml` | `tests/python/test_grpo_vprm_lineage_retirement.py` | Implemented |
| REQ-REPORT-024 | `python/carnot/reporting/agent_usage.py`, `scripts/agent_plan_usage.py` | `tests/python/test_agent_plan_usage.py` | Implemented |
| REQ-PUBLISH-003 | `scripts/experiment_317_hf_publish.py` | `tests/python/test_experiment_317_hf_publish.py` | Implemented |
| REQ-PUBLISH-004 | `scripts/experiment_330_hf_live_publish.py` | `tests/python/test_experiment_330_hf_live_publish.py` | Implemented |
