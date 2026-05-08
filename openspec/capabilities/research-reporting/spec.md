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

### REQ-REPORT-043: WOPR Puzzle Cartridge Lineage Retirement

The Exp 1457 WOPR puzzle-cartridge retirement workflow shall write
`results/experiment_1457_wopr_puzzle_cartridge_retirement.json` with
`status="in_progress"` before terminal consolidation. It shall then review the
WOPR puzzle and gallery records named by the scope-reduction directive,
including Slitherlink, Connect Four, Hex, Nonogram, Futoshiki, Kakuro, Masyu,
and related gallery deploy/update tasks, using `research-complete.yaml`,
`ops/known-issues.md`, `ops/experiment_signal_noise_classification.csv`,
available `results/experiment_*.json` artifacts, and public docs references.

The workflow shall write
`ops/lineage-retirements/wopr_puzzle_cartridges_retired.md` with the reviewed
experiment ids, shipped or blocked outcomes, demo assets that remain preserved,
known-issues id discrepancies, reopen conditions, and the final retirement
decision. It shall update `ops/exclusion_manifest.yaml` or the active exclusion
mechanism with a planner-visible block for future WOPR puzzle-cartridge/gallery
research tasks unless an operator explicitly reopens gallery work with a new
verify-repair or Phase-3 substrate thesis link and falsifiable acceptance gate.

The terminal artifact shall include:

- `status`
- `cartridge_experiments_reviewed`
- `retirement_note_path`
- `wopr_puzzle_lineage_retired`
- `exclusion_manifest_updated`
- `preserved_assets`
- `future_reopen_conditions`
- `honest_verdict`

The workflow shall preserve working demo assets and historical documentation;
retirement here means active research-scope closure, not destructive cleanup.

### SCENARIO-REPORT-043: Exp 1457 Retires WOPR Puzzle-Cartridge Research Scope

Given the .112 scope-reduction directive identifies WOPR puzzle cartridges as a
non-thesis demo lineage, and the evidence record contains shipped E=0 cartridges,
gate-blocked precursor attempts, known-issues references, docs/gallery mentions,
and preserved demo code under `python/carnot/games` and `spaces/wopr-games`,
when Exp 1457 runs for run date `20260507`, then it writes all required
REQ-REPORT-043 fields, lists at least the Hex, Connect Four, Nonogram,
Futoshiki, Kakuro, Masyu, and Slitherlink records, writes the markdown
retirement note, records preserved assets, updates the exclusion manifest with a
future-puzzle-cartridge block, sets `wopr_puzzle_lineage_retired == true`, and
reports an honest retirement verdict.

### REQ-REPORT-044: HardNet++/DSP Repair Stack Consolidation and Retirement

The Exp 1458 HardNet++/DSP repair-stack consolidation workflow shall write
`results/experiment_1458_hardnet_dsp_repair_stack_consolidation.json` with
`status="in_progress"` before terminal consolidation. It shall then review the
HardNet++, FSNet, SnareNet, DSP feasibility-channel, conservative replay, and
learned stop-policy records using `research-complete.yaml`,
`research-references.md`, `ops/experiment_signal_noise_classification.csv`,
`ops/exclusion_manifest.yaml`, and available `results/experiment_*.json`
artifacts.

The workflow shall write
`ops/lineage-retirements/hardnet_dsp_repair_stack_retired.md` with the reviewed
experiment ids, measured positives, repeated non-headline lesson, cited recent
hard-constraint papers, future reopen conditions, and the final retirement
decision. It shall update `ops/exclusion_manifest.yaml` or the active exclusion
mechanism with a planner-visible block for future HardNet++/DSP repair variants
unless an operator explicitly reopens the line with new evidence beyond
conservative replay.

The terminal artifact shall include:

- `status`
- `hardnet_dsp_experiments_reviewed`
- `consolidation_note_path`
- `hardnet_dsp_lineage_retired`
- `exclusion_manifest_updated`
- `lessons_retained`
- `cited_recent_constraint_papers`
- `future_reopen_conditions`
- `honest_verdict`

The workflow shall preserve the hard-constraint lesson: projection/repair layers
are useful for enforcing feasibility in continuous numeric domains, but the
existing Carnot HardNet++/DSP line is retired as active headline scope because
the latest stop-policy evidence matched conservative replay and did not prove a
broad learned general rule.

### SCENARIO-REPORT-044: Exp 1458 Retires HardNet++/DSP Variant Proliferation

Given the .112 scope-reduction directive identifies HardNet++/DSP repair as a
lineage to consolidate, and the evidence record contains HardNet++ projection
success, FSNet/SnareNet feasibility improvements, marginal DSP feasibility
prediction, conservative replay stop-policy utility, and learned stop-policy
evidence that adds no delta over replay, when Exp 1458 runs for run date
`20260507`, then it writes all required REQ-REPORT-044 fields, lists the
reviewed HardNet++/DSP-related experiments, writes the markdown consolidation
note, cites HardNet++, KKT-Hardnet, SnareNet, and DSP feasibility-channel work,
updates the exclusion manifest with a future-variant block, sets
`hardnet_dsp_lineage_retired == true`, and reports an honest retirement verdict.

### REQ-REPORT-045: Comparator Integration Cite/Retire Audit

The Exp 1461 comparator-integration audit workflow shall write
`results/experiment_1461_comparator_integration_cite_retire_audit.json` with
`status="in_progress"` before terminal audit completion. It shall then review
the comparator records in `ops/known-issues.md`, `research-references.md`, and
the relevant `docs/` research notes without modifying
`scripts/research_conductor.py`.

The workflow shall write
`docs/research-notes/comparator_cite_retire_audit.md` with one row per audited
comparator. Each row shall include comparator name, one of `cite`, `retire`, or
`future_watchlist`, a one-line rationale, impacted paper section, and, for any
retired comparator, a concise future reopen condition. The comparator set shall
include at least Abstract-CoT, Meta-Harness, Autodata, LARQL, Skillify, GStack,
EBT/NRGPT, ARM-as-EBM, BEAVER, and ontology-constrained reasoning.

The workflow may update `research-references.md` only to clarify the
cite/retire/watchlist status of already-recorded comparator items; it shall not
add unrelated broad references. The terminal artifact shall include:

- `status`
- `comparator_decision_count`
- `cite_count`
- `retire_count`
- `watchlist_count`
- `decision_table_path`
- `references_updated`
- `paper_related_work_implications`
- `honest_verdict`

The counts shall match the decision rows in the markdown table, and
`honest_verdict` shall state that the audit narrowed paper-v6 comparator scope
when every comparator has an explicit decision.

### SCENARIO-REPORT-045: Exp 1461 Narrows Comparator Scope

Given the .112 scope-reduction directive names comparator integrations as
ambiguous scope and the local evidence record contains paper-v6 citation
candidates, deferred infrastructure ideas, and weakly evidenced comparator
names, when Exp 1461 runs for run date `20260507`, then it writes all required
REQ-REPORT-045 artifact fields, writes the comparator decision table, classifies
each audited comparator as cite, retire, or future_watchlist, records a future
reopen condition for every retired comparator, updates `research-references.md`
only with status clarification, and reports an honest scope-narrowing verdict.

### REQ-REPORT-046: External Verifier Benchmark Fit Audit

The Exp 1465 external verifier benchmark fit audit workflow shall write
`results/experiment_1465_external_verifier_benchmark_fit_audit.json` with
`status="in_progress"` before terminal decision work. It shall then review
VNNLIB/VNN-COMP, BEAVER-style deterministic bounds, and one smaller existing
benchmark option without implementing a new broad benchmark runner.

The workflow shall write
`docs/research-notes/external_verifier_benchmark_fit.md` with one row per
reviewed benchmark family. Each row shall include benchmark name, decision
(`adopt`, `defer`, or `retire`), rationale, fit risks, and reopen or next
condition. The terminal artifact shall include:

- `status`
- `benchmarks_reviewed`
- `benchmark_decision_table_path`
- `benchmark_adoption_decision`
- `adopted_benchmark`
- `deferred_benchmarks`
- `retired_benchmarks`
- `next_minimal_benchmark_task`
- `honest_verdict`

If any benchmark is adopted, `next_minimal_benchmark_task` shall define exactly
one future task with inputs, expected artifact fields, and an applicable
end-to-end check. If none are adopted, the artifact shall explain why all
families are deferred or retired.

### SCENARIO-REPORT-046: Exp 1465 Selects One Minimal External Verifier Task

Given the run date is `20260507` and the current comparator audit plus verifier
specifications are available, when Exp 1465 runs, then it writes all required
REQ-REPORT-046 artifact fields, writes the external verifier benchmark fit note,
adopts no more than one benchmark family for the next minimal task, and defers
or retires the remaining reviewed families with explicit reasons.

### REQ-REPORT-047: Milestone .112 Terminal Retrospective

The Exp 1466 `.112` milestone retrospective workflow shall write
`results/experiment_1466_milestone_112_retro.json` with
`status="in_progress"` before scoring source evidence. It shall then read the
`.112` roadmap success criteria from
`openspec/change-proposals/research-roadmap-vNEXT.md`, `research-roadmap.yaml`,
`research-roadmap-next.yaml` when present, `ops/conductor-log.md`, and every
available Exp 1453 through Exp 1465 source artifact without modifying
`scripts/research_conductor.py` or `research-roadmap.yaml`.

The terminal artifact shall include:

- `status`
- `milestone`
- `criteria_met`
- `criteria_total`
- `scope_reduction_required`
- `scope_reduction_tasks_completed`
- `scope_reduction_compliance_met`
- `blocked_tasks`
- `retired_lineages`
- `carry_forward_tracks`
- `missing_artifacts`
- `research_roadmap_yaml_modified`
- `scripts_research_conductor_modified`
- `ops_docs_updated`
- `honest_verdict`

The workflow shall score all 14 roadmap success criteria as `met`, `unmet`, or
`gate_blocked_with_evidence`. Only `met` criteria shall contribute to
`criteria_met`. Scope-reduction criteria shall count as `met` only when the
terminal artifact exists and the required markdown, CSV, paper, or exclusion
manifest evidence named by that artifact also exists or is explicitly recorded
as updated. Runtime and repair gates shall count as `met` only when their exact
fields pass: `exp1463.local_sota_runtime_ready == true` for runtime, and for
repair either `exp1464.acceptance_delta_pp > 0` or
`exp1464.repair_executor_lineage_retired == true` after the runtime gate is on.
Gate-blocked evidence shall be surfaced, but it shall not count as met unless
the criterion explicitly allows a blocker or retirement outcome.

The retrospective shall record all retired lineages from Exp 1456, Exp 1457,
Exp 1458, and Exp 1464, plus carry-forward rules for runtime, repair,
self-learning, paper claims, and benchmark adoption. It shall report whether
`research-roadmap.yaml` and `scripts/research_conductor.py` were modified by the
retro workflow. If an operator stop rule delegates ops-doc reconciliation to a
separate conductor pass, the artifact shall set `ops_docs_updated=false` and
explain the delegation instead of claiming an update that did not occur.

### SCENARIO-REPORT-047: .112 Retro Scores Scope Reduction and Repair Honestly

Given Exp 1453 through Exp 1465 exist and include completed scope-reduction
artifacts, lineages retired into notes and exclusion-manifest updates, Exp 1463
with `local_sota_runtime_ready == true`, Exp 1464 with
`acceptance_delta_pp == 0.0` and `repair_executor_lineage_retired == true`, and
`research-roadmap-next.yaml` is missing at retro time, when Exp 1466 runs for
run date `20260507`, then it writes all required REQ-REPORT-047 fields, reports
`criteria_total == 14`, counts only source-field and documentation-backed
criteria as met, records the completed scope-reduction task ids, lists
`research-roadmap-next.yaml` in `missing_artifacts`, records runtime, repair,
self-learning, paper-claim, and benchmark carry-forward rules, confirms
`research-roadmap.yaml` and `scripts/research_conductor.py` were not modified by
the retro workflow, and writes an honest verdict with the final score.

### REQ-REPORT-048: Milestone .113 Activation Manifest

The Exp 1467 `.113` activation workflow shall write
`results/experiment_1467_112_completion_archive_113_activation.json` with
`status="in_progress"` before terminal completion. It shall then read
`results/experiment_1466_milestone_112_retro.json`, the Exp 1453 through
Exp 1466 entries in `ops/conductor-log.md`, `research-complete.yaml`,
`research-roadmap.yaml`, `ops/exclusion_manifest.yaml`, and
`ops/active-priorities.md` without modifying `scripts/research_conductor.py` or
`research-roadmap.yaml`.

The workflow shall write `ops/milestone_113_activation_manifest.md` with the
allowed `.113` tracks:

- live SOTA telemetry
- BEAVER-lite bounds
- one self-learning pivot
- T-SKM/STATIC smokes
- KV260 RTL regression
- THRML simulation

The manifest shall explicitly forbid reopening GRPO/VPRM, WOPR puzzle
cartridges, HardNet++/DSP, validation-error repair, broad VNN-COMP runners, and
hardware execution claims unless an operator reopens the track with a new root
cause and falsifiable gate. The terminal artifact shall include:

- `status`
- `milestone`
- `predecessor_milestone`
- `criteria_met`
- `criteria_total`
- `research_complete_has_112_entry`
- `activation_manifest_complete`
- `retired_lineages_preserved`
- `allowed_113_tracks`
- `forbidden_reopen_tracks`
- `honest_verdict`

`research_complete_has_112_entry` shall reflect whether
`research-complete.yaml` already contains a `2026.04.112` archive row. If the
row is absent, the artifact shall record the archive gap rather than claiming
that `.112` is fully archived. The final verdict shall be complete only when
the `.112` retro reports all 14 criteria met, the `.112` archive row exists or
is explicitly reported as a gap, the `.113` manifest is written, every retired
lineage block is preserved, and no forbidden file modification is reported.

### SCENARIO-REPORT-048: Exp 1467 Activates .113 Without Reopening Retired Work

Given Exp 1466 reports `.112` as 14 of 14 criteria met, `research-complete.yaml`
contains a `2026.04.112` archive row, the conductor log records OK outcomes for
Exp 1453 through Exp 1466, and the exclusion manifest preserves the GRPO/VPRM,
WOPR puzzle-cartridge, and HardNet++/DSP retirement blocks, when Exp 1467 runs
for run date `20260507`, then it writes all required REQ-REPORT-048 fields,
writes the `.113` activation markdown, reports
`research_complete_has_112_entry == true`, lists only the allowed `.113`
tracks, forbids the retired reopen tracks, confirms `research-roadmap.yaml` and
`scripts/research_conductor.py` were not modified, and writes an honest
activation verdict.

### REQ-REPORT-049: Milestone .113 Terminal Retrospective

The Exp 1478 `.113` retrospective workflow shall write
`results/experiment_1478_milestone_113_retro.json` with `status="in_progress"`
before it loads source artifacts. It shall then read
`openspec/change-proposals/research-roadmap-vNEXT.md`, `research-roadmap.yaml`,
`research-roadmap-next.yaml` when present, `ops/conductor-log.md`, and the
authoritative Exp 1467 through Exp 1477 result artifacts that exist.

The workflow shall score the 12 roadmap success criteria from exact source
fields:

- Exp 1467 activation manifest completion and predecessor summary.
- Exp 1468 live SOTA inference plus recorded top-k/logprob availability.
- Exp 1469 terminal HALT/spilled-energy diagnostic completion when gated on,
  or a terminal skip explaining missing logprobs when gated off.
- Exp 1470 sound BEAVER-lite bounds with live/mock logprob provenance labeled.
- Exp 1471 positive self-learning growth with at least one promotion and
  nonforgetting at or above 0.99, or an explicit pivot retirement.
- Exp 1472 soundness and completeness mistakes plus an asymmetric-cost
  decision.
- Exp 1473 terminal telemetry-validity verdict with superficial-confound
  checks or named blockers.
- Exp 1474 zero-violation T-SKM toy projection or a recorded blocker.
- Exp 1475 exact STATIC CSR acceptance equivalence with latency reported.
- Exp 1476 source-level KV260 RTL regression completion with no board, bitfile,
  or latency claim.
- Exp 1477 no hardware claim plus simulator parity/sample-quality fields.
- Exp 1478 closure with `criteria_total == 12`, all required lineage decisions
  and carry-forward rules recorded, and both `research-roadmap.yaml` and
  `scripts/research_conductor.py` unchanged.

The terminal artifact shall include at least:

- `status`
- `milestone`
- `criteria_met`
- `criteria_total`
- `blocked_tasks`
- `retired_lineages`
- `preserved_lineages`
- `carry_forward_tracks`
- `missing_artifacts`
- `research_roadmap_yaml_modified`
- `scripts_research_conductor_modified`
- `ops_docs_updated`
- `honest_verdict`

It shall record the HALT/spilled diagnostic, self-learning pivot, T-SKM,
STATIC, KV260 regression, and THRML/NPIM tracks as either retired or preserved.
Failed conductor attempts and terminal environmental blockers shall be recorded
with their exact available reasons, but prior failed attempts shall not make a
criterion fail when the authoritative terminal artifact satisfies the roadmap
criterion. `ops_docs_updated` shall truthfully reflect whether the retrospective
workflow edited `ops/status.md` and `ops/changelog.md`; when an operator stop
rule delegates ops reconciliation to the conductor, it shall remain false with a
note rather than fabricating an update.

### SCENARIO-REPORT-049: .113 Retro Scores Terminal Evidence Without Reopening Scope

Given Exp 1467 through Exp 1477 terminal artifacts exist for run date
`20260507`, `research-roadmap-next.yaml` is absent, Exp 1469 retires the
HALT/spilled diagnostic as non-headline telemetry, Exp 1471 and Exp 1472
preserve the narrow self-learning claim, Exp 1473 blocks the telemetry headline
claim, Exp 1474 through Exp 1477 preserve bounded CPU/simulator tracks, and
the forbidden files `research-roadmap.yaml` and `scripts/research_conductor.py`
are unchanged, when Exp 1478 runs, then it writes all required REQ-REPORT-049
fields, reports `criteria_total == 12`, records the missing next-roadmap file,
sets `ops_docs_updated == false` when docs reconciliation is delegated, and
writes an honest verdict with the final score.

### REQ-REPORT-050: Milestone .114 Activation Manifest

The Exp 1479 `.114` activation workflow shall write
`results/experiment_1479_113_completion_archive_114_activation.json` with
`status="in_progress"` before terminal completion. It shall then read
`results/experiment_1478_milestone_113_retro.json`, the Exp 1467 through
Exp 1478 entries in `ops/conductor-log.md`, `ops/status.md`,
`ops/changelog.md`, and `research-complete.yaml` without modifying
`scripts/research_conductor.py` or `research-roadmap.yaml`.

The workflow shall write `ops/milestone_114_activation_manifest.md` with the
allowed `.114` tracks:

- adversarial balanced telemetry
- BEAVER-lite calibration
- HalluGuard-style risk-bound fit
- FR-11 query-time self-learning
- CCTU-style executable constraints
- V_1 pairwise verification
- THRML preflight/parity
- partial-trace localization

The manifest shall explicitly preserve the blocks on telemetry headline claims,
repair-executor reruns, GRPO/VPRM, WOPR puzzle cartridges, HardNet++/DSP, broad
VNN-COMP runners, KV260 board claims, and THRML/TSU hardware claims. Hardware
evidence boundaries shall remain limited to dual RTX 3090 runtime, KV260 RTL
source/simulation, and THRML simulator preflight/parity unless an operator
reopens the scope with live evidence and a falsifiable gate. The terminal
artifact shall include:

- `status`
- `milestone`
- `predecessor_milestone`
- `predecessor_criteria_met`
- `predecessor_criteria_total`
- `activation_manifest_complete`
- `telemetry_headline_block_preserved`
- `self_learning_followup_allowed`
- `hardware_claim_boundaries`
- `allowed_114_tracks`
- `forbidden_reopen_tracks`
- `research_complete_has_113_entry`
- `honest_verdict`

`research_complete_has_113_entry` shall reflect whether
`research-complete.yaml` already contains a `2026.04.113` archive row. If the
row is absent, the artifact shall record the archive gap rather than claiming
that `.113` is fully archived. The final verdict shall be complete only when
the `.113` retro reports 12 of 12 criteria met, the `.114` manifest is written,
the telemetry-headline block is preserved, the bounded self-learning follow-up
is allowed without broad headline claims, all hardware claim boundaries are
recorded, and no forbidden file modification is reported.

### SCENARIO-REPORT-050: Exp 1479 Activates .114 With Guardrails Preserved

Given Exp 1478 reports `.113` as 12 of 12 criteria met, the conductor log
records terminal entries for Exp 1467 through Exp 1478, `research-complete.yaml`
may or may not already contain a `2026.04.113` archive row, and the .113 ops
docs preserve the telemetry, self-learning, and hardware guardrails, when
Exp 1479 runs for run date `20260507`, then it writes all required
REQ-REPORT-050 fields, writes the `.114` activation markdown, reports the
archive-row state honestly, lists only the allowed `.114` tracks, forbids the
blocked reopen tracks, confirms `research-roadmap.yaml` and
`scripts/research_conductor.py` were not modified, and writes an honest
activation verdict.

### REQ-REPORT-051: HalluGuard Risk-Bound Fit Audit

The Exp 1483 HalluGuard risk-bound fit audit workflow shall write
`results/experiment_1483_halluguard_risk_bound_fit_audit.json` with
`status="in_progress"` before source-artifact loading. It shall then read the
HalluGuard entry in `research-references.md`, the relevant `.113` and `.114`
live telemetry artifacts, the BEAVER-lite deterministic-bound artifacts, and
`docs/research-notes/paper_v6_anchored_claim_matrix.md` without modifying
`scripts/research_conductor.py`.

The workflow shall map Carnot's available evidence into a HalluGuard-style
risk-bound fit audit by separating:

- data-driven/evidence-availability risk fields, including live telemetry
  availability, logits/top-k availability, known verifier labels, balanced
  label counts, superficial-confound audit status, and missing-evidence
  caveats; and
- reasoning-driven/reasoning-step risk fields, including BEAVER-lite bound
  soundness, unsafe-mass bounds, empirical violation rates, prefix-closed
  constraint counts, and the limitation that the current artifacts do not
  certify every reasoning step.

The terminal artifact shall include:

- `status`
- `source_artifacts`
- `risk_decomposition_complete`
- `data_driven_fields_available`
- `reasoning_driven_fields_available`
- `implemented_assumptions`
- `missing_assumptions`
- `claim_allowed`
- `audit_note_path`
- `honest_verdict`

`claim_allowed` shall be `false` for full HalluGuard reproduction unless every
formal HalluGuard assumption, including the required NTK/certification
conditions and full data-driven and reasoning-driven risk-bound checks, is
implemented and checked locally. The workflow shall write
`docs/research-notes/halluguard_carnot_risk_bound_fit.md` with implemented
assumptions, missing assumptions, and allowed wording.

### SCENARIO-REPORT-051: Exp 1483 Blocks Full HalluGuard Reproduction Claim

Given Exp 1470 reports sound BEAVER-lite live-logprob bounds, Exp 1473 blocks
telemetry headline claims under adversarial validity checks, Exp 1480 reports
balanced live SOTA telemetry with logits/top-k availability and verifier
labels, Exp 1482 reports calibrated live prefix bounds, and the HalluGuard
reference entry warns not to claim full reproduction without NTK/certification
assumptions, when Exp 1483 runs for run date `20260507`, then it writes all
required REQ-REPORT-051 fields, completes the data-driven versus
reasoning-driven decomposition, writes the audit note, records a non-empty
`missing_assumptions` list, sets `claim_allowed == false`, and reports an
honest verdict that Carnot has only a HalluGuard-style fit audit rather than a
full HalluGuard reproduction.

### REQ-REPORT-052: Milestone .115 Activation Manifest

The Exp 1492 `.115` activation workflow shall write
`results/experiment_1492_114_completion_archive_115_activation.json` with
`status="in_progress"` before terminal completion. It shall then read
`results/experiment_1491_milestone_114_retro.json`, the Exp 1479 through
Exp 1491 entries in `ops/conductor-log.md`, `research-complete.yaml`,
`ops/status.md`, and `ops/changelog.md` without modifying
`scripts/research_conductor.py` or `research-roadmap.yaml`.

The workflow shall write `ops/milestone_115_activation_manifest.md` with the
allowed `.115` tracks:

- trigger-token certificate export
- prompt-to-validator compilation
- interwhen-style monitoring
- HoVer safe-prefix continuation
- FR-11 trace2skill daily eval
- artifact reachability
- verifier orthogonality
- graph-energy adapters
- KAN hardware accounting
- gated THRML import/parity

The manifest shall explicitly preserve the blocks on Semantic Energy/logit
telemetry headline claims, V_1 pairwise headline claims, decoded-quality claims
from injected-failure localization, THRML parity before import readiness, KV260
board claims, TSU hardware claims, GRPO/VPRM reopenings, WOPR puzzle
cartridges, and legacy small-model headline results. The terminal artifact
shall include:

- `status`
- `milestone`
- `predecessor_milestone`
- `predecessor_criteria_met`
- `predecessor_criteria_total`
- `activation_manifest_complete`
- `retired_headline_signals`
- `allowed_115_tracks`
- `gated_115_tracks`
- `continuous_self_learning_required`
- `mandated_sota_models`
- `research_complete_has_114_entry`
- `honest_verdict`

`research_complete_has_114_entry` shall reflect whether
`research-complete.yaml` already contains a `2026.04.114` archive row. If the
row is absent, the artifact shall record the archive gap rather than claiming
that `.114` is fully archived. The final verdict shall use a conductor-accepted
success prefix and shall be complete only when the `.114` retro reports 12 of
13 criteria met with the structured THRML gate skip recorded, the `.115`
manifest is written, the retired headline signals and guardrail blocks are
preserved, the continuous self-learning requirement and mandated local SOTA
models are recorded, and no forbidden file modification is reported.

### SCENARIO-REPORT-052: Exp 1492 Activates .115 With .114 Guardrails Preserved

Given Exp 1491 reports `.114` as 12 of 13 criteria met with one honest
structured gate skip, the conductor log records terminal entries for Exp 1479
through Exp 1491, `research-complete.yaml` may or may not already contain a
`2026.04.114` archive row, and the .115 planning docs preserve the Semantic
Energy, V_1, THRML, hardware, self-learning, and legacy-small-model guardrails,
when Exp 1492 runs for run date `20260507`, then it writes all required
REQ-REPORT-052 fields, writes the `.115` activation markdown, reports the
archive-row state honestly, lists only the allowed `.115` tracks, records the
structured gates, confirms `research-roadmap.yaml` and
`scripts/research_conductor.py` were not modified, and writes an honest
activation verdict with an accepted success prefix.

### REQ-REPORT-053: Milestone .115 Terminal Retrospective

The Exp 1505 `.115` retrospective workflow shall write
`results/experiment_1505_milestone_115_retro.json` with `status="in_progress"`
before terminal completion. It shall then read the authoritative Exp 1492
through Exp 1504 result JSON artifacts, the Exp 1492 through Exp 1504 conductor
log entries, `openspec/change-proposals/research-roadmap-vNEXT.md`,
`research-complete.yaml`, `ops/status.md`, `ops/changelog.md`, and
`ops/known-issues.md` without modifying `research-roadmap.yaml` or
`scripts/research_conductor.py`.

The workflow shall score every `.115` success criterion from the roadmap
source fields, including honest structured gate skips where a prerequisite gate
prevents an experiment from running. It shall summarize each experiment verdict
and classify milestone lines as graduated, carry-forward, or retired while
preserving claim boundaries around Semantic Energy/logit telemetry, V_1
pairwise self-verification, bounded continuous self-learning, simulator-only
THRML evidence, and no-synthesis/no-board hardware accounting. The terminal
artifact shall include:

- `status`
- `milestone`
- `criteria_met`
- `criteria_total`
- `experiments_reviewed`
- `completed_experiments`
- `honest_gate_skips`
- `retired_lines`
- `graduated_lines`
- `carry_forward_lines`
- `continuous_self_learning_outcome`
- `hardware_claim_boundaries`
- `ops_docs_updated`
- `research_complete_updated`
- `protected_files_unchanged`
- `honest_verdict`

`research_complete_updated` shall be true only when the workflow appends a
concise `2026.04.115` archive row to `research-complete.yaml` from terminal
artifacts; otherwise the artifact shall record why the archive was not updated.
For conductor stop-when-done retro runs where a separate reconciliation agent
owns `ops/status.md`, `ops/changelog.md`, and `_bmad/traceability.md`,
`ops_docs_updated` shall remain false with an explicit deferred-reconciliation
reason rather than modifying those files. The final verdict shall start with a
conductor-accepted success prefix.

### SCENARIO-REPORT-053: Exp 1505 Closes .115 With Claim Boundaries Preserved

Given Exp 1492 through Exp 1504 terminal artifacts exist, Exp 1493 through Exp
1496 report trigger certificates, validator compilation, monitor intervention,
and safe-prefix continuation readiness with zero false accepts, Exp 1497 and
Exp 1498 report bounded FR-11 daily-eval and reachable evidence, Exp 1499 and
Exp 1500 report verifier orthogonality and deterministic-first discipline, Exp
1501 reports deterministic plan-graph energy readiness, Exp 1502 reports
no-synthesis KAN accounting, Exp 1503 reports `thrml_import_ready=true`, and
Exp 1504 reports simulator-only parity with no hardware claim, when Exp 1505
runs for run date `20260507`, then it writes all required REQ-REPORT-053
fields, reports the success-criteria score from source fields, records any
honest gate skips explicitly, updates `research-complete.yaml` only when enough
terminal artifacts exist, leaves `research-roadmap.yaml` and
`scripts/research_conductor.py` unchanged, and writes an honest retrospective
verdict with an accepted success prefix.

### REQ-REPORT-054: Milestone .116 Activation Manifest

The Exp 1506 `.116` activation workflow shall write
`results/experiment_1506_115_completion_archive_116_activation.json` with
`status="in_progress"` before terminal completion. It shall then read
`results/experiment_1505_milestone_115_retro.json`, the Exp 1492 through
Exp 1505 entries in `ops/conductor-log.md`, `research-complete.yaml`,
`ops/status.md`, `ops/changelog.md`, `research-roadmap.yaml`,
`openspec/change-proposals/research-roadmap-vNEXT.md`,
`research-hardware-wishlist.md`, `_bmad/architecture.md`, and the Exp 1502 and
Exp 1504 result artifacts without modifying `scripts/research_conductor.py` or
`research-roadmap.yaml`.

The workflow shall write `ops/milestone_116_activation_manifest.md` with the
allowed `.116` tracks:

- safe-DSL verifier induction
- trigger+grammar certificate decoding
- executable monitor runtime
- plan-graph structural contracts
- product-line solver oracle
- FR-11 verifier-feedback replay
- trace2skill portable pack
- THRML SamplerBackend conformance
- KAN shape normalization
- KV260 source-level RTL properties

The manifest shall explicitly preserve the blocks on Semantic Energy/logit
telemetry headline claims, V_1 pairwise headline claims, decoded-quality claims
from injected-failure localization, arbitrary generated-Python verifier trust,
TSU hardware claims, KV260 board claims, synthesis claims, and legacy
small-model headline results. The terminal artifact shall include:

- `status`
- `milestone`
- `predecessor_milestone`
- `predecessor_criteria_met`
- `predecessor_criteria_total`
- `activation_manifest_complete`
- `prior_trigger_certificates_ready`
- `prior_validator_compiler_ready`
- `prior_monitor_replay_ready`
- `prior_fr11_daily_eval_ready`
- `prior_thrml_parity_ready`
- `prior_kan_shape_blocker_recorded`
- `prior_kv260_source_track_active`
- `mandated_sota_models`
- `continuous_self_learning_required`
- `retired_headline_signals`
- `allowed_116_tracks`
- `gated_116_tracks`
- `research_complete_has_115_entry`
- `honest_verdict`

`prior_thrml_parity_ready` shall be true only when Exp 1504 reports simulator
parity passed without a hardware claim. `prior_kan_shape_blocker_recorded`
shall be true only when Exp 1502 records the KAN/KAEM proxy shape-normalization
carry-forward. `prior_kv260_source_track_active` shall be true only when the
hardware wishlist or architecture record KV260 source-level RTL/lint/simulation
work as active while deferring board, bitfile, and latency claims. The final
verdict shall use a conductor-accepted success prefix and shall be complete
only when Exp 1505 reports 12 of 12 criteria met, `research-complete.yaml`
contains a `2026.04.115` archive row, all prior-readiness booleans above are
recorded honestly, the `.116` activation markdown is written, and no forbidden
file modification is reported.

### SCENARIO-REPORT-054: Exp 1506 Activates .116 With .115 Evidence Archived

Given Exp 1505 reports `.115` as 12 of 12 criteria met, `research-complete.yaml`
contains a `2026.04.115` archive row, Exp 1504 reports simulator-only THRML
parity with no hardware claim, Exp 1502 records the KAN shape-normalization
carry-forward blocker, and the hardware wishlist or architecture keeps KV260
source-level RTL/lint/simulation work active while deferring board claims, when
Exp 1506 runs for run date `20260507`, then it writes all required
REQ-REPORT-054 fields, writes the `.116` activation markdown, lists only the
allowed `.116` tracks, records the structured gates for downstream `.116`
tasks, confirms `research-roadmap.yaml` and `scripts/research_conductor.py`
were not modified, and writes an honest activation verdict with an accepted
success prefix.

### REQ-REPORT-055: Milestone .116 Terminal Retrospective

The Exp 1518 `.116` retrospective workflow shall write
`results/experiment_1518_milestone_116_retro.json` with `status="in_progress"`
before terminal completion. It shall then read the authoritative Exp 1506
through Exp 1517 result JSON artifacts, the `.116` success criteria in
`openspec/change-proposals/research-roadmap-vNEXT.md`, `research-roadmap.yaml`,
`research-complete.yaml`, `ops/conductor-log.md`, `ops/status.md`, and
`ops/changelog.md` without modifying `research-roadmap.yaml` or
`scripts/research_conductor.py`.

The workflow shall score every `.116` success criterion from source artifact
fields, count honest gate-blocked tasks separately from failed tasks, and record
whether any source artifact reports changes to `research-roadmap.yaml` or
`scripts/research_conductor.py`. It shall preserve the `.116` claim boundaries:
Semantic Energy/logit telemetry and V_1 are not headline signals, generated
verifier code is trusted only after safe-DSL compilation, and THRML, KAN, and
KV260 evidence is software/source conformance rather than hardware execution.
The terminal artifact shall include:

- `status`
- `milestone`
- `criteria_met`
- `criteria_total`
- `completed_tasks`
- `gated_or_blocked_tasks`
- `failed_tasks`
- `verifier_runtime_contract_ready`
- `continuous_self_learning_result`
- `substrate_conformance_result`
- `retired_or_demoted_claims`
- `carry_forward_gates`
- `ops_docs_updated`
- `research_complete_entry_recommended`
- `honest_verdict`

For conductor stop-when-done retro runs where a separate reconciliation agent
owns `ops/status.md`, `ops/changelog.md`, and `_bmad/traceability.md`,
`ops_docs_updated` shall remain false with an explicit deferred-reconciliation
reason rather than modifying those files. `research_complete_entry_recommended`
shall describe the recommended `.116` archive row without appending it. The
final verdict shall start with a conductor-accepted success prefix.

### SCENARIO-REPORT-055: Exp 1518 Closes .116 With Runtime And Substrate Gates Preserved

Given Exp 1506 through Exp 1517 terminal artifacts exist, Exp 1507 through Exp
1511 report verifier induction, grammar decoding, monitor runtime, structural
contracts, and product-line solver-oracle readiness, Exp 1512 through Exp 1514
report bounded FR-11 policy-cache, rollback replay, and portable skill-pack
readiness, and Exp 1515 through Exp 1517 report simulator/source-only THRML,
KAN, and KV260 conformance without hardware claims, when Exp 1518 runs for run
date `20260508`, then it writes all required REQ-REPORT-055 fields, reports the
success-criteria score from source fields, records gate-blocked and failed tasks
separately, confirms `research-roadmap.yaml` and `scripts/research_conductor.py`
were not modified, recommends but does not append the `research-complete.yaml`
entry, and writes an honest retrospective verdict with an accepted success
prefix.

### REQ-REPORT-056: Milestone .117 Activation Manifest

The Exp 1519 `.116` completion archive and `.117` activation workflow shall
write `results/experiment_1519_116_completion_archive_117_activation.json` with
`status="in_progress"` before terminal completion. It shall then read Exp 1518,
`research-complete.yaml`, conductor log evidence for Exp 1506 through Exp 1518,
`ops/status.md`, `ops/changelog.md`, `ops/known-issues.md`, and the `.117`
roadmap plan without modifying `research-roadmap.yaml` or
`scripts/research_conductor.py`.

The terminal artifact shall include:

- `status`
- `milestone`
- `predecessor_milestone`
- `predecessor_criteria_met`
- `predecessor_criteria_total`
- `activation_manifest_complete`
- `prior_runtime_contract_ready`
- `prior_fr11_rollback_ready`
- `prior_product_line_benchmark_ready`
- `prior_thrml_conformance_ready`
- `prior_kan_shape_manifest_ready`
- `prior_kv260_property_pack_ready`
- `research_complete_has_116_entry`
- `mandated_sota_models`
- `continuous_self_learning_required`
- `allowed_117_tracks`
- `gated_117_tracks`
- `retired_headline_signals`
- `honest_verdict`

`prior_runtime_contract_ready` shall be true only when Exp 1507, Exp 1508,
Exp 1509, and Exp 1510 are complete and record zero false accepts.
`prior_fr11_rollback_ready` shall be true only when Exp 1512 and Exp 1513 are
complete and record zero soundness mistakes. `prior_product_line_benchmark_ready`
shall be true when Exp 1511 produced the product-line benchmark manifest, even
if its parse, feasibility, and oracle-agreement metrics are weak.
`prior_thrml_conformance_ready` shall be true only when Exp 1515 completed
simulator-only THRML conformance. `prior_kan_shape_manifest_ready` and
`prior_kv260_property_pack_ready` shall preserve the `.116` no-synthesis,
no-board, source-level claim boundaries.

The activation markdown shall list allowed `.117` tracks for runtime-contract
E2E, live SOTA contract-guided repair, CDG root-cause repair, product-line
rescue/retirement, FR-11 live policy promotion, MARCH-style claim isolation,
and THRML/Carnot parity scaling. It shall preserve blocks on Semantic
Energy/logit telemetry headline claims, pairwise LLM verifier headline claims,
arbitrary generated-Python verifier trust, TSU hardware claims, KV260 board
claims, KAN synthesis claims, and legacy small-model headline results. The
final verdict shall use a conductor-accepted success prefix and shall be
complete only when Exp 1518 reports 13 of 13 criteria met, the activation
markdown is written, all readiness gates are recorded honestly, and protected
roadmap/conductor files remain unchanged.

### SCENARIO-REPORT-056: Exp 1519 Activates .117 With .116 Evidence Archived

Given Exp 1518 reports `.116` as 13 of 13 criteria met, Exp 1507 through Exp
1510 report zero-false-accept runtime-contract evidence, Exp 1512 and Exp 1513
report zero soundness mistakes, Exp 1511 produced the product-line benchmark
manifest, and Exp 1515 through Exp 1517 record simulator/source-only substrate
evidence, when Exp 1519 runs for run date `20260508`, then it writes all
required REQ-REPORT-056 fields, writes `ops/milestone_117_activation_manifest.md`,
lists only the allowed `.117` tracks and same-roadmap gates, records whether
`research-complete.yaml` already contains a `2026.04.116` archive row, confirms
`research-roadmap.yaml` and `scripts/research_conductor.py` were not modified,
and writes an honest activation verdict with an accepted success prefix.

### REQ-REPORT-057: Milestone .117 Terminal Retrospective

The Exp 1532 `.117` retrospective workflow shall write
`results/experiment_1532_milestone_117_retro.json` with `status="in_progress"`
before terminal completion. It shall then read the authoritative Exp 1519
through Exp 1531 result JSON artifacts, the `.117` success criteria in
`openspec/change-proposals/research-roadmap-vNEXT.md`, `research-roadmap.yaml`,
`research-complete.yaml`, `ops/conductor-log.md`, `ops/status.md`, and
`ops/changelog.md` without modifying `research-roadmap.yaml` or
`scripts/research_conductor.py`.

The workflow shall score every `.117` success criterion from source artifact
fields as `MET`, `NOT_MET`, or `GATE_BLOCKED`, record missing or gated source
artifacts explicitly, and preserve the `.117` claim boundaries: no TSU hardware
claim, no KAN synthesis claim, no KV260 board claim, no arbitrary
generated-Python trust, no legacy small-model headline result, and no LLM judge
as final authority. It shall decide whether the product-line branch continues
or retires from Exp 1523, and it shall decide the next THRML scaling gate from
Exp 1530 and Exp 1531. The terminal artifact shall include:

- `status`
- `milestone`
- `criteria_met`
- `criteria_total`
- `runtime_contract_e2e_outcome`
- `live_contract_repair_outcome`
- `cdg_root_cause_outcome`
- `product_line_decision`
- `continuous_self_learning_outcome`
- `claim_isolation_outcome`
- `thrml_scaling_outcome`
- `claim_boundaries_preserved`
- `carry_forward_gates`
- `research_complete_entry_recommended`
- `ops_docs_reconciled`
- `honest_verdict`

For conductor stop-when-done retro runs where a separate reconciliation agent
owns `ops/status.md`, `ops/changelog.md`, and `_bmad/traceability.md`,
`ops_docs_reconciled` shall remain false with an explicit deferred
reconciliation reason rather than modifying those files.
`research_complete_entry_recommended` shall describe the recommended `.117`
archive row without appending it. The final verdict shall start with a
conductor-accepted success prefix.

### SCENARIO-REPORT-057: Exp 1532 Closes .117 With Carry-Forward Gates

Given Exp 1519 through Exp 1531 terminal artifacts exist, Exp 1520 through Exp
1525 report zero false accepts or soundness mistakes at their trust boundary,
Exp 1523 reports either `product_line_rescue_ready=true` or
`product_line_branch_retired=true`, and Exp 1526 through Exp 1531 report
software/simulator-only THRML parity outcomes without hardware claims, when Exp
1532 runs for run date `20260508`, then it writes all required REQ-REPORT-057
fields, reports the success-criteria score from source fields, records
carry-forward gates for `.118` with exact artifact fields future tasks must
gate on, confirms `research-roadmap.yaml` and `scripts/research_conductor.py`
were not modified, recommends but does not append the `research-complete.yaml`
entry, and writes an honest retrospective verdict with an accepted success
prefix.

### REQ-REPORT-058: Milestone .118 Activation Manifest

The Exp 1533 `.117` completion archive and `.118` activation workflow shall
write `results/experiment_1533_117_completion_archive_118_activation.json` with
`status="in_progress"` before terminal completion. It shall then read Exp 1532,
the `.117` source artifacts for Exp 1520, Exp 1521, Exp 1522, Exp 1523, Exp
1524, Exp 1525, Exp 1530, and Exp 1531, `research-complete.yaml`,
`ops/conductor-log.md`, `ops/status.md`, `ops/changelog.md`,
`ops/known-issues.md`, and the `.118` roadmap materials without modifying
`research-roadmap.yaml` or `scripts/research_conductor.py`.

The terminal artifact shall include:

- `status`
- `milestone`
- `predecessor_milestone`
- `predecessor_criteria_met`
- `predecessor_criteria_total`
- `activation_manifest_complete`
- `prior_runtime_contract_e2e_ready`
- `prior_live_sota_repair_ready`
- `prior_cdg_ready`
- `prior_product_line_ready`
- `prior_fr11_promotion_ready`
- `prior_claim_isolation_ready`
- `prior_thrml_n128_ready`
- `prior_thrml_diverse_ready`
- `prior_orphan_test_incident_recorded`
- `research_complete_has_117_entry`
- `mandated_sota_models`
- `continuous_self_learning_required`
- `allowed_118_tracks`
- `gated_118_tracks`
- `retired_headline_signals`
- `honest_verdict`

`prior_runtime_contract_e2e_ready` shall be true only when Exp 1520 is complete,
reports `runtime_contract_e2e_ready=true`, and records `false_accept_rate=0.0`.
`prior_live_sota_repair_ready` shall be true only when Exp 1521 reports a
mandated local SOTA GGUF model. `prior_cdg_ready` shall be true only when Exp
1522 reports `cdg_root_cause_repair_ready=true`. `prior_product_line_ready`
shall be true only when Exp 1523 either rescues the branch or explicitly retires
it. `prior_fr11_promotion_ready` shall be true only when Exp 1524 reports
`soundness_mistakes=0` and `no_model_weight_mutation=true`.
`prior_claim_isolation_ready` shall be true only when Exp 1525 reports
deterministic validator outcomes with explicit budget metrics.
`prior_thrml_n128_ready` and `prior_thrml_diverse_ready` shall come from Exp
1530 and Exp 1531 software-only parity fields. `prior_orphan_test_incident_recorded`
shall be true only when `ops/known-issues.md` or `ops/conductor-log.md`
documents the `.117` orphan-test wedge.

The activation markdown shall list allowed `.118` tracks for orphan-test guard,
automata/XGrammar/ABS contract decoding, SATQuest CNF benchmark, BEAVER-lite
prefix-risk audit, residual-drift ledger, external-feedback FR-11 skill graph,
product-line scale, claim-isolation uncertainty routing, ARM/EBT soft-value
diagnostics, THRML n=256/n=64-diverse stress, Extropic Z1 readiness packet, and
milestone retro. It shall preserve blocks on legacy small-model headline
claims, BEAVER/logprob acceptance authority, ARM/EBT soft-value acceptance
authority, Extropic TSU/Z1 hardware execution claims, KV260 board claims, and
model-weight mutation. The final verdict shall use a conductor-accepted success
prefix and shall be complete only when Exp 1532 reports 14 of 14 criteria met,
the activation markdown is written, all readiness gates are recorded honestly,
the orphan-test incident is recorded, and protected roadmap/conductor files
remain unchanged.

### SCENARIO-REPORT-058: Exp 1533 Activates .118 With .117 Evidence Archived

Given Exp 1532 reports `.117` as 14 of 14 criteria met, Exp 1520 reports
zero-false-accept runtime-contract E2E readiness, Exp 1521 uses a mandated local
SOTA GGUF, Exp 1522 through Exp 1525 report their `.117` readiness fields at
the correct trust boundary, Exp 1530 and Exp 1531 report software-only THRML
parity readiness, and the `.117` orphan-test wedge is documented, when Exp 1533
runs for run date `20260508`, then it writes all required REQ-REPORT-058
fields, writes `ops/milestone_118_activation_manifest.md`, lists only the
allowed `.118` tracks and same-roadmap gates, records whether
`research-complete.yaml` already contains a `2026.04.117` archive row, confirms
`research-roadmap.yaml` and `scripts/research_conductor.py` were not modified,
and writes an honest activation verdict with an accepted success prefix.

### REQ-REPORT-059: Milestone .118 Terminal Retrospective

The Exp 1546 `.118` retrospective workflow shall write
`results/experiment_1546_milestone_118_retro.json` with `status="in_progress"`
before terminal completion. It shall then read the authoritative Exp 1533
through Exp 1545 result JSON artifacts, the `.118` success criteria in
`openspec/change-proposals/research-roadmap-vNEXT.md`, `research-roadmap.yaml`,
`research-roadmap-next.yaml`, `research-complete.yaml`, `ops/conductor-log.md`,
`ops/status.md`, `ops/changelog.md`, and `ops/known-issues.md` without
modifying `research-roadmap.yaml` or `scripts/research_conductor.py`.

The workflow shall score every `.118` success criterion from source artifact
fields as `MET`, `NOT_MET`, or `HONESTLY_TERMINAL`, record missing, blocked, or
criterion-failing source artifacts explicitly, and preserve the `.118` claim
boundaries: automata and SAT evidence may improve contract generation or
diagnose false accepts, but deterministic runtime contracts and solver oracles
remain the authority; FR-11 may claim safe query-time promotion only when
`no_model_weight_mutation=true` and `soundness_mistakes=0`, and may claim
positive utility only when `utility_delta > 0`; ARM/EBT soft-value signals
remain diagnostic-only; THRML and Extropic work shall not claim hardware
execution without authenticated device evidence. The terminal artifact shall
include:

- `status`
- `milestone`
- `criteria_met`
- `criteria_total`
- `completed_tasks`
- `honestly_terminal_tasks`
- `failed_or_blocked_tasks`
- `automata_contract_gate`
- `satquest_verifier_gate`
- `residual_drift_gate`
- `fr11_positive_utility_gate`
- `product_line_carry_forward_gate`
- `claim_isolation_router_gate`
- `arm_ebm_diagnostic_boundary`
- `thrml_next_scaling_gate`
- `extropic_access_readiness_gate`
- `recommended_119_focus`
- `ops_reconciliation_needed`
- `active_roadmap_modified`
- `conductor_modified`
- `honest_verdict`

For conductor stop-when-done retro runs where a separate reconciliation agent
owns `research-complete.yaml`, `ops/status.md`, `ops/changelog.md`, and
`_bmad/traceability.md`, `ops_reconciliation_needed` shall identify the needed
follow-up but the workflow shall not edit those files. The final verdict shall
start with a conductor-accepted success prefix.

### SCENARIO-REPORT-059: Exp 1546 Closes .118 With .119 Gates

Given Exp 1533 through Exp 1545 terminal artifacts exist, Exp 1535 improves
contract parsing through automata/ABS constraints, Exp 1536 reports SATQuest
solver-oracle false accepts, Exp 1539 reports safe FR-11 promotion but
`utility_delta=0.0`, Exp 1540 and Exp 1541 report product-line and
claim-isolation budget evidence with zero deterministic false accepts, Exp 1542
keeps ARM/EBT diagnostic-only, Exp 1543 and Exp 1544 report software-only THRML
parity, and Exp 1545 reports readiness without hardware execution, when Exp
1546 runs for run date `20260508`, then it writes all required REQ-REPORT-059
fields, reports the success-criteria score from source fields, records SATQuest
and FR-11 limits without inventing success, records `.119` carry-forward gates,
confirms `research-roadmap.yaml` and `scripts/research_conductor.py` were not
modified, and writes an honest retrospective verdict with an accepted success
prefix.

### REQ-REPORT-060: Milestone .119 Activation Manifest

The Exp 1547 `.118` completion archive and `.119` activation workflow shall
write `results/experiment_1547_118_completion_archive_119_activation.json` with
`status="in_progress"` before terminal completion. It shall then read Exp 1546,
the listed `.118` source artifacts for Exp 1535, Exp 1536, Exp 1538, Exp 1539,
Exp 1540, Exp 1541, Exp 1542, Exp 1543, Exp 1544, and Exp 1545,
`research-complete.yaml`, `research-references.md`, `ops/conductor-log.md`,
`ops/status.md`, `ops/changelog.md`, `ops/known-issues.md`, and `.119`
roadmap materials without modifying `research-roadmap.yaml` or
`scripts/research_conductor.py`. Missing listed artifacts or text inputs shall
be recorded by path rather than replaced with invented evidence.

The terminal artifact shall include:

- `status`
- `milestone`
- `predecessor_milestone`
- `predecessor_criteria_met`
- `predecessor_criteria_total`
- `activation_manifest_complete`
- `prior_automata_ready`
- `prior_satquest_benchmark_ready`
- `prior_satquest_solver_oracle_false_accepts`
- `prior_satquest_zero_solver_false_accepts`
- `prior_residual_drift_ready`
- `prior_fr11_safe_only`
- `prior_fr11_positive_utility`
- `prior_product_line_ready`
- `prior_claim_router_ready`
- `prior_arm_ebm_diagnostic_ready`
- `prior_thrml_n256_ready`
- `prior_thrml_diverse_n64_ready`
- `thrml_independent_rng_required`
- `prior_extropic_packet_ready`
- `research_complete_has_118_entry`
- `mandated_sota_models`
- `continuous_self_learning_required`
- `allowed_119_tracks`
- `retired_headline_signals`
- `honest_verdict`

`predecessor_criteria_met` and `predecessor_criteria_total` shall be set to 13
and 14 only when Exp 1546 supports those values. `prior_satquest_solver_oracle_false_accepts`
shall come from Exp 1546's SATQuest carry-forward gate when available, and
`prior_satquest_zero_solver_false_accepts` shall be false whenever any solver
oracle false accepts remain. `prior_fr11_safe_only` shall be true only when
FR-11 evidence reports `soundness_mistakes=0` and `no_model_weight_mutation=true`.
`prior_fr11_positive_utility` shall be true only when Exp 1546 reports
`positive_utility_achieved=true`. `thrml_independent_rng_required` shall be
true only when `ops/known-issues.md` contains the mandatory THRML/Carnot
independent-RNG audit entry.

The activation markdown shall list allowed `.119` tracks for THRML
independent-RNG audit, SATQuest oracle repair, SATQuest SOTA re-evaluation,
unified automata/SAT/runtime contract gate, residual-drift repair,
claim-isolation scale, product-line scale, FR-11 positive-utility-or-retire,
ARM/EBT telemetry repair, Weaver-style verification routing, THRML/Extropic
packet update, and milestone retro. It shall preserve blocks on legacy
small-model headline claims, SATQuest acceptance before oracle repair, ARM/EBT
soft-value acceptance authority, Extropic TSU/Z1/XTR-0 hardware execution
claims, KV260 board claims, and model-weight mutation. The final verdict shall
use a conductor-accepted success prefix and shall be complete only when the
Exp 1546 13-of-14 predecessor evidence is supported, all required carry-forward
fields are explicit, the activation markdown is written, and protected
roadmap/conductor files remain unchanged.

### SCENARIO-REPORT-060: Exp 1547 Activates .119 With .118 Limits Archived

Given Exp 1546 reports `.118` as 13 of 14 criteria met, Exp 1535 reports ready
automata/ABS contract decoding, Exp 1536 reports three SATQuest solver-oracle
false accepts, Exp 1539 reports safe FR-11 promotion with `utility_delta=0.0`,
Exp 1540 and Exp 1541 report product-line and claim-router readiness, Exp 1542
reports diagnostic-only ARM/EBT evidence without logprob telemetry, Exp 1543
and Exp 1544 report software-only THRML readiness, Exp 1545 reports an
Extropic packet without hardware execution, `research-complete.yaml` already
contains a `.118` archive row, and `ops/known-issues.md` contains the mandatory
THRML/Carnot independent-RNG audit entry, when Exp 1547 runs for run date
`20260508`, then it writes all required REQ-REPORT-060 fields, writes
`ops/milestone_119_activation_manifest.md`, exposes same-roadmap gate fields
for `.119`, confirms `research-roadmap.yaml` and `scripts/research_conductor.py`
were not modified, and writes an honest activation verdict with an accepted
success prefix.

### REQ-REPORT-061: Milestone .119 Terminal Retrospective

The Exp 1559 `.119` retrospective workflow shall write
`results/experiment_1559_milestone_119_retro.json` with `status="in_progress"`
before terminal completion. It shall then read the authoritative Exp 1547
through Exp 1558 result JSON artifacts, the `.119` success criteria in
`openspec/change-proposals/research-roadmap-vNEXT.md`, `research-roadmap.yaml`,
`research-roadmap-next.yaml`, `research-complete.yaml`, `ops/conductor-log.md`,
`ops/status.md`, `ops/changelog.md`, and `ops/known-issues.md` without
modifying `research-roadmap.yaml` or `scripts/research_conductor.py`.

The workflow shall score every `.119` success criterion from source artifact
fields as `MET`, `NOT_MET`, or `HONESTLY_TERMINAL`. It shall record missing,
blocked, skipped, or criterion-failing source artifacts explicitly rather than
inferring success from downstream tasks. THRML independent-RNG work shall only
pass the RNG criterion when `independent_rng_audit_ready=true`,
`rng_path_independent=true`, byte-identical stochastic pairs are absent, and no
hardware claim is made. THRML/Extropic readiness shall be recorded as honestly
terminal, not ready, when the conductor blocks Exp 1558 because Exp 1548 failed
the independent-RNG or bounded-KL gate. ARM/EBT and Weaver soft signals shall
remain diagnostic or routing-only below deterministic validator authority. The
terminal artifact shall include:

- `status`
- `milestone`
- `criteria_met`
- `criteria_total`
- `completed_tasks`
- `honestly_terminal_tasks`
- `failed_or_blocked_tasks`
- `thrml_independent_rng_gate`
- `satquest_oracle_repair_gate`
- `satquest_sota_gate`
- `unified_contract_gate`
- `residual_drift_repair_gate`
- `claim_isolation_scale_gate`
- `product_line_scale_gate`
- `fr11_positive_utility_or_retire_gate`
- `arm_ebt_telemetry_gate`
- `verification_compute_router_gate`
- `extropic_readiness_gate`
- `recommended_120_focus`
- `ops_reconciliation_needed`
- `active_roadmap_modified`
- `conductor_modified`
- `honest_verdict`

For conductor stop-when-done retro runs where a separate reconciliation agent
owns `research-complete.yaml`, `ops/status.md`, `ops/changelog.md`, and
`_bmad/traceability.md`, `ops_reconciliation_needed` shall identify the needed
follow-up but the workflow shall not edit those files. The final verdict shall
start with a conductor-accepted success prefix.

### SCENARIO-REPORT-061: Exp 1559 Closes .119 With .120 Gates

Given Exp 1547 activates `.119`, Exp 1548 proves independent RNG paths but
reports `independent_rng_audit_ready=false` because bounded KL fails, Exp 1549
repairs SATQuest to zero oracle false accepts, Exp 1550 runs local SOTA
SATQuest with zero solver false accepts, Exp 1551 through Exp 1554 report
unified contract, residual-drift, claim-router, and product-line scale
readiness with zero deterministic false accepts, Exp 1555 reports positive
FR-11 utility without model-weight mutation, Exp 1556 reports ARM/EBT logprob
telemetry while keeping deterministic validators final, Exp 1557 reports
Weaver-style routing with soft signals used only for routing, and Exp 1558 is
conductor-blocked by the failed Exp 1548 RNG/KL gate, when Exp 1559 runs for
run date `20260508`, then it writes all required REQ-REPORT-061 fields,
reports the success-criteria score from source fields, records the THRML and
Extropic blocks without inventing readiness, records `.120` carry-forward
gates, confirms `research-roadmap.yaml` and `scripts/research_conductor.py`
were not modified, and writes an honest retrospective verdict with an accepted
success prefix.

### REQ-REPORT-062: Milestone .120 Activation Manifest

The Exp 1560 `.119` completion archive and `.120` activation workflow shall
write `results/experiment_1560_119_completion_archive_120_activation.json`
with `status="in_progress"` before terminal completion. It shall then read
Exp 1559, Exp 1548, Exp 1543, Exp 1544, the `.119` archive in
`research-complete.yaml`, the conductor log entries for Exp 1547 through
Exp 1559, `ops/status.md`, `ops/changelog.md`, `ops/known-issues.md`, the
nine Deep Think verdicts in
`docs/research-notes/iclr26-deep-think-responses.md`, and
`docs/research-notes/iclr26-integration-plan.md` without modifying
`research-roadmap.yaml` or `scripts/research_conductor.py`.

The terminal artifact shall include:

- `status`
- `milestone`
- `predecessor_milestone`
- `predecessor_criteria_met`
- `predecessor_criteria_total`
- `research_complete_has_119_entry`
- `exp1559_reports_criteria_met`
- `activation_manifest_complete`
- `allowed_120_tracks`
- `kinetic_defense_validation_ready`
- `brain_linear_ar_validation_ready`
- `thrml_vendoring_ready`
- `soft_gibbs_residual_ready`
- `rho_C_measurement_ready`
- `paper_v6_drafting_ready`
- `preserved_headline_blocks`
- `thrml_scaling_sweep_lineage_retired`
- `research_roadmap_yaml_modified`
- `scripts_research_conductor_modified`
- `honest_verdict`

`kinetic_defense_validation_ready` shall be true only when Exp 1543 prior
THRML parity data exists. `brain_linear_ar_validation_ready` shall be true
only when both Exp 1543 and Exp 1544 THRML data exist. `thrml_vendoring_ready`
shall be true only when the Exp 1548 KL≈0.17 mismatch finding is recorded.
`soft_gibbs_residual_ready` shall be true for the `.120` n=8 prototype track.
`rho_C_measurement_ready` shall be true only when a k=6 calibration/evaluation
corpus exists. `paper_v6_drafting_ready` shall be true only when
`docs/research-notes/iclr26-integration-plan.md` exists.

The activation markdown shall list only these `.120` tracks:
kinetic-defense-in-depth validation, BRAIN+Linear-AR rescue, SpecAnn rejection
record, THRML vendoring + candidate-warm-start, Soft-Gibbs Residual
implementation + coverage bound, ρ(C) measurement, FR-11 v14 retention audit,
paper-v6 §3 sampler drafting, AR-REINFORCE step-wise baseline, and `.120`
retro. The workflow shall preserve blocks on Semantic Energy/logit headline
claims, pairwise LLM verifier headline claims, arbitrary generated-Python
verifier trust, TSU hardware claims, KV260 board claims, KAN synthesis claims,
and legacy small-model headline results. It shall update
`ops/exclusion_manifest.yaml` to retire the THRML scaling sweep lineage
(Exp 1526 through Exp 1531 plus Exp 1543 and Exp 1544 patterns) because THRML
vendoring makes parity constructive and moves the scaling sweep into the
paper-v6 retrospective record rather than active research.

### SCENARIO-REPORT-062: Exp 1560 Activates .120 With ICLR-26 Gates

Given Exp 1559 reports `.119` as 12 of 13 criteria met, `research-complete.yaml`
contains the `2026.04.119` archive, Exp 1543 and Exp 1544 contain THRML parity
data, Exp 1548 records the KL≈0.17 mismatch, a k=6 calibration corpus exists,
`docs/research-notes/iclr26-integration-plan.md` exists, and the Deep Think
response file contains nine verdict sections, when Exp 1560 runs for run date
`20260508`, then it writes all required REQ-REPORT-062 fields, writes
`ops/milestone_120_activation_manifest.md`, exposes the same-roadmap gate
fields, updates the exclusion manifest with the THRML scaling sweep retirement,
confirms `research-roadmap.yaml` and `scripts/research_conductor.py` were not
modified, and writes an honest activation verdict with an accepted success
prefix.

### REQ-REPORT-063: Milestone .120 Terminal Retrospective

The Exp 1572 `.120` retrospective workflow shall read the authoritative
Exp 1560 through Exp 1571 result JSON artifacts plus the Exp 1573 blocked-gate
artifact and write `results/experiment_1572_milestone_120_retro.json` with
`status="in_progress"` before terminal completion. The terminal artifact shall
audit every `.120` activation, Tier 1, Tier 2, Tier 3, Extropic readiness, and
retrospective criterion from source artifact fields rather than from terminal
verdict prefixes alone.

The terminal artifact shall include:

- `status`
- `milestone`
- `next_milestone`
- `criteria_results`
- `criteria_met`
- `criteria_total` set to `14`
- `criteria_met_fraction`
- `criteria_score_pct`
- `paper_v6_section_3_drafted`
- `all_4_carnot_contributions_validated`
- `rho_C_curve_published_ready`
- `terminal_verdicts`
- `notable_successes`
- `failures_or_partials`
- `bottlenecks_identified`
- `carry_forward_gates_121`
- `retro_complete`
- `honest_verdict`

Missing source artifacts shall count as `MISSING`. Blocked pre-gate artifacts
shall count as `BLOCKED`. Falsified acceptance gates shall count as `NOT_MET`
even when their source artifact uses a conductor-accepted `complete:` honest
verdict prefix. THRML vendoring may satisfy the sampler-replacement criterion
when the vendored sampler, KL=0 parity, candidate warm start, focused
regression, and applicable E2E gates pass, but the terminal artifact shall
retain any full-suite regression caveat from the source artifact. FR-11 v14
retention audit may satisfy the audit criterion when it completes the retained
policy review and files a reversal recommendation, but the terminal artifact
shall retain any target-count caveat from the source artifact.

`all_4_carnot_contributions_validated` shall be true only when all four
paper-v6 contribution gates are met: C-parameterized rho(C) curve, Soft-Gibbs
Residual, kinetic defense-in-depth, and BRAIN+Linear-AR rescue. `rho_C_curve_
published_ready` shall be derived from the Exp 1567 fitted curve, confidence
intervals, and inversion validation fields. The final `honest_verdict` shall
start with `complete:` and encode the met/total criteria count.

### SCENARIO-REPORT-063: Exp 1572 Closes .120 With Carry-Forward Gates

Given Exp 1560 activates `.120`, Exp 1561 falsifies THRML kinetic-security
parity, Exp 1562 falsifies the predicted BRAIN+Linear-AR widening, Exp 1563
documents the SpecAnn rejection, Exp 1564 vendors THRML with candidate warm
start while retaining the full-suite regression caveat, Exp 1565 implements
Soft-Gibbs Residual, Exp 1566 validates candidate warm-start, Exp 1567 fits the
rho(C) curve, Exp 1568 completes the retained-policy audit with a reversal
recommendation and target-count caveat, Exp 1569 is blocked by the
prior-failure pre-gate, Exp 1570 verifies the Soft-Gibbs coverage bound,
Exp 1571 passes the step-wise AR-REINFORCE baseline gate, and Exp 1573 is
blocked by the prior-failure pre-gate, when Exp 1572 runs for run date
`20260508`, then it writes all required REQ-REPORT-063 fields, reports
`criteria_total == 14`, counts only met acceptance gates in `criteria_met`,
sets `paper_v6_section_3_drafted == false`, sets
`all_4_carnot_contributions_validated == false`, records the required `.121`
carry-forward gates for paper-v6 Section 3 finalization, SpecAnn rejection
record verification, Phase 5 PCD divergence audit, Soft-Gibbs Residual at
production scale n=128, and MCMC-Layer-free Phase 5 architecture, and writes an
honest retrospective verdict with an accepted success prefix.

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

### REQ-REPORT-063: Milestone .121 Activation Manifest

The Exp 1574 activation workflow shall archive milestone `2026.05.120` and
activate milestone `2026.05.121` by writing
`results/experiment_1574_120_completion_archive_121_activation.json` and
`ops/milestone_121_activation_manifest.md`.

The workflow shall first write the JSON artifact with `status="in_progress"`.
The terminal artifact shall include these top-level fields:

- `status`
- `activation_manifest_complete`
- `prior_failure_autofill_ready`
- `paper_v6_sampler_resume_ready`
- `extropic_packet_resume_ready`
- `brain_reinforce_training_ready`
- `ot_framework_adoption_ready`
- `dccd_jsonschema_smoke_ready`
- `fr11_v15_patch_ready`
- `phase1_ship_readiness_ready`
- `hardware_eval_ready`
- `honest_verdict`

The workflow shall summarize what `.120` proved, falsified, and carried
forward from source artifacts, including the blocked Exp 1569 paper-v6 sampler
draft and blocked Exp 1573 Extropic Z1 readiness packet. It shall expose .121
allowed tracks for prior-failure repair, paper-v6 sampler drafting, Extropic
Z1 readiness update, BRAIN REINFORCE training dynamics, OT verification
framework adoption, DCCD/JSONSchemaBench SOTA smoke, FR-11 lambda-GRPO
retention repair, Phase-1 ship readiness, Z1 drift correction,
Tenstorrent/PolarFire/Strix/KV260 hardware portfolio correction, and retro.

The workflow shall preserve blocks on TSU/Z1 hardware execution claims, KV260
board claims without transcripts, legacy-small-model headline results, and soft
energy/logprob scores as acceptance authority. It shall report
`research-roadmap.yaml` and `scripts/research_conductor.py` as unchanged when
`git diff --quiet --` confirms no activation-workflow edits to those paths.

### SCENARIO-REPORT-063: Exp 1574 Activates .121 Carry-Forward Gates

**Given** Exp 1572 reports milestone `.120` as complete with `10/14` criteria
met and carries forward Exp 1569, Exp 1573, BRAIN REINFORCE training dynamics,
and FR-11 lambda-GRPO reversal work
**And** Exp 1569 and Exp 1573 are blocked at the conductor prior-failure gate
**And** Exp 1571 provides the step-wise AR-REINFORCE baseline evidence
**When** the Exp 1574 activation workflow runs
**Then** it writes the required REQ-REPORT-063 fields
**And** `activation_manifest_complete`,
`brain_reinforce_training_ready`, `phase1_ship_readiness_ready`, and
`hardware_eval_ready` are true
**And** `ops/milestone_121_activation_manifest.md` lists the allowed .121
tracks and the preserved claim blocks
**And** `research-roadmap.yaml` and `scripts/research_conductor.py` remain
unchanged by the activation workflow.

### REQ-REPORT-064: Exp 1575 Carry-Forward Prior-Failure Audit

The Exp 1575 audit workflow shall write
`results/experiment_1575_carry_forward_prior_failures_autofill_audit.json` to
prove that the Exp 1576 and Exp 1577 `.121` carry-forward roadmap entries cite
real prior failures with exact source-artifact verdicts before either task is
dispatched.

The workflow shall first write the JSON artifact with `status="in_progress"`.
The terminal artifact shall include these top-level fields:

- `status`
- `autofill_dry_run_completed`
- `validate_prior_failures_passed`
- `audit_roadmap_gates_passed`
- `exp1576_prior_failures_valid`
- `exp1577_prior_failures_valid`
- `carryforward_prior_failures_ready`
- `honest_verdict`

The workflow shall request `research-roadmap-next.yaml` and fall back to the
active `research-roadmap.yaml` when the next-roadmap handoff file is absent.
It shall dry-run `scripts/conductor_priors_autofill.py`, run
`scripts/validate_prior_failures.py`, run `scripts/audit_roadmap_gates.py`, and
record command outputs plus dry-run task/stub counts. It shall independently
inspect Exp 1576 and Exp 1577 prior-failure entries in the selected roadmap,
confirm that Exp 1576 cites Exp 1569's exact `honest_verdict`, confirm that Exp
1577 cites Exp 1573's exact `honest_verdict`, and confirm that every additional
`expNNNN-*` prior listed by those tasks has a real source artifact whose
`honest_verdict` exactly equals the roadmap `verdict`.

### SCENARIO-REPORT-064: Exp 1575 Blocks Carry-Forward Gaps

**Given** the selected `.121` roadmap contains Exp 1576 and Exp 1577
**And** the source artifacts for Exp 1569 and Exp 1573 both report
`honest_verdict="blocked_gate_check_failed"`
**When** the Exp 1575 audit workflow runs
**Then** it writes the required REQ-REPORT-064 fields
**And** `carryforward_prior_failures_ready` is true only when the autofill
dry-run, prior-failure validator, roadmap-gate audit, and both target task
inspections all pass
**And** any missing task, missing prior entry, missing source artifact, or
verdict mismatch is recorded with the exact task id and field.


### REQ-REPORT-065: Exp 1577 Extropic Z1 Readiness Packet THRML Alignment

The Exp 1577 workflow shall update the Extropic Z1 readiness packet after
THRML vendoring without upgrading simulator or readiness evidence into a
Z1/XTR/TSU hardware claim.

The workflow shall first write
`results/experiment_1577_extropic_z1_readiness_packet_thrml_alignment_resumed.json`
with `status="in_progress"`. The terminal workflow shall write
`docs/research-notes/extropic-z1-readiness-packet-2026-05-121.md` and then
write the terminal JSON artifact with these top-level fields:

- `status`
- `packet_path`
- `extropic_z1_packet_updated`
- `thrml_vendoring_reflected`
- `analog_drift_correction_required`
- `simulator_only_no_hardware_claim`
- `honest_verdict`

The markdown packet shall reflect THRML 0.1.3 vendoring from Exp 1564, the
candidate warm-start API requirement from Exp 1566, Soft-Gibbs Residual
relevance from Exp 1565, and the existing Exp 1545 access-readiness boundary.
It shall contain a `pre-silicon correction prerequisites` section naming
detailed-balance drift correction as required before any Z1 claim. The packet
shall explicitly state that the current status is simulator-only and shall not
claim access to Z1, XTR, or TSU hardware.

### SCENARIO-REPORT-065: Exp 1577 Updates Z1 Packet Without Hardware Claim

**Given** Exp 1545 produced an access-readiness packet with no hardware
execution claim
**And** Exp 1564 reports `thrml_vendoring_complete=true`,
`thrml_version="0.1.3"`, `simulator_only=true`, and
`no_tsu_hardware_claim=true`
**And** Exp 1565 reports Soft-Gibbs Residual implemented
**And** Exp 1566 reports `candidate_warm_start_validated=true`
**When** the Exp 1577 workflow runs
**Then** it writes the required REQ-REPORT-065 fields, writes the markdown
packet, sets `thrml_vendoring_reflected=true`,
`analog_drift_correction_required=true`, and
`simulator_only_no_hardware_claim=true`, and records an honest verdict that
keeps Z1/XTR/TSU hardware execution unclaimed.


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
| REQ-REPORT-043 | `python/carnot/reporting/wopr_puzzle_cartridge_retirement.py`, `results/experiment_1457_wopr_puzzle_cartridge_retirement.json`, `ops/lineage-retirements/wopr_puzzle_cartridges_retired.md`, `ops/exclusion_manifest.yaml` | `tests/python/test_wopr_puzzle_cartridge_retirement.py` | Implemented |
| REQ-REPORT-044 | `python/carnot/reporting/hardnet_dsp_repair_stack_retirement.py`, `results/experiment_1458_hardnet_dsp_repair_stack_consolidation.json`, `ops/lineage-retirements/hardnet_dsp_repair_stack_retired.md`, `ops/exclusion_manifest.yaml` | `tests/python/test_hardnet_dsp_repair_stack_retirement.py` | Implemented |
| REQ-REPORT-045 | `python/carnot/reporting/comparator_cite_retire_audit.py`, `results/experiment_1461_comparator_integration_cite_retire_audit.json`, `docs/research-notes/comparator_cite_retire_audit.md`, `research-references.md` | `tests/python/test_comparator_cite_retire_audit.py` | Implemented |
| REQ-REPORT-046 | `python/carnot/reporting/external_verifier_benchmark_fit_audit.py`, `results/experiment_1465_external_verifier_benchmark_fit_audit.json`, `docs/research-notes/external_verifier_benchmark_fit.md` | `tests/python/test_external_verifier_benchmark_fit_audit.py` | Implemented |
| REQ-REPORT-047 | `python/carnot/reporting/milestone_retro_112.py`, `results/experiment_1466_milestone_112_retro.json` | `tests/python/test_milestone_retro_112.py` | Implemented |
| REQ-REPORT-048 | `python/carnot/reporting/milestone_113_activation_manifest.py`, `results/experiment_1467_112_completion_archive_113_activation.json`, `ops/milestone_113_activation_manifest.md` | `tests/python/test_milestone_113_activation_manifest.py` | Implemented |
| REQ-REPORT-049 | `python/carnot/reporting/milestone_retro_113.py`, `results/experiment_1478_milestone_113_retro.json` | `tests/python/test_milestone_retro_113.py` | Implemented |
| REQ-REPORT-050 | `python/carnot/reporting/milestone_114_activation_manifest.py`, `results/experiment_1479_113_completion_archive_114_activation.json`, `ops/milestone_114_activation_manifest.md` | `tests/python/test_milestone_114_activation_manifest.py` | Implemented |
| REQ-REPORT-051 | `python/carnot/reporting/halluguard_risk_bound_fit_audit.py`, `results/experiment_1483_halluguard_risk_bound_fit_audit.json`, `docs/research-notes/halluguard_carnot_risk_bound_fit.md` | `tests/python/test_halluguard_risk_bound_fit_audit.py` | Implemented |
| REQ-REPORT-052 | `python/carnot/reporting/milestone_115_activation_manifest.py`, `results/experiment_1492_114_completion_archive_115_activation.json`, `ops/milestone_115_activation_manifest.md` | `tests/python/test_milestone_115_activation_manifest.py` | Implemented |
| REQ-REPORT-053 | `python/carnot/reporting/milestone_retro_115.py`, `results/experiment_1505_milestone_115_retro.json`, `research-complete.yaml` | `tests/python/test_milestone_retro_115.py` | Implemented |
| REQ-REPORT-054 | `python/carnot/reporting/milestone_116_activation_manifest.py`, `results/experiment_1506_115_completion_archive_116_activation.json`, `ops/milestone_116_activation_manifest.md` | `tests/python/test_milestone_116_activation_manifest.py` | Implemented |
| REQ-REPORT-055 | `python/carnot/reporting/milestone_retro_116.py`, `results/experiment_1518_milestone_116_retro.json` | `tests/python/test_milestone_retro_116.py` | Implemented |
| REQ-REPORT-056 | `python/carnot/reporting/milestone_117_activation_manifest.py`, `results/experiment_1519_116_completion_archive_117_activation.json`, `ops/milestone_117_activation_manifest.md` | `tests/python/test_milestone_117_activation_manifest.py` | Implemented |
| REQ-REPORT-057 | `python/carnot/reporting/milestone_retro_117.py`, `results/experiment_1532_milestone_117_retro.json` | `tests/python/test_milestone_retro_117.py` | Implemented |
| REQ-REPORT-058 | `python/carnot/reporting/milestone_118_activation_manifest.py`, `results/experiment_1533_117_completion_archive_118_activation.json`, `ops/milestone_118_activation_manifest.md` | `tests/python/test_milestone_118_activation_manifest.py` | Implemented |
| REQ-REPORT-059 | `python/carnot/reporting/milestone_retro_118.py`, `results/experiment_1546_milestone_118_retro.json` | `tests/python/test_milestone_retro_118.py` | Implemented |
| REQ-REPORT-060 | `python/carnot/reporting/milestone_119_activation_manifest.py`, `results/experiment_1547_118_completion_archive_119_activation.json`, `ops/milestone_119_activation_manifest.md` | `tests/python/test_milestone_119_activation_manifest.py` | Implemented |
| REQ-REPORT-061 | `python/carnot/reporting/milestone_retro_119.py`, `results/experiment_1559_milestone_119_retro.json` | `tests/python/test_milestone_retro_119.py` | Implemented |
| REQ-REPORT-062 | `python/carnot/reporting/milestone_120_activation_manifest.py`, `results/experiment_1560_119_completion_archive_120_activation.json`, `ops/milestone_120_activation_manifest.md`, `ops/exclusion_manifest.yaml` | `tests/python/test_milestone_120_activation_manifest.py` | Implemented |
| REQ-REPORT-063 | `python/carnot/reporting/milestone_121_activation_manifest.py`, `results/experiment_1574_120_completion_archive_121_activation.json`, `ops/milestone_121_activation_manifest.md` | `tests/python/test_milestone_121_activation_manifest.py` | Implemented |
| REQ-REPORT-064 | `scripts/experiment_1575_carry_forward_prior_failures_autofill_audit.py`, `results/experiment_1575_carry_forward_prior_failures_autofill_audit.json` | `tests/python/test_experiment_1575_carry_forward_prior_failures_audit.py` | Implemented |
| REQ-REPORT-024 | `python/carnot/reporting/agent_usage.py`, `scripts/agent_plan_usage.py` | `tests/python/test_agent_plan_usage.py` | Implemented |
| REQ-PUBLISH-003 | `scripts/experiment_317_hf_publish.py` | `tests/python/test_experiment_317_hf_publish.py` | Implemented |
| REQ-PUBLISH-004 | `scripts/experiment_330_hf_live_publish.py` | `tests/python/test_experiment_330_hf_live_publish.py` | Implemented |
