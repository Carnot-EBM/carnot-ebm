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
`/home/ianblenke/github.com/Carnot-EBM/carnot-ebm`, carry the Exp 1293 blocked
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
`/home/ianblenke/github.com/Carnot-EBM/carnot-ebm`.

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
and project root `/home/ianblenke/github.com/Carnot-EBM/carnot-ebm`.

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
and project root `/home/ianblenke/github.com/Carnot-EBM/carnot-ebm`.

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


### REQ-REPORT-066: Milestone .122 Archive and .123 State Artifact

The Exp 1601 workflow shall archive milestone `2026.05.122` and confirm
milestone `2026.05.123` is the active roadmap state by writing
`results/experiment_1601_archive.json`.

The workflow shall first write the JSON artifact with `status="in_progress"`.
The terminal artifact shall include these top-level fields:

- `status`
- `milestone`
- `predecessor_milestone`
- `predecessor_archived`
- `predecessor_task_count`
- `predecessor_tasks_terminal`
- `active_roadmap_milestone`
- `active_roadmap_task_count`
- `first_active_task_id`
- `status_moved_to_changelog`
- `setup_123_state`
- `missing_task_deliverables`
- `research_roadmap_yaml_modified`
- `scripts_research_conductor_modified`
- `honest_verdict`

The workflow shall treat `research-complete.yaml` as the milestone archive
source of truth, record every `.122` task id/title/deliverable/result, confirm
the active roadmap is `2026.05.123`, confirm Exp 1601 is the first active task,
and confirm `ops/changelog.md` contains the milestone `.122` status transfer.
It shall record missing task deliverables without fabricating their contents.
It shall not modify `research-roadmap.yaml` or `scripts/research_conductor.py`.

### SCENARIO-REPORT-066: Exp 1601 Archives .122 and Activates .123 State

**Given** `research-complete.yaml` contains milestone `2026.05.122` with
Exp 1588 through Exp 1600 tasks
**And** the active `research-roadmap.yaml` declares milestone `2026.05.123`
with Exp 1601 as its first task
**And** `ops/changelog.md` contains the milestone `.122` status entry
**When** the Exp 1601 archive workflow runs
**Then** it writes the required REQ-REPORT-066 fields
**And** `predecessor_archived`, `predecessor_tasks_terminal`,
`status_moved_to_changelog`, and `setup_123_state` are true
**And** any missing task deliverable is listed in `missing_task_deliverables`
without blocking the archive when `research-complete.yaml` reports the task
terminal.
**And** `research-roadmap.yaml` and `scripts/research_conductor.py` remain
unchanged by the workflow.


### REQ-REPORT-067: Milestone .123 Archive and .124 State Artifact

The Exp 1614 workflow shall archive milestone `2026.05.123` and confirm
milestone `2026.05.124` is the active roadmap state by writing
`results/experiment_1614_archive.json`.

The workflow shall first write the JSON artifact with `status="in_progress"`.
The terminal artifact shall include these top-level fields:

- `status`
- `milestone`
- `predecessor_milestone`
- `predecessor_archived`
- `predecessor_task_count`
- `predecessor_tasks_terminal`
- `active_roadmap_milestone`
- `active_roadmap_task_count`
- `first_active_task_id`
- `status_moved_to_changelog`
- `setup_124_state`
- `missing_task_deliverables`
- `research_roadmap_yaml_modified`
- `scripts_research_conductor_modified`
- `honest_verdict`

The workflow shall treat `research-complete.yaml` as the milestone archive
source of truth, record every `.123` task id/title/deliverable/result, confirm
the active roadmap is `2026.05.124`, confirm Exp 1614 is the first active task,
and confirm `ops/changelog.md` contains the milestone `.123` status transfer.
It shall record missing task deliverables without fabricating their contents.
It shall not modify `research-roadmap.yaml` or `scripts/research_conductor.py`.

### SCENARIO-REPORT-067: Exp 1614 Archives .123 and Activates .124 State

**Given** `research-complete.yaml` contains milestone `2026.05.123` with
Exp 1601 through Exp 1613 tasks
**And** the active `research-roadmap.yaml` declares milestone `2026.05.124`
with Exp 1614 as its first task
**And** `ops/changelog.md` contains the milestone `.123` status entry
**When** the Exp 1614 archive workflow runs
**Then** it writes the required REQ-REPORT-067 fields
**And** `predecessor_archived`, `predecessor_tasks_terminal`,
`status_moved_to_changelog`, and `setup_124_state` are true
**And** any missing task deliverable is listed in `missing_task_deliverables`
without blocking the archive when `research-complete.yaml` reports the task
terminal.
**And** `research-roadmap.yaml` and `scripts/research_conductor.py` remain
unchanged by the workflow.


### REQ-REPORT-068: Milestone .126 Archive and .127 Initialization Artifact

The Exp 1652 workflow shall archive milestone `2026.05.126` and confirm
milestone `2026.05.127` is the active roadmap state by writing
`results/experiment_1652_archive.json` from `scripts/experiment_1652_archive.py`.

The workflow shall first write the JSON artifact with `status="in_progress"`.
The terminal artifact shall include these top-level fields:

- `status`
- `milestone`
- `predecessor_milestone`
- `predecessor_archived`
- `predecessor_task_count`
- `predecessor_tasks_terminal`
- `active_roadmap_milestone`
- `active_roadmap_task_count`
- `first_active_task_id`
- `status_moved_to_changelog`
- `setup_127_state`
- `nsvif_dsl_landed`
- `kv260_potts_synthesis_landed`
- `cerce_ledger_landed`
- `missing_task_deliverables`
- `research_roadmap_yaml_modified`
- `scripts_research_conductor_modified`
- `honest_verdict`

The workflow shall treat `research-complete.yaml` as the milestone archive
source of truth, record every `.126` task id/title/deliverable/result, confirm
the active roadmap is `2026.05.127`, confirm Exp 1652 is the first active task,
and confirm `ops/changelog.md` contains the milestone `.126` status transfer.
It shall record that NSVIF DSL, KV260 Potts synthesis, and CerCE ledger landed
when those tracks are present in the `.126` archive, the `.127` roadmap, and
their available result artifacts. It shall also preserve lower-level evidence
such as Vivado availability or missing bring-up deliverables without fabricating
hardware execution success. It shall not modify `research-roadmap.yaml` or
`scripts/research_conductor.py`.

### SCENARIO-REPORT-068: Exp 1652 Archives .126 and Initializes .127 State

**Given** `research-complete.yaml` contains milestone `2026.05.126` with
Exp 1640 through Exp 1651 tasks
**And** the active `research-roadmap.yaml` declares milestone `2026.05.127`
with Exp 1652 as its first task
**And** `ops/changelog.md` contains the milestone `.126` status entry
**When** the Exp 1652 archive workflow runs
**Then** it writes the required REQ-REPORT-068 fields
**And** `predecessor_archived`, `predecessor_tasks_terminal`,
`status_moved_to_changelog`, `setup_127_state`, `nsvif_dsl_landed`,
`kv260_potts_synthesis_landed`, and `cerce_ledger_landed` are true
**And** any missing task deliverable is listed in `missing_task_deliverables`
without blocking the archive when `research-complete.yaml` reports the task
terminal.
**And** `research-roadmap.yaml` and `scripts/research_conductor.py` remain
unchanged by the workflow.

### REQ-REPORT-069: Milestone .127 Operational Retrospective

The Exp 1665 workflow shall close out milestone `2026.05.127` by writing
`results/operational_retro_2026_05_127.json` from
`scripts/experiment_1665_retro.py`.

The workflow shall first write the JSON artifact with `status="in_progress"`.
The terminal artifact shall include these top-level fields:

- `status`
- `schema`
- `milestone`
- `generated_at`
- `retro_type`
- `summary`
- `total_wall_time_minutes`
- `experiments_completed`
- `task_attempts`
- `completed_task_count`
- `blocked_task_count`
- `task_outcomes`
- `slowest_experiments`
- `bottlenecks_identified`
- `improvements_suggested`
- `top_3_highest_leverage_actions`
- `estimated_time_savings_pct`
- `meta_reflection`
- `research_roadmap_yaml_modified`
- `scripts_research_conductor_modified`
- `honest_verdict`

The workflow shall read the active `.127` roadmap, the conductor log, and
available Exp 1652 through Exp 1664 deliverables. It shall classify terminal
task state from the conductor log without inventing missing successes, report
pre-gate blocks separately from completed tasks, preserve the fact that KV260
hardware execution fell back to software when hardware was unavailable, and
summarize the high-leverage operational fixes for the next milestone. It shall
not modify `research-roadmap.yaml` or `scripts/research_conductor.py`.

### SCENARIO-REPORT-069: Exp 1665 Summarizes Milestone .127 Findings

**Given** `research-roadmap.yaml` declares milestone `2026.05.127` with
Exp 1652 through Exp 1665 tasks
**And** `ops/conductor-log.md` contains Exp 1652 through Exp 1664 terminal
events with OK, GATE_BLOCK, DOOMED_RERUN_BLOCK, and recovered FAIL statuses
**When** the Exp 1665 retrospective workflow runs
**Then** it writes the required REQ-REPORT-069 fields
**And** completed tasks, blocked tasks, failed-then-completed tasks, slowest
experiments, bottlenecks, and high-leverage next actions are derived from those
source events
**And** `research-roadmap.yaml` and `scripts/research_conductor.py` remain
unchanged by the workflow.


### REQ-REPORT-1876: Milestone .146 Completion Ledger and .147 Gate Contract

The Exp 1876 workflow shall archive actionable milestone `2026.05.146`
evidence into `results/experiment_1876_146_completion_147_gate_contract.json`
without modifying `research-roadmap.yaml` or `scripts/research_conductor.py`.

The workflow shall read the Exp 1864, 1868, 1869, 1871, and 1872 result
artifacts, the Exp 1864 through Exp 1875 conductor-log entries, the `.147`
roadmap, the `.147` planning proposal, and the changelog. It shall classify
schema-complete evidence separately from malformed-but-actionable evidence,
record downstream blocks caused by missing standard gate fields, and list
scopes that must not be rerun without a changed root cause.

The terminal artifact shall include these top-level fields:

- `status`
- `honest_verdict`
- `milestone_146_archived`
- `artifact_schema_contract_ready`
- `prior_failure_carryforward_ready`
- `gate_contract_ready`
- `blocked_scope_summary`

The artifact shall set `artifact_schema_contract_ready=true` only when the
missing-field failures are explicitly named and the next milestone has an
explicit gate field for artifact normalization. It shall set
`prior_failure_carryforward_ready=true` only when blocked or retired scopes are
carried forward with a changed-root-cause policy instead of being silently
rerun.

### SCENARIO-REPORT-1876: Exp 1876 Preserves Gate-Field Failures

**Given** `.146` contains useful ROCE, LTLZinc, HILED, S2KAN, and Ising
consensus evidence
**And** the conductor log records missing `status` field gate blocks for ROCE
and HILED downstream tasks
**When** the Exp 1876 workflow runs
**Then** it writes the required REQ-REPORT-1876 fields
**And** records usable evidence, malformed evidence, missing-field blocks, and
do-not-rerun scopes in machine-readable lists
**And** `research-roadmap.yaml` and `scripts/research_conductor.py` remain
unchanged by the workflow.


### REQ-REPORT-1877: ROCE/HILED Artifact Contract Normalization

The Exp 1877 workflow shall read malformed-but-actionable ROCE and HILED source
artifacts from `results/experiment_1864_roce.json` and
`results/experiment_1869_hiled.json`, preserve their raw metrics, and write a
standard Carnot wrapper artifact to
`results/experiment_1877_artifact_contract_normalization.json`.

The terminal artifact shall include these top-level fields:

- `status`
- `honest_verdict`
- `gate_contract_normalization_ready`
- `roce_success_rate`
- `hiled_simulator_ready`
- `normalized_artifacts`
- `tests_run`

The workflow shall set `gate_contract_normalization_ready=true` only when both
source artifacts are readable, both normalized wrappers expose standard
`status` and `honest_verdict` fields, the top-level ROCE success-rate gate is
numeric, and the top-level HILED simulator gate is true.

### SCENARIO-REPORT-1877: Exp 1877 Normalizes ROCE and HILED Gate Fields

**Given** Exp 1864 and Exp 1869 produced useful evidence without standard
Carnot gate fields
**When** the Exp 1877 workflow normalizes those source artifacts
**Then** it writes the required REQ-REPORT-1877 fields
**And** the normalized ROCE wrapper preserves the raw ROCE metrics while
exposing `status`, `honest_verdict`, and `roce_success_rate`
**And** the normalized HILED wrapper preserves the raw HILED metrics while
exposing `status`, `honest_verdict`, and `hiled_simulator_ready`.


### REQ-REPORT-1889: Milestone .147 Research and Operational Retrospective

The Exp 1889 workflow shall read the available milestone `2026.05.147`
artifacts and conductor-log events, reconcile completed work versus blocked,
retired, and missing scopes, and write
`results/experiment_1889_milestone_147_retro.json` without modifying
`research-roadmap.yaml`, `ops/status.md`, `ops/changelog.md`, or
`scripts/research_conductor.py`.

The workflow shall classify prompt-to-validator, telemetry, FR-11, and
hardware-accounting gates from explicit source fields or gate-block evidence
only. Missing artifacts shall be reported as blocked or missing evidence, not
as successful work.

The terminal artifact shall include these top-level fields:

- `status`
- `honest_verdict`
- `milestone_147_retro_complete`
- `completed_task_count`
- `blocked_task_count`
- `next_gate_recommendations`
- `tests_run`

### SCENARIO-REPORT-1889: Exp 1889 Reconciles .147 Evidence Without Planning .148

**Given** `.147` completed contract normalization, ROCE validator-tree, and
BEAVER-lite artifacts
**And** live SOTA, telemetry, FR-11, hardware-accounting, and integrated E2E
work is blocked or absent in the artifact set
**When** the Exp 1889 retrospective workflow runs
**Then** it writes the required REQ-REPORT-1889 fields
**And** records the gate readiness for prompt-to-validator, telemetry, FR-11,
and hardware accounting from source evidence
**And** it recommends next gate fields without creating a `.148` plan.


### REQ-REPORT-1890: Milestone .147 Completion to .148 Activation Contract

The Exp 1890 workflow shall read the `.147` closeout artifacts, current ops
entries, and `.148` roadmap proposal, then write
`results/experiment_1890_147_completion_148_activation_contract.json` without
modifying `research-roadmap.yaml`, `ops/status.md`, `ops/changelog.md`, or
`scripts/research_conductor.py`.

The workflow shall carry forward ready substrate fields from explicit source
artifacts only: validator-tree readiness from Exp 1878 and BEAVER-lite bounds
readiness from Exp 1879. It shall carry forward blocked gates from Exp 1889 and
ops evidence only: missing mandated live SOTA cache/runtime models, missing
terminal telemetry artifact, missing FR-11 ledger artifact, and missing
hardware-accounting artifact.

The terminal artifact shall include these top-level fields:

- `status`
- `honest_verdict`
- `milestone_147_archived`
- `validator_tree_ready`
- `beaver_bounds_ready`
- `live_sota_blocked_missing_models`
- `telemetry_missing_terminal_artifact`
- `fr11_ledger_missing_terminal_artifact`
- `hardware_accounting_missing_terminal_artifact`
- `same_title_compute_dedupe_required`
- `next_gate_contract_ready`
- `tests_run`

The artifact shall set `same_title_compute_dedupe_required=true` only when the
operational retrospective records same-title compute-bound terminal-state
dedupe, GPU/model-count telemetry, and the `.147` 11 percent wall-time savings
target. It shall set `next_gate_contract_ready=true` only when `.147` is
archived, the validator-tree and BEAVER-lite substrate is ready, every blocked
gate above is explicitly captured, and `.148` gate-planning text names the
activation contract path.

### SCENARIO-REPORT-1890: Exp 1890 Activates .148 Gates From .147 Evidence

**Given** Exp 1878 and Exp 1879 completed the validator-tree and BEAVER-lite
substrate
**And** Exp 1889 reports live SOTA, telemetry, FR-11, and hardware-accounting
gates as blocked or missing
**And** the operational retrospective records the 11 percent dedupe and
GPU/model-count telemetry speed target
**When** the Exp 1890 workflow runs
**Then** it writes the required REQ-REPORT-1890 fields
**And** records ready substrate, blocked gate fields, and operational speedups
in machine-readable structures for `.148` downstream gates.


### REQ-REPORT-1903: Milestone .148 Retrospective Artifact

The Exp 1903 workflow shall read all available Exp 1890 through Exp 1902 result
JSON artifacts, the active `.148` roadmap, and milestone conductor-log entries,
then write `results/experiment_1903_milestone_148_retro.json` without
modifying `research-roadmap.yaml`, `ops/status.md`, `ops/changelog.md`, or
`scripts/research_conductor.py`.

The workflow shall classify completed tasks, structured blocked tasks, retired
pre-emptive gate skips, and unexpected technical failures separately. Missing
artifacts caused by expected structured gate skips shall not be conflated with
unexpected missing-artifact failures from failed synthesis or pre-test runs.

The terminal artifact shall include these top-level fields:

- `status`
- `honest_verdict`
- `milestone_148_retro_complete`
- `completed_task_count`
- `blocked_task_count`
- `failed_task_count`
- `same_title_compute_dedupe_result`
- `next_gate_recommendations`
- `tests_run`

The artifact shall report whether terminal telemetry, FR-11, and
hardware-accounting artifacts exist, whether the SOTA cache/runtime gap closed,
and whether the `.147` target for same-title compute-bound terminal-state dedupe
plus GPU/model-count telemetry produced the expected operational speedup.

### SCENARIO-REPORT-1903: Exp 1903 Separates .148 Gate Skips From Failures

**Given** Exp 1890 completed, Exp 1894 and Exp 1901 wrote structured blocked
gate artifacts, several downstream tasks were pre-emptively skipped because
upstream tasks retired, and several ungated tasks failed without terminal
artifacts
**When** the Exp 1903 retrospective workflow runs
**Then** it writes the required REQ-REPORT-1903 fields
**And** separates expected structured gate skips from unexpected
missing-artifact failures
**And** reports unresolved SOTA cache/runtime, telemetry, FR-11, and
hardware-accounting gaps without claiming the `.147` operational speedup target
was proven.


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
| REQ-REPORT-066 | `python/carnot/reporting/milestone_123_archive.py`, `results/experiment_1601_archive.json` | `tests/python/test_milestone_123_archive.py` | Implemented |
| REQ-REPORT-067 | `python/carnot/reporting/milestone_124_archive.py`, `results/experiment_1614_archive.json` | `tests/python/test_milestone_124_archive.py` | Implemented |
| REQ-REPORT-068 | `scripts/experiment_1652_archive.py`, `results/experiment_1652_archive.json` | `tests/python/test_experiment_1652_archive.py` | Planned |
| REQ-REPORT-069 | `scripts/experiment_1665_retro.py`, `results/operational_retro_2026_05_127.json` | `tests/python/test_experiment_1665_retro.py` | Implemented |
| REQ-REPORT-1876 | `python/carnot/reporting/milestone_146_completion_147_gate_contract.py`, `results/experiment_1876_146_completion_147_gate_contract.json` | `tests/python/test_milestone_146_completion_147_gate_contract.py` | Implemented |
| REQ-REPORT-1877 | `python/carnot/reporting/artifact_contract_normalization.py`, `results/experiment_1877_artifact_contract_normalization.json` | `tests/python/test_artifact_contract_normalization.py` | Implemented |
| REQ-REPORT-1889 | `python/carnot/reporting/milestone_retro_147.py`, `results/experiment_1889_milestone_147_retro.json` | `tests/python/test_milestone_retro_147.py` | Implemented |
| REQ-REPORT-1890 | `python/carnot/reporting/milestone_147_completion_148_activation_contract.py`, `results/experiment_1890_147_completion_148_activation_contract.json` | `tests/python/test_milestone_147_completion_148_activation_contract.py` | Implemented |
| REQ-REPORT-1903 | `python/carnot/reporting/milestone_retro_148.py`, `results/experiment_1903_milestone_148_retro.json` | `tests/python/test_milestone_retro_148.py` | Implemented |
| REQ-REPORT-2008 | `python/carnot/reporting/milestone_156_archive_157_activation.py`, `results/experiment_2008_archive_156_activate_157.json` | `tests/python/test_milestone_156_archive_157_activation.py` | Implemented |
| REQ-REPORT-024 | `python/carnot/reporting/agent_usage.py`, `scripts/agent_plan_usage.py` | `tests/python/test_agent_plan_usage.py` | Implemented |
| REQ-PUBLISH-003 | `scripts/experiment_317_hf_publish.py` | `tests/python/test_experiment_317_hf_publish.py` | Implemented |
| REQ-PUBLISH-004 | `scripts/experiment_330_hf_live_publish.py` | `tests/python/test_experiment_330_hf_live_publish.py` | Implemented |

### REQ-REPORT-124: Milestone .124 Terminal Retrospective

The pipeline SHALL evaluate all tasks from the .124 milestone and generate a single consolidated retrospective artifact containing outcome status, gate failures, and the overall criteria\_met count.

### SCENARIO-REPORT-124

Given the .124 artifact sources in `results/` (experiments 1614 to 1625), when Exp 1626 runs, then it writes all required REQ-REPORT-124 fields, reports the honest\_verdict, and scores the milestone pass/fail ratio accurately based on terminal artifact presence and `status` flag.

### REQ-REPORT-1744: Phase 4 Synthesis Latency Tradeoff

The pipeline SHALL evaluate the results of Exp 1743 and produce a JSON artifact containing the scatter data for token latency vs repair success.

### SCENARIO-REPORT-1744

Given the Exp 1743 artifact source, when Exp 1744 runs, then it evaluates the latency overhead and accuracy gain, writes the findings to `results/experiment_1744_impact.json`, and records the scatter plot data.


### REQ-REPORT-1745: Milestone .134 Synthesis Retrospective

The Exp 1745 milestone .134 retrospective workflow shall parse the results of Phase 1-3, summarize findings regarding hardware resolution, continuous learning scale-up, and System-2 EqM accuracy, and identify gaps for milestone .135. It shall write `results/experiment_1745_retro.json` containing:

- `milestone` set to `2026.05.134`
- `hardware_resolution` summarizing hardware findings
- `continuous_learning_scale_up` summarizing continuous learning
- `system_2_eqm_accuracy` summarizing System-2 EqM accuracy
- `gaps_for_135` listing gaps for the next milestone
- `honest_verdict` recording the overall outcome

### SCENARIO-REPORT-1745: Exp 1745 Generates Phase 4 Synthesis Retrospective

**Given** the completion of Phase 1-3 experiments up to 1744
**When** the Exp 1745 workflow runs
**Then** it writes all required REQ-REPORT-1745 fields
**And** identifies clear gaps for milestone .135.

### REQ-REPORT-1758: Milestone .135 Synthesis Retrospective

The Exp 1758 milestone .135 retrospective workflow shall parse the results of Phase 5 Operations (experiments 1746 to 1757), summarize findings, and identify new gaps. It shall write `results/experiment_1758_retro.json` containing:

- `milestone` set to `2026.05.135`
- `honest_verdict` recording the overall outcome
- `new_gaps` listing gaps for the next milestone
- `details` extracted from the aggregated JSONs.

### SCENARIO-REPORT-1758: Exp 1758 Generates Phase 5 Synthesis Retrospective

**Given** the completion of Phase 5 Operations experiments (1746-1757)
**When** the Exp 1758 workflow runs
**Then** it writes all required REQ-REPORT-1758 fields to `results/experiment_1758_retro.json`
**And** details `honest_verdict` and `new_gaps`.

### REQ-REPORT-1770: Milestone Phase 4 Operations Retrospective

The Exp 1770 retrospective workflow shall read the authoritative artifacts from Exp 1759 through Exp 1769.
The workflow shall parse the result JSONs and write an aggregate retrospective to `results/experiment_1770_retro.json` detailing the `honest_verdict`.



### REQ-REPORT-1798: Milestone .138 Phase 4 Operations Retrospective

The Exp 1798 retrospective workflow shall read the authoritative artifacts from Exp 1785 through Exp 1797.
The workflow shall parse the result JSONs and write an aggregate retrospective to `results/experiment_1798_retro.json` detailing the `honest_verdict`.

### SCENARIO-REPORT-1798: Exp 1798 Generates Phase 4 Synthesis Retrospective

**Given** the completion of Phase 4 Operations experiments (1785-1797)
**When** the Exp 1798 workflow runs
**Then** it writes all required REQ-REPORT-1798 fields to `results/experiment_1798_retro.json`
**And** details `honest_verdict`.

### REQ-REPORT-1824: Milestone .141 Final Evaluation Retrospective

The Exp 1824 retrospective workflow shall read the authoritative artifacts from Exp 1814 through Exp 1823.
The workflow shall parse the result JSONs and write an aggregate retrospective to `results/experiment_1824_retro.json` detailing the `honest_verdict`, hardware integration results, online distillation metrics, and `top_3_gaps`.

### SCENARIO-REPORT-1824: Exp 1824 Generates Phase 18 Final Evaluation Retrospective

**Given** the completion of Phase 18 Phase 4 Operations experiments (1814-1823)
**When** the Exp 1824 workflow runs
**Then** it writes all required REQ-REPORT-1824 fields to `results/experiment_1824_retro.json`
**And** details `honest_verdict` and top 3 gaps.

### REQ-REPORT-1838: Milestone .142 Retrospective

The Exp 1838 retrospective workflow shall read the authoritative artifacts from Exp 1825 through Exp 1837.
The workflow shall parse the result JSONs and write an aggregate retrospective to `results/experiment_1838_retro.json` detailing the `honest_verdict`.

### SCENARIO-REPORT-1838: Exp 1838 Generates Milestone .142 Retrospective

**Given** the completion of experiments (1825-1837)
**When** the Exp 1838 workflow runs
**Then** it writes all required REQ-REPORT-1838 fields to `results/experiment_1838_retro.json`
**And** details `honest_verdict`.

### REQ-REPORT-1850: Milestone .143 Retrospective

The Exp 1850 retrospective workflow shall read the authoritative artifacts from Exp 1839 through Exp 1849.
The workflow shall parse the result JSONs and write an aggregate retrospective to `results/experiment_1850_retro.json` detailing the `honest_verdict`.

### SCENARIO-REPORT-1850: Exp 1850 Generates Milestone .143 Retrospective

**Given** the completion of experiments (1839-1849)
**When** the Exp 1850 workflow runs
**Then** it writes all required REQ-REPORT-1850 fields to `results/experiment_1850_retro.json`
**And** details `honest_verdict` and metrics from the results.


### REQ-REPORT-1853: Milestone .144 Retrospective

The Exp 1853 retrospective workflow shall read the authoritative artifacts from Exp 1849 through Exp 1852.
The workflow shall parse the result JSONs and write an aggregate retrospective to `results/experiment_1853_retro.json` detailing the `honest_verdict`, `tasks_summary`, `gates_passed_count`, `gates_failed_count`, and `paper_v6_carryforward_items`.

### SCENARIO-REPORT-1853: Exp 1853 Generates Milestone .144 Retrospective

**Given** the completion of experiments (1849-1852)
**When** the Exp 1853 workflow runs
**Then** it writes all required REQ-REPORT-1853 fields to `results/experiment_1853_retro.json`
**And** details `honest_verdict` and synthesized findings.

### REQ-REPORT-0863: Milestone .145 Retrospective

The Exp 1863 milestone .145 retrospective workflow shall write
`results/experiment_1863_retro.json` summarizing the pass/fail rates of the VL proxy and S2KAN tests.

The terminal artifact shall include:
- `schema` set to `carnot.milestone_research_retro.v1`
- `milestone` set to `2026.05.145`
- `vl_proxy_pass_rate`
- `s2kan_pass_rate`
- `honest_verdict` formatted as a concise milestone outcome

Missing artifacts shall count as unmet criteria.

### REQ-REPORT-1931: Milestone .150 Retrospective

The Exp 1931 milestone .150 retrospective workflow shall write
`results/experiment_1931_milestone_150_retro.json` summarizing the success and failure rates of the .150 milestone.

The terminal artifact shall include:
- `schema` set to `carnot.milestone_150_retro.v1`
- `milestone_150_retro_complete`
- `completed_task_count`
- `blocked_task_count`
- `failed_task_count`
- `next_gate_recommendations`
- `tests_run`
- `honest_verdict` formatted as a concise milestone outcome

### SCENARIO-REPORT-1931: Exp 1931 Generates Milestone .150 Retrospective

**Given** the completion of experiments (1918-1930)
**When** the Exp 1931 workflow runs
**Then** it writes all required REQ-REPORT-1931 fields to `results/experiment_1931_milestone_150_retro.json`
**And** details `honest_verdict` and synthesized findings.

### REQ-REPORT-152: Milestone .152 Terminal Retrospective

The Exp 1955 milestone .152 retrospective workflow shall write `results/experiment_1955_milestone_152_retro.json` compiling the preceding .152 experiments. It shall parse honest_verdicts from Exp 1944 through Exp 1954 and record completed, blocked, and failed task metrics. It shall include recommendations for the next milestone.

### REQ-REPORT-1967: Milestone .153 Pre-Retro Audit

The Exp 1967 milestone .153 pre-retro audit workflow shall write `results/experiment_1967_milestone_153_pre_retro_audit.json` by scanning artifacts from Exp 1956 through Exp 1966. It shall verify logprobs, artifact formatting, and deterministic check compliance, and record the missing result files or violated gates.

The terminal artifact shall include:
- `schema` set to `carnot.milestone_pre_retro_audit.v1`
- `milestone` set to `153`
- `missing_files`
- `violated_gates`
- `compliant_artifacts`
- `non_compliant_artifacts`
- `honest_verdict` formatted as a concise summary.

### SCENARIO-REPORT-1967: Exp 1967 Generates Milestone .153 Pre-Retro Audit

**Given** the completion of experiments (1956-1966)
**When** the Exp 1967 workflow runs
**Then** it writes all required REQ-REPORT-1967 fields to `results/experiment_1967_milestone_153_pre_retro_audit.json`
**And** verifies artifact compliance and details `honest_verdict`.

### REQ-REPORT-154: Exp 1980 Milestone .154 Pre-Retro Audit

The pipeline SHALL evaluate all tasks from the .154 milestone (Exp 1969 through 1979) and generate a single pre-retro artifact `results/experiment_1980_milestone_154_pre_retro.json`. It SHALL skip intentionally retired experiments 1971 and 1979 and evaluate the formatting, logprobs, and zero-false-accept bounds for the remaining experiments.

### SCENARIO-REPORT-154: Exp 1980 audits .154 artifacts

**Given** the completion of experiments 1969 through 1979 (with 1971 and 1979 retired)
**When** the Exp 1980 audit workflow runs
**Then** it writes all required REQ-REPORT-154 fields to `results/experiment_1980_milestone_154_pre_retro.json`
**And** reports the compliance and missing files accurately.


### REQ-REPORT-075: Milestone .156 Pre-Retro Audit

The Exp 2006 milestone .156 pre-retro audit workflow shall read the authoritative
Exp 1996 through Exp 2005 result JSON artifacts and write
`results/experiment_2006_milestone_156_pre_retro.json` with:

- `experiment` set to 2006
- `status` set to `success` or `failure`
- `artifacts_exist`, boolean indicating if all .156 artifacts exist
- `valid_schema_confirmed`, boolean indicating if they contain a valid schema
- `sota_models_utilized`, boolean indicating if SOTA models were confirmed to be used
- `honest_verdict` formatted as a concise audit outcome

The workflow shall verify that the .156 artifacts (1996-2005) exist and have basic schema keys, and confirm at least one utilizes SOTA models in its `model_specs` or `models_utilized` fields.

### REQ-REPORT-2008: Milestone .156 Archive and .157 Activation Artifact

The Exp 2008 milestone activation workflow shall read the authoritative
Exp 1996 through Exp 2007 result JSON artifacts, the active roadmap, the
research-complete archive, and the conductor log, then write
`results/experiment_2008_archive_156_activate_157.json` with:

- `schema` set to `carnot.milestone_156_archive_157_activation.v1`
- `milestone` set to `2026.05.157`
- `predecessor_milestone` set to `2026.05.156`
- `success`, true only when the predecessor artifacts are archived and the
  `.157` environment is active
- `previous_milestone_artifacts_archived`
- `archive_move_required`, false when `research-complete.yaml` already records
  all `.156` deliverables and the canonical `results/` artifacts remain in place
- `archive_artifacts`, listing every Exp 1996 through Exp 2007 artifact path
- `missing_artifacts`
- `milestone_environment_ready`
- `roadmap_157_active`
- `conductor_activation_logged`
- `protected_files_unchanged`
- `handoff_requirements`
- `tests_run`
- `honest_verdict` formatted as a concise terminal outcome with a
  `complete:` or `blocked:` prefix

The workflow shall not modify `scripts/research_conductor.py`. It shall not
move canonical result artifacts out of `results/` when the milestone is already
represented in `research-complete.yaml`; in that case, the terminal artifact
shall record that no archive-directory move was required.

### SCENARIO-REPORT-2008: Exp 2008 Archives .156 and Activates .157

**Given** the `.156` retro reports completion, Exp 1996 through Exp 2007
artifacts exist, `research-complete.yaml` records the `.156` deliverables, and
the active roadmap/log show milestone `2026.05.157`
**When** the Exp 2008 workflow runs
**Then** it writes all required REQ-REPORT-2008 fields to
`results/experiment_2008_archive_156_activate_157.json`
**And** `success` is true, `archive_move_required` is false, and
`milestone_environment_ready` is true.


### REQ-REPORT-157: Milestone .157 Retrospective Artifact

The Exp 2017 milestone .157 retrospective workflow shall read the authoritative
Exp 2008 through Exp 2016 result JSON artifacts, plus the `results/` artifacts 
for that range. It shall write `results/experiment_2017_milestone_157_retro.json`
with:

- `schema` set to `carnot.milestone_retro.v1`
- `milestone` set to `2026.05.157`
- `experiment_id` set to `2017`
- `status`
- `completed_experiments`
- `blocked_experiments`
- `failed_experiments`
- `completed_task_count`
- `blocked_task_count`
- `failed_task_count`
- `experiment_honest_verdicts`
- `recommendations`
- `bottlenecks_identified`
- `retro_complete`
- `honest_verdict`

Missing or unreadable artifacts in the range [2008, 2016] shall be counted as failed
unless they are known exceptions. The workflow shall identify execution bottlenecks
and gating behavior based on the blocked artifacts (specifically prior_failures missing).

### SCENARIO-REPORT-157-A: Milestone .157 Retrospective Handles Blocked and Missing Artifacts

**Given** the .157 milestone artifacts (2008-2016) contain blocked artifacts due to gate checks and missing artifacts
**When** the Exp 2017 workflow runs
**Then** it writes all required REQ-REPORT-157 fields to `results/experiment_2017_milestone_157_retro.json`
**And** it accurately categorizes 2009, 2010, 2011, 2015 as blocked, 2008, 2012, 2013, 2014 as failed (missing), and 2016 as failed.

### REQ-ORCH-RETRO-001: Milestone .158 Pre-Retro Audit

The Exp 2026 milestone .158 pre-retro audit workflow shall write `results/experiment_2026_milestone_158_pre_retro.json` by auditing the conductor log for SEAL generation and STKAN tasks, and reporting their completion statuses.

### REQ-REPORT-158: Milestone .158 Retrospective Artifact

The Exp 2027 milestone .158 retrospective workflow shall read the authoritative
Exp 2026 result JSON artifact. It shall write `results/experiment_2027_milestone_158_retro.json`
with:

- `schema` set to `carnot.milestone_retro.v1`
- `milestone` set to `2026.05.158`
- `experiment_id` set to `2027`
- `status`
- `seal_success`
- `stkan_success`
- `recommendations`
- `retro_complete`
- `honest_verdict`

### SCENARIO-REPORT-158-A: Milestone .158 Retrospective Analyzes SEAL and STKAN

**Given** the .158 milestone pre-retro audit artifact (2026) contains the SEAL and STKAN task completion statuses
**When** the Exp 2027 workflow runs
**Then** it writes all required REQ-REPORT-158 fields to `results/experiment_2027_milestone_158_retro.json`
**And** it accurately sets `seal_success` and `stkan_success` based on the pre-retro audit, and provides recommendations for the next milestone.

### REQ-REPORT-2065: Milestone 161 Retrospective

The Exp 2065 milestone 161 retrospective workflow shall write
`results/experiment_2065_retro.json` documenting the findings of the EBT,
Soft Bellman, and TSU integrations. The artifact shall include:

- `milestone` set to `2026.05.161`
- A summary of the performance of the separated Mouth/Brain system, clarifying
  that verification (Brain) can run independently of language generation (Mouth).
- A summary of TSU hardware readiness, affirming that the Extropic TSU interface
  is validated via simulation, while authentic hardware execution claims are deferred
  until physical deployment.
- Formatting adhering to the standard `carnot.milestone_retro.v1` schema.
- `honest_verdict` reflecting the successful integration analysis.

### REQ-REPORT-2089: Milestone 163 Retrospective

The Exp 2089 milestone 163 retrospective workflow shall summarize the performance of SMT/LLM constraint extraction and JEPA scaffolding. It shall write `results/experiment_2089_retro.json` with the milestone schema, `criteria_met`, `notable_successes`, and `honest_verdict`.


### REQ-REPORT-168: Milestone 2026.05.168 Terminal Retrospective

The Exp 1679 `.168` milestone retrospective workflow shall write `results/experiment_1679_retro.json` containing:
- schema set to `carnot.milestone_research_retro.v1`
- milestone set to `2026.05.168`
- `tasks_summary` extracting hypothesis, gate threshold, empirical result, and surprising finding for Exp 1675 through 1678.
- `gates_passed_count` and `gates_failed_count`
- `actual_agent_backend_distribution`
- `paper_v6_carryforward_items` highlighting the first physical-board sovereignty data point (Exp 1676)
- `hardware_sovereignty_data_points` mapping board to gate_passed
- `adversarial_verify_flag_count` from `scripts/adversarial_verify.py`
- `honest_verdict` prefixed with `complete:`

### REQ-REPORT-169: Milestone 2026.05.169 Retrospective Artifact
The Exp 1684 `.169` milestone retrospective workflow shall write `results/experiment_1684_retro.json` containing:
- schema set to `carnot.milestone_research_retro.v1`
- milestone set to `2026.05.169`
- `tasks_summary` extracting hypothesis, gate threshold, empirical result, and carryforward for Exp 1680 through 1683.
- `gates_passed_count` and `gates_failed_count`
- `actual_agent_backend_distribution`
- `paper_v6_carryforward_items` highlighting the paper-v6 §6 disclosure required for THRML bias systematic underestimate (Exp 1682)
- `phase1_ship_progress_pp_remaining` showing delta from .168 to .169
- `adversarial_verify_flag_count` from `scripts/adversarial_verify.py`
- `honest_verdict` prefixed with `complete:`

### REQ-REPORT-1717: Findings Audit 1717

The repository shall provide an audit script `scripts/audit_1717.py` that verifies experiments in the `.174` and `.175` ranges, classifies any adversarial flags, and appends a `corrigendum_2026_05_176_audit` to flagged artifacts. It must output `results/experiment_1717_findings_audit.json`.


### REQ-REPORT-1796: Findings Audit 1796
**Given** the artifact sources in `results/` for .186 and .187,
**When** Exp 1796 runs,
**Then** it writes all required REQ-REPORT-1796 fields, identifies flagged artifacts using `adversarial_verify.py`, applies correct classifications, appends corrigenda fields to the flagged artifacts, and produces an honest_verdict prefixed with `complete:`.

### REQ-REPORT-194: Milestone .194 Operational Retrospective
The repository shall provide a script to generate the .194 operational retrospective JSON, outputting to `results/operational_retro_2026_05_194.json`. It must contain a schema of `carnot.operational_retro.v64`, milestone `2026.05.194`, and specific flag fields `adversarial_confirmation_result`, `pypi_ship_result`, and `phase4_closure_result`.

### REQ-REPORT-195: Milestone .195 Initialization Artifact
The repository shall provide a module `carnot.reporting.experiment_1914_init` to generate the .195 initialization artifact JSON, outputting to `results/experiment_1914_init.json`. It must contain a schema of `carnot.init.v1`, experiment `1914`, `status_updated` boolean, and `honest_verdict` formatted as `complete: initialized .195`.

### REQ-REPORT-1924: Milestone .195 Terminal Retrospective
The repository shall provide a module `carnot.reporting.experiment_1924_retro` and a test `tests/python/test_experiment_1924_retro.py` that outputs the milestone retrospective to `results/experiment_1924_retro.json` containing schema `carnot.retro.v1`, experiment `1924`, `retrospective_summary`, and `honest_verdict` formatted as `complete: .195 finished`.

### REQ-REPORT-2001: Findings Audit 2001
The repository shall provide a module `carnot.pipeline.findings_audit_2001` and a test `tests/python/test_experiment_2001_findings_audit_198.py` to verify the `.198` artifact range (experiments 1980 through 1986). It shall output `results/experiment_2001_findings_audit_198.json` with schema `carnot.findings_audit_corrigenda.v11`, experiment `2001`, and `honest_verdict` starting with a terminal prefix (e.g. `complete:`).


### REQ-REPORT-2010: Consolidated Phase 1 Audit
The pipeline SHALL run a consolidated audit for milestones .198 to .201, generate a Phase 1 dashboard, and write results/experiment_2010_consolidated_audit.json.


### REQ-REPORT-2010: Consolidated Phase 1 Audit
The pipeline SHALL run a consolidated audit for milestones .198 to .201, generate a Phase 1 dashboard, and write results/experiment_2010_consolidated_audit.json.

### REQ-REPORT-2265: Milestone 2026.05.223 Operational Retrospective

The repository shall provide `scripts/experiment_2265_retro.py` to generate `results/experiment_2265_retro.json` with schema `carnot.operational_retro.v66`.
The artifact must record `total_wall_time_min`, `n_experiments_completed`, `n_gate_blocks`, `n_compute_bound`, `criteria_met`, `top_gaps_resolved`, and `next_milestone_speedup_target_pct`.
The `honest_verdict` field must start with a terminal prefix and the `.222` gap resolution analysis must explicitly cover the pre-test fix, KAN-CL n=256 validation, and live generation beyond the one-token probe.

#### SCENARIO-REPORT-2265: Generate .223 Retrospective Artifact

**Given** a conductor log containing the milestone 2026.05.223 activation and terminal task rows
**And** result artifacts for the completed or blocked .223 experiments
**When** the Exp 2265 retrospective generator runs
**Then** it writes `results/experiment_2265_retro.json` with schema `carnot.operational_retro.v66`, terminal-prefixed `honest_verdict`, completion fraction in `criteria_met`, all three `.222` gap closure records, and a quantified speedup target for milestone `.224`.

### REQ-REPORT-2609: Sklearn Prerequisite Fix Artifact

The repository shall provide a helper module `carnot.reporting.sklearn_prereq_fix_2609` and a focused test `tests/python/test_experiment_2609_sklearn_prereq_fix.py` for generating the Exp 2609 prerequisite artifact. The artifact must be written to `results/experiment_2609_sklearn_prereq_fix.json`, use a terminal-prefixed `honest_verdict`, record whether sklearn was already installed, record the sklearn version, verify the Carnot import chain, and resolve the FoVer corpus path when a corpus with more than 100 rows is available.

#### SCENARIO-REPORT-2609: Generate sklearn prerequisite artifact

**Given** the active conductor Python environment and project root
**When** the Exp 2609 helper builds the prerequisite artifact
**Then** it writes all required sklearn, Carnot import, FoVer corpus, installation audit, and `preconditions_checked` fields needed by downstream verifier-recovery tasks.

### REQ-REPORT-2321: Milestone 2026.05.227 Operational Retrospective

The repository shall provide `scripts/experiment_2321_retro.py` to generate
`results/experiment_2321_retro.json` with schema
`carnot.operational_retro.v70`.

The artifact must record `total_wall_time_min`, `n_experiments_completed`,
`n_gate_blocks`, `n_compute_bound`, `criteria_met`, `top_gaps_resolved`,
`pretest_cascade_status`, and `next_milestone_speedup_target_pct`. The
`honest_verdict` field must start with a terminal prefix. The gap-resolution
analysis must explicitly cover the `.226` carry-forward gaps: full pre-test
cascade resolution, FST live generation beyond one-token probing, and NSVIF
neuro-symbolic extraction execution. If Exp 2309 did not set
`pretest_fixed=true`, the artifact must include direct operator escalation
commands for the two named pre-test failures.

#### SCENARIO-REPORT-2321: Generate .227 Retrospective Artifact

**Given** a conductor log containing the milestone 2026.05.227 activation and terminal task rows
**And** result artifacts or missing-artifact evidence for Exp 2309, Exp 2310, Exp 2312, Exp 2313, and Exp 2320
**When** the Exp 2321 retrospective generator runs
**Then** it writes `results/experiment_2321_retro.json` with schema `carnot.operational_retro.v70`, terminal-prefixed `honest_verdict`, completion fraction in `criteria_met`, all three `.226` gap closure records, explicit `pretest_cascade_status`, and a quantified speedup target for milestone `.228`.

### REQ-REPORT-2335: Milestone 2026.05.228 Operational Retrospective

The repository shall provide `scripts/experiment_2335_retro.py` to generate
`results/experiment_2335_retro.json` with schema
`carnot.operational_retro.v71`.

The artifact must record `total_wall_time_min`, `n_experiments_completed`,
`n_gate_blocks`, `n_compute_bound`, `criteria_met`, `top_gaps_resolved`,
`pretest_cascade_status`, and `next_milestone_speedup_target_pct`. The
`honest_verdict` field must start with a terminal prefix. The gap-resolution
analysis must explicitly cover the `.228` design gaps: full pre-test cascade
resolution, NSVIF neuro-symbolic extraction first actual run, and FST live
generation beyond one-token probing. If Exp 2323 did not set
`pretest_fixed=true`, the artifact must include direct operator escalation
commands for the Exp 2309 and Exp 2323 pre-test failures, plus a recommendation
to consider a no-xdist pre-test fallback.

#### SCENARIO-REPORT-2335: Generate .228 Retrospective Artifact

**Given** a conductor log containing the milestone 2026.05.228 activation and terminal task rows
**And** result artifacts or missing-artifact evidence for Exp 2323, Exp 2324, Exp 2326, Exp 2327, and Exp 2334
**When** the Exp 2335 retrospective generator runs
**Then** it writes `results/experiment_2335_retro.json` with schema `carnot.operational_retro.v71`, terminal-prefixed `honest_verdict`, completion fraction in `criteria_met`, all three `.228` gap closure records, explicit `pretest_cascade_status`, and a quantified speedup target for milestone `.229`.

### REQ-REPORT-2349: Milestone 2026.05.229 Operational Retrospective

The repository shall provide `scripts/experiment_2349_retro.py` to generate
`results/experiment_2349_retro.json` with schema
`carnot.operational_retro.v72`.

The artifact must record `total_wall_time_min`, `n_experiments_completed`,
`n_gate_blocks`, `n_compute_bound`, `criteria_met`, `top_gaps_resolved`,
`pretest_cascade_status`, `ungated_tasks_completed`, and
`next_milestone_speedup_target_pct`. The `honest_verdict` field must start
with a terminal prefix. The gap-resolution analysis must explicitly cover the
`.229` design gaps: full pre-test cascade resolution, Semantic Energy Tier 0g
prototype landing, and NSVIF neuro-symbolic extraction first actual run. If Exp
2337 did not set `pretest_fixed=true`, the artifact must include operator
manual inspection commands and recommend manual inspection before milestone
`.230` activation.

#### SCENARIO-REPORT-2349: Generate .229 Retrospective Artifact

**Given** a conductor log containing the milestone 2026.05.229 activation and terminal task rows
**And** result artifacts or missing-artifact evidence for Exp 2337, Exp 2338, Exp 2339, Exp 2341, Exp 2342, and Exp 2348
**When** the Exp 2349 retrospective generator runs
**Then** it writes `results/experiment_2349_retro.json` with schema `carnot.operational_retro.v72`, terminal-prefixed `honest_verdict`, completion fraction in `criteria_met`, all three `.229` gap closure records, explicit `pretest_cascade_status`, `ungated_tasks_completed`, and a quantified speedup target for milestone `.230`.

### REQ-REPORT-2389: Paper-v6 Real-Data Results Table

The repository shall provide a paper-v6 results-table compiler that reads the
available milestone `.231` and `.232` result artifacts, records missing source
artifacts honestly, writes `docs/paper_v6_results_table.md`, and emits
`results/experiment_2389_paperv6_table.json`.

The terminal artifact must include:

- `honest_verdict`, prefixed with `complete:` and including
  `n_paper_ready_results`
- `n_paper_ready_results`, counting only local results with `n_examples >= 30`
  and no `IMPLAUSIBLE_PERFECT` provenance
- `best_auroc_achieved`, the highest AUROC across available Carnot Tier 0
  verifier rows
- `hallscan_gap`, computed as `0.88 - best_auroc_achieved`
- `results_table_written`
- `n_missing_results`
- `duration_s`

The compiler must include HalluScan AUROC `0.88` and HIVE AUROC `0.9236` as
external baselines without counting them as local paper-ready Carnot results.

#### SCENARIO-REPORT-2389: Compile Paper-v6 Table From Available Artifacts

**Given** some expected `.231` and `.232` result artifacts may be absent from
`results/`
**When** the Exp 2389 compiler runs
**Then** it writes `docs/paper_v6_results_table.md` with columns
`metric_name`, `value`, `n_examples`, `paper_ready`, `external_baseline`, and
`gap_to_baseline`
**And** it writes `results/experiment_2389_paperv6_table.json` with all
REQ-REPORT-2389 required fields, source-artifact availability, methodology
notes, and a terminal-prefixed honest verdict.

### REQ-REPORT-2378: Archive Milestone 2026.05.231 and Activate 2026.05.232

The repository shall provide an Exp 2378 archive generator that writes
`results/experiment_2378_archive.json` with schema
`carnot.archive_activation.v1`.

The generator must read the active `research-roadmap.yaml` milestone before
making archive decisions. If the active milestone is `2026.05.231`, it must
append the `2026.05.231` entry to `research-complete.yaml` when absent and copy
`research-roadmap-next.yaml` over `research-roadmap.yaml` to activate
`2026.05.232`. If the active milestone is already `2026.05.232`, it must avoid
duplicate archive entries and set `archive_ready=true` only when
`research-complete.yaml` contains `id: 2026.05.231`. If the active milestone is
anything else, it must leave roadmap files unchanged and report the unexpected
roadmap state honestly.

The terminal artifact must include:

- `honest_verdict`, prefixed with `complete:`
- `archive_ready`, true only when `research-complete.yaml` records
  `id: 2026.05.231`
- `milestone_archived`, equal to `2026.05.231`
- precondition, archive, activation, acceptance-gate, and field-principle
  evidence sufficient for conductor reconciliation

#### SCENARIO-REPORT-2378: Idempotent .231 Archive and .232 Activation

**Given** `research-roadmap.yaml` may already have been swapped by the
conductor
**And** `research-complete.yaml` may already contain `id: 2026.05.231`
**When** the Exp 2378 archive generator runs
**Then** it writes `results/experiment_2378_archive.json` with a
terminal-prefixed `honest_verdict`, `archive_ready=true`,
`milestone_archived=2026.05.231`, no duplicate archive entry, and no
modification to `scripts/research_conductor.py`.

### REQ-REPORT-2391: Milestone 2026.05.232 Operational Retrospective

The repository shall provide `scripts/experiment_2391_retro.py` to generate
`results/experiment_2391_retro.json` with schema
`carnot.operational_retro.v75`.

The artifact must read the conductor log and milestone roadmap, record one
terminal status for each planned Exp 2378 through Exp 2391 task, and include
`honest_verdict`, `n_experiments_completed`, `n_gate_blocks`, `n_failed`,
`total_wall_time_min`, `fr11_satisfied`, `fst_live_path_ab_completed`,
`auroc_gap_to_hallscan_at_232_close`, `kv260_yosys_synthesis_succeeded`,
`phase1_ship_criteria_met`, `top_3_successes`, `top_3_gaps_for_233`, and
`retro_complete`.

Missing source artifacts must be treated as unavailable evidence rather than
as successful gates. The `honest_verdict` field must start with a terminal
prefix.

#### SCENARIO-REPORT-2391: Generate .232 Retrospective Artifact

**Given** the .232 conductor log contains repeated failed task attempts, one
paper-v6 table success, and a capstone gate block
**And** the key Exp 2382, Exp 2383, Exp 2384, Exp 2388, and Exp 2390 source
artifacts may be absent from `results/`
**When** the Exp 2391 retrospective generator runs
**Then** it writes `results/experiment_2391_retro.json` with schema
`carnot.operational_retro.v75`, terminal-prefixed `honest_verdict`, accurate
task counts, close-of-milestone AUROC gap accounting, false FR-11/FST/KV260
ship-gate booleans when evidence is absent, and `retro_complete=true`.

### REQ-REPORT-2543: Archive Milestone 2026.05.244 and Confirm 2026.05.245 Activation

The repository shall provide an Exp 2543 archive generator that writes
`results/experiment_2543_archive.json` with schema
`carnot.archive_activation.v1`.

The generator must read the active `research-roadmap.yaml` milestone before
making archive decisions. If the active milestone is `2026.05.244`, it must
append the `2026.05.244` entry to `research-complete.yaml` when absent and copy
`research-roadmap-next.yaml` over `research-roadmap.yaml` to activate
`2026.05.245`. If the active milestone is already `2026.05.245`, it must avoid
duplicate archive entries, confirm that `research-complete.yaml` contains
`id: 2026.05.244`, and report `archive_ready=true` only after that confirmation.
If the active milestone is anything else, it must leave roadmap files unchanged
and report the unexpected roadmap state honestly.

The terminal artifact must include:

- `honest_verdict`, prefixed with `complete:`
- `archive_ready`, true only when `research-complete.yaml` records
  `id: 2026.05.244`
- `milestone_archived`, equal to `2026.05.244`
- `execution_gap_diagnosis`, preserving the exp2530-exp2534 root-cause
  hypothesis for .245 process improvement
- `preconditions_checked`, recording the milestone and archive checks that
  guarded activation
- `duration_s`, a wall-clock duration for the archive generator run

#### SCENARIO-REPORT-2543: Idempotent .244 Archive and .245 Activation

**Given** `research-roadmap.yaml` may already have been swapped by the
conductor
**And** `research-complete.yaml` may already contain `id: 2026.05.244`
**When** the Exp 2543 archive generator runs
**Then** it writes `results/experiment_2543_archive.json` with a
terminal-prefixed `honest_verdict`, `archive_ready=true`,
`milestone_archived=2026.05.244`, the execution-gap diagnosis, no duplicate
archive entry, and no modification to `scripts/research_conductor.py`.

### REQ-REPORT-2827: Archive Milestone 2026.05.267 and Confirm 2026.05.268 Activation

The repository shall provide an Exp 2827 archive generator that writes
`results/experiment_2827_archive_v267.json` with schema
`carnot.archive_activation.v1`.

The generator must read the active `research-roadmap.yaml` milestone before
making archive decisions and must confirm `2026.05.268` is active. It must read
`research-complete.yaml`, avoid duplicate `2026.05.267` archive entries, and
ensure the archive row honestly records the partial .267 outcome:

- Exp 2819 through Exp 2822 were skipped because of the pre-restart
  gemini-cli crash storm.
- Exp 2823 was retired as fabricated and moved to `legacy/fabricated/` with an
  exclusion-manifest entry.
- Exp 2824, Exp 2825, and Exp 2826 produced non-fabricated artifacts.

The terminal artifact must include:

- `honest_verdict`, prefixed with `complete:` or `complete_`
- `archived_milestone`, equal to `2026.05.267`
- `archived_milestone_experiments_completed`, equal to `3`
- `activated_milestone`, equal to `2026.05.268`
- `duration_s`, a wall-clock duration for the archive generator run

The generator must not modify `scripts/research_conductor.py`.

#### SCENARIO-REPORT-2827: Idempotent .267 Partial Archive and .268 Activation

**Given** `research-roadmap.yaml` is already active on `2026.05.268`
**And** `research-complete.yaml` may already contain a generic
`2026.05.267` archive row
**When** the Exp 2827 archive generator runs
**Then** it writes `results/experiment_2827_archive_v267.json` with a
terminal-prefixed `honest_verdict`, `archived_milestone=2026.05.267`,
`archived_milestone_experiments_completed=3`,
`activated_milestone=2026.05.268`, no duplicate archive entry, and no
modification to `scripts/research_conductor.py`.

### REQ-REPORT-2835: Archive Milestone 2026.05.268 and Confirm 2026.05.269 Activation

The repository shall provide an Exp 2835 archive generator that writes
`results/experiment_2835_archive_v268.json` with schema
`carnot.archive_activation.v1`.

The generator must read `research-roadmap.yaml` before making archive
decisions and must report the observed milestone without modifying the
roadmap. It must read `research-complete.yaml`, avoid duplicate
`2026.05.268` archive entries, and ensure the archive row honestly records
the .268 outcomes:

- Exp 2827 completed the .267 archive and .268 activation.
- Exp 2828 was blocked by missing system `torch`/CUDA and an uncached
  mandated SOTA GGUF.
- Exp 2829, Exp 2830, and Exp 2831 were blocked by unavailable CUDA and
  uncached/missing corpus prerequisites.
- Exp 2832 completed with an empty verifier matrix because upstream corpora
  produced no measured per-verifier AUROC rows.
- Exp 2833 completed the paper table integration but was not cite-ready.
- Exp 2834 completed the capstone while leaving the FoVer-overfit thesis and
  FR-11 learning delta unconfirmed.

The terminal artifact must include:

- `honest_verdict`, prefixed with `complete:` or `success:`
- `archived_milestone`, equal to `2026.05.268`
- `activated_milestone`, equal to `2026.05.269`
- `archived_task_summary`, a dictionary that preserves each Exp 2827 through
  Exp 2834 status without converting blocked artifacts into successes
- `runtime_root_cause`, documenting why .268 live evaluations did not run
- `duration_s`, a wall-clock duration for the archive generator run

The generator must not modify `research-roadmap.yaml` or
`scripts/research_conductor.py`. Ops status, changelog, and traceability
reconciliation remain the conductor's follow-up step for this archive task.

#### SCENARIO-REPORT-2835: Idempotent .268 Blocked Archive and .269 Activation

**Given** `research-roadmap.yaml` is active on `2026.05.269`
**And** `research-complete.yaml` may already contain a generic
`2026.05.268` archive row
**When** the Exp 2835 archive generator runs
**Then** it writes `results/experiment_2835_archive_v268.json` with a
terminal-prefixed `honest_verdict`, `archived_milestone=2026.05.268`,
`activated_milestone=2026.05.269`, honest blocked counts and source links in
`archived_task_summary`, no duplicate archive entry, and no modification to
`research-roadmap.yaml` or `scripts/research_conductor.py`.

### REQ-REPORT-2846: Milestone 2026.05.269 Capstone Claim-Boundary Artifact

The repository shall provide an Exp 2846 capstone generator that writes
`results/experiment_2846_capstone_v269.json` with schema
`carnot.milestone_capstone.v269`.

The generator must read all expected `.269` upstream result artifacts, classify
missing artifacts separately from `blocked_*` artifacts, exclude adversarially
flagged source metrics from headline claims, and synthesize the precise claim
boundary for milestone `2026.05.269`. It must not run external model inference,
must not submit or upload publication artifacts, and must not modify
`scripts/research_conductor.py`.

The terminal artifact must include:

- `honest_verdict`, prefixed with `complete:`, `success:`, or `blocked_`
- `milestone`, equal to `2026.05.269`
- `sota_runtime_ready`, copied from Exp 2836 while preserving any adversarial
  warning context
- `primary_corpus_results`, containing only real upstream fields for FoVer,
  MBPP, HumanEval, and TruthfulQA, with flagged FoVer metrics labeled
  non-headline and blocked corpus measurements left null
- `self_learning_result`, derived from Exp 2844 and reporting blocks without
  converting zero sentinel fields into measured improvement
- `paper_ready`, true only when Exp 2845 exists, is not adversarially flagged,
  and reports a ready paper gate
- `top_3_next_actions`
- `docs_updated`, with honest reconciliation status
- `duration_s`, a wall-clock duration for the synthesis run

#### SCENARIO-REPORT-2846: Ungated .269 Capstone Preserves Blocked and Flagged Boundaries

**Given** Exp 2836 through Exp 2844 artifacts may include blocked verdicts or
adversarial flags
**And** Exp 2842 and Exp 2845 may be absent from `results/`
**When** the Exp 2846 capstone generator runs
**Then** it writes `results/experiment_2846_capstone_v269.json` with a
terminal-prefixed `honest_verdict`, `milestone=2026.05.269`,
`sota_runtime_ready` copied from Exp 2836, `paper_ready=false` unless the
paper artifact is present and ready, missing Exp 2842/2845 recorded as missing,
blocked Exp 2838/2839/2840/2844 recorded as blocked, and no modification to
`scripts/research_conductor.py`.

### REQ-REPORT-2847: Archive Milestone 2026.05.269 and Confirm 2026.05.270 Activation

The repository shall provide an Exp 2847 archive/activation generator that
writes `results/experiment_2847_archive_v269_activate_v270.json`.

The generator must read `research-complete.yaml` and determine whether
milestone `2026.05.269` already has a completed archive block. If the block is
present, the generator must set `archive_already_present=true` and avoid
modifying `research-complete.yaml`. If the block is absent, the generator may
append a minimal completed archive row for milestone `2026.05.269` from the
Exp 2846 capstone source without modifying `research-roadmap.yaml`.

The generator must confirm milestone `2026.05.270` and milestone doc
`openspec/change-proposals/research-roadmap-vNEXT.md` from
`research-roadmap-next.yaml` when that file exists. If
`research-roadmap-next.yaml` has already been activated and removed, it must
read the active `research-roadmap.yaml` as a fallback while still leaving that
roadmap unmodified.

The terminal artifact must include:

- `honest_verdict`, prefixed with `complete:` or `blocked:`
- `archived_milestone`, equal to `2026.05.269`
- `activated_milestone`, equal to `2026.05.270`
- `archive_already_present`
- `capstone_source`, equal to `results/experiment_2846_capstone_v269.json`
- `paper_ready_from_capstone`
- `blocked_artifacts_from_capstone`
- `missing_artifacts_from_capstone`
- `top_3_next_actions`
- `run_date`, equal to `20260522`
- `duration_s`, a real wall-clock duration for the archive generator run

#### SCENARIO-REPORT-2847: Existing .269 Archive Confirms Active .270 Roadmap

**Given** `research-complete.yaml` already contains a completed
`2026.05.269` archive row
**And** `research-roadmap-next.yaml` may already have been activated into
`research-roadmap.yaml`
**When** the Exp 2847 archive generator runs
**Then** it writes `results/experiment_2847_archive_v269_activate_v270.json`
with `archive_already_present=true`, `archived_milestone=2026.05.269`,
`activated_milestone=2026.05.270`, capstone blocked and missing artifact lists
copied from `results/experiment_2846_capstone_v269.json`, exactly three next
actions, and no modification to `research-roadmap.yaml` or
`scripts/research_conductor.py`.

### REQ-REPORT-2861: Archive Milestone 2026.05.270 and Confirm 2026.05.271 Activation

The repository shall provide an Exp 2861 archive/activation generator that
writes `results/experiment_2861_archive_v270_activate_v271.json`.

The generator must read `research-complete.yaml` and determine whether
milestone `2026.05.270` already has a completed archive block. If the block is
present, the generator must set `archive_already_present=true` and avoid
modifying `research-complete.yaml`. If the block is absent, the generator may
append a minimal completed archive row for milestone `2026.05.270` from
`results/experiment_2860_capstone_v270.json` without modifying
`research-roadmap.yaml`.

The generator must confirm milestone `2026.05.271` and milestone doc
`openspec/change-proposals/research-roadmap-vNEXT.md` from
`research-roadmap-next.yaml` when that file exists. If
`research-roadmap-next.yaml` has already been activated and removed, it must
read the active `research-roadmap.yaml` as a fallback while still leaving that
roadmap unmodified.

The terminal artifact must include:

- `honest_verdict`, prefixed with `complete:` or `blocked:`
- `archived_milestone`, equal to `2026.05.270`
- `activated_milestone`, equal to `2026.05.271`
- `archive_already_present`
- `capstone_source`, equal to `results/experiment_2860_capstone_v270.json`
- `paper_ready_from_capstone`
- `sota_runtime_ready_v2_from_capstone`
- `blocked_artifacts_from_capstone`
- `missing_artifacts_from_capstone`
- `top_3_next_actions`
- `field_principles`
- `run_date`, equal to `20260522`
- `duration_s`, a real wall-clock duration for the archive generator run

#### SCENARIO-REPORT-2861: Existing .270 Archive Confirms Active .271 Roadmap

**Given** `research-complete.yaml` already contains a completed
`2026.05.270` archive row
**And** `research-roadmap-next.yaml` may already have been activated into
`research-roadmap.yaml`
**When** the Exp 2861 archive generator runs
**Then** it writes `results/experiment_2861_archive_v270_activate_v271.json`
with `archive_already_present=true`, `archived_milestone=2026.05.270`,
`activated_milestone=2026.05.271`, capstone paper and SOTA-runtime readiness
booleans copied from `results/experiment_2860_capstone_v270.json`, blocked and
missing artifact lists copied from that capstone, exactly three next actions,
field principles describing the archive fields, and no modification to
`research-roadmap.yaml` or `scripts/research_conductor.py`.

### REQ-REPORT-2855: Clean .270 Cross-Corpus Matrix Without Imputation

The repository shall provide an Exp 2855 matrix generator that writes
`results/experiment_2855_cross_corpus_matrix_v4.json`.

The generator must read the available Exp 2850 through Exp 2854 `.270` corpus
artifacts, classify every expected corpus row as `clean`, `blocked`,
`flagged`, or `missing`, and build `verifier_corpus_dual_matrix` from source
fields only. It must not fabricate missing AUROC, learning-contribution,
sample-count, or seed-count values. Blocked and flagged corpora must remain
visible in the matrix with their row status rather than disappearing from the
artifact.

The terminal artifact must include:

- `honest_verdict`
- `cross_corpus_matrix_built`, true only when FoVer and at least one non-FoVer
  corpus have clean rows
- `verifier_corpus_dual_matrix`, with production AUROC, architecture-only
  AUROC, learning contribution, `n_examples`, `n_seeds`, and `row_status`
- `row_status_by_corpus`
- clean, blocked, flagged, and missing corpus counts
- `paper_eligible_rows`
- `claim_boundary_notes`
- `source_artifacts`
- `duration_s`
- `run_date`, equal to `20260522`

The generator must not modify `scripts/research_conductor.py`.

#### SCENARIO-REPORT-2855: Blocked and Missing .270 Rows Stay Visible

**Given** Exp 2850 is a clean FoVer dual-condition artifact
**And** one non-FoVer artifact is blocked
**And** other expected non-FoVer artifacts may be absent
**When** the Exp 2855 matrix generator runs
**Then** it writes `results/experiment_2855_cross_corpus_matrix_v4.json` with
FoVer marked `clean`, blocked corpora marked `blocked`, absent corpora marked
`missing`, null metric fields for non-clean rows, `cross_corpus_matrix_built`
set to false until a clean non-FoVer row exists, and no modification to
`scripts/research_conductor.py`.

### REQ-REPORT-2865: Clean .271 Cross-Corpus Matrix From FoVer Plus Non-FoVer Evidence

The repository shall provide an Exp 2865 matrix generator that writes
`results/experiment_2865_cross_corpus_matrix_v5.json`.

The generator must read the clean FoVer artifact
`results/experiment_2850_fover_dual_condition_integrity_v4.json` and the
HaluEval/FEVER artifact
`results/experiment_2864_halueval_fever_full_calibration_v3.json`, classify
FoVer, HaluEval/FEVER, MBPP, HumanEval, and TruthfulQA rows as `clean`,
`blocked`, `flagged`, or `missing`, and build `verifier_corpus_dual_matrix`
only from rows classified `clean`. Missing MBPP, HumanEval, and TruthfulQA
metrics must not be inferred from prior artifacts, placeholder values, or
neighboring corpora.

`cross_corpus_matrix_built` must be true only when FoVer is clean and at least
one non-FoVer row is clean. The terminal artifact must include
`honest_verdict`, `cross_corpus_matrix_built`, `verifier_corpus_dual_matrix`,
`row_status_by_corpus`, `paper_eligible_rows`, clean/blocked/flagged/missing
corpus counts, `source_artifacts`, `excluded_from_headline`,
`claim_boundary_notes`, `field_principles`, `run_date="20260522"`, and a real
`duration_s`.

#### SCENARIO-REPORT-2865: HaluEval/FEVER Creates The First Clean Non-FoVer Row

**Given** Exp 2850 is a clean FoVer dual-condition artifact
**And** Exp 2864 is a clean HaluEval/FEVER calibration artifact
**And** MBPP, HumanEval, and TruthfulQA source artifacts are absent
**When** the Exp 2865 matrix generator runs
**Then** it writes `results/experiment_2865_cross_corpus_matrix_v5.json` with
FoVer and HaluEval/FEVER marked `clean`, the absent corpora marked `missing`,
`cross_corpus_matrix_built=true`, `paper_eligible_rows` containing exactly the
two clean rows, `verifier_corpus_dual_matrix` containing only clean rows, and
`excluded_from_headline` explaining why every non-clean corpus is excluded.

### REQ-REPORT-2880: Clean .272 Cross-Corpus Matrix V6 With Pilot Boundaries

The repository shall provide an Exp 2880 matrix generator that writes
`results/experiment_2880_cross_corpus_matrix_v6.json`.

The generator MUST read the clean Exp 2865 cross-corpus matrix and the three
gated `.272` artifacts:

- `results/experiment_2877_exact_frontier_expansion_halueval_fever_v2.json`
- `results/experiment_2878_halueval_fever_error_verifiability_v1.json`
- `results/experiment_2879_code_corpus_manifest_execution_pilot_v1.json`

Rows are eligible only when their source artifact is clean, no synthetic row is
created, the row has valid label evidence or an explicit pilot status, and no
source artifact has an unresolved `blocked_*` verdict. The generator MUST NOT
infer unavailable metrics, MUST keep missing TruthfulQA and unsupported metrics
as `null` with reasons, and MUST keep MBPP/HumanEval execution-pilot rows out of
headline claims.

The terminal artifact MUST include `honest_verdict`,
`cross_corpus_matrix_built`, `source_artifacts`, `clean_row_count`,
`headline_eligible_rows`, `pilot_only_rows`, `missing_rows`, `matrix_rows`,
`markdown_table`, `synthetic_rows_created=false`, `field_principles`,
`run_date="20260522"`, and a measured `duration_s`. Each matrix row MUST expose
expanded columns for `exact_frontier_support`, `error_verifiability`,
`label_consistency`, `code_execution_pilot`, and `residual_gap`.

#### SCENARIO-REPORT-2880: Pilot Rows Stay Outside Headline Claims

**Given** Exp 2865 contains clean FoVer and HaluEval/FEVER rows
**And** Exp 2877 and Exp 2878 provide clean HaluEval/FEVER exact-frontier and
error-verifiability evidence
**And** Exp 2879 provides a clean MBPP/HumanEval manifest-only execution pilot
**When** the Exp 2880 matrix generator runs
**Then** it writes `results/experiment_2880_cross_corpus_matrix_v6.json` with
FoVer and HaluEval/FEVER in `headline_eligible_rows`, MBPP and HumanEval in
`pilot_only_rows`, TruthfulQA in `missing_rows`, no synthetic rows, null fields
with reasons for unsupported metrics, and a compact markdown table matching the
machine-readable rows.

### REQ-REPORT-2872: Milestone 2026.05.271 Capstone Claim Boundary

The repository shall provide an Exp 2872 capstone generator that writes
`results/experiment_2872_capstone_v271.json` with schema
`carnot.milestone_capstone.v271`.

The generator must read every expected Exp 2861 through Exp 2871 result
artifact, classify each artifact as `clean`, `blocked`, `missing`, or
`adversarially_flagged`, and decide paper readiness only from clean evidence.
It must not infer missing metrics, must not count adversarially flagged runtime,
micro-panel, or verifier artifacts as headline evidence, must not modify
`research-roadmap.yaml`, and must not modify `scripts/research_conductor.py`.

The terminal artifact must include:

- `honest_verdict`, prefixed with `complete:`
- `milestone`, equal to `2026.05.271`
- `paper_ready`, true only when Exp 2865 built a cross-corpus matrix from clean
  FoVer plus at least one clean non-FoVer row and the headline rows themselves
  are not adversarially flagged
- `sota_runtime_ready_v3`, preserving the Exp 2862 readiness boundary while
  separately marking Exp 2862 as adversarially flagged when its artifact is
  flagged
- `manifest_contract_ready`
- `cross_corpus_matrix_built`
- `fr11_self_learning_ready`
- `continuous_self_learning_completed`
- `headline_eligible_rows`
- clean, blocked, missing, and adversarially flagged artifact lists
- `primary_corpus_results`, copied from clean source matrix/calibration fields
  without imputation
- `self_learning_summary`, including energy delta, correctness delta, memory
  hashes, and no-model-weight-mutation status from Exp 2869
- `runtime_summary`, including whether Exp 2870 invoked a mandated SOTA model
  and whether any runtime or micro-panel source was adversarially flagged
- `claim_boundary_notes`
- exactly three `top_3_next_actions`
- `pushed=false`
- `scripts_research_conductor_modified=false`
- `field_principles`
- `run_date`, equal to `20260522`
- a real `duration_s`

#### SCENARIO-REPORT-2872: Capstone Preserves Clean, Missing, and Flagged Boundaries

**Given** Exp 2864 and Exp 2865 provide clean HaluEval/FEVER and matrix rows
**And** Exp 2862, Exp 2870, or Exp 2871 may be adversarially flagged
**And** MBPP, HumanEval, and TruthfulQA rows may remain missing from the matrix
**When** the Exp 2872 capstone generator runs
**Then** it writes `results/experiment_2872_capstone_v271.json` with
`paper_ready=true` only from the clean Exp 2865 headline rows, flagged runtime
or formal-verifier artifacts listed in `adversarially_flagged_artifacts`, missing
non-clean corpus rows retained in `primary_corpus_results`, Exp 2869's
energy/correctness and mutation boundary preserved in `self_learning_summary`,
exactly three next actions, and no modification to `research-roadmap.yaml` or
`scripts/research_conductor.py`.

### REQ-REPORT-2873: Archive Milestone 2026.05.271 and Confirm 2026.05.272 Activation

The repository shall provide an Exp 2873 archive/activation generator that
writes `results/experiment_2873_archive_v271_activate_v272.json`.

The generator must read `research-complete.yaml` and determine whether
milestone `2026.05.271` already has a completed archive block. If the block is
present, the generator must set `archive_already_present=true` and avoid
modifying `research-complete.yaml`. If the block is absent, the generator may
append a minimal completed archive row for milestone `2026.05.271` from
`results/experiment_2872_capstone_v271.json` without modifying
`research-roadmap.yaml`.

The generator must confirm milestone `2026.05.272` and milestone doc
`openspec/change-proposals/research-roadmap-vNEXT.md` from
`research-roadmap-next.yaml` when that file exists. If
`research-roadmap-next.yaml` has already been activated and removed, it must
read the active `research-roadmap.yaml` as a fallback while still leaving that
roadmap unmodified.

The terminal artifact must include:

- `honest_verdict`, prefixed with `complete:` or `blocked:`
- `archived_milestone`, equal to `2026.05.271`
- `activated_milestone`, equal to `2026.05.272`
- `archive_already_present`
- `capstone_source`, equal to `results/experiment_2872_capstone_v271.json`
- `paper_ready_from_capstone`
- `clean_artifacts_from_capstone`
- `blocked_artifacts_from_capstone`
- `missing_artifacts_from_capstone`
- `adversarially_flagged_artifacts_from_capstone`
- `headline_eligible_rows_from_capstone`
- `top_3_next_actions`
- `field_principles`
- `run_date`, equal to `20260522`
- `duration_s`, a real wall-clock duration for the archive generator run

#### SCENARIO-REPORT-2873: Existing .271 Archive Confirms Active .272 Roadmap

**Given** `research-complete.yaml` already contains a completed
`2026.05.271` archive row
**And** `research-roadmap-next.yaml` may already have been activated into
`research-roadmap.yaml`
**When** the Exp 2873 archive generator runs
**Then** it writes `results/experiment_2873_archive_v271_activate_v272.json`
with `archive_already_present=true`, `archived_milestone=2026.05.271`,
`activated_milestone=2026.05.272`, paper readiness, clean artifact, blocked
artifact, missing artifact, adversarially flagged artifact, headline-eligible
row, and next-action fields copied from
`results/experiment_2872_capstone_v271.json`, exactly three next actions, field
principles describing the archive fields, and no modification to
`research-roadmap.yaml` or `scripts/research_conductor.py`.

### REQ-REPORT-2894: Clean .273 Cross-Corpus Matrix V7 From Completed Support Artifacts

The repository shall provide an Exp 2894 matrix generator that writes
`results/experiment_2894_cross_corpus_matrix_v7.json`.

The generator MUST read the clean Exp 2880 cross-corpus matrix v6 and the
available `.273` support artifacts:

- `results/experiment_2888_truthfulqa_inficheck_taxonomy_manifest_v1.json`
- `results/experiment_2889_mbpp_humaneval_generated_code_clean_row_v1.json`
- `results/experiment_2890_code_structural_dependency_verifier_v1.json`
- `results/experiment_2891_cctu_executable_constraint_validator_pilot_v1.json`
- `results/experiment_2892_vericot_exact_frontier_expansion_v1.json`
- `results/experiment_2893_kan_hardware_complexity_accounting_v1.json`

Rows are eligible only when the source artifact is clean, no synthetic row is
created, headline code metrics have valid generated-output labels and tests,
taxonomy or pilot rows have explicit non-headline status, and no source used for
that row has unresolved adversarial or corrigendum flags. The generator MUST
NOT infer unavailable AUROC, pass@k, generated-answer, CCTU, VeriCoT, KAN, or
hardware metrics. Unsupported metrics MUST remain `null` with explicit reasons.

The terminal artifact MUST include `honest_verdict`,
`cross_corpus_matrix_built`, `source_artifacts`, `clean_row_count`,
`headline_eligible_rows`, `pilot_only_rows`, `taxonomy_only_rows`,
`blocked_rows`, `missing_rows`, `matrix_rows`, `markdown_table`,
`synthetic_rows_created=false`, `field_principles`, `run_date="20260523"`, and
a measured `duration_s`. Each matrix row MUST expose columns for
`truthfulqa_taxonomy`, `generated_code_status`,
`structural_dependency_verification`, `cctu_constraint_category_coverage`,
`vericot_exact_support`, `kan_complexity`, and `residual_gap`.

#### SCENARIO-REPORT-2894: V7 Preserves Headline, Pilot, And Taxonomy Boundaries

**Given** Exp 2880 contains clean FoVer and HaluEval/FEVER headline rows plus
MBPP/HumanEval pilot-only rows
**And** Exp 2888 provides a clean TruthfulQA taxonomy manifest without
generated-answer metrics
**And** Exp 2889 may contain generated-code rows with unresolved adversarial or
corrigendum flags
**And** Exp 2890 through Exp 2893 provide clean support metadata
**When** the Exp 2894 matrix generator runs
**Then** it writes `results/experiment_2894_cross_corpus_matrix_v7.json` with
FoVer and HaluEval/FEVER in `headline_eligible_rows`, MBPP and HumanEval in
`pilot_only_rows`, TruthfulQA in `taxonomy_only_rows`, any unresolved
generated-code source flags preserved in `blocked_rows`, no synthetic rows,
null unsupported metrics with reasons, and a compact markdown table matching
the machine-readable rows.

### REQ-REPORT-2895: Paper-v6 Evidence Table And Claim Boundary From Matrix V7

The repository shall provide an Exp 2895 evidence-table generator that writes
`results/experiment_2895_paper_v6_evidence_table_v4.json`.

The generator MUST read `results/experiment_2894_cross_corpus_matrix_v7.json`
and `results/experiment_2884_capstone_v272.json`, separate paper-v6 statements
into safe headline claims, pilot-only statements, taxonomy-only statements,
blocked claims, and forbidden claims, and produce a compact markdown table
suitable for operator review. The generator MUST NOT modify the operator-
curated landing page, submit externally, or infer metrics that matrix v7 marks
as null, pilot-only, taxonomy-only, blocked, or missing.

The terminal artifact MUST include `honest_verdict`,
`paper_evidence_table_ready`, `source_artifacts`, `headline_claims`,
`pilot_only_statements`, `taxonomy_only_statements`, `forbidden_claims`,
`markdown_table`, `arxiv_submission_performed=false`,
`landing_page_modified=false`, `field_principles`, `run_date="20260523"`, and
a measured `duration_s`. The artifact MAY include additional internal fields,
but any such fields MUST preserve the same claim-boundary discipline and MUST
NOT create a publication action.

#### SCENARIO-REPORT-2895: Evidence Table Preserves Claim Boundaries

**Given** Exp 2894 has FoVer and HaluEval/FEVER headline rows, MBPP and
HumanEval pilot-only rows with blocked generated-code support, and a
TruthfulQA taxonomy-only row
**And** Exp 2884 contains safe and forbidden paper-v6 claims from the prior
capstone
**When** the Exp 2895 evidence-table generator runs
**Then** it writes `results/experiment_2895_paper_v6_evidence_table_v4.json`
with headline claims only for clean headline rows, pilot-only statements only
for MBPP/HumanEval pilot evidence, taxonomy-only statements only for
TruthfulQA taxonomy evidence, blocked or forbidden claims for unresolved
generated-code flags and prior capstone exclusions, a markdown table that
labels every row boundary explicitly, and no arXiv submission or landing-page
modification.

### REQ-REPORT-2884: Milestone 2026.05.272 Capstone Claim Boundary

The repository shall provide an Exp 2884 capstone generator that writes
`results/experiment_2884_capstone_v272.json` with schema
`carnot.milestone_capstone.v272`.

The generator MUST read every expected `.272` deliverable from Exp 2873 through
Exp 2883 when present, list missing deliverables explicitly, and classify each
source artifact as `clean`, `flagged`, `blocked`, `missing`, or `pilot-only`.
Flagged artifacts MUST remain excluded from paper readiness even when they also
report success booleans. Pilot-only artifacts MAY support operational follow-up
claims but MUST NOT become headline rows.

The terminal artifact MUST compare the `.272` outputs against the `.271`
flagged runtime, SOTA micro-panel, and KAN PWA/MILP artifacts; decide whether
matrix v6 contains more clean evidence than matrix v5; evaluate FR-11 recurrence
trigger, token reduction, energy, correctness/AUROC, drift, and
non-forgetting; and evaluate the THRML sampler branch without making hardware
claims. It MUST NOT modify `research-roadmap.yaml` or
`scripts/research_conductor.py`.

The terminal artifact MUST include `honest_verdict`, `milestone`,
`paper_ready`, `clean_artifacts`, `flagged_artifacts`, `blocked_artifacts`,
`missing_artifacts`, `pilot_only_artifacts`, `corrected_271_flags`,
`sota_runtime_clean`, `micro_panel_clean`, `kan_tautology_cleared`,
`cross_corpus_matrix_built`, `headline_eligible_rows`,
`continuous_self_learning_result`, `thrml_sampler_status`,
`paper_v6_safe_claims`, `paper_v6_forbidden_claims`, `top_3_next_actions`,
`field_principles`, `run_date="20260522"`, and a measured `duration_s`.

#### SCENARIO-REPORT-2884: Flagged Branches Stay Out of Paper-Ready Claims

**Given** Exp 2874 clears the `.271` runtime flag
**And** Exp 2875 or Exp 2882 may still contain adversarial/corrigendum flags
**And** Exp 2879 contributes only MBPP/HumanEval pilot rows
**And** Exp 2883 may be a clean dependency block with local fallback but no THRML
or hardware result
**When** the Exp 2884 capstone generator runs
**Then** it writes `results/experiment_2884_capstone_v272.json` with paper
readiness derived only from clean matrix evidence, flagged artifacts excluded
from safe claims, pilot-only rows excluded from `headline_eligible_rows`,
FR-11 scale-up claims forbidden when the source is flagged, no hardware claim
for THRML fallback-only evidence, exactly three next actions, and no
modification to `research-roadmap.yaml` or `scripts/research_conductor.py`.

### REQ-REPORT-2885: Archive Milestone 2026.05.272 and Confirm 2026.05.273 Activation

The repository shall provide an Exp 2885 archive/activation generator that
writes `results/experiment_2885_archive_v272_activate_v273.json`.

The generator must read `research-complete.yaml` and determine whether
milestone `2026.05.272` already has a completed archive block. If the block is
present, the generator must set `archive_already_present=true` and avoid
modifying `research-complete.yaml`. If the block is absent, the generator may
append a minimal completed archive row for milestone `2026.05.272` from
`results/experiment_2884_capstone_v272.json` without modifying
`research-roadmap.yaml`.

The generator must confirm milestone `2026.05.273` and milestone doc
`openspec/change-proposals/research-roadmap-vNEXT.md` from
`research-roadmap-next.yaml` when that file exists. If
`research-roadmap-next.yaml` has already been activated and removed, it must
read the active `research-roadmap.yaml` as a fallback while still leaving that
roadmap unmodified.

The terminal artifact must include:

- `honest_verdict`, prefixed with `complete:` or `blocked:`
- `archived_milestone`, equal to `2026.05.272`
- `activated_milestone`, equal to `2026.05.273`
- `archive_already_present`
- `capstone_source`, equal to `results/experiment_2884_capstone_v272.json`
- `paper_ready_from_capstone`
- `clean_artifacts_from_capstone`
- `flagged_artifacts_from_capstone`
- `blocked_artifacts_from_capstone`
- `missing_artifacts_from_capstone`
- `pilot_only_artifacts_from_capstone`
- `paper_v6_safe_claims_from_capstone`
- `paper_v6_forbidden_claims_from_capstone`
- `top_3_next_actions`
- `field_principles`
- `run_date`, equal to `20260522`
- `duration_s`, a real wall-clock duration for the archive generator run

#### SCENARIO-REPORT-2885: Existing .272 Archive Confirms Active .273 Roadmap

**Given** `research-complete.yaml` already contains a completed
`2026.05.272` archive row
**And** `research-roadmap-next.yaml` may already have been activated into
`research-roadmap.yaml`
**When** the Exp 2885 archive generator runs
**Then** it writes `results/experiment_2885_archive_v272_activate_v273.json`
with `archive_already_present=true`, `archived_milestone=2026.05.272`,
`activated_milestone=2026.05.273`, paper readiness, clean artifact, flagged
artifact, blocked artifact, missing artifact, pilot-only artifact, paper-v6 safe
claim, paper-v6 forbidden claim, and next-action fields copied from
`results/experiment_2884_capstone_v272.json`, exactly three next actions, field
principles describing the archive fields, and no modification to
`research-roadmap.yaml` or `scripts/research_conductor.py`.

### REQ-REPORT-2897: Archive Milestone 2026.05.273 and Confirm 2026.05.274 Activation

The repository shall provide an Exp 2897 archive/activation generator that
writes `results/experiment_2897_archive_v273_activate_v274.json`.

The generator must first require
`results/experiment_2896_capstone_v273.json`. If that capstone is absent or
malformed, it must write a terminal artifact with
`honest_verdict="blocked_capstone_missing"` and must not modify
`research-complete.yaml`, `research-roadmap.yaml`, or
`scripts/research_conductor.py`.

When the capstone is present, the generator must read
`research-complete.yaml` and determine whether milestone `2026.05.273` already
has a completed archive block. If the block is present, the generator must set
`archive_already_present=true` and avoid modifying `research-complete.yaml`. If
the block is absent, the generator may append a minimal completed archive row
for milestone `2026.05.273` from
`results/experiment_2896_capstone_v273.json` without modifying
`research-roadmap.yaml`.

The generator must confirm milestone `2026.05.274` from
`research-roadmap-next.yaml` when that file exists. If
`research-roadmap-next.yaml` has already been activated and removed, it must
read the active `research-roadmap.yaml` as a fallback while still leaving that
roadmap unmodified.

The terminal artifact must include:

- `honest_verdict`, prefixed with `complete:` on successful archival
- `archived_milestone`, equal to `2026.05.273`
- `activated_milestone`, equal to `2026.05.274`
- `capstone_source`, equal to `results/experiment_2896_capstone_v273.json`
- `paper_ready_from_capstone`
- `clean_artifacts_from_capstone`
- `flagged_artifacts_from_capstone`
- `blocked_artifacts_from_capstone`
- `missing_artifacts_from_capstone`
- `pilot_only_artifacts_from_capstone`
- `inference_substrate`, equal to `aggregation_from_upstream_artifacts`
- `duration_s`, a real wall-clock duration for the archive generator run

#### SCENARIO-REPORT-2897: Existing .273 Archive Confirms Active .274 Roadmap

**Given** `research-complete.yaml` already contains a completed
`2026.05.273` archive row
**And** `research-roadmap-next.yaml` may already have been activated into
`research-roadmap.yaml`
**When** the Exp 2897 archive generator runs
**Then** it writes `results/experiment_2897_archive_v273_activate_v274.json`
with `archive_already_present=true`, `archived_milestone=2026.05.273`,
`activated_milestone=2026.05.274`, paper readiness, clean artifact, flagged
artifact, blocked artifact, missing artifact, and pilot-only artifact fields
copied from `results/experiment_2896_capstone_v273.json`,
`inference_substrate=aggregation_from_upstream_artifacts`, field principles
describing the required artifact fields, and no modification to
`research-roadmap.yaml` or `scripts/research_conductor.py`.

### REQ-REPORT-2902: Cross-Corpus Matrix V8 With Forward-Only Provenance

The repository shall provide an Exp 2902 cross-corpus matrix v8 generator that
writes `results/experiment_2902_cross_corpus_matrix_v8.json`.

The generator MUST first require
`results/experiment_2894_cross_corpus_matrix_v7.json`. If matrix v7 is absent,
it MUST write a terminal artifact with `honest_verdict="blocked_v7_missing"`
and MUST NOT infer rows from later support artifacts alone.

When matrix v7 is present, the generator MUST load it, identify v7 rows that
are clean, flagged, blocked, and pilot-only, and then add explicit support rows
from clean existing Exp 2890 Code Structural Dependency, Exp 2891 CCTU, Exp
2892 VeriCoT, and Exp 2898 KV260 hardware artifacts. It MUST keep CCTU as
pilot-only support, MUST keep unresolved generated-code flags visible, and MUST
NOT promote pilot-only, taxonomy-only, or hardware-smoke evidence into
cross-corpus AUROC or pass@k metrics.

Every emitted row MUST cite the direct upstream artifact path, experiment id,
fields imported, and SHA256 hash used to build that row. The terminal artifact
MUST include `honest_verdict`, `inference_substrate` equal to
`aggregation_from_upstream_artifacts`, `rows_clean`, `rows_flagged`,
`rows_blocked`, `rows_pilot_only`, `cited_upstream_artifacts`, and measured
`duration_s`. The `cited_upstream_artifacts` field MUST declare the
forward-only provenance discipline and expose a list shaped as
`{experiment_id, fields_imported, sha256}` so a third party can verify the
aggregation did not synthesize values from missing sources.

#### SCENARIO-REPORT-2902: V8 Adds Clean Code, CCTU, VeriCoT, And KV260 Rows

**Given** Exp 2894 matrix v7 exists with clean headline, taxonomy-only, and
pilot-only row boundaries
**And** Exp 2890, Exp 2891, Exp 2892, and Exp 2898 artifacts exist cleanly
**When** the Exp 2902 matrix v8 generator runs
**Then** it writes `results/experiment_2902_cross_corpus_matrix_v8.json` with
forward-only aggregation provenance, clean rows for v7 headline/taxonomy
evidence plus Exp 2890, Exp 2892, and Exp 2898, pilot-only rows for v7 code
pilot rows and Exp 2891 CCTU, flagged rows for unresolved generated-code
support, no synthesized metrics, and per-row artifact SHA256 citations.

### REQ-REPORT-2909: Archive Milestone 2026.05.274 and Confirm 2026.05.275 Activation

The repository shall provide an Exp 2909 archive/activation generator that
writes `results/experiment_2909_archive_v274_activate_v275.json`.

The generator must first require
`results/experiment_2908_capstone_v274.json`. If that capstone is absent or
malformed, it must write a terminal artifact with
`honest_verdict="blocked_capstone_missing"` and must not modify
`research-complete.yaml`, `research-roadmap.yaml`, or
`scripts/research_conductor.py`.

When the capstone is present, the generator must read
`research-complete.yaml` and determine whether milestone `2026.05.274` already
has a completed archive block. If the block is present, the generator must set
`archive_ready=true` without modifying `research-complete.yaml`. If the block
is absent, the generator may append a minimal completed archive row for
milestone `2026.05.274` from `results/experiment_2908_capstone_v274.json`
without modifying `research-roadmap.yaml`.

The generator must confirm milestone `2026.05.275` from
`research-roadmap-next.yaml` when that file exists. If
`research-roadmap-next.yaml` has already been activated and removed, it must
read the active `research-roadmap.yaml` as a fallback while still leaving that
roadmap unmodified.

The terminal artifact must include:

- `honest_verdict`, equal to
  `complete: archive_ready=true; archived_milestone=2026.05.274; activated_milestone=2026.05.275`
  on successful archival
- `archive_ready`
- `archived_milestone`, equal to `2026.05.274`
- `activated_milestone`, equal to `2026.05.275`
- `capstone_source`, equal to `results/experiment_2908_capstone_v274.json`
- `paper_ready_from_capstone`
- `clean_artifacts_from_capstone`
- `flagged_artifacts_from_capstone`
- `blocked_artifacts_from_capstone`
- `missing_artifacts_from_capstone`
- `pilot_only_artifacts_from_capstone`
- `gaps_for_275`
- `inference_substrate`, equal to `aggregation_from_upstream_artifacts`
- `duration_s`, a real wall-clock duration for the archive generator run
- `run_date`, equal to `20260523`

#### SCENARIO-REPORT-2909: Existing .274 Archive Confirms Active .275 Roadmap

**Given** `research-complete.yaml` already contains a completed
`2026.05.274` archive row
**And** `research-roadmap-next.yaml` may already have been activated into
`research-roadmap.yaml`
**When** the Exp 2909 archive generator runs
**Then** it writes `results/experiment_2909_archive_v274_activate_v275.json`
with `archive_ready=true`, `archived_milestone=2026.05.274`,
`activated_milestone=2026.05.275`, paper readiness, clean artifact, flagged
artifact, blocked artifact, missing artifact, pilot-only artifact, and
`gaps_for_275` fields copied from
`results/experiment_2908_capstone_v274.json`,
`inference_substrate=aggregation_from_upstream_artifacts`, and no modification
to `research-roadmap.yaml` or `scripts/research_conductor.py`.

### REQ-REPORT-2921: Cross-Corpus Matrix V9 And Paper-v6 Claim Boundary

The repository shall provide an Exp 2921 cross-corpus matrix v9 generator that
writes `results/experiment_2921_cross_corpus_matrix_v9_paper_boundary_v1.json`.

The generator MUST first verify that every gated `.275` upstream artifact is
present and has the readiness field required by `research-roadmap.yaml`:
Exp 2911 `code_hallucination_verifier_ready`, Exp 2913
`kv260_claim_boundary_ready`, Exp 2918 `online_self_learning_ready`, Exp 2919
`constraintbench_mini_ready`, and Exp 2920 `state_verifier_harness_ready`. If
any gated artifact is absent or the readiness field is not true, it MUST write
a terminal artifact with `honest_verdict="blocked_gate_inconsistent"` and MUST
NOT promote rows from downstream evidence.

When the gates are consistent, the generator MUST read matrix v8 and the
available `.275` artifacts, classify every candidate row as `clean`,
`flagged`, `blocked`, `pilot_only`, `diagnostic_only`, or `missing`, and keep
flagged, blocked, pilot-only, simulator-only, and diagnostic-only rows out of
headline eligibility. Clean rows MAY become headline eligible only when their
own direct artifact supports a bounded paper-v6 claim. GateMate and THRML rows
MUST remain non-headline unless their own artifacts are clean hardware evidence;
simulator-only THRML parity may be preserved only as diagnostic context.

The terminal artifact MUST include `honest_verdict`,
`cross_corpus_matrix_v9_built`, `paper_claim_boundary_ready`,
`headline_eligible_rows`, `clean_rows`, `flagged_rows`, `blocked_rows`,
`pilot_only_rows`, `diagnostic_only_rows`, `missing_rows`, `matrix_v9_path`,
`paper_v6_claim_boundary`, `inference_substrate` equal to
`aggregation_from_upstream_artifacts`, measured `duration_s`, and
`run_date="20260523"`.

#### SCENARIO-REPORT-2921: V9 Promotes Only Clean Bounded .275 Evidence

**Given** matrix v8 exists with clean headline rows and flagged code-pilot rows
**And** the `.275` gated artifacts are present and readiness fields are true
**And** Exp 2911 and Exp 2919 are adversarially flagged despite their ready
fields
**When** the Exp 2921 matrix v9 generator runs
**Then** it writes
`results/experiment_2921_cross_corpus_matrix_v9_paper_boundary_v1.json` with
clean bounded rows for SOTA codegen, KV260 claim boundary, FR-11 process
rewards, and the state verifier harness, flagged rows for the code
hallucination verifier and ConstraintBench mini, a blocked GateMate row,
diagnostic-only THRML and spilled-energy rows, no headline promotion from
flagged or simulator-only evidence, and a paper-v6 claim-boundary block that
lists exactly the headline-eligible and non-headline rows.

### REQ-REPORT-2922: Milestone .275 Capstone Claim Boundary

The repository shall provide an Exp 2922 milestone .275 capstone generator that
writes `results/experiment_2922_capstone_v275.json` by aggregating only upstream
artifact JSON files from the `.275` milestone. The generator MUST NOT rerun
models, verifiers, hardware, or sampler measurements; its
`inference_substrate` MUST be `aggregation_from_upstream_artifacts`.

The generator MUST inspect every expected `.275` deliverable from Exp 2909
through Exp 2921, including Exp 2915 even when its artifact is absent, and
classify each artifact as exactly one of `clean`, `flagged`, `blocked`,
`missing`, `pilot_only`, or `diagnostic_only`. Missing artifacts MUST be listed
explicitly and MUST NOT be inferred as successful from downstream rows.

The terminal artifact MUST include `honest_verdict`, `paper_ready`,
`hardware_baselines_ready`, `hardware_speedup_claim_eligible`,
`sota_code_row_repaired`, `fr11_self_learning_clean`, `clean_artifacts`,
`flagged_artifacts`, `blocked_artifacts`, `missing_artifacts`,
`pilot_only_artifacts`, `diagnostic_only_artifacts`, `headline_eligible_rows`,
`hardware_claim_boundary`, `codegen_claim_boundary`, `fr11_claim_boundary`,
`top_3_next_actions`, `inference_substrate` equal to
`aggregation_from_upstream_artifacts`, measured `duration_s`, and
`run_date="20260523"`.

`paper_ready` MUST remain true only when Exp 2921 reports
`paper_claim_boundary_ready=true` and no headline-eligible row depends on a
flagged, blocked, missing, pilot-only, or diagnostic-only `.275` source.
`hardware_speedup_claim_eligible` MUST mirror Exp 2913 only when Exp 2912 and
Exp 2913 are both clean. `sota_code_row_repaired` MUST be true only when Exp
2910 is clean, has `codegen_corrigendum_ready=true`, and does not have
unresolved adversarial/corrigendum flags. `fr11_self_learning_clean` MUST be
true only when Exp 2918 is clean, `online_self_learning_ready=true`, an online
update or replay-scheduler update was performed, and forgetting metrics are
reported.

#### SCENARIO-REPORT-2922: Capstone Closes .275 With Missing GateMate Listed

**Given** Exp 2909 through Exp 2921 source artifacts exist except the gated Exp
2915 GateMate bitstream artifact
**And** Exp 2911 and Exp 2919 carry adversarial/corrigendum flags
**And** Exp 2916 and Exp 2917 are diagnostic-only rows
**And** Exp 2912, Exp 2913, Exp 2918, Exp 2920, and Exp 2921 are clean
**When** the Exp 2922 capstone generator runs
**Then** it writes `results/experiment_2922_capstone_v275.json` with the
GateMate bitstream artifact in `missing_artifacts`, Exp 2911 and Exp 2919 in
`flagged_artifacts`, Exp 2916 and Exp 2917 in `diagnostic_only_artifacts`,
`paper_ready=true`, `hardware_baselines_ready=true`,
`hardware_speedup_claim_eligible=true`, `sota_code_row_repaired=true`,
`fr11_self_learning_clean=true`, and a top-three `.276` action list that
prioritizes resolving flagged code-taxonomy/ConstraintBench evidence and the
missing GateMate bitstream.

## Implementation Status (REQ-REPORT-2922)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-2922 | Implemented (`python/carnot/reporting/capstone_v275_2922.py`) | Implemented (`tests/python/test_capstone_v275_2922.py`) |

### REQ-REPORT-2923: Archive Milestone 2026.05.275 and Confirm 2026.05.276 Activation

The repository shall provide an Exp 2923 archive/activation generator that
writes `results/experiment_2923_archive_v275_activate_v276.json`.

The generator MUST first require
`results/experiment_2922_capstone_v275.json`. If that capstone is absent or
malformed, it MUST write a terminal artifact with
`honest_verdict="blocked_capstone_missing"` and MUST NOT modify
`research-complete.yaml`, `research-roadmap.yaml`, or
`scripts/research_conductor.py`.

When the capstone is present, the generator MUST read
`research-complete.yaml` and determine whether milestone `2026.05.275` already
has a completed archive block. If the block is present, the generator MUST set
`archive_ready=true` without modifying `research-complete.yaml`. If the block
is absent, the generator MAY append a minimal completed archive row for
milestone `2026.05.275` from `results/experiment_2922_capstone_v275.json`
without modifying `research-roadmap.yaml`.

The generator MUST confirm milestone `2026.05.276` from
`research-roadmap-next.yaml` when that file exists. If
`research-roadmap-next.yaml` has already been activated and removed, it MUST
read the active `research-roadmap.yaml` as a fallback while still leaving that
roadmap unmodified.

The terminal artifact MUST include:

- `honest_verdict`, equal to
  `complete: archive_ready=true; archived_milestone=2026.05.275; activated_milestone=2026.05.276`
  on successful archival
- `archive_ready`
- `archived_milestone`, equal to `2026.05.275`
- `activated_milestone`, equal to `2026.05.276`
- `capstone_source`, equal to `results/experiment_2922_capstone_v275.json`
- `paper_ready_from_capstone`
- `hardware_speedup_claim_eligible_from_capstone`
- `clean_artifacts_from_capstone`
- `flagged_artifacts_from_capstone`
- `blocked_artifacts_from_capstone`
- `missing_artifacts_from_capstone`
- `diagnostic_artifacts_from_capstone`
- `recommended_next_actions`
- `inference_substrate`, equal to `aggregation_from_upstream_artifacts`
- `duration_s`, a real wall-clock duration for the archive generator run
- `run_date`, equal to `20260523`

`recommended_next_actions` MUST be copied from the capstone's
`recommended_next_actions` list when present. When the capstone uses the prior
`top_3_next_actions` field instead, the generator MUST copy that list into the
required `recommended_next_actions` artifact field without inventing new
actions.

#### SCENARIO-REPORT-2923: Existing .275 Archive Confirms Active .276 Roadmap

**Given** `research-complete.yaml` already contains a completed
`2026.05.275` archive row
**And** `research-roadmap-next.yaml` may already have been activated into
`research-roadmap.yaml`
**When** the Exp 2923 archive generator runs
**Then** it writes
`results/experiment_2923_archive_v275_activate_v276.json` with
`archive_ready=true`, `archived_milestone=2026.05.275`,
`activated_milestone=2026.05.276`, paper readiness, hardware speedup claim
eligibility, clean artifact, flagged artifact, blocked artifact, missing
artifact, diagnostic artifact, and recommended next-action fields copied from
`results/experiment_2922_capstone_v275.json`,
`inference_substrate=aggregation_from_upstream_artifacts`, and no modification
to `research-roadmap.yaml` or `scripts/research_conductor.py`.

## Implementation Status (REQ-REPORT-2923)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-2923 | Implemented (`python/carnot/reporting/milestone_275_archive_276_activation.py`) | Implemented (`tests/python/test_experiment_2923_archive_v275.py`) |

### REQ-REPORT-2924: Matrix V9 And .275 Capstone Aggregation Corrigendum

The repository shall provide an Exp 2924 aggregation-metadata corrigendum
workflow that writes
`results/experiment_2924_aggregation_metadata_corrigendum_v1.json` by reading
only the existing Exp 2921 matrix v9 artifact and Exp 2922 `.275` capstone
artifact. The workflow MUST NOT call an LLM, rerun a verifier, rerun a sampler,
or launch hardware.

If either upstream artifact is absent or malformed, the workflow MUST write a
terminal artifact with `honest_verdict="blocked_upstream_artifact_missing"`,
`aggregation_metadata_clean=false`, `no_new_llm_call=true`,
`no_new_hardware_run=true`, and the missing upstream paths listed explicitly.

When both upstream artifacts are present, the workflow MUST compute SHA256
checksums for every source artifact directly cited by Exp 2921 or Exp 2922,
plus Exp 2921 and Exp 2922 themselves. Missing cited artifacts MUST remain
listed with a null checksum rather than being inferred from downstream rows.

The workflow MUST build `aggregation_provenance` rows containing
`artifact_path`, `checksum`, `row_role`, `source_inference_substrate`, and
`current_task_reran_compute=false`. The provenance MUST distinguish
`corrigendum_subject` artifacts from matrix-row and capstone-source artifacts.

The workflow MUST preserve upstream flagged-row facts separately from
current-artifact audit findings. In particular, Exp 2911, Exp 2919, and Exp
2921 MUST remain listed in `upstream_flagged_rows_preserved` when they are
flagged by the `.275` artifacts. Exp 2921 and Exp 2922 DURATION_TOO_SHORT or
METHODOLOGY_MISSING findings caused by inherited compute-bound metadata MUST be
listed in `metadata_false_positive_findings` without laundering Exp 2911 or
Exp 2919.

The workflow MUST run the local adversarial artifact audit when available. If
the audit still flags the corrigendum artifact, it MUST record the exact audit
findings in `adversarial_audit_rerun` and set
`aggregation_metadata_clean=false`; otherwise it MUST set
`aggregation_metadata_clean=true`.

The terminal artifact MUST include `honest_verdict`,
`aggregation_metadata_clean`, `no_new_llm_call=true`,
`no_new_hardware_run=true`, `aggregation_from_upstream_artifacts=true`,
`source_artifact_checksums`, `aggregation_provenance`,
`upstream_flagged_rows_preserved`, `metadata_false_positive_findings`,
`adversarial_audit_rerun`, `inference_substrate` equal to
`aggregation_from_upstream_artifacts`, measured `duration_s`, and
`run_date="20260523"`.

#### SCENARIO-REPORT-2924: Corrigendum Separates Upstream Flags From Metadata False Positives

**Given** Exp 2921 and Exp 2922 artifacts exist
**And** they cite upstream source artifacts with mixed clean, flagged, blocked,
missing, and diagnostic roles
**And** Exp 2911, Exp 2919, and Exp 2921 are flagged in the `.275` artifacts
**When** the Exp 2924 corrigendum workflow runs
**Then** it writes
`results/experiment_2924_aggregation_metadata_corrigendum_v1.json` with all
required fields, SHA256 checksums for present cited artifacts, null checksums
for missing cited artifacts, `current_task_reran_compute=false` on every
provenance row, Exp 2911/2919/2921 preserved as upstream flags, Exp 2921 and
Exp 2922 aggregation-only audit findings classified as metadata false
positives, and the local adversarial audit result recorded for the corrigendum
itself.

## Implementation Status (REQ-REPORT-2924)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-2924 | Planned (`python/carnot/reporting/aggregation_metadata_corrigendum_2924.py`) | Planned (`tests/python/test_aggregation_metadata_corrigendum_2924.py`) |

### REQ-REPORT-2935: Cross-Corpus Matrix V10 Paper-Boundary Corrigendum

The repository shall provide an Exp 2935 cross-corpus matrix v10 generator that
writes
`results/experiment_2935_cross_corpus_matrix_v10_paper_boundary_corrigendum_v1.json`
using only upstream artifact JSON files. The generator MUST NOT call an LLM,
rerun a verifier, run a sampler, launch hardware, or modify
`scripts/research_conductor.py`.

The generator MUST first require the structured `.276` gate sources: Exp 2924
`aggregation_metadata_clean=true`, Exp 2925 `taxonomy_corrigendum_clean=true`,
Exp 2926 `constraintbench_corrigendum_ready=true`, and Exp 2933
`kan_cl_self_learning_ready=true`. If any required source artifact is absent,
malformed, or has a false/missing gate field, it MUST write a terminal artifact
with `honest_verdict="blocked_required_corrigendum_missing"`,
`matrix_v10_ready=false`, `matrix_v10_paper_boundary_ready=false`, the exact
gate errors, `no_new_llm_call=true`, and `no_new_hardware_run=true`.

When the required gates pass, the generator MUST load the Exp 2921 matrix v9
artifact and every expected `.276` artifact from Exp 2924 through Exp 2934 when
present, with Exp 2928 counted as missing when its expected GateMate bitstream
artifact is absent. It MUST classify rows as `clean`, `flagged`, `blocked`,
`missing`, `projection_only`, `diagnostic_only`, or `pilot_only`; preserve the
source checksum for every cited artifact; and preserve upstream flags from v9
and `.276` artifacts without laundering them through later clean corrigenda.

The v10 matrix rows MUST include columns for corpus or task, verifier type,
inference substrate, hardware substrate, live-LLM model specs only when the row
source declares live LLM inference, headline eligibility, and paper-claim
eligibility. Flagged, blocked, missing, projection-only, diagnostic-only, and
pilot-only rows MUST NOT become headline eligible by implication from a clean
aggregate or corrigendum artifact. Projection-only rows MAY be retained for
planning context but MUST remain outside paper-claim eligibility.

The generator MUST run the local adversarial artifact audit when available and
record the exact rerun result. If `scripts/adversarial_artifact_audit.py` is
absent, the generator MAY use the existing `scripts/adversarial_verify.py`
artifact audit fallback and MUST record which tool was used or why no audit ran.

The terminal artifact MUST include `honest_verdict`, `matrix_v10_ready`,
`matrix_v10_paper_boundary_ready`, `no_new_llm_call=true`,
`no_new_hardware_run=true`, `row_classification_counts`,
`headline_eligible_rows`, `flagged_rows`, `blocked_rows`,
`projection_only_rows`, `pilot_only_rows`, `paper_claim_boundary`,
`source_artifact_checksums`, `adversarial_audit_rerun`, `inference_substrate`
equal to `aggregation_from_upstream_artifacts`, measured `duration_s`, and
`run_date="20260523"`.

#### SCENARIO-REPORT-2935: V10 Consumes Corrigenda Without Promoting Flagged Boundaries

**Given** Exp 2924, Exp 2925, Exp 2926, and Exp 2933 gate artifacts are present
with their required ready fields true
**And** Exp 2921 matrix v9 contains flagged MBPP/HumanEval, flagged Exp 2911
and Exp 2919 rows, pilot-only code rows, diagnostic-only hardware/probe rows,
and a missing GateMate bitstream row
**And** the `.276` artifacts include a clean taxonomy corrigendum, a clean
ConstraintBench rerun, a projection-only KV260 scaling row, blocked GateMate and
LLMEval-Logic rows, and flagged citation/reformulation rows
**When** the Exp 2935 generator runs
**Then** it writes
`results/experiment_2935_cross_corpus_matrix_v10_paper_boundary_corrigendum_v1.json`
with all required fields, checksums for present sources, null checksums for
missing sources, `matrix_v10_ready=true`,
`matrix_v10_paper_boundary_ready=true`, no new LLM or hardware run, flagged
rows preserved, projection-only and pilot-only rows excluded from headline and
paper-claim eligibility, and the adversarial audit rerun recorded.

## Implementation Status (REQ-REPORT-2935)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-2935 | Implemented (`python/carnot/reporting/cross_corpus_matrix_v10_2935.py`) | Implemented (`tests/python/test_experiment_2935_cross_corpus_matrix_v10_paper_boundary_corrigendum.py`) |

### REQ-REPORT-2936: Milestone 2026.05.276 Terminal Capstone

The repository shall provide an Exp 2936 milestone capstone generator that
writes `results/experiment_2936_capstone_v276.json` using only the existing
`.276` upstream artifacts from Exp 2923 through Exp 2935. The generator MUST
NOT call an LLM, run hardware, rerun a verifier, launch synthesis, or modify
`scripts/research_conductor.py`.

The generator MUST count every expected `.276` artifact, including missing
artifacts, rather than deriving the source set only from files that exist on
disk. It MUST classify each expected artifact as one of `clean`, `flagged`,
`blocked`, `missing`, `projection_only`, `diagnostic_only`, or `pilot_only`.
Clean corrigendum artifacts MAY preserve upstream flags, but unresolved
current-artifact flags, blocked verdicts, missing artifacts, projections,
diagnostics, and pilots MUST NOT become headline or paper claim evidence.

The terminal artifact MUST summarize whether `.276` repaired the `.275`
evidence-boundary gaps for aggregation metadata, code-taxonomy provenance, and
ConstraintBench non-tautology. It MUST summarize GateMate status using the
corrected himbaechel/gmpack preflight, bitstream, and flash-smoke artifacts,
including the exact blocker when the branch is blocked. It MUST summarize
continuous self-learning status from Exp 2933 with utility and forgetting
fields.

The terminal artifact MUST include `honest_verdict`, `milestone` equal to
`2026.05.276`, `paper_ready`, `hardware_speedup_claim_eligible`,
`gate_mate_speedup_claim_eligible=false`, `evidence_boundary_repaired`,
`sota_structured_generation_clean`, `fr11_self_learning_clean`,
`clean_artifacts`, `flagged_artifacts`, `blocked_artifacts`,
`missing_artifacts`, `projection_only_artifacts`,
`diagnostic_only_artifacts`, `pilot_only_artifacts`,
`row_classification_counts`, `top_three_next_actions`,
`source_artifact_checksums`, `no_new_llm_call=true`,
`no_new_hardware_run=true`, `inference_substrate` equal to
`aggregation_from_upstream_artifacts`, measured `duration_s`, and
`run_date="20260523"`.

`paper_ready` MUST be true only when matrix v10 is ready, the v10
paper-boundary is ready, and all headline rows reported by matrix v10 are
clean. `hardware_speedup_claim_eligible` MAY remain true only for the existing
KV260 same-basis evidence boundary already preserved by matrix v10; GateMate
MUST remain speedup-ineligible unless a matched GateMate hardware-vs-CPU basis
exists. `evidence_boundary_repaired` MUST be true only when Exp 2924, Exp
2925, and Exp 2926 pass their clean gate fields.

#### SCENARIO-REPORT-2936: Capstone Closes .276 With Blocked GateMate Preserved

**Given** expected `.276` artifacts Exp 2923 through Exp 2935
**And** Exp 2928 is absent
**And** Exp 2927 records a corrected GateMate himbaechel preflight but blocked
constraints
**And** Exp 2929 is blocked by the missing bitstream
**And** Exp 2930 is projection-only
**And** Exp 2931 is blocked
**And** Exp 2932 and Exp 2934 remain flagged
**And** Exp 2935 matrix v10 preserves flagged, blocked, missing,
projection-only, diagnostic-only, and pilot-only rows
**When** the Exp 2936 capstone generator runs
**Then** it writes `results/experiment_2936_capstone_v276.json` with all
required fields, Exp 2928 listed in `missing_artifacts`, Exp 2930 listed in
`projection_only_artifacts`, blocked and flagged rows preserved in
`row_classification_counts`, `paper_ready=true`, the existing KV260
`hardware_speedup_claim_eligible=true`, `gate_mate_speedup_claim_eligible=false`,
`evidence_boundary_repaired=true`, `sota_structured_generation_clean=false`,
`fr11_self_learning_clean=true`, and a top-three `.277` action list that
prioritizes KV260 claim revalidation, same-schedule CPU comparison, and code
corpus AUPRC/base-rate validation.

## Implementation Status (REQ-REPORT-2936)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-2936 | Implemented (`python/carnot/reporting/capstone_v276_2936.py`) | Implemented (`tests/python/test_experiment_2936_capstone_v276.py`) |

### REQ-REPORT-2937: Archive Milestone 2026.05.276 and Confirm 2026.05.277 Activation

The repository shall provide an Exp 2937 archive/activation generator that
writes `results/experiment_2937_archive_v276_activate_v277.json`.

The generator MUST first require
`results/experiment_2936_capstone_v276.json`. If that capstone is absent or
malformed, it MUST write a terminal artifact with
`honest_verdict="blocked_capstone_missing"` and MUST NOT modify
`research-complete.yaml`, `research-roadmap.yaml`, or
`scripts/research_conductor.py`.

When the capstone is present, the generator MUST read
`research-complete.yaml` and determine whether milestone `2026.05.276` already
has a completed archive block. If the block is present, the generator MUST set
`archive_ready=true` without modifying `research-complete.yaml`. If the block
is absent, the generator MAY append a minimal completed archive row for
milestone `2026.05.276` from `results/experiment_2936_capstone_v276.json`
without modifying `research-roadmap.yaml`.

The generator MUST confirm milestone `2026.05.277` from
`research-roadmap-next.yaml` when that file exists. If
`research-roadmap-next.yaml` has already been activated and removed, it MUST
read the active `research-roadmap.yaml` as a fallback while still leaving that
roadmap unmodified.

The terminal artifact MUST include:

- `honest_verdict`, equal to
  `complete: archive_ready=true; archived_milestone=2026.05.276; activated_milestone=2026.05.277`
  on successful archival
- `archive_ready`
- `archived_milestone`, equal to `2026.05.276`
- `activated_milestone`, equal to `2026.05.277`
- `capstone_source`, equal to `results/experiment_2936_capstone_v276.json`
- `paper_ready_from_capstone`
- `clean_artifacts_from_capstone`
- `flagged_artifacts_from_capstone`
- `blocked_artifacts_from_capstone`
- `missing_artifacts_from_capstone`
- `pilot_only_artifacts_from_capstone`
- `artifact_classification_counts_from_capstone`
- `capstone_honest_verdict`
- `field_principles.honest_verdict`, equal to
  `Self-declared terminal state per Verdict Terminal-Prefix Discipline.`
- `inference_substrate`, equal to `aggregation_from_upstream_artifacts`
- `duration_s`, a real wall-clock duration for the archive generator run
- `run_date`, equal to `20260523`

The generator MUST copy required classification lists directly from the capstone
without reclassification or imputation. It MAY include additional capstone
summary fields, such as projection-only artifacts and headline readiness
booleans, as long as they remain direct aggregation from the capstone.

#### SCENARIO-REPORT-2937: Existing .276 Archive Confirms Active .277 Roadmap

**Given** `research-complete.yaml` already contains a completed
`2026.05.276` archive row
**And** `research-roadmap-next.yaml` may already have been activated into
`research-roadmap.yaml`
**When** the Exp 2937 archive generator runs
**Then** it writes
`results/experiment_2937_archive_v276_activate_v277.json` with
`archive_ready=true`, `archived_milestone=2026.05.276`,
`activated_milestone=2026.05.277`, paper readiness, clean artifact, flagged
artifact, blocked artifact, missing artifact, pilot-only artifact,
classification-count, capstone-verdict, field-principle, and duration fields
copied or derived from `results/experiment_2936_capstone_v276.json`,
`inference_substrate=aggregation_from_upstream_artifacts`, and no modification
to `research-roadmap.yaml` or `scripts/research_conductor.py`.

## Implementation Status (REQ-REPORT-2937)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-2937 | Implemented (`python/carnot/reporting/milestone_276_archive_277_activation.py`) | Implemented (`tests/python/test_experiment_2937_archive_v276.py`) |

### REQ-REPORT-2940: Verifier-Ensemble AUPRC Code-Corpus Base-Rate Audit

The repository shall provide an Exp 2940 AUPRC/base-rate audit generator that
writes
`results/experiment_2940_verifier_ensemble_auprc_code_corpora_v1.json` using
the checked-in Exp 2910 code-generation candidate artifact and the Exp 2837
FoVer comparison artifact. The generator MUST NOT call an LLM, rerun code
generation, launch hardware, or modify conductor/ops status files.

The generator MUST compute a precision-recall curve over Exp 2910 k=8
candidate rows using per-candidate verifier/status energy and pass/fail labels,
report code-corpus AUPRC against the empirical paper-v6 random baseline of
0.075, and report PPV, recall, and F1 at the max-F1, PPV>=0.5, and
recall>=0.8 operating points. It MUST recompute FoVer AUPRC through the same
precision-recall implementation from raw Exp 2837 score rows when present, or
from the local FoVer scoring path used by Exp 2837 when the checked-in artifact
contains only AUROC summaries.

The terminal artifact MUST include `honest_verdict`,
`inference_substrate="aggregation_from_upstream_artifacts"`,
`preconditions_checked`, `code_corpus_auprc`,
`code_corpus_baseline_random_auprc`, `fover_corpus_auprc`,
`max_f1_operating_point`, `ppv_50_operating_point`,
`recall_80_operating_point`, `paper_v6_recommendation`,
`cited_upstream_artifacts`, `methodology_note`, and measured `duration_s`.
It MUST cite SHA256 checksums for at least Exp 2910 and Exp 2837. It MUST set
`paper_v6_recommendation` to `retain` only when `code_corpus_auprc > 0.15` and
the max-F1 operating point has `f1 > 0.30`; otherwise it MUST recommend
retracting the code-corpus active-inference claim.

#### SCENARIO-REPORT-2940: AUPRC Replaces AUROC For Extreme Code Base Rate

**Given** Exp 2910 contains k=8 generated code candidates with verifier/status
scores and pass/fail labels
**And** Exp 2837 is present for the FoVer comparison baseline
**When** the Exp 2940 generator runs
**Then** it writes
`results/experiment_2940_verifier_ensemble_auprc_code_corpora_v1.json` with
code-corpus AUPRC in `[0, 1]`, a non-tautological value not exactly `0.5`, the
random-baseline AUPRC fixed at `0.075`, all three required operating points,
FoVer AUPRC from the same precision-recall code path, at least two cited
upstream artifacts with SHA256 checksums, and a paper-v6 recommendation derived
from the stated AUPRC/F1 gate.

## Implementation Status (REQ-REPORT-2940)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-2940 | Implemented (`python/carnot/reporting/verifier_ensemble_auprc_code_corpora_2940.py`) | Implemented (`tests/python/test_experiment_2940_verifier_ensemble_auprc_code_corpora.py`) |

### REQ-REPORT-2943: Cross-Corpus Matrix V11 Corrigenda And AUPRC Columns

The repository shall provide an Exp 2943 cross-corpus matrix v11 generator that
writes `results/experiment_2943_cross_corpus_matrix_v11.json` using only
checked-in upstream artifact JSON files. The generator MUST NOT call an LLM,
rerun code generation, run a verifier, launch hardware, or modify conductor,
ops, changelog, or traceability files.

The generator MUST require the Exp 2935 matrix v10 artifact and the Deep Think
corrigenda artifacts Exp 2938, Exp 2939, Exp 2940, and Exp 2942. If any required
artifact is absent, malformed, or missing a required field, it MUST write a
terminal artifact with `honest_verdict="blocked_required_upstream_missing"`,
`inference_substrate="aggregation_from_upstream_artifacts"`, the available
row buckets, `per_corpus_auprc` as an object, numeric
`kv260_same_schedule_speedup_recorded`, integer `kv260_n_crossover_measured`,
`cited_upstream_artifacts`, and measured `duration_s`.

When all required upstream artifacts are available, the generator MUST carry
forward the v10 clean, flagged, and blocked row lists, append clean corrigenda
rows for Exp 2938, Exp 2939, Exp 2940, and Exp 2942, and preserve v10 flagged
and blocked rows without promoting them. The generator MUST add per-corpus AUPRC
columns from Exp 2940 for at least FoVer and code corpora, using the Exp 2940
code-corpus AUPRC and FoVer AUPRC values without recomputation. It MUST record
the Exp 2939 same-schedule KV260 speedup from
`kv260_speedup_vs_same_schedule_cpu.value` and the Exp 2942 measured crossover
as an integer, using `0` when the hardware artifact honestly reports that no
crossover was measured.

The terminal artifact MUST include `honest_verdict`,
`inference_substrate="aggregation_from_upstream_artifacts"`, `rows_clean`,
`rows_flagged`, `rows_blocked`, `per_corpus_auprc`,
`kv260_same_schedule_speedup_recorded`, `kv260_n_crossover_measured`,
`cited_upstream_artifacts`, and measured `duration_s`. It MAY include
additional matrix rows, claim-boundary notes, checksums, run date, and schema
fields, as long as they remain direct aggregation from the upstream artifacts.

#### SCENARIO-REPORT-2943: V11 Adds AUPRC And Deep Think Corrigenda Outcomes

**Given** Exp 2935 matrix v10 is present with clean, flagged, and blocked row
lists
**And** Exp 2938 reports the KV260 MMD/KS sampling corrigendum
**And** Exp 2939 reports the same-schedule CPU/KV260 speedup value
**And** Exp 2940 reports code-corpus and FoVer AUPRC values
**And** Exp 2942 reports the measured n-scaling profile or a fixed-n
limitation
**When** the Exp 2943 generator runs
**Then** it writes `results/experiment_2943_cross_corpus_matrix_v11.json` with
all required fields, v10 row buckets preserved, clean rows added for the four
corrigenda artifacts, `per_corpus_auprc` populated from Exp 2940, the exact
same-schedule speedup recorded from Exp 2939, the measured crossover recorded
from Exp 2942 with `0` representing not measured, upstream checksums cited, and
no new model, verifier, sampler, or hardware execution.

## Implementation Status (REQ-REPORT-2943)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-2943 | Implemented (`python/carnot/reporting/cross_corpus_matrix_v11_2943.py`) | Implemented (`tests/python/test_experiment_2943_cross_corpus_matrix_v11.py`) |

### REQ-REPORT-2944: Paper-v6 Narrowing Discipline Mechanical Audit

The repository shall provide an Exp 2944 paper-v6 narrowing audit that reads
the `CLAUDE.md` "Paper-v6 Narrowing Discipline" section, represents its seven
retracted-claim forbidden phrasings as explicit regular expressions, and scans
only these targets: `docs/arxiv-paper/main.tex`, `docs/technical-report.md`,
`docs/technical-report.html`, `docs/index.html`, and the ten most recent
`results/experiment_*capstone*.json` artifacts selected deterministically by
experiment number.

The audit MUST NOT modify operator-curated documentation. It MAY auto-rewrite
autonomous capstone artifacts only by applying the post-narrowing replacement
phrasing to matched string content. The terminal artifact
`results/experiment_2944_paper_v6_narrowing_audit_v1.json` MUST include
`honest_verdict`, `inference_substrate="aggregation_from_upstream_artifacts"`,
`files_scanned`, `per_file_hits`, `n_total_hits`,
`n_operator_curated_hits_left_for_operator`,
`n_autonomous_artifact_hits_auto_fixed`, `suggested_lint_script_path`,
`cited_upstream_artifacts`, and measured `duration_s`.

#### SCENARIO-REPORT-2944: Audit Records Operator Hits And Fixes Capstone Hits

**Given** a paper-v6 policy source, operator-curated docs, and capstone JSON
artifacts containing forbidden phrases from the seven retracted claims
**When** the Exp 2944 audit runs
**Then** it records every hit with file, line, matched phrase, retracted claim
ID, and suggested fix
**And** leaves operator-curated docs unchanged
**And** rewrites only matched autonomous capstone string content to the
post-narrowing phrasing
**And** writes the required JSON deliverable fields without creating or
committing the proposed pre-commit hook.

## Implementation Status (REQ-REPORT-2944)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-2944 | Implemented (`python/carnot/reporting/paper_v6_narrowing_audit_2944.py`) | Implemented (`tests/python/test_experiment_2944_paper_v6_narrowing_audit.py`) |

### REQ-REPORT-2945: Phase-4 VFE Firewall Verification

The repository shall provide an Exp 2945 Phase-4 VFE firewall verifier that
scans the paper-v6 LaTeX source and the ten most recent capstone JSON artifacts
for Phase-4 citations that are used in a hardware, FPGA, KV260, or Glauber
context. The verifier MUST build explicit regular expressions for these
Phase-4 citations: `exp2550`, `exp2748`, `exp2753`, `exp2766`,
`Phase-4 active inference`, `variational free energy`, `FEP factor graph`, and
`FEP aggregator`.

For every Phase-4 citation hit, the verifier MUST inspect a bounded surrounding
context window for hardware references. Any co-occurrence is a firewall
violation because Phase-4 VFE bounds apply only to the RTX 3090
continuous-sampler deployment and MUST NOT defend KV260 synchronous-Glauber or
other FPGA-deployment claims. The terminal artifact
`results/experiment_2945_phase4_vfe_firewall_verification_v1.json` MUST include
`honest_verdict`, `inference_substrate="aggregation_from_upstream_artifacts"`,
`files_scanned`, `firewall_violations`, `n_violations`,
`firewall_paragraph_draft`, `cited_upstream_artifacts`, and measured
`duration_s`.

#### SCENARIO-REPORT-2945: Hardware Co-occurrences Produce Firewall Violations

**Given** paper-v6 LaTeX and recent capstone inputs contain Phase-4 citations
**When** the Exp 2945 verifier scans each citation context
**Then** it records each hardware-context co-occurrence with `file`, `line`,
`phase_4_citation`, and `hardware_context_snippet`
**And** writes an operator-integrable LaTeX firewall paragraph draft stating
that Phase-4 VFE bounds apply only to RTX 3090 continuous-sampler deployment
and cannot support KV260 synchronous-Glauber, FPGA, or hardware deployment
claims.

## Implementation Status (REQ-REPORT-2945)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-2945 | Implemented (`python/carnot/reporting/phase4_vfe_firewall_verification_2945.py`) | Implemented (`tests/python/test_experiment_2945_phase4_vfe_firewall_verification.py`) |

### REQ-REPORT-2948: Milestone 2026.05.277 Terminal Capstone

The repository shall provide an Exp 2948 milestone capstone generator that
writes `results/experiment_2948_capstone_v277.json` using only the existing
`.277` upstream artifacts from Exp 2937 through Exp 2947. The generator MUST
NOT call an LLM, run hardware, rerun a verifier, launch synthesis, or modify
`scripts/research_conductor.py`.

The generator MUST classify every expected `.277` artifact as one of `clean`,
`flagged`, `blocked`, or `missing`. The classifier MUST treat known
adversarial-verify false-positives (exp2939 TAUTOLOGY from
`experiment_id == random_seed`, exp2941 same pattern, exp2943 aggregation-
substrate methodology/duration flags) as `clean` and document the override
reason in module comments.

The terminal artifact MUST synthesize the three Deep Think Corrigenda
outcomes into a `deep_think_corrigenda_outcomes` block with at least
`mmd_distinguishable`, `same_schedule_speedup`, and
`code_auprc_recommendation` fields, plus a `headline_outcome` field whose
value is one of `narrow` / `rescue` / `additional_rounds_needed`.

The terminal artifact MUST include `paper_v6_safe_claims` and
`paper_v6_forbidden_claims` lists. The forbidden list MUST cite every
CLAUDE.md Paper-v6 Narrowing Discipline retracted-claim id `(#2)`, `(#3)`,
`(#6)`, `(#7)`, `(#8)`, `(#9)`, `(#10)` so paper-v6 LaTeX-side narrowing
work can grep for them.

The terminal artifact MUST include a
`narrowing_discipline_compliance_audit` field shaped as a list of
`{file, hits, fixes_applied}` rows derived from the Exp 2944 audit's
`per_file_hits` and `audit_resolution_by_operator` blocks.

`paper_ready` MUST be true only when: (a) the three Deep Think Corrigenda
artifacts landed clean, (b) the cross-corpus matrix v11 reports
`matrix_v11_ready=true`, (c) the Phase-4 VFE firewall reports zero
violations, (d) every narrowing-audit hit's resolution is operator-
authorized and matches a terminal token (`resolved`, `applied`, or
`false_positive`), and (e) the `headline_outcome` is `narrow` or
`rescue`.

The terminal artifact MUST include `honest_verdict`, `milestone` equal to
`2026.05.277`, `inference_substrate` equal to
`aggregation_from_upstream_artifacts`, `clean_artifacts`,
`flagged_artifacts`, `blocked_artifacts`, `missing_artifacts`,
`artifact_classification_counts`, `deep_think_corrigenda_outcomes`,
`paper_v6_safe_claims`, `paper_v6_forbidden_claims`,
`narrowing_discipline_compliance_audit`, `top_3_next_actions`,
`gaps_for_278`, `cited_upstream_artifacts`, `source_artifact_status`,
`field_principles`, `no_new_llm_call=true`, `no_new_hardware_run=true`,
measured `duration_s`, and `run_date="20260523"`.

#### SCENARIO-REPORT-2948: Capstone Closes .277 With Narrowing Confirmed

**Given** expected `.277` artifacts Exp 2937 through Exp 2947 are all present
**And** Exp 2938 reports `distributions_distinguishable=true` with all three
seed MMD p-values <= 0.001
**And** Exp 2939 reports a `kv260_speedup_vs_same_schedule_cpu` value < 1.0
**And** Exp 2940 reports `paper_v6_recommendation.value="retain"` with
code-corpus AUPRC well above the 0.075 base rate
**And** Exp 2943 reports `matrix_v11_ready=true`
**And** Exp 2944 records the Paper-v6 narrowing audit with operator-authorized
resolutions for every hit
**And** Exp 2945 reports zero firewall violations
**When** the Exp 2948 capstone generator runs
**Then** it writes `results/experiment_2948_capstone_v277.json` with all
required fields, `paper_ready=true`, `headline_outcome="narrow"`,
`deep_think_corrigenda_outcomes.mmd_distinguishable=true`,
`deep_think_corrigenda_outcomes.same_schedule_speedup` below 1.0,
`deep_think_corrigenda_outcomes.code_auprc_recommendation="retain"`,
and a `top_3_next_actions` list that prioritizes operator narrowing edits,
stronger candidate generation for the next AUPRC re-measure, and an OOD
seventh cross-corpus row.

## Implementation Status (REQ-REPORT-2948)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-2948 | Implemented (`python/carnot/reporting/capstone_v277_2948.py`) | Implemented (`tests/python/test_experiment_2948_capstone_v277.py`) |

### REQ-REPORT-2949: Archive Milestone 2026.05.277 and Confirm 2026.05.278 Activation

The repository shall provide an Exp 2949 archive/activation generator that
writes `results/experiment_2949_archive_v277_activate_v278.json` using only
checked-in roadmap, archive-ledger, and Exp 2948 capstone artifacts. The
generator MUST NOT modify `research-roadmap.yaml` or
`scripts/research_conductor.py`, MUST NOT push, MUST NOT call an LLM, and MUST
NOT run hardware or synthesis.

The generator MUST read `results/experiment_2948_capstone_v277.json` and copy
the `.277` archive evidence without reclassifying it: `paper_ready`,
`deep_think_corrigenda_outcomes.headline_outcome`, clean artifacts, flagged
artifacts, blocked artifacts, missing artifacts, and the `.278` next-gap list.
If `deep_think_corrigenda_outcomes.headline_outcome` is missing, it MUST use an
empty string rather than fabricating a headline outcome.

The generator MUST ensure `research-complete.yaml` contains exactly one
completed `2026.05.277` milestone archive row after the run. If the row already
exists, it MUST leave the archive ledger unchanged. If the row is absent, it
MUST append a minimal archive row that cites the Exp 2948 capstone deliverable
without rewriting unrelated historical milestone entries.

The generator MUST confirm that milestone `2026.05.278` is activated from
roadmap state. It MUST prefer `research-roadmap-next.yaml` when that file is
present; when it is absent because activation has already occurred, it MUST use
the active `research-roadmap.yaml` as a read-only fallback and record that
fallback in artifact metadata.

The terminal artifact MUST include `honest_verdict`,
`archived_milestone="2026.05.277"`,
`activated_milestone="2026.05.278"`,
`capstone_source="results/experiment_2948_capstone_v277.json"`,
`paper_ready_from_capstone`, `headline_outcome_from_capstone`,
`clean_artifacts_from_capstone`, `flagged_artifacts_from_capstone`,
`blocked_artifacts_from_capstone`, `missing_artifacts_from_capstone`,
`next_gaps_from_capstone`, `archive_ready`,
`inference_substrate="aggregation_from_upstream_artifacts"`, measured
`duration_s`, and `run_date="20260523"`. It MAY include additional
audit-trace fields as long as they remain direct aggregation from upstream
artifacts and document that `research-roadmap.yaml`,
`scripts/research_conductor.py`, `ops/changelog.md`, `ops/status.md`, and
`_bmad/traceability.md` were not modified by this task.

#### SCENARIO-REPORT-2949: Existing .277 Archive Confirms Active .278 Roadmap

**Given** `results/experiment_2948_capstone_v277.json` exists
**And** `research-complete.yaml` already contains a completed
`2026.05.277` archive row
**And** `research-roadmap-next.yaml` is absent because
`research-roadmap.yaml` is already activated at `2026.05.278`
**When** the Exp 2949 generator runs
**Then** it writes `results/experiment_2949_archive_v277_activate_v278.json`
with all required fields, `archive_ready=true`,
`paper_ready_from_capstone=true`, `headline_outcome_from_capstone` copied from
`deep_think_corrigenda_outcomes.headline_outcome`, zero duplicate `.277`
archive rows, `activation.used_active_roadmap_fallback=true`, and unchanged
`research-roadmap.yaml` and `scripts/research_conductor.py`.

## Implementation Status (REQ-REPORT-2949)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-2949 | Implemented (`python/carnot/reporting/milestone_277_archive_278_activation.py`) | Implemented (`tests/python/test_experiment_2949_archive_v277.py`) |

### REQ-REPORT-2960: Cross-Corpus Matrix V12 From .278 Artifacts

The repository shall provide an Exp 2960 cross-corpus matrix v12 generator
that writes `results/experiment_2960_cross_corpus_matrix_v12.json` using only
checked-in upstream artifact JSON files. The generator MUST start from
`results/experiment_2943_cross_corpus_matrix_v11.json`, carry forward the
`.277` narrowing facts from `results/experiment_2948_capstone_v277.json`, and
read every available completed `results/experiment_295*.json` `.278` artifact
without rerunning live inference, verifier scoring, synthesis, board flashing,
or hardware smoke tests.

The generator MUST require Exp 2954 to report
`self_learning_utility_artifact_ready=true`; if this precondition is not met,
it MUST write a terminal blocked artifact rather than fabricating a self-
learning utility row. Missing or blocked hardware branches MUST be reported in
their own row classes instead of promoted to clean evidence.

The matrix v12 artifact MUST preserve the Paper-v6 Narrowing Discipline claim
boundary from `.277`: no KV260 speedup claim, no KV260 Boltzmann or
thermalization claim, no TSU/Kona performance claim, and no broad verifier
generalization beyond the measured rows. Rows added in v12 MUST include
structured repair, threshold policy, self-learning utility, GateMate,
PolarFire, and NL-to-Z3 evidence and each row MUST be labeled as one of
`clean`, `flagged`, `blocked`, `gated-skipped`, `pilot-only`, or
`aggregation-only`.

The terminal artifact MUST include `honest_verdict`, `matrix_v12_ready`,
`inference_substrate="aggregation_from_upstream_artifacts"`,
`upstream_artifacts_read`, `upstream_checksums`, `clean_rows`, `flagged_rows`,
`blocked_rows`, `gated_skipped_rows`, `pilot_only_rows`,
`forbidden_claims_absent`, `code_repair_delta_summary`,
`self_learning_delta_summary`, `hardware_state_summary`,
`solver_state_summary`, and measured `duration_s`. It MAY include compact
`matrix_rows`, `aggregation_only_rows`, schema, run date, and no-new-execution
booleans as long as every value is derived from upstream JSON fields.

#### SCENARIO-REPORT-2960: V12 Aggregates .278 Rows Without New Execution

**Given** matrix v11 and the .277 capstone are present
**And** Exp 2950 through Exp 2959 artifacts are present or honestly blocked
**And** Exp 2954 reports `self_learning_utility_artifact_ready=true`
**When** the Exp 2960 matrix v12 generator runs
**Then** it writes `results/experiment_2960_cross_corpus_matrix_v12.json`
with all required fields, upstream checksums for every source it read, v11 row
buckets preserved, v12 rows added for structured repair, threshold policy,
self-learning utility, GateMate, PolarFire, and NL-to-Z3, flagged rows kept
separate from clean rows, blocked or gated-skipped hardware rows kept separate
from clean rows, compact deltas relative to `.277`, and no forbidden Paper-v6
claim phrasing.

## Implementation Status (REQ-REPORT-2960)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-2960 | Implemented (`python/carnot/reporting/cross_corpus_matrix_v12_2960.py`) | Implemented (`tests/python/test_experiment_2960_cross_corpus_matrix_v12.py`) |

### REQ-REPORT-2961: Milestone 2026.05.278 Terminal Capstone

The repository shall provide an Exp 2961 milestone capstone generator that
writes `results/experiment_2961_capstone_v278.json` by aggregating only
checked-in `.278` roadmap, upstream result, and matrix artifacts. The
generator MUST NOT call an LLM, rerun tests for an upstream experiment, run
hardware, launch synthesis, push, or modify `scripts/research_conductor.py`.

The generator MUST read every available `results/experiment_295*.json` and
`results/experiment_2960*.json` artifact at closeout, plus
`results/experiment_2948_capstone_v277.json` and the active milestone roadmap.
It MUST classify every planned `.278` task as exactly one of `clean`,
`flagged`, `blocked`, `gated-skipped`, `missing`, `pilot-only`, or
`aggregation-only`. Missing branch outcomes MUST remain `missing` rather than
being inferred from downstream artifacts.

The capstone MUST summarize code repair, self-learning, solver, GateMate, and
PolarFire outcomes from source artifact fields. It MUST restate forbidden
claims that remain forbidden, including KV260 speedup, KV260 Boltzmann or
thermalization, TSU/Kona performance, broad hardware acceleration, and broad
verifier generalization beyond measured rows.

`paper_ready` MUST be true only when the `.278` evidence preserves or improves
paper-v6's narrowed `.277` claim set, `results/experiment_2960_cross_corpus_matrix_v12.json`
is ready, forbidden claims are absent, and no new unresolved flagged, blocked,
gated-skipped, missing, or pilot-only `.278` artifact affects the narrowed paper
claim set. Planning-only or aggregation-only artifacts MAY remain outside the
paper-ready gate when they explicitly make no paper claim.

The terminal artifact MUST include `honest_verdict`,
`milestone="2026.05.278"`, `paper_ready`, `headline_outcome`,
`clean_artifacts`, `flagged_artifacts`, `blocked_artifacts`,
`gated_skipped_artifacts`, `missing_artifacts`, `pilot_only_artifacts`,
`aggregation_only_artifacts`, `gaps_closed`, `gaps_remaining`,
`forbidden_claims_absent`, `next_milestone_recommendations`,
`inference_substrate="aggregation_from_upstream_artifacts"`, and measured
`duration_s`. It MAY include compact outcome summaries, source checksums,
classification details, and no-new-execution booleans as long as they are
derived from upstream artifacts.

#### SCENARIO-REPORT-2961: Capstone Closes .278 Without Fabricating Branches

**Given** the active roadmap describes milestone `2026.05.278`
**And** the `.277` capstone and every available `.278` result artifact through
Exp 2960 are present or honestly absent
**When** the Exp 2961 capstone generator runs
**Then** it writes `results/experiment_2961_capstone_v278.json` with all
required fields, classifies every planned `.278` task into one terminal bucket,
records clean code-repair and threshold-policy evidence separately from flagged
or blocked artifacts, keeps GateMate flash smoke blocked when the board outcome
is blocked, keeps planning and matrix rows aggregation-only, sets
`paper_ready=false` when unresolved flags or blocked/missing paper-relevant
branches remain, reports `forbidden_claims_absent=true`, and recommends two to
four concrete `.279` gaps.

## Implementation Status (REQ-REPORT-2961)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-2961 | Implemented (`python/carnot/reporting/capstone_v278_2961.py`) | Implemented (`tests/python/test_experiment_2961_capstone_v278.py`) |

### REQ-REPORT-2962: Archive Milestone 2026.05.278 and Confirm 2026.05.279 Activation

The repository shall provide an Exp 2962 archive/activation generator that
writes `results/experiment_2962_archive_v278_activate_v279.json` using only
checked-in roadmap, archive-ledger, and Exp 2961 capstone artifacts. The
generator MUST NOT modify `research-roadmap.yaml`, MUST NOT modify
`scripts/research_conductor.py`, MUST NOT push, MUST NOT call an LLM, and MUST
NOT run hardware, synthesis, verifier scoring, or solver execution.

The generator MUST read `results/experiment_2961_capstone_v278.json` and copy
the `.278` archive evidence without reclassifying it: `paper_ready`,
`headline_outcome`, clean artifacts, flagged artifacts, blocked artifacts,
missing artifacts, artifact-classification counts, and next-gap evidence from
`gaps_remaining` or the next-milestone recommendations. Missing optional fields
MUST become empty strings, empty lists, or empty mappings rather than fabricated
values.

The generator MUST ensure `research-complete.yaml` contains exactly one
completed `2026.05.278` milestone archive row after the run. If the row already
exists, it MUST leave the archive ledger unchanged. If the row is absent, it
MUST append a minimal archive row that cites the Exp 2961 capstone deliverable
without rewriting unrelated historical milestone entries.

The generator MUST confirm that milestone `2026.05.279` is activated from
roadmap state. It MUST prefer `research-roadmap-next.yaml` when that file is
present; when it is absent because activation has already occurred, it MUST use
the active `research-roadmap.yaml` as a read-only fallback and record that
fallback in artifact metadata.

The terminal artifact MUST include `honest_verdict`,
`archived_milestone="2026.05.278"`,
`activated_milestone="2026.05.279"`,
`capstone_source="results/experiment_2961_capstone_v278.json"`,
`paper_ready_from_capstone`, `headline_outcome_from_capstone`,
`clean_artifacts_from_capstone`, `flagged_artifacts_from_capstone`,
`blocked_artifacts_from_capstone`, `missing_artifacts_from_capstone`,
`archive_ready`, `inference_substrate="aggregation_from_upstream_artifacts"`,
and measured `duration_s`. It MAY include additional audit-trace fields,
including `next_gaps_from_capstone`, as long as they remain direct aggregation
from upstream artifacts and document that `research-roadmap.yaml`,
`scripts/research_conductor.py`, `ops/changelog.md`, `ops/status.md`, and
`_bmad/traceability.md` were not modified by this task.

#### SCENARIO-REPORT-2962: Existing .278 Archive Confirms Active .279 Roadmap

**Given** `results/experiment_2961_capstone_v278.json` exists
**And** `research-complete.yaml` already contains a completed
`2026.05.278` archive row
**And** `research-roadmap-next.yaml` is absent because
`research-roadmap.yaml` is already activated at `2026.05.279`
**When** the Exp 2962 generator runs
**Then** it writes `results/experiment_2962_archive_v278_activate_v279.json`
with all required fields, `archive_ready=true`,
`paper_ready_from_capstone=false`, `headline_outcome_from_capstone` copied
from the capstone `headline_outcome`, zero duplicate `.278` archive rows,
`activation.used_active_roadmap_fallback=true`, and unchanged
`research-roadmap.yaml` and `scripts/research_conductor.py`.

## Implementation Status (REQ-REPORT-2962)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-2962 | Implemented (`python/carnot/reporting/milestone_278_archive_279_activation.py`) | Implemented (`tests/python/test_experiment_2962_archive_v278.py`) |

### REQ-REPORT-2973: Cross-Corpus Matrix V13 From .279 Artifacts

The repository shall provide an Exp 2973 cross-corpus matrix v13 generator
that writes `results/experiment_2973_cross_corpus_matrix_v13.json` using only
checked-in upstream artifact JSON files. The generator MUST start from
`results/experiment_2960_cross_corpus_matrix_v12.json`, carry forward the
`.278` capstone facts from `results/experiment_2961_capstone_v278.json`, and
read every available completed `.279` artifact from Exp 2962 through Exp 2972
without rerunning live inference, verifier scoring, solver execution, synthesis,
board flashing, or hardware smoke tests.

The generator MUST require Exp 2969 to report
`non_tautological_self_learning_ready=true`; if this precondition is not met,
it MUST write a terminal blocked artifact rather than fabricating self-learning
or KAN-memory rows. Missing or blocked `.279` branches MUST be reported in
their own row classes instead of promoted to clean evidence.

The matrix v13 artifact MUST preserve the `.277` and `.278` claim boundaries:
no KV260 speedup claim, no KV260 Boltzmann or thermalization claim, no TSU or
Kona performance claim, and no native EBT training claim. Rows added in v13
MUST include DCCD repair, BEAVER-style certificates, solver-frontier
formalization, partial monitors, non-tautological FR-11, KAN forgetting guard,
and GateMate flash state, with each row labeled as one of `clean`, `flagged`,
`blocked`, `gated-skipped`, `pilot-only`, or `aggregation-only`.

The terminal artifact MUST include `honest_verdict`, `matrix_v13_ready`,
`inference_substrate="aggregation_from_upstream_artifacts"`,
`upstream_artifacts_read`, `upstream_checksums`, `clean_rows`, `flagged_rows`,
`blocked_rows`, `gated_skipped_rows`, `pilot_only_rows`,
`forbidden_claims_absent`, `repair_replication_summary`,
`solver_frontier_summary`, `self_learning_summary`, `kan_memory_summary`,
`hardware_state_summary`, and measured `duration_s`. It MAY include compact
`matrix_rows`, `aggregation_only_rows`, schema, run date, and no-new-execution
booleans as long as every value is derived from upstream JSON fields.

#### SCENARIO-REPORT-2973: V13 Aggregates .279 Rows Without New Execution

**Given** matrix v12 and the .278 capstone are present
**And** Exp 2962 through Exp 2972 artifacts are present or honestly blocked
**And** Exp 2969 reports `non_tautological_self_learning_ready=true`
**When** the Exp 2973 matrix v13 generator runs
**Then** it writes `results/experiment_2973_cross_corpus_matrix_v13.json`
with all required fields, upstream checksums for every source it read, v12 row
buckets preserved, v13 rows added for DCCD repair, BEAVER-style certificates,
solver-frontier formalization, partial monitors, non-tautological FR-11, KAN
forgetting guard, and GateMate flash state, flagged rows kept separate from
clean rows, blocked or gated-skipped hardware rows kept separate from clean
rows, compact deltas relative to `.278`, and no forbidden Paper-v6 claim
phrasing.

## Implementation Status (REQ-REPORT-2973)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-2973 | Implemented (`python/carnot/reporting/cross_corpus_matrix_v13_2973.py`) | Implemented (`tests/python/test_experiment_2973_cross_corpus_matrix_v13.py`) |

### REQ-REPORT-2974: Milestone 2026.05.279 Terminal Capstone

The repository shall provide an Exp 2974 milestone capstone generator that
writes `results/experiment_2974_capstone_v279.json` by aggregating only
checked-in `.279` roadmap, upstream result, and matrix artifacts. The
generator MUST NOT call an LLM, rerun tests for an upstream experiment, run
hardware, launch synthesis, push, modify `scripts/research_conductor.py`, or
modify ops/changelog/status/traceability files during the conductor closeout
task.

The generator MUST read every available `.279` task artifact from Exp 2962
through Exp 2973, the active milestone roadmap, the `.278` terminal capstone,
and the v13 matrix when present. It MUST classify every planned `.279` task as
exactly one of `clean`, `flagged`, `blocked`, `gated-skipped`, `missing`,
`pilot-only`, or `aggregation-only`. Missing branch outcomes MUST remain
`missing` or `gated-skipped` according to explicit roadmap gates rather than
being inferred from downstream artifacts.

The capstone MUST summarize DCCD code repair, BEAVER-style certificates,
solver-frontier formalization, partial monitors, FR-11 non-tautology, KAN
memory, and GateMate outcomes from source artifact fields. It MUST identify
which of the three biggest `.279` gaps closed and which remain open. It MUST
restate forbidden claims that remain forbidden while keeping those
re-statements separate from paper-safe claim prose.

`paper_ready` MUST be true only when `.279` clears `.278`'s flagged code,
FR-11, and solver rows or preserves them as explicitly non-headline without new
adversarial flags, the v13 matrix is ready, forbidden claims are absent, and
there are no unresolved current `.279` flagged, blocked, gated-skipped,
missing, or pilot-only artifacts affecting the narrowed paper claim set.

The terminal artifact MUST include `honest_verdict`,
`milestone="2026.05.279"`, `paper_ready`, `headline_outcome`,
`clean_artifacts`, `flagged_artifacts`, `blocked_artifacts`,
`gated_skipped_artifacts`, `missing_artifacts`, `pilot_only_artifacts`,
`aggregation_only_artifacts`, `gaps_closed`, `gaps_remaining`,
`forbidden_claims_absent`, `next_milestone_recommendations`,
`inference_substrate="aggregation_from_upstream_artifacts"`, and measured
`duration_s`. It MAY include compact outcome summaries, source checksums,
classification details, safe/forbidden claim lists, and no-new-execution
booleans as long as they are derived from upstream artifacts.

#### SCENARIO-REPORT-2974: Capstone Closes .279 Without Promoting Flagged Rows

**Given** the active roadmap describes milestone `2026.05.279`
**And** the `.278` capstone and every available `.279` result artifact through
Exp 2973 are present or honestly absent
**And** Exp 2964, Exp 2967, Exp 2969, and Exp 2973 remain adversarially flagged
**And** Exp 2968 is a pilot-only partial monitor harness
**When** the Exp 2974 capstone generator runs
**Then** it writes `results/experiment_2974_capstone_v279.json` with all
required fields, classifies every planned `.279` task into one terminal bucket,
records GateMate board contact/flash and KAN forgetting guard as bounded clean
outcomes, keeps DCCD repair, solver formalization, FR-11, and matrix v13 flags
out of headline claims, sets `paper_ready=false`, reports
`forbidden_claims_absent=true`, and recommends two to four concrete `.280`
gaps.

## Implementation Status (REQ-REPORT-2974)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-2974 | Implemented (`python/carnot/reporting/capstone_v279_2974.py`) | Implemented (`tests/python/test_experiment_2974_capstone_v279.py`) |

### REQ-REPORT-2975: Archive Milestone 2026.05.279 and Confirm 2026.05.280 Activation

The repository shall provide an Exp 2975 archive/activation generator that
writes `results/experiment_2975_archive_v279_activate_v280.json` using only
checked-in roadmap, archive-ledger, and Exp 2974 capstone artifacts. The
generator MUST NOT modify `research-roadmap.yaml`, MUST NOT modify
`scripts/research_conductor.py`, MUST NOT push, MUST NOT call an LLM, and MUST
NOT run hardware, synthesis, verifier scoring, solver execution, or fresh
research experiments.

The generator MUST read `results/experiment_2974_capstone_v279.json` and copy
the `.279` archive evidence without reclassifying it: `paper_ready`,
`headline_outcome`, clean artifacts, flagged artifacts, blocked artifacts,
missing artifacts, pilot-only artifacts, artifact-classification counts, and
next-gap evidence from `gaps_remaining` or the next-milestone
recommendations. Missing optional fields MUST become empty strings, empty
lists, or empty mappings rather than fabricated values.

The generator MUST ensure `research-complete.yaml` contains exactly one
completed `2026.05.279` milestone archive row after the run. If the row already
exists, it MUST leave the archive ledger unchanged. If the row is absent, it
MUST append a minimal archive row that cites the Exp 2974 capstone deliverable
without rewriting unrelated historical milestone entries.

The generator MUST confirm that milestone `2026.05.280` is activated from
roadmap state. It MUST prefer `research-roadmap-next.yaml` when that file is
present; when it is absent because activation has already occurred, it MUST use
the active `research-roadmap.yaml` as a read-only fallback and record that
fallback in artifact metadata.

The terminal artifact MUST include `honest_verdict`,
`archived_milestone="2026.05.279"`,
`activated_milestone="2026.05.280"`,
`capstone_source="results/experiment_2974_capstone_v279.json"`,
`paper_ready_from_capstone`, `headline_outcome_from_capstone`,
`clean_artifacts_from_capstone`, `flagged_artifacts_from_capstone`,
`blocked_artifacts_from_capstone`, `missing_artifacts_from_capstone`,
`pilot_only_artifacts_from_capstone`, `archive_ready`,
`inference_substrate="aggregation_from_upstream_artifacts"`, and measured
`duration_s`. It MAY include additional audit-trace fields, including
`artifact_classification_counts_from_capstone`, `next_gaps_from_capstone`, and
read-only roadmap hashes, as long as every value is direct aggregation from
upstream artifacts and the artifact documents that `research-roadmap.yaml`,
`scripts/research_conductor.py`, `ops/changelog.md`, `ops/status.md`, and
`_bmad/traceability.md` were not modified by this task.

#### SCENARIO-REPORT-2975: Existing .279 Archive Confirms Active .280 Roadmap

**Given** `results/experiment_2974_capstone_v279.json` exists
**And** `research-complete.yaml` already contains a completed
`2026.05.279` archive row
**And** `research-roadmap-next.yaml` is absent because
`research-roadmap.yaml` is already activated at `2026.05.280`
**When** the Exp 2975 generator runs
**Then** it writes `results/experiment_2975_archive_v279_activate_v280.json`
with all required fields, `archive_ready=true`,
`paper_ready_from_capstone=false`, `headline_outcome_from_capstone` copied
from the capstone `headline_outcome`, zero duplicate `.279` archive rows,
`activation.used_active_roadmap_fallback=true`, and unchanged
`research-roadmap.yaml` and `scripts/research_conductor.py`.

## Implementation Status (REQ-REPORT-2975)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-2975 | Implemented (`python/carnot/reporting/milestone_279_archive_280_activation.py`) | Implemented (`tests/python/test_experiment_2975_archive_v279.py`) |

### REQ-REPORT-2986: Cross-Corpus Matrix V14 From .280 Artifacts

The repository shall provide an Exp 2986 cross-corpus matrix v14 generator
that writes `results/experiment_2986_cross_corpus_matrix_v14.json` using only
checked-in upstream artifact JSON files. The generator MUST read
`results/experiment_2973_cross_corpus_matrix_v13.json`,
`results/experiment_2974_capstone_v279.json`, and every available `.280`
artifact from Exp 2975 through Exp 2985 without rerunning live inference,
verifier scoring, solver execution, synthesis, board flashing, readback, or
hardware smoke tests.

The generator MUST require
`results/experiment_2982_fr11_independent_metric_utility_gate_v4.json` to
report `fr11_independent_metrics_evaluated=true`. If that precondition is not
met, it MUST write a terminal blocked artifact rather than fabricating a clean
self-learning row.

The matrix v14 rows MUST classify every source artifact by terminal status
(`clean`, `flagged`, `blocked`, `gated-skipped`, `pilot-only`, or
`projection-only`), claim class, evidence type, model compliance, hardware
compliance, and prior-failure outcome. Repair, solver, FR-11, monitor, and
hardware rows MUST be clean only when their own clean/readiness booleans pass
and their claim-boundary guards do not allow unsupported headline, model,
hardware-speedup, sampler, thermodynamic, or full-streaming claims. Missing or
gated `.280` rows MUST remain explicit gated-skipped rows rather than being
inferred from downstream artifacts.

The terminal artifact MUST include `honest_verdict`, `matrix_v14_ready`,
`milestone="2026.05.280"`, `clean_count`, `flagged_count`, `blocked_count`,
`gated_skipped_count`, `pilot_only_count`, `projection_only_count`,
`row_count`, `rows`, `repair_claim_status`, `solver_claim_status`,
`fr11_claim_status`, `hardware_claim_status`, `model_compliance_summary`,
`claim_boundary_violations`, `next_milestone_recommendations`,
`inference_substrate="aggregation_from_upstream_artifacts"`, and measured
`duration_s`. It MAY include source-artifact checksums and no-new-execution
booleans as long as every value is derived from upstream JSON fields.

#### SCENARIO-REPORT-2986: V14 Aggregates .280 Rows With Explicit Claim Boundaries

**Given** matrix v13 and the `.279` capstone are present
**And** Exp 2975 through Exp 2985 artifacts are present or honestly absent
**And** Exp 2982 reports `fr11_independent_metrics_evaluated=true`
**When** the Exp 2986 matrix v14 generator runs
**Then** it writes `results/experiment_2986_cross_corpus_matrix_v14.json`
with all required fields, carry-forward rows for prior `.279` buckets,
explicit rows for every `.280` artifact, flagged code-repair and solver rows
kept separate from clean rows, Exp 2982 classified clean only when its
independent metric and guard fields pass, GateMate readback/smoke limitations
classified as blocked hardware evidence, Exp 2985 classified projection-only,
and next milestone recommendations that preserve those claim boundaries.

## Implementation Status (REQ-REPORT-2986)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-2986 | Planned (`python/carnot/reporting/cross_corpus_matrix_v14_2986.py`) | Planned (`tests/python/test_experiment_2986_cross_corpus_matrix_v14.py`) |

### REQ-REPORT-2987: Milestone .280 Terminal Capstone

The repository shall provide an Exp 2987 milestone capstone generator that
writes `results/experiment_2987_capstone_v280.json` by aggregating only local
upstream artifacts from milestone `2026.05.280`, including every available
artifact from Exp 2975 through Exp 2986 and matrix v14 when present. The
generator MUST NOT rerun live inference, verifier scoring, solver execution,
synthesis, board flashing, readback, hardware smoke tests, or the conductor.

The capstone MUST classify local artifacts as `clean`, `flagged`, `blocked`,
`missing`, `gated-skipped`, `pilot-only`, or `projection-only`. It MUST verify
each loaded artifact's deliverable path, `honest_verdict`, required claim
booleans, model-compliance evidence when applicable, hardware claim-boundary
fields when applicable, and prior-failure outcome as recorded by matrix v14.

The terminal artifact MUST include `honest_verdict`, `milestone="2026.05.280"`,
`paper_ready`, `headline_outcome`, `clean_artifacts`, `flagged_artifacts`,
`blocked_artifacts`, `missing_artifacts`, `gated_skipped_artifacts`,
`pilot_only_artifacts`, `projection_only_artifacts`, `gaps_closed`,
`gaps_remaining`, `repair_ready`, `solver_ready`, `fr11_ready`,
`hardware_ready`, `model_compliance_summary`,
`hardware_claim_boundary_summary`, `retirement_recommendations`,
`next_milestone_recommendations`,
`inference_substrate="aggregation_from_upstream_artifacts"`, and measured
`duration_s`. It MAY include artifact audit rows, source checksums, matrix row
counts, and no-new-execution booleans as long as every value is derived from
local artifacts.

`paper_ready` MUST be true only when all of these local-evidence gates pass:
repair is ready from a clean intent-preserving repair rerun, solver feedback is
ready without unresolved artifact flags, FR-11 independent self-learning is
ready, GateMate hardware readback or smoke-vector evidence is ready, no
required capstone source is missing or blocked, and matrix v14 reports no claim
boundary violations. Projection-only and pilot-only artifacts MUST remain out
of headline claims.

#### SCENARIO-REPORT-2987: Capstone Closes .280 Without Promoting Flagged Rows

**Given** every available `.280` artifact from Exp 2975 through Exp 2986 is
present or honestly absent
**And** matrix v14 records clean FR-11 evidence but blocked repair and hardware
rows plus flagged solver model-compliance evidence
**When** the Exp 2987 capstone generator runs
**Then** it writes `results/experiment_2987_capstone_v280.json` with all
required fields, `paper_ready=false`, `repair_ready=false`,
`solver_ready=false`, `fr11_ready=true`, `hardware_ready=false`, classification
lists copied from local artifact and matrix evidence, explicit model and
hardware claim-boundary summaries, next-milestone recommendations, and
retirement recommendations for repeated failed or overclaimed scopes.

## Implementation Status (REQ-REPORT-2987)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-2987 | Planned (`python/carnot/reporting/capstone_v280_2987.py`) | Planned (`tests/python/test_experiment_2987_capstone_v280.py`) |

### REQ-REPORT-2988: Archive Milestone 2026.05.280 and Confirm 2026.05.281 Activation

The repository shall provide an Exp 2988 archive/activation generator that
writes `results/experiment_2988_archive_v280_activate_v281.json` using only
checked-in roadmap, archive-ledger, Exp 2987 capstone, and local upstream
result artifacts. The generator MUST NOT modify `scripts/research_conductor.py`,
MUST NOT push, MUST NOT call an LLM, and MUST NOT rerun live inference,
verifier scoring, solver execution, synthesis, board flashing, readback,
hardware smoke tests, or fresh research experiments.

The generator MUST read `results/experiment_2987_capstone_v280.json` and every
local `.280` result artifact referenced by its `artifact_audit` rows. It MUST
summarize clean, flagged, blocked, projection-only, pilot-only, and missing
counts without repairing or reclassifying historical artifacts. Flagged and
blocked rows MUST be carried forward explicitly with experiment id, path,
classification, honest verdict, upstream flags, and prior-failure outcome.

The generator MUST ensure `research-complete.yaml` contains exactly one
completed `2026.05.280` milestone archive row after the run. If the row already
exists, it MUST leave the archive ledger unchanged. If the row is absent, it
MUST append the completed `.280` task list and capstone outcome without
rewriting unrelated historical milestone entries.

The generator MUST confirm that milestone `2026.05.281` is activated from
roadmap state. It MUST prefer `research-roadmap-next.yaml` when that file is
present; when it is absent because activation has already occurred, it MUST use
the active `research-roadmap.yaml` as a read-only fallback, require non-empty
tasks, and record that fallback in artifact metadata.

The terminal artifact MUST include `archive_ready`,
`archived_milestone="2026.05.280"`,
`activated_milestone="2026.05.281"`, `research_complete_updated`,
`status_updates_written`, `n_tasks_archived`,
`blocked_or_flagged_rows_carried_forward`, `validation_commands`, and
`honest_verdict` with an accepted terminal prefix. It MAY include additional
audit-trace fields such as artifact classification counts, capstone source
metadata, source-read counts, read-only roadmap hashes, and notes, as long as
every value is direct aggregation from local artifacts. When the conductor
prompt directs ops-doc reconciliation to a separate step, the generator MUST
leave `ops/status.md`, `ops/changelog.md`, and `_bmad/traceability.md`
unchanged and report `status_updates_written=false` honestly.

#### SCENARIO-REPORT-2988: Existing .280 Archive Confirms Active .281 Roadmap

**Given** `results/experiment_2987_capstone_v280.json` exists
**And** `research-complete.yaml` already contains a completed
`2026.05.280` archive row with the completed `.280` task list
**And** `research-roadmap-next.yaml` is absent because
`research-roadmap.yaml` is already activated at `2026.05.281` with non-empty
tasks
**When** the Exp 2988 generator runs
**Then** it writes `results/experiment_2988_archive_v280_activate_v281.json`
with all required fields, `archive_ready=true`,
`research_complete_updated=true`, `n_tasks_archived=13`,
`activation.used_active_roadmap_fallback=true`, classification counts copied
from the capstone audit, flagged and blocked rows carried forward, and
unchanged `research-roadmap.yaml`, `scripts/research_conductor.py`,
`ops/status.md`, and `ops/changelog.md`.

## Implementation Status (REQ-REPORT-2988)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-2988 | Implemented (`python/carnot/reporting/milestone_280_archive_281_activation.py`) | Implemented (`tests/python/test_experiment_2988_archive_v280.py`) |

### REQ-REPORT-2998: Cross-Corpus Matrix V15 From .281 Artifacts

The repository shall provide an Exp 2998 cross-corpus matrix v15 generator
that writes `results/experiment_2998_cross_corpus_matrix_v15.json` using only
checked-in upstream artifact JSON files. The generator MUST read
`results/experiment_2986_cross_corpus_matrix_v14.json`,
`results/experiment_2987_capstone_v280.json`, and every available `.281`
artifact from Exp 2988 through Exp 2997 without rerunning live inference,
verifier scoring, solver execution, synthesis, board flashing, readback, or
hardware smoke tests.

The generator MUST build an aggregation artifact even when upstream `.281`
tasks are flagged, blocked, gated-skipped, projection-only, pilot-only, or
missing. Missing or gated upstream tasks MUST remain explicit rows rather than
being inferred from downstream artifacts. The matrix rows MUST separately track
SOTA cache provenance, verifier-backed hard-code manifest readiness,
intent-preserving repair, solver-provenance reproduction, AquaForte/BEAVER
substrate corrigendum, prompt-validator protocol evidence, FR-11
self-learning evidence, GateMate host-visible smoke/readback evidence, and
SSQA dual-BRAM RTL/PnR/resource evidence.

The generator MUST preserve claim-boundary distinctions across PRD
requirements, OpenSpec reporting requirements, paper-v6 anchored claim
boundaries, hardware boundaries, and FR-11 self-learning boundaries. Live LLM
inference, verifier-only or deterministic artifacts, aggregation artifacts,
and hardware smoke artifacts MUST remain distinguishable by row evidence type
and `inference_substrate`. Rows MUST NOT promote pilot-only, projection-only,
flagged, blocked, gated-skipped, or missing evidence into paper-v6 headline
claims. Hardware rows MUST NOT claim sampler speedup, Boltzmann
thermalization, same-basis CPU-vs-FPGA speedup, Extropic execution, NPU
acceleration, photonic execution, or broad hardware sovereignty without
authenticated source evidence. FR-11 rows MUST keep verifier-grounded trace
memory separate from broad autonomous self-improvement or model-weight update
claims.

The terminal artifact MUST include `matrix_v15_ready`, `n_clean`,
`n_flagged`, `n_blocked`, `n_gated_skipped`, `n_pilot_only`,
`n_projection_only`, `claim_rows`, `hardware_claim_boundary`,
`self_learning_claim_boundary`, `unresolved_blockers`, and `honest_verdict`.
It MAY include `n_missing`, `rows`, row-count metadata, source checksums,
paper-v6 boundary summaries, PRD/OpenSpec boundary summaries, no-new-execution
booleans, and next-milestone recommendations as long as every value is derived
from upstream JSON fields or checked-in claim-boundary documents.

#### SCENARIO-REPORT-2998: V15 Aggregates .281 Rows Without Requiring All Upstream Successes

**Given** matrix v14 and the `.280` capstone are present
**And** Exp 2988 through Exp 2997 artifacts are present, flagged, blocked,
gated-skipped, projection-only, pilot-only, or honestly absent
**When** the Exp 2998 matrix v15 generator runs
**Then** it writes `results/experiment_2998_cross_corpus_matrix_v15.json`
with all required fields, `matrix_v15_ready=true`, explicit counts for every
terminal status bucket, explicit claim rows for SOTA cache, hard-code manifest,
repair, solver provenance, AquaForte/BEAVER substrate, prompt-validator
protocol, FR-11 self-learning, GateMate, and SSQA, hardware and self-learning
claim-boundary dictionaries, unresolved blockers for real gaps, and an
`honest_verdict` that states readiness and counts without promoting blocked,
flagged, pilot-only, projection-only, gated-skipped, or missing evidence.

## Implementation Status (REQ-REPORT-2998)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-2998 | Planned (`python/carnot/reporting/cross_corpus_matrix_v15_2998.py`) | Planned (`tests/python/test_experiment_2998_cross_corpus_matrix_v15.py`) |

### REQ-REPORT-2999: Milestone 2026.05.281 Terminal Capstone

The repository shall provide an Exp 2999 milestone capstone generator that
writes `results/experiment_2999_capstone_v281.json` using only checked-in
local artifacts. The generator MUST read
`results/experiment_2998_cross_corpus_matrix_v15.json`,
`results/experiment_2987_capstone_v280.json`, and every available `.281`
artifact from Exp 2988 through Exp 2997 without rerunning live inference,
verifier scoring, solver execution, synthesis, board flashing, readback,
hardware smoke tests, the conductor, or external publication tooling.

The capstone MUST decide `paper_ready` from local evidence only. It MUST treat
flagged repair evidence, flagged AquaForte/BEAVER substrate evidence, blocked
GateMate readback/smoke evidence, missing SSQA evidence, unresolved matrix
blockers, and paper-v6 claim-boundary violations as paper-readiness blockers.
It MAY report that paper-v6 is closer to readiness when clean SOTA cache,
hard-code manifest, solver provenance, prompt-validator, or FR-11 evidence has
landed, but that proximity MUST NOT trigger external publication.

The capstone MUST enumerate matrix rows by terminal status (`clean`,
`flagged`, `blocked`, `missing`, and `gated-skipped`) and preserve
pilot-only/projection-only evidence as non-headline context. It MUST state
which `.280` gaps were closed, which remain, and the exact next-milestone
actions required for conductor planning. Hardware rows MUST keep GateMate and
SSQA bounded to host-visible readback/smoke, RTL/PnR/resource, and explicit
no-speedup/no-thermodynamic-claim evidence. FR-11 rows MUST stay bounded to
verifier-grounded trace memory rather than broad autonomous self-improvement or
model-weight update claims.

The terminal artifact MUST include `capstone_ready`, `paper_ready`,
`clean_artifacts`, `flagged_artifacts`, `blocked_artifacts`,
`missing_artifacts`, `gated_skipped_artifacts`, `gaps_closed`,
`gaps_remaining`, `next_milestone_recommendations`,
`external_publication_triggered`, and `honest_verdict`. It MAY include source
checksums, paper-readiness blockers, proof summaries, no-new-execution
booleans, matrix counts, and measured `duration_s` as long as every value is
derived from local artifacts.

#### SCENARIO-REPORT-2999: Capstone Synthesizes .281 Without Publication

**Given** matrix v15 is present and reports `matrix_v15_ready=true`
**And** the `.280` capstone is present
**And** Exp 2988 through Exp 2997 artifacts are present, flagged, blocked,
gated-skipped, or honestly absent
**When** the Exp 2999 capstone generator runs
**Then** it writes `results/experiment_2999_capstone_v281.json` with all
required fields, `capstone_ready=true`, `external_publication_triggered=false`,
row-status lists copied from matrix v15, `paper_ready=false` when any required
claim row remains flagged/blocked/missing/gated-skipped, explicit `.280` gap
closure and remaining-gap lists, hardware and FR-11 boundaries preserved, and
an `honest_verdict` that states capstone readiness and paper status.

## Implementation Status (REQ-REPORT-2999)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-2999 | Implemented (`python/carnot/reporting/capstone_v281_2999.py`) | Implemented (`tests/python/test_experiment_2999_capstone_v281.py`) |

### REQ-REPORT-3000: Archive Milestone 2026.05.281 and Confirm 2026.05.282 Activation

The repository shall provide an Exp 3000 archive/activation generator that
writes `results/experiment_3000_archive_v281_activate_v282.json` using only
checked-in roadmap state, `research-complete.yaml`,
`results/experiment_2999_capstone_v281.json`, and the local `.281` result
artifacts referenced by that capstone. The generator MUST NOT modify
`scripts/research_conductor.py`, MUST NOT push, MUST NOT call an LLM, and MUST
NOT rerun live inference, verifier scoring, solver execution, synthesis, board
flashing, readback, hardware smoke tests, or historical experiments.

The generator MUST read `results/experiment_2999_capstone_v281.json` and all
`.281` result artifacts referenced by its `source_artifacts_read` entries when
they exist locally. Missing referenced artifacts MUST remain missing and MUST
NOT be repaired or synthesized. The generator MUST summarize clean, flagged,
blocked, gated-skipped, projection-only, pilot-only, and missing counts from
the capstone or matrix evidence and MUST carry blocked, flagged, and missing
rows forward explicitly.

The generator MUST ensure `research-complete.yaml` contains exactly one
completed `2026.05.281` milestone archive row after the run. If the row already
exists, it MUST leave the archive ledger unchanged. If the row is absent and
the capstone is present, it MAY append the completed `.281` task list and
capstone outcome without rewriting unrelated historical milestone entries.

The generator MUST confirm that milestone `2026.05.282` is activated from
roadmap state. It MUST prefer `research-roadmap-next.yaml` when that file is
present; when it is absent because activation has already occurred, it MUST use
the active `research-roadmap.yaml` as a read-only fallback, require non-empty
tasks, and record that fallback in artifact metadata. When an outer conductor
prompt directs ops-doc reconciliation to a separate step, the generator MUST
leave `ops/status.md`, `ops/changelog.md`, and `_bmad/traceability.md`
unchanged and report `status_updates_written=false` honestly unless those docs
already contain both milestone identifiers.

The terminal artifact MUST include `archive_ready`,
`archived_milestone="2026.05.281"`,
`activated_milestone="2026.05.282"`, `research_complete_updated`,
`status_updates_written`, `n_tasks_archived`,
`blocked_or_flagged_rows_carried_forward`, `validation_commands`, and
`honest_verdict` with an accepted terminal prefix. It MAY include source-read
summaries, capstone and matrix classification counts, read-only roadmap hashes,
ops-doc mutation checks, and notes as long as every value is direct aggregation
from local artifacts.

#### SCENARIO-REPORT-3000: Existing .281 Archive Confirms Active .282 Roadmap

**Given** `results/experiment_2999_capstone_v281.json` exists
**And** `research-complete.yaml` already contains a completed
`2026.05.281` archive row with the completed `.281` task list
**And** `research-roadmap-next.yaml` is absent because
`research-roadmap.yaml` is already activated at `2026.05.282` with non-empty
tasks
**When** the Exp 3000 generator runs
**Then** it writes `results/experiment_3000_archive_v281_activate_v282.json`
with all required fields, `archive_ready=true`,
`research_complete_updated=true`, `n_tasks_archived=12`,
`activation.used_active_roadmap_fallback=true`, classification counts copied
from the capstone evidence, blocked, flagged, and missing rows carried
forward, and unchanged `research-roadmap.yaml`,
`scripts/research_conductor.py`, `ops/status.md`, `ops/changelog.md`, and
`_bmad/traceability.md`.

## Implementation Status (REQ-REPORT-3000)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-3000 | Implemented (`python/carnot/reporting/milestone_281_archive_282_activation.py`) | Implemented (`tests/python/test_experiment_3000_archive_v281.py`) |

### REQ-REPORT-3010: Cross-Corpus Matrix V16 From .282 Artifacts

The repository shall provide an Exp 3010 cross-corpus matrix v16 generator
that writes `results/experiment_3010_cross_corpus_matrix_v16.json` using only
checked-in upstream artifact JSON files and checked-in claim-boundary
documents. The generator MUST read
`results/experiment_2998_cross_corpus_matrix_v15.json`,
`results/experiment_2999_capstone_v281.json`, every available `.282`
artifact from Exp 3000 through Exp 3009, the active `.282` roadmap, and the
paper-v6/OpenSpec/PRD hardware-boundary documents needed to classify claims.
It MUST NOT rerun live inference, verifier scoring, solver execution,
synthesis, board flashing, readback, hardware smoke tests, the conductor, or
external publication tooling.

The generator MUST build a terminal aggregation artifact even when upstream
`.282` tasks are flagged, blocked, gated-skipped, projection-only,
pilot-only, or missing. Missing artifacts MUST remain explicit in source-read
metadata. Structured gates MUST remain visible: Exp 3009 is
`gated-skipped` when Exp 3008 reports `host_visible_io_ready=false`, even if
the Exp 3009 JSON artifact is absent. The matrix rows MUST separately track
SOTA GGUF cache readiness, metamorphic oracle readiness, SOTA hard-set repair,
AquaForte/BEAVER substrate provenance, validator-tree expansion,
fixed-point diagnostics, FR-11 trace-memory stability, GateMate host-visible
IO, and SSQA dual-BRAM RTL/PnR/resource status.

The generator MUST preserve claim-boundary distinctions across PRD FR-11 and
FR-12, OpenSpec reporting requirements, paper-v6 narrowing, hardware
boundaries, and the `.282` roadmap acceptance criteria. Rows MUST NOT promote
flagged, blocked, gated-skipped, projection-only, pilot-only, or missing
evidence into paper-v6 headline claims. The matrix MUST detect and report
unsupported boundary claims including LLM-as-verifier authority, false SOTA
headline use, TSU/Kona access or parity claims, and GateMate/KV260 hardware
speedup or thermodynamic claims. FR-11 evidence MUST stay bounded to
verifier-grounded trace-memory stability unless independent held-out metrics,
negative controls, drift checks, and forgetting checks are clean and
unflagged.

The terminal artifact MUST include `matrix_v16_ready`, `clean_count`,
`flagged_count`, `blocked_count`, `gated_skipped_count`, `missing_count`,
`projection_only_count`, `repaired_claims`, `still_blocked_claims`,
`claim_boundary_violations`, `recommended_next_actions`, and
`honest_verdict`. It MAY include `pilot_only_count`, `rows`, row-count
metadata, claim-row summaries, source checksums, missing-artifact metadata,
paper-v6/PRD/OpenSpec boundary summaries, no-new-execution booleans, and
measured `duration_s` as long as every value is derived from upstream JSON
fields or checked-in claim-boundary documents. When a conductor prompt assigns
ops reconciliation to a separate step, the generator MUST leave
`ops/status.md`, `ops/changelog.md`, and `_bmad/traceability.md` unchanged.

#### SCENARIO-REPORT-3010: V16 Aggregates .282 Rows Without Requiring Upstream Success

**Given** matrix v15 and the `.281` capstone are present
**And** Exp 3000 through Exp 3009 artifacts are present, flagged, blocked,
gated-skipped, projection-only, pilot-only, or honestly absent
**And** Exp 3008 reports `host_visible_io_ready=false`
**When** the Exp 3010 matrix v16 generator runs
**Then** it writes `results/experiment_3010_cross_corpus_matrix_v16.json`
with `matrix_v16_ready=true`, explicit counts for every terminal status
bucket, explicit source-read metadata for missing artifacts, Exp 3009
classified as `gated-skipped`, claim rows for the .282 repair, substrate,
FR-11, GateMate, and SSQA branches, repaired-claim names for clean claim
repairs, still-blocked claim names for unresolved rows, exact recommended
next actions, no unsupported paper-v6/hardware/Kona/TSU/LLM-verifier claim
promotion, and an `honest_verdict` that states readiness and counts.

## Implementation Status (REQ-REPORT-3010)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-3010 | Implemented (`python/carnot/reporting/cross_corpus_matrix_v16_3010.py`) | Implemented (`tests/python/test_experiment_3010_cross_corpus_matrix_v16.py`) |

### REQ-REPORT-3011: Milestone 2026.05.282 Terminal Capstone

The repository shall provide an Exp 3011 milestone capstone generator that
writes `results/experiment_3011_capstone_v282.json` using only checked-in
upstream JSON artifacts and checked-in claim-boundary documents. The generator
MUST read `results/experiment_3010_cross_corpus_matrix_v16.json`,
`results/experiment_2999_capstone_v281.json`, and every available `.282`
artifact from Exp 3000 through Exp 3009. It MUST NOT rerun live inference,
verifier scoring, solver execution, synthesis, board flashing, readback,
hardware smoke tests, the conductor, or external publication tooling.

The capstone MUST use matrix v16 as the authority for row status, while still
counting the terminal task set explicitly. It MUST classify Exp 3000 through
Exp 3010 as `clean`, `flagged`, `blocked`, `gated-skipped`, `missing`,
`pilot-only`, or `projection-only`, and MUST keep carried-forward matrix
blockers visible separately from task-scoped status rows. Missing Exp 3009
evidence MUST remain visible even when its task row is classified
`gated-skipped` because Exp 3008 did not open the host-visible IO gate.

The capstone MUST decide whether `.282` repaired the `.281` blockers without
promoting claim boundaries. Exp 3003 MUST repair the Exp 2991 repair-methodology
blocker only when its matrix row is clean. Exp 3004 MAY repair only the
AquaForte/BEAVER substrate-provenance blocker when its matrix row is clean; it
MUST NOT be treated as solving the BEAVER task itself. Exp 3007 MUST remain
bounded to verifier-grounded FR-11 trace-memory stability and MUST NOT promote
flagged held-out or tautology evidence. Exp 3008 and Exp 3009 MUST remain
bounded to GateMate host-visible IO and SSQA RTL/PnR/resource evidence without
sampler, speedup, thermodynamic, KV260, TSU, Kona, or hardware-sovereignty
claims.

The capstone MUST set `paper_ready=false` unless every claimed result has
durable verifier evidence, no false SOTA substitution, no live/substrate
ambiguity, no unresolved flagged/blocked/gated-skipped/missing matrix row, and
no hardware claim-boundary breach. External publication MUST remain disallowed
by the artifact even if a future synthetic input makes `paper_ready=true`.
When a conductor prompt assigns ops reconciliation to a separate step, the
generator MUST leave `ops/status.md`, `ops/changelog.md`, and
`_bmad/traceability.md` unchanged.

The terminal artifact MUST include `capstone_ready`, `paper_ready`,
`n_tasks_evaluated`, `repaired_rows`, `flagged_rows`, `blocked_rows`,
`gated_skipped_rows`, `missing_rows`, `publication_action_allowed`,
`next_milestone_recommendation`, and `honest_verdict`. It MAY include
task-scoped rows, matrix-wide status lists, repaired and unrepaired blocker
decisions, source checksums, paper-readiness blockers, no-new-execution
booleans, and measured `duration_s` as long as every value is derived from
upstream artifacts or checked-in claim-boundary documents.

#### SCENARIO-REPORT-3011: Capstone Synthesizes .282 Go/No-Go Without Publication

**Given** matrix v16 is present and reports `matrix_v16_ready=true`
**And** the `.281` capstone is present
**And** Exp 3000 through Exp 3009 artifacts are present, flagged, blocked,
gated-skipped, projection-only, pilot-only, or honestly absent
**When** the Exp 3011 capstone generator runs
**Then** it writes `results/experiment_3011_capstone_v282.json` with
`capstone_ready=true`, task classifications for Exp 3000 through Exp 3010,
matrix-wide flagged/blocked/gated-skipped/missing row lists, Exp 3004 promoted
only as substrate provenance, Exp 3003/3007/3008/3009 kept non-promotable,
`paper_ready=false`, `publication_action_allowed=false`, an exact next
milestone recommendation, no ops-doc mutations, and an `honest_verdict` that
states capstone readiness and paper readiness.

## Implementation Status (REQ-REPORT-3011)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-3011 | Implemented (`python/carnot/reporting/capstone_v282_3011.py`) | Implemented (`tests/python/test_experiment_3011_capstone_v282.py`) |

### REQ-REPORT-3012: Archive Milestone 2026.05.282 and Confirm 2026.05.283 Activation

The repository shall provide an Exp 3012 archive/activation generator that
writes `results/experiment_3012_archive_v282_activate_v283.json` using only
checked-in roadmap state, `research-complete.yaml`,
`results/experiment_3011_capstone_v282.json`,
`results/experiment_3010_cross_corpus_matrix_v16.json`, and the local `.282`
result artifacts referenced by the capstone. The generator MUST NOT modify
`scripts/research_conductor.py`, MUST NOT modify `research-roadmap.yaml`, MUST
NOT push, MUST NOT call an LLM, and MUST NOT rerun live inference, verifier
scoring, solver execution, synthesis, board flashing, readback, hardware smoke
tests, or historical experiments.

The generator MUST read the Exp 3011 capstone and all `.282` result artifacts
referenced by its `source_artifacts_read` entries when they exist locally.
Missing referenced artifacts MUST remain missing and MUST NOT be repaired or
synthesized. The generator MUST summarize clean, flagged, blocked,
gated-skipped, projection-only, pilot-only, missing, and adversarially flagged
counts from the capstone and matrix evidence. Blocked, flagged, missing,
gated-skipped, and adversarially flagged rows MUST be carried forward
explicitly so downstream planning can see unresolved work.

The generator MUST ensure `research-complete.yaml` contains exactly one
completed `2026.05.282` milestone archive row after the run. If the row already
exists, it MUST leave the archive ledger unchanged. If the row is absent and
the capstone is present, it MAY append the completed `.282` task list and
capstone outcome without rewriting unrelated historical milestone entries.

The generator MUST confirm that milestone `2026.05.283` is activated from
roadmap state. It MUST prefer `research-roadmap-next.yaml` when that file is
present; when it is absent because activation has already occurred, it MUST use
the active `research-roadmap.yaml` as a read-only fallback, require non-empty
tasks, and record that fallback in artifact metadata. When an outer conductor
prompt directs ops-doc reconciliation to a separate step, the generator MUST
leave `ops/status.md`, `ops/changelog.md`, and `_bmad/traceability.md`
unchanged and report `status_updates_written=false` honestly unless those docs
already contain both milestone identifiers.

The terminal artifact MUST include `archive_ready`,
`archived_milestone="2026.05.282"`,
`activated_milestone="2026.05.283"`, `research_complete_updated`,
`status_updates_written`, `n_tasks_archived`,
`blocked_or_flagged_rows_carried_forward`,
`adversarial_flags_carried_forward`,
`inference_substrate="aggregation_from_upstream_artifacts"`,
`validation_commands`, and `honest_verdict` with an accepted terminal prefix.
It MUST NOT include a top-level `model_specs` field. It MAY include source-read
summaries, capstone and matrix classification counts, read-only roadmap hashes,
ops-doc mutation checks, and notes as long as every value is direct aggregation
from local artifacts.

#### SCENARIO-REPORT-3012: Existing .282 Archive Confirms Active .283 Roadmap

**Given** `results/experiment_3011_capstone_v282.json` exists
**And** `research-complete.yaml` already contains a completed
`2026.05.282` archive row with the completed `.282` task list
**And** `research-roadmap-next.yaml` is absent because
`research-roadmap.yaml` is already activated at `2026.05.283` with non-empty
tasks
**When** the Exp 3012 generator runs
**Then** it writes `results/experiment_3012_archive_v282_activate_v283.json`
with all required fields, `archive_ready=true`,
`research_complete_updated=true`, `n_tasks_archived=12`,
`activation.used_active_roadmap_fallback=true`, classification counts copied
from the capstone and matrix evidence, blocked, flagged, missing,
gated-skipped, and adversarially flagged rows carried forward, no top-level
`model_specs`, and unchanged `research-roadmap.yaml`,
`scripts/research_conductor.py`, `ops/status.md`, `ops/changelog.md`, and
`_bmad/traceability.md`.

## Implementation Status (REQ-REPORT-3012)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-3012 | Implemented (`python/carnot/reporting/milestone_282_archive_283_activation.py`) | Implemented (`tests/python/test_experiment_3012_archive_v282.py`) |

### REQ-REPORT-3024: Cross-Corpus Matrix V17 From .283 Artifacts

The repository shall provide an Exp 3024 cross-corpus matrix v17 generator
that writes `results/experiment_3024_cross_corpus_matrix_v17.json` using only
checked-in upstream artifact JSON files and checked-in claim-boundary
documents. The generator MUST read
`results/experiment_3011_capstone_v282.json`, every available `.283` artifact
from Exp 3012 through Exp 3023, the active `.283` roadmap or recorded
activation fallback, and the paper-v6/OpenSpec/PRD hardware-boundary documents
needed to classify claims. It MUST NOT rerun live inference, verifier scoring,
solver execution, synthesis, board flashing, readback, hardware smoke tests,
the conductor, or external publication tooling.

The generator MUST build a terminal aggregation artifact even when upstream
`.283` tasks are flagged, blocked, gated-skipped, projection-only, pilot-only,
or missing. Missing artifacts and missing roadmap documents MUST remain
explicit in source-read metadata. Rows MUST separately track `.283` archive
activation, SOTA GGUF telemetry, repair taxonomy, acceptance controller,
acceptance-controlled repair, NSVIF validator trees, BEAVER-style frontier
certificates, FR-11 feasibility diagnostics, FR-11 verifier-feedback
self-learning, GateMate transport, GateMate flash/smoke, and SSQA explicit
gate status.

The generator MUST preserve claim-boundary distinctions across PRD FR-11 and
FR-12, OpenSpec reporting requirements, paper-v6 narrowing, hardware
boundaries, and the `.283` roadmap acceptance criteria. Rows MUST NOT promote
flagged, blocked, gated-skipped, projection-only, pilot-only, or missing
evidence into paper-v6 headline claims. The matrix MUST report whether Exp
3016 repair, Exp 3020 FR-11 self-learning, Exp 3022 GateMate IO, and Exp 3023
SSQA are promotable, flagged, blocked, or gate-skipped. The matrix MUST detect
and report unsupported boundary claims including LLM-as-verifier authority,
false SOTA headline use, TSU/Kona access or parity claims, and GateMate/KV260
hardware speedup, sampler, Boltzmann, or thermodynamic claims.

Because Exp 3024 is aggregation-only, the terminal artifact MUST set
`inference_substrate=aggregation_from_upstream_artifacts` and MUST NOT include
top-level `model_specs`, `target_model`, CUDA, or GGUF fields. Source model and
hardware provenance MAY be cited only under `cited_upstream_artifacts`.

The terminal artifact MUST include `matrix_v17_ready`, `clean_count`,
`flagged_count`, `blocked_count`, `gated_skipped_count`, `missing_count`,
`projection_only_count`, `repaired_claims`, `still_blocked_claims`,
`claim_boundary_violations`, `cited_upstream_artifacts`,
`inference_substrate`, `recommended_next_actions`, and `honest_verdict`. It
MAY include `pilot_only_count`, `rows`, row-count metadata, claim-row
summaries, source checksums, missing-artifact metadata, paper-v6/PRD/OpenSpec
boundary summaries, no-new-execution booleans, and measured `duration_s` as
long as every value is derived from upstream JSON fields or checked-in
claim-boundary documents. When a conductor prompt assigns ops reconciliation
to a separate step, the generator MUST leave `ops/status.md`,
`ops/changelog.md`, and `_bmad/traceability.md` unchanged.

#### SCENARIO-REPORT-3024: V17 Aggregates .283 Rows Without Live Metadata Leakage

**Given** the `.282` capstone is present
**And** Exp 3012 through Exp 3023 artifacts are present, flagged, blocked,
gated-skipped, projection-only, pilot-only, or honestly absent
**And** Exp 3022 is blocked by a structured GateMate transport gate
**When** the Exp 3024 matrix v17 generator runs
**Then** it writes `results/experiment_3024_cross_corpus_matrix_v17.json`
with `matrix_v17_ready=true`, explicit counts for every terminal status
bucket, explicit source-read metadata for missing artifacts and missing
roadmap documents, claim rows for Exp 3016, Exp 3020, Exp 3022, and Exp 3023,
repaired-claim names for clean or explicitly repaired non-headline claim
repairs, still-blocked claim names for unresolved rows, exact recommended next
actions, no unsupported paper-v6/hardware/Kona/TSU/LLM-verifier claim
promotion, source model/hardware details only under `cited_upstream_artifacts`,
and an `honest_verdict` that states readiness and counts.

## Implementation Status (REQ-REPORT-3024)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-3024 | Implemented (`python/carnot/reporting/cross_corpus_matrix_v17_3024.py`) | Implemented (`tests/python/test_experiment_3024_cross_corpus_matrix_v17.py`) |

### REQ-REPORT-3025: Milestone 2026.05.283 Terminal Capstone

The repository shall provide an Exp 3025 milestone capstone generator that
writes `results/experiment_3025_capstone_v283.json` using only checked-in
upstream JSON artifacts and checked-in claim-boundary documents. The generator
MUST read `results/experiment_3024_cross_corpus_matrix_v17.json`,
`results/experiment_3011_capstone_v282.json`, every available `.283` artifact
from Exp 3012 through Exp 3023, the active `.283` roadmap or recorded
activation fallback, and the paper-v6/OpenSpec/PRD/roadmap claim-boundary
documents needed to make the go/no-go decision. It MUST NOT rerun live
inference, verifier scoring, solver execution, synthesis, board flashing,
readback, hardware smoke tests, the conductor, or external publication tooling.

The capstone MUST use matrix v17 as the authority for row status. It MUST
classify Exp 3012 through Exp 3024 as `clean`, `flagged`, `blocked`,
`gated-skipped`, `missing`, `pilot-only`, or `projection-only`, while keeping
matrix-wide carried-forward flagged, blocked, gated-skipped, missing,
pilot-only, and projection-only rows visible separately from task-scoped rows.
The capstone MUST decide whether repair, FR-11 self-learning, GateMate IO, and
SSQA are promotable from matrix v17 claim rows without broadening their claim
boundaries.

The capstone MUST decide whether `.283` repaired the `.282` blockers. Exp 3016
MUST repair the Exp 3003 repair-methodology blocker only when the matrix row is
clean and no adversarial/methodology flags remain. Exp 3020 MAY repair the Exp
3007 FR-11 stability blocker only as verifier-feedback controller utility over
exact cached traces; it MUST NOT promote native LLM training, broad autonomous
self-improvement, or tautological feasibility evidence. Exp 3022 MUST repair
the Exp 3008 hardware IO blocker only when host-visible deterministic output is
captured. Exp 3023 MAY repair only the Exp 3009 missing-artifact blocker when
the explicit SSQA artifact is written; SSQA remains non-promotable until
GateMate host-visible IO is ready and bounded RTL/PnR/resource evidence exists.
Exp 3024 MAY repair the Exp 3011 aggregation false-positive risk only when the
capstone keeps top-level `inference_substrate` equal to
`aggregation_from_upstream_artifacts` and keeps source model/hardware details
under cited upstream-artifact provenance rather than top-level live-inference
metadata.

The capstone MUST set `paper_ready=false` unless every claimed result has
durable verifier evidence, no false SOTA substitution, no live/substrate
ambiguity, no aggregation-live-inference false positive, no unresolved
flagged/blocked/gated-skipped/missing matrix row, and no hardware
claim-boundary breach. External publication MUST remain disallowed by the
artifact even if a future synthetic input makes `paper_ready=true`. When a
conductor prompt assigns ops reconciliation to a separate step, the generator
MUST leave `ops/status.md`, `ops/changelog.md`, and `_bmad/traceability.md`
unchanged.

The terminal artifact MUST include `capstone_ready`, `paper_ready`,
`n_tasks_evaluated`, `repaired_rows`, `flagged_rows`, `blocked_rows`,
`gated_skipped_rows`, `missing_rows`, `cited_upstream_artifacts`,
`inference_substrate`, `publication_action_allowed`,
`next_milestone_recommendation`, and `honest_verdict`. It MAY include
task-scoped rows, promotion decisions, repaired/unrepaired blocker decisions,
source checksums, paper-readiness blockers, no-new-execution booleans, ops-doc
mutation flags, and measured `duration_s` as long as every value is derived
from upstream artifacts or checked-in claim-boundary documents.

#### SCENARIO-REPORT-3025: Capstone Synthesizes .283 Go/No-Go Without Publication

**Given** matrix v17 is present and reports `matrix_v17_ready=true`
**And** the `.282` capstone is present
**And** Exp 3012 through Exp 3023 artifacts are present, flagged, blocked,
gated-skipped, projection-only, pilot-only, or honestly absent
**When** the Exp 3025 capstone generator runs
**Then** it writes `results/experiment_3025_capstone_v283.json` with
`capstone_ready=true`, task classifications for Exp 3012 through Exp 3024,
matrix-wide flagged/blocked/gated-skipped/missing row lists, repair kept
non-promotable while adversarial/methodology flags remain, Exp 3020 promoted
only as bounded verifier-feedback controller utility, GateMate and SSQA kept
non-promotable until host-visible IO exists, `paper_ready=false`,
`publication_action_allowed=false`, source model/hardware details only under
`cited_upstream_artifacts`, an exact next milestone recommendation, no
ops-doc mutations, and an `honest_verdict` that states capstone readiness and
paper readiness.

## Implementation Status (REQ-REPORT-3025)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-3025 | Implemented (`python/carnot/reporting/capstone_v283_3025.py`) | Implemented (`tests/python/test_experiment_3025_capstone_v283.py`) |

### REQ-REPORT-3026: Archive Milestone 2026.05.283 and Confirm 2026.05.284 Activation

The repository shall provide an Exp 3026 archive/activation generator that
writes `results/experiment_3026_archive_v283_activate_v284.json` using only
checked-in roadmap state, `research-complete.yaml`,
`results/experiment_3024_cross_corpus_matrix_v17.json`, and
`results/experiment_3025_capstone_v283.json`. The generator MUST NOT modify
`research-roadmap.yaml`, MUST NOT modify `scripts/research_conductor.py`, MUST
NOT rename roadmap files, MUST NOT push, MUST NOT call an LLM, and MUST NOT
rerun live inference, verifier scoring, solver execution, synthesis, board
flashing, readback, hardware smoke tests, or historical experiments.

The generator MUST confirm that the Exp 3025 capstone exists, reports
`capstone_ready=true`, and records the previous paper-readiness decision from
the explicit `paper_ready` field. It MUST summarize `.283` clean, flagged,
blocked, gated-skipped, projection-only, pilot-only, missing, adversarially
flagged, and paper-ready status from Exp 3024 and Exp 3025 without repairing or
rewriting historical rows. Blocked, flagged, gated-skipped, projection-only,
pilot-only, missing, and adversarially flagged rows MUST be carried forward
explicitly in the activation artifact.

The generator MUST confirm that milestone `2026.05.284` is available from
roadmap state and points to
`openspec/change-proposals/research-roadmap-vNEXT.md`. It MUST prefer
`research-roadmap-next.yaml` when that file is present. When
`research-roadmap-next.yaml` is absent because activation has already occurred,
it MUST use the active `research-roadmap.yaml` as read-only activation
evidence, require non-empty tasks, and record the fallback honestly rather than
recreating, renaming, or editing roadmap files.

The terminal artifact MUST include `milestone_archived`,
`next_milestone="2026.05.284"`, `next_roadmap_path`, `capstone_ready`,
`previous_paper_ready`, `carry_forward_blockers`,
`protected_files_unchanged`, `inference_substrate`, and `honest_verdict` with
an accepted terminal prefix unless a precondition is honestly blocked. The
`inference_substrate` field MUST be an object declaring aggregation-only
evidence and MUST NOT include top-level live model metadata such as
`model_specs`, CUDA, GGUF, or target-model fields. The artifact MAY include
source checksums, protected-file hashes, roadmap metadata, next execution
order, source-read summaries, and no-new-execution booleans as long as every
value is derived from checked-in artifacts or read-only roadmap state.

#### SCENARIO-REPORT-3026: Completed .283 Archive Confirms Active .284 Roadmap

**Given** `results/experiment_3025_capstone_v283.json` exists and reports
`capstone_ready=true`
**And** `research-complete.yaml` already contains a completed `2026.05.283`
archive row with the completed `.283` task list
**And** `research-roadmap-next.yaml` is absent because `research-roadmap.yaml`
is already activated at `2026.05.284` with non-empty tasks and
`milestone_doc=openspec/change-proposals/research-roadmap-vNEXT.md`
**When** the Exp 3026 generator runs
**Then** it writes `results/experiment_3026_archive_v283_activate_v284.json`
with all required fields, `milestone_archived=true`,
`next_milestone="2026.05.284"`, `next_roadmap_path="research-roadmap.yaml"`,
`capstone_ready=true`, `previous_paper_ready=false`,
`protected_files_unchanged=true`, explicit status summaries and carry-forward
blockers, aggregation-only `inference_substrate`, no top-level live-model
metadata, and unchanged `research-roadmap.yaml` and
`scripts/research_conductor.py`.

## Implementation Status (REQ-REPORT-3026)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-3026 | Planned (`python/carnot/reporting/milestone_283_archive_284_activation.py`) | Planned (`tests/python/test_experiment_3026_archive_v283.py`) |

### REQ-REPORT-3027: Adversarial Flag Methodology Corrigendum

The repository shall provide an Exp 3027 methodology-corrigendum generator
that writes
`results/experiment_3027_adversarial_flag_methodology_corrigendum_v1.json`
using only checked-in upstream artifacts. The generator MUST read Exp 3013
SOTA GGUF preflight evidence, Exp 3014 repair taxonomy, Exp 3015 acceptance
controller, Exp 3016 acceptance-controlled repair rerun, Exp 3018 validator
frontier certificate, Exp 3024 matrix v17, and Exp 3025 capstone v283. It MUST
NOT run live LLM inference, verifier scoring, solver execution, synthesis,
board flashing, readback, historical experiment rewrites, the conductor, or
external publication tooling.

The corrigendum MUST collect and cite source fields for duration,
`model_specs`, inference substrate, transcript paths or hashes, seeds,
checksums, adversarial flags, SOTA headline readiness, and paper readiness. It
MUST apply a MARCH-style information-asymmetry rule: a source row may not grade
itself. Classifications MUST cite direct source-artifact fields when available,
and matrix/capstone fields may only supply aggregation context or row counts.

The corrigendum MUST classify all matrix/capstone flagged rows that affect
`.283` paper readiness into exactly one primary class from:
`true_methodology_blocker`, `aggregation_false_positive`, `missing_metadata`,
`unresolved_bound`, `hardware_blocked`, or `clean_but_not_headline`. It MUST
keep prior carry-forward aggregation rows visible without treating them as new
live-repair failures. It MUST separately expose real methodology blockers,
aggregation/substrate-schema false positives, missing metadata rows,
unresolved-bound rows, hardware-blocked rows, and clean-but-not-headline rows.

The Exp 3028 decision MUST be conservative. If Exp 3016 lacks any required
live transcript, live model specification, random seed, or model/transcript
checksum evidence needed to reconstruct clean repair evidence, then Exp 3027
MUST set `repair_rerun_required=true`; otherwise it MAY allow transcript
reconstruction. Exp 3027 MUST carry forward Exp 3013's explicit
`sota_headline_ready` field for downstream live-repair gates while still
reporting any metadata caveats.

Because Exp 3027 is aggregation-only, its terminal artifact MUST set
`inference_substrate` to an object declaring aggregation-only evidence and
MUST NOT include top-level `model_specs`, `target_model`, CUDA, GGUF, or
live-model fields. Source model details MAY appear only inside cited source
artifact summaries.

The terminal artifact MUST include `methodology_corrigendum_ready`,
`sota_headline_ready`, `repair_rerun_required`, `flagged_rows_reviewed`,
`true_methodology_blockers`, `aggregation_false_positive_rows`,
`missing_metadata_rows`, `unresolved_bound_rows`, `source_artifacts`,
`inference_substrate`, and `honest_verdict`. It MAY include row
classifications, repair-rerun decision details, hardware-blocked rows,
clean-but-not-headline rows, source checksums, no-new-execution booleans, and
measured `duration_s` as long as every value is derived from upstream JSON
artifacts or file presence/checksum checks.

#### SCENARIO-REPORT-3027: Corrigendum Gates Exp 3028 Without Live Inference

**Given** Exp 3013, Exp 3014, Exp 3015, Exp 3016, Exp 3018, Exp 3024, and Exp
3025 artifacts are present
**And** matrix v17 reports flagged, blocked, gated-skipped, and missing rows
**And** Exp 3016 has live transcript paths and model checksum evidence but no
required random seed
**When** the Exp 3027 corrigendum generator runs
**Then** it writes
`results/experiment_3027_adversarial_flag_methodology_corrigendum_v1.json`
with `methodology_corrigendum_ready=true`, `sota_headline_ready` copied from
Exp 3013, `repair_rerun_required=true`, `flagged_rows_reviewed` matching the
matrix/capstone flagged-row count, direct citations for every classification,
Exp 3016 in `missing_metadata_rows`, Exp 3018 in
`unresolved_bound_rows`, deterministic cached-replay duration flags separated
as aggregation/substrate false positives, hardware gate blockers preserved
outside repair evidence, aggregation-only `inference_substrate`, no top-level
live-model metadata, and an honest verdict that starts with `complete:`.

## Implementation Status (REQ-REPORT-3027)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-3027 | Implemented (`python/carnot/reporting/adversarial_flag_methodology_corrigendum_3027.py`) | Implemented (`tests/python/test_experiment_3027_adversarial_flag_methodology_corrigendum.py`) |

### REQ-REPORT-3029: Repair Promotion Boundary Audit

The repository shall provide an Exp 3029 aggregation-only repair promotion
boundary audit that writes
`results/experiment_3029_repair_promotion_boundary_audit_v2.json`. The audit
MUST read Exp 3027, Exp 3028, Exp 3016, matrix v17, and capstone v283 before
deciding whether the repair claim is promotable, bounded, blocked, or retired.
It MUST NOT run live LLM inference, verifier scoring, solver execution,
synthesis, board flashing, readback, historical experiment rewrites, the
conductor, or external publication tooling.

The audit MUST build a claim-boundary table. Each row MUST include the proposed
repair claim, required support fields, observed support fields, blockers, and
allowed wording. A repair claim may be promotable only when the evidence has
clean deltas, clean metadata, non-vacuous tautology gates, no false-accept
increase, no intent drift, no legacy smoke-only model use, and enough live or
reconstructed evidence. Partial evidence MUST be bounded in wording, and
unsupported headline language MUST be retired or blocked so failed reruns do not
recur as public claims.

Because Exp 3029 is aggregation-only, its top-level `inference_substrate` MUST
be an object declaring aggregation-only evidence and MUST NOT include top-level
`model_specs`, `target_model`, CUDA, GGUF, GPU inventory, headline-model, or
live-model fields. Source model details MAY appear only under
`cited_upstream_artifacts`.

The terminal artifact MUST include `repair_promotion_boundary_ready`,
`repair_claim_status`, `promotable_claims`, `bounded_claims`,
`retired_or_blocked_claims`, `cited_upstream_artifacts`,
`inference_substrate`, and `honest_verdict`. It MAY include the
claim-boundary table, required source errors, source checksums, no-new-execution
booleans, status-update booleans, and measured `duration_s` as long as every
value is derived from upstream JSON artifacts or file presence/checksum checks.

#### SCENARIO-REPORT-3029: Clean Repair Evidence Is Bounded Against Prior Matrix Flags

**Given** Exp 3027 reports the prior repair metadata gap and Exp 3028 reports
clean reconstructed repair evidence
**And** matrix v17 and capstone v283 still contain older flagged repair
promotion decisions from before Exp 3028
**When** the Exp 3029 repair promotion boundary audit runs
**Then** it writes
`results/experiment_3029_repair_promotion_boundary_audit_v2.json` with
`repair_promotion_boundary_ready=true`, `repair_claim_status="bounded"`, clean
artifact-level repair claims only in `bounded_claims`, unsupported headline
language in `retired_or_blocked_claims`, an explicit claim-boundary table,
source model details only under `cited_upstream_artifacts`, aggregation-only
top-level `inference_substrate`, no top-level live-model metadata, and an
honest verdict that starts with `complete:`.

## Implementation Status (REQ-REPORT-3029)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-3029 | Implemented (`python/carnot/reporting/repair_promotion_boundary_audit_3029.py`) | Implemented (`tests/python/test_experiment_3029_repair_promotion_boundary_audit.py`) |

### REQ-REPORT-3030: Validator Frontier Corrigendum

The repository shall provide an Exp 3030 validator-frontier corrigendum
generator that writes
`results/experiment_3030_validator_frontier_corrigendum_v2.json`. The
generator MUST read Exp 3017 validator-tree evidence, Exp 3018 BEAVER-style
frontier-certificate evidence, and Exp 3027 methodology-corrigendum evidence
before deciding which frontier regions are verified, irrelevant, unresolved,
fallback-only, or missing-authority. It MUST NOT run live LLM inference,
verifier scoring, solver execution, synthesis, board flashing, readback,
historical experiment rewrites, the conductor, or external publication tooling.

The corrigendum MUST build an inspectable frontier table. Each row MUST include
the source artifact path, source row identifier, authority type, classification,
bound status, and allowed claim wording. Cached deterministic candidate rows
MAY be counted as verified only when the Exp 3017 exact authority and Exp 3018
deterministic validator outcome are present and no live LLM or fallback
evidence was used. Non-authoritative semantic boundaries and rejected ambiguous
space MUST be separated as irrelevant/clipped regions. Rows with unresolved
probability bounds, nondeterministic validators, or LLM-only labels MUST remain
visible and MUST NOT be promoted as exact BEAVER probability bounds. Enumerator
fallback evidence MUST be reported as fallback-only and cannot be promoted as
exact authority. Missing provenance or absent authority MUST block promotion
instead of being folded into verified counts.

Because Exp 3030 is aggregation-only, its top-level `inference_substrate` MUST
be an object declaring aggregation-only evidence and MUST NOT include top-level
`model_specs`, `target_model`, CUDA, GGUF, GPU inventory, headline-model, or
live-model fields. Source model details, if any, MAY appear only under
`cited_upstream_artifacts`.

The terminal artifact MUST include `validator_frontier_corrigendum_ready`,
`verified_region_count`, `irrelevant_region_count`,
`unresolved_region_count`, `fallback_only_count`,
`missing_authority_count`, `frontier_rows`, `cited_upstream_artifacts`,
`inference_substrate`, and `honest_verdict`. It MAY include required source
errors, source checksums, no-new-execution booleans, status-update booleans,
and measured `duration_s` as long as every value is derived from upstream JSON
artifacts, JSONL manifests, or file presence/checksum checks.

#### SCENARIO-REPORT-3030: Validator Frontier Regions Stay Inspectable

**Given** Exp 3017, Exp 3018, and Exp 3027 artifacts are present
**And** Exp 3018 contains certified candidate rows, non-prefix rows,
unresolved source rejections, probability-bound placeholders, and separated
enumerator fallback provenance
**When** the Exp 3030 validator-frontier corrigendum generator runs
**Then** it writes
`results/experiment_3030_validator_frontier_corrigendum_v2.json` with
`validator_frontier_corrigendum_ready=true`, every frontier row classified,
verified exact-authority rows counted separately from irrelevant, unresolved,
fallback-only, and missing-authority regions, unresolved rows still present in
`frontier_rows`, fallback-only evidence prevented from promotion, source
artifact citations for Exp 3017, Exp 3018, and Exp 3027, aggregation-only
top-level `inference_substrate`, no top-level live-model metadata, and an
honest verdict that starts with `complete:`.

## Implementation Status (REQ-REPORT-3030)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-3030 | Implemented (`python/carnot/reporting/validator_frontier_corrigendum_3030.py`) | Implemented (`tests/python/test_experiment_3030_validator_frontier_corrigendum.py`) |

### REQ-REPORT-3038: Cross-Corpus Matrix V18 From .284 Artifacts

The repository shall provide an Exp 3038 cross-corpus matrix v18 generator
that writes `results/experiment_3038_cross_corpus_matrix_v18.json` using only
checked-in upstream artifact JSON files, the active `.284` roadmap, and the
Exp 3024 matrix v17 baseline. The generator MUST read every available `.284`
artifact from Exp 3026 through Exp 3037, represent Exp 3038 itself as the
current aggregation task, and represent Exp 3039 as missing until the capstone
artifact exists. It MUST NOT run live LLM inference, verifier scoring, solver
execution, synthesis, board flashing, readback, hardware smoke tests, the
conductor, external publication tooling, or historical experiment rewrites.

The generator MUST classify each of the 14 `.284` tasks as exactly one of
`clean`, `flagged`, `blocked`, `gated_skipped`, `projection_only`,
`pilot_only`, `missing`, or `retired`. Missing, gated, blocked, flagged,
projection-only, pilot-only, and retired evidence MUST remain visible instead
of being collapsed into successful rows. The Exp 3035 gate-check artifact MAY
be read from the actual gate-check path when the planned deliverable path is
absent, but the missing planned path MUST remain inspectable in the source
metadata. Exp 3036 MUST remain `gated_skipped` when the Exp 3035 shim gate did
not pass, even when the Exp 3036 deliverable is absent.

The matrix rows MUST include specialized columns for `repair_claim_status`,
`fr11_self_learning_promotable`, `gatemate_output_contract_ready`,
`host_visible_output_observed`, and `ssqa_gate_status`. The matrix MUST keep
repair evidence bounded when Exp 3029 reports `repair_claim_status="bounded"`,
MUST keep FR-11 promotion controller-only unless the held-out and
nonforgetting controls are clean, MUST keep GateMate output-contract and
host-visible smoke evidence distinct, and MUST keep SSQA as gate-skipped when
host-visible GateMate output is absent.

Because Exp 3038 is aggregation-only, the terminal artifact MUST expose
top-level live-model metadata only as absence: no top-level `model_specs`,
`target_model`, CUDA, GGUF, GPU inventory, headline-model, or live-model fields
are allowed. Source model and hardware details MAY appear only under
`cited_upstream_artifacts`. The top-level `inference_substrate` MUST be a
small aggregation dictionary that does not claim model execution.

The terminal artifact MUST include `matrix_v18_ready`, `rows_total`, `clean`,
`flagged`, `blocked`, `gated_skipped`, `projection_only`, `pilot_only`,
`missing`, `retired`, `matrix_rows`, `cited_upstream_artifacts`,
`inference_substrate`, and `honest_verdict`. It MAY include baseline v17
status summaries, source checksums, missing planned artifacts, no-new-execution
booleans, status-update booleans, recommended next actions, and measured
`duration_s` as long as every value is derived from upstream JSON artifacts,
roadmap task metadata, or file presence/checksum checks. When a conductor
prompt assigns ops reconciliation to a separate step, the generator MUST leave
`ops/status.md`, `ops/changelog.md`, and `_bmad/traceability.md` unchanged.

#### SCENARIO-REPORT-3038: V18 Represents All .284 Tasks Without Live Metadata Leakage

**Given** matrix v17 is present and reports `matrix_v17_ready=true`
**And** Exp 3026 through Exp 3037 artifacts are present, flagged, blocked,
gated-skipped, projection-only, pilot-only, retired, or honestly absent
**And** the Exp 3035 planned deliverable is absent but a conductor gate-check
artifact records the failed Exp 3034 output-contract gate
**When** the Exp 3038 matrix v18 generator runs
**Then** it writes `results/experiment_3038_cross_corpus_matrix_v18.json`
with `matrix_v18_ready=true`, `rows_total=14`, explicit counts for every
terminal status bucket, one `matrix_rows` entry for each `.284` task from Exp
3026 through Exp 3039, Exp 3036 classified as `gated_skipped` rather than
silently missing, Exp 3039 classified as `missing`, repair, FR-11, GateMate,
host-visible output, and SSQA specialized columns filled from upstream
evidence, source model/hardware details only under `cited_upstream_artifacts`,
no top-level live-model metadata, and an `honest_verdict` that states readiness
and counts.

## Implementation Status (REQ-REPORT-3038)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-3038 | Planned (`python/carnot/reporting/cross_corpus_matrix_v18_3038.py`) | Planned (`tests/python/test_experiment_3038_cross_corpus_matrix_v18.py`) |

### REQ-REPORT-3039: Milestone 2026.05.284 Capstone

The repository shall provide an Exp 3039 milestone capstone generator that
writes `results/experiment_3039_capstone_v284.json` using only checked-in
upstream JSON artifacts, the active `.284` roadmap, and matrix v18. The
generator MUST read `results/experiment_3038_cross_corpus_matrix_v18.json`
and every available `.284` artifact from Exp 3026 through Exp 3037 before
deciding paper readiness, repair promotion, FR-11 self-learning promotion,
GateMate status, SSQA status, and the natural `.285` milestone focus. It MUST
NOT run live LLM inference, verifier scoring, solver execution, synthesis,
board flashing, readback, hardware smoke tests, the conductor, external
publication tooling, or historical experiment rewrites.

The capstone MUST set `capstone_ready=true` only when matrix v18 exists,
reports `matrix_v18_ready=true`, and represents all 14 `.284` task rows from
Exp 3026 through Exp 3039. It MUST reconcile the matrix counts for `clean`,
`flagged`, `blocked`, `gated_skipped`, `projection_only`, `pilot_only`,
`missing`, and `retired` against the matrix rows, preserving non-clean rows
instead of converting them into milestone success.

The capstone MUST set `paper_ready=true` only when repair evidence is clean and
promotable, FR-11 self-learning is non-tautological and explicitly scoped, and
hardware blockers are either resolved or explicitly bounded without unsupported
speedup, sampler, thermodynamic, annealing, latency, energy, or board
performance claims. Bounded controller-only FR-11 evidence MAY be promoted only
as verifier-feedback controller utility; it MUST NOT be broadened into native
LLM weight learning or unconstrained autonomous self-improvement. GateMate and
SSQA statuses MUST remain blocked or gate-skipped when the output contract,
host-visible transcript, or bounded resource evidence is absent.

Because Exp 3039 is aggregation-only, the terminal artifact MUST expose no
top-level live-model metadata such as `model_specs`, `target_model`, CUDA,
GGUF, GPU inventory, headline-model, or live-model fields. Source model and
hardware details MAY appear only under cited upstream-artifact provenance. When
a conductor prompt assigns ops reconciliation to a separate step, the generator
MUST leave `ops/status.md`, `ops/changelog.md`, and `_bmad/traceability.md`
unchanged.

The terminal artifact MUST include `capstone_ready`, `paper_ready`,
`repair_claim_status`, `fr11_self_learning_status`, `gatemate_status`,
`ssqa_status`, `matrix_v18_summary`, `blockers_remaining`,
`next_milestone_focus`, `recommended_next_actions`, `inference_substrate`, and
`honest_verdict`. It MAY include source checksums, cited upstream artifacts,
paper-readiness checks, what-the-milestone-proved summaries, no-new-execution
booleans, status-update booleans, and measured `duration_s` as long as every
value is derived from upstream JSON artifacts, roadmap metadata, or file
presence/checksum checks.

#### SCENARIO-REPORT-3039: Capstone Closes .284 Without Paper Overclaim

**Given** matrix v18 is present and reports `matrix_v18_ready=true`
**And** Exp 3026 through Exp 3037 artifacts are present, flagged, blocked,
gated-skipped, projection-only, pilot-only, retired, or honestly absent
**And** matrix v18 reports bounded repair evidence, controller-only FR-11
self-learning, a missing GateMate output contract, missing host-visible output,
and gate-skipped SSQA
**When** the Exp 3039 capstone generator runs
**Then** it writes `results/experiment_3039_capstone_v284.json` with
`capstone_ready=true`, `paper_ready=false`, repair carried forward as bounded
rather than headline-promotable, FR-11 carried forward only as controller-only
non-tautological self-learning, GateMate bounded as blocked on the missing
pinout/output contract, SSQA bounded as gate-skipped without performance
claims, reconciled matrix v18 counts, visible blockers, three to five concrete
`.285` actions, source model/hardware details only under
`cited_upstream_artifacts`, no top-level live-model metadata, and an
`honest_verdict` that starts with `complete:`.

## Implementation Status (REQ-REPORT-3039)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-3039 | Implemented (`python/carnot/reporting/capstone_v284_3039.py`) | Implemented (`tests/python/test_experiment_3039_capstone_v284.py`) |

### REQ-REPORT-3040: Archive .284 And Confirm .285 Roadmap Handoff

The repository shall provide an Exp 3040 archive/handoff generator that writes
`results/experiment_3040_archive_v284_activate_v285.json` using only checked-in
aggregation artifacts, roadmap YAML/Markdown files, and file presence/checksum
metadata. The generator MUST read
`results/experiment_3039_capstone_v284.json` as the authority artifact for the
completed `.284` milestone and MUST set `prior_capstone_ready` directly from
that artifact's `capstone_ready` field. It MUST set `prior_paper_ready`
directly from the authority artifact's `paper_ready` field rather than
inferring paper readiness from capstone completion, matrix cleanliness, or
roadmap status.

The generator MUST read `results/experiment_3038_cross_corpus_matrix_v18.json`
and summarize `.284` `clean`, `flagged`, `blocked`, `gated_skipped`,
`missing`, and bounded carry-forward status. The bounded carry-forward status
MUST keep the unresolved claims visible as exactly these three blockers when
they are present in the capstone: `repair_claim_status=bounded`,
`fr11_self_learning_status=controller_only_promotable`, and
`gatemate_status=blocked_pinout_missing_bounded`.

The generator MUST verify the `.285` roadmap handoff without modifying
`research-roadmap.yaml` or `scripts/research_conductor.py`. It MUST prefer
`research-roadmap-next.yaml` when that staged file exists and otherwise MAY use
the already-activated `research-roadmap.yaml` as a read-only fallback, provided
the artifact records that the requested staged roadmap is absent. Whichever
roadmap source is used MUST target milestone `2026.05.285`, point
`milestone_doc` to `openspec/change-proposals/research-roadmap-vNEXT.md`, and
contain at least one task. The generator MUST also confirm that the vNEXT
planning Markdown file exists. It MUST NOT activate the roadmap itself, run the
conductor, run live LLM inference, verifier scoring, solver execution,
synthesis, board flashing, readback, hardware smoke tests, external
publication tooling, or historical experiment rewrites.

The terminal artifact MUST include `archive_v284_activate_v285_ready`,
`prior_capstone_ready`, `prior_paper_ready`, `carry_forward_blockers`,
`next_milestone`, `source_artifacts`, `inference_substrate`, and
`honest_verdict`. It MAY include `.284` status summaries, roadmap handoff
metadata, source checksums, no-new-execution booleans, protected-file state,
and measured `duration_s` as long as every value is derived from upstream JSON
artifacts, roadmap files, or file presence/checksum checks. When a conductor
prompt assigns ops reconciliation to a separate step, the generator MUST leave
`ops/status.md`, `ops/changelog.md`, and `_bmad/traceability.md` unchanged.

#### SCENARIO-REPORT-3040: Archive .284 And Audit .285 Activation Readiness

**Given** `results/experiment_3039_capstone_v284.json` exists and reports
`capstone_ready=true`
**And** `results/experiment_3038_cross_corpus_matrix_v18.json` reports matrix
v18 counts for `clean`, `flagged`, `blocked`, `gated_skipped`, and `missing`
**And** the `.285` roadmap source, either staged or already active, targets
milestone `2026.05.285` and points to
`openspec/change-proposals/research-roadmap-vNEXT.md`
**When** the Exp 3040 archive/handoff generator runs
**Then** it writes `results/experiment_3040_archive_v284_activate_v285.json`
with `archive_v284_activate_v285_ready=true`, `prior_capstone_ready=true`,
`prior_paper_ready=false`, `.284` status counts preserved, the three
carry-forward blockers visible, `next_milestone="2026.05.285"`, concrete
source artifact provenance, aggregation-only inference substrate metadata, no
roadmap activation performed by this task, and an `honest_verdict` that starts
with `complete:`.

## Implementation Status (REQ-REPORT-3040)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-3040 | Implemented (`python/carnot/reporting/archive_v284_activate_v285_3040.py`) | Implemented (`tests/python/test_experiment_3040_archive_v284_activate_v285.py`) |

### REQ-REPORT-3054: Archive .285 And Confirm .286 Roadmap Handoff

The repository shall provide an Exp 3054 archive/handoff generator that writes
`results/experiment_3054_archive_v285_activate_v286.json` using only checked-in
aggregation artifacts, roadmap YAML/Markdown files, and file presence/checksum
metadata. The generator MUST read
`results/experiment_3053_capstone_v285.json` as the authority artifact for the
completed `.285` milestone and MUST set `prior_capstone_ready` directly from
that artifact's `capstone_ready` field. It MUST set `prior_paper_ready`
directly from the authority artifact's `paper_ready` field rather than
inferring paper readiness from capstone completion, matrix cleanliness, roadmap
status, or publication recommendations.

The generator MUST read `results/experiment_3052_cross_corpus_matrix_v19.json`
and summarize `.285` `paper_ready`, `repair_claim_status`,
`fr11_self_learning_status`, `gatemate_status`, and `ssqa_status` from the
matrix/capstone chain. The carry-forward blocker list MUST keep these unresolved
claim boundaries visible: repair remains bounded, GateMate remains blocked on
the output contract, SSQA remains blocked or gate-skipped until host-visible
smoke exists, and model-weight self-learning remains out of scope when FR-11
evidence is controller-only.

The generator MUST verify the `.286` roadmap handoff without modifying
`research-roadmap.yaml` or `scripts/research_conductor.py`. It MUST prefer
`research-roadmap-next.yaml` when that staged file exists and otherwise MAY use
the already-activated `research-roadmap.yaml` as a read-only fallback, provided
the artifact records that the requested staged roadmap is absent. Whichever
roadmap source is used MUST target milestone `2026.05.286`, point
`milestone_doc` to `openspec/change-proposals/research-roadmap-vNEXT.md`, and
contain at least one task. The generator MUST also confirm that the vNEXT
planning Markdown file exists. It MUST NOT activate the roadmap itself, run the
conductor, run live LLM inference, verifier scoring, solver execution,
synthesis, board flashing, readback, hardware smoke tests, external
publication tooling, or historical experiment rewrites.

The terminal artifact MUST include `archive_v285_activate_v286_ready`,
`prior_capstone_ready`, `prior_paper_ready`, `carry_forward_blockers`,
`next_milestone`, `source_artifacts`, `inference_substrate`, and
`honest_verdict`. It MAY include `.285` status summaries, roadmap handoff
metadata, source checksums, no-new-execution booleans, protected-file state,
and measured `duration_s` as long as every value is derived from upstream JSON
artifacts, roadmap files, or file presence/checksum checks. When a conductor
prompt assigns ops reconciliation to a separate step, the generator MUST leave
`ops/status.md`, `ops/changelog.md`, and `_bmad/traceability.md` unchanged.

#### SCENARIO-REPORT-3054: Archive .285 And Audit .286 Activation Readiness

**Given** `results/experiment_3053_capstone_v285.json` exists and reports
`capstone_ready=true`
**And** `results/experiment_3052_cross_corpus_matrix_v19.json` reports the
`.285` paper, repair, FR-11, GateMate, and SSQA statuses
**And** the `.286` roadmap source, either staged or already active, targets
milestone `2026.05.286` and points to
`openspec/change-proposals/research-roadmap-vNEXT.md`
**When** the Exp 3054 archive/handoff generator runs
**Then** it writes `results/experiment_3054_archive_v285_activate_v286.json`
with `archive_v285_activate_v286_ready=true`, `prior_capstone_ready=true`,
`prior_paper_ready=false`, the `.285` matrix/capstone statuses preserved, the
four carry-forward blocker categories visible, `next_milestone="2026.05.286"`,
concrete source artifact provenance, aggregation-only inference substrate
metadata, no roadmap activation performed by this task, and an
`honest_verdict` that starts with `complete:`.

## Implementation Status (REQ-REPORT-3054)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-3054 | Implemented (`python/carnot/reporting/archive_v285_activate_v286_3054.py`) | Implemented (`tests/python/test_experiment_3054_archive_v285_activate_v286.py`) |

### REQ-REPORT-3067: Archive .286 And Confirm .287 Roadmap Handoff

The repository shall provide an Exp 3067 archive/handoff generator that writes
`results/experiment_3067_archive_v286_activate_v287.json` using only checked-in
aggregation artifacts, roadmap YAML/Markdown files, and file presence/checksum
metadata. The generator MUST read
`results/experiment_3066_capstone_v286.json` as the authority artifact for the
completed `.286` milestone and MUST set `prior_capstone_ready` directly from
that artifact's `capstone_ready` field. It MUST set `prior_paper_ready`
directly from the authority artifact's `paper_ready` field rather than
inferring paper readiness from capstone completion, matrix cleanliness, roadmap
status, or publication recommendations.

The generator MUST read `results/experiment_3065_cross_corpus_matrix_v20.json`
and summarize `.286` `paper_ready`, `repair_claim_status`,
`solver_grounding_status`, `fr11_self_learning_status`, `gatemate_status`,
`ssqa_status`, and `publication_blocker_count` from the matrix/capstone chain.
If `publication_blocker_count` is absent as a top-level JSON field but present
in the authority artifact's terminal verdict, the generator MAY recover the
integer from that verdict and MUST mark the recovery source explicitly. The
carry-forward blocker list MUST keep these unresolved claim boundaries visible:
negative verifier gain, repair gated skipped, FR-11 controller-only flagged,
KAN/PWA not promoted, GateMate operator actions missing, and SSQA host-visible
smoke missing.

The generator MUST verify the `.287` roadmap handoff without modifying
`research-roadmap.yaml` or `scripts/research_conductor.py`. It MUST prefer
`research-roadmap-next.yaml` when that staged file exists and otherwise MAY use
the already-activated `research-roadmap.yaml` as a read-only fallback, provided
the artifact records that the requested staged roadmap is absent. Whichever
roadmap source is used MUST target milestone `2026.05.287`, point
`milestone_doc` to `openspec/change-proposals/research-roadmap-vNEXT.md`, and
contain at least one task. The generator MUST also confirm that the vNEXT
planning Markdown file exists. It MUST NOT activate the roadmap itself, run the
conductor, run live LLM inference, verifier scoring, solver execution,
synthesis, board flashing, readback, hardware smoke tests, external
publication tooling, or historical experiment rewrites.

The terminal artifact MUST include `archive_v286_activate_v287_ready`,
`prior_capstone_ready`, `prior_paper_ready`, `carry_forward_blockers`,
`next_milestone`, `source_artifacts`, `inference_substrate`, and
`honest_verdict`. It MAY include `.286` status summaries, roadmap handoff
metadata, source checksums, no-new-execution booleans, protected-file state,
and measured `duration_s` as long as every value is derived from upstream JSON
artifacts, roadmap files, or file presence/checksum checks. When a conductor
prompt assigns ops reconciliation to a separate step, the generator MUST leave
`ops/status.md`, `ops/changelog.md`, and `_bmad/traceability.md` unchanged.

#### SCENARIO-REPORT-3067: Archive .286 And Audit .287 Activation Readiness

**Given** `results/experiment_3066_capstone_v286.json` exists and reports
`capstone_ready=true`
**And** `results/experiment_3065_cross_corpus_matrix_v20.json` reports the
`.286` paper, repair, solver-grounding, FR-11, GateMate, SSQA, and blocker
statuses
**And** the `.287` roadmap source, either staged or already active, targets
milestone `2026.05.287` and points to
`openspec/change-proposals/research-roadmap-vNEXT.md`
**When** the Exp 3067 archive/handoff generator runs
**Then** it writes `results/experiment_3067_archive_v286_activate_v287.json`
with `archive_v286_activate_v287_ready=true`, `prior_capstone_ready=true`,
`prior_paper_ready=false`, the `.286` matrix/capstone statuses preserved, the
six carry-forward blocker categories visible, `next_milestone="2026.05.287"`,
concrete source artifact provenance, aggregation-only inference substrate
metadata, no roadmap activation performed by this task, and an
`honest_verdict` that starts with `complete:`.

## Implementation Status (REQ-REPORT-3067)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-3067 | Implemented (`python/carnot/reporting/archive_v286_activate_v287_3067.py`) | Implemented (`tests/python/test_experiment_3067_archive_v286_activate_v287.py`) |

### REQ-REPORT-3055: Repair Headline Retirement And Blocker Ledger

The repository shall provide an Exp 3055 repair headline retirement and
blocker ledger generator that writes
`results/experiment_3055_repair_headline_retirement_and_blocker_ledger_v1.json`
using only checked-in Exp 3041, Exp 3042, matrix v19, capstone v285, and the
exclusion manifest. The generator MUST NOT run live LLM inference, verifier
scoring, solver execution, synthesis, board flashing, readback, hardware smoke
tests, the conductor, external publication tooling, or historical artifact
rewrites.

The ledger MUST extract every repair-related blocker and retired or bounded
repair row visible in the four source artifacts. Unsupported repair headline
wording that lacks support under matrix v19 and capstone v285 MUST be marked
`retired_for_headline_use` in a machine-readable retired-claim list. Bounded
repair evidence MUST remain in a separate bounded-claim list and MUST NOT be
promoted by wording. Future repair reruns MUST be gated by explicit evidence:
deterministic fingerprint, seed, duration sanity, de-tautology metrics,
verifier gain, and exact checker authority. Manifest changes, if required by
CLAUDE.md failed-rerun and exclusion-manifest discipline, MUST be minimal and
cited by entry id and path.

The terminal artifact MUST include `repair_headline_retirement_ready`,
`retired_repair_claims`, `still_bounded_repair_claims`,
`rerun_prerequisites`, `manifest_updates`, `source_artifacts`,
`inference_substrate`, and `honest_verdict`. It MAY include extracted blocker
rows, source checksums, no-new-execution booleans, status-update booleans, and
measured `duration_s` as long as every value is derived from checked-in
artifacts, manifest fields, or file presence/checksum checks. The
`repair_headline_retirement_ready` field MUST be true only when matrix v20 can
consume the retired and bounded repair decisions by claim id, row id, status,
source artifact, source field, and required evidence.

#### SCENARIO-REPORT-3055: Unsupported Repair Headlines Retire Before Rerun

**Given** Exp 3041, Exp 3042, matrix v19, and capstone v285 are present
**And** matrix v19 and capstone v285 preserve repair as bounded with two
retired repair headline rows
**And** CLAUDE.md discipline requires retired headline-rerun scope to be
traceable in the exclusion manifest
**When** the Exp 3055 ledger generator runs
**Then** it writes
`results/experiment_3055_repair_headline_retirement_and_blocker_ledger_v1.json`
with `repair_headline_retirement_ready=true`, retired repair claims marked
`retired_for_headline_use`, bounded repair claims still separate from retired
claims, rerun prerequisites covering deterministic fingerprint, seed, duration
sanity, de-tautology metrics, verifier gain, and exact checker authority, the
manifest update cited by id and source file, source artifact provenance for all
four source artifacts, aggregation-only inference substrate metadata, no live
model or hardware execution by the ledger task, and an `honest_verdict` that
starts with `complete:`.

## Implementation Status (REQ-REPORT-3055)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-3055 | Implemented (`python/carnot/reporting/repair_headline_blocker_ledger_3055.py`) | Implemented (`tests/python/test_experiment_3055_repair_headline_blocker_ledger.py`) |

### REQ-REPORT-3056: Repair De-Tautology Protocol

The repository shall provide an Exp 3056 repair de-tautology protocol
generator that writes
`results/experiment_3056_repair_de_tautology_protocol_v1.json` using only
checked-in Exp 3028, Exp 3042, Exp 3043, Exp 3055, and research-reference
metadata. The generator MUST NOT run live LLM inference, model loading,
verifier scoring, solver execution, synthesis, board flashing, readback,
hardware smoke tests, the conductor, or historical artifact rewrites.

The protocol MUST extract the exact prior repair blocker fields for Exp 3028
and Exp 3042: tautological pass@1/pass@k evidence, implausible exact-zero
repair side effects, too-short live-run duration, missing seed metadata, and
unresolved methodology or bounded-promotion blockers. Each extracted blocker
MUST cite source artifact, source field, blocker kind or classification, and
the observed field names that a future run must clear.

The protocol MUST define mechanically checkable acceptance checks for all
future local SOTA repair runs before live rerun execution: separate pass@1 and
pass@k derivation, non-vacuous per-task outcomes, wall-clock sanity,
top-level and per-transcript seed/logging, model_specs identity and checksum
evidence, transcript fingerprint linkage, and named checker authority. The
intent-preservation requirement MAY cite Approximately Aligned Decoding
(AprAD) as design inspiration for preserving draft intent while applying hard
verifier gates, but it MUST NOT claim AprAD was implemented unless a later
artifact records an actual local implementation.

The terminal artifact MUST include
`repair_de_tautology_protocol_ready`, `blocked_prior_fields`,
`required_live_run_fields`, `intent_preservation_checks`,
`duration_sanity_rule`, `fingerprint_requirements`,
`promotion_disqualifiers`, `inference_substrate`, and `honest_verdict`. It MAY
include an Exp 3059 matrix-v20 field contract, source artifact provenance,
source checksums, blocked reasons, no-new-execution booleans, and measured
`duration_s` as long as every value is derived from checked-in artifacts or
file presence/checksum checks. `repair_de_tautology_protocol_ready` MUST be
true only when Exp 3059 can consume the protocol without live inference, using
machine-readable fields that declare every live-run JSON field required for
matrix v20 promotion.

#### SCENARIO-REPORT-3056: Future Repair Rerun Is Gated By Predeclared Protocol

**Given** Exp 3028, Exp 3042, Exp 3043, Exp 3055, and research-reference
metadata exist
**And** Exp 3028 and Exp 3042 preserve repair blockers for tautology,
implausible perfect deltas, duration sanity, missing seed metadata, and
unresolved methodology
**When** the Exp 3056 repair de-tautology protocol generator runs
**Then** it writes
`results/experiment_3056_repair_de_tautology_protocol_v1.json` with
`repair_de_tautology_protocol_ready=true`, every prior blocker mapped to an
explicit future-run check, Exp 3059 live-run JSON fields listed for matrix v20
promotion eligibility, AprAD-inspired intent-preservation checks that do not
claim an AprAD implementation, transcript fingerprint requirements, promotion
disqualifiers, aggregation-only inference substrate metadata, no live model or
hardware execution by the protocol task, and an `honest_verdict` that starts
with `complete:`.

## Implementation Status (REQ-REPORT-3056)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-3056 | Implemented (`python/carnot/reporting/repair_de_tautology_protocol_3056.py`) | Implemented (`tests/python/test_experiment_3056_repair_de_tautology_protocol.py`) |

### REQ-REPORT-3041: Matrix/Capstone Adversarial Flag Hygiene

The repository shall provide an Exp 3041 flag-hygiene generator that writes
`results/experiment_3041_matrix_capstone_adversarial_flag_hygiene_v1.json`
using only checked-in Exp 3027, Exp 3028, Exp 3029, Exp 3038, and Exp 3039
artifacts. The generator MUST NOT run live LLM inference, verifier scoring,
solver execution, synthesis, board flashing, readback, hardware smoke tests,
the conductor, or historical artifact rewrites.

The generator MUST collect every adversarial flag, methodology flag, missing
row, blocked row, and gate-skipped row visible in the .284 repair, matrix, and
capstone artifacts. Every emitted classification row MUST cite the exact source
artifact path and source field that supports the classification. Aggregation
false positives MAY be emitted only when the cited source artifact explicitly
declares aggregation-only semantics and the flag came from treating that
aggregation artifact as a compute-bound live model run. Missing metadata and
unresolved verifier bounds MUST remain blockers; they MUST NOT be converted to
clean rows. Hardware-blocked and gate-skipped rows MUST remain bounded and MUST
NOT become performance claims.

The terminal artifact MUST include `flag_hygiene_ready`, `rows_reviewed`,
`true_blocker_rows`, `aggregation_false_positive_rows`,
`missing_metadata_rows`, `unresolved_bound_rows`, `hardware_blocked_rows`,
`gate_skipped_rows`, `source_artifacts`, `inference_substrate`, and
`honest_verdict`. It MAY include grouped summaries, downstream consumer
metadata, source checksums, no-new-execution booleans, status-update booleans,
and measured `duration_s` as long as every value is derived from upstream JSON
artifacts or file presence/checksum checks. `flag_hygiene_ready` MUST be true
only when downstream Exp 3042 and Exp 3043 can mechanically consume the
classification lists by row id, classification, source artifact, source field,
and blocking status. When a conductor prompt assigns ops reconciliation to a
separate step, the generator MUST leave `ops/status.md`, `ops/changelog.md`,
and `_bmad/traceability.md` unchanged.

#### SCENARIO-REPORT-3041: Flag Hygiene Separates Aggregation False Positives From Blockers

**Given** Exp 3027, Exp 3028, Exp 3029, Exp 3038, and Exp 3039 artifacts exist
and the matrix/capstone report flagged, blocked, missing, bounded, and
gate-skipped .284 rows
**When** the Exp 3041 flag-hygiene generator runs
**Then** it writes
`results/experiment_3041_matrix_capstone_adversarial_flag_hygiene_v1.json`
with aggregation-only duration/methodology flags classified separately from
true blockers, missing metadata, unresolved bounds, hardware blocks, and
gate-skipped rows; preserves the capstone `paper_ready=false` blockers;
records source artifact and field citations for every classification row;
declares aggregation-only inference substrate with no live model or hardware
execution; sets `flag_hygiene_ready=true` only when the rows are mechanically
consumable by Exp 3042 and Exp 3043; and emits an `honest_verdict` starting
with `complete:`.

## Implementation Status (REQ-REPORT-3041)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-3041 | Implemented (`python/carnot/reporting/matrix_capstone_adversarial_flag_hygiene_3041.py`) | Implemented (`tests/python/test_experiment_3041_matrix_capstone_adversarial_flag_hygiene.py`) |

### REQ-REPORT-3042: Repair Promotion Reconciliation For Matrix V19

The repository shall provide an Exp 3042 repair-promotion reconciliation
generator that writes
`results/experiment_3042_repair_promotion_reconciliation_v3.json` using only
checked-in Exp 3028, Exp 3029, Exp 3038, Exp 3039, and Exp 3041 artifacts. The
generator MUST NOT run live LLM inference, verifier scoring, solver execution,
synthesis, board flashing, readback, hardware smoke tests, the conductor, or
historical artifact rewrites.

The generator MUST verify that Exp 3028 contains model specifications,
transcript counts, repair deltas, checker fields, and clean
acceptance-controller evidence before considering any repair row candidate for
matrix v19. The generator MUST remove only aggregation false positives that Exp
3041 explicitly classified as `aggregation_false_positive`. Missing metadata,
unresolved bounds, intent drift, false accepts, syntax/schema regressions,
source gaps, and non-aggregation adversarial flags MUST remain visible blockers
and MUST NOT be converted into clean repair evidence.

The generator MUST emit exactly one repair decision for matrix v19:
`clean_candidate`, `bounded`, `blocked`, or `retired`. It MAY classify the row
as `clean_candidate` only when Exp 3028's clean-evidence gates pass and no
repair-relevant blockers remain after the Exp 3041 false-positive removal.
Candidate repair promotion MUST remain separate from final capstone or paper
promotion: a clean candidate decision is not a paper-ready decision, and a
bounded decision MUST keep the residual blocker citations attached.

The terminal artifact MUST include `repair_reconciliation_ready`,
`repair_promotion_candidate`, `repair_claim_status`,
`accepted_source_artifacts`, `remaining_blockers`,
`aggregation_false_positives_removed`, `repair_delta_summary`,
`inference_substrate`, and `honest_verdict`. It MAY include source checksums,
Exp 3028 evidence checks, prior matrix/capstone status, no-new-execution
booleans, status-update booleans, and measured `duration_s` as long as every
value is derived from upstream JSON artifacts or file presence/checksum checks.
When a conductor prompt assigns ops reconciliation to a separate step, the
generator MUST leave `ops/status.md`, `ops/changelog.md`, and
`_bmad/traceability.md` unchanged.

#### SCENARIO-REPORT-3042: Repair Row Remains Bounded After False-Positive Cleanup

**Given** Exp 3028, Exp 3029, Exp 3038, Exp 3039, and Exp 3041 artifacts exist
**And** Exp 3028 reports clean acceptance-controlled repair deltas but still
has non-aggregation repair flags or missing metadata
**And** Exp 3041 reports aggregation false positives for aggregation-only
matrix/capstone artifacts
**When** the Exp 3042 repair-promotion reconciliation generator runs
**Then** it writes
`results/experiment_3042_repair_promotion_reconciliation_v3.json` with
`repair_reconciliation_ready=true`, only Exp 3041-confirmed aggregation false
positives in `aggregation_false_positives_removed`, the remaining repair
blockers still cited by exact source artifact and source field,
`repair_claim_status="bounded"`, `repair_promotion_candidate=false`, Exp 3028
deltas preserved in `repair_delta_summary`, aggregation-only inference
substrate metadata, no live model or hardware execution, and an
`honest_verdict` that starts with `complete:`.

## Implementation Status (REQ-REPORT-3042)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-3042 | Planned (`python/carnot/reporting/repair_promotion_reconciliation_3042.py`) | Planned (`tests/python/test_experiment_3042_repair_promotion_reconciliation.py`) |

### REQ-REPORT-3052: Cross-Corpus Matrix V19 From .284 And .285 Artifacts

The repository shall provide an Exp 3052 cross-corpus matrix v19 generator
that writes `results/experiment_3052_cross_corpus_matrix_v19.json` using only
checked-in `.284` and `.285` artifact JSON files. The generator MUST read the
available matrix v18, capstone v284, flag-hygiene, repair-reconciliation,
verified-speculation fingerprint, FR-11 solver-feedback, KAN locality,
GateMate output-contract, GateMate host-visible smoke, and SSQA gate artifacts.
It MUST NOT run live LLM inference, verifier scoring, solver execution,
synthesis, board flashing, readback, hardware smoke tests, the conductor, or
historical artifact rewrites.

The generator MUST classify every emitted matrix claim row as exactly one of
`clean`, `flagged`, `bounded`, `blocked`, `gated_skipped`, `projection_only`,
`missing`, or `retired`. Every row MUST record its `source_artifact`, `status`,
`evidence_class`, and `blocker_class`, and missing optional artifacts MUST be
represented as missing rows rather than silently ignored. Repair may be
classified as clean only when Exp 3042 reports
`repair_promotion_candidate=true` and no required blocker remains. FR-11
self-learning may be promoted only within Exp 3046/3047's controller-side
scope and MUST NOT become a model-weight-learning claim. GateMate and SSQA may
be promoted only when a host-visible transcript and SSQA gate artifact exist;
otherwise their rows MUST remain blocked, gated-skipped, or missing.

The terminal artifact MUST include `matrix_v19_ready`, `rows_total`,
`clean_count`, `flagged_count`, `bounded_count`, `blocked_count`,
`gated_skipped_count`, `projection_only_count`, `missing_count`,
`retired_count`, `repair_claim_status`, `fr11_self_learning_status`,
`gatemate_status`, `ssqa_status`, `rows`, `source_artifacts`,
`inference_substrate`, and `honest_verdict`. `matrix_v19_ready` MUST be true
when all emitted rows have been classified, even when paper readiness remains
false because bounded, blocked, gated-skipped, missing, flagged, projection-only,
or retired rows remain visible. When a conductor prompt assigns ops
reconciliation to a separate step, the generator MUST leave `ops/status.md`,
`ops/changelog.md`, and `_bmad/traceability.md` unchanged.

#### SCENARIO-REPORT-3052: V19 Aggregates .284 And .285 Rows Without Over-Promotion

**Given** matrix v18, capstone v284, flag-hygiene, repair-reconciliation,
verified-speculation, FR-11, KAN-locality, GateMate output-contract, and SSQA
gate artifacts are present or honestly absent
**And** Exp 3042 reports a bounded repair decision with remaining blockers
**And** Exp 3046/3047 report controller-side FR-11 evidence without model-weight
training or mutation
**And** GateMate host-visible smoke is missing and the SSQA gate artifact
reports a structured upstream gate failure
**When** the Exp 3052 matrix v19 generator runs
**Then** it writes `results/experiment_3052_cross_corpus_matrix_v19.json` with
`matrix_v19_ready=true`, explicit counts for clean, flagged, bounded, blocked,
gated-skipped, projection-only, missing, and retired rows, repair preserved as
bounded, FR-11 scoped to controller-side evidence only, GateMate blocked or
missing until host-visible transcript evidence exists, SSQA gate-skipped until
the GateMate smoke gate passes, source provenance on every row, aggregation-only
inference substrate metadata, no live model or hardware execution by the matrix
task, and an `honest_verdict` that starts with `complete:`.

## Implementation Status (REQ-REPORT-3052)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-3052 | Implemented (`python/carnot/reporting/cross_corpus_matrix_v19_3052.py`) | Implemented (`tests/python/test_experiment_3052_cross_corpus_matrix_v19.py`) |

### REQ-REPORT-3053: Milestone 2026.05.285 Capstone From Matrix V19

The repository shall provide an Exp 3053 milestone capstone generator that
writes `results/experiment_3053_capstone_v285.json` using only checked-in
matrix v19 and source artifact JSON files. The generator MUST read
`results/experiment_3052_cross_corpus_matrix_v19.json` and the source
artifacts cited by that matrix before deciding capstone readiness, publication
readiness, repair claim status, FR-11 self-learning status, GateMate status,
SSQA status, promoted claims, bounded claims, blocked claims, gated-skipped
claims, and the exact next-milestone recommendation. It MUST NOT run live LLM
inference, verifier scoring, solver execution, synthesis, board flashing,
readback, hardware smoke tests, the conductor, external publication tooling, or
historical artifact rewrites.

The capstone MUST set `capstone_ready=true` only when matrix v19 exists,
reports `matrix_v19_ready=true`, all emitted matrix rows are classified, row
counts reconcile with their statuses, and the required matrix source artifacts
are readable JSON objects. It MUST set `paper_ready=true` only when repair
evidence is clean and promotable, FR-11 evidence stays within PRD-aligned claim
boundaries, GateMate has host-visible transcript evidence, SSQA has a consumed
GateMate smoke gate, and matrix v19 contains no non-clean publication blockers.
Controller-only FR-11 evidence MAY be carried as bounded self-learning
evidence, but it MUST NOT be broadened into model-weight learning or an
unconstrained autonomous self-improvement claim.

The terminal artifact MUST include `capstone_ready`, `paper_ready`,
`repair_claim_status`, `fr11_self_learning_status`, `gatemate_status`,
`ssqa_status`, `matrix_v19_summary`, `promoted_claims`, `bounded_claims`,
`blocked_claims`, `gated_skipped_claims`, `next_milestone_recommendation`,
`source_artifacts`, `inference_substrate`, and `honest_verdict`. It MAY include
paper-readiness checks, source checksums, missing source artifacts,
no-new-execution booleans, ops-reconciliation booleans, and measured
`duration_s` as long as every value is derived from checked-in artifacts or
file presence/checksum checks. When a conductor prompt assigns ops
reconciliation to a separate step, the generator MUST leave `ops/status.md`,
`ops/changelog.md`, and `_bmad/traceability.md` unchanged.

#### SCENARIO-REPORT-3053: Capstone Closes .285 Without Paper Overclaim

**Given** matrix v19 is present and reports `matrix_v19_ready=true`
**And** matrix v19 contains clean, bounded, blocked, gated-skipped, missing,
projection-only, flagged, and retired rows with source provenance
**And** repair is bounded rather than headline-promotable
**And** FR-11 evidence is controller-side only without model-weight training or
mutation
**And** GateMate lacks a host-visible transcript while SSQA reports a
structured upstream gate skip
**When** the Exp 3053 capstone generator runs
**Then** it writes `results/experiment_3053_capstone_v285.json` with
`capstone_ready=true`, `paper_ready=false`, matrix v19 counts reconciled,
promoted/bounded/blocked/gated-skipped claims enumerated separately, repair
carried as bounded, FR-11 carried only as controller-side evidence, GateMate
blocked until host-visible transcript evidence exists, SSQA gate-skipped until
the GateMate smoke gate passes, an exact next-milestone recommendation naming
which blockers should be retired, gated, or rerun, aggregation-only inference
substrate metadata, no live model or hardware execution by the capstone task,
and an `honest_verdict` that starts with `complete:`.

## Implementation Status (REQ-REPORT-3053)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-3053 | Implemented (`python/carnot/reporting/capstone_v285_3053.py`) | Implemented (`tests/python/test_experiment_3053_capstone_v285.py`) |

### REQ-REPORT-3065: Cross-Corpus Matrix V20 From .285 And .286 Artifacts

The repository shall provide an Exp 3065 cross-corpus matrix v20 generator
that writes `results/experiment_3065_cross_corpus_matrix_v20.json` using only
checked-in `.285` and `.286` artifact JSON files. The generator MUST read
matrix v19, capstone v285, and every available milestone `.286` artifact from
Exp 3054 through Exp 3064 before classifying rows. The generator MUST NOT run
live repair, live LLM inference, verifier scoring, solver execution, synthesis,
board flashing, readback, hardware smoke tests, the conductor, or historical
artifact rewrites. Missing optional artifact paths requested by the conductor
prompt, including the `experiment_3059_*_v1.json` alias when absent, MUST be
recorded as missing rows and missing source-artifact records rather than
silently ignored.

The v20 artifact MUST classify rows into exactly these row-class lists:
`clean_rows`, `flagged_rows`, `bounded_rows`, `blocked_rows`,
`gated_skipped_rows`, `projection_only_rows`, `missing_rows`, and
`retired_rows`. Every row in those lists MUST be machine-readable and include
at least `row_id`, `status`, `source_artifact`, `source_field`,
`evidence_class`, `blocker_class`, `claim_scope`, and `summary`. Repair,
solver-grounded verification, FR-11, KAN/PWA, GateMate, and SSQA statuses MUST
cite concrete source artifact paths and fields. Repair evidence MAY remain
bounded or gate-skipped without requiring a live repair rerun. GateMate and
SSQA rows MAY remain blocked or gate-skipped without requiring hardware success.
FR-11 rows MUST preserve the controller-only boundary and MUST NOT promote
model-weight learning without a source artifact that explicitly supports it.

The terminal artifact MUST include `matrix_v20_ready`, `rows_total`,
`clean_rows`, `flagged_rows`, `bounded_rows`, `blocked_rows`,
`gated_skipped_rows`, `projection_only_rows`, `missing_rows`, `retired_rows`,
`publication_blocker_count`, `source_artifacts`, `inference_substrate`, and
`honest_verdict`. It MUST also include a machine-readable list of
`publication_blockers` whose length equals `publication_blocker_count`.
`matrix_v20_ready` MUST be true only when all row-class lists exist, all rows
have valid classes and source-field citations, required source artifacts are
readable JSON objects, and optional missing artifacts are explicitly represented
as missing rows and source records. When a conductor prompt assigns ops
reconciliation to a separate step, the generator MUST leave `ops/status.md`,
`ops/changelog.md`, and `_bmad/traceability.md` unchanged.

#### SCENARIO-REPORT-3065: V20 Aggregates .285 And .286 Rows With Honest Gates

**Given** matrix v19, capstone v285, and `.286` artifacts Exp 3054 through Exp
3064 are present or honestly absent
**And** the actual Exp 3059 blocked-gate artifact exists while the requested
`experiment_3059_gated_sota_repair_de_tautology_rerun_v1.json` alias is absent
**And** repair remains bounded or gate-skipped, solver-grounded verification has
flagged local SOTA rows, FR-11 remains controller-only, KAN/PWA is an exact
controller-anchor audit rather than a promoted model-weight verifier, GateMate
does not allow rerun, and SSQA remains host-visible-smoke gated
**When** the Exp 3065 matrix v20 generator runs
**Then** it writes `results/experiment_3065_cross_corpus_matrix_v20.json` with
`matrix_v20_ready=true`, all eight row-class lists populated or explicitly
empty, source artifact and source-field citations on every row, the missing
Exp 3059 alias recorded as a missing row, every publication blocker listed and
counted, aggregation-only inference substrate metadata, no live repair or
hardware execution by the matrix task, and an `honest_verdict` that starts with
`complete:`.

## Implementation Status (REQ-REPORT-3065)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-3065 | Implemented (`python/carnot/reporting/cross_corpus_matrix_v20_3065.py`) | Implemented (`tests/python/test_experiment_3065_cross_corpus_matrix_v20.py`) |

### REQ-REPORT-3066: Milestone 2026.05.286 Capstone From Matrix V20

The repository shall provide an Exp 3066 milestone capstone generator that
writes `results/experiment_3066_capstone_v286.json` using only checked-in
matrix v20 and source artifact JSON files. The generator MUST read
`results/experiment_3065_cross_corpus_matrix_v20.json` and the source
artifacts cited by that matrix before deciding capstone readiness, paper
readiness, repair claim status, solver-grounded verification status, FR-11
self-learning status, KAN/PWA status, GateMate status, SSQA status,
publication blockers, and the exact next-milestone recommendation. It MUST NOT
run live repair, live LLM inference, verifier scoring, solver execution,
synthesis, board flashing, readback, hardware smoke tests, the conductor,
external publication tooling, or historical artifact rewrites.

The capstone MUST set `capstone_ready=true` only when matrix v20 exists,
reports `matrix_v20_ready=true`, all eight row-class lists are present, row
counts reconcile with their statuses, publication blockers are counted
consistently, and required source artifacts are readable JSON objects. It MUST
set `paper_ready=true` only when matrix v20 has zero publication blockers and
every promoted clean claim has a concrete present source artifact and source
field. Controller-side FR-11 evidence MAY be carried as bounded self-learning
evidence, but it MUST NOT be broadened into model-weight learning without a
source artifact that explicitly trained and verified model weights. GateMate,
SSQA, and hardware speedup claims MUST remain blocked or gate-skipped unless
host-visible output evidence exists.

The terminal artifact MUST include `capstone_ready`, `paper_ready`,
`repair_claim_status`, `solver_grounding_status`, `fr11_self_learning_status`,
`kan_pwa_status`, `gatemate_status`, `ssqa_status`, `publication_blockers`,
`next_milestone_recommendation`, `source_artifacts`, `inference_substrate`, and
`honest_verdict`. It MAY include matrix-v20 summaries, paper-readiness checks,
promoted or blocked claim rows, source checksums, missing source artifacts,
no-new-execution booleans, ops-reconciliation booleans, and measured
`duration_s` as long as every value is derived from checked-in artifacts or
file presence/checksum checks. When a conductor prompt assigns ops
reconciliation to a separate step, the generator MUST leave `ops/status.md`,
`ops/changelog.md`, and `_bmad/traceability.md` unchanged.

#### SCENARIO-REPORT-3066: Capstone Closes .286 From Matrix V20 Without Paper Overclaim

**Given** matrix v20 is present and reports `matrix_v20_ready=true`
**And** matrix v20 contains clean, flagged, bounded, blocked, gated-skipped,
missing, projection-only, and retired row-class lists with source provenance
**And** matrix v20 reports nonzero publication blockers for bounded repair,
flagged solver-grounded verification, controller-only FR-11, bounded KAN/PWA,
blocked GateMate, and host-visible-smoke-gated SSQA
**When** the Exp 3066 capstone generator runs
**Then** it writes `results/experiment_3066_capstone_v286.json` with
`capstone_ready=true`, `paper_ready=false`, matrix v20 counts reconciled,
repair carried as bounded, solver-grounded verification carried as flagged
solver-authority evidence with no positive gain, FR-11 carried only as
controller-side evidence, KAN/PWA carried as bounded controller-anchor audit
evidence, GateMate blocked until host-visible output evidence exists, SSQA
gate-skipped until the GateMate smoke gate passes, every publication blocker
listed and counted, aggregation-only inference substrate metadata, no live
model or hardware execution by the capstone task, and an `honest_verdict` that
starts with `complete:`.

## Implementation Status (REQ-REPORT-3066)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-3066 | Implemented (`python/carnot/reporting/capstone_v286_3066.py`) | Implemented (`tests/python/test_experiment_3066_capstone_v286.py`) |

### REQ-REPORT-3068: Matrix V20 Artifact-Alias And Blocker Normalization Ledger

The repository shall provide an Exp 3068 normalization-ledger generator that
writes
`results/experiment_3068_matrix_v20_artifact_alias_blocker_normalization_v1.json`
using only checked-in matrix v20, capstone v286, conductor-log, ops-status,
ops-changelog, and result-filename evidence. The generator MUST read
`results/experiment_3065_cross_corpus_matrix_v20.json`,
`results/experiment_3066_capstone_v286.json`, the actual
`results/experiment_3059_gated_sota_repair_de_tautology_rerun.json`, conductor
log entries for Exp 3054 through Exp 3066, and actual result filenames before
writing the ledger. It MUST NOT rewrite any prior result artifact, run live
model inference, run repair, invoke verifier scoring, run solver execution,
run synthesis, flash hardware, perform readback, run the conductor, or mark any
research claim clean solely because a filename alias was resolved.

The ledger MUST identify source-artifact path mismatches, missing-source rows,
duplicate rows, bounded rows, blocked rows, retired rows, and projection-only
rows from matrix v20. The Exp 3059 requested
`experiment_3059_gated_sota_repair_de_tautology_rerun_v1.json` path MUST be
recorded as an explicit non-destructive alias to the actual checked-in
gate-blocked artifact
`experiment_3059_gated_sota_repair_de_tautology_rerun.json` when that artifact
is present. This alias MAY remove only the artifact-hygiene missing-file
blocker; it MUST preserve the Exp 3059 gate-blocked/gated-skipped research
status and all other missing evidence.

The terminal artifact MUST include
`matrix_v20_normalization_ready`, `artifact_aliases`,
`missing_artifacts_after_aliasing`, `blocker_categories`,
`exp3059_alias_status`, `publication_blocker_count_before`,
`normalized_blocker_count_estimate`, `source_artifacts`,
`inference_substrate`, and `honest_verdict`. `blocker_categories` MUST separate
research blockers, artifact hygiene blockers, true missing evidence, honest
bounded rows, retired rows, duplicate rows, blocked rows, and projection-only
rows. `normalized_blocker_count_estimate` MUST be auditable from matrix v20's
publication-blocker count minus only the resolved artifact-hygiene alias
blockers; bounded, flagged, blocked, gated-skipped, projection-only, and true
missing evidence rows MUST remain visible for matrix v21.

#### SCENARIO-REPORT-3068: Exp 3059 Alias Does Not Clean Research Blockers

**Given** matrix v20 and capstone v286 are present
**And** matrix v20 reports the requested Exp 3059 `_v1` artifact path as
missing while the actual gate-blocked Exp 3059 artifact exists
**And** matrix v20 has bounded, blocked, retired, projection-only, duplicate,
and true missing-evidence rows
**When** the Exp 3068 normalization-ledger generator runs
**Then** it writes
`results/experiment_3068_matrix_v20_artifact_alias_blocker_normalization_v1.json`
with `matrix_v20_normalization_ready=true`, an Exp 3059 alias record mapping
the requested `_v1` path to the actual gate-blocked artifact without rewriting
either path, the true missing artifacts still listed after aliasing, blocker
categories separated into research, artifact-hygiene, and honest
bounded/retired groups, `publication_blocker_count_before` copied from matrix
v20, `normalized_blocker_count_estimate` reduced only by the resolved alias
blocker, aggregation-only inference-substrate metadata, and an `honest_verdict`
that starts with `complete:`.

## Implementation Status (REQ-REPORT-3068)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-3068 | Implemented (`python/carnot/reporting/matrix_v20_artifact_alias_blocker_normalization_3068.py`) | Implemented (`tests/python/test_experiment_3068_matrix_v20_artifact_alias_blocker_normalization.py`) |

### REQ-REPORT-3069: Solver-Verifier Failure Autopsy And Recovery Protocol

The repository shall provide an Exp 3069 artifact-only autopsy generator that
writes
`results/experiment_3069_solver_verifier_failure_autopsy_protocol_v1.json`
before any local SOTA verifier repair, repair promotion, or LLM-guided SMT
promotion is retried. The generator MUST read Exp 3057, Exp 3058, matrix v20
Exp 3065, capstone v286 Exp 3066, CODEX/CLAUDE workflow instructions, and
`research-references.md`. It MUST NOT run live LLM inference, run a fresh
solver experiment, invoke verifier scoring, modify
`scripts/research_conductor.py`, push, or rewrite prior result artifacts.

The autopsy MUST extract the Exp 3057 and Exp 3058 metrics that explain the
solver-grounded failure: Exp 3057 false-negative rate, false-positive rate,
one-shot solver accuracy, verifier-selected accuracy, verifier-gain delta,
exact-solver agreement, adversarial flags, and Exp 3058 guided success,
solver-only success, guided-minus-solver-only lift, invalid proposal count,
formal fallback preservation, and adversarial flags. It MUST classify failure
modes including false negatives, no verifier gain, no SMT lift,
self-verification risk, and solver-only equivalence. It MUST translate the
research references into predeclared local diagnostics for first-token entropy,
abstention precision, rejection recall, confidence coverage,
Lyapunov-style perturbation sensitivity when logits or trajectories are
accessible, and VERGE/MCS feedback.

The terminal artifact MUST include
`verifier_failure_autopsy_ready`, `root_cause_hypotheses`,
`recovery_protocol`, `abstention_policy`, `candidate_signals`,
`promotion_disqualifiers`, `source_artifacts`, `inference_substrate`, and
`honest_verdict`. The recovery protocol MUST define minimum artifact fields,
exact-solver authority requirements, acceptance gates, and consumer-ready
blocking conditions for Exp 3070, Exp 3071, Exp 3072, and Exp 3075.
`verifier_failure_autopsy_ready` shall be true only when all required sources
are present as readable artifacts or text references, all five failure modes
are classified, candidate diagnostics are predeclared, promotion
disqualifiers cover Exp 3070/3071/3072/3075, the recovery protocol can be
consumed directly by next experiments, the inference substrate explicitly
declares no live model inference, and `honest_verdict` starts with a terminal
success prefix.

#### SCENARIO-REPORT-3069: Failed Verifier Gain Blocks Promotion Until Explained

**Given** Exp 3057 reports negative verifier gain with a false-negative
failure
**And** Exp 3058 reports zero guided SMT lift relative to solver-only fallback
**And** matrix v20 and capstone v286 carry solver-grounded verification as
flagged no-gain evidence
**When** the Exp 3069 autopsy generator runs
**Then** it writes
`results/experiment_3069_solver_verifier_failure_autopsy_protocol_v1.json`
with `verifier_failure_autopsy_ready=true`, root-cause hypotheses tied to
concrete source artifacts, a bounded verifier-gain recovery protocol with
minimum fields and exact-solver authority gates, an abstention policy that
prevents forced accept/reject on uncertain cases, candidate diagnostic signals
for confidence and feedback calibration, promotion disqualifiers for Exp
3070, Exp 3071, Exp 3072, and Exp 3075, artifact-only inference-substrate
metadata, and an `honest_verdict` that starts with `complete:`.

## Implementation Status (REQ-REPORT-3069)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-3069 | Implemented (`python/carnot/reporting/solver_verifier_failure_autopsy_protocol_3069.py`) | Implemented (`tests/python/test_experiment_3069_solver_verifier_failure_autopsy_protocol.py`) |

### REQ-REPORT-3074: LLGuidance/AprAD Repair Micro-Panel Protocol

The repository shall provide an Exp 3074 artifact-only repair micro-panel
protocol generator that writes
`results/experiment_3074_llguidance_aprad_repair_protocol_v1.json` before any
Exp 3075 SOTA repair generation is attempted. The generator MUST read Exp 3056
and Exp 3059, and SHOULD also cite capstone v286 and research-reference
metadata when present. It MUST preserve all de-tautology disqualifiers from Exp
3056, preserve the Exp 3059 verifier-gain gate-blocking reason, and MUST NOT
run live LLM inference, model loading, verifier scoring, solver execution,
synthesis, board flashing, the conductor, pushes, or historical artifact
rewrites.

The protocol MUST translate LLGuidance-style grammar constraints into local
protocol fields for grammar source, constrained syntax target, schema
validation, parse failures, and fallback behavior. It MUST translate
AprAD-style intent preservation into local protocol fields for task intent
hash, behavioral tests, semantic drift checks, and verifier authority. Syntax
validity alone MUST NOT authorize a repair: the protocol must require exact
semantic validation by deterministic tests, exact solver or verifier authority,
or an explicitly blocked outcome when that authority is unavailable.

The terminal artifact MUST include
`grammar_constrained_repair_protocol_ready`,
`schema_syntax_failure_targets`, `exact_semantic_validation_required`,
`aprad_intent_preservation_rules`, `llguidance_runtime_plan`,
`de_tautology_disqualifiers`, `exp3075_required_fields`,
`inference_substrate`, and `honest_verdict`. The Exp 3075 field contract MUST
include a clean blocked outcome for verifier-gain gate failures so matrix v21
can distinguish "not run because a required verifier-gain gate failed" from a
failed or missing repair artifact. `grammar_constrained_repair_protocol_ready`
shall be true only when Exp 3075 can consume the protocol directly from
machine-readable field lists, failure targets, grammar runtime plan,
intent-preservation rules, carried-forward disqualifiers, and no-live-inference
substrate metadata.

#### SCENARIO-REPORT-3074: Grammar And Intent Gates Precede SOTA Repair

**Given** Exp 3056 reports a ready de-tautology protocol with promotion
disqualifiers
**And** Exp 3059 reports a gate-blocked SOTA repair rerun because verifier gain
failed
**When** the Exp 3074 protocol generator runs
**Then** it writes
`results/experiment_3074_llguidance_aprad_repair_protocol_v1.json` with
`grammar_constrained_repair_protocol_ready=true`, all Exp 3056 de-tautology
disqualifiers preserved, schema and syntax failure targets declared for
matrix-v21 measurement, an explicit LLGuidance runtime plan with deterministic
schema fallback, AprAD-inspired intent-preservation rules that require task
intent hashes, behavioral tests, semantic-drift checks, and independent
verifier authority, `exact_semantic_validation_required=true`, an Exp 3075
required-field contract including clean blocked verifier-gain outcomes,
artifact-only inference-substrate metadata, and an `honest_verdict` that starts
with `complete:`.

## Implementation Status (REQ-REPORT-3074)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-3074 | Planned (`python/carnot/reporting/llguidance_aprad_repair_protocol_3074.py`) | Planned (`tests/python/test_experiment_3074_llguidance_aprad_repair_protocol.py`) |

### REQ-REPORT-3079: Cross-Corpus Matrix V21 From .286 And .287 Artifacts

The repository shall provide an Exp 3079 cross-corpus matrix v21 generator
that writes `results/experiment_3079_cross_corpus_matrix_v21.json` using only
checked-in `.286` and `.287` artifact JSON files plus the conductor log. The
generator MUST read matrix v20, capstone v286, the Exp 3068 artifact-alias and
blocker-normalization ledger, and every available `.287` artifact before
classifying rows. It MUST apply Exp 3068 artifact aliases non-destructively:
alias normalization MAY retire the resolved artifact-hygiene missing row, but
MUST NOT mark the underlying research gate, repair, verifier, FR-11, or
hardware claim clean solely because a filename mismatch was resolved. The
generator MUST NOT modify any prior result file, run live model inference, run
repair, invoke verifier scoring, run solver execution, run synthesis, flash
hardware, perform readback, run the conductor, push, or modify
`scripts/research_conductor.py`.

The v21 artifact MUST classify evidence-first rows into exactly these row
statuses: `clean`, `flagged`, `bounded`, `blocked`, `gated_skipped`,
`projection_only`, `missing`, and `retired`. Every row in the top-level `rows`
list MUST include at least `row_id`, `status`, `source_artifact`,
`source_field`, `evidence_class`, `blocker_class`, `claim_scope`, and
`summary`. Matrix v20 unresolved publication blockers MUST be carried forward
unless Exp 3068 explicitly resolves them as artifact-hygiene aliases. New
`.287` rows MUST add blockers when conductor gates fail, adversarial flags are
present, methodology corrigenda remain open, verifier-gain or abstention
metrics fail gates, FR-11 soundness/completeness budgets are exceeded, repair
micro-panel execution is gate-skipped, or GateMate/SSQA operator preconditions
remain absent. Future-context audits such as EBT/ARM adapter feasibility MUST
remain `projection_only` or bounded unless a checked-in implementation and
test evidence support a stronger row.

The terminal artifact MUST include `matrix_v21_ready`, `rows_total`,
`clean_rows`, `flagged_rows`, `bounded_rows`, `blocked_rows`,
`gated_skipped_rows`, `projection_only_rows`, `missing_rows`, `retired_rows`,
`publication_blocker_count`, `rows`, `source_artifacts`,
`inference_substrate`, and `honest_verdict`. The row-count fields MUST be
integers, not row lists. `matrix_v21_ready` MUST be true only when all required
sources are readable, all row statuses are valid, row counts reconcile with
`rows_total`, `publication_blocker_count` reconciles with all non-clean and
non-retired rows, every row traces to a source artifact or conductor-log
source field, and the inference substrate declares aggregation from upstream
artifacts with no live model or hardware execution.

#### SCENARIO-REPORT-3079: V21 Aggregates .287 Evidence Without Alias Overclaim

**Given** matrix v20, capstone v286, and the Exp 3068 normalization ledger are
present
**And** Exp 3068 maps the requested Exp 3059 `_v1` artifact alias to the actual
gate-blocked artifact without changing the research status
**And** available `.287` artifacts include archive activation, normalization,
failure autopsy, first-token and VERGE/MCS panels, the Exp 3072 blocked gate
record, EBT/ARM adapter feasibility, repair protocol, FR-11 budget and pilot
artifacts, and the GateMate/SSQA refresh
**And** Exp 3075 is absent but the conductor log records it as a structured
gate skip because the verifier-gain precondition failed
**When** the Exp 3079 matrix v21 generator runs
**Then** it writes `results/experiment_3079_cross_corpus_matrix_v21.json` with
`matrix_v21_ready=true`, the Exp 3059 artifact-hygiene row retired rather than
cleaned, all unresolved v20 blockers carried forward, new flagged/gated/
blocked/projection-only `.287` blockers added from the checked-in evidence,
integer status counts that sum to `rows_total`, `publication_blocker_count`
equal to the non-clean/non-retired row count, source citations for every row,
aggregation-only inference substrate metadata, and an `honest_verdict` that
starts with `complete:`.

## Implementation Status (REQ-REPORT-3079)

| Requirement | Implementation | Tests |
|---|---|---|
| REQ-REPORT-3079 | Planned (`python/carnot/reporting/cross_corpus_matrix_v21_3079.py`) | Planned (`tests/python/test_experiment_3079_cross_corpus_matrix_v21.py`) |
