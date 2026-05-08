# Publication Readiness Capability Specification

**Capability:** publication
**Version:** 0.1.0
**Status:** Draft
**Traces to:** FR-PUB

## Overview

Defines the criteria that must be satisfied before any Carnot headline result
can be published, shared publicly, or cited in external communications.

The core invariant: every number in a headline claim must trace to a live-GPU
inference run recorded in a result JSON. Simulated, mocked, or CPU-only results
may appear in the repo as reference points but must be excluded from headline
claims.

## Requirements

### REQ-PUBLISH-001: Live-GPU Provenance for Headline Numbers

All headline VR numbers (signed_improvement, cross_model_delta, grammar_recall,
safety AUROC) MUST have `inference_mode == "live_gpu"` in the source result
file. A result with `inference_mode != "live_gpu"` (e.g. "blocked", "cpu",
"simulated") MUST be excluded from headline claims and marked as a caveat or
negative result in the publication.

Why: The project's credibility rests on the energy function being ground truth.
Allowing simulated numbers into headlines would undermine the anti-hallucination
claim that motivates the project (see _bmad/prd.md Phase 1).

### REQ-PUBLISH-002: Negative Results Section Required

The model card and technical report MUST include a dedicated "Negative Results"
section documenting all significant failures alongside successes. Specifically:

- JEPA v15 OOD AUC = 0.4751 (below random, GSM8K 500-699)
- JEPA v16 OOD AUC = 0.4759 (InfoNCE contrastive loss did not fix root cause)
- Adversarial VR: blocked / not yet measured on live GPU
- Cross-model Gemma-4-E4B-it: signed_improvement = -0.8 (regression)

Why: Publishing only positive results is selection bias. Negative results help
the community understand where constraint-based EBMs succeed and where they
currently fail. They also protect the project's long-term credibility: if
failures are discovered by third parties they should already be documented.

### REQ-PUBLISH-003: Gate Check Before Publication Actions

A publication readiness audit script MUST load all gate result files and compute
a boolean `publication_ready` flag before any publication action is taken. The
script MUST emit an `honest_verdict` field in the result artifact with one of:

- `"publication_ready"` — all provenance valid AND cross-model result exists
- `"publication_ready_with_caveats"` — provenance valid but cross-model blocked
- `"publication_blocked_no_primary_result"` — Exp 679 gate fails

### REQ-PUBLISH-004: Position Paper v3 Findings Update Artifact

The arXiv position-paper update script MUST verify that
`docs/arxiv-paper/main.tex` incorporates the milestone .87-.88 findings before
emitting `results/experiment_1135_position_paper_v3_findings_update.json`.
The artifact MUST record:

- the paper sections modified
- whether the GRPO + ThinkPRM v2 result was integrated
- whether the exp1120 energy-inversion fix was integrated
- whether the exp1130 Zenil `alpha_t` post-retrain result was integrated
- whether HIVE arXiv:2604.26139 was added to Related Work
- a closed-set `honest_verdict`

The script MUST derive the reported numerical values from the checked-in
experiment result JSON files rather than hard-coding them in the artifact.

### REQ-PUBLISH-005: arXiv Final Submission v4 Bundle Artifact

The arXiv final-submission runner MUST verify that
`docs/arxiv-paper/main.tex` contains the GRPO v2 `+8.51` pp result and the
milestone .89 projection-repair and MetaCluster compression summaries before
emitting `results/experiment_1153_arxiv_final_submission_v4.json`. The runner
MUST recompile `docs/arxiv-paper/main.pdf`, repack
`results/carnot-arxiv-v4.tar.gz`, and record browser-ready manual arXiv upload
steps. The artifact MUST include closed-set booleans for paper integrations,
PDF and bundle verification, submission state, and an `honest_verdict` from the
allowed v4 final-submission verdict set.

### REQ-PUBLISH-006: Phase 4 Active-Inference Section 7 Artifact

The arXiv Phase 4 revision runner MUST verify that
`docs/arxiv-paper/main.tex` expands Section 7 from the placeholder
Themesis/Seed IQ acknowledgment into a substantive empirical-comparison
section before emitting
`results/experiment_1167_paper_v4_phase4_section.json`. The revised section
MUST include four numbered subsections covering theoretical equivalence,
Exp 1165 pilot results, Exp 1166 ARC-AGI-3 leaderboard context, and an honest
gap analysis. The runner MUST verify that the paper includes the Exp 1165
`action_count_ratio`, Seed IQ leaderboard context, and the statement that
Carnot's `F(z)=sum_k w_k E_k(z)` is a variational-free-energy approximation.
It MUST recompile `docs/arxiv-paper/main.pdf`, repack
`results/carnot-arxiv-v5.tar.gz`, and emit the required closed-set
`honest_verdict`.

## Scenarios

### SCENARIO-PUBLISH-001: All Headline Numbers Have Live-GPU Provenance

**Given** all source result files have `inference_mode == "live_gpu"`
**When** the provenance audit runs
**Then** every entry in the provenance table has `provenance_valid = True`
 AND `publication_ready = True`

### SCENARIO-PUBLISH-002: Negative Results Documented in Model Card

**Given** JEPA v15 OOD AUC = 0.4751 is in the source result
**When** the model card draft is written
**Then** the model card contains a "Negative Results" section that includes
 the JEPA v15 OOD AUC value AND the adversarial VR blocked status

### SCENARIO-PUBLISH-003: Primary Result Gate Fails

**Given** Exp 679 result file is absent or `vr_200q_validated` is False
**When** the publication readiness script runs
**Then** `honest_verdict == "publication_blocked_no_primary_result"`
 AND `publication_ready == False`

### SCENARIO-PUBLISH-004: Position Paper Findings Fully Integrated

**Given** the exp1118, exp1120, exp1121, exp1129, and exp1130 result files exist
AND the position paper includes the milestone .87-.88 findings
**When** the findings-update script runs
**Then** the deliverable JSON reports the Abstract, Results, and Related Work
sections as modified
AND `honest_verdict == "fully_updated"`
AND every required integration boolean is true

### SCENARIO-PUBLISH-005: Final Submission Bundle Ready For Manual Upload

**Given** the position paper source exists and the exp1147 and exp1148 result
artifacts exist
**When** the exp1153 final-submission runner executes
**Then** the paper contains the GRPO v2, projection-repair, and MetaCluster
summaries
AND `docs/arxiv-paper/main.pdf` is recompiled
AND `results/carnot-arxiv-v4.tar.gz` is verified
AND the deliverable JSON contains exact manual arXiv upload steps

### SCENARIO-PUBLISH-006: Phase 4 Section Ready For Hold-Lift Review

**Given** the position paper source and Exp 1165/1166 result artifacts exist
AND Section 7 contains the four Phase 4 active-inference comparison
subsections
**When** the exp1167 Phase 4 paper-revision runner executes
**Then** `docs/arxiv-paper/main.pdf` is recompiled
AND `results/carnot-arxiv-v5.tar.gz` is verified
AND the deliverable JSON reports `paper_ready_for_arxiv_hold_lift = True`
AND `honest_verdict == "paper_v4_phase4_complete_arxiv_ready"`

### REQ-PUBLISH-007: High-Severity Integrity Fixes (ISSUE-6 through ISSUE-10)

The paper-v5 high-severity remediation script MUST verify that
`docs/arxiv-paper/main.tex` contains all five fixes before emitting
`results/experiment_1181_paper_v5_high_issues_6_10.json`. The artifact MUST
record:

- `issue_6_grpo_cis_added`: GRPO delta claims include inline sample size and
  Clopper-Pearson 95% CI annotations (n=25/n=47 small-sample binomial CIs).
- `issue_6_small_sample_caveat_added`: a footnote warning that GRPO delta
  estimates on n=25-47 are preliminary indicators, not definitive accuracy claims.
- `issue_7_humaneval_reframed`: the HumanEval 0.0% baseline is explained as a
  harness extraction failure, result moved to anomaly context.
- `issue_8_alpha_t_rejection_rate_added`: the 24/100 false rejection rate is
  disclosed alongside the alpha_t=0.38 claim.
- `issue_9_phase4_baseline_caveat_added`: the Phase-4 74.7% action reduction is
  annotated as compared against a random-legal-greedy baseline with a forward
  reference to a stronger BFS-to-goal comparison.
- `issue_10_seed_iq_footnote_added`: a footnote on the Seed IQ table row states
  the value was not independently re-fetched (exp1166: seed_iq_score_confirmed=false).
- `high_severity_fixed`: count of fixes applied (must equal 5).
- `4_test_passes_high`: all fixes satisfy the paper-v5 4-test.
- `honest_verdict`: one of "all_5_high_resolved" | "partial_fix" | "blocked".

### SCENARIO-PUBLISH-007: All Five High-Severity Fixes Verified

**Given** main.tex is accessible and the five high-severity issues exist
**When** the exp1181 remediation script runs
**Then** all five issue booleans are True
  AND `high_severity_fixed == 5`
  AND `honest_verdict == "all_5_high_resolved"`

### REQ-PUBLISH-008: Medium/Low-Severity Integrity Fixes (ISSUE-11 through ISSUE-18)

The paper-v5 medium/low remediation script MUST verify that
`docs/arxiv-paper/main.tex` and `docs/arxiv-paper/carnot.bib` contain all
eight fixes before emitting
`results/experiment_1182_paper_v5_medium_low_issues_11_18.json`. The
artifact MUST record:

- `issue_11_thinkprm_citation_fixed`: the ThinkPRM AUROC=0.9885 claim names
  exp1111 v1 and the exp1111 v2 retrain artifact.
- `issue_12_holdout_n_stated`: FoVer holdout claims disclose `n=50 holdout`
  and cross-reference exp1121's production-corpus AUROC=0.3333 reading.
- `issue_13_nrgpt_disclosure_added`: any NRGPT citation discloses
  AUROC_n1=0.9209, AUROC_n3=0.9158, and `n_iters_monotone=False`, or the
  artifact records that NRGPT is not cited in the paper.
- `issue_14_soskan_auroc_reconciled`: every SOS-KAN/SOSKAN AUROC claim names
  its corpus and sample size.
- `issue_15_fig2_caveat_added`: the Figure 2 caption states that the binormal
  curve is fit from published AUROC rather than re-evaluated on held-out data.
- `issue_16_bib_stubs_removed`: count of stub/fabricated bibliography entries
  removed after audit.
- `issue_17_k15_caption_tightened`: Table 1 explains that k=15 is the
  theoretical maximum from Theorem 3.2 and not an experimentally achieved
  result.
- `issue_18_hardware_scope_added`: the hardware-portability theorem states
  that only KV260 FPGA has been empirically verified at submission time.
- `medium_low_issues_fixed`: count of issues resolved (must equal 8).
- `honest_verdict`: one of "all_8_medium_low_resolved" | "partial_fix" |
  "blocked".

### REQ-PUBLISH-009: Paper Numerical Claim Audit

The paper numerical-claim audit script MUST scan `docs/arxiv-paper/main.tex`
for numerical claims using the configured paper-claim regex, count how many
claims have a following `(expNNNN)` artifact citation within 200 characters,
load the corresponding `results/experiment_NNNN_*.json` artifacts, and verify
that cited numerical values match artifact fields after documented LaTeX and
unit normalization. The script MUST report:

- `n_claims_total`
- `n_claims_with_artifact_citation`
- `n_claims_verified`
- `n_mismatches`

The script MUST exit nonzero when any mismatch exists or when
`n_claims_with_artifact_citation / n_claims_total < 0.8`.

### REQ-PUBLISH-010: Paper v5 Recompile And arXiv Bundle v6 Gate Record

The paper-v5 recompile runner MUST refuse to compile or bundle the paper unless
the Exp 1180 critical-fix gate and Exp 1181 high-severity gate are both present
and true. When the prerequisite gates pass, the runner MUST execute the figure
integrity audit and paper numerical-claim audit, attempt the `pdflatex` +
`bibtex` compile pipeline, build the v6 arXiv source bundle, run the final
banned-string grep checks, and emit
`results/experiment_1183_paper_v5_recompile_arxiv_bundle_v6.json`.

The artifact MUST record:

- `pdf_compiles_without_error`
- `arxiv_bundle_v6_ready`
- `arxiv_bundle_path`
- `figure_audit_untraced_constants`
- `claim_audit_n_mismatches`
- `known_remaining_issues`
- `fabricated_constants_remaining`
- `paper_word_count`
- `4_test_full_pass`
- `honest_verdict`, one of `"arxiv_bundle_v6_ready"`,
  `"compilation_failed"`, or `"audit_failures_remain"`

Audit failures MUST be documented in the artifact without preventing source
bundle creation once the Exp 1180 and Exp 1181 prerequisite gates are true.

### REQ-PUBLISH-011: Figure 3 Measured-Only FPGA Latency

Figure 3 MUST display only measured latency data sourced from
`results/experiment_1068_kv260_smoke_test_v9.json`. It MUST NOT include the
unmeasured 290 ms CPU baseline, per-200-sample CPU sweep comparisons, or any
derived CPU-vs-FPGA speedup badge unless the CPU baseline is measured on the
same per-sample basis and recorded in a result artifact. The figure-rendering
module MUST expose the measured FPGA latency and render the PNG/PDF outputs
from that measured value.

### REQ-PUBLISH-012: Paper v6 Critical Integrity Fix Gate

The paper-v6 critical integrity fixer MUST verify that
`docs/arxiv-paper/main.tex` resolves all five arXiv-blocking integrity issues
before emitting `results/experiment_1257_paper_v6_critical_issues_fix.json`.
The artifact MUST record boolean fields `issue_1_fix_applied` through
`issue_5_fix_applied`, `critical_issues_fixed == 5`,
`issues_fixed_list == ["ISSUE-1", "ISSUE-2", "ISSUE-3", "ISSUE-4", "ISSUE-5"]`,
`status == "complete"`, and
`honest_verdict == "paper_v6_5_of_5_critical_issues_fixed"`.

The five fixes are:

- ISSUE-1: any Figure 3 or hardware-latency CPU baseline caveat MUST state
  that the 290 ms CPU reference was an order-of-magnitude estimate and direct
  readers to the exp1094 measured CPU baseline.
- ISSUE-2: the KL=3.07 finding MUST be labeled as a software-simulated Glauber
  dynamics proxy, with FPGA bitstream measurement explicitly deferred.
- ISSUE-3: the `15.6x` speedup and `CPU_GIBBS_PER_SWEEP_NS = 1000.0` baseline
  MUST be absent from `main.tex`; CPU latency prose MUST cite the measured
  exp1094 value of approximately 15.964 us per sweep.
- ISSUE-4: the 76,130x HumanEval/HardNet++ speedup headline MUST be absent;
  any retained latency comparison MUST be framed as a different task class
  rather than a single speedup metric.
- ISSUE-5: SOSKANEnergyV3 AUROC claims MUST distinguish in-distribution,
  production/OOD, and post-fix production measurements instead of presenting
  multiple AUROCs as one comparable number.

### SCENARIO-PUBLISH-008: All Medium/Low Fixes And Claim Audit Verified

**Given** the position paper source, bibliography, and local experiment result
artifacts exist
**When** the exp1182 remediation script runs
**Then** all eight issue booleans are True
  AND `paper_claim_audit_script_active` is True
  AND `paper_claim_audit_n_mismatches == 0`
  AND `medium_low_issues_fixed == 8`
  AND `honest_verdict == "all_8_medium_low_resolved"`

### SCENARIO-PUBLISH-009: Exp 1183 Blocks Before Recompile When Prior Gates Are Missing

**Given** Exp 1180 has not emitted a successful critical-fix artifact
**When** the exp1183 recompile runner executes
**Then** it writes the required gate-record schema with
`prerequisites_met == False`
AND `arxiv_bundle_v6_ready == False`
AND it does not run the audit, compile, or bundle steps.

### SCENARIO-PUBLISH-010: Exp 1183 Records Bundle And Audit Status

**Given** Exp 1180 and Exp 1181 have both emitted successful gate artifacts
**When** the exp1183 recompile runner executes
**Then** it records the audit counts, banned-string count, paper word count,
bundle path, compile status, and a closed-set `honest_verdict`.

### SCENARIO-PUBLISH-011: Figure 3 Removes Unmeasured Speedup Claim

**Given** Exp 1068 records the measured KV260 FPGA latency
**When** Figure 3 is rendered
**Then** the plotted data contains exactly the measured FPGA latency bar
AND no unmeasured CPU baseline or derived CPU-vs-FPGA speedup annotation is
rendered.

### SCENARIO-PUBLISH-012: All Five Paper v6 Critical Fixes Verified

**Given** `docs/arxiv-paper/main.tex`, the exp1094 sampler-correctness artifact,
and the exp1257 deliverable path exist
**When** the exp1257 paper-v6 critical fixer runs
**Then** all five issue booleans are True
AND `critical_issues_fixed == 5`
AND `issues_fixed_list` names ISSUE-1 through ISSUE-5
AND `honest_verdict == "paper_v6_5_of_5_critical_issues_fixed"`
AND `status == "complete"`.

### REQ-PUBLISH-013: Paper v6 Critical Fixes v2 Terminal Artifact

The paper-v6 critical-fixes v2 auditor MUST verify that
`docs/arxiv-paper/main.tex` removes or explicitly caveats the five
publication-blocking claim classes before emitting
`results/experiment_1269_paper_v6_critical_fixes_v2.json`.
The five claim classes are estimated CPU/FPGA speedups, KL measurement
provenance, hand-typed CPU constants, apples-to-oranges HumanEval latency,
and SOS-KAN AUROC ambiguity.

The artifact MUST record:

- `critical_issues_fixed == 5`
- `issues_fixed_list` naming the five fixed claim classes
- `measured_artifacts_cited` including exp1256, exp1264, exp1265, and exp1266
- `old_claims_remaining == []`
- `status == "complete"`
- `honest_verdict == "paper_v6_critical_fixes_v2_complete"`

### SCENARIO-PUBLISH-013: Critical Fixes v2 Audit Is Clean

**Given** `docs/arxiv-paper/main.tex` and the exp1256/1264/1265/1266
measured artifacts exist
**When** the exp1269 critical-fixes v2 auditor runs
**Then** all five claim classes are fixed
AND the paper cites all four measured artifacts
AND the banned-string audit has no remaining old unsupported claim strings
AND the deliverable JSON reports `status == "complete"` and
`honest_verdict == "paper_v6_critical_fixes_v2_complete"`.

### REQ-PUBLISH-014: Gated arXiv Bundle v10 Artifact

The arXiv bundle-v10 runner MUST create an in-progress deliverable at
`results/experiment_1270_arxiv_bundle_v10_gated.json` before attempting any
compile or packaging command. It MUST refuse to produce a completed bundle
unless `results/experiment_1269_paper_v6_critical_fixes_v2.json` records
`critical_issues_fixed >= 5`.

When the prerequisite gate passes, the runner MUST inspect the local paper
directory for available narrow build paths in this order: `tectonic`,
`latexmk`, and Makefile targets under `docs/arxiv-paper`. If a compile or
package path is available, it MUST run only that narrow command, record whether
`docs/arxiv-paper/main.pdf` exists after the attempt, create or verify a bundle
path, and set `arxiv_submitted` to false unless a local submission receipt
already exists.

If no local compile or package command can run, the artifact MUST be honest:
`status == "blocked"`, `pdf_compiled == false`, `missing_tool` names the exact
missing command names, and `honest_verdict` reports the local TeX-tooling
block. A completed artifact MUST record `status == "complete"`, a non-empty
`bundle_path`, `pdf_compiled == true`, and `honest_verdict` from the bundle-v10
closed set.

### SCENARIO-PUBLISH-014: Bundle v10 Runs Only After Critical Fix Gate

**Given** Exp 1269 records at least five critical issues fixed
AND `docs/arxiv-paper/main.tex` exists
**When** the exp1270 arXiv bundle-v10 runner executes
**Then** it writes
`results/experiment_1270_arxiv_bundle_v10_gated.json`
AND records `run_date == "20260504"`
AND records `arxiv_submitted == false` when no local submission receipt exists
AND reports either a complete compiled/package artifact or a blocked artifact
with exact missing tool names.

### REQ-PUBLISH-015: Terminal arXiv v10 Hold/Receipt Artifact

The arXiv v10 hold/receipt runner MUST create
`results/experiment_1307_arxiv_v10_hold_receipt_v2.json` with
`status == "in_progress"` before doing any receipt evaluation. It MUST perform
only local repository file checks and MUST NOT attempt arXiv login, upload,
submission, or any other credentialed operation.

The terminal artifact MUST include:

- `status`
- `publication_state`
- `arxiv_receipt_present`
- `operator_hold_active`
- `credentialed_submission_attempted`
- `blocker`
- `honest_verdict`

The runner MUST set `credentialed_submission_attempted == false`. If a local
receipt is already recorded, it MUST set `arxiv_receipt_present == true` and
`publication_state == "submitted"`. If no local receipt is recorded, it MUST
set `operator_hold_active == true` when the operator publication hold is active,
otherwise it MUST record the exact local blocker.

### SCENARIO-PUBLISH-015: No Receipt Leaves Publication On Operator Hold

**Given** the operator publication hold is active in `ops/known-issues.md`
AND no checked-in local arXiv receipt exists
**When** the exp1307 hold/receipt runner executes
**Then** it writes a complete artifact with
`arxiv_receipt_present == false`
AND `operator_hold_active == true`
AND `credentialed_submission_attempted == false`
AND `publication_state == "operator_hold"`.

### SCENARIO-PUBLISH-016: Local Receipt Makes State Submitted

**Given** a checked-in local arXiv submission receipt exists
**When** the exp1307 hold/receipt runner executes
**Then** it writes a complete artifact with
`arxiv_receipt_present == true`
AND `publication_state == "submitted"`
AND `credentialed_submission_attempted == false`.

### REQ-PUBLISH-016: Publication-Hold Related-Work Delta Artifact

The Exp 1321 publication-hold related-work runner MUST create
`results/experiment_1321_publication_hold_related_work_delta_v11.json` with
`status == "in_progress"` before reading the hold evidence or literature
references. It MUST inspect only local repository files and MUST NOT attempt
arXiv login, upload, submission, or any other credentialed operation.

The terminal artifact MUST read
`results/experiment_1307_arxiv_v10_hold_receipt_v2.json`,
`ops/known-issues.md`, and `research-references.md`, identify the 2026-05-05
2025--2026 references that materially affect related work, and write a compact
related-work delta either into a local paper/related-work notes file or, when
no suitable file exists, directly into the artifact. The artifact MUST include:

- `status`
- `publication_state`
- `operator_hold_active`
- `credentialed_submission_attempted`
- `related_work_delta_written`
- `new_references_count`
- `honest_verdict`

The runner MUST set `credentialed_submission_attempted == false` and preserve
`publication_state == "operator_hold"` when the local Exp 1307 artifact and
known-issues evidence show the operator publication hold is still active.

### SCENARIO-PUBLISH-017: Related-Work Delta Does Not Lift Operator Hold

**Given** Exp 1307 records `publication_state == "operator_hold"`
AND `ops/known-issues.md` contains an active publication-hold section
AND `research-references.md` contains 2026-05-05 2025--2026 reference entries
**When** the Exp 1321 related-work-delta runner executes
**Then** it writes a complete artifact with
`publication_state == "operator_hold"`
AND `operator_hold_active == true`
AND `credentialed_submission_attempted == false`
AND `new_references_count > 0`
AND `related_work_delta_written == true`.

### REQ-PUBLISH-017: Publication Hold v16 Claim-Boundary Review

The Exp 1378 publication-hold v16 runner MUST create
`results/experiment_1378_publication_hold_v16_claim_boundary.json` with
`status == "in_progress"` before loading prior hold or milestone `.106`
evidence. The terminal artifact MUST read Exp 1362, Exp 1366, Exp 1369,
Exp 1370, Exp 1371, Exp 1372, and Exp 1374 source artifacts from the local
repository only. It MUST summarize whether the three primary hold blockers
have local evidence:

- certificate parsing recovered with `certificate_parse_rate >= 0.75`
- semantic validation, MCS repair, and scheduler triage all allow their local
  claims without false acceptance
- headline self-learning is allowed by fresh primary semantic-verifier evidence

The artifact MUST include:

- `status`
- `certificate_evidence_summary`
- `semantic_repair_evidence_summary`
- `kan_formal_evidence_summary`
- `self_learning_evidence_summary`
- `hold_blocker_resolved_certificate`
- `hold_blocker_resolved_semantic_repair`
- `hold_blocker_resolved_self_learning`
- `all_primary_blockers_resolved`
- `publication_hold_state`
- `paper_changes_needed_for_lift`
- `ebt_arm_claim_boundary`
- `dvi_ready`
- `external_dependency_claim_allowed`
- `honest_verdict`

The runner MUST set `publication_hold_state == "lift_recommended"` only when
all three primary blocker booleans are true. It MUST keep
`external_dependency_claim_allowed == false` unless external parity was locally
demonstrated by source artifacts.

### SCENARIO-PUBLISH-018: Full .106 Evidence Recommends Hold Lift Without External Parity

**Given** Exp 1366 reports `certificate_parse_rate == 1.0`,
`prefix_injection_supported == true`, and `headline_result_allowed == true`
AND Exp 1369 reports `validator_execution_pass_rate == 1.0` and
`semantic_validator_claim_allowed == true`
AND Exp 1370 reports `repair_claim_allowed == true`
AND Exp 1371 reports `triage_claim_allowed == true` and
`false_acceptance_rate == 0.0`
AND Exp 1374 reports `headline_result_allowed == true`,
`path_used == "primary_semantic_verified"`, and `dvi_ready == true`
**When** the Exp 1378 publication-hold v16 runner executes
**Then** it writes a complete artifact with
`all_primary_blockers_resolved == true`
AND `publication_hold_state == "lift_recommended"`
AND `external_dependency_claim_allowed == false`.

### REQ-PUBLISH-018: Audited arXiv Bundle v11 Submission Artifact

The Exp 1380 arXiv bundle-v11 runner MUST create
`results/experiment_1380_arxiv_bundle_v11_submission.json` with
`status == "in_progress"` before loading Exp 1379 or attempting any compile,
packaging, or submission command. It MUST read
`results/experiment_1379_paper_integrity_audit_v2.json` and refuse to compile
or bundle when `arxiv_submission_ready != true`.

When the Exp 1379 gate passes, the runner MUST find the audited paper file,
compile the LaTeX source with available local TeX tooling, package only the
active arXiv source files into `results/arxiv_bundle_v11.tar.gz`, and omit
unused or placeholder figures from the source archive. It MUST attempt
submission only when a local non-interactive arXiv upload command is available;
otherwise it MUST record manual submission steps and leave
`submission_attempted == false`.

The artifact MUST include:

- `status`
- `paper_file_found`
- `latex_compile_success`
- `bundle_file_path`
- `bundle_size_bytes`
- `figures_included`
- `submission_attempted`
- `submission_result`
- `arxiv_id_if_submitted`
- `remaining_blocker`
- `honest_verdict`

### SCENARIO-PUBLISH-019: Exp 1380 Produces a Submission-Ready Archive After Exp 1379

**Given** Exp 1379 records `arxiv_submission_ready == true`
AND `paper_file_path` points to an existing `main.tex`
AND local TeX tooling compiles the paper without errors
**When** the Exp 1380 bundle-v11 runner executes
**Then** it writes `results/arxiv_bundle_v11.tar.gz`
AND the bundle contains `main.tex`, `carnot.bib`, and every active figure
referenced by the paper except unused placeholder figures
AND the deliverable records `latex_compile_success == true`,
`bundle_size_bytes > 0`, submission status, and a complete honest verdict.

### REQ-PUBLISH-019: arXiv SWORD API Submission Or Manual Checklist

The Exp 1390 arXiv submission runner MUST create
`results/experiment_1390_arxiv_submission_sword_api.json` with
`status == "in_progress"` before checking credentials or attempting any network
submission. It MUST verify that `results/arxiv_bundle_v11.tar.gz` exists and is
non-empty, and it MUST use the audited paper metadata from
`docs/arxiv-paper/main.tex` plus the required submission metadata:

- title
- abstract
- author `Ian Blenke <ian@blenke.com>`
- primary category `cs.LG`
- license `CC-BY-4.0`

When non-interactive arXiv SWORD credentials are available, the runner MUST use
Python's `requests` library to POST the bundle and metadata to
`https://arxiv.org/sword/deposit`, record `submission_attempted == true`, and
extract any returned arXiv identifier into `arxiv_id_if_submitted`.

When credentials are unavailable, the runner MUST NOT fabricate a submission.
It MUST write `docs/arxiv-manual-submission-checklist.md` with the exact upload
URL, ready bundle path, pre-filled metadata, and step-by-step manual upload
instructions that let an operator complete submission quickly in the browser.

The artifact MUST include:

- `status`
- `bundle_path`
- `submission_attempted`
- `submission_method`
- `arxiv_id_if_submitted`
- `submission_result`
- `manual_checklist_generated`
- `manual_checklist_path`
- `honest_verdict`

### SCENARIO-PUBLISH-020: Missing SWORD Credentials Produce Manual Checklist

**Given** `results/arxiv_bundle_v11.tar.gz` exists and is non-empty
AND no arXiv SWORD credentials are configured
**When** the Exp 1390 runner executes
**Then** it writes `docs/arxiv-manual-submission-checklist.md`
AND the deliverable records `submission_attempted == false`,
`submission_result == "manual_checklist_generated"`,
`manual_checklist_generated == true`, and a complete honest verdict.

### SCENARIO-PUBLISH-021: SWORD Credentials Trigger API Submission Attempt

**Given** `results/arxiv_bundle_v11.tar.gz` exists and is non-empty
AND non-interactive arXiv SWORD credentials are configured
**When** the Exp 1390 runner executes
**Then** it POSTs the bundle and metadata to
`https://arxiv.org/sword/deposit`, records `submission_attempted == true`, and
stores any returned arXiv identifier in `arxiv_id_if_submitted`.

### REQ-PUBLISH-020: arXiv Operator Action Sheet

The Exp 1412 arXiv operator-action-sheet runner MUST create
`results/experiment_1412_arxiv_operator_action_sheet_v3.json` with
`status == "in_progress"` before validating the ready bundle or writing the
operator sheet. It MUST verify that `results/arxiv_bundle_v11.tar.gz` exists
and is non-empty, read `docs/arxiv-manual-submission-checklist.md`, and write a
terse browser-only action sheet at `docs/arxiv-submit-now.md`.

The action sheet MUST include the exact upload URL `https://arxiv.org/submit`,
the source bundle path, primary category `cs.LG`, license `CC-BY-4.0`, title,
author, and a one-line pointer to the checklist abstract. The runner MUST NOT
attempt a SWORD/API submission and MUST record
`credentialed_submission_attempted == false`.

The artifact MUST include:

- `status`
- `bundle_path`
- `bundle_exists`
- `bundle_size_bytes`
- `manual_checklist_path`
- `operator_action_sheet_path`
- `submission_ready_for_operator`
- `credentialed_submission_attempted`
- `honest_verdict`

### SCENARIO-PUBLISH-022: Ready Bundle Produces Browser Action Sheet

**Given** `results/arxiv_bundle_v11.tar.gz` exists and is non-empty
AND `docs/arxiv-manual-submission-checklist.md` contains the upload URL and
pre-filled metadata
**When** the Exp 1412 runner executes
**Then** it writes `docs/arxiv-submit-now.md`
AND the deliverable records `submission_ready_for_operator == true`,
`credentialed_submission_attempted == false`, the bundle size, and a complete
honest verdict.

### REQ-PUBLISH-021: Paper v6 Anchored-Claims Narrowing

The Exp 1462 paper-v6 anchored-claims narrowing workflow MUST create
`results/experiment_1462_paper_v6_anchored_claims_narrowing.json` with
`status == "in_progress"` before reading paper sources or changing paper text.
It MUST inspect the local publication hold evidence, Exp 1454 signal/noise
summary, Exp 1455 priority audit, Exp 1459 self-learning decision, Exp 1460
hardware narrowing decision, and Exp 1461 comparator cite/retire audit.

The workflow MUST locate the active paper-v6 source when present. If a source
exists, it MUST add or update an "Anchored Claims" section containing between
three and five explicit claims. Each anchored claim MUST include empirical
artifact paths and theoretical support. Unsupported territory MUST be preserved
as appendix or future-work notes rather than deleted. If no paper source exists,
the workflow MUST still write the claim matrix and set `paper_updated == false`.

The terminal artifact MUST include:

- `status`
- `paper_source_path`
- `anchored_claim_count`
- `anchored_claims`
- `unanchored_claims_moved`
- `claim_matrix_path`
- `paper_updated`
- `arxiv_submission_triggered`
- `honest_verdict`

The workflow MUST set `arxiv_submission_triggered == false` and MUST NOT run any
publish, bundle, upload, or submit command.

### SCENARIO-PUBLISH-023: Exp 1462 Narrows Paper v6 Claims Without Submission

**Given** publication is on hold, the scope-reduction directive is active, and
the local Exp 1454, 1455, 1459, 1460, and 1461 evidence artifacts exist
**When** the Exp 1462 anchored-claims narrowing workflow runs for run date
`20260507`
**Then** it writes the in-progress artifact first, writes a claim matrix,
records between three and five anchored claims, records empirical artifacts and
theoretical support for each anchored claim, moves unsupported territory to
appendix or future-work notes, leaves `arxiv_submission_triggered == false`, and
reports a complete honest verdict.

### REQ-PUBLISH-022: Paper v6 Section 3 Sampler Draft Resumption

The Exp 1576 paper-v6 Section 3 sampler-draft workflow MUST create
`results/experiment_1576_paper_v6_section_3_sampler_draft_resumed.json` with
`status == "in_progress"` before drafting the sampler/verifier subsection. It
MUST write `docs/research-notes/paper-v6-section-3-sampler-draft.md` as a
focused paper-v6 Section 3 update, not a wholesale paper rewrite.

The draft MUST include subsections for:

- THRML vendored sampler
- candidate warm-start
- Soft-Gibbs Residual
- kinetic-security caveat
- SpecAnn rejection
- BRAIN expressivity vs training-dynamics open question

The draft MUST anchor its claims to the local `.120`/resumed evidence artifacts
for Exp 1561, Exp 1562, Exp 1563 or the available SpecAnn rejection source,
Exp 1564, Exp 1565, Exp 1566, Exp 1570, and Exp 1571. It MUST explicitly avoid
claiming THRML security parity or Extropic hardware execution. It MUST include
a short paper-v6 integration checklist with exact insertion points for the
active paper source and record whether the draft is ready for Exp 1579 OT
framework integration.

The terminal artifact MUST include:

- `status`
- `draft_path`
- `paper_v6_sampler_section_draft_ready`
- `kinetic_security_caveat_included`
- `brain_training_dynamics_open_question_included`
- `no_hardware_execution_claim`
- `honest_verdict`

### SCENARIO-PUBLISH-024: Exp 1576 Draft Is Honest And OT-Ready

**Given** the Exp 1561, Exp 1562, Exp 1564, Exp 1565, Exp 1566, Exp 1570, and
Exp 1571 evidence artifacts exist locally
AND the requested Exp 1563 SpecAnn rejection artifact may be absent but the
draft cites the available SpecAnn rejection source explicitly
**When** the Exp 1576 sampler-draft workflow completes
**Then** it writes the Section 3 sampler/verifier draft with all required
subsections, includes the kinetic-security caveat and BRAIN training-dynamics
open question, avoids THRML security-parity and hardware-execution claims,
records exact active-paper insertion points, sets
`paper_v6_sampler_section_draft_ready == true`, and reports a complete honest
verdict that states whether Exp 1579 OT framework integration can consume the
draft.

### REQ-PUBLISH-023: Paper v6 OT Verification Framework Adoption

The Exp 1579 paper-v6 OT verification framework adoption workflow MUST create
`results/experiment_1579_iclr26_ot_verification_framework_paper_v6_adoption.json`
with `status == "in_progress"` before writing the adoption note. It MUST write
`docs/research-notes/paper-v6-ot-verification-framework-adoption.md` as a
focused paper-v6 vocabulary and claim-boundary note using the ICLR 2026
framework from arXiv:2510.18982.

The note MUST map coverage, verifier ROC, and sampling sub-optimality onto
Carnot's verifier cascade without claiming a new Carnot theorem. It MUST include
a conflict ledger that records every paper-v6 claim that must be softened
because finite-K sampling, verifier ROC, or out-of-distribution verifier
calibration does not support the stronger reading. If
`docs/papers/paper-v6/main.tex` is absent or lacks a clearly isolated related
work or Section 3 insertion point, the workflow MUST not patch the paper source;
instead it MUST include a patch plan in the note.

The terminal artifact MUST include:

- `status`
- `adoption_note_path`
- `ot_framework_adopted`
- `claim_conflict_count`
- `paper_patch_applied`
- `no_publication_trigger`
- `honest_verdict`

`no_publication_trigger` MUST remain true, and the workflow MUST not trigger any
arXiv submission, release, or push action.

### SCENARIO-PUBLISH-025: Exp 1579 Adopts OT Vocabulary Without Overclaiming

**Given** the Exp 1576 sampler/verifier draft exists and arXiv:2510.18982 has
been reviewed
**When** the Exp 1579 OT adoption workflow completes
**Then** it writes the adoption note with explicit coverage, ROC, and
sub-optimality mappings, records at least four claim conflicts, records whether
a paper patch was applied, keeps `no_publication_trigger == true`, and reports
a complete honest verdict that preserves the finite-K and verifier-ROC
boundaries.

## Implementation Status

| Requirement | Status | Notes |
|-------------|--------|-------|
| REQ-PUBLISH-001 | Implemented | Exp 700 provenance audit |
| REQ-PUBLISH-002 | Implemented | Exp 700 model card draft |
| REQ-PUBLISH-003 | Implemented | Exp 700 gate logic |
| REQ-PUBLISH-004 | Implemented | Exp 1135 position paper findings update |
| REQ-PUBLISH-005 | Proposed | Exp 1153 final arXiv v4 bundle artifact |
| REQ-PUBLISH-006 | Proposed | Exp 1167 Phase 4 Section 7 revision artifact |
| REQ-PUBLISH-007 | Implemented | Exp 1181 paper v5 high-severity fixes ISSUE-6..10 |
| REQ-PUBLISH-008 | Proposed | Exp 1182 paper v5 medium/low fixes ISSUE-11..18 |
| REQ-PUBLISH-009 | Proposed | Exp 1182 paper numerical-claim audit script |
| REQ-PUBLISH-010 | Proposed | Exp 1183 paper v5 recompile and arXiv bundle v6 gate artifact |
| REQ-PUBLISH-011 | Proposed | Exp 1245 fig3 measured-only FPGA latency fix |
| REQ-PUBLISH-012 | Proposed | Exp 1257 paper v6 five critical integrity fixes |
| REQ-PUBLISH-013 | Proposed | Exp 1269 paper v6 critical fixes v2 terminal artifact |
| REQ-PUBLISH-014 | Proposed | Exp 1270 gated arXiv bundle v10 artifact |
| REQ-PUBLISH-015 | Proposed | Exp 1307 arXiv v10 hold/receipt terminal artifact |
| REQ-PUBLISH-016 | Proposed | Exp 1321 publication-hold related-work delta artifact |
| REQ-PUBLISH-017 | Implemented | Exp 1378 publication-hold v16 claim-boundary review |
| REQ-PUBLISH-018 | Implemented | Exp 1380 audited arXiv bundle v11 submission artifact |
| REQ-PUBLISH-019 | Implemented | Exp 1390 arXiv SWORD API submission or manual checklist |
| REQ-PUBLISH-020 | Implemented | Exp 1412 arXiv operator action sheet |
| REQ-PUBLISH-021 | Proposed | Exp 1462 paper-v6 anchored-claims narrowing artifact |
| REQ-PUBLISH-022 | Proposed | Exp 1576 paper-v6 Section 3 sampler/verifier draft |
| REQ-PUBLISH-023 | Proposed | Exp 1579 ICLR 2026 OT verification framework adoption |
