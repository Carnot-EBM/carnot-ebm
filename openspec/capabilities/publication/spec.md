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

### REQ-PUBLISH-007: G-Gate Status Synthesis

The G-gate synthesis script MUST aggregate the findings of `publication_gate.py` with the verifier cross-domain synthesis. The script MUST evaluate `publication_gate.py --json` to get the G1-G4 status, `paper_ready`, and `unmet_gates`. It MUST also synthesize the `verifier_generalization_scope` from the verifier cross-domain synthesis experiment (e.g., experiment 3576). The script MUST record `p01_status` as `honest-negative`. It MUST NOT cite any artifact that is flagged adversarial (e.g., experiment 3574). It MUST record `cited_upstream_artifacts` as a list of valid experiment artifacts. The resulting JSON MUST have an `honest_verdict` with the `complete: g_gate_synthesis_v329_paper_ready_{bool}_verifier_generalization_{scope}` prefix.

### REQ-PUBLISH-3664: v335 Real-NLI Facts Capstone

The Exp 3664 v335 capstone script MUST aggregate the stable G1-G4 publication
gate with Exp 3654 through Exp 3660 artifacts and write
`results/experiment_3664_capstone_and_g_gate_v335.json`. The artifact MUST
record the corrected math/code/facts generalization table, where math is the
frozen FoVer 0.9131 headline, code is hardened only when Exp 3658 replicates on
a balanced second corpus, and facts is measured from Exp 3655 only when the
real-NLI grounding verifier exists, is leak-free, and is not an implausible
AUROC leak. Missing or gate-blocked facts fields MUST be reported as
`not_measured_real_nli` rather than inferred from `None`. The capstone MUST
exclude any `flagged_adversarial` upstream artifact from citations and safe
claims, preserve P0.1 as `honest-negative`, and emit a terminal verdict with
the `complete: capstone_v335_facts_{generalize_or_domain_bound}_with_real_nli_verifier_value_{scope}_paper_ready_true`
prefix.

### REQ-PUBLISH-3677: v336 Dependency-Aware/Facts-Real Capstone

The Exp 3677 v336 capstone script MUST aggregate `publication_gate.py --json`
with Exp 3667 through Exp 3673 artifacts and write
`results/experiment_3677_capstone_and_g_gate_v336.json`. The workflow MUST be
aggregation-only: it SHALL read upstream artifacts, run the artifact summarizer,
and SHALL NOT perform live inference or modify `scripts/research_conductor.py`.

The artifact MUST record whether Exp 3667/3668 form a
`dependency_aware_headline_candidate_status` of
`clean_and_heldout_validated`, `clean_but_overfit`, `no_significant_gain`, or
`flagged_still`. Exp 3667 may be cited only when
`adversarial_verify_clean == true` and the artifact is not
`flagged_adversarial`; Exp 3668 skipped/missing fields MUST be reported as
`not_measured` logic rather than inferred from `None`. The frozen FoVer headline
MUST remain `0.9131`; any dependency-aware win MUST be described only as a
headline-advancement candidate pending re-freeze and re-reproduction.

The artifact MUST record the Exp 3670 facts real-benchmark verdict as one of
`generalizes_real`, `auroc_parity_with_catch_value`,
`domain_bound_real_earned`, or `not_measured`; any grounding AUROC greater than
or equal to `0.99` MUST be treated as a leak unless `grounding_leak_free == true`
is proven. It MUST record the Exp 3671 shipped-detector boolean, Exp 3672
SC-weak selection direction, Exp 3673 FR-11 v10 result, `p01_status` as
`honest-negative`, and `trained_judge_ood_retired == true`.

The workflow MUST exclude `flagged_adversarial` upstream artifacts from
`cited_upstream_artifacts`, record G1-G4 and `paper_ready` directly from the
publication gate, include narrowing-clean `paper_v6_safe_claims` and
`paper_v6_forbidden_claims`, and emit the terminal verdict prefix
`complete: capstone_v336_dependency_aware_<status>_facts_real_<verdict>_detector_shipped_<bool>_paper_ready_true`.

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

### SCENARIO-PUBLISH-007: G-Gate Status Synthesis Complete

**Given** publication gate is ready and verifier cross-domain synthesis artifacts are available
**When** the G-gate status synthesis script runs
**Then** the script executes `publication_gate.py`, excludes flagged adversarial artifacts, and emits a correctly-formatted JSON with all required fields.

### SCENARIO-PUBLISH-3664: Real-NLI Facts Capstone Complete

**Given** publication gate is ready and Exp 3654 through Exp 3660 artifacts are available
**When** the Exp 3664 v335 capstone script runs
**Then** the script records whether facts generalize under the real-NLI verifier,
excludes flagged adversarial upstream artifacts, preserves P0.1 as
honest-negative, and emits a paper-ready artifact with all required fields.

### SCENARIO-PUBLISH-3677: v336 Capstone Complete

**Given** publication gate is ready and clean Exp 3667 through Exp 3673
artifacts are available
**When** the Exp 3677 v336 capstone script runs
**Then** it emits the required paper-ready artifact, excludes
`flagged_adversarial` citations, classifies skipped upstream work as
`not_measured`, preserves the frozen FoVer headline and P0.1 honest negative,
retires the trained-judge-OOD hypothesis, and records the dependency-aware win
only as a future headline-advancement candidate pending re-freeze and
re-reproduction.

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

### REQ-PUBLISH-024: Phase 1 Software Ship Readiness Ledger

The Exp 1582 Phase 1 ship-readiness workflow MUST create
`results/experiment_1582_phase1_ship_readiness_ledger.json` with
`status == "in_progress"` before inspecting release metadata. It MUST inspect
the software-only Phase 1 ship gates that the operator decoupled from paper,
arXiv, and hardware validation:

- PyPI/package readiness for `pip install .`, console script exposure,
  versioning, Apache-2.0 license metadata, and package data.
- HuggingFace/model-card readiness, including missing local model or dataset
  artifacts needed by the documented mirror plan.
- Second-channel mirror readiness, preferring content-addressed IPFS CIDs.
- MCP and CLI external-integrator quick-start completeness.
- Independent reproducer readiness using a fresh venv or CI path, with any
  safe local smoke command recorded when run.

The workflow MUST NOT publish packages, upload credentials, push releases, or
modify `scripts/research_conductor.py`. The terminal artifact MUST include:

- `status`
- `phase1_ship_readiness_ledger_ready`
- `pypi_package_ready`
- `hf_mirror_ready`
- `second_mirror_ready`
- `mcp_cli_docs_ready`
- `independent_reproducer_path_ready`
- `safe_local_smoke_ran`
- `blocking_items_count`
- `ledger_path`
- `honest_verdict`

The workflow MUST write `ops/phase1_ship_readiness.md` with a pass/fail
checklist, exact blockers, and concrete commands that can be run without
publishing or uploading credentials unless explicitly marked as operator-only.

### SCENARIO-PUBLISH-026: Phase 1 Ledger Blocks Ship Until Software Gates Pass

**Given** the repository contains local package metadata, model-card references,
mirror records, MCP/CLI docs, and Phase 1 reproducer evidence
**When** the Exp 1582 readiness workflow completes
**Then** it writes the markdown ledger and terminal JSON artifact
AND the artifact contains every REQ-PUBLISH-024 required field
AND `blocking_items_count` equals the number of unresolved ship blockers
AND `honest_verdict` is `phase1_software_ship_ready` only when all five
software gates are ready.

### REQ-PUBLISH-025: PyPI Publish Dry Run Artifact

The PyPI publish dry run MUST produce an artifact at `results/experiment_2103_pypi_publish_dry_run.json` that conforms to the `carnot.phase1_pypi_publish_dry_run.v1` schema. The artifact MUST record the sdist and wheel sizes, the result of `twine check`, and an honest verdict indicating readiness without actually publishing to PyPI.

### REQ-PUBLISH-026: PyPI Workflow Status Re-check

The PyPI workflow status re-check MUST produce an artifact at `results/experiment_1987_pypi_status_recheck.json` conforming to `carnot.pypi_workflow_status_recheck.v1`. It MUST report whether the workflow run has transitioned from 'waiting' to 'succeeded' or 'failed', and verify external install if 'succeeded'.

### REQ-PUBLISH-029: Phase 1 Recovery Audit

The Phase 1 Recovery task MUST produce an artifact at `results/experiment_1989_phase1_recovery.json` that conforms to the `carnot.phase1_recovery.v1` schema. The artifact MUST check the shipping status of MCP/CLI Integrator Docs (exp1981) and Independent Reproducer (exp1982), reporting their individual status and passing its acceptance gate if both have successfully shipped.

### SCENARIO-PUBLISH-027: PyPI Publish Dry Run Passes

**Given** the package build is deterministic and produces valid metadata
**When** the PyPI publish dry run executes
**Then** it writes a complete artifact with `twine_check_passed == true`,
`acceptance_gate_passed == true`, and an `honest_verdict` indicating the dry run was successful.

### SCENARIO-PUBLISH-029: Phase 1 Recovery succeeds

**Given** the exp1981 and exp1982 artifacts both exist and passed their acceptance gates
**When** the Phase 1 Recovery audit executes
**Then** it writes a complete artifact with `acceptance_gate_passed == true` and an honest verdict indicating success.

### REQ-PUBLISH-030: Exp 2553 arXiv Package v3 Readiness Artifact

The Exp 2553 arXiv package v3 runner MUST produce
`results/experiment_2553_arxiv_package_v3.json` without attempting any
credentialed arXiv submission. The runner MUST verify
`docs/arxiv-paper/main.tex` exists, detect local TeX tooling in the order
`tectonic` then `pdflatex`, compile `main.tex` with the detected tool, count
abstract words from the LaTeX abstract environment, and load
`results/experiment_2544_phase4_option_b.json` to compute the redefined Gate 3
as `phase4_validated_any OR phase4_honest_negative_documented`.

The artifact MUST include `honest_verdict`, `arxiv_ready`,
`submission_package_ready`, `gate_3_phase4_resolved`,
`latex_compile_success`, `abstract_word_count`,
`operator_submission_checklist`, `preconditions_checked`, and `duration_s`.
`arxiv_ready` MUST be true only when the four publication gates pass, the
LaTeX compile succeeds, and the abstract word count is at most 250.

### SCENARIO-PUBLISH-030: Honest Negative Resolves Gate 3 For Operator Submission

**Given** the paper source exists, local TeX tooling compiles it, the abstract
contains at most 250 words, and Exp 2544 records
`phase4_honest_negative_documented == true`
**When** the Exp 2553 arXiv package v3 runner executes
**Then** it writes `results/experiment_2553_arxiv_package_v3.json`
AND `gate_3_phase4_resolved == true`
AND `arxiv_ready == true`
AND `submission_package_ready == true`
AND `operator_submission_checklist` contains browser-only operator actions.

### REQ-PUBLISH-031: Exp 2554 Milestone .245 Capstone Synthesis Artifact

The Exp 2554 capstone runner MUST produce
`results/experiment_2554_capstone_v245.json` by reading the 11 .245 task
artifacts (exp2543 through exp2553) and synthesizing a single coherent
milestone report. The runner MUST NOT attempt any external submission, MUST
NOT modify any source paper or paper bibliography, and MUST be a pure-function
synthesis with no network access.

The artifact MUST include `honest_verdict`, `n_experiments_completed`,
`best_245_auroc`, `auroc_adversarially_verified`, `phase4_final_status`,
`arxiv_ready`, `operator_recommendation`, `hardware_terminal_states`,
`gatemate_status`, `kv260_status`, `jepa_discrimination_improved`,
`top_3_successes`, `top_3_gaps_for_246`, `external_baselines`,
`process_flags`, `preconditions_checked`, and `duration_s`.

`arxiv_ready` MUST be read directly from `exp2553.arxiv_ready` and MUST NOT
be softened or inferred. `phase4_final_status` MUST be
`retired_negative_option_b` when `exp2544.phase4_section_expanded == true`
AND `exp2544.phase4_honest_negative_documented == true`; otherwise it MUST
be `blocked_precondition`. `best_245_auroc` MUST be the cite-safe headline:
ensemble v7b's 5-seed mean when adversarially clean, adaptive conformal's
mean only when it is both higher AND adversarially clean, otherwise the
carry-forward `0.9750` baseline from exp2498.

### SCENARIO-PUBLISH-031: All Eleven .245 Artifacts Land And ArXiv Is Ready

**Given** results/experiment_2543_archive.json through
results/experiment_2553_arxiv_package_v3.json all exist on disk
AND exp2544 records `phase4_section_expanded == true` and
`phase4_honest_negative_documented == true`
AND exp2546 records `ensemble_v7b_auroc >= 0.975` across at least 3 seeds
without `flagged_adversarial`
AND exp2553 records `arxiv_ready == true`
**When** the Exp 2554 capstone runner executes
**Then** it writes `results/experiment_2554_capstone_v245.json`
AND `honest_verdict` starts with `complete:`
AND `arxiv_ready == true`
AND `phase4_final_status == "retired_negative_option_b"`
AND `operator_recommendation == "submit_now"`
AND `best_245_auroc` is the ensemble v7b headline number
AND `auroc_adversarially_verified == true`.

### REQ-PUBLISH-032: Exp 2826 Milestone .267 Multi-Corpus Capstone Synthesis Artifact

The Exp 2826 capstone runner MUST produce
`results/experiment_2826_capstone_v267.json` by reading the prior milestone
capstone (exp2818) and the seven .267 task artifacts (exp2819 through exp2825)
and synthesizing a single coherent milestone report.  The runner MUST NOT
attempt any external submission or model inference, and MUST be a pure-function
synthesis that degrades gracefully when upstream artifacts are missing or
adversarially flagged.

The artifact MUST include `honest_verdict` (terminal-prefix per CLAUDE.md
Verdict Terminal-Prefix Discipline), `corpora_headline_table`,
`fover_shape_overfit_confirmed`, `self_learning_contribution_confirmed`,
`architecture_transfer_verifiers`, `memory_augmented_verifiers`,
`corpus_specific_verifiers`, `low_signal_verifiers`,
`recommended_headline_repin`, `gaps_for_268`, `acceptance_criteria_met`,
and `duration_s`.

`fover_shape_overfit_confirmed` MUST be `true` only when a non-FoVer
architecture-only AUROC AND a FoVer architecture-only AUROC are both present
in unflagged upstream artifacts AND the FoVer value exceeds every non-FoVer
value by more than 0.10.  A missing exp2820 (FoVer leakage isolation) artifact
MUST yield `false`.

`self_learning_contribution_confirmed` MUST be `true` only when
`exp2820.learning_contribution > 0.05` from an unflagged artifact.  A missing
exp2820 MUST yield `false`.

`recommended_headline_repin` MUST be `false` whenever fewer than two
adversarially-clean non-FoVer AUROC values are available.

### SCENARIO-PUBLISH-032: Nominal Run — All Seven .267 Artifacts Land Clean

**Given** exp2819 through exp2825 all exist on disk without `flagged_adversarial`
AND exp2820 records `learning_contribution >= 0.06`
AND exp2820 records `condition_b_architecture_only_auroc_mean >= 0.93`
AND at least two non-FoVer architecture-only AUROCs exceed 0.83
**When** the Exp 2826 capstone runner executes
**Then** it writes `results/experiment_2826_capstone_v267.json`
AND `honest_verdict` starts with `complete:`
AND `fover_shape_overfit_confirmed == true`
AND `self_learning_contribution_confirmed == true`
AND `recommended_headline_repin == true`.

### SCENARIO-PUBLISH-032B: Degraded Run — Most Upstream Artifacts Missing or Flagged

**Given** exp2820, exp2821, and exp2822 are absent from disk
AND exp2823 has `flagged_adversarial == true`
**When** the Exp 2826 capstone runner executes
**Then** it writes `results/experiment_2826_capstone_v267.json`
AND `honest_verdict` starts with `complete:`
AND `fover_shape_overfit_confirmed == false`
AND `self_learning_contribution_confirmed == false`
AND `recommended_headline_repin == false`
AND `gaps_for_268` contains at least three items.

### REQ-PUBLISH-033: Exp 2833 Paper-v6 Multi-Corpus Table v2

The Exp 2833 paper-v6 table integrator MUST read the authoritative Exp 2828,
Exp 2829, Exp 2830, Exp 2831, and Exp 2832 artifacts and update
`docs/arxiv-paper/main.tex` Section 5 with a dual-condition multi-corpus table.
The table MUST include exactly the FoVer, MBPP, HumanEval, and TruthfulQA rows
with columns for `N`, architecture-only AUROC, production AUROC,
learning-delta, and peer baseline. Numeric AUROC fields MUST come from the
source artifacts; missing or blocked upstream measurements MUST remain visibly
unmeasured and MUST NOT be replaced with placeholders or carry-forward values.

The runner MUST update the Section 5.1 self-learning disclosure from Exp 2828's
actual `learning_contribution` and `per_verifier_learning_contribution` fields,
including an explicit unavailable disclosure when Exp 2828 did not measure
those fields. The runner MUST update the Section 5.2 per-verifier breakdown
from Exp 2832's verifier-category fields and matrix status.

The runner MUST compile `docs/arxiv-paper/main.tex` with `pdflatex`, MUST NOT
submit or upload to any external publication venue, and MUST write
`results/experiment_2833_paper_v6_multicorpus_table_v2.json` with
`honest_verdict`, `paper_v6_compile_success`, `corpora_in_table`,
`submission_package_ready`, `arxiv_ready_v7`, `duration_s`, and
operator-only submission guard fields.

### SCENARIO-PUBLISH-033: Measured Dual-Condition Artifacts Render Numerically

**Given** Exp 2828 through Exp 2831 contain numeric production and
architecture-only AUROC means and Exp 2832 contains verifier-category data
**When** the Exp 2833 paper-v6 table integrator runs
**Then** it replaces the old multi-corpus table with values derived from those
artifacts
AND writes an Exp 2833 artifact whose `honest_verdict` starts with `complete:`
AND whose `corpora_in_table` is exactly `["FoVer", "MBPP", "HumanEval",
"TruthfulQA"]`
AND whose submission guard fields record that no external submission was
attempted.

### SCENARIO-PUBLISH-033B: Blocked Upstream Artifacts Stay Unmeasured

**Given** one or more Exp 2828 through Exp 2831 artifacts have blocked verdicts
or null AUROC values
**When** the Exp 2833 paper-v6 table integrator runs
**Then** the corresponding paper table cells say the measurements are
unmeasured
AND the Exp 2833 artifact sets `submission_package_ready == false` and
`arxiv_ready_v7 == false`
AND no placeholder values such as `<peer>` or synthetic AUROC defaults are
inserted.

### REQ-PUBLISH-034: Exp 2841 Paper-v6 Multi-Corpus Table v3

The Exp 2841 paper-v6 table integrator MUST read the post-.268 dual-condition
corpus artifacts and the Exp 2840 verifier matrix, then update
`docs/arxiv-paper/main.tex` Section 5 with the authoritative v3
dual-condition table. The table MUST contain exactly the FoVer, MBPP,
HumanEval, and TruthfulQA rows with columns for `N`, architecture-only AUROC,
production AUROC, learning delta, and peer baseline. Numeric AUROC and
learning-delta cells MUST be derived from the loaded source artifacts; blocked,
missing, or null source measurements MUST remain visibly unmeasured and MUST
NOT be replaced by Exp 2825, Exp 2833, placeholders, or synthetic defaults.

The runner MUST update Section 5.1 from the FoVer source artifact's actual
`learning_contribution` and `per_verifier_learning_contribution` fields, and
MUST update Section 5.2 from the Exp 2840 verifier matrix categories and
matrix status. The runner MUST compile `docs/arxiv-paper/main.tex` with
`pdflatex`, MUST NOT submit or upload to any external publication venue, and
MUST write `results/experiment_2841_paper_v6_multicorpus_table_v3.json` with
`honest_verdict`, `paper_v6_compile_success`, `corpora_in_table`,
`submission_package_ready`, `arxiv_ready_v8`, and `duration_s`.

`submission_package_ready` and `arxiv_ready_v8` MUST be true only when the
paper compiles and every corpus row has both production and architecture-only
AUROC measured. The terminal artifact MUST record the source artifact paths and
operator-only publication guard fields.

### SCENARIO-PUBLISH-034: v3 Integrates Measured Post-.268 Rows

**Given** the post-.268 source artifacts contain numeric production and
architecture-only AUROC means for all four corpora and Exp 2840 contains a
verifier matrix
**When** the Exp 2841 paper-v6 table integrator runs
**Then** it replaces the existing Section 5 multi-corpus block with source
derived values
AND writes an Exp 2841 artifact whose `honest_verdict` starts with `complete:`
AND whose `corpora_in_table` is exactly `["FoVer", "MBPP", "HumanEval",
"TruthfulQA"]`
AND whose operator-only guard fields record that no external submission was
attempted.

### SCENARIO-PUBLISH-034B: v3 Blocks Readiness For Missing Real AUROCs

**Given** one or more post-.268 source artifacts have blocked verdicts or null
AUROC values
**When** the Exp 2841 paper-v6 table integrator runs
**Then** the corresponding paper table cells say the measurements are
unmeasured
AND the Exp 2841 artifact sets `submission_package_ready == false` and
`arxiv_ready_v8 == false`
AND no placeholder values such as `<peer>` or carry-forward Exp 2825/2833
values are inserted.

### REQ-PUBLISH-035: Exp 2903 Paper-v6 Hardware Validation Section v1

The Exp 2903 paper-v6 hardware-validation section builder MUST read
`results/experiment_2898_kv260_ising_sampler_hardware_latency_benchmark_v1.json`
and stage a standalone LaTeX subsection at
`docs/arxiv-paper/sections/hardware-validation-v1.tex` without adding an
`\input` to `docs/arxiv-paper/main.tex`. The subsection MUST include the KV260
board identity, bitstream SHA256, loaded overlay name, `n_spins`, per-seed
median and p95 latencies, and an explicit disclosure that no same-basis CPU
comparison or FPGA speedup claim is available yet.

The builder MUST fail closed unless the Exp 2898 artifact exists, its
`honest_verdict` starts with `complete:` or `success:`, all checked
preconditions are available, and no sample or acceptance gate reports failure.
It MUST write
`results/experiment_2903_paper_v6_hardware_validation_section_v1.json` with
`honest_verdict`, `inference_substrate="aggregation_from_upstream_artifacts"`,
`latex_snippet_path`, `kv260_latency_cited_p50_us`,
`kv260_latency_cited_p95_us`, `bitstream_sha256_cited`,
`cited_upstream_artifacts`, and `duration_s`.

### SCENARIO-PUBLISH-035: KV260 Artifact Stages Hardware Validation Snippet

**Given** Exp 2898 completed successfully with available board preconditions,
zero failed samples, and three n=64 per-seed latency rows
**When** the Exp 2903 builder runs
**Then** it writes the standalone hardware-validation subsection with the
source-derived KV260 board, overlay, bitstream, spin-count, median, and p95
values
AND writes the Exp 2903 JSON artifact with a `complete:` verdict and upstream
artifact provenance
AND leaves `docs/arxiv-paper/main.tex` unchanged.

### SCENARIO-PUBLISH-035B: Failed Upstream Gate Blocks The Snippet

**Given** Exp 2898 is missing, non-terminal, has unavailable preconditions, or
contains failed samples or failed acceptance-gate status
**When** the Exp 2903 builder runs
**Then** it writes a blocked Exp 2903 JSON artifact
AND does not create or update the LaTeX snippet.

### REQ-PUBLISH-036: Exp 3451 FoVer G2 CI Workflow + Docker Clean-Room

The Exp 3451 G2 mechanism builder MUST ship the mechanism a non-operator
environment uses to close gate G2 (independent reproduction of the FoVer
headline) without itself claiming G2 is met.

It MUST author a GitHub Actions workflow at
`.github/workflows/reproduce-fover-headline.yml` that, on a clean
`ubuntu-latest` runner, checks out the repo, sets up Python, runs
`pip install -e .`, executes `scripts/reproduce_fover_headline.py`, and ASSERTS
condition-A mean AUROC in `[0.9027, 0.9235]` AND learning_contribution mean in
`[0.0125, 0.0245]` (exiting non-zero when either falls outside its published
CI). The workflow file is committed to the working tree only; the builder MUST
NOT push.

When Docker is available (`command -v docker` AND `docker info` succeed), the
builder MUST write a minimal Dockerfile on a clean Python base image (NOT the
operator's venv), build it, and run the harness inside the container, capturing
the containerized condition-A AUROC and learning_contribution. When Docker is
unavailable, the builder MUST fall back to a fresh-venv clean-room run rather
than failing the task — the CI workflow file is still the primary deliverable.

The builder MUST NOT push, MUST NOT set `g2_independent_reproducer=true` (only
an actual external/CI run by a non-operator may flip that), and MUST NOT modify
`scripts/research_conductor.py`. It MUST write
`results/experiment_3451_fover_g2_ci_workflow_and_docker_cleanroom_v1.json` with
`honest_verdict` (terminal `complete:` prefix),
`inference_substrate="verifier_ensemble_against_cached_candidates"`,
`ci_workflow_path`, `docker_available`, `g2_docker_cleanroom_reproduced`,
`condition_a_auroc_isolated`, `learning_contribution_isolated`, `g2_status`,
`g2_independent_reproducer` (always false), `reproducibility_checksum`,
`random_seed`, and `duration_s`.

### SCENARIO-PUBLISH-036: CI Workflow + Docker Clean-Room Both Ready

**Given** the reproducer harness and FoVer corpus are present and Docker is
available
**When** the Exp 3451 builder runs
**Then** it writes `.github/workflows/reproduce-fover-headline.yml` asserting
both published CIs
AND builds + runs the harness inside a clean-room Docker container that
recomputes condition-A AUROC in `[0.9027, 0.9235]` and learning_contribution in
`[0.0125, 0.0245]`
AND writes the Exp 3451 JSON artifact with a `complete:` verdict,
`g2_docker_cleanroom_reproduced == true`, and `g2_independent_reproducer == false`.

### SCENARIO-PUBLISH-036B: Docker Unavailable Falls Back To Fresh Venv

**Given** the harness and corpus are present but Docker is not available
**When** the Exp 3451 builder runs
**Then** it still authors the CI workflow file
AND falls back to a fresh-venv clean-room recompute
AND records `docker_available == false` with an honest `g2_status` of
`ci_ready_docker_unavailable` (when the fresh-venv clean-room reproduced) or a
`still_failing_<cause>` string otherwise
AND never sets `g2_independent_reproducer == true`.

### REQ-PUBLISH-037: Exp 3463 FoVer G2 CI Dry-Run + External-Reproducer Handoff

The Exp 3463 G2 dry-run builder MUST prove the
`.github/workflows/reproduce-fover-headline.yml` workflow (authored by Exp 3451)
runs green in a non-operator environment, and MUST assemble a one-command
external-reproducer handoff package — bringing gate G2 as close to
closeable-by-a-non-operator as autonomous work allows, without itself claiming
G2 is met.

It MUST STATICALLY VALIDATE the workflow: parse the YAML, and assert it pins a
Python version, installs `pip install -e .`, runs
`scripts/reproduce_fover_headline.py`, and (transitively, via that harness)
asserts condition-A mean AUROC in `[0.9027, 0.9235]` AND learning_contribution
mean in `[0.0125, 0.0245]` with a non-zero exit on failure. The static-validation
result is recorded in `ci_workflow_validated`.

It MUST DRY-RUN the workflow in an isolated runner. When `act` (nektos/act) is
available it MAY use it; otherwise it MUST execute the workflow's steps inside a
fresh Docker container (a clean Python base image, NOT the operator's venv) when
Docker is available, or a fresh venv when Docker is unavailable. The dry-run MUST
run the exact assertion command the workflow runs
(`python3 scripts/reproduce_fover_headline.py`) and capture its exit code; a
zero exit with both numbers in their published CIs sets `g2_ci_dryrun_green=true`.
The method used is recorded in `ci_dryrun_method`
(`act` | `stepwise_docker` | `stepwise_venv`).

It MUST write a NEW (non-operator-curated) external-reproducer handoff document
at `docs/g2-external-reproducer-handoff.md` containing a one-command
reproduction, the expected CI assertions, the corpus checksum, and the exact
steps a non-operator must take to close G2; and it MUST append (never delete) the
dry-run result to `ops/reproduction-runbook-fover-headline.md`.

The builder MUST NOT push, MUST NOT set `g2_independent_reproducer=true` (only an
actual external/CI run by a non-operator may flip that), and MUST NOT modify
`scripts/research_conductor.py`. It MUST write
`results/experiment_3463_fover_g2_ci_dryrun_and_external_handoff_v1.json` with
`honest_verdict` (terminal `complete:` prefix),
`inference_substrate="verifier_ensemble_against_cached_candidates"`,
`ci_workflow_validated`, `ci_dryrun_method`, `g2_ci_dryrun_green`,
`condition_a_auroc_isolated`, `learning_contribution_isolated`,
`g2_handoff_package_ready`, `handoff_doc_path`, `g2_status`,
`g2_independent_reproducer` (always false), `reproducibility_checksum`,
`random_seed`, and `duration_s`.

### SCENARIO-PUBLISH-037: CI Dry-Run Green And Handoff Package Ready

**Given** the workflow, reproducer harness, and FoVer corpus are present and
Docker is available
**When** the Exp 3463 dry-run builder runs
**Then** it statically validates the workflow asserts both published CIs
(`ci_workflow_validated == true`)
AND dry-runs the workflow's assertion command inside a clean-room container that
exits zero with condition-A AUROC in `[0.9027, 0.9235]` and learning_contribution
in `[0.0125, 0.0245]` (`g2_ci_dryrun_green == true`,
`ci_dryrun_method == "stepwise_docker"`)
AND writes `docs/g2-external-reproducer-handoff.md`
(`g2_handoff_package_ready == true`)
AND writes the Exp 3463 JSON artifact with a `complete:` verdict and
`g2_independent_reproducer == false`.

### SCENARIO-PUBLISH-037B: No Isolated Runner Falls Back To Validation Only

**Given** the workflow and harness are present but neither `act` nor Docker is
available
**When** the Exp 3463 dry-run builder runs
**Then** it still statically validates the workflow (`ci_workflow_validated`)
AND falls back to a `stepwise_venv` dry-run (or records the dry-run as
unavailable) rather than failing the task
AND still writes the handoff package (`g2_handoff_package_ready == true`)
AND reports an honest `g2_status` and never sets
`g2_independent_reproducer == true`.

### REQ-PUBLISH-038: Exp 3476 FoVer G2 Self-Contained External Reproduction Package

The Exp 3476 G2 package builder MUST assemble a single self-contained
reproduction package that a true stranger (no repo checkout, zero Carnot
knowledge) can run in one command to independently recompute the FoVer headline,
and MUST verify that package reproduces from a clean environment — without
itself claiming gate G2 is met.

It MUST build a package directory containing: the reproducer harness
(`scripts/reproduce_fover_headline.py`), the labeled corpus
(`data/fover_corpus.jsonl`), the FR-11 session-memory state files required for
condition A (production), the `carnot` package source, a pinned
`requirements.txt` (exact installed versions), the build metadata
(`pyproject.toml` + license/readme), a single one-command entry point
(`run.sh`) that installs the pinned dependencies, installs the package, runs the
harness, and propagates the harness non-zero exit on CI-band failure, and a
package `README` explaining the one command and the expected output. It MUST tar
the directory to `dist/g2-fover-repro.tar.gz` and compute its sha256. When an
IPFS node is available it MUST record the tarball CID (decentralization rule 3);
otherwise it MUST record `ipfs_available=false` and MUST NOT fail.

It MUST VERIFY the package by extracting the tarball into a fresh temporary
directory and running the one command inside a clean environment (a stock Docker
base image, NOT the operator's venv, when Docker is available; otherwise a fresh
venv), confirming condition-A mean AUROC lands in `[0.9027, 0.9235]` AND
learning_contribution mean in `[0.0125, 0.0245]`. The clean-environment method
is recorded in `clean_env_method` (`docker` | `fresh_venv`). When no clean
environment is available it MUST report `package_built_verification_unavailable`
rather than fail.

The builder MUST append (never delete) the package path + checksum to
`ops/reproduction-runbook-fover-headline.md`, MUST NOT edit operator-curated docs
(`ops/north-star.md`, `docs/index.html`, `README.md`), MUST NOT push, MUST NOT
trigger external CI, MUST NOT set `g2_independent_reproducer=true` (only an
actual external/CI run by a non-operator may flip that), and MUST NOT modify
`scripts/research_conductor.py`. It MUST write
`results/experiment_3476_fover_g2_self_contained_external_package_v1.json` with
`honest_verdict` (terminal `complete:` prefix),
`inference_substrate="verifier_ensemble_against_cached_candidates"`,
`package_path`, `package_sha256`, `package_cid`, `ipfs_available`,
`one_command_repro`, `clean_env_method`, `condition_a_auroc_isolated`,
`learning_contribution_isolated`, `package_verified_reproduces`, `g2_status`,
`g2_independent_reproducer` (always false), `operator_action_required`,
`reproducibility_checksum`, `random_seed`, and `duration_s`.

### SCENARIO-PUBLISH-038: Self-Contained Package Built And Verified

**Given** the reproducer harness and FoVer corpus are present and a clean
environment (Docker or fresh venv) is available
**When** the Exp 3476 package builder runs
**Then** it builds `dist/g2-fover-repro.tar.gz` with a non-null sha256
(`package_sha256 != null`)
AND the package includes a one-command entry point (`one_command_repro != null`)
AND extracting the tarball into a fresh temp dir and running the one command in a
clean environment recomputes condition-A AUROC in `[0.9027, 0.9235]` and
learning_contribution in `[0.0125, 0.0245]` (`package_verified_reproduces == true`)
AND writes the Exp 3476 JSON artifact with a `complete:` verdict and
`g2_independent_reproducer == false`.

### SCENARIO-PUBLISH-038B: No Clean Environment Reports Built-But-Unverified

**Given** the harness and corpus are present but neither Docker nor a usable
fresh venv is available
**When** the Exp 3476 package builder runs
**Then** it still builds and checksums the package (`package_sha256 != null`)
AND reports `g2_status == "package_built_verification_unavailable"` with
`package_verified_reproduces == false`
AND never sets `g2_independent_reproducer == true`.

### REQ-PUBLISH-039: Exp 3488 FoVer G2 Clean-Room Regression Verify + Lowest-Friction External Ask

The Exp 3488 G2 regression verifier MUST re-run the self-contained reproduction
package built by Exp 3476 from an environment isolated from the working repo,
confirm it still lands condition-A mean AUROC inside the published CI, re-verify
the package's content integrity, and prepare the lowest-friction external ask —
all WITHOUT itself claiming gate G2 is met.

It MUST check preconditions first: the package `dist/g2-fover-repro.tar.gz` MUST
be present (or rebuildable from Exp 3476's builder); if absent and not
rebuildable it MUST report `blocked_g2_package_unavailable`. An isolated runner
(a fresh `python -m venv`, or a temp directory unpacked from the tarball and run
outside the working repo) MUST be constructible; if not it MUST report
`blocked_fresh_env_unavailable`.

It MUST extract the on-disk tarball into a temporary directory OUTSIDE the
working repo, run the package's single reproduce command in an isolated
environment (a fresh venv when one can be built, otherwise the unpacked temp
directory run with the package source on the path — never the working repo's
import path), and capture the reproduced condition-A AUROC and its CI band. The
REGRESSION GATE is `package_auroc_within_ci`: the reproduced condition-A mean
AUROC MUST land inside `[0.9027, 0.9235]`. It MUST re-compute the tarball sha256
and compare it to the sha256 recorded by Exp 3476; `package_sha256_verified` is
true only on an exact match. It MUST record the IPFS CID when a node is
available (decentralization rule 3) and MUST NOT fail when IPFS is absent.

It MUST author the lowest-friction external ask as files in the working tree
ONLY (never pushed, never triggered): a public `workflow_dispatch` reproduction
workflow (`.github/workflows/fover-g2-repro.yml`) an external party can one-click
run, a one-paragraph reproducer invite (`docs/g2-reproducer-invite.md`), and an
operator checklist (`ops/g2-external-ask-operator-checklist.md`) whose terminal
step is the single operator action (clicking "Run workflow" / sending the
invite). It MUST append (never delete) the regression result to
`ops/reproduction-runbook-fover-headline.md`.

It MUST NOT edit operator-curated docs (`ops/north-star.md`, `docs/index.html`,
`README.md`), MUST NOT push, MUST NOT trigger external CI, MUST NOT set
`g2_met=true` / `g2_independent_reproducer=true` (only an actual external/CI run
by a non-operator may flip those — Operator-Only External Publication), and MUST
NOT modify `scripts/research_conductor.py`. It MUST write
`results/experiment_3488_fover_g2_clean_room_regression_verify_external_ask_v1.json`
with `honest_verdict` (terminal `complete:` prefix),
`inference_substrate="verifier_ensemble_against_cached_candidates"`,
`package_reproduced_auroc`, `package_auroc_within_ci`, `package_sha256_verified`,
`package_cid`, `external_ask_workflow_path`, `operator_checklist_path`,
`g2_met` (always false), `external_run_pending` (true), `random_seed`,
`reproducibility_checksum`, and `duration_s`.

### SCENARIO-PUBLISH-039: Package Regression-Clean And External Ask Ready

**Given** the Exp 3476 package `dist/g2-fover-repro.tar.gz` is present and an
isolated runner is constructible
**When** the Exp 3488 regression verifier runs
**Then** it re-runs the package from a temp directory outside the working repo
and the reproduced condition-A mean AUROC lands in `[0.9027, 0.9235]`
(`package_auroc_within_ci == true`)
AND the tarball sha256 matches Exp 3476's recorded checksum
(`package_sha256_verified == true`)
AND it writes the `workflow_dispatch` workflow, reproducer invite, and operator
checklist to the working tree (`external_ask_workflow_path != null` and
`operator_checklist_path != null`)
AND writes the Exp 3488 JSON artifact with a `complete:` verdict, `g2_met == false`,
and `external_run_pending == true`.

### SCENARIO-PUBLISH-039B: Missing Package Reports Blocked

**Given** the Exp 3476 package is absent and cannot be rebuilt
**When** the Exp 3488 regression verifier runs
**Then** it reports `honest_verdict` starting with
`complete: blocked_g2_package_unavailable`
AND never sets `g2_met == true`.

### REQ-PUBLISH-039C: Exp 3499 FoVer G2 Regression Drift Check (post-.322)

Same guarantees as REQ-PUBLISH-039 but run AFTER milestone .322's changes.  It
MUST re-verify the on-disk package produces AUROC within `[0.9027, 0.9235]` from
a fresh isolated environment, re-confirm the SHA256, refresh the external-ask
artifacts, and append a v2 section to the runbook.  It MUST write
`results/experiment_3499_fover_g2_regression_verify_external_ask_refresh_v2.json`
and MUST NOT set `g2_met=true`.

### SCENARIO-PUBLISH-039C: v2 Post-.322 Drift Check Produces v2 Artifact

**Given** the package `dist/g2-fover-repro.tar.gz` is present
**When** the Exp 3499 regression verifier (v2) runs
**Then** it produces
`results/experiment_3499_fover_g2_regression_verify_external_ask_refresh_v2.json`
with `package_auroc_within_ci == true`, `package_sha256_verified == true`,
`g2_met == false`, `external_run_pending == true`, and a `complete:` verdict.

### REQ-PUBLISH-040: Exp 3681 G2 Reproducer Prep For Operator Re-Freeze

The Exp 3681 prep runner MUST prepare, but not perform, a dependency-aware
FoVer headline re-freeze. It MUST first verify that Exp 3680 records
`dependency_aware_g1_rigor_confirmed == true` and that
`scripts/reproduce_fover_headline.py` is importable. If either precondition is
false, it MUST write
`results/experiment_3681_g2_reproducer_prep_operator_refreeze_package.json`
with the terminal verdict
`complete: blocked_g1_candidate_not_confirmed_or_reproducer_unavailable`.

When preconditions pass, the runner MUST use an ADDITIVE extension of
`scripts/reproduce_fover_headline.py` to recompute the dependency-aware
production AUROC and learning contribution from the cached FoVer corpus, assert
that those values land inside Exp 3680-derived confidence bounds, and confirm
that the unchanged frozen 0.9131 reproduction path still passes. The default
workflow-facing 0.9131 path MUST remain unchanged.

The artifact MUST draft, but not apply, the exact CI-workflow assertion bounds
an operator would set for the candidate re-freeze. It MUST include the
dependency-aware production AUROC CI from Exp 3680 and a learning-contribution
CI derived from Exp 3680 per-seed rows. It MUST include an ordered
`operator_checklist` whose steps are marked `OPERATOR-ACTION` for the north-star
Section 1 headline edit, the CI-workflow assertion update, and triggering the
independent reproducer run.

The runner MUST NOT edit `ops/north-star.md`, MUST NOT edit
`.github/workflows/reproduce-fover-headline.yml`, MUST NOT trigger GitHub
Actions, MUST NOT change the frozen 0.9131 publication-gate source, MUST NOT
change `paper_ready`, and MUST NOT modify `scripts/research_conductor.py`.

### SCENARIO-PUBLISH-040: Re-Freeze Package Ready But Headline Unchanged

**Given** Exp 3680 confirms the dependency-aware G1 candidate and the FoVer
headline reproducer imports
**When** the Exp 3681 prep runner executes
**Then** it writes
`results/experiment_3681_g2_reproducer_prep_operator_refreeze_package.json`
with `honest_verdict ==
"complete: refreeze_package_ready_for_operator_frozen_headline_unchanged"`
AND `reproducer_extended == true`
AND `existing_0_9131_reproduction_still_green == true`
AND `candidate_reproduction_asserts_in_ci == true`
AND `north_star_unmodified_assert == true`
AND `ci_workflow_unmodified_assert == true`
AND `frozen_headline_unchanged_assert == true`.

### SCENARIO-PUBLISH-040B: Missing Candidate Or Reproducer Blocks

**Given** Exp 3680 is missing, does not confirm the G1 candidate, or the FoVer
headline reproducer cannot be imported
**When** the Exp 3681 prep runner executes
**Then** it writes the same artifact path with `honest_verdict ==
"complete: blocked_g1_candidate_not_confirmed_or_reproducer_unavailable"`
AND does not edit `ops/north-star.md`
AND does not edit `.github/workflows/reproduce-fover-headline.yml`
AND does not trigger GitHub Actions.

### REQ-PUBLISH-3692: Clean Re-Emit Of Operator Re-Freeze Package

The Exp 3692 runner MUST re-emit the dependency-aware FoVer headline re-freeze
package after Exp 3681 was flagged by the adversarial verifier. It MUST verify
the same preconditions as Exp 3681: Exp 3680 records
`dependency_aware_g1_rigor_confirmed == true` and
`scripts/reproduce_fover_headline.py` is importable. If either precondition is
false, it MUST write
`results/experiment_3692_refreeze_package_clean_reemit.json` with the terminal
verdict `complete: blocked_g1_candidate_not_confirmed_or_reproducer_unavailable`.

When preconditions pass, the runner MUST preserve the additive
dependency-aware candidate path in `scripts/reproduce_fover_headline.py`,
confirm the unchanged frozen 0.9131 reproduction path still passes, confirm the
candidate reproduction lands inside Exp 3680-derived confidence bounds, and
draft the exact CI-workflow assertion bounds for operator use without applying
them. The artifact MUST use the bare `inference_substrate` value
`verifier_ensemble_against_cached_candidates`, MUST remove cached reproduction
marker fields that would cause verifier-scoring artifacts to look compute-bound,
MUST run `scripts/adversarial_verify.py` on the written artifact, and MUST set
`adversarial_verify_clean == true` only when no critical flag remains.

The runner MUST NOT edit `ops/north-star.md`, MUST NOT edit
`.github/workflows/reproduce-fover-headline.yml`, MUST NOT trigger GitHub
Actions, MUST NOT change the frozen 0.9131 publication-gate source, MUST NOT
change `paper_ready`, and MUST NOT modify `scripts/research_conductor.py`.

### SCENARIO-PUBLISH-3692: Clean Re-Freeze Package Ready But Headline Unchanged

**Given** Exp 3680 confirms the dependency-aware G1 candidate and the FoVer
headline reproducer imports
**When** the Exp 3692 clean re-emit runner executes
**Then** it writes
`results/experiment_3692_refreeze_package_clean_reemit.json`
with `honest_verdict ==
"complete: refreeze_package_reemitted_clean_for_operator_frozen_headline_unchanged"`
AND `adversarial_verify_clean == true`
AND `reproducer_extended == true`
AND `existing_0_9131_reproduction_still_green == true`
AND `candidate_reproduction_asserts_in_ci == true`
AND `north_star_unmodified_assert == true`
AND `ci_workflow_unmodified_assert == true`
AND `frozen_headline_unchanged_assert == true`.

### SCENARIO-PUBLISH-3692B: Missing Candidate Or Reproducer Blocks Clean Re-Emit

**Given** Exp 3680 is missing, does not confirm the G1 candidate, or the FoVer
headline reproducer cannot be imported
**When** the Exp 3692 clean re-emit runner executes
**Then** it writes the same artifact path with `honest_verdict ==
"complete: blocked_g1_candidate_not_confirmed_or_reproducer_unavailable"`
AND does not edit `ops/north-star.md`
AND does not edit `.github/workflows/reproduce-fover-headline.yml`
AND does not trigger GitHub Actions.

### REQ-PUBLISH-041: Exp 3689 v337 Dependency-Aware Capstone And G-Gate

The Exp 3689 capstone runner MUST aggregate `publication_gate.py --json` with
Exp 3680 through Exp 3685 artifacts and write
`results/experiment_3689_capstone_and_g_gate_v337.json`. The workflow MUST be
aggregation-only: it SHALL run the artifact summarizer for each upstream
artifact, SHALL NOT perform live inference, and SHALL NOT modify
`scripts/research_conductor.py`.

The artifact MUST preserve the frozen FoVer headline at `0.9131` and record
`frozen_headline_unchanged == true` when the publication gate still sources the
frozen FoVer artifact. A dependency-aware improvement from Exp 3680 MUST be
reported only as a `headline-advancement candidate with an operator-ready
re-freeze package pending operator action + CI re-reproduction`; it MUST NOT
silently replace the frozen headline. Exp 3680's number may be cited only when
`adversarial_verify_clean == true`, the artifact is not `flagged_adversarial`,
the acceptance gate passes, `leak_free == true`, and any AUROC greater than or
equal to `0.99` has explicit leak-free evidence.

The artifact MUST exclude any `flagged_adversarial` upstream artifact from
`cited_upstream_artifacts`. A flagged or missing Exp 3681 MUST block
`refreeze_package_status == "ready_for_operator"` even when its local verdict
string is celebratory. A flagged or missing Exp 3682 MUST record
`selection_gap_verdict == "not_measured"` rather than synthesizing a
fundamental-selection conclusion from missing or quarantined fields.

The artifact MUST record Exp 3683 as `recovered_math_and_code`,
`math_only_earned`, or `not_measured`; Exp 3684 as
`robust_beats_self_certainty`, `narrowed_collapses_vs_self_certainty`, or
`not_measured`; and Exp 3685 as the drift-aware FR-11 v11 result. It MUST
preserve `p01_status == "honest-negative"`, set
`facts_generalization_retired == true`, set
`trained_judge_ood_retired == true`, include narrowing-clean
`paper_v6_safe_claims` and `paper_v6_forbidden_claims`, and emit a terminal
verdict with the prefix
`complete: capstone_v337_dependency_aware_<status>_selection_<verdict>_detector_code_<status>_paper_ready_true_frozen_headline_unchanged`.

### SCENARIO-PUBLISH-041: v337 Capstone Preserves Gate And Excludes Flagged Artifacts

**Given** the publication gate reports G1-G4 pass and Exp 3680 through Exp 3685
artifacts are available
**When** the Exp 3689 capstone runner executes
**Then** it writes
`results/experiment_3689_capstone_and_g_gate_v337.json`
with `paper_ready == true`, `frozen_headline_unchanged == true`,
`p01_status == "honest-negative"`, facts-generalization and trained-judge-OOD
retired, flagged upstream artifacts excluded from citations, skipped or
flagged gated tasks recorded as `not_measured`, and the dependency-aware win
kept as a candidate pending operator action plus CI re-reproduction.


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
| REQ-PUBLISH-024 | Proposed | Exp 1582 Phase 1 software ship readiness ledger |
| REQ-PUBLISH-025 | Implemented | Exp 2103 PyPI publish dry run artifact |
| REQ-PUBLISH-030 | Implemented | Exp 2553 arXiv package v3 readiness artifact |
| REQ-PUBLISH-031 | Implemented | Exp 2554 milestone .245 capstone synthesis artifact |
| REQ-PUBLISH-032 | Implemented | Exp 2826 milestone .267 multi-corpus capstone synthesis artifact |
| REQ-PUBLISH-033 | Planned | Exp 2833 paper-v6 multi-corpus table v2 |
| REQ-PUBLISH-034 | Planned | Exp 2841 paper-v6 multi-corpus table v3 |
| REQ-PUBLISH-035 | Planned | Exp 2903 paper-v6 hardware-validation snippet |
| REQ-PUBLISH-036 | Implemented | Exp 3451 FoVer G2 CI workflow + Docker clean-room |
| REQ-PUBLISH-037 | Implemented | Exp 3463 FoVer G2 CI dry-run + external-reproducer handoff |
| REQ-PUBLISH-038 | Implemented | Exp 3476 FoVer G2 self-contained external reproduction package |
| REQ-PUBLISH-039 | Implemented | Exp 3488 FoVer G2 clean-room regression verify + lowest-friction external ask |
| REQ-PUBLISH-040 | Proposed | Exp 3681 G2 reproducer prep for operator re-freeze |
| REQ-PUBLISH-041 | Proposed | Exp 3689 v337 dependency-aware capstone and G-gate |

### REQ-PUBLISH-026: HuggingFace Publish Retry
The experiment 1750 huggingface retry runner MUST attempt to upload the smallest model in models/ with a no-emoji model card. If credentials pass, it MUST upload and record hf_upload_succeeded = True. If blocked, it MUST emit an honest verdict of "blocked_credentials".

### REQ-PUBLISH-027: Position Paper Nexus
The experiment 1913 architecture paper workflow MUST draft the position paper nexus and update the architecture document. It MUST produce an artifact at `results/experiment_1913_arch_paper.json` with the schema `carnot.arch_paper.v1` and record the status and an honest verdict indicating readiness.
