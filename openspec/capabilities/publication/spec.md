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
