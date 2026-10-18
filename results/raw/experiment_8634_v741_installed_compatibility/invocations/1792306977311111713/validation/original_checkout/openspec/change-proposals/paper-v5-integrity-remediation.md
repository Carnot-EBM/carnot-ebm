# Paper v5 — Integrity Remediation

**Status:** DRAFT — operator-directed 2026-05-02 18:50Z
**Replaces:** the assumption that paper v4 is arXiv-ready
**Blocks:** arXiv submission until all 18 audit findings are resolved

## Why this exists

Operator audit + adversarial sub-agent review found 18 integrity issues
in `docs/arxiv-paper/main.tex` and the 7 figures. The paper-v4 artifact
(exp1167) reported `paper_ready_for_arxiv_hold_lift: true`, but the
exp1167 codex had not audited pre-existing figures or numerical claims
for integrity — only added the Phase-4 section structure and recompiled.

**5 critical issues each individually block submission per CLAUDE.md
"All headline results must have live GPU provenance":**

1. fig3 — fabricated CPU baseline + apples-to-oranges + buried caveat
2. main.tex KL=3.07 — software-proxy claim presented as FPGA-measured
3. fig7 / main.tex 15.6x — CPU baseline is hand-typed code constant
4. main.tex 76,130x HardNet++ — apples-to-oranges (CPU code vs LLM API)
5. main.tex exp1121 — hides SOSKANEnergyV3 collapse to 0.3333 AUROC

5 high-severity, 5 medium, 3 low. Full punch-list in `ops/known-issues.md`.

## Decision-making framework

Every claim in paper v5 must satisfy this 4-test:

```
1. SOURCE-ARTIFACT: every numerical constant traces to a specific
   results/experiment_NNNN_*.json field (no docstring estimates, no
   hand-typed code constants in figure scripts that lack artifact citation).
2. SAME-BASIS COMPARISON: speedup/improvement comparisons use the same
   measurement basis on both sides (per-sample vs per-sample, same N,
   same hardware class, same algorithm class).
3. PROMINENT CAVEATS: any "extrapolated", "estimated", "synthetic",
   "small-scale" caveat is rendered at least as visually prominent as
   the claim it qualifies. Buried bottom-of-figure caveats fail.
4. NO HALLUCINATED CITES: every \cite{...} entry in carnot.bib must
   correspond to a real published or verifiably-existing work.
```

Violations of test 1, 2, or 3 in the existing paper become MANDATORY
fixes. Violations of test 4 require bibliography validation in the
.92 plan.

## Phase 1 — Critical Fixes (.92 mandatory pickup, blocks submission)

Each ISSUE-1 through ISSUE-5 ships as a separate task with full
`prior_failures` block citing the audit finding. Each MUST verify
its fix against the 4-test before marking complete.

### exp11XX-paper-v5-fig3-fix
**Scope:** Resolve ISSUE-1 (fig3 11680x speedup integrity).
**Three valid paths:**
- (a) Pull `fig3_fpga_latency.py` entirely; rewrite `main.tex` lines 506-515 to drop the figure
- (b) Re-render fig3 as a SINGLE BAR showing only the measured exp1068 24.83µs at 64 spins, no CPU comparison, no speedup claim
- (c) Run a real CPU benchmark for the same N=64 / per-sample basis as exp1068 (likely cite exp1094 measurement of 15.96µs which is the same C++ Glauber on real hardware), re-render comparison honestly. **Headline becomes ~250x not 11,680x or 58x.**
**Acceptance gate:** `figure_passes_4_test = True` AND `headline_matches_artifact_field = True`.

### exp11XX-paper-v5-kl-claim-fix
**Scope:** Resolve ISSUE-2 (KL=3.07 software-proxy framing).
**Action:** sweep `main.tex` for every "FPGA KL=3.07" mention; rewrite to "software-proxy KL=3.07; bitstream KL has not been measured on-board". Add a paragraph at end of Section 4 explicitly stating which measurements are bitstream-live and which are software-proxy.
**Acceptance gate:** `n_misleading_kl_claims_remaining == 0` (cross-checked by grep).

### exp11XX-paper-v5-15p6x-baseline-fix
**Scope:** Resolve ISSUE-3 (15.6x speedup CPU baseline).
**Two valid paths:**
- (a) Run a real optimized C++ Gibbs benchmark on the same 12-spin ring exp1094 used; cite the artifact ID; recompute the ratio.
- (b) Retract the 15.6x speedup headline; rewrite to cite exp1094's actual measured 15.96µs CPU vs FPGA bitstream timing as the only defensible comparison; let the new ratio (~249x) be the honest number with proper caveat.
**Acceptance gate:** `cpu_baseline_artifact_id != null` AND `cpu_baseline_value_matches_artifact_field == True`.

### exp11XX-paper-v5-hardnet-reframe
**Scope:** Resolve ISSUE-4 (76,130x HardNet++ apples-to-oranges).
**Action:** rewrite line 730 from "ran 76,130x faster than prompt repair" to "handled the constraint family in 117µs per violation vs 8.9s for prompt repair on the same 20 cases" (cite exp1147). Drop the multiplicative speedup framing entirely.
**Acceptance gate:** no "76,130" or "76130" anywhere in main.tex.

### exp11XX-paper-v5-and-compose-honest-framing
**Scope:** Resolve ISSUE-5 (exp1121 SOSKANEnergyV3 collapse hidden).
**Action:** rewrite Section 5 (or wherever exp1121 is cited) to ADD the SOSKANEnergyV3=0.3333 production-corpus finding. **This is OFFENSIVE remediation, not defensive** — the OOD collapse strengthens Wall 3 (verifier null space) by providing a concrete numerical example. Frame as: "On the FoVer corpus, SOSKANEnergyV3 reaches AUROC=0.9545 (exp1072). On the exp1121 production corpus, the SAME verifier reads AUROC=0.3333 — a striking confirmation of the OOD energy-inversion pattern." Two AUROCs, two corpora, one verifier; the difference IS the contribution.
**Acceptance gate:** `wall_3_narrative_includes_concrete_ood_example == True`.

## Phase 2 — High-Severity Fixes (.92 strongly preferred, .93 hard deadline)

### exp11XX-paper-v5-grpo-confidence-intervals
ISSUE-6: rewrite GRPO claims to include n=25/n=47 sample sizes inline; add binomial CIs (which would span zero on n=25); add small-sample caveat at least as prominent as +8.51pp number.

### exp11XX-paper-v5-humaneval-reframe
ISSUE-7: HumanEval baseline 0.0% is harness failure, not model failure. Either compare against an alternative-extraction-harness pass@1, or reframe as "after extraction-fix, +36pp" with explicit pre/post extraction caveat. Move from headline to anomaly section.

### exp11XX-paper-v5-alpha-t-disagreement-caveat
ISSUE-8: add the 24/100 ground-truth-correct rejection rate to alpha_t=0.38 framing. The current Zenil interpretation depends on the verifier's calibration to ground truth, not just positivity.

### exp11XX-paper-v5-phase4-pilot-stronger-baseline
ISSUE-9: re-run exp1165 Phase-4 pilot with a non-trivial baseline (BFS to goal, OR shortest-action heuristic that's harder than random-legal-greedy). The 98%-solve baseline reveals the puzzles are too easy. ALSO: capture full free_energy_values trace across all 10 puzzles (currently 3); state monotone fraction over the right N.

### exp11XX-paper-v5-seed-iq-table-footnote
ISSUE-10: add footnote to Table 5 explicitly stating "Seed IQ row is documented fallback evidence (ops/known-issues.md); not independently re-fetched in this work" — per exp1166's own `seed_iq_score_confirmed: false`.

## Phase 3 — Medium-Severity Fixes (.93 strongly preferred)

### exp11XX-paper-v5-thinkprm-citation-fix
ISSUE-11: name the predecessor experiment ID for ThinkPRM AUROC=0.9885, OR move the number to a labeled "from prior milestone, artifact retired" status.

### exp11XX-paper-v5-retrained-verifier-holdout-disclosure
ISSUE-12: state n=50 holdout size; cross-reference exp1121's contradictory production reading.

### exp11XX-paper-v5-nrgpt-non-monotone-disclosure
ISSUE-13: **CRITICAL FOR PHASE-4 NARRATIVE.** If NRGPT is referenced, cite as "AUROC_n1=0.9209, AUROC_n3=0.9158 (energy iteration NOT monotone — `n_iters_monotone=False`)". The non-monotonicity is a finding, not something to hide.

### exp11XX-paper-v5-sos-kan-auroc-reconciliation
ISSUE-14: reconcile two SOS-KAN AUROCs (0.9902 vs 0.9545) by naming the corpus and sample size each was measured on.

### exp11XX-paper-v5-fig2-binormal-caveat
ISSUE-15: add "(binormal fit from published AUROC; not a re-evaluation)" to fig2 paper caption.

## Phase 4 — Low-Severity Fixes (.93 nice-to-have)

### exp11XX-paper-v5-bibliography-audit
ISSUE-16: audit `docs/arxiv-paper/carnot.bib` for stub/placeholder entries. Verify each \cite{...} corresponds to a real published or verifiably-existing work. Suspect entries: `themesis2026seediq`, `hive2026`, `llmsgamingverifiers2026`, `rewardunderattack2026`.

### exp11XX-paper-v5-table1-caption-clarity
ISSUE-17: tighten Table 1 caption around the retracted k=15 row.

### exp11XX-paper-v5-hardware-portability-theorem-coverage
ISSUE-18: add "no substrate other than KV260 has been empirically verified" to the hardware-portability theorem paragraph.

## Phase 5 — Process Hardening (one-shot)

### exp11XX-figure-integrity-audit-script
**Scope:** ship a `scripts/figure_integrity_audit.py` that:
- Scans every `docs/figures/*.py` script
- Extracts every numerical constant
- Greps `results/` for an artifact field matching the constant
- Flags any constant that doesn't trace to a measured artifact
- Runs as a pre-commit hook before any paper-rebuild task can mark complete

**Why now:** the same audit pattern that caught fig3 will catch the next fig3. Codifying it as a script + hook prevents recurrence.

### exp11XX-claim-traceability-pre-commit-hook
**Scope:** ship a `scripts/paper_claim_audit.py` that:
- Reads main.tex
- Extracts every numerical claim (regex: `\d+(\.\d+)?\s*(×|x|%|pp|µs|ms|fold)`)
- Verifies each is followed within 200 chars by a `(exp\d+)` citation
- Reads the cited artifact JSON
- Verifies the claimed value matches an actual field
- Flags mismatches

**Why now:** prevents drift between the paper and the experimental record.

## Acceptance criteria for hold-lift

The publication hold lifts when ALL of the following are true:

```
phase_1_critical_fixes_landed = 5/5    (ISSUE-1 through ISSUE-5)
phase_2_high_severity_landed = 5/5     (ISSUE-6 through ISSUE-10)
phase_3_medium_landed         = 5/5     (ISSUE-11 through ISSUE-15)
phase_4_low_landed            = 3/3     (ISSUE-16 through ISSUE-18)
figure_integrity_script_active = True
paper_claim_audit_hook_active  = True
4_test_passes_for_every_claim  = True
operator_explicit_approval     = True   (operator must say "submit it")
```

**Estimated wall time:** Phase 1 alone is ~6-10 hours of careful work
(LaTeX edits + figure re-renders + potentially one new CPU benchmark
experiment). Phase 2-4 add another ~6-8 hours. Phase 5 hardening is
~3-4 hours. Total: ~15-22 hours of integrity work distributed across
.92 and .93.

## Decentralization implications

This proposal preserves rules 1-7 of the decentralization-respecting
design constraints in CLAUDE.md. No new closed-weight dependencies;
all integrity audits run on local artifacts; bibliography validation
uses publicly-citable sources only.

## Cross-references

- Audit punch-list: `ops/known-issues.md` PUBLICATION HOLD section (extended 2026-05-02 18:50Z)
- exp1167 manual override: `results/experiment_1167_paper_v4_phase4_section.json#manual_override_2026_05_02T18_40Z`
- Memory: `feedback_paper_integrity_audit.md`
- 4-test framework: see "Decision-making framework" above
