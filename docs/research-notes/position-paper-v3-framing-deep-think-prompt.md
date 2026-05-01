# Deep Think Round 3 Prompt — Position Paper v3 Framing Strategy

**Status:** Ready to send. Strategic framing question for the
Carnot position paper, given .85's empirical findings have
recalibrated several headline claims.
**Date drafted:** 2026-05-01
**Publication deadline:** 2026-05-15 (2 weeks).
**How to use:** open a fresh Deep Think session. Paste the section
labelled `## Prompt to send (verbatim)`.

---

## Prompt to send (verbatim)

### Background

The Carnot project is preparing a position paper (v2 currently
arxiv-ready as of exp1091, deadline 2026-05-15). Today's empirical
work (2026-05-01) recalibrated several load-bearing claims that
the v2 draft cites:

**Recalibration 1 — FPGA speedup headline.** The v2 cites a 13,061×
CPU-vs-FPGA speedup at N=64 from exp1081. Subsequent Deep Think
analysis (FPGA Round 2) revealed:
- The CPU baseline was unoptimized Python (~52 µs).
- Against optimized C++ CPU (~1 µs), the realized speedup is
  ~15.6× at k=4 chromatic colors.
- Empirical chromatic number χ of Carnot's actual J matrices is
  unmeasured. For numeric-SAT constraints with K-clique structure,
  χ ≥ 8 is mathematically guaranteed.
- The synchronous parallel Glauber that produced the 13,061×
  result violates detailed balance (KL=3.07 nats vs 0.05 threshold,
  empirically confirmed exp1094).

**Recalibration 2 — AND-composition over k=15 verifiers.** The v2
cites the architecture's k=15 AND-composition for exponential
joint-null-space shrinkage. Subsequent Deep Think analysis
(k-ceiling Round 2 + Phase-3 Round 2) revealed:
- Exp1093 measured pairwise r=0.66 across 3 deployed text probes.
- Participation Ratio of exp1093's correlation matrix:
  D_eff = 1.603 (not 5).
- Welch/Rankin Simplex bound at α²=0.66, r_max=0.5: k* ≤ 3.125
  within homogeneous text-probe cluster.
- With cross-mechanism diversity (Z3, gVisor, etc.), realistic
  k_max ≈ 8.
- Geometric-mean cos^k(θ_F) approximation for joint volume is off
  by ~3.2 × 10⁹× on mixed-correlation matrices; correct math is
  √det(Σ).

**Recalibration 3 — FPGA production hardware path.** v2 frames the
KV260 as proof-of-concept with Extropic Z1 + photonic as
production. FPGA Round 2 analysis:
- Z1 has zero independent peer-reviewed silicon validation that
  it samples from exp(-E/T) at scale.
- Pivoting production to Z1 is "vendor marketing, not science."
- Realistic Phase-2: ship 1-milestone χ ≤ 4 Fast-Path bitstream
  with CPU sequential fallback for χ > 4 ("Sparse-Constraint
  Accelerator").
- Z1 deferred to "future architecture pending independent
  silicon benchmarking"; build CPU emulator only for now.

**Recalibration 4 — Phase-3 prototype readiness.** v2 frames
Phase-3 (DBAE-EBM) as the architectural endgame. Phase-3 Round 1
recommended BLOCKING Phase-3 prototype work; Phase-3 Round 2
refuted the block in favor of "ship Phase-3 NOW with documented
D_int=1.6 limitation, build Mock Cascade for parallel-track
engineering."

**Recalibration 5 — energy ordering inversion finding.** The .85
retro reported `mean_correct=0.689 > mean_incorrect=0.621` on
SOTA outputs — a possible **energy-ordering inversion** that
contradicts the v2's claim that low-energy = correct. Empirical
investigation pending; the cause is unknown.

### What v2 currently claims (excerpt summary)

The v2 draft cites:
- `arxiv_ready` verdict
- All 5 .85-scan papers (2508.17440, 2603.06621, 2604.15149,
  2510.23972, 2512.15605)
- Live results: `alpha_t=0.38, +36pp HumanEval, AUROC=0.9545,
  FPGA=24.83µs at N=64`
- The 9-round Zenil derivation chain (Phase-3 → Phase-7 defense
  stack)
- 13,061× FPGA speedup
- Open-source local-first architecture (Apache 2.0)
- Decentralization-respecting design (CLAUDE.md rules 1-7)

### The question

**Given the 5 recalibrations above, what is the most defensible
v3 paper structure for the 2026-05-15 deadline?**

The paper must:
- Cite all .85-scan papers and theoretical foundations honestly.
- Recalibrate numerical claims that don't survive scrutiny.
- Maintain the survivability argument (the architectural defense
  stack is sound).
- Acknowledge empirical limitations without abandoning the
  contribution claim.
- Respect decentralization rules and open-source positioning.
- Remain shippable in 2 weeks of operator time.

### Sub-questions to address

#### Q1. Numerical claim recalibration matrix

For each suspect v2 claim, recommend the v3 framing:

| v2 claim | v3 framing options |
|---|---|
| 13,061× FPGA speedup | (a) retract entirely; (b) recalibrate to ~15.6× vs optimized C++; (c) contextualize as "vs vendor-equivalent unoptimized baseline"; (d) defer FPGA section to follow-up paper |
| k=15 AND-composition | (a) reframe as "k≈7-8 with strict mechanism orthogonality"; (b) document D_int=1.6 measurement as structural finding; (c) replace with Welch-bound theoretical contribution |
| Z1 production pivot | (a) reframe as "future hardware target"; (b) cite vendor without silicon claims; (c) remove entirely; (d) frame as "research direction" |
| Phase-3 prototype readiness | (a) "preliminary prototype, working scaffolding"; (b) "architectural framework, prototype in progress"; (c) "complete vision, validation experiments pending" |
| Energy ordering on SOTA outputs | (a) acknowledge as open question; (b) defer until investigated; (c) frame as evaluation methodology question |

For each row, recommend the option that **maximizes scientific
defensibility** while preserving the core contribution.

#### Q2. Section-by-section v3 outline

Propose a v3 paper outline that:

- Opens with the problem (LLM verification) and Carnot's thesis
  (energy-based verification + repair).
- Presents the architectural framework (Phase-3 through Phase-7)
  as the contribution.
- Cites the empirical findings honestly (including the Welch
  bound, D_int=1.603 measurement, FPGA correctness audit, energy
  inversion) as evidence the framework can be empirically
  validated.
- Frames the recalibrations as **methodological insights**, not
  failures.
- Closes with the publication-ready evidence + roadmap for
  future work.

How many sections, in what order, with what relative space
allocation?

#### Q3. The "honest negative findings" angle

Several .85 findings are honest negatives:
- exp1099 RLVR-SSD `no_improvement_honest_negative` (corpus pre-
  filtering masked signal)
- exp1100 `cascade_validated_sota_inefficient` (Pareto-suboptimal
  cost)
- exp1093 `verifiers_correlated_diversity_needed` (D_int=1.6
  ceiling)
- exp1094 `fpga_sampler_distribution_mismatch_confirmed` (KL=3.07)

How should these be presented? Options:

- **A) Hide them** — only cite positive findings. Risks paper
  failing peer review when reviewers ask about these.
- **B) Bury them in appendices** — present positive headline,
  technical details in supplementary material.
- **C) Foreground them as the contribution** — "we built the
  framework, applied phase-validation discipline, and these
  are the empirical findings that emerged."
- **D) Frame as future-work limitations** — "these findings
  motivate the v3 architecture amendments below."

Recommend the most defensible approach for a 2-week deadline.

#### Q4. Reviewer adversarial defense

Anticipate the top 3 reviewer concerns. For each:
- The likely critique
- The strongest defense available given the .85 empirical state
- The sentences that should appear in v3 to preempt the critique

Concerns to consider:
- "Your FPGA speedup is misleading."
- "Your verifier ensemble is correlated, why does the scheme
  work?"
- "Your Z1 claim is vaporware."
- "You haven't actually trained Phase-3."
- "What about Goodfire Silico's white-box approach?"
- "Why do mean_correct > mean_incorrect on SOTA outputs?"

#### Q5. Decentralization-and-open-source positioning

The paper positions Carnot as Apache-2.0 + local-first +
multi-mirror (gitea + GitHub) + sovereignty-respecting. Goodfire
Silico is closed-source + Goodfire-service-required. How should
this contrast be framed?

- (a) Aggressive contrast (positions Carnot's approach as
  philosophically superior).
- (b) Complementary framing (different layers of the LLM-
  reliability stack; both have value).
- (c) Engineering-first framing (Carnot solves a different
  problem; positioning is incidental).

#### Q6. The 2-week timeline reality

What is the realistic v3 work plan? Each item should be
2-3 hours of operator time:

- (1) Update numerical claims per Q1.
- (2) Restructure outline per Q2.
- (3) Add honest-negatives section per Q3.
- (4) Add reviewer-defense paragraphs per Q4.
- (5) Resolve energy-ordering inversion (empirical investigation
   first; if architectural concern surfaces, paper acknowledges).
- (6) Update bibliography with newly-relevant work (Goodfire,
   Welch bound, Participation Ratio references).
- (7) Final pass for consistency and tone.

What is the priority order? Which items can be parallelized?
Which is the load-bearing critical-path item?

### Constraints

- 2026-05-15 deadline is hard. Latex compilation + arXiv submission
  takes a day; effective work deadline is 2026-05-13.
- Operator has ~2 hours/day for v3 work, total ~28 hours.
- Co-authors (if any) will not contribute substantial revisions
  in this timeframe.
- Any claim that requires new experiments to support cannot
  appear in v3 (those are .86+ work).

### What I am NOT asking

- I am NOT asking for new architectural defense layers beyond
  Phase-7.
- I am NOT asking to retract the paper or defer publication.
- I am NOT asking for theoretical contributions beyond what's
  already derived.
- I am NOT asking for marketing copy or rhetorical sophistry —
  the goal is scientific defensibility.

### Output format

1. **Executive summary** (1-2 paragraphs naming the recommended
   v3 framing strategy).
2. **Q1 answer** — recalibration matrix with recommendations.
3. **Q2 answer** — section-by-section outline.
4. **Q3 answer** — honest-negative-findings positioning.
5. **Q4 answer** — top 3 reviewer adversarial defenses.
6. **Q5 answer** — Goodfire Silico positioning.
7. **Q6 answer** — 28-hour work plan with critical-path item.
8. **Risk register** — top 3 ways v3 still fails peer review with
   the recommended framing, plus mitigation.

### Honesty requirement

If the .85 empirical state does not support a defensible v3 by
2026-05-15, say so explicitly. Recommend either (a) deferring to
a workshop track, (b) targeting a later venue, or (c) acknowledging
the timeline mismatch and shipping v3 as a "preliminary findings"
paper. The honest "v3 isn't ready for the headline venue but
there's a defensible alternative path" answer is more valuable
than overclaiming.

Per the project's documented Deep-Think prediction pattern, prefer
qualitative recommendations that bound the strategic space over
specific prescriptions (e.g., "use 30 vs 40 pages, this exact
abstract phrasing"). The user will translate qualitative guidance
into the actual draft.
