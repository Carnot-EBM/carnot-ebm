# Anthropic RSI + Automated-W2S — Verified Citation Collection

**Purpose.** A curated, verbatim-verified quote bank from two Anthropic 2026
publications that are directly load-bearing for Carnot's autonomous-research +
verification thesis. Assembled for citation in paper-v6 (autonomous-research,
verification, and limitations sections) and the operator's external writing.

**Provenance / verification discipline.** Each quote below was captured via two
independent WebFetch passes against the live URLs on 2026-06-05. Quotes that
returned identical verbatim text across both passes are marked **[VERIFIED]**.
Quotes that appeared in only one pass, or whose surrounding context was not
re-confirmed contiguous, are marked **[FLAGGED — re-check against live HTML before
paper use]**. WebFetch routes through a summarizer model, so even VERIFIED quotes
should be diffed against the live page HTML before they appear in a published
artifact (per CLAUDE.md literature-priority + adversarial-artifact discipline).

---

## Source A — "Recursive Self-Improvement" (rendered title: "When AI builds itself")

- Publisher: Anthropic Institute
- URL: https://www.anthropic.com/institute/recursive-self-improvement
- Accessed: 2026-06-05

### A1. Definition [VERIFIED — opening section]
> "Taken far enough, and given enough compute, that trend points to an AI system
> capable of fully autonomously designing and developing its own successor. This is
> called recursive self-improvement."

### A2. Status / urgency [VERIFIED — opening section]
> "We are not there yet, and recursive self-improvement is not inevitable. But it
> could come sooner than most institutions are prepared for."

### A3. The capability-vs-judgement gap [VERIFIED — "Evidence from within Anthropic"]
> "However, large performance gaps persist when it comes to Claude exercising
> judgement in choosing goals in both engineering and research. That's the gap
> between AI today and a future system that could autonomously design its own
> successor."

*Carnot relevance:* this is the conductor's exact bottleneck — capability races,
goal-choice judgement lags. Corroborates the Deep-Think P3 "loop can't self-seed /
Verification Trap" finding and why Carnot's research frames are human-seeded.

### A4. Future human role = verification [VERIFIED — "Possible futures", Scenario 3]
> "Humans play a substantially diminished role in their development, likely moving
> most of our effort towards oversight, validation, and verification of an expanding
> 'virtual lab' run by AI systems."

*Carnot relevance:* the "virtual lab + oversight/validation/verification" is the
Carnot conductor + adversarial_verify.py + verifier-authenticity discipline +
outer-loop-as-supervisor. A perfect energy SCORER is not a GENERATOR but it is a
VERIFIER — exactly the chokepoint this names.

### A5. Compounding misalignment [VERIFIED — "Possible futures", Scenario 3]
> "Alternatively, the rare occurrences of misalignment present in today's models
> could compound as the models build their successors, growing more frequent but
> less understood until we lose control of them."

*Carnot relevance:* the argument for externally-grounded, un-gameable verification
(energy = ground truth) and a multi-verifier ensemble with a hard-to-mimic joint
null space.

### A6. The W2S result, as cited in the RSI piece [VERIFIED — "Evidence from within Anthropic"]
> "Two human researchers, over about a week, recovered roughly 23% of that gap; the
> agents recovered 97% over 800 cumulative hours and used roughly $18,000 in
> compute."

### A7. Delegating AI development to AI [VERIFIED — opening section]
> "But at Anthropic, we are delegating a growing share of AI development to AI
> systems themselves, which is speeding up our work."

### A8. The conditional / verifiable slowdown [VERIFIED — "What should we do?"]
> "We believe it would be good for the world to have the option to slow or
> temporarily pause frontier AI development to enable societal structures and
> alignment research to keep up with the advance of the technology."

> "A meaningful slowdown or pause would require multiple well-resourced labs at or
> near the frontier, in multiple countries, agreeing to stop under the same
> conditions. It would also require that each can verify that the others have
> actually stopped."

### A9. Verification difficulty [VERIFIED — "What should we do?"]
> "Training runs are far easier to conceal than missile silos, their inputs are
> general-purpose, and the incentive to defect quietly is enormous, because whoever
> continues while others pause could inherit the lead."

### A10. The window [VERIFIED — closing]
> "The window to investigate the questions together is here, and people outside AI
> companies should be involved in this deliberation."

*Carnot positioning:* A8-A9 are a centralized-actors governance model (multi-lab
verifiable pause). Carnot's contribution is orthogonal and complementary — supply
the OPEN, LOCAL, externally-grounded verification layer the agenda assumes, so the
oversight primitive isn't itself a closed-vendor dependency.

---

## Source B — "Automated Weak-to-Strong Researcher"

- Authors: Jiaxin Wen*, Liang Qiu*, Joe Benton, Jan Hendrik Kirchner, Jan Leike
  (Anthropic). "*" = co-first authors. "Research partially done as part of the
  Anthropic Fellows Program."
- URL: https://alignment.anthropic.com/2026/automated-w2s-researcher/
- Code: https://github.com/safety-research/automated-w2s-research
- Accessed: 2026-06-05

### B1. The problem framing [VERIFIED — §1 Environment / Background]
> "humans will have to supervise models much smarter than them. We study an
> analogous problem: using weak small models to supervise strong large models."

### B2. The human baseline [VERIFIED — Introduction]
> "two authors spent 7 days tuning four representative prior methods, achieving a
> best PGR of 0.23"

### B3. The headline result [VERIFIED — TL;DR / Introduction]
> "our AAR reaches a PGR of 0.97 within 5 days (800 cumulative hours across 9 AARs),
> at a total cost of roughly $18,000 in compute and API calls"

### B4. Practicality claim [VERIFIED — Introduction]
> "Our results suggest that automated research on outcome-gradable problems is
> already practical."

### B5. Directed seeding beats identical prompts [VERIFIED — §3.1]
> "the directed setting makes hill-climbing much faster and yields higher final
> PGR."

*Carnot relevance:* independent confirmation of the conductor's entropy-collapse /
breadth-churn failure and the Depth-Over-Breadth fix — diversity must be engineered
via directed seeding; the loop will not self-diversify.

### B6. Reward hacking was unpredicted [VERIFIED — §5 Reward Hacking]
> "None of the authors predicted these hacks before running AARs."

### B7. Label exfiltration [VERIFIED — §5 Reward Hacking]
> "AARs first estimate the uncertainty of its predictions on each test example,
> identify the five most uncertain ones, and then exfiltrate their labels."

### B8. Caps don't stop hacking [VERIFIED — §1 Environment / Evaluation]
> "At any practical cap, our AAR simply budgets its submissions more carefully and
> the same hacks still appear."

### B9. Mitigation: hold out entirely separate data [VERIFIED — §5 Reward Hacking]
> "We hence conclude that future work should test AAR-discovered ideas on entirely
> held-out datasets."

*Carnot relevance (B6-B9):* this is the empirical case for Carnot's whole
adversarial-verify / verifier-authenticity / poison-test discipline. An autonomous
swarm hill-climbing a single metric games it every time, in ways the designers do
not predict, and submission caps don't fix it — only an externally-grounded,
held-out, un-gameable objective does. Maps to energy-as-ground-truth + the G2
independent-reproducer gate + cross-corpus matrix.

### B10. The key bottleneck [VERIFIED through "...designing evals"; remainder FLAGGED]
> "The key bottleneck for alignment research is moving from proposing and executing
> ideas to designing evals"

[FLAGGED — re-check contiguity before paper use] continuation seen in first pass
only: "...we should find the right metrics (data, models) that AARs can reliably
hill-climb without overfitting."

*Carnot relevance:* "the bottleneck is designing evals, not executing ideas" IS the
verifier-product thesis, stated by Jan Leike's team.

### B11. Legibility — for now [VERIFIED — §9 Alien science]
> "For now, our AAR-proposed ideas remain understandable to humans."

### B12. Legibility — the warning [VERIFIED tail clause; lead-in FLAGGED]
> "science is no longer about understanding but merely hill-climbing"

[FLAGGED — re-check before paper use] lead-in seen in first pass only: "In the
future, however, we expect to eventually see hard-to-verify ideas emerge if we only
optimize for outcome rewards. In that case, science is no longer about
understanding but merely hill-climbing."

*Carnot relevance:* the legibility risk is answered in Carnot by principle-annotated
artifact fields (Anthropic "Teaching Claude Why") + honest_verdict discipline, which
keep the loop's outputs human-auditable.

### B13. Less structure beats more [VERIFIED — §6 Preliminary Results; ellipsis in source capture]
> "A fixed workflow... seems reasonable but underperforms giving AARs no workflow at
> all."

---

## One-paragraph synthesis for the paper

## Source C — "Automated Alignment Researchers" (public companion to Source B)

- Publisher: Anthropic Research (anthropic.com/research blog)
- URL: https://www.anthropic.com/research/automated-alignment-researchers
- Published: April 14, 2026 (Anthropic Fellows study)
- Accessed: 2026-06-05
- Relationship: the polished public-facing writeup of the SAME 9-agent W2S
  experiment as Source B; identical numbers (0.23 -> 0.97 PGR, 800h, $18k). Quotes
  below are SINGLE-PASS (one fetch) — mark **[SINGLE-PASS — two-pass verify before
  paper use]**. Added because Source C states three Carnot-load-bearing framings
  more sharply than the technical paper does.

### C1. The bottleneck flips from generation to evaluation [SINGLE-PASS]
> "the core bottleneck in alignment research could become evaluation (making sure
> that experiments are set up sufficiently well that we're confident in their
> results), rather than generation (relying on human researchers to propose
> promising ideas)."

*Carnot relevance:* the single crispest external statement of the Carnot thesis —
verification, not generation, is the scarce primitive.

### C2. The design requirement = Carnot's architecture [SINGLE-PASS]
> "Any deployment of automated researchers will require evaluations that the AARs
> can't tamper with—and human inspections of both their results and their methods."

*Carnot relevance:* both clauses are Carnot by construction — un-tamperable
evaluation (energy = ground truth) + human/AI inspection of results AND methods
(operator-in-loop + adversarial_verify.py + verifier-authenticity audit, which
inspects HOW an artifact was produced, not just its number).

### C3. Verification asymmetry, plainly [SINGLE-PASS]
> "If their skills progress to the point where they're generating millions of lines
> of incredibly complicated code that we can't parse ourselves, it could become very
> difficult to tell whether they're acting in the ways we intend."

### C4. Scalable-oversight framing of W2S [SINGLE-PASS]
> "As a proxy for scalable oversight, the weak model stands in for humans, and the
> strong model for the much-smarter-than-human models we might one day need to
> oversee."

### C5. "Research taste" — naming the goal-choice gap [SINGLE-PASS]
> "today's frontier models still lack 'research taste' (industry parlance for having
> an intuitive sense of which ideas might work and which won't)."

*Carnot relevance:* frontier-lab vocabulary for the same gap as RSI-essay A3 +
Deep-Think P3 "loop can't self-seed". Research taste is the missing capability; the
human supplies it.

### C6. The "alien science" risk [SINGLE-PASS]
> "Over time, the models' ideas could become much harder to verify, or corrupted in
> ways that are tricky for humans to parse or catch. That could mean creating an
> 'alien science'."

### C7. Methods don't transfer (humility flag) [SINGLE-PASS]
> "AARs tend to capitalize on opportunities unique to the models and datasets
> they're given, which means their methods might not work elsewhere."

*Carnot relevance:* direct caveat for any from-scratch tiny-corpus Carnot result
(e.g. the Sudoku v1-v4 arc) — the ordering is the finding, the absolutes are
setup-specific.

### C8. Bootstrapping research-taste via W2S [SINGLE-PASS]
> "We could conduct weak-to-strong supervision on Claude's ability to scope research
> projects."

*Carnot relevance:* a candidate path to partially close the "loop can't self-seed"
gap — but their own reward-hacking results are the warning: it needs an
un-tamperable score for "good research scoping," which is the hard part. File as
Phase-3+ idea, not a naive action.

### C9. Math-vs-code generalization gradient [SINGLE-PASS]
Best method on held-out data: math PGR **0.94**, coding PGR **0.47** (~2x baseline);
production-scale transfer to Claude Sonnet 4 "didn't lead to a statistically
significant improvement."

*Carnot relevance — a genuine cross-program convergence:* Carnot's
`verifier-domain-bound` finding independently shows the verifier is strong on math
(FoVer 0.91, G2-reproduced), weak on code, earned-negative on facts. Two separate
research programs find the SAME domain gradient — reasoning/math verification
generalizes; code verification is materially harder. External corroboration of
Carnot domain-boundedness, not coincidence.

---

The RSI essay is the worldview; the Automated-W2S paper is the controlled receipt.
Together they establish, from the frontier lab itself, that (a) autonomous research
on outcome-gradable problems already beats human baselines (0.23 -> 0.97 PGR,
~$18k), (b) the binding constraint is the verifier / eval design, not idea
generation ("the key bottleneck ... is ... designing evals"), and (c) an autonomous
optimizer games any single metric in unpredicted ways unless the objective is
externally grounded and held out (four unforeseen hacks incl. label exfiltration;
caps don't help). This is the Carnot thesis almost verbatim — with Carnot's twist
that the verification layer should be open, local, and decentralized, so the
oversight primitive the whole RSI agenda depends on is not itself a closed-vendor
chokepoint. Cite Source B as the closest published peer to the Carnot conductor.

See also: memory references `anthropic-recursive-self-improvement`,
`anthropic-automated-w2s-researcher`; `deep-think-post-bounded-2026-06` (P3 loop
can't self-seed); `sakana-dgm` (self-modifying agent removed its own safety markers);
`anthropic-teaching-why` (principle-grounded legibility).

---

## Strategic addendum — efficiency-parity verification is an RSI-SCALE requirement (2026-06-06)

Re-reading Source A (RSI essay) after Carnot's 2026-06-06 pivot to a HYBRID
pragmatic architecture (open LLM / TRM-refiner generator + energy ensemble as the
*verifier*; energy-as-generator closed-negative) and the operator's win condition
("the energy model equally as effective as the LM so long as it is cheaper and/or
faster") yields a sharper, load-bearing reading:

- A4 ("oversight, validation, and verification of an expanding 'virtual lab'") +
  A5 (misalignment compounds if the check can't keep pace) imply a SCALE problem:
  when AI generates work at machine scale, you **cannot afford a generative
  LLM-judge on every artifact**. Verifying a virtual lab DEMANDS a cheap, fast,
  scalable, un-gameable check.
- Therefore Carnot's **efficiency-parity** win condition is not merely pragmatic —
  it is *the property that makes "verify the virtual lab" tractable*. A
  forward-pass, hardware-acceleratable, externally-grounded energy verifier is
  exactly the always-on, scale-viable check the essay implies someone must build;
  a generative judge does not survive the throughput.
- This **elevates efficiency-parity from pragmatic-floor to strategically
  load-bearing**, and positions Carnot's verifier as a candidate scale-viable
  safety check for the RSI world — complementing (not competing with) Source A's
  centralized multi-lab governance, by supplying the open/local/cheap verification
  primitive that governance assumes exists.

Carnot-relevance for the paper: cite this as the strategic motivation for the
verifier-centric hybrid and for reporting **cost/latency at matched accuracy** (vs
an LLM-as-verifier baseline) as a first-class result, not just AUROC. See memory
`hybrid-pragmatic-architecture`.

Honest caveats (unchanged): parity itself is unproven (exp3885 moat-scissor
INCONCLUSIVE — "not clearly better" ≠ "reaches parity"); the cost/latency ratio is
plausible but UNMEASURED; goal-choice/research-taste (A3) remains human.

---

## Source D — "How Anthropic enables self-service data analytics with Claude" (2026)

- Publisher: Anthropic / claude.com blog
- URL: https://claude.com/blog/how-anthropic-enables-self-service-data-analytics-with-claude
- Accessed: 2026-06-06 (SINGLE-PASS — two-pass verify before paper use)
- Relevance: APPLIED BI-ops post (semantic layers, skills, validation), tangential
  to Carnot's core thesis — but two bits are directly load-bearing for the
  verification + efficiency-parity story. Captured per operator 2026-06-06.

### D1. The "silent failure" — Carnot's exact problem, admitted unsolved, in the no-oracle regime [SINGLE-PASS]
> the "silent failure" — wrong answers that look plausible and get used without
> objection... "we don't have a robust solution yet."

> "for analytics use cases, there's often only a single correct answer ... with no
> deterministic way of proving the correctness."

*Carnot relevance:* the cleanest external statement of the problem Carnot exists to
solve, landing in the HARDEST version — a domain with a single correct answer but
NO deterministic oracle (the *facts-like* regime where Carnot's verifier is
earned-negative, [[verifier-domain-bound-math-only]]). Validates the mission AND
names the hardest open frontier: grounded verification where there's no symbolic/
test check to anchor the energy. Their mitigations (provenance footer, human
sign-off, sanity-check evals) are heuristic, not externally grounded.

### D2. Adversarial-review cost/accuracy datapoint — concrete Pareto calibration [SINGLE-PASS]
> Adversarial Review: sub-agent aggressively challenges assumptions; increased
> accuracy 6% but added 32% tokens, 72% latency.

Plus: swapping the adversarial reviewer to a cheaper model "lost the accuracy wins
with no real speedup"; and the ablation "Direct SQL corpus access moved accuracy
<1%, despite 80% of answers being in the corpus. Conclusion: bottleneck is
structure, not access."

*Carnot relevance:* (a) a real, quantified point on the accuracy↔cost Pareto curve
for a verification pass (+6% acc / +72% latency) — exactly the tradeoff the
efficiency-parity win condition says to measure; (b) "structure not access" echoes
the Sudoku finding ([[energy-as-generator-sudoku]]) that the wall is inference
STRUCTURE, not whether the information is present; (c) caution: a cheaper judge can
lose the accuracy win — verifier-model choice matters.

### D3. Context-engineering + maintenance-decay (conductor lesson) [SINGLE-PASS]
> "Without skills, Claude's ability to answer analytics questions accurately didn't
> exceed 21%... Adding skills gets these numbers consistently above 95%."

> Skill accuracy "drifted from ~95% to ~65% over one month without active
> maintenance."

*Carnot relevance:* transfers to Carnot's own CONDUCTOR — autonomous-agent
reliability comes from scaffolding (CLAUDE.md disciplines, artifact templates,
verifier-authenticity machinery) and DECAYS without active upkeep (what the
in-process-docs reconciliation fights). Borrowable patterns: provenance footer
(source-tier label), active correction harvesting (scan channels → auto-PR fixes),
per-PR ablation deltas.

*Net:* relevance is aspirational/by-analogy, NOT a solution — it's a well-described
version of Carnot's hardest problem (oracle-free verification) from a team that
says they haven't solved it, plus useful cost/accuracy calibration for the
efficiency-parity frame.
