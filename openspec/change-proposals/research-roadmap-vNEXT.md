# Research Roadmap — Milestone 2026.06.336

**Status:** Pre-staged by outer-loop Claude (Opus 4.8), 2026-06-01.
**Predecessor:** 2026.06.335 (FACTS-made-real milestone — all 14 tasks landed).
**Milestone doc for:** `research-roadmap-next.yaml` (`milestone: 2026.06.336`)

---

## 1. What the previous milestone (.335) proved

`.335` was the milestone that **made the FACTS row real** and hardened the
math+code cross-domain scope. All 14 tasks landed real artifacts (codex,
28/28 clean across .334+.335). Earned outcomes:

| Result | Finding | Artifact |
|---|---|---|
| **Facts row, real NLI** | A real model-based NLI atomic-claim grounding verifier (arXiv:2504.18639 recipe) was built and scored. Facts is **domain-bound even with a real model**: grounding AUROC **0.7437** (CI [0.70,0.785]) ~= confidence baseline **0.7446** (delta -0.0009, CI straddles 0). | exp3654, exp3655 |
| **...but a complementary catch-signal exists** | At fixed confidence FPR the grounding verifier catches **~40%** of errors confidence misses (McNemar **p=0.00031**) -- AUROC-parity hides real second-pair-of-eyes value. **Measured on a SYNTHETIC v3 corpus.** | exp3655 |
| **Dependency-aware weighting BEATS Carnot** | Learned label-conditional dependency-structure weighting (arXiv:1903.05844) reached **0.9326** vs Carnot's **0.919** (+0.013) and recovered the -0.236 naive-penalty regression (+0.297). **BUT the artifact was tautology-flagged** (false positive: same AUROC stored under two field names). | exp3656 (flagged) |
| **Code generalization replicated** | math->code verifier transfer held on a balanced 2nd corpus -- code claim hardened. | exp3658 |
| **Second-pair-of-eyes detector** | Calibrated fused detector **wins on math** (0.954 vs 0.5 confidence; Brier 0.015/ECE 0.020) but is **weak+uncalibrated on the imbalanced code corpus** (0.45, ECE 0.37). | exp3657 |
| **Trained judge OOD -- RETIRED** | A real-substrate trained EBM judge ALSO failed OOD (0.572 < confidence 0.882), same verdict as the .334 toy head. `retire_if_same_verdict` fired -> the trained-judge-as-cross-domain-fix hypothesis is retired. | exp3659 |
| **FR-11 v9** | Online fusion-weight learning held with no collapse, +0.168 AUROC gain. | exp3660 |
| **Publication gate** | **paper_ready = TRUE.** G1 (FoVer 0.9131, 5-seed, CI), G2 (CI reproducer run 26725185125), G3 (narrowing-clean), G4 (traces to artifacts) ALL met. | exp3664 |
| **Backend** | gemini quota RECOVERED per a single probe; codex routing proven across 28 tasks. | exp3653 |

**Strategic position:** the paper is over the line (paper_ready true). Per
`ops/north-star.md` sec 1, the Depth-Over-Breadth forcing function is retired
(P0.1 answered honest-negative; G2 closed). The mandate now: **resume research
breadth toward NEW directions -- the verifier ensemble's discriminating value
where self-consistency is NOT near-optimal -- while every milestone either
advances the headline, ships product value, or earns a trustworthy negative.**

---

## 2. The three biggest gaps (PRD vision vs current state)

1. **The headline AUROC has a live improvement lead that is quarantined.**
   exp3656 showed dependency-aware weighting beats Carnot's weighting (0.9326 vs
   0.919) -- the first result in many milestones that could *raise* the headline.
   It is blocked only by a false-positive tautology flag. Resolving it cleanly
   (de-tautologized, multi-seed, DeLong, held-out validated) is the single
   highest-value thing .336 can do toward sec 1 ("raises the AUROC").

2. **The facts negative rests on a synthetic corpus.** The core mission is
   "escape LLM hallucinations." `.335` earned "facts is domain-bound" -- but only
   on a synthetic v3 corpus where confidence already reached 0.745. The real
   benchmarks (RAGTruth arXiv:2401.00396, FELM, HaluEval) were never tried. Until
   the facts verdict is measured on a REAL hallucination corpus, "facts does not
   generalize" is not fully earned -- and the complementary catch-signal (40% at
   fixed FPR) hints the AUROC framing may be the wrong lens.

3. **The strongest product result is still an experiment artifact, not a
   product.** The second-pair-of-eyes detector (exp3657) wins on math and is
   calibrated, but it is not wired into the shipped Phase-1 surface
   (pipeline / MCP `score_candidates` / CLI). Phase 1's ship gate is software-
   operational; turning the validated detector into a deployable, tested API is
   the product-headline path (north-star sec 1 product advancement).

---

## 3. Milestone architecture (4 phases, 13 tasks)

```
Phase 0 -- Transition + routing safety (exp3665, exp3666)
    archive .335 / activate .336  -->  backend-state diagnostic (gemini stability, 2nd probe)

Phase 1 -- ADVANCE THE HEADLINE (the dependency-aware weighting lead)
    exp3667 clean de-tautologized dependency-aware weighting (1903.05844 + Weaver 2506.18203)
              |  dependency_aware_beats_carnot (BARE bool)
              v
    exp3668 held-out / cross-split validation (guard against overfitting the weighting)

Phase 2 -- FACTS ON A REAL BENCHMARK (stress the earned negative)
    exp3669 build a REAL factual-hallucination corpus (RAGTruth / FELM, evidence+labels+confidence)
              |  real_factual_corpus_built (BARE bool)
              v
    exp3670 re-measure facts row (real NLI verifier exp3654) on the REAL corpus
              (AUROC + the complementary catch-rate lens; retire_if_same)

Phase 3 -- PRODUCT + SELF-LEARNING + NEW DIRECTION
    exp3671 ship the second-pair-of-eyes detector into the Phase-1 surface (pipeline/MCP/CLI) + E2E
    exp3672 NEW DIRECTION -- ensemble selection value where SC is WEAK (flip the P0.1 premise)
    exp3673 FR-11 v10 -- online dependency-aware verifier weighting (forward diff from v9 fusion)

Phase 4 -- HARDWARE CONTINUITY + CAPSTONE
    exp3674 KV260 continuity   exp3675 PolarFire continuity   exp3676 GateMate audit
    exp3677 Capstone v336 + G1-G4 gate synthesis (paper_ready must stay TRUE)
```

### Dependency graph

- exp3668 **gated_on** exp3667.`dependency_aware_beats_carnot == true`
- exp3670 **gated_on** exp3669.`real_factual_corpus_built == true`
- exp3677 (capstone) aggregates exp3667/3668/3669/3670/3671/3672/3673
- All `gated_on` upstream fields are emitted as **BARE scalars**
  (per `feedback_gated_fields_must_be_bare` -- a `{value,principle}` dict breaks
  the conductor gate, the .330 cascade).

---

## 4. Invariants (do NOT regress)

- **paper_ready = TRUE** (G1-G4 closed 2026-05-31). The capstone re-checks; no
  task may regress the gate. Headline = FoVer **0.9131** (frozen, G2-reproduced).
  A dependency-aware improvement is reported as a *candidate* for a future
  re-freeze, NOT silently substituted for the frozen headline.
- **P0.1 stays honest-negative.** Do not re-test Route-1/Route-2 energy-descent
  (Depth-Over-Breadth retired; the question is answered).
- **Trained-judge-as-cross-domain-fix is RETIRED** (exp3659 same-verdict). Do not
  re-propose a trained-judge OOD vN.
- **Anti-poison test discipline (.325/.326/.332 cascade):** every shipped pytest
  parametrizes over the script's honest verdicts on realistic synthetic fixtures;
  never hard-assert one success string against a real corpus; never use
  Q/R/H-number placeholder tokens; run your own test green before finishing.
- **Verifier authenticity:** docstring must match implementation; a heuristic
  must carry the `pcib_probe.py`-style disclosure, never be named "model-based".
- **Leak guards:** any grounding verifier scores `(model_answer, evidence)` only,
  never the gold answer/label; AUROC >= 0.99 on n>=200 is a RED-FLAG leak.

## 5. Backend routing decision

exp3653 reported gemini *recovered* via a single trivial probe, but `.333` was a
total gemini-crash wipeout (zero artifacts) and codex ran **28/28** tasks cleanly
across `.334`+`.335`. Asymmetric risk (a gemini relapse wipes the whole
milestone) favors **one more codex+requires_codex milestone** while gemini
stability is confirmed across a *second* consecutive probe (exp3666). Once
exp3666 confirms gemini stable across .335 AND .336, `.337` may flip to
gemini-default per the standing Gemini-Default rule. The `requires_codex` on each
task cites this anti-wipeout rationale. **Operator may override to gemini-default
if quota preservation outweighs the wipeout risk.**

## 6. Hardware requirements

- exp3667/3668/3670/3671/3672/3673: CPU-only verifier-scoring against cached
  corpora (`verifier_ensemble_against_cached_candidates`). GPU opportunistic.
- exp3669 (real corpus build) may fetch RAGTruth/FELM -- needs network; precondition-gated.
- Hardware continuity (exp3674-3676): SSH-reachability preconditions only
  (KV260 = `ssh kria`, NOT host SD card; PolarFire = `ssh polarfire`; GateMate =
  `command -v openFPGALoader` audit-only). KV260 has been unreachable 5
  milestones -- surface as operator-action.

## 7. New references filed (research-references.md)

Weaver (arXiv:2506.18203), Learning Dependency Structures for Weak Supervision
(arXiv:1903.05844), Dependency Structure Misspecification (arXiv:2106.10302),
RAGTruth (arXiv:2401.00396) + RAGTruth++, HalluLens/HalluScan (arXiv:2605.02443),
Energy-Based Calibration for Implicit CoT (arXiv:2511.07124).
