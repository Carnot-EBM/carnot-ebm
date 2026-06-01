# Research Roadmap — Milestone 2026.06.337

**Status:** Pre-staged by outer-loop Claude (Opus 4.8), 2026-06-01.
**Predecessor:** 2026.06.336 (first headline-advancing lead — all 12 tasks landed clean on codex).
**Milestone doc for:** `research-roadmap-next.yaml` (`milestone: 2026.06.337`)

---

## 1. What the previous milestone (.336) proved

`.336` produced the **first headline-advancing lead in many milestones** and
turned three open hypotheses into earned verdicts. All 12 tasks landed real
artifacts (codex, anti-wipeout routing). Read via `scripts/summarize_artifact.py`:

| Result | Finding | Artifact |
|---|---|---|
| **Dependency-aware weighting BEATS Carnot — CLEAN** | De-tautologized, 5-seed, DeLong: dependency-aware AUROC **0.9332** vs Carnot-current **0.9194** (+0.0138) on FoVer, adversarial-verify clean. The exp3656 tautology flag is resolved. | exp3667 |
| **…and it GENERALIZES held-out** | ≥5 disjoint train/test splits: held-out dependency-aware **0.9332** vs Carnot **0.9200**. Verdict: `headline_re_freeze_candidate_for_v337`. | exp3668 |
| **FACTS domain-bound on a REAL benchmark — earned-negative** | Real RAGTruth corpus (non-degenerate, confidence baseline 0.708) + real NLI grounding verifier (leak-free, n≥200): `facts_domain_bound_on_real_benchmark_335_negative_genuinely_earned`. `retire_if_same` fired → facts-generalization RETIRED. | exp3669, exp3670 |
| **Ensemble adds NO best-of-N SELECTION value** | Positive control valid (oracle 0.607 > SC 0.459, flips 28): ensemble-selection 0.344 = confidence-selection 0.344, both **below SC 0.459**; fusion 0.279. Paired Δ vs SC −0.115 (CI [−0.197,−0.049], McNemar p=0.016). Earned-negative. | exp3672 |
| **Second-pair-of-eyes detector SHIPPED — code-weak** | Wired `python/carnot/pipeline/second_pair_detector.py` → score_candidates (MCP/CLI), E2E green. Math fused AUROC **0.980** (Brier 0.014, ECE 0.009) but CODE AUROC **0.5** (ECE 0.27) even balanced. | exp3671 |
| **FR-11 v10 online dependency-aware weighting** | Holds, no collapse, gain +0.0018 (tiny). | exp3673 |
| **Publication gate** | **paper_ready = TRUE.** G1 (FoVer 0.9131, exp2850), G2 (CI reproducer run 26725185125), G3 (narrowing-clean), G4 (traces) all met. | exp3677 |
| **Backend** | gemini stable for 2 consecutive probes (exp3653/3666); eligible to flip, but `.333` was a total gemini-crash wipeout. | exp3666 |

**Strategic position:** the paper is over the line and now has a validated lead
that could *raise* the frozen headline. Per `ops/north-star.md` §1, every
milestone must either advance the headline, ship product value, or earn a
trustworthy negative. `.337` does all three — with a clear, honest scope:
**Carnot's verifier ensemble DISCRIMINATES step-errors well (math, generalizes
to code) but does NOT generalize to facts (real, earned) and does NOT
rank-within-a-question for best-of-N selection (earned).**

---

## 2. The three biggest gaps (PRD vision vs current state)

1. **The headline lead is validated but not promoted.** The dependency-aware
   weighting (0.9332) beats Carnot's weighting clean AND held-out, but the frozen
   headline is still 0.9131. Promoting it requires full **G1-rigor dual-condition
   integrity** (mirror exp2837/2850: production + architecture-only, ≥5 seeds, CI,
   reproducibility_checksum, adversarial-clean, leak-free) PLUS a re-reproduced
   **G2** (the CI reproducer asserts the old number) — and the north-star §1 edit
   is operator-curated. The `.337` job is to produce the evidence + the operator
   re-freeze package, never a silent substitution.

2. **Discrimination decouples from selection — is it fixable?** exp3672 earned
   that the ensemble (AUROC 0.93) selects worse than SC. Independent precedent
   (arXiv:2512.23067, "Reward Model Selection Crisis": discrimination↔selection
   decoupling, Kendall τ 0.08–0.31) says this is a known phenomenon. The open
   question for an *energy*-based verifier: can per-question calibration / a
   ranking objective / a stronger confidence signal (self-certainty,
   arXiv:2502.18581) recover selection value, or is the decoupling fundamental?
   This is the candidate-ranker product (Tier A `score_candidates`).

3. **The shipped detector is math-strong, code-blind.** exp3671 ships but code
   AUROC = 0.5 (no signal) and is badly miscalibrated even on the balanced
   corpus. Phase-1's ship gate is software-operational; a detector that returns
   noise on code is half-shipped. Test whether the validated dependency-aware
   weighting transfers to code discrimination and re-calibrate the code operating
   point — or honestly document code as math-only.

---

## 3. Milestone architecture (4 phases, 12 tasks)

```
Phase 0 — Transition + routing safety (exp3678, exp3679)
    archive .336 / activate .337  -->  backend-state diagnostic v3 (3rd gemini probe; gates a .338 flip)

Phase 1 — PROMOTE THE HEADLINE (the dependency-aware weighting re-freeze)
    exp3680 dependency-aware dual-condition integrity at G1 rigor (1903.05844 + Weaver 2506.18203)
              |  dependency_aware_g1_rigor_confirmed (BARE bool)
              v
    exp3681 G2 reproducer prep + OPERATOR re-freeze package (prepare-only; frozen 0.9131 stays)

Phase 2 — SCOPE THE PRODUCT HONESTLY (discrimination vs selection; code; stronger baseline)
    exp3682 diagnose + try to close the discrimination-vs-selection gap (2512.23067 + 2502.18581)
    exp3683 harden the detector's CODE operating point (+ dependency-aware weighting on code)
    exp3684 adversarial product re-baseline vs self-certainty (2502.18581) — is the value robust?

Phase 3 — SELF-LEARNING + HARDWARE + SYNTHESIS
    exp3685 FR-11 v11 online dependency-aware weighting with drift detection (no collapse)
    exp3686 KV260 continuity   exp3687 PolarFire continuity   exp3688 GateMate audit
    exp3689 capstone + G1-G4 v337
```

### Dependency graph (cascade-proof)

- exp3681 **gated_on** exp3680.`dependency_aware_g1_rigor_confirmed == true` (BARE bool).
- Every other science task is **ungated** and self-contained.
- exp3689 (capstone) is **ungated**; it aggregates exp3680–exp3685 and records
  `not_measured` for any skipped gated task — never reads a missing field as None
  and synthesizes around it.
- All `gated_on` upstream fields are emitted as **BARE scalars**
  (`feedback_gated_fields_must_be_bare` — a `{value,principle}` dict breaks the
  conductor gate, the .330 cascade).

---

## 4. Invariants (carried from .322–.336, do NOT regress)

- **paper_ready = TRUE (G1∧G2∧G3∧G4).** The frozen FoVer headline **0.9131**
  stays frozen. A dependency-aware win is a **headline-advancement CANDIDATE**
  pending a future re-freeze + re-reproduction — never a silent substitution.
  North-star §1 is operator-curated; the agent prepares the package only
  (Operator-Only External Publication + Public Documentation Discipline).
- **P0.1 stays honest-negative.** Depth-Over-Breadth is retired; do not re-test
  the answered energy-descent existential question.
- **facts-generalization is RETIRED** (exp3670 same-verdict on real RAGTruth). Do
  NOT re-propose a "does facts generalize" experiment.
- **trained-judge-as-cross-domain-fix is RETIRED** (exp3659). Do not re-propose.
- **SC-best-of-N-selection-by-ensemble-energy is an earned-negative** (exp3672).
  `.337` DIAGNOSES the gap (a different question — why, and is it fixable), it
  does not re-run the same selection test hoping for a different answer.
- **Every gated_on is a BARE scalar** (`feedback_gated_fields_must_be_bare`).
- **NO poison tests** (.325/.326/.332 cascade): parametrize pytests over honest
  verdicts on realistic synthetic fixtures; never hard-assert one success string
  against a real corpus; never use Q/R/H-number placeholder tokens; run your own
  test green before finishing.
- **De-tautology discipline:** store each conceptually-distinct metric under
  exactly ONE field name; aggregations use a fixed `random_seed`, measurements
  are content-derived.
- **Verifier authenticity + leak guards:** docstring matches implementation; a
  grounding verifier scores `(model_answer, evidence)` only, never the label;
  AUROC ≥ 0.99 on n≥200 is a RED-FLAG leak.

## 5. Backend routing decision (.337)

exp3666 reports gemini stable for **2 consecutive probes** (`recommended_routing:
gemini_default_eligible_for_v337`). The standing Gemini-Default rule and the
pre-stated `.336` flip criterion ("stable across .335 AND .336 → .337 may flip")
are both satisfied. **However**, `.333` was a total gemini-crash wipeout (all 14
tasks lost) and `.337` carries the highest-value work in months (the headline
re-freeze). Given the asymmetric risk, `.337` **keeps codex + requires_codex**
for one more milestone — matching the proven `.334/.335/.336` routing — and runs
a **3rd consecutive gemini probe (exp3679)** that gates a `.338` flip. The
`requires_codex` on each task cites this anti-wipeout rationale. **The operator
may override to gemini-default if quota preservation outweighs the wipeout risk**
(the diagnostic green light is already on file in exp3666).

## 6. Hardware requirements

Per Hardware-Task Continuity Discipline, one task per attached board, SSH-only
preconditions:
- exp3686 KV260 — `ssh kria` reachability (host SD-card checks PERMANENTLY
  retired). Unreachable since `.331`; records the outage streak as operator-action.
- exp3687 PolarFire — opportunistic `ssh polarfire` reachability + continuity.
- exp3688 GateMate — documentation/audit only (`command -v openFPGALoader`;
  flash/smoke host-IO hang is a known blocker — do NOT run the hanging path).

The science (exp3680–exp3685) is CPU-only verifier-scoring against cached corpora
(`verifier_ensemble_against_cached_candidates`); GPU opportunistic.

## 7. SOTA model usage

The `.337` science scores cached corpora (FoVer, RAGTruth, balanced code,
multi-candidate selection sets) — no live LLM load is headline this milestone.
Where a task does invoke a model, it must use a mandated SOTA GGUF
(`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`, or
`unsloth/gemma-4-26B-A4B-it-GGUF`) via the `.gguf` path (embedded tokenizer;
never `AutoTokenizer` on a GGUF repo).

## 8. New references filed (research-references.md, Post-.336 sweep)

The Reward Model Selection Crisis (arXiv:2512.23067 — discrimination↔selection
decoupling), Scalable Best-of-N via Self-Certainty (arXiv:2502.18581),
Self-Consistency Boosts Calibration for Math Reasoning (arXiv:2403.09849),
Budget-aware Test-time Scaling via Discriminative Verification (arXiv:2510.14913).
