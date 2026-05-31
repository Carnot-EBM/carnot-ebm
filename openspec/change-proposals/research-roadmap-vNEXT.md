# Research Roadmap — Milestone 2026.05.330

**Verifier Cross-Domain Value, DE-CONTAMINATED — was ".329 math-only" a real
limitation or a failed positive control? Build the missing factual-grounding
verifier and re-test on realistic corpora.**

Planned 2026-05-31 (Claude Opus 4.8, outer-loop, on the autonomous-planner
directive). Pre-staged per the Pre-Staged Roadmap Convention.

---

## 1. What the previous milestone (.329) actually proved

`.329` asked the right product question — does the G2-reproduced verifier
ensemble (FoVer math step-error **AUROC 0.9131**, `paper_ready=true`) catch
errors a strong baseline misses, and **generalize beyond math** to code bugs
and factual hallucinations? Its headline verdict was **"math-only,
domain-bound"** (exp3576). Reading the underlying artifacts shows that verdict
is **a contaminated null — the positive control failed** (FALSE_NEGATIVE_RISK,
CLAUDE.md Reading-Results Discipline):

| .329 exp | domain | ensemble AUROC | best single verifier | model-confidence baseline | what it really means |
|---|---|---|---|---|---|
| exp3573 | code | **0.44** (worse than random) | 0.50 (inert) | 0.8992 | code-applicable verifiers **never fired** |
| exp3574 | facts | **0.50** (inert) | 0.50 (inert) | **1.0 (perfect)** | degenerate corpus + only 2 inert verifiers scored |

Two independent contamination signals:

1. **Degenerate corpora.** A model-confidence AUROC of **1.0** on facts (and
   0.90 on code) is the IMPLAUSIBLE_PERFECT signature: the negatives were
   trivially separable, so there was **no real headroom** for any verifier to
   demonstrate value. A null measured where confidence already scores 1.0 is
   uninformative.
2. **Inert / wrong verifier set.** Every per-verifier AUROC was exactly **0.50**
   — the verifiers did not fire. exp3574 scored only `SemanticConsistencyVerifier`
   + `IsingVerifier`; the actually factual-applicable verifiers
   (`semantic_energy`, `nla_verifier_v3`,
   `canonical_answer_vericot_grounding_pilot_v1`, `suppressed_retrieval_probe`,
   `tier0u/v/w` consistency) were **never in the ensemble**. exp3573's code
   ensemble at 0.44 likewise means the execution-applicable verifiers
   (`controlled_invariance_executor_v2`, `executable_monitor_runtime_adapter`,
   `ast_structure_verifier`) were not scoring the completions.

There is also an unresolved internal tension: exp3575 reported "second pair of
eyes confirmed" (`code_conditional_catch_rate=0.75`) **from an ensemble whose
code AUROC was 0.44** — a result that cannot both be true under a fair test and
must be re-derived once the verifiers actually fire.

**So `.329`'s real, defensible findings are narrower than its headline:** the
FoVer math headline (0.9131, G2-reproduced) stands; the "doesn't generalize"
claim is **not yet earned** — it was measured against a failed positive control.
This milestone earns (or refutes) it honestly.

### What stands unchanged going into .330
- **G1–G4 all met; `paper_ready=true`** (FoVer headline reproduced on a clean
  CI runner, G2 closed). North-star §2.
- **P0.1 is honest-negative** (energy *selection* has no headroom where
  self-consistency / strong classical baselines are near-optimal). Do NOT
  re-test Route-1/Route-2.
- Hardware: KV260 repeatedly SSH-unreachable (board down), PolarFire reachable,
  GateMate flash/smoke host-IO hangs (known blocker).

---

## 2. The three biggest gaps between current state and the PRD vision

The PRD north star is **"escape LLM hallucinations + autonomous directed
self-learning, energy as ground truth."** Against that:

1. **No working factual-grounding verifier (the core-motivation gap).** Carnot's
   *entire reason to exist* is catching factual hallucinations, yet `.329` shows
   the ensemble has **no verifier that fires on factual claims**. Constraint
   verifiers (SAT/Z3/AST/Ising) structurally cannot check facts. The SOTA recipe
   (retrieval + atomic-claim NLI entailment, per Mu-SHROOM / HalluSearch) is
   not yet wired into the ensemble. **This is the milestone's primary build.**

2. **The cross-domain generalization question is unanswered, not answered
   "no".** A peer EBM judge (arXiv:2505.14999) claims energy verifiers DO
   generalize OOD — but theirs is *trained* on reasoning-validity, where
   Carnot's is a fixed constraint ensemble. We must run the fair test (realistic
   corpora + applicable verifiers) before scoping the paper to "math-only".

3. **The product value prop ("second pair of eyes") is unmeasured under a fair
   test.** Whether the ensemble is *additive* to a strong model-confidence
   baseline — catching errors confidence misses — is the genuine commercial
   claim, and `.329` measured it against a degenerate corpus.

---

## 3. Architecture of the milestone

```
                 .329 NULL (contaminated)
                 "verifier is math-only"
                          |
        +-----------------+------------------+
        |  Phase A - DE-CONTAMINATE          |
        |  A1 diagnose the null (positive-   |
        |     control + verifier-inertia)    |
        |  A2 build REALISTIC corpora where  |
        |     confidence is NOT perfect      |
        +-----------------+------------------+
                          | realistic, headroom-bearing corpora
        +-----------------+------------------+
        |  Phase B - BUILD / WIRE THE        |
        |  RIGHT VERIFIERS                    |
        |  B1 score factual-APPLICABLE        |
        |     verifiers (which fire?)         |
        |  B2 prototype retrieval/NLI factual-|
        |     grounding verifier (the gap)    |
        +-----------------+------------------+
                          | verifiers that actually fire
        +-----------------+------------------+
        |  Phase C - THE FAIR PRODUCT TEST   |
        |  C1 corrected cross-domain re-      |
        |     measurement (math|code|facts)   |
        |  C2 additivity / second-pair-of-eyes|
        |     vs strong confidence (McNemar)  |
        +-----------------+------------------+
                          |
        +-----------------+------------------+
        |  Phase D - SELF-LEARNING, SYNTH,   |
        |  HARDWARE, OPS                      |
        |  FR-11 . synthesis . G-gate .       |
        |  KV260/PolarFire/GateMate . capstone|
        +------------------------------------+
```

**Either outcome is genuine learning** (and neither re-grinds an answered
question):
- *Verifiers fire on realistic corpora and beat/augment confidence* → the
  `.329` "math-only" verdict was an artifact; the verifier value generalizes →
  **a broader, stronger paper claim.**
- *Verifiers still don't fire even with the right set + realistic corpora* →
  "math-only" is now earned against a valid positive control → **a precise,
  defensible limitation that honestly scopes the verifier claim** (and motivates
  the trained-judge direction of arXiv:2505.14999 as future work).

---

## 4. Phases and experiments (14 tasks, exp3583–exp3596)

**Phase 0 — ops**
- exp3583 — archive .329, activate .330.

**Phase A — de-contaminate the .329 null**
- exp3584 — diagnose the null: confirm corpus degeneracy (confidence AUROC≈1.0)
  + verifier inertia (per-verifier=0.5); enumerate which verifiers are
  *applicable* per domain. Positive-control audit (FALSE_NEGATIVE_RISK).
- exp3585 — build a REALISTIC factual hallucination corpus where
  model-confidence is NOT a perfect detector (SOTA GGUF generation over
  TruthfulQA-style / open QA with fact-level labels, or fetch Mu-SHROOM/SHROOM).

**Phase B — build / wire the right verifiers**
- exp3586 — score the factual-APPLICABLE verifiers on the realistic corpus:
  which fire (per-verifier AUROC materially > 0.5)? Is the ensemble inert
  because of composition, or genuinely domain-bound?
- exp3587 — prototype a retrieval/NLI atomic-claim grounding verifier (the SOTA
  recipe) and evaluate it vs the confidence baseline on the realistic corpus.

**Phase C — the fair product test**
- exp3588 — corrected cross-domain re-measurement: math (0.9131) | code | facts,
  using the per-domain APPLICABLE verifier set + realistic corpora + strong
  confidence baseline. Honest verdict with a valid positive control.
- exp3589 — additivity / "second pair of eyes": does ensemble⊕confidence beat
  confidence alone? Conditional catch-rate + McNemar on realistic corpora.
  Resolves the exp3575/exp3576 tension.

**Phase D — self-learning, synthesis, hardware, ops**
- exp3590 — FR-11 continuous self-learning (mandatory): conservative-default on
  a fresh non-degenerate corpus, optionally calibrating the new factual verifier.
- exp3591 — cross-domain synthesis v2 + paper-claim scoping (does the corrected
  test broaden, confirm, or re-scope "math-only"?).
- exp3592 — G1–G4 gate-status synthesis v330 (paper_ready stays true; record the
  corrected verifier-generalization result).
- exp3593 — KV260 SSH continuity (Hardware-Task Continuity).
- exp3594 — PolarFire opportunistic reachability + continuity audit.
- exp3595 — GateMate continuity audit (documentation-only; flash/smoke hangs).
- exp3596 — capstone v330.

---

## 5. Dependency graph

```
exp3583 (activate)
   +-> exp3584 (diagnose) --+
   +-> exp3585 (corpus)  ---+--> exp3586 (score applicable verifiers)
                                  +-> exp3587 (factual grounding verifier)
                                         +-> exp3588 (corrected cross-domain)
                                                +-> exp3589 (additivity / McNemar)
                                                       +-> exp3591 (synthesis)
                                                              +-> exp3592 (G-gate)
                                                                     +-> exp3596 (capstone)
exp3590 (FR-11)            -- independent (self-learning mandate)
exp3593/3594/3595 (HW)     -- independent (hardware continuity)
```

Gates: exp3588 gated on exp3586 landing a non-inert verifier signal; exp3589
gated on exp3588. Corpus-availability handled by in-prompt PRECONDITIONS with
`blocked_*` fallbacks (not structural gates), so a missing corpus degrades
gracefully rather than cascade-blocking.

---

## 6. Hardware requirements

- **CPU-only** for all verifier-scoring + aggregation tasks (the
  `verifier_ensemble_against_cached_candidates` / `aggregation_from_upstream`
  substrates — cheap, externally reproducible, no GPU).
- **GPU / llama.cpp (SOTA GGUF)** ONLY for exp3585 corpus generation if a
  realistic labeled corpus must be synthesized (precondition-gated; prefers an
  existing/fetched corpus). MODEL_SPECS use the mandated SOTA GGUFs
  (`unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/gemma-4-31B-it-GGUF`,
  `unsloth/gemma-4-26B-A4B-it-GGUF`) via `cached_sota_pair()`.
- **FPGA boards** (KV260 via `ssh kria`, PolarFire via `ssh polarfire`, GateMate
  via DirtyJTAG): SSH-reachability / detect preconditions only, per the KV260
  SSH-Not-SD-Card Discipline. No bitstream rebuild this milestone.

---

## 7. Disciplines honored

- **Reading-Results / FALSE_NEGATIVE_RISK** — the milestone exists to supply the
  positive control the `.329` null lacked.
- **Adversarial Artifact Verification + Sample-Size Rigor** — all AUROC claims
  carry n≥100, ≥3 seeds, bootstrap CIs, class balance; surprising results
  cross-checked.
- **Inference-Substrate Declaration** — every task declares its substrate.
- **Pre-Launch Preconditions** — every compute/corpus/hardware resource is
  checked before use with a `blocked_*` fallback.
- **Principle-Annotated Artifact Fields** — every required field + gate carries a
  `principle:`.
- **Gemini-Default** — all tasks `agent_type: gemini`.
- **Paper-v6 Narrowing** — synthesis/capstone emit `paper_safe_claims` /
  `paper_forbidden_claims`; a domain-bound ensemble is a *scoped* claim, never a
  foundation-model claim; P0.1 stays honest-negative.
- **Hardware-Task Continuity** — one task per attached board, operator_override
  on the routine continuations.
- **Verdict Terminal-Prefix** — every `honest_verdict` starts `complete:`.
