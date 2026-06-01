# Research Roadmap — Milestone 2026.06.332

**Title:** Finish the de-contamination *for real* — RUN the facts + code rows (not block them),
answer cross-domain generalization with a VALID positive control, and resume NEW breadth:
position Carnot vs the SOTA weak-verifier peer (Weaver) and find where a verifier actually beats
self-consistency.

**Author:** Outer-loop / planning agent (Claude Opus 4.8), 2026-05-31.
**Status:** pre-staged roadmap (preserved by the conductor per Pre-Staged Roadmap Convention).
**Milestone doc consumed by:** `research-roadmap-next.yaml`.

---

## 1. What the previous milestone (2026.06.331) actually proved

`.331` was the THIRD attempt (after `.329`, `.330`) to answer one question:
**does the verifier ensemble's error-detection value generalize beyond MATH — to CODE and FACTS?**
It again **did not finish honestly**.

| Exp | Intent | Honest outcome |
|---|---|---|
| exp3598 | audit the `.330` AUROC=1.0 grounding number | **LEAK CONFIRMED** — verifier keyed on the answer string containing 'H' (a perfect label-correlate in the toy corpus), not real grounding |
| exp3599 | build factual corpus v2 with held-out evidence | **wrote an empty `{}` artifact** — but the corpus DATA landed (`data/realistic_factual_corpus_v2.jsonl`, 200 records, real `evidence_passage`) |
| exp3600 | real NLI grounding verifier | `blocked_gate_check_failed` (exp3599 gated field read `None`) |
| exp3601 | **CENTERPIECE** cross-domain re-measurement | `blocked_gate_check_failed` (same cascade) — never ran |
| exp3602 | math→code PRM transfer | `blocked_no_labeled_code_corpus`; `verifiers_fired_on_code=false` |
| exp3603 | additivity / second-pair-of-eyes / McNemar | **never ran** |
| exp3605 / exp3610 | synthesis + capstone | DECLARED "`.329` math-only CONFIRMED / value math_only_earned" — **from blocked non-math rows** |

**The load-bearing finding:** the capstone asserted a NULL ("math-only earned") while the facts and
code rows were **blocked, not measured**. That is exactly the FALSE_NEGATIVE_RISK failure mode the
project's own discipline forbids: *a null is not a finding unless a valid positive control ran.* Only
MATH (FoVer 0.9131, frozen, G2-reproduced) was actually measured. The de-contamination question is
**STILL OPEN.**

**What's genuinely true after `.331`:**
- `paper_ready = TRUE` (G1-G4 met; FoVer 0.9131 G2-reproduced on a clean CI runner, run 26725185125).
  This milestone must NOT regress it.
- P0.1 is **honest-negative** (energy-vs-AR bounded on tested corpora because SC/strong classical
  baselines are near-optimal). The Depth-Over-Breadth forcing function **retired 2026-05-31**. Do NOT
  re-test Route-1/Route-2.
- The `.330` gate-cascade root cause is known: a `gated_on` field emitted as a `{value, principle}`
  dict instead of a bare scalar (`conductor_gates.py:_eval_op` does not unwrap). **Forward rule: every
  gated field is a BARE top-level scalar** (`feedback_gated_fields_must_be_bare`).

## 2. The three biggest gaps between current state and the PRD vision

1. **The core-motivation claim (escape LLM hallucinations) is still unmeasured on FACTS.** The product
   thesis — a "second pair of eyes" that catches errors the model's own confidence misses — has only
   ever been measured on math. Facts and code have been blocked three milestones running by
   *infrastructure*, not by a real limitation. Closing this is the single highest-value move and it is
   now tractable (the corpus exists; the cascade bug is understood).
2. **No positioning against the SOTA peer.** Weaver (arXiv:2506.18203) executes Carnot's exact premise
   (weighted weak-verifier ensemble, 15 verifiers) and beats o3-mini. Carnot has two clean
   differentiators Weaver leaves open — **correlation-aware weighting** (Weaver never measures the
   inter-verifier correlation matrix it assumes away) and **online weight adaptation** (Weaver fits
   weights once). Until measured, "Carnot adds something Weaver doesn't" is an unproven claim.
3. **We don't know WHERE a verifier beats self-consistency.** P0.1 showed SC is near-optimal on the
   tested corpora, so a verifier has no headroom there. arXiv:2510.14913 frames this precisely (hybrid
   verification+SC is best under budget) and says the productive move is corpora where **oracle > SC**.
   We have never built such a positive-control corpus.

## 3. Architecture / dependency shape

```
            ┌─────────────────────────────────────────────────────────────┐
 Phase 0    │ exp3611  archive .331 (record the blocked-not-measured truth) │
            └───────────────────────────────┬─────────────────────────────┘
                                             │
 Phase 1 — FINISH THE DE-CONTAMINATION (run the rows, don't block them)
   exp3612  validate facts corpus v2 (200 recs, evidence) -> emit BARE fields   [claude]
   exp3613  build/label CODE corpus + confirm exec verifiers FIRE + math->code  [claude]
            transfer stress-test (2506.00027 vs ThinkPRM 2504.16828)
   exp3614  CENTERPIECE: cross-domain re-measurement math|code|facts vs         [claude/opus]
            strong confidence baseline — graceful per-row degradation,
            VALID positive control (rows RUN, headroom present), leak-free
            real NLI grounding verifier on corpus v2
   exp3615  additivity / second pair of eyes / McNemar  (gated_on 3614 valid)
                                             │
 Phase 2 — NEW BREADTH (sanctioned by north-star §retirement)
   exp3616  Weaver peer comparison (2506.18203) + inter-verifier CORRELATION matrix
   exp3617  headroom + hybrid: build a corpus where oracle > SC; measure
            verifier-vs-SC + the hybrid detector under compute budget (2510.14913)
                                             │
 Phase 3 — CONTINUOUS SELF-LEARNING (mandatory, every milestone)
   exp3618  FR-11 v7: online CORRELATION-AWARE verifier weighting without collapse
            (Weaver's no-online-adaptation gap + RC2RLHF buffer + uncertainty gate)
                                             │
 Phase 4 — HARDWARE CONTINUITY (one task per attached board)
   exp3619  KV260 (was SSH-unreachable in .331 — re-check + honest state)
   exp3620  PolarFire (reachable in .331 — continuity)
   exp3621  GateMate (openFPGALoader missing in .331 — detect/audit only)
                                             │
 Phase 5 — SYNTHESIS
   exp3622  cross-domain synthesis v4 — CORRECT the .331 record (declared
            "confirmed" from blocked rows); state the now-fairly-measured scope
   exp3623  capstone v332 + G1-G4 gate synthesis (paper_ready must stay true)
```

Only one `gated_on` edge in the whole milestone (exp3615 on exp3614's bare `positive_control_valid`),
deliberately — over-gating is exactly what cascade-blocked `.330`/`.331`. The centerpiece (exp3614)
is NOT gated; it degrades gracefully per-row so MATH always lands even if a non-math corpus is missing.

## 4. Phase descriptions

**Phase 1 — finish the de-contamination.** This is the milestone's spine. The apparatus already
exists; the failures were artifact-write + over-gating. exp3612 validates the 200-record facts corpus
(evidence independence: the passage must not be a function of the label; confidence headroom AUROC in
(0.5, 0.95)) and re-emits a clean artifact with **bare** gated booleans. exp3613 builds the labeled
code corpus from the existing `experiment_1999_code_verification_humaneval.json` and **wires the four
execution-applicable verifiers** (`controlled_invariance_executor_v2`, `executable_monitor_runtime_adapter`,
`ast_structure_verifier`, `code_structural_dependency_verifier`) so they actually score completions
(`.331` had them inert), folding in the math->code transfer stress-test. exp3614 is the centerpiece:
the math|code|facts generalization table vs a strong confidence baseline, with a VALID positive
control (each non-math row actually runs and has a confidence-baseline AUROC < 0.95). exp3615 is the
product claim — does fusing the ensemble with confidence catch errors confidence alone misses
(conditional catch-rate + McNemar).

**Phase 2 — new breadth.** exp3616 reproduces Weaver-style weak-supervision weighting on a shared
corpus and measures the **inter-verifier correlation matrix** Weaver assumes away — the differentiation
evidence. exp3617 builds a positive-control corpus where the oracle materially beats self-consistency
(the headroom P0.1 lacked) and measures the verifier-vs-SC margin + the hybrid detector under a fixed
compute budget.

**Phase 3 — self-learning (mandatory).** exp3618 extends FR-11 to **online correlation-aware**
verifier weighting (down-weight verifiers that are redundant given others, not just noisy ones),
gated by a conservative-default / uncertainty rule that provably prevents collapse (control arm
collapses; deploy arm holds).

**Phase 4 — hardware continuity.** One task per attached board (KV260 / PolarFire / GateMate) per the
Hardware-Task Continuity Discipline; SSH-reachability preconditions only for KV260/PolarFire,
detect-only for GateMate.

**Phase 5 — synthesis.** exp3622 corrects the `.331` record (which declared "confirmed" from blocked
rows) and states the now-fairly-measured scope. exp3623 capstones + re-checks G1-G4 (paper_ready must
remain true).

## 5. Self-learning coverage

exp3618 (FR-11 v7, online correlation-aware weighting without collapse) satisfies the mandatory
"every milestone advances continuous self-learning" requirement, and is tied to a NEW research thread
(Weaver's no-online-weighting gap) rather than a routine vN+1 re-run.

## 6. Hardware requirements

- exp3614 facts row uses a small NLI cross-encoder (transformers) on the existing corpus — GPU
  helpful, CPU acceptable. No live 35B generation required (corpus already built).
- exp3613/3616/3617 score cached corpora — CPU.
- KV260 (`ssh kria`), PolarFire (`ssh polarfire`), GateMate (`openFPGALoader -c dirtyJtag --detect`).

## 7. Models

Headline-eligible LLM work uses the mandated SOTA GGUFs via `cached_sota_pair()`
(`Qwen3.6-35B-A3B-GGUF` / `gemma-4-31B-it-GGUF` / `gemma-4-26B-A4B-it-GGUF`). Most `.332` tasks score
**cached corpora** and load no LLM (the honest, cheap, externally-reproducible path); the facts row's
NLI verifier is a small entailment checkpoint, not an LLM.

## 8. Discipline compliance

- **FALSE_NEGATIVE_RISK:** the centerpiece refuses to assert a null without a valid positive control
  (rows run + headroom). This is the explicit repair of the `.329`/`.330`/`.331` failure mode.
- **Gated-fields-bare:** every `gated_on` field is emitted as a bare scalar.
- **Verifier Authenticity:** the NLI grounding verifier must invoke a real entailment model OR disclose
  a text-statistical proxy in its docstring; an AUROC of exactly 1.0 is treated as a leak.
- **Gemini-Default:** experiment tasks default to gemini; exp3612/3613/3614 carry `requires_claude:
  true` because gemini demonstrably failed this exact category in `.331` (empty artifact, inert
  verifiers, gate cascade) and the work is multi-file + judgment-heavy (leak detection, evidence
  independence).
- **Paper-v6 Narrowing / paper_ready:** a domain-bound ensemble is a SCOPED claim, never a
  foundation-model claim; never cite a leaking 1.0; G1-G4 must not regress.
- **Hardware-Task Continuity:** one task per attached board.
- **P0.1 honest-negative + Depth-Over-Breadth retired:** no energy-vs-AR re-test.
