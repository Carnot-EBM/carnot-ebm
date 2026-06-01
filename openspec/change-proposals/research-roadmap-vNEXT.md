# Research Roadmap — Milestone 2026.06.335

**Status:** Proposed (outer-loop / Opus 4.8 pre-staged, 2026-06-01)
**Milestone doc for:** `research-roadmap-next.yaml` (`milestone: 2026.06.335`)
**Prior milestone:** 2026.06.334 (the cross-domain de-contamination science that FINALLY ran)

---

## 1. What the previous milestone proved

`.334` was the milestone that — after FIVE stalled attempts (`.329`–`.333`, the last a total
gemini-quota wipeout) — finally executed the cross-domain de-contamination science, routed entirely to
**codex (`requires_codex`)** on a verified-working backend. All 14 tasks landed real artifacts. The
honest scientific outcomes (read via `scripts/summarize_artifact.py`):

| Finding | Evidence | Verdict |
|---|---|---|
| The `.329` **"math-only" was a contamination ARTIFACT** | exp3642 centerpiece, valid positive control | `generalizes_to_code_not_facts` |
| Math headline holds, frozen | exp2837/2850, AUROC **0.9131** | G1 met |
| **CODE generalizes** | exp3641 verifiers fire; math->code PRM transfer (arXiv:2506.00027) | `code_generalizes=true` |
| **FACTS does NOT generalize** | grounding AUROC **0.6495** < confidence **0.7446** | `facts_generalize=false` |
| - but facts used a **text-statistical PROXY, not a real NLI model** | `nli_substrate=disclosed_text_statistical_proxy` | **the #1 gap** |
| **Second pair of eyes is REAL** | exp3643 fused 0.822 vs confidence 0.536 (delta +0.286) | `fusion_wins` |
| **Verifier beats SC where headroom exists** | exp3645 oracle 0.867 / SC 0.70 / hybrid 0.767 | `hybrid_wins_under_budget` |
| **Trained EBM judge does NOT solve OOD** | exp3646 OOD 0.673 < confidence 0.882 (CPU-only, tiny) | `also_math_only` |
| **Correlation-aware weighting HURT** | exp3644 corr-aware 0.635 << Carnot 0.919 (delta -0.236) | paradox, needs diagnosis |
| `paper_ready` stays **TRUE** | G1-G4 met; FoVer G2-reproduced on CI run 26725185125 | invariant |
| P0.1 stays **honest-negative** | Depth-Over-Breadth retired 2026-05-31 | invariant |

**One-line summary:** Verifier value is now defensibly **math + code** (not math-only); the
second-pair-of-eyes product value is **real and strong**; but **facts** — the original
escape-LLM-hallucinations mission — has only been tested with a weak text-statistical proxy, the
trained-judge cross-domain path was under-resourced, and the correlation-aware weighting result is a
paradox. `paper_ready` is true and must not regress.

## 2. The three biggest gaps between current state and the PRD vision

1. **FACTS is untested with a real grounding verifier (CORE MISSION).** The PRD mission is "escape LLM
   hallucinations." The one domain that maps directly to factual hallucination — grounding a model
   answer against held-out evidence — was measured only with a text-statistical token-support proxy
   (AUROC 0.6495, lost to confidence). The literature recipe (Span-Level / DeBERTa-NLI atomic-claim
   decomposition, arXiv:2504.18639; FinGround type-routed verification, arXiv:2604.23588) was never
   tried. **Until a real model-based NLI grounding verifier is measured, "facts does not generalize" is
   not an earned conclusion.**

2. **The now-defensible claims are single-corpus / under-resourced.** Code generalization rests on ONE
   imbalanced corpus (exp3641: 296 errors / 24 correct, n=320). The trained-judge OOD test used a
   CPU-only tiny numpy head. Both need a second corpus / real substrate before they harden into paper
   claims.

3. **The correlation-aware weighting paradox is unresolved.** exp3644 + exp3647 down-weighted "redundant"
   verifiers and it HURT (-0.236 AUROC). Either inter-verifier correlation is genuinely harmless on this
   ensemble (refuting the joint-null-space concern), or the redundancy penalty is mis-specified vs proper
   dependency-aware weak supervision (arXiv:1903.05844). This blocks the Weaver-differentiation claim.

## 3. Milestone theme

> **Make the FACTS row REAL, harden the defensible math+code + second-pair-of-eyes claims, and resolve
> the correlation-aware weighting paradox — without regressing `paper_ready`.**

Every task either (a) advances the headline (extends cross-domain scope or the product detector), or
(b) closes a measurement gap that the `.334` findings exposed. No `vN+1` re-measurement of an already-
measured artifact unless it answers a question the prior version did not (north-star section 1).

## 4. Phase structure

```
Phase A — Transition & backend (exp3652, exp3653)
  3652 archive .334 / activate .335 (records the math+code+second-pair-of-eyes wins honestly)
  3653 backend-state diagnostic (probe gemini; recommend routing; science stays on codex)

Phase B — FACTS made real (THE core gap) (exp3654, exp3655)
  3654 build a REAL model-based NLI atomic-claim grounding verifier (DeBERTa/MiniLM-NLI +
       SRL/atomic decomposition, arXiv:2504.18639/2604.23588); leak guards; standalone AUROC on v3 corpus
  3655 re-measure the FACTS row with the real NLI verifier vs confidence — does factual grounding
       generalize once a real model is used?   [gated_on 3654 nli_grounding_built]

Phase C — Harden defensible claims + product (exp3656, exp3657, exp3658)
  3656 diagnose the correlation-aware weighting paradox with dependency-aware weak supervision (1903.05844)
  3657 build the deployable fused "second pair of eyes" detector (calibrated, per-domain operating points)
  3658 replicate code generalization on a SECOND balanced code corpus (MBPP/LiveCodeBench)

Phase D — Foundation-model path + self-learning (exp3659, exp3660)
  3659 trained-EBM-judge OOD v3 with a REAL model substrate (cached_sota_pair GGUF embedding, GPU)
       — does a properly-resourced trained judge beat confidence OOD?   [retire_if_same_verdict]
  3660 FR-11 continuous self-learning v9 — online learning of the FUSION weights per-domain (forward
       diff from v8's online correlation-aware weighting); collapse control arm mandatory

Phase E — Hardware continuity + capstone (exp3661-exp3664)
  3661 KV260 SSH-reachability continuity (UNREACHABLE 4 milestones -> operator-action)
  3662 PolarFire opportunistic continuity
  3663 GateMate continuity audit (doc-only; openFPGALoader missing)
  3664 Capstone v335 + G1-G4 gate synthesis (does facts generalize w/ real NLI? paper_ready stays true)
```

### Dependency graph

```
3652 - 3653 -+- 3654 -- 3655 -----------------------+
             +- 3656                                  |
             +- 3657                                  +- 3664 (capstone)
             +- 3658                                  |
             +- 3659                                  |
             +- 3660                                  |
             +- 3661 / 3662 / 3663 (hardware) --------+
```
Only 3655 has a hard `gated_on` (3654's bare `nli_grounding_built`). All other Phase B-E tasks run
independently and degrade gracefully per the `.334` lesson (never assert a null from a blocked row).

## 5. Architecture touchpoints

- **`python/carnot/verify/`** — a NEW model-based NLI grounding verifier (exp3654). The leaky
  `retrieval_nli_grounding_verifier.py` (exp3587 'H'-token proxy) is NOT reused; its substring logic was
  already deleted in `.334`. The new verifier must invoke a real transformers NLI checkpoint OR carry the
  explicit `pcib_probe.py`-style disclosure if a real checkpoint is unavailable. Verifier Authenticity
  Discipline applies (docstring matches implementation; no adversarial gaming).
- **`python/carnot/pipeline/`** — the fused "second pair of eyes" detector (exp3657) is a deployable
  combination of ensemble energy + model confidence; it is the Phase-1 product surface.
- **FR-11 self-learning module** — exp3660 extends the online-weighting loop to fusion weights.
- No changes to `scripts/research_conductor.py` (forbidden).

## 6. Hardware requirements

- exp3654/3655/3659 prefer GPU (RTX 3090) for the NLI checkpoint + trained-judge substrate; all degrade
  to CPU with an honest note. exp3659 declares `requires_gpu: true`.
- KV260 / PolarFire / GateMate continuity tasks per Hardware-Task Continuity Discipline (SSH reachability
  only for KV260 — host SD-card checks permanently retired).

## 7. Models

SOTA GGUF per CLAUDE.md where live LLM inference is genuinely needed: `unsloth/Qwen3.6-35B-A3B-GGUF`,
`unsloth/gemma-4-31B-it-GGUF`, `unsloth/gemma-4-26B-A4B-it-GGUF` via `cached_sota_pair()` (exp3659
trainable substrate). Most `.335` tasks are `verifier_ensemble_against_cached_candidates` (score cached
corpora, no LLM load) or `aggregation_from_upstream_artifacts`; the NLI grounding verifier uses a
DeBERTa/MiniLM-NLI checkpoint (a verifier substrate, not a generation model).

## 8. Backend routing

Gemini quota was exhausted through `.333` (the wipeout). `.334` routed every experiment to
**codex (`gpt-5.5`, `requires_codex: true`)** and all 14 tasks succeeded. `.335` keeps that routing
(CLAUDE.md Gemini-Default `requires_codex` positive criterion #3 — "gemini quota exhausted") until the
exp3653 diagnostic confirms gemini recovery and the operator flips `CODEX_FORCE_EXPERIMENTS` off.

## 9. Invariants (do NOT regress)

- `paper_ready == true` (G1-G4 met). The capstone re-runs `scripts/publication_gate.py --json` and asserts.
- P0.1 stays **honest-negative** — do NOT re-test energy-vs-AR Route-1/Route-2 (Depth-Over-Breadth retired).
- Every `gated_on` value is a **bare scalar** (`feedback_gated_fields_must_be_bare`).
- **No poison tests** (the `.325/.326/.332` cascade): parametrize pytests over the script's honest
  verdicts on realistic synthetic fixtures; never hard-assert one success string against a real corpus;
  never use `Q\d+/R\d+/H\d+` placeholder tokens. Run your own test green before finishing.
- Leak guards on the facts row: the grounding verifier scores `(model_answer, evidence_passage)` ONLY,
  never the gold answer or label; an AUROC >= 0.99 on n>=200 is a RED-FLAG leak, not a win.
