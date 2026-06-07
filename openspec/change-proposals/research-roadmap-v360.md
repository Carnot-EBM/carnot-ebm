# Research Roadmap — Milestone 2026.06.360

**HARNESS-FIRST: bank the EBT energy-as-GENERATOR negative, and finish the two
forward bets the agent kept fabricating by building + unit-testing the live-model
harness BEFORE the measurement.**

Outer-loop plan (Claude Opus 4.8, 2026-06-06).

---

## 1. What the previous milestone (.359) proved

.359 re-issued the three operator-seeded forward bets with the .358 infra fixed
(import bug + poison-pretest cascade). The verdicts:

| Bet | Verdict | Trustworthy? |
|---|---|---|
| **EBT energy-as-GENERATOR (Thesis A, Phase-3)** | **FUNDAMENTAL** — at matched inference FLOPs, EBT global-beam=0.0 AND greedy-argmin=0.0 vs AR=0.94 (exp3882, clean 3673s, positive control passed); System-2 K-curve PLATEAU at 0.0 (exp3883). | **YES — clean, real, decisive.** |
| **Verifier MOAT scissor (DT-P2)** | INCONCLUSIVE + **FLAGGED** (exp3885, 35s). Corpus is now good (exp3884: 150 errors, ensemble AUROC 0.967) but the reasoner self-verify caught **0/150** errors -> degenerate AUROC=0.5. | **NO — broken harness.** |
| **Facts via graph-grounding** | `blocked_graph_verifier_not_invoked` + **FLAGGED** (exp3886, 11s, `model_invoked=False`, `n_items=0`). The graph verifier module exists but was never exercised. | **NO — stub, not invoked.** |

Standing mandates landed clean: FR-11 v24 INVARIANT_HELD (exp3888, AUROC 0.9075,
mem-contrib +0.0185); GateMate de-flagged (exp3889 OK); PolarFire/KV260 continuity
(exp3890 OK); capstone exp3891 `paper_ready=TRUE`, frozen 0.9131 unchanged, G1-G4 met.

**The decisive science:** **energy-as-GENERATOR is bounded** (FUNDAMENTAL), the
complement of the already-settled **energy-as-SELECTION** negative (P0.1). Both
operator-seeded "energy is the core mechanism for the foundation model" theses now
have a trustworthy negative. **What survives and is proven is the VERIFIER**
(FoVer 0.9131, frozen, independently reproduced — G2 closed).

**The decisive operational lesson:** the agent reliably succeeds when it reuses a
pre-built, working harness (EBT reused `thesis_a_part_b_scaled.py` -> real 3673s run)
and reliably FABRICATES when asked to implement a live-model invocation inside a thin
wrapper in one turn (moat scissor 35s degenerate; graph verifier 11s not-invoked;
the same mode as exp3862's 1.02s stub). **.360 is designed around this lesson.**

## 2. The three biggest gaps (current state vs PRD vision)

1. **The decisive Phase-3 negative is single-shot.** EBT FUNDAMENTAL (exp3882) is the
   verdict that closes the operator-seeded energy-as-generator thesis. Per the
   Adversarial-Confirmation discipline, a negative that feeds a *strategic* claim
   ("abandon the energy-core foundation-model thesis") deserves ONE independent
   replication before it is banked. exp3882 measured FUNDAMENTAL across seeds {1,2,3}
   (seeds 1,2 had AR headroom; seed 3 AR collapsed) in 3673s. The replication re-runs
   the *exp3882 measurement script verbatim* at 2 fresh seeds — an honest retrain (the
   scaled harness has no checkpoint reuse; the model is seed-coupled to its corpus),
   bounded to 2 seeds (~2500s) to stay under the 4800s codex cap.

2. **The verifier's DURABILITY (moat) and BREADTH (facts) are unmeasured** because the
   live-model harnesses fabricate. These are the two questions that decide the
   verifier's forward value — and the verifier is now the project's *only* surviving
   asset. They have never gotten a trustworthy verdict (moat: OOD .357, cascade .358,
   degenerate-reasoner .359; facts: stub .356/.359). The fix is not a new corpus or a
   new technique — it is a **tested harness**.

3. **The operator next-thesis decision is now forced.** With BOTH energy mechanisms
   bounded-negative and the verifier proven, the realistic forward direction is
   "verifier as a durable, broad external second-opinion layer," not "energy as a
   generator/selector." The loop cannot choose the next thesis; the capstone must
   surface this crossroads with the .360 evidence attached.

## 3. Design principle — HARNESS-FIRST (the load-bearing difference vs .356-.359)

For each live-model bet, .360 splits the work into two tasks:

- **(a) BUILD + UNIT-TEST the harness as a module** with a deliverable that is a
  *passing test asserting the harness produces non-degenerate output on a fixture*
  (a positive control on the HARNESS itself, not on the science). The
  reasoner-self-verify harness must catch a known injected arithmetic error on a tiny
  fixture and yield AUROC > 0.6; the graph-grounding verifier must flag a known
  hallucinated relation. These are CPU/fast and cannot fabricate a science result.
- **(b) RUN the measurement** by importing the tested harness, with the live SOTA
  model and a real duration floor.

This mirrors the only thing that worked: the EBT bet reused a real, debugged harness.
A thin-wrapper one-turn implementation is forbidden for live-model steps.

NO hard `gated_on` on the critical path (the .340 proven-safe disk-fallback pattern):
each downstream task reads its upstream artifact off disk in-script and emits
`blocked_upstream_*` if absent — a skipped upstream costs ONE task, never a cascade.
Every REQUIRED ARTIFACT FIELD value is emitted as a BARE scalar (the `principle:`
lines guide the agent; they are NOT a `{value,principle}` wrapper — the exp3871 bug).

## 4. Architecture / dependency graph

```
exp3892  archive .359 -> activate .360 ; GREEN-GATE (yaml parses, core pretest green,
         harnesses importable)                              [infra, codex]
   |
   |-- PHASE 1  Bank the EBT negative
   |     exp3893  EBT FUNDAMENTAL adversarial replication (reuse exp3882 script, 2 fresh
   |              seeds, retrain ~2500s < cap) -> confirms beam=argmin=0 vs AR>0.4   [GPU]
   |
   |-- PHASE 2  Verifier MOAT (harness-first)
   |     exp3894  BUILD+UNIT-TEST reasoner_self_verification harness module
   |              (fixture positive control: AUROC>0.6, catches injected error)   [CPU/smoke]
   |     exp3895  RUN moat scissor: tested harness (3894) x in-distribution corpus (3884),
   |              live Qwen3.6-35B, residual_catch + CI95 + Jaccard         [GPU] (disk-reads 3894+3884)
   |
   |-- PHASE 3  Facts via graph-grounding (harness-first)
   |     exp3896  BUILD+UNIT-TEST graph_grounding_fact_verifier module
   |              (fixture: flags a known hallucinated relation; locates RAGTruth corpus) [CPU/smoke]
   |     exp3897  RUN graph-grounding on facts corpus via tested module (3896),
   |              facts_catch_delta + per-item/span scores, real duration     [GPU] (disk-reads 3896)
   |     exp3898  facts COMPLEMENTARITY (disk-reads 3897 per-item scores)      [CPU]
   |
   |-- PHASE 4  Mandates + hardware + capstone
         exp3899  FR-11 v25 online independence-reweighting (research-program MANDATE) [CPU]
         exp3900  GateMate terminal-state confirmation (graduate if CLEAN_TERMINAL)    [HW]
         exp3901  PolarFire + KV260 consolidated continuity                            [HW]
         exp3902  Capstone .360 + FORCED operator next-thesis decision                 [aggregation]
```

## 5. Hardware requirements

- 2x RTX 3090 (CUDA) for exp3893 (EBT decode), exp3895 (35B self-verify), exp3897
  (graph-grounding model invocation). All GPU tasks run via `{project_root}/.venv/bin/python`
  (bare `python` has no torch -> silent CPU drop).
- GateMate A1-EVB-2M (DirtyJTAG) for exp3900; PolarFire + KV260 over SSH for exp3901.

## 6. Invariants carried

`paper_ready` stays TRUE (the milestone adds replication + durability + breadth
lenses, not a new headline); FoVer **0.9131 frozen**, never silently substituted;
verifier is **math-domain-bound** until facts-breadth is proven with a non-fabricated
run; energy-as-selection (P0.1) and energy-as-generation (EBT) are both **bounded
negatives** (different mechanisms, both real); never aggregate `flagged_adversarial`
artifacts; no external publication (operator-only).

## 7. Routing

All tasks `agent_type: codex` + `requires_codex: true` + `model: gpt-5.5`
(anti-wipeout; gemini crashes GPU workloads and 429-wiped .333/.355; standing
operator gemini<->codex flip authority 2026-06-05). GPU tasks add `requires_gpu: true`.
The EBT replication (exp3893) reuses the exp3882 measurement script verbatim at 2
fresh seeds (honest retrain, ~2500s) and is bounded to 2 seeds to stay under the
codex 4800s wall-clock cap (exp3882 ran 3 seeds in 3673s).

## 8. New literature folded in (Post-.359 sweep, see research-references.md)

- **arXiv:2606.05976 "The Self-Correction Illusion: LLMs Correct Others but Not
  Themselves"** — fresh direct corroborator of the moat premise; LLMs correct external
  claims but not their own traces. The repeated moat INCONCLUSIVE is a broken harness,
  not a refuted premise.
- **arXiv:2602.07594 "Learning to Self-Verify Makes LMs Better Reasoners"** —
  adversarial baseline: the moat positive control must be a STRONG (non-degenerate)
  reasoner self-verify; the .359 AUROC=0.5 control was correctly rejected by the gate.
- **arXiv:2603.27752 "Retromorphic Testing / Hierarchical Verification for RAG"** —
  persist SPAN/per-item scores in exp3897 so complementarity (exp3898) is computable.
- **arXiv:2512.01659 HalluGraph** — the entity-grounding + relation-preservation
  decomposition exp3896 must actually implement (not a 1-second stub).
