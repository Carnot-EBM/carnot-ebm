# Research Roadmap v355 Change Proposal — Adjudicate the Verifier-MOAT Durability (P2), properly this time

**Milestone:** 2026.06.355
**Planned:** 2026-06-05 (Claude Opus 4.8, outer-loop planning agent)
**Milestone doc:** this file

## What the previous milestones proved (the converged state)

The project is **converged** on its product. After ~70 milestones of energy-foundation
exploration:

- **The sole defensible headline is the FoVer 0.9131 math step-level error verifier** (G1–G4 all
  pass, `paper_ready=TRUE`, G2 independently reproduced on a clean CI runner 2026-05-31). Frozen.
- **Both energy-foundation routes are BOUNDED.** Energy-as-selector (P0.1) does not beat AR/SC.
  Energy-as-generator / Thesis-A (EBT) is discriminative-PASS but generative-BOUNDED at scale
  (operator's direct runs; exp3766). Findings, not doomed-rerun ids.
- **The verifier is math-domain-bound** — earned-negative on facts (RAGTruth + NLI), weak on code.
  It is a *reasoning* verifier, not a general fact-checker.
- **The loop will NOT self-seed the next paradigm** (Verification Trap, Deep-Think P3, 2026-06-03).
  The operator seeds the next foundation-model route or explicitly freezes; the loop scaffolds.

## The ONE load-bearing question left unanswered: does the moat survive?

The Deep-Think post-bounded round (2026-06-03) reframed the whole project's forward value as a
single question (P2): **is Carnot's verifier a durable moat, or is it about to be subsumed by
frontier reasoners' own self-verification?** The product is worth something only if Carnot
*independently* catches the step-errors a strong reasoner's self-verification MISSES. If it mostly
re-catches what the reasoner already catches, the value erodes the moment o1/o3-class models ship.

`.354 tried to answer this at scale (exp3844: the error-independence scissor, N≥1000) — and
**BLOCKED**: `blocked_fover_balanced_corpus_not_available`. The cached FoVer v4 corpus is 98%
correct-heavy (1286/1308 correct), so the *residual error set* it could produce is ~22 items —
far too few for a defensible residual-catch CI. The only prior signal is exp3827's 100-item
subset (`residual_catch=0.9000`, `overlap=0.5000`) — suspiciously round numbers that the
Adversarial-Confirmation Discipline explicitly forbids citing until replicated at scale.

So the project's single most important forward claim rests on one small-sample run with a missing
at-scale confirmation. `.355 fixes that — and does it the way the 2026 literature says it must be
done, not the naive way.

## What the literature says about how to measure this (the .355 planning sweep)

The `.355 research sweep (research-references.md, 2026-06-05) surfaced both the support and the
load-bearing refutation risk:

- **SUPPORTS the moat — arXiv:2602.03485 (Self-Verification Dilemma):** strong reasoners'
  self-verification is "the vast majority confirmatory rather than corrective, rarely identifying
  errors" → a residual error set demonstrably persists.
- **SUPPORTS — arXiv:2603.17775 (CoVerRL):** self-consistency collapses into a destructive
  consensus that reinforces confident-wrong majorities → exactly where an *independent* external
  verifier is required.
- **THE REFUTATION RISK — arXiv:2604.07650 (Behavioral Entanglement):** verifiers that share an
  LLM lineage with the reasoner produce *correlated failure masquerading as independence*. **A
  naive scissor that just counts "residual caught" can report a fake moat if the verifier and the
  reasoner share blind spots.** The independence must be MEASURED, not assumed. Carnot's claim
  survives only because its core verifiers are NON-LLM substrates (Z3/AST/SAT/energy) — but that
  decorrelation has to be demonstrated.
- **THE COMPLEMENTARITY BAR — arXiv:2504.16828 (ThinkPRM):** a single strong generative PRM is
  hard to out-AUROC. The moat must therefore rest on *complementary catch* (error-independence),
  not on beating a strong single judge on AUROC.
- **THE CORPUS — arXiv:2501.03124 (PRMBench):** 6,216 problems / 83,456 step labels across 9
  fine-grained error axes — a balanced successor to ProcessBench whose axes also enable a
  per-error-axis independence decomposition.

`.355 turns this into a four-experiment adjudication: build the right corpus, run the scissor at
scale with bootstrap CI + positive controls, **measure the independence** (not assume it), and
test complementarity against a strong-single-PRM baseline.

## The three biggest gaps between current state and the PRD vision

1. **The moat is asserted from one 100-item run with the at-scale confirmation blocked.** Build
   the balanced corpus that blocked exp3844, then run the scissor properly (N≥1000, bootstrap
   CI95, two positive controls) AND audit verifier-vs-reasoner independence per 2604.07650 so the
   moat is *measured*, not assumed. (`.355 Phase 1 — the spine.)
2. **The verifier is earned-negative on FACTS and re-testing without a NEW architecture is
   forbidden** (memory project_verifier_domain_bound). Graph-RAG / KG-alignment grounding
   (MemGraphRAG 2606.00610, HalluGraph 2512.01659, GraphRAG-attention 2512.09148) is a
   mechanistically *different* verifier — the one route with a plausible mechanism for the facts
   domain. Prototype-first. (`.355 Phase 2.)
3. **Continuous self-learning (research-program.md MANDATE) has never operated on the moat
   itself.** 2604.07650's independence-reweighting (+4.5% over majority vote) is the exact Tier-1
   online mechanism Carnot should run on its own ensemble: upweight verifiers that catch the
   residual *other* verifiers miss. The mandated self-learning task and the moat thesis become the
   same task. (`.355 Phase 3.)

## Architecture (what .355 touches)

```
  PRMBench (2501.03124, 9 error axes) + FoVer v3 (balanced ~47% incorrect)
                              │
                exp3846  build BALANCED step-error corpus
                (N>=1000, >=100 incorrect, 9-axis labels, seed-controlled)
                              │  (gated: n_incorrect_steps >= 100)
                              ▼
   ┌───────────── exp3847  SCISSOR AT SCALE (SUPERSEDES exp3844) ─────────────┐
   │  Qwen3.6-35B self-verify  ──┐                                            │
   │                             ├─► residual = gold-incorrect MISSED by reasoner
   │  Carnot k=15 ensemble    ───┘   residual_catch_rate + bootstrap CI95     │
   │  positive controls: reasoner AUROC in [.55,.95]; ensemble AUROC ~0.913   │
   └──────────────┬───────────────────────────────┬──────────────────────────┘
                  │ (gated: n_residual_errors>=30) │
                  ▼                                 ▼
   exp3848 INDEPENDENCE AUDIT (2604.07650)   exp3849 ThinkPRM COMPLEMENTARITY
   error-mask correlation reasoner vs         does the cheap NON-LLM ensemble add
   ensemble; per-9-axis decomposition;        catch a strong generative PRM misses?
   real decorrelation or shared blind spot?   (complementary, not head-to-head AUROC)

  FACTS domain (earned-negative) ── NEW architecture ──────────────────────────
   exp3850 graph-grounding fact-verifier PROTOTYPE (RAGTruth)  ──(gated:delta>0)──►
   exp3851 facts complementarity with the existing ensemble

  SELF-LEARNING (mandate) ─ exp3852 FR-11 v22 online independence-reweighting (Tier-1, CPU)
  LDT sharpening ─────────── exp3853 is the 0.010 margin-over-random real? (score-matched control)
  HARDWARE (continuity) ──── exp3854 GateMate Ising flash · exp3855 PolarFire smoke v3
  CAPSTONE ───────────────── exp3856 moat-durability verdict; paper_ready stays TRUE; 0.9131 frozen
```

## Phases

**Phase 0 — Activation (1 task).** exp3845 archive `.354 honestly (LDT-lattice LATTICE_VIABLE but
margin-over-random=0.010; scissor-at-scale BLOCKED on corpus) and activate `.355.

**Phase 1 — Verifier-moat durability, the spine (4 tasks).**
- exp3846 — build a BALANCED step-error corpus (PRMBench primary, FoVer v3 fallback): N≥1000,
  ≥100 gold-incorrect steps, 9 error-axis labels preserved, seed-controlled. The precondition that
  blocked exp3844. (codex, CPU/network.)
- exp3847 — SCISSOR AT SCALE (SUPERSEDES exp3844): Qwen3.6-35B self-verification vs Carnot k=15
  ensemble; residual_catch_rate with bootstrap CI95; two positive controls (reasoner AUROC
  non-degenerate, ensemble AUROC reproduces frozen 0.9131±0.02). (codex, live GPU.)
- exp3848 — INDEPENDENCE AUDIT per 2604.07650: error-mask correlation between reasoner and
  ensemble, per-9-axis decomposition; is the independence real or a shared blind spot? (claude,
  CPU re-score.)
- exp3849 — ThinkPRM COMPLEMENTARITY: does the cheap NON-LLM ensemble catch errors a strong
  generative PRM (the Qwen3.6 long-CoT self-verify, reused) misses? Complementary catch, not
  head-to-head AUROC. (codex, CPU reuse.)

**Phase 2 — Facts domain via a NEW architecture (2 tasks).**
- exp3850 — graph-grounding fact-verifier PROTOTYPE on RAGTruth (MemGraphRAG/HalluGraph-inspired
  KG-alignment), prototype-first. (claude+opus.)
- exp3851 — facts complementarity: does the graph verifier catch factual hallucinations the
  math-bound ensemble misses? (claude, gated on exp3850 signal.)

**Phase 3 — Self-learning + LDT sharpening (2 tasks).**
- exp3852 — FR-11 v22 ONLINE independence-reweighting (the mandated self-learning task; Tier-1 CPU
  counter updates): upweight verifiers that catch the residual other verifiers miss, per 2604.07650.
- exp3853 — LDT-lattice margin sharpening: exp3833 landed LATTICE_VIABLE but
  margin-over-random=0.010 — is the sound-elimination edge REAL (preferential sparing of correct
  candidates beyond the score distribution) under a SCORE-MATCHED random control + bootstrap CI?

**Phase 4 — Hardware continuity (MANDATORY, 2 tasks).**
- exp3854 — GateMate A1 Ising-tile flash + sample-level timing (terminal-state attempt; toolchain
  verified, himbaechel invocation corrected). (claude+opus, hardware.)
- exp3855 — PolarFire SoC smoke v3, precondition-gated SSH round-trip + hash-verified workload
  (terminal-state attempt). (claude+opus, hardware.)

**Phase 5 — Capstone (1 task).** exp3856 — aggregate the moat-durability verdict (the headline:
does the moat survive at scale with MEASURED independence?), graph prototype outcome, self-learning,
hardware. `paper_ready` stays TRUE; frozen 0.9131 unchanged.

## Dependency graph

```
exp3845 (activate)
  └─ exp3846 (corpus) ─gate:n_incorrect>=100─► exp3847 (scissor) ─gate:n_residual>=30─┬─► exp3848 (independence)
                                                                                       └─► exp3849 (ThinkPRM compl.)
exp3850 (graph prototype) ─gate:facts_catch_delta>0─► exp3851 (facts complementarity)
exp3852 (FR-11 v22)   exp3853 (LDT sharpen)   exp3854 (GateMate)   exp3855 (PolarFire)
  └────────────────── all feed ──────────────────► exp3856 (capstone)
```

## Hardware requirements

- **exp3847** — internal RTX 3090 (CUDA) + cached `unsloth/Qwen3.6-35B-A3B-GGUF` (fallback
  `gemma-4-26B-A4B-it-GGUF`) via llama.cpp. Single heavy GPU task; sequenced first among GPU work
  to avoid contention with hardware tasks.
- **exp3850** — optional small GPU / CPU embeddings for KG alignment; designed CPU-first to avoid
  GPU contention with the scissor.
- **exp3854** — GateMate A1-EVB-2M over DirtyJTAG (`openFPGALoader -c dirtyJtag`); yosys 0.64
  `synth_gatemate` → `nextpnr-himbaechel --device CCGM1A1` → `gmpack` (NOT `nextpnr-gatemate`).
- **exp3855** — PolarFire SoC Discovery Kit via `ssh polarfire` (precondition: ssh round-trip).
- KV260 is GRADUATED to terminal state (kv260_terminal=True, 2026-05-21) — no `.355 task required.

## Routing

- **codex (gpt-5.5):** exp3846 (dataset pipeline), exp3847 (live GPU; gemini crashes GPU workloads
  per exp3703; verdict is a deterministic CI gate so codex suffices), exp3849 (formulaic reuse/agg).
- **claude (opus):** exp3850 (NEW-architecture prototype, open-ended judgment), exp3854/exp3855
  (hardware tool choreography, opus + max_turns 100 per the hardware-rescue routing rule).
- **claude (sonnet, default):** exp3845, exp3848 (interpret decorrelation — judgment), exp3851,
  exp3852, exp3853, exp3856.

## Invariants (must hold at capstone)

- `paper_ready` stays TRUE (G1∧G2∧G3∧G4). FoVer **0.9131 stays frozen** — `.355 adds a new
  durability LENS on the same ensemble; it does not re-measure or move the headline AUROC.
- No artifact carrying `flagged_adversarial: true` is aggregated into the capstone or any
  forward-facing claim (Fabrication Gate).
- Every compute-bound artifact carries `inference_substrate`, `random_seed(s)`,
  `reproducibility_checksum`, `preconditions_checked`, and a plausible `duration_s`.
- The scissor's residual numbers are VOID unless BOTH positive controls pass (reasoner
  non-degenerate; ensemble AUROC reproduces frozen 0.9131±0.02) — exp3820 mode.
- Null/negative moat claims require the positive controls to have passed (FALSE_NEGATIVE_RISK
  discipline): a "moat subsumed" verdict is only valid if the reasoner genuinely self-verified.

## Risks / discipline notes

- **GPU contention.** exp3847 (scissor) is the single load-bearing GPU task; the hardware tasks use
  FPGA/SSH boards, not the 3090. exp3850 is CPU-first. If the rig is busy, exp3847 must exit
  cleanly to `blocked_no_free_gpu` (never queue, never fabricate) — the corpus + downstream
  CPU-reuse tasks still bank value next milestone.
- **Fake-independence trap (2604.07650).** The scissor alone can report a fake moat; exp3848 is
  NOT optional — it is the measurement that makes the moat claim defensible. The capstone verdict
  must be conditioned on exp3848's correlation result.
- **Round-number replication.** exp3827's 0.90/0.50 are exactly the kind of small-sample artifact
  the Adversarial-Confirmation Discipline targets; exp3847's bootstrap CI95 lower bound is the
  paper-citable field, not the point estimate.
- **No re-grind.** This is NOT a re-test of energy-as-selector/generator (both bounded) nor a
  naive facts-generalization rerun (forbidden without a new architecture — exp3850 supplies one).
  It is depth on the single open, load-bearing product question.
</content>
</invoke>
