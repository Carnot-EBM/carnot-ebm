# Research Roadmap v358 — Adjudicate the Forward Bets

**Milestone:** 2026.06.358
**Planned:** 2026-06-05 (outer-loop, Claude Opus 4.8)
**Supersedes draft:** identical copy committed as `research-roadmap-next.yaml`
**Prior milestone doc:** `docs/research-notes/verifier-moat-scissor-plot-design.md`

---

## 1. What the previous milestone(s) proved

The project is **converged on every self-generable thread** and `paper_ready =
TRUE` (G1–G4 met; the FoVer methods headline **AUROC 0.9131**, n=1000, 5-seed,
CI95 [0.9027, 0.9235], is frozen and was reproduced on a clean CI runner for
G2). What is left are **operator-seeded forward bets** and **measurement gaps**
that the autonomous loop cannot resolve by manufacturing breadth (north-star
§1). The last several milestones sharpened exactly three of them:

1. **The verifier MOAT-at-scale is still unproven (DT-P2 o1-subsumption).**
   The error-independence scissor — *of the step-errors a strong frontier
   reasoner's own self-verification MISSES, what fraction does Carnot
   independently catch?* — has now been blocked/inconclusive **four times**,
   each for a different reason. The latest (`.357 exp3869) was the most
   informative: it finally ran end-to-end against the 500-error PRMBench corpus
   and returned **INCONCLUSIVE** because *both* positive controls were
   degenerate — the reasoner self-verify AUROC = 0.5 (caught zero gold-incorrect
   steps) and Carnot's ensemble AUROC = **0.55** (near chance). Root cause:
   `data/step_error_balanced_v2.json` (PRMBench) is **out-of-distribution** for
   the FoVer-trained, math-domain-bound ensemble. The moat can only be
   adjudicated on an **in-distribution** error-rich corpus where the ensemble
   actually discriminates (AUROC ≥ 0.65). That corpus does not yet exist at
   sufficient error count (FoVer-v4 has 6548 items but only 114 incorrect).

2. **The Phase-3 energy-as-GENERATOR bet (Thesis A) is unadjudicated.** The
   operator seeded EBT (arXiv:2507.02092) as the candidate foundation-model
   paradigm. Part-(a) PASSED (a tiny 38M byte-EBT trained stably and learned a
   *generalizing* energy landscape, pos/neg margin 8.6× untrained). Part-(b) —
   *does energy-descent GENERATION beat autoregression at matched compute?* —
   is the kill-gate. A scaled run cleanly produced **AR = 0.84, EBT-argmin =
   0.0** with confirmed headroom (`thesis_a_part_b_scaled_seed1.json`, verdict
   BOUNDED). The Deep-Think-P1 *decisive probe* (replace greedy per-token
   energy-argmin with a GLOBAL discrete beam search minimizing cumulative EBT
   energy — does global search recover AR-level accuracy → ARTIFACT, or also
   fail → FUNDAMENTAL) then ran (`thesis_a_p1_discrete_search_v2`) and returned
   **INCONCLUSIVE** only because the *re-trained* AR control collapsed to 0.01
   (< 0.3 guard). The adjudication is one disciplined run away: do it on a
   **confirmed-headroom checkpoint** (AR ∈ [0.4, 0.95]) instead of retraining
   and gambling on AR stability.

3. **Facts is the one domain with a NEW-architecture mechanism, but the signal
   was fabrication-flagged.** The math-bound ensemble is earned-negative on
   facts; graph-grounding (MemGraphRAG, arXiv:2606.00610) is the one route with
   a plausible mechanism. The `.356 prototype (exp3862) showed **AUROC 0.643 vs
   math-baseline 0.411, facts_catch_delta = 0.232** — but it was
   `flagged_adversarial` (1.02 s wall-clock for a claimed model-invoke →
   DURATION_TOO_SHORT) and the complementarity follow-up (exp3863) blocked
   because per-item scores were never persisted. The signal is promising but
   **un-bankable** until a non-fabricated re-run reproduces it with real
   wall-clock and per-item scores.

Carried invariants: `paper_ready = TRUE`, frozen FoVer 0.9131 **never silently
substituted**, verifier math-domain-bound, energy-SELECTION (P0.1)
honest-negative (this milestone's EBT work is *generation*, a different
mechanism), KV260 terminal, GateMate/PolarFire opportunistic-continuity.

---

## 2. The three biggest gaps (current state → PRD vision)

| Gap | PRD link | This milestone's move |
|---|---|---|
| **G-A. The Phase-3 foundation-model paradigm is unvalidated.** Energy-as-generator is the operator's chosen bet and the only live path to "continuous, non-autoregressive, self-correcting reasoning" (PRD Phase 3). It is stuck one disciplined run from a verdict. | PRD Phase 3 / Kona parity | Phase 1: adjudicate Thesis-A part-(b) on a **confirmed-headroom** checkpoint (DT-P1 global beam search) + an energy-descent System-2 scaling diagnostic. |
| **G-B. The moat (product value) is unproven at scale.** "Escape LLM hallucinations" is only defensible if Carnot catches what the frontier reasoner misses. 4× blocked; the blocker is now a *corpus distribution* problem, not a mechanical one. | PRD FR-* verification, north-star §1 (moat) | Phase 2: build an **in-distribution** error-rich corpus (ensemble AUROC ≥ 0.65), then finally run the scissor against it. |
| **G-C. The verifier is math-only.** A general "second pair of eyes" needs at least one more domain. Facts via graph-grounding is the only mechanism with signal — but the signal is fabrication-flagged. | PRD factual-grounding gate (Tier C) | Phase 3: de-fabricate the graph-grounding signal (real wall-clock + per-item scores) and measure facts complementarity. |

Plus the two standing mandates: **continuous self-learning** (FR-11, every
milestone — research-program.md) and **hardware continuity** (one task per
non-terminal attached board — Hardware-Task Continuity Discipline).

---

## 3. Architecture (where each task acts)

```
                    ┌─────────────────────────────────────────────┐
                    │  FROZEN HEADLINE (paper_ready=TRUE)          │
                    │  FoVer 4-verifier AUROC 0.9131 — DO NOT MOVE │
                    └─────────────────────────────────────────────┘
   PHASE 1 — Phase-3 forward bet (energy as GENERATOR, EBT)
     exp3871  part-(b) DT-P1 adjudication ON A CONFIRMED-HEADROOM checkpoint
              (AR in [0.4,0.95]) → global beam search vs greedy argmin
              → ARTIFACT | FUNDAMENTAL | INCONCLUSIVE
     exp3872  energy-descent System-2 diagnostic: accuracy(K steps) curve
              (gated on exp3871 positive_control_passed)
   PHASE 2 — Moat at scale (DT-P2), IN-DISTRIBUTION this time
     exp3873  BUILD in-distribution error-rich step-error corpus
              (>=150 incorrect, ensemble AUROC >= 0.65) → emits the gate field
     exp3874  RUN error-independence scissor on it (gated on AUROC>=0.65)
              residual_catch + bootstrap CI95 + reasoner positive control
   PHASE 3 — Broaden the verifier (facts, NEW architecture)
     exp3875  graph-grounding fact verifier, DE-FABRICATED (real wall-clock,
              per-item scores persisted) → reproduce facts_catch_delta
     exp3876  facts complementarity vs math ensemble (gated on delta>0)
   PHASE 4 — Mandates + continuity + record
     exp3877  FR-11 v24 online independence-reweighting (loads v23 state)
     exp3878  GateMate continuity — corrigendum the TAUTOLOGY flag + readback
     exp3879  PolarFire + KV260 consolidated opportunistic continuity audit
     exp3880  capstone v358 (paper_ready stays TRUE; conditioned verdicts)
   PHASE 0 — exp3870  archive .357 / activate .358 + backend routing diag
```

## 4. Dependency graph

```
exp3870 (activate) ─► everything
exp3871 (EBT adjudication) ──gated──► exp3872 (positive_control_passed==true)
exp3873 (build corpus) ──gated──► exp3874 (carnot_ensemble_auroc_on_corpus>=0.65)
exp3875 (graph re-run) ──gated──► exp3876 (facts_catch_delta>0)
exp3877, exp3878, exp3879 independent
exp3880 (capstone) consumes all
```

## 5. Hardware requirements

- **RTX 3090 (internal, cuda:1)** — exp3871/3872 (EBT train+generate),
  exp3873/3874 (Qwen3.6-35B GGUF self-verification + k=15 ensemble scoring),
  exp3875 (graph-grounding verifier model invocation). All via
  `.venv/bin/python` (infra discipline — bare `python` has no torch and silently
  drops to CPU, the fault that blocked the EBT kill-gate twice).
- **GateMate A1-EVB-2M** (DirtyJTAG `1209:c0ca`) — exp3878.
- **PolarFire SoC** (`ssh polarfire`) + **KV260** (`ssh kria`, SSH-not-SD-card) —
  exp3879.
- CPU-only — exp3870, exp3877 (Tier-1 counter updates), exp3880.

## 6. Routing & anti-wipeout

All experiment tasks route **`agent_type: codex` + `model: gpt-5.5` +
`requires_codex: true`** per the proven `.337/`.340/`.356/`.357 anti-wipeout
pattern (gemini-CLI crashes on GPU workloads and caused the `.333/`.355
whole-milestone wipeouts; operator granted standing gemini↔codex flip
authority 2026-06-05). GPU tasks additionally set `requires_gpu: true`. Verdicts
are deterministic gates, so codex suffices. The operator may flip any task to
gemini at activation if quota inverts.

## 7. Discipline checklist

- Terminal-prefix verdicts (`complete:` / `success:` / `blocked_<resource>`).
- PRECONDITIONS step 0 on every compute-bound task (GPU / GGUF / board / corpus).
- `inference_substrate` declared per task (live vs scoring vs aggregation).
- Principle-annotated REQUIRED ARTIFACT FIELDS — **except gated fields, which
  are emitted BARE scalars** (`feedback_gated_fields_must_be_bare`: a
  `{value,principle}` dict breaks the `gated_on` resolver).
- `prior_failures:` (4 sub-fields) on the genuine reruns (EBT, moat, graph);
  `operator_override:` on the routine/lineage continuations (archive, FR-11
  vN+1, hardware, capstone) per the 2026-05-29 auto-override rule.
- No `flagged_adversarial:true` artifact aggregated into the capstone.
- Frozen 0.9131 never moved; this milestone adds a *durability + generation*
  lens, not a new headline.
