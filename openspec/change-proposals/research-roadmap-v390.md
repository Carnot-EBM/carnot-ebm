# Research Roadmap — Milestone 2026.06.390

**Author:** Claude Opus 4.8 (outer-loop planning agent)
**Date:** 2026-06-14
**Status:** proposed
**Supersedes input:** `research-roadmap-v389.md` (the just-closed .389)
**North star:** `ops/north-star.md` §0 (solve ARC-AGI-3, accurately + efficiently),
§5 (energy VERIFIES, refinement GENERATES) — and the **2026-06-14 P0 operator
directive ("2+3+1"): re-aim the verifier program at the ORACLE-DISTINCT frontier.**

---

## 0. One-paragraph thesis

`.389 tried to run the de-confounded verifier-as-reward A-vs-B on code. The code
operating point **CLEARED** (Phase-0 certification precision **0.956**, Youden
**J=0.414**, training headroom **0.600**, corpora N-matched A=776 / B=776 / C=742) —
but the **background** LoRA training process **exited before its first checkpoint**
(exp4198 `blocked_training_process_exited_before_checkpoint`), so the decisive
A-vs-B was never collected and the capstone, skipping four DURATION_TOO_SHORT-flagged
artifacts, read "NO-OPERATING-POINT." That headline is an INFRA artifact, not a
scientific one — it is the recurring *powering-run background mechanism fails*
pattern (memory `incident_powering_run_background_mechanism_fails`). Meanwhile the
operator's **newest** directive (2026-06-14) re-aims the program: the verifier's
EXECUTION wins (code, ARC GAP-4) are CIRCULAR (verifier == the executable oracle) —
valid, but NOT a moat and NOT headline/gate-eligible. The deep, defensible,
still-UNPROVEN claim is an **oracle-DISTINCT** (learned/energy) verifier that
captures headroom where execution is NOT trivially the oracle: closing the
GAP-3-energy-ties-vote-on-ARC gap. So `.390 does two things at once. **HEADLINE
(Phase A):** does a LEARNED (oracle-distinct) verifier — the in-repo V-STaR class
(exp4176) applied to ARC — beat majority vote where execution is not the oracle,
with a MATCHED no-verifier control (CI95-excl-0)? Framed by the freshly-ingested
ARBITER ("wrong-majority failures": vote picks the largest BASIN, not the most
accurate) and SCOPE (a fine-grained learned signal recovers minority-vote-correct
answers). Plus the cheap **verifier-as-DETECTOR** measurement (detection AUROC where
selection headroom is ~0). **OWED (Phase B):** FINISH the de-confounded
verifier-as-reward A-vs-B on code by resuming the .389 checkpoint and running the
3-arm LoRA-RFT **synchronously, resume-accumulate** (the infra fix). Then ARC
north-star progress (Phase C) and the reserved infra/SOTA/hardware/capstone slots
(Phase D).

---

## 1. What .389 produced (the inputs to this plan)

| Result | Artifact | Status |
|---|---|---|
| **Verifier-as-reward code operating point CLEARED** — Phase-0 precision 0.956, Youden J 0.414, headroom 0.600, harness ready | exp4197 | ⚠️ flagged DURATION_TOO_SHORT (false-positive on a fast verifier-scoring step), but the numbers are the input to .390 |
| **3-arm corpora built + N-matched** — A(certified)=776, B(random-label)=776, C(gold)=742, base_passrate=0.6 | exp4198 | ❌ **infra**: `blocked_training_process_exited_before_checkpoint` — the background LoRA process died before checkpoint. Stable checkpoint dir `code_verifier_reward_lora_rft_a83b52882c198954` exists on disk |
| **Decisive A-vs-B** — never collected (gate-blocked on exp4198.training_launched=False) | exp4199 | ❌ blocked (downstream of the infra failure) |
| **Certified ARC corpus** — 16 demo-perfect programs, certification precision 0.9375; in-context distill-lift uninformative (seeded checkpoint missing) | exp4200 | ⚠️ flagged; corpus jsonl on disk |
| **ARC incremental** — no new solve (lp85 L4 no observed level-up candidate); held at total_levels=15 / total_games=13 | exp4201 | ✅ honest no-solve |
| **ARC live solver vs floor** — efficiency-only "win" (5 vs 6 actions) but **0 levels completed** by either (score 0.0); NO accuracy win, NO leaderboard submission | exp4202 | ✅ grounding, weak |
| **SOTA flagged for .390** = `non_qwen_same_generator_random_label_ablation_v390` | exp4203 | ✅ ingested, 8 methods |
| **Hardware** — GateMate unreachable (blocked); PolarFire hash-verified CPU dispatch; KV260 terminal-confirmed via SSH | exp4205 | ✅ |
| **Capstone** — headline = NO-OPERATING-POINT (artifact of skipping 4 flagged); ARC 15 levels | exp4206 | ✅ honest aggregation |

**The honest read of .389:** the verifier-as-reward test is **90% done** and
blocked ONLY on the background-training infra; the operating point is real. The
operator's 2026-06-14 directive then re-aims the HEADLINE at oracle-distinct work.
`.390 serves both.

---

## 2. The oracle-distinct frontier, made executable (why ARC, why V-STaR, why now)

The 2026-06-14 P0 directive's three planner directions:

1. **HEADLINE = an ORACLE-DISTINCT (learned/energy) verifier** that captures
   ARC-class headroom where execution is NOT trivially the oracle — close the
   GAP-3-energy-ties-vote-on-ARC gap. Every such task sets `verifier_is_oracle:
   false` and reports a MATCHED no-verifier control with CI95-excl-0.
2. **COMPLEMENTARY = the verifier-as-DETECTOR measurement** — detection AUROC where
   SELECTION headroom is ~0 (`docs/research-notes/verifier-as-detector-measurement-spec.md`).
3. **STOP re-running CIRCULAR confirmations** — code/HumanEval test-pass SELECTION,
   efficiency-vs-LLM-judge on code, Sudoku-at-convergence (the lint WARNs them).

**Why this is executable now, not aspirational:**

- The oracle-distinct verifier CLASS already exists in-repo: the `.386 **V-STaR
  learned selector** (exp4176, arXiv:2402.06457) trains on ACCEPTED + REJECTED
  traces — a learned correctness boundary, not the execution oracle — and reached
  AUROC 0.9792 on code. `.390 applies that class to the **ARC GAP-4 candidate pool**
  (`results/arc3_trm_verifier_rerank.json`), where the question becomes oracle-distinct.
- The literature freshly ingested (`research-references.md` `.390 sweep) tells us
  WHERE the headroom is: **ARBITER (2605.26172)** "wrong-majority failures" — the
  correct answer is in the pool but loses to a LARGER wrong basin; **SCOPE
  (2512.15146)** recovers minority-vote-correct answers a flat vote discards. A
  learned ARC verifier that recovers those is the oracle-distinct win.
- The detector measurement is **cheap** (cached pools, no new generation/training;
  `scripts/exp_verifier_detector_auroc.py` already exists) and decisive about the
  OTHER axis of verifier value (flag/reject, abstention) that the selector metric
  cannot see.

**On the OWED verifier-as-reward (Phase B) vs the "stop circular" directive.** The
de-confounded A-vs-B on code is the REWARD axis (does the execution LABEL *train* a
better model), NOT the SELECTION axis the directive halts. It is the operator's own
2026-06-11 pivot, it is HALF-DONE (operating point cleared, corpora built) and
blocked only on infra, and it is the explicit SOTA-flagged-for-.390 work. It is
honestly framed as `verifier_is_oracle: true` (execution oracle as reward — RLVR/
RLEF; cf. ExecVerify 2603.11226, EVOM 2604.00442) and is NOT claimed as a moat.
Finishing it converts a 90%-done decisive test into an answer; it does not compete
with the oracle-distinct headline.

---

## 3. Architecture (what executes)

```
                    .390  ORACLE-DISTINCT FRONTIER  (Phase A — HEADLINE)
   ┌──────────────────────────────────────────────────────────────────────┐
   │  A1 detector AUROC (cached)        A2 build V-STaR-on-ARC (learned)    │
   │  detection where headroom~0   ───▶ off-fold AUROC, verifier_is_oracle  │
   │  (Sudoku/code/ARC/GSM8K)           =false   ──▶  A3 measure: does it    │
   │  the OTHER axis of value                         beat vote OFF-ORACLE   │
   │                                                  vs MATCHED control,    │
   │                                                  CI95-excl-0 ?          │
   └──────────────────────────────────────────────────────────────────────┘
                    OWED  VERIFIER-AS-REWARD  (Phase B — self-learning)
   ┌──────────────────────────────────────────────────────────────────────┐
   │  B1 RESUME .389 checkpoint  ──SYNCHRONOUS, resume-accumulate──▶         │
   │     3-arm LoRA-RFT on code: A(verifier-cert) vs B(same-gen random-label │
   │     = Spurious-Rewards control) vs C(gold) vs D(cold); gold-control +   │
   │     truncation guards; Youden-J; memorization-shortcut diagnostic       │
   │  B2 certified-ARC-corpus distill-lift (Invisible-Leash latent/absent)   │
   └──────────────────────────────────────────────────────────────────────┘
                    NORTH-STAR ARC  (Phase C)        RESERVED  (Phase D)
   ┌─────────────────────────────────────┐  ┌───────────────────────────────┐
   │  C1 ARC incremental +1 (>=16)        │  │ D1 SOTA-ingestion (oracle-     │
   │  C2 ARC live solver-vs-floor         │  │    distinct track)            │
   │     (ACCURACY-seeking; no submit)    │  │ D2 registry+gaps hygiene      │
   └─────────────────────────────────────┘  │ D3 hardware continuity        │
                                            │ D4 capstone .390              │
                                            └───────────────────────────────┘
```

Reused, not rebuilt: `results/experiment_4176_vstar_selector_model.json` +
its trainer (V-STaR class); `python/carnot/agentic/arc_gap4_execution_verifier.py`
+ `arc_agi3_world_model.py`; `results/arc3_trm_verifier_rerank.json` (ARC pool);
`scripts/exp_verifier_detector_auroc.py`; `scripts/experiments/process_reward_weighted_sft_onpolicy_powered.py`
+ the resumable checkpoint `code_verifier_reward_lora_rft_a83b52882c198954`;
`scripts/gap4_program_induction_stack.py` + the certified corpus jsonl.

---

## 4. Phases & tasks (12 tasks)

**Infra:** exp4207 archive .389 → activate .390.

**Phase A — ORACLE-DISTINCT FRONTIER (headline, 2026-06-14 P0):**
- exp4208 **[A1] verifier-as-DETECTOR measurement** (cheap, cached) — detection
  AUROC per domain vs SELECTION headroom; declare `verifier_is_oracle` per domain.
- exp4209 **[A2] BUILD the oracle-distinct learned ARC verifier** (V-STaR-on-ARC) —
  train on accepted+rejected ARC candidate traces; off-fold AUROC; `verifier_is_oracle: false`.
- exp4210 **[A3] MEASURE it beats vote OFF-ORACLE** (gated_on A2) — learned-verifier
  rerank vs majority vote vs MATCHED no-verifier control on held-out ARC; CI95-excl-0;
  `verifier_is_oracle: false`. The headline gate (close GAP-3-ties-vote).

**Phase B — OWED verifier-as-reward + self-learning (2026-06-11 pivot):**
- exp4211 **[B1] FINISH the de-confounded verifier-as-reward A-vs-B on code** —
  RESUME `code_verifier_reward_lora_rft_a83b52882c198954`; run the 3-arm LoRA-RFT
  **SYNCHRONOUSLY, resume-accumulate** with per-step progress prints (the infra fix);
  A(certified) vs B(same-generator random-label) vs C(gold) vs D(cold); gold-control
  + truncation guards; Youden-J; memorization-shortcut diagnostic. `verifier_is_oracle: true`
  (honest — reward axis, not a moat). Self-learning / FR-11.
- exp4212 **[B2] certified-ARC-corpus distill-lift** (Invisible-Leash latent-vs-absent)
  — extend exp4200's certified corpus; cheap in-context lift of the LOCAL base's ARC
  induction from certified exemplars. Self-learning / sovereignty.

**Phase C — ARC north-star progress:**
- exp4213 **[C1] ARC incremental +1** (monotonic; total_levels >= 16; hardened GAP-4).
- exp4214 **[C2] ARC live solver-vs-floor, ACCURACY-seeking** (complete a level on an
  easy live game vs the floor; NO leaderboard submission).

**Phase D — reserved slots:**
- exp4215 **[D1] SOTA-ingestion** (oracle-distinct learned-verifier track; flag for .391).
- exp4216 **[D2] verifier-registry + gaps hygiene** (bit-exact GAP-4 regression replay;
  record the .390 oracle-distinct + reward outcomes; open a GAP-ORACLE-DISTINCT entry).
- exp4217 **[D3] hardware continuity** (GateMate + PolarFire drive-to-terminal; KV260
  opportunistic; SSH/USB-detect preconditions only).
- exp4218 **[D4] capstone .390** (UNGATED; headline = did the oracle-distinct learned
  verifier beat vote on ARC?).

---

## 5. Dependency graph

```
exp4207 (archive/activate, runs first)
exp4208 [A1 detector]              ── independent (cached)
exp4209 [A2 build V-STaR-on-ARC]   ── independent (build)
   └─▶ exp4210 [A3 measure beats-vote]   gated_on A2.selector_trained == true
exp4211 [B1 verifier-as-reward]    ── resumes .389 checkpoint (synchronous)
exp4212 [B2 ARC distill-lift]      ── resumes exp4200 corpus
exp4213 [C1 ARC +1]                ── independent
exp4214 [C2 ARC live]              ── independent (offline-validate first)
exp4215 [D1 SOTA]  exp4216 [D2 registry]  exp4217 [D3 hardware]  ── independent
exp4218 [D4 capstone]              ── UNGATED; reads all upstream, skips flagged
```

Only one structured `gated_on` (A3 on A2). Everything else is order-independent so a
single blocked task cannot cascade the milestone.

---

## 6. Hardware & model requirements

- **RTX 3090 (GPU rig):** B1 LoRA-RFT (small NON-Qwen base — gemma-4-E4B-it or
  MiniCPM5-1B; **Qwen is FORBIDDEN as the trained base** per the Spurious-Rewards
  confound arXiv:2506.10947, Qwen GGUF fine as an off-policy teacher only). A2 V-STaR
  training (small MLP/probe on cached ARC candidate features — light). A1 detector is
  CPU/GPU-light (cached scoring). The outer-loop owns any TRM training — **no .390 task
  touches the TRM checkpoint** (val 0.8227, SIGTERM'd, conductor stood-down).
- **SOTA local GGUF** (`cached_sota_pair()` / `.gguf` path, NOT AutoTokenizer):
  Qwen3.6-35B-A3B-GGUF / gemma-4-31B-it-GGUF / gemma-4-26B-A4B-it-GGUF / gemma-4-12B-it.
  B2's certified-corpus generation uses codex (induces 0.94) + the local GGUF for the
  in-context lift base.
- **FPGA boards (Phase D3):** GateMate (USB DirtyJTAG `--detect`), PolarFire (`ssh
  polarfire`), KV260 (`ssh kria`, opportunistic/terminal). SSH/USB-detect preconditions
  ONLY (KV260 SSH-Not-SD-Card Discipline).
- **ARC-AGI-3 (Phase C):** offline fixtures (C1) + the live SDK anonymous key (C2,
  `arc_agi` v0.9.8) — NO leaderboard submission (operator-only external publication).

**Architecture freshness watch:** `_bmad/architecture.md` "Last Reconciled" =
2026-05-16 (~29 days). Crossing the 30-day flag threshold next milestone — a
reconciliation pass is due in .391 if not sooner.

---

## 7. What this milestone is NOT

- **NOT a re-grind of circular SELECTION wins.** No code/HumanEval test-pass
  selection, no efficiency-vs-LLM-judge on code, no Sudoku-at-convergence rerank
  (2026-06-14 directive item 3; the lint WARNs them). The detector (A1) and reward
  (B1) axes are DIFFERENT from selection; the ARC oracle-distinct test (A3) is
  explicitly `verifier_is_oracle: false`.
- **NOT a moat over-claim.** Every verifier-value task declares `verifier_is_oracle`
  honestly; a circular (execution-oracle) result is labeled `execution_grounded` and
  may NOT flip the DiffusionGemma gate (which stays STILL-PENDING until an
  oracle-distinct win lands with a matched control — CLAUDE.md Circularity Discipline).
- **NOT a background powering run.** B1 is SYNCHRONOUS, resume-accumulate, per-step
  progress-printed (the .389 infra failure mode is banned).
- **NOT a TRM-training task.** No task launches train.py, kills it, or writes
  `results/trm_runs/sudoku_extreme_baseline/`.
- **NOT an external publication.** ARC live (C2) submits NO scorecard.

---

## 8. Acceptance — the decision-grade questions

1. **[A3, HEADLINE] Does a LEARNED (oracle-distinct) verifier beat majority vote on
   ARC where execution is not the oracle, vs a matched control, CI95-excl-0?**
   YES → the first oracle-distinct verifier win (the defensible moat; informs the
   DiffusionGemma gate). TIE → honest null; the GAP-3-ties-vote frontier persists and
   the detector axis (A1) becomes the honest reframing of "where the verifier adds value."
2. **[B1, OWED] Does the verifier's execution LABEL carry training signal beyond the
   spurious-reward confound — A(certified) vs B(same-generator random-label),
   CI95-excl-0, with the gold-control + truncation guards passing?** The project's
   first clean verifier-as-reward read, either way (REAL or distillation-null).
3. **[A1] Where selection headroom is ~0, does the verifier still DETECT errors
   (AUROC ≫ 0.5, CI95-excl)?** The orthogonal-value reframing.
4. **[C] Does ARC progress advance (total_levels >= 16) and/or does the live solver
   complete a level vs the floor?**

A milestone that answers (1) or (2) decisively — positive OR honest-negative —
advances the north star (north-star §0 rule). The capstone (exp4218) records the
single headline read.
