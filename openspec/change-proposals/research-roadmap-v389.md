# Research Roadmap — Milestone 2026.06.389

**Title:** RUN THE VERIFIER-AS-REWARD TEST THE LITERATURE LEAVES OPEN — the de-confounded
A-vs-B on CODE, where the execution verifier's Youden J = TPR−FPR ≫ 0 clears the
training-reward precondition that the grid / noisy content-EBMs failed

**Planned:** 2026-06-14 (UTC) · **Supersedes:** 2026.06.388 · **Planner:** Opus 4.8 (outer-loop)

---

## 0. One-paragraph thesis

The operator's 2026-06-11 TOP-PRIORITY pivot is unambiguous: **stop grinding
verifier-as-SELECTOR (commodity, answered) and prove verifier-as-REWARD** — use the
un-hallucinating execution verifier as an *automated ground-truth engine* to TRAIN a
generator, not to filter at inference. Every attempt at that pivot has BLOCKED on the same
wall: the **Phase-0 precision gate** fails on grid / Sudoku domains (exp4159: training-model
base 0.501 < 0.85 with **zero** demo-perfect coverage; exp4100: no grid discrimination
signal), so the decisive A-vs-B training test has never actually run. Meanwhile two facts
now make the test executable on **code**: (1) the SAME shared execution-consistency primitive
certifies code at **precision 0.96** (exp4093), trivially clearing Phase-0 because the
verifier is the unit-test oracle — an *exact*, un-hallucinating signal with Youden
**J ≫ 0**, the precondition arXiv:2601.04411 proves is required and the noisy trained
content-EBMs lacked; and (2) the 2026 literature makes the de-confounded ablation the
load-bearing OPEN question — arXiv:2506.10947 shows *random* rewards recover ~74% of RLVR
gains **on Qwen** (a spurious-reward confound), failing on Llama/OLMo, so a clean test must
ablate **verifier-certified vs random-label from the SAME generator on a NON-Qwen base**.
So .389 runs exactly that: **Phase A** the de-confounded 3-arm on-policy LoRA-RFT on code
(A = verifier-certified, B = same-generator random-label [the Spurious-Rewards control],
C = gold-SFT oracle) with the load-bearing **A-vs-B** gate — does the verifier's *label*
carry training signal beyond distillation/spurious-reward? **Phase B** carries the pivot to
the north star (scale the GAP-4-certified ARC program corpus + the cheap Invisible-Leash
in-context-distillation lift test). **Phase C** banks monotonic ARC +1 and runs the Carnot
solver vs the random/greedy floor on the now-reachable LIVE ARC-AGI-3 env (no leaderboard
submission — operator-gated). **Phase D** reserves SOTA-ingestion / registry-hygiene /
hardware / capstone. DiffusionGemma stays GATED (gate STILL-PENDING per the .388 correction;
weights not cached). GAP-3 trained content energies stay RETIRED.

## 1. What .388 produced (the inputs to this plan)

| Result | Artifact | Reading |
|---|---|---|
| Efficiency moat "WON" but **semi-circular** — verifier 0.84 vs LLM-judge 0.66 (+0.18, CI95[0.08,0.30]), verifier ~free vs judge 5270 tok / 48s | exp4186 | On CODE the verifier ≈ the unit-test oracle, so this does NOT show a *non-trivial* verifier adds value; it's the easy case. |
| GAP-4 graded execution gate **production-safe** — holds +4/−0 (vote 0.4516→0.5806), vote-aware guard blocked the 25094a63 mis-promotion; graded relaxation adds nothing beyond exact-match | exp4187 | The execution gate is safe to ship as the commodity abstention wrapper; no new headroom from relaxation. |
| Sovereign local generator **UNDER-induces** — local 0.2258 (7/31 demo-perfect) vs codex 0.9355 (29/31); self-distillation corpus only **7** | exp4188 | A naive local generator can't yet feed a verifier-as-reward corpus on ARC; the certified-corpus must come from a stronger generator. |
| DiffusionGemma **blocked — no weights** (only 31M config/tokenizer cached) | exp4189 | The verifier-as-GUIDANCE gate stays STILL-PENDING; not a .389 task. |
| ARC monotonic +1 — lp85 → L3, **total_levels_solved = 15**, real-env-confirmed | exp4190 | Progress-not-perfection intact. |
| ARC-AGI-3 LIVE env **reachable** — anonymous key, 25 envs, random/greedy baseline established | exp4191 | The §0 grounding is unblocked; the next step is the Carnot solver vs the floor. |

**The blocker the whole pivot kept hitting** (exp4159 / exp4100, .377–.388): the verifier-as-reward
Phase-0 precision gate fails on grid/Sudoku (base too weak + zero demo-perfect coverage + no
discrimination). **The fix is not a better grid verifier — it is to run the test on the domain
where the execution verifier is already exact and precise: code.**

## 2. The pivot, made executable (why CODE clears what grids failed)

| Precondition (from the operator pivot + the literature) | Grids / Sudoku | CODE (execution) |
|---|---|---|
| Phase-0 certification precision P(gold \| certified) ≥ 0.85 | ✗ 0.501 base, 0 coverage (exp4159) | ✓ 0.96 (exp4093) |
| Youden J = TPR − FPR > 0 (arXiv:2601.04411) | ✗ noisy content-EBM (chance) | ✓ J≫0, exact output-match |
| Base has training headroom | corpus-dependent | ✓ harder code corpora (base < ceiling) |
| Generator produces certifiable own-traces (on-policy) | ✗ no demo-perfect coverage | ✓ visible-test-perfect traces |
| De-confound vs Spurious-Rewards (arXiv:2506.10947) | — | ✓ NON-Qwen base + same-generator random-label ablation |

The load-bearing GATE is **A vs B**, not A-vs-cold-base: A = verifier-certified
(visible-test-perfect) own-traces; B = the **same generator's** non-certified / random-label
traces, |B| = |A| — this is exactly the spurious-reward control that arXiv:2506.10947 and
arXiv:2509.20837 leave open. **A ≫ B (CI excl 0) → the verifier's label carries training
signal → verifier-as-reward is REAL. A ≈ B → "RFT helps" is just distillation / spurious-reward
elicitation → the verifier adds nothing (honest null, NOT a partial win).** Either outcome is
decision-grade and is the first clean answer to the operator's pivot.

## 3. Architecture (what executes)

```
                       ┌─────────────────────────────────────────────┐
   PHASE A (headline)  │  on-policy 3-arm LoRA-RFT on CODE            │
                       │  NON-Qwen base (gemma-4-12B-it / E4B-it)     │
   base ── gen K ──────┤    Arm A: visible-test-perfect (certified)   │
   traces              │    Arm B: same-gen random-label (spurious)   │── held-out
        │              │    Arm C: hidden-test-gold (oracle SFT)      │   hidden-test
        ▼              │    Arm D: cold base (no train)               │   pass@1
   execution verifier  └─────────────────────────────────────────────┘   ⇒ A vs B gate
   (sandbox.py +                       │ certifies / Youden J
    arc_gap4_execution_verifier.py)    ▼
   ── reused, model-free, exact ──>  Phase-0 precision gate (≥0.85, J>0)

   PHASE B (north star)   codex induces ARC programs → GAP-4-certify → CERTIFIED corpus
                          → in-context-distill into local base → induction lift vs 0.23?

   PHASE C (north star)   ARC +1 (hardened GAP-4)  ·  LIVE-env solver vs random/greedy floor
```

The energy/execution **verifier is unchanged and model-free** — it is reused as the
*reward-labeling* engine. The novelty is the layer it operates at (training data, not
inference filter) and the de-confounded ablation that isolates its label's value.

## 4. Phases & tasks

- **Archive/activate (exp4196).** Close .388, record the close-state truthfully, activate .389.
- **PHASE A — verifier-as-reward headline (exp4197 build → exp4198 launch → exp4199 collect).**
  Split per the long-codex rule: A1 measures Phase-0 precision + Youden J + headroom +
  gen-suitability and BUILDS the on-policy 3-corpus harness; A2 LAUNCHES the 3-arm LoRA-RFT
  backgrounded (resume-not-restart, truncation guard); A3 COLLECTS the decisive A-vs-B verdict.
- **PHASE B — pivot to the north star (exp4200).** Scale the GAP-4-certified ARC corpus +
  the cheap Invisible-Leash in-context-distillation lift test (local induction vs the 0.23 ceiling).
- **PHASE C — ARC north-star progress (exp4201 +1; exp4202 LIVE solver-vs-floor).**
- **PHASE D — reserved slots (exp4203 SOTA-ingestion; exp4204 registry/gaps hygiene;
  exp4205 hardware continuity; exp4206 capstone).**

## 5. Dependency graph

```
exp4196 (archive/activate)
  └─ exp4197 (A1 build + Phase-0/J/headroom)
        └─[gated phase0_precision≥0.85 ∧ harness_ready]─ exp4198 (A2 launch 3-arm RFT)
              └─[gated training_launched]─ exp4199 (A3 collect + A-vs-B verdict)
  ├─ exp4200 (B: certified ARC corpus + distill-lift)   [independent]
  ├─ exp4201 (C: ARC +1)                                [independent]
  ├─ exp4202 (C: LIVE solver vs floor)                  [independent]
  ├─ exp4203 (D: SOTA-ingestion)                        [independent]
  ├─ exp4204 (D: registry/gaps hygiene)                 [reads A3/B]
  ├─ exp4205 (D: hardware continuity)                   [independent]
  └─ exp4206 (D: capstone)                              [UNGATED aggregate]
```

## 6. Hardware & model requirements

- **RTX 3090 ×2 (CUDA)** — Phase A LoRA-RFT training + held-out eval (the headline GPU work).
- **Trainable NON-Qwen base (cached):** `google/gemma-4-12B-it` (primary; QLoRA fits the 3090s)
  or a smaller `gemma-4-E4B-it` if the precheck shows enough code headroom; `openbmb/MiniCPM5-1B`
  (Llama-arch) as the lightweight non-Qwen fallback. Non-Qwen is MANDATORY (spurious-reward
  confound, arXiv:2506.10947). The SOTA-GGUF rule is satisfied by naming a SOTA GGUF
  (`unsloth/gemma-4-26B-A4B-it-GGUF`) as the certification-precision reference / optional
  off-policy teacher arm.
- **Cached code corpora:** MBPP, HumanEval, EvalPlus(+) hidden tests, LiveCodeBench-lite — the
  precheck picks the (base, corpus) operating point where Phase-0 clears AND headroom exists.
- **Model-free verifier:** `python/carnot/verify/sandbox.py` (restricted exec) +
  `python/carnot/agentic/arc_gap4_execution_verifier.py` (demo-fit / content-hash) — reused, no GPU.
- **No FPGA dependency** for the headline; GateMate/PolarFire are the opportunistic continuity slot.

## 7. What this milestone is NOT

- NOT more verifier-as-SELECTOR grinding (operator pivot; the selection question is answered).
- NOT a DiffusionGemma headline (gate STILL-PENDING + weights not cached — stays gated).
- NOT a GAP-3 trained-content-energy revival (retired; operator-authorization required; CEM
  arXiv:2510.20607 re-flagged to the operator only).
- NOT a heroic single monolithic training run — Phase A is split build/launch/collect per the
  long-codex + resume-not-restart disciplines.

## 8. Acceptance — the one decision-grade question

**Does the execution verifier's LABEL carry training signal beyond distillation / the
spurious-reward confound?** A-vs-B held-out hidden-test delta with bootstrap CI95 excluding 0
(exp4199), with the gold-control gate passing (C ≥ base) and the truncation guard < 5%. A clean
YES is the project's first verifier-as-reward POSITIVE; a clean A≈B null bounds the pivot
honestly. Both are reported without spin.
