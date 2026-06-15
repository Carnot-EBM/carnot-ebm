# Research Roadmap — Milestone 2026.06.392

**STRENGTHEN THE ORACLE-DISTINCT VERIFIER — the .391 beats-vote gate RAN and TIED vote; diagnose the null, fix all three causes, and re-test with power.**

- **Status:** proposed (planner: Claude Opus 4.8, outer-loop, 2026-06-15)
- **Prior milestone:** 2026.06.391 (oracle-distinct verifier TIES vote on n=14; verifier-as-reward HARNESS-DEFERRED; ARC at 17 levels)
- **Milestone doc (this file):** `openspec/change-proposals/research-roadmap-v392.md`
- **YAML:** `research-roadmap-next.yaml`
- **Tasks:** 12 (exp4230–exp4241)
- **North star:** solve ARC-AGI-3 accurately and efficiently; the verifier is the existential value-add (`ops/north-star.md` §0/§5)

---

## 0. One-paragraph thesis

`.391 did the right thing: it finally RAN the oracle-distinct beats-vote gate (the P0 directive of
2026-06-14), with the `.390 infra blockers fixed. The honest read is a clean **TIES-VOTE-NULL**: on a
held-out ARC split, a LEARNED oracle-distinct verifier (`verifier_is_oracle=false`) scored
`verifier@1 − vote@1 = −0.0714`, bootstrap CI95 `[−0.214, 0.0]` — it did not beat vote (it slightly lost).
But this is **not a settled refutation** of the oracle-distinct thesis; it is an *under-powered, weakly-
built* first read with three separable, diagnosable causes: **(1)** the verifier was a per-candidate
**logistic regression that scored each candidate in ISOLATION** — blind to the competing candidate set,
the vote basin, and cross-candidate agreement; **(2)** it trained on **14 ACCEPTED / 1782 REJECTED rows**
(base-rate 0.008) — an extreme imbalance that collapses a binary discriminator (off-fold AUROC only
0.779); **(3)** the held-out gate was **n=14 tasks**, below the CLT floor. `.392 attacks all three with
the SOTA the `.391 ingestion flagged: a **cross-candidate set-encoder aggregator** (Set-Encoder
2404.06912 / MSV 2603.03417 / AggLM 2509.06870) + a **calibrated imbalance-aware loss** (2509.19681) on a
**grown labeled pool** (target ≥30 held-out), with a **margin-triggered ARBITER override** (2606.04323),
on **ARC** AND a higher-power, less-imbalanced **CODE** replication (learned pass-predictor, no test
execution at inference → still oracle-distinct) to disambiguate "ARC data-sparsity" from "the selection
thesis is bounded." We also FINISH the owed verifier-as-reward — re-scoped to FIT the conductor window
with a harness-first smoke that asserts REAL training (so it cannot short-circuit to a fake-short
artifact) and a hard auto-retire — keep ARC monotonic (+1 → 18), and probe live-env solve accuracy.

---

## 1. What .391 produced (the inputs to this plan)

| Result | Artifact | Read |
|---|---|---|
| **Oracle-distinct verifier TIES vote on ARC** (the HEADLINE gate, CLEAN/un-flagged) | exp4221 | `verifier@1 − vote@1 = −0.0714`, CI95 `[−0.214, 0.0]`; `oracle@K=1.0` (headroom EXISTS); ARBITER override degenerated to keep-vote (0.643=0.643); matched control = verifier (0.5714). **n=14 held-out tasks.** `verifier_is_oracle=false`. |
| **The learned ARC verifier was weak + imbalanced** | exp4220 (flagged TAUTOLOGY, false-positive) | plain per-candidate **logistic regression**, **14 accepted / 1782 rejected** (base-rate 0.008), off-fold AUROC **0.779**, `wrong_majority_n=5`. |
| **Detector ≫ Selector divergence** (carried from exp4208) | exp4229 capstone | detection AUROC ≈ 1.0 on math/sudoku/code, 0.90 on ARC; **selection headroom only on ARC (0.129) and code (0.18)**; math/sudoku headroom ≈ 0 (SC near-ceiling). |
| **Verifier-as-reward: 4th/5th infra failure** | exp4222 (smoke, 14s), exp4223 (3-arm, 36.7s) — both DURATION_TOO_SHORT-flagged | the live LoRA training **short-circuited to a fake-short "progress" artifact** instead of training; operating point (Phase-0 0.956, Youden-J 0.414) + N-matched corpora (A=776/B=776/C=742) intact. **Accumulate-floor reached.** |
| **ARC monotonic +1** | exp4224 | sc25 advanced to L3; `total_levels_solved=17`, `total_games_solved=13`, real-env-confirmed. |
| **Live solver: efficient, 0 levels** | exp4225 | beats the random/greedy floor on EFFICIENCY; `levels_completed=0` on the live env (no ACCURACY win yet). |
| **SOTA flagged for .392** | exp4226 | **AggLM-style review-and-reconcile cross-candidate aggregator** — "train an ARC aggregator that recovers minority-correct answers instead of only assigning independent candidate scores." |
| **DiffusionGemma gate** | — | **STILL-PENDING** — activates only on an oracle-distinct win with a matched control (`verifier_is_oracle=false`, CI95-excl-0). |

**The one-line lesson:** the gate ran and the verifier tied vote, but the verifier was the *weakest
possible build* (isolated logistic regression on 14 positives) tested at the *lowest possible power*
(n=14). Before concluding "oracle-distinct selection can't beat vote on ARC," we owe it the strongest
build (cross-candidate aggregator + imbalance-aware loss) at adequate power — and a cross-domain (code)
control. That is depth, not churn (north-star §1): the technique and corpus change to address the named
root cause.

---

## 2. Why these three gaps, and why now

**Gap 1 — The oracle-distinct moat is the existential, still-UNPROVEN claim, and the `.391 null is
diagnosably premature.** Per `ops/north-star.md` §5, with the generator commodity, the VERIFIER is
Carnot's entire value-add, and its oracle-distinct value is unproven. `.391 produced the first clean read
(TIES) but with a build/power so weak that the null carries little information. The disciplined next move
is the strongest build the literature offers, at power, on the two domains where headroom exists (ARC +
code). **This is Phase A — the headline.**

**Gap 2 — The verifier-as-reward (self-learning / FR-11) is a TOP-PRIORITY operator directive that is
owed but infra-cursed.** Four/five consecutive INFRA failures (background-process death → gate-block →
PEFT attach → two duration-flag short-circuits), now at the operator's own accumulate-floor. The honest
move is ONE re-scoped attempt that is *guaranteed to either produce a real training run in-window or fail
loudly* — harness-first with a real-training assertion — and a hard retire mechanic so a fifth no-usable
window auto-retires the live-LoRA path (a clean signal to take it off the conductor). **This is Phase B —
the owed self-learning experiment.**

**Gap 3 — ARC-AGI-3 live-solve ACCURACY is still 0.** The solver is efficient but completes 0 live
levels; the north star's primary axis (accuracy) is unmet on the real env. Keep the offline solved-level
count monotonic (+1 → 18) and apply the margin-triggered override to goal-predicate induction to try to
complete a live level. **This is Phase C — the north star.**

---

## 3. Architecture (what executes)

```
                       ┌──────────────────────────────────────────────────────────┐
                       │  PHASE A — ORACLE-DISTINCT AGGREGATOR (the headline)       │
                       │                                                            │
 grown labeled ARC ───▶│  A1 cross-candidate SET-ENCODER aggregator                 │
 pool (load_arc_rows,  │      + calibrated imbalance-aware loss (2509.19681)        │
 ≥30 held-out tasks)   │      + cross-candidate features (MSV/AggLM/Set-Encoder)    │
                       │      out-of-fold; verifier_is_oracle=false                 │
                       │            │ aggregator_trained (bare bool)                │
                       │            ▼                                               │
                       │  A2 ARC beats-vote gate (gated_on A1):                     │
                       │      aggregator@1 vs vote@1 vs matched control vs          │
                       │      MARGIN-TRIGGERED override (2606.04323) vs oracle@K    │
                       │      task-level bootstrap CI95-excl-0  ← THE HEADLINE      │
                       │                                                            │
                       │  A3 CODE oracle-distinct replication (independent):        │
                       │      learned pass-predictor (NO execution) — higher-power, │
                       │      less-imbalanced disambiguation of the ARC null        │
                       └──────────────────────────────────────────────────────────┘
                       ┌──────────────────────────────────────────────────────────┐
                       │  PHASE B — VERIFIER-AS-REWARD (owed; self-learning/FR-11)  │
                       │  B1 harness-first smoke: REAL-training assertion           │
                       │      (duration floor + loss-moved) → harness_smoke_passed  │
                       │            │ (bare bool)                                   │
                       │            ▼                                               │
                       │  B2 de-confounded 3-arm A-vs-B (gated_on B1), window-      │
                       │      boxed, retire_if_same_verdict; verifier_is_oracle=true│
                       └──────────────────────────────────────────────────────────┘
                       ┌──────────────────────────────────────────────────────────┐
                       │  PHASE C — ARC NORTH STAR                                  │
                       │  C1 incremental +1 (total_levels ≥ 18)                     │
                       │  C2 live-env accuracy probe (margin-triggered override)    │
                       └──────────────────────────────────────────────────────────┘
                       ┌──────────────────────────────────────────────────────────┐
                       │  PHASE D — RESERVED + CLOSE                                │
                       │  D1 SOTA-ingestion  D2 registry/gaps  D3 hardware  D4 cap  │
                       └──────────────────────────────────────────────────────────┘
```

---

## 4. Phases & tasks (12 tasks, exp4230–exp4241)

| # | id | phase | what | gate / override |
|---|---|---|---|---|
| 0 | exp4230 | infra | archive .391 → activate .392; record close-state (TIES-VOTE-NULL n=14; reward harness-deferred; ARC 17) | operator_override (class 1) |
| 1 | exp4231 | A1 | BUILD the cross-candidate set-encoder aggregator on a GROWN ARC pool + calibrated imbalance-aware loss; out-of-fold AUROC; `aggregator_trained` | prior_failures exp4220 + override |
| 2 | exp4232 | A2 | ARC beats-vote gate: aggregator vs vote vs matched control vs margin-triggered override vs oracle@K; CI95-excl-0 — **THE HEADLINE** | gated_on A1; prior_failures exp4221 + override |
| 3 | exp4233 | A3 | CODE oracle-distinct beats-vote (learned pass-predictor, NO execution) — higher-power, less-imbalanced disambiguation | operator_override (class 3) |
| 4 | exp4234 | B1 | verifier-as-reward LoRA harness-first smoke: REAL-training assertion (duration + loss-moved); `harness_smoke_passed` | prior_failures exp4222 |
| 5 | exp4235 | B2 | verifier-as-reward 3-arm A-vs-B, window-boxed, retire mechanic | gated_on B1; prior_failures exp4223/exp4211 |
| 6 | exp4236 | C1 | ARC incremental +1 (total_levels ≥ 18) | operator_override (class 3) |
| 7 | exp4237 | C2 | ARC live-env solver accuracy (margin-triggered override on goal-predicate induction); no submission | prior_failures exp4225 + override |
| 8 | exp4238 | D1 | SOTA-ingestion slot (.392 sweep → .393) | operator_override (class 1) |
| 9 | exp4239 | D2 | verifier registry + gaps hygiene (GAP-4 regression guard + record .392 outcomes; declare methodology) | operator_override (class 1) |
| 10 | exp4240 | D3 | hardware continuity (GateMate + PolarFire drive-toward-terminal; KV260 opportunistic) | operator_override (class 2) |
| 11 | exp4241 | D4 | capstone .392 (UNGATED) | operator_override (class 1) |

The single load-bearing question: **does the STRONGER oracle-distinct verifier (cross-candidate
aggregator + imbalance-aware loss, at power) beat vote on ARC and/or code (`verifier_is_oracle=false`,
matched control, CI95-excl-0)?** A clean win closes GAP-3-ties-vote and makes the DiffusionGemma gate
resolvable; a clean ties-with-headroom-at-power is a much stronger (and decision-grade) null than `.391's
underpowered one.

---

## 5. Dependency graph

```
exp4230 (archive/activate) ──▶ everything

PHASE A:  exp4231 (A1 build) ──aggregator_trained──▶ exp4232 (A2 ARC gate, gated)
          exp4233 (A3 code replication) — independent of A1 (different domain/pool)

PHASE B:  exp4234 (B1 smoke) ──harness_smoke_passed──▶ exp4235 (B2 3-arm, gated)

PHASE C:  exp4236 (C1 +1), exp4237 (C2 live) — independent

PHASE D:  exp4238 (SOTA), exp4239 (registry), exp4240 (hardware) — independent
          exp4241 (capstone) — UNGATED; reads all upstream, skips flagged artifacts
```

Two structured gates (`gated_on`) let A2/B2 skip their Sonnet call when the prerequisite verdict fails:
- exp4232 `gated_on` exp4231 `aggregator_trained == true`
- exp4235 `gated_on` exp4234 `harness_smoke_passed == true`

---

## 6. Hardware & model requirements

- **Phase A** is CPU-sufficient (cached candidate pools + a small set-encoder/aggregator; no GGUF/GPU
  inference). `inference_substrate: verifier_ensemble_against_cached_candidates`.
- **Phase B** needs a single RTX 3090 (CUDA) + a NON-Qwen SOTA base. Default **`unsloth/gemma-4-12B-it`**
  (the lightweight SOTA per CLAUDE.md — fast iteration / higher batch throughput on one 3090, the right
  pick when an experiment makes many LLM calls). Qwen is FORBIDDEN as the trained base (Spurious-Rewards
  confound); a Qwen GGUF as an off-policy teacher/certifier reference is fine. `inference_substrate:
  live_llm_inference`. PRECONDITIONS gate CUDA + the cached base BEFORE training; cache the 12B model in a
  PRECONDITIONS step (newly released 2026-06-05 — do not assume it is already cached).
- **Phase C** is offline/air-gapped for C1; C2 uses the live ARC SDK anonymous key (no leaderboard
  submission — operator-only).
- **Hardware continuity (D3):** GateMate (`openFPGALoader -c dirtyJtag --detect`) + PolarFire
  (`ssh polarfire`) drive-toward-terminal; KV260 (`ssh kria`) opportunistic. SSH/USB-detect preconditions
  ONLY (KV260 SSH-Not-SD-Card Discipline).
- **The TRM Sudoku checkpoint is DONE (val 0.8227, SIGTERM'd) and the conductor stays stood-down.** NO
  task may launch TRM training, `pkill`/`kill` against `train.py`, or WRITE
  `results/trm_runs/sudoku_extreme_baseline/`.

---

## 7. What this milestone is NOT

- **NOT a re-run of the `.391 null.** Every Phase-A task changes the TECHNIQUE (cross-candidate aggregator,
  not isolated logistic regression), the LOSS (calibrated imbalance-aware), the CORPUS (grown ≥30 held-out
  / cross-domain code), or the OVERRIDE (margin-triggered). This is the Failed-Experiment Rerun
  Discipline's "address the root cause," not churn.
- **NOT a circular execution win.** The CODE replication (A3) uses a LEARNED pass-predictor that does NOT
  execute tests at inference (`verifier_is_oracle=false`); `adversarial_verify.py:check_circular_moat_overclaim`
  must stay clean. The verifier-as-reward (B) declares `verifier_is_oracle=true` honestly (the reward is
  the execution oracle — an RLVR reward axis, NOT a moat claim).
- **NOT a 5th identical doomed verifier-as-reward run.** B is re-scoped to fit the window with a
  real-training assertion + hard auto-retire; a fifth no-usable window retires the live-LoRA path.
- **NOT an off-ARC-significance SELECTOR re-grind** (the operator de-prioritized that 2026-06-11).
- **NOT a leaderboard submission** (C2 submits no scorecard; external publication is operator-only).
- **NOT a flip of the DiffusionGemma gate** unless an oracle-distinct win lands with a matched control.

---

## 8. Acceptance — the decision-grade questions

1. **Does the stronger oracle-distinct verifier beat vote on ARC?** (exp4232) `aggregator@1 − vote@1`,
   CI95-excl-0, headroom present, `verifier_is_oracle=false`, matched control reported. WIN / TIES-AT-POWER /
   NO-HEADROOM-UNINFORMATIVE.
2. **Does it beat vote on CODE (less-imbalanced, higher-power)?** (exp4233) — disambiguates whether the
   `.391 ARC null was data-sparsity (code wins) or a thesis limit (code also ties).
3. **Does the verifier's LABEL carry training signal?** (exp4235, gated on a REAL B1 smoke) A-vs-B (certified
   vs same-generator random-label) CI95-excl-0, with the gold-control + truncation guards passing — or an
   honest harness-deferred / auto-retired verdict.
4. **Is the ARC solved-level count monotonic?** (exp4236) `total_arc_levels_solved ≥ 18`.
5. **Can the live solver complete a level?** (exp4237) `solver_completes_level`, vs the `.391 efficiency-only
   floor; no submission.
6. **Is the DiffusionGemma gate now resolvable?** (exp4241) true ONLY on an oracle-distinct win with a
   matched control; otherwise STILL-PENDING.

The capstone (exp4241) emits a single `headline_outcome` from the enumerated set and decides the standing
of the oracle-distinct frontier + the verifier-as-reward pivot after `.392.
