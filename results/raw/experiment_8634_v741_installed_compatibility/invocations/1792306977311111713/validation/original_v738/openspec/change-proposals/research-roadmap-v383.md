# Research Roadmap v383 — Converge the Sudoku-Extreme TRM baseline to ~0.87 with the FIXED LR schedule, THEN run the decisive verifier-as-reward graft (the test never yet run on a faithful baseline)

**Milestone:** 2026.06.383
**Planned:** 2026-06-13 (planning agent, Claude Opus 4.8)
**Prior:** 2026.06.382 (openspec/change-proposals/research-roadmap-v382.md — inline)
**North star:** solve ARC-AGI-3 accurately AND efficiently (ops/north-star.md §0).
Post-reframe (§5): energy VERIFIES, a learned recursive refiner GENERATES. The
verifier is Carnot's existential value-add and its value is still UNPROVEN. This
milestone runs the cheapest DECISIVE test of that value: does the executable Carnot
verifier add value as reward/reranker on a FAITHFUL recursive-reasoner baseline?

---

## 1. What .382 proved (the honest read, from the artifacts not the prose)

.382's headline question was: **did the LR-resume fix make resumable training
ACCUMULATE, and did the Sudoku-Extreme baseline converge toward 0.87?** Capstone
exp4133 verdict: `lr_fixed_accumulating_v383_continues`. The answer is a clean,
load-bearing YES on the mechanism:

| Track | .382 task | Outcome | Evidence |
|---|---|---|---|
| **LR-resume FIX (headline)** | exp4126 | **FIXED — schedule now continues across resume** | Root cause: `TRMModule.manual_step` reset to 0 each process + `training_step` pinned LR to `base_lr` after warmup instead of calling the cosine schedule. After fix, a resumed pass starts at `train/lr=9.999e-05` (continues), not the `2.45e-6` fresh-warmup reset. `lr_continuous_across_resume=true`. |
| **ACCUMULATION** | exp4127 | **val 0.106 -> 0.278 in ONE corrected pass** | `delta=+0.172` vs .381's `+0.01/pass` = **~17x faster**. `per_pass_delta_vs_v381.beats_v381=true`. Bounded (~3396s). Still `matches_published_087=false` (0.278 vs 0.87). |
| **GRAFT** | exp4128 | **DEFERRED — baseline not yet faithful** | `graft_deferred=true`, `baseline_val=0.2782`, `estimated_passes_to_converge_for_383=4`. `verifier_value_added=false` (because deferred, NOT because tested). |
| **ACCURACY (ARC)** | exp4129 | **+1 game, real-env-confirmed** | `total_games_solved=13`, a new non-spatial game solved at action 16, `real_env_confirmed=true`. |
| SOTA / registry / hardware | exp4130/4131/4132 | recorded | registry regression-guard green; GateMate unreachable, PolarFire CPU-dispatch hash-verified, KV260 terminal. |

**The diagnosis (load-bearing for this milestone).** Two facts define .383:

1. **The bottleneck is now PURELY convergence, and it is no longer a mechanism
   bug — it is wall-clock.** The LR fix changed the regime from `+0.01/pass`
   (glacial, the .381 re-warm) to `+0.172/pass` (real accumulation). At the
   .382-measured rate, the baseline needs an estimated **~4 more bounded passes**
   (exp4128) from 0.278 to reach the published ~0.87. **.383 therefore schedules
   FOUR accumulate passes, not three** — three would land ~0.79 (and convergence
   decelerates near the ceiling), so the decisive RFT arm would defer a 4th time.
   This is depth, not churn: continuing accumulation is the prerequisite for the
   decisive graft, and the regime change (proven 17x) makes it tractable.

2. **The verifier-as-reward graft has NEVER been tested on a faithful baseline.**
   exp4109 (.380) ran a "graft" on the *interrupted* 0.0232 checkpoint — not a real
   test. exp4119 (.381, val=0.106) and exp4128 (.382, val=0.278) both honestly
   DEFERRED. So the central question of the whole post-reframe program — *does the
   executable Carnot verifier add value over the vote on a domain where it provably
   discriminates?* — is still OPEN. .383 exists to answer it on a faithful baseline.

This sequencing is mandated by the operator's 2026-06-13 "queued but not competing"
directive: the DiffusionGemma energy-guided-diffusion scale-up
(`docs/research-notes/diffusiongemma-energy-guided-diffusion-spec.md`) is GATED on
the TRM graft reporting `verifier_value_added==true`. The TRM-Sudoku test is the
cheapest DECISIVE version of that same question; it must close first.

---

## 2. The three biggest gaps between current state and the PRD vision

1. **No faithful recursive-reasoner baseline yet (the convergence gap).** The PRD's
   Phase-3 endgame and the north-star §5 hybrid both need a *working generator* to
   verify. The TRM baseline is 0.278/0.87. Closing this is the gating prerequisite
   for every downstream verifier-value claim on this domain.

2. **The verifier's value is UNPROVEN where it should be strongest (the moat gap).**
   On non-executable ARC grids the verifier *anti*-discriminated (exp4099, captured
   -0.23pp). Sudoku is the opposite — row/col/box validity is an EXACT executable
   check, essentially a perfect oracle. If the verifier cannot add value HERE, the
   verifier-as-reward thesis is in serious trouble. This is the decisive measurement.

3. **The headroom question is unanswered (the false-negative gap).** Every prior
   single-verifier "no value" finding (exp4099/4109/4128 ~0.0pp) risks being a
   FALSE_NEGATIVE_RISK: if TRM's pass@K pool has no headroom (oracle == vote), NO
   reranker could win and the null is uninformative. .383 makes the oracle-vs-vote
   positive control a FIRST-CLASS, mandatory step before any verifier verdict.

---

## 3. Architecture — the converge-then-graft pipeline

```
  STABLE CHECKPOINT  results/trm_runs/sudoku_extreme_baseline/last.ckpt
  (val=0.278, FIXED LR schedule, exp4127)
          |
          v
  PHASE A — BASELINE CONVERGENCE (depth; the prerequisite)
  +----------------------------------------------------------------+
  | exp4135 pass1  resume (fixed LR) <=60min -> val, save back      |
  | exp4136 pass2  defensive read; resume <=60min -> val            |
  |   PLATEAU FLOOR: if val did NOT improve vs prior -> config      |
  |   audit instead of blind-training (Failed-Rerun Discipline)     |
  | exp4137 pass3  defensive; resume <=60min -> matches_087?        |
  +----------------------------------------------------------------+
          |  faithful baseline (target val ~0.87; gate val>=0.85)
          v
  PHASE B — THE DECISIVE GRAFT (the payoff)
  +----------------------------------------------------------------+
  | exp4138  sample K candidates per held-out puzzle from the      |
  |          faithful TRM                                            |
  |   (0) HEADROOM POSITIVE CONTROL (mandatory, FIRST):             |
  |        oracle(best-of-K, executable validity) vs vote pass@1.   |
  |        oracle==vote -> NO HEADROOM -> uninformative null, STOP   |
  |        the verifier verdict (FALSE_NEGATIVE_RISK).               |
  |   (A) RERANK arm: executable-verifier rerank pass@1 vs vote,    |
  |        bootstrap CI; (Weaver fallback: weighted ensemble of     |
  |        executable + text-stat verifiers if single is headroom-  |
  |        limited).                                                 |
  |   (B) RFT arm (de-confound): A=verifier-certified corpus vs     |
  |        B=vote-certified corpus, N-matched, same TRM, same pool, |
  |        resume-train each (FIXED schedule); held-out exact-acc,  |
  |        CI95 excl 0.                                              |
  |   -> verifier_value_added (bool). Unlocks/kills DiffusionGemma. |
  +----------------------------------------------------------------+

  PHASE C — NORTH-STAR BREADTH + MANDATED SLOTS
    exp4139  ARC-AGI-3 incremental +1 (monotonic; offline explore-induce-verify)
    exp4140  SOTA-ingestion (GRAM 97%, RLVR-TRM-thinking-reward, Weaver)
    exp4141  verifier-registry + gaps hygiene (GAP-4 regression guard)
    exp4142  hardware continuity (GateMate + PolarFire; KV260 terminal)

  PHASE D — CLOSE
    exp4134  archive .382 -> activate .383  (runs FIRST, mechanical)
    exp4143  capstone .383 (UNGATED aggregation; does verifier_value_added flip?)
```

**Why bounded passes, not a detached contiguous run.** exp4126 raised the
contiguous-run alternative, but two facts close it: (a) bounded passes now WORK
(proven +0.172 with the fix), and (b) the split-BUILD-background-COLLECT mechanism
has failed three milestones (memory `incident_powering_run_background_mechanism_fails`).
The proven-robust pattern is **single synchronous resume-accumulate per task with
per-task progress prints** (so codex's idle-timeout never fires) and a stable
(corpus+model)-keyed checkpoint. Each of exp4135/4136/4137 is exactly that.

---

## 4. Phase descriptions

### Phase A — Baseline convergence (exp4135 / 4136 / 4137) — `track: trm-baseline`
Three bounded (≤60min, `trainer.max_time`) resume-accumulate passes from the stable
checkpoint with the FIXED LR schedule, each saving back to the stable path so the
next pass continues. Each pass is DEFENSIVE: it reads the prior pass's val and, if
val did **not** improve, runs a SHORT config audit instead of blind-training the next
pass (Failed-Experiment Rerun Discipline — no blind plateau-burning). Deliverable per
pass: `val_exact_accuracy`, `delta_vs_previous`, `matches_published_087`,
`stable_checkpoint_path`. Target val ≈ 0.87; the Phase-B gate is val ≥ 0.85.

### Phase B — The decisive verifier-as-reward graft (exp4138) — `track: verifier-as-reward`
On the faithful baseline, the single most important experiment of the milestone.
Built-in **headroom positive control first** (oracle best-of-K vs vote — if equal,
the null is uninformative and we say so). Then the RERANK arm (executable-verifier
rerank vs vote, bootstrap CI; Weaver weighted-ensemble fallback) and the RFT arm
(A=verifier-certified vs B=vote-certified, de-confounded, CI95 excl 0). Defensive: if
val ≥ 0.85 → full graft; if 0.75 ≤ val < 0.85 → rerank+headroom arm only (the rerank
question is answerable at 0.75+), defer RFT; if val < 0.75 → defer with the
passes-to-converge estimate for .384. `verifier_value_added` is the headline answer
and the DiffusionGemma gate.

### Phase C — North-star breadth + mandated slots (exp4139–4142)
- **exp4139 ARC-AGI-3 incremental** (`track: arc`): advance the solved-level count
  monotonically by ≥1 — next-unsolved non-spatial game's L1, OR (the non-spatial pool
  is thinning: `n_nonspatial=6`, exp4120 reported "no unsolved strict non-spatial
  candidates") advance an already-started game (e.g. r11l, the survey top-pick) to its
  next level. Offline explore→induce→GAP-4-verify→act. Honest no-solve is COMPLETE.
- **exp4140 SOTA-ingestion** (`track: research`): ingest GRAM (97% Sudoku-Extreme,
  the new SOTA baseline), the RLVR+GRPO TRM-as-thinking-reward precedent, and Weaver
  (weak-verifier ensemble); flag the strongest for .384. Real arXiv IDs/URLs only.
- **exp4141 verifier-registry + gaps hygiene** (`track: infra`): bit-exact GAP-4 ARC-1
  regression replay (vote 0.4516 → gated 0.5806) + record the .383 graft outcome.
- **exp4142 hardware continuity** (`track: hardware`): GateMate + PolarFire
  drive-to-terminal; KV260 opportunistic-confirm. SSH/USB-detect preconditions ONLY.

### Phase D — Close (exp4134 archive, exp4143 capstone)
exp4134 runs FIRST (archive .382 → activate .383, mechanical, codex). exp4143
aggregates the milestone into one honest headline: did the baseline reach ~0.87, and
did `verifier_value_added` flip true (unlocking DiffusionGemma) or honestly null?

---

## 5. Dependency graph

```
  exp4134 (archive/activate)  ── runs first
  exp4135 ─> exp4136 ─> exp4137   (accumulate chain; each resumes the prior's ckpt)
                          │
                          v
                       exp4138  (graft; defensive read of exp4137 baseline val)
  exp4139, exp4140, exp4141, exp4142   (independent; any order)
  exp4143 (capstone) ── reads 4135-4142, runs last, UNGATED
```

No `gated_on` fields: the accumulate chain and the graft are DEFENSIVE (they read
upstream artifacts and branch internally), and the capstone aggregates whatever
exists — the proven .380/.381/.382 pattern. This keeps every task producing
decision-grade signal even when an upstream pass under-converges.

---

## 6. Hardware requirements

- **2x RTX 3090 (CUDA), GPU 0 primary** — Phase A training (3 bounded ≤60min nano-trm
  passes) and the Phase B RFT arm. Phase B rerank + headroom are CPU verifier-scoring.
- **GateMate A1-EVB-2M + PolarFire SoC Discovery Kit** — continuity (exp4142),
  SSH/USB-detect preconditions only. KV260 is terminal (opportunistic confirm).
- No new hardware required. DiffusionGemma (the next scale-up) would need the
  Apache-2.0 DiffusionGemma 26B/4B-active weights cached — but it is GATED behind this
  milestone's graft verdict and is NOT part of .383.

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| 3 passes under-converge (land ~0.79–0.85, not 0.87) | Phase-B graft is graded/defensive: rerank+headroom answerable at val≥0.75; full RFT at ≥0.85; else defer to .384 with a passes estimate. Convergence still advanced materially. |
| Convergence plateaus below target | Each accumulate pass has a PLATEAU FLOOR: no-improvement → config audit, not blind retrain (Failed-Rerun Discipline). |
| Verifier "no value" is a false negative (no headroom) | Mandatory oracle-vs-vote positive control FIRST; an uninformative null is labeled FALSE_NEGATIVE_RISK, never propagated as "verifier fails". |
| Single executable verifier headroom-limited | Weaver weighted-ensemble fallback (executable validity + text-stat verifiers). |
| ARC non-spatial pool exhausted | exp4139 may advance an already-started game's level (incremental-progress discipline allows +1 level), not only first-solves. |

---

## 8. Compliance with mandated disciplines

- **Codex-Default v2** — all 10 tasks `agent_type: codex` + `model: gpt-5.5`. Planner
  (this doc) stays Opus via env.
- **Failed-Experiment Rerun / Exclusion-Manifest Cross-Check** — every
  lineage-continuation task (accumulate, graft, ARC, archive, capstone, SOTA, registry,
  hardware) carries an `operator_override:` per the 2026-05-29 standing auto-override
  (classes 1/2/3), citing the false-positive scope-match + the forward difference. No
  task scope-matches a retired exp_id in `ops/exclusion_manifest.yaml`.
- **Verdict Terminal-Prefix** — every `honest_verdict` spec starts `complete:`/etc.
- **Principle-Annotated Artifact Fields** — every REQUIRED ARTIFACT FIELD + gate
  carries a `principle:`.
- **Pre-Launch Preconditions** — every compute-bound task has a PRECONDITIONS step 0
  (uv + nano-trm + CUDA + stable checkpoint; SSH/USB for hardware) → `blocked_*` on miss.
- **Inference-Substrate Declaration** — each task declares `inference_substrate`.
- **ARC-AGI-3 Incremental-Progress Scoping** — exp4139 targets +1 level, never a
  FULL/all-levels solve.
- **SOTA-Ingestion Cycle** — exp4140 reserved; reliable channel, not /deep-research.
- **Hardware-Task Continuity** — exp4142 reserves one task for the non-terminal boards.
- **Reserved infra slots** — exp4134 (archive), exp4141 (registry hygiene), exp4143
  (capstone) cover the ≥2 infra-slot reservation.
- **Depth-Over-Breadth ethos (north-star §1)** — the milestone is weighted to the
  load-bearing converge+graft depth (4 of 10 tasks), not vN+1 re-measurement churn.
