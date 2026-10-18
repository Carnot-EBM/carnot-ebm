# Research Roadmap v417 — ACTION EFFICIENCY (the ARC live wall)

**Milestone:** 2026.06.417
**Planned:** 2026-06-20 (UTC) — ARC-AGI-3 Submission Sprint, 10 days to the 2026-06-30 deadline
**Predecessor:** 2026.06.416 (`openspec/change-proposals/research-roadmap-v416.md`)
**Planner:** Claude Opus 4.8 (sprint: experiments codex/gpt-5.5; planner/retro stay Opus)

---

## TL;DR

The live ARC agent's wall is **action efficiency**, not solve-rate and not config
tuning. The leaderboard score is `min(human_actions/agent_actions, 1)^2`, so the live
StepwiseExplorer's **~7760 actions to find a ~21-action solution** scores ~0.08 even
when it SOLVES. `.414/.415/.416 confirmed this three independent ways: the guidance
score-levers (value_weight, frame-change predictor, energy-augmented ranking) all
returned honest NULLs, and three cascade fixes restored speed + solve-rate (1→4 solved,
6→0 timeouts) but left median actions **unchanged at ~7760**. The gap is the explorer's
**exploration strategy**, which is architectural.

`.417 has a single question: **make the live explorer find solutions with FEWER
actions.** Every ARC task is measured by median actions-to-first-levelup on the 8-game
gate (`scripts/kaggle/arc_local_submission_gate.py`, baseline 7760), at equal-or-better
solve-rate. This is the StochasticGoose lever (the leaderboard leader's whole edge) and
it directly multiplies the score.

---

## What `.416 proved (and the three-milestone pattern it closed)

| `.416 phase | Result | Read |
|---|---|---|
| A1 value_weight re-measure | **NULL** — keep weight 0 (1/7 held-out) | the v3 head (LOO 0.674) at weight>0 does NOT beat bare-BFS within the ~6.5 min/game eval budget at FULL per-node cost |
| A2 frame-change predictor re-run | **NULL** — "staged corpus shortfall" | the staged human-replay mirror was attribution-format, not usable training examples |
| A3 energy-augmented ranking | **NULL** — `P(frame_change)·(−ΔE)` ranking | ranking re-orders but does not PRUNE → no action saved |
| A4 ka59 HUD-register deepen L2 | honest residual (not reproduced) | grid-only state omits HUD registers (the deepening-tail root cause) |
| A5 cd82 adapter deepen L2 | **SUCCESS** — offline-reproduced (+1 level) | the level-up that grew `reproducible_total_levels` to 47 |
| B2 lazy value-eval prototype | **232.69x speedup**, quality preserved 80/80 | top-K / frame-hash cache makes value-head eval cheap → unblocks best-first at weight>0 |

**The closed pattern:** RANKING with a slow-but-good signal does not reduce actions —
because the explorer still EXPANDS every salient candidate per node (breadth = the
7760). The lesson is unambiguous: **PRUNE, don't rank**, and **don't expand on easy
frames**. `.416 B2 (lazy eval) removes the only remaining excuse for the value head
("too slow per node").

**Current ARC state (authoritative):** `reproducible_total_levels: 47`,
`reproducible_total_games: 24`, first live submission 2026-06-17 = 13 levels / 11 games
(score 0.08); the next-online-play gate is "beat 13 levels." Held-out solve-rate
**0.143** (1/7); variant-transfer **0.28** (7/25). Leaderboard: Tufa StochasticGoose
1.21 (CNN frame-change predictor), a wall at 0.63–0.70, Carnot 0.08. All winners < 13%.

---

## The three biggest gaps vs the PRD vision (this milestone's frame)

1. **Action efficiency — the score lever (PRIMARY).** The explorer is effect-blind:
   centroid-click + RESET-replay, expanding all salient candidates. Closing this is the
   only thing that moves 0.08. `.417 attacks it four ways: prune no-ops with a working
   frame-change predictor (A1), a human-imitation action prior (A2), an adaptive
   per-step budget that doesn't expand on easy frames (A3), and best-first over the now-
   cheap lazy value head (A4).
2. **Cross-game generalization — the unsolved part (the moat).** Held-out solve-rate
   0.143, verifier LOO-AUROC 0.503 (chance). The frame-change predictor is trained
   POOLED across all games (the "persist action-effect across games" idea) — a
   transfer-relevant signal, measured held-out.
3. **Hidden-state world models — the deepening-tail.** ar25/ka59 L2 stall on grid-only
   state (HUD registers invisible). Not the `.417 headline (action efficiency is), but
   the level-up guarantee (A5) deepens a clean derive-from-env game instead, and the
   gap stays logged for the moat track.

---

## Architecture — where `.417 intervenes

```
                         ARC live agent (submitted)  —  the thing the leaderboard scores
                                    │
        make_carnot_agent → E3AgentPolicy  (frame-only, env._game BLOCKED live)
                                    │
                          StepwiseExplorer  ←──────────────── THE 7760-ACTION WALL
                                    │
              rich_action_candidates(frame) → {salient candidates}
                                    │
   ┌────────────────────────────────┼─────────────────────────────────────────────┐
   │  A1 PRUNE          A2 PRIOR        A3 ADAPTIVE BUDGET     A4 BEST-FIRST         │
   │  frame-change      imitation       ACT/PonderNet gate    lazy value head        │
   │  predictor →       prior (342      (energy/value margin   (.416 B2, 232x) →      │
   │  drop no-op        human replays    + predicted-no-op     value_weight>0         │
   │  candidates        + self-superv.   + frame novelty):     re-measured affordably │
   │  BEFORE expansion  marginal)        easy→1, hard→expand                          │
   └────────────────────────────────┼─────────────────────────────────────────────┘
                                    │
                          A6 INTEGRATION: wire whichever levers beat 7760 into the
                          SUBMITTED agent + re-measure the 8-game gate (the headline)
                                    │
                    arc_solver_kit.reproduce  ← every claim offline-reproduction-gated
```

**Training data for A1/A2 (robust, no external dependency):** the PRIMARY corpus is
**self-supervised `(frame, action, next_frame)` transitions generated from the offline
arcade** — zero quota, deterministic, all 25 games, no network. The human-replay corpus
(`data/arc_public_demo_human_replay_corpus`) is an OPTIONAL bootstrap. This design
removes the external-corpus dependency that nulled the predictor in `.415 AND `.416.

---

## Phases & tasks (12 tasks)

### PHASE 0 — Transition
- **exp4510** archive `.416 → activate `.417 (mechanical; codex; assert YAML parses + pre-test gate green; record the true `.416 close-state: levels 47/24, held-out 0.143, variant 0.28, three score-levers NULL, lazy-eval 232x shipped).

### PHASE A — ARC NORTH STAR (action efficiency; the sprint majority)
- **exp4511 (A1, PRIMARY, SCORE-LEVER)** — frame-change / clickability predictor that
  **PRUNES** no-op candidates. Self-supervised corpus from the offline arcade (human
  replays optional bootstrap), small conv predictor, wired into `rich_action_candidates`
  to DROP candidates predicted no-op BEFORE expansion. Gate: median actions-to-first-
  levelup STRICTLY < 7760 on the 8-game gate, solve-rate not reduced; positive control +
  FALSE_NEGATIVE_RISK null guard. *prior_failures: exp4490/exp4501.*
- **exp4512 (A2, SCORE-LEVER)** — imitation / behavior-cloning ACTION PRIOR from the 342
  human replays (+ self-supervised marginal fallback) so the explorer tries human-like
  efficient actions FIRST. Gate: actions saved at equal solve-rate.
- **exp4513 (A3, SCORE-LEVER)** — verifier-grounded **ADAPTIVE per-step budget**
  (ACT/PonderNet; LoopWM arXiv:2606.18208 citable). Cheap per-step gate (energy/value
  margin + predicted-no-op + frame novelty): easy frame → commit 1 candidate, ambiguous →
  expand. Cuts actions by NOT expanding when the frame is easy. Zero new model/training.
- **exp4514 (A4, SCORE-LEVER)** — best-first with the **LAZY value head** (`.416 B2,
  232x). Re-measure `value_weight` ∈ {0, 0.5, 1, 2, 5} with the cheap eval; raise
  `SUBMITTED_VALUE_WEIGHT` ONLY if a weight beats 0 on solve-rate AND stays in budget.
  *prior_failures: exp4500 (the `.416 null was at FULL per-node cost).*
- **exp4515 (A5, LEVEL-UP GUARANTEE; MANDATORY)** — DEEPEN a shallow (L1) game to L2 via
  a derive-from-env GameAdapter; **bank +1 offline-reproduced level**
  (`offline_reproduced=true`, new deeper level). Target the graph-explore family
  (su15/sp80/cn04/m0r0/sk48) — clean derive-from-env L2 predicate, NOT the HUD-register
  stall games. *prior_failures: exp4503; operator_override: level-up guarantee.*
- **exp4516 (A6, INTEGRATION + HEADLINE METRIC)** — wire whichever of A1–A4 beat 7760
  into the SUBMITTED agent (parity test updated), close the forward-edge navigation loop
  (candidate 5: confirm `_shortest_path` is actually used vs replay), and re-measure the
  8-game gate end-to-end. If NO lever beats 7760, keep the bare explorer and report the
  honest null. *gated_on A1 actions-delta (soft).*

### PHASE B — Reserved infrastructure (2 slots)
- **exp4517 (B1, INFRA; OVERDUE)** — repair the milestone-scoped timing detector. The
  false-zero observability gap has recurred every retro `.363→`.416 and is flagged as THE
  operational bottleneck. mtime scan + ops/changelog-window fallback + `detector_gap_
  suspected` emission + write-time `duration_s`/`compute_bound` stamping.
- **exp4518 (B2, INFRA)** — canonicalize the action-efficiency metric harness:
  `arc_local_submission_gate.py` as the CI-guarded median-actions-to-solve dashboard
  across the 8 games + **per-lever attribution** so each A-task's actions-saved is
  measured the same way (kills cherry-picked baselines).

### PHASE C — Hardware continuity (MANDATORY; 1 per attached board)
- **exp4519 (C)** — KV260 (SSH reachability, NOT SD-card) + GateMate (USB detect) +
  PolarFire (SSH) audit + next forward step. GateMate has been DirtyJTAG-unreachable
  for many milestones; record reachability honestly, do not fabricate.

### PHASE D — SOTA ingestion (reserved; bleeding-edge track = action efficiency)
- **exp4520 (D)** — ingest SOTA for **learned action-effect models / affordance learning
  / experience-replay-for-search** (StochasticGoose CNN, PersistentAEM, prioritized
  replay, IDA*), map the strongest 3–5 onto the `.418 roadmap. Reliable channel only
  (`sweep_*.py` + low-concurrency WebSearch/WebFetch); `/deep-research` is BANNED in-loop.

### PHASE E — Capstone
- **exp4521 (E)** — capstone v417: the action-efficiency scorecard. Did median actions-
  to-solve drop off 7760? Which lever moved it? Held-out solve-rate + variant transfer;
  submission-readiness (>13 levels, operator-only submit). Skip any `flagged_adversarial`
  artifact; honor `verifier_is_oracle`.

---

## Dependency graph

```
exp4510 (transition)
   ├─ exp4511 (A1 prune predictor) ──────────────┐
   ├─ exp4512 (A2 imitation prior) ──────────────┤
   ├─ exp4513 (A3 adaptive budget) ──────────────┼─→ exp4516 (A6 integration + 8-game gate)
   ├─ exp4514 (A4 lazy best-first) ──────────────┘        │
   ├─ exp4515 (A5 deepen +1 level; level-up guarantee)    │
   ├─ exp4517 (B1 timing detector)                        │
   ├─ exp4518 (B2 metric harness) ───────────────────────→┘ (canonical metric A1–A4/A6 report against)
   ├─ exp4519 (C hardware continuity)
   └─ exp4520 (D SOTA ingestion) ────────────────→ exp4521 (E capstone) ← reads all A/B results
```

A1–A4 are independent score-levers (parallel-safe). A6 integrates whichever win. B2
ships the canonical metric the A-tasks report against (run early). E aggregates.

---

## Hardware requirements

- **RTX 3090 ×2 (CUDA):** A1's small conv predictor trains here if a GPU path is wired;
  CPU is acceptable (the predictor is small, the corpus is self-generated). Not a GGUF
  load — no `live_llm_inference`.
- **Offline arcade (CPU, zero quota):** the substrate for A1–A6 and A5's reproduction
  gate — `verifier_ensemble_against_cached_candidates`.
- **KV260 (SSH) / GateMate (USB) / PolarFire (SSH):** continuity audit only (PHASE C).
- **16GB / offline / kernels-only safe:** every A-lever must respect the Kaggle eval
  envelope (no network, time-bounded). A3 (adaptive budget) is explicitly zero-new-model.

---

## Compliance with standing disciplines

- **ARC-AGI-3 Submission Sprint Forcing Function (through 2026-06-30):** MAJORITY ARC
  live-solving — A1–A6 + B2 (ARC metric) + D (ARC SOTA) + E (ARC capstone) = 9/12 ARC.
  All experiments `codex`/`gpt-5.5`; planner/retro stay Opus.
- **ARC Level-Up Attempt Guarantee:** A5 banks +1 offline-reproduced level
  (`scripts/arc_levelup_guarantee_lint.py` ≥1).
- **ARC Solve Reproducibility:** every A-claim is `arc_solver_kit.reproduce`-gated;
  `offline_reproduced`/`reproduced_levels` emitted; registry updated.
- **Overdue-Priority + reserved-infra slots:** B1 (timing detector, overdue `.363→`.416),
  B2 (metric harness) = 2 infra slots.
- **Hardware-Task Continuity:** PHASE C, 1 per attached board (KV260 SSH-not-SD-card).
- **SOTA-Ingestion Cycle:** PHASE D, reliable channel, real arXiv IDs, `/deep-research`
  banned in-loop.
- **Failed-Experiment Rerun + Inference-Substrate + Verdict Terminal-Prefix +
  Principle-Annotated Fields + Circularity/Oracle-Distinctness:** honored per-task (see
  YAML `prior_failures`, `operator_override`, `inference_substrate`, principle notes).
- **No leaderboard submission in-loop** — submission is operator-only; tasks end at
  `submission_package_ready`.
