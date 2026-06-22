# Research Roadmap — Milestone 2026.06.424

**GENERATION COMPLETENESS — finally attack the candidate-GENERATION wall directly: WIRE the toolkit approaches into the held-out generation harness so the selected approach actually RUNS and GENERATES the winner (`winner_generated` 1/25 → up), the un-tried fix after re-ranking / routing / verifier-expansion all nulled; keep the 53-level package operator-resubmit-ready (beat 33) and bank +1 more level**

- **Planned by:** outer-loop (Claude Opus 4.8 planner), 2026-06-22.
- **Milestone doc for:** `research-roadmap-next.yaml` (2026.06.424).
- **Sprint:** ARC-AGI-3 submission sprint through **2026-06-30** (CLAUDE.md
  "ARC-AGI-3 Submission Sprint Forcing Function" — majority ARC; ≥1 level-up
  bank; 2 reserved infra; 1 per-board hardware; 1 SOTA-ingestion; all
  experiments `codex`/`gpt-5.5`; planner+retro stay Claude Opus). **8 days to deadline.**

---

## 1. What the previous milestone (.423) proved

`.423` ("CLOSE THE LIVE-SUBMISSION GAP, flanked by the feature-ROUTER") landed
**one decisive win, one capability bank, and a FOURTH confirmation of the generation wall.**

| Phase | Result | Verdict |
|---|---|---|
| **A1 close live-submission gap (exp4580)** | `live_submittable_level_count` 33 → **53** (`count_delta=20`); banked replayable trajectories for all 24 reproduced games; env-adaptive re-solve recovered sc25; `ready_for_operator_submit=True` | ✅ **the headline win** |
| **A2 level-up self-play (exp4581)** | **ar25 L1→L2 banked → `reproducible_total_levels` 53 → 54**; learned verifier checkpointed | ✅ the capability bank |
| A3 feature-ROUTER for seen→hidden transfer (exp4582) | `generic_transfer` flat at **0.04** (== baseline == random-route control → TAUTOLOGY-flagged, false-negative-risk open) | honest null (4th confirmation) |
| A4 diversity-floor transfer (exp4583) | no transfer (first-win not up on held-out variants) | honest null |
| A5 persist primitive (exp4584) | `env_adaptive_resolve_operator` persisted; drift-recovery transfer on s5i5/ft09/sb26, no new bank | ordering/recovery-only |
| A6 integration (exp4585) | `integrated_live_submittable_54_above_33` (A1+A2 wired) | ✅ integrated |
| B1/B2/C/D/E | live-submittable co-headline metric; offline-arc METHODOLOGY guard; hardware audit; skill-routing SOTA ingested; capstone | shipped |

**The decisive diagnostic the planner must build on (exp4582 `winner_generated`):**
across 25 held-out variants the winning candidate was **GENERATED for only 1** (24 not
generated). The residual is **mechanical**, not modeling:

| mechanic class | routed approach | wired into variant harness? | unsolved |
|---|---|---|---|
| avatar_navigation | `goal_distance_astar` | **NO (`variant_wired=False`)** | 12 |
| click_connect | `goal_distance_astar` | **NO (`variant_wired=False`)** | 3 |
| keyboard_graph | `systematic_bfs` | yes | 7 |

**15 of the 24 misses are because the right approach is NOT WIRED into the variant
generation harness.** The `.423 feature-ROUTER correctly *classified* the mechanic and
*selected* the approach — but selecting an approach that never runs generates nothing.
This is "integration, not modeling" applied to GENERATION, and it is the un-tried lever.

## 2. Settled facts — do NOT re-derive or re-build (rerun-discipline)

1. **`cross_game_features_v3` ALREADY transfers — LOO-AUROC 0.674** (exp4545 `.418;
   in-sample 0.871). GAP-ARCH-FEATURES (the structural-feature verifier-transfer item)
   is **closed above the 0.6 bar.** Do NOT re-propose "build v3 to beat chance." The OPEN
   question is using that transferring energy to **guide GENERATION**, not to re-rank.
2. **Verifier-guided best-first EXPANSION REGRESSED** (.422 A2 exp4569, `transfer_delta=-0.04`;
   weekend `value_weight>0` 2/11 vs diversity 4/11; reset-replay navigation tax). Do NOT
   re-run a learned-value EXPANSION priority as a headline (honor the operator heads-up).
3. **Re-ranking / routing a FIXED pool adds 0** (.421 A6 `ordering_gain=0`, .423 A3): the
   winner is never in the pool. Generation, not ranking, is the binding constraint — now
   confirmed FOUR times. **`.424 stops re-ranking and fixes generation.**

## 3. The `.424 strategy — three thrusts, all genuine sprint progress

**Thrust 1 (HEADLINE — A1): GENERATION COMPLETENESS.** Wire each mechanic-class approach
into the held-out variant generation harness (`measure_generic_transfer_over_variants`'s
`variant_runner`) so the SELECTED approach actually RUNS and GENERATES candidates:
`arc_goal_distance.goal_distance_solve` for avatar_navigation + click_connect (the 15
`variant_wired=False` misses), `graph_explore_solve_v2`+diversity for graph classes, the
LLM reasoner for the residual tail. Headline metric: `winner_generated_rate` (1/25 baseline)
+ `generic_transfer` (0.04 baseline, CI must exclude it) + median actions-to-first-levelup.
`verifier_is_oracle: false`. This is high-confidence (mechanical wiring grounded in exp4582's
exact residual) and not a re-run of any failed approach (those re-RANKED or EXPANDED a fixed
pool; this WIRES generators in).

**Thrust 2 (DEEP MOAT — A3): generation guidance via an objective goal-distance / structural
energy** for the wired-but-still-failing classes (keyboard_graph BFS, 7 unsolved) and the
no-avatar config/click tail. The operator's 2026-06-20 energy-augmented-ARC spine: objective
energy over game-agnostic STRUCTURE that biases the explorer's action proposals toward the
goal (a GENERATION prior), NOT a learned-value best-first EXPANSION priority (which regressed).
`verifier_is_oracle: false`; measure `winner_generated_rate` + action cost; honest null if it
does not help.

**Thrust 3 (CAPABILITY + SCORE): bank more, keep the package fresh.** A2 runs the standing
self-play loop EVERY milestone (operator 2026-06-21) to bank +1 NEW reproducible level on a
rotated game (`reproducible_total_levels` 54 → 55+) and train+checkpoint the learned verifier;
A4 folds the new bank into the refreshed 53→54+ package and keeps it `ready_for_operator_submit`
so the operator can resubmit and beat 33. Submission stays operator-only.

## 4. Architecture (where each task plugs in)

```
   ARC-AGI-3 live env (25 public + hidden) ── offline arcade (deterministic sim, zero quota)
                                  │
        ┌─────────────────────────┼──────────────────────────────────────────┐
        │  GENERATION (the wall)   │   VERIFICATION (transfers: v3 LOO 0.674)  │
        │                          │                                          │
   feature classifier ──► APPROACH DISPATCH ──► candidate pool ──► verifier-routed search
   (recommend_approach)   (A1: WIRE goal_distance_astar /        (rank within pool;
                           graph_explore+diversity / LLM into     re-ranking already
                           variant_runner — the un-tried fix)     proven 0-value)
                                  │                                          ▲
                          (A3) goal-distance / structural ENERGY prior ──────┘
                               biases proposals toward the goal (generation, not expansion)
                                  │
        reproduction gate (arc_solver_kit.reproduce) ──► only reproduced levels count
                                  │
   A2 self-play loop ──► bank +1 level (54→55+) + train learned verifier checkpoint
   A4 refresh package (53→54+, operator-resubmit-ready, beat 33) ── A6 wire winners into SUBMITTED_AGENT_CONFIG
```

Co-headline metrics (capstone E, formalized by B1): `reproducible_total_levels` (capability,
54) · live-submittable count (honest leaderboard score, 53; baseline 33) · `generic_transfer`
(seen→hidden, 0.04) · action efficiency · **`winner_generated_rate` (the generation-vs-ranking
gap, NEW in B1).**

## 5. Phases & tasks (12 tasks)

- **Phase 0 — transition (exp4591):** archive `.423 → activate `.424; record the true `.423
  close-state (A1 closed gap to 53; A2 ar25 L2 → 54; A3 router null `winner_generated=1/25`;
  generation-not-ranking quadruply-confirmed).
- **Phase A — ARC north star (the majority):**
  - **A1 (HEADLINE, exp4592):** GENERATION COMPLETENESS — wire the toolkit into the variant
    `variant_runner`; measure `winner_generated_rate` + `generic_transfer` + actions.
  - **A2 (LEVEL-UP GUARANTEE + self-play, exp4593):** standing loop, bank +1 level (54→55+),
    rotate target (prefer sk48 L1→L2 graph family; skip ka59 / cd82-L3 / sp80-L3 / su15-L3
    dead-ends), train+checkpoint the learned verifier.
  - **A3 (DEEP MOAT, exp4594):** goal-distance / structural-progress ENERGY as a GENERATION
    prior for the wired-but-failing classes; NOT a learned-value EXPANSION priority.
  - **A4 (SCORE — keep package fresh, exp4595):** fold A2's new bank into the refreshed
    package, keep `ready_for_operator_submit` (beat 33), extend env-adaptive recovery.
  - **A5 (SELF-LEARNING persist + transfer, exp4596):** persist `.424's winning primitive
    (A1 dispatcher OR A3 goal-energy prior) into `arc_solver_kit` + registry; measure transfer.
  - **A6 (INTEGRATION + headline metric, exp4597):** wire winners into `SUBMITTED_AGENT_CONFIG`
    + the refreshed package; re-measure `winner_generated_rate` + `generic_transfer` +
    live-submittable count; keep `test_arc_submitted_agent_parity.py` green.
- **Phase B — reserved infra:**
  - **B1 (exp4598):** formalize `winner_generated_rate` (generation-vs-ranking gap) as a
    capstone co-headline metric. Asserting tests.
  - **B2 (exp4599):** TAUTOLOGY null-delta false-flag guard in `adversarial_verify.py` — an
    artifact declaring an explicit null-delta (== 0) WITH a `null_delta_methodology_note` +
    a passing positive control is downgraded from CRITICAL TAUTOLOGY to annotated-WARN (so
    the capstone reads honest nulls instead of excluding them); keep TAUTOLOGY CRITICAL for
    genuinely-distinct-metric bit-identity. Asserting tests.
- **Phase C — hardware continuity (exp4600):** per-board reachability audit (KV260 SSH,
  GateMate USB, PolarFire SSH).
- **Phase D — SOTA-ingestion (exp4601):** ingest candidate-GENERATION / world-model-induction
  / exploration SOTA (Code World Models 2510.04542; Adaptive World Models in Novel Games
  2507.12821; One Life to Learn 2510.12088; predictive-WM exploration 2502.13200; ScreenExplorer
  2505.19095; ARC-AGI-3 report 2603.24621) mapped onto A1/A3; flag the strongest for `.425.
- **Phase E — capstone (exp4602):** the scorecard — did A1 wiring raise `winner_generated_rate`
  + `generic_transfer` above 0.04 / lower actions? Did A3 goal-energy help the wired-but-failing
  classes? Did A2 grow `reproducible_total_levels` (54→55+)? Report all co-headline metrics +
  `winner_generated_rate` (B1). `verifier_is_oracle:false` on every value claim.

## 6. Dependency graph

```
exp4591 (P0 transition)
   ├─► exp4592 (A1 generation completeness — HEADLINE)
   ├─► exp4593 (A2 level-up + self-play — independent, guarantees a bank)
   ├─► exp4594 (A3 goal-energy generation prior — independent of A1)
   ├─► exp4598 (B1 winner_generated_rate metric — independent infra)
   ├─► exp4599 (B2 TAUTOLOGY null-delta guard — independent infra)
   ├─► exp4600 (C hardware — independent)
   └─► exp4601 (D SOTA-ingestion — independent)
exp4592/4593/4594 ──► exp4595 (A4 refresh package; folds A2 bank)
exp4592/4593/4594/4595 ──► exp4596 (A5 persist the winning primitive + transfer)
exp4592..4596 ──► exp4597 (A6 integration: wire winners into SUBMITTED_AGENT_CONFIG)
ALL ──► exp4602 (E capstone scorecard)
```

## 7. Hardware requirements

- **A1–A6, B1–B2:** offline arcade (deterministic sim) + the learned verifier (CPU forward
  pass) — `verifier_ensemble_against_cached_candidates`, zero LLM/GPU, zero quota. If A3's
  generation prior optionally invokes the LLM proposer, it MUST use **Qwen3.5-9B-MTP on the
  iGPU** (the frozen live-submission generator, [[project_arc_live_generator]]), NEVER the 3090s.
- **C:** SSH/USB board reachability only (KV260 `ssh kria`, GateMate `openFPGALoader --detect`,
  PolarFire `ssh polarfire`). No bitstream build this milestone unless a board is mid-bringup.
- **D:** network for the reliable arXiv / Semantic-Scholar sweep (no `/deep-research`).

## 8. Honest accounting / discipline

- `verifier_is_oracle: false` on every value claim (A1/A3/A4/A5/A6) — these are generation +
  packaging levers, oracle-distinct from the executable win-check (no circular moat claim).
- Every newly-solved variant/level must **offline-reproduce** (`arc_solver_kit.reproduce`) to
  count — only reproduced levels touch `reproducible_total_levels` or the live-submittable count.
- Every transfer/efficiency null carries an explicit delta + `null_delta_methodology_note` + a
  WORKING positive control (the `.423 A3 broken-control / false-negative-risk-open trap must NOT
  recur — B2 makes the honest-null carve-out mechanical so the capstone reads it).
- Submission is **operator-only** (External Publication); `.424 PREPARES + offline-validates the
  resubmit package and emits `ready_for_operator_submit`. It never submits.
- The capstone (E) skips `flagged_adversarial` artifacts EXCEPT the annotated null-delta
  carve-out; honors the `.423 B2 offline-arc-METHODOLOGY guard, the `.422 B2 learned-CNN DURATION
  guard, and the positive-control-failed guard.
