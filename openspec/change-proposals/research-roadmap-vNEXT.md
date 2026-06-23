# Research Roadmap — Milestone 2026.06.429

**Planned by:** outer-loop (Claude Opus 4.8 planner), 2026-06-23.
**Sprint:** ARC-AGI-3 submission sprint through 2026-06-30 (ARC Prize Milestone #1, $25K). **7 days left.**

---

## 1. What the previous milestone (.428) proved

`.428` headline verdict: **`complete: capability_grew_56_to_57`**. Reproducible solve
capability grew (A3 banked **ft09 L3**, registry 56 → 57), the operator-resubmit package
stayed fresh (A4 live-submittable 57, beats the standing 33), but **both energy-driven
*generation* levers NULLED on live solve-rate**:

| .428 task | lever | live outcome |
|---|---|---|
| A1 exp4640 | graded `is_goal` as a **live goal-ENERGY heuristic** (operator menu #1) | **null** — `uniform_energy_ablation_failed` (goal-energy did not beat random energy → excluded from headline); live solve-rate 0.04 == 0.04 |
| A2 exp4641 | action-effect predictor **ranker → search EXPANSION PRIOR** (SOTA-D #3) | **null** — `no_deeper_solve`; `depth_of_live_solve_delta = 0`, `live_multi_level_solve_rate = 0.0`, first-win held at 0.59 |
| A3 exp4642 | self-play level-up bank | **win** — ft09 L3 offline-reproduced (56 → 57) |

This continues the now multiply-confirmed pattern from `.427`: a **GENERATION** lever (the
action-effect predictor) crossed the offline→live bridge for **efficiency / first-win**
(0.407 → 0.591, actions 2 → 1), but **nothing has crossed the bridge for SOLVE-RATE /
DEPTH**. The agent reaches the **first** live level-up at 0.59 but the **multi-level**
(≥2) live solve-rate is stuck at **0.0–0.04**.

## 2. The convergent diagnosis (sharpened by a week of outer-loop falsification)

Across 2026-06-23 the outer loop prototyped and **falsified four candidate levers**, each
on real code/data — turning the wall from "unknown" to **localized**:

| candidate | disposition | why dead |
|---|---|---|
| **Macro-action vocabulary / horizon collapse** (was PURSUE_HIGH) | **RETIRED** (`macro-vocab-prototype-finding-2026-06-23.md`) | the corpus solutions are *interleaved fixed-count runs of different actions* → the search is **breadth/guidance-bound, not depth-bound**; macros multiply branching without a guiding signal → strictly worse |
| **Click-heatmap-as-generator** (off-centroid clicks; was PURSUE_MED) | **RETIRED** (`click-heatmap-generator-falsified-2026-06-23.md`) | non-circular human-replay falsifier: **99.1%** of effective human clicks land on/near an object centroid; only **0.9%** truly off-object → coverage is not the problem |
| **just-explore exploration-schedule extraction** | **CLOSED** (`h2h-just-explore-vs-bare-explorer-2026-06-23.md`) | apparent head-to-head wins were ~⅗ offline replay-overhead budget artifact + ⅖ un-extractable frontier/navigation gap (bp35 = seed-noise, m0r0 = replay-overhead depth we already adapter-solve); the live `E3AgentPolicy` is stateful, so it never had the handicap |
| **goal-energy heuristic** (menu #1) | **NULL** (`.428` A1) | `uniform_energy_ablation_failed` — the single-game `is_goal` well did not beat random energy on the live search |

**The localized wall: GENERATION GUIDANCE.** The right *candidates* are already in the
pool (centroids enumerate every object; coverage is not the gap), and the wall is not
*depth* (macros lose). The binding constraint is **which action / which object, in which
order, to chain a *second* live level-up** — and a *learned guidance signal that actually
helps the live search at affordable cost*. The representation is **not** the bottleneck
either: `cross_game_features_v3` is **LOO-AUROC 0.725** (not the 0.503 the docs once said)
— "add more features" is dead too (`arc-representation-not-the-bottleneck-2026-06-23.md`).

## 3. The .429 bet — the two surviving, best-motivated levers

Both attack the **generation-guidance** wall on the **SCORED** `E3AgentPolicy`, both
`verifier_is_oracle: false`, both live-path-reachable (`arc_orphan_solver_lint`).

### A1 (HEADLINE, de-risked) — make the 0.725-AUROC discriminator AFFORDABLE live, then prove it GUIDES

The live value head was reverted (`value_weight` 5.0 → 0.0, `arc_competition_agent.py:68-78`)
**not because the representation is weak** (it is 0.725 AUROC, above chance) but because
computing `cross_game_features_v3` per frontier node cost **13 ms** (a pure-python
4-connectivity flood fill) → the 25-game sim **timed out** → value-routing solved fewer
games in bounded time. An outer-loop prototype (worktree `outer-loop/repr-prototype`,
commit `0de16b09e`, `arc-feature-cost-prototype-handoff-2026-06-23.md`) **proved the fix**:
replace the flood fill in `arc_agi3_world_model.objects()` +
`arc_value_learner._component_stats_from_grid` with `scipy.ndimage.label` (vectorised C, with
a pure-python fallback for the Kaggle kernel), **output verified identical** over 40 random
grids, **AUROC unchanged 0.725**, **13 ms → 0.64–0.87 ms/node** (0.64 ms × 5000 = 3.2 s, was
the 38 s timeout). The code comment already prescribes the sequel: *"raise value_weight ONLY
[after the cost fix]."*

A1 **productionizes the proven fix** into the live modules, routes by the cheap **v2+frame-Δ**
subset (0.64 ms, the highest-AUROC 0.742 class — dropping the dead-weight `action`/0.488 and
`predicate_distance`/0.536 classes), **raises `value_weight` off the 0.0 floor on the SCORED
agent**, and proves the live first-win / solve-rate goes **UP** with **no timeout**, vs the
`value_weight=0` baseline (bootstrap CI). This is the most de-risked lever available and a
direct outer-loop→conductor handoff. The honest gate: a live lift (CI excludes the baseline)
AND no timeout — **OR** an honest null that *disambiguates the prior value-head nulls as
distribution-shift / calibration (not cost)*, which B1 then localizes for `.430`.
`verifier_is_oracle: false` (the value head is oracle-distinct). `retire_if_same_verdict`.

### A2 (operator menu #2, the generation bet) — energy-as-fitness QD evolution over action-sequences

The operator's energy-config-space directive (TOP-PRIORITY, `2026-06-22`,
`arc-generation-wall-energy-config-space-2026-06-22.md`) sequences a 3-step menu: **#1**
goal-energy (done, `.428`, nulled) → **#6** macro vocabulary (done, retired) → **#2
energy-as-fitness QD evolution** — "*THIRD, depends on #1+#6 ... highest direct leverage on
`winner_generated` ... the generative-fitness tier.*" Both prerequisites are now resolved, so
**#2 is the operator's explicit next sequenced lever**, independently SOTA-flagged #1 for
`.429` by the `.428` ingestion (exp4649: arXiv:2605.28814 BES + 2308.05483 QD-under-sparsity +
2504.01915 unsupervised-QD).

A2 wraps the goal-energy + action-effect rollouts in a **MAP-Elites-style archive over
multi-action sequences**: mutation (insert/delete/swap/splice) + shared-state crossover;
**fitness = goal-energy delta + action-effect cell-recall + first-win action-efficiency**;
behavior descriptors from action-effect predictions for diverse niches. Unlike the dead macro
lever (horizon-collapse) and the dead goal-energy heuristic (single rollout), QD is
**population-based generation with diversity** — it *generates NEW winning sequences* the
best-first search never proposed. The operator's note states the bet plainly: it **"lives
under the P0.1 shadow"** (energy provably fails to generate *de-novo*); the entire wager is
that **population-seeding + non-null QD priority + behavioral diversity escapes it**.
The falsifiable gate is sharp: on ≥1 hard game (pre-confirmed reachable via `cell_recall`),
QD puts a **winner in the pool** (`winner_generated`, offline-reproduced) that primitive
best-first does **not** reach at **equal budget**, AND **beats a random-mutation / no-energy-
fitness ablation** (else it is the search, not the energy). `verifier_is_oracle: false`.
`retire_if_same_verdict`.

## 4. Milestone shape (12 tasks, exp4651–exp4662)

```
Phase 0  exp4651  archive .428 -> activate .429 (mechanical, codex)

Phase A  ARC NORTH STAR (majority; SCORED path; verifier_is_oracle:false)
  A1 exp4652  HEADLINE  compute-cost value-routing fix -> affordable discriminator GUIDES live
  A2 exp4653  MENU #2   energy-as-fitness QD evolution (generation lever; winner_generated gate)
  A3 exp4654  GUARANTEE self-play bank +1 reproducible level (57 -> 58+) + train learned verifier
  A4 exp4655  SCORE     refresh operator-resubmit package (stay > 33)
  A5 exp4656  LEARN     persist+transfer the winning A1/A2 primitive (cross-game)
  A6 exp4657  INTEGRATE fold the winning config into SUBMITTED_AGENT_CONFIG + parity green

Phase B  INFRA (2 reserved slots)
  B1 exp4658  value-routing live CI-gate + distribution-shift-vs-calibration diagnostic (feeds .430)
  B2 exp4659  adversarial_verify hardening: QD-without-random-mutation-ablation + value-routing-cost-control guards

Phase C  HARDWARE (1 per-board continuity slot)
  C  exp4660  KV260 / PolarFire / GateMate reachability audit (SSH-not-SD-card)

Phase D  SOTA-INGESTION (1 slot -> .430)
  D  exp4661  focused pass on the surviving generation-guidance class (hierarchical subgoal / factored executable model)

Phase E  CAPSTONE
  E  exp4662  capstone .429 + headline decision (did the affordable value head / QD cross the bridge?) + G1-G4
```

## 5. Architecture — where the two levers attach (the live generation-guidance pipeline)

```
                          SCORED E3AgentPolicy (live, stateful, hidden game)
                                         |
   perception (object/frame-D features, NOW 0.64 ms/node via scipy.ndimage.label  <-- A1)
                                         |
                 candidate pool  =  rich_action_candidates (object centroids; coverage OK)
                                         |
        +--------------------------------+---------------------------------+
        |  A2: energy-as-fitness QD       |  A1: value-routing (value_weight>0)
        |  GENERATES new action-sequences |  GUIDES which frontier branches expand
        |  (MAP-Elites archive, mutation/ |  (cross_game_features_v3, 0.725 AUROC,
        |   crossover; fitness = goal-    |   now AFFORDABLE -> no timeout)
        |   energy + cell-recall + eff)   |
        +--------------------------------+---------------------------------+
                                         |
                       graph_explore_solve_v2  (best-first search)
                                         |
                    arc_solver_kit.reproduce  (offline reproduction gate)
                                         |
              registry bump + learned-verifier checkpoint (A3) + package refresh (A4)
```

A1 attacks the **guidance** half (make the learned discriminator affordable so it can route
the live search); A2 attacks the **generation** half (population-search that adds the missing
winning sequence to the pool). Complementary: A1 says *which branch*, A2 says *propose the
branch that was never there*.

## 6. Dependency graph

```
exp4651 (archive/activate)
   |
   +-- exp4652 A1 (value-routing)   --\
   +-- exp4653 A2 (QD evolution)    ---+--> exp4655 A4 (package)  --> exp4657 A6 (integrate)
   +-- exp4654 A3 (self-play bank)  --/        |                         |
   |                                            +--> exp4656 A5 (persist+transfer)
   +-- exp4658 B1 (CI-gate + diagnostic; reads A1)
   +-- exp4659 B2 (adversarial_verify hardening; reads A1/A2 fixtures)
   +-- exp4660 C  (hardware continuity)
   +-- exp4661 D  (SOTA-ingestion -> .430)
   |
   +-- exp4662 E  (capstone; reads A1..A6, B1, B2, registry)
```

A3 / A4 are **independent** of A1/A2 so the level-up guarantee and the operator-resubmit
package hold even if both headline levers null.

## 7. Hardware requirements

- **No 3090.** Any LLM arm runs on the iGPU **Qwen3.5-9B-MTP** and declares
  `live_llm_inference` for that arm only. A1/A2 are offline-arcade live-search measurements
  (`verifier_ensemble_against_cached_candidates`); the value head + QD scorer are small CPU/iGPU
  computations.
- **C (continuity):** KV260 via `ssh kria` (SSH-reachability precondition, NEVER a host SD-card
  device check), PolarFire via `ssh polarfire`, GateMate via `openFPGALoader -c dirtyJtag
  --detect`. `.428` C recorded kv260+polarfire reachable, gatemate USB undetected.

## 8. Discipline invariants carried

- **ARC sprint forcing function** (through 2026-06-30): majority ARC; >=1 level-up attempt that
  BANKS a reproducible level (A3); 2 reserved infra (B1/B2); 1 per-board hardware (C); 1
  SOTA-ingestion (D). All experiments `codex`/`gpt-5.5`; planner/retro stay Claude Opus.
- `verifier_is_oracle: false` on every value claim; `solve_provenance` on every ARC solve task.
- `live_path_reachable` HARD gate on A1/A2 (`arc_orphan_solver_lint`) — no orphaned
  `scripts/experiments` solver; the lever must be reachable from `E3AgentPolicy` / `arc_graph_explore`.
- Every energy/QD value claim must beat its **ablation** (A1: no-timeout cost control; A2:
  random-mutation / no-energy-fitness) — the `.425-`.428 B2 guard lineage extended in B2.
- Submission stays **operator-only** (External Publication Discipline) — tasks PREPARE +
  offline-validate, never submit.
- `paper_ready` (G1–G4, FoVer 0.9131) is a frozen invariant — `.429` adds the
  bridge/generation-guidance lens, not a new headline.

## 9. Honest risks / null-paths (pre-registered)

- **A1 may partially null** even with the cost fixed — per-game LOO variance is 0.379→1.0, and
  the value head is a *reranker* (the class that has nulled). Then A1's value is **diagnostic**:
  it removes the timeout confound and disambiguates the prior nulls as distribution-shift /
  calibration (B1 localizes which). That is real progress, not a wasted slot.
- **A2 lives under the P0.1 shadow** (energy fails to generate de-novo). If QD nulls
  `winner_generated` with a passing random-mutation ablation, the energy-as-fitness generator is
  retired for the sprint and the residual logged as a Missing-Verifier / bridge gap. The
  `retire_if_same_verdict` gate makes that retirement mechanical.
- The dead levers (macro, click-heatmap, just-explore schedule, goal-energy heuristic) are
  **not re-proposed** — A2 is explicitly *not* macro-vocab (population generation with diversity,
  not horizon-collapse) and *not* the goal-energy heuristic (population search, not single rollout).
