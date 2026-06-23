# HANDOFF: ARC feature-cost prototype (12-15x cheaper routing features) — proven, ready to productionize

Date: 2026-06-23 · Outer-loop prototype (worktree `outer-loop/repr-prototype`, commit `0de16b09e`) → conductor productionizes.

## What was proven (in the worktree, isolated from the conductor)

The live value head regressed (`value_weight` reverted 5→0: "slower than bare BFS, the 25-game sim timed
out") NOT because the representation is weak — `cross_game_features_v3` is **LOO-AUROC 0.725, above chance**
(not the 0.503 the docs said; that was the v2 frame-only class). It regressed because computing it per
frontier node cost **13 ms** (a pure-python 4-connectivity flood fill, 4.5M `list.append`/node).

Fix (prototype, verified): replace the flood fill in `arc_agi3_world_model.objects()` +
`arc_value_learner._component_stats_from_grid` with `scipy.ndimage.label` (vectorised C), with a
pure-python FALLBACK when scipy is absent (the live Kaggle kernel may lack it). Output **identical**
(verified equal over 40 random grids; downstream features order-invariant; LOO-AUROC unchanged 0.7248;
tests pass). Per-node cost:

| feature set | before | after | AUROC |
|---|---|---|---|
| v2 | 2.3 ms | 0.30 ms | 0.515 |
| **v2 + frame-Δ (routing)** | 7.6 ms | **0.64 ms** | **0.742** |
| v3_full (current live) | 13 ms | 0.87 ms | 0.725 |

0.64 ms/node × 5000 nodes = 3.2 s (was 38 s = the timeout). Per-node value-routing is now AFFORDABLE.

## Productionization steps (conductor)

1. **Merge `outer-loop/repr-prototype`** (pure speedup, identical output, fallback-safe) into main.
2. Route by the cheap **v2 + frame-Δ** subset (highest AUROC, 0.64 ms) — drop the dead-weight `action`
   (0.488) / `predicate_distance` (0.536) classes; `object_relational` (5 ms, 0.657) is optional.
3. Raise `value_weight` off the 0.0 floor (`arc_competition_agent.py:60`) and **prove the live
   first-win-rate / solve-rate goes UP** on the 25-game sim (it should no longer time out). CI-gate it.
4. Optional further headroom: a frame-hash feature cache (BFS revisits frames) → amortizes most calls to ~0.

Honest caveat: may still PARTIALLY null — per-game LOO variance is 0.379→1.0 (the verifier misleads on some
games regardless of cost). Affordability removes the timeout; it does not fix per-game transfer.

## The OTHER highest-leverage move (from the SubQ/energy synthesis workflow, same session)

The synthesis ranked the energy roles; the cheapest highest-leverage one is **NOT this** — it is **wiring
`exp4020` `is_goal` as a graded GOAL-ENERGY target** (held-out precision 1.0, BUILT but UNWIRED), which
closes the dominant `GAP-ARCH-GOAL-NOT-VERIFIED` (a 99%-accurate dynamics model pointed at the wrong
win-condition plans confidently to the wrong state). Nearly free. See the synthesis (workflow weryz9i6n) +
`docs/research-notes/arc-representation-not-the-bottleneck-2026-06-23.md`. SubQ premise REFUTED: SubQ
(subq.ai) is a long-context LANGUAGE model (RULER/SWE-Bench), NOT on the ARC-AGI-3 board — the 10M-context
angle is a category error for ARC-AGI-3 (64×64 grids fit in normal context). Graph-RAG memory + structural
verifier features is the realistic literature-validated win (AriGraph 2407.04363, AGWM 2605.06841).
