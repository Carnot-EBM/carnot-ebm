# ARC-AGI-3 engine A/B: v2 systematic-BFS vs v3 best-first vs v3+learned-verifier (2026-06-17)

Operator: "build the engine A/B." This extends the trained router's question from "which
HEURISTIC" to "which ENGINE", and tests the prize: can a different search engine (v3 best-first /
novelty, the engine built for DEEP games) crack a game that v2-BFS exhausts its budget on?

## Result (reproduction-gated, offline arcade)

| game | type | v2_bfs (acts/exp) | v3_novelty (acts/exp) | v3+verifier (acts/exp) |
|------|------|-------------------|-----------------------|------------------------|
| su15 | solved | **7 / 1746** | 29 / 8901 | (no checkpoint) |
| r11l | solved | **3** / 2236 | 58 / 3918 | 6 / **1064** |
| cd82 | solved | **5 / 525** | 15 / 2445 | (no checkpoint) |
| wa30 | deep | FAIL 0/20000 | FAIL 0/20000 | — |
| g50t | deep | FAIL 0/5575* | FAIL 0/5020 | — |
| sb26 | deep | FAIL 0/20000 | FAIL 0/20000 | — |

(*g50t's v2 exhausted at 5575 expansions, not the 20k budget — its reachable state space from L0
is small but the win is not in it: a mechanic/action-schema gap like tn36, not a search-depth gap.)

## Findings

1. **v2-BFS wins every solvable game.** It finds the OPTIMAL (shortest) path. v3-novelty solves the
   same games but with far LONGER paths (29/58/15 vs 7/3/5 actions) because best-first-by-novelty
   does not minimise path length. For ARC (action-efficiency is scored), v2-BFS is the correct
   first-contact engine — there is no game in the solvable set where v3 beats it.

2. **v3 cracks NO deep game.** wa30/g50t/sb26 resist BOTH engines at the 20k budget. A different
   GENERIC search engine is NOT the lever for the deep tail — confirming the registry's HARD-TAIL
   finding. The deep games need per-game RE (a GameAdapter) or a well-trained per-game
   verifier/representation, not a swap of blind-search strategy.

3. **A learned verifier guiding v3 cuts expansions (−52% on r11l: 1064 vs 2236)** but loses
   path-optimality (6 vs 3 actions). This is the most useful signal: the efficiency lever is a
   GOOD per-game verifier guiding best-first — which is exactly the OfflineSolver / verifier-routed
   path that already works for adaptered games (lp85, self-improving). It is NOT a generic novelty
   engine. So "v3+a trained verifier" ≈ the verifier-routed solver we already have; its quality is
   bounded by the verifier, i.e. the TRM / learned-representation track.

## Conclusion for the router

Engine routing COLLAPSES to "v2-BFS for first-contact": v2 dominates every solvable game on
path-optimality, and engine choice does not unlock the deep tail. So the router does not need a
separate learned engine-selector — the honest policy is `engine = v2-BFS` (then the heuristic
router picks v2's goal-distance heuristic), and the verifier-routed OfflineSolver is the efficiency
upgrade once a game is adaptered. The deep tail's lever is `GAP-ARC-TRM-TRAINED-ON-ARC` (a good
per-game verifier/representation) + per-game RE, NOT engine choice. `GAP-ARC-ROUTER-ENGINE-LABELS`
is answered: labels collected (v2 dominant), engine choice ruled out as the deep-tail lever.

## Shipped
- `graph_explore_solve_v3` gained a `stats` param (expansions/states) mirroring v2, so the A/B can
  measure v3's search cost.
- `results/arc_engine_ab.json` — the full A/B record.

Cross-refs: `docs/research-notes/arc-trained-router-2026-06-17.md` (the heuristic router this
extends); `ops/verifier_gaps.md` (the engine/TRM gaps this answers/refines).
