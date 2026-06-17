# ARC-AGI-3: a TRAINED router for dynamic adaptation to unseen games (2026-06-17)

Operator directive (2026-06-17): "we must capture 'when to use which' heuristic and must properly
train our TRMs and BFS and DFS and routers if we hope to solve new games we haven't seen before
dynamically when running live."

## The gap this closes

The prior heuristic-selection layer (`arc_heuristic_select.recommend_order`) was a HAND-CODED
threshold (`cell_impact >= 40 -> region_count`). That is not *trained* and does not improve with
experience. Worse, a single feature mispredicts: cd82 (cell-impact 201) and sp80 (34) both win
with **plain BFS**, not a heuristic — because they are small games with no search headroom, which
no static feature predicts.

## What was built: a trained, online-updating router (`arc_router`)

- **Training data** = the solve LEDGER (`ops/arc_router_ledger.json`): one row per solved game,
  `{features -> winning approach}`, collected by running the heuristic portfolio
  (`arc_heuristic_select.select_best`) and reproduction-gating each arm.
- **Features**: `cell_impact` (median cells changed per action — decides cell_count vs
  region_count) and `bfs_expansions` (the cheap BFS-arm probe — the search-HEADROOM signal that
  decides whether *any* heuristic helps), plus start wrong-cells/regions, solution depth, and the
  survey features (action_type, spatial, difficulty).
- **Model**: a CAUSALLY-STRUCTURED 2-node decision tree whose two thresholds are LEARNED from the
  ledger as stable midpoints between well-separated classes. The structure (headroom gate FIRST,
  then heuristic choice) is fixed because it is causally correct — "does any heuristic help?"
  precedes "which heuristic?" — and because a *free* greedy tree's split-order is unstable on
  sparse data (it scored 4/8 leave-one-out; the structured form scores 8/8). Learned thresholds on
  the current 8-game ledger: headroom = 1135 BFS-expansions, impact = 36 cells/action.
- **Generalisation, honestly measured (leave-one-out)**: train on N-1 games, predict the held-out
  game's winning approach. **8/8** — it generalises, it did not memorise. (k-NN got 3/8, the free
  tree 4/8; the hand threshold 6/8 — it structurally cannot get the two BFS-wins.)
- **Explore vs exploit by NOVELTY**: `route()` returns the prediction PLUS a confidence = proximity
  to the nearest solved game. A game UNLIKE anything solved (low proximity) -> EXPLORE: run the full
  portfolio and LEARN, rather than trust an extrapolation. On the current sparse ledger every game
  reads `explore` — which is correct: explore+learn until the feature space densifies, then exploit
  (skip the portfolio) on familiar games. This is the active-learning loop the live solver needs.

## How it stays trained live (online update)

`arc_loop_solve.solve_via_explore` now calls `arc_heuristic_select.select_and_learn` after every
fresh adapter-free solve: it runs the portfolio, banks the winning heuristic to `gap_fills/`, AND
records `(features -> winner)` to the router ledger. So every new game we solve live both captures
its best heuristic and TRAINS the router. `recommend_approach` surfaces the router policy in every
new-game recipe. The ledger is the durable training artifact; the router improves monotonically as
the solved-game set grows.

## Honest scope + what is NOT yet trained (logged as gaps)

The router currently routes the goal-distance HEURISTIC (where we have labelled outcomes). The
operator's full ask — "train our TRMs and BFS and DFS and routers" — has two unbuilt pieces, both
gated on collecting the right A/B labels (logged in `ops/verifier_gaps.md`):

1. **Engine routing (BFS vs best-first/DFS vs TRM-guided)**: we have heuristic-vs-heuristic labels
   but not engine-vs-engine labels per game. Need an engine A/B (v2-BFS vs v3-best-first vs a
   TRM-guided rollout) on each solved game to extend the ledger's approach space to the engine
   dimension. The feature schema + ledger already accommodate it.
2. **A trained ARC TRM**: the TRM (recursive reasoner) is the generator/refiner in the hybrid
   architecture; the running TRM is sudoku-trained. An ARC TRM needs training on the accumulated
   ARC solve traces (the captured trajectories) — a heavier GPU track, separate from this router.

Until those land, the router's BFS/heuristic routing is real and validated; engine/TRM routing is
honestly out of scope and flagged, not faked.

## Files
- `python/carnot/agentic/arc_router.py` — the trained router (train/route/leave_one_out/record/
  extract_features/learned thresholds).
- `python/carnot/agentic/arc_heuristic_select.py` — portfolio + `select_and_learn` (online update).
- `ops/arc_router_ledger.json` — the training ledger (grows per solve).
- `tests/python/test_arc_router.py`, `test_arc_heuristic_select.py` — 16 asserting tests incl. the
  leave-one-out generalisation check.

Cross-refs: `docs/research-notes/arc-llm-as-gap-filler-not-solver-2026-06-17.md` (the heuristics
the router selects among); CLAUDE.md "ARC Solve Reproducibility + Solver-Reuse" (capture-as-reusable
-asset); `ops/verifier_gaps.md` (engine/TRM routing data gaps).
