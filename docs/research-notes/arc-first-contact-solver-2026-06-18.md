# A First-Contact ARC-AGI-3 Solver: Architecture, Results, and Failure Map (2026-06-18)

Outer-loop session record. This note consolidates a single night's investigation that began
from one operator question — "how does a per-game engine help us when we go to play live games
we have never seen before?" — and ended with a working, validated first-contact solver plus a
fully-measured map of exactly what limits it. It is a research record, not a public-facing
document.

## Headline

A first-contact ARC-AGI-3 solver that, with **zero prior knowledge** (no banked solve, nothing
game-specific pre-loaded), **solves 5 of 10 unseen movement-class games** — near-optimal action
counts — and whose 5 non-solves are each pinned to a specific, named component rather than a
mystery.

| Game | First-contact solve | Agent colour | Induced goal | Notes |
|---|---|---|---|---|
| sp80 | yes, 4 actions | 9 | 6 | short navigation |
| cn04 | yes, 14 actions | 0 | 8 | deep navigation |
| ar25 | yes, 15 actions | 4 | 11 | win involves reflection at the goal |
| tu93 | yes, 18 actions | 4 | 9 | NEW game (generalization) |
| ls20 | yes, 13 actions | 12 | 3/5 | NEW game (generalization) |
| ka59 | no | 14 | — | push + click; combinatorial |
| cd82 | no | 4 (growth) | — | growth/shrink dynamics |
| m0r0 | no | 10 | — | ambiguous multi-instance agent + exploration depth |
| sk48 | no | 14 | 1 | full pipeline ran; solve search exhausted |
| wa30 | no | 14 | — | exploration depth |

"First-contact" means the pipeline never reads a banked solution. Banked solves are used only to
confirm a game is solvable; the solver explores, identifies the agent, induces the goal, plans,
and executes entirely from its own observations. tu93 and ls20 were never iterated on — they are
the proof the solver generalizes rather than overfitting the development sample.

## The architecture (the pipeline)

The solver is a sequence of components, each of which was built and empirically validated this
session. The path to a solve is:

1. **Object-centric perception.** Segment frames into objects via per-colour connected components
   (`arc_world_model_dsl._color_components`). This is the load-bearing representation choice — see
   "Why representation, not engines" below.
2. **Reliable agent identification** (`identify_agent`, extended to `identify_agent_rich`). The
   agent is the connected object that most consistently **translates** under directional actions
   (per-object, not per-colour, so it works even when the agent shares the background colour, e.g.
   sp80). If no object translates, fall back to a **growth/shrink** agent (the colour whose cell
   count changes incrementally, e.g. cd82). This is a *general test-time method* — it re-derives
   each game's agent from that game's own observations, with no pretraining.
3. **Curious/directed exploration** (`curious_explore_for_win`). BFS over the reachable
   **agent-position** graph (dedup on the agent centroid, not the full grid). For navigation games
   the agent position is the state variable that matters, so coverage is ~O(#positions) instead of
   the exponential O(branching^depth) of random walks — it reaches deep wins random play cannot
   stumble (cn04 at 14, ar25 at 15). Adaptive granularity: escalate to a multi-object "pieces" key
   (+ piece-centroid clicks) only when the minimal agent key dries up.
4. **Test-time goal induction** (`induce_goal_color`). When a win is stumbled, the goal object is
   the non-background, non-agent colour the agent reached. This yields a goal *colour* (not a fixed
   coordinate), so it generalizes across layouts: "reach colour G, wherever it is."
5. **Goal-directed multi-step planning** (`best_first_search` over deep-copied real envs, the
   existing planner reused). Heuristic = nearest agent-colour cell to nearest goal-colour cell
   (cell-based, so it is robust to growth agents and to multiple same-colour agent instances).
   Because successors are real-env copies, a level-up in search *is* a real win.

## Why representation, not engines (the four foundational probes)

The session opened with four proposer-free probes that ruled out the obvious shortcuts and
located the real bridge to unseen games:

- **Whole-engine transfer is zero**, even between deliberately-similar games (ar25 ↔ ka59):
  an induced engine encodes a game's specific object/layout, so it predicts wrong on another game
  even when the *mechanic* is shared. (`arc3_logo_induction_transfer.json`)
- **A pixel-level translation primitive fits 0/9 games** — at the 64×64 pixel representation the
  dynamics are not clean parameterizable primitives. (`arc3_mechanic_primitive_transfer.json`)
- **At the object level, movement games become clean** — sp80's transitions are 70/70 single-object
  translations where pixels gave 0. The representation was the blocker, not the games.
  (`arc3_object_centric_repr.json`)
- These motivated extending the existing **M2-v2 ObjectDeltaModel** (`arc_world_model_dsl.py`) with
  per-object translate and composite move+recolor rules, lifting cn04 dynamics accuracy from 0.000
  to 0.477. (Two commits; tests in `tests/python/test_arc_world_model_dsl.py`.)

The throughline: the parts that **generalize** are *methods applied fresh at test time* (segment,
identify the agent, induce the goal, search), not *artifacts reused across games* (engines, value
heads — both measured at chance transfer). This directly answers the opening question.

## What "solving" actually required (the planning thread)

Measured, in order:

1. A good dynamics model alone does **not** solve — goal-induction is necessary
   (`arc3_m2_solve_objectdelta.json`, early arms).
2. Naive novelty/change-seeking guidance can be **worse than random** — on sp80 it avoided the
   low-visual-change "commit" action that wins.
3. Goal-direction with a perfect simulator solves the short game (sp80) optimally — goal-direction
   is sound; the gap is model accuracy / planning depth.
4. Multi-step `best_first_search` + an **agent-to-goal-distance heuristic** unlocks the deeper
   games where greedy and a whole-grid-mismatch heuristic both failed.

## The failure map (each non-solve pinned to one component)

This is the most reusable output: every non-solve is a specific, separable next target.

- **ka59 — combinatorial hardness.** Measured: its win genuinely involves 7 piece-objects, all
  active (relevance filtering keeps colours [0,1,4,14] — there is no static clutter to shed), and
  the win mixes keyboard with a setup click. Brute BFS over positions^7 × click-branching is
  intractable. *Needs:* a fundamentally smarter solver (learned heuristic / abstraction / subgoal
  decomposition) — the open research problem, not a state-key mechanic.
- **cd82 — growth-appropriate exploration.** Its agent grows/shrinks cell-by-cell (color 2:
  30→43; color 4: 64→63→62→61), not rigid translation. Growth *identification* now works, but the
  agent-centroid exploration key is wrong for a growing object (worse than the grid-hash fallback).
  *Needs:* an exploration state key suited to growth (e.g. frontier/extent rather than centroid).
- **m0r0 — multi-instance agent + exploration depth.** Two color-10 objects (25 cells each) plus two
  ~1300-cell regions; the centroid is ambiguous and the win is not stumbled. *Needs:* agent-instance
  disambiguation in the *exploration* key (not just the goal heuristic) and/or deeper search.
- **sk48 — solve-heuristic depth.** The full pipeline runs (win stumbled, agent=14, goal=1) but
  `best_first_search` exhausts 5000 nodes — the goal heuristic does not guide to the win. *Needs:*
  a stronger heuristic or larger budget, possibly a wrong induced goal colour.
- **wa30 — exploration depth.** Win not stumbled within budget. *Needs:* deeper/curiosity-weighted
  exploration.

## Honest scope and the long-tail lesson

The solver works on the *navigation/movement class*. The five failures span five *distinct*
mechanics, and the last several build steps (multi-object state → click-capable → relevance
filtering → richer dynamics → disambiguation) each added a correct, no-regression improvement that
did **not** unlock its target game, because each target had *further* idiosyncratic structure.

The lesson, stated plainly: **chasing the remaining games one mechanic at a time is the
GameAdapter anti-pattern, not convergence on a general solver.** A genuinely general ARC-AGI-3
solver for the hard tail (ka59-class combinatorial puzzles) needs learned/abstraction-based search,
which is the actual open research problem. The value delivered here is (a) a working solver for the
tractable class with measured ~50% first-contact coverage, and (b) a precise, component-level map of
what each remaining game requires — so future work can target a *class* of failure deliberately
rather than rediscover it per game.

## Artifacts and code

- Pipeline: `scripts/experiments/arc3_test_time_goal_induction.py` (explore → identify → induce →
  plan → solve), `scripts/experiments/arc3_m2_solve_objectdelta.py` (planning arms + agent ID).
- Inducer: `python/carnot/agentic/arc_world_model_dsl.py` (M2-v2 ObjectDeltaModel + per-object
  translate + composite rules); tests in `tests/python/test_arc_world_model_dsl.py`.
- Foundational probes: `results/arc3_logo_induction_transfer.json`,
  `results/arc3_mechanic_primitive_transfer.json`, `results/arc3_object_centric_repr.json`.
- Solve + generalization: `results/arc3_m2_solve_objectdelta.json`,
  `results/arc3_test_time_goal_induction.json`, `results/arc3_generalization_validation.json`.
- Reused planner: `python/carnot/agentic/arc_heuristic_search_over_verified_wm.py`
  (`best_first_search`); goal induction precedent: `python/carnot/agentic/arc_agi3_goal_induction.py`.
- Plan context: `docs/research-notes/arc-agi3-agent-research-plan.md` (M2 gate).

## Suggested next directions (deliberate, not per-game)

1. **Growth-class exploration** (a state key suited to growing agents) — unlocks cd82's class.
2. **Multi-instance disambiguation in exploration** — unlocks m0r0's class.
3. **Learned/abstraction search** for the combinatorial tail (ka59-class) — the open research
   problem, and the only thing that moves the hard games.
4. **Wire the first-contact pipeline into the live submission path** — it is the live-legal,
   no-prior-knowledge solver the M2/M3/M4 plan needs (operator-gated per Operator-Only External
   Publication).
