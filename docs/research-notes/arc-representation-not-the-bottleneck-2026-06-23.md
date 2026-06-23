# ARC live agent: the representation is NOT the bottleneck (re-diagnosis)

Date: 2026-06-23 · Outer-loop (interactive, worktree `outer-loop/repr-prototype`) · measurement, not opinion.

## Correction

The architecture-gaps analysis (and my relayed answer) concluded the binding constraint was the
**representation** — "features at chance, LOO-AUROC 0.503, the live agent reasons over a bag of frame-stats."
A fresh run of the dedicated harness **refutes that**:

`python/carnot/experiment_4545_cross_game_discrimination_v3.py` (794-row cached corpus, 18 games, seed 0):

| feature class | LOO-AUROC |
|---|---|
| v2 (frame-only order-1) | 0.515 ← the "0.503/chance" number |
| v2 + **frame-Δ** | **0.742** (carries the signal) |
| v2 + object-relational | 0.657 |
| v2 + action-conditioned | 0.488 (dead weight) |
| v2 + predicate-distance | 0.536 (marginal) |
| **v3_full (the LIVE features)** | **0.725** (in-sample 0.86, CI [0.649, 0.806]) |

The frame-Δ / relational / action / predicate features the analysis "recommended adding" are **already
implemented** in `cross_game_features_v3` (`arc_value_learner.py:394`) and already lift LOO-AUROC from 0.515
to 0.725. **"Add more features" is not the lever.** (The 0.503 the workflow cited was the v2 baseline / the
older bridge_v2 artifact; the workflow didn't realize v3 already exists.)

## What the real binding constraint is

Two issues the 0.725 mean hides:

1. **Per-game variance: LOO-AUROC ranges 0.379 → 1.0 across the 18 games.** The verifier transfers strongly
   to some games and *anti-correlates* on others — the cross-game generalization is uneven, not uniformly
   above chance.
2. **The OFFLINE → LIVE bridge fails anyway.** A 0.725-AUROC discriminator *regressed* the live search:
   `value_weight` reverted 5.0 → 0.0 (`arc_competition_agent.py:60`) because it was "slower than bare BFS
   and solved fewer games in bounded time (the 25-game sim timed out)."

So the gap is **not** "can the representation discriminate" (it can, 0.725) — it's **"why does a decent
offline discriminator make the live search worse."** Three candidate causes, to disambiguate:

- **COMPUTE-COST** (the live comment points here): computing v3 per frontier node slows the bounded-time
  search → fewer nodes explored → fewer solves, even with better rankings. Fix = cheap/cached/incremental
  features, or apply the value head only at decision points, not every node.
- **DISTRIBUTION-SHIFT**: the verifier is trained on *winning-path* states (`collect_trajectory_data` →
  steps-to-go) but the live frontier is *off-path* states it never saw → ~chance where it matters. Fix =
  train on search-distribution states (DAgger-style), or use it as bounded pruning not an A* value.
- **CALIBRATION**: a 0.725 *ranking* is not a usable A* cost; a wrong rank on the decisive node misroutes a
  depth-first search that was otherwise systematic. Fix = isotonic/Platt calibration to a cost.

## Next outer-loop step

Disambiguate compute-cost vs distribution-shift vs calibration (the live `value_weight` evidence points at
compute-cost first). This re-points the effort away from "more features" — the representation is done.

## Provenance

Run in worktree `outer-loop/repr-prototype` (isolated from the conductor's `git add -A`). Conductor keeps
its full roadmap; this prototype hands off when a cause is confirmed. Supersedes the "representation is the
binding constraint" framing in `project_arc_live_agent_learning_gaps` (memory corrected). Cross-refs:
`experiment_4545_cross_game_discrimination_v3.py`, `arc_value_learner.py:394` (cross_game_features_v3),
`arc_competition_agent.py:60` (value_weight reversion), `docs/research-notes/arc-live-agent-learning-gaps`.
