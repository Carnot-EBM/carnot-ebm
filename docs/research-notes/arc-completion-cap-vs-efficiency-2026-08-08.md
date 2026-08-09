# ARC-AGI-3 Score Decomposition: Completion Cap vs Efficiency — 2026-08-08

**Phase 0d of the live-agent improvement plan** (`docs/research-notes/arc-live-agent-improvement-
plan-2026-08-08.md`). Question: how much of our 0.08 hidden leaderboard score is capped by NOT
completing levels at all, versus limited by being INEFFICIENT (too many actions) on the levels we
do complete? The answer decides the standing budget split between depth work (reach more levels)
and efficiency work (reach the same levels faster).

## The real scoring formula

Kaggle does not hand back a per-game breakdown of our hidden score, so this note works from two
things we DO have: the authoritative scorer source
(`arc_agi.scorecard.EnvironmentScoreCalculator`, vendored in the local ARC-AGI reference checkout)
and our own registry of the 25 public games' real level counts (`ops/arc_solve_registry.yaml`).

Per level `i` (1-indexed) in a game:

```
if not completed:
    level_score = 0
else:
    level_score = min((baseline_actions / agent_actions) ** 2 * 100, 115)
```

A game's score is a **weighted average over ALL attempted levels, weighted by level index**:

```
weight_i = i                     # level 3 counts 3x as much as level 1
game_score = min(
    sum(weight_i * level_score_i) / sum(weight_i for every attempted level),
    sum(weight_i for COMPLETED levels only) / sum(weight_i for every attempted level) * 100,
)
```

Two consequences that drive everything below:

1. **An incomplete level contributes exactly 0**, no matter how close the agent got or how many
   actions it spent trying. There is no partial credit for effort on the level itself.
2. **The second `min(...)` term is a hard completion ceiling.** However efficient the agent is on
   the levels it DID complete, the game's score cannot exceed
   `(sum of completed levels' weights) / (sum of all attempted levels' weights) * 100`. A game
   where the agent only ever reaches level 1 out of 6 is capped at `1/(1+2+...+6)*100 = 4.76`,
   regardless of efficiency. Efficiency only moves the score WITHIN that ceiling.

## The 25-game level-count structure

`ops/arc_solve_registry.yaml` records the real, fully-cleared level count for every public game
(`levels_reproduced` with `full_game_clear: true`):

| Levels per game (N) | Number of games |
|---|---|
| 6 | 9 |
| 7 | 5 |
| 8 | 6 |
| 9 | 4 |
| 10 | 1 |

Total: 25 games, 183 levels. This is a public-game proxy for level-count structure, not the true
hidden roster (Kaggle does not publish the hidden games' level counts) — the caveat in the last
section names exactly what this proxy cannot tell us.

## What efficiency alone can buy, versus what completion depth can buy

Using the formula above, holding the completion pattern fixed and varying only the per-level
efficiency score (60 = well under baseline pace, 100 = exactly at the human baseline, 115 = the
formula's own cap):

| Scenario | Efficiency swing tested | Resulting leaderboard-scale score |
|---|---|---|
| All 25 games reach level 1 only | eff 60 | 0.0211 |
| All 25 games reach level 1 only | eff 100 | 0.0352 |
| All 25 games reach level 1 only | eff 115 | 0.0352 (capped) |

The full efficiency range (worst plausible to the formula's own maximum bonus) moves the score by
**less than 1.7x**. Now hold efficiency fixed at 100 and vary only how DEEP the agent gets:

| Scenario | Resulting leaderboard-scale score |
|---|---|
| 0% of games reach level 1 | 0.0000 |
| Documented live first-win rate (59%), level 1 only, 0% deepen | 0.0208 |
| 100% of games reach level 1 only | 0.0352 |
| 59% of games reach level 1, ALL of those also reach level 2 | 0.0624 |
| 100% of games reach level 1 AND level 2 | 0.1057 |

Going from "nobody reaches level 1" to "everybody reaches level 1" is a jump from 0 to 0.0352.
Going from "level 1 only" to "level 1 and level 2" roughly **triples** the score at the same
first-win rate. Depth moves the score by an order of magnitude more than efficiency does, on this
game-count structure, because of the weighting scheme: a level-2 completion is worth at least 2x a
level-1 completion before efficiency even enters, and every additional level compounds that.

## Honest caveat — the model does not fully explain the observed 0.08

Plugging in our own documented internal numbers is informative but does not cleanly reproduce the
actual hidden score:

- Documented live first-win rate (~0.59, offline/familiar-game measurement) with the documented
  "live multi-level rate ~0" (no level-2+ deepening) predicts **0.0208** — well under the observed
  **0.08**.
- The more hidden-game-like held-out proxy (`first_win_rate_integrated = 0.04`,
  `experiment_4605_live_integration_scored_agent.py`) with 0% deepening predicts **0.0014** — off
  by more than 50x.
- Even the extreme case of a 100% first-win rate with zero deepening tops out at **0.0352** —
  still under the observed 0.08. Matching 0.08 on this game-count structure requires most games to
  reach at least level 2, and some to reach level 3.

Three explanations are consistent with this gap, and this note cannot distinguish between them
with data on hand:

1. The true hidden game roster has a shallower level-count structure than our own 25-game public
   proxy (fewer levels per game raises the completion ceiling for the same first-win rate).
2. Our "live multi-level rate ~0" figure is stale for the CURRENT scored configuration and the
   agent is deepening more than we currently believe on hidden games specifically.
3. Some combination of both, plus sampling noise (a small hidden set means a few lucky deep
   completions move the average a lot).

This uncertainty does not change the decision below — every scenario tested, including the ones
that don't fully explain 0.08, shows the same ordering: completion depth dominates efficiency by
5-50x depending on the comparison. A calibration that fully explained 0.08 would only make that
gap larger, since it requires MORE depth than the conservative scenarios already modeled here, not
less.

## Decision

**The standing budget split favors depth (getting the agent past level 1, and past level 2) over
efficiency (reducing action count on levels already being completed), by roughly an order of
magnitude in expected score impact.** This matches Phase 1 of the improvement plan, which is
already scoped as depth work: re-deriving the 0/296 world-model closure under honest metrics
(1a) and promoting trajectory transfer (1b) both attack REACHING more levels, not doing the same
levels faster. Nothing in this note argues to cut efficiency work entirely — the review's Speed
findings (frontier BFS redundancy, uncached candidate-scorer keys) cost wall-clock that could
otherwise fund more search depth per action budget, which is itself a depth lever, not a pure
efficiency one. But a task whose sole benefit is "the same completed levels, fewer actions" should
lose priority to any task that plausibly unlocks one more level on one more game, until the
level-2/level-3 wall is meaningfully narrower than it is today.

## How this was computed

`ops/arc_solve_registry.yaml`'s `levels_reproduced` field per game, combined with a direct
reimplementation of `arc_agi.scorecard.EnvironmentScoreCalculator.add_level` /
`EnvironmentScoreCalculator.to_score` (read from the vendored ARC-AGI reference package, not
guessed) as a short local Python script. No new experiment artifact was produced — this is
arithmetic over already-committed data plus a well-known scoring rule, matching the "one-page
note" scope of Phase 0d.

## Cross-references

- `docs/research-notes/arc-live-agent-improvement-plan-2026-08-08.md` — the plan Phase 0d belongs to
- `docs/research-notes/live-agent-adversarial-review-2026-08-08.md` — the review whose Phase 1
  items (1a, 1b) this note's decision endorses continuing to prioritize
- `ops/arc_solve_registry.yaml` — the 25-game level-count structure used here
- `ops/known-issues.md` (2026-06-19 ref 53862349) — the 0.08 first scored hidden submission
- `ops/known-issues.md` (A4 retarget) — `first_win_rate_integrated = 0.04`, the held-out proxy
  closer to hidden-game conditions
- `ops/known-issues.md` (2026-06-24, L2-goal-predicate wall) — "live first-win rate is ~0.59 but
  the live multi-level rate is ~0"
