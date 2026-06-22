# Hazard-aware world model — deepens tu93 L2 under induced imagination planning

Date: 2026-06-22 · Outer-loop (interactive, ultracode) · OFFLINE, zero quota · `verifier_is_oracle: false`

Follow-on to `mechanic-conditioned-reinduction-trigger-2026-06-22.md`. That note found tu93 L2 stalls
because a Level-2 charging-enemy HAZARD removes the avatar — a transition the pure-nav induced world model
cannot represent (it only translates/blocks), so it plans straight into the enemy and dies. The forward need
was a **hazard-aware model class**. This note builds it and shows it deepens tu93 L2.

All claims were put through an adversarial review (reproduction-reality, hazard-model causality, scoping)
before write-up.

## Scope (stated up front, honestly — sharpened after adversarial review)

- This is **not** a first solve of tu93 L2. tu93 is already reproduced to L5 in the registry via a
  hand-built Manhattan-distance routed best-first solver. **The contribution is that an INDUCED world model
  + imagination planning** (the Executable-World-Models lever, the thing that stalled at the hazard) **now
  handles a hazard level** — the world-model path reaches where it previously died.
- **Demonstrated on exactly ONE hazard instance** — tu93 L2's horizontal line-charger — across 5 seeds of
  that one env, plus one synthetic unit test (a colour-6 charger, which proves no tu93-hardcoding and that a
  different colour/axis is learnable). Generality across hazard TYPES is **not** proven here. The hazard
  model is a **LINE-CHARGER primitive**: an object that sits still until the avatar approaches along a shared
  line (row/column) within a charge range, then charges to intercept and removes the avatar. The class learns
  the axis from data (so a vertical charger is covered by construction), but it does **not** model pursuers,
  proximity bombs, multi-hazard interactions, or hazards whose lethality depends on non-rendered state. tu93
  L2 is grid-deterministic (0.0 nondeterminism over 12012 transitions), so its hazard is grid-expressible.
- **Honest caveats** (from the review; neither refutes the result): (1) the death transitions the hazard
  learner trains on are GUARANTEED by running the nav model's own suicidal plan (`nav_death_transitions`) —
  this is fair (that death is exactly the re-induction trigger's signal) but the pipeline is not tested for
  whether unguided exploration alone surfaces enough deaths to fit the hazard. (2) `charge_range` is the max
  post-move death distance; on tu93 all deaths share distance 6 so it is well-anchored, but noisier death
  data could mis-estimate the lethal radius.

## What the hazard is (measured from the live env)

tu93 L2 introduces colours 8 + 15 (absent in L1): a **charging enemy**, a 3×3 colour-8 block with a
colour-15 centre — structurally a mirror of the avatar (colour-9 block + colour-4 centre). Measured
behaviour (60-episode probe): the enemy is **stationary** until the avatar moves **along its row toward it**
and ends **within ~6 cells**; then it charges to intercept and the avatar is removed (game over). All 4
observed deaths share the geometry `same_row, moving_toward, post-move |Δcol| = 6`; being on its row far away
(|Δcol| = 12, 18, 24) is safe. A **safe path exists**: go up off the enemy's row, then right to the goal.

## The hazard-aware model class

`python/carnot/agentic/arc_nav_world_model.py:HazardAwareNavWorldModel` (extends `InducedNavWorldModel`).
`fit(transitions, goal_color=...)` learns, FROM TRANSITIONS (no hardcoding):

- the nav params (inherited), with the **goal colour inherited from L1** (level-invariant — the L2 data has
  no level-up to anchor goal detection on its own);
- the **hazard object** = the non-structural colour blob that **MOVES (charges) at the instant of death**
  (this disambiguates the charging enemy from the static door, which a naive "nearest blob" picks wrongly);
- the **charge axis** (row/col) and **charge range**, from the avatar→hazard geometry across death transitions.

`engine` predicts **avatar-REMOVAL** for a lethal move (a move that leaves the avatar on the hazard's line,
moving toward it, within charge range) — yielding an avatar-less, dead-end grid that `plan_in_model` will
never route through to the goal. So the planner finds the safe detour automatically.

On tu93 L2 the learned fit is stable across seeds: `hazard_colors={8,15}, axis=row, charge_range=6, goal=14`.

## Result — deepens tu93 L2, reproduced, on every seed

`scripts/experiments/experiment_hazard_aware.py` (head-to-head, both arms re-induced from the SAME L2
transitions; the only difference is the model class):

```
seed 7,20260622,3,42,100   NAV: deepened=False (game_over, dies at the enemy)
                           HAZARD_AWARE: deepened=True, reproduced_level=2, plan_len=10
VERDICT: hazard-aware deepens tu93 L2 + reproduces on 5/5 seeds where the pure-nav model dies
```

The hazard-aware plan weaves off the enemy's row (e.g. `[1,4,4,2,4,4,1,4,4,1]` — up/down dodges) and reaches
the L2 goal. **Reproduction-gated on a fresh env** (`reproduced_level=2`). The review independently confirmed
the solve is **parity-ROBUST** (stronger than bare fresh-env reproduction, and material given tu93's
non-idempotent-reset gotcha #7): replaying the banked sequence on 8 independent fresh envs, on one env reset
8×, and on a fresh arcade per replay all return level 2 — **no [1,2,1,2] toggle** — so this path is not
parity-contingent.

To guarantee the learner always has its signal, the harness augments the (random) L2 collection with the
nav model's own lethal plan execution — the deterministic death at the enemy IS the re-induction trigger's
signal, so it is the natural, always-available hazard datum.

## Causality (not luck)

The NAV and HAZARD_AWARE arms differ ONLY by the model class (same transitions, goal, planner). Direct test:
**disabling `is_lethal`** on the hazard-aware model (so it stops avoiding the hazard) reverts it to the nav
plan (plan_len 10→8) and it **dies (game_over)** — exactly like the nav arm. So the hazard avoidance is
causally responsible for the deepening:

```
HAZARD_AWARE (is_lethal active):   deepened=True,  reproduced_level=2, plan_len=10, stop=level_up
HAZARD_AWARE (is_lethal DISABLED): deepened=False, reproduced_level=0, plan_len=8,  stop=game_over
```

## Forward

- This closes the tu93-L2 deepening wall for the world-model path and supplies a reusable hazard primitive.
  Wire it into the re-induction loop: when the trigger fires (deterministic game-over after a level-up), fit
  the hazard-aware model (its signal is the death that fired the trigger) before re-planning.
- Extend the hazard vocabulary beyond the line-charger (pursuer, proximity, multi-hazard) as new games
  surface them — each new hazard is a learnable primitive, captured once and reused.

## Artifacts

- `python/carnot/agentic/arc_nav_world_model.py:HazardAwareNavWorldModel` (+ `tests/python/test_arc_nav_world_model.py`, 6 pass)
- `scripts/experiments/experiment_hazard_aware.py` — head-to-head NAV vs HAZARD_AWARE (tu93, 5 seeds)
- `results/experiment_hazard_aware.json` (adversarial-verify clean)
