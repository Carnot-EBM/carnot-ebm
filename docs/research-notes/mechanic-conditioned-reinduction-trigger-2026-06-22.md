# Mechanic-conditioned re-induction trigger for cross-level deepening

Date: 2026-06-22 · Outer-loop (interactive, ultracode) · OFFLINE, zero quota · `verifier_is_oracle: false`

Follow-on to `program-generalization-first-swing-2026-06-22.md`. That note showed the Executable-World-Models
deepening lever works at L1 (a faithful world model plans a level in imagination and reproduces it) but that
a model frozen at L1 does not deepen when the next level's mechanic shifts. The forward lever proposed there
was: **treat a deterministic, budget-unexhausted env game-over after a level-up as a re-induction trigger** —
re-fit the world model at the new level and re-plan. This note implements and tests that trigger.

All claims here were put through an adversarial review (3 independent skeptics, re-running the experiments
and reading the env source) before write-up; two first-draft over-claims were caught and corrected — those
corrections are folded in below and flagged.

## What was built

1. **An auto-fitting nav world model** — `python/carnot/agentic/arc_nav_world_model.py:InducedNavWorldModel`.
   `fit(transitions)` learns, FROM TRANSITIONS ALONE (no hardcoding), the per-action displacement, the avatar
   colours (by co-translation grouping — the colours that always shift together), the floor colour, the
   swept-gap wall colours (by a blocked-vs-moved discrimination score), and the goal colour. It exposes
   `engine` + `is_level_complete`, the `plan_in_model` interface. Re-induction = call `fit` again on
   transitions collected at the new level. Unit-tested (`tests/python/test_arc_nav_world_model.py`).
   - **Adversarial verdict: SURVIVES.** No hardcoding (proven by colour-permutation invariance — remapping
     tu93's colours makes `fit` recover the permuted values); seed-stable across 5 seeds; recovers tu93 L1 at
     **100% movement accuracy IN- AND OUT-OF-SAMPLE** (no leakage); degrades gracefully on non-nav games
     (sb26 → empty fit, ls20 → 0.84 < the 0.9 clean-fit bar). Caveat: wall-recovery needs enough blocked
     samples per action (it is unreliable on tiny corpora — fine on real games, where every action both moves
     and blocks).

2. **The re-induction trigger + a FROZEN-vs-REINDUCT head-to-head** —
   `scripts/experiments/experiment_reinduction.py`. Reaches L1 with an auto-induced model, then deepens two
   ways: FROZEN keeps the L1 model; REINDUCT re-collects transitions at the current level (`collect_at_level`,
   which replays the banked prefix to reach the level then explores) and re-fits. The trigger condition is a
   deterministic, budget-unexhausted game-over after a level-up.

## Result 1 — the operator provably deepens past a frozen model (controlled positive proof)

`scripts/experiments/experiment_reinduction_synthetic_control.py`. On the REAL tu93 L1 maze (rich layout →
robust fitting), shift **only the wall colour** 5→7 to make a clean grid-expressible mechanic shift. The
avatar/door/floor/goal colours are unchanged, so the frozen model LOCATES the avatar and knows the goal — it
fails purely because it learned wall=5 and treats L2's colour-7 walls as passable, planning a path straight
through walls that the real colour-7-walled env blocks.

```
FROZEN   L1=solved  L2=FAILED (plan_executed_no_advance)   frozen_locates_avatar=True
REINDUCT L1=solved  L2=SOLVED (advanced)                   m1.wall=[5] -> m2.wall=[7]
```

REINDUCT deepens where FROZEN cannot, and FROZEN fails for a LEGITIMATE reason (mis-modelled walls, avatar
findable) — not the trivial avatar-relabel a first draft used. **This is the positive proof that the
re-induction operator cracks a grid-expressible nav-mechanic shift.**

> **Correction (caught in review).** The first draft shifted EVERY param including the avatar colour, which
> made FROZEN fail trivially (it could not even find the L2 avatar — a relabelling tautology, not mechanic
> generalization). The headline is now a wall-colour-only shift on the real maze, where the failure is
> attributable to the mechanic. Scope: the claim is about grid-expressible NAV shifts; `GroundTruthNav` is an
> independently-coded simulator but shares the nav mechanic FORM, so this validates parameter-recovery +
> plan-divergence, not arbitrary-mechanic modelling.

## Result 2 — on the real reproduced game set, the trigger correctly DIAGNOSES (tu93)

`results/experiment_reinduction_tu93.json`. Both arms reach L1; REINDUCT re-fits an L2 model that is
**movement-accurate (1.0) on L2 navigation** — and yet deepening still stalls at a deterministic game-over
(step 4, budget 47/50 unexhausted) where **the avatar is REMOVED** (`stop: game_over_..._avatar_REMOVED`).

The precise, adversarially-corrected reading:

- tu93 **L2 navigation is grid-deterministic** (an independent determinism probe over 12012 L2 transitions
  found 0.0 nondeterminism — sb26-class, NOT wa30-class) and is re-inducible (the 1.0 movement accuracy).
- But tu93 **L2 adds a single charging-wall-sprite hazard** (per the env source) that arms, charges, and
  **removes the avatar**. The pure-nav engine only translates/blocks — it is **structurally unable to
  represent avatar removal**, so it plans a path into the hazard. Re-inducing the NAV model cannot fix this:
  the missing capability is a model CLASS (hazard-aware), not a parameter.
- The trigger does its job: it distinguishes "re-fit a movement-accurate model that still dies" (→ the level
  adds a non-nav mechanic; escalate to a hazard-aware model) from "re-fit and now deepen" (Result 1).

> **Corrections (caught in review).** (a) "movement_accuracy 1.0 = clean refit" was overclaimed — the metric
> only scores avatar-bbox-changed, so the fatal removal transitions score as "moves"; the model is in fact
> WRONG on the death, which the metric cannot see. The verdict now says "movement-accurate" and the harness
> separately detects avatar-REMOVAL. (b) The mechanic is a charging-wall sprite, NOT "enemy/box multi-phase"
> (enemy/box counts are 0 at L2). (c) "Hidden state" was unscoped — L2 navigation is grid-deterministic; only
> the hazard/death interaction is non-grid.

## Why no real-game REINDUCT>FROZEN exists yet (survey)

Among reproduced ARC games, a clean reach-goal nav game that imagination-planning fully drives is rare:
tu93 is the one, and its L2 adds the hazard above. A nav-fit survey (auto-inducer on the reproduced set):
dc22/ar25/wa30 fit L1 navigation (1.0) but are L1-only (dc22), hidden-state (wa30), or have a non-reach-goal
win (ar25 reflection) that the reach-goal predicate cannot express, so `plan_in_model` cannot even reach
their L1. m0r0's L1 is not rigid-avatar nav at all. So a grid-expressible reach-goal L2 shift — the precondition
for a real-game positive — is absent from the current set; hence the controlled synthetic proof (Result 1).

## Forward

- **Build the hazard-aware model class** (the tu93 L2 need): extend the induced model to represent
  object-removal / lethal-contact, so the planner avoids the hazard. This, plus the re-induction trigger,
  would deepen tu93 L2.
- **Extend the inducer's win vocabulary** beyond reach-goal (reflection/coalescence/toggle) so more reproduced
  L2s become plannable — then more real-game re-induction tests become possible.
- The re-induction trigger itself is ready to wire into the standing solver loop as the level-boundary action:
  on a deterministic budget-unexhausted game-over after a level-up, re-collect + re-fit before re-planning.

## Artifacts

- `python/carnot/agentic/arc_nav_world_model.py` — the auto-fitting inducer (+ `tests/python/test_arc_nav_world_model.py`)
- `scripts/experiments/experiment_reinduction.py` — the trigger + FROZEN-vs-REINDUCT harness (tu93 artifact)
- `scripts/experiments/experiment_reinduction_synthetic_control.py` — the controlled positive proof
- `results/experiment_reinduction_tu93.json`, `results/experiment_reinduction_synthetic_control.json` (both adversarial-verify clean)
