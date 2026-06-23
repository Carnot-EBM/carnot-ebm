# tu93 L3 SOLVED — the hazard interception zone, calibrated against the BFS path

Date: 2026-06-22 · Outer-loop (interactive, ultracode) · OFFLINE, zero quota · `verifier_is_oracle: false`

Closes the open problem from `hazard-aware-L3-next-rung-2026-06-22.md`, which (after an adversarial review
retracted a "needs dynamic modelling" over-claim) established that tu93 L3 is **statically solvable** and the
next step was a **correctly-calibrated** static interception lethal-zone. This note does that calibration —
against the real-env BFS ground truth — and the escalation loop now auto-deepens **tu93 L1 → L2 → L3**.

## How the lethal-zone was calibrated (reproducible)

`scripts/experiments/experiment_hazard_l3_calibration.py` (committed; backs every number here):

1. Reach L3 (nav → hazard[toward]); run a **position-keyed real-env BFS** over L3 (sound because the
   chargers are static-until-triggered, i.e. position-deterministic). It finds a **19-action winning path**
   and labels every explored `(state, action)` as died/safe — **88 moves, 5 real deaths**.
2. Identify, per death, the **killer charger** (the one that *moved/charged* in the death frame) and its
   geometry. The result is unambiguous: every death is the avatar's destination ending **exactly aligned**
   with the killer (row OR col offset 0) at distance **6** (one charge step) **on the side the charger
   faces** — deaths 1,2 on a down-facing charger's column; deaths 3,4,5 on right-facing chargers' rows.
3. The charger's **facing is readable from the grid**: its centre-marker (the least-common hazard colour,
   tu93's colour-15) is **offset within the block in the direction it faces** — verified: `(25,25)`→+col→
   faces right, `(25,31)`→+row→faces down, `(37,13)`→+col→faces right, all matching the killer directions.

## The calibrated rule (`lethal_mode='omni'`)

`python/carnot/agentic/arc_nav_world_model.py:HazardAwareNavWorldModel`. A move is lethal iff, for some
charger, the avatar's **destination** is on the charger's **facing line** (aligned on the perpendicular
axis), **on the side it faces** (signed facing from the centre-marker offset), at distance **1..reach** —
**collision-exempt** (landing exactly on a charger is *not* a kill; it is defeated/passed). The facing
direction (not just axis) is what stops the over-pruning: a charger does not kill what is behind it.

This was reached by removing three earlier wrong assumptions, each caught by the BFS labels: the `enter`
rung's "perpendicular step-on is lethal" (backwards — approach direction is irrelevant; *alignment* is what
matters), the facing-agnostic "any aligned charger kills" (over-prunes — only the *facing* charger kills),
and "collision is lethal" (the 3 final win-path FPs were all collisions, which are safe).

Calibration result (reproducible, `experiment_hazard_l3_calibration.json`): **FN=0, FP=0,
win_path_pruned=0** over the 88 labelled moves. Clean.

## Result — the loop auto-deepens tu93 to L3

`results/experiment_reinduction_hazard_loop.json` (ladder nav → hazard[toward] → hazard[omni]):

```
seeds 7, 20260622, 3:  deepest L3, reproduced L3   |   L1(nav) -> L2(hazard_aware[toward]) -> L3(hazard_aware[omni])
```

L2's single horizontal charger cracks on `toward`; L3's three facing-charger configuration cracks on `omni`.
**Reproduced on a fresh env** at level 3 every seed.

## Adversarial review — SURVIVES (with the scope it forced)

A hostile reviewer independently replayed the banked 47-action sequence on **12 pristine envs — all reached
level 3** (parity-robust; no gotcha-#7 toggle). No hardcoding (facing is data-read; no layout literals in
`is_lethal`), no leak (`_levels_completed` is the env's own counter). Two honesty fixes it forced, now done:
- **Scope: robust-but-single-layout.** tu93 L3's charger layout is **byte-identical across seeds** (the seed
  only perturbs re-induction sampling), so 3 seeds = **one layout × 3** — deterministic-robust reproduction,
  not cross-configuration generalization. This is validated on **one level of one game**, NOT a general
  hazard solver.
- **The FN=0 claim is now backed** by the committed `experiment_hazard_l3_calibration.py` (was an inline
  throwaway before).

## Scope / honesty

Not a first solve (tu93 was already reproduced to L4+ via Manhattan routing). The contribution: the
induced-world-model imagination path now deepens through a hazard level, with the interception lethal-zone
**calibrated against real-env ground truth** rather than assumed — and the per-charger **facing read from the
grid** is a general mechanic, even though it is validated here on tu93 L3's single static layout.

## Forward

- Test the facing-aware `omni` rung on a SECOND charger game/level to convert "single-layout" into general
  evidence.
- Push to L4+ (a new mechanic surfaces a new rung), and/or drop the loop into the standing solver.

## Artifacts

- `python/carnot/agentic/arc_nav_world_model.py` — `lethal_mode='omni'` (facing-directional, calibrated) + `_charger_facing`
- `scripts/experiments/experiment_hazard_l3_calibration.py` + `results/experiment_hazard_l3_calibration.json` — the reproducible calibration (FN=0, FP=0)
- `scripts/experiments/experiment_reinduction_hazard_loop.py` + `results/experiment_reinduction_hazard_loop.json` — the loop deepening tu93 to L3
- `tests/python/test_arc_nav_world_model.py` — 8 tests incl. `test_omni_mode_is_facing_directional`
