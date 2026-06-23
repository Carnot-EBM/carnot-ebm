# Hazard-aware escalation rung for tu93 L3 — robustness fixes + an honest static wall

Date: 2026-06-22 · Outer-loop (interactive, ultracode) · OFFLINE, zero quota · `verifier_is_oracle: false`

Follow-on to `reinduction-loop-hazard-escalation-2026-06-22.md`, which auto-deepened tu93 L1→L2 (nav →
hazard[toward]) and stalled at L3. This note adds the next rung and characterises L3. **An adversarial review
caught and corrected an over-claim in the first draft — that correction is the headline finding.**

## What L3 is

Reaching L3 (solving L1+L2) and probing the live env: tu93 L3 has **THREE charging enemies** (colour 8 +
colour 15 centres, 3 blocks) — vs L2's single enemy — and they are **VERTICAL chargers** (they charge along
columns; L2's was horizontal). The goal sits on the avatar's row with the enemies clustered above.

## Robustness fixes (real, shipped, L2 intact)

1. **Door-colour exclusion.** `InducedNavWorldModel` now learns `door_color` (the dominant passable colour
   the avatar moves THROUGH on a successful move) and `HazardAwareNavWorldModel` excludes it from hazard
   candidates. Without this, L3's doors were flagged as hazards → every move looked lethal → `no_plan`. Pure
   data-derived, no hardcoding; unit-tested.
2. **Conservative charge-range floor.** `charge_range = max(observed death distance, move step)`. The L3 fit
   under-estimated the range (5 vs the true 6 = one charge step); flooring at the move step fixes it
   (under-estimating is fatal — the planner walks a "safe" move into a charge).
3. **Charger line-of-sight.** A charge is only lethal if its straight path to the avatar is wall-free
   (`_charge_unobstructed`) — a wall shields the avatar. Correct physics; part of a more precise interception
   model.

All three keep tu93 L2 solving (the loop still deepens L1→L2 and reproduces). Unit tests: 8 pass.

## The escalation rung: `lethal_mode` toward → enter

`HazardAwareNavWorldModel` gains `lethal_mode ∈ {toward, enter}`. `toward` flags a move lethal only on an
along-axis approach (tu93 L2's horizontal charger tolerates perpendicular step-ons). `enter` ALSO flags a
perpendicular step ONTO the charge line (tu93 L3's vertical chargers kill those). The loop escalates the
ladder **nav → hazard[toward] → hazard[enter]**, taking the first rung that deepens.

## The honest L3 finding (over-claim RETRACTED)

The loop deepens tu93 L1→L2 (toward rung, reproduced, every seed) and at L3 exhausts both current static rungs:
`toward` plans-but-dies (UNDER-prunes — misses the perpendicular step-on kills), `enter` no-plans
(OVER-prunes — forbids safe moves).

> **Correction (adversarial review, 2026-06-22).** The first draft concluded "L3 needs DYNAMIC charge-state
> modelling." A hostile reviewer probed the live env and **refuted** that: the L3 chargers are
> **static-until-triggered** (they don't move while the avatar is far/unaligned; the outcome is a
> deterministic function of avatar POSITION). A **position-keyed real-env BFS finds a verified 19-action
> winning path** (won, level 3, on a fresh env, robust across seeds). So **L3 is STATICALLY solvable** — the
> two current rungs are merely **mis-parameterised** lethal-zones (`enter`'s `align_tol × charge_range` band
> forbids ~6 genuinely-safe moves), not evidence of a dynamic mechanic. Tightening `align_tol`, the range
> floor, and adding line-of-sight did NOT alone find the path, so the exact calibration remains open — but
> the next step is a **correctly-calibrated static interception lethal-zone**, NOT dynamic modelling.

So L3 is **not** the world-model path's wall in principle (a static path exists); it is a **calibration**
problem in the lethal-zone geometry. The honest verdict string and methodology note say exactly this.

## Scope / honesty

Not a first solve (tu93 was already reproduced to L4+ via the Manhattan-routed solver). The contribution is
the induced-world-model imagination path + the robustness fixes + the escalation-rung mechanism, plus a
precisely-characterised, reviewer-corrected L3 status. The over-claim ("dynamic") is retracted; the open
problem is static calibration (or falling back to the position-keyed real-env search the existing tu93 solver
uses).

## Forward

- **Calibrate the static interception zone** against the known-safe path (the reviewer's 19-action BFS
  solution): mark a move lethal only at the EXACT interception cells, not the whole band. This should let the
  `enter` rung crack L3.
- Alternatively, when imagination-planning over-prunes, **fall back to position-keyed real-env search** for
  that level (the chargers are position-deterministic, so it is sound) — a pragmatic rung.

## Artifacts

- `python/carnot/agentic/arc_nav_world_model.py` — door_color, charge_range floor, line-of-sight, `lethal_mode`
- `scripts/experiments/experiment_reinduction_hazard_loop.py` — ladder nav → hazard[toward] → hazard[enter]
- `results/experiment_reinduction_hazard_loop.json` — L2 reproduced + L3 rung-exhaustion (adversarial-verify clean)
- `tests/python/test_arc_nav_world_model.py` — 8 tests (incl. door-exclusion + enter-mode)
