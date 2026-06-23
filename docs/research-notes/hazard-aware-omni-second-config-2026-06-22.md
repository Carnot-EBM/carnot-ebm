# Facing-aware omni — validated on a second charger configuration

Date: 2026-06-22 · Outer-loop (interactive, ultracode) · OFFLINE, zero quota · `verifier_is_oracle: false`

Directly addresses the scope limitation the adversarial review forced on
`hazard-aware-L3-calibrated-2026-06-22.md`: that the facing-aware `omni` rule was "validated on ONE level of
ONE game (tu93 L3's single static layout)." This note tests it on a **second charger configuration** and
shows the facing mechanism is not overfit to L3's specific layout.

## Result — omni is CLEAN on tu93 L2 AND L3 (two different configs)

The generalized calibration harness (`scripts/experiments/experiment_hazard_l3_calibration.py`, now
`--game/--target-level`) runs the same reproducible test on each charger level: position-keyed real-env BFS →
win path + per-move died/safe labels → score the `omni` `is_lethal` predicate (FN / FP / win-path-pruned).

| level | chargers | facings observed | BFS labels (deaths) | omni FN / FP / win-pruned |
|---|---|---|---|---|
| **tu93 L2** | **1** (horizontal) | left `(0,-1)` | 41 (2) | **0 / 0 / 0** |
| **tu93 L3** | **3** | right `(0,+1)` + down `(1,0)` | 88 (5) | **0 / 0 / 0** |

So the *same* facing-aware rule — each charger kills only on its facing line, on the side it faces, at
distance 1..reach, collision-exempt — is **clean (FN=0, FP=0, win-path-unpruned)** on:

- a **different charger count** (1 vs 3),
- a **different maze layout** (L2 ≠ L3),
- and **three of the four facing directions** (left at L2; right + down at L3),

with the facing read from the centre-marker offset in each case. The rule was *calibrated* against L3's BFS
labels but is *not* overfit to L3 — it transfers cleanly to L2's different configuration. Artifacts:
`results/experiment_hazard_l3_calibration.json` (L3) and `..._tu93_L2.json`.

The escalation loop confirms the same end-to-end: `omni` deepens tu93 L3 (where it is needed), and forced on
L2 it also solves L2 (`level_up`, plan_len 10) — though in the live ladder L2 is taken by the cheaper
`toward` rung first.

## Scope (still honest)

This is a second CONFIGURATION (count / layout / facings) but still within tu93 — same game, same charger
sprite encoding. A second *game* (different art / encoding) would be stronger evidence still. A parallel scan
of the reproduced-game set probed for a nav-reachable charging-enemy level (the signature: a non-avatar block
that *translates/charges* at a death, with a rigid-nav avatar). The completed batches found **none** —
wa30, ls20, dc22, ar25, cn04 have no charging-enemy mechanic (no block moves at a death) — so charging
enemies appear **tu93-specific within the reachable reproduced set** (most other games are click/paint/match,
not nav, so the omni loop cannot reach any charger level they might have). The cross-game test is therefore
**not currently available**; the layout/count/direction generalization within tu93 (L2 vs L3) is the
achievable evidence, and the facing-from-centre-marker mechanism is general by construction.

## Artifacts

- `scripts/experiments/experiment_hazard_l3_calibration.py` — generalized (`--game --target-level`); reproducible FN/FP scoring
- `results/experiment_hazard_l3_calibration.json` (tu93 L3) + `results/experiment_hazard_l3_calibration_tu93_L2.json` (tu93 L2)
- `python/carnot/agentic/arc_nav_world_model.py` — the facing-aware `omni` rule + `_charger_facing`
