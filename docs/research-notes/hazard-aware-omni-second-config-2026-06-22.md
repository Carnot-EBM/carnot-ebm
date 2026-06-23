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

## Third config: an INDEPENDENT constructed game — new encoding + the UP facing

The tu93 L2/L3 evidence is two configs but ONE game, ONE sprite encoding, and only three of the four facings
(left at L2; right + down at L3 — **never UP**). To get the stronger cross-encoding evidence, I built an
INDEPENDENT charger game, `GroundTruthChargerNav` (`scripts/experiments/experiment_constructed_charger.py`) —
its own state + physics, **not** the model's `is_lethal` — with:

- a **deliberately non-tu93 palette**: avatar 6/7, charger 11/13, wall 1, goal 12 (none of tu93's 9/4/8/15/5/14),
- a **different step** (5, vs tu93's 6),
- an **UP-facing charger** (the facing tu93 never exposed), resting below a corridor and charging up into it,
- a **walled maze** (corridors + a detour bay, structurally like tu93) so the avatar is confined and its 3x3
  footprint never overlaps the charger's — the sprite-occlusion edge case an *open field* exposes (an avatar
  drawn over a charger erases it and the facing is unreadable; tu93's own walls prevent this for free).

`HazardAwareNavWorldModel` is fit from the constructed env's transitions and tested the same way as the real
games: position-keyed BFS → win path + died/safe labels → score the `omni` predicate.

| test | result |
|---|---|
| charger detected from new encoding | `hazard_colors={11,13}`, `charge_range=8` learned |
| **UP facing recovered** from the centre-marker on the new sprite | **`(-1,0)` ✓** (the untested 4th direction) |
| omni FN / FP / win-path-pruned vs BFS ground truth | **0 / 0 / 0** |
| charger-blind frozen nav | plans the straight path `[4,4,4,4]` and **DIES at col 26** (avatar removed) |
| omni | **deepens** (L0→L1, reproduced) via the bay detour: row 16 → up to row 11 → over the charger → down at col 31 → goal |

So the facing-aware omni mechanism — read the facing from the centre-marker offset; each charger kills only on
its facing line, on the side it faces, within reach, collision-exempt — **transfers to a brand-new encoding
and recovers the previously-untested UP facing**, predicting deaths with FN=0/FP=0 against an independent
ground truth, and the planner converts that into a solve a charger-blind nav cannot reach.

**Honest scope of this control.** `GroundTruthChargerNav`'s kill rule is the *same physics family* as omni's
predicate (facing-line / side / reach / collision-exempt) with *different constants* (alignment tolerance 1
vs the model's 2; the avatar travel grid) — it is a **controlled** test of generalization across **encoding +
the UP facing + maze layout**, not an adversarial test of whether an arbitrarily-different charger physics
could fool the model. It complements (does not replace) the tu93 real-game evidence: tu93 is the genuine
unseen-game physics; the constructed game isolates the encoding+UP-facing axis the real games could not.

## Scope (still honest)

The tu93 L2/L3 pair is a second CONFIGURATION (count / layout / facings) but still within tu93 — same game,
same charger sprite encoding. A real second *game* with a charging-enemy mechanic would be stronger still: a
parallel scan of the reproduced-game set probed for a nav-reachable charging-enemy level (the signature: a
non-avatar block that *translates/charges* at a death, with a rigid-nav avatar). The completed batches found
**none** — wa30, ls20, dc22, ar25, cn04 have no charging-enemy mechanic (no block moves at a death) — so
charging enemies appear **tu93-specific within the reachable reproduced set** (most other games are
click/paint/match, not nav, so the omni loop cannot reach any charger level they might have). The real
cross-game test is therefore **not currently available**; the **constructed game above** is the achievable
encoding+UP-facing generalization evidence, and the layout/count/direction generalization within tu93 (L2 vs
L3) is the achievable real-game evidence. The facing-from-centre-marker mechanism is general by construction.

## Artifacts

- `scripts/experiments/experiment_hazard_l3_calibration.py` — generalized (`--game --target-level`); reproducible FN/FP scoring
- `results/experiment_hazard_l3_calibration.json` (tu93 L3) + `results/experiment_hazard_l3_calibration_tu93_L2.json` (tu93 L2)
- `scripts/experiments/experiment_constructed_charger.py` — the INDEPENDENT constructed game (new encoding + UP facing) + `results/experiment_constructed_charger.json`
- `python/carnot/agentic/arc_nav_world_model.py` — the facing-aware `omni` rule + `_charger_facing`
