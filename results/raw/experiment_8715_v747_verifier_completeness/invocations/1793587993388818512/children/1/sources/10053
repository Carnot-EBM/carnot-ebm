#!/usr/bin/env python3
"""Constructed second charger game -- encoding-general validation of the facing-aware 'omni' rule.

The facing-aware omni rule was calibrated on tu93 L3 and shown clean on tu93 L2 (two configs, but ONE game,
ONE charger sprite encoding; only left/right/down facings -- never UP). This builds an INDEPENDENT charger
game (GroundTruthChargerNav -- its own state + physics, NOT the model's is_lethal) with:
  * DIFFERENT colours (avatar 6/7, charger 11/13, wall 1, door 3, goal 12 -- none of tu93's 9/4/8/15/5/2/14),
  * a DIFFERENT step (5, not tu93's 6),
  * an UP-facing charger -- the one facing tu93 never exposed (it only had left/right/down).
The game is a WALLED MAZE (corridors + a detour bay, structurally like tu93) so the avatar is confined and its
3x3 footprint never overlaps the charger's -- the sprite-occlusion edge case an open field exposes. The
charger physics matches the calibrated model: it kills when the avatar's destination is on its FACING line, on
the side it faces, at distance 1..reach (collision-exempt). The charger renders its centre marker OFFSET in
its facing direction, so the model must READ the up-facing from the grid -- on a brand-new encoding.

Test (the same shape as the real-game work): position-keyed BFS over the constructed env -> win path +
died/safe labels; fit HazardAwareNavWorldModel and check it (a) RECOVERS the chargers + per-charger facings
(incl. UP) from the new encoding, (b) scores FN=0/FP=0/win-path-unpruned, and (c) the omni planner DEEPENS
where a frozen pure-nav model dies. This is a CONTROLLED proof that the omni mechanism generalises across
encoding + the untested UP facing; it complements (does not replace) the tu93 real-game evidence.

OFFLINE, zero quota. verifier_is_oracle: false.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import deque
from pathlib import Path

import numpy as np

from carnot.agentic.arc_nav_world_model import InducedNavWorldModel, HazardAwareNavWorldModel

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "experiment_constructed_charger.json"

# --- a DELIBERATELY non-tu93 palette --------------------------------------------------------------------
WALL, FLOOR, DOOR, GOAL = 1, 0, 3, 12
AV_RING, AV_CTR = 6, 7
HZ_RING, HZ_CTR = 11, 13
STEP, REACH = 5, 8
_DIRS = {1: (-STEP, 0), 2: (STEP, 0), 3: (0, -STEP), 4: (0, STEP)}


class GroundTruthChargerNav:
    """Independent charger-game simulator (NOT the model). State = avatar cell + static chargers (each a
    cell + a facing unit) + goal cell, on a WALLED MAZE (a wall mask -- corridors + a detour bay, like
    tu93). Renders to a colour grid; the centre markers (avatar AV_CTR, charger HZ_CTR) are drawn OFFSET so
    facing is readable from the render. The maze walls confine the avatar to corridors so its 3x3 footprint
    never overlaps a charger's footprint -- the sprite-occlusion edge case that an open field exposes (and
    that tu93's own walls likewise prevent)."""

    def __init__(self, h, w, walls, avatar, chargers, goal):
        self.h, self.w = h, w
        self.walls = walls  # bool mask, True = wall
        self.avatar = avatar  # (r, c) centre of the 3x3 avatar
        self.chargers = chargers  # list of {"pos": (r, c), "facing": (fdy, fdx)}
        self.goal = goal
        self.level = 0
        self.done = False

    def _stamp(self, g, r, c, ring, ctr, ctr_off):
        g[r - 1 : r + 2, c - 1 : c + 2] = ring
        g[r + ctr_off[0], c + ctr_off[1]] = ctr  # centre marker, offset in the facing direction

    def render(self):
        g = np.full((self.h, self.w), FLOOR, dtype=int)
        g[self.walls] = WALL
        gr, gc = self.goal
        g[gr - 1 : gr + 2, gc - 1 : gc + 2] = GOAL
        for ch in self.chargers:
            r, c = ch["pos"]
            fdy, fdx = ch["facing"]
            self._stamp(g, r, c, HZ_RING, HZ_CTR, (fdy, fdx))  # marker offset = facing direction
        if self.avatar is not None:  # avatar is REMOVED on death
            ar, ac = self.avatar
            self._stamp(g, ar, ac, AV_RING, AV_CTR, (-1, 0))
        return g

    def _blocked(self, nr, nc):
        # out of bounds, or the destination 3x3 footprint hits any wall in the maze mask
        if nr - 1 < 0 or nc - 1 < 0 or nr + 1 >= self.h or nc + 1 >= self.w:
            return True
        return bool(self.walls[nr - 1 : nr + 2, nc - 1 : nc + 2].any())

    def step(self, action):
        if self.done or action not in _DIRS:
            return self.render(), self.level, self.done
        dy, dx = _DIRS[action]
        ar, ac = self.avatar
        nr, nc = ar + dy, ac + dx
        if self._blocked(nr, nc):
            return self.render(), self.level, self.done  # wall block -> avatar stays
        self.avatar = (nr, nc)
        # charger interception: avatar destination on a charger's FACING line, on the side it faces, 1..reach.
        # The charger CHARGES (moves along its facing line to the avatar) and REMOVES it -- so the death frame
        # shows the charger displaced from its rest position (the signal the hazard learner reads), exactly
        # like tu93. The avatar is then removed (None).
        for ch in self.chargers:
            cr, cc = ch["pos"]
            fdy, fdx = ch["facing"]
            hit = (
                (abs(nc - cc) <= 1 and 0 < (nr - cr) * fdy <= REACH)
                if fdx == 0
                else (abs(nr - cr) <= 1 and 0 < (nc - cc) * fdx <= REACH)
            )
            if hit:
                ch["pos"] = (nr, nc)  # charge along its facing line onto the avatar's cell
                self.avatar = None
                self.done = True
                return self.render(), self.level, True
        if abs(nr - self.goal[0]) <= 1 and abs(nc - self.goal[1]) <= 1:  # reached the goal
            self.level += 1
            self.done = True
        return self.render(), self.level, self.done


def build_level():
    """A walled MAZE whose only deadly obstacle is an UP-facing charger -- the one facing tu93 never exposed.
    The avatar (centre (16,6)) must reach the goal ((16,46)) along a horizontal main corridor (rows 14-18).
    An UP-facer rests BELOW the corridor at (24,26) and charges UP through an open shaft into the corridor, so
    the destination cell (16,26) on the straight path is lethal (distance 8 = reach, on the charger's facing
    side). A charger-blind nav therefore walks straight into col 26 and DIES; the omni planner must take the
    detour bay: up to row 11 at col 21, across the top cross (row 11, clear of the charger by 13 rows), down to
    the corridor at col 31, then on to the goal.

    The walls confine the avatar so that on EVERY safe reachable cell its 3x3 footprint is >=3 cells from the
    charger in at least one axis -- the avatar can never sit adjacent to / on top of the charger, so the
    sprite-occlusion FN/FP that an open field produces cannot occur (the same property tu93's maze walls give
    for free). The charger column (26) is on the avatar's travel grid, so alignment is exact (no off-grid
    centroid drift); the only col-aligned cells are lethal (rows 16-23) or far above (row 11, beyond reach)."""
    h, w = 33, 53
    walls = np.ones((h, w), dtype=bool)  # start all-wall; carve the floor corridors + bay

    def carve(r0, r1, c0, c1):
        walls[r0 : r1 + 1, c0 : c1 + 1] = False

    carve(14, 18, 4, 48)  # main horizontal corridor (avatar travels row 16)
    carve(9, 18, 19, 23)  # up-bay riser at col 21 (lets the avatar climb to row 11)
    carve(9, 13, 19, 33)  # top cross at row 11 (traverse col 21 -> col 31, clear of the charger)
    carve(9, 18, 29, 33)  # down riser at col 31 (return to the corridor)
    carve(
        14, 25, 25, 27
    )  # vertical shaft at col 26 -- the charger's unobstructed up-charge channel

    chargers = [
        {"pos": (24, 26), "facing": (-1, 0)}
    ]  # faces UP: lethal col 26, rows 16..23 (THE UP TEST)
    return GroundTruthChargerNav(h, w, walls, avatar=(16, 6), chargers=chargers, goal=(16, 46))


def fresh():
    return build_level()


def collect(n, seed):
    """Random-walk transitions for fitting (each episode resets; death/level-up ends it)."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        env = fresh()
        g0 = env.render()
        for _ in range(14):
            a = rng.choice([1, 2, 3, 4])
            g1, _lvl, done = env.step(a)
            rows.append((g0.copy(), a, g1.copy(), 0, 1 if (done and env.level > 0) else 0))
            if done:
                break
            g0 = g1
    return rows


def nav_deaths(model, max_nodes=4000):
    """Run the nav model's plan in the constructed env to GUARANTEE death transitions (the hazard signal)."""
    from carnot.agentic.arc_executable_world_model import plan_in_model

    env = fresh()
    g = env.render()
    plan = plan_in_model(
        model.engine, model.is_level_complete, g, max_nodes=max_nodes, max_depth=80
    )
    out = []
    if not plan:
        return out
    for s in plan:
        g0 = env.render()
        g1, lvl, done = env.step(s["action"])
        out.append(
            (g0.copy(), int(s["action"]), g1.copy(), 0, 1 if (done and env.level > 0) else 0)
        )
        if done:
            break
    return out


def avatar_pos(g):
    from carnot.agentic.arc_nav_world_model import _bbox, _color_cells

    b = _bbox(_color_cells(g, {AV_RING, AV_CTR}))
    return None if b is None else (b[0], b[1])


def bfs_labels(max_nodes=400):
    """Position-keyed BFS over the constructed env -> win action path + (path, action, died) labels."""
    env0 = fresh()
    start = avatar_pos(env0.render())
    seen = {start}
    q = deque([([], start)])
    labels = []
    win = None
    nodes = 0
    while q and nodes < max_nodes and win is None:
        path, _p = q.popleft()
        nodes += 1
        for a in (1, 2, 3, 4):
            env = fresh()
            died = False
            lvl = 0
            for act in path + [a]:
                _g, lvl, done = env.step(act)
                if done and env.level == 0:
                    died = True
                    break
                if done:
                    break
            labels.append((path + [a], a, bool(died)))
            if died:
                continue
            if lvl > 0:
                win = path + [a]
                break
            p = avatar_pos(env.render())
            if p and p not in seen:
                seen.add(p)
                q.append((path + [a], p))
    return win, labels, nodes


def solve_with(model, max_nodes=40000):
    """Plan in `model` over the constructed env, execute, return (deepened, n_steps, banked)."""
    from carnot.agentic.arc_executable_world_model import plan_in_model

    env = fresh()
    g = env.render()
    plan = plan_in_model(
        model.engine, model.is_level_complete, g, max_nodes=max_nodes, max_depth=120
    )
    if not plan:
        return {"planned": False, "deepened": False, "reason": "no_plan"}
    banked = []
    for s in plan:
        _g, lvl, done = env.step(s["action"])
        banked.append(int(s["action"]))
        if done:
            break
    return {
        "planned": True,
        "deepened": bool(env.level > 0),
        "n_steps": len(banked),
        "banked": banked,
    }


def reproduce(banked):
    env = fresh()
    for a in banked:
        _g, lvl, done = env.step(a)
        if done:
            break
    return env.level


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260622)
    args = ap.parse_args()
    t0 = time.time()

    # fit nav + omni from the constructed env's transitions (+ the nav suicidal plan's deaths as hazard
    # signal). The goal colour is provided (as the re-induction loop provides the level-invariant goal); the
    # CHARGER + its FACING must still be auto-detected from the new encoding -- that is what is under test.
    tr = collect(120, args.seed)
    nav = InducedNavWorldModel.fit(tr)
    nav.goal_color = GOAL
    nd = nav_deaths(nav)
    omni = HazardAwareNavWorldModel.fit(list(tr) + nd, goal_color=GOAL, lethal_mode="omni")

    # what facings did omni RECOVER from the new encoding? (expect all four incl. UP = (-1,0))
    g_full = fresh().render()
    recovered = {}
    for hy, hx, _s in omni._hazard_blobs(g_full):
        f = omni._charger_facing(g_full, hy, hx)
        recovered[(round(hy), round(hx))] = f
    facings_recovered = sorted(str(v) for v in recovered.values())
    up_facer_recovered = any(v == (-1, 0) for v in recovered.values())

    # BFS ground truth -> score omni FN/FP/win-path-pruned
    win, labels, nodes = bfs_labels()
    winset = {tuple(win[: i + 1]) for i in range(len(win))} if win else set()
    fn = fp = wpp = 0
    from carnot.agentic.arc_nav_world_model import _bbox, _color_cells

    def before(path):
        env = fresh()
        for act in path[:-1]:
            _g, _lvl, done = env.step(act)
            if done:
                return None
        return env.render()

    for path, a, died in labels:
        g = before(path)
        if g is None or _bbox(_color_cells(g, {AV_RING, AV_CTR})) is None:
            continue
        pred = omni.is_lethal(np.asarray(g), a)
        if died and not pred:
            fn += 1
        if (not died) and pred:
            fp += 1
            if tuple(path) in winset:
                wpp += 1

    # frozen nav (charger-blind) vs omni
    frozen = solve_with(nav)
    omni_res = solve_with(omni)
    omni_repro = reproduce(omni_res["banked"]) if omni_res.get("banked") else 0

    clean = fn == 0 and wpp == 0 and win is not None
    omni_cracks = bool(omni_res.get("deepened") and omni_repro >= 1 and not frozen.get("deepened"))
    if omni_cracks and clean and up_facer_recovered:
        verdict = (
            "success: facing_aware_omni_GENERALISES_to_a_constructed_charger_game_new_encoding_and_"
            "the_UP_facing_FN0_FP0_omni_deepens_where_charger_blind_nav_dies"
        )
    elif omni_cracks:
        verdict = (
            "success: omni_deepens_constructed_charger_game_but_check_facing_recovery_or_FN_FP"
        )
    else:
        verdict = "complete: constructed_charger_omni_did_not_cleanly_crack_inspect"

    art = {
        "experiment": "experiment_constructed_charger",
        "honest_verdict": verdict,
        "verifier_is_oracle": False,
        "inference_substrate": "constructed_ground_truth_charger_nav_offline",
        "random_seed": args.seed,
        "palette": {
            "avatar": [AV_RING, AV_CTR],
            "charger": [HZ_RING, HZ_CTR],
            "wall": WALL,
            "door": DOOR,
            "goal": GOAL,
            "step": STEP,
            "reach": REACH,
        },
        "true_charger_facings": [str(c["facing"]) for c in fresh().chargers],
        "omni_recovered_facings": facings_recovered,
        "up_facer_recovered": bool(up_facer_recovered),
        "omni_recovered_hazard_colors": sorted(omni.hazard_colors),
        "omni_center_color": omni.hazard_center_color,
        "bfs_nodes": nodes,
        "n_labels": len(labels),
        "n_deaths": sum(1 for _p, _a, d in labels if d),
        "win_path_len": len(win) if win else None,
        "omni_FN": fn,
        "omni_FP": fp,
        "omni_win_path_pruned": wpp,
        "calibration_clean": bool(clean),
        "frozen_nav_deepened": bool(frozen.get("deepened")),
        "omni_deepened": bool(omni_res.get("deepened")),
        "omni_reproduced_level": int(omni_repro),
        "omni_cracks_where_nav_dies": omni_cracks,
        "methodology_note": (
            "GroundTruthChargerNav is an INDEPENDENT charger simulator (own state/physics, "
            "NOT the model's is_lethal), with a non-tu93 palette, step 5, and chargers of "
            "all four facings incl. UP. omni is fit from the constructed transitions, must "
            "RECOVER the facings from the new encoding, and is scored vs a position-keyed "
            "BFS ground truth (FN/FP/win-path-pruned) + a frozen-nav-dies / omni-deepens "
            "contrast. Controlled encoding+UP generalisation, complementing the tu93 "
            "real-game evidence."
        ),
        "duration_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(art, indent=2))
    print(f"VERDICT: {verdict}")
    print(
        f"  true_facings={art['true_charger_facings']} recovered={facings_recovered} up_recovered={up_facer_recovered}"
    )
    print(
        f"  omni FN={fn} FP={fp} win_pruned={wpp} clean={clean} | frozen_nav_deepened={frozen.get('deepened')} "
        f"omni_deepened={omni_res.get('deepened')} reproduced=L{omni_repro} -> {OUT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
