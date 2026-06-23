#!/usr/bin/env python3
"""Calibrate (and VALIDATE, reproducibly) the hazard 'omni' lethal-zone against tu93 L3's real-env BFS path.

This is the committed, reproducible backing for the calibration claim in
docs/research-notes/hazard-aware-L3-calibrated-2026-06-22.md. It:

  1. reaches tu93 L3 (nav -> hazard[toward]),
  2. runs a POSITION-KEYED real-env BFS over L3 -> finds a winning action path AND labels every explored
     (state, action) as died / safe (the ground truth; the chargers are static-until-triggered, so a
     position key is sound),
  3. fits HazardAwareNavWorldModel(lethal_mode='omni') and scores its is_lethal predicate against the labels:
     FN (missed real deaths -- MUST be 0, else the planner walks into a charge) and FP (safe moves wrongly
     pruned, of which the load-bearing subset is those ON the win path -- MUST be 0, else the planner cannot
     route the verified path).

A clean calibration is FN==0 AND win_path_pruned==0. verifier_is_oracle: false. OFFLINE, zero quota.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_agi3_world_model import grid_of
from carnot.agentic.arc_agi3_live_adapter import _game_action, _levels_completed, _game_over
from carnot.agentic.arc_graph_explore import _warm
from carnot.agentic.arc_executable_world_model import to_logical
from carnot.agentic.arc_nav_world_model import InducedNavWorldModel, HazardAwareNavWorldModel, _bbox, _color_cells

import importlib.util
_here = Path(__file__).resolve().parent
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, str(_here / f)); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m
_ri = _load("exp_reind", "experiment_reinduction.py")
_ha = _load("exp_haz", "experiment_hazard_aware.py")
_loop = _load("exp_loop", "experiment_reinduction_hazard_loop.py")

REPO = _here.parents[1]
OUT = REPO / "results" / "experiment_hazard_l3_calibration.json"


def _ok(f):
    try:
        return np.asarray(grid_of(f)).ndim == 2
    except Exception:
        return False


def _avatar_pos(g, av_colors):
    b = _bbox(_color_cells(g, av_colors))
    return None if b is None else (b[0], b[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-nodes", type=int, default=80)
    args = ap.parse_args()
    t0 = time.time()
    game, seed = "tu93", args.seed

    # --- reach L3: L1 (nav) -> L2 (hazard toward); bank the L1->L2 action prefix -----------------------
    m0, pre, cell, _ = _ri.reach_level_one(game, 120000, 80, seed=seed)
    goal = m0.goal_color
    tr2, _ = _ri.collect_at_level(game, pre, 150, seed=seed + 1)
    nav = InducedNavWorldModel.fit(tr2); nav.goal_color = goal
    nd = _ha.nav_death_transitions(nav, game, pre, cell, 120000, 80)
    haz2 = HazardAwareNavWorldModel.fit(list(tr2) + nd, goal_color=goal, lethal_mode="toward")
    l1l2 = _loop.execute_from(game, pre, cell, haz2, 120000, 80)["banked"]
    av_colors = nav.avatar_colors

    def run(actions):
        arc = kit.offline_arcade(); env = arc.make(game, scorecard_id=arc.open_scorecard()); f = _warm(env, False)
        for a in list(l1l2) + list(actions):
            f = env.step(_game_action(GameAction, int(a if not isinstance(a, dict) else a["action"])), data=None)
            if f is None or _game_over(f):
                return f, None
        return f, (to_logical(grid_of(f), cell) if _ok(f) else None)

    f0, g0 = run([])
    start_lvl = _levels_completed(f0)
    start = _avatar_pos(g0, av_colors)

    # --- position-keyed real-env BFS over L3: win path + (state, action, died) labels ------------------
    seen = {start}; q = deque([([], start)]); labels = []; winpath = None; nodes = 0
    while q and nodes < args.max_nodes and winpath is None:
        path, _pos = q.popleft(); nodes += 1
        for a in (1, 2, 3, 4):
            f, g = run(path + [a])
            died = f is None or _game_over(f)
            lvl = _levels_completed(f) if f is not None else -1
            labels.append((path + [a], a, bool(died)))
            if died:
                continue
            if lvl > start_lvl:
                winpath = path + [a]; break
            p = _avatar_pos(g, av_colors) if g is not None else None
            if p and p not in seen:
                seen.add(p); q.append((path + [a], p))

    # --- score the calibrated 'omni' is_lethal against the labels --------------------------------------
    tr3, _ = _ri.collect_at_level(game, l1l2, 200, seed=99)
    nav3 = InducedNavWorldModel.fit(tr3); nav3.goal_color = goal
    nd3 = _ha.nav_death_transitions(nav3, game, l1l2, cell, 120000, 80)
    omni = HazardAwareNavWorldModel.fit(list(tr3) + nd3, goal_color=goal, lethal_mode="omni")
    winset = {tuple(winpath[:i + 1]) for i in range(len(winpath))} if winpath else set()

    fn = fp = wpp = 0
    facings = Counter()
    for path, a, died in labels:
        f, _g = run(path[:-1])
        if f is None or not _ok(f):
            continue
        g = np.asarray(to_logical(grid_of(f), cell))
        pred = omni.is_lethal(g, a)
        for hy, hx, _s in omni._hazard_blobs(g):
            fc = omni._charger_facing(g, hy, hx)
            if fc is not None:
                facings[("row" if fc[0] == 0 else "col", fc)] += 1
        if died and not pred:
            fn += 1
        if (not died) and pred:
            fp += 1
            if tuple(path) in winset:
                wpp += 1

    n_deaths = sum(1 for _p, _a, d in labels if d)
    clean = (fn == 0 and wpp == 0 and winpath is not None)
    verdict = ("success: omni_lethal_zone_CALIBRATED_clean_FN0_winpath_unpruned_on_tu93_L3_single_layout"
               if clean else f"complete: calibration_not_clean_FN{fn}_winpathpruned{wpp}_inspect")

    art = {"experiment": "experiment_hazard_l3_calibration", "game": game, "honest_verdict": verdict,
           "verifier_is_oracle": False, "inference_substrate": "offline_arc_search_plus_induced_world_model",
           "random_seed": seed, "n_bfs_nodes": nodes, "n_labelled_moves": len(labels),
           "n_real_deaths": n_deaths, "win_path_len": len(winpath) if winpath else None,
           "omni_FN_missed_deaths": fn, "omni_FP_over_prune": fp, "omni_win_path_moves_pruned": wpp,
           "calibration_clean": bool(clean),
           "hazard_center_color": omni.hazard_center_color, "charge_range": omni.charge_range,
           "distinct_charger_facings_observed": {str(k): v for k, v in facings.items()},
           "methodology_note": ("Position-keyed real-env BFS over tu93 L3 gives the ground-truth win path + "
                                "per-move died/safe labels (the chargers are static-until-triggered, so a "
                                "position key is sound). The 'omni' is_lethal rule (per-charger facing read "
                                "from the centre-marker offset; directional; collision-exempt; dist 1..reach) "
                                "is scored against those labels. CLEAN = FN==0 (no real death missed) AND "
                                "win_path_pruned==0 (the verified path is not over-pruned). SCOPE: tu93 L3 is "
                                "a SINGLE static layout (seed-invariant); this validates the calibration on "
                                "that one level, not a general hazard solver."),
           "duration_s": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(art, indent=2))
    print(f"VERDICT: {verdict}")
    print(f"  BFS nodes={nodes} labels={len(labels)} deaths={n_deaths} win_path_len={art['win_path_len']}")
    print(f"  omni: FN={fn} FP={fp} win_path_pruned={wpp} | facings={dict(facings)} -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
