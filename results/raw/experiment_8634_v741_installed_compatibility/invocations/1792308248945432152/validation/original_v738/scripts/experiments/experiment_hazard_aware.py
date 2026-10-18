#!/usr/bin/env python3
"""Hazard-aware world model DEEPENS tu93 L2 where the pure-nav model dies.

The re-induction work (docs/research-notes/mechanic-conditioned-reinduction-trigger-2026-06-22.md) showed
tu93 L2 stalls because a Level-2 charging-enemy HAZARD removes the avatar -- a transition the pure-nav engine
cannot represent (it only translates/blocks), so it plans straight into the enemy. This experiment builds on
the hazard-aware model class (carnot.agentic.arc_nav_world_model.HazardAwareNavWorldModel): it LEARNS the
line-charger hazard from the death transitions in L2 data (the enemy is the object that CHARGES = moves at
the instant of death), predicts avatar-REMOVAL for a lethal move, and so its planner routes AROUND the enemy
to the goal.

Head-to-head on tu93, both re-induced from L2 transitions (goal colour inherited from L1, which is
level-invariant):
  * NAV          -- InducedNavWorldModel: plans into the enemy and dies (the prior wall).
  * HAZARD_AWARE -- HazardAwareNavWorldModel: learns the charger, plans the safe detour, deepens L2.

Reproduction-gated (the banked L1->L2 action sequence is replayed on a FRESH env) and run over several seeds
for robustness. OFFLINE, zero quota. verifier_is_oracle: false (the world model is an induced predictive
model, not the executable correctness oracle).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_agi3_world_model import grid_of
from carnot.agentic.arc_agi3_live_adapter import _game_action, _levels_completed, _game_over
from carnot.agentic.arc_graph_explore import _warm
from carnot.agentic.arc_executable_world_model import to_logical, plan_in_model
from carnot.agentic.arc_nav_world_model import InducedNavWorldModel, HazardAwareNavWorldModel

# reuse the L1-reach + level-targeted collection from the re-induction harness
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "exp_reind", str(Path(__file__).resolve().parent / "experiment_reinduction.py"))
_ri = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ri)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "experiment_hazard_aware.json"


def _ok(frame):
    try:
        return np.asarray(grid_of(frame)).ndim == 2
    except Exception:
        return False


def deepen_with(model, game, l1_prefix, cell, max_plan, max_depth):
    """Replay the L1 prefix to reach L2, plan L2 INSIDE `model`, execute, and report whether it deepened
    (and the banked L1->L2 actions for the reproduction gate)."""
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = _warm(env, False)
    for a in l1_prefix:
        f = env.step(_game_action(GameAction, int(a["action"])), data=a.get("data"))
        if f is None or not _ok(f):
            return {"reached_L1": False}
    lvl0 = _levels_completed(f)
    g = to_logical(grid_of(f), cell)
    plan = plan_in_model(model.engine, model.is_level_complete, g, max_nodes=max_plan, max_depth=max_depth)
    if not plan:
        return {"reached_L1": True, "planned": False, "deepened": False, "reason": "no_plan_in_model"}
    banked = list(l1_prefix)
    stop = "plan_exhausted"
    for s in plan:
        f = env.step(_game_action(GameAction, int(s["action"])), data=s.get("data"))
        if f is None or not _ok(f):
            stop = "env_none"
            break
        banked.append({"action": int(s["action"]), "data": s.get("data")})
        if _levels_completed(f) > lvl0:
            stop = "level_up"
            break
        if _game_over(f):
            stop = "game_over"
            break
    deepened = _levels_completed(f) > lvl0
    # reproduction gate on a FRESH env
    repro = 0
    if deepened:
        envr = arc.make(game, scorecard_id=arc.open_scorecard())
        fr = _warm(envr, False)
        base = _levels_completed(fr)
        for a in banked:
            fr = envr.step(_game_action(GameAction, int(a["action"])), data=a.get("data"))
            if fr is None or not _ok(fr):
                break
        repro = _levels_completed(fr) - base
    return {"reached_L1": True, "planned": True, "plan_len": len(plan), "deepened": bool(deepened),
            "reproduced_level": int(repro), "stop": stop, "n_actions": len(banked)}


def nav_death_transitions(model, game, l1_prefix, cell, max_plan, max_depth):
    """Execute the NAV model's L2 plan (which deterministically walks into the hazard and dies) and record
    the transitions. This GUARANTEES the death transition the hazard learner needs -- it is exactly the
    re-induction trigger's own signal (the deterministic game-over after the level-up). Returns Transitions."""
    from carnot.agentic.arc_executable_world_model import Transition
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = _warm(env, False)
    for a in l1_prefix:
        f = env.step(_game_action(GameAction, int(a["action"])), data=a.get("data"))
        if f is None or not _ok(f):
            return []
    lvl0 = _levels_completed(f)
    g = to_logical(grid_of(f), cell)
    plan = plan_in_model(model.engine, model.is_level_complete, g, max_nodes=max_plan, max_depth=max_depth)
    out = []
    if not plan:
        return out
    for s in plan:
        g0 = to_logical(grid_of(f), cell); l0 = _levels_completed(f)
        nf = env.step(_game_action(GameAction, int(s["action"])), data=s.get("data"))
        g1 = to_logical(grid_of(nf), cell) if (nf is not None and _ok(nf)) else g0
        l1 = _levels_completed(nf) if (nf is not None and _ok(nf)) else l0
        out.append(Transition(g0, int(s["action"]), s.get("data"), g1, l0, l1))
        if nf is None or not _ok(nf) or _game_over(nf) or l1 > l0:
            break
        f = nf
    return out


def run_seed(game, seed, args):
    model0, l1_prefix, cell, _t = _ri.reach_level_one(game, args.max_plan, args.max_depth, seed=seed)
    if l1_prefix is None:
        return {"seed": seed, "reached_L1": False}
    tr2, _ = _ri.collect_at_level(game, l1_prefix, args.n_reinduce, seed=seed)
    nav = InducedNavWorldModel.fit(tr2)
    nav.goal_color = model0.goal_color                      # goal is level-invariant
    # augment with the nav-plan death transition(s) so the hazard learner always has its signal
    nav_deaths = nav_death_transitions(nav, game, l1_prefix, cell, args.max_plan, args.max_depth)
    haz = HazardAwareNavWorldModel.fit(list(tr2) + nav_deaths, goal_color=model0.goal_color)
    nav_res = deepen_with(nav, game, l1_prefix, cell, args.max_plan, args.max_depth)
    haz_res = deepen_with(haz, game, l1_prefix, cell, args.max_plan, args.max_depth)
    return {"seed": seed, "reached_L1": True, "hazard_fit": haz.hazard_fit,
            "nav_arm": nav_res, "hazard_aware_arm": haz_res,
            "hazard_aware_deepens_where_nav_dies": bool(haz_res.get("reproduced_level", 0) >= 2
                                                        and not nav_res.get("deepened"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="tu93")
    ap.add_argument("--seeds", default="7,20260622,3")
    ap.add_argument("--n-reinduce", type=int, default=150)
    ap.add_argument("--max-plan", type=int, default=120000)
    ap.add_argument("--max-depth", type=int, default=120)
    args = ap.parse_args()
    t0 = time.time()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    rows = [run_seed(args.game, s, args) for s in seeds]
    wins = [r for r in rows if r.get("hazard_aware_deepens_where_nav_dies")]
    repro_seeds = [r["seed"] for r in rows if r.get("hazard_aware_arm", {}).get("reproduced_level", 0) >= 2]

    if len(wins) == len(seeds) and seeds:
        verdict = ("success: HAZARD_AWARE_model_class_DEEPENS_tu93_L2_and_reproduces_where_the_pure_nav_model_"
                   "dies_on_every_seed")
    elif wins:
        verdict = ("success: hazard_aware_model_class_deepens_tu93_L2_reproduced_on_"
                   f"{len(wins)}_of_{len(seeds)}_seeds_where_nav_dies")
    else:
        verdict = "complete: hazard_aware_model_did_not_cleanly_deepen_inspect_rows"

    art = {"experiment": "experiment_hazard_aware", "game": args.game, "honest_verdict": verdict,
           "verifier_is_oracle": False, "inference_substrate": "offline_arc_search_plus_induced_world_model",
           "random_seeds": seeds, "n_seeds_hazard_aware_deepens_where_nav_dies": len(wins),
           "reproduced_seeds": repro_seeds, "rows": rows,
           "methodology_note": ("Both arms are re-induced from the SAME L2 transitions (goal colour inherited "
                                "from L1, which is level-invariant). NAV plans into the charging enemy and dies; "
                                "HAZARD_AWARE learns the line-charger (the object that MOVES at the instant of "
                                "death) and predicts avatar-removal for a lethal move, so plan_in_model routes "
                                "the safe detour. Reproduction-gated on a fresh env (reproduced_level>=2 = L2 "
                                "solved + reproduced)."),
           "duration_s": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(art, indent=2))
    print(f"\nVERDICT: {verdict}")
    for r in rows:
        h = r.get("hazard_aware_arm", {}); n = r.get("nav_arm", {})
        print(f"  seed {r['seed']}: NAV deepened={n.get('deepened')}({n.get('stop')}) | "
              f"HAZARD_AWARE deepened={h.get('deepened')} reproduced_L={h.get('reproduced_level')} "
              f"plan_len={h.get('plan_len')} | fit={r.get('hazard_fit',{}).get('hazard_colors')}/"
              f"{r.get('hazard_fit',{}).get('hazard_axis')}/range{r.get('hazard_fit',{}).get('charge_range')}")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
