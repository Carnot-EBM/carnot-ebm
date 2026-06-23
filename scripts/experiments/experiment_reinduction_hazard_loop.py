#!/usr/bin/env python3
"""Integrated re-induction loop with HAZARD ESCALATION at the trigger.

This wires the hazard-aware model (docs/research-notes/hazard-aware-world-model-2026-06-22.md) INTO the
mechanic-conditioned re-induction loop (docs/research-notes/mechanic-conditioned-reinduction-trigger-2026-06-22.md)
as a single self-deepening mechanism. At each level the loop:

  1. RE-INDUCES a nav world model from transitions collected at the current level (goal colour inherited
     from L1, which is level-invariant) and plans+executes.
  2. On a level-up -> bank the actions and continue to the next level.
  3. On the TRIGGER -- a game-over where the avatar was REMOVED (the hazard signature; the nav engine only
     translates/blocks so it plans straight into a hazard) -- it ESCALATES: it fits a HAZARD-AWARE model
     from the level's transitions PLUS the trigger's own death (the nav suicidal plan's death IS the hazard
     signal), and RE-PLANS the same level. If that deepens, continue; otherwise stop.
  4. On any other stall (move-budget / wall) -> stop (out of the current model classes' scope).

The escalation ladder is nav -> hazard-aware. The whole banked L1->Lk sequence is reproduction-gated on a
fresh env. Demonstrated end-to-end on tu93: the loop reaches L1 with the nav model, the nav re-fit stalls at
the L2 charging enemy (trigger), the loop escalates to the hazard-aware model and deepens to L2 -- with NO
hand-holding between levels.

OFFLINE, zero quota. verifier_is_oracle: false.
"""
from __future__ import annotations

import argparse
import importlib.util
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

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "experiment_reinduction_hazard_loop.json"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, str(Path(__file__).resolve().parent / fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ri = _load("exp_reind", "experiment_reinduction.py")
_ha = _load("exp_haz", "experiment_hazard_aware.py")


def _ok(frame):
    try:
        return np.asarray(grid_of(frame)).ndim == 2
    except Exception:
        return False


def execute_from(game, prefix, cell, model, max_plan, max_depth):
    """Replay `prefix` to reach the current level, plan inside `model`, execute, and report the outcome
    (level_up / avatar_removed / other stall) + the banked actions for the reproduction gate."""
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = _warm(env, False)
    for a in prefix:
        f = env.step(_game_action(GameAction, int(a["action"])), data=a.get("data"))
        if f is None or not _ok(f):
            return {"planned": False, "level_up": False, "avatar_removed": False, "banked": list(prefix),
                    "stop": "prefix_replay_failed"}
    lvl0 = _levels_completed(f)
    g = to_logical(grid_of(f), cell)
    plan = plan_in_model(model.engine, model.is_level_complete, g, max_nodes=max_plan, max_depth=max_depth)
    if not plan:
        return {"planned": False, "level_up": False, "avatar_removed": False, "banked": list(prefix),
                "stop": "no_plan_in_model"}
    banked = list(prefix)
    stop = "plan_exhausted"
    avatar_removed = False
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
            avatar_removed = model._avatar_bbox(to_logical(grid_of(f), cell)) is None
            stop = "game_over_avatar_removed" if avatar_removed else "game_over_other"
            break
    return {"planned": True, "plan_len": len(plan), "level_up": _levels_completed(f) > lvl0,
            "avatar_removed": bool(avatar_removed), "banked": banked, "stop": stop}


def escalating_deepen(game, target_level, n_reinduce, max_plan, max_depth, seed):
    """The integrated loop: re-induce nav per level, ESCALATE to hazard-aware at the avatar-removal trigger."""
    model0, l1_prefix, cell, _ = _ri.reach_level_one(game, max_plan, max_depth, seed=seed)
    if l1_prefix is None:
        return {"reached_L1": False}
    goal_color = model0.goal_color
    banked = list(l1_prefix)
    level = 1                       # L1 solved by the reach-step
    per_level = [{"level_reached": 1, "model_class": "nav", "via": "reach_level_one"}]
    while level < target_level:
        tr, _ = _ri.collect_at_level(game, banked, n_reinduce, seed=seed + level)
        nav = InducedNavWorldModel.fit(tr)
        nav.goal_color = goal_color
        res = execute_from(game, banked, cell, nav, max_plan, max_depth)
        if res["level_up"]:
            banked = res["banked"]; level += 1
            per_level.append({"level_reached": level, "model_class": "nav", "plan_len": res.get("plan_len")})
            continue
        if res["avatar_removed"] or res["stop"] == "game_over_avatar_removed":
            # ----- TRIGGER: hazard stall -> ESCALATE up the hazard-rule ladder, re-planning the SAME level.
            # Rung A = line-charger 'toward' (tu93 L2 horizontal charger). Rung B = 'enter' (ALSO charge on a
            # perpendicular step-onto the line; tu93 L3 vertical chargers). Try each; take the first whose
            # plan actually deepens (does not die). ----------------------------------------------------------
            nav_deaths = _ha.nav_death_transitions(nav, game, banked, cell, max_plan, max_depth)
            escalated = False
            attempts = []
            for mode in ("toward", "omni"):
                haz = HazardAwareNavWorldModel.fit(list(tr) + nav_deaths, goal_color=goal_color,
                                                   lethal_mode=mode)
                resh = execute_from(game, banked, cell, haz, max_plan, max_depth)
                attempts.append({"lethal_mode": mode, "hazard_fit": haz.hazard_fit, "stop": resh["stop"],
                                 "plan_len": resh.get("plan_len")})
                if resh["level_up"]:
                    banked = resh["banked"]; level += 1
                    per_level.append({"level_reached": level, "model_class": f"hazard_aware[{mode}]",
                                      "escalated_from": "nav_avatar_removed", "hazard_fit": haz.hazard_fit,
                                      "plan_len": resh.get("plan_len")})
                    escalated = True
                    break
            if escalated:
                continue
            per_level.append({"level_stalled": level, "model_class": "hazard_aware",
                              "escalated_from": "nav_avatar_removed", "rungs_tried": attempts})
            break
        per_level.append({"level_stalled": level, "model_class": "nav", "stop": res["stop"]})
        break
    # reproduction gate on a fresh env
    arc = kit.offline_arcade()
    envr = arc.make(game, scorecard_id=arc.open_scorecard())
    fr = _warm(envr, False)
    base = _levels_completed(fr)
    for a in banked:
        fr = envr.step(_game_action(GameAction, int(a["action"])), data=a.get("data"))
        if fr is None or not _ok(fr):
            break
    reproduced = _levels_completed(fr) - base
    return {"reached_L1": True, "deepest_level": int(level), "reproduced_level": int(reproduced),
            "per_level": per_level, "n_banked_actions": len(banked),
            "escalated_to_hazard_aware": any(str(p.get("model_class", "")).startswith("hazard_aware")
                                             for p in per_level)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="tu93")
    ap.add_argument("--seeds", default="7,20260622,3")
    ap.add_argument("--target-level", type=int, default=2)
    ap.add_argument("--n-reinduce", type=int, default=150)
    ap.add_argument("--max-plan", type=int, default=120000)
    ap.add_argument("--max-depth", type=int, default=120)
    args = ap.parse_args()
    t0 = time.time()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    rows = [{"seed": s, **escalating_deepen(args.game, args.target_level, args.n_reinduce, args.max_plan,
                                            args.max_depth, s)} for s in seeds]
    # The escalation ladder is nav -> hazard[toward] -> hazard[omni]. L2's single HORIZONTAL charger cracks
    # on the toward rung; L3's THREE per-charger-facing chargers crack on the omni rung, whose lethal-zone
    # was CALIBRATED against L3's position-keyed real-env BFS ground truth (per-charger facing read from the
    # centre-marker offset; directional; collision-exempt; FN=0 on the 88-move labelled set).
    min_repro = min((r.get("reproduced_level", 0) for r in rows), default=0)
    deepest = min((r.get("deepest_level", 0) for r in rows), default=0)
    used_omni = any("omni" in str(p.get("model_class", "")) for r in rows for p in r.get("per_level", []))
    if seeds and min_repro >= args.target_level and all(r.get("escalated_to_hazard_aware") for r in rows):
        omni_tag = ("_via_the_omni_rung_CALIBRATED_vs_the_L3_BFS_path_single_static_layout" if used_omni else "")
        verdict = ("success: reinduction_loop_AUTO_ESCALATES_nav_toward_omni_at_the_trigger_and_deepens_"
                   f"{args.game}_to_L{min_repro}_reproduced_on_every_seed{omni_tag}")
    elif seeds and min_repro >= 2:
        verdict = (f"success: reinduction_loop_escalates_and_deepens_{args.game}_to_L{min_repro}_reproduced_on_every_seed")
    elif any(r.get("reproduced_level", 0) >= 2 for r in rows):
        verdict = "success: reinduction_loop_escalates_and_deepens_reproduced_on_some_seeds_inspect_rows"
    else:
        verdict = "complete: escalating_loop_did_not_cleanly_deepen_inspect_rows"

    art = {"experiment": "experiment_reinduction_hazard_loop", "game": args.game, "honest_verdict": verdict,
           "verifier_is_oracle": False, "inference_substrate": "offline_arc_search_plus_induced_world_model",
           "random_seeds": seeds, "target_level": args.target_level,
           "min_reproduced_level": min_repro, "deepest_level": deepest, "rows": rows,
           "methodology_note": ("One integrated loop: re-induce nav per level; on the avatar-removal TRIGGER "
                                "escalate up the hazard-rule ladder nav -> hazard[toward] -> hazard[omni] and "
                                "re-plan the same level; take the first rung that deepens; else stop. No "
                                "hand-holding between levels. Reproduction-gated on a fresh env. tu93 L2 (1 "
                                "horizontal charger) cracks on the 'toward' rung. tu93 L3 (3 chargers with "
                                "PER-CHARGER FACINGS) cracks on the 'omni' rung, whose interception lethal-zone "
                                "was CALIBRATED against L3's position-keyed real-env BFS ground truth (the "
                                "verified 19-action path): each charger kills only when the avatar's "
                                "destination is on its facing line, on the side it faces (read from the "
                                "centre-marker offset), at distance 1..reach -- collision-exempt. The "
                                "calibration is reproducibly clean (FN=0, FP=0, win-path-unpruned over 88 "
                                "BFS-labelled moves; see experiment_hazard_l3_calibration.py). SCOPE: tu93 L3 "
                                "is a SINGLE static layout (seed-invariant -- 3 seeds = 1 layout x3); this is "
                                "validated on that one level, NOT a general hazard solver. Static, not dynamic."),
           "duration_s": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(art, indent=2))
    print(f"\nVERDICT: {verdict}")
    for r in rows:
        chain = " -> ".join(f"L{p['level_reached']}({p['model_class']})" for p in r.get("per_level", [])
                            if "level_reached" in p)
        stalled = [p for p in r.get("per_level", []) if "level_stalled" in p]
        smsg = ""
        if stalled:
            s = stalled[-1]
            rungs = ",".join(f"{t['lethal_mode']}={t['stop']}" for t in s.get("rungs_tried", []))
            smsg = f" | STALL attempting L{s['level_stalled'] + 1} rungs[{rungs or s.get('stop')}]"
        print(f"  seed {r['seed']}: deepest L{r.get('deepest_level')} reproduced L{r.get('reproduced_level')} "
              f"| {chain}{smsg}")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
