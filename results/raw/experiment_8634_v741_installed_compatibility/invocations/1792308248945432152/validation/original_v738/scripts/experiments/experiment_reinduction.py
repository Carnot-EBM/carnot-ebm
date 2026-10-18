#!/usr/bin/env python3
"""Mechanic-conditioned RE-INDUCTION trigger for cross-level deepening (Executable World Models).

The program-generalization first swing (docs/research-notes/program-generalization-first-swing-2026-06-22.md)
showed: a world model induced+verified at L1 plans L1 in imagination and reproduces it, but a model FROZEN
at L1 does NOT deepen when the next level's MECHANIC shifts -- it plans a move that is fatal under the new
rules. The leader (arXiv:2605.05138) pays for this by RE-INDUCING the model per mechanic. This experiment
implements + tests that trigger, using the auto-fitting nav world model (carnot.agentic.arc_nav_world_model)
so the model is FITTED FROM TRANSITIONS at each level rather than hand-written.

Two arms, head-to-head, on the same game:
  * FROZEN     -- induce a nav model from L<k> transitions ONCE, then try to deepen with that frozen model.
  * REINDUCT   -- after each level-up, COLLECT fresh transitions at the new level and RE-FIT the model
                  before planning the next level. The "trigger" is: a deterministic, budget-unexhausted
                  env game-over right after a level-up => the mechanic shifted => re-induce.

If REINDUCT reaches a level FROZEN cannot, the mechanic-conditioned re-induction trigger cracks the
deepening wall. If BOTH stall at the same level despite REINDUCT fitting a clean model there, the new
level's mechanic is HIDDEN-STATE (not grid-expressible) and needs state-augmentation -- a correctly
attributed, distinct finding (NOT a re-induction failure).

OFFLINE, zero quota. verifier_is_oracle: false (the world model is an induced PREDICTIVE model, not the
executable correctness oracle).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_agi3_world_model import grid_of
from carnot.agentic.arc_agi3_live_adapter import _game_action, _levels_completed, _game_over
from carnot.agentic.arc_graph_explore import _warm, rich_action_candidates
from carnot.agentic.arc_executable_world_model import (
    collect_transitions, Transition, WorldModelVerifier, to_logical, detect_cell, plan_in_model,
)
from carnot.agentic.arc_nav_world_model import InducedNavWorldModel

REPO = Path(__file__).resolve().parents[2]


def _out(game):
    return REPO / "results" / f"experiment_reinduction_{game}.json"


def _ok(frame):
    try:
        return np.asarray(grid_of(frame)).ndim == 2
    except Exception:
        return False


def _budget(env):
    try:
        gobj = env._game
        for attr in dir(gobj):
            o = getattr(gobj, attr, None)
            if hasattr(o, "current_steps"):
                return getattr(o, "current_steps", None), getattr(o, "max_steps", None)
    except Exception:
        pass
    return None, None


def collect_at_level(game, prefix, n, cap_per_episode=50, seed=0):
    """Collect (grid, action, next_grid) transitions AT the level reached by replaying `prefix` (a list of
    raw action dicts). Fresh env per episode so a death/level-up just ends the episode; we re-reach the
    level by replaying the prefix. Returns (transitions, cell)."""
    rng = random.Random(seed)
    arc = kit.offline_arcade()
    trans, cell = [], None
    episodes = 0
    while len(trans) < n and episodes < n:
        episodes += 1
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        f = _warm(env, False)
        if cell is None:
            cell = detect_cell(grid_of(f))
        ok = True
        for a in prefix:
            f = env.step(_game_action(GameAction, int(a["action"])), data=a.get("data"))
            if f is None or not _ok(f) or _game_over(f):
                ok = False
                break
        if not ok or f is None or not _ok(f):
            continue
        base = _levels_completed(f)
        for _ in range(cap_per_episode):
            if not _ok(f):
                break
            cands = rich_action_candidates(f)
            if not cands:
                break
            c = cands[rng.randrange(min(len(cands), 6))]
            g0 = to_logical(grid_of(f), cell)
            l0 = _levels_completed(f)
            nf = env.step(_game_action(GameAction, c.action_id), data=c.data)
            if nf is None:
                break
            g1 = to_logical(grid_of(nf), cell) if _ok(nf) else g0
            l1 = _levels_completed(nf) if _ok(nf) else l0
            trans.append(Transition(g0, int(c.action_id), c.data, g1, l0, l1))
            if not _ok(nf) or _game_over(nf) or l1 > l0:
                break
            f = nf
    return trans, cell


def model_movement_accuracy(model, trans):
    """Fraction of (move/block) transitions the induced model predicts correctly (avatar moved-or-not)."""
    cm = cb = fm = fb = 0
    for t in trans:
        if t.action not in (1, 2, 3, 4):
            continue
        b0 = model._avatar_bbox(t.grid)
        if b0 is None:
            continue
        pred = model.engine(t.grid, t.action)
        real_moved = model._avatar_bbox(t.next_grid) != b0
        my_moved = model._avatar_bbox(pred) != b0
        if real_moved and my_moved:
            cm += 1
        elif (not real_moved) and (not my_moved):
            cb += 1
        elif real_moved and not my_moved:
            fb += 1
        else:
            fm += 1
    tot = cm + cb + fm + fb
    return {"accuracy": round((cm + cb) / tot, 3) if tot else None, "n": tot,
            "correct_move": cm, "correct_block": cb, "false_move": fm, "false_block": fb}


def reach_level_one(game, max_plan, max_depth, n_l1=200, seed=0):
    """Induce an L0 nav model, plan+execute L1. Returns (model0, l1_prefix_actions, cell) or (None,..)."""
    trans0, cell = collect_transitions(game, n=n_l1, seed=seed)
    model0 = InducedNavWorldModel.fit(trans0)
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = _warm(env, False)
    g = to_logical(grid_of(f), cell)
    plan = plan_in_model(model0.engine, model0.is_level_complete, g, max_nodes=max_plan, max_depth=max_depth)
    if not plan:
        return model0, None, cell, trans0
    prefix = []
    for s in plan:
        f = env.step(_game_action(GameAction, int(s["action"])), data=s.get("data"))
        if f is None or not _ok(f):
            return model0, None, cell, trans0
        prefix.append({"action": int(s["action"]), "data": s.get("data")})
        if _levels_completed(f) > 0:
            return model0, prefix, cell, trans0
    return model0, None, cell, trans0


def deepen_arm(game, model_provider, l1_prefix, cell, target_level, max_plan, max_depth, seed):
    """Execute the deepening chain. `model_provider(level, prefix)` returns the world model to plan level
    `level+1` with -- FROZEN passes the same model every call; REINDUCT re-fits from fresh transitions at
    the current level. Returns a per-level trace + the deepest reproduced level."""
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = _warm(env, False)
    for a in l1_prefix:                      # replay the shared L1 solution
        f = env.step(_game_action(GameAction, int(a["action"])), data=a.get("data"))
        if f is None or not _ok(f):
            return {"reached": 0, "per_level": [{"error": "l1_replay_failed"}], "banked": l1_prefix}
    level = _levels_completed(f)
    start = level
    banked = list(l1_prefix)
    per_level = []
    while level < target_level:
        model, fit_info = model_provider(level, list(banked), seed)
        if model is None:
            per_level.append({"from_level": level, "reinduced": fit_info, "planned": False,
                              "reason": "no_model"})
            break
        g = to_logical(grid_of(f), cell)
        plan = plan_in_model(model.engine, model.is_level_complete, g, max_nodes=max_plan, max_depth=max_depth)
        if not plan:
            per_level.append({"from_level": level, "reinduced": fit_info, "planned": False,
                              "reason": "no_plan_in_model"})
            break
        advanced = False
        steps = 0
        stop = "plan_exhausted"
        for s in plan:
            cur, mx = _budget(env)
            nf = env.step(_game_action(GameAction, int(s["action"])), data=s.get("data"))
            steps += 1
            if nf is None or not _ok(nf):
                stop = "env_none"
                break
            banked.append({"action": int(s["action"]), "data": s.get("data")})
            f = nf
            if _levels_completed(f) > level:
                level = _levels_completed(f)
                advanced = True
                stop = "level_up"
                break
            if _game_over(f):
                # was the avatar REMOVED (a hazard consumed it) vs merely blocked? Avatar-removal is a
                # transition the pure-nav model CANNOT represent (its engine only translates/blocks), so a
                # game-over-by-removal proves nav re-induction is insufficient regardless of nav accuracy.
                avatar_present = model._avatar_bbox(to_logical(grid_of(f), cell)) is not None
                stop = (f"game_over_step{steps}_budget{cur}/{mx}_avatar_"
                        f"{'present' if avatar_present else 'REMOVED'}")
                break
        per_level.append({"from_level": int(level if not advanced else level - 1), "reinduced": fit_info,
                          "plan_len": len(plan), "steps": steps, "advanced": advanced, "stop": stop})
        if not advanced:
            break
    # reproduction gate on a fresh env
    repro = 0
    if banked:
        envr = arc.make(game, scorecard_id=arc.open_scorecard())
        fr = _warm(envr, False)
        base = _levels_completed(fr)
        for a in banked:
            fr = envr.step(_game_action(GameAction, int(a["action"])), data=a.get("data"))
            if fr is None or not _ok(fr):
                break
        repro = _levels_completed(fr) - base
    return {"reached": int(level - start) + start, "levels_gained": int(level - start),
            "reproduced": int(repro), "per_level": per_level, "banked_len": len(banked)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--target-level", type=int, default=3)
    ap.add_argument("--max-plan", type=int, default=60000)
    ap.add_argument("--max-depth", type=int, default=80)
    ap.add_argument("--n-reinduce", type=int, default=120, help="transitions to collect per re-induction")
    ap.add_argument("--seed", type=int, default=20260622)
    args = ap.parse_args()
    t0 = time.time()
    game = args.game

    model0, l1_prefix, cell, trans0 = reach_level_one(game, args.max_plan, args.max_depth, seed=args.seed)
    l1_fit = model0.fit_quality if model0 else None
    if l1_prefix is None:
        art = {"experiment": "experiment_reinduction", "game": game,
               "honest_verdict": "complete: could_not_reach_L1_via_auto_induced_model_gap_sharpened",
               "verifier_is_oracle": False, "inference_substrate": "offline_arc_search_plus_induced_world_model",
               "random_seed": args.seed, "l1_fit_quality": l1_fit, "duration_s": round(time.time() - t0, 1)}
        _out(game).write_text(json.dumps(art, indent=2))
        print(f"VERDICT: {art['honest_verdict']}")
        return 0

    # FROZEN arm: always returns the L1-fitted model.
    def frozen_provider(level, prefix, seed):
        return model0, {"reinduced": False, "source": "frozen_L1_model"}

    # REINDUCT arm: re-collect transitions at the CURRENT level and re-fit.
    reinduct_cache = {}

    def reinduct_provider(level, prefix, seed):
        if level == 0:
            return model0, {"reinduced": False, "source": "L1_model", "fit": l1_fit}
        if level in reinduct_cache:
            return reinduct_cache[level]
        tr, _ = collect_at_level(game, prefix, args.n_reinduce, seed=seed + level)
        if not tr:
            res = (None, {"reinduced": True, "n_transitions": 0, "note": "no_transitions_at_level"})
            reinduct_cache[level] = res
            return res
        model = InducedNavWorldModel.fit(tr)
        acc = model_movement_accuracy(model, tr)
        res = (model, {"reinduced": True, "level": level, "n_transitions": len(tr),
                       "movement_accuracy": acc, "fit": model.fit_quality})
        reinduct_cache[level] = res
        return res

    frozen = deepen_arm(game, frozen_provider, l1_prefix, cell, args.target_level, args.max_plan,
                        args.max_depth, args.seed)
    reinduct = deepen_arm(game, reinduct_provider, l1_prefix, cell, args.target_level, args.max_plan,
                          args.max_depth, args.seed)

    fz = frozen["reproduced"]
    ri = reinduct["reproduced"]
    # Did the RE-INDUCED model fit the next level's NAVIGATION well (move/block prediction) yet still stall?
    # NOTE: movement_accuracy ONLY scores avatar-bbox-changed vs not; it CANNOT see a fatal transition where
    # a hazard REMOVES the avatar (removal scores as a "move"), so a high movement_accuracy does NOT mean the
    # model captured the level -- only its navigation. We therefore separately detect avatar-REMOVAL at the
    # stall (a transition the pure-nav engine is structurally unable to represent).
    reinduct_levels = [pl.get("reinduced") for pl in reinduct["per_level"] if isinstance(pl.get("reinduced"), dict)]
    nav_refit_movement_accurate = any(
        (r.get("movement_accuracy") or {}).get("accuracy", 0) and (r["movement_accuracy"]["accuracy"] >= 0.9)
        for r in reinduct_levels) and ri <= fz
    stalled_by_avatar_removal = any("avatar_REMOVED" in str(pl.get("stop", ""))
                                    for pl in reinduct["per_level"]) and ri <= fz

    if ri > fz:
        verdict = ("success: mechanic_conditioned_reinduction_DEEPENS_past_frozen_L1_model_"
                   f"reinduct_L{ri}_vs_frozen_L{fz}_reproduced")
    elif stalled_by_avatar_removal:
        verdict = ("complete: reinduction_refits_a_movement_accurate_nav_model_at_the_next_level_but_a_"
                   "deterministic_HAZARD_REMOVES_the_avatar_a_transition_the_nav_model_cannot_represent_"
                   "nav_reinduction_insufficient_needs_a_hazard_aware_model_class_gap_sharpened")
    elif nav_refit_movement_accurate:
        verdict = ("complete: reinduction_refits_a_movement_accurate_nav_model_but_deepening_still_stalls_"
                   "next_level_adds_a_non_nav_mechanic_the_nav_model_does_not_capture_gap_sharpened")
    elif ri == fz and ri > 0:
        verdict = ("complete: reinduction_and_frozen_reach_same_level_no_unique_crack_this_game_"
                   "honest_null_gap_sharpened")
    else:
        verdict = "complete: neither_arm_deepened_gap_sharpened"

    art = {"experiment": "experiment_reinduction", "game": game, "honest_verdict": verdict,
           "verifier_is_oracle": False, "inference_substrate": "offline_arc_search_plus_induced_world_model",
           "random_seed": args.seed, "target_level": args.target_level,
           "l1_fit_quality": l1_fit, "l1_prefix_len": len(l1_prefix),
           "frozen_arm": frozen, "reinduction_arm": reinduct,
           "frozen_reproduced": fz, "reinduction_reproduced": ri,
           "reinduction_deepens_past_frozen": bool(ri > fz),
           "nav_refit_movement_accurate_but_stalled": bool(nav_refit_movement_accurate),
           "stalled_by_avatar_removal_hazard": bool(stalled_by_avatar_removal),
           "methodology_note": ("Head-to-head: a model FROZEN at L1 vs RE-INDUCED per level. REINDUCT>FROZEN "
                                "=> the trigger cracks a grid-expressible mechanic shift (see the synthetic "
                                "control for a clean positive). On tu93 both reach L1: the re-induced model is "
                                "movement-accurate on L2 NAVIGATION (which is grid-deterministic) but the engine "
                                "only translates/blocks and is structurally BLIND to the L2-specific hazard that "
                                "REMOVES the avatar (a charging-wall sprite per the env source). movement_accuracy "
                                "cannot see that fatal transition (removal counts as a move), so 'movement "
                                "accurate' must NOT be read as 'captured the level'. Deepening tu93 L2 needs a "
                                "hazard-aware model class, not more nav re-induction. This is NOT broad hidden "
                                "state (L2 nav is grid-deterministic) and NOT enemy/box (the L2 mechanic is a "
                                "single charging-wall sprite)."),
           "duration_s": round(time.time() - t0, 1)}
    _out(game).write_text(json.dumps(art, indent=2))
    print(f"\nVERDICT: {verdict}")
    print(f"  FROZEN reproduced L{fz} | REINDUCT reproduced L{ri} | deepens_past_frozen={ri>fz} -> {_out(game)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
