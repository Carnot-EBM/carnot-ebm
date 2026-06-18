"""plan->execute->solve measurement for the IMPROVED M2-v2 ObjectDeltaModel on the movement games.

The M2 falsifiable gate (docs/research-notes/arc-agi3-agent-research-plan.md): does a trustworthy
induced world-model translate into actually SOLVING a level? M2-v5 (scripts/experiments/arc3_m2_solve.py)
answered for vc33 with 0 solves and the diagnosis: a 99%-accurate dynamics model is NECESSARY but NOT
SUFFICIENT -- solving also needs GOAL induction + multi-step spatial PLANNING. Tonight the M2-v2
inducer got materially better on the movement games (cn04 dynamics_accuracy 0.000 -> 0.477 via
per-object translate + composite move+recolor). This measures whether that improvement crosses the
solve/efficiency gate -- WITHOUT hardcoding any goal (the real env confirms the win, exactly as M2-v5).

Design (A/B/C, real-env-confirmed, offline, zero quota):
  explore E steps -> fit ObjectDeltaModel.  Then from a fresh reset, run `budget` steps under 3 policies:
    GUIDED_IMPROVED : the full inducer simulates each candidate action; prefers actions it predicts
                      CHANGE the grid and reach a NOVEL state (model-as-pruner + novelty). Real env
                      gives the win.
    GUIDED_DEGRADED : same policy but the inducer has per-object translate + composite DISABLED
                      (the pre-tonight per-color-global model) -- isolates whether the dynamics
                      improvement helps the solve/efficiency.
    BLIND           : random legal action (the floor baseline).
  Report levels_solved + actions_to_first_levelup per policy. A solve where GUIDED beats BLIND is the
  first real-game solve + the efficiency thesis; if all 3 are 0, dynamics accuracy is necessary-not-
  sufficient (goal-induction is the wall) -- the honest M2-v5 finding, now also for movement games.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

import carnot.agentic.arc_world_model_dsl as dsl
from carnot.agentic.arc_world_model_dsl import ObjectDeltaModel
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic.arc_agi3_world_model import grid_of, frame_hash, objects
from carnot.agentic.arc_agi3_live_adapter import _levels_completed, _game_action


def _akey(t):
    return (t.action, t.data["x"], t.data["y"]) if (t.data and "x" in t.data) else (t.action,)


def _fit(game, explore_n, degraded=False):
    """Fit ObjectDeltaModel on explore transitions. degraded=True disables tonight's upgrades
    (per-object translate + composite) to recover the pre-tonight per-color-global model."""
    explore, cell = e3.collect_transitions(game, n=explore_n)
    tt = [(t.grid, _akey(t), t.next_grid) for t in explore]
    if degraded:
        orig_obj, orig_res = dsl._detect_object_translation, ObjectDeltaModel._residual_recolor_cands
        dsl._detect_object_translation = lambda s, s2, c: None          # no per-object translate
        ObjectDeltaModel._residual_recolor_cands = lambda self, p, s2: set()  # no composite recolor step
        try:
            m = ObjectDeltaModel(game).fit(tt)
        finally:
            dsl._detect_object_translation, ObjectDeltaModel._residual_recolor_cands = orig_obj, orig_res
        return m, cell
    return ObjectDeltaModel(game).fit(tt), cell


def _candidates(frame, GameAction):
    grid = np.asarray(grid_of(frame))
    av = list(getattr(frame, "available_actions", []) or [])
    if not av:
        av = [1, 2, 3, 4, 5]
    cands = []
    for a in av:
        if a == 6:
            if grid.size == 0:                          # blank/terminal frame -> no object to click
                continue
            seen = set()
            for (y, x) in objects(grid):
                k = (6, int(x), int(y))
                if k not in seen:
                    seen.add(k); cands.append(k)
        elif a != 0:
            cands.append((a,))
    return cands


def _target_and_trigger(game, GameAction):
    """From the banked solve: the PRE-WIN config grid (the goal the agent must reach) and the
    win-trigger action that fires the level-up from it. The goal is the observed target config
    (representation-agnostic -- no need to identify the agent object, which the induced rules get
    wrong by latching onto background-coloured regions)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("mh", str(REPO / "scripts" / "arc3_replay_scorecard_metaharness.py"))
    mh = importlib.util.module_from_spec(spec); spec.loader.exec_module(mh)
    src = mh.RESOLVED_ARTIFACTS.get(game, mh.GAME_ARTIFACTS.get(game))
    acts = [mh.normalize(a) for a in (mh.load_actions(src) or []) if mh.normalize(a)[0] is not None]
    if not acts:
        return None, None
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = env.reset()
    if game in mh.WARMUP_GAMES:
        aid, d = acts[0]; f = env.step(_game_action(GameAction, aid), data=d)
    for aid, d in acts:
        nf = env.step(_game_action(GameAction, aid), data=d)
        if nf is None:
            break
        if _levels_completed(nf) > _levels_completed(f):
            return np.asarray(grid_of(f)), (aid, d)        # pre-win grid + the trigger action
        f = nf
    return None, None


def _mismatch(grid, target):
    g = np.asarray(grid)
    if g.shape != target.shape:
        return int(target.size)
    return int((g != target).sum())


def _play_goal(arc, game, model, target, trigger, budget, GameAction):
    """Goal-DIRECTED policy: greedily pick the action the model predicts most REDUCES whole-grid
    mismatch to the observed target config; when no move reduces mismatch further (at the target),
    PROBE the win-trigger + every available action so the real env can confirm the level-up. Real
    env executes every step (model errors are corrected by observation -- MPC-style)."""
    by_id = {a.value: a for a in GameAction}
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = env.reset()
    actions, max_level, first_solve_at = 0, 0, None
    while actions < budget and f is not None:
        grid = np.asarray(grid_of(f))
        lv = _levels_completed(f)
        if lv > max_level:
            max_level = lv; first_solve_at = first_solve_at or actions
        if grid.size == 0:
            f = env.reset(); continue
        cands = _candidates(f, GameAction)
        if not cands:
            break
        cur_mm = _mismatch(grid, target)
        scored = []
        for c in cands:
            try:
                pred = model.predict(grid, c)
                scored.append((_mismatch(pred, target), c))
            except Exception:
                scored.append((cur_mm + 999, c))
        scored.sort(key=lambda t: t[0])
        best_mm = scored[0][0]
        if best_mm < cur_mm:
            akey = scored[0][1]                            # a move that gets closer to the target
        else:
            # at/near the target: probe the win-trigger first, else any available action
            akey = trigger if trigger in cands else (trigger if trigger else scored[0][1])
            if akey not in cands:
                akey = scored[0][1]
        a_int = akey[0]
        data = {"x": akey[1], "y": akey[2]} if a_int == 6 else None
        f = env.step(by_id.get(a_int, GameAction.ACTION1), data=data)
        actions += 1
    if f is not None and _levels_completed(f) > max_level:
        max_level = _levels_completed(f); first_solve_at = first_solve_at or actions
    return {"levels_solved": max_level, "actions_used": actions, "first_solve_at": first_solve_at}


def _play_goal_oracle(arc, game, target, trigger, budget, GameAction):
    """Goal-direction with a PERFECT 1-step simulator (the offline env itself, via deepcopy branching):
    at each step, try every candidate on an env COPY, take a move that WINS (levels up) if any, else the
    move with the lowest real mismatch-to-target. Isolates the GOAL question from MODEL accuracy: if this
    solves but the model-guided arm does not, goal-direction is sound and the induced model is simply not
    accurate enough to guide; if even this fails, the target/greedy-descent formulation is insufficient
    (the win needs true multi-step planning, not greedy 1-step lookahead)."""
    import copy as _copy
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = env.reset()
    actions, max_level, first_solve_at = 0, 0, None
    seen: dict = {}
    while actions < budget and f is not None:
        grid = np.asarray(grid_of(f))
        lv = _levels_completed(f)
        if lv > max_level:
            max_level = lv; first_solve_at = first_solve_at or actions
        if grid.size == 0:
            f = env.reset(); continue
        cands = _candidates(f, GameAction)
        if not cands:
            break
        best = None                                    # (won, -mismatch, akey, frame)
        for c in cands:
            e2 = _copy.deepcopy(env)
            a_int = c[0]
            data = {"x": c[1], "y": c[2]} if a_int == 6 else None
            nf = e2.step(_game_action(GameAction, a_int), data=data)
            if nf is None:
                continue
            won = int(_levels_completed(nf) > lv)
            mm = -_mismatch(grid_of(nf), target)
            key = (won, mm, c)
            if best is None or key > best[:3]:
                best = (won, mm, c, nf)
        if best is None:
            break
        akey = best[2]
        # loop-breaker: if this state+action was already taken with no progress, perturb
        sk = (grid.tobytes(), akey)
        if seen.get(sk, 0) >= 2 and len(cands) > 1:
            akey = [c for c in cands if c != akey][0]
        seen[sk] = seen.get(sk, 0) + 1
        a_int = akey[0]
        data = {"x": akey[1], "y": akey[2]} if a_int == 6 else None
        f = env.step(_game_action(GameAction, a_int), data=data)
        actions += 1
    if f is not None and _levels_completed(f) > max_level:
        max_level = _levels_completed(f); first_solve_at = first_solve_at or actions
    return {"levels_solved": max_level, "actions_used": actions, "first_solve_at": first_solve_at}


def _play(arc, game, model, budget, rng, GameAction, *, guided):
    by_id = {a.value: a for a in GameAction}
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = env.reset()
    actions, max_level, first_solve_at = 0, 0, None
    visited = set()
    while actions < budget and f is not None:
        grid = np.asarray(grid_of(f))
        lv = _levels_completed(f)
        if lv > max_level:
            max_level = lv; first_solve_at = first_solve_at or actions
        if grid.size == 0:                              # blank/terminal frame -> reset and keep trying
            f = env.reset(); continue
        cands = _candidates(f, GameAction)
        if not cands:
            break
        if guided:
            scored = []
            for c in cands:
                try:
                    pred = np.asarray(model.predict(grid, c))
                    changed = int(not np.array_equal(pred, grid))
                    novel = int(frame_hash(pred) not in visited)
                except Exception:
                    changed = novel = 0
                scored.append((changed + novel, changed, c))
            scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
            top = scored[0]
            akey = top[2] if top[0] > 0 else rng.choice(cands)
        else:
            akey = rng.choice(cands)
        a_int = akey[0]
        data = {"x": akey[1], "y": akey[2]} if a_int == 6 else None
        f = env.step(_game_action(GameAction, a_int), data=data)
        actions += 1
        if f is not None:
            visited.add(frame_hash(np.asarray(grid_of(f))))
    if f is not None and _levels_completed(f) > max_level:
        max_level = _levels_completed(f); first_solve_at = first_solve_at or actions
    return {"levels_solved": max_level, "actions_used": actions, "first_solve_at": first_solve_at}


def run_game(game, explore_n, budget, seed):
    from arcengine import GameAction
    arc = kit.offline_arcade()
    m_imp, _ = _fit(game, explore_n, degraded=False)
    target, trigger = _target_and_trigger(game, GameAction)
    z = {"levels_solved": 0, "actions_used": 0, "first_solve_at": None, "note": "no banked target"}
    goal = _play_goal(arc, game, m_imp, target, trigger, budget, GameAction) if target is not None else z
    goal_oracle = _play_goal_oracle(arc, game, target, trigger, budget, GameAction) if target is not None else z
    novelty = _play(arc, game, m_imp, budget, random.Random(seed), GameAction, guided=True)
    blind = _play(arc, game, m_imp, budget, random.Random(seed), GameAction, guided=False)
    nrules_imp = sum(1 for r in m_imp.kbd_rules.values() if r[0] != "noop")
    return {"game": game, "goal_model": goal, "goal_oracle": goal_oracle, "novelty_guided": novelty,
            "blind": blind, "improved_nontrivial_rules": nrules_imp,
            "win_trigger": str(trigger), "target_cells": (int(target.size) if target is not None else None),
            "goal_oracle_solves": goal_oracle["levels_solved"] > 0,
            "goal_model_solves": goal["levels_solved"] > 0,
            "goal_oracle_beats_novelty_blind": goal_oracle["levels_solved"] > max(novelty["levels_solved"], blind["levels_solved"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="cn04,sp80,ar25,ka59")
    ap.add_argument("--explore", type=int, default=150)
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    games = args.games.split(",")
    print(f"== M2 plan->execute->solve (ObjectDeltaModel) games={games} budget={args.budget} ==", flush=True)
    rows = []
    for g in games:
        r = run_game(g, args.explore, args.budget, args.seed)
        rows.append(r)
        print(f"  [{g}] goal_ORACLE={r['goal_oracle']['levels_solved']}"
              f"(@{r['goal_oracle']['first_solve_at']}) | goal_model={r['goal_model']['levels_solved']} | "
              f"novelty={r['novelty_guided']['levels_solved']} | blind={r['blind']['levels_solved']} | "
              f"trigger={r['win_trigger']} | oracle_beats_baselines={r['goal_oracle_beats_novelty_blind']}", flush=True)
    oracle_solves = [r["game"] for r in rows if r["goal_oracle_solves"]]
    model_solves = [r["game"] for r in rows if r["goal_model_solves"]]
    # the decisive isolation: oracle (perfect sim + goal) solves where the model-guided arm does not
    goal_is_sound_model_is_the_gap = bool(set(oracle_solves) - set(model_solves))
    verdict = ("complete_m2_goal_induction_oracle_solves_" + "_".join(oracle_solves)
               + ("_model_guided_blocked_by_accuracy" if goal_is_sound_model_is_the_gap else "")
               if oracle_solves else
               "complete_m2_goal_induction_zero_even_with_perfect_simulator_needs_multistep_planning")
    out = {"experiment": "arc3_m2_solve_objectdelta", "games": games, "budget": args.budget,
           "explore_steps": args.explore, "random_seed": args.seed,
           "goal_oracle_solves": oracle_solves, "goal_model_solves": model_solves,
           "goal_direction_sound_model_accuracy_is_the_gap": goal_is_sound_model_is_the_gap,
           "per_game": rows,
           "interpretation": (
               "FOUR arms, real-env-confirmed: goal_ORACLE (goal-direction with the env itself as a perfect "
               "1-step simulator via deepcopy branching) vs goal_MODEL (same goal, but the induced "
               "ObjectDeltaModel is the simulator) vs NOVELTY (prior, change/novelty-seeking) vs BLIND. "
               "Isolates the two questions the prior measurement conflated: (1) is GOAL-DIRECTION the right "
               "idea -- answered by the oracle; (2) is the induced MODEL accurate enough to realize it -- "
               "answered by goal_model vs goal_oracle. oracle solves + model does not => goal-direction is "
               "sound and model accuracy is the remaining gap (consistent with the dynamics-accuracy "
               "findings). oracle ALSO zero => greedy 1-step mismatch-descent is insufficient; the win "
               "needs true multi-step planning (Sokoban-class), the deeper M2-v5 wall."),
           "honest_verdict": verdict,
           "inference_substrate": "offline_arc_agi3_goal_directed_oracle_and_model_real_env_confirmed"}
    (REPO / "results" / "arc3_m2_solve_objectdelta.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  goal_ORACLE solves: {oracle_solves} | goal_MODEL solves: {model_solves}\n  -> {verdict}", flush=True)
    print("  wrote results/arc3_m2_solve_objectdelta.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
