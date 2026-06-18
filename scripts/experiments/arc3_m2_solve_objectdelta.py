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
    m_deg, _ = _fit(game, explore_n, degraded=True)
    imp = _play(arc, game, m_imp, budget, random.Random(seed), GameAction, guided=True)
    deg = _play(arc, game, m_deg, budget, random.Random(seed), GameAction, guided=True)
    blind = _play(arc, game, m_imp, budget, random.Random(seed), GameAction, guided=False)
    nrules_imp = sum(1 for r in m_imp.kbd_rules.values() if r[0] != "noop")
    return {"game": game, "guided_improved": imp, "guided_degraded": deg, "blind": blind,
            "improved_nontrivial_rules": nrules_imp,
            "improved_solves": imp["levels_solved"] > 0,
            "improvement_helped": imp["levels_solved"] > max(deg["levels_solved"], blind["levels_solved"])}


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
        print(f"  [{g}] improved={r['guided_improved']['levels_solved']} "
              f"(@{r['guided_improved']['first_solve_at']}) | degraded={r['guided_degraded']['levels_solved']} | "
              f"blind={r['blind']['levels_solved']} | solves={r['improved_solves']} helped={r['improvement_helped']}", flush=True)
    any_solve = any(r["improved_solves"] for r in rows)
    any_helped = any(r["improvement_helped"] for r in rows)
    verdict = ("complete_m2_objectdelta_solve_" + ("levels_solved" if any_solve else
               "zero_solves_dynamics_necessary_not_sufficient_goal_induction_is_the_wall"))
    out = {"experiment": "arc3_m2_solve_objectdelta", "games": games, "budget": args.budget,
           "explore_steps": args.explore, "random_seed": args.seed,
           "any_solve": any_solve, "improvement_helped_any": any_helped, "per_game": rows,
           "interpretation": (
               "Model-guided (improved ObjectDeltaModel as forward simulator + pruner/novelty) vs the "
               "pre-tonight degraded model vs blind, real-env-confirmed wins, NO hardcoded goal. A solve "
               "where improved>blind = first real-game solve + efficiency thesis. All-zero = the improved "
               "dynamics model is necessary-not-sufficient and goal-induction is the binding wall "
               "(confirms/extends the M2-v5 vc33 finding to the movement games)."),
           "honest_verdict": verdict,
           "inference_substrate": "offline_arc_agi3_objectdelta_model_guided_real_env_confirmed"}
    (REPO / "results" / "arc3_m2_solve_objectdelta.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  any solve: {any_solve} | improvement helped: {any_helped}\n  -> {verdict}", flush=True)
    print("  wrote results/arc3_m2_solve_objectdelta.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
