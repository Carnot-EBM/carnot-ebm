"""TEST-TIME GOAL INDUCTION -- the actual unseen-game unlock: induce the goal with NO banked solve.

The agent-distance heuristic needs a TARGET. So far the target came from a banked win, which a
never-seen game does not have. This closes that gap with a fully first-contact pipeline (nothing
game-specific is pre-loaded):

  1. EXPLORE the real env (random legal actions) until a level-up is STUMBLED -> record the grid just
     BEFORE the winning action (the pre-win config). No banked solve used.
  2. IDENTIFY the agent (the general test-time method: the object that consistently translates).
  3. INDUCE THE GOAL OBJECT: at the pre-win config, the goal = the non-background, non-agent colour
     whose nearest cell is CLOSEST to the agent (the thing the agent reached to win). This is a goal
     COLOUR, so it GENERALIZES across layouts/levels: 'reach the colour-G object, wherever it is'.
  4. SOLVE via best_first_search with heuristic = agent-to-nearest-G-cell distance (the induced goal),
     searching for a real level-up. Compare to the banked-target heuristic + blind.

Validates the first-contact claim on the games where exploration can stumble a win (short solves).
Honest limit: a win that is never stumbled cannot have its goal induced this way -- deep-solve games
need curious/directed exploration or goal PERCEPTION, the remaining frontier. Proposer-free, zero quota.
"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_agi3_world_model import grid_of, frame_hash
from carnot.agentic.arc_agi3_live_adapter import _levels_completed, _game_action
from carnot.agentic.arc_world_model_dsl import _background


def _harness():
    spec = importlib.util.spec_from_file_location(
        "h", str(REPO / "scripts" / "experiments" / "arc3_m2_solve_objectdelta.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _kbd(frame, GameAction):
    av = [a for a in (getattr(frame, "available_actions", []) or [1, 2, 3, 4, 5]) if a not in (0, 6)]
    return [(a,) for a in av] or [(a,) for a in (1, 2, 3, 4, 5)]


def explore_for_win(arc, game, GameAction, budget, seed, episode_len=25):
    """EPISODIC random exploration until a level-up is stumbled: take short random walks from reset
    (many short episodes from a fresh start hit a short win far more often than one long drifting
    walk). Returns the grid BEFORE the winning action (the pre-win config), or None. NO banked solve."""
    rng = random.Random(seed)
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    steps = 0
    while steps < budget:
        f = env.reset()
        prev = np.asarray(grid_of(f)); lv = _levels_completed(f)
        for _ in range(episode_len):
            if steps >= budget:
                break
            if f is None or np.asarray(grid_of(f)).size == 0:
                break
            c = rng.choice(_kbd(f, GameAction))
            nf = env.step(_game_action(GameAction, c[0]), data=None)
            steps += 1
            if nf is None:
                break
            if _levels_completed(nf) > lv:
                return prev                             # the config from which the win fired
            prev = np.asarray(grid_of(nf)); lv = _levels_completed(nf); f = nf
    return None


def induce_goal_color(prewin, agent, H):
    """The goal object = the non-background, non-agent colour whose nearest cell is closest to the
    agent in the pre-win config (what the agent reached to win). Returns a colour G, or None."""
    if agent is None:
        return None
    bg = _background(prewin)
    ac = H._agent_centroid(prewin, agent)
    if ac is None:
        return None
    best = None
    for color in set(int(v) for v in np.unique(prewin)) - {bg, agent["color"]}:
        cells = np.argwhere(prewin == color)
        if len(cells) == 0:
            continue
        d = float(np.min(np.abs(cells[:, 0] - ac[0]) + np.abs(cells[:, 1] - ac[1])))
        if best is None or d < best[1]:
            best = (color, d)
    return best[0] if best else None


def agent_to_color_dist(grid, agent, G, H):
    ac = H._agent_centroid(grid, agent)
    cells = np.argwhere(np.asarray(grid) == G)
    if ac is None or len(cells) == 0:
        return 9999.0
    return float(np.min(np.abs(cells[:, 0] - ac[0]) + np.abs(cells[:, 1] - ac[1])))


def solve_with_induced_goal(arc, game, agent, G, GameAction, H, max_expansions=5000):
    """best_first_search over deepcopied real envs, guided by agent-to-colour-G distance (the INDUCED
    goal -- no banked target), searching for a real level-up."""
    import copy as _copy
    from carnot.agentic.arc_heuristic_search_over_verified_wm import best_first_search
    env0 = arc.make(game, scorecard_id=arc.open_scorecard())
    f0 = env0.reset()
    if f0 is None:
        return {"solved": False, "actions": None, "nodes": 0}
    start_level = _levels_completed(f0)
    reg = {}
    g0 = np.asarray(grid_of(f0)); gh0 = frame_hash(g0)
    reg[gh0] = (env0, f0)
    start = {"gh": gh0, "h": agent_to_color_dist(g0, agent, G, H), "level": start_level}

    def is_goal(s):
        return s["level"] > start_level

    def heuristic(s):
        return float(s["h"])

    def next_states(s):
        cur = reg.get(s["gh"])
        if cur is None:
            return []
        env, frame = cur
        out = []
        for c in _kbd(frame, GameAction):
            e2 = _copy.deepcopy(env)
            nf = e2.step(_game_action(GameAction, c[0]), data=None)
            if nf is None:
                continue
            g = np.asarray(grid_of(nf))
            if g.size == 0:
                continue
            gh = frame_hash(g)
            if gh not in reg:
                reg[gh] = (e2, nf)
            out.append((c, {"gh": gh, "h": agent_to_color_dist(g, agent, G, H),
                            "level": _levels_completed(nf)}))
        return out

    res = best_first_search(start, next_states=next_states, is_goal=is_goal,
                            heuristic=heuristic, max_expansions=max_expansions)
    return {"solved": res.solved, "actions": (len(res.actions) if res.solved else None),
            "nodes": res.nodes_expanded}


def run_game(game, explore_budget, max_exp, seed):
    from arcengine import GameAction
    H = _harness()
    arc = kit.offline_arcade()
    explore_trans, _ = __import__("carnot.agentic.arc_executable_world_model",
                                  fromlist=["collect_transitions"]).collect_transitions(game, n=150)
    agent = H.identify_agent(explore_trans)
    prewin = explore_for_win(arc, game, GameAction, explore_budget, seed)
    rec = {"game": game, "agent": (None if agent is None else {"color": agent["color"]}),
           "win_stumbled": prewin is not None}
    if prewin is None or agent is None:
        rec["verdict"] = "no_win_stumbled_or_no_agent_cannot_induce_goal"
        return rec
    G = induce_goal_color(prewin, agent, H)
    rec["induced_goal_color"] = (int(G) if G is not None else None)
    if G is None:
        rec["verdict"] = "win_stumbled_but_no_goal_object_induced"
        return rec
    sol = solve_with_induced_goal(arc, game, agent, G, GameAction, H, max_expansions=max_exp)
    rec["induced_goal_solve"] = sol
    rec["solves_first_contact"] = bool(sol["solved"])
    return rec


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="sp80,cn04,ar25,ka59")
    ap.add_argument("--explore-budget", type=int, default=400)
    ap.add_argument("--max-exp", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    games = args.games.split(",")
    print(f"== TEST-TIME GOAL INDUCTION (no banked solve) games={games} ==", flush=True)
    rows = []
    for g in games:
        r = run_game(g, args.explore_budget, args.max_exp, args.seed)
        rows.append(r)
        print(f"  [{g}] win_stumbled={r.get('win_stumbled')} agent={r.get('agent')} "
              f"induced_goal_color={r.get('induced_goal_color')} "
              f"first_contact_solve={r.get('induced_goal_solve')}", flush=True)
    solved = [r["game"] for r in rows if r.get("solves_first_contact")]
    verdict = ("complete_test_time_goal_induction_first_contact_solves_" + "_".join(solved) if solved
               else "complete_test_time_goal_induction_zero_first_contact_solves")
    out = {"experiment": "arc3_test_time_goal_induction", "games": games,
           "first_contact_solves": solved, "per_game": rows,
           "interpretation": (
               "Fully first-contact (NO banked solve): explore -> stumble a win -> induce the agent "
               "(general method) -> induce the GOAL OBJECT colour (what the agent reached) -> solve via "
               "best_first_search guided by agent-to-goal-colour distance. A first-contact solve = the "
               "unseen-game pipeline works end to end on that game. Honest limit: only games whose win "
               "is STUMBLE-ABLE within the explore budget can have a goal induced this way; deep-solve "
               "games need curious/directed exploration or goal PERCEPTION (the remaining frontier)."),
           "honest_verdict": verdict,
           "inference_substrate": "offline_arc_agi3_test_time_goal_induction_no_banked_solve"}
    (REPO / "results" / "arc3_test_time_goal_induction.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  first-contact solves: {solved}\n  -> {verdict}", flush=True)
    print("  wrote results/arc3_test_time_goal_induction.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
