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


def identify_movers(transitions, top_k=3, min_moves=3):
    """Generalizes identify_agent: the top-K connected objects that translate under actions (most-
    frequent first), filtered to those that moved >= min_moves times (drops one-off / HUD-flicker
    movers). For a push game this returns the agent AND the pushed block -- the objects whose JOINT
    positions define the game state. Returns a list of {'color','shape_sig'}."""
    from collections import Counter, defaultdict
    from carnot.agentic.arc_world_model_dsl import _color_components, _object_shape_sig, _background
    tally, sizes = Counter(), defaultdict(list)
    for t in transitions:
        if t.level_after != t.level_before or np.array_equal(t.grid, t.next_grid):
            continue
        bg = _background(t.grid)
        for color in set(int(v) for v in np.unique(t.grid)) - {bg}:
            cs = set(_color_components(t.grid, color)); cs2 = set(_color_components(t.next_grid, color))
            gone = [c for c in cs if c not in cs2]; came = [c for c in cs2 if c not in cs]
            if len(gone) == 1 and len(came) == 1 and len(gone[0]) == len(came[0]):
                a, b = gone[0], came[0]
                dy = min(y for y, _ in b) - min(y for y, _ in a)
                dx = min(x for _, x in b) - min(x for _, x in a)
                if (dy, dx) != (0, 0) and {(y + dy, x + dx) for y, x in a} == set(b):
                    sig = (color, _object_shape_sig(a)); tally[sig] += 1; sizes[sig].append(len(a))
    ranked = sorted([s for s in tally if tally[s] >= min_moves],
                    key=lambda s: (tally[s], -float(np.mean(sizes[s]))), reverse=True)[:top_k]
    return [{"color": c, "shape_sig": sig} for (c, sig) in ranked]


def _movers_key(grid, movers, H):
    """A state key = the centroid of EACH tracked mover (agent + pushed objects), so two states with
    the same agent position but a differently-placed pushed block are DISTINCT (the fix for push games
    that agent-position-only coverage wrongly merged)."""
    parts = []
    for m in movers:
        ac = H._agent_centroid(grid, m)
        parts.append(None if ac is None else (int(round(ac[0])), int(round(ac[1]))))
    return tuple(parts)


def _pieces_key(grid, min_size=4, max_size=30):
    """A state key = the positions of all PIECE-sized objects (size band excludes big walls/frames).
    Captures the agent AND a pushable block WITHOUT having to observe the block move first (random
    explore rarely pushes it, so it never registers as a 'mover'). Static pieces are constant -> no
    state-space blowup; only objects that actually move add dimensions."""
    from carnot.agentic.arc_world_model_dsl import _color_components, _background
    g = np.asarray(grid)
    bg = _background(g)
    parts = []
    for color in set(int(v) for v in np.unique(g)) - {bg}:
        for comp in _color_components(g, color):
            if min_size <= len(comp) <= max_size:
                ys = [y for y, _ in comp]; xs = [x for _, x in comp]
                parts.append((color, int(round(float(np.mean(ys)))), int(round(float(np.mean(xs))))))
    return tuple(sorted(parts))


def curious_explore_for_win(arc, game, agent, GameAction, H, max_expansions=8000, movers=None,
                            key_mode="agent"):
    """CURIOUS/DIRECTED exploration: BFS over the reachable AGENT-POSITION graph (dedup on the agent's
    centroid, not the full grid), trying every action at each newly-reached position, until a level-up.
    For navigation/movement games the agent position is the state variable that matters, so covering
    reachable positions is ~O(#positions) instead of the exponential O(branching^depth) of random walks
    -- it reaches deep wins random play cannot stumble. Returns the pre-win grid, or None. NO banked
    solve. (Falls back to a coarse grid hash when no agent / agent centroid is unavailable.)"""
    import copy as _copy
    from collections import deque
    env0 = arc.make(game, scorecard_id=arc.open_scorecard())
    f0 = env0.reset()
    if f0 is None:
        return None
    start_level = _levels_completed(f0)

    def key(grid):
        # 'agent' (default): minimal key = agent position -- right for NAVIGATION games (cn04/sp80/ar25),
        #   which over-fragment under a multi-object key. 'pieces': all piece-sized objects -- needed for
        #   PUSH games (ka59) where the pushed block's position is part of the state, but it explodes the
        #   state space of multi-object navigation puzzles, so it is opt-in (the granularity is genuinely
        #   game-dependent; the principled fix is to ESCALATE granularity only when 'agent' search dries up).
        if key_mode == "pieces":
            pk = _pieces_key(grid)
            if pk:
                return ("pieces",) + pk
        if agent is not None:
            ac = H._agent_centroid(grid, agent)
            if ac is not None:
                return ("p", int(round(ac[0])), int(round(ac[1])))
        return ("g", frame_hash(np.asarray(grid)))

    frontier = deque([(env0, f0)])
    visited = {key(grid_of(f0))}
    expansions = 0
    while frontier and expansions < max_expansions:
        env, f = frontier.popleft()
        grid = np.asarray(grid_of(f))
        if grid.size == 0:
            continue
        for c in _kbd(f, GameAction):
            e2 = _copy.deepcopy(env)
            nf = e2.step(_game_action(GameAction, c[0]), data=None)
            expansions += 1
            if nf is None:
                continue
            if _levels_completed(nf) > start_level:
                return grid                             # the pre-win config we acted from
            g = np.asarray(grid_of(nf))
            if g.size == 0:
                continue
            k = key(g)
            if k not in visited:
                visited.add(k); frontier.append((e2, nf))
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


def run_game(game, explore_budget, max_exp, seed, curious=True):
    from arcengine import GameAction
    H = _harness()
    arc = kit.offline_arcade()
    explore_trans, _ = __import__("carnot.agentic.arc_executable_world_model",
                                  fromlist=["collect_transitions"]).collect_transitions(game, n=150)
    agent = H.identify_agent(explore_trans)
    movers = identify_movers(explore_trans)
    key_used = None
    if curious:
        # ADAPTIVE granularity: try the minimal AGENT-position key first (right for navigation, no
        # over-fragmentation); only if it dries up, ESCALATE to the multi-object PIECES key (push games).
        prewin = curious_explore_for_win(arc, game, agent, GameAction, H,
                                         max_expansions=explore_budget, key_mode="agent")
        key_used = "agent"
        if prewin is None:
            prewin = curious_explore_for_win(arc, game, agent, GameAction, H,
                                             max_expansions=explore_budget, key_mode="pieces")
            key_used = "pieces_escalated"
    else:
        prewin = explore_for_win(arc, game, GameAction, explore_budget, seed)
    rec = {"game": game, "explore_mode": ("curious_adaptive_bfs" if curious else "random"),
           "state_key_used": key_used,
           "agent": (None if agent is None else {"color": agent["color"]}),
           "movers": [m["color"] for m in movers], "win_stumbled": prewin is not None}
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
    ap.add_argument("--explore-budget", type=int, default=8000)
    ap.add_argument("--max-exp", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random", action="store_true", help="use random exploration instead of curious")
    args = ap.parse_args()
    games = args.games.split(",")
    mode = "RANDOM" if args.random else "CURIOUS (agent-position BFS)"
    print(f"== TEST-TIME GOAL INDUCTION (no banked solve) explore={mode} games={games} ==", flush=True)
    rows = []
    for g in games:
        r = run_game(g, args.explore_budget, args.max_exp, args.seed, curious=not args.random)
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
