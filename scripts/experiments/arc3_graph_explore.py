"""M1: Family-A GameGraph explorer — the no-induction path to the first ARC-AGI-3 level-up.

Plan: docs/research-notes/arc-agi3-agent-research-plan.md. Every prior policy (random, object_click,
codex-text, Gemma E2B/E4B direct+reasoning) scored 0 on vc33. The SOTA-3rd training-free method
(arXiv:2512.24156) shows a directed state-graph explorer with NO rule induction beats frontier LLMs.
This builds the cheapest credible instance: a persistent per-game GameGraph + structured, multi-
episode exploration that (a) tries untested actions at each state, (b) PRUNES deadly (GAME_OVER) and
no-effect actions (the verifier-as-pruner role, here as deterministic graph bookkeeping), (c)
navigates back to frontier states via BFS when the local state is exhausted, (d) persists across
episode RESETS so knowledge accumulates. M1 success = reach levels_completed >= 1 on vc33, offline.

Fully offline + air-gapped (SDK OperationMode.OFFLINE + local environment_files/). No LLM, no GPU.
Emits the same quota-gate artifact shape as arc3_offline_eval.py for comparability.

  .venv/bin/python scripts/experiments/arc3_graph_explore.py --game vc33-5430563c --episodes 10 --budget 250
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENVDIR = str(REPO / "environment_files")
import sys
sys.path.insert(0, str(REPO / "python"))
from carnot.agentic.arc_agi3_world_model import (  # noqa: E402
    GameGraph, grid_of, frame_hash, compute_grid_delta, objects, action_key)


def _candidate_akeys(grid, available, coarse_step=7):
    """Discrete action candidates from a state: object centroids + a coarse click grid (for click
    games), and each keyboard action. Object centroids are the perceptual-priority click targets."""
    cands = []
    av = list(available or [])
    if 6 in av:
        h, w = grid.shape
        seen = set()
        for (y, x) in objects(grid):                 # priority 1: object centroids
            k = (6, x, y)
            if k not in seen:
                seen.add(k); cands.append(k)
        for y in range(0, h, coarse_step):           # priority 2: coarse grid sweep
            for x in range(0, w, coarse_step):
                k = (6, x, y)
                if k not in seen:
                    seen.add(k); cands.append(k)
    for a in av:                                     # keyboard actions
        if a != 6 and a != 0:
            cands.append((a,))
    return cands


def _pick(graph, fh, untested, rng):
    """Among untested non-deadly candidates, prefer click-on-object (centroid, listed first) then a
    random untested one. Productive-action preference emerges as the graph fills (we deprioritize
    candidates whose action-int previously caused 0 change anywhere)."""
    # deprioritize action-ints that have only ever produced no-effect transitions
    no_effect_ints = set()
    eff_ints = set()
    for e in graph.edges.values():
        ai = e["akey"][0]
        (eff_ints if e["n_changed"] not in (0, None) else no_effect_ints).add(ai)
    good = [k for k in untested if k[0] in eff_ints or k[0] not in no_effect_ints]
    pool = good or untested
    return pool[0] if len(pool) <= 2 else rng.choice(pool)


def run(game="vc33-5430563c", episodes=10, budget=250, seed=0, write=True, persist=True):
    from arc_agi import Arcade
    from arc_agi.base import OperationMode
    from arcengine.enums import GameAction, GameState
    started = time.time()
    rng = random.Random(seed)
    by_id = {a.value: a for a in GameAction}
    arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=ENVDIR)
    info = {getattr(e, "game_id", None): (getattr(e, "baseline_actions", None) or []) for e in arc.get_environments()}
    baseline = info.get(game, [])
    graph = GameGraph(game)
    total_actions = 0
    max_levels = 0
    first_solve_action = None
    ep_summ = []

    for ep in range(episodes):
        env = arc.make(game)
        f = env.reset()
        prev = None  # (fh, akey, grid, levels)
        ep_actions = 0
        while total_actions < budget:
            grid = grid_of(f)
            fh = frame_hash(grid)
            graph.see_node(fh, f)
            cur_lv = int(getattr(f, "levels_completed", 0) or 0)
            if prev is not None:                        # record the transition we just took
                delta = compute_grid_delta(prev[2], grid)
                ld = cur_lv - prev[3]
                graph.record(prev[0], prev[1], fh, delta, ld, game_over=False)
                if cur_lv > max_levels:
                    max_levels = cur_lv
                    first_solve_action = first_solve_action or total_actions
            st = getattr(f, "state", None)
            if st in (GameState.WIN, GameState.GAME_OVER):
                break
            cands = _candidate_akeys(grid, getattr(f, "available_actions", []))
            untested = graph.untested(fh, cands)
            if untested:
                akey = _pick(graph, fh, untested, rng)
            else:                                       # local exhausted -> navigate to a frontier
                frontier = graph.frontier_states(lambda h, n: _candidate_akeys_for_node(graph, h))
                nav = graph.shortest_path_action(fh, frontier - {fh}) if frontier else None
                if nav is None:
                    break                               # stuck -> reset episode (graph persists)
                akey = nav
            a_int = akey[0]
            data = {"x": akey[1], "y": akey[2]} if a_int == 6 else None
            prev = (fh, akey, grid, cur_lv)
            f = env.step(by_id.get(a_int, GameAction.ACTION1), data=data)
            total_actions += 1
            ep_actions += 1
            if getattr(f, "state", None) == GameState.GAME_OVER:   # record deadly + reset
                ng = grid_of(f)
                graph.record(prev[0], prev[1], frame_hash(ng), compute_grid_delta(prev[2], ng), 0, True)
                break
            if getattr(f, "state", None) == GameState.WIN:
                max_levels = max(max_levels, int(getattr(f, "levels_completed", 0) or 0))
                break
        ep_summ.append({"episode": ep, "actions": ep_actions, "max_levels_so_far": max_levels,
                        "nodes": len(graph.nodes), "deadly": len(graph.deadly)})
        if total_actions >= budget:
            break

    graph.max_levels = max_levels
    if persist:
        graph.persist(REPO / "results" / f"world_model_{game.split('-')[0]}.json")
    win_levels = len(baseline)
    art = {
        "experiment": "arc3_graph_explore", "title": f"arc3_graph_explore_{game.split('-')[0]}",
        "honest_verdict": (f"complete: graph_explore_{game.split('-')[0]}_levels{max_levels}of{win_levels}"
                           f"_beats_floor={max_levels > 0}"),
        "inference_substrate": "offline_arc_agi3_graph_explore", "policy": "graph_explore_familyA",
        "game": game, "episodes_run": len(ep_summ), "total_actions": total_actions,
        "ACCURACY_total_levels_solved": max_levels, "ACCURACY_total_win_levels": win_levels,
        "first_solve_at_action": first_solve_action,
        "graph_nodes": len(graph.nodes), "graph_edges": len(graph.edges), "deadly_actions": len(graph.deadly),
        "n_transitions": len(graph.transition_store), "baseline_actions": baseline,
        "episode_summary": ep_summ, "no_llm_used": True, "no_gpu_used": True,
        "submitted_to_leaderboard": False, "random_seed": seed,
        "duration_s": round(time.time() - started, 1),
        "note": ("M1: Family-A persistent-graph explorer (no induction). M1 success = >=1 level. "
                 "If 0 after the budget, escalate to M1b (graft the deterministic-delta DSL inducer). "
                 "world_model_<game>.json persists the graph for cross-episode + Family-B reuse."),
    }
    if write:
        (REPO / "results" / f"arc3_graph_explore_{game.split('-')[0]}.json").write_text(
            json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"-> {art['honest_verdict']}")
    print(f"   actions={total_actions} episodes={len(ep_summ)} nodes={len(graph.nodes)} "
          f"deadly={len(graph.deadly)} first_solve_at={first_solve_action}")
    return art


def _candidate_akeys_for_node(graph, fh):
    """Candidates for a KNOWN node, reconstructed from its stored available_actions (for frontier
    detection without the live grid). Click games: we can't re-derive object centroids offline from
    the hash, so use the already-recorded edges' action space as the candidate proxy."""
    n = graph.nodes.get(fh, {})
    av = n.get("available_actions", [])
    # proxy candidates = keyboard actions + any click akeys already seen from this node's edges
    cands = [(a,) for a in av if a not in (0, 6)]
    if 6 in av:
        for ek, e in graph.edges.items():
            if e["from"] == fh and e["akey"][0] == 6:
                cands.append(tuple(e["akey"]))
    return cands


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", default="vc33-5430563c")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--budget", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    art = run(game=args.game, episodes=args.episodes, budget=args.budget, seed=args.seed)
    raise SystemExit(0 if art["ACCURACY_total_levels_solved"] > 0 else 1)
