"""Exploration-failure diagnostic for the hard-tail ARC-AGI-3 games.

For each game it measures, on the OFFLINE arcade (zero quota):
  (a) candidate-set size + composition: how many clicks-on-N-objects vs
      keyboard actions (action ids 1-5), and which action TYPES appear.
  (b) distinct states reached over ~N random/hybrid steps vs whether ANY
      level-up fires.
  (c) whether the explorer EXHAUSTS candidates (stuck repeating the same
      (state, action) pairs) or reaches diverse states but never the win.

GUARD: grid_of can return a NON-2D array on a game-over / degenerate frame
(0-D scalar or 1-D empty). We detect g.ndim != 2 and treat it as a dead
frame -> _warm reset, never hashing/segmenting a degenerate grid.

Classification per game is emitted as a hint; the decisive call is made in
the report. CANDIDATES = winning action TYPE absent from the candidate set;
STRUCTURE = depth-first/greedy misses but the state space is diverse and a
systematic BFS would find it; BUDGET = more random actions would find it.
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter

sys.path.insert(0, "python")

from carnot.agentic.arc_solver_kit import offline_arcade
from carnot.agentic.arc_agi3_world_model import frame_hash, grid_of
from carnot.agentic.arc_agi3_live_adapter import (
    _available_action_ids,
    _game_action,
    _game_over,
    _levels_completed,
)
from carnot.agentic.arc_graph_explore import rich_action_candidates

from arcengine import GameAction

GAMES = {
    "ls20": "ls20-9607627b",
    "wa30": "wa30-ee6fef47",
    "su15": "su15-1944f8ab",
    "tu93": "tu93-0768757b",
    "cn04": "cn04-2fe56bfb",
    "m0r0": "m0r0-492f87ba",
    "sk48": "sk48-d8078629",
}


def is_degenerate(frame) -> bool:
    """The GUARD: grid_of returns 0-D/1-D on game-over/degenerate frames."""
    try:
        g = grid_of(frame)
    except Exception:
        return True
    return g.ndim != 2 or g.size == 0


def warm(env):
    """Reset (no warmup-action burn; we want the pristine candidate set)."""
    return env.reset()


def candidate_composition(frame):
    """(a) size + composition of the rich candidate set at a given frame."""
    cands = rich_action_candidates(frame)
    types = Counter()
    for c in cands:
        if c.action_id == 6:
            types["click"] += 1
        else:
            types[f"key{c.action_id}"] += 1
    return len(cands), dict(types)


def explore(env, gid, steps=800, seed=0, max_actions_per_episode=140):
    """Hybrid random/greedy explorer (the standing graph_explore policy):
    prefer a globally-untested (state, action) pair; else random. Restart on
    game-over / dead frame. Instruments distinct states, level-ups, and
    whether candidates EXHAUST (every fresh (state,action) pair tried)."""
    rng = random.Random(seed)
    global_tested: set = set()           # (state_hash, action_key) tried ever
    distinct_states: set = set()
    level_ups = 0
    best_level = 0
    steps_done = 0
    episodes = 0
    degenerate_hits = 0
    none_step_hits = 0
    gameovers = 0
    fresh_available_log = []             # how many fresh candidates at each step
    n_changed_zero = 0                   # no-op (frame unchanged) actions
    transitions = 0
    distinct_edges: set = set()          # (state_hash, action_key) that CHANGED state

    while steps_done < steps:
        f = warm(env)
        episodes += 1
        if is_degenerate(f):
            degenerate_hits += 1
            continue
        cur = frame_hash(grid_of(f))
        distinct_states.add(cur)
        ep_steps = 0
        while steps_done < steps and ep_steps < max_actions_per_episode:
            cands = rich_action_candidates(f)
            if not cands:
                break
            keyed = {c.key: c for c in cands}
            fresh = [c for c in cands if (cur, c.key) not in global_tested]
            fresh_available_log.append(len(fresh))
            if fresh:
                sel = fresh[0]                    # greedy: take an untested action
            else:
                sel = cands[rng.randrange(len(cands))]   # all tested -> random
            global_tested.add((cur, sel.key))
            nf = env.step(_game_action(GameAction, sel.action_id), data=sel.data,
                          reasoning={"policy": "diag_hybrid_explore"})
            steps_done += 1
            ep_steps += 1
            if nf is None:
                none_step_hits += 1
                break
            if is_degenerate(nf):
                degenerate_hits += 1
                break
            lvl = _levels_completed(nf)
            if lvl > best_level:
                level_ups += 1
                best_level = lvl
            nh = frame_hash(grid_of(nf))
            if nh != cur:
                transitions += 1
                distinct_edges.add((cur, sel.key))
            else:
                n_changed_zero += 1
            distinct_states.add(nh)
            if _game_over(nf):
                gameovers += 1
                break
            f = nf
            cur = nh

    # EXHAUSTION metric: fraction of steps where NO fresh (state,action) pair
    # was available (everything already tested) -> the explorer is stuck
    # cycling. High => candidate-exhaustion (STRUCTURE/CANDIDATES); low =>
    # still expanding (BUDGET).
    n_log = len(fresh_available_log)
    steps_with_no_fresh = sum(1 for x in fresh_available_log if x == 0)
    exhaust_frac = (steps_with_no_fresh / n_log) if n_log else 0.0

    return {
        "steps_done": steps_done,
        "episodes": episodes,
        "distinct_states": len(distinct_states),
        "distinct_changing_edges": len(distinct_edges),
        "level_ups": level_ups,
        "best_level": best_level,
        "global_tested_pairs": len(global_tested),
        "transitions_state_changed": transitions,
        "noop_actions_state_unchanged": n_changed_zero,
        "gameovers": gameovers,
        "degenerate_frame_hits": degenerate_hits,
        "none_step_hits": none_step_hits,
        "exhaustion_fraction": round(exhaust_frac, 4),
        "mean_fresh_candidates_per_step": round(
            sum(fresh_available_log) / n_log, 3) if n_log else 0.0,
    }


def diagnose_game(short, gid, steps=800):
    ar = offline_arcade()
    env = ar.make(gid, save_recording=False, include_frame_data=True)
    f0 = env.reset()
    # (a) candidate composition at the initial state
    init_size, init_types = candidate_composition(f0)
    init_action_ids = _available_action_ids(f0)
    init_shape = list(grid_of(f0).shape)

    t0 = time.time()
    exp = explore(env, gid, steps=steps)
    exp["wall_s"] = round(time.time() - t0, 2)

    # classification hint
    if exp["level_ups"] > 0:
        cls = "SOLVED-IN-WINDOW"
    elif exp["distinct_states"] <= 3 and exp["exhaustion_fraction"] > 0.5:
        cls = "CANDIDATES?"   # tiny state space + exhausted -> winning type likely missing
    elif exp["distinct_states"] >= 50:
        cls = "STRUCTURE?"    # diverse states, never win -> systematic/BFS needed (or deep)
    else:
        cls = "BUDGET-or-STRUCTURE?"

    return {
        "game": short,
        "game_id": gid,
        "init_candidate_size": init_size,
        "init_candidate_composition": init_types,
        "init_available_action_ids": init_action_ids,
        "init_grid_shape": init_shape,
        "explore": exp,
        "classification_hint": cls,
    }


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    only = sys.argv[2] if len(sys.argv) > 2 else None
    out = {}
    for short, gid in GAMES.items():
        if only and short != only:
            continue
        print(f"### {short} ({gid}) ...", flush=True)
        try:
            res = diagnose_game(short, gid, steps=steps)
        except Exception as e:
            import traceback
            res = {"game": short, "game_id": gid, "error": f"{type(e).__name__}: {e}",
                   "trace": traceback.format_exc()[-800:]}
        out[short] = res
        print(json.dumps(res, indent=2), flush=True)
    with open("results/diag_hardtail_exploration.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("WROTE results/diag_hardtail_exploration.json", flush=True)


if __name__ == "__main__":
    main()
