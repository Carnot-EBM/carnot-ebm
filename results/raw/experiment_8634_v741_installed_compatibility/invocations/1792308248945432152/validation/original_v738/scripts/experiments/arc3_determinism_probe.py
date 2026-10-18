"""Determinism probe: is each ARC-AGI-3 game grid-MARKOV (visible grid + action determines next grid)
or HIDDEN-STATE (same visible grid + same action -> different next grid)? This decides whether the
Family-B world-model can be selected by an EXACT-match reproduction check (deterministic games -> a
trained soft ARC-energy is premature) or REQUIRES a soft/learned energy that ranks near-misses
(hidden-state games -> a trained ARC-energy is the necessary enabler, worth building earlier).

Two measurements per game, fully offline (no LLM/GPU):
  1) REPLAY DETERMINISM (from reset): replay the SAME random action sequence twice from two fresh
     resets; if the visible-state sequences match, the env is deterministic-from-reset (no RNG). This
     is the easy baseline; these puzzle envs are expected to pass.
  2) GRID-MARKOV / HIDDEN-STATE: over a multi-episode exploration's transition store, group every
     transition by (visible_frame_hash, action). If any (frame_hash, action) key sampled >=2 times
     (reached via different histories) maps to >1 distinct next frame_hash, the visible grid does NOT
     determine the dynamics -> HIDDEN STATE. nondeterminism_rate = nondet_keys / revisited_keys.
     This is the plan's own top-risk detector ("same (frame_hash, action) -> different deltas").

  .venv/bin/python scripts/experiments/arc3_determinism_probe.py --budget 1500 --episodes 40
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENVDIR = str(REPO / "environment_files")
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))
from carnot.agentic.arc_agi3_world_model import (  # noqa: E402
    GameGraph, grid_of, frame_hash, compute_grid_delta)
import arc3_graph_explore as gx  # noqa: E402


def _replay(arc, game, seq, by_id, GameAction, GameState):
    env = arc.make(game)
    f = env.reset()
    states = [frame_hash(grid_of(f))]
    for akey in seq:
        a_int = akey[0]
        data = {"x": akey[1], "y": akey[2]} if a_int == 6 else None
        f = env.step(by_id.get(a_int, GameAction.ACTION1), data=data)
        states.append(frame_hash(grid_of(f)))
        if getattr(f, "state", None) in (GameState.WIN, GameState.GAME_OVER):
            break
    return states


def _explore_collect(arc, game, budget, episodes, rng, GameAction, GameState):
    """Multi-episode exploration accumulating one GameGraph; return it (transition_store populated)."""
    by_id = {a.value: a for a in GameAction}
    graph = GameGraph(game)
    total = 0
    for _ in range(episodes):
        env = arc.make(game)
        f = env.reset()
        prev = None
        while total < budget:
            grid = grid_of(f)
            fh = frame_hash(grid)
            graph.see_node(fh, f)
            if prev is not None:
                graph.record(prev[0], prev[1], fh, compute_grid_delta(prev[2], grid), 0, False)
            if getattr(f, "state", None) in (GameState.WIN, GameState.GAME_OVER):
                break
            cands = gx._candidate_akeys(grid, getattr(f, "available_actions", []))
            untested = graph.untested(fh, cands)
            # bias toward REVISITING tried (s,a) ~30% of the time to sample determinism; else explore
            tried_here = [k for k in cands if graph.tried(fh, k) and not graph.is_deadly(fh, k)]
            if tried_here and rng.random() < 0.3:
                akey = rng.choice(tried_here)
            elif untested:
                akey = gx._pick(graph, fh, untested, rng)
            elif tried_here:
                akey = rng.choice(tried_here)
            else:
                break
            a_int = akey[0]
            data = {"x": akey[1], "y": akey[2]} if a_int == 6 else None
            prev = (fh, akey, grid)
            f = env.step(by_id.get(a_int, GameAction.ACTION1), data=data)
            total += 1
            if getattr(f, "state", None) == GameState.GAME_OVER:
                ng = grid_of(f)
                graph.record(prev[0], prev[1], frame_hash(ng), compute_grid_delta(prev[2], ng), 0, True)
                break
        if total >= budget:
            break
    return graph


def run(games=None, budget=1500, episodes=40, seed=0, n_replays=12, replay_len=8, write=True):
    from arc_agi import Arcade
    from arc_agi.base import OperationMode
    from arcengine.enums import GameAction, GameState
    started = time.time()
    rng = random.Random(seed)
    by_id = {a.value: a for a in GameAction}
    arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=ENVDIR)
    all_ids = sorted(getattr(e, "game_id", None) for e in arc.get_environments())
    if games:
        sel = [g for g in all_ids if g.split("-")[0] in set(games)]
    else:
        sel = all_ids
    per_game = []
    for game in sel:
        short = game.split("-")[0]
        # (1) replay determinism from reset
        env0 = arc.make(game); f0 = env0.reset()
        start_cands = gx._candidate_akeys(grid_of(f0), getattr(f0, "available_actions", []))
        replay_mismatch = 0
        replay_pairs = 0
        for _ in range(n_replays):
            seq = [rng.choice(start_cands) for _ in range(replay_len)] if start_cands else []
            a = _replay(arc, game, seq, by_id, GameAction, GameState)
            b = _replay(arc, game, seq, by_id, GameAction, GameState)
            for sa, sb in zip(a, b):
                replay_pairs += 1
                if sa != sb:
                    replay_mismatch += 1
        # (2) grid-Markov / hidden-state over an exploration's transition store
        graph = _explore_collect(arc, game, budget, episodes, rng, GameAction, GameState)
        outcomes = defaultdict(set)
        samples = defaultdict(int)
        for t in graph.transition_store:
            key = (t["s"], tuple(t["akey"]))
            outcomes[key].add(t["s2"])
            samples[key] += 1
        revisited = [k for k, n in samples.items() if n >= 2]
        nondet = [k for k in revisited if len(outcomes[k]) > 1]
        rate = round(len(nondet) / len(revisited), 4) if revisited else None
        per_game.append({
            "game": short, "n_transitions": len(graph.transition_store),
            "distinct_state_action_keys": len(samples),
            "revisited_keys": len(revisited), "nondeterministic_keys": len(nondet),
            "nondeterminism_rate_among_revisited": rate,
            "replay_step_pairs": replay_pairs, "replay_mismatches": replay_mismatch,
            "deterministic_from_reset": replay_mismatch == 0,
        })
        print(f"  {short:6s} revisited={len(revisited):5d} nondet={len(nondet):4d} "
              f"rate={rate} | replay_det={replay_mismatch == 0} (mism {replay_mismatch}/{replay_pairs})",
              flush=True)

    # aggregate
    rated = [g for g in per_game if g["nondeterminism_rate_among_revisited"] is not None]
    hidden = [g for g in rated if g["nondeterminism_rate_among_revisited"] > 0.01]
    all_det_reset = all(g["deterministic_from_reset"] for g in per_game)
    macro = round(sum(g["nondeterminism_rate_among_revisited"] for g in rated) / len(rated), 4) if rated else None
    verdict = ("complete: determinism_probe_hidden_state_games={}of{}_macro_nondet_rate={}"
               "_det_from_reset={}").format(len(hidden), len(rated), macro, all_det_reset)
    art = {
        "experiment": "arc3_determinism_probe", "title": "arc3_determinism_probe",
        "honest_verdict": verdict,
        "inference_substrate": "offline_arc_agi3_determinism_analysis",
        "games_measured": len(per_game), "games_with_revisit_data": len(rated),
        "hidden_state_games": [g["game"] for g in hidden],
        "n_hidden_state_games": len(hidden),
        "macro_nondeterminism_rate": macro,
        "all_deterministic_from_reset": all_det_reset,
        "budget_per_game": budget, "episodes_per_game": episodes,
        "n_replays": n_replays, "replay_len": replay_len, "random_seed": seed,
        "no_llm_used": True, "no_gpu_used": True, "submitted_to_leaderboard": False,
        "duration_s": round(time.time() - started, 1), "per_game": per_game,
        "interpretation": (
            "nondeterminism_rate>0 among REVISITED (frame_hash,action) keys = the visible grid does NOT "
            "determine dynamics = HIDDEN STATE = exact-match M2 model selection fails = a trained SOFT "
            "ARC-energy (rank near-misses) is the necessary enabler, worth building earlier. rate~0 = "
            "grid-Markov = exact-match reproduction suffices for M2 model selection = trained ARC-energy "
            "is premature for first-solve (still useful later for planning into unvisited states)."),
    }
    if write:
        (REPO / "results" / "arc3_determinism_probe.json").write_text(
            json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"\n-> {verdict}")
    print(f"   hidden-state games (nondet>1%): {len(hidden)}/{len(rated)} | macro rate {macro} | "
          f"det-from-reset all={all_det_reset}")
    return art


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", default="", help="comma-separated short ids; empty = all 25")
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    gl = [g.strip() for g in args.games.split(",") if g.strip()] or None
    run(games=gl, budget=args.budget, episodes=args.episodes, seed=args.seed)
