"""SMOKE (north-star stage-3 grounding): random-agent baseline on LIVE ARC-AGI-3.

Confirms the real env loop works end-to-end (Arcade.make -> reset -> step -> state/
levels) against the live anonymous-key API, and establishes the RANDOM FLOOR: how many
levels a random agent solves and at what action cost. This is the floor the Carnot
verifier-routed / action-pruner harness must beat on BOTH north-star axes:
  ACCURACY   = levels_completed (random ~0 expected)
  EFFICIENCY = actions used vs EnvironmentInfo.baseline_actions[level] (reference count)

NOT a submission (operator-only) — anonymous play + local scoring only.

  .venv/bin/python scripts/experiments/arc_agi3_random_baseline_smoke.py --games 5 --budget 40
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "results" / "arc_agi3_random_baseline_smoke.json"


def _play_random(env, info, *, budget, rng):
    from arcengine.enums import GameAction, GameState
    f = env.reset()
    by_id = {a.value: a for a in GameAction}
    actions = 0
    max_levels = int(getattr(f, "levels_completed", 0) or 0)
    state = str(getattr(f, "state", "?"))
    while f is not None and actions < budget:
        st = getattr(f, "state", None)
        if st in (GameState.WIN, GameState.GAME_OVER):
            state = str(st)
            break
        avail = list(getattr(f, "available_actions", []) or [])
        if not avail:
            break
        a_int = rng.choice(avail)
        action = by_id.get(a_int, GameAction.ACTION1)
        # click-type actions (ACTION6) need grid coords; others take no data.
        data = {"x": rng.randrange(64), "y": rng.randrange(64)} if a_int == 6 else None
        f = env.step(action, data=data)
        actions += 1
        if f is not None:
            max_levels = max(max_levels, int(getattr(f, "levels_completed", 0) or 0))
            state = str(getattr(f, "state", state))
    return {"actions_used": actions, "max_levels_completed": max_levels, "final_state": state}


def run(n_games=5, budget=40, seed=0, write=True):
    from arc_agi import Arcade
    started = time.time()
    rng = random.Random(seed)
    arc = Arcade(arc_api_key="")
    envs = arc.get_environments()
    infos = {getattr(e, "game_id", None): e for e in envs}
    picks = list(infos)[:n_games]

    per_game = []
    for gid in picks:
        info = infos[gid]
        base = getattr(info, "baseline_actions", None) or []
        env = arc.make(gid, seed=seed)
        if env is None:
            per_game.append({"game_id": gid, "error": "make_returned_none"})
            continue
        try:
            r = _play_random(env, info, budget=budget, rng=rng)
        except Exception as e:  # live API hiccup on one game shouldn't kill the smoke
            per_game.append({"game_id": gid, "error": f"{type(e).__name__}: {e}"})
            continue
        r.update({"game_id": gid, "tags": getattr(info, "tags", None),
                  "win_levels": getattr(info, "win_levels", None),
                  "n_levels": len(base), "baseline_actions_level0": base[0] if base else None,
                  "action_budget": budget})
        per_game.append(r)
        print(f"[{gid}] levels={r['max_levels_completed']} actions={r['actions_used']}/{budget} "
              f"state={r['final_state']} baseline_L0={base[0] if base else '?'}", flush=True)

    played = [g for g in per_game if "error" not in g]
    total_levels = sum(g["max_levels_completed"] for g in played)
    env_ok = len(played) > 0
    verdict = (f"complete: arc_agi3_random_baseline_{'ENV_LOOP_OK' if env_ok else 'ENV_LOOP_FAILED'}"
               f"_games{len(played)}_randomlevels{total_levels}_floor_established")
    art = {
        "experiment": "arc_agi3_random_baseline_smoke",
        "title": "arc_agi3_live_random_agent_floor",
        "honest_verdict": verdict,
        "inference_substrate": "live_arc_agi3_remote_api_random_agent",
        "run_date": "2026-06-08", "random_seed": seed,
        "n_games_attempted": len(picks), "n_games_played": len(played),
        "total_levels_solved_random": total_levels,
        "per_game": per_game,
        "no_llm_used": True, "no_verifier_used": True, "submitted_to_leaderboard": False,
        "duration_s": round(time.time() - started, 1),
        "interpretation": (
            "Grounds the live env loop + sets the RANDOM FLOOR. Random agent levels_solved "
            "(expected ~0) and action cost are the baseline the Carnot verifier-routed / "
            "action-pruner harness must beat: ACCURACY = more levels, EFFICIENCY = fewer "
            "actions vs baseline_actions. This is the first REAL (non-synthetic) ARC-AGI-3 "
            "interaction; prior exp1165/3919/3929 were synthetic."
        ),
    }
    if write:
        OUT.write_text(json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
    return art


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=5)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    art = run(n_games=args.games, budget=args.budget, seed=args.seed)
    print(f"\n-> {art['honest_verdict']}")
    print(f"   games_played={art['n_games_played']} random_levels_solved={art['total_levels_solved_random']} "
          f"dur={art['duration_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
