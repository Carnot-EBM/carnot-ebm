"""GROUNDING (north-star stage-3 prep): characterize all 25 live ARC-AGI-3 games.

Descriptive reference for the eventual verifier-routed harness: per-game action space,
level count, baseline-action distribution, and reset-frame grid structure (active cells,
palette). Ranks games by a simple "start-here" difficulty proxy so the first real
verifier-agent runs target the most tractable games. NOT a submission, no LLM, no
verifier scoring -- pure characterization via reset() on the anonymous-key API.

  .venv/bin/python scripts/experiments/arc_agi3_game_characterization.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "results" / "arc_agi3_game_characterization.json"


def _grid_stats(frame_obj):
    """Active-cell + palette stats from a reset frame (frame is (stack,H,W) int8)."""
    try:
        import numpy as np
        arr = np.array(frame_obj.frame)
        if arr.ndim == 3:
            arr = arr[-1]  # last frame in the stack
        h, w = arr.shape
        vals, counts = np.unique(arr, return_counts=True)
        bg = int(vals[counts.argmax()])
        active = int((arr != bg).sum())
        return {"h": h, "w": w, "palette_size": int(len(vals)),
                "background_color": bg, "active_cells": active,
                "active_fraction": round(active / (h * w), 4)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def run(write=True):
    from arc_agi import Arcade
    started = time.time()
    arc = Arcade(arc_api_key="")
    envs = arc.get_environments()
    games = []
    for e in envs:
        gid = getattr(e, "game_id", None)
        base = list(getattr(e, "baseline_actions", None) or [])
        rec = {
            "game_id": gid, "title": getattr(e, "title", None),
            "tags": getattr(e, "tags", None),
            "n_levels": len(base),
            "baseline_actions": base,
            "baseline_L0": base[0] if base else None,
            "baseline_min": min(base) if base else None,
            "baseline_max": max(base) if base else None,
            "baseline_total": sum(base) if base else None,
        }
        try:
            env = arc.make(gid, seed=0)
            f = env.reset() if env is not None else None
            if f is not None:
                rec["reset_available_actions"] = list(getattr(f, "available_actions", []) or [])
                rec["win_levels"] = getattr(f, "win_levels", None)
                rec["reset_grid"] = _grid_stats(f)
        except Exception as ex:
            rec["reset_error"] = f"{type(ex).__name__}: {ex}"
        games.append(rec)
        print(f"[{gid}] tags={rec['tags']} levels={rec['n_levels']} "
              f"baseL0={rec['baseline_L0']} avail={rec.get('reset_available_actions')} "
              f"grid={rec.get('reset_grid',{}).get('h')}x{rec.get('reset_grid',{}).get('w')}", flush=True)

    # "start-here" ranking: fewer baseline actions for level 0 + smaller action space =
    # more tractable first target for the verifier-routed agent.
    def _difficulty(g):
        return (g.get("baseline_L0") or 1e9, len(g.get("reset_available_actions") or [1] * 9))
    ranked = sorted([g for g in games if g.get("baseline_L0")], key=_difficulty)
    start_here = [{"game_id": g["game_id"], "baseline_L0": g["baseline_L0"],
                   "n_actions": len(g.get("reset_available_actions") or []),
                   "tags": g["tags"]} for g in ranked[:8]]

    action_space_dist = {}
    for g in games:
        key = ",".join(str(a) for a in (g.get("reset_available_actions") or []))
        action_space_dist[key] = action_space_dist.get(key, 0) + 1

    art = {
        "experiment": "arc_agi3_game_characterization",
        "title": "arc_agi3_25_game_reference",
        "honest_verdict": f"complete: arc_agi3_characterized_{len(games)}_games_start_here_ranked",
        "inference_substrate": "live_arc_agi3_remote_api_characterization",
        "run_date": "2026-06-08",
        "n_games": len(games),
        "reset_action_space_distribution": action_space_dist,
        "start_here_top8": start_here,
        "games": games,
        "no_llm_used": True, "no_verifier_used": True, "submitted_to_leaderboard": False,
        "duration_s": round(time.time() - started, 1),
        "interpretation": (
            "Reference for the verifier-routed harness build. start_here_top8 = the most "
            "tractable first targets (lowest level-0 baseline action count + smallest action "
            "space). reset_action_space_distribution shows how interactive the games are "
            "(single-action = pure-click; multi-action = keyboard+click). The verifier-agent "
            "(router + action-pruner) should be developed against start_here games first, "
            "measured vs the random floor (results/arc_agi3_random_baseline_smoke.json)."
        ),
    }
    if write:
        OUT.write_text(json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
    return art


if __name__ == "__main__":
    a = run()
    print(f"\n-> {a['honest_verdict']}")
    print(f"   action_space_distribution: {a['reset_action_space_distribution']}")
    print("   START-HERE top 8 (easiest first targets):")
    for g in a["start_here_top8"]:
        print(f"     {g['game_id']}  baseL0={g['baseline_L0']}  n_actions={g['n_actions']}  tags={g['tags']}")
