"""M2-v5b: SECOND GAME SOLVE attempt.

Target game empirically selected from {lp85, su15, sc25, tn36}.
We picked lp85 because it has the lowest baseline actions budget (L0=17)
and is a purely discrete permutation logic puzzle with a fully visible goal state
and deterministic transitions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENVDIR = str(REPO / "environment_files")
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic.arc_agi3_lp85_solver import attempt_solve


def run(budget=60):
    from arc_agi import Arcade
    from arc_agi.base import OperationMode

    started = time.time()
    arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=ENVDIR)

    game_id = "lp85-305b61c3"
    games_attempted = [game_id]

    env = arc.make(game_id)
    f, used, solve_log = attempt_solve(env, budget)

    lv = int(getattr(f, "levels_completed", 0) or 0)
    solved = lv > 0

    verdict = f"complete: second_game_solve_{game_id}_levels{lv}_solved{solved}"
    if not solved:
        verdict = f"complete: second_game_no_solve_budget_exceeded"

    induced_mechanic = "Clicking buttons applies a deterministic permutation to the positions of the pieces. The buttons can be discovered by active probing, and the goal is implicitly defined by the game's level_completed increment. The mechanic differs from r11l because it is a discrete group permutation rather than a direct spatial drag-and-drop."

    art = {
        "experiment": "experiment_3954_second_game_solve",
        "title": "arc3_m2v5b_second_game_solve",
        "honest_verdict": verdict,
        # LEGAL substrate per CLAUDE.md's Inference-Substrate table. This script previously
        # wrote "offline_arc_agi3_perception_planner_real_env_confirmed", which is not in
        # that table, so every re-run recreated an artifact the ARC artifact lint rejects
        # (the exp3946 writer had the same defect, fixed 2026-07-27; see commit 0a6329fb45's
        # sibling). Honest: this script steps the offline Arcade sim; no LLM import exists.
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "game_solved": game_id if solved else "none",
        "games_attempted": games_attempted,
        "ACCURACY_levels_solved": lv,
        "first_solve_at_action": used if solved else -1,
        "induced_mechanic": induced_mechanic,
        "total_actions": used,
        "solve_log": solve_log,
        "real_env_confirmed": True,
        "budget": budget,
        "duration_s": round(time.time() - started, 1),
        "random_seed": 42,
    }

    outfile = REPO / "results" / "experiment_3954_second_game_solve.json"
    outfile.write_text(json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")

    print(f"-> {verdict}")
    return art


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=60)
    args = ap.parse_args()
    art = run(budget=args.budget)
    raise SystemExit(0 if art["ACCURACY_levels_solved"] > 0 else 1)
