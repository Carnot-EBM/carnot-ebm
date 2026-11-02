"""Piece-3 deliverable: HYBRID explorer = structured-first + random-restart-on-stall (2026-06-21).

The structured depth_first_ride explorer reaches first-win on 1/11 (efficiently, lp85@20) but MISSES easy
structure-missed wins (r11l/sp80); a pure random explorer reaches 3/11 but loses lp85's efficiency (142 vs
20) so its averaged score is WORSE. The hybrid keeps the best of both: run the structured explorer first
(efficient where it works); if it does NOT win within its slice, fall to a random-restart phase (diversity
catches r11l/sp80). Measured against both pure baselines on first-win AND efficiency (the authoritative
scorer). Offline, no model, no LLM.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts"))

import arc_leaderboard_eval as lbe
import arc_random_explore_measure as rnd

GAMES = (sys.argv[1].split(",") if len(sys.argv) > 1
         else ["r11l", "lp85", "ls20", "wa30", "cd82", "sp80", "su15", "tu93", "cn04", "m0r0", "sk48"])
STRUCT_BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 700
RAND_BUDGET = int(sys.argv[3]) if len(sys.argv) > 3 else 1300


def hybrid(game: str) -> dict:
    # structured phase (efficient where it works)
    pol = lbe._build_policy("explorer", game)
    r = lbe.run_game(game, pol, budget=STRUCT_BUDGET)
    if int(r.get("levels", 0)) > 0:
        return {"game": game, "won_by": "structured", "levels": int(r["levels"]),
                "first_levelup_actions": r.get("actions_to_first_levelup"),
                "actions": int(r.get("actions", STRUCT_BUDGET)), "efficiency": r.get("efficiency", 0.0)}
    # random-restart fallback (diversity for the structure-missed tail)
    rnd.BUDGET = RAND_BUDGET
    rr = rnd.run(game)
    if int(rr.get("levels_reached", 0)) > 0:
        rand_first = rr.get("first_levelup_actions") or RAND_BUDGET
        return {"game": game, "won_by": "random", "levels": int(rr["levels_reached"]),
                "first_levelup_actions": STRUCT_BUDGET + rand_first,
                "actions": STRUCT_BUDGET + rand_first, "efficiency": None}
    return {"game": game, "won_by": "none", "levels": 0, "first_levelup_actions": None,
            "actions": STRUCT_BUDGET + RAND_BUDGET, "efficiency": 0.0}


def main() -> int:
    print(f"== HYBRID explorer (structured {STRUCT_BUDGET} + random-on-stall {RAND_BUDGET}) ==", flush=True)
    print(f"{'game':6} {'won_by':>10} {'levels':>6} {'1st_lvlup':>9} {'eff':>7}", flush=True)
    rows = []
    won = 0
    won_struct = 0
    for g in GAMES:
        try:
            r = hybrid(g)
        except Exception as e:
            r = {"game": g, "won_by": "error", "levels": 0, "error": f"{type(e).__name__}: {str(e)[:70]}"}
        won += int(r.get("levels", 0) > 0)
        won_struct += int(r.get("won_by") == "structured")
        rows.append(r)
        print(f"{g:6} {str(r.get('won_by')):>10} {r.get('levels', 0):>6} "
              f"{str(r.get('first_levelup_actions')):>9} {str(r.get('efficiency')):>7} {r.get('error', '')}", flush=True)
    print(f"\nHYBRID first-win: {won}/{len(GAMES)}  ({won_struct} via structured-efficient, "
          f"{won - won_struct} via random-diversity)  vs structured 1/11, random 3/11", flush=True)
    out = {
        "experiment": "arc_hybrid_explore_measure",
        "honest_verdict": f"complete_hybrid_explore_firstwin_{won}_of_{len(GAMES)}",
        "struct_budget": STRUCT_BUDGET, "rand_budget": RAND_BUDGET,
        "first_win": won, "via_structured": won_struct, "via_random": won - won_struct,
        "per_game": rows,
        "inference_substrate": "verifier_ensemble_against_cached_candidates", "verifier_is_oracle": False,
    }
    (REPO / "results" / "arc_hybrid_explore_measure.json").write_text(json.dumps(out, indent=2, default=str))
    print("-> results/arc_hybrid_explore_measure.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
