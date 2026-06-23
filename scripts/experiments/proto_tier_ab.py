"""A/B: graft just-explore's 5-tier salience schedule into rich_action_candidates and test whether OUR
graph explorer reaches first-wins its flat area*rarity sort misses -- on the 5 games the 2026-06-23
head-to-head showed just-explore winning (bp35[UNSOLVED]/ft09/m0r0/r11l/vc33), plus a regression set.

Same env, same budget, graph_explore_solve_v2; the ONLY difference is CARNOT_ARC_TIER_SCHEDULE
(off = flat order, on = tier order). Gate: tier reaches a STRICTLY deeper level than flat on >=2 of the
5 win-games, with ZERO regression (tier >= flat) on every game. bp35 is the prize (we don't solve it).
"""

from __future__ import annotations

import json
import os
import sys
import time

from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_graph_explore import graph_explore_solve_v2

WIN_GAMES = ["bp35", "ft09", "m0r0", "r11l", "vc33"]   # just-explore beat our flat explorer here
REGRESSION = ["lp85", "sp80", "su15", "sc25", "cd82", "ar25", "cn04"]  # must not regress


def _gid(arc, short):
    for e in arc.get_environments():
        g = getattr(e, "game_id", "")
        if g.split("-")[0] == short:
            return str(g)
    raise RuntimeError(f"{short} unavailable")


def _run(arc, gid, budget, tier: bool) -> int:
    if tier:
        os.environ["CARNOT_ARC_TIER_SCHEDULE"] = "1"
    else:
        os.environ.pop("CARNOT_ARC_TIER_SCHEDULE", None)
    env = arc.make(gid, scorecard_id=arc.open_scorecard())
    _traj, lvl = graph_explore_solve_v2(env, start_level=0, max_expansions=budget,
                                        warmup=True, max_depth=60)
    return int(lvl)


def main() -> int:
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    arc = kit.offline_arcade()
    out = {"budget": budget, "win_games": {}, "regression": {}}
    for label, games in (("win_games", WIN_GAMES), ("regression", REGRESSION)):
        for short in games:
            gid = _gid(arc, short)
            t0 = time.time()
            flat = _run(arc, gid, budget, tier=False)
            tier = _run(arc, gid, budget, tier=True)
            out[label][short] = {"flat": flat, "tier": tier, "delta": tier - flat,
                                 "wall_s": round(time.time() - t0, 1)}
            print(f"  {label[:3]} {short:6} flat={flat} tier={tier} delta={tier-flat:+d}  ({round(time.time()-t0,1)}s)", flush=True)
    win_deltas = {g: v["delta"] for g, v in out["win_games"].items()}
    reg_deltas = {g: v["delta"] for g, v in out["regression"].items()}
    out["tier_deeper_win_games"] = sorted(g for g, d in win_deltas.items() if d > 0)
    out["regressions"] = sorted(g for g, d in {**win_deltas, **reg_deltas}.items() if d < 0)
    out["bp35_first_contact"] = out["win_games"].get("bp35", {}).get("tier", 0) > out["win_games"].get("bp35", {}).get("flat", 0)
    gate = len(out["tier_deeper_win_games"]) >= 2 and not out["regressions"]
    out["VERDICT"] = "TIER_SCHEDULE_WINS_extract" if gate else (
        "TIER_REGRESSES" if out["regressions"] else "TIER_NULL_no_win")
    json.dump(out, open("results/proto_tier_ab.json", "w"), indent=2)
    print(f"\ntier_deeper_win_games={out['tier_deeper_win_games']}  regressions={out['regressions']}  "
          f"bp35_first_contact={out['bp35_first_contact']}  VERDICT={out['VERDICT']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
