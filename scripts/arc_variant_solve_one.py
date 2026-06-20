"""Try to SOLVE manufactured held-out VARIANTS of one game with the generic solver — a real
generalization test (a variant solve = a held-out-layout solve, the dev-side proxy for the unseen eval).

Color-permutation recolors the observation (positions fixed → no click remap); reflection moves positions
(click data inverse-remapped by VariantEnv). The REAL env keeps win-logic, so `solved` means the real
level advanced on a layout the solver has never seen. Bounded so it finishes fast. Prints one JSON line.

Usage: .venv/bin/python scripts/arc_variant_solve_one.py --game lf52 [--budget 2500]
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic import arc_solver_kit as kit  # noqa: E402
from carnot.agentic.arc_graph_explore import graph_explore_solve_v2  # noqa: E402
from carnot.agentic.arc_variant_generator import VariantEnv  # noqa: E402


def _try(game, budget, **kw):
    arc = kit.offline_arcade()
    env = VariantEnv(arc.make(game, scorecard_id=arc.open_scorecard()), game, **kw)
    env.reset()
    t = time.time()
    try:
        traj, lvl = graph_explore_solve_v2(env, 0, max_expansions=budget, max_depth=30)
        return {"solved": bool(traj and lvl >= 1), "level": int(lvl),
                "moves": len(traj) if traj else 0, "s": round(time.time() - t, 1)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "s": round(time.time() - t, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--budget", type=int, default=2500)
    a = ap.parse_args()
    res = {
        "game": a.game,
        "color": _try(a.game, a.budget, variant=1, reflect=None),
        "reflect_h": _try(a.game, a.budget, variant=1, reflect=1),
        "reflect_v": _try(a.game, a.budget, variant=1, reflect=0),
    }
    res["any_variant_solved"] = any(v.get("solved") for k, v in res.items() if isinstance(v, dict))
    res["reflect_solved"] = bool(res["reflect_h"].get("solved") or res["reflect_v"].get("solved"))
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
