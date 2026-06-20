"""Overnight variant-transfer probe (outer-loop, operator-requested 2026-06-20).

Does the GENERIC solver RE-DERIVE a solution on a manufactured HELD-OUT layout variant? A color-
permutation recolors the observation (positions unchanged -> a color-robustness check); a REFLECTION moves
positions -> the banked layout-specific solution no longer applies, so a solve means the solver genuinely
re-searched the new layout. That generalization (re-derive, not replay) is the dev-side proxy for the
unseen ~110 eval games -- and a reflection-variant solve is itself a held-out-layout solve.

Reproduction-honest: VariantEnv keeps the REAL win-logic (levels_completed passes through), so a "solved"
flag means the real level advanced on the variant. Bounded per-variant so it finishes overnight.
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_graph_explore import graph_explore_solve_v2
from carnot.agentic.arc_variant_generator import VariantEnv

# games the generic explore re-derives (adapter-free / L1). Reflection variants test whether the solver
# generalizes; we EXCLUDE the deep adaptered games (lp85/tu93/tr87/tn36) whose solves are banked-replay.
GAMES = ["lf52", "cd82", "sp80", "su15", "cn04", "m0r0", "sk48", "r11l", "ls20",
         "wa30", "s5i5", "ft09", "vc33", "g50t", "dc22", "sb26"]


def main() -> int:
    arc = kit.offline_arcade()
    out = {}
    for g in GAMES:
        res = {}
        for label, kw in [("color", dict(variant=1, reflect=None)),
                          ("reflect_h", dict(variant=1, reflect=1)),
                          ("reflect_v", dict(variant=1, reflect=0))]:
            try:
                env = VariantEnv(arc.make(g, scorecard_id=arc.open_scorecard()), g, **kw)
                env.reset()
                t = time.time()
                # bounded so the overnight sweep finishes; a solve within this budget == EASILY re-derived
                # (robust), a no-solve == the variant is harder for the generic solver (layout-sensitive).
                traj, lvl = graph_explore_solve_v2(env, 0, max_expansions=3000, max_depth=40)
                res[label] = {"solved": bool(traj and lvl >= 1), "level": int(lvl),
                              "moves": len(traj) if traj else 0, "s": round(time.time() - t, 1)}
            except Exception as e:
                res[label] = {"error": f"{type(e).__name__}: {e}"}
        solved = sum(1 for v in res.values() if v.get("solved"))
        out[g] = {"variants": res, "transfer_solved": f"{solved}/3"}
        flags = "  ".join(
            f"{k}:{'OK-L' + str(v.get('level')) if v.get('solved') else ('ERR' if 'error' in v else 'no')}"
            for k, v in res.items())
        print(f"[{g}] transfer {solved}/3   {flags}", flush=True)
    # summary: reflection-transfer rate = how many games still solve under a position-moving variant
    refl_ok = sum(1 for g in out if out[g]["variants"].get("reflect_h", {}).get("solved")
                  or out[g]["variants"].get("reflect_v", {}).get("solved"))
    summary = {"games": len(out), "reflection_generalizing_games": refl_ok,
               "interpretation": "games whose generic solver re-derives under a reflected (position-moved) "
                                 "layout -- these generalize (robust to relayout), the trait the unseen eval "
                                 "rewards; the rest replay layout-specific solutions and would not transfer.",
               "per_game": out}
    (REPO / "results" / "arc_variant_transfer_probe.json").write_text(json.dumps(summary, indent=2))
    print(f"=== DONE: {refl_ok}/{len(out)} games generalize under reflection ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
