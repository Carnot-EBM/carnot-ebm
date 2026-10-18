"""Measure the divergence-tolerant TTT solve loop (piece 2) vs the explorer floor (2026-06-21).

Runs arc_ttt_solve_loop.ttt_solve on the harness games and reports levels-reached + first-win + the
divergence-learn / replan diagnostics. Compares against the undirected explorer floor (1/11, lp85 only,
results/arc_compete_sim.json). The honest question: does plan-through-the-cell-recall-model with
replan-on-divergence reach a first win where undirected exploration cannot -- or is first-win
exploration-bound (piece 3), leaving the loop's value to deepening on already-winnable games?
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic.arc_ttt_solve_loop import ttt_solve

GAMES = (sys.argv[1].split(",") if len(sys.argv) > 1 else ["lp85", "cd82", "r11l", "ls20"])
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 2000


def main() -> int:
    print(f"== TTT divergence-tolerant solve loop: games={GAMES} budget={BUDGET} ==", flush=True)
    print(f"{'game':6} {'levels':>6} {'1st_lvlup':>9} {'actions':>7} {'wins':>4} {'plans':>5} "
          f"{'replans':>7} {'div_learn':>9}", flush=True)
    rows = []
    solved = 0
    for g in GAMES:
        try:
            r = ttt_solve(g, budget=BUDGET)
        except Exception as e:
            r = {"game": g, "error": f"{type(e).__name__}: {str(e)[:80]}", "levels_reached": 0}
            print(f"{g:6} ERROR {r['error']}", flush=True)
            rows.append(r); continue
        solved += int(r.get("levels_reached", 0) > 0)
        rows.append(r)
        print(f"{g:6} {r['levels_reached']:>6} {str(r.get('first_levelup_actions')):>9} {r['actions']:>7} "
              f"{r.get('n_win_states',0):>4} {r.get('plans_found',0):>5} {r.get('replans_on_divergence',0):>7} "
              f"{r.get('divergence_learns',0):>9}", flush=True)
    print(f"\nFIRST-WIN: {solved}/{len(GAMES)} games reached >=1 level (explorer floor on these = lp85 only)",
          flush=True)
    out = {
        "experiment": "arc_ttt_solve_loop_measure",
        "honest_verdict": f"complete_ttt_solve_loop_firstwin_{solved}_of_{len(GAMES)}",
        "budget": BUDGET, "per_game": rows,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "verifier_is_oracle": False,
    }
    (REPO / "results" / "arc_ttt_solve_loop_measure.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"-> results/arc_ttt_solve_loop_measure.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
