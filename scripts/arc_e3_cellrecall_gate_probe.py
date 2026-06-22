"""Piece-1 measurement of the coordinated redesign: does the CELL-RECALL verify gate let e3's LLM-induced
world-models PASS (and the plan step fire) where the exact-match gate blocked them? (2026-06-21)

Runs the FULL e3 cascade with CARNOT_ARC_TRUST_METRIC=cell_recall on the explorer-failing games and dumps
each induction attempt's verify_accuracy (exact-full-grid, recorded regardless) vs verify_cell_recall
(graded) + whether the plan fired. Exact-match was the 0.08-wall chokepoint (both TTT + e3 induction died
there); this measures whether re-metricing the gate UN-gates the induced models. NOTE (honest): passing the
gate is necessary-not-sufficient -- pieces 2 (divergence-tolerant execute) + 3 (directed first-win) are
still missing, so first-win is NOT expected to move from this piece alone; the signal is gate-pass + plan-fire.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts"))

os.environ["CARNOT_ARC_TRUST_METRIC"] = "cell_recall"

import arc_leaderboard_eval as lbe

GAMES = (sys.argv[1].split(",") if len(sys.argv) > 1 else ["cd82", "ls20"])
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 3000


def main() -> int:
    print(f"== e3 cell-recall verify-gate probe: games={GAMES} budget={BUDGET} (CARNOT_ARC_TRUST_METRIC=cell_recall) ==",
          flush=True)
    for game in GAMES:
        pol = lbe._build_policy("e3", game)
        r = lbe.run_game(game, pol, budget=BUDGET)
        attempts = getattr(pol, "induction_attempts", [])
        print(f"\n{game}: levels={r.get('levels')} first_levelup_actions={r.get('actions_to_first_levelup')} "
              f"actions={r.get('actions')} | {len(attempts)} induction attempts", flush=True)
        for i, a in enumerate(attempts):
            acc = a.get("verify_accuracy")
            cr = a.get("verify_cell_recall")
            flip = (acc is not None and cr is not None and acc < 0.5 <= cr)  # gate flips skip->pass
            print(f"  attempt {i}: verify_accuracy={acc} verify_cell_recall={cr} "
                  f"trust_metric={a.get('trust_metric')} skipped={a.get('skipped')} planned={a.get('planned')} "
                  f"plan_len={a.get('plan_length')} {'<- GATE FLIPS exact->cellrecall' if flip else ''}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
