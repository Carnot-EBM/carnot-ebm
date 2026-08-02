"""Robustness check: does the conclusion survive changing the root grid?

THE WEAKNESS THIS PROBES. `capture_roots.py` takes each game's OPENING board from the
offline arcade, because that is the only start grid available for all 20 games. The
grid the LIVE planner searches from is `E3AgentPolicy.root_grid` -- wherever the agent
had got to when reinduction fired -- and only 3 of these games have one recorded
(`results/arc_induce_bestofn_20260731/harness/capture/<game>/root_grid1.pkl`).

So the headline sweep is run on a start grid that is FAITHFUL IN KIND but not identical
to the live one. That is a real limitation, and the honest way to bound it is to
measure it rather than argue about it: on the cells where both roots exist, re-run the
gate from the captured root and check whether any per-cell verdict flips.

HOW TO READ THE RESULT. If no verdict flips, the arcade root is not doing the work and
the headline numbers stand on their own. If verdicts DO flip, the arcade numbers are
not evidence about the live planner and the artifact must say exactly that -- the point
of this file is to be able to tell those two cases apart, not to reach the first one.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CAPTURED = REPO / "results" / "arc_induce_bestofn_20260731" / "harness" / "capture"


def _verdict(arm: dict[str, Any]) -> str:
    """Collapse an arm to the one thing the comparison turns on."""
    if arm.get("outcome") != "gate_ran":
        return f"outcome:{arm.get('outcome')}"
    return f"satisfiable:{arm.get('satisfiable')}/{arm.get('counterexample_kind')}"


def main() -> int:
    arcade = json.loads((HERE / "gate_arcade.json").read_text())
    cap_path = HERE / "gate_captured.json"
    if not cap_path.exists():
        print("gate_captured.json missing -- run run_measure.py --root-source captured first")
        return 1
    captured = json.loads(cap_path.read_text())

    by_cell_a = {c["cell"]: c for c in arcade["cells"] if "arms" in c}
    by_cell_c = {c["cell"]: c for c in captured["cells"] if "arms" in c}
    overlap = sorted(set(by_cell_a) & set(by_cell_c))

    grid_cmp = []
    for game in sorted({by_cell_c[c]["game"] for c in overlap}):
        a = pickle.loads((HERE / "roots" / f"{game}.pkl").read_bytes())
        c = pickle.loads((CAPTURED / game / "root_grid1.pkl").read_bytes())
        a, c = np.asarray(a), np.asarray(c)
        same_shape = a.shape == c.shape
        grid_cmp.append(
            {
                "game": game,
                "arcade_shape": list(a.shape),
                "captured_shape": list(c.shape),
                "identical": bool(same_shape and np.array_equal(a, c)),
                "frac_cells_differing": (round(float(np.mean(a != c)), 4) if same_shape else None),
            }
        )

    flips = []
    for cell in overlap:
        for arm_a, arm_c in zip(by_cell_a[cell]["arms"], by_cell_c[cell]["arms"], strict=True):
            va, vc = _verdict(arm_a), _verdict(arm_c)
            flips.append(
                {
                    "cell": cell,
                    "role": arm_a["role"],
                    "arcade": va,
                    "captured": vc,
                    "flipped": va != vc,
                }
            )

    n_flip = sum(1 for f in flips if f["flipped"])
    out = {
        "n_cells_with_both_roots": len(overlap),
        "cells": overlap,
        "root_grid_comparison": grid_cmp,
        "arm_verdicts": flips,
        "n_arms_compared": len(flips),
        "n_arms_flipped": n_flip,
        "conclusion_robust_to_root_source": n_flip == 0,
    }
    (HERE / "root_robustness.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "arm_verdicts"}, indent=2))
    for f in flips:
        if f["flipped"]:
            print(f"  FLIP {f['cell']:24s} {f['role']:9s} {f['arcade']} -> {f['captured']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
