#!/usr/bin/env python3
"""Rebuild and pickle ONE game's induction window + prefix/held-out split, in a killable process.

WHY A SUBPROCESS FOR SOMETHING THAT LOOKS LIKE ARITHMETIC. `build_progress_window` steps a real
game environment and builds a scorecard. It has no internal bound. The 2026-08-01 rescue pass
(`results/arc_object_perception_ab_change_fidelity_20260801/window_worker.py`) records that
`build_progress_window("tr87")` span at 100% CPU without returning and took an entire driver
down, twice, in two independently-written sweeps -- in a driver that had carefully isolated
ENGINE execution and then made the unbounded call itself, one indirection out. This file exists
so that mistake is not repeated here.

A game whose window cannot be rebuilt inside the timeout is DROPPED with its reason recorded.
Its candidates then carry static-only classification (no dry run) and are counted as a stated
coverage gap, never silently as "no defect found".
"""

from __future__ import annotations

import json
import os
import pathlib
import pickle
import sys

# Derive the repo root from this file's own location -- never a hardcoded absolute path
# (CLAUDE.md "Test-Run Record Integrity Discipline" rule 4). This file lives at
# <repo>/results/arc_generation_taxonomy_20260801/window_worker.py, so the root is two up.
REPO = pathlib.Path(__file__).resolve().parents[2]

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CARNOT_ARC_E3_DIR", "/tmp/arc_gentax/e3_build")
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)
sys.path.insert(0, str(REPO / "python"))


def main() -> int:
    job = json.loads(pathlib.Path(sys.argv[1]).read_text())
    from carnot.agentic import arc_actions_to_progress as atp
    from carnot.agentic import arc_world_model_trust_energy as wmte

    w = atp.build_progress_window(job["game"])
    if w is None:
        print(json.dumps({"status": "no_window", "game": job["game"]}))
        return 0
    win, _full, cell = w
    # The SAME deterministic split the induce path makes, so `shown` here is the set of
    # transitions the model was actually shown when the frozen candidate was generated.
    shown, held = wmte._split_prefix_heldout(list(win))  # noqa: SLF001
    with open(job["window_pkl"], "wb") as fh:
        pickle.dump({"shown": shown, "held": held, "cell": int(cell)}, fh)
    print(
        json.dumps(
            {
                "status": "ok",
                "game": job["game"],
                "n_shown": len(shown),
                "n_heldout": len(held),
                "cell": int(cell),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
