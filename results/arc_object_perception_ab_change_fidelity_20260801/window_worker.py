#!/usr/bin/env python3
"""Rebuild and pickle ONE game's progress window + explicit split, in a killable process.

WHY THIS IS ITS OWN PROCESS, found the hard way. The first version of `rescore.py` isolated
ENGINE execution (LLM-written code, the obvious hazard) but built the windows inline in the
driver. `build_progress_window("tr87")` then span at 100% CPU without returning and took the
whole pass down with it -- twice, in two independently-written sweeps. So the driver had
exactly the defect it was written to guard against, one indirection out: the unbounded call
was not the one that looked dangerous.

`build_progress_window` steps a real game environment and creates a scorecard, so it is not
pure arithmetic and it has no internal bound. A game whose window cannot be rebuilt inside the
timeout is reported as such and DROPPED from the post-hoc pass -- its A/B cells keep the
numbers `run_ab.py` already recorded, they simply gain no added channels. That is a stated
coverage gap, never a zero.
"""

from __future__ import annotations

import json
import os
import pathlib
import pickle
import sys

REPO = pathlib.Path(os.environ["CARNOT_REPO"])
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CARNOT_ARC_E3_DIR", "/tmp/arc_rescore/e3_build")
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
