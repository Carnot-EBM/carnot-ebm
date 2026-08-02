#!/usr/bin/env python3
"""Rebuild ONE game's progress window + the canonical prefix/held-out split, in a killable
process, and pickle it.

WHY A SUBPROCESS. `build_progress_window` steps a real game environment; it is not pure
arithmetic and has no internal bound. The sibling experiment
(results/arc_object_perception_ab_change_fidelity_20260801/window_worker.py) records that
building windows inline took a whole pass down twice, at 100% CPU, on tr87. A game whose
window cannot be rebuilt inside the driver's timeout is reported as a MISSING OBSERVATION
and dropped from the pass -- never scored zero.

This is a byte-for-byte re-use of that sibling's contract (same `build_progress_window`,
same `wmte._split_prefix_heldout`) so this experiment's windows ARE the windows the frozen
engine corpus was induced from. That is what makes the pre-flight on those engines
meaningful rather than a re-derivation.
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
os.environ.setdefault("CARNOT_ARC_E3_DIR", "/tmp/arc_goalab/e3_build")
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
    win, full, cell = w
    shown, held = wmte._split_prefix_heldout(list(win))  # noqa: SLF001
    with open(job["window_pkl"], "wb") as fh:
        pickle.dump({"shown": shown, "held": held, "full": list(full), "cell": int(cell)}, fh)
    n_levelup = sum(1 for t in win if getattr(t, "level_after", 0) > getattr(t, "level_before", 0))
    print(
        json.dumps(
            {
                "status": "ok",
                "game": job["game"],
                "n_shown": len(shown),
                "n_heldout": len(held),
                "n_full": len(full),
                "n_levelup_in_window": n_levelup,
                "cell": int(cell),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
