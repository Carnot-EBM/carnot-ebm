#!/usr/bin/env python3
"""Rebuild and pickle ONE A/B game's progress window + split, in a killable process.

Identical construction to the A/B's own `window_worker.py` -- `build_progress_window` then
`_split_prefix_heldout` -- because a window built any other way is a different experiment, and
the unmasked arm would then fail its reproduction check against the A/B's recorded numbers.
That check is the whole reason this is safe to re-derive rather than copy.

WHY IT IS ITS OWN PROCESS. `build_progress_window` steps a real game environment and creates a
scorecard; it is not arithmetic and it has no internal bound. `build_progress_window("tr87")`
span at 100% CPU without returning in two independently-written sweeps on 2026-07-31, taking
the whole driver down. A game that cannot be rebuilt inside the timeout is reported, and the
driver falls back to a previously-pickled window for it -- which is only sound BECAUSE the
unmasked reproduction check would catch a window that is not the one the A/B graded.
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
os.environ.setdefault("CARNOT_ARC_E3_DIR", "/tmp/arc_hudms/e3_build")
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
