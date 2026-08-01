#!/usr/bin/env python3
"""Rebuild ONE game's progress window + explicit split + plan roots, in a killable process.

WHY A SUBPROCESS FOR SOMETHING THAT LOOKS LIKE ARITHMETIC. `build_progress_window` steps a real
offline environment and opens a scorecard; it has no internal bound. The sibling post-hoc pass
(`results/arc_object_perception_ab_change_fidelity_20260801/window_worker.py`) records that
`build_progress_window("tr87")` span at 100% CPU without returning and took a whole driver down,
twice, in two independently-written sweeps -- after the first version of that driver had already
isolated engine execution on the reasoning that LLM-written code was the only hazard. The
unbounded call was not the one that looked dangerous. A game that cannot be rebuilt inside the
driver's timeout is DROPPED with its reason recorded; it is never a zero.

WHAT THIS ADDS OVER THE SIBLING: the two PLAN ROOTS.

`plan_in_model` is a search rooted at a start grid, and the frozen best-of-N run rooted it at
`E3AgentPolicy.root_grid` -- a field the live agent carries and the object-perception A/B never
recorded. So for the A/B corpus a root has to be reconstructed, and this pickles the two honest
candidates rather than silently picking one:

  * `window_root` = the first grid of the level-up-straddling window (`shown[0].grid`). This is
    the earliest state in the level the window ends by clearing, so it is the closest
    reconstructible analogue of the agent's level root.
  * `held_root`   = the first grid of the held-out tail (`held[0].grid`). Nearer the level-up, so
    a plan from here is strictly easier to find.

Reporting both is not indecision. The best-of-N corpus has the REAL `root_grid` on disk, so the
same engines can be planned from the real root AND from `window_root`, which measures how much
the proxy costs. A conclusion that survives all three roots does not rest on the substitution.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import pickle
import sys

REPO = pathlib.Path(os.environ["CARNOT_REPO"])
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CARNOT_ARC_E3_DIR", "/tmp/arc_metric_validity/e3_build")
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)
sys.path.insert(0, str(REPO / "python"))


def main() -> int:
    job = json.loads(pathlib.Path(sys.argv[1]).read_text())

    import numpy as np

    from carnot.agentic import arc_actions_to_progress as atp
    from carnot.agentic import arc_world_model_trust_energy as wmte

    w = atp.build_progress_window(job["game"])
    if w is None:
        print(json.dumps({"status": "no_window", "game": job["game"]}))
        return 0
    win, _full, cell = w
    shown, held = wmte._split_prefix_heldout(list(win))  # noqa: SLF001
    if not shown or not held:
        print(json.dumps({"status": "empty_split", "game": job["game"]}))
        return 0

    window_root = np.asarray(shown[0].grid)
    held_root = np.asarray(held[0].grid)

    def _sha(a) -> str:
        return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:16]

    with open(job["window_pkl"], "wb") as fh:
        pickle.dump(
            {
                "shown": shown,
                "held": held,
                "cell": int(cell),
                "window_root": window_root,
                "held_root": held_root,
            },
            fh,
        )
    print(
        json.dumps(
            {
                "status": "ok",
                "game": job["game"],
                "n_shown": len(shown),
                "n_heldout": len(held),
                "cell": int(cell),
                "window_root_shape": list(window_root.shape),
                "window_root_sha256_16": _sha(window_root),
                "held_root_shape": list(held_root.shape),
                "held_root_sha256_16": _sha(held_root),
                "window_root_equals_held_root": bool(
                    np.array_equal(window_root, held_root)
                ),
                # How many held-out rows actually carry a change. `change_fidelity` averages over
                # exactly these, so a game with few of them supports a very coarse metric.
                "n_heldout_changing": int(
                    sum(
                        1
                        for t in held
                        if not np.array_equal(np.asarray(t.grid), np.asarray(t.next_grid))
                    )
                ),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
