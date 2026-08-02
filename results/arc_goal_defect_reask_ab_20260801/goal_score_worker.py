#!/usr/bin/env python3
"""Evaluate ONE induced `is_level_complete` against REAL OBSERVED FRAMES, in a killable process.

NEVER run induced code in the driver's own interpreter. This worker exists so that a
predicate that hangs, recurses, or allocates is a bounded MISSING OBSERVATION for one cell
rather than a dead pass. Every per-frame call is additionally wrapped so that a raising
predicate is recorded as `raised` on that frame instead of aborting the cell.

WHAT IT SCORES, and the frame vocabulary (established by `pre/boundary_anatomy.json`, which
measured all 20 windows before any outcome was chosen):

  * `open`     -- the window's first observed grid: the level's OPENING BOARD.
  * `shown`    -- every grid in the prefix the induce prompt was built from.
  * `held`     -- every grid in the held-out tail, which the prompt never saw.
  * `pre_win`  -- the grid immediately BEFORE the real level-up: the last within-level state,
                  the one the winning action was taken from. It is in `held` in 20/20 games.
  * `post_win` -- `next_grid` of the real level-up transition. The anatomy measured this as a
                  WHOLESALE BOARD REPLACEMENT (median 25.8x an ordinary step's cell-change),
                  i.e. the NEXT level's opening board -- NOT a picture of the level that was
                  just completed. `arc_actions_to_progress._levelup_positive_recall`
                  (REQ-ARC-WMTE-5714) scores exactly this frame, which is why it is reported
                  here too rather than quietly replaced.

Outputs raw per-frame booleans. No outcome is defined here; the analysis defines outcomes
from these, so the same evaluation feeds every candidate and none of them can drift.
"""

from __future__ import annotations

import json
import pathlib
import pickle
import sys

import numpy as np


def _call(fn, grid) -> object:
    """True/False, or the string 'raised'/'nonbool'. Never propagates."""
    try:
        v = fn(np.asarray(grid))
    except Exception:  # noqa: BLE001
        return "raised"
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if v is None:
        return "nonbool"
    try:
        return bool(v)
    except Exception:  # noqa: BLE001
        return "nonbool"


def main() -> int:
    job = json.loads(pathlib.Path(sys.argv[1]).read_text())
    with open(job["window_pkl"], "rb") as fh:
        w = pickle.load(fh)
    shown, held, full = w["shown"], w["held"], w["full"]

    src = pathlib.Path(job["engine_path"]).read_text()
    ns: dict = {}
    try:
        exec(compile(src, job["engine_path"], "exec"), ns)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "exec_failed", "error": f"{type(exc).__name__}: {exc}"[:300]}))
        return 0
    fn = ns.get("is_level_complete")
    if fn is None:
        print(json.dumps({"status": "no_goal_fn"}))
        return 0

    lu = [
        i
        for i, t in enumerate(full)
        if getattr(t, "level_after", 0) > getattr(t, "level_before", 0)
    ]
    out: dict = {"status": "ok"}
    out["open"] = _call(fn, full[0].grid)
    out["shown_before"] = [_call(fn, t.grid) for t in shown]
    out["shown_after"] = [_call(fn, t.next_grid) for t in shown]
    out["held_before"] = [_call(fn, t.grid) for t in held]
    out["held_after"] = [_call(fn, t.next_grid) for t in held]
    if lu:
        t = full[lu[0]]
        out["pre_win"] = _call(fn, t.grid)
        out["post_win"] = _call(fn, t.next_grid)
    else:
        out["pre_win"] = None
        out["post_win"] = None
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
