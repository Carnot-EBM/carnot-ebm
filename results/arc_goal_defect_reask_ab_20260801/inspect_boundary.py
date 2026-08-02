#!/usr/bin/env python3
"""What does the real level-up boundary LOOK like? -- a pure-data pre-flight, no LLM.

THE QUESTION THIS ANSWERS, and why it has to be answered before any outcome is chosen.
`arc_actions_to_progress._levelup_positive_recall` (REQ-ARC-WMTE-5714) scores an induced
`is_level_complete` on `t.next_grid` at a real level-up. But the 2026-07-29 win-state-poison
correction established that `previous_level_complete_grid` -- captured from the frame AFTER
the counter incremented -- is the NEXT level's OPENING BOARD, not a win state. If
`t.next_grid` at a level-up is the same kind of frame, then "fires at the level-up frame" is
asking the predicate to be True on an opening board, which is exactly what
`_goal_satisfiability_check` REJECTS as `goal_predicate_true_at_root`. An outcome that
demands the thing the shipped gate forbids would be incoherent.

So: measure, do not reason. For each game's window, dump how similar the level-up
transition's `prev_grid` and `next_grid` are to (a) each other, (b) the level's own opening
board, and (c) the frames on either side. Report cell-difference counts, not prose.
"""

from __future__ import annotations

import json
import pathlib
import pickle
import sys

import numpy as np

SCRATCH = pathlib.Path(
    "/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/"
    "87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/goalab"
)
HERE = pathlib.Path(__file__).resolve().parent


def diff(a, b) -> int:
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return -1
    return int(np.sum(a != b))


def main() -> int:
    rows = []
    for pkl in sorted((SCRATCH / "windows").glob("*.pkl")):
        game = pkl.stem
        with open(pkl, "rb") as fh:
            d = pickle.load(fh)
        full = d["full"]
        shown, held = d["shown"], d["held"]
        idx = [
            i
            for i, t in enumerate(full)
            if getattr(t, "level_after", 0) > getattr(t, "level_before", 0)
        ]
        rec: dict = {
            "game": game,
            "n_full": len(full),
            "n_shown": len(shown),
            "n_held": len(held),
            "levelup_idx_in_full": idx,
        }
        # is the level-up inside the SHOWN prefix or the HELD-OUT tail of the window?
        wshown = [
            i
            for i, t in enumerate(shown)
            if getattr(t, "level_after", 0) > getattr(t, "level_before", 0)
        ]
        wheld = [
            i
            for i, t in enumerate(held)
            if getattr(t, "level_after", 0) > getattr(t, "level_before", 0)
        ]
        rec["levelup_in_shown"] = wshown
        rec["levelup_in_held"] = wheld
        if idx:
            i = idx[0]
            t = full[i]
            rec["levels"] = [
                int(getattr(t, "level_before", -1)),
                int(getattr(t, "level_after", -1)),
            ]
            rec["prev_vs_next_at_levelup"] = diff(t.grid, t.next_grid)
            rec["grid_shape"] = list(np.asarray(t.grid).shape)
            rec["n_cells"] = int(np.asarray(t.grid).size)
            # the level's own opening board = prev_grid of the FIRST transition of the window
            rec["next_at_levelup_vs_window_open"] = diff(t.next_grid, full[0].grid)
            rec["prev_at_levelup_vs_window_open"] = diff(t.grid, full[0].grid)
            # the frame AFTER the boundary (the first ordinary frame of the new level)
            if i + 1 < len(full):
                rec["next_at_levelup_vs_following_frame"] = diff(t.next_grid, full[i + 1].grid)
                rec["following_step_change"] = diff(full[i + 1].grid, full[i + 1].next_grid)
            # a typical ordinary step's change size, for scale
            ordinary = [diff(u.grid, u.next_grid) for j, u in enumerate(full) if j != i]
            ordinary = [x for x in ordinary if x >= 0]
            rec["ordinary_step_change_median"] = int(np.median(ordinary)) if ordinary else None
            rec["ordinary_step_change_max"] = int(max(ordinary)) if ordinary else None
        rows.append(rec)
        print(json.dumps(rec))
    out = HERE / "pre" / "boundary_anatomy.json"
    out.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
