#!/usr/bin/env python3
"""Score the DYNAMICS-FREE baselines for ONE game, in a killable process.

WHAT A BASELINE IS FOR HERE. The headroom artifact disqualified four object metrics with a
single criterion: does the metric rank a NON-MODEL above a real engine? It only ever tested
ONE non-model -- the INERT identity engine. Adversarial review named two more that were never
tested, and both are cheap:

  * DELTA REPLAY -- apply the most common rewrite seen in the SHOWN rows, unconditionally.
    Uses only prompt-visible data, so it is a thing a language model can produce by copying,
    with no dynamics understood at all.
  * ACTION / COORDINATE BLINDNESS -- an engine that is correct but cannot see the action or
    the click. Measured per engine in `rescore_worker.py`, not here.

plus ORACLE, which is not a baseline but the metric's reachable CEILING on this window. If the
oracle does not score 1.0 the window contains aliased states (the same grid bytes with two
different outcomes) and no engine can be perfect -- which is a fact about the corpus a reader
needs before interpreting any per-game value.

Same subprocess isolation and the same reason as `rescore_worker.py`.
"""

from __future__ import annotations

import collections
import json
import os
import pathlib
import pickle
import sys

REPO = pathlib.Path(os.environ["CARNOT_REPO"])
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CARNOT_ARC_E3_DIR", "/tmp/arc_rescore/e3b")
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)
sys.path.insert(0, str(REPO / "python"))


def main() -> int:
    job = json.loads(pathlib.Path(sys.argv[1]).read_text())
    import numpy as np
    from carnot.agentic import arc_executable_world_model as e3

    with open(job["window_pkl"], "rb") as fh:
        win = pickle.load(fh)
    held, shown = list(win["held"]), list(win["shown"])

    def sc(engine) -> dict:
        vr = e3.WorldModelVerifier(list(held)).score(engine)
        return {
            "change_fidelity": round(float(vr.change_fidelity), 6),
            "accuracy": round(float(vr.accuracy), 6),
            "cell_recall": round(float(vr.cell_recall), 6),
            "spurious_changed_cells": int(vr.spurious_changed_cells),
            "invented_changed_cells": int(vr.invented_changed_cells),
            "n_noop": int(vr.n_noop),
            "n_noop_hallucinated": int(vr.n_noop_hallucinated),
            "noop_channel_measurable": bool(vr.noop_channel_measurable),
        }

    # --- DELTA REPLAY: the most-voted (row, col, value) rewrites among SHOWN rows ---------
    # Capped at 8 rewrites and at 64 cells per shown delta so a single level-up-sized rewrite
    # cannot dominate the vote. Reads only what the prompt shows -- never a held-out row.
    votes: collections.Counter = collections.Counter()
    for t in shown:
        g0, g1 = np.asarray(t.grid), np.asarray(t.next_grid)
        m = g0 != g1
        if m.any() and int(m.sum()) <= 64:
            for y, x in zip(*np.where(m), strict=False):
                votes[(int(y), int(x), int(g1[y, x]))] += 1
    modal = [k for k, _ in votes.most_common(8)]

    def modal_delta(grid, action, data):
        g = np.asarray(grid).copy()
        for y, x, v in modal:
            if y < g.shape[0] and x < g.shape[1]:
                g[y, x] = v
        return g

    # --- ORACLE: exact lookup of the recorded outcome. Its score is the CEILING. ----------
    lut = {
        np.asarray(t.grid).tobytes(): np.asarray(t.next_grid)
        for t in held
        if t.level_after <= t.level_before
    }

    def oracle(grid, action, data):
        return lut.get(np.asarray(grid).tobytes(), np.asarray(grid))

    n_alias = len([t for t in held if t.level_after <= t.level_before]) - len(lut)

    print(
        json.dumps(
            {
                "game": job["game"],
                "status": "ok",
                "n_shown": len(shown),
                "n_heldout": len(held),
                "n_modal_rewrites": len(modal),
                "n_aliased_heldout_states": int(n_alias),
                "baselines": {
                    "IDENTITY": sc(lambda g, a, d: np.asarray(g)),
                    "MODAL_SHOWN_DELTA_REPLAY": sc(modal_delta),
                    "ORACLE_ceiling": sc(oracle),
                },
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
