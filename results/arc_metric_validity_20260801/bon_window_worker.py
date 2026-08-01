#!/usr/bin/env python3
"""Rebuild ONE best-of-N game's PROVEN split + its three plan roots, in a killable process.

WHY THE FROZEN HARNESS'S OWN SPLITTER IS IMPORTED RATHER THAN REIMPLEMENTED. The best-of-N run's
held-out set is not a fraction rule -- it is `full \\ shown` where `shown` is the rows the prompt
actually rendered, proven row-by-row against the prompt TEXT, with rows whose rendered delta line
is ambiguous DROPPED rather than scored as unseen. Rewriting that here would produce a second,
subtly different definition of "held out", and the whole value of this corpus is that its split
carries a proof. `results/arc_induce_bestofn_20260731/harness/split.py` is on disk and frozen, so
it is imported and called.

CALL_INDEX 1 is pinned, not defaulted. `split.py` defaults `SPLIT_CALL_INDEX` to 2, but
`bestofn_scored.json` records `call_index: 1` -- the bounded-reinduction STALL site. Taking the
module default would silently score a different induction call than the one whose `plan_found`
values this run is re-deriving.

`CARNOT_ARC_INDUCE_TRANSITIONS_K=8` IS ALSO PINNED, AND FINDING OUT WHY IS THE REASON THIS FILE
RUNS A REPRODUCTION GATE. `split.py` derives `shown` as `changed[:k-2] + noop[:2]` where `k` comes
from the LIVE resolver `e3._induce_transitions_k()`. That resolver's default changed from 8 to
None ("show ALL transitions") on 2026-08-01 -- the day AFTER this corpus was frozen. Imported
unpinned it now raises `TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'`, which
is the LOUD failure. The quiet one is what matters: had the resolver returned some other integer
instead of None, the split would have silently reshaped and every held-out number here would have
been graded against a different set of rows than the frozen run graded, with nothing to say so. So
the pin restores the as-run prompt shape byte-for-byte, and the rebuilt split is then CHECKED
row-count-by-row-count against the frozen `split.json`. A mismatch voids the game rather than
being averaged in.

THREE ROOTS, and this corpus is the only place all three exist. `real_root` is
`E3AgentPolicy.root_grid` as captured live; `window_root`/`held_root` are the reconstructions the
object-perception corpus is forced to use. Running the same engines from all three is what
licenses the substitution over there -- or refuses to.
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
os.environ.setdefault("CARNOT_ARC_E3_DIR", "/tmp/arc_metric_validity/e3_bon")
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)
os.environ["SPLIT_CALL_INDEX"] = "1"
# See the docstring: restores the as-frozen prompt shape. Without it the live resolver's new
# "ALL" default reshapes `shown` and every held-out number would grade a different row set.
os.environ["CARNOT_ARC_INDUCE_TRANSITIONS_K"] = "8"
sys.path.insert(0, str(REPO / "python"))


def main() -> int:
    job = json.loads(pathlib.Path(sys.argv[1]).read_text())
    harness = pathlib.Path(job["harness_dir"])
    sys.path.insert(0, str(harness))

    import numpy as np

    from split import load_split  # type: ignore[import-not-found]

    s = load_split(job["game"], 1)
    shown, held = list(s["_shown"]), list(s["_heldout"])
    if not shown or not held:
        print(
            json.dumps(
                {
                    "status": "empty_split",
                    "game": job["game"],
                    "n_shown": len(shown),
                    "n_heldout": len(held),
                }
            )
        )
        return 0

    def _sha(a) -> str:
        return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:16]

    # REPRODUCTION GATE against the frozen split.json. The rebuild imports live repo code, and
    # the pinned `k` above exists because that code has already drifted once since this corpus was
    # frozen. If the row counts disagree, this is not the split the best-of-N graded and the game
    # is VOID -- reported, never quietly used.
    frozen = None
    fp = pathlib.Path(job["frozen_split_json"])
    if fp.exists():
        for row in json.loads(fp.read_text()).get("rows", []):
            if row.get("game") == job["game"] and int(row.get("call_index", 1)) == 1:
                frozen = row
                break
    repro = None
    if frozen is not None:
        repro = {
            "n_shown": (len(shown), frozen["n_shown"]),
            "n_heldout": (len(held), frozen["n_heldout"]),
            "heldout_n_changing": (
                int(s.get("heldout_n_changing") or 0),
                frozen["heldout_n_changing"],
            ),
        }
        repro_ok = all(a == b for a, b in repro.values())
    else:
        repro_ok = False

    window_root = np.asarray(shown[0].grid)
    held_root = np.asarray(held[0].grid)
    with open(job["window_pkl"], "wb") as fh:
        pickle.dump(
            {
                "shown": shown,
                "held": held,
                "cell": None,
                "window_root": window_root,
                "held_root": held_root,
            },
            fh,
        )
    print(
        json.dumps(
            {
                "status": "ok" if repro_ok else "reproduction_mismatch",
                "reproduces_frozen_split": bool(repro_ok),
                "reproduction_detail_rebuilt_vs_frozen": repro,
                "game": job["game"],
                "n_shown": len(shown),
                "n_heldout": len(held),
                "split_proven": bool(s.get("split_proven")),
                "heldout_n_changing": int(s.get("heldout_n_changing") or 0),
                "heldout_can_grade_change": bool(s.get("heldout_can_grade_change")),
                "n_ambiguous_dropped": int(s.get("n_ambiguous_dropped") or 0),
                "window_root_sha256_16": _sha(window_root),
                "held_root_sha256_16": _sha(held_root),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
