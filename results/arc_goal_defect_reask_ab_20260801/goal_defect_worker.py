#!/usr/bin/env python3
"""Run `LocalGGUFProposer._goal_defects` on ONE frozen engine, both flag states, in a killable
process.

BOTH FLAG STATES IN ONE WORKER on purpose: `defects_off` is the inertness proof measured on
REAL corpus code rather than on a fixture. The unit test asserts inertness on a hand-written
`return False`; this asserts it on 115 engines the model actually wrote, which is the sample
that could contain a shape the fixture does not.

No LLM, no GPU, no server -- this only executes the induced predicate against observed frames.
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
sys.path.insert(0, str(REPO / "python"))


def main() -> int:
    job = json.loads(pathlib.Path(sys.argv[1]).read_text())
    from carnot.agentic import arc_executable_world_model as e3

    with open(job["window_pkl"], "rb") as fh:
        w = pickle.load(fh)
    # The detector probes the SHOWN grids -- the ones the induce prompt contained -- which is
    # what keeps it disjoint from the held-out tail the outcome is scored on.
    shown = w["shown"]
    code = pathlib.Path(job["engine_path"]).read_text()
    prop = e3.LocalGGUFProposer()

    os.environ.pop("CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK", None)
    off = prop._goal_defects(code, shown)  # noqa: SLF001
    os.environ["CARNOT_ARC_INDUCE_GOAL_DEFECT_CHECK"] = "1"
    on = prop._goal_defects(code, shown)  # noqa: SLF001
    print(json.dumps({"status": "ok", "defects_off": off, "defects_on": on}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
