#!/usr/bin/env python3
"""Is `engine_changes_anything` -- the check this artifact recommends wiring -- itself bounded?

WHY THIS PROBE EXISTS. The recommendation that comes out of this taxonomy is to reject INERT
engines at generation time, which means calling `arc_engine_static_validation.engine_changes_
anything` on the live induce path. Recommending that without checking whether it can hang would
be reintroducing the 2026-07-31 incident -- a non-terminating induced engine wedged a generation
loop for 13 minutes -- inside the fix for a different problem.

The 2026-08-01 hardening put `dry_run_defects` behind a killable subprocess. It did NOT put
`engine_changes_anything` behind one: that function calls `_exec_namespace` and then the engine
directly, in this interpreter, with no timeout parameter and no subprocess. Reading the source is
suggestive; this probe MEASURES it, on the one frozen candidate known to be non-terminating
(ft09 candidate 5), by timing the two calls separately.

EXPECTED, IF THE GAP IS REAL: `validate_engine_code` returns within its own bound carrying
`engine_nonterminating`, and `engine_changes_anything` does not return at all.

This probe runs the engine in a killable child of its own, so the probe cannot hang the session
it is diagnosing.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRATCH = pathlib.Path(
    os.environ.get(
        "ARC_GENTAX_SCRATCH",
        "/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/"
        "87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/gentax",
    )
)
BOUND_S = 45.0

CHILD = r"""
import json, os, pickle, sys, time
sys.path.insert(0, os.environ["CARNOT_PY"])
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
from carnot.agentic import arc_engine_static_validation as sv
code = open(sys.argv[2]).read()
shown = list(pickle.load(open(sys.argv[3], "rb"))["shown"])
which = sys.argv[1]
t = time.monotonic()
if which == "validate":
    d = sv.validate_engine_code(code, transitions=shown, required=("engine",))
    print(json.dumps({"call": which, "s": round(time.monotonic()-t, 2),
                      "kinds": sorted({x.kind for x in d})}))
else:
    r = sv.engine_changes_anything(code, shown)
    print(json.dumps({"call": which, "s": round(time.monotonic()-t, 2), "result": r}))
"""


def run(which: str, code_path: str, shown_pkl: str) -> dict:
    script = SCRATCH / "_probe_child.py"
    script.write_text(CHILD)
    env = dict(os.environ, CARNOT_PY=str(REPO / "python"))
    t = time.monotonic()
    try:
        pr = subprocess.run(  # noqa: S603
            [sys.executable, str(script), which, code_path, shown_pkl],
            capture_output=True,
            text=True,
            timeout=BOUND_S,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "call": which,
            "outcome": "DID_NOT_RETURN",
            "killed_after_s": BOUND_S,
            "elapsed_s": round(time.monotonic() - t, 2),
        }
    line = (pr.stdout or "").strip().splitlines()
    if not line:
        return {"call": which, "outcome": "no_output", "stderr": (pr.stderr or "")[-300:]}
    return {"call": which, "outcome": "returned", **json.loads(line[-1])}


def main() -> int:
    code = SCRATCH / "bon_extracted" / "ft09_k5.py"
    shown = SCRATCH / "bon_shown" / "ft09_shown.pkl"
    if not code.exists() or not shown.exists():
        print(json.dumps({"status": "inputs_missing", "code": str(code), "shown": str(shown)}))
        return 1
    out = {
        "what_this_is": __doc__,
        "candidate": (
            "ft09_k5 -- the frozen non-terminating engine (bestofn score_status=gate_timeout)"
        ),
        "outer_bound_s": BOUND_S,
        "validate_engine_code": run("validate", str(code), str(shown)),
        "engine_changes_anything": run("changes", str(code), str(shown)),
    }
    v, c = out["validate_engine_code"], out["engine_changes_anything"]
    out["verdict"] = (
        "GAP CONFIRMED: validate_engine_code returns inside its own subprocess bound while "
        "engine_changes_anything does not return at all. Wiring the inertness check onto the "
        "live induce path AS-IS would reintroduce the 2026-07-31 hang. It must be given the "
        "same killable-subprocess treatment dry_run_defects already has."
        if v.get("outcome") == "returned" and c.get("outcome") == "DID_NOT_RETURN"
        else "NO GAP OBSERVED on this candidate; the recommendation's hazard caveat is not "
        "supported by this probe and should be weakened accordingly."
    )
    (HERE / "probe_unbounded_inertness.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
