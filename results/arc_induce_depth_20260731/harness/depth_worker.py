#!/usr/bin/env python3
"""DEPTH SWEEP -- re-run the goal gate and the planner for ONE candidate at ONE max_depth.

WHY A SUBPROCESS, AGAIN. Identical reasoning to the best-of-N `gate_worker.py` this is derived
from, and it is not defensive programming -- it is a measured failure. On 2026-07-31 the
best-of-N generation loop wedged for 13 minutes spinning inside a dry run of a non-terminating
induced engine. `_goal_satisfiability_check`, `plan_in_model` and `dry_run_defects` all wrap
engine invocation in a broad `except Exception: continue`, so a signal-based in-process alarm
would be SWALLOWED by the very code meant to be interrupted, turning a hang into a silent false
CLEAN. So no generated code runs in the driver's interpreter; the driver kills by PID.

WHY THE GATE AND THE PLANNER GET THE **SAME** max_depth, always.
`arc_llm_reinduction.py` states the invariant plainly: the gate's veto on
`goal_unreached_within_depth` is SOUND precisely because "plan_in_model is bounded by the same
depth cap and therefore could not reach this goal either". Deepen the gate alone and it starts
certifying goals the planner then fails on -- converting an honest veto into a false accept one
layer down. Deepen the planner alone and the gate keeps vetoing goals the planner could now
reach. Only the coupled pair is a meaningful measurement, so `max_depth` is one job field
applied to both, never two.

WHAT IS NOT WIDENED. `max_nodes` stays at the shipped default on BOTH (the gate reads
`_goal_gate_max_nodes_default()`; the planner its own 20000). Depth is the single manipulated
variable. If a candidate converts here, it converted because the search was allowed to go
deeper -- not because it was given more budget to go wider. Any candidate that terminates
`budget_exhausted` / `max_nodes_reached` is recorded as UNDECIDED and is NOT counted as a
conversion, because at that termination depth is not what stopped it.

`goal_energy` is deliberately NOT supplied to `plan_in_model`, matching the best-of-N run this
extends. The live stall path may install a goal-energy heuristic, and a best-first search can
only reach a goal in FEWER nodes, never more -- so every plan-found count here is a LOWER BOUND
on the live planner. Conservative direction for a measurement whose headline is a conversion.

Criterion (i) -- held-out dynamics -- is deliberately NOT recomputed. It is a function of
(engine, held-out rows) with no search in it, so it cannot depend on `max_depth`; re-deriving it
would only add a way for this sweep to disagree with the artifact it extends. It is joined in
from `bestofn_scored.json` by `(game, candidate)`.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("CARNOT_ARC_E3_DIR", "/tmp/arc_depth_gate/e3")
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "python"))


def main() -> int:
    with open(sys.argv[1]) as fh:
        job = json.loads(fh.read())
    depth = int(job["max_depth"])
    out: dict = {"status": "ok", "max_depth": depth}

    import numpy as np
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic.arc_llm_reinduction import _goal_satisfiability_check

    with open(job["code_path"]) as fh:
        code = fh.read()
    ns: dict = {"np": np, "numpy": np}
    try:
        exec(compile(code, job["code_path"], "exec"), ns)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        out["status"] = f"unrunnable:{type(exc).__name__}"
        out["error"] = str(exc)[:240]
        print(json.dumps(out))
        return 0

    engine = ns.get("engine")
    goal = ns.get("is_level_complete")
    if not callable(engine):
        out["status"] = "no_engine"
        print(json.dumps(out))
        return 0

    with open(job["root_pkl"], "rb") as fh:
        root = np.asarray(pickle.load(fh))

    # ---- criterion (ii): the goal gate, shipped max_nodes, SWEPT max_depth ----------------
    t = time.monotonic()
    try:
        check = _goal_satisfiability_check(
            engine=engine,
            goal=goal if callable(goal) else None,
            start_grid=root,
            max_depth=depth,
        )
        out["goal_satisfiable"] = bool(check.get("satisfiable"))
        out["goal_kind"] = str((check.get("counterexample") or {}).get("kind") or "satisfiable")
        for k in (
            "engine_calls",
            "engine_errors",
            "max_nodes",
            "max_depth",
            "termination",
            "frontier_remaining",
            "depth_truncated_nodes",
            "reachable_grids_evaluated",
        ):
            if k in check:
                out[f"goal_{k}"] = check[k]
    except Exception as exc:  # noqa: BLE001
        out["goal_satisfiable"] = False
        out["goal_kind"] = f"gate_raised:{type(exc).__name__}"
        out["goal_error"] = str(exc)[:240]
    out["goal_wall_s"] = round(time.monotonic() - t, 3)

    # ---- criterion (iii): the planner, shipped max_nodes, SAME SWEPT max_depth -------------
    t = time.monotonic()
    diag: dict = {}
    try:
        plan = e3.plan_in_model(
            engine,
            goal if callable(goal) else None,
            root,
            max_depth=depth,
            diagnostics=diag,
        )
        out["plan_found"] = bool(plan)
        out["plan_length"] = len(plan) if plan else 0
    except Exception as exc:  # noqa: BLE001
        out["plan_found"] = False
        out["plan_length"] = 0
        out["plan_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    out["plan_diagnostics"] = {
        k: v for k, v in diag.items() if isinstance(v, (int, float, str, bool))
    }
    out["plan_wall_s"] = round(time.monotonic() - t, 3)

    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
