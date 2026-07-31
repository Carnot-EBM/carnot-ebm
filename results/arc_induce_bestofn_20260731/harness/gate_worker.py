#!/usr/bin/env python3
"""BEST-OF-N, STEP 4a -- run the goal gate and the planner for ONE candidate, in isolation.

WHY THIS IS A SEPARATE PROCESS rather than a function the scorer calls in-line.

Criteria (ii) and (iii) execute LLM-WRITTEN CODE inside a search loop -- up to `max_nodes`
engine calls for the gate and `max_nodes` more for the planner. Two failure modes make in-process
evaluation unsafe, and both are realistic rather than theoretical:

  * NON-TERMINATION. `_goal_satisfiability_check` and `plan_in_model` bound the SEARCH, not the
    engine: a single `engine()` call that loops forever is unbounded, and both functions wrap
    engine invocation in `except Exception: continue`, so a signal-based in-process alarm would
    be SWALLOWED by the very code it is meant to interrupt and the timeout would silently become
    a skipped node. A subprocess killed from outside cannot be swallowed.
  * MEMORY / INTERPRETER DAMAGE. Generated code has already been observed in this project to be
    mechanically defective in 22 of 36 attempts; an allocation blow-up or a C-level crash in a
    numpy call takes the whole scoring run with it if it shares the interpreter.

The cost of isolation is one interpreter start per candidate (~1s against a gate that is allowed
minutes), which is not worth optimising away.

WHAT IS AND IS NOT MEASURED HERE. The gate and planner are called at the SHIPPED defaults --
`_goal_satisfiability_check` with no `max_nodes` argument (so it reads
`_goal_gate_max_nodes_default()`, exactly as `execute_bounded_llm_reinduction` does) and
`plan_in_model` with its own defaults. Nothing is widened. A candidate that fails because the
budget ran out is recorded as `goal_unreached_within_budget` / `goal_unreached_within_depth`,
NOT as a degenerate predicate -- that distinction was fixed in the repo on 2026-07-31 and
throwing it away here would reintroduce the mislabel one level up.

`goal_energy` is deliberately NOT supplied to `plan_in_model`. The live stall path calls
`self._guided_plan_in_model(e3.plan_in_model)`, which may install a goal-energy heuristic; using
the plain FIFO BFS here makes criterion (iii) a LOWER BOUND on what the live planner could find,
which is the conservative direction for a phase whose headline may be "(iii) yield is zero". A
best-first search can only reach the goal in FEWER nodes, never in more.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time

REPO = "/home/ianblenke/github.com/ianblenke/carnot"
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("CARNOT_ARC_E3_DIR", "/tmp/arc_bon_gate/e3")
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)
sys.path.insert(0, os.path.join(REPO, "python"))


def main() -> int:
    code_path, root_path = sys.argv[1], sys.argv[2]
    out: dict = {"status": "ok"}

    import numpy as np

    from carnot.agentic.arc_executable_world_model import plan_in_model
    from carnot.agentic.arc_llm_reinduction import _goal_satisfiability_check

    code = open(code_path).read()
    with open(root_path, "rb") as fh:
        root = np.asarray(pickle.load(fh))

    ns: dict = {"np": np, "numpy": np}
    try:
        exec(compile(code, code_path, "exec"), ns)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        out["status"] = f"unrunnable:{type(exc).__name__}"
        out["error"] = str(exc)[:240]
        print(json.dumps(out))
        return 0
    engine = ns.get("engine")
    goal = ns.get("is_level_complete")
    out["has_engine"] = callable(engine)
    out["has_goal"] = callable(goal)
    if not callable(engine):
        out["status"] = "no_engine"
        print(json.dumps(out))
        return 0

    # ---- criterion (ii): the goal gate, at the SHIPPED budget ---------------------------
    t = time.monotonic()
    try:
        check = _goal_satisfiability_check(
            engine=engine, goal=goal if callable(goal) else None, start_grid=root
        )
        out["goal_satisfiable"] = bool(check.get("satisfiable"))
        out["goal_kind"] = str((check.get("counterexample") or {}).get("kind") or "satisfiable")
        for k in (
            "reachable_grids_evaluated",
            "engine_calls",
            "engine_errors",
            "max_nodes",
            "max_depth",
            "termination",
            "frontier_remaining",
            "depth_truncated_nodes",
        ):
            if k in check:
                out[f"goal_{k}"] = check[k]
    except Exception as exc:  # noqa: BLE001
        out["goal_satisfiable"] = False
        out["goal_kind"] = f"gate_raised:{type(exc).__name__}"
        out["goal_error"] = str(exc)[:240]
    out["goal_wall_s"] = round(time.monotonic() - t, 2)

    # ---- criterion (iii): the planner, at the SHIPPED budget ----------------------------
    t = time.monotonic()
    diag: dict = {}
    try:
        plan = plan_in_model(engine, goal if callable(goal) else None, root, diagnostics=diag)
        out["plan_found"] = bool(plan)
        out["plan_length"] = len(plan) if plan else 0
    except Exception as exc:  # noqa: BLE001
        out["plan_found"] = False
        out["plan_length"] = 0
        out["plan_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    out["plan_diagnostics"] = {k: v for k, v in diag.items() if isinstance(v, (int, float, str, bool))}
    out["plan_wall_s"] = round(time.monotonic() - t, 2)

    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
