#!/usr/bin/env python3
"""PHASE 2, depth probe -- run the SHIPPED gate and planner with ONLY the depth cap moved.

WHAT THIS IS FOR. The greedy rollout in `worker.py` establishes that four tn36 candidates have a
goal predicate their own engine reaches at depth 61. That is a statement about the MODEL, made
with a search of my own construction. It does not by itself prove that the shipped machinery
would certify and plan those goals if its horizon reached that far -- the shipped gate and
planner are breadth-first with their own dedup and their own budgets. This probe closes that gap
by calling `_goal_satisfiability_check` and `plan_in_model` THEMSELVES.

WHAT IS AND IS NOT CHANGED, AND WHY THIS IS NOT GATE-WIDENING. Exactly one parameter moves:
`max_depth`, from its default 40 to a sweep. `max_nodes` stays at the shipped 20000. Every
quality check inside the gate is untouched -- `goal_predicate_true_at_root` still rejects a
trivially-true predicate, `degenerate_goal_predicate` still fires on a genuinely drained
frontier, and a candidate that fails for any reason other than the horizon fails here too.
Moving a HORIZON cannot admit a goal the predicates reject; it can only turn an UNDECIDED
verdict into a DECIDED one, in either direction. That is the same reasoning the repo already
applies to `CARNOT_ARC_GOAL_GATE_MAX_NODES`, which exists precisely so this half of the pipeline
can be budgeted in a diagnostic run.

This is DIAGNOSTIC OUTPUT ONLY. Nothing here is counted as a Phase-1 criterion (ii)/(iii) pass;
the Phase-1 yields stand exactly as reported at the shipped defaults. No production default is
edited, and nothing is proposed for the live path on the strength of this probe alone -- see the
artifact's `what_it_would_take_to_bank_a_level` for the boundary on what it does and does not
license.
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
os.environ.setdefault("CARNOT_ARC_E3_DIR", "/tmp/arc_phase2_depth/e3")
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)
sys.path.insert(0, os.path.join(REPO, "python"))

SHIPPED_DEPTH = 40
DEPTHS = (40, 50, 61, 70, 100)


def main() -> int:
    job = json.loads(open(sys.argv[1]).read())
    import numpy as np

    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic.arc_llm_reinduction import _goal_satisfiability_check

    ns: dict = {"np": np, "numpy": np}
    exec(compile(open(job["code_path"]).read(), job["code_path"], "exec"), ns)  # noqa: S102
    engine, goal = ns.get("engine"), ns.get("is_level_complete")
    with open(job["root_pkl"], "rb") as fh:
        root = np.asarray(pickle.load(fh))

    out: dict = {"game": job["game"], "candidate": job["candidate"], "sweep": []}
    for depth in DEPTHS:
        row: dict = {"max_depth": depth, "is_shipped_default": depth == SHIPPED_DEPTH}
        t = time.monotonic()
        try:
            chk = _goal_satisfiability_check(
                engine=engine, goal=goal, start_grid=root, max_depth=depth
            )
            row["gate_satisfiable"] = bool(chk.get("satisfiable"))
            row["gate_kind"] = str((chk.get("counterexample") or {}).get("kind") or "satisfiable")
            row["gate_first_true_depth"] = chk.get("first_true_depth")
            row["gate_grids_evaluated"] = chk.get("reachable_grids_evaluated")
            row["gate_engine_calls"] = chk.get("engine_calls")
            row["gate_termination"] = chk.get("termination")
        except Exception as exc:  # noqa: BLE001
            row["gate_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        row["gate_wall_s"] = round(time.monotonic() - t, 2)

        t = time.monotonic()
        diag: dict = {}
        try:
            plan = e3.plan_in_model(engine, goal, root, max_depth=depth, diagnostics=diag)
            row["plan_found"] = bool(plan)
            row["plan_length"] = len(plan) if plan else 0
            row["plan_nodes_expanded"] = diag.get("nodes_expanded")
            row["plan_termination"] = diag.get("termination_reason")
        except Exception as exc:  # noqa: BLE001
            row["plan_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
        row["plan_wall_s"] = round(time.monotonic() - t, 2)
        out["sweep"].append(row)

    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
