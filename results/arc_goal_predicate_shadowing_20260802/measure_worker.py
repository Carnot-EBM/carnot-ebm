"""Run ONE (cell, definition) through the SHIPPED goal gate. One process, killable.

INVOKED AS: measure_worker.py <analysis.json> <cell> <def_index> <root_pkl> <out_json>

WHY A SEPARATE PROCESS PER PREDICATE, NON-NEGOTIABLE. The code under test is
LLM-generated and unreviewed. Three of the definitions in this corpus reach the end of
their body without returning; one raises `NameError` on a variable it never bound; and
several run unbounded `while queue:` loops whose termination depends on a `visited` set
the model may or may not have updated correctly. `exec`ing any of that in the
measuring interpreter risks a non-terminating loop taking the whole sweep with it --
which is exactly what cost 13 minutes on a previous run of this project. The parent
enforces the wall clock with `subprocess.run(timeout=...)` and `kill()`, because a
signal-based timeout inside the worker cannot interrupt a tight numpy loop that never
yields to the interpreter's signal check.

WHAT IS MEASURED. `arc_llm_reinduction._goal_satisfiability_check` -- the SHIPPED gate,
imported, not reimplemented. It answers: starting from `root`, using THIS FILE'S OWN
engine, does a bounded search reach any grid where the predicate is True? Its verdict
is the comparison, so a predicate is graded by the same machinery that would gate it in
the live agent. The two definitions of a cell are graded against the SAME engine and
the SAME root, so the only thing that varies between the two runs is the predicate.

WHY THE ENGINE IS HELD FIXED. Both definitions ship inside one file next to one engine.
Swapping the predicate alone isolates the variable this task is about. It also means a
cell whose engine is broken yields `engine_unusable` for BOTH arms and contributes
nothing to the comparison in either direction, rather than silently scoring 0-0 as
though the predicates had been tested.
"""

from __future__ import annotations

import ast
import json
import pickle
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "python"))

GOAL_FN = "is_level_complete"


def _source_with_only(code: str, keep_index: int) -> str:
    """The file, with every top-level `is_level_complete` REMOVED except number `keep_index`.

    Surgery by line-range on the original text rather than `ast.unparse`, so comments,
    formatting and every unrelated helper survive byte-identically. Only the competing
    definition's own lines are dropped; nothing else in the module moves.
    """
    tree = ast.parse(code)
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == GOAL_FN]
    drop: set[int] = set()
    for i, fn in enumerate(defs):
        if i == keep_index:
            continue
        # `decorator_list` can start above `lineno`; include it so nothing dangles.
        start = min([fn.lineno] + [d.lineno for d in fn.decorator_list])
        for ln in range(start, (fn.end_lineno or fn.lineno) + 1):
            drop.add(ln)
    lines = code.splitlines()
    return "\n".join(text for n, text in enumerate(lines, start=1) if n not in drop) + "\n"


def main() -> int:
    analysis_path, cell, def_index_s, root_pkl, out_json = sys.argv[1:6]
    def_index = int(def_index_s)
    out: dict[str, Any] = {"cell": cell, "def_index": def_index}

    analysis = json.loads(Path(analysis_path).read_text())
    row = next(
        r for r in analysis["rows"] if r["cell"] == cell and r["corpus"] == "ab_change_fidelity"
    )
    code = (REPO / row["path"]).read_text()
    trimmed = _source_with_only(code, def_index)

    ns: dict[str, Any] = {"__name__": "candidate_world_model"}
    try:
        exec(compile(trimmed, f"<{cell}#def{def_index}>", "exec"), ns)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        out["outcome"] = "module_exec_failed"
        out["error"] = f"{type(exc).__name__}: {exc}"[:200]
        Path(out_json).write_text(json.dumps(out) + "\n")
        return 0

    engine = ns.get("engine")
    goal = ns.get(GOAL_FN)
    out["engine_present"] = callable(engine)
    out["goal_present"] = callable(goal)
    if not callable(engine):
        out["outcome"] = "engine_unusable"
        Path(out_json).write_text(json.dumps(out) + "\n")
        return 0

    root = pickle.loads(Path(root_pkl).read_bytes())

    # A cheap direct call FIRST, so "the predicate is not even a predicate" is reported as
    # itself rather than as whatever the gate happens to say about it. The gate maps an
    # exception to `goal_predicate_error` and a None return to falsey, which would blur the
    # three distinct defects (raises / returns None / returns False) into one bucket.
    try:
        raw = goal(root) if callable(goal) else None
        out["root_call"] = {
            "ok": True,
            "returned_type": type(raw).__name__,
            "is_bool_like": isinstance(raw, (bool, int)) or hasattr(raw, "item"),
            "value": bool(raw) if raw is not None else None,
            "returned_none": raw is None,
        }
    except Exception as exc:  # noqa: BLE001
        out["root_call"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]}

    from carnot.agentic.arc_llm_reinduction import _goal_satisfiability_check

    try:
        res = _goal_satisfiability_check(engine=engine, goal=goal, start_grid=root)
        out["outcome"] = "gate_ran"
        out["satisfiable"] = bool(res.get("satisfiable"))
        out["reachable_grids_evaluated"] = int(res.get("reachable_grids_evaluated") or 0)
        ce = res.get("counterexample") or {}
        out["counterexample_kind"] = ce.get("kind")
    except Exception as exc:  # noqa: BLE001
        out["outcome"] = "gate_raised"
        out["error"] = f"{type(exc).__name__}: {exc}"[:200]

    Path(out_json).write_text(json.dumps(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
