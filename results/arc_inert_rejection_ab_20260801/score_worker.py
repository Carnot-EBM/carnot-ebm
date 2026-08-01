"""Score ONE emitted engine, in its OWN process, so a hostile engine can be killed from outside.

WHY A WORKER AT ALL. `WorldModelVerifier.score` and the state-graph probe both EXECUTE generated
code, and unlike `arc_engine_static_validation`'s dry run neither is internally bounded. A
non-terminating induced engine wedged a generation loop for 13 minutes on 2026-07-31; the same
code reached from a scoring pass would wedge it just as thoroughly. `score.py` runs this file
with `subprocess.run(timeout=...)`, so a hang becomes a recorded `worker_timeout` status -- a
MISSING OBSERVATION, never a zero.

It reads its job from a JSON path and prints one JSON line. Nothing is imported from this file.
"""

from __future__ import annotations

import json
import os
import pathlib
import pickle
import sys
import time

REPO = pathlib.Path(os.environ["CARNOT_REPO"])
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# This process runs untrusted code and needs no accelerator; denying it one also stops a generated
# engine from grabbing a card another session on this shared machine owns.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CARNOT_ARC_E3_DIR", "/tmp/arc_inert_rejection_ab/e3")
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)
sys.path.insert(0, str(REPO / "python"))

STATE_PROBE_MAX_CALLS = 600


def _state_graph_probe(engine, root, e3, np) -> dict:
    """Is the engine's reachable state graph a TREE or a PATH?

    COPIED FROM `results/arc_metric_validity_20260801/score_worker.py`, and the copy is checked
    rather than asserted: `test_probe_is_the_measure_that_was_validated` compares this function's
    executable AST against that file's, docstrings excluded, and fails the build if they diverge.
    That test exists because the first draft of this file SAID "copied verbatim" while having
    renamed `n_distinct_changing_successors_at_root`, dropped `engine_changes_anything_at_root`
    (itself a reported predictor at AUC 0.694) and reshaped the error field -- a docstring claim
    the code did not honour, in the one place where the whole value of the code is that it is
    the same code. Caught by an adversarial review pass, not by writing it.

    Being the same code is the point: `probe_depth_reached` is the measure that run found PREDICTS
    plannability (pooled AUC 0.787, game-clustered CI [0.675, 0.859] excluding chance,
    FWER-adjusted p = 0.0030 over a family of 14) where `change_fidelity` does not (AUC 0.609,
    CI [0.381, 0.751], containing chance). Re-implementing it here from the description would
    produce a number that is not comparable to the one that was validated, which is the whole
    reason for quoting it. Its own docstring, preserved:

        WHY THIS IS A FIRST-CLASS PREDICTOR AND NOT A CURIOSITY. `plan_max_depth_default`'s own
        measurement records that tn36's 32 changing root actions collapse to ONE distinct
        successor -- the engine fills the next cell wherever the click lands -- and both search
        sites dedup by state key, so the search tree degenerates to a PATH. On a path the node
        budget buys depth one-for-one and only `max_depth` can stop the search, which is why
        raising the horizon and not `max_nodes` was the correct fix there. That makes branching a
        structural determinant of whether a plan is findable at all, entirely separate from how
        accurate the engine is.

        Bounded at `STATE_PROBE_MAX_CALLS` engine calls, far below the planner's 20000, so this
        is a cheap shape probe and not a second search.
    """
    out: dict = {}
    try:
        root = np.asarray(root)
        seen = {e3._state_key(root)}  # noqa: SLF001
        frontier = [root]
        calls = 0
        expanded = 0
        depth = 0
        new_per_expansion: list[int] = []
        root_successors: set = set()
        root_changing_successors: set = set()
        root_key = e3._state_key(root)  # noqa: SLF001
        while frontier and calls < STATE_PROBE_MAX_CALLS:
            nxt = []
            for grid in frontier:
                if calls >= STATE_PROBE_MAX_CALLS:
                    break
                expanded += 1
                fresh = 0
                for c in e3._model_candidates(grid):  # noqa: SLF001
                    if calls >= STATE_PROBE_MAX_CALLS:
                        break
                    try:
                        ng = np.asarray(engine(grid.copy(), c["action"], c["data"]))
                    except Exception:  # noqa: BLE001
                        continue
                    calls += 1
                    if ng.shape != root.shape:
                        continue
                    k = e3._state_key(ng)  # noqa: SLF001
                    if depth == 0:
                        root_successors.add(k)
                        if k != root_key:
                            root_changing_successors.add(k)
                    if k in seen:
                        continue
                    seen.add(k)
                    fresh += 1
                    nxt.append(ng)
                new_per_expansion.append(fresh)
            frontier = nxt
            depth += 1
        out["probe_engine_calls"] = calls
        out["probe_nodes_expanded"] = expanded
        out["probe_distinct_states"] = len(seen)
        out["probe_depth_reached"] = depth
        # The headline shape number: how many NEW states an expansion yields on average. ~1.0 is
        # a path (depth-bound), >>1 is a tree (breadth-bound), 0.0 is inert.
        out["probe_mean_new_states_per_expansion"] = (
            round(sum(new_per_expansion) / len(new_per_expansion), 4) if new_per_expansion else None
        )
        out["n_distinct_successors_at_root"] = len(root_successors)
        out["n_distinct_changing_successors_at_root"] = len(root_changing_successors)
        out["engine_changes_anything_at_root"] = bool(root_changing_successors)
        out["probe_status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        out["probe_status"] = f"raised:{type(exc).__name__}"
        out["probe_error"] = str(exc)[:200]
    return out


def main() -> int:
    t0 = time.time()
    job = json.loads(pathlib.Path(sys.argv[1]).read_text())
    import numpy as np

    from carnot.agentic import arc_executable_world_model as e3

    row: dict = {"cell_id": job["cell_id"], "status": "ok"}
    with open(job["window_pkl"], "rb") as fh:
        win = pickle.load(fh)  # noqa: S301 - written by this run's own collection pass
    held = win["held"]

    code = pathlib.Path(job["engine_path"]).read_text()
    ns: dict = {}
    try:
        exec(compile(code, job["engine_path"], "exec"), ns)  # noqa: S102 - THE POINT of this process
    except Exception as exc:  # noqa: BLE001
        row["status"] = "engine_does_not_import"
        row["error"] = f"{type(exc).__name__}: {exc}"[:200]
        print(json.dumps(row, default=str))
        return 0
    engine = ns.get("engine")
    if not callable(engine):
        row["status"] = "no_engine_symbol"
        print(json.dumps(row, default=str))
        return 0

    try:
        vr = e3.WorldModelVerifier(list(held)).score(engine)
        row["heldout"] = {
            "measurable": True,
            "n": int(vr.n),
            "n_correct": int(vr.n_correct),
            "accuracy": round(float(vr.accuracy), 6),
            "cell_recall": round(float(vr.cell_recall), 6),
            "n_changing": int(vr.n_changing),
            "n_changes_correct": int(vr.n_changes_correct),
            "change_accuracy": round(float(vr.change_accuracy), 6),
            "change_fidelity": round(float(vr.change_fidelity), 6),
            "correct_changed_cells": int(vr.correct_changed_cells),
            "spurious_changed_cells": int(vr.spurious_changed_cells),
            "error": vr.error,
        }
    except Exception as exc:  # noqa: BLE001
        row["heldout"] = {"measurable": False, "error": f"{type(exc).__name__}: {exc}"[:200]}

    # THE PROBE ROOT, named explicitly because it is a harness choice. `held[0].grid` is the state
    # immediately after the evidence the prompt showed -- where an agent that had just induced
    # this engine would actually plan from. It is NOT the same root the metric-validity run used
    # (that run rebuilt windows with their own roots), so probe_depth values here are comparable
    # WITHIN this experiment and should not be pooled with that one's.
    root = np.asarray(held[0].grid) if len(held) else None
    row["state_graph"] = _state_graph_probe(engine, root, e3, np) if root is not None else None
    row["worker_wall_s"] = round(time.time() - t0, 2)
    print(json.dumps(row, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
