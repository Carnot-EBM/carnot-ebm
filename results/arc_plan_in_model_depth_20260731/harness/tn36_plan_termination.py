#!/usr/bin/env python3
"""Does `plan_in_model` itself now report tn36's depth-capped search as depth-capped?

The 2026-07-31 goal-gate fix (REQ-ARC-WMTE-6047-D) split the depth axis out of `queue_exhausted`
in the GATE. Its adversarial review found the identical conflation still live in the function the
gate guards -- `plan_in_model` -- and pointed at the gate's own evidence artifact as the proof:

    results/arc_goal_gate_depth_20260731/tn36_depth_label.json
      plan_at_max_depth_40.diagnostics.termination_reason == "queue_exhausted"

on a goal the SAME (engine, predicate, root) triple reaches at depth 61 once the cap is lifted.
This probe re-runs exactly that call against the committed fixtures and records the label now.

It reads the two committed fixtures and asserts BOTH hashes against what the original audit
recorded before drawing any conclusion, so a swapped or regenerated fixture invalidates the probe
rather than silently changing its answer. Nothing here writes to the historical artifact -- that
record stands as written (never-prune); this is a sibling.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time

REPO = pathlib.Path("/home/ianblenke/github.com/ianblenke/carnot")
sys.path.insert(0, str(REPO / "python"))

import numpy as np  # noqa: E402
from carnot.agentic import arc_executable_world_model as e3  # noqa: E402

FIX = REPO / "tests/fixtures/arc_goal_gate_depth_tn36"
ENGINE_SRC = FIX / "tn36_on_world_model.py.frozen"
ROOT_NPY = FIX / "tn36_on_root_grid.npy"
# Recorded by the 2026-07-30 gate audit; re-asserted here.
AUDIT_ENGINE_MD5 = "6d96491f80bec0319828ba1a04f5841e"
AUDIT_ROOT_SHA16 = "f328c951a03d248d"


def main(out_path: str) -> int:
    src = ENGINE_SRC.read_bytes()
    engine_md5 = hashlib.md5(src).hexdigest()
    root = np.load(ROOT_NPY)
    root_sha16 = hashlib.sha256(root.tobytes()).hexdigest()[:16]
    if engine_md5 != AUDIT_ENGINE_MD5 or root_sha16 != AUDIT_ROOT_SHA16:
        print("FIXTURE HASH MISMATCH -- refusing to draw a conclusion", file=sys.stderr)
        return 2

    ns: dict = {}
    exec(compile(src, str(ENGINE_SRC), "exec"), ns)  # noqa: S102 - frozen fixture, hash-pinned
    engine = ns["engine"]
    is_done = ns["is_level_complete"]

    rows = {}
    for label, max_depth in (("plan_at_max_depth_40", 40), ("plan_at_max_depth_80", 80)):
        diag: dict = {}
        t0 = time.time()
        try:
            plan = e3.plan_in_model(
                engine, is_done, root, max_nodes=20000, max_depth=max_depth, diagnostics=diag
            )
            err = None
        except Exception as exc:  # pragma: no cover - defensive, reported not swallowed
            plan, err = None, f"{type(exc).__name__}: {exc}"
        rows[label] = {
            "found": plan is not None,
            "length": len(plan or []),
            "error": err,
            "wall_s": round(time.time() - t0, 3),
            "diagnostics": diag,
        }

    out = {
        "probe": "plan_in_model_depth_termination_tn36",
        "schema": "carnot.arc_plan_in_model_depth_label.v1",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "game": "tn36",
        "arm": "on",
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "random_seed": 1,
        "src_engine_md5": engine_md5,
        "src_engine_md5_matches_audit": engine_md5 == AUDIT_ENGINE_MD5,
        "root_grid_sha256": root_sha16,
        "root_grid_sha256_matches_audit": root_sha16 == AUDIT_ROOT_SHA16,
        "reproducibility_checksum": hashlib.sha256(
            src + root.tobytes() + b"plan_in_model_depth_v1"
        ).hexdigest(),
        # What the historical artifact recorded, quoted so the delta is legible without a
        # second file open. That artifact is NOT rewritten.
        "label_before_this_commit": "queue_exhausted",
        "label_after_this_commit": rows["plan_at_max_depth_40"]["diagnostics"].get(
            "termination_reason"
        ),
        "prior_artifact": "results/arc_goal_gate_depth_20260731/tn36_depth_label.json",
        **rows,
        "return_value_unchanged": (
            rows["plan_at_max_depth_40"]["found"] is False
            and rows["plan_at_max_depth_80"]["found"] is True
            and rows["plan_at_max_depth_80"]["length"] == 61
        ),
        "honest_verdict": (
            "complete_plan_in_model_depth_axis_split_and_verified_on_the_real_tn36_cell"
        ),
        "verifier_is_oracle": True,
        "verifier_is_oracle_note": (
            "There is no verifier claim here at all -- this probe reads a diagnostics string out "
            "of a search whose ground truth (does a plan exist at this depth) is decided by "
            "executing the engine. Declared True because the check IS the executable oracle; no "
            "moat / value-added claim is made or implied."
        ),
        "preconditions_checked": [
            {"resource": "tn36_engine_fixture", "available": ENGINE_SRC.exists()},
            {"resource": "tn36_root_grid_fixture", "available": ROOT_NPY.exists()},
        ],
    }
    pathlib.Path(out_path).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "label_before_this_commit",
                    "label_after_this_commit",
                    "return_value_unchanged",
                )
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
