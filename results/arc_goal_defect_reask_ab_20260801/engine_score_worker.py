#!/usr/bin/env python3
"""GUARDRAIL worker: score ONE cell's induced ENGINE on the held-out tail, in a killable process.

WHY THE GUARDRAIL EXISTS. On the combined induce call ONE answer carries both `engine` and
`is_level_complete`, so a goal-triggered re-ask regenerates the ENGINE as well. If the goal
gets better while the engine gets worse, the intervention is not free, and reporting only the
goal would be reporting half a trade. This scores the engine with the SAME verifier and the
SAME held-out tail the sibling A/B used for change_fidelity, so the two runs' engine numbers
are directly comparable.

Subprocess for the usual reason: this executes LLM-written `engine()` code, and
`engine_changes_anything` is documented as UNBOUNDED with a measured non-terminating case
(ft09_k5). A hang here costs one cell.
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
        held = pickle.load(fh)["held"]

    eng_path = pathlib.Path(job["engine_path"])
    game = eng_path.parent.name
    try:
        engine, _is_done = e3._load_engine_from(eng_path.parent.parent, game)  # noqa: SLF001
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"measurable": False, "reason": f"{type(exc).__name__}: {exc}"[:200]}))
        return 0
    if engine is None:
        # A complete response that produced no usable engine is a REAL ZERO on every
        # similarity metric, not a missing observation -- the model answered, the answer was
        # unusable. Missingness is decided upstream from the server/exception counters.
        print(
            json.dumps(
                {
                    "measurable": True,
                    "engine_loaded": False,
                    "change_fidelity": 0.0,
                    "accuracy": 0.0,
                    "cell_recall": 0.0,
                }
            )
        )
        return 0
    try:
        vr = e3.WorldModelVerifier(list(held)).score(engine)
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"measurable": False, "reason": f"{type(exc).__name__}: {exc}"[:200]}))
        return 0
    print(
        json.dumps(
            {
                "measurable": True,
                "engine_loaded": True,
                "n": int(vr.n),
                "n_correct": int(vr.n_correct),
                "accuracy": round(float(vr.accuracy), 6),
                "cell_recall": round(float(vr.cell_recall), 6),
                "n_changing": int(vr.n_changing),
                "change_accuracy": round(float(vr.change_accuracy), 6),
                "change_fidelity": round(float(vr.change_fidelity), 6),
                "error": vr.error,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
