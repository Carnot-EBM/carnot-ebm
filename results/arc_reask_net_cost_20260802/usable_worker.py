#!/usr/bin/env python3
"""GUARDRAIL worker: decide whether ONE cell's emitted engine is USABLE, in a killable process.

WHAT "USABLE" MEANS HERE, stated precisely because the whole primary rests on it. It is exactly
`arc_engine_static_validation.validate_engine_code(...) == []` -- the SAME definition the shipped
gate itself uses to decide whether to re-ask. That is deliberate: the gate's own historical
headline (13/36 -> 22/36) was counted with this definition, so scoring the arms with it is what
makes this run's `usable` column directly comparable to the number being audited. It is NOT a
quality claim. A clean engine can still be completely wrong about the game; `usable` is not
`good`, and the sibling out-of-sample run found a game where every arm produced clean-but-wrong
engines on 19 of 19 held-out transitions.

WHY POST-HOC RATHER THAN READ OFF THE LIVE RUN. In the gate-OFF arm `_engine_defects` is never
called at all, so there is no live number to read. Scoring every arm's FINAL emitted engine with
one identical scorer is the only symmetric option.

ONE DECLARED ASYMMETRY vs the live gate: `stop_type` is not available post-hoc (induce makes
several calls and only the last one's is retained), so it is passed as None and the
`truncated_before_required_symbols` check cannot fire here. Applied IDENTICALLY to every arm, so
it cannot confound the contrast -- it can only make every arm's `usable` count slightly generous
in the same direction.

SUBPROCESS FOR THE USUAL REASON: this executes LLM-written `engine()` code. A non-terminating
engine wedged a sibling run for 13 minutes. A hang here costs one cell and nothing else.
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
# The scorer must never inherit an arm's knobs: `engine_inertness_defect` is opt-in via
# CARNOT_ARC_INDUCE_REJECT_INERT and would change the DEFINITION of usable between cells if it
# leaked in from the driver's environment.
os.environ.pop("CARNOT_ARC_INDUCE_REJECT_INERT", None)
sys.path.insert(0, str(REPO / "python"))


def main() -> int:
    job = json.loads(pathlib.Path(sys.argv[1]).read_text())
    from carnot.agentic import arc_engine_static_validation as sv

    with open(job["window_pkl"], "rb") as fh:
        shown = pickle.load(fh)["shown"]

    code = pathlib.Path(job["engine_path"]).read_text()
    try:
        defects = sv.validate_engine_code(
            code,
            transitions=list(shown),
            stop_type=None,  # see the module docstring: declared, identical across arms
            required=("engine",),
            budget=int(job["budget"]),
        )
    except Exception as exc:  # noqa: BLE001
        # A scorer that cannot run is a MISSING OBSERVATION, never a "defective" verdict.
        # Conflating "I could not check" with "I checked and it is broken" is how a guard
        # starts inventing findings.
        print(json.dumps({"scored": False, "reason": f"{type(exc).__name__}: {exc}"[:200]}))
        return 0
    kinds = sorted({d.kind for d in defects})
    print(json.dumps({"scored": True, "usable": not kinds, "defect_kinds": kinds}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
