#!/usr/bin/env python3
"""BEST-OF-N -- run the SHIPPED mechanical defect check for ONE candidate, in isolation.

WHY THIS EXISTS, and it is a finding rather than a convenience.

`arc_engine_static_validation.validate_engine_code` ends in `dry_run_defects`, which EXECUTES the
LLM-written `engine()` against the observed transitions. That call is bounded in the number of
transitions (`limit=25`) but NOT in wall-clock, and the generated code is arbitrary Python. On
2026-07-31 this harness's generation loop wedged on ft09 candidate 5: the process sat in state R
at 32% CPU for 13 minutes with no open socket and both GPUs idle, having already received the
completion from the server. It was spinning inside the dry run of a generated engine that does
not terminate.

The SHIPPED induce path runs the same check in-process, on the same kind of code, with the same
absence of a timeout -- `generate()` calls `validate_engine_code` directly. So this is not a
harness-only hazard: a non-terminating induced engine hangs the live agent's induction, and no
existing timeout catches it (the server-side call has already returned, so the HTTP timeout is
irrelevant, and the agent's own wall budget is checked between actions rather than inside one).

A subprocess killed from OUTSIDE cannot be swallowed. An in-process signal alarm could be:
`dry_run_defects` catches broad exceptions around the engine invocation, so a SIGALRM-raised
exception inside the engine call would be absorbed and silently recorded as a passing check --
turning a hang into a false CLEAN, which is worse than the hang.

A candidate whose dry run does not terminate is recorded as `validation_timeout` and is NOT
usable. That is the honest reading: an engine the pipeline cannot even finish checking is not a
candidate the pipeline could have used.
"""

from __future__ import annotations

import json
import os
import pickle
import sys

REPO = "/home/ianblenke/github.com/ianblenke/carnot"
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("CARNOT_ARC_E3_DIR", "/tmp/arc_bon_validate/e3")
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)
sys.path.insert(0, os.path.join(REPO, "python"))


def main() -> int:
    code_path, shown_path, stop_type, budget = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

    from carnot.agentic import arc_engine_static_validation as sv

    code = open(code_path).read()
    with open(shown_path, "rb") as fh:
        shown = pickle.load(fh)

    defects = sv.validate_engine_code(
        code,
        transitions=shown,
        stop_type=stop_type or None,
        required=("engine", "is_level_complete"),
        budget=budget,
    )
    try:
        import ast

        ast.parse(code)
        parses = True
    except SyntaxError:
        parses = False
    out = {
        "defect_kinds": sorted({d.kind for d in defects}),
        "defect_details": [d.detail[:240] for d in defects],
        "parses": parses,
        "engine_changes_anything": sv.engine_changes_anything(code, shown),
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
