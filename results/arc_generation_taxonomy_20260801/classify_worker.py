#!/usr/bin/env python3
"""Classify ONE frozen induced-engine candidate with the SHIPPED detector, in a killable process.

THE WHOLE POINT: this calls `arc_engine_static_validation.validate_engine_code` and reports what
it returns. It does NOT reimplement any check. A parallel classifier would measure the classifier,
not the corpus, and the corpus is the thing under study -- the shipped detector's own blind spots
(documented in its FALSE-POSITIVE DIRECTION section) are part of what a taxonomy needs to expose,
and they are only exposed by using it.

WHY A SUBPROCESS, restated because it is not optional. `validate_engine_code` ends in
`dry_run_defects`, which EXECUTES LLM-written code. The detector already runs its dry run in its
own killable child, but two things are still unbounded from here: `engine_changes_anything` (a
separate entry point with NO subprocess of its own -- it calls `_exec_namespace` and then the
engine directly, in-process) and module-level execution inside `_exec_namespace` itself. A
non-terminating generated engine wedged a live loop for 13 minutes on 2026-07-31. So the
classification of one candidate is bounded from OUTSIDE by the driver, and the two nested bounds
compose rather than conflict: the inner one turns a hang into an `engine_nonterminating` defect,
the outer one turns anything the inner one misses into a recorded `worker_timeout` row.

WHAT COMES OUT. One JSON line: the defect kinds, whether the code parses, whether the engine ever
changes anything on transitions it was shown, and the raw byte/line size. Everything else --
grouping, rates, shares -- is the driver's job, on data that is already recorded.
"""

from __future__ import annotations

import ast
import json
import os
import pathlib
import pickle
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CARNOT_ARC_E3_DIR", "/tmp/arc_gentax/e3_validate")
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)
sys.path.insert(0, str(REPO / "python"))


def main() -> int:
    job = json.loads(pathlib.Path(sys.argv[1]).read_text())
    from carnot.agentic import arc_engine_static_validation as sv

    code = pathlib.Path(job["code_path"]).read_text()

    shown = None
    wp = job.get("window_pkl")
    if wp and pathlib.Path(wp).exists():
        with open(wp, "rb") as fh:
            shown = list(pickle.load(fh)["shown"])

    defects = sv.validate_engine_code(
        code,
        transitions=shown,
        stop_type=job.get("stop_type") or None,
        required=("engine", "is_level_complete"),
        budget=job.get("budget"),
    )

    try:
        ast.parse(code)
        parses = True
        parse_error = None
        parse_error_type = None
    except SyntaxError as exc:
        parses = False
        # IndentationError is a SUBCLASS of SyntaxError, and the shipped detector folds both
        # into the single kind `syntax_error`. The distinction matters for the taxonomy (an
        # indentation failure is a different generation pathology from an unmatched brace), so
        # the concrete exception class is recorded HERE rather than by widening the detector.
        parse_error_type = type(exc).__name__
        parse_error = f"{exc.msg} (line {exc.lineno})"

    out = {
        "cell": job["cell"],
        "status": "ok",
        "game": job.get("game"),
        "corpus": job.get("corpus"),
        "code_bytes": len(code.encode("utf-8")),
        "code_lines": code.count("\n") + 1,
        "parses": parses,
        "parse_error_type": parse_error_type,
        "parse_error": parse_error,
        "dry_run_transitions": 0 if shown is None else len(shown),
        "defect_kinds": sorted({d.kind for d in defects}),
        "defects": [
            {
                "kind": d.kind,
                "detail": d.detail[:400],
                "line": d.line,
                "retryable": bool(d.retryable),
                "repairable": bool(d.repairable),
                "evidence": {k: str(v)[:200] for k, v in (d.evidence or {}).items()},
            }
            for d in defects
        ],
        "any_repairable": any(d.repairable for d in defects),
        "any_retryable": any(d.retryable for d in defects),
    }

    # INERTNESS. Recorded, never gated -- the detector's own docstring is emphatic that this is
    # a quality judgement belonging to the trust gate. It is here because "no defects found" and
    # "the engine models nothing" are the two largest classes in this corpus and an operator
    # reading only the defect list would see the first and miss the second.
    if shown is not None:
        out["engine_changes_anything"] = sv.engine_changes_anything(code, shown)
    else:
        out["engine_changes_anything"] = None

    print(json.dumps(out, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
