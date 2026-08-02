"""Step 1 (dynamic half): grade BOTH definitions of every double-definition cell.

For each of the 23 cells this launches two worker subprocesses -- one per competing
`is_level_complete` -- against the SAME engine and the SAME root grid, and records what
the shipped goal gate says about each. The parent owns the wall clock: `timeout=` plus
`kill()` on expiry, because the code being graded is unreviewed LLM output and at least
one definition in this corpus runs an unbounded loop.

`--root-source` selects which start grid to search from:
  `arcade`   (default) the opening board from the offline `environment_files` env,
             available for all 20 games, captured by `capture_roots.py`.
  `captured` `E3AgentPolicy.root_grid` as recorded during a real bounded-progress run,
             available for 3 of the 20 games. This is what the LIVE planner searches
             from, so it is the faithful one -- just far scarcer.
Running both is the robustness check: if the verdict flips with the root source, the
`arcade` numbers do not support a conclusion and the artifact must say so.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CAPTURED_ROOTS = REPO / "results" / "arc_induce_bestofn_20260731" / "harness" / "capture"

# One predicate on one 64x64 board under a 20000-node gate budget is seconds of work when
# it terminates at all. Anything past this is a non-terminating generated loop, which is a
# RESULT (recorded as `timeout`), not an error to retry.
PER_CALL_TIMEOUT_S = 120


def _root_for(game: str, source: str) -> Path | None:
    if source == "arcade":
        p = HERE / "roots" / f"{game}.pkl"
        return p if p.exists() else None
    p = CAPTURED_ROOTS / game / "root_grid1.pkl"
    return p if p.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-source", choices=("arcade", "captured"), default="arcade")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    analysis_path = HERE / "analysis.json"
    analysis = json.loads(analysis_path.read_text())
    cells = [
        r
        for r in analysis["rows"]
        if r["corpus"] == "ab_change_fidelity" and (r.get("n_goal_defs") or 0) > 1
    ]
    scratch = Path(
        "/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/"
        "87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/gps/work"
    )
    scratch.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for row in cells:
        root = _root_for(row["game"], args.root_source)
        if root is None:
            results.append({"cell": row["cell"], "game": row["game"], "skipped": "no_root_grid"})
            continue
        entry: dict[str, Any] = {
            "cell": row["cell"],
            "game": row["game"],
            "root_source": args.root_source,
            "root_pkl": str(root.relative_to(REPO)),
            "bound_index": row["bound_index"],
            "arms": [],
        }
        for i, d in enumerate(row["defs"]):
            out_json = scratch / f"{row['cell']}__def{i}__{args.root_source}.json"
            t0 = time.monotonic()
            cmd = [
                sys.executable,
                str(HERE / "measure_worker.py"),
                str(analysis_path),
                row["cell"],
                str(i),
                str(root),
                str(out_json),
            ]
            try:
                proc = subprocess.run(  # noqa: S603
                    cmd, timeout=PER_CALL_TIMEOUT_S, capture_output=True, text=True
                )
                elapsed = time.monotonic() - t0
                if out_json.exists():
                    arm = json.loads(out_json.read_text())
                else:
                    arm = {
                        "outcome": "worker_no_output",
                        "returncode": proc.returncode,
                        "stderr": proc.stderr[-300:],
                    }
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - t0
                arm = {"outcome": "timeout", "timeout_s": PER_CALL_TIMEOUT_S}
            arm.update(
                {
                    "def_index": i,
                    "role": d["role"],
                    "classification": d["classification"],
                    "constant_false": d["constant_false"],
                    "static_defects": [x["kind"] for x in d["static_defects"]],
                    "elapsed_s": round(elapsed, 2),
                }
            )
            entry["arms"].append(arm)
        results.append(entry)
        tags = [f"{a['role']}={a.get('outcome')}/{a.get('satisfiable')}" for a in entry["arms"]]
        print(f"{row['cell']:24s} {' '.join(tags)}", flush=True)

    out = Path(args.out) if args.out else HERE / f"gate_{args.root_source}.json"
    out.write_text(json.dumps({"root_source": args.root_source, "cells": results}, indent=2) + "\n")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
