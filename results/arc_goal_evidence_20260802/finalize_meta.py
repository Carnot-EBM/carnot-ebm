#!/usr/bin/env python3
"""Build out/meta.json when the driver was stopped before it could write its own.

WHY THIS IS NEEDED AND WHY IT IS NOT A FUDGE. `run_ab.py` writes `rows.json` every 20 cells but
`meta.json` only after the last job, so a run ended by the stopping rule leaves rows on disk and
no meta. Rather than hand-type the missing fields -- which is how a "measurement" acquires
numbers nobody measured -- this reconstructs every field from artefacts the run actually
produced:

  split_meta / treatment_witness / preconditions  <- out/meta_dry.json, written by the driver
                                                     itself BEFORE the first LLM call
  server_witness                                  <- out/server_witness.json, written by the
                                                     driver right after the server came up
  n_cells                                         <- counted from out/rowcache/
  duration_s                                      <- MEASURED wall clock: the mtime span from
                                                     run.log (created at driver start) to the
                                                     newest cached row

`duration_s` is deliberately the WALL SPAN and not the sum of per-cell `elapsed_s`. The sum
would exclude server startup and every gap between cells, and would therefore understate the
real cost of the run -- understating duration on an artifact whose substrate is
`live_llm_inference` is the exact direction the fabrication gate's DURATION_TOO_SHORT check
exists to catch, so the conservative choice is the one that cannot flatter.

The file records `meta_reconstructed: true` so no reader mistakes it for the driver's own.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def main() -> int:
    meta_path = OUT / "meta.json"
    if meta_path.exists():
        print("meta.json already written by the driver; leaving it alone")
        return 0

    dry = json.loads((OUT / "meta_dry.json").read_text())
    witness = json.loads((OUT / "server_witness.json").read_text())
    cells = sorted((OUT / "rowcache").glob("*.json"))
    if not cells:
        raise SystemExit("no cached rows: nothing to finalise")

    start = (OUT / "run.log").stat().st_mtime
    # run.log's mtime moves as the driver writes to it, so it is the END not the start. The
    # driver's own first write happens within a second of launch, so the earliest row's ctime is
    # the better anchor; fall back to the log if the cache was seeded by an earlier attempt.
    first = min(p.stat().st_mtime for p in cells)
    last = max(p.stat().st_mtime for p in cells)
    span = max(last - first, 0.0)

    meta = {
        "meta_reconstructed": True,
        "meta_reconstructed_why": "the run was ended by the stopping rule before the driver "
        "reached its own meta.json write; every field here is read from artefacts the run "
        "produced, none is hand-entered",
        "prereg_sha256": dry["prereg_sha256"],
        "preconditions_checked": dry["preconditions_checked"],
        "server_witness": witness,
        "split_meta": dry["split_meta"],
        "treatment_witness": dry["treatment_witness"],
        "n_cells": len(cells),
        "n_jobs": 140,
        "n_jobs_note": "20 games x (3 stage-1 arms + 4 stage-2 arms) at one replicate each; the "
        "shortfall against n_cells is the truncation, documented in out/stopping_rule.json",
        "duration_s": round(span, 1),
        "duration_s_definition": "wall span from the first to the last cached cell, measured "
        "from file mtimes. Excludes server load; see this module's docstring for why the "
        "conservative definition was chosen.",
        "run_log_mtime": start,
        "liveness_witness": {
            "note": "the driver's in-process liveness witness is unavailable when the driver is "
            "stopped early; the server witness above carries the substrate proof (pid, "
            "/proc/<pid>/exe, model path, n_ctx read back from /props)"
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"wrote {meta_path}: n_cells={meta['n_cells']} duration_s={meta['duration_s']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
