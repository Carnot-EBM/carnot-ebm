#!/usr/bin/env python3
"""Score every A/B cell's induced goal against REAL OBSERVED FRAMES, and the engine as a guardrail.

SEPARATE FROM run_ab.py ON PURPOSE. Scoring executes LLM-written code, which can hang or
allocate without bound; keeping it out of the GPU driver means a pathological predicate costs
one cell, not the run and not the server. Every evaluation goes through
`goal_score_worker.py` in a killable subprocess with a timeout, and a timeout is recorded as a
MISSING OBSERVATION -- never as a zero.

The outcome definitions live in `preflight_outcomes.outcomes` and are IMPORTED here rather than
restated, so the pre-flight that chose the primary and the run that tests it cannot drift into
computing two different things under one name.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
# Derived, never hardcoded: CLAUDE.md Test-Run Record Integrity rule 4 -- an absolute path
# baked into source means a fresh clone writes into the operator's checkout, which is
# independently a G2 reproducibility defect. This file lives at <repo>/results/<exp>/, so the
# repo root is two parents up.
REPO = HERE.parents[1]
PY = "/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python"
SCRATCH = pathlib.Path(
    "/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/"
    "87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/goalab"
)
TIMEOUT = 120

_spec = importlib.util.spec_from_file_location("_pf", HERE / "preflight_outcomes.py")
_pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pf)
outcomes = _pf.outcomes


def score_goal(row: dict) -> dict:
    game = row["game"]
    eng = pathlib.Path(row["e3_dir"]) / game / "world_model.py"
    if not eng.exists():
        return {"status": "no_engine_file"}
    pkl = SCRATCH / "windows" / f"{game}.pkl"
    if not pkl.exists():
        return {"status": "no_window"}
    tag = f"{game}__r{row['replicate']}__{row['tag']}"
    job = SCRATCH / f"abscore_{tag}.json"
    job.write_text(json.dumps({"engine_path": str(eng), "window_pkl": str(pkl)}))
    try:
        p = subprocess.run(
            [PY, str(HERE / "goal_score_worker.py"), str(job)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            env={"PATH": "/usr/bin:/bin", "HOME": "/home/ianblenke", "JAX_PLATFORMS": "cpu"},
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    for line in reversed((p.stdout or "").strip().splitlines()):
        try:
            return json.loads(line)
        except Exception:  # noqa: BLE001,S112
            continue
    return {"status": "worker_error", "stderr": (p.stderr or "")[-300:]}


def score_engine(row: dict) -> dict:
    """GUARDRAIL. On the combined call one answer carries both functions, so a goal-triggered
    re-ask regenerates the ENGINE too. If the goal improves while the engine degrades, the
    intervention is not free. Uses the SAME verifier the sibling A/B scored change_fidelity
    with, in its own killable subprocess for the same reason."""
    game = row["game"]
    eng = pathlib.Path(row["e3_dir"]) / game / "world_model.py"
    if not eng.exists():
        return {"measurable": False, "reason": "no_engine_file"}
    pkl = SCRATCH / "windows" / f"{game}.pkl"
    tag = f"{game}__r{row['replicate']}__{row['tag']}"
    job = SCRATCH / f"abeng_{tag}.json"
    job.write_text(json.dumps({"engine_path": str(eng), "window_pkl": str(pkl)}))
    try:
        p = subprocess.run(
            [PY, str(HERE / "engine_score_worker.py"), str(job)],
            capture_output=True,
            text=True,
            timeout=300,
            env={
                "PATH": "/usr/bin:/bin",
                "HOME": "/home/ianblenke",
                "JAX_PLATFORMS": "cpu",
                "CARNOT_REPO": str(REPO),
                "CUDA_VISIBLE_DEVICES": "",
            },
        )
    except subprocess.TimeoutExpired:
        return {"measurable": False, "reason": "timeout"}
    for line in reversed((p.stdout or "").strip().splitlines()):
        try:
            return json.loads(line)
        except Exception:  # noqa: BLE001,S112
            continue
    return {"measurable": False, "reason": "worker_error", "stderr": (p.stderr or "")[-300:]}


def one(row: dict) -> dict:
    out = dict(row)
    g = score_goal(row)
    out["goal_raw"] = g
    out["outcomes"] = outcomes(g)
    out["engine"] = score_engine(row)
    return out


def _load_rows() -> list[dict]:
    """Prefer `out/rows.json`, else assemble from the per-cell caches in `out/cells/`.

    WHY THE FALLBACK. `run_ab.py` writes rows.json only at the very END, but it caches every
    cell as it goes. Reading the caches lets an INTERIM analysis run against a job still in
    flight, without signalling, pausing or otherwise touching the running process -- which
    matters on a machine where the GPU window is not ours to schedule.

    A partial read is still a BALANCED design: the job loop is
    `for replicate: for game: for arm`, so truncation drops whole GAMES with all their arms
    rather than leaving an arm short. Cells are dropped, never zero-filled.
    """
    rows_json = HERE / "out" / "rows.json"
    if rows_json.exists():
        return json.loads(rows_json.read_text())
    rows = []
    for p in sorted((HERE / "out" / "cells").glob("*.json")):
        try:
            rows.append(json.loads(p.read_text()))
        except Exception:  # noqa: BLE001,S112
            continue  # a cell mid-write is skipped, not half-read
    return rows


def main() -> int:
    rows = _load_rows()
    with ThreadPoolExecutor(max_workers=6) as ex:
        scored = list(ex.map(one, rows))
    (HERE / "out" / "scored.json").write_text(json.dumps(scored, indent=1))
    ok = sum(1 for r in scored if r["outcomes"])
    print(f"scored {ok}/{len(scored)} cells with a measurable goal")
    return 0


if __name__ == "__main__":
    sys.exit(main())
