#!/usr/bin/env python3
"""Driver: rebuild all 20 game windows via killable subprocesses, in parallel, with a timeout.

A game that times out is recorded as a MISSING OBSERVATION (status timeout) and is absent
from every downstream count. It is never a zero.
"""

from __future__ import annotations

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
WINDOWS = SCRATCH / "windows"
TIMEOUT = 900

GAMES = [
    "ar25",
    "bp35",
    "cd82",
    "cn04",
    "g50t",
    "ka59",
    "lf52",
    "lp85",
    "ls20",
    "m0r0",
    "re86",
    "s5i5",
    "sb26",
    "sc25",
    "sk48",
    "su15",
    "tn36",
    "tr87",
    "tu93",
    "wa30",
]


def one(game: str) -> dict:
    WINDOWS.mkdir(parents=True, exist_ok=True)
    pkl = WINDOWS / f"{game}.pkl"
    if pkl.exists():
        return {"status": "cached", "game": game}
    job = SCRATCH / f"job_{game}.json"
    job.write_text(json.dumps({"game": game, "window_pkl": str(pkl)}))
    env = {
        "CARNOT_REPO": str(REPO),
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/ianblenke",
        "CARNOT_ARC_OFFLINE": "1",
        "CARNOT_ARC_E3_DIR": f"/tmp/arc_goalab/e3_build_{game}",
    }
    try:
        p = subprocess.run(
            [PY, str(HERE / "window_worker.py"), str(job)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "game": game}
    tail = (p.stdout or "").strip().splitlines()
    for line in reversed(tail):
        try:
            return json.loads(line)
        except Exception:  # noqa: BLE001,S112
            continue
    return {"status": "error", "game": game, "stderr": (p.stderr or "")[-400:]}


def main() -> int:
    with ThreadPoolExecutor(max_workers=6) as ex:
        rows = list(ex.map(one, GAMES))
    out = HERE / "pre" / "windows.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    ok = sum(1 for r in rows if r.get("status") in ("ok", "cached"))
    print(f"windows ok={ok}/{len(rows)}")
    for r in rows:
        print(" ", json.dumps(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
