#!/usr/bin/env python3
"""Driver for the best-of-N corpus: extract frozen completions to .py, score per game."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "out"
CELLS = OUT / "cells_bestofn"
CELLS.mkdir(parents=True, exist_ok=True)
SCRATCH = pathlib.Path(os.environ["SCRATCH_E3"])
BON = REPO / "results" / "arc_induce_bestofn_20260731"
CODE = SCRATCH / "bon_code"
CODE.mkdir(parents=True, exist_ok=True)
GAME_TIMEOUT_S = int(os.environ.get("GVS_GAME_TIMEOUT_S", "1800"))

sys.path.insert(0, str(REPO / "python"))
from carnot.agentic import arc_executable_world_model as e3  # noqa: E402


def main() -> int:
    scored = json.loads((BON / "bestofn_scored.json").read_text())
    by_game = defaultdict(list)
    n_missing = 0
    for c in scored["candidates"]:
        g, k = c["game"], int(c["candidate"])
        tag = c.get("tag") or "gpu1"
        txt_p = BON / "harness" / "bon" / tag / f"{g}_k{k}.txt"
        if not txt_p.exists():
            n_missing += 1
            continue
        cp = CODE / f"{g}_k{k}.py"
        if not cp.exists():
            txt = txt_p.read_text(errors="replace")
            cp.write_text(e3._extract_python(txt) or txt.strip())  # noqa: SLF001
        by_game[g].append({"cell": f"{g}__k{k}", "path": str(cp)})
    print(
        f"bestofn: {sum(len(v) for v in by_game.values())} candidates over "
        f"{len(by_game)} games ({n_missing} completion texts missing)",
        flush=True,
    )
    env = dict(os.environ, CARNOT_REPO=str(REPO), SCRATCH_E3=str(SCRATCH))
    status = {}
    for i, g in enumerate(sorted(by_game), 1):
        cell_out = CELLS / f"{g}.json"
        if cell_out.exists():
            status[g] = "cached"
            print(f"[{i}] {g} cached", flush=True)
            continue
        jf = OUT / f".bonjobs_{g}.json"
        jf.write_text(json.dumps(by_game[g]))
        t0 = time.time()
        try:
            r = subprocess.run(
                [
                    str(REPO / ".venv/bin/python"),
                    str(HERE / "bestofn_worker.py"),
                    g,
                    str(jf),
                    str(cell_out),
                ],
                env=env,
                timeout=GAME_TIMEOUT_S,
                capture_output=True,
                text=True,
            )
            ok = r.returncode == 0 and cell_out.exists()
            status[g] = "ok" if ok else f"worker_rc{r.returncode}"
            if not ok:
                (OUT / f"err_bon_{g}.log").write_text(
                    (r.stdout or "")[-3000:] + "\n---STDERR---\n" + (r.stderr or "")[-8000:]
                )
        except subprocess.TimeoutExpired:
            status[g] = "game_timeout"
        print(f"[{i}] {g} {status[g]} n={len(by_game[g])} {time.time() - t0:.1f}s", flush=True)
        jf.unlink(missing_ok=True)
    (OUT / "collect_status_bestofn.json").write_text(
        json.dumps({"status": status, "n_missing_completions": n_missing}, indent=1)
    )
    print("bestofn collection done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
