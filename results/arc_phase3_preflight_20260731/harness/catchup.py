#!/usr/bin/env python3
"""Finish the A/B census the main driver did not reach before it died.

WHY THIS EXISTS AND WHY IT IS NOT SIMPLY "RUN THE REST". The verdict was already settled when
the driver stopped: 4 of 4 comparable cells byte-IDENTICAL, so the best attainable attributable
rate was 2/6 = 0.33 against the 0.5 the 12-cell grid needs, and the pre-flight's own doctrine
says every cell after that point is pure cost. Two things nonetheless justify these cells:

  1. The two unrun games (tu93, lp85) are not a random sample of the six -- they are the two
     SLOWEST, i.e. the deepest runs with the most post-induction actions, which is exactly where
     a treatment acting through the induced engine would have the most room to express itself.
     Refusing on 4 games while silently omitting the 2 most favourable ones would be a selection
     effect pointing in the treatment's disfavour, and this pre-flight exists to catch precisely
     that class of error in other people's experiments.
  2. `vc33/trtb` was lost to a bug in this harness (no `trtb` entry in `ARM_ENV`), not to the
     science. A gap left by my own defect should be filled rather than reported as a datum.

The A/A floors for games whose A/B came back IDENTICAL are deliberately NOT re-run: a floor
exists to decide whether a PERTURBED cell is attributable, and an identical pair has nothing to
attribute. Running them would be cost with no possible effect on the verdict.

Ordered cheapest-first per worker so a wall-clock cut leaves the most complete census.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python"
CELL = os.path.join(HERE, "cell.py")
CELLS = os.path.join(HERE, "pf", "cells")
LOG = os.path.join(HERE, "pf", "catchup.log")
MAIN_REPO = "/home/ianblenke/github.com/ianblenke/carnot"
SEED = 1
HARD_TIMEOUT_S = 1600

# (arm, game, gpu, port). tu93 completes the missing A/B pair; lp85/trt completes the pair whose
# control already exists (as a 1500s partial, so that pair may still resolve as a MISSING
# observation -- which is an honest outcome, not a zero). vc33/trtb repairs the harness bug.
WORK = {
    "0": [("ctl", "tu93", "0", 8971), ("trt", "tu93", "0", 8971)],
    "1": [("trtb", "vc33", "1", 8972), ("trt", "lp85", "1", 8972)],
}

_lock = threading.Lock()
_T0 = time.monotonic()


def _log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} (+{round(time.monotonic() - _T0)}s) {msg}"
    with _lock:
        with open(LOG, "a") as fh:
            fh.write(line + "\n")
    print(line, flush=True)


def _worker(key: str) -> None:
    for arm, game, gpu, port in WORK[key]:
        out = os.path.join(CELLS, f"{arm}__{game}__s{SEED}.json")
        if os.path.exists(out):
            _log(f"SKIP {arm}/{game}")
            continue
        env = dict(os.environ)
        env.update({"CELL_GPU": gpu, "CELL_PORT": str(port),
                    "CELL_SAMPLER_SEED": str(1000 + sum(ord(c) for c in game) * 7)})
        t = time.monotonic()
        try:
            p = subprocess.run([PY, CELL, arm, game, str(SEED)], cwd=MAIN_REPO, env=env,
                               capture_output=True, text=True, timeout=HARD_TIMEOUT_S)
            tail = (p.stdout or p.stderr or "").strip().splitlines()[-1:] or [""]
            _log(f"DONE {arm}/{game} rc={p.returncode} {round(time.monotonic()-t)}s :: {tail[0][:200]}")
        except subprocess.TimeoutExpired:
            _log(f"TIMEOUT {arm}/{game} -- the cell writes its own partial record")


def main() -> int:
    _log(f"CATCHUP START {WORK}")
    ts = [threading.Thread(target=_worker, args=(k,)) for k in WORK]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    _log("CATCHUP END")
    return 0


if __name__ == "__main__":
    sys.exit(main())
