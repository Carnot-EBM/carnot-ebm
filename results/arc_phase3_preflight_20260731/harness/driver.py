#!/usr/bin/env python3
"""Driver for the PHASE-3 treatment-activation pre-flight of the Phase-2 wired induce fix.

Three arms per game -- ``ctl``, ``trt``, ``ctlb`` -- run BACK TO BACK on the SAME worker, and
therefore through the SAME llama-server process. That is not a scheduling convenience: the
sampler seed does not reach across server processes (measured this session -- identical config
on a second server gives different output, while within one process it holds byte-exactly
across a 4x budget range), so arms split across processes would be confounded by sampler
variance alone.

WHY ALL THREE ARMS PER GAME rather than the whole A/B first and the A/A afterwards. A cell only
earns "attributable" when the harness demonstrably repeats itself there (A/A byte-identical) AND
the treatment nonetheless changed the trace. Doing the A/B sweep first and the A/A second leaves
every perturbation unattributable if the second pass is cut short -- which is exactly how the
2026-07-29 retention grid ended up with one perturbed cell nobody could attribute. Interleaving
means the answer is COMPLETE for every game that finishes and the wall budget can bite anywhere.

EARLY STOP. Only a PERTURBED-and-attributable cell can ever be discordant, so the verdict is
fixed the moment ``attributable >= REQUIRED`` (PASS certain) or
``attributable + games_outstanding < REQUIRED`` (REFUSE certain). Every cell after that is pure
cost. On global wall expiry the verdict is INCONCLUSIVE, never REFUSE: a probe halted by the
clock has not shown the treatment inert, and conflating those two is this pre-flight's own
central error committed one level up.

GAME ORDER is ascending measured wall time (from the 2026-07-29 retention grid) within each
worker, so a wall-clock cut leaves the largest number of COMPLETE games rather than a set of
half-run ones.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python"
CELL = os.path.join(HERE, "cell.py")
# See cell.py: `HERE/cells` belongs to a prior session's LLM-on/off A/B. This probe keeps its
# own subtree so no census can ever pick up the other experiment's records.
CELLS = os.path.join(HERE, "pf", "cells")
LOG = os.path.join(HERE, "pf", "driver.log")
MAIN_REPO = "/home/ianblenke/github.com/ianblenke/carnot"

# FOUR arms per game, not three. An A/B comparison spans TWO arms, and a single A/A floor
# witnesses the determinism of only ONE of them -- the hole this module shipped with and that
# its own 2026-07-30 review found on the first real grid to use it. Here it is not a
# theoretical worry: the treatment CHANGES THE SAMPLER (`repeat_penalty`), which is exactly the
# kind of change that can alter how repeatable an arm is, so "the control arm repeats itself"
# licenses nothing about the treatment arm. Both floors are measured instead of argued.
ARMS = ("ctl", "trt", "ctlb", "trtb")
SEED = 1
HARD_TIMEOUT_S = 1500
TERM_GRACE_S = 25
GLOBAL_WALL_S = float(os.environ.get("PROBE_GLOBAL_WALL_S", str(4.0 * 3600)))
# The size of the banked-levels grid this probe SCREENS -- the 12-cell grid Phase 2 refused on
# an analytic bound. The probe runs 6 cells; `planned_n_cells` is what turns its measured
# attributable RATE into a statement about the grid that would actually be run.
PLANNED_N_CELLS = 12

# One worker per RTX 3090. Non-default ports: 8919 is the default and a stale server there is
# silently adopted, which would hand both workers the same process.
WORKERS = [
    {"gpu": "0", "port": 8971, "games": ["ft09", "tn36", "tu93"]},
    {"gpu": "1", "port": 8972, "games": ["vc33", "sc25", "lp85"]},
]
ALL_GAMES = [g for w in WORKERS for g in w["games"]]

sys.path.insert(0, os.path.join(MAIN_REPO, "python"))
from carnot.analysis.treatment_activation_preflight import (  # noqa: E402
    IDENTICAL,
    PERTURBED,
    classify_trace_pair,
    min_one_way_discordant_pairs,
)

REQUIRED = min_one_way_discordant_pairs(0.05)

_lock = threading.Lock()
_T0 = time.monotonic()
_stop = threading.Event()


def _log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} (+{round(time.monotonic() - _T0)}s) {msg}"
    with _lock:
        with open(LOG, "a") as fh:
            fh.write(line + "\n")
    print(line, flush=True)


def _sampler_seed(game: str) -> str:
    """A per-GAME sampler seed, IDENTICAL across arms.

    Per-game so two games cannot land on the same sampler state; identical across arms because
    the arms must differ only in the treatment -- a per-arm seed would reintroduce exactly the
    confound the seeding exists to remove.

    Derived from the game name by a STABLE formula rather than ``hash()``: Python salts
    ``hash()`` per process (PYTHONHASHSEED), so a resumed run would seed the same game
    differently from the cells already on disk and silently break the pairing.
    """
    return str(1000 + sum(ord(c) for c in game) * 7)


def _read(arm: str, game: str):
    p = os.path.join(CELLS, f"{arm}__{game}__s{SEED}.json")
    if not os.path.exists(p):
        return None, False
    with open(p) as fh:
        d = json.load(fh)
    res = d.get("result") or {}
    trace = res.get("action_trace")
    complete = d.get("status") == "ok" and not res.get("timed_out")
    return trace, complete


def _classify(game: str, a: str, b: str):
    ta, ca = _read(a, game)
    tb, cb = _read(b, game)
    return classify_trace_pair(ta, tb, a_complete=ca, b_complete=cb)


def _attributable() -> tuple[int, list]:
    """Cells that PERTURB under A/B and are byte-IDENTICAL under BOTH arms' A/A replicates.

    Raw A/B perturbation is not the honest quantity: a nondeterministic harness makes every
    cell perturb and the pre-flight passes trivially while nothing is attributable to anything.
    BOTH floors are required on the SAME cell -- a cell witnessed on only one side is
    unwitnessed on the other, which is a missing observation and never a pass.
    """
    cells = []
    for g in ALL_GAMES:
        ab = _classify(g, "ctl", "trt")
        aa_ctl = _classify(g, "ctl", "ctlb")
        aa_trt = _classify(g, "trt", "trtb")
        if (ab.get("cls") == PERTURBED
                and aa_ctl.get("cls") == IDENTICAL
                and aa_trt.get("cls") == IDENTICAL):
            cells.append(g)
    return len(cells), cells


def _run_cell(arm: str, game: str, gpu: str, port: int) -> None:
    out = os.path.join(CELLS, f"{arm}__{game}__s{SEED}.json")
    if os.path.exists(out):
        _log(f"SKIP  {arm}/{game} (already on disk)")
        return
    env = dict(os.environ)
    env.update({"CELL_GPU": gpu, "CELL_PORT": str(port),
                "CELL_SAMPLER_SEED": _sampler_seed(game)})
    # The canonical repo path, never $PWD: $PWD is a symlink alias here and a worktree run
    # once produced a phantom 5-action regression that survived 6 replicates across 2 seeds.
    t = time.monotonic()
    try:
        p = subprocess.run([PY, CELL, arm, game, str(SEED)], cwd=MAIN_REPO, env=env,
                           capture_output=True, text=True, timeout=HARD_TIMEOUT_S)
        tail = (p.stdout or p.stderr or "").strip().splitlines()[-1:] or [""]
        _log(f"DONE  {arm}/{game} rc={p.returncode} {round(time.monotonic()-t)}s :: {tail[0][:220]}")
    except subprocess.TimeoutExpired:
        # The cell's own SIGTERM handler writes a partial record; the hard kill below is only
        # for a cell that ignores it. A missing observation must stay VISIBLE.
        _log(f"TIMEOUT {arm}/{game} after {HARD_TIMEOUT_S}s -- cell writes its own partial")
        time.sleep(TERM_GRACE_S)


def _worker(w: dict) -> None:
    for game in w["games"]:
        if _stop.is_set():
            _log(f"STOP before {game} on gpu{w['gpu']}")
            return
        for arm in ARMS:
            if time.monotonic() - _T0 > GLOBAL_WALL_S:
                _log(f"GLOBAL WALL reached; abandoning {arm}/{game}")
                _stop.set()
                return
            _run_cell(arm, game, w["gpu"], w["port"])
        n, cells = _attributable()
        # A game counts as OUTSTANDING until its LAST arm is on disk -- a game with three of
        # four arms cannot yet be attributed, so counting it as done would let the early stop
        # fire on an incomplete census.
        outstanding = sum(
            1 for g in ALL_GAMES
            if not os.path.exists(os.path.join(CELLS, f"{ARMS[-1]}__{g}__s{SEED}.json"))
        )
        # The probe screens a LARGER grid than it runs, so the quantity that decides is the
        # attributable RATE projected to PLANNED_N_CELLS -- not the count at the probed size.
        # Requiring 6 attributable out of 6 probed would refuse a treatment that perturbs half
        # its cells, which is precisely the grid the projection says IS worth running.
        need_rate = REQUIRED / PLANNED_N_CELLS
        best_rate = (n + outstanding) / len(ALL_GAMES)
        _log(f"AFTER {game}: attributable={n} {cells} outstanding={outstanding} "
             f"rate={n/len(ALL_GAMES):.3f} best_possible={best_rate:.3f} need={need_rate:.3f}")
        # NOTE: there is deliberately no early stop on the PASS side. The refusal side saves a
        # doomed remainder and is worth taking early; a PASS, by contrast, is a RATE estimate
        # that the remaining games sharpen, and stopping the instant it crosses 0.5 would
        # report the threshold back as the measurement with the widest possible interval around
        # it. The full grid is ~2h, so the precision is cheap.
        if best_rate < need_rate:
            _log("REFUSE CERTAIN -- even all-remaining-attributable misses the projected rate")
            _stop.set()
            return


def main() -> int:
    os.makedirs(CELLS, exist_ok=True)
    _log(f"START required={REQUIRED} games={ALL_GAMES} arms={ARMS} "
         f"global_wall={GLOBAL_WALL_S}s head={subprocess.run(['git','-C',MAIN_REPO,'rev-parse','--short','HEAD'],capture_output=True,text=True).stdout.strip()}")
    ts = [threading.Thread(target=_worker, args=(w,), daemon=False) for w in WORKERS]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    n, cells = _attributable()
    _log(f"END attributable={n} {cells}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
