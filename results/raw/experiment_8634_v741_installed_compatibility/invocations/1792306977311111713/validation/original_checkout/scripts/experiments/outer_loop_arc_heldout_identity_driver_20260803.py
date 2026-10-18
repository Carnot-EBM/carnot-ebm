#!/usr/bin/env python3
"""Driver for the leave-one-game-out ADAPTER-FREE / held-out-identity measurement.

Fans `outer_loop_arc_heldout_identity_cell_20260803.py` out over the 25-game public roster x
2 arms x N seeds, one SUBPROCESS per cell (see that module for why), and collects the rows.

It does NOT analyse. Analysis lives in
`outer_loop_arc_heldout_identity_analyse_20260803.py` so the pre-registered test cannot be
tuned against the numbers while they are being collected.

CPU ONLY. `CUDA_VISIBLE_DEVICES=""` is set by each cell; GPU 1 belongs to a concurrent
workflow and this measurement needs no generator.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CELL = REPO / "scripts" / "experiments" / "outer_loop_arc_heldout_identity_cell_20260803.py"
PYBIN = "/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python"

# The 25 tracked public games, taken from ops/arc_solve_registry.yaml rather than hardcoded, so
# a game added to the survey set cannot silently drop out of the denominator.
ARMS = ("control_identity_on", "heldout_identity_off")


def roster() -> list[str]:
    import yaml

    d = yaml.safe_load((REPO / "ops" / "arc_solve_registry.yaml").read_text())
    return sorted(str(e["game"]) for e in d["games"])


def run_cell(game: str, arm: str, seed: int, outdir: Path, budget: int, wall_s: float) -> dict:
    out = outdir / f"{game}__{arm}__s{seed}.json"
    if out.exists():
        try:
            return json.loads(out.read_text())
        except Exception:
            pass
    cmd = [
        PYBIN,
        str(CELL),
        "--game",
        game,
        "--arm",
        arm,
        "--seed",
        str(seed),
        "--budget",
        str(budget),
        "--wall-s",
        str(wall_s),
        "--out",
        str(out),
    ]
    t0 = time.time()
    try:
        subprocess.run(
            cmd, capture_output=True, text=True, timeout=wall_s + 900, cwd=str(REPO), check=False
        )
    except subprocess.TimeoutExpired:
        # A hard kill leaves no row; write one so the cell is in the denominator BY NAME.
        out.write_text(
            json.dumps(
                {
                    "game": game,
                    "arm": arm,
                    "seed": seed,
                    "status": "blocked_driver_timeout",
                    "cell_wall_s": round(time.time() - t0, 2),
                },
                indent=2,
            )
        )
    if out.exists():
        try:
            return json.loads(out.read_text())
        except Exception as exc:
            return {"game": game, "arm": arm, "seed": seed, "status": f"blocked_unreadable_{exc}"}
    return {"game": game, "arm": arm, "seed": seed, "status": "blocked_no_row_written"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--budget", type=int, default=400)
    ap.add_argument("--wall-s", type=float, default=1800.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--games", nargs="*", default=None)
    ap.add_argument("--outdir", default=str(REPO / "results" / "arc_heldout_identity_20260803"))
    args = ap.parse_args(argv)

    games = args.games or roster()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cells = [(g, a, s) for g in games for a in ARMS for s in args.seeds]
    print(f"cells={len(cells)} games={len(games)} arms={len(ARMS)} seeds={args.seeds}", flush=True)

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(run_cell, g, a, s, outdir, args.budget, args.wall_s): (g, a, s)
            for (g, a, s) in cells
        }
        for f in as_completed(futs):
            g, a, s = futs[f]
            try:
                row = f.result()
            except Exception as exc:
                row = {"status": f"blocked_driver_exception_{type(exc).__name__}"}
            done += 1
            res = row.get("result") or {}
            print(
                f"[{done}/{len(cells)}] {g} {a} s{s} status={row.get('status')} "
                f"lv={res.get('levels_gained')} rep={row.get('reproduced')} "
                f"act={res.get('total_actions')} eo={res.get('explored_out')} "
                f"wall={row.get('cell_wall_s')}",
                flush=True,
            )
    print(f"ALL DONE in {round(time.time() - t0, 1)}s -> {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
