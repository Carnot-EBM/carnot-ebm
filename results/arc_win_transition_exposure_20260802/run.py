#!/usr/bin/env python3
"""DRIVER: how often is `_win_transition` even AVAILABLE at a live induce call? CPU only.

EXPOSURE BEFORE EFFECT. The 2026-08-01 change (arc_competition_agent.py:4710/:4946/:6433 plus
the `win_transition=` kwarg through arc_executable_world_model) can only alter behaviour at the
LIVE induce call, and only when `self._win_transition` is non-None. Its sole writer is
`_begin_level_goal_episode`, which runs only after a level-up. So the change's exposure is
bounded by how often the live agent has already banked a level when it inducts. This measures
that, and NOTHING ELSE -- no effect claim is made anywhere in this harness.

THE REPLICATE AXIS IS `frontier_discipline_seed`, NOT A GLOBAL SEED. Every RNG the live explorer
uses is `random.Random(<constructor default>)` (arc_competition_agent.py:1310, :1397), which a
worker-level `random.seed()` cannot reach. Replicating over argv seeds would produce an
identically-zero A/A by construction and would be a design bug, not a noise floor. The
constructor argument is the reachable knob, so that is what varies here; `--argv-seed-probe`
DEMONSTRATES the inertness of the unreachable one rather than assuming it.

25 public games, each in a killable bounded subprocess. A game that does not return inside the
bound is a COVERAGE GAP, never a zero.

NO GPU, NO GENERATOR, NO LLM, NO SUBMISSION, NO SCORED GAME. The proposer is the project's own
`_NoOpProposer` llm_off arm definition (results/first_win_llm_on_20260727 `arm_definitions`).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKER = HERE / "worker.py"
PYBIN = "/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python"

REGISTRY = REPO / "ops" / "arc_solve_registry.yaml"
GAMES = sorted(str(e["game"]) for e in yaml.safe_load(REGISTRY.read_text())["games"])

ARGV_SEED = 20260802  # held FIXED; it is provably inert (see module docstring)
SHIPPED_FD_SEED = 20260724  # arc_competition_agent.py:1310 default
PER_RUN_TIMEOUT_S = 900


def one(game: str, budget: int, fd_seed: int, scratch: Path, argv_seed: int = ARGV_SEED) -> dict:
    out = scratch / f"{game}__b{budget}__fd{fd_seed}__av{argv_seed}.json"
    if out.exists():
        try:
            cached = json.loads(out.read_text())
            # Stamp the gap flag on the CACHED path too. Returning the raw record left
            # `coverage_gap` ABSENT on every resumed cell, and absent is not False.
            cached["coverage_gap"] = bool(cached.get("error"))
            if cached["coverage_gap"]:
                cached["gap_reason"] = cached.get("error")
            return cached
        except Exception:
            out.unlink(missing_ok=True)
    env = dict(os.environ)
    env.update(
        {
            # SINGLE-THREADED BLAS + PASSIVE WAIT. Under 11 concurrent workers with
            # OMP_NUM_THREADS=2 the OpenMP runtime BUSY-WAITS between parallel regions, which
            # inflated CPU time ~6x over an uncontended run of the same cell and made the sweep
            # look pathologically slow. Nothing in this measurement is BLAS-bound.
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_WAIT_POLICY": "PASSIVE",
            "JAX_PLATFORMS": "cpu",
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    t0 = time.time()
    try:
        proc = subprocess.run(
            [PYBIN, str(WORKER), game, str(budget), str(out), str(argv_seed), str(fd_seed)],
            cwd=str(REPO),
            env=env,
            capture_output=True,
            text=True,
            timeout=PER_RUN_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return {
            "game": game,
            "budget": budget,
            "seed": argv_seed,
            "frontier_discipline_seed": fd_seed,
            "coverage_gap": True,
            "gap_reason": f"exceeded {PER_RUN_TIMEOUT_S}s wall bound",
            "elapsed_s": round(time.time() - t0, 1),
        }
    if not out.exists():
        return {
            "game": game,
            "budget": budget,
            "seed": argv_seed,
            "frontier_discipline_seed": fd_seed,
            "coverage_gap": True,
            "gap_reason": f"worker exit {proc.returncode}; stderr tail: {proc.stderr[-400:]}",
            "elapsed_s": round(time.time() - t0, 1),
        }
    rec = json.loads(out.read_text())
    rec["coverage_gap"] = bool(rec.get("error"))
    if rec["coverage_gap"]:
        rec["gap_reason"] = rec.get("error")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="400,2000")
    ap.add_argument("--fd-seeds", default=str(SHIPPED_FD_SEED))
    ap.add_argument(
        "--replicate-budget",
        type=int,
        default=0,
        help="if set, extra fd-seed replicates are run at THIS budget only",
    )
    ap.add_argument("--replicate-fd-seeds", default="")
    ap.add_argument(
        "--argv-seed-probe",
        default="",
        help="comma-separated games: rerun at 2 argv seeds to DEMONSTRATE inertness",
    )
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument(
        "--scratch",
        default=os.environ.get(
            "WTX_SCRATCH",
            "/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/"
            "87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/wtx/cells",
        ),
    )
    ap.add_argument("--out", default=str(HERE / "rows.json"))
    args = ap.parse_args()

    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    budgets = [int(b) for b in args.budgets.split(",") if b]
    fd_seeds = [int(s) for s in args.fd_seeds.split(",") if s]

    jobs = [(g, b, fd, ARGV_SEED) for b in budgets for fd in fd_seeds for g in GAMES]
    if args.replicate_budget and args.replicate_fd_seeds:
        for fd in (int(s) for s in args.replicate_fd_seeds.split(",") if s):
            jobs += [(g, args.replicate_budget, fd, ARGV_SEED) for g in GAMES]
    if args.argv_seed_probe:
        for g in args.argv_seed_probe.split(","):
            jobs.append((g, 400, SHIPPED_FD_SEED, ARGV_SEED + 1))

    print(f"== {len(jobs)} cells over {len(GAMES)} games ==", flush=True)
    rows: list[dict] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one, g, b, fd, scratch, av): (g, b, fd, av) for (g, b, fd, av) in jobs}
        for f in as_completed(futs):
            rec = f.result()
            rows.append(rec)
            print(
                f"  [{len(rows):3}/{len(jobs)}] {rec['game']:5} b={rec['budget']:5} "
                f"fd={rec.get('frontier_discipline_seed')} av={rec.get('seed')} "
                f"levels={rec.get('levels')} induce={rec.get('n_induce_calls')} "
                f"win={rec.get('n_with_win_transition')} gap={rec.get('coverage_gap')} "
                f"[{rec.get('elapsed_s')}s]",
                flush=True,
            )
    Path(args.out).write_text(json.dumps(rows, indent=2, default=str))
    print(f"wrote {args.out} in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
