#!/usr/bin/env python3
"""DEPTH SWEEP driver -- does raising max_depth fill the empty intersection?

THE QUESTION, stated so the null is as reportable as the win. The best-of-N run
(`results/arc_induce_bestofn_20260731/`) found criterion (iii) = 0 at every N, and found that
the zero is an EMPTY INTERSECTION rather than an absent capability: 9 of 40 stall-path
candidates clear criterion (i) (held-out dynamics), 2 reach a satisfiable goal with a found
plan, and NO candidate is in both sets. Of the 9 (i)-passers, the gate's own census says 6 were
rejected UNDECIDED as `goal_unreached_within_depth` -- the depth cap -- and only 3 were
DISPROVED as `degenerate_goal_predicate`.

So the pre-registered hypothesis is: **the intersection is empty at max_depth=40 because the
cap, not because good dynamics and plannability are actually incompatible.** If raising the cap
converts even one (i)-passer to plan-found, the "select on dynamics OR select on plannability"
trade-off dissolves rather than having to be resolved.

FALSIFIABLE, both ways, decided before the sweep runs:
  * CONFIRMED if |{(i)} INTERSECT {(iii)}| > 0 at some swept depth. Report the depth, which
    candidates, and the node cost.
  * REFUTED if the intersection is still empty at max_depth=200. That is a real finding and is
    to be reported as the headline, not buried: it would mean the 6 depth-capped (i)-passers
    are unreachable for a reason deeper than the cap, and that selecting on dynamics is
    genuinely anti-selective rather than incidentally so.

WHAT IS MANIPULATED, AND WHAT IS HELD FIXED. `max_depth` only, applied to the gate and the
planner TOGETHER (see depth_worker.py for why decoupling them is unsound in both directions).
`max_nodes` stays shipped on both. Same code, same root grids, same call index (1) as the
best-of-N run -- the completions are frozen on disk and are re-read, not re-generated, so this
sweep costs zero GPU and cannot be contaminated by sampling noise.

A CANDIDATE THAT TERMINATES ON BUDGET IS NOT A CONVERSION. At `budget_exhausted` /
`max_nodes_reached`, depth is not what stopped the search, so such a candidate is recorded
UNDECIDED at that depth and excluded from the conversion count. Counting it would let a
node-budget effect be reported as a depth effect.

CONCURRENCY NOTE. A separate Phase-2 run holds the GPUs and imports the same shipped modules.
This sweep is CPU-only and READ-ONLY with respect to the repo (it writes only under its own
results dir and a private scratch dir), and it does not edit shipped source -- deliberately, so
that it cannot perturb the measurement running alongside it.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
OUT_DIR = HERE.parent
REPO = OUT_DIR.parent.parent
BON = REPO / "results" / "arc_induce_bestofn_20260731"
SCRATCH = pathlib.Path(os.environ.get("DEPTH_SCRATCH", "/tmp/arc_depth_sweep"))
SCRATCH.mkdir(parents=True, exist_ok=True)

PY = os.environ.get("DEPTH_PY", "/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python")
CALL_INDEX = 1
# 40 is the shipped cap and the sweep's own control -- it must reproduce the best-of-N result or
# the harness is wrong. 61 is tn36's measured exact requirement. 200 is the pre-registered
# refutation point.
DEPTHS = [int(x) for x in os.environ.get("DEPTH_LADDER", "40,61,80,120,200").split(",")]
WORKERS = int(os.environ.get("DEPTH_WORKERS", "6"))
# Generous: the point of the subprocess is that a non-terminating engine cannot wedge the run.
TIMEOUT_S = int(os.environ.get("DEPTH_TIMEOUT_S", "300"))


def _load_bon() -> tuple[dict, dict]:
    scored = json.loads((BON / "bestofn_scored.json").read_text())
    by_key = {}
    for c in scored["candidates"]:
        if c.get("score_status") == "generation_failed":
            continue
        by_key[f"{c['game']}|{c['candidate']}"] = c
    return scored, by_key


def _write_code(scored: dict) -> dict[str, pathlib.Path]:
    """Re-extract each candidate's code from its FROZEN completion .txt.

    Deliberately re-derived from the completion rather than from any .py left in a scratch dir:
    the completion is the artifact of record and carries a sha, a stray .py does not.
    """
    sys.path.insert(0, str(REPO / "python"))
    from carnot.agentic import arc_executable_world_model as e3

    paths: dict[str, pathlib.Path] = {}
    for tag_dir in sorted((BON / "harness" / "bon").iterdir()):
        if not tag_dir.is_dir():
            continue
        bon_json = tag_dir / "bon.json"
        if not bon_json.exists():
            continue
        run = json.loads(bon_json.read_text())
        for row in run.get("rows", []):
            if row.get("status") != "ok":
                continue
            text = (tag_dir / row["completion_file"]).read_text(errors="replace")
            code = e3._extract_python(text) or text.strip()
            key = f"{row['game']}|{row['candidate']}"
            cp = SCRATCH / f"{row['game']}_k{row['candidate']}.py"
            cp.write_text(code)
            paths[key] = cp
    return paths


def _run_one(args) -> dict:
    key, depth, job_path = args
    t = time.monotonic()
    try:
        proc = subprocess.run(  # noqa: S603
            [PY, str(HERE / "depth_worker.py"), str(job_path)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
        raw = (proc.stdout or "").strip().splitlines()
        payload = json.loads(raw[-1]) if raw else {"status": "no_output"}
    except subprocess.TimeoutExpired:
        payload = {"status": "worker_timeout", "max_depth": depth}
    except Exception as exc:  # noqa: BLE001
        payload = {"status": f"driver_error:{type(exc).__name__}", "error": str(exc)[:200]}
    payload["_key"] = key
    payload["_depth"] = depth
    payload["_wall_s"] = round(time.monotonic() - t, 3)
    return payload


def main() -> int:
    scored, by_key = _load_bon()
    stall = set(scored["stall_games"])
    code_paths = _write_code(scored)

    jobs = []
    for key, cand in sorted(by_key.items()):
        cp = code_paths.get(key)
        if cp is None:
            continue
        game = cand["game"]
        root = BON / "harness" / "capture" / game / f"root_grid{CALL_INDEX}.pkl"
        if not root.exists():
            continue
        for depth in DEPTHS:
            job = {
                "code_path": str(cp),
                "root_pkl": str(root),
                "max_depth": depth,
            }
            jp = SCRATCH / f"{game}_k{cand['candidate']}_d{depth}.job.json"
            jp.write_text(json.dumps(job))
            jobs.append((key, depth, jp))

    print(
        f"{len(jobs)} jobs ({len(by_key)} candidates x {len(DEPTHS)} depths), "
        f"{WORKERS} at a time ...",
        flush=True,
    )
    t0 = time.monotonic()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, r in enumerate(ex.map(_run_one, jobs), 1):
            results.append(r)
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)}  ({time.monotonic() - t0:.0f}s)", flush=True)

    raw_path = OUT_DIR / "depth_sweep_raw.json"
    raw_path.write_text(
        json.dumps(
            {
                "call_index": CALL_INDEX,
                "depths": DEPTHS,
                "stall_games": sorted(stall),
                "postbank_games": sorted(scored["postbank_games"]),
                "wall_s": round(time.monotonic() - t0, 2),
                "results": results,
            },
            indent=1,
        )
    )
    print(f"wrote {raw_path}  ({time.monotonic() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
