#!/usr/bin/env python3
"""Drive the masked re-score: 116 A/B cells + 31 best-of-N candidates, every mask arm.

CPU ONLY. No LLM, no GPU, no llama-server, no generation. Every input is already frozen on
disk: engine text under `engines/` and `arc_induce_bestofn_20260731/harness/bon/`, plus a
deterministically rebuilt window. This pass adds a GRADING SETTING to frozen engines; it does
not re-induce anything, so the treatment is untouched.

THE INTEGRITY GATE. Each cell's `unmasked` arm is re-derived here and compared to the
`change_fidelity` the A/B recorded for that same cell (and, for the best-of-N side, to the
value in the frozen `fidelity_vs_plan.json`). If a cell disagrees, its window is not the window
the original graded and the whole pass is void rather than partly merged -- the same gate the
A/B's own post-hoc pass used, for the same reason. It is what makes rebuilding (or falling back
to a cached) window sound instead of hopeful.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

REPO = pathlib.Path("/home/ianblenke/github.com/ianblenke/carnot")
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent
AB = REPO / "results" / "arc_object_perception_ab_change_fidelity_20260801"
BON = REPO / "results" / "arc_induce_bestofn_20260731"
SCRATCH = pathlib.Path(
    "/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/"
    "87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/hudms"
)
CACHED_WINDOWS = pathlib.Path(
    "/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/"
    "87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/op_ab/rescore_scratch"
)
WINDOW_TIMEOUT_S = 240.0
CELL_TIMEOUT_S = 120.0
N_PARALLEL = 6


def run_worker(worker: str, job: dict, tag: str, timeout: float) -> dict:
    jp = SCRATCH / f"job_{tag}.json"
    jp.write_text(json.dumps(job))
    env = dict(os.environ, CARNOT_REPO=str(REPO))
    try:
        pr = subprocess.run(  # noqa: S603
            [sys.executable, str(HERE / worker), str(jp)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        # UNDETERMINED, not a zero. Nothing about this engine was measured under any arm, so it
        # leaves both numerator and denominator. Scoring a hang as 0.0 would make a
        # non-terminating engine look like a bad model rather than an unmeasured one.
        return {"status": "undetermined_worker_timeout"}
    lines = (pr.stdout or "").strip().splitlines()
    if not lines:
        return {"status": "worker_no_output", "stderr": (pr.stderr or "")[-400:]}
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return {"status": "worker_bad_output", "stdout": lines[-1][:400]}


def build_windows(roster: list[str]) -> tuple[dict[str, pathlib.Path], dict[str, dict]]:
    paths: dict[str, pathlib.Path] = {}
    status: dict[str, dict] = {}
    for game in roster:
        p = SCRATCH / f"{game}_window.pkl"
        r = run_worker(
            "window_worker.py",
            {"game": game, "window_pkl": str(p)},
            f"win_{game}",
            WINDOW_TIMEOUT_S,
        )
        if r.get("status") == "ok" and p.exists():
            r["window_source"] = "rebuilt_here"
            paths[game] = p
        else:
            # FALLBACK, disclosed. tr87's rebuild is a known non-terminator. A cached window is
            # only admissible because the unmasked reproduction gate re-derives that cell's
            # change_fidelity and refuses the pass if it is not the window the A/B graded.
            cached = CACHED_WINDOWS / f"{game}_window.pkl"
            if cached.exists():
                shutil.copy(cached, p)
                paths[game] = p
                r["window_source"] = "cached_from_prior_rebuild_reproduction_gated"
            else:
                r["window_source"] = "none"
        status[game] = r
        print(f"  {game:<6} {r.get('status')} src={r.get('window_source')}")
    return paths, status


def main() -> int:  # noqa: C901
    t0 = time.time()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    rows = json.loads((AB / "rows.json").read_text())
    roster = json.loads((AB / "meta.json").read_text())["roster"]

    print(f"rebuilding {len(roster)} A/B windows")
    windows, window_status = build_windows(roster)

    # ---- A/B cells -----------------------------------------------------------------
    jobs = []
    for r in rows:
        game, rep, tag = r["game"], r["replicate"], r["tag"]
        cell = f"{game}__r{rep}__{tag}"
        code = AB / "engines" / cell / game / "world_model.py"
        if not code.exists():
            continue
        jobs.append((cell, game, code, r))
    print(f"\nscoring {len(jobs)} A/B cells ({N_PARALLEL} parallel)")

    def do_ab(j):
        cell, game, code, r = j
        wp = windows.get(game)
        if wp is None:
            return {"cell": cell, "game": game, "status": "no_window"}
        res = run_worker(
            "ab_worker.py",
            {"cell": cell, "game": game, "window_pkl": str(wp), "code_path": str(code)},
            cell,
            CELL_TIMEOUT_S,
        )
        res.setdefault("cell", cell)
        res["game"], res["replicate"], res["arm"] = game, r["replicate"], r["tag"]
        h = r.get("heldout") or {}
        res["ab_recorded_change_fidelity"] = h.get("change_fidelity")
        if res.get("status") == "ok" and h.get("measurable"):
            a = h.get("change_fidelity")
            b = (res.get("arms") or {}).get("unmasked", {}).get("change_fidelity")
            res["reproduces_ab_unmasked"] = bool(
                a is not None and b is not None and abs(float(a) - float(b)) < 1e-6
            )
        return res

    with ThreadPoolExecutor(max_workers=N_PARALLEL) as ex:
        ab_cells = list(ex.map(do_ab, jobs))
    n_repro = sum(1 for c in ab_cells if c.get("reproduces_ab_unmasked") is False)
    print(
        f"  A/B: ok={sum(1 for c in ab_cells if c.get('status') == 'ok')}/{len(ab_cells)} "
        f"reproduction_mismatch={n_repro}"
    )

    # ---- best-of-N candidates (the corpus that carries plan_found) -------------------
    d = json.loads((BON / "bestofn_scored.json").read_text())
    stall = set(d["stall_games"])
    cands = [c for c in d["candidates"] if c["game"] in stall]
    frozen_join = {
        (r["game"], r["cand"]): r for r in json.loads((AB / "fidelity_vs_plan.json").read_text())
    }
    print(f"\nscoring {len(cands)} best-of-N candidates")

    def do_bon(c):
        cp = BON / "harness" / "bon" / "gpu1" / f"{c['game']}_k{c['candidate']}.txt"
        if not cp.exists():
            m = sorted((BON / "harness" / "bon").rglob(f"{c['game']}_k{c['candidate']}.txt"))
            cp = m[0] if m else None
        rec = {
            "game": c["game"],
            "cand": c["candidate"],
            "plan_found": c.get("plan_found"),
            "goal_satisfiable": c.get("goal_satisfiable"),
            "usable": c.get("usable"),
            "engine_changes_anything": c.get("engine_changes_anything"),
        }
        if cp is None:
            rec["status"] = "no_code_file"
            return rec
        res = run_worker(
            "bon_worker.py",
            {"game": c["game"], "cand": c["candidate"], "code_path": str(cp)},
            f"bon_{c['game']}_{c['candidate']}",
            CELL_TIMEOUT_S,
        )
        rec.update(res)
        fz = frozen_join.get((c["game"], c["candidate"]))
        rec["frozen_join_change_fidelity"] = (fz or {}).get("change_fidelity")
        if rec.get("status") == "ok" and fz and fz.get("change_fidelity") is not None:
            b = (rec.get("arms") or {}).get("unmasked", {}).get("change_fidelity")
            rec["reproduces_frozen_join_unmasked"] = bool(
                b is not None and abs(float(fz["change_fidelity"]) - float(b)) < 1e-6
            )
        return rec

    with ThreadPoolExecutor(max_workers=N_PARALLEL) as ex:
        bon_rows = list(ex.map(do_bon, cands))
    n_bon_mm = sum(1 for c in bon_rows if c.get("reproduces_frozen_join_unmasked") is False)
    print(
        f"  BoN: ok={sum(1 for c in bon_rows if c.get('status') == 'ok')}/{len(bon_rows)} "
        f"reproduction_mismatch={n_bon_mm}"
    )

    payload = {
        "duration_s": round(time.time() - t0, 2),
        "window_status": window_status,
        "ab_cells": ab_cells,
        "bon_candidates": bon_rows,
    }
    (OUT / "rescore_masked_raw.json").write_text(json.dumps(payload, indent=1))
    print(f"\nwrote {OUT / 'rescore_masked_raw.json'} in {payload['duration_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
