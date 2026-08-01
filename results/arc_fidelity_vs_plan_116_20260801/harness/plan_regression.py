#!/usr/bin/env python3
"""Does change_fidelity predict PLANNABILITY? The 116-engine version.

WHY THIS EXISTS. The object-perception A/B moved `change_fidelity` (p=0.0192), but the same
artifact flagged that the metric's link to anything downstream is UNVERIFIED, and the frozen
48-candidate join was suggestive in the WRONG direction: both plannable candidates sat in the
bottom half of the fidelity ranking and all six perfect-1.0 candidates were unplannable, because
those six are tn36's progress-BAR TICKERS -- they model the status indicator exactly and the
playfield not at all. That join had n_plannable = 2, and its own artifact says so: "far too few to
claim a relationship, let alone a negative one... a reason to MEASURE the link, not a finding that
the link is absent."

This measures it at the larger n. If fidelity does not predict plannability, then the
object-perception result is a signal about a number that does not matter, and so is every future
representation A/B scored on it.

EVERY ENGINE RUNS IN A KILLABLE SUBPROCESS. A non-terminating induced engine wedged a run for 13
minutes on 2026-07-31, and the search functions bound the SEARCH, not the engine: a single
`engine()` call that loops forever is unbounded. An in-process signal alarm would be swallowed by
`plan_in_model`'s own broad `except Exception`, turning a hang into a silent false "no plan" --
which here would look exactly like a real negative result. So the bound is a process boundary.

SHIPPED DEFAULTS, DELIBERATELY. `plan_in_model` is called with no `max_depth`/`max_nodes`, so it
resolves `plan_max_depth_default()` (80 since 2026-08-01) and 20000 nodes -- the same values the
live agent uses. Widening either here would measure a planner nobody runs. `goal_energy` is not
supplied, matching every prior measurement in this line, which makes plan_found a LOWER bound on
what the live best-first planner could find.
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
OUT = HERE.parent
REPO = OUT.parent.parent
AB = REPO / "results" / "arc_object_perception_ab_change_fidelity_20260801"
PY = os.environ.get("PLAN_PY", "/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python")
TIMEOUT_S = int(os.environ.get("PLAN_TIMEOUT_S", "180"))
WORKERS = int(os.environ.get("PLAN_WORKERS", "6"))

CHILD = r"""
import json, os, pickle, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("CARNOT_ARC_E3_DIR", "/tmp/arc_planreg/e3")
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)
sys.path.insert(0, os.path.join(sys.argv[2], "python"))
import numpy as np
from carnot.agentic import arc_executable_world_model as e3

job = json.loads(open(sys.argv[1]).read())
out = {"status": "ok"}
code = open(job["engine_path"]).read()
ns = {"np": np, "numpy": np}
try:
    exec(compile(code, job["engine_path"], "exec"), ns)  # noqa: S102
except Exception as exc:
    print(json.dumps({"status": f"unrunnable:{type(exc).__name__}"}))
    raise SystemExit(0)
engine, goal = ns.get("engine"), ns.get("is_level_complete")
out["has_engine"] = callable(engine)
out["has_goal"] = callable(goal)
if not callable(engine):
    out["status"] = "no_engine"
    print(json.dumps(out)); raise SystemExit(0)
with open(job["root_pkl"], "rb") as fh:
    root = np.asarray(pickle.load(fh))
diag = {}
try:
    plan = e3.plan_in_model(engine, goal if callable(goal) else None, root, diagnostics=diag)
    out["plan_found"] = bool(plan)
    out["plan_length"] = len(plan) if plan else 0
except Exception as exc:
    out["plan_found"] = False
    out["plan_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
out["plan_diagnostics"] = {k: v for k, v in diag.items() if isinstance(v, (int, float, str, bool))}
# Distinct successors from the root: the tn36 tree was a PATH (one distinct successor per state),
# which is why max_depth and not max_nodes was its binding constraint. Recorded because it is a
# candidate predictor of plannability in its own right.
try:
    seen = set()
    for c in e3._model_candidates(root):
        try:
            ng = np.asarray(engine(root.copy(), c["action"], c["data"]))
        except Exception:
            continue
        if ng.shape == root.shape:
            seen.add(e3._state_key(ng))
    out["distinct_successors_at_root"] = len(seen)
except Exception:
    out["distinct_successors_at_root"] = None
print(json.dumps(out))
"""


_ROOT_CACHE: dict[str, pathlib.Path | None] = {}


def _root_for(game: str) -> pathlib.Path | None:
    """Root grid for a game, REBUILT the way the A/B built its own window.

    The frozen best-of-N capture only covers 4 of this roster's 20 games, and those 4 include
    tn36 -- the game whose progress-bar tickers are the known degeneracy. Restricting to them
    would bias the regression toward exactly the cluster under suspicion. `build_progress_window`
    is offline, CPU-only and cached, and is the same function the A/B used, so rebuilding keeps
    the planning start state consistent with the engines being scored.
    """
    if game in _ROOT_CACHE:
        return _ROOT_CACHE[game]
    out = None
    try:
        import pickle

        sys.path.insert(0, str(REPO / "python"))
        from carnot.agentic import arc_actions_to_progress as atp

        w = atp.build_progress_window(game)
        if w is not None:
            win, _full, _cell = w
            if win:
                scratch = pathlib.Path(os.environ.get("PLAN_SCRATCH", "/tmp/arc_planreg"))
                scratch.mkdir(parents=True, exist_ok=True)
                out = scratch / f"root_{game}.pkl"
                with open(out, "wb") as fh:
                    pickle.dump(win[0].grid, fh)
    except Exception as exc:  # noqa: BLE001 - a game we cannot rebuild is EXCLUDED, never zeroed
        print(f"  [root] {game}: {type(exc).__name__}: {str(exc)[:80]}", flush=True)
        out = None
    _ROOT_CACHE[game] = out
    return out


def _run(args) -> dict:
    key, engine_path, root_pkl = args
    scratch = pathlib.Path(os.environ.get("PLAN_SCRATCH", "/tmp/arc_planreg"))
    scratch.mkdir(parents=True, exist_ok=True)
    jp = scratch / f"{key}.job.json"
    jp.write_text(json.dumps({"engine_path": str(engine_path), "root_pkl": str(root_pkl)}))
    child = scratch / "child.py"
    child.write_text(CHILD)
    t = time.monotonic()
    try:
        proc = subprocess.run(  # noqa: S603
            [PY, str(child), str(jp), str(REPO)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )
        raw = (proc.stdout or "").strip().splitlines()
        payload = (
            json.loads(raw[-1])
            if raw
            else {"status": "no_output", "stderr": (proc.stderr or "")[-200:]}
        )
    except subprocess.TimeoutExpired:
        # A non-terminating engine. NOT a plain "no plan" -- recorded distinctly, because
        # collapsing it into False would let a hang masquerade as a measured negative.
        payload = {"status": "engine_nonterminating", "plan_found": False}
    except Exception as exc:  # noqa: BLE001
        payload = {"status": f"driver_error:{type(exc).__name__}", "error": str(exc)[:200]}
    payload["_key"] = key
    payload["_wall_s"] = round(time.monotonic() - t, 2)
    return payload


def main() -> int:
    rows = json.loads((AB / "rows.json").read_text())
    jobs, meta = [], {}
    for r in rows:
        game, arm = r.get("game"), r.get("arm")
        rep = r.get("replicate", r.get("r", 0))
        key = f"{game}__r{rep}__{arm}"
        eng = AB / "engines" / key / str(game) / "world_model.py"
        root = _root_for(str(game))
        if not eng.exists() or root is None:
            continue
        jobs.append((key, eng, root))
        meta[key] = r
    print(
        f"{len(jobs)} engines with both a committed engine file and a frozen root grid", flush=True
    )

    results = []
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, res in enumerate(ex.map(_run, jobs), 1):
            results.append(res)
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)} ({time.monotonic() - t0:.0f}s)", flush=True)

    joined = []
    for res in results:
        r = meta.get(res["_key"], {})
        h = r.get("heldout") or {}
        joined.append(
            {
                "key": res["_key"],
                "game": r.get("game"),
                "arm": r.get("arm"),
                "status": res.get("status"),
                "plan_found": bool(res.get("plan_found")),
                "plan_length": res.get("plan_length"),
                "plan_termination": (res.get("plan_diagnostics") or {}).get("termination_reason"),
                "distinct_successors_at_root": res.get("distinct_successors_at_root"),
                "change_fidelity": h.get("change_fidelity"),
                "cell_recall": h.get("cell_recall"),
                "accuracy": h.get("accuracy"),
                "spurious_changed_cells": h.get("spurious_changed_cells"),
                "n_changing": h.get("n_changing"),
                "measurable": h.get("measurable"),
                "wall_s": res.get("_wall_s"),
            }
        )

    (OUT / "plan_regression_raw.json").write_text(
        json.dumps(
            {"n_jobs": len(jobs), "wall_s": round(time.monotonic() - t0, 1), "rows": joined},
            indent=2,
        )
        + "\n"
    )
    n_plan = sum(1 for j in joined if j["plan_found"])
    print(f"\nwrote {OUT / 'plan_regression_raw.json'}  plannable={n_plan}/{len(joined)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
