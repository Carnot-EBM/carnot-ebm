#!/usr/bin/env python3
"""PHASE 2, STEP 1 -- run the static + dry-run checks over EVERY generated engine on disk.

WHY A CORPUS SCAN RATHER THAN A UNIT TEST. A validator that rejects code is only as good as its
false-positive rate, and a unit test can only measure that against cases its author thought of.
The repo holds 439 real world-model files written by a real generator across months of runs
(`results/arc_e3`, `results/arc_e3_origin_fixtures`, `results/arc_logo_snapshot`, and the
per-experiment `e3_store` directories). Scanning all of them measures the false-positive rate
against the actual distribution of generated code, including every shape the author did not
anticipate.

WHAT IT PROVES, AND WHAT IT DOES NOT. Every `missing_return` flag is cross-checked by EXECUTING
the engine and recording the (action, data-shape) pairs on which it really does return `None`.
A flag with no confirming execution is reported as `UNCONFIRMED` and counted against the check,
not quietly dropped -- an AST claim that no execution can reproduce is exactly the kind of
plausible-looking evidence this project's QA-layer discipline exists to catch.

The execution cross-check uses a SYNTHETIC probe grid, so a `None` it does not reproduce is not
proof the path is unreachable, only that this probe did not reach it. That is why the AST result
is the finding and the execution is corroboration, not the other way round.

Read-only with respect to every evidence directory: it opens `world_model*.py` files and never
writes one. `CARNOT_ARC_E3_DIR` is redirected so that `collect_transitions` cannot write into
`results/arc_e3`.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

REPO = "/home/ianblenke/github.com/ianblenke/carnot"
sys.path.insert(0, os.path.join(REPO, "python"))

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
_SCRATCH = os.environ.get("P2_SCRATCH", "/tmp/arc_p2_scratch")
os.environ.setdefault("CARNOT_ARC_E3_DIR", os.path.join(_SCRATCH, "e3"))
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)

import numpy as np  # noqa: E402

from carnot.agentic.arc_engine_static_validation import (  # noqa: E402
    dry_run_defects,
    engine_changes_anything,
    missing_return_defects,
)

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "corpus_scan.json"

# The five games the 2026-07-30 activation grid measured, in the order the audit reports them.
GAMES = ["ft09", "tu93", "lp85", "tn36", "sc25"]
N_TRANS = int(os.environ.get("P2_N_TRANS", "25"))
SEED = int(os.environ.get("P2_SEED", "1"))


def _probe_none_returns(src: str, path: str) -> dict:
    """Execute the engine on a synthetic grid and record where it returns None.

    Corroboration for the AST flag. Deliberately broad over (action, data) rather than clever:
    the point is to find ANY reachable None, not to characterise the engine.
    """
    ns: dict = {"np": np, "numpy": np}
    try:
        exec(compile(src, path, "exec"), ns)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        return {"exec_ok": False, "exec_error": f"{type(exc).__name__}: {exc}"[:200]}
    engine = ns.get("engine")
    if not callable(engine):
        return {"exec_ok": True, "engine_callable": False}
    grid = np.zeros((64, 64), dtype=int)
    grid[10, 10] = 3
    grid[20, 20] = 8
    none_on = []
    raised_on = []
    for action in (1, 2, 3, 4, 5, 6, 7):
        for data in (None, {"x": 10, "y": 10}):
            try:
                res = engine(grid.copy(), action, data)
            except Exception as exc:  # noqa: BLE001
                raised_on.append([action, data is not None, type(exc).__name__])
                continue
            if res is None:
                none_on.append([action, data is not None])
    return {
        "exec_ok": True,
        "engine_callable": True,
        "returns_none_on": none_on,
        "n_none": len(none_on),
        "raised_on": raised_on[:6],
        "n_raised": len(raised_on),
    }


def main() -> int:
    t0 = time.time()
    results_root = pathlib.Path(REPO) / "results"
    files = sorted(results_root.rglob("world_model*.py"))

    # --- static pass over the whole corpus -------------------------------------------------
    rows = []
    for p in files:
        src = p.read_text(errors="replace")
        defects = missing_return_defects(src)
        kinds = sorted({d.kind for d in defects})
        row = {
            "path": str(p.relative_to(REPO)),
            "n_lines": src.count("\n") + 1,
            "defect_kinds": kinds,
        }
        if "missing_return" in kinds:
            row["missing_return_line"] = next(
                d.line for d in defects if d.kind == "missing_return"
            )
            row["execution_cross_check"] = _probe_none_returns(src, str(p))
        rows.append(row)

    flagged = [r for r in rows if "missing_return" in r["defect_kinds"]]
    confirmed = [
        r for r in flagged if int(r.get("execution_cross_check", {}).get("n_none", 0) or 0) > 0
    ]
    unconfirmed = [r for r in flagged if r not in confirmed]

    # --- dry run against REAL offline transitions, per game --------------------------------
    from carnot.agentic import arc_executable_world_model as e3

    per_game = {}
    for game in GAMES:
        entry: dict = {"game": game}
        try:
            trans, cell = e3.collect_transitions(game, n=N_TRANS, seed=SEED)
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"collect_transitions: {type(exc).__name__}: {exc}"[:200]
            per_game[game] = entry
            continue
        entry["n_transitions"] = len(trans)
        entry["cell"] = int(cell)
        entry["n_changing"] = int(
            sum(1 for t in trans if not np.array_equal(t.grid, t.next_grid))
        )
        engines = []
        for p in files:
            # Only score engines whose directory names this game, so a dry run is always run
            # against transitions from the game the engine was induced on.
            if f"/{game}/" not in str(p) and not str(p.parent).endswith(game):
                continue
            src = p.read_text(errors="replace")
            defects = dry_run_defects(src, trans)
            engines.append(
                {
                    "path": str(p.relative_to(REPO)),
                    "static_kinds": sorted({d.kind for d in missing_return_defects(src)}),
                    "dry_run_kinds": sorted({d.kind for d in defects}),
                    "dry_run_details": [d.detail[:220] for d in defects],
                    "changes_anything": engine_changes_anything(src, trans),
                }
            )
        entry["engines"] = engines
        per_game[game] = entry

    out = {
        "generated_by": "results/arc_engine_validation_20260731/harness/scan_corpus.py",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_s": round(time.time() - t0, 2),
        "n_world_model_files_scanned": len(files),
        "static": {
            "n_clean": len(rows) - len(flagged),
            "n_missing_return": len(flagged),
            "n_missing_return_execution_confirmed": len(confirmed),
            "n_missing_return_UNCONFIRMED": len(unconfirmed),
            "flagged": flagged,
        },
        "dry_run_per_game": per_game,
        "pins": {"n_transitions": N_TRANS, "seed": SEED, "games": GAMES},
    }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(
        f"scanned {len(files)} files in {out['duration_s']}s -- "
        f"missing_return {len(flagged)} "
        f"(execution-confirmed {len(confirmed)}, UNCONFIRMED {len(unconfirmed)})"
    )
    for g, e in per_game.items():
        if "error" in e:
            print(f"  {g}: {e['error']}")
            continue
        n_raise = sum(1 for r in e["engines"] if "engine_raised" in r["dry_run_kinds"])
        n_none = sum(1 for r in e["engines"] if "engine_returned_none" in r["dry_run_kinds"])
        n_inert = sum(1 for r in e["engines"] if r["changes_anything"] is False)
        print(
            f"  {g}: {len(e['engines'])} engines, {e['n_transitions']} transitions "
            f"({e['n_changing']} changing) -- raised {n_raise}, returned-None {n_none}, "
            f"inert {n_inert}"
        )
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
