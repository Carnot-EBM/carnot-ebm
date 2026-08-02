#!/usr/bin/env python3
"""PRE-FLIGHT: does a candidate outcome have VARIANCE on the already-induced corpus?

WHY THIS RUNS BEFORE THE A/B AND BEFORE PRE-REGISTRATION. The 138-engine taxonomy that
motivates this session says plan_found is an EXACT function of the goal gate's kind (0
mismatches in 138), so scoring a GOAL intervention against plan_found grades the goal gate
with the goal gate. The replacement outcome therefore has to be chosen on evidence, and the
one thing that would waste a GPU run is picking an outcome that is 0/116 or 116/116 under
the control arm -- an outcome with no variance cannot move, so the A/B would be guaranteed
null for a reason that has nothing to do with the intervention.

The corpus scored here is the FROZEN engine set from
results/arc_object_perception_ab_change_fidelity_20260801/engines (116 cells, 20 games, 3
replicates, two arms of a DIFFERENT intervention). It is used ONLY to characterise the
outcome's marginal distribution. No treatment effect is estimated from it.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from collections import Counter
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
CORPUS = REPO / "results/arc_object_perception_ab_change_fidelity_20260801/engines"
TIMEOUT = 120


def score_cell(cell_dir: pathlib.Path) -> dict:
    name = cell_dir.name
    game = name.split("__")[0]
    eng = cell_dir / game / "world_model.py"
    rec = {"cell": name, "game": game, "arm": name.split("__")[-1]}
    if not eng.exists():
        rec["status"] = "no_engine_file"
        return rec
    pkl = SCRATCH / "windows" / f"{game}.pkl"
    if not pkl.exists():
        rec["status"] = "no_window"
        return rec
    job = SCRATCH / f"score_{name}.json"
    job.write_text(json.dumps({"engine_path": str(eng), "window_pkl": str(pkl)}))
    try:
        p = subprocess.run(
            [PY, str(HERE / "goal_score_worker.py"), str(job)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            env={"PATH": "/usr/bin:/bin", "HOME": "/home/ianblenke", "JAX_PLATFORMS": "cpu"},
        )
    except subprocess.TimeoutExpired:
        rec["status"] = "timeout"
        return rec
    for line in reversed((p.stdout or "").strip().splitlines()):
        try:
            rec.update(json.loads(line))
            return rec
        except Exception:  # noqa: BLE001,S112
            continue
    rec["status"] = "worker_error"
    rec["stderr"] = (p.stderr or "")[-300:]
    return rec


def outcomes(rec: dict) -> dict:
    """Every candidate outcome, derived from the same raw per-frame booleans."""
    if rec.get("status") != "ok":
        return {}
    held = list(rec["held_before"]) + list(rec["held_after"])
    shown = list(rec["shown_before"]) + list(rec["shown_after"])
    allf = held + shown + [rec["open"]]
    tru = [x for x in allf if x is True]
    fal = [x for x in allf if x is False]
    heldt = [x for x in held if x is True]
    heldf = [x for x in held if x is False]
    return {
        # O1 the shipped REQ-5714 metric: fires on the post-level-up frame
        "O1_fires_post_win": rec["post_win"] is True,
        # O2 fires on the last WITHIN-level state (the frame the winning action came from)
        "O2_fires_pre_win": rec["pre_win"] is True,
        # O3 fires anywhere in the held-out tail
        "O3_fires_heldout": bool(heldt),
        # O4 discriminates within the held-out tail (true somewhere, false somewhere)
        "O4_discriminates_heldout": bool(heldt) and bool(heldf),
        # O5 O1 plus the discrimination the gate demands: not true at the level's own root
        "O5_post_win_and_not_open": (rec["post_win"] is True) and (rec["open"] is False),
        # O6 O2 plus not-true-at-root
        "O6_pre_win_and_not_open": (rec["pre_win"] is True) and (rec["open"] is False),
        # O7 the treatment's own probe, for the circularity diagnosis: constant over
        #    every observed frame (this is what an accept-time constancy check catches)
        "O7_constant_on_observed": not (tru and fal),
        "O7b_all_false_observed": not tru,
        "O7c_all_true_observed": not fal,
        # execution health
        "raised_any": any(x == "raised" for x in allf),
        "nonbool_any": any(x == "nonbool" for x in allf),
    }


def main() -> int:
    cells = sorted(d for d in CORPUS.iterdir() if d.is_dir())
    with ThreadPoolExecutor(max_workers=8) as ex:
        recs = list(ex.map(score_cell, cells))
    for r in recs:
        r["outcomes"] = outcomes(r)
    out = HERE / "pre" / "preflight_outcomes.json"
    out.write_text(json.dumps(recs, indent=1))

    st = Counter(r.get("status") for r in recs)
    print(f"cells={len(recs)} status={dict(st)}")
    ok = [r for r in recs if r.get("status") == "ok"]
    print(f"scored={len(ok)}")
    keys = list(ok[0]["outcomes"]) if ok else []
    print(f"\n{'outcome':32s} {'n_true':>7s} {'/n':>5s} {'rate':>7s}  games_with_variance")
    for k in keys:
        n = sum(1 for r in ok if r["outcomes"][k])
        bygame: dict = {}
        for r in ok:
            bygame.setdefault(r["game"], []).append(r["outcomes"][k])
        var = sum(1 for v in bygame.values() if len(set(v)) > 1)
        print(f"{k:32s} {n:7d} {len(ok):5d} {n / len(ok):7.3f}  {var}/{len(bygame)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
