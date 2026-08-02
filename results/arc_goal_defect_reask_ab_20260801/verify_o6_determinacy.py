#!/usr/bin/env python3
"""POST-HOC (2026-08-02): does the SHIPPED goal gate DETERMINE the pre-registered primary?

WHY THIS EXISTS. The pre-registration swapped the primary from O4 to O6 for one stated reason:
O4 failed a determinacy test -- every predicate the accept check would KEEP scored O4-positive
(6 of 6, FN=0), so the gate's accept decision determined the outcome and grading the goal
intervention against it was circular one indirection out. O6 was chosen because "2 of the 6
predicates the gate would keep still FAIL O6".

Adversarial review disputed that number. This settles it by running the ACTUAL SHIPPED
`LocalGGUFProposer._goal_defects` -- not a reimplementation of it -- over the same 115 frozen
engines, and cross-tabbing against O6 read from `pre/preflight_outcomes.json`.

THE ANSWER, and it is not the pre-registration's. The shipped gate keeps 4, and 4 of 4 are
O6-positive: FN=0, P(O6 | accept) = 1.000 against P(O6 | reject) = 0.027. O6 therefore carries
the EXACT defect that disqualified O4.

THE ROOT CAUSE, which is the part worth keeping. `pre/circularity_gap.json` computes constancy
with a LOCAL `const()` helper over ALL shown frames. The shipped gate probes at most
`_GOAL_PROBE_MAX_GRIDS = 12` grids -- the first 6 transitions. Capping can only make a
predicate look MORE constant, so the shipped gate is strictly the stricter of the two, and the
two cells that made O6 look undetermined (tu93__r1__on, tu93__r2__on) are exactly the two the
cap rejects. The pre-registration validated the circularity of a SHIPPED treatment using a more
permissive stand-in for it. That is the transferable lesson: when a design decision turns on
what a gate would do, call the gate.

Determinism was checked -- two consecutive runs on an idle machine give identical verdicts and
an identical kept set. One cell (sc25__r2__off, excluded from the 115-cell join) sits at ~11s
against the gate's 10s SIGALRM watchdog, and a timeout returns "accept, as before"; so the
gate's verdict on THAT cell is load-dependent. It cannot affect the cross-tab below.

Every evaluation runs in the killable subprocess `goal_defect_worker.py`, never here.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
# Derived, never hardcoded: CLAUDE.md Test-Run Record Integrity rule 4 -- an absolute path
# baked into source means a fresh clone writes into the operator's checkout.
REPO = HERE.parents[1]
PY = os.environ.get(
    "CARNOT_PY", "/home/ianblenke/github.com/Carnot-EBM/carnot-ebm/.venv/bin/python"
)
# The frozen windows are session scratch, so they are overridable rather than assumed present.
WINDOWS = pathlib.Path(
    os.environ.get(
        "CARNOT_GOALAB_WINDOWS",
        "/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/"
        "87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/goalab/windows",
    )
)
JOBS = pathlib.Path(os.environ.get("CARNOT_GOALAB_JOBS", "/tmp")) / "goal_o6_verify"
CORPUS = REPO / "results/arc_object_perception_ab_change_fidelity_20260801/engines"


def one(cell_dir: pathlib.Path) -> dict:
    name = cell_dir.name
    game = name.split("__")[0]
    eng = cell_dir / game / "world_model.py"
    rec = {"cell": name, "game": game}
    if not eng.exists():
        rec["status"] = "no_engine_file"
        return rec
    pkl = WINDOWS / f"{game}.pkl"
    if not pkl.exists():
        rec["status"] = "no_window"
        return rec
    JOBS.mkdir(parents=True, exist_ok=True)
    job = JOBS / f"defect_{name}.json"
    job.write_text(json.dumps({"engine_path": str(eng), "window_pkl": str(pkl)}))
    try:
        p = subprocess.run(
            [PY, str(HERE / "goal_defect_worker.py"), str(job)],
            capture_output=True,
            text=True,
            timeout=180,
            env={
                "PATH": "/usr/bin:/bin",
                "HOME": os.environ.get("HOME", "/home/ianblenke"),
                "JAX_PLATFORMS": "cpu",
                "CARNOT_REPO": str(REPO),
                "CUDA_VISIBLE_DEVICES": "",
            },
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


def main() -> int:
    pre = {r["cell"]: r for r in json.loads((HERE / "pre" / "preflight_outcomes.json").read_text())}
    cells = sorted(d for d in CORPUS.iterdir() if d.is_dir())
    with ThreadPoolExecutor(max_workers=8) as ex:
        recs = list(ex.map(one, cells))

    joined = []
    for r in (x for x in recs if x.get("status") == "ok"):
        o = pre.get(r["cell"], {}).get("outcomes") or {}
        if not o:
            continue
        joined.append(
            {
                **r,
                "O6": bool(o["O6_pre_win_and_not_open"]),
                "O4": bool(o["O4_discriminates_heldout"]),
            }
        )

    kept = [r for r in joined if not r["defects_on"]]
    rej = [r for r in joined if r["defects_on"]]

    # The pre-registration's stand-in, recomputed here so the two are comparable side by side.
    def const(vals: list) -> bool:
        return len({v for v in vals if isinstance(v, bool)}) <= 1

    unc_kept = [
        r
        for r in pre.values()
        if r.get("status") == "ok" and not const(list(r["shown_before"]) + list(r["shown_after"]))
    ]

    out = {
        "what_this_is": "post-hoc determinacy check of the SHIPPED goal gate against the "
        "pre-registered primary O6, run with the shipped `_goal_defects` rather than a "
        "reimplementation of it.",
        "n_scored": len(joined),
        "shipped_gate": {
            "n_rejected": len(rej),
            "n_kept": len(kept),
            "kept_cells": sorted(r["cell"] for r in kept),
            "defect_kind_engine_counts": dict(Counter(k for r in joined for k in r["defects_on"])),
            "n_engines_with_more_than_one_defect_kind": sum(
                1 for r in joined if len(r["defects_on"]) > 1
            ),
            "inert_when_flag_off": sum(1 for r in joined if r["defects_off"]),
        },
        "O6_determinacy": {
            "kept_and_O6_positive": sum(1 for r in kept if r["O6"]),
            "kept_and_O6_negative_FN": sum(1 for r in kept if not r["O6"]),
            "rejected_and_O6_positive": sum(1 for r in rej if r["O6"]),
            "P_O6_given_accept": round(sum(1 for r in kept if r["O6"]) / len(kept), 4)
            if kept
            else None,
            "P_O6_given_reject": round(sum(1 for r in rej if r["O6"]) / len(rej), 4)
            if rej
            else None,
            "verdict": "FN=0 -- the gate's accept decision DETERMINES O6 in the keep "
            "direction, which is the same defect that disqualified O4.",
        },
        "O4_determinacy_for_comparison": {
            "kept_and_O4_positive": sum(1 for r in kept if r["O4"]),
            "kept_and_O4_negative_FN": sum(1 for r in kept if not r["O4"]),
        },
        "root_cause_the_prereg_used_a_more_permissive_stand_in": {
            "uncapped_const_over_all_shown_frames_kept": len(unc_kept),
            "uncapped_kept_cells": sorted(r["cell"] for r in unc_kept),
            "uncapped_kept_and_O6_negative": sorted(
                r["cell"] for r in unc_kept if not r["outcomes"]["O6_pre_win_and_not_open"]
            ),
            "shipped_probe_grid_cap": "_GOAL_PROBE_MAX_GRIDS = 12 (the first 6 transitions)",
            "reading": "capping can only make a predicate look MORE constant, so the shipped "
            "gate is strictly stricter. The cells the pre-registration counted as evidence "
            "that the gate does NOT determine O6 are exactly the cells the cap rejects.",
        },
    }
    (HERE / "out" / "o6_determinacy_20260802.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
