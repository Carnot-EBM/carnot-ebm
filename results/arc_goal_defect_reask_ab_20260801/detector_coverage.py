#!/usr/bin/env python3
"""How much of the goal-failure population would the accept check actually CATCH?

OBSERVATIONAL, NOT CAUSAL, and the distinction is the whole reading. This runs the detector
over the 115 frozen engines from the sibling A/B and reports what it would have REJECTED. It
says nothing about whether the re-ask then produces a better predicate -- that is what the GPU
A/B is for, and no number here should be read as an effect.

WHY IT IS WORTH MEASURING ANYWAY. The taxonomy that motivated this work estimated the
catchable slice at 37 of 71 (52%), reasoning from the two SYNTACTIC classes: 34 unconditional
`return False` (A_DECLINED) plus 3 with no return at all (D_NO_PREDICATE). But the detector's
`goal_constant` probe is a RUNTIME check over observed frames, and a whole-board trope
(`np.all(grid == 1)`) is just as constant on those frames as `return False` is -- it is simply
constant for a different reason. So the ceiling on DETECTION may be materially higher than the
syntactic estimate. If it is, that is worth knowing before reading the A/B, because it separates
"the gate cannot see the problem" from "the gate sees it and the re-ask cannot fix it" -- two
very different negative results.

Every evaluation runs in the killable subprocess `goal_defect_worker.py`, never in this
interpreter.
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


def one(cell_dir: pathlib.Path) -> dict:
    name = cell_dir.name
    game = name.split("__")[0]
    eng = cell_dir / game / "world_model.py"
    rec = {"cell": name, "game": game}
    if not eng.exists():
        rec["status"] = "no_engine_file"
        return rec
    pkl = SCRATCH / "windows" / f"{game}.pkl"
    if not pkl.exists():
        rec["status"] = "no_window"
        return rec
    job = SCRATCH / f"defect_{name}.json"
    job.write_text(json.dumps({"engine_path": str(eng), "window_pkl": str(pkl)}))
    try:
        p = subprocess.run(
            [PY, str(HERE / "goal_defect_worker.py"), str(job)],
            capture_output=True,
            text=True,
            timeout=180,
            env={
                "PATH": "/usr/bin:/bin",
                "HOME": "/home/ianblenke",
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

    ok = [r for r in recs if r.get("status") == "ok"]
    # Join to the pre-flight outcome so the cross-tab is on the SAME cells.
    joined = []
    for r in ok:
        o = pre.get(r["cell"], {}).get("outcomes") or {}
        if not o:
            continue
        joined.append({**r, "discriminates": bool(o["O4_discriminates_heldout"])})

    n = len(joined)
    rejected = [r for r in joined if r["defects_on"]]
    kinds = Counter(k for r in joined for k in r["defects_on"])
    # The cross-tab that matters: does the gate reject the GOOD ones too?
    tp = sum(1 for r in joined if r["defects_on"] and not r["discriminates"])
    fp = sum(1 for r in joined if r["defects_on"] and r["discriminates"])
    fn = sum(1 for r in joined if not r["defects_on"] and not r["discriminates"])
    tn = sum(1 for r in joined if not r["defects_on"] and r["discriminates"])

    out = {
        "what_this_is": "OBSERVATIONAL detector coverage over 115 frozen engines. It measures "
        "what the accept check would REJECT, not whether the re-ask then helps.",
        "n_scored": n,
        "n_would_be_rejected": len(rejected),
        "rejection_rate": round(len(rejected) / n, 4) if n else None,
        "defect_kinds": dict(kinds),
        "inert_when_flag_off": sum(1 for r in joined if r["defects_off"]),
        "cross_tab_vs_primary_outcome": {
            "rejected_and_non_discriminating_TP": tp,
            "rejected_but_discriminating_FP": fp,
            "kept_and_non_discriminating_FN": fn,
            "kept_and_discriminating_TN": tn,
            "reading": "FP is the cost that matters: a predicate the gate throws away even "
            "though it DID discriminate on held-out frames. Each FP is one wasted sample, "
            "bounded, and the re-ask can only replace it -- but a large FP count would mean "
            "the gate is not selective, it is just re-rolling everything.",
        },
        "vs_the_syntactic_estimate": {
            "taxonomy_estimate": "37 of 71 (52%), from A_DECLINED (34) + D_NO_PREDICATE (3)",
            "why_this_can_differ": "goal_constant is a RUNTIME probe over observed frames, so a "
            "whole-board trope (C_UNIFORMITY) or a colour-elimination predicate "
            "(B_COLOUR_ELIMINATION) is caught too -- constant for a different reason, but "
            "just as uninformative to the search.",
        },
    }
    (HERE / "pre" / "detector_coverage.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
