#!/usr/bin/env python3
"""PHASE 2, STEP 6 -- score every A/B completion with the PRODUCTION trust gate's own metrics.

WHY, AND WHAT THIS CAN AND CANNOT SAY.

`repair_ab.py`'s `usable` bar is "mechanically clean AND not inert". That is the right bar for
the question it answers -- did the repair produce code that WORKS -- but it is not the gate's
bar. This script runs the SHIPPED `WorldModelVerifier` over each completion so the same engines
carry the numbers the live path would judge them by: `accuracy` (the full-grid byte-exact match
the live call gates at `min_heldout_accuracy=1.0`), `cell_recall`, `change_fidelity`,
`invented_changed_cells` and the no-op hallucination rate.

**THESE NUMBERS ARE IN-SAMPLE AND MUST NOT BE READ AS GATE OUTCOMES.** The live gate scores a
HELD-OUT suffix: `arc_llm_reinduction._proposal_prefix` keeps `round(n/3)` transitions out of the
proposer's prompt and the verifier grades on those. Every transition available here was IN the
induce prompt, so a high `accuracy` here is partly memorisation and a genuine held-out score
requires the live loop. What these numbers CAN do is separate an engine that models something
from one that models nothing -- an in-sample `cell_recall` of 0.0 is damning regardless of the
split, and that is the discrimination this adds over the boolean `usable`.

Nothing here changes any threshold. It reports what the shipped verifier says.
"""

from __future__ import annotations

import json
import os
import pathlib
import pickle
import sys
import time

REPO = "/home/ianblenke/github.com/ianblenke/carnot"
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "gate_scores.json"

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault(
    "CARNOT_ARC_E3_DIR", os.environ.get("P2_SCRATCH", "/tmp/arc_p2_scratch") + "/e3_score"
)
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)
sys.path.insert(0, os.path.join(REPO, "python"))

GAMES = ["ft09", "tu93", "lp85", "tn36", "sc25"]


def main() -> int:
    import numpy as np

    from carnot.agentic import arc_executable_world_model as e3

    rows = []
    for game in GAMES:
        ab_path = HERE / "ab" / game / "ab.json"
        tr_path = HERE / "capture" / game / "transitions2.pkl"
        if not ab_path.exists() or not tr_path.exists():
            rows.append({"game": game, "status": "NOT_RUN"})
            continue
        ab = json.loads(ab_path.read_text())
        if ab.get("status") != "ok":
            rows.append({"game": game, "status": ab.get("status")})
            continue
        with open(tr_path, "rb") as fh:
            trans = pickle.load(fh)
        for r in ab.get("rows", []):
            if r.get("status") != "ok":
                continue
            text = (HERE / "ab" / game / r["completion_file"]).read_text(errors="replace")
            code = e3._extract_python(text) or text.strip()
            ns: dict = {"np": np, "numpy": np}
            row = {
                "game": game,
                "tag": r["tag"],
                "arm": r["tag"].split("_", 1)[1],
                "usable": r["usable"],
                "defect_kinds": r["defect_kinds"],
            }
            try:
                exec(compile(code, r["completion_file"], "exec"), ns)  # noqa: S102
                engine = ns.get("engine")
                assert callable(engine)
            except Exception as exc:  # noqa: BLE001
                row["status"] = f"unrunnable:{type(exc).__name__}"
                rows.append(row)
                continue
            vr = e3.WorldModelVerifier(trans).score(engine)
            row.update(
                status="ok",
                # `accuracy` is what the live call gates at min_heldout_accuracy=1.0 -- but on a
                # HELD-OUT suffix, which this is not. See the module docstring.
                in_sample_accuracy=round(float(getattr(vr, "accuracy", 0.0) or 0.0), 4),
                in_sample_cell_recall=round(float(getattr(vr, "cell_recall", 0.0) or 0.0), 4),
                in_sample_change_fidelity=round(
                    float(getattr(vr, "change_fidelity", 0.0) or 0.0), 4
                ),
                n_changing=int(getattr(vr, "n_changing", 0) or 0),
                n_changes_correct=int(getattr(vr, "n_changes_correct", 0) or 0),
                invented_changed_cells=int(getattr(vr, "invented_changed_cells", 0) or 0),
                n_noop=int(getattr(vr, "n_noop", 0) or 0),
                n_noop_hallucinated=int(getattr(vr, "n_noop_hallucinated", 0) or 0),
            )
            rows.append(row)

    out = {
        "generated_by": "results/arc_engine_validation_20260731/harness/score_with_gate.py",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "IN_SAMPLE_WARNING": (
            "Every transition scored here was IN the induce prompt. The live gate grades a "
            "HELD-OUT suffix (arc_llm_reinduction._proposal_prefix keeps round(n/3) out). These "
            "are NOT gate outcomes; they separate an engine that models something from one that "
            "models nothing."
        ),
        "rows": rows,
    }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    scored = [r for r in rows if r.get("status") == "ok"]
    for r in scored:
        print(
            f"{r['game']:5s} {r['arm']:8s} usable={str(r['usable']):5s} "
            f"acc={r['in_sample_accuracy']:.3f} recall={r['in_sample_cell_recall']:.3f} "
            f"fid={r['in_sample_change_fidelity']:.3f} "
            f"changes {r['n_changes_correct']}/{r['n_changing']} "
            f"invented={r['invented_changed_cells']} "
            f"noop_halluc={r['n_noop_hallucinated']}/{r['n_noop']}"
        )
    print(f"\nwrote {OUT}  ({len(scored)} completions scored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
