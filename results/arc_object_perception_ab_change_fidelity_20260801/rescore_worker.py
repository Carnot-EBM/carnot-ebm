#!/usr/bin/env python3
"""Re-score ONE frozen A/B cell under the FULL VerifyResult field set, in a killable process.

WHY THIS EXISTS AT ALL. `run_ab.py` recorded seven of `VerifyResult`'s channels and dropped
the four that name the one failure mode `change_fidelity` is structurally blind to:

    n_noop / n_noop_hallucinated / noop_hallucination_rate / noop_channel_measurable
    invented_changed_cells

Adversarial review constructed the engine that exploits the gap -- perfect on every CHANGING
transition AND hallucinating a change on every NO-OP -- and it was reproduced here on frozen
data: on sc25 it scores change_fidelity 1.0000 with full-grid accuracy 0.0714, and
`spurious_changed_cells` (the secondary an operator would reach for) reads a clean 0, because
that counter is only accumulated INSIDE changing transitions. Only `noop_hallucination_rate`
names it, and that field was not recorded.

WHY A POST-HOC PASS RATHER THAN AN EDIT TO run_ab.py. run_ab.py is mid-collection and its
bytes are the as-run record (`run_ab.py.frozen`). Editing a harness while it runs would make
the artifact's provenance a lie. Every input this pass needs is already frozen on disk: the
engine text in `e3_store/<game>__r<rep>__<tag>/`, and a window rebuilt by the SAME
deterministic `build_progress_window` + `_split_prefix_heldout` calls run_ab.py makes. So this
adds fields without touching the treatment, and it RE-DERIVES the two fields run_ab.py already
recorded as a reproduction check -- if `change_fidelity` here disagrees with the value in the
cell, the rebuild is not deterministic and the whole pass is void rather than quietly averaged
in.

WHY A SUBPROCESS. This executes LLM-written code. On 2026-07-31 a non-terminating induced
engine (ft09 candidate 5) wedged a generation loop for 13 minutes, and an in-process alarm
would be SWALLOWED here -- the scoring loop wraps `engine(...)` in `except Exception`, so a
SIGALRM-raised exception is caught and recorded as an ordinary per-transition failure. A hang
would silently become a CLEAN ZERO, which is worse than the hang because it is invisible.
"""

from __future__ import annotations

import ast
import json
import os
import pathlib
import pickle
import sys

REPO = pathlib.Path(__file__).resolve()
while REPO.name != "carnot" and REPO.parent != REPO:
    REPO = REPO.parent
# The scratchpad is not inside the repo, so derive the repo from the env the driver sets.
REPO = pathlib.Path(os.environ["CARNOT_REPO"])

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CARNOT_ARC_E3_DIR", "/tmp/arc_rescore/e3")
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)
sys.path.insert(0, str(REPO / "python"))


def reads_data_param(src: str) -> bool:
    """Does `engine`'s body ever READ its third parameter?

    AST, never a substring scan: the word `data` appears in the prose docstring of nearly
    every induced engine, so `"data" in src` would call a coordinate-BLIND engine aware.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == "engine":
            if len(fn.args.args) < 3:
                return False
            nm = fn.args.args[2].arg
            return any(
                isinstance(n, ast.Name) and n.id == nm and isinstance(n.ctx, ast.Load)
                for n in ast.walk(fn)
            )
    return False


def main() -> int:  # noqa: C901
    job = json.loads(pathlib.Path(sys.argv[1]).read_text())
    out: dict = {"status": "ok", "cell": job["cell"]}

    import numpy as np
    from carnot.agentic import arc_executable_world_model as e3

    with open(job["window_pkl"], "rb") as fh:
        win = pickle.load(fh)
    held = list(win["held"])

    code_path = pathlib.Path(job["code_path"])
    if not code_path.exists():
        out["status"] = "no_engine_file"
        print(json.dumps(out))
        return 0
    src = code_path.read_text()
    out["reads_data_param"] = reads_data_param(src)

    ns: dict = {"np": np, "numpy": np}
    try:
        exec(compile(src, str(code_path), "exec"), ns)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        out["status"] = f"unrunnable:{type(exc).__name__}"
        out["error"] = str(exc)[:200]
        print(json.dumps(out))
        return 0
    engine = ns.get("engine")
    if not callable(engine):
        out["status"] = "no_engine_symbol"
        print(json.dumps(out))
        return 0

    vr = e3.WorldModelVerifier(list(held)).score(engine)
    out["full"] = {
        # --- the seven run_ab.py already recorded (re-derived as a reproduction check) ---
        "n": int(vr.n),
        "n_correct": int(vr.n_correct),
        "accuracy": round(float(vr.accuracy), 6),
        "cell_recall": round(float(vr.cell_recall), 6),
        "n_changing": int(vr.n_changing),
        "n_changes_correct": int(vr.n_changes_correct),
        "change_accuracy": round(float(vr.change_accuracy), 6),
        "change_fidelity": round(float(vr.change_fidelity), 6),
        "correct_changed_cells": int(vr.correct_changed_cells),
        "spurious_changed_cells": int(vr.spurious_changed_cells),
        # --- THE FOUR THAT WERE MISSING (adversarial review, 2026-08-01) ---------------
        "n_noop": int(vr.n_noop),
        "n_noop_hallucinated": int(vr.n_noop_hallucinated),
        "noop_hallucination_rate": round(float(vr.noop_hallucination_rate), 6),
        "noop_channel_measurable": bool(vr.noop_channel_measurable),
        "invented_changed_cells": int(vr.invented_changed_cells),
        "invented_change_rate": round(float(vr.invented_change_rate), 6),
        # --- provenance of the grading itself ------------------------------------------
        "hud_mask_status": str(vr.hud_mask_status),
        "n_levelup_rows_excluded": int(vr.n_levelup_rows_excluded),
    }

    # ---- ACTION / COORDINATE BLINDNESS PROBE ------------------------------------------
    # The headroom artifact disqualified four object metrics for ranking the INERT engine
    # above a real one. It never tested the other way a non-model can score well: an engine
    # that is CORRECT but cannot see the action or the click. Feed the same start grid every
    # (action, data) pair that actually occurs in the held-out rows -- arbitrary probes are
    # not enough, because an engine that keys on the two specific clicks the corpus contains
    # is a no-op on every other coordinate and would look constant.
    seen, probes = set(), []
    for t in held:
        k = (int(t.action), json.dumps(t.data, sort_keys=True))
        if k not in seen:
            seen.add(k)
            probes.append((int(t.action), t.data))
    probes += [(6, {"x": 0, "y": 0}), (6, {"x": 63, "y": 63})]
    grad = [
        t
        for t in held
        if t.level_after <= t.level_before
        and not np.array_equal(np.asarray(t.grid), np.asarray(t.next_grid))
    ]
    if grad:
        start = np.asarray(grad[0].grid)
        outs = set()
        for a, d in probes:
            try:
                outs.add(np.asarray(engine(start.copy(), a, d)).tobytes())
            except Exception:  # noqa: BLE001
                outs.add(b"<raise>")
        out["n_probes"] = len(probes)
        out["distinct_outputs_over_probes"] = len(outs)
        out["behaviourally_blind"] = len(outs) == 1
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
