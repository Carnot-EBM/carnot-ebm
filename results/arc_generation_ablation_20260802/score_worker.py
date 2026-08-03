"""Score ONE engine in its OWN process, so a hostile or non-terminating engine can be killed.

WHY A WORKER. `WorldModelVerifier.score` EXECUTES generated code and is not internally bounded. A
non-terminating induced engine wedged a generation loop for 13 minutes on 2026-07-31 in this
project; reached from a scoring pass it would wedge this one just as thoroughly. `score.py` runs
this file with `subprocess.run(timeout=...)`, so a hang becomes a recorded `worker_timeout` --
a MISSING OBSERVATION, never a zero.

It also serves the REACHABILITY PROBE (`--oracle` / `--identity`), deliberately through the SAME
code path: a probe that proves the metric reachable using a different scorer than the one the arms
are graded with has proved nothing about the arms.

Reads a JSON job from argv[1], prints one JSON line on stdout.
"""

from __future__ import annotations

import json
import os
import pathlib
import pickle
import sys
import time

REPO = pathlib.Path(os.environ["CARNOT_REPO"])
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# This process runs untrusted code and needs no accelerator; denying it one also stops a generated
# engine from grabbing a card another session on this shared machine owns.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(REPO / "python"))


def score_block(e3, engine, transitions) -> dict:
    """One held-out block's full VerifyResult, flattened. `measurable: False` when the block is
    EMPTY -- which is a coverage gap, not a zero, and the caller must not average it in."""
    if not transitions:
        return {"measurable": False, "n": 0, "reason": "empty_block"}
    vr = e3.WorldModelVerifier(list(transitions)).score(engine)
    return {
        "measurable": True,
        "n": int(vr.n),
        "n_correct": int(vr.n_correct),
        "accuracy": round(float(vr.accuracy), 6),
        "n_changing": int(vr.n_changing),
        "n_changes_correct": int(vr.n_changes_correct),
        # THE PRIMARY. n_changes_correct / n_changing, where n_changes_correct counts WHOLE-GRID
        # EXACT matches on changing rows -- an exact-match rate, not a cell fraction.
        "change_accuracy": round(float(vr.change_accuracy), 6),
        # The cell measures. cell_recall is the quantity the brief's prose describes;
        # change_fidelity scores over the UNION of changed-by-reality and changed-by-engine, so a
        # spurious write costs what a miss costs.
        "cell_recall": round(float(vr.cell_recall), 6),
        "change_fidelity": round(float(vr.change_fidelity), 6),
        "correct_changed_cells": int(vr.correct_changed_cells),
        "spurious_changed_cells": int(vr.spurious_changed_cells),
        "invented_changed_cells": int(getattr(vr, "invented_changed_cells", 0)),
        "n_noop": int(vr.n_noop),
        "n_noop_hallucinated": int(vr.n_noop_hallucinated),
        "n_levelup_rows_excluded": int(getattr(vr, "n_levelup_rows_excluded", 0)),
        "error": getattr(vr, "error", None),
    }


def substantive_split(transitions, min_changed_cells: int = 2):
    """Split gradable changing rows into SUBSTANTIVE (reality changed >= min_changed_cells) and
    TRIVIAL (exactly one cell changed).

    WHY THIS EXISTS -- it was added because of a measured result, not a hunch. `bp35__r1__antiid`
    scored change_accuracy 0.5662 on the fresh block, clearing the brief's >= 0.5 target on a
    plain-branch game over 219 rows, with zero leakage and 120 distinct correct next_grids. It
    looked like the first target hit in the record. Decomposing WHICH rows it got right showed all
    124 of them change EXACTLY ONE CELL (min = median = max = 1), while every row where reality
    changed more than one cell (median 47) was wrong. The engine had induced the single-cell
    progress-counter tick at row 63 and none of the game's dynamics.

    `change_accuracy` weights a one-cell HUD tick identically to a 47-cell state transition, so an
    engine that models only the counter can clear the headline bar without modelling the game. The
    substantive stratum is the same metric restricted to rows where something other than a counter
    moved. It is REPORTED ALONGSIDE the full metric, never as a replacement, and both denominators
    are recorded -- no row is silently censored, it is only stratified.
    """
    import numpy as np

    sub, triv = [], []
    for t in transitions:
        g0, g1 = np.asarray(t.grid), np.asarray(t.next_grid)
        if t.level_after > t.level_before or np.array_equal(g0, g1):
            continue
        (sub if int((g0 != g1).sum()) >= min_changed_cells else triv).append(t)
    return sub, triv


def distance_to_exact(engine, transitions, limit: int = 260) -> dict:
    """How FAR from an exact whole-grid match is the engine on each changing held-out row?

    WHY THIS IS NOT OPTIONAL. `change_accuracy` is an EXACT-match rate, so it reports 0.0000 both
    for an engine that returns its input and for an engine that gets 50 of 52 changed cells right
    and misses two. Those are different findings with different fixes -- the 2026-08-01 census
    found exactly that case (ls20, wrong on a two-row marker advancing one column per action, i.e.
    a step counter) sitting at change_accuracy 0.0000 with cell_recall 0.9615. Without this
    channel a null on the primary cannot distinguish 'the model declines to model' from 'the model
    is one named mechanic away'.

    Counts cells where the prediction differs from reality, over the FULL grid (so a spurious
    write in an untouched region costs as much as a missed change -- the same symmetry
    `change_fidelity` uses). A shape mismatch is recorded as such rather than as a huge distance.
    """
    import numpy as np

    dists, shape_bad, raised = [], 0, 0
    for t in list(transitions)[:limit]:
        g0, g1 = np.asarray(t.grid), np.asarray(t.next_grid)
        if t.level_after > t.level_before or np.array_equal(g0, g1):
            # not a gradable changing row -- excluded from BOTH sides, as the verifier does
            continue
        try:
            pred = np.asarray(engine(g0.copy(), t.action, t.data))
        except Exception:  # noqa: BLE001
            raised += 1
            continue
        if pred.shape != g1.shape:
            shape_bad += 1
            continue
        dists.append(int((pred != g1).sum()))
    out = {
        "n_rows": len(dists),
        "n_shape_mismatch": shape_bad,
        "n_raised": raised,
        "n_true_changed_cells_median": None,
    }
    if dists:
        s = sorted(dists)
        n = len(s)
        out.update(
            {
                "wrong_cells_min": s[0],
                "wrong_cells_median": s[n // 2],
                "wrong_cells_max": s[-1],
                "n_rows_exact": sum(1 for d in s if d == 0),
                "n_rows_within_1": sum(1 for d in s if d <= 1),
                "n_rows_within_2": sum(1 for d in s if d <= 2),
                "n_rows_within_5": sum(1 for d in s if d <= 5),
                "n_rows_within_10": sum(1 for d in s if d <= 10),
            }
        )
    return out


def identity_probe(engine, transitions, limit: int = 40) -> dict:
    """Does the engine change ANYTHING on the actions it was shown? This is the mechanism witness
    for the anti-identity arm, and the census's top-level class. `return grid` on every branch and
    `out = grid.copy(); return out` are both identity here -- behaviour, not syntax, because an
    AST reading of this misclassified 12 of 27 live-store engines that write THROUGH the parameter
    (`grid[py,px] = 15; return grid`) as identity and had to be corrected."""
    import numpy as np

    changed = probed = raised = 0
    for t in list(transitions)[:limit]:
        probed += 1
        try:
            pred = np.asarray(engine(np.asarray(t.grid).copy(), t.action, t.data))
        except Exception:  # noqa: BLE001
            raised += 1
            continue
        if pred.shape != np.asarray(t.grid).shape or not np.array_equal(pred, np.asarray(t.grid)):
            changed += 1
    return {
        "n_probed": probed,
        "n_changed": changed,
        "n_raised": raised,
        "is_identity": bool(probed > 0 and changed == 0 and raised < probed),
    }


def main() -> int:
    t0 = time.time()
    job = json.loads(pathlib.Path(sys.argv[1]).read_text())
    from carnot.agentic import arc_executable_world_model as e3

    with open(job["windows_pkl"], "rb") as fh:
        store = pickle.load(fh)
    s = store[job["game"]]
    out: dict = {"cell_id": job.get("cell_id"), "game": job["game"], "mode": job["mode"]}

    if job["mode"] == "oracle":
        # REACHABILITY PROBE. A HAND-WRITTEN lookup table over the rows being scored -- NOT an
        # induced engine, NOT a solve, and NOT a claim about any capability. Its only job is to
        # answer the question a prior arm failed to ask: can this metric, on these exact rows,
        # with this exact scorer, return anything other than 0? If it cannot, every "zero" in the
        # run is arithmetically forced rather than measured.
        import numpy as np

        table = {}
        for blk in ("tail", "fresh"):
            for t in s[blk]:
                key = (np.asarray(t.grid).tobytes(), int(t.action), repr(t.data))
                table[key] = np.asarray(t.next_grid)

        def engine(grid, action, data=None):
            g = np.asarray(grid)
            return table.get((g.tobytes(), int(action), repr(data)), g)

    elif job["mode"] == "identity":
        # The floor: an engine that returns its input. Shows the metric DISCRIMINATES rather than
        # merely being non-constant -- identity must score 0 on changing rows by construction.
        import numpy as np

        def engine(grid, action, data=None):
            return np.asarray(grid)

    else:
        # A real induced engine, loaded through the SHIPPED loader so a defect that the live path
        # would hit is hit here too.
        engine_src = pathlib.Path(job["engine_path"]).read_text()
        tmp = pathlib.Path(job["tmp_e3"]) / job["game"]
        tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "world_model.py").write_text(engine_src)
        e3.E3_DIR = pathlib.Path(job["tmp_e3"])
        try:
            engine, ilc = e3.load_engine(job["game"])
        except Exception as exc:  # noqa: BLE001
            print(
                json.dumps(
                    {**out, "status": "load_failed", "error": f"{type(exc).__name__}: {exc}"[:300]}
                )
            )
            return 0
        if engine is None:
            print(json.dumps({**out, "status": "no_engine_symbol"}))
            return 0
        out["is_level_complete_present"] = ilc is not None

    try:
        out["tail"] = score_block(e3, engine, s["tail"])
        out["fresh"] = score_block(e3, engine, s["fresh"])
        out["shown_train"] = score_block(e3, engine, s["shown"])  # TRAINING, never generalization
        out["identity_probe"] = identity_probe(engine, s["shown"])
        out["distance_tail"] = distance_to_exact(engine, s["tail"])
        out["distance_fresh"] = distance_to_exact(engine, s["fresh"])
        # STRATIFIED, not censored: both halves are scored and both denominators recorded.
        for blk in ("tail", "fresh"):
            sub, triv = substantive_split(s[blk])
            out[f"{blk}_substantive"] = score_block(e3, engine, sub)
            out[f"{blk}_substantive"]["n_rows_in_stratum"] = len(sub)
            out[f"{blk}_trivial_1cell"] = score_block(e3, engine, triv)
            out[f"{blk}_trivial_1cell"]["n_rows_in_stratum"] = len(triv)
        out["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        out["status"] = "score_raised"
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    out["worker_s"] = round(time.time() - t0, 2)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
