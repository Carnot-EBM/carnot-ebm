#!/usr/bin/env python3
"""METRIC-HEADROOM, STEP 1a -- score ONE frozen induction candidate under EVERY candidate metric.

WHY THIS RUNS IN ITS OWN KILLABLE SUBPROCESS, and why that is not optional here.

This module executes LLM-WRITTEN code. On 2026-07-31 the best-of-N generation loop wedged for 13
minutes in state R at 32% CPU with both GPUs idle, spinning inside `validate_engine_code`'s dry
run of a non-terminating induced engine (ft09 candidate 5 -- which is one of the 48 candidates
this very module re-scores). The shipped fix was `arc_engine_static_validation.dry_run_defects`:
bound the engine with a process that can be KILLED FROM OUTSIDE rather than with anything inside
the interpreter. This module reuses that pattern directly, and for the same two reasons the
sibling `gate_worker.py` states:

  * NON-TERMINATION. Every metric below invokes `engine(...)` once per transition. A single call
    that loops forever is unbounded, and a signal-based in-process alarm would be SWALLOWED --
    the scoring loops wrap engine invocation in `except Exception` because the exception IS an
    observation, so a SIGALRM-raised exception is caught and recorded as an ordinary failure. A
    hang would silently become a CLEAN ZERO, which is worse than the hang because it is invisible.
  * INTERPRETER DAMAGE. Generated code was mechanically defective in 22 of 36 attempts in the
    preceding phase. A C-level fault inside a numpy call takes down whatever interpreter it
    shares.

So: NO generated code is executed in the driver's interpreter at all. The driver
(`score_metrics.py`) starts one of these per candidate and kills it on timeout.

WHAT IS MEASURED, and why each metric is here rather than being an obvious variant of another.

The problem this phase exists to solve: `experiment_6018` A/B'd object perception and returned
`unmeasurable_instrument_floor` because its PRE-REGISTERED primary -- held-out exact-full-grid
match -- was exactly 0.0 in both arms on all 168 cells, giving zero discordant per-game pairs and
no possible test. A metric that cannot vary has not measured a null. So the question is not "does
object perception help" but "is there ANY graded held-out metric with headroom", and that must be
answered honestly, including the answer "no".

Five families are computed, deliberately spanning different granularities so that a floor at one
granularity does not silently propagate to all of them:

  1. TRANSITION-EXACT (the control, and the thing that floored):
       `exact_match_accuracy`   full-grid equality over ALL held-out rows
       `change_exact_accuracy`  full-grid equality over CHANGING rows only
     Both are all-or-nothing per transition. The second is the honest one: the first can be
     earned on a no-op-heavy split by an engine that predicts nothing ever changes.

  2. CELL-GRANULAR, from the SHIPPED `WorldModelVerifier` (`cell_recall`, `change_fidelity`,
     `correct_changed_cells`, `spurious_changed_cells`, `invented_changed_cells`). These are not
     reinvented here -- the task is to start from what exists. `change_fidelity` is the symmetric
     one (a spurious write costs what a miss costs); `cell_recall` is explicitly documented in the
     repo as blind to spurious writes.

  3. WHOLE-GRID DISTANCE (`grid_agreement_all`, `grid_agreement_changing`): 1 - normalised
     Hamming between predicted and observed next grid. The most forgiving metric available, and
     included precisely so that "even THIS is floored" is a statement the run can make.

  4. CHANGE-SET GEOMETRY (`changed_cell_jaccard`): Jaccard between the set of cells the engine
     WROTE and the set reality CHANGED, ignoring the values written. Asks the strictly weaker
     question "did you change the right cells" and separates "knows where the action lands but
     not what it draws" from "has no idea".

  5. OBJECT-GRANULAR (`object_match_iou`, `object_inventory_jaccard`, `object_positional_jaccard`).
     These are the ones this phase ADDS, and they exist because the treatment under test is
     object perception. The documented top-10 ARC-AGI-3 pipeline feeds its solver 4-connected
     object segmentation with translation-invariant shape hashes rather than the raw numeric grid,
     and `objects_block()` in this repo already builds exactly that table. If a representation
     change helps at all, the level at which it should first show up is the OBJECT level -- so
     grading only at cell or whole-grid granularity could floor a real effect. `object_match_iou`
     is the graded one (each true object scores its best pixel-IoU against a predicted object of
     the same colour, so an object off by one pixel scores ~0.97 rather than 0); the two Jaccards
     are all-or-nothing set comparisons kept alongside as the harsher reads.

     Colour equality is REQUIRED for an object match. Blobs are same-colour connected components
     by construction, so matching a red object to a blue one of identical shape would be scoring a
     coincidence.

WHAT IS NOT DONE HERE. No GPU, no generation, no model of any kind. Every candidate's completion
text was frozen on 2026-07-31 and is read off disk. This is arithmetic over recorded transitions.
"""

from __future__ import annotations

import json
import os
import pathlib
import pickle
import sys
import time

# NO HARDCODED ABSOLUTE PATH. `.../results/arc_metric_headroom_20260801/harness/x.py` -> up 3.
# CLAUDE.md "Test-Run Record Integrity" rule 4: an absolute path baked into source is a defect,
# because a fresh clone then writes into the original author's checkout.
REPO = pathlib.Path(__file__).resolve().parents[3]
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("CARNOT_ARC_E3_DIR", "/tmp/arc_metric_headroom/e3")
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)
sys.path.insert(0, str(REPO / "python"))


def _round(x, n=6):
    return None if x is None else round(float(x), n)


def main() -> int:  # noqa: C901
    with open(sys.argv[1]) as fh:
        job = json.loads(fh.read())
    out: dict = {"status": "ok"}

    import numpy as np
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic.arc_color_blob_salience import connected_color_blobs, object_hash

    with open(job["code_path"]) as fh:
        code = fh.read()
    ns: dict = {"np": np, "numpy": np}
    try:
        exec(compile(code, job["code_path"], "exec"), ns)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        # No engine exists. This is a genuine False on every metric, NOT a missing observation --
        # the same distinction score_bon.py drew between `unrunnable:*` and `gate_timeout`.
        out["status"] = f"unrunnable:{type(exc).__name__}"
        out["error"] = str(exc)[:240]
        print(json.dumps(out))
        return 0
    engine = ns.get("engine")
    out["has_engine"] = callable(engine)
    if not callable(engine):
        out["status"] = "no_engine"
        print(json.dumps(out))
        return 0

    with open(job["heldout_pkl"], "rb") as fh:
        rows = list(pickle.load(fh))
    if not rows:
        out["status"] = "no_heldout_rows"
        out["heldout_n"] = 0
        print(json.dumps(out))
        return 0

    # ---- family 2: the SHIPPED verifier, unmodified ---------------------------------------
    # Started from what exists rather than reimplemented, per the task's explicit instruction.
    # Defaults are the submitted-path defaults (HUD mask flag off), matching gate_worker.py so
    # the two runs' shared fields are directly comparable rather than two different measurements
    # wearing one name.
    vr = e3.WorldModelVerifier(rows).score(engine)
    n_changing = int(getattr(vr, "n_changing", 0) or 0)
    out.update(
        {
            "heldout_n": int(getattr(vr, "n", 0) or 0),
            "exact_match_accuracy": _round(getattr(vr, "accuracy", 0.0)),
            "heldout_n_changing": n_changing,
            "heldout_n_correct": int(getattr(vr, "n_correct", 0) or 0),
            "heldout_n_changes_correct": int(getattr(vr, "n_changes_correct", 0) or 0),
            # None, not 0.0, where nothing changing was held out: the value meaning "perfect" and
            # the value meaning "not measurable" must not be the same number.
            "change_exact_accuracy": (
                _round(getattr(vr, "change_accuracy", 0.0)) if n_changing else None
            ),
            "cell_recall": (_round(getattr(vr, "cell_recall", 0.0)) if n_changing else None),
            "change_fidelity": (
                _round(getattr(vr, "change_fidelity", 0.0)) if n_changing else None
            ),
            "correct_changed_cells": int(getattr(vr, "correct_changed_cells", 0) or 0),
            "spurious_changed_cells": int(getattr(vr, "spurious_changed_cells", 0) or 0),
            "invented_changed_cells": int(getattr(vr, "invented_changed_cells", 0) or 0),
            "heldout_n_noop": int(getattr(vr, "n_noop", 0) or 0),
            "heldout_n_noop_hallucinated": int(getattr(vr, "n_noop_hallucinated", 0) or 0),
            "hud_mask_status": str(getattr(vr, "hud_mask_status", "disabled")),
        }
    )

    # ---- families 1, 3, 4, 5: computed per transition here ---------------------------------
    def _objects(grid: np.ndarray):
        """Full 4-connected same-colour partition + translation-invariant shape id.

        `connected_color_blobs(min_pixels=1, max_component_fraction=1.0)` is the same call
        `blob_topology` makes, so every pixel belongs to exactly one returned blob. The
        containment/adjacency flood-fills that `blob_topology` adds on top are ~140x more
        expensive and are not needed to compare two partitions, so they are skipped.
        """
        blobs = connected_color_blobs(np.asarray(grid), min_pixels=1, max_component_fraction=1.0)
        labels = np.full(np.asarray(grid).shape, -1, dtype=np.int32)
        colors, sizes, keys, inv_keys = [], [], [], []
        for i, b in enumerate(blobs):
            for y, x in b.cells:
                labels[y, x] = i
            colors.append(int(b.color))
            sizes.append(int(b.pixel_count))
            h = object_hash(b)
            inv_keys.append(h)  # translation-INVARIANT: colour + normalised shape
            keys.append((h, int(b.bbox[0]), int(b.bbox[1])))  # + absolute position
        return (
            labels,
            np.asarray(colors, dtype=np.int64),
            np.asarray(sizes, dtype=np.int64),
            keys,
            inv_keys,
        )

    def _multiset_jaccard(a: list, b: list) -> float:
        """|A n B| / |A u B| over MULTISETS. Multiset, not set: three identical wall segments are
        three objects, and collapsing them would let an engine that deletes two of them score
        perfectly."""
        from collections import Counter

        ca, cb = Counter(a), Counter(b)
        inter = sum((ca & cb).values())
        union = sum((ca | cb).values())
        return float(inter / union) if union else 1.0

    def _object_match_iou(pred: np.ndarray, true: np.ndarray) -> tuple[float, float, float]:
        """GRADED object correspondence: (recall, precision, f1) over best pixel-IoU matches.

        For each TRUE object, its score is the best pixel-IoU achieved by any PREDICTED object of
        the SAME COLOUR; recall is the pixel-count-weighted mean of those. Precision is the mirror
        image (each predicted object scored against the true partition), which is what makes an
        engine that shatters the board into invented objects lose -- a recall-only read would not
        see it, the same asymmetry the repo already documents for `cell_recall`.

        Weighting by pixel count rather than counting objects uniformly is deliberate: an ARC
        frame's partition is dominated by single-pixel specks, and an unweighted mean would be a
        measurement of speck reproduction rather than of object dynamics.
        """
        lt, ct, st, _, _ = _objects(true)
        lp, cp, sp, _, _ = _objects(pred)
        nt, npd = len(ct), len(cp)
        if nt == 0 or npd == 0:
            return (1.0, 1.0, 1.0) if (nt == 0 and npd == 0) else (0.0, 0.0, 0.0)
        both = (lt >= 0) & (lp >= 0)
        pair = lt[both].astype(np.int64) * npd + lp[both].astype(np.int64)
        inter = np.bincount(pair, minlength=nt * npd).reshape(nt, npd).astype(np.float64)
        union = st[:, None] + sp[None, :] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
        # Same-colour constraint: blobs are single-colour components, so a cross-colour "match"
        # would be scoring a coincidence of shape.
        iou = np.where(ct[:, None] == cp[None, :], iou, 0.0)
        best_t = iou.max(axis=1)
        best_p = iou.max(axis=0)
        rec = float((best_t * st).sum() / max(1, st.sum()))
        prec = float((best_p * sp).sum() / max(1, sp.sum()))
        f1 = float(2 * rec * prec / (rec + prec)) if (rec + prec) > 0 else 0.0
        return rec, prec, f1

    agree_all: list[float] = []
    agree_chg: list[float] = []
    jacc: list[float] = []
    obj_f1: list[float] = []
    obj_rec: list[float] = []
    obj_inv: list[float] = []
    obj_pos: list[float] = []
    n_engine_raised = 0
    n_shape_mismatch = 0
    n_levelup_excluded = 0
    t0 = time.monotonic()

    for t in rows:
        # Same exclusion the shipped verifier makes: the completing action re-lays out the board,
        # so `next_grid` on a level-up is the NEXT level's opening screen and no engine induced
        # from THIS level can predict it. Grading it measures the renderer, not the engine.
        if t.level_after > t.level_before:
            n_levelup_excluded += 1
            continue
        g0 = np.asarray(t.grid)
        g1 = np.asarray(t.next_grid)
        changed = not np.array_equal(g0, g1)
        try:
            pred = np.asarray(engine(t.grid.copy(), t.action, t.data))
        except Exception:  # noqa: BLE001
            n_engine_raised += 1
            agree_all.append(0.0)
            if changed:
                agree_chg.append(0.0)
                jacc.append(0.0)
                obj_f1.append(0.0)
                obj_rec.append(0.0)
                obj_inv.append(0.0)
                obj_pos.append(0.0)
            continue
        if pred.shape != g1.shape:
            n_shape_mismatch += 1
            agree_all.append(0.0)
            if changed:
                agree_chg.append(0.0)
                jacc.append(0.0)
                obj_f1.append(0.0)
                obj_rec.append(0.0)
                obj_inv.append(0.0)
                obj_pos.append(0.0)
            continue

        a = float((pred == g1).mean())
        agree_all.append(a)
        if not changed:
            continue
        agree_chg.append(a)

        m = g1 != g0  # what reality changed
        wrote = pred != g0  # what the engine changed
        inter = int((m & wrote).sum())
        union = int((m | wrote).sum())
        jacc.append(float(inter / union) if union else 1.0)

        rec, _prec, f1 = _object_match_iou(pred, g1)
        obj_rec.append(rec)
        obj_f1.append(f1)
        _lp, _cp, _sp, kp, ip = _objects(pred)
        _lt, _ct, _st, kt, it = _objects(g1)
        obj_inv.append(_multiset_jaccard(ip, it))
        obj_pos.append(_multiset_jaccard(kp, kt))

    def _mean(xs):
        return _round(sum(xs) / len(xs)) if xs else None

    out.update(
        {
            "grid_agreement_all": _mean(agree_all),
            "grid_agreement_changing": _mean(agree_chg),
            "changed_cell_jaccard": _mean(jacc),
            "object_match_iou": _mean(obj_f1),
            "object_match_recall": _mean(obj_rec),
            "object_inventory_jaccard": _mean(obj_inv),
            "object_positional_jaccard": _mean(obj_pos),
            "n_engine_raised": n_engine_raised,
            "n_shape_mismatch": n_shape_mismatch,
            "n_levelup_excluded": n_levelup_excluded,
            "n_graded_changing": len(agree_chg),
            "worker_wall_s": round(time.monotonic() - t0, 3),
        }
    )
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
