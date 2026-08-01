#!/usr/bin/env python3
"""SECOND-CORPUS over-masking check, plus a same-row comparison against the explorer's mask.

WHY THIS EXISTS -- it is the direct answer to a red flag, not a bonus.
    The capture accepted a band on lf52 and su15. Those are EXACTLY the two games
    REQ-ARC-WMTE-6015 documents as over-masking cases:

        game   changed-cells-inside-mask   changing transitions, raw -> masked
        lf52          1.0000                       60 -> 0   (the entire game is deleted)
        su15          0.7568                       28 -> 1

    That table was measured on a DIFFERENT corpus (`collect_transitions(n=60, seed=0)` --
    RANDOM actions from reset) and on a DIFFERENT mask (the explorer's frame-space
    `edge_bar_hud_mask`). REQ-ARC-WMTE-6017's own corrigendum states the rule that makes
    both facts compatible: "a verdict is a statement about (mask, corpus), never about the
    mask alone" -- and records lf52's verdict FLIPPING between a random-action corpus
    (REFUSED) and a live episode (applied).

    So an accept on the winning-route corpus does NOT license masking generally. This
    script asks the two questions that decide it:

      1. Is my band the SAME band the explorer's detector picks? If it is a different row,
         the documented refusal is about a different set of cells and does not transfer.
      2. Does my mask survive the swallow guard on the RANDOM-ACTION corpus -- the corpus
         that produced the documented refusal?

    A mask that fails (2) is DEMOTED to suspected_false_positive, because the asymmetry is
    unchanged: over-masking destroys correctness, under-masking only costs efficiency. This
    check can only ever REMOVE a mask, never add one.

Spec: REQ-ARC-WMTE-6015 / REQ-ARC-WMTE-6017.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import numpy as np

REPO = "/home/ianblenke/github.com/ianblenke/carnot"
sys.path.insert(0, os.path.join(REPO, "python"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

OUT_DIR = os.path.join(REPO, "results", "arc_hud_mask_capture_20260801")

# The corpus scope REQ-ARC-WMTE-6015's table was measured at, quoted verbatim by
# REQ-ARC-WMTE-6019: "collect_transitions(n=60, seed=0) -- SIXTY actions, not 120".
RANDOM_CORPUS_N = 60
RANDOM_CORPUS_SEED = 0


def _mask_from(shape: list[int], rows: list[int], cols: list[int]) -> np.ndarray:
    m = np.zeros(tuple(shape), dtype=bool)
    for r in rows:
        m[r, :] = True
    for c in cols:
        m[:, c] = True
    return m


# A row/col counts as a BAND when the explorer covers at least this fraction of it. A strict
# `.all()` test does not work and briefly produced a FALSE CLAIM in this artifact: on tn36 the
# explorer masks row 1 columns 1..61 -- 61 of 64 cells, unmistakably the same progress bar this
# capture's detector finds -- and an `.all()` summary reported it as "no rows", from which the
# prose concluded "the explorer proposes nothing on tn36". It proposes almost exactly the same
# band. 0.9 is not tuned: it is simply low enough that a band missing its end-caps still reads
# as a band, and high enough that a single masked cell per row (lp85's column-0 mask, which
# touches all 64 rows with 1 cell each) never does. The raw cell-level overlap is reported
# alongside so no conclusion depends on this threshold.
BAND_COVERAGE_MIN = 0.9


def explorer_mask_bands(game: str) -> dict[str, Any]:
    """What does the EXPLORER's shipped frame-space detector mask on this game?

    Resolved through the same two calls the live path uses: `edge_bar_hud_mask` on the reset
    frame, then `logical_hud_mask` to bring it into logical coordinates. Returns BOTH the
    strictly-full rows/cols and the >=BAND_COVERAGE_MIN ones, plus the mask itself so the
    caller can compute a cell-level comparison that no band summary can distort.
    """
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_agi3_world_model import grid_of
    from carnot.agentic.arc_executable_world_model import detect_cell, logical_hud_mask
    from carnot.agentic.arc_hud_bar_detector import edge_bar_hud_mask

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = env.reset()
    cell = detect_cell(grid_of(f))
    fm = edge_bar_hud_mask(f)
    lm = logical_hud_mask(fm, cell) if fm is not None else None
    if lm is None:
        return {
            "resolved": False,
            "rows": [],
            "cols": [],
            "rows_full": [],
            "cols_full": [],
            "cell": int(cell),
            "n_masked_cells": 0,
            "_mask": None,
        }
    lm = np.asarray(lm, dtype=bool)
    h, w = lm.shape
    rows_full = [int(r) for r in range(h) if bool(lm[r, :].all())]
    cols_full = [int(c) for c in range(w) if bool(lm[:, c].all())]
    rows_band = [int(r) for r in range(h) if lm[r, :].sum() >= BAND_COVERAGE_MIN * w]
    cols_band = [int(c) for c in range(w) if lm[:, c].sum() >= BAND_COVERAGE_MIN * h]
    return {
        "resolved": True,
        "rows": rows_band,
        "cols": cols_band,
        "rows_full": rows_full,
        "cols_full": cols_full,
        "band_coverage_min": BAND_COVERAGE_MIN,
        "row_coverage_counts": {str(r): int(lm[r, :].sum()) for r in rows_band or rows_full},
        "col_coverage_counts": {str(c): int(lm[:, c].sum()) for c in cols_band or cols_full},
        "cell": int(cell),
        "logical_shape": [int(h), int(w)],
        "n_masked_cells": int(lm.sum()),
        "note": (
            "`rows`/`cols` are bands covered at least BAND_COVERAGE_MIN; `rows_full`/`cols_full` "
            "are the strictly-complete ones. They differ whenever the explorer masks a band "
            "minus its end-caps -- tn36 row 1 is 61 of 64 cells, a band by any reading and "
            "invisible to a strict test."
        ),
        "_mask": lm,
    }


def main() -> int:
    from carnot.agentic.arc_executable_world_model import (
        collect_transitions,
        hud_mask_swallow_check,
        hud_mask_swallow_clean,
    )

    t0 = time.time()
    with open(os.path.join(OUT_DIR, "per_game.json")) as fh:
        games = json.load(fh)["games"]

    out: list[dict] = []
    for g in games:
        if g.get("status") != "ok":
            continue
        game = g["game"]
        rows, cols = g["hud_rows"], g["hud_cols"]
        # The detector-vs-detector comparison must use every DETECTED band, not just the
        # stage-1 accepted ones: on cn04 and lp85 the detected band was excluded by the audit,
        # and comparing an empty accepted-mask against the explorer would report a
        # "disagreement" that is really this capture's own audit decision, not a difference
        # between the two detectors.
        det_rows = sorted({b["index"] for b in g["bands"] if b["axis"] == "row"})
        det_cols = sorted({b["index"] for b in g["bands"] if b["axis"] == "col"})
        rec: dict[str, Any] = {
            "game": game,
            "capture_mask_status": g["mask_status"],
            "capture_hud_rows": rows,
            "capture_hud_cols": cols,
            "capture_detected_rows": det_rows,
            "capture_detected_cols": det_cols,
        }
        try:
            rec["explorer_edge_bar_mask"] = explorer_mask_bands(game)
        except Exception as exc:
            rec["explorer_edge_bar_mask"] = {"error": repr(exc)[:300]}

        if rows or cols:
            try:
                trans, _cell = collect_transitions(game, n=RANDOM_CORPUS_N, seed=RANDOM_CORPUS_SEED)
                mask = _mask_from(g["logical_shape"], rows, cols)
                chk = hud_mask_swallow_check(trans, mask)
                rec["random_action_corpus"] = {
                    "n": RANDOM_CORPUS_N,
                    "seed": RANDOM_CORPUS_SEED,
                    "n_transitions_collected": len(trans),
                    "swallow_check": chk,
                    "clean": bool(hud_mask_swallow_clean(chk)),
                }
            except Exception as exc:
                rec["random_action_corpus"] = {"error": repr(exc)[:300]}
        else:
            rec["random_action_corpus"] = {
                "skipped": "no mask to judge -- the capture shipped nothing for this game"
            }

        exp = rec.get("explorer_edge_bar_mask") or {}
        exp_mask = exp.pop("_mask", None)
        same = None
        if exp.get("resolved"):
            same = (
                sorted(exp.get("rows", [])) == det_rows and sorted(exp.get("cols", [])) == det_cols
            )
        rec["same_band_as_explorer"] = same

        # Cell-level agreement, which no band summary can distort. Reported unconditionally so
        # a reader can re-judge `same_band_as_explorer` without trusting BAND_COVERAGE_MIN.
        if exp_mask is not None and (det_rows or det_cols):
            mine = _mask_from(g["logical_shape"], det_rows, det_cols)
            inter = int((mine & exp_mask).sum())
            union = int((mine | exp_mask).sum())
            rec["detected_band_vs_explorer_cells"] = {
                "n_cells_detected_by_this_capture": int(mine.sum()),
                "n_cells_masked_by_explorer": int(exp_mask.sum()),
                "n_cells_in_both": inter,
                "jaccard": round(inter / union, 6) if union else 0.0,
                "fraction_of_my_band_the_explorer_also_masks": (
                    round(inter / int(mine.sum()), 6) if int(mine.sum()) else None
                ),
            }
        else:
            rec["detected_band_vs_explorer_cells"] = None
        out.append(rec)
        cells = rec["detected_band_vs_explorer_cells"] or {}
        print(
            f"[xcorpus] {game}: detected rows={det_rows} cols={det_cols} | explorer "
            f"rows={exp.get('rows')} cols={exp.get('cols')} same={same} "
            f"jaccard={cells.get('jaccard')} | "
            f"random-corpus clean={rec['random_action_corpus'].get('clean')} "
            f"reason={(rec['random_action_corpus'].get('swallow_check') or {}).get('reason')}",
            flush=True,
        )

    payload = {
        "corpus": {
            "kind": "collect_transitions random actions from reset",
            "n": RANDOM_CORPUS_N,
            "seed": RANDOM_CORPUS_SEED,
            "why_this_scope": (
                "REQ-ARC-WMTE-6019 records that REQ-ARC-WMTE-6015's over-masking table was "
                "measured at n=60, seed=0, and that n=60 is a strict prefix of n=120 under "
                "the same seed. Matching the scope is what makes a comparison to that table "
                "meaningful rather than a fresh unrelated measurement."
            ),
        },
        "games": out,
        "duration_s": round(time.time() - t0, 3),
    }
    path = os.path.join(OUT_DIR, "cross_corpus_check.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    print(json.dumps({"written": path, "n_games": len(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
