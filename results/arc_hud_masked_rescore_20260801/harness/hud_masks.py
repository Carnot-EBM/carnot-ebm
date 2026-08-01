#!/usr/bin/env python3
"""Materialise the HUD masks the capture artifact published, in LOGICAL-grid coordinates.

ONE PLACE, because a mask built two different ways in two workers is a difference that would
show up as a "result". Both the A/B worker and the best-of-N worker import this.

THE CAPTURE ARTIFACT PUBLISHES TWO DIFFERENT THINGS AND THEY MUST NOT BE CONFLATED.

  `hud_rows` / `hud_cols`      the DEFAULT mask. Every band in it cleared BOTH corpora of the
                               capture's over-masking audit. 6 of 20 games have one; the other
                               14 ship an EMPTY mask, which that artifact is explicit is a
                               REFUSAL with a recorded reason, not an absence.

  `conditionally_clean_bands`  bands the audit detected and then WITHHELD at its stage-2
                               second-corpus gate. Its own `warning` field says using one is
                               "an explicit opt-in to a mask whose safety is corpus-conditional,
                               and any result obtained with it must report the refusal
                               alongside". Three games have one: tn36 row 1, lf52 row 0,
                               su15 row 63.

tn36 row 1 is the ONLY band that can address the question this re-score exists for -- the six
perfect-`change_fidelity` bar-tickers are ticking exactly that row -- and it is precisely the
band the capture withheld. So the DEFAULT arm structurally cannot answer Q1, and an honest
answer needs the conditional arm, carrying the withholding with it. Both are computed here and
reported side by side; neither is presented as the other.

COORDINATE SPACE. The capture's `coordinate_space` is `logical_grid` with `cell: 1` on all 20
games, i.e. the indices are already in the same space as `Transition.grid`. `logical_hud_mask`
is therefore NOT called: at cell==1 it is the identity, and calling it would only add a
downsample that cannot fire. This is the whole reason this path is reconstructible where the
A/B's was not -- the A/B needed a FRAME-coordinate mask it never recorded.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np

CAPTURE = pathlib.Path("results/arc_hud_mask_capture_20260801")


def _repo() -> pathlib.Path:
    import os

    return pathlib.Path(os.environ["CARNOT_REPO"])


def load_mask_record(game: str) -> dict | None:
    p = _repo() / CAPTURE / "masks" / f"{game}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _materialise(shape: Any, rows: list[int], cols: list[int]) -> np.ndarray | None:
    """Build the boolean mask, or None when there is nothing to mask.

    None rather than an all-False array is deliberate and load-bearing. An all-False mask would
    reach `WorldModelVerifier` as a real mask, pass the swallow check trivially, and be recorded
    as `hud_mask_status == "applied"` with 0 cells -- a masked-looking record that masked
    nothing. That is the exact "second column of identical numbers wearing a different name"
    failure this whole re-score is meant to avoid, and it would be indistinguishable in the
    artifact from a mask that really was applied. None routes to `unresolved` instead, which
    says out loud that no mask was available.
    """
    if not rows and not cols:
        return None
    m = np.zeros(tuple(int(x) for x in shape), dtype=bool)
    for r in rows:
        m[int(r), :] = True
    for c in cols:
        m[:, int(c)] = True
    return m if bool(m.any()) else None


def masks_for(game: str) -> dict:
    """Return {'default': mask|None, 'conditional': mask|None, 'meta': {...}}.

    `conditional` is default PLUS the withheld bands -- a superset, never a replacement, so the
    two arms differ by exactly the withheld bands and nothing else.
    """
    rec = load_mask_record(game)
    if rec is None:
        return {
            "default": None,
            "conditional": None,
            "meta": {"game": game, "no_capture_record": True},
        }
    shape = rec.get("logical_shape") or [64, 64]
    d_rows = [int(r) for r in (rec.get("hud_rows") or [])]
    d_cols = [int(c) for c in (rec.get("hud_cols") or [])]
    c_rows, c_cols = list(d_rows), list(d_cols)
    cond = []
    for b in rec.get("conditionally_clean_bands") or []:
        cond.append(
            {
                "axis": b.get("axis"),
                "index": int(b.get("index")),
                "stage": b.get("stage"),
                "refusal_class": b.get("refusal_class"),
                "random_action_corpus_reason": (b.get("random_action_corpus") or {}).get("reason"),
                "winning_route_corpus_reason": (b.get("winning_route_corpus") or {}).get("reason"),
                "warning": b.get("warning"),
            }
        )
        if b.get("axis") == "row":
            c_rows.append(int(b.get("index")))
        elif b.get("axis") == "col":
            c_cols.append(int(b.get("index")))
    default = _materialise(shape, d_rows, d_cols)
    conditional = _materialise(shape, c_rows, c_cols)
    return {
        "default": default,
        "conditional": conditional,
        "meta": {
            "game": game,
            "logical_shape": [int(x) for x in shape],
            "cell": int(rec.get("cell") or 1),
            "capture_mask_status": rec.get("mask_status"),
            "default_rows": d_rows,
            "default_cols": d_cols,
            "default_cells": int(default.sum()) if default is not None else 0,
            "conditional_rows": c_rows,
            "conditional_cols": c_cols,
            "conditional_cells": int(conditional.sum()) if conditional is not None else 0,
            "withheld_bands_folded_into_conditional_arm": cond,
            "n_excluded_bands_never_used_in_any_arm": len(rec.get("excluded_bands") or []),
        },
    }
