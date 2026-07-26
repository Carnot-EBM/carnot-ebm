"""DOSE WITNESS for the convention-perturbation battery.

Measures, per game and per condition, how much of each lever's assumed convention the
perturbation actually destroys -- BEFORE any expensive A/B is run.  A condition with zero
dose on a lever cannot possibly move that lever's behaviour, so a flat result under it would
be uninterpretable rather than reassuring.

Two doses, one per lever, each measured on the lever's OWN causal channel:

  FRONTIER  the tier-ordered click-point sequence produced by
            arc_graph_explore._tier_ordered_click_points on the reset frame.  This is
            literally the list the tier barrier consumes, so a change here is a change the
            mechanism must see.  Reported as (a) the fraction of connected components whose
            tier changes, and (b) whether the ordered point list differs at all.

  HUD       the Stage-1 mask arc_hud_bar_detector.edge_bar_hud_mask returns on the reset
            frame.  Reported as the masked-cell count before and after and whether the
            masked cell SET changes.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cptb_perturb as P  # noqa: E402


def _tiers(grid):
    """Per-component tier, using the module's OWN constants (imported, never re-typed)."""
    from carnot.agentic.arc_graph_explore import (
        _TIER_MAX_WIDTH,
        _TIER_MIN_WIDTH,
        _TIER_SALIENT_COLORS,
        _TIER_STATUS_BAR_COLOR,
    )
    from carnot.agentic.arc_solver_kit import object_centric_digest

    out = []
    for comp in object_centric_digest(grid)["components"]:
        bb = comp["bbox"]
        h = bb[2] - bb[0] + 1
        w = bb[3] - bb[1] + 1
        color = int(comp["color"])
        salient = color in _TIER_SALIENT_COLORS
        medium = _TIER_MIN_WIDTH <= w <= _TIER_MAX_WIDTH and _TIER_MIN_WIDTH <= h <= _TIER_MAX_WIDTH
        if color == _TIER_STATUS_BAR_COLOR:
            t = 4
        elif salient and medium:
            t = 0
        elif medium:
            t = 1
        elif salient:
            t = 2
        else:
            t = 3
        out.append(((int(comp["centroid"][0]), int(comp["centroid"][1])), t))
    return out


def _mask_cells(grid):
    from carnot.agentic.arc_hud_bar_detector import edge_bar_hud_mask

    m = edge_bar_hud_mask(grid)
    if m is None:
        return None
    arr = np.asarray(m)
    return frozenset(map(tuple, np.argwhere(arr)))


def main() -> int:
    from carnot.agentic import arc_solver_kit as kit
    from carnot.experiment_5836_frontier_discipline_ab import ALL_GAMES

    arc = kit.offline_arcade()
    sc = arc.open_scorecard()

    palettes = {}
    frames = {}
    for g in ALL_GAMES:
        env = arc.make(g, scorecard_id=sc)
        f = env.reset()
        st = np.array(f.frame)
        if st.ndim == 2:
            st = st[None, ...]
        frames[g] = st[-1]
        palettes[g] = sorted({int(c) for c in np.unique(st)})

    P.install(palettes)
    from carnot.agentic import arc_variant_generator as vg

    rows = []
    for g in ALL_GAMES:
        base = frames[g]
        # C1: salience inversion (no reflect)
        c1 = vg.transform_frame_grid(base, g, P.VARIANT_SALIENCE_INVERSION, reflect=None)
        # C2: row roll, colour identity
        c2 = vg.transform_frame_grid(base, g, P.VARIANT_IDENTITY_COLOR, reflect=P.REFLECT_DIAG_ROLL)
        # C3: the DOSE AXIS of the same roll (added 2026-07-25).  The k=3 roll was measured to
        # raze the corpus, which auto-falsifies any narrow-support lever under it regardless of
        # mechanism.  Measuring the HUD mask at smaller k separates "the magnitude at which the
        # Stage-1 predicate stops firing" from "the magnitude at which the games stop being
        # winnable", which the single-magnitude design conflated.
        c3 = {
            f"C3_roll_k{k}": vg.transform_frame_grid(
                base, g, P.VARIANT_IDENTITY_COLOR, reflect=P.reflect_code_for_roll_k(k))
            for k in (1, 2)
        }

        t0 = _tiers(base)
        m0 = _mask_cells(base)
        row = {
            "game": g,
            "palette": palettes[g],
            "salience_map_pairs": [
                [int(a), int(P._MAPS[g][a])]
                for a in palettes[g]
                if int(P._MAPS[g][a]) != int(a)
            ],
            "n_components": len(t0),
            "baseline_hud_mask_cells": (len(m0) if m0 is not None else None),
            "bottom_k_rows_nonbg_cells_that_wrap": int((base[-P.ROLL_K :] != 0).sum()),
        }
        for label, gridx in [("C1_salience_inversion", c1), ("C2_diag_roll", c2), *c3.items()]:
            tx = _tiers(gridx)
            mx = _mask_cells(gridx)
            # tier change fraction: match components by centroid where possible (C1 keeps
            # positions exactly; C2 moves them, so compare the MULTISET of tiers + the
            # ordered click sequence instead of pretending a positional match exists).
            base_by_pt = dict(t0)
            if label == "C1_salience_inversion":
                changed = sum(
                    1 for pt, t in tx if base_by_pt.get(pt) is not None and base_by_pt[pt] != t
                )
                frac = changed / max(1, len(tx))
            else:
                frac = None
            order0 = [t for _, t in sorted(t0, key=lambda z: (z[1],))]
            orderx = [t for _, t in sorted(tx, key=lambda z: (z[1],))]
            row[label] = {
                "frontier_tier_change_fraction": frac,
                "frontier_tier_histogram_before": {
                    str(k): order0.count(k) for k in range(5) if order0.count(k)
                },
                "frontier_tier_histogram_after": {
                    str(k): orderx.count(k) for k in range(5) if orderx.count(k)
                },
                "frontier_tier_multiset_changed": sorted(order0) != sorted(orderx),
                "hud_mask_cells_after": (len(mx) if mx is not None else None),
                "hud_mask_changed": (m0 != mx),
            }
        rows.append(row)
        print(json.dumps(row))

    out = Path(os.environ.get("CPTB_WORKDIR") or Path(__file__).resolve().parent) / "cptb_dose.json"
    out.write_text(json.dumps({"rows": rows}, indent=1))
    print("WROTE", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
