#!/usr/bin/env python3
"""POSITIVE evidence that the six shipped HUD bands are progress bars, not playfield.

WHY THIS EXISTS
---------------
The original capture defended each band NEGATIVELY: three alarms did not fire (A1 the
winning route never clicks inside, A2 the mover never occupies it, A3 the shipped swallow
guard says the mask does not eat the corpus's changed cells) and at least one positive
discriminator passed (P1 the band changes when the action is far away, P2 several distinct
action ids each change it). Every one of those is a statement about what the band does NOT
do, or about how it CORRELATES with actions.

An adversarial review of that capture made the point that this is the weakest available
defence of the one claim that matters -- "the mask does not delete real playfield" -- and
that a much stronger, entirely independent argument is available from the band's SHAPE.
A progress bar has a morphology no board region has:

  M1  exactly two distinct colours ever appear in the band (background + fill)
  M2  the minority colour forms a SINGLE CONTIGUOUS RUN in every frame (a bar, not scatter)
  M3  the row/col immediately INSIDE the band is constant across the whole trajectory
      (a divider separating the readout from the board). Reported in two strengths,
      because one band (ka59) is constant for every within-level frame and redraws only
      at the level boundary, where the bar also empties -- which is MORE bar-like, not
      less, and is invisible if the test is only ever reported as a single bool.
  M4  the mover entity never occupies the band
  M5  the winning route never clicks inside the band
  M6  the band resets toward uniform at a level boundary (the bar empties on level-up)

M1-M3 are pure shape and say nothing about actions, so they cannot be produced by the same
mechanism that produces the detector's own action-correlation signals. If a band satisfies
all of them, "this is a status readout" stops being an inference from absence.

WHAT THIS SCRIPT IS NOT
-----------------------
It cannot ADD a band to any mask, and it does not re-run the audit or change a decision.
It only measures shape, on the same offline corpus the capture used, so a reader can weigh
the bands on evidence the original artifact did not carry. A band that FAILS a morphology
test is reported as failing; nothing here is tuned to make the shipped set look good.

CPU-only. No LLM, no GPU, no network, no scored/online game. Reads the offline arcade over
environment_files exactly as capture.py does.
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

# The six games that carry a default mask, and the band each one ships.
# Read from masks/_index.json at runtime rather than hardcoded, so this cannot drift
# from what actually shipped.


def band_cells(grid: np.ndarray, axis: str, index: int) -> np.ndarray:
    return grid[index, :] if axis == "row" else grid[:, index]


def inner_neighbour_index(axis: str, index: int, shape: tuple[int, int]) -> int | None:
    """The row/col one step TOWARD the board interior from the band."""
    n = shape[0] if axis == "row" else shape[1]
    if index <= 1:
        return index + 1
    if index >= n - 2:
        return index - 1
    return None


def longest_run_is_whole(mask: np.ndarray) -> bool:
    """True when every True cell in `mask` lies in one contiguous run."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return True
    return bool(idx[-1] - idx[0] + 1 == idx.size)


def frames_of(window: list[Any], traj: list[Any]) -> list[np.ndarray]:
    """Every distinct grid the trajectory visits, in order (before-frames + final after)."""
    seq = traj if traj else window
    out = [np.asarray(t.grid) for t in seq]
    if seq:
        out.append(np.asarray(seq[-1].next_grid))
    return out


def check_game(game: str, axis: str, index: int) -> dict:
    from carnot.agentic.arc_actions_to_progress import build_progress_window
    from carnot.agentic.arc_entity_hud_perception import (
        Transition as PTransition,
    )
    from carnot.agentic.arc_entity_hud_perception import detect_mover

    t0 = time.time()
    out = build_progress_window(game)
    if out is None:
        return {"game": game, "status": "no_window"}
    window, traj, cell = out

    def to_pt(t: Any) -> PTransition:
        d = t.data if isinstance(t.data, dict) else {}
        return PTransition(
            before=t.grid, action=int(t.action), after=t.next_grid, x=d.get("x"), y=d.get("y")
        )

    traj_pts = [to_pt(t) for t in traj]
    win_pts = [to_pt(t) for t in window]
    mover = detect_mover(traj_pts) or detect_mover(win_pts)
    mover_color = int(mover.color) if mover is not None else None

    frames = frames_of(window, traj)
    shape = tuple(int(v) for v in frames[0].shape)

    # ---- M1: how many distinct colours ever appear in the band -----------------------
    seen: set[int] = set()
    for f in frames:
        if tuple(f.shape) != shape:
            continue
        seen.update(int(v) for v in np.unique(band_cells(f, axis, index)))
    colours = sorted(seen)

    # ---- M2: is the minority colour one contiguous run, in every frame? --------------
    contiguous = 0
    checked = 0
    minority_colour = None
    if len(colours) == 2:
        # majority = the colour occupying more cells summed over the trajectory
        totals = {c: 0 for c in colours}
        for f in frames:
            if tuple(f.shape) != shape:
                continue
            b = band_cells(f, axis, index)
            for c in colours:
                totals[c] += int((b == c).sum())
        minority_colour = min(totals, key=lambda c: totals[c])
        for f in frames:
            if tuple(f.shape) != shape:
                continue
            checked += 1
            if longest_run_is_whole(band_cells(f, axis, index) == minority_colour):
                contiguous += 1

    # ---- M3: is the inner neighbour constant across the trajectory? -------------------
    nb = inner_neighbour_index(axis, index, shape)
    nb_constant = None
    nb_values = None
    nb_change_at: list[int] = []
    if nb is not None:
        vals = set()
        seq_nb: list[tuple[int, ...]] = []
        for f in frames:
            if tuple(f.shape) != shape:
                continue
            v = tuple(int(x) for x in band_cells(f, axis, nb))
            vals.add(v)
            seq_nb.append(v)
        nb_constant = len(vals) == 1
        nb_values = len(vals)
        nb_change_at = [
            i for i, (a, b) in enumerate(zip(seq_nb, seq_nb[1:], strict=False)) if a != b
        ]

    # ---- M4: does the mover ever enter the band? --------------------------------------
    mover_in_band = 0
    if mover_color is not None:
        for f in frames:
            if tuple(f.shape) != shape:
                continue
            if bool((band_cells(f, axis, index) == mover_color).any()):
                mover_in_band += 1

    # ---- M5: does the winning route ever click inside the band? -----------------------
    clicks_total = 0
    clicks_in_band = 0
    for t in traj_pts:
        if t.x is None or t.y is None:
            continue
        clicks_total += 1
        # arc click coords are (x=col, y=row) in the detector's own convention
        pos = int(t.y) if axis == "row" else int(t.x)
        if pos == index:
            clicks_in_band += 1

    # ---- M6: does the band reset toward uniform at a level boundary? -------------------
    # A level boundary is a transition whose grid shape changes, or where the band's
    # minority-fill count DROPS by more than half. Report the fill series so a reader can
    # judge rather than trusting a threshold.
    fills = []
    if minority_colour is not None:
        for f in frames:
            if tuple(f.shape) != shape:
                continue
            fills.append(int((band_cells(f, axis, index) == minority_colour).sum()))
    reset_at = [i for i, (a, b) in enumerate(zip(fills, fills[1:], strict=False)) if b * 2 < a]
    resets = len(reset_at)

    # M3b: every change in the divider coincides with the bar emptying, i.e. the whole
    # HUD strip redraws at the level boundary and is otherwise frozen. This is a WEAKER
    # test than M3 and is reported separately, never substituted for it.
    nb_constant_within_level = None
    if nb is not None:
        nb_constant_within_level = all(i in reset_at for i in nb_change_at)

    passed = {
        "M1_two_colours": len(colours) == 2,
        "M2_minority_contiguous_every_frame": (checked > 0 and contiguous == checked),
        "M3_inner_neighbour_constant": bool(nb_constant) if nb_constant is not None else None,
        "M4_mover_never_enters": mover_in_band == 0,
        "M5_route_never_clicks_inside": clicks_in_band == 0,
    }
    return {
        "game": game,
        "status": "ok",
        "axis": axis,
        "index": index,
        "shape": list(shape),
        "n_frames": len(frames),
        "M1_distinct_colours_in_band": colours,
        "M2_minority_colour": minority_colour,
        "M2_frames_with_single_contiguous_run": contiguous,
        "M2_frames_checked": checked,
        "M3_inner_neighbour_index": nb,
        "M3_inner_neighbour_distinct_values_over_trajectory": nb_values,
        "M3_inner_neighbour_changes_at_transitions": nb_change_at,
        "M3b_inner_neighbour_changes_only_when_the_bar_resets": nb_constant_within_level,
        "M4_frames_with_mover_inside_band": mover_in_band,
        "M4_mover_colour": mover_color,
        "M5_route_clicks_total": clicks_total,
        "M5_route_clicks_inside_band": clicks_in_band,
        "M6_band_fill_series": fills,
        "M6_n_halving_drops": resets,
        "M6_reset_at_transitions": reset_at,
        "M6_band_changing_positions": sorted(
            int(i)
            for i in np.flatnonzero(
                np.logical_or.reduce(
                    [
                        band_cells(a, axis, index) != band_cells(b, axis, index)
                        for a, b in zip(frames, frames[1:], strict=False)
                        if tuple(a.shape) == shape and tuple(b.shape) == shape
                    ]
                )
            )
        ),
        "morphology": passed,
        "morphology_all_pass": all(v for v in passed.values() if v is not None),
        "duration_s": round(time.time() - t0, 3),
    }


def ls20_two_row_widget() -> dict:
    """Is ls20's masked row 62 half of a two-row widget whose other half is row 61?

    The adversarial review claims rows 61 and 62 are byte-identical across the whole
    trajectory, which would mean the shipped mask covers half a widget and leaves the
    other half graded -- not because anyone judged row 61 to be playfield, but because
    `arc_entity_hud_perception.detect_hud_registers(edge_margin=2)` structurally cannot
    PROPOSE row 61 on a 64-row grid. That is a completeness defect plus a detector
    limitation, and it is only worth recording if the byte-identity is real.
    """
    from carnot.agentic.arc_actions_to_progress import build_progress_window

    out = build_progress_window("ls20")
    if out is None:
        return {"status": "no_window"}
    window, traj, _cell = out
    frames = frames_of(window, traj)
    shape = tuple(int(v) for v in frames[0].shape)

    identical = 0
    checked = 0
    for f in frames:
        if tuple(f.shape) != shape:
            continue
        checked += 1
        if np.array_equal(f[61, :], f[62, :]):
            identical += 1

    # change masks per transition
    same_change_mask = 0
    trans = 0
    n61_changed_cells = 0
    n62_changed_cells = 0
    n61_touching = 0
    n62_touching = 0
    seq = traj if traj else window
    for t in seq:
        a = np.asarray(t.grid)
        b = np.asarray(t.next_grid)
        if a.shape != b.shape or tuple(a.shape) != shape:
            continue
        trans += 1
        c61 = a[61, :] != b[61, :]
        c62 = a[62, :] != b[62, :]
        if np.array_equal(c61, c62):
            same_change_mask += 1
        n61_changed_cells += int(c61.sum())
        n62_changed_cells += int(c62.sum())
        n61_touching += int(bool(c61.any()))
        n62_touching += int(bool(c62.any()))

    # what would a two-row mask do to the swallow guard?
    total_changed = 0
    inside = 0
    for t in seq:
        a = np.asarray(t.grid)
        b = np.asarray(t.next_grid)
        if a.shape != b.shape or tuple(a.shape) != shape:
            continue
        ch = a != b
        total_changed += int(ch.sum())
        inside += int(ch[61, :].sum() + ch[62, :].sum())
    return {
        "status": "ok",
        "n_frames_checked": checked,
        "n_frames_row61_byte_identical_to_row62": identical,
        "n_transitions": trans,
        "n_transitions_with_identical_change_masks": same_change_mask,
        "row61_changed_cells_total": n61_changed_cells,
        "row62_changed_cells_total": n62_changed_cells,
        "row61_transitions_touching": n61_touching,
        "row62_transitions_touching": n62_touching,
        "two_row_mask_changed_cells_inside": inside,
        "corpus_changed_cells_total": total_changed,
        "two_row_mask_changed_cell_overlap": round(inside / total_changed, 6)
        if total_changed
        else None,
        "swallow_overlap_threshold": 0.5,
        "two_row_mask_would_still_pass_the_swallow_guard": (
            total_changed > 0 and inside / total_changed < 0.5
        ),
    }


def main() -> int:
    index_path = os.path.join(OUT_DIR, "masks", "_index.json")
    with open(index_path) as fh:
        idx = json.load(fh)
    shipped = []
    for game, rec in sorted(idx.get("games", idx).items()):
        if not isinstance(rec, dict):
            continue
        rows = rec.get("hud_rows") or []
        cols = rec.get("hud_cols") or []
        for r in rows:
            shipped.append((game, "row", int(r)))
        for c in cols:
            shipped.append((game, "col", int(c)))

    results = []
    for game, axis, index in shipped:
        rec = check_game(game, axis, index)
        results.append(rec)
        print(
            f"{game} {axis}{index}: colours={rec.get('M1_distinct_colours_in_band')} "
            f"contig={rec.get('M2_frames_with_single_contiguous_run')}/"
            f"{rec.get('M2_frames_checked')} "
            f"nb_vals={rec.get('M3_inner_neighbour_distinct_values_over_trajectory')} "
            f"mover_in={rec.get('M4_frames_with_mover_inside_band')} "
            f"clicks_in={rec.get('M5_route_clicks_inside_band')}/"
            f"{rec.get('M5_route_clicks_total')} "
            f"ALL={rec.get('morphology_all_pass')}",
            flush=True,
        )

    ls20 = ls20_two_row_widget()
    print("ls20 two-row widget:", json.dumps(ls20), flush=True)

    payload = {
        "shipped_bands": shipped,
        "per_band": results,
        "n_bands_all_morphology_pass": sum(1 for r in results if r.get("morphology_all_pass")),
        "ls20_two_row_widget": ls20,
    }
    path = os.path.join(OUT_DIR, "morphology_check.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
