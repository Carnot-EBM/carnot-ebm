#!/usr/bin/env python3
"""Capture per-game LOGICAL HUD masks for the 20-game object-perception A/B roster, and
EARN the right to use them with an explicit over-masking audit.

WHY THIS EXISTS
---------------
The object-perception A/B (results/arc_object_perception_ab_change_fidelity_20260801)
moved change_fidelity (p=0.0192) but recorded its own limitation:

    hud_mask_limitation.status_on_every_cell = "disabled"
    "both this A/B and the headroom harness construct the verifier bare, so
     REQ-ARC-WMTE-6010's compare-time HUD masking is OFF and any HUD/status strip is
     INSIDE the graded comparison."

That is not cosmetic. The known degeneracy is that all six candidates scoring a perfect
change_fidelity 1.0 are UNPLANNABLE: they are tn36's progress-BAR tickers. They model the
status indicator exactly and the playfield not at all. The bar is inside the metric, so
modelling only the bar maxes the metric.

The A/B said the masked arm was "not reconstructible from this run's evidence" because
`arc_executable_world_model.logical_hud_mask(frame_mask, cell)` needs a FRAME-coordinate
mask that the graded cells never recorded. That is true of THAT path and not of this one:
`arc_entity_hud_perception.detect_hud_registers()` consumes TRANSITIONS and returns
HudBand(axis, index, ...) in the SAME coordinate space as the transition grids. No frame
mask is required, so the mask is reconstructible from the window alone.

WHY THIS DETECTOR RATHER THAN A CONVENIENT ONE
----------------------------------------------
`detect_hud_registers`'s own docstring records that a count-monotonicity test MISSES real
ARC counters -- "measured on tu93/bp35: row63 changed on 100% of actions yet net count
change was 0" -- so it keys on `changed_fraction` and position-independence instead. And
`edge_margin=2` confines candidates to the outer two rows/cols, so it structurally cannot
mask interior playfield.

THE FAILURE MODE THAT MATTERS MORE THAN THE RESULT
--------------------------------------------------
OVER-MASKING IS INVISIBLE AND CATASTROPHIC. If a flagged band is really playfield, masking
deletes real dynamics: every engine's score IMPROVES while it models LESS, and nothing in
the numbers says so. `logical_hud_mask`'s own docstring names the asymmetry -- "over-masking
destroys CORRECTNESS while under-masking only costs efficiency". So the over-masking audit
is the FIRST deliverable, not a closing caveat, and it GATES which bands reach the mask.

NO THRESHOLD TUNING. `detect_hud_registers` is called with its SHIPPED DEFAULTS on every
game. If the defaults flag nothing on a game, that is the measurement, not a reason to
adjust a knob. The audit can only ever REMOVE a band from the mask, never add one.

verifier_is_oracle: N/A -- this is perception capture, not a verifier, and it emits no
score. Every signal comes from the agent's own (grid, action, grid) transitions plus the
offline arcade's own winning route; nothing here reads a game's source or win predicate.

Spec: REQ-ARC-WMTE-6010 (the compare-time HUD masking this makes reconstructible),
REQ-ARC-WMTE-6015 (the swallow guard reused here as audit alarm A3),
REQ-ARC-WMTE-5833 (the detector).
"""

from __future__ import annotations

import hashlib
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
MASK_DIR = os.path.join(OUT_DIR, "masks")

# ---------------------------------------------------------------------------------------
# PRE-REGISTERED AUDIT CONSTANTS. Fixed before any per-game number was read; they are
# reported next to every measurement so a reviewer can re-judge a band under their own bar.
# ---------------------------------------------------------------------------------------

# "Far" = the acted cell is at least this many cells from the band, measured along the
# band's own axis. `edge_margin=2` already confines a band to the outer two rows/cols, so
# a distance of 4 puts the acted cell outside the edge band PLUS a two-cell buffer -- it
# cannot be explained as "the action landed on/next to the band".
FAR_MIN = 4

# A band must change on at least this fraction of the FAR-acted opportunities to count as
# position-decoupled. Deliberately reused from the detector's own `pos_independent_min`
# rather than chosen fresh, so this introduces no new tunable into the project.
FAR_RATE_MIN = 0.7

# Minimum far-acted opportunities before P1 may be claimed at all. Below this the rate is
# an artefact of two or three draws, so the discriminator is recorded UNAVAILABLE rather
# than passed -- refuse-on-doubt, the same direction as the swallow guard.
FAR_MIN_OPPORTUNITIES = 3

# P2: a band is action-decoupled if at least two DISTINCT action ids each change it on at
# least half their frame-changing transitions. 0.5 mirrors the detector's own
# `mixed_changed_min`, again avoiding a fresh number.
ACT_TYPE_RATE_MIN = 0.5
ACT_TYPE_MIN_OBS = 2
ACT_TYPE_MIN_TYPES = 2

ROSTER_SOURCE = os.path.join(
    REPO, "results", "arc_object_perception_ab_change_fidelity_20260801", "meta.json"
)

SEED = 20260801


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _canon_sha256(obj: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()
    )


def _band_index_range(axis: str, index: int, shape: tuple[int, int]) -> bool:
    h, w = shape
    return index < (h if axis == "row" else w)


def _band_cells_of(grid: np.ndarray, axis: str, index: int) -> np.ndarray:
    return grid[index, :] if axis == "row" else grid[:, index]


def _band_mask(shape: tuple[int, int], bands: list[dict]) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    for b in bands:
        if not _band_index_range(b["axis"], b["index"], shape):
            continue
        if b["axis"] == "row":
            m[b["index"], :] = True
        else:
            m[:, b["index"]] = True
    return m


def _acted_location(t: Any, mover_color: int | None) -> tuple[tuple[float, float] | None, str]:
    """Where did this action act, in LOGICAL (row, col)?

    A click carries its own target. A directional action does not, so the best available
    proxy for "where the agent acted" is the PLAYER's position before the action -- which
    is exactly what `detect_mover` gives us, and is the prompt's third suggested
    discriminator turned into a location. When neither exists the transition contributes
    NO far-action evidence rather than a guessed one.
    """
    data = t.data if isinstance(t.data, dict) else {}
    if int(t.action) == 6 and data.get("x") is not None and data.get("y") is not None:
        return (float(data["y"]), float(data["x"])), "click"
    if mover_color is not None:
        g = np.asarray(t.grid)
        ys, xs = np.nonzero(g == int(mover_color))
        if ys.size:
            return (float(ys.mean()), float(xs.mean())), "mover_centroid"
    return None, "unavailable"


def _axis_distance(axis: str, index: int, loc: tuple[float, float]) -> float:
    return abs(loc[0] - index) if axis == "row" else abs(loc[1] - index)


# ---------------------------------------------------------------------------------------
# the over-masking audit -- computed per (band, corpus)
# ---------------------------------------------------------------------------------------


def audit_band_on_corpus(
    band: dict, transitions: list, mover_color: int | None, shape: tuple[int, int]
) -> dict:
    """Evidence that this band is a COUNTER and not playfield, measured on one corpus.

    Returns every quantity, availability flag, and verdict. Nothing here decides on its
    own; `decide_band` combines the two corpora.
    """
    from carnot.agentic.arc_executable_world_model import (
        hud_mask_swallow_check,
        hud_mask_swallow_clean,
    )

    axis, index = band["axis"], band["index"]
    changing = [
        t
        for t in transitions
        if np.asarray(t.grid).shape == np.asarray(t.next_grid).shape
        and bool(np.any(np.asarray(t.grid) != np.asarray(t.next_grid)))
    ]

    n_changing = len(changing)
    band_changed_flags: list[bool] = []
    far_opps = far_changed = 0
    far_distances: list[float] = []
    loc_kinds: dict[str, int] = {}
    per_action: dict[int, list[int]] = {}
    acts_in_band = 0
    mover_in_band_frames = 0

    for t in changing:
        g0, g1 = np.asarray(t.grid), np.asarray(t.next_grid)
        if not _band_index_range(axis, index, g0.shape):
            band_changed_flags.append(False)
            continue
        changed = not np.array_equal(
            _band_cells_of(g0, axis, index), _band_cells_of(g1, axis, index)
        )
        band_changed_flags.append(changed)
        per_action.setdefault(int(t.action), []).append(int(changed))

        loc, kind = _acted_location(t, mover_color)
        loc_kinds[kind] = loc_kinds.get(kind, 0) + 1
        if loc is not None:
            d = _axis_distance(axis, index, loc)
            if d >= FAR_MIN:
                far_opps += 1
                far_distances.append(d)
                if changed:
                    far_changed += 1

        # A1 -- did the winning route CLICK inside this band? (a band you must act in is
        # interactive, i.e. playfield)
        data = t.data if isinstance(t.data, dict) else {}
        if int(t.action) == 6:
            if axis == "row" and data.get("y") is not None and int(data["y"]) == index:
                acts_in_band += 1
            if axis == "col" and data.get("x") is not None and int(data["x"]) == index:
                acts_in_band += 1

        # A2 -- does the PLAYER ever occupy the band?
        if mover_color is not None:
            for g in (g0, g1):
                if _band_index_range(axis, index, g.shape) and bool(
                    np.any(_band_cells_of(g, axis, index) == int(mover_color))
                ):
                    mover_in_band_frames += 1
                    break

    n_band_changed = int(sum(band_changed_flags))
    changed_fraction = round(n_band_changed / n_changing, 6) if n_changing else 0.0

    far_rate = round(far_changed / far_opps, 6) if far_opps else None
    p1_available = far_opps >= FAR_MIN_OPPORTUNITIES
    p1_pass = bool(p1_available and far_rate is not None and far_rate >= FAR_RATE_MIN)

    act_rates = {
        str(a): {
            "n": len(v),
            "rate": round(sum(v) / len(v), 6),
        }
        for a, v in sorted(per_action.items())
    }
    qualifying = [
        a
        for a, r in act_rates.items()
        if r["n"] >= ACT_TYPE_MIN_OBS and r["rate"] >= ACT_TYPE_RATE_MIN
    ]
    p2_available = (
        sum(1 for r in act_rates.values() if r["n"] >= ACT_TYPE_MIN_OBS) >= ACT_TYPE_MIN_TYPES
    )
    p2_pass = bool(len(qualifying) >= ACT_TYPE_MIN_TYPES)

    # A3 -- the project's OWN shipped over-masking guard, on this band alone.
    swallow = hud_mask_swallow_check(transitions, _band_mask(shape, [band]))
    a3_clean = bool(hud_mask_swallow_clean(swallow))

    return {
        "n_transitions": len(transitions),
        "n_changing_transitions": n_changing,
        "band_changed_on": n_band_changed,
        "changed_fraction_on_this_corpus": changed_fraction,
        "acted_location_kinds": loc_kinds,
        "P1_far_action_decoupling": {
            "far_min_cells": FAR_MIN,
            "far_opportunities": far_opps,
            "far_changed": far_changed,
            "far_action_change_rate": far_rate,
            "median_far_distance": (
                round(float(np.median(far_distances)), 3) if far_distances else None
            ),
            "available": p1_available,
            "pass": p1_pass,
            "principle": (
                "a counter advances no matter where you act; a board region changes only "
                "where you act on it, so a band that keeps changing when the action is "
                f">={FAR_MIN} cells away is decoupled from the action's LOCATION"
            ),
        },
        "P2_action_type_decoupling": {
            "per_action": act_rates,
            "qualifying_action_ids": qualifying,
            "available": p2_available,
            "pass": p2_pass,
            "principle": (
                "a counter ticks on the mere fact of acting, so it changes under several "
                "DIFFERENT action types; a board region responds to the specific action "
                "that touches it"
            ),
        },
        "A1_solution_acts_in_band": {
            "n_click_actions_targeting_band": acts_in_band,
            "alarm": acts_in_band > 0,
            "principle": (
                "if the offline arcade's own winning route has to ACT inside the band, the "
                "band is interactive playfield and masking it deletes a required mechanic"
            ),
        },
        "A2_mover_occupies_band": {
            "mover_color": mover_color,
            "n_frames_with_mover_in_band": mover_in_band_frames,
            "available": mover_color is not None,
            "alarm": mover_in_band_frames > 0,
            "principle": (
                "if the PLAYER entity can be inside the band, the band is somewhere the "
                "game is played, not a status readout"
            ),
        },
        "A3_swallow_guard": {
            "record": swallow,
            "clean": a3_clean,
            "alarm": not a3_clean,
            "principle": (
                "REQ-ARC-WMTE-6015's shipped guard: a mask that covers the majority of the "
                "corpus's changed cells, or that leaves no changing transition at all, is "
                "refused on doubt -- over-masking destroys correctness, under-masking only "
                "costs efficiency"
            ),
        },
    }


def decide_band(window_audit: dict, traj_audit: dict) -> dict:
    """Combine the two corpora into one verdict.

    ALARMS are OR-ed: a band that looks like playfield on EITHER corpus is excluded. That
    is the refuse-on-doubt direction the swallow guard already established, and it is the
    only direction that is safe given the stated asymmetry.

    POSITIVE evidence is required on the PRIMARY (full winning trajectory) corpus, which
    is the richer of the two; the window's numbers are recorded next to it so a reader can
    see whether the shorter corpus would have reached the same place. A band with NO
    positive discriminator available is NOT defensible -- `changed_fraction` alone cannot
    tell a counter row from the row the player happens to walk along, so "the detector
    flagged it" is not by itself a reason to delete cells.
    """
    alarms: list[str] = []
    for name, aud in (("window", window_audit), ("trajectory", traj_audit)):
        if aud["A1_solution_acts_in_band"]["alarm"]:
            alarms.append(f"A1_solution_acts_in_band@{name}")
        if aud["A2_mover_occupies_band"]["alarm"]:
            alarms.append(f"A2_mover_occupies_band@{name}")
        if aud["A3_swallow_guard"]["alarm"]:
            alarms.append(
                f"A3_swallow_guard@{name}:{aud['A3_swallow_guard']['record'].get('reason')}"
            )

    p1 = traj_audit["P1_far_action_decoupling"]
    p2 = traj_audit["P2_action_type_decoupling"]
    positives = [n for n, p in (("P1", p1), ("P2", p2)) if p["pass"]]
    any_available = bool(p1["available"] or p2["available"])

    if alarms:
        return {
            "accepted": False,
            "suspected_false_positive": True,
            "reason": "playfield_alarm",
            "alarms": alarms,
            "positives_passed": positives,
        }
    if not positives:
        return {
            "accepted": False,
            "suspected_false_positive": True,
            "reason": (
                "undefendable_no_positive_evidence"
                if any_available
                else "undefendable_no_discriminator_available"
            ),
            "alarms": [],
            "positives_passed": [],
        }
    return {
        "accepted": True,
        "suspected_false_positive": False,
        "reason": "defended:" + "+".join(positives),
        "alarms": [],
        "positives_passed": positives,
    }


# ---------------------------------------------------------------------------------------
# per-game capture
# ---------------------------------------------------------------------------------------


def capture_game(game: str) -> dict:
    from carnot.agentic.arc_actions_to_progress import build_progress_window
    from carnot.agentic.arc_entity_hud_perception import (
        Transition as PTransition,
    )
    from carnot.agentic.arc_entity_hud_perception import (
        detect_hud_registers,
        detect_mover,
    )
    from carnot.agentic.arc_executable_world_model import (
        hud_mask_swallow_check,
        hud_mask_swallow_clean,
    )

    t0 = time.time()
    out = build_progress_window(game)
    if out is None:
        return {
            "game": game,
            "status": "no_window",
            "note": "build_progress_window returned None -- the game did not reach L1 offline",
            "duration_s": round(time.time() - t0, 3),
        }
    window, traj, cell = out

    def to_pt(t: Any) -> PTransition:
        d = t.data if isinstance(t.data, dict) else {}
        return PTransition(
            before=t.grid,
            action=int(t.action),
            after=t.next_grid,
            x=d.get("x"),
            y=d.get("y"),
        )

    win_pts = [to_pt(t) for t in window]
    traj_pts = [to_pt(t) for t in traj]

    shape = tuple(int(v) for v in np.asarray(window[0].grid).shape)  # type: ignore[assignment]

    # DETECTION: shipped defaults, on the WINDOW -- the corpus the A/B actually graded and
    # the corpus a re-score has in hand. The trajectory read is a robustness column only;
    # it never adds a band to the mask.
    bands_window = detect_hud_registers(win_pts)
    bands_traj = detect_hud_registers(traj_pts)
    mover = detect_mover(traj_pts) or detect_mover(win_pts)
    mover_color = int(mover.color) if mover is not None else None

    def band_dict(b: Any) -> dict:
        return {
            "axis": b.axis,
            "index": int(b.index),
            "direction": b.direction,
            "changed_fraction": float(b.changed_fraction),
            "monotone_ratio": float(b.monotone_ratio),
        }

    win_keys = {(b.axis, int(b.index)) for b in bands_window}
    traj_keys = {(b.axis, int(b.index)) for b in bands_traj}

    band_records: list[dict] = []
    for b in bands_window:
        bd = band_dict(b)
        w_aud = audit_band_on_corpus(bd, window, mover_color, shape)
        t_aud = audit_band_on_corpus(bd, traj, mover_color, shape)
        decision = decide_band(w_aud, t_aud)
        band_records.append(
            {
                **bd,
                "detected_on_window": True,
                "detected_on_full_trajectory": (b.axis, int(b.index)) in traj_keys,
                "audit_window": w_aud,
                "audit_full_trajectory": t_aud,
                "decision": decision,
            }
        )

    accepted = [b for b in band_records if b["decision"]["accepted"]]
    excluded = [b for b in band_records if not b["decision"]["accepted"]]

    mask = _band_mask(shape, accepted)
    masked_cells = int(mask.sum())
    masked_cell_fraction = round(masked_cells / float(shape[0] * shape[1]), 6)

    # FINAL combined-mask swallow check: the per-band alarms are necessary but the mask that
    # is actually APPLIED is the union, and a union can swallow where no single band does.
    combined_window = hud_mask_swallow_check(window, mask if masked_cells else None)
    combined_traj = hud_mask_swallow_check(traj, mask if masked_cells else None)
    combined_clean = bool(
        masked_cells
        and hud_mask_swallow_clean(combined_window)
        and hud_mask_swallow_clean(combined_traj)
    )
    if masked_cells and not combined_clean:
        # Refuse the whole mask rather than ship a union that deletes the dynamics.
        accepted_final: list[dict] = []
        for b in accepted:
            b["decision"] = {
                **b["decision"],
                "accepted": False,
                "suspected_false_positive": True,
                "reason": "combined_mask_swallows_dynamics",
            }
            excluded.append(b)
        accepted = accepted_final
        mask = _band_mask(shape, [])
        masked_cells = 0
        masked_cell_fraction = 0.0
        mask_status = "refused_combined_mask_swallows"
    elif masked_cells:
        mask_status = "accepted"
    elif band_records:
        mask_status = "refused_all_bands_excluded"
    else:
        mask_status = "no_bands_detected"

    # Does the game's own dynamics reach the outer two rows/cols OUTSIDE the detected bands?
    # A "yes" is a standing risk for the whole edge-band approach on that game, independent
    # of whether any band was accepted.
    edge = np.zeros(shape, dtype=bool)
    edge[:2, :] = True
    edge[-2:, :] = True
    edge[:, :2] = True
    edge[:, -2:] = True
    all_band_mask = _band_mask(shape, band_records)
    edge_outside_bands = edge & ~all_band_mask
    touches = False
    for t in traj:
        g0, g1 = np.asarray(t.grid), np.asarray(t.next_grid)
        if g0.shape != g1.shape or g0.shape != shape:
            continue
        if bool(((g0 != g1) & edge_outside_bands).any()):
            touches = True
            break
    if not touches:
        for t in traj:
            d = t.data if isinstance(t.data, dict) else {}
            if int(t.action) == 6 and d.get("x") is not None and d.get("y") is not None:
                r, c = int(d["y"]), int(d["x"])
                if 0 <= r < shape[0] and 0 <= c < shape[1] and bool(edge_outside_bands[r, c]):
                    touches = True
                    break

    return {
        "game": game,
        "status": "ok",
        "cell": int(cell),
        "logical_shape": [int(shape[0]), int(shape[1])],
        "n_window_transitions": len(window),
        "n_trajectory_transitions": len(traj),
        "mover": (
            None
            if mover is None
            else {
                "color": int(mover.color),
                "alignment": float(mover.alignment),
                "evidence": int(mover.evidence),
            }
        ),
        "bands_detected_window": [band_dict(b) for b in bands_window],
        "bands_detected_full_trajectory": [band_dict(b) for b in bands_traj],
        "detection_agrees_window_vs_trajectory": sorted(map(list, win_keys))
        == sorted(map(list, traj_keys)),
        "bands": band_records,
        "n_bands_detected": len(band_records),
        "n_bands_accepted": len(accepted),
        "n_bands_suspected_false_positive": len(excluded),
        "mask_status": mask_status,
        "hud_rows": sorted(b["index"] for b in accepted if b["axis"] == "row"),
        "hud_cols": sorted(b["index"] for b in accepted if b["axis"] == "col"),
        "masked_cells": masked_cells,
        "masked_cell_fraction": masked_cell_fraction,
        "combined_mask_swallow_check_window": combined_window,
        "combined_mask_swallow_check_trajectory": combined_traj,
        "combined_mask_swallow_clean": combined_clean,
        "playfield_touches_border_outside_detected_bands": bool(touches),
        "duration_s": round(time.time() - t0, 3),
    }


def write_mask_file(rec: dict) -> str | None:
    """One file per game, in the form a re-score can consume without re-deriving anything.

    Bands are whole rows/columns, so (shape, hud_rows, hud_cols) is a LOSSLESS encoding of
    the boolean mask -- no RLE, no 4096-entry blob, and a reader can see at a glance what
    is being deleted. The excluded bands travel WITH the mask: a consumer that disagrees
    with an exclusion can see exactly what was withheld and why, rather than finding a
    silently smaller mask.
    """
    if rec.get("status") != "ok":
        return None
    path = os.path.join(MASK_DIR, f"{rec['game']}.json")
    payload = {
        "game": rec["game"],
        "coordinate_space": "logical_grid",
        "coordinate_space_note": (
            "Indices are LOGICAL-grid rows/cols -- the same space as Transition.grid and "
            "Transition.next_grid. No frame-coordinate mask and no logical_hud_mask() "
            "downsample is required; that is the whole reason this path is reconstructible "
            "where the A/B's was not."
        ),
        "cell": rec["cell"],
        "logical_shape": rec["logical_shape"],
        "hud_rows": rec["hud_rows"],
        "hud_cols": rec["hud_cols"],
        "masked_cells": rec["masked_cells"],
        "masked_cell_fraction": rec["masked_cell_fraction"],
        "mask_status": rec["mask_status"],
        # NULL, not False, when there is no mask to judge. `False` here would read as "the
        # swallow guard examined this mask and refused it", which is a different and much
        # stronger statement than "there was nothing to examine" -- exactly the
        # clean-vs-unmeasurable conflation `hud_mask_swallow_clean` was written to stop
        # consumers making. `mask_status` carries the real reason.
        "combined_mask_swallow_clean": (
            rec["combined_mask_swallow_clean"] if rec["masked_cells"] else None
        ),
        "excluded_bands": [
            {
                "axis": b["axis"],
                "index": b["index"],
                "changed_fraction": b["changed_fraction"],
                "reason": b["decision"]["reason"],
                "alarms": b["decision"]["alarms"],
            }
            for b in rec["bands"]
            if not b["decision"]["accepted"]
        ],
        "how_to_apply": (
            "mask = np.zeros(tuple(logical_shape), bool); mask[hud_rows, :] = True; "
            "mask[:, hud_cols] = True; then "
            "arc_executable_world_model.apply_hud_mask(grid, mask) on BOTH sides of any "
            "exact-match comparison. An empty mask means MASKING IS REFUSED for this game "
            "-- apply nothing, and record mask_status as the reason."
        ),
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def main() -> int:
    t_start = time.time()
    with open(ROSTER_SOURCE) as fh:
        meta = json.load(fh)
    roster = list(meta["split_meta"].keys())

    preconditions = []

    def prec(resource: str, ok: bool, detail: str = "") -> None:
        preconditions.append({"resource": resource, "available": bool(ok), "detail": detail})

    prec("roster_source_meta_json", os.path.exists(ROSTER_SOURCE), ROSTER_SOURCE)
    env_dir = os.path.join(REPO, "environment_files")
    prec("offline_environment_files", os.path.isdir(env_dir), env_dir)
    try:
        import carnot.agentic.arc_entity_hud_perception as _m  # noqa: F401

        prec("detector_module_arc_entity_hud_perception", True, "")
    except Exception as exc:  # pragma: no cover - precondition path
        prec("detector_module_arc_entity_hud_perception", False, repr(exc))
    try:
        import carnot.agentic.arc_executable_world_model as _w  # noqa: F401

        prec("swallow_guard_arc_executable_world_model", True, "")
    except Exception as exc:  # pragma: no cover - precondition path
        prec("swallow_guard_arc_executable_world_model", False, repr(exc))
    prec("no_gpu_required", True, "CPU-only: offline arcade env-stepping plus numpy detectors")

    if not all(p["available"] for p in preconditions):
        print(json.dumps({"blocked": preconditions}, indent=2))
        return 1

    games: list[dict] = []
    for g in roster:
        print(f"[capture] {g} ...", flush=True)
        try:
            rec = capture_game(g)
        except Exception as exc:  # pragma: no cover - per-game failure is data, not a crash
            rec = {"game": g, "status": "error", "error": repr(exc)[:500]}
        games.append(rec)
        p = write_mask_file(rec)
        print(
            f"[capture] {g}: status={rec.get('status')} "
            f"bands={rec.get('n_bands_detected')} accepted={rec.get('n_bands_accepted')} "
            f"mask={rec.get('mask_status')} frac={rec.get('masked_cell_fraction')} "
            f"mask_file={'yes' if p else 'no'}",
            flush=True,
        )

    with open(os.path.join(OUT_DIR, "per_game.json"), "w") as fh:
        json.dump({"games": games}, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")

    index = {
        g["game"]: {
            "mask_file": f"masks/{g['game']}.json",
            "mask_status": g.get("mask_status"),
            "hud_rows": g.get("hud_rows", []),
            "hud_cols": g.get("hud_cols", []),
            "masked_cell_fraction": g.get("masked_cell_fraction", 0.0),
        }
        for g in games
        if g.get("status") == "ok"
    }
    with open(os.path.join(MASK_DIR, "_index.json"), "w") as fh:
        json.dump(index, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(json.dumps({"n_games": len(games), "elapsed_s": round(time.time() - t_start, 2)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
