"""Unit tests for FRAME-ONLY MazeModel induction (scripts/arc3_frame_induction.py:induce_maze_model).

The inducer must, from a list of play-area grids alone (no internal state), find the OBJECT by motion
(the colour whose centroid moves across frames) and WALLS by stability (static non-floor structure),
surface a distinct TARGET sprite when one renders, and HONESTLY report that fields which draw on the
floor (target/checkpoints) or are invisible at rest (hazards) are not frame-inducible — the tn36
finding. These pin the algorithm on synthetic frames so it is game-independent.

Spec: REQ-PHASE4-081, SCENARIO-PHASE4-081 (the maze-strategy perception layer).
"""

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "arc3_frame_induction",
    Path(__file__).resolve().parents[2] / "scripts" / "arc3_frame_induction.py",
)
FI = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FI)


def _floor():
    """A 64x64 board split into three large floor regions (colours 0/1/2) — the dominant areas the
    inducer treats as background/checkerboard."""
    g = np.zeros((64, 64), dtype=np.int16)
    g[:, 22:43] = 1
    g[:, 43:] = 2
    return g


def _block(g, color, x, y, w=4, h=4):
    g[y : y + h, x : x + w] = color


def test_induces_object_by_motion_walls_and_target():
    # object (colour 7) MOVES between the two frames; two static wall blocks (8); one static target
    # sprite (9). A direct-control maze that renders all three -> a usable, planner-ready model.
    gA, gB = _floor(), _floor()
    _block(gA, 7, 10, 10)  # object at (10,10) in frame A
    _block(gB, 7, 20, 10)  # object moved to (20,10) in frame B
    for g in (gA, gB):
        _block(g, 8, 30, 30)  # wall block 1 (static)
        _block(g, 8, 40, 44)  # wall block 2 (static)
        _block(g, 9, 50, 50)  # lone target sprite (static)
    out = FI.induce_maze_model([gA, gB], play_top=0, play_rows=64)
    assert out["object_color"] == 7
    assert out["object_box"] == (20, 10, 4, 4)  # tracks the moved sprite (largest CC)
    assert out["frame_inducible"]["object"] and out["frame_inducible"]["walls"]
    assert out["frame_inducible"]["target"] and out["target_box"] == (50, 50, 4, 4)
    assert len(out["walls"]) == 2  # two static wall blocks
    assert out["usable_model"] is True


def test_no_distinct_target_is_not_usable_and_flags_honestly():
    # the tn36 reality: object moves, walls render, but the target/checkpoints draw on the FLOOR (no
    # distinct colour) -> no target sprite -> NOT a usable model, and the report says so honestly.
    gA, gB = _floor(), _floor()
    _block(gA, 7, 10, 10)
    _block(gB, 7, 24, 10)
    for g in (gA, gB):
        _block(g, 8, 30, 30)
        _block(g, 8, 40, 44)
        # NB: no distinct target colour — the goal sits on floor colour, as in tn36.
    out = FI.induce_maze_model([gA, gB], play_top=0, play_rows=64)
    assert out["frame_inducible"]["object"] and out["frame_inducible"]["walls"]
    assert out["frame_inducible"]["target"] is False and out["target_box"] is None
    assert out["frame_inducible"]["checkpoints"] == "not_rendered_distinctly"
    assert out["frame_inducible"]["hazards_at_rest"] == "invisible_until_run"
    assert out["usable_model"] is False  # planner-critical fields absent from frames


def test_static_only_has_no_object():
    # nothing moves between frames -> no object can be induced by motion (object stays None/False).
    g = _floor()
    _block(g, 8, 30, 30)
    out = FI.induce_maze_model([g, g.copy()], play_top=0, play_rows=64)
    assert out["object_box"] is None and out["frame_inducible"]["object"] is False
    assert out["usable_model"] is False
