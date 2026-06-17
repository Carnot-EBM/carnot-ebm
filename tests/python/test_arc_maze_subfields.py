"""Unit tests for FRAME-ONLY maze sub-field induction
(scripts/arc3_frame_induction.py:induce_maze_sub_fields).

Checkpoints + hazards are not in a single static frame the naive way (checkpoints draw on the floor
checkerboard, spikes are invisible at rest) — but both leave a STATIC marking: a checkpoint is a
DITHERED 4x4 of the OBJECT's colour (distinct from the solid object / hollow target), and the hazard
band renders a distinct low-area MARKER colour in a tight horizontal band. These pin both reads + the
dither/solid disambiguation on synthetic frames. (The decisive check is tn36 L6/L7: induced
checkpoints + hazard band == internal truth, exact.)

Spec: REQ-PHASE4-081, SCENARIO-PHASE4-081 (the maze planner's perception inputs).
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
    g = np.zeros((64, 64), dtype=np.int16)
    g[:, 22:43] = 1
    g[:, 43:] = 2
    return g


def _solid_object(g, color, x, y):
    """A solid 4x4 sprite with a bottom notch (so it is the OBJECT, notched, not a wall)."""
    g[y : y + 4, x : x + 4] = color
    g[y + 3, x + 1] = 0
    g[y + 3, x + 2] = 0


def _dither_checkpoint(g, color, x, y):
    """A checkpoint pad: a DITHERED 4x4 checkerboard of the object colour (8 diagonal pixels)."""
    for dy in range(4):
        for dx in range(4):
            if (dx + dy) % 2 == 0:
                g[y + dy, x + dx] = color


def test_induces_checkpoints_and_hazard_band():
    g = _floor()
    _solid_object(g, 7, 10, 20)  # object colour 7 (notched solid)
    _dither_checkpoint(g, 7, 40, 8)  # a checkpoint pad in the object colour (dithered)
    _dither_checkpoint(g, 7, 14, 8)  # a second checkpoint pad
    for x in (37, 48, 59):  # hazard band: marker colour 8 in a tight row at y=16
        g[16, x] = 8
    out = FI.induce_maze_sub_fields(g, play_top=2, maze_bottom=32)
    assert (40, 8, 4, 4) in out["checkpoints"] and (14, 8, 4, 4) in out["checkpoints"]
    assert len(out["checkpoints"]) == 2  # the solid object is NOT a checkpoint (it was removed)
    hx, hy, hw, hh = out["hazard_band"]
    assert hy == 16 and hx == 37 and hx + hw - 1 == 59  # the marker-colour band, bbox exact


def test_no_hazard_when_no_marker_colour():
    g = _floor()
    _solid_object(g, 7, 10, 20)
    _dither_checkpoint(g, 7, 40, 8)
    out = FI.induce_maze_sub_fields(g, play_top=2, maze_bottom=32)
    assert out["hazard_band"] is None and (40, 8, 4, 4) in out["checkpoints"]


def test_empty_maze_has_no_subfields():
    out = FI.induce_maze_sub_fields(_floor(), play_top=2, maze_bottom=32)
    assert out["checkpoints"] == [] and out["hazard_band"] is None
