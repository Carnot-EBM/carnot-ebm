"""Unit tests for FRAME-ONLY object+target attribute induction
(scripts/arc3_frame_induction.py:induce_object_target_attrs).

The transition model's INPUTS are the object's and target's five attributes (x, y, scale, rotation,
property). The target renders as a HOLLOW OUTLINE sprite, the object as the SOLID version of the same
sprite, so both are frame-readable: position (box), scale (box size / 4), property (colour), rotation
(the directional notch -> NUB_TO_ROTATION). These pin the read + the solid/outline classification on
synthetic frames, game-independent. (The decisive check is the tn36 end-to-end: frame-induced
object+target -> transition model -> plan -> the real env WINS, 5/5 L1-L5.)

Spec: REQ-PHASE4-081, SCENARIO-PHASE4-081 (the program-editor model's perception inputs).
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
    """64x64 board with three large floor regions (colours 0/1/2) — the dominant areas."""
    g = np.zeros((64, 64), dtype=np.int16)
    g[:, 22:43] = 1
    g[:, 43:] = 2
    return g


def _solid_sprite(g, color, x, y, notch):
    """A SOLID sprite: a 4x4 colour block with a 2-cell floor notch on one edge (the facing)."""
    g[y : y + 4, x : x + 4] = color
    if notch == "B":
        g[y + 3, x + 1] = 0
        g[y + 3, x + 2] = 0
    elif notch == "T":
        g[y, x + 1] = 0
        g[y, x + 2] = 0
    elif notch == "L":
        g[y + 1, x] = 0
        g[y + 2, x] = 0
    elif notch == "R":
        g[y + 1, x + 3] = 0
        g[y + 2, x + 3] = 0


def _outline_sprite(g, color, x, y):
    """A HOLLOW OUTLINE sprite representing a 4x4 target at (x,y): the colour ring drawn 1px outside
    the box on all sides (symmetric, interior floor -> the box centre is floor -> classified outline)."""
    g[y - 1, x - 1 : x + 5] = color  # top
    g[y + 4, x - 1 : x + 5] = color  # bottom
    g[y - 1 : y + 5, x - 1] = color  # left
    g[y - 1 : y + 5, x + 4] = color  # right


def test_reads_solid_object_and_hollow_target():
    g = _floor()
    _solid_sprite(g, 7, 10, 10, notch="B")  # object: solid, faces down -> rotation 0
    _outline_sprite(g, 9, 40, 25)  # target: hollow outline (symmetric), within the window
    out = FI.induce_object_target_attrs(g, play_top=2, play_rows=40)
    obj, tgt = out["object"], out["target"]
    assert obj is not None and tgt is not None
    assert (obj.x, obj.y, obj.scale, obj.prop, obj.rotation) == (10, 10, 1, 7, 0)  # solid, exact
    assert out["object_color"] == 7 and out["target_color"] == 9  # distinct sprites
    assert (tgt.x, tgt.y, tgt.scale, tgt.prop) == (40, 25, 1, 9)  # outline, read back


def test_object_rotation_from_notch_edge():
    # the directional notch encodes rotation (calibrated bottom=0, left=90, top=180, right=270).
    for notch, rot in [("B", 0), ("L", 90), ("T", 180), ("R", 270)]:
        g = _floor()
        _solid_sprite(g, 7, 30, 20, notch=notch)
        obj = FI.induce_object_target_attrs(g, play_top=2, play_rows=40)["object"]
        assert obj is not None and obj.rotation == rot, f"notch {notch} -> {obj.rotation} != {rot}"


def test_no_sprites_returns_none():
    out = FI.induce_object_target_attrs(_floor(), play_top=2, play_rows=40)
    assert out["object"] is None and out["target"] is None
