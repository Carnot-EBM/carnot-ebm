"""Unit tests for FULL frame-only maze-model assembly
(scripts/arc3_frame_induction.py: _frame_walls + frame_to_maze_model).

frame_to_maze_model assembles a complete arc_maze_planner.MazeModel from a single frame: object start,
target, walls, checkpoints, and the hazard band -- all frame-induced -- plus the offline move-codes.
The load-bearing detail is that walls are decomposed into ROW-RUNS (not bounding boxes) so a concave
wall's interior PASSAGE is preserved (a bbox would fill the gap and over-block the planner -- the L7
failure mode). These pin that on synthetic frames. (The decisive check is tn36 L6+L7: frame_to_maze_
model -> planner -> the real env WINS.)

Spec: REQ-PHASE4-081, SCENARIO-PHASE4-081 (the full frame-only maze solve).
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

_MOVES = [((0, -4), 33), ((0, 4), 3), ((-4, 0), 1), ((4, 0), 2)]


def test_frame_walls_preserves_concave_gap():
    # two wall segments in one row with a gap between them (a passage). The row-run decomposition must
    # cover BOTH segments and leave the gap open (a bounding box would fill it -> over-block).
    maze = np.zeros((30, 64), dtype=np.int16)
    maze[:, 22:43] = 1
    maze[:, 43:] = 2  # three floor colours 0/1/2
    maze[5, 2:6] = 6  # left wall segment
    maze[5, 10:14] = 6  # right wall segment (gap at x[6,10))
    maze[9, 2:6] = 6  # a second row so colour 6 has >=2 CCs (structural)
    boxes = FI._frame_walls(maze, obj_color=7, floor={0, 1, 2}, play_top=2)
    assert (2, 7, 4, 1) in boxes and (10, 7, 4, 1) in boxes  # both segments, row-run boxes
    # no box covers the gap cell (x in [6,10), y=7) -> the passage stays open
    assert not any(bx <= 7 < bx + bw and by == 7 for bx, by, bw, bh in boxes)


def _floor():
    g = np.zeros((64, 64), dtype=np.int16)
    g[:, 22:43] = 1
    g[:, 43:] = 2
    return g


def _solid(g, color, x, y):
    g[y : y + 4, x : x + 4] = color
    g[y + 3, x + 1] = 0
    g[y + 3, x + 2] = 0  # bottom notch -> object


def _outline(g, color, x, y):
    g[y - 1, x - 1 : x + 5] = color
    g[y + 4, x - 1 : x + 5] = color
    g[y - 1 : y + 5, x - 1] = color
    g[y - 1 : y + 5, x + 4] = color


def _dither(g, color, x, y):
    for dy in range(4):
        for dx in range(4):
            if (dx + dy) % 2 == 0:
                g[y + dy, x + dx] = color


def test_frame_to_maze_model_assembles_from_a_single_frame():
    g = _floor()
    _solid(g, 7, 10, 20)  # object (solid, notched)
    _outline(g, 9, 40, 8)  # target (hollow outline)
    _dither(g, 7, 20, 8)  # a checkpoint pad (object-colour dither)
    g[12, 30:34] = 6  # two wall blocks (color 6, >=2 CCs -> structural)
    g[20, 30:34] = 6
    model = FI.frame_to_maze_model(g, 6, _MOVES)
    assert model is not None
    assert model.start == (10, 20) and model.target == (40, 8)
    assert (20, 8) in model.checkpoints
    assert len(model.walls) >= 2 and model.spikes_visible == []  # walls present, no hazard band
    assert model.n_slots == 6 and model.move_codes == _MOVES


def test_frame_to_maze_model_none_when_no_object():
    assert FI.frame_to_maze_model(_floor(), 6, _MOVES) is None
