import numpy as np

import numpy as np


def _decode_delta(delta):
    """Parse a delta string into (row, col_start, [(value, count), ...]) runs."""
    if not delta or delta == "(no change)":
        return []
    runs = []
    for part in delta.split(","):
        # each run is like r<row>c<col0>:<v0>x<n0>,<v1>x<n1>...
        # but the format is r<row>c<col0>:<v0>x<n0>,<v1>x<n1>,...
        # Actually re-reading: "r<row>c<col0>:<v0>x<n0>,<v1>x<n1>,..."
        # So each comma-separated segment after splitting by ',' at top level
        # is tricky because values also have commas? No - the format uses
        # commas between value-count pairs within a run too.
        pass
    return runs


def engine(grid, action, data):
    g = grid.copy()
    return g


def is_level_complete(grid):
    return False

def is_level_complete(grid):
    if grid is None:
        return False
    try:
        rows = grid.shape[0]
        cols = grid.shape[1]
    except Exception:
        return False
    if rows == 0 or cols == 0:
        return False
    # Count occurrences of each color
    counts = {}
    for r in range(rows):
        for c in range(cols):
            v = grid[r, c]
            counts[v] = counts.get(v, 0) + 1
    # Win condition: color 14 (the special color) must be the majority
    # In the start state, color 14 appears only in a small region (rows 49-63, col 0-13)
    # In the win state, the special color should dominate the board
    total = rows * cols
    if 14 in counts:
        if counts[14] > total // 2:
            return True
    return False
