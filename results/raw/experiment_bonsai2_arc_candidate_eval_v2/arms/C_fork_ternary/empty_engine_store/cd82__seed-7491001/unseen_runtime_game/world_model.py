import numpy as np


def _decode_delta(delta_str):
    """Parse a delta string into a list of (row, col_start, [(value, count), ...])."""
    if not delta_str or delta_str == "(no change)":
        return []
    runs = []
    # Split on spaces
    parts = delta_str.split()
    for part in parts:
        # Format: r<row>c<col>:<v>x<n>,<v>x<n>,...
        row_col_part, value_part = part.split(':')
        # Parse row and col from "r<row>c<col>"
        rc = row_col_part[1:]
        c_idx = rc.find('c')
        row = int(rc[:c_idx])
        col = int(rc[c_idx + 1:])
        # Parse values
        vals = []
        for vpart in value_part.split(','):
            v, n = vpart.split('x')
            vals.append((int(v), int(n)))
        runs.append((row, col, vals))
    return runs


def _apply_delta(grid, delta_runs):
    """Apply delta runs to grid."""
    g = grid.copy()
    for row, col, vals in delta_runs:
        c = col
        for val, cnt in vals:
            g[row, c:c+cnt] = val
            c += cnt
    return g


def engine(grid, action, data=None):
    H, W = grid.shape
    out = grid.copy()

    if action == 6:
        # Click -- increment a counter (bottom-right corner)
        px, py = data['x'], data['y']
        # The click seems to just change r63c63 -> r63c62 -> ... (decrementing column index)
        # Actually looking at the pattern: each ACTION6 changes exactly one cell in row 63
        # from 4 to 5. The column decreases by 1 each time.
        # This looks like a "score" or "progress" indicator.
        # Let's track it differently: find the leftmost 4 in row 63 and set it to 5?
        # No - looking more carefully, the bottom row starts as all 4s (r63:4x64)
        # After clicks, cells turn from 4 to 5 going right-to-left.
        # So we need to track how many cells have been clicked.
        # Simplest: count number of 5s in row 63, then set next 4 to 5.
        # But wait - the delta shows only ONE cell changing per click.
        # r63c63:5x1 means cell (63,63) changed to 5. Then r63c62:5x1, etc.
        # So each click sets the next cell (from right to left) in row 63 from 4 to 5.
        # We can just find the first 4 from the right in row 63 and change it to 5.
        if out[63].any():  # row exists
            # Find rightmost 4 in row 63
            for c in range(W-1, -1, -1):
                if out[63, c] == 4:
                    out[63, c] = 5
                    break
        return out

    if action == 1 or action == 2 or action == 3 or action == 4 or action == 5:
        # Movement actions
        # Looking at the pattern more carefully...
        # The game seems to involve moving a "ball" or object around a maze
        # with gravity pulling objects down
        
        pass
    
    return out


def is_level_complete(grid):
    """Return True if level is complete."""
    H, W = grid.shape
    # Check if all of row 63 is color 5 (all clicks registered)
    return np.all(grid[63] == 5)