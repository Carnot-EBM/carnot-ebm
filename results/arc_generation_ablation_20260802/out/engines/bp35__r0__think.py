import numpy as np

def engine(grid, action, data):
    """
    World model for ARC game bp35.
    The grid contains patterns of colors. Action 3 seems to shift a pattern left.
    Action 6 clicks a location and fills/clears an area.
    Action 4 shifts a pattern right.
    Based on the observed transitions, this looks like a puzzle where specific blocks
    of color are moved or toggled.
    """
    new_grid = grid.copy()
    h, w = new_grid.shape

    if action == 3:
        # ACTION 3: Shift a specific pattern (likely the 'active' object) to the left.
        # In the observations, it moves a 5x5-ish complex structure left by 6 units.
        # We look for the unique pattern (colors 9, 11) and move it.
        for r in range(h):
            for c in range(w - 1, 5, -1):
                if new_grid[r, c] == 9: # Marker for the moving block
                    # This is a simplification; we apply the delta logic from observation
                    pass
        # Since inducing exact coordinate-based movement without clear anchors is hard,
        # we simulate the observed behavior: shifting the "object" at row 37-41.
        # The object is roughly columns [c, c+5]. It moves to [c-6, c-1].
        obj_mask = (grid == 9) | (grid == 11)
        if np.any(obj_mask):
            coords = np.argwhere(obj_mask)
            min_c = np.min(coords[:, 1])
            max_c = np.max(coords[:, 1])
            rows = coords[:, 0]
            
            # Clear old
            new_grid[np.ix_(rows, range(min_c, max_c + 1))] = 5
            # Shift left by 6
            start_c = max(0, min_c - 6)
            end_c = max(0, max_c - 6)
            # Note: this is a rough approximation of the complex pattern shift
            for r in rows:
                for c in range(min_c, max_c + 1):
                    new_grid[r, max(0, c-6)] = grid[r, c]

    elif action == 4:
        # ACTION 4: Shift right. Inverse of Action 3.
        obj_mask = (grid == 9) | (grid == 11)
        if np.any(obj_mask):
            coords = np.argwhere(obj_mask)
            min_c = np.min(coords[:, 1])
            max_c = np.max(coords[:, 1])
            rows = coords[:, 0]
            new_grid[np.ix_(rows, range(min_c, max_c + 1))] = 5
            for r in rows:
                for c in range(min_c, max_c + 1):
                    target_c = min(w - 1, c + 6)
                    new_grid[r, target_c] = grid[r, c]

    elif action == 6:
        # ACTION 6: Click at x, y. Fills a block with color 10.
        px, py = data['x'], data['y']
        # The observed click at (24, 36) filled r37-41, c25-30 with 10s.
        # Logic: Fill a 5x6 area around the clicked point.
        # Based on observations: y is row, x is col.
        row_start = py - 1 # approx
        col_start = px + 1 # approx
        # Adjusting to match observation: Action 6 {x:24, y:36} -> r37-41, c25-30
        # This suggests row = y+1, col = x+1
        rs, re = py + 1, py + 6
        cs, ce = px + 1, px + 7
        if rs < h and re <= h and cs < w and ce <= w:
            new_grid[rs:re, cs:ce] = 10

    return new_grid

def is_level_complete(grid):
    """
    The win state isn't explicitly provided as a full grid, but usually involves
    clearing certain colors or reaching a specific configuration.
    Given the data, we check if the 'target' color (e.g., 15) has reached a goal.
    In the initial grid, r63c0 is 15x5. The transitions show r63cx changing.
    If r63c63 becomes 15, it might be complete.
    """
    # Based on observed ACTION3/6 deltas: r63c5, r63c6... are being set to 15.
    # It looks like a progress bar at the bottom.
    return np.sum(grid[63, :] == 15) >= 60