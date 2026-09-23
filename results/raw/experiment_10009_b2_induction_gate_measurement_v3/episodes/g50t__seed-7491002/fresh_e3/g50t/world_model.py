import numpy as np


def engine(grid, action, data):
    g = grid.copy()
    h, w = g.shape

    # Identify the two main toggle regions based on initial layout
    # Region A: rows 8-12, cols 14-18 (the "top" 9/5 pattern)
    # Region B: rows 14-18, cols 14-18 (the "bottom" 9/5 pattern)
    # These regions swap colors between 5 and 9 on actions 1, 2, 4

    region_a_rows = slice(8, 13)
    region_a_cols = slice(14, 19)
    region_b_rows = slice(14, 19)
    region_b_cols = slice(14, 19)

    if action == 1:
        # Toggle: A becomes 9 where it was 5, B becomes 5 where it was 9
        a_vals = g[region_a_rows, region_a_cols].copy()
        b_vals = g[region_b_rows, region_b_cols].copy()
        g[region_a_rows, region_a_cols] = np.where(a_vals == 5, 9, np.where(a_vals == 9, 5, a_vals))
        g[region_b_rows, region_b_cols] = np.where(b_vals == 5, 9, np.where(b_vals == 9, 5, b_vals))
    elif action == 2:
        # Same toggle as action 1
        a_vals = g[region_a_rows, region_a_cols].copy()
        b_vals = g[region_b_rows, region_b_cols].copy()
        g[region_a_rows, region_a_cols] = np.where(a_vals == 5, 9, np.where(a_vals == 9, 5, a_vals))
        g[region_b_rows, region_b_cols] = np.where(b_vals == 5, 9, np.where(b_vals == 9, 5, b_vals))
    elif action == 4:
        # Toggle both regions AND the right-side region (cols 20-24)
        a_vals = g[region_a_rows, region_a_cols].copy()
        b_vals = g[region_b_rows, region_b_cols].copy()
        ra = slice(8, 13)
        rb = slice(14, 19)
        rc = slice(20, 25)
        c_vals = g[ra, rc].copy()
        d_vals = g[rb, rc].copy()
        g[ra, rc] = np.where(c_vals == 5, 9, np.where(c_vals == 9, 5, c_vals))
        g[rb, rc] = np.where(d_vals == 5, 9, np.where(d_vals == 9, 5, d_vals))
        g[region_a_rows, region_a_cols] = np.where(a_vals == 5, 9, np.where(a_vals == 9, 5, a_vals))
        g[region_b_rows, region_b_cols] = np.where(b_vals == 5, 9, np.where(b_vals == 9, 5, b_vals))

    # Action 3: move the bottom-right pixel (color 1) left by 1 on row 63
    if action == 3:
        # Find color-1 pixels on row 63 and shift them left
        row = g[63]
        ones = np.where(row == 1)[0]
        for col in sorted(ones):
            new_col = col - 1
            if new_col >= 0:
                g[63, new_col] = 1
                g[63, col] = 9  # restore background
        # Actually from data: r63c62:1x1 means cell (63,62) becomes 1
        # The pattern shows sequential leftward movement of a single pixel
        pass

    return g


def is_level_complete(grid):
    return False