import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action 2: Move Down
    if action == 2:
        # Find all non-background cells (color 9) and move them down if possible.
        # This looks like a "gravity" or "push" mechanism where blocks are moved.
        # The observed transitions show that groups of blocks are shifting downwards.
        #
        # Based on the same pattern in ACTION2, it's a<|channel>thought
        # shift of everything except background color 9.
        # new_grid = grid.copy()
        # For each column, we find indices of non-9 cells.
        # For instance, if there's a block of colors [5, 0, 5], they maintain relative order.
        # Shift them all down to the bottommost available space.
        # Fill top with background color 9.
        
        new_grid = grid.copy()
        H, W = grid.shape
        for c in range(W):
            col = grid[:, c]
            non_bg = col[col != 9]
            new_col = np.full(H, 9, dtype=int)
            new_col[H - len(non_bg):] = non_bg
            new_grid[:, c] = new_col
        return new_grid

    # Action 3: Move Right (or Left? Let's check delta).
    # In ACTION3, r15c6 becomes 5x3, whereas initially r15 was 9x9, 5x9...
    # It seems like blocks are shifting horizontally.
    if action == 3:
        # Find all non-background cells and move them right.
        new_grid = grid.copy()
        H, W = grid.shape
        for r in range(H):
            row = grid[r, :]
            non_bg = row[row != 9]
            new_row = np.full(W, 9, dtype=int)
            new_row[W - len(non_bg):] = non_bg
            new_grid[r, :] = new_row
        return new_grid

    return grid

def is_level_complete(grid):
    # The win state isn't explicitly given, but typically it involves 
    # moving objects to a target area or clearing the board.
    # Looking at the INITIAL GRID, there are some specific colors (10, 11, 4, 5, 0).
    # A common goal is to align these colors with targets.
    # Let's assume completion when certain patterns are formed.
    # Since no WIN STATE was provided, we return False unless a clear condition is met.
    # In many ARC games, "win" means reaching a specific configuration.
    # Without a target, this is hard. However, usually if the blocks reach the bottom/right, it might be complete.
    # We will check for a simple condition: any cell of color 11 being in a specific place?
    # No, let's just return False as we don't have the win state.
    return False