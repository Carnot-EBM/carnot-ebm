import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action 2: Move Down
    if action == 2:
        # The game seems to be a puzzle where blocks of colors are shifted down.
        # We identify "objects" which are contiguous regions of non-background color (color 9)
        # and based on thes observed transitions, it<|channel>thought
        # we shift everything that isn't background or static walls (color 10)
        # Shift all cells except those of color 10 (walls) and potentially some other fixed elements.
        # Fixed elements in this case seem to be color 10.
        new_grid = grid.copy()
        
        # Simple gravity/shift down for objects
        for c in range(grid.shape[1]):
            col = grid[:, c]
            # Find indices of non-wall, non-background cells
            # In these specific observations, only certain columns have moving parts.
            # 30 <= c < 33 is wall (color 10)
            # Background is 9.
            # Colors 5, 4, 0, 11 are likely moving parts.
            
            # To simulate ACTION 2 (Move Down), we move the 'active' pixels down if they can.
            # The logic seems to be a block movement rather than single pixel gravity.
            # We need to find blocks of colors [0, 4, 5, 11] and shift them down by 3 rows.
            # Since it's hard to induce exact physics from limited data, let's use the delta patterns.
            # The deltas show that when action 2 is called, things move down by 3 rows.
            # Shift everything not background (9) or wall (10) down by 3.
            # But wait, the observed transitions show a very specific pattern:
            # r15->r18, r16->r19, etc. This is exactly +3 rows.
            # Let's implement a simple "shift non-background/non-wall cells down by 3".
            
            # However, looking closer at the deltas:
            # Action 2 moves objects down by 3 units.
            # It also changes some values in column 63.
            
            # For simplicity and based on the provided examples, we will simulate the shift.
            # We only shift pixels that are part of the 'active' regions.
            # Active regions are those with colors other than 9 or 10.
            mask = (grid != 9) & (grid != 10)
            # To avoid overwriting, we process from bottom up.
            new_grid[3:] = grid[:-3] # This is too naive.
            
            # Correct logic for ACTION 2:
            # 1. Identify all active pixels (not 9, not 10).
            # 2. Move them down by 3.
            # 3. Fill old positions with background (9).
            # 4. Handle boundaries.
            
            temp_grid = np.full(grid.shape, 9)
            # Keep walls
            temp_grid[grid == 10] = 10
            # Shift others
            for r in range(grid.shape[0]):
                for c in range(grid.shape[1]):
                    if grid[r, c] != 9 and grid[r, c] != 10:
                        nr = r + 3
                        if nr < grid.shape[0]:
                            temp_grid[nr, c] = grid[r, c]
            
            # Special case for column 63 based on observed deltas:
            # Action 2 increments the row of a color-5 pixel in col 63.
            # The initial grid has color 5 at r0c63, r1c63, r2c63.
            # Transitions show ACTION 2 moves it to r5, r6, r7...
            # This is consistent with shift-down-by-3 if we consider only one active pixel there.
            
            return temp_grid

    # Action 3: Move Left/Right (based on observations, it shifts things horizontally)
    if action == 3:
        # Observed transitions for ACTION 3 show horizontal movement.
        # In the first transition, objects move from c6 to c12? No, let's look closer.
        # r15c6:5x3 -> r15c3:5x3. That' same as -3 columns.
        # Let's implement "shift non-background/non-wall cells left by 3".
        temp_grid = np.full(grid.shape, 9)
        temp_grid[grid == 10] = 10
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] != 9 and grid[r, c] != 10:
                    nc = c - 3
                    if nc >= 0:
                        temp_grid[r, nc] = grid[r, c]
        return temp_grid

    return grid

def is_level_complete(grid):
    # Win state usually involves reaching a certain configuration or collecting items.
    # Without a win state example, we assume it's complete if no active pixels are left
    # or they reach a specific area. Given the data, we can't be sure.
    # However, typically these games end when objects hit a target.
    # Let's check for any color other than 9 or 10 in the bottom rows.
    return False # Default to False as no WIN STATE was provided.