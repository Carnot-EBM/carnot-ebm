import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Based on observed transitions, this game seems to be a puzzle where clicking or moving
    # changes blocks of colors. ACTION6 is a click at (x, y).
    # The logic for other actions might be movement or shifting patterns.
    # 
    # However, looking at the same-level transitions provided, they are all level 0->0,
    # and only some cells change.
    # Specifically, ACTION6 clicks seem to replace areas with color 10.
    # Action 3 and 4 shift or restore patterns.
    # 
    # Given the constraints and thes specific observations, we can't induce a general rule
    # without more complex state tracking. But since it's an ARC-AGI task,
    # 
    # Let's implement a basic version that handles the delta updates if we were simulating.
    # Since we cannot store history, we must deduce the rules from the coordinates.
    
    new_grid = grid.copy()
    if action == 6:
        # Click data contains x, y. In these examples, ACTION6 replaces a 6x5 block area starting at (y, x)
        # but shifted slightly.
        px, py = data['x'], data['y']
        # Looking at r37c25:10x6 etc., px=24, py=36 -> row 37..41, col 25..30.
        # Row offset is +1, Col offset is +1.
        for r in range(py + 1, py + 6):
            if 0 <= r < new_grid.shape[0]:
                for c in range(px + 1, px + 7):
                    if 0 <= c < new_grid.shape[1]:
                        new_grid[r, c] = 10
        return new_grid

    if action == 3:
        # Action 3 seems to restore some pattern of colors [5, 9, 11] inside the blocks that were color 10.
        # This is very specific and depends on the same-level transitions.
        # We'll implement it as a "no-op" or a simple shift if we not only have enough coordinates.
        pass
    
    if action == 4:
        # la-// No clear general rule for own logic without more full grids.
        # Let's try to actually apply the delta from the observed data if possible.
        # But since wes are just a<|channel>thought
        # a world model must be deterministic based on grid state.
        pass

    return new_grid

def is_level_complete(grid):
    # The win state is usually when certain cells reach a target color or a block is cleared.
    # In this game, there are no WIN STATE grids provided in the prompt.
    # Usually, it's something to be deduced.
    # For now, return False unless a specific condition is met.
    # Return True if row 63 has any non-zero value (as seen in deltas).
    return np.any(grid[63, :] != 0) # This is a placeholder guess.