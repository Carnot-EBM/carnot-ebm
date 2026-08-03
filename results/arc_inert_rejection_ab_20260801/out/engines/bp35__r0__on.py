import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The game seems to be a puzzle where clicking or moving changes blocks of colors.
    # Based on the observed transitions, ACTION 6 (click) and ACTION 3/4 (directional)
    # appear to shift or toggle patterns of cells.
    # Specifically, ACTION 6 at (x, y) often replaces a region around that point with color 10.
    # Color 10 acts like a 'wall' or 'filler'.
    #
    # Let's implement a basic version based on the same logic seen in the la-//
    # In this case, we<|channel>thought
    #
    # Let's refine the transition rules from the observations.
    # ACTION 6 (Click): Replaces a region around (x, y) with color 10.
    # ACTION 3 (Directional - Left?): Shifts some specific pattern of colors (5, 9, 11)
    #   around. It also increments a counter in row 63.
    # ACTION 4 (Directional - Right?): Shifts the opposite way.
    #
    # Since the exact geometry is complex, let's focus on the most salient features:
    # The grid contains regions of color 10 (walls), color 5 (background),
    # and other colors (objects).
    #
    # Given the constraints and the provided transitions, it looks like the game
    # involves moving an object (the block of colors 5, 9, 11) across the board.
    #
    # Let's try to simulate the movement of the "object" (the block of colors 5, 9, 11).
    # Object coordinates are found by looking for values not equal to 5 or 10.
    #
    # For simplicity, we will implement the engine based on the observed deltas.
    #
    # Note: Row 63 column index seems to be tracking progress/moves.

    new_grid = grid.copy()
    
    if action == 6:
        # Click at (x, y). In observations, this often fills a region with 10.
        px, py = data['x'], data['y']
        # Based on ACTION 6 data={'x': 24, 'y': 36}, r37-r41 c25-c30 became 10.
        # This is roughly x+1, y+1 to x+6, y+5.
        for r in range(py + 1, py + 6):
            for c in range(px + 1, px + 6):
                if 0 <= r < new_grid.shape[0] and 0 <= c < new_grid.shape[1]:
                    new_grid[r, c] = 10
        # Update counter in row 63
        count = np.sum(new_grid[63, :] != 15) # Simplified counter logic
        new_grid[63, count] = 15 if count < 64 else 15
        
    elif action == 3:
        # Directional shift left?
        # Find the "object" (colors other than 5 or 10).
        # Let's look for color 9 or 11 specifically.
        obj_coords = np.argwhere((grid != 5) & (grid != 10))
        if len(obj_coords) > 0:
            min_c = obj_coords[:, 1].min()
            max_c = obj_coords[:, 1].max()
            # Shift object left by 6 columns
            # This is a very rough approximation of the observed ACTION 3 behavior.
            # We need to restore background before shifting.
            for r, c in obj_coords:
                new_grid[r, c] = 5
            for r, c in obj_coords:
                if c - 6 >= 0:
                    new_grid[r, c - 6] = grid[r, c]
        # Update counter in row 63
        count = np.sum(new_grid[63, :] == 15) # Simplified
        if count < 64:
            new_grid[63, count] = 15
            
    elif action == 4:
        # Directional shift right?
        obj_coords = np.argwhere((grid != 5) & (grid != 10))
        if len(obj_coords) > 0:
            for r, c in obj_coords:
                new_grid[r, c] = 5
            for r, c in obj_coords:
                if c + 6 >= 0 and c + 6 < new_grid.shape[1]:
                    new_grid[r, c + 6] = grid[r, c]
        # No counter update observed for ACTION 4? Or different logic.

    return new_grid

def is_level_complete(grid):
    # The win state isn't explicitly given, but usually it involves a specific configuration.
    # In many ARC games, the goal is to clear objects or reach a target.
    # Let's assume completion if row 63 is filled with color 15 up to some point.
    return np.sum(grid[63, :] == 15) >= 10 # Heuristic based on observations.