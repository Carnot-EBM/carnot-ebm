import numpy as np

def engine(grid, action, data):
    """
    The game appears to be a puzzle where clicking on specific areas toggles colors 
    or shifts patterns. Based on the observed transitions, ACTION6 (click) at 
    specific coordinates changes blocks of cells from color 14 back to color 1 or 3, 
    and vice versa, while also incrementing a counter in the top row (r0).
    
    Looking closely at the deltas:
    - Clicking x=18, y=19 affects columns around 17-21 and rows 17-22.
    - Clicking x=30, y=19 affects columns around 29-33 and rows 17-22.
    - The pattern seems to involve replacing 'color 14' segments with 'color 1' or 'color 3'.
    - There is a persistent change in grid[0, col] which acts as a click counter.
    """
    if action != 6:
        return grid.copy()

    new_grid = grid.copy()
    px, py = data['x'], data['y']
    
    # Increment the click counter in the first row
    # Find the first cell in r0 that is 0 and set it to 1
    for c in range(64):
        if new_grid[0, c] == 0:
            new_grid[0, c] = 1
            break

    # Logic for updating blocks based on clicks
    # This part of the world model approximates the observed delta patterns.
    # In ARC games like this, clicking often toggles state or clears obstacles.
    # We observe that clicking at (18, 19), (30, 19), (42, 19) targets specific columns.
    
    target_cols = [17, 18, 19, 20, 21, 29, 30, 31, 32, 33, 41, 42, 43, 44, 45]
    rows_affected = [17, 18, 19, 20, 21, 22]
    
    # Determine which column group was clicked
    clicked_group = None
    if 15 <= px <= 22:
        clicked_group = range(17, 22)
    elif 27 <= px <= 35:
        clicked_group = range(29, 34)
    elif 39 <= px <= 47:
        clicked_group = range(41, 46)

    if clicked_group:
        for r in rows_affected:
            for c in clicked_group:
                if 0 <= r < 64 and 0 <= c < 64:
                    # Toggle between color 14 (yellow/orange) and others
                    if new_grid[r, c] == 14:
                        new_grid[r, c] = 1 # Reset to base pattern
                    elif new_grid[r, c] == 1:
                        new_grid[r, c] = 14 # Set to active state
    
    return new_grid

def is_level_complete(grid):
    """
    The level is complete when a specific target configuration is reached.
    Given the lack of a WIN STATE grid, we assume completion occurs when 
    the top row counter reaches a certain value or all 'obstacles' are cleared.
    """
    # Based on typical ARC patterns, check if a goal condition is met.
    # Since no win state was provided, return False unless a clear signal exists.
    return np.sum(grid[0, :]) >= 5  # Example threshold for clicks