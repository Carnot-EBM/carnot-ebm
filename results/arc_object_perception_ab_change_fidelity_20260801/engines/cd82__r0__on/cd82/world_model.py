import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action mapping based on observed transitions:
    # ACTION4: Likely a "down-right" or similar movement/interaction.
    # ACTION2: Likely a "left" or "up" movement/interaction.
    # The provided transitions show complex changes in color and values.
    # It's a<|channel>thought
    # Let's analyze the delta patterns.
    # In ACTION4, cells change from 5 to 2, then some 15s appear.
    # In ACTION2, cells change back to 5, and other areas shift.
    # This looks like a puzzle where an object (color 2) moves and interacts with others.
    # The game state involves colors [0, 2, 3, 4, 5, 15].
    # Color 5 seems to be the background.
    # Based on the deltas, it's a movement of a specific entity own by color 2.
    # We need to find the position of the 'player' object (color 2) and move it.
    #
    # However, the prompt asks for SIMPLE GENERAL rules.
    # Looking at the deltas again:
    # ACTION4: r21c39:2x1 ... this is moving something towards the bottom right.
    # ACTION2: r21c39:5x1 ... this is restoring background.
    #
    # Given the limited data and high complexity of the deltas, we will implement a basic
    # version that handles the observed actions as movements of the "active" object.

    if action == 6: # Click
        return grid
    
    # Find all cells of color 2 (the active object)
    coords = np.argwhere(grid == 2)
    if coords.size == 0:
        return grid
    
    # Simple translation based on action
    dy, dx = 0, 0
    if action == 1: dy, dx = -1, 0 # Up
    elif action == 2: dy, dx = 0, -1 # Left
    elif action == 3: dy, dx = 0, 1  # Right
    elif action == 4: dy, dx = 1, 0  # Down
    
    # This doesn't match the complex deltas perfectly but follows general ARC logic.
    # The actual transitions are far more complex (changing shapes).
    # Let's try to be more specific about the observed ACTION4 and ACTION2.
    
    # Since I cannot deduce the exact physics from 3 frames, I will return the grid
    # unless it's a click, which is not provided in data.
    
    # For this specific task, we must provide an executable world model.
    # If the rules are too complex to induce, returning the same grid is a fallback,
    # but let's implement basic movement for color 2.

    new_grid = grid.copy()
    for y, x in coords:
        ny, nx = y + dy, x + dx
        if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
            # Move object cell if destination is background (5)
            if new_grid[ny, nx] == 5:
                new_grid[ny, nx] = 2
                new_grid[y, x] = 5
                
    return new_grid

def is_level_complete(grid):
    # Win state usually involves reaching a target or clearing objects.
    # In many ARC games, it's when certain colors disappear or align.
    # Looking at INITIAL GRID, there are several "target" like areas (color 0).
    # We'll assume completion if all cells of color 0 are replaced by something else.
    return np.sum(grid == 0) == 0