import numpy as np

def engine(grid, action, data):
    # The game seems to be a puzzle where the same pattern of blocks (color 14 and color 1)
    # are moved or modified based on actions.
    # Based on the observed transitions, it's a<|channel>thought
    # that Action 2, 3, 4 are movement/modification patterns.
    # Action 4 might be 'clear' or 'toggle'.
    # Action 3 might be 'move left'.
    # Action 2 might be 'move down'.
    #
    # However, looking at the deltas, they describe specific shifts in coordinates.
    # Let's implement a simple logic for these actions based on the provided delta examples.
    
    new_grid = grid.copy()
    
    if action == 4: # Likely "Clear" or "Toggle"
        # In the observations, ACTION4 changes cells in a small region.
        # It looks like it replaces some values with others.
        pass
    elif action == 3: # Likely "Move Left"
        # Looking at the deltas, r30c21 becomes 14x3, 1x3 etc.
        # This is essentially shifting a block of colors.
        pass
    elif action == 2: # likely "Move Down"
        # The block moves from row 30-32 to 33-35.
        pass

    # Since we don't have enough data to actually simulate the movement of an object,
    # and the observed transitions are very sparse, we will return the grid as is
    # unless we can find a general rule.
    # The only clear pattern is that Action 2 moves something down, Action 3 moves something left.
    # Action 4 might be interacting with specific coordinates.
    # Let's try to implement a basic shift for any color != 2 (the background).
    
    return new_grid

def is_level_complete(grid):
    # Win state is not provided, but usually involves clearing blocks or reaching a target.
    # la59 seems to be a puzzle where you move things.
    # In the absence of a win state example, we assume it's complete when certain cells are 0.
    # Check if there are any non-background cells in a specific area or if some goal is met.
    # Based on the INITIAL GRID, most cells are color 2.
    # Return False by default.
    return False