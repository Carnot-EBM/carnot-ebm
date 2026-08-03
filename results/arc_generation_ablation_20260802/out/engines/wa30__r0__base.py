import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Based on observed transitions, this looks like a puzzle game where blocks are moved or modified.
    # The provided observations are very limited and specific to own coordinates.
    # Inducing general rules for movement/action effects based on the same coordinate changes.
    
    new_grid = grid.copy()
    
    if action == 6: # Click action
        px, py = data['x'], data['y']
        # In most ARC games, clicking modifies a cell or toggles it.
        # if grid[py, px] == 0:
        #     // logic here
        #     # new_grid[py, px] = 1
        #     # pass
        #     pass
        # { "type": "type_a", "id": "id_a" }
        pass
    
    elif action == 4: # Right move / Action 4
        # Looking at ACTION4 deltas: r32c32:1x4,14x3,0x1... etc.
        # These cells are changing from original values to something else.
        # It's essentially shifting a block of colors (1, 14, 0) horizontally.
        # This kind of behavior is 
        #
        #<|channel>thought
        # The observed transitions show specific blocks of pixels moving or changing color.
        # Let's try to implement a simple shift for the same coordinates mentioned in the observations.
        # 
        # 
        # 
        # 
        #  uma own coordinate-based approach since general rules are not clear.
        pass

    return new_grid

def is_level_complete(grid):
    # return True if `grid` is a level-complete / win state, else False.
    # In these games, usually completion is when certain conditions are met.
    # Often it involves clearing a board or reaching a target configuration.
    # Based on the provided data, we don't have a WIN STATE grid.
    # However, often "is_level_complete" can be checked by looking at the same cell that changes last.
    # return False
    return False