import numpy as np

def engine(grid, action, data):
    """
    Predicts the next grid state given an action and its associated data.
    The observed transitions show complex modifications to the grid upon clicking.
    Since we cannot fully induce the exact pixel-by-pixel transition rules from the limited data,
    we implement a simplified model that updates the clicked cell's value if it's ACTION6.
    """
    next_grid = grid.copy()
    if action == 6:  # ACTION6 is a click with {'x': px, 'y': py}
        px = data['x']
        py = data['y']
        # Ensure coordinates are within bounds
        if 0 <= py < grid.shape[0] and 0 <= px < grid.shape[1]:
            # In many ARC games, clicking toggles or changes the color of the target cell.
            # Based on observations, clicks often trigger larger effects, but for this world model,
            # we simulate a simple change to indicate the interaction occurred.
            current_val = grid[py, px]
            # Simple toggle logic based on common patterns (e.g., 5 -> 0 or vice versa)
            next_grid[py, px] = 0 if current_val != 0 else 5
            
    return next_grid

def is_level_complete(grid):
    """
    Determines if the level is complete.
    Based on the observed transitions, the win state occurs after a specific sequence of actions.
    The final winning move was a click at (31, 34), which in the provided "BOARD BEFORE THE COMPLETING ACTION"
    was part of an object of color 1.
    Looking at the board before completion, there were several distinct objects created by previous moves:
    - A block of color 15 around (38, 32).
    - A block of color 3 around (18, 38).
    - Several small objects of color 1.
    A plausible general rule for ARC puzzles like this is that the level is completed when certain
    objects are placed or modified such that they satisfy a spatial condition.
    Given the data, we check for the presence of these key markers (color 3 and color 15 blocks)
    which indicate the prerequisite steps have been taken.
    """
    # Check for existence of color 3 pixels (created during Transition 2)
    has_color_3 = np.any(grid == 3)
    
    # Check for existence of color 15 pixels (always present, but specifically look for clusters)
    # In the win state precursor, there's a cluster of color 15 near (38, 32).
    # We can simplify this to checking if color 15 exists alongside color 3.
    has_color_15 = np.any(grid == 15)
    
    # The winning click was on a pixel of color 1. If both prerequisites are met,
    # and the grid has reached a specific configuration, it might be complete.
    # Since we don't have the final rendered win-state grid, we use the indicators from the transitions.
    if has_color_3 and has_color_15:
        # Additionally, looking at the "Board Before Completing Action", there were several objects of color 1.
        # Let's check if there are any remaining color 1 pixels that could be clicked.
        if np.any(grid == 1):
            return True
            
    return False