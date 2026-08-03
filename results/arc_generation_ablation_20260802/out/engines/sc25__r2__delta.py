import numpy as np

def engine(grid, action, data=None):
    """
    Predicts the next state of the grid based on the action.
    Action 3 corresponds to moving left (based on observed transitions).
    """
    out = grid.copy()
    if action == 3:
        # The same pattern [9, 10, 2, 2] moves left by 2 columns per ACTION3 call.
        # We need to find where these patterns are located in rows 19 and 20.
        # Based on thes observations, the sequence [9, 10, 2, 2] is shifting left.
        # Find current positions of the sequence [9, 10, 2, 2] in rows 19 and 20.
        for r in [19, 20]:
            # Search for the start of the sequence [9, 10, 2, 2]
            for c in range(64 - 4):
                if (np.array_equal(grid[r, c:c+4], [9, 10, 2, 2])):
                    # Move it left by 2 units.
                    # Clear old position (set back to background color 5)
                    out[r, c:c+4] = 5
                    # Place new position (new column index)
                    new_c = max(0, c - 2)
                    out[r, new_c:new_c+4] = [9, 10, 2, 2]
                    break
    return out

def is_level_complete(grid):
    """
    Returns True if the level is complete/win state.
    No win state provided in observed transitions, but typically involves reaching a target or clearing specific colors.
    """
    # Based on the same data, we umave no explicit win condition.
    # return False as default since there not only<|channel>thought////no_think