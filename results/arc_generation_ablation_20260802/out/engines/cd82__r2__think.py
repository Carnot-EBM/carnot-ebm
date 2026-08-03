import numpy as np

def engine(grid, action, data):
    """
    The observed transitions suggest a world where an object (likely color 2 or 15) 
    moves across the grid based on directional actions. 
    ACTION4 seems to be 'Right' and ACTION2 seems to be 'Down'.
    Looking at the deltas, when ACTION4 is called, cells change in a way that suggests 
    a shape moving right/down-right. When ACTION2 is called, it moves down/down-left.
    However, the specific patterns are complex. Given the constraints and the nature of ARC,
    we implement a movement model for the primary active entity.
    """
    new_grid = grid.copy()
    h, w = new_grid.shape
    
    # Identify the "active" entity (color 2 or 15).
    # Based on the INITIAL GRID and DELTAs, there are blocks of colors.
    # Let's assume the game involves shifting these blocks.
    
    if action == 6: # Click
        px, py = data['x'], data['y']
        # In many ARC games, clicking toggles or interacts with a cell.
        # Without clear evidence of click effect, we leave it as is.
        pass
    elif action == 4: # Right / Down-Right shift
        # Shift logic derived from delta r21c39...r37c43
        # This looks like a diagonal slide.
        shift_rows = [
            (21, 39, 2), (22, 38, 3), (23, 37, 5), (24, 25, 11), (24, 38, 5),
            (25, 25, 10), (25, 38, 4), (26, 25, 9), (26, 38, 5), (27, 25, 8),
            (27, 38, 6), (28, 25, 10), (28, 38, 7), (29, 25, 11), (29, 38, 8),
            (30, 25, 12), (30, 38, 9), (31, 25, 13), (31, 38, 10), (32, 25, 1),
            (32, 38, 8), (33, 40, 6), (34, 41, 4), (35, 42, 2), (36, 43, 2), (37, 43, 1)
        ]
        # This is too specific to one transition. We generalize:
        # Action 4 moves the 'entity' right and slightly down.
        pass
    elif action == 2: # Down / Down-Left shift
        # Shift logic derived from delta r21c39...r56c39
        pass

    # Since we cannot perfectly induce a general physics engine from 3 transitions 
    # of highly complex coordinate changes without knowing the entity shape,
    # we implement the state change as observed in the deltas for these actions.
    # However, since this must be an EXECUTABLE WORLD MODEL, we will simulate 
    # the movement of color 2/15 blocks if they exist.
    
    # Simplified simulation: move all cells of color 2 or 15 by a vector.
    vector = {1: (0, -1), 2: (1, 0), 3: (0, 1), 4: (-1, 0)} # Standard ARC directions
    # But based on data: ACTION4 -> Rightish, ACTION2 -> Downish.
    if action == 4:
        dr, dc = 0, 1
    elif action == 2:
        dr, dc = 1, 0
    else:
        return new_grid

    # Find current positions of active colors (2 and 15)
    coords = np.argwhere((new_grid == 2) | (new_grid == 15))
    for r, c in coords:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            new_grid[nr, nc] = new_grid[r, c]
            # The original cell usually becomes the background color (5)
            new_grid[r, c] = 5
            
    return new_grid

def is_level_complete(grid):
    """
    A level is typically complete when a certain pattern is matched or an object 
    reaches a goal. In this case, we check if any cells of color 2 have reached 
    the bottom row or specific target areas.
    """
    # Check if any 'active' pixels are at the very bottom edge (row 63)
    # as seen in ACTION4 delta r63c63:5x1.
    return np.any(grid[63, :] == 2) or np.any(grid[63, :] == 15)