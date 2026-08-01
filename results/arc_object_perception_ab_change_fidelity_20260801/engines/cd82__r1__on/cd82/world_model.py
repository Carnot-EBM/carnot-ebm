import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The game seems to be a puzzle where an object (color 2) moves and interacts with other elements.
    # Based on the observed transitions, ACTION4 (down/right) and ACTION2 (up/left) move the same object.
    # Action 4 is often associated with moving 'down-right' or similar.
    # Action 2 is often associated moving 'up-left'.
    # Action 6 is click.
    
    # Let's identify the "player" object which is color 2.
    # In the initial state, obj11 is color 2.
    # In the first delta, it moves from bbox=(24, 25, 32, 38) to something else.
    # In the second delta, it {something} happens.
    # la own logic for finding the player object.
    
    # new_grid = grid.copy()
    # return new_grid
    
    # This is not even a<|channel>thought//no_think
    # I will implement a simple movement model based on the provided deltas.
    # Since the actual rules are complex and only few examples are given,
    # I will focus on the most likely behavior: an object of color 2 moves in directions.
    # The action integers correspond to directional movements.
    # Action 1: Up, 2: Left, 3: Right, 4: Down (or variations).
    # Based on ACTION4 (down/right) and ACTION2 (up/left), let's map them.
    
    # However, looking at the deltas, the changes are quite large.
    # It looks like "painting" or "filling" areas as the object moves.
    # Let's try to find the object of color 2 and move it.
    
    # Find all cells of color 2.
    player_cells = np.argwhere(grid == 2)
    if player_cells.size == 0:
        return grid

    # Define direction vectors for actions.
    # Action 1: Up, 2: Left, 3: Right, 4: Down.
    # In this specific game, they might be diagonal.
    # Action 4: Down-Right (+1, +1)
    # Action 2: Up-Left (-1, -1)
    # Action 5: Down-Left (+1, -1)
    # Action 3: Up-Right (-1, +1)
    
    directions = {
        1: (-1, 0), # Up
        2: (-1, -1), # Up-Left (based on ACTION2 delta)
        3: (1, 1),   # Down-Right (based on ACTION4 delta)
        4: (1, 1),   # Down-Right (ACTION4 is clearly moving things down/right)
        5: (1, 0),   # Down
        6: (0, 0),   # Click
        7: (0, 0)
    }
    
    dy, dx = directions.get(action, (0, 0))
    
    # Move the player object and potentially "clear" or "paint" cells.
    # The deltas show that as color 2 moves, it leaves a trail of color 15 or changes colors.
    # Let's implement a simple shift for all color 2 cells.
    
    new_grid = grid.copy()
    
    # To avoid overwriting while moving, we find current positions first.
    current_player_cells = np.argwhere(grid == 2)
    for r, c in current_player_cells:
        nr, nc = r + dy, c + dx
        if 0 <= nr < new_grid.shape[0] and 0 <= nc < new_grid.shape[1]:
            # In some transitions, the old cell becomes color 5 (background).
            new_grid[r, c] = 5
            new_grid[nr, nc] = 2
        else:
            # If it hits a boundary, it might stay or stop.
            pass

    return new_grid

def is_level_complete(grid):
    # Level complete usually means certain objects are gone or in specific places.
    # Based on common ARC patterns, maybe when color 2 reaches a target.
    # For now, return False as no win state was provided.
    return False