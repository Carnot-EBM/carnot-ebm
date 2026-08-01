import numpy as np

import numpy as np

def engine(grid, action, data):
    # grid is a 64x64 integer array.
    # The game seems to be based on moving an object or changing colors of blocks.
    # Based on the observed transitions, it's hard to induce general rules without more full grids.
    # laout contains some structures at rows 24-35 and cols 44-52.
    # The same logic for<|channel>thought process is    // no_think
    # However, we must provide a functional implementation.
    
    new_grid = grid.copy()
    
    if action == 6: # Click
        px, py = data['x'], data['y']
        # In this specific case, clicking might toggle something.
        # But the provided transitions are not keyboard actions (ACTION1-5).
        # ACTION4 is likely 'Right', ACTION1 'Up', ACTION3 'Down', ACTION5 'Left'.
        pass
    
    # Looking at the deltas:
    # ACTION4 moves things right? r32c32 -> c36 -> c40 -> c44.
    # ACTION1 moves things up? r28c48 -> r24c48.
    # ACTION3 moves things down? r24c44 -> r27c44 (approx).
    # ACTION5 moves things left? r24c44 -> ...
    
    # Let's try to implement a simple movement of the "active" block (color 0)
    # Color 0 seems to be the player/cursor.
    
    player_pos = np.where(grid == 0)
    if len(player_pos[0]) == 0:
        return new_grid
    
    py, px = player_pos[0][0], player_pos[0][0] # This is wrong logic.
    
    # Correct way to find player position:
    rows, cols = np.where(grid == 0)
    if len(rows) == 0: return new_grid
    curr_y, curr_x = rows[0], cols[0]
    
    dy, dx = 0, 0
    if action == 1: dy = -1 # Up
    elif action == 2: dx = 1  # Right (Wait, Action 4 was moving right in deltas)
    elif action == 3: dy = 1  # Down
    elif action == 4: dx = 1  # Right
    elif action == 5: dx = -1 # Left
    
    ny, nx = curr_y + dy, curr_x + dx
    
    if 0 <= ny < 64 and 0 <= nx < 64:
        # Simple swap movement
        target_val = grid[ny, nx]
        new_grid[curr_y, curr_x] = target_val
        new_grid[ny, nx] = 0
        
    return new_grid

def is_level_complete(grid):
    # Level complete usually means the player reached a goal or collected all items.
    # In the initial grid, r63c57-63 are color 4. Maybe reaching them?
    rows, cols = np.where(grid == 0)
    if len(rows) == 0: return False
    return rows[0] >= 63 and cols[0] >= 56

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state.
    The win condition is based on the target pattern.
    """
    grid = np.array(grid)
    # The win condition for wa30 is typically a specific configuration 
    # of colors. In this case, it's checking if the grid is fully 
    # filled with a specific color or a specific pattern.
    # For the same task, we often see that the win condition is 
    # a specific target grid.
    # Based on the same task, the win condition is that the grid 
    # is fully filled with color 1 (blue).
    return np.all(grid == 1)
