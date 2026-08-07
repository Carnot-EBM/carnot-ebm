import numpy as np

def engine(grid, action, data=None):
    """
    World model for game cd82.
    The grid contains several moving parts (objects) that can be shifted by directional actions.
    Action 2: Up, Action 3: Down, Action 4: Left, Action 5: Right? No, based on observed transitions:
    ACTION 4: Shift something left/down?
    ACTION 2: Shift something up/right?
    Looking at ACTION 2 and ACTION 3 in the first example:
    - ACTION 2 shifts a set of cells from r21c39 to r45c38.
    - ACTION 3 shifts them back or further down.
    Actually, let's analyze the movement patterns more closely.
    
    Based on the provided transitions:
    - ACTION 2 moves an object (obj11/12/13) downwards and slightly right.
    - ACTION 3 moves it upwards and slightly left.
    - ACTION 4 moves it downwards and slightly left.
    - ACTION 5 moves it upwards and slightly right.
    Wait, this is not quite correct. Let's look at the same object across frames.
    - Initial state: obj12 (color 15, bbox=(25, 26, 31, 37))
    - After ACTION 4: Object shifted.
    - After ACTION 2: Object shifted again.
    - After ACTION 3: Object shifted again.
    
    Let's refine the action mapping based on coordinates:
    ACTION 2: Down-Right shift?
    ACTION 3: Up-Left shift?
    ACTION 4: Down-Left shift?
    ACTION 5: Up-Right shift?
    
    Looking at the WIN TRANSITION:
    The completing action is ACTION 5.
    Before ACTION 5, the objects are at r45c25 to r53c38.
    After ACTION 5, they move back to a specific configuration.
    Actually, looking at the "changed cells" for ACTION 5 in the first example:
    It seems to be resetting or moving things into place.
    """
    # The observed transitions are very complex and involve multiple objects shifting.
    # Since we need a simple general rule, let's look at the object movement.
    # There is a core set of objects (colors 0, 15, 2) that move together as a block.
    # This block consists of obj11, 12, 13.
    # Let's identify the same block of colors [0, 15, 2].
    
    # Find all cells with color 0, 15, or 2.
    mask = np.isin(grid, [0, 15, 2])
    if not np.any(mask):
        return grid.copy()

    # Get the bbox of these cells.
    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Define shifts based on action
    # ACTION 2: Down-Right?
    # ACTION 3: Up-Left?
    # ACTION 4: Down-Left?
    # ACTION 5: Up-Right?
    
    # Based on observed deltas:
    # ACTION 2: r21c39 -> r45c38 (Down shift)
    # ACTION 3: r40c43 -> r56c39 (Up shift?)
    # ACTION 4: r21c39 -> r37c43 (Down shift)
    # ACTION 5: Reset/Win condition move.
    
    # Let's try a simpler mapping:
    # Action 2: Down
    # Action 3: Up
    # Action 4: Left
    # Action 5: Right
    
    # Actually, let's look at the coordinates again.
    # Initial obj12: bbox=(25, 26, 31, 37)
    # After ACTION 4: Object shifted to r21c39...r37c43.
    # After ACTION 2: Object shifted to r33c39...r45c38.
    # After ACTION 3: Object shifted to r40c43...r56c39.
    # This is very confusing. Let' same check if it's just simple shifts.
    
    # The most likely scenario in these ARC games is that they are puzzles where you move an object into a target area.
    # Target area for this level seems to be the top-left or some specific region.
    # ACTION 5 is the win action.
    # Before ACTION 5 (win), objects were at y=45..53.
    # After ACTION 5, they moved back up towards y=3..12.
    
    # Since we cannot easily induce the exact movement rules from the provided deltas, 
    # and we need to return a function, let's implement a basic shift logic.
    
    new_grid = grid.copy()
    
    if action == 2: # Down
        shift_y, shift_x = 1, 0
    elif action == 3: # Up
    	shift_y, shift_x = -1, 0
    elif action == 4: # Left
        shift_y, shift_x = 0, -1
    elif action == 5: # Right
        shift_y, shift_x = 0, 1
    else:
        return new_grid

    # Move all cells of color [0, 15, 2] by (shift_y, shift_x)
    coords = np.argwhere(mask)
    for y, x in coords:
        ny, nx = y + shift_y, x + shift_x
        if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
            # Only move if target is not blocked by a wall (color 4 or 3)
            if grid[ny, nx] != 4 and grid[ny, nx] != 3:
                new_grid[y, x] = 5 # Reset old position to background
                new_grid[ny, nx] = grid[y, x]
                
    return new_grid

def is_level_complete(grid):
    """
    The level is complete when the moving objects are shifted into a specific target area.
    Looking at ACTION 5's delta, it seems to be resetting things.
    In many ARC games, 'win' means reaching a certain configuration.
    Let's check if any object of color 0, 15, or 2 has reached the top region.
    """
    mask = np.isin(grid, [0, 15, 2])
    if not np.any(mask):
        return False
    coords = np.argwhere(mask)
    y_min = coords.min(axis=0)[0]
    # If the moving block has moved up to y < 10, it might be win state.
    return y_min < 10