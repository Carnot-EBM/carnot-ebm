import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The game seems to be a puzzle where objects are moved or modified by actions.
    # Based on the observed transitions, it's difficult to determine exact rules.
    # However, we can actually see that row 63 has some values changing.
    # 
    # Let's look at the<|channel>thought process for moreing own logic based on the provided deltas.
    # Since the specific movement and other rules aren't fully clear from only a few frames,
    # I will implement a basic state machine if possible or try to find common patterns.
    # In this case, the ACTION4 moves something horizontally.
    # 
    # Looking at the deltas:
    # ACTION4 shifts cells in rows 32-35 columns 32, 36, 40, 44.
    # It looks like it's moving a 4x4 block of color 14/0.
    # 
    # Given the constraints and limited data, I'll provide a general structure.
    
    new_grid = grid.copy()
    
    if action == 4: # Right shift?
        # Observed: r32c32 -> r32c36 -> r32c40 -> r32c44
        # This is likely a player character or object move.
        # Find the current position of the 'object' (color 0)
        # Let's assume the object is the one with value 0.
        # The delta shows that where the object was, it becomes 1, and where it goes, it becomes 0.
        obj_pos = np.where(grid == 0)
        if len(obj_pos[0]) > 0:
            # Simplified logic for Action 4 based on observed transitions:
            # Move the 0-block right by 4 units if possible.
            for r in range(len(obj_pos[0])):
                new_grid[obj_pos[0][r], obj_pos[1][r]] = 1
                new_grid[obj_pos[0][r], obj_pos[1][r] + 4] = 0
    elif action == 1: # Up?
        # Observed: r28c48 -> r24c48
        # Shift block of color 14/0 up.
        obj_pos = np.where(grid == 0)
        if len(obj_pos[0]) > 0:
            for r in range(len(obj_pos[0])):
                new_grid[obj_pos[0][r], obj_pos[1][r]] = 1
                new_grid[obj_pos[0][r]-4, obj_pos[1][r]] = 0
    elif action == 3: # Down?
        # Actually ACTION3 shifted something from row 24 to 27 area.
        # Let's try a simple shift logic.
        pass
    elif action == 6: # Click
        # Handle click data if needed.
        pass
        
    return new_grid

def is_level_complete(grid):
    # The win state isn't explicitly provided but usually involves collecting items or reaching a goal.
    # Row 63 has some changes (e.g., r63c56:4x1).
    # This might be a progress bar or count.
    # Check if the object reached a certain position or color was changed.
    # return True if grid[63, 63] == 4 else False
    return np.any(grid[63, :] == 4) and np.sum(grid[63, :] == 4) >= 7