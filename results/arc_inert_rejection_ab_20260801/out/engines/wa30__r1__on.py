import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Based on observed transitions, this looks like a puzzle game where blocks are moved or toggled.
    # The provided observations are very sparse and limited.
    # However, we can induce that Action 4 moves something to the right.
    # Action 1 moves something to the left/up? No, let's look at the coordinates.
    #
    # Let's analyze the same region of<|channel>thought
    # Initial Grid:
    # r24-r27: c44-c47 (color 4), c48-c49 (color 9) - wait, no.
    # r28-r31: c28-c35 (color 4), c36-c39 (color 9)
    # r32-r35: c32-c34 (color 14), c35 (color 0)
    #
    # Looking at ACTION 4:
    # It changes cells in rows 32-35, columns 32, 36, 40, 44... shifting color 14 block.
    # This suggests Action 4 is 'Right'.
    #
    # Looking at ACTION 1:
    # It shifts blocks vertically or horizontally.
    #
    # Given the constraints and the lack of clear patterns, we will implement a basic movement system.
    # The most likely scenario for ARC games like this is that the player controls an object (the 0-cell).
    #
    # Let's identify the "player" as the cell with value 0.
    # Find all zeros.
    # In initial grid: r32-35, c35 is 0.
    # After first ACTION 4: r32-35, c36 becomes 0? No, delta says r32c32:1x4, 14x3, 0x1.
    # That means col 32=1, 33=1, 34=1, 35=1, 36=14, 37=14, 38=14, 39=0.
    # So the 0-block shifted from c35 to c39.
    # Action 4 = Right.
    # Action 1 = Up.
    # Action 3 = Down.
    # Action 2 = Left.
    #
    # Let's refine:
    # Action 4: Shift 0-block right by 4 units.
    # Action 1: Shift 0-block up.
    # Action 3: Shift 0-block down.
    # Action 2: Shift 0-block left.
    #
    # The "player" is a block of zeros (size 4x1).
    # Find the top-left corner of the zero-block.
    #
    # In initial grid: r32, c35.
    # After ACTION 4: r32, c39.
    # After another ACTION 4: r32, c43? No, delta says r32c40... so it moves in steps of 4.
    #
    # Let's implement this movement logic.

    new_grid = grid.copy()
    
    # Identify player position (top-left of the 4x1 zero block)
    zeros = np.argwhere(grid == 0)
    if len(zeros) == 0:
        return new_grid
    
    # Assume the player is the first group of 4 vertical zeros
    py, px = zeros[0]
    
    # Movement vectors
    movements = {
        1: (-4, 0), # Up
        2: (0, -4), # Left
        3: (4, 0),  # Down
        4: (0, 4),  # Right
    }
    
    if action in movements:
        dy, dx = movements[action]
        ny, nx = py + dy, px + dx
        
        # Check boundaries
        if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
            # Move the 4x1 zero block
            for i in range(4):
                if 0 <= py + i < grid.shape[0]:
                    new_grid[py+i, px] = 1 # Fill old position with background color 1
                if 0 <= ny + i < grid.shape[0]:
                    new_grid[ny+i, nx] = 0 # Set new position to 0
                    
    return new_grid

def is_level_complete(grid):
    # Level complete if player reaches a target area.
    # In observed transitions, r63c56 becomes 4 when ACTION 4 is used multiple times.
    # This suggests a goal at the bottom right.
    # Let's check for any cell of value 4 in the same row as the goal (r63).
    return np.any(grid[63, :] == 4)