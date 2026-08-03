import numpy as np

import numpy as np

def engine(grid, action, data):
    # The game seems to be a puzzle where the same pattern of blocks is repeated or shifted.
    # Based on the observed transitions, ACTION1-ACTION4 are directional movements (Up, Down, Left, Right) 
    # and ACTION6 is clicking.
    # Action mapping based on common ARC patterns: 1=Up, 2=Down, 3=Left, 4=Right.
    
    # Identify the "player" object - usually a small distinct group of cells.
    # In this case, we look for the color that moves.
    # Let's assume the player is defined by any cell not equal to background (color 5 or 4).
    # Find all coordinates of non-background cells.
    coords = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] not in [4, 5]:
                coords.append((r, c))
            elif grid[r, c] == 4:
                # We treat 4 as part of the environment/wall.
                pass
    
    # This is a a simple movement simulation.
    # Since the exact rules aren't fully clear from the limited data, 
    # we will implement a basic shift of the same pattern of blocks.
    
    # The a<|channel>thought
    # Based on the transitions:
    # ACTION1 shifts the block pattern Downwards? No, let's check.
    # r18c11... becomes r23c11... and soes on. That looks like it shifted DOWN.
    # ACTION2 shifts the block pattern further DOWN.
    # ACTION3 shifts something LEFT?
    # ACTION4 shifts something RIGHT?
    
    # Let's refine action mapping based on observed deltas:
    # ACTION1: (r+5, c) - Shifted down by 6 rows approx.
    # ACTION2: (r+6, c) - Shifted down by 6 rows approx.
    # ACTION3: (r, c-something) - Changed cells in columns 36-42.
    # ACTION4: (r, c+something) - Small changes in cols 23, 29, 35, 41.
    
    # This is a very specific puzzle. Given the constraints and the lack of clear "player" object,
    # we will implement a simple translation of all non-background pixels.
    
    # The background colors are 4 and 5.
    # We assume any cell with color not in [4, 5] is part of the moving entity.
    # move_map = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    # But wait, the data shows Action 1 shifted things DOWN.
    # Let's try to map them as: 1=Down, 2=Down(more), 3=Left, 4=Right.
    
    # Actually, looking at the deltas again:
    # ACTION1: r18 -> r23 (down 5)
    # ACTION2: r24 -> r30 (down 6)
    # ACTION3: columns change from 42 to 36 (left 6)
    # ACTION4: columns change from 23 to 29 etc (right 6)
    
    # So: 1=Down, 2=Down, 3=Left, 4=Right.
    
    # To simulate this perfectly without knowing the "player" coordinates, 
    # we can just shift all non-background pixels by a fixed amount.
    
    # However, the most robust way in these ARC games is usually identifying the object and moving it.
    
    # The background colors are 4 and 5.
    # We define an object as any connected component of cells not equal to 4 or 5.
    
    # For simplicity, let's implement the shifts observed.
    new_grid = grid.copy()
    
    # Define movement vectors based on observations
    # Action 1 & 2 move things down. Action 3 moves left. Action 4 moves right.
    move_vec = {1: (6, 0), 2: (6, 0), 3: (0, -6), 4: (0, 6)}
    
    if action in move_vec:
        dr, dc = move_vec[action]
        
        # Find all current positions of the "object"
        obj_coords = []
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] not in [4, 5]:
                    obj_coords.append((r, c))
        
        # Clear old positions
        for r, c in obj_coords:
            new_grid[r, c] = 5 if r < 42 else 4 # Restore background
            
        # Move to new positions
        for r, c in obj_coords:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                new_grid[nr, nc] = grid[r, c]
                
    return new_grid

def is_level_complete(grid):
    # Win state usually involves a specific pattern or object reaching a goal.
    # Without a win state example, we check for common completion patterns (e.g., no objects left).
    # Or just return False as it's hard to induce without data.
    return False

import numpy as np

def is_level_complete(grid):
    grid = np.array(grid)
    # The win condition for sk48 is that all cells in the grid are the same color
    # (excluding the background color 0).
    # We check if there is only one unique non-zero color present in the grid.
    non_zero_colors = np.unique(grid[grid != 0])
    if len(non_zero_colors) == 1:
        return True
    return False
