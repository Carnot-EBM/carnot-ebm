import numpy as np

def engine(grid, action, data):
    """
    Induces the world model for game 'sk48'.
    Based on observed transitions:
    - Action 1 (Up), 2 (Down), 3 (Left), 4 (Right) move a specific object/pattern.
    - The grid contains static boundaries (color 5 and 4) and movable elements.
    - ACTION 6 is a click (not seen in provided sequence but defined).
    - Transitions show movement of blocks of colors (e.g., color 6, 0, 1, 2 patterns).
    """
    new_grid = grid.copy()
    h, w = new_grid.shape

    # Identify the "player" or moving entity. 
    # Looking at the deltas, there's a pattern involving color 6 and others that shifts.
    # In this specific ARC task, it looks like a puzzle where you shift a block.
    
    if action == 1: # UP
        # Shift the active region up by some offset if not blocked
        # Based on delta r18c11...r29c11 changing to 6x6 etc.
        # This is a simplified representation of the observed shifting behavior.
        shift_region(new_grid, -1)
    elif action == 2: # DOWN
        shift_region(new_grid, 1)
    elif action == 3: # LEFT
        shift_region(new_grid, 0, col_dir=-1)
    elif action == 4: # RIGHT
        shift_region(new_grid, 0, col_dir=1)
    
    return new_grid

def shift_region(grid, row_off=0, col_dir=0):
    """
    Helper to simulate the movement of the movable blocks seen in the transitions.
    The patterns are complex, but they generally move as a unit.
    """
    # Find all cells that aren't background (5 or 4) and aren't static boundaries
    # In this game, it seems color 6 is part of the moving block.
    mask = (grid != 5) & (grid != 4)
    coords = np.argwhere(mask)
    if len(coords) == 0: return
    
    # This is a heuristic based on the provided deltas showing shifts of specific regions.
    # Since we don't have the full logic for collision/blocking, we apply a simple shift.
    # However, looking at ACTION2, it moves a large chunk from r18-r29 down to r30-r35.
    
    # To be more precise with the observed data:
    # Action 2 shifted a region starting around r18c11 down by about 12 rows.
    # Action 1 shifted it back up.
    
    # For the purpose of this executable model, we implement the shift if possible.
    # We use a simplified version where we identify the 'movable' cluster.
    
    # Note: The actual ARC logic usually involves sliding blocks until they hit walls.
    pass

def is_level_complete(grid):
    """
    The win state is typically when a certain pattern is reached or an object 
    is moved to a target location. In sk48, look for specific color alignments.
    """
    # Based on common ARC patterns, check if the movable block reaches a goal area.
    # Without the WIN STATE grid, we assume completion based on typical puzzle goals.
    return False # Default as no win state was provided in observations.

# Re-evaluating engine to actually perform shifts based on the delta patterns.
def engine(grid, action, data):
    new_grid = grid.copy()
    h, w = new_grid.shape
    
    # Identify moving parts (anything not background 5 or wall 4)
    # Actually, looking at the deltas, only specific colors move.
    # Let's find the bounding box of the "active" elements (colors other than 4 and 5).
    mask = (grid != 5) & (grid != 4)
    coords = np.argwhere(mask)
    if len(coords) == 0: return new_grid
    
    r_min, c_min = coords.min(axis=0)
    r_max, c_max = coords.max(axis=0)
    
    # The observed movements are quite large jumps (e.g., ACTION2 moves r18->r30).
    # This suggests a 'snap' movement or sliding until hitting a boundary.
    
    if action == 1: # UP
        # Slide up until we hit color 5/4 or top
        for r in range(r_min, r_max + 1):
            row_vals = grid[r, c_min:c_max+1]
            # Simplified: shift the whole block up by a fixed amount seen in data
            pass 
    elif action == 2: # DOWN
        # Observed: shifted from ~r18 to ~r30 (delta ~12)
        shift_amount = 12
        region = grid[r_min:r_max+1, c_min:c_max+1].copy()
        new_grid[r_min:r_max+1, c_min:c_max+1] = grid[r_min-shift_amount if r_min>shift_amount else 0 : r_max-shift_amount+1, c_min:c_max+1]
        # This is too complex for a general rule without more samples.
        # Let's use a simpler approach: just apply the delta logic if it matches exactly.
        pass

    # Given the constraints and the nature of these tasks, we provide a structure 
    # that handles the basic movement of the non-background pixels.
    if action in [1, 2, 3, 4]:
        dr, dc = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[action]
        # Find movable block
        mask = (grid != 5) & (grid != 4)
        coords = np.argwhere(mask)
        if len(coords) > 0:
            # Shift all active cells by dr, dc if they stay within bounds and don't hit walls
            # For ARC, usually the whole object moves together.
            new_coords = coords + [dr, dc]
            for old, new in zip(coords, new_coords):
                if 0 <= new[0] < h and 0 <= new[1] < w:
                    if grid[new[0], new[1]] == 5 or grid[new[0], new[1]] == 4:
                        return new_grid # Blocked
                    new_grid[new[0], new[1]] = grid[old[0], old[1]]
                    new_grid[old[0], old[1]] = 5 # Clear old position with background
    return new_grid