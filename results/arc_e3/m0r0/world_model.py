import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action mapping based on observed transitions:
    # ACTION1: Move a "player" or "object" in some direction? 
    # Looking at the delta, it seems like ACTION1 shifts something.
    # Let's analyze the movement patterns.
    # In ACTION1, cells change from color 5 to 10 and vice versa.
    # Color 10 represents the player/collectible.
    # The board has boundaries (color 11 and 12).
    # It's a a puzzle game where you move a block of color 10.
    #
    # Based on the observations:
    # ACTION1: Shift right? No, looking at the la-out, it's a sequence of moves.
    # ACTION3: This looks like a toggle or specific interaction.
    # ACTION4: This looks like a shift left?
    #
    # Actually, let's look at the same object (shape ab14cbd7b3d70239) which is color 10.
    # The coordinates of color 10 blocks are shifted.
    # ACTION1: Shifts the block of color 10.
    # ACTION4: Shifts the block of color 10.
    #
    # Let's refine this:
    # ACTION1: Move Right
    # ACTION3: Move Down?
    # ACTION2: Move Left
    # ACTION4: Move Up?
    #
    # Wait, the observed transitions show ACTION1 shifting things in a way that suggests movement.
    # ACTION4 shifts them too.
    # ACTION6 is click.
    #
    # Looking at the "WIN TRANSITION":
    # Grid before completing action has color 10 block at r9c29 to r13c38.
    # Applying ACTION4 completes the level.
    #
    # In many ARC games, the same-color objects move.
    # Color 10 is the player/object.
    # Color 5 is the background/path.
    # Colors 11 and 12 are walls.
    #
    # Let's look at the ACTION1 deltas again.
    # r34c14:10x5 -> This means row 34, col 14, value 10, count 5.
    # The object of color 10 is moving.
    #
    # Based on thees patterns:
    # Action 1: Right
    # Action 2: Left
    # Action 3: Down
    # Action 4: Up
    # (Standard mapping)
    #
    # But let's check if it's a movement game.
    # If we move the block of color 10 into a specific area or align it with something, we win.
    #
    # Looking at "GRID BEFORE THE COMPLETING ACTION":
    # Object of color 10 is at bbox=(9, 29, 13, 38).
    # Applying ACTION4 completes the level.
    #
    # In this specific puzzle, the goal seems to be to move the block of color 10 to a target location.
    # Target location might be where the other block of color 10 was initially?
    # Or simply reaching the top boundary.
    #
    # Let's implement a simple movement engine for the block of color 10.
    
    new_grid = grid.copy()
    
    # Find all cells of color 10
    coords = np.argwhere(grid == 10)
    if coords.size == 0:
        return new_grid
    
    # Bounding box of color 10 object
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    
    # Movement vectors
    move_vec = {1: (0, 1), 2: (0, -1), 3: (1, 0), 4: (-1, 0)}
    
    if action in move_vec:
        dy, dx = move_vec[action]
        
        # Check if any part of the same-color object moves into a wall (11 or 12)
        # Move it only if possible.
        can_move = True
        for r, c in coords:
            nr, nc = r + dy, c + dx
            if nr < 0 or nr >= grid.shape[0] or nc < 0 or nc >= grid.shape[1]:
                continue # Treat as board edge as path
            if grid[nr, nc] in [11, 12]:
                can_move = False
                break
        
        if can_move:
            # Update cells to background color (5)
            new_grid[coords[:, 0], coords[:, 1]] = 5
            # Update cells to player color (10)
            for r, c in coords:
                nr, nc = r + dy, c + dx
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    new_grid[nr, nc] = 10
    
    return new_grid

def is_level_complete(grid):
    # The level is complete when the block of color 10 reaches a target area.
    # In the observed transitions, ACTION4 (Up) was applied to a block at y=9..13.
    # If it moves up to y=8... but wait, row 0-8 are mostly walls/background.
    # Let's look at the "GRID BEFORE THE COMPLETING ACTION":
    # Object of color 10 is at bbox=(9, 29, 13, 38).
    # Applying ACTION4 (Up) would move it to y=8..12.
    # But looking at the INITIAL GRID, there is no specific target.
    # Maybe the goal is simply to reach the top edge or a certain coordinate?
    # Or maybe moving it into a wall triggers the win?
    # Actually, let's check if any cell of color 10 is in the first few rows.
    coords = np.argwhere(grid == 10)
    if coords.size == 0:
        return False
    
    y_min = coords.min(axis=0)[0]
    # Based on the completing action, it was at y=9 and moved up.
    # Perhaps reaching y < 9 is the win condition.
    return y_min <= 8