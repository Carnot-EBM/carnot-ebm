import numpy as np

import numpy as np

def engine(grid, action, data):
    # Induce rules based on the same transitions provided.
    # The game seems to involve moving blocks of colors (specifically color 14)
    # and potentially interacting with other cells.
    # The same ACTION4, ACTION3, ACTION2 etc. result in different changes.
    # This means the state depends on some internal cursor/position.
    # Find current position of something unique or identify patterns.
    
    # Let's assume there is a 'cursor' or 'player' represented by color 0.
    # Color 0 appears in r31c21, r34c21, etc.
    # In the initial grid, color 0 only exists at r31c21 (approx).
    # Let's look at the<|channel>thought
    # Initial Grid: r31: ... 0x1 ...
    # Action 4 (Right?): Changes occur around c18-c26.
    # Action 3 (Left?): Changes occur around c15-c21.
    # Action 2 (Up?): Changes occur around r30-r32 vs r33-r35.
    # Action 3 again: Shifts things left.
    
    # Based on the transitions:
    # ACTION 2: Up
    # ACTION 3: Left
    # ACTION 4: Right
    # ACTION 1: Down (implied)
    # ACTION 5: ?
    # ACTION 6: Click
    # ACTION 7: ?
    
    # Find current position of color 0.
    pos = np.where(grid == 0)
    if len(pos[0]) == 0:
        return grid
    
    y, x = pos[0][0], pos[1][0]
    
    # Define movement vectors for actions
    moves = {
        1: (1, 0),  # Down
        2: (-1, 0), # Up
        3: (0, -1), # Left
        4: (0, 1),  # Right
    }
    
    if action in moves:
        dy, dx = moves[action]
        ny, nx = y + dy, x + dx
        
        # Simple rule: move the 'player' (color 0) and swap with target cell.
        # The observed deltas are more complex blocks of colors changing.
        # It looks like a block of size 3x3 or similar is moving.
        # Let's try to implement a simple shift of the player and a local change.
        
        new_grid = grid.copy()
        
        # In this specific game, it seems ACTION 4 shifts something right, 
        # ACTION 3 shifts left, ACTION 2 shifts up.
        # This matches standard directional keys.
        
        # We need to simulate the "block" movement seen in the deltas.
        # The deltas show changes in 3-row spans.
        # The color 14 is often involved.
        # Color 15 is background/wall?
        # Color 2 is outer boundary.
        
        # Let's assume the "entity" is a 3x3 block centered at (y, x).
        # Move the entity by shifting its contents.
        
        # Identify the block center (the 0 cell)
        center_y, center_x = y, x
        
        # Define the block boundaries
        by1, by2 = center_y - 1, center_y + 2
        bx1, bx2 = center_x - 1, center_x + 2
        
        # To move: we shift the values of the 3x3 area and update the 'player' position.
        # But looking at the deltas, it's more like swapping blocks or painting.
        # Given the constraints, let's implement a simple shift for the player 0.
        
        new_grid[ny, nx] = 0
        new_grid[y, x] = grid[ny, nx] if 0 <= ny < 64 and 0 <= nx < 64 else grid[y, x]
        
        return new_grid

    return grid

def is_level_complete(grid):
    # Level complete usually means color 0 reaches a target or disappears.
    # In many ARC games, it's about filling a region or reaching a goal.
    # Since no win state was provided, we guess based on common patterns.
    # Let's assume it's complete when color 0 is at some specific location (e.g., bottom right).
    pos = np.where(grid == 0)
    if len(pos[0]) == 0:
        return False
    return pos[0][0] >= 60 and pos[1][0] >= 60

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state.
    The win condition is that all cells are the same color (uniform grid).
    """
    grid = np.array(grid)
    if grid.size == 0:
        return False
    return np.all(grid == grid[0, 0])
