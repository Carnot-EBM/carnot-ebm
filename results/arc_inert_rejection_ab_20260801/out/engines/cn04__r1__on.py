import numpy as np

import numpy as np

def engine(grid, action, data):
    """
    The game involves a character (color 0) moving through a maze.
    Action 4 is 'Right'.
    Based on the observed transitions, Action 4 shifts the pattern of color 0 cells to the right by 1 column.
    It also seems to update some other colors in the same relative positions.
    Looking at thes deltas, it's a simple horizontal shift of the entire grid content that is not background (color 10).
    Wait, looking closer at the same-colored blocks shifting together.
    Actually, the most consistent rule for ACTION 4 is:
    Shift all non-background pixels (non-10) to the right by 1 pixel, wrapping around or being clipped.
    However, only specific areas are affected.
    Let's refine: The characters/objects (colors 0, 8, 14) are shifted right.
    """
    new_grid = grid.copy()
    if action == 4: # Right
        # Find all non-background cells
        # We want to move everything that isn't 10.
        # Mask for non-background
        mask = (grid != 10)
        
        # Shift mask and values
        # Create a temporary grid to store the new positions
        temp_grid = np.full(grid.shape, 10)
        
        # For each cell that is not 10, move it to (r, c+1)
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1] - 1):
                if grid[r, c] != 10:
                    temp_grid[r, c + 1] = grid[r, c]
                    
        # Handle wrap-around or clipping? 
        # In the observed data, color 0 at r0c16 shifts to r0c17, then r0c18...
        # The deltas show r0c16 becomes 0x1 (which is background 10), and r0c17 becomes 0x1.
        # Wait, the delta says "r0c16:0x1". This means the value at r0c16 becomes 0.
        # Let's re-read the run-length encoding of deltas.
        # "changed cells ... r0c16:0x1" -> row 0, col 16 becomes value 0.
        # But in INITIAL GRID, r0 was 10x16, 4x32, 10x16. So r0c16 is color 4.
        # If ACTION 4 makes it 0, that's a change.
        # Then next ACTION 4 makes r0c17 become 0.
        # Looking at the pattern: Action 4 moves everything right by 1.
        # Background is 10.
        
        # Correct logic for shifting non-background pixels:
        # Shift all values to the right by 1 column.
        # Fill new vacancies with background color 10.
        # The observed data shows specific blocks moving.
        # Let's implement a simple shift.
        
        shifted_grid = np.full(grid.shape, 10)
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1] - 1):
                if grid[r, c] != 10:
                    shifted_grid[r, c + 1] = grid[r, c]
        
        # Now we need to handle the "holes" (color 0).
        # In this game, color 0 often represents the player or an empty space.
        # The deltas show that when something shifts right, the cell it left becomes 10?
        # No, look at "r0c16:0x1". That means value 0.
        # Wait, if r0c16 was 4 and now it's 0... then 0 is the "player" or "empty".
        # Let's try shifting everything including 0.
        
        final_grid = np.full(grid.shape, 10)
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1] - 1):
                final_grid[r, c+1] = grid[r, c]
        
        # Column 0 remains background 10.
        # Col 63 is clipped.
        return final_grid

    elif action == 2: # Left
        final_grid = np.full(grid.shape, 10)
        for r in range(grid.shape[0]):
            for c in range(1, grid.shape[1]):
                final_grid[r, c-1] = grid[r, c]
        return final_grid
    
    elif action == 1: # Up
        final_grid = np.full(grid.shape, 10)
        for r in range(1, grid.shape[0]):
            for c in range(grid.shape[1]):
                final_grid[r-1, c] = grid[r, c]
        return final_grid
        
    elif action == 3: # Down
        final_grid = np.full(grid.shape, 10)
        for r in range(1, grid.shape[0]):
            for c in range(grid.shape[1]):
                final_grid[r+1, c] = grid[r, c] # This will be out of bounds for the last row.
        # Correcting loop
        final_grid = np.full(grid.shape, 10)
        for r in range(grid.shape[0]-1):
            for c in range(grid.shape[1]):
                final_grid[r+1, c] = grid[r, c]
        return final_grid

    return grid

def is_level_complete(grid):
    # Level complete when a certain condition is met.
    # In many ARC games, it's about reaching a goal or clearing blocks.
    # Since no win state was provided, we assume a simple check.
    # Often, color 0 (player) reaches a target area.
    # Let's return False unless we can detect a specific pattern.
    return False

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state.
    The win condition is that all cells of the same color (excluding background 0)
    must be connected (4-connectivity) and each color must form a single contiguous block.
    """
    grid = np.array(grid)
    unique_colors = np.unique(grid)
    unique_colors = unique_colors[unique_colors != 0]
    
    if len(unique_colors) == 0:
        return False
        
    for color in unique_colors:
        # Find all coordinates of the current color
        coords = np.argwhere(grid == color)
        if len(coords) == 0:
            return False
            
        # Start BFS to find all reachable cells of the same color
        start_node = tuple(coords[0])
        visited = {start_node}
        queue = [start_node]
        
        while queue:
            curr = queue.pop(0)
            r, c = tuple(curr)
            # Check 4-neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = (r + dr, c + dc)
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    if grid[nr, nc] == color and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        
        # If the number of visited cells equals the total number of cells of that color,
        # the color forms a single contiguous block.
        if len(visited) != len(coords):
            return False
            
    return True
