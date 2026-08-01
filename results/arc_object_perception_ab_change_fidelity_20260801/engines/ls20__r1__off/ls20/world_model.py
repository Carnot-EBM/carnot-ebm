import numpy as np

import numpy as np

def engine(grid, action, data):
    # grid is 64x64. Colors are [0, 1, 3, 4, 5, 8, 9, 11, 12]
    # Action mapping based on observed transitions:
    # ACTION1: Up, ACTION2: Down, ACTION3: Left, ACTION4: Right, ACTION6: Click
    # The game seems to be a puzzle where some objects move in blocks of 5 columns wide.
    # The object being moved is the set of colors {12, 9} (or similar) and it's associated with 
    # a specific region/block.
    # The same time-t movement happens to a "cursor" or "indicator own" at rows 61, 62.
    # Let's analyze the movements from the delta changes.
    # In ACTION3 (Left), cells at c19-23 change to something else, and cursor moves left.
    # In ACTION1 (Up), the block shifts up by 5 rows.
    # In ACTION4 (Right), the block shifts right by 5 cols.
    #
    # Based on the<|channel>thought process, we identify that there is a 'player' block (colors 9, 12)
    # moving within a boundary defined by color 3.
    # The indicator at r61, 62 is just reflecting the x-coordinate of the player block.
    
    new_grid = grid.copy()
    
    # Identify the current position of the player block
    # We search for the presence of color 12 or 9 in the main area.
    # Find all coordinates where value is 12 or 9.
    coords = np.argwhere((grid == 12) | (grid == 9))
    if coords.size == 0:
        return new_grid

    # The player block is typically 5 columns wide.
    # Let's find the top-left corner of this block.
    min_r, min_c = coords.min(axis=0)
    max_r, max_c = coords.max(axis=0)
    
    # Movement logic
    dr, dc = 0, 0
    if action == 1: # Up
        dr = -5
    elif action == 2: # Down
        dr = 5
    elif action == 3: # Left
        dc = -5
    elif action == 4: # Right
        dc = 5
    
    if dr != 0 or dc != 0:
        # Define the bounding box of the moving object
        # It seems to be a specific set of cells that move together.
        # We need to identify which cells are part of the "moving entity".
        # Moving entity consists of colors {9, 12} and potentially others in its local area.
        # Find all coordinates where value is 12 or 9.
        entity_coords = np.argwhere((grid == 12) | (grid == 9))
        
        # To avoid shifting everything, we only shift the identified block.
        # The player block is usually located between rows 8-60 and cols 4-60.
        # We find the current top-left corner of the block.
        block_min_r, block_min_c = coords.min(axis=0)
        
        # Create a mask for the entity
        mask = np.zeros_like(grid, dtype=bool)
        # Based on observed deltas, the entity is roughly 5x5 or similar.
        # The entity is defined by values 9 or 12.
        mask[(grid == 12) | (grid == 9)] = True
        
        # Shift the entity
        for r, c in zip(*np.where(mask)):
            new_grid[r + dr, c + dc] = grid[r, c] if 0 <= r+dr < 64 and 0 <= c+dc < 64 else 4
            # Clear old position
            if not ((r + dr == r) and (c + dc == c)):
                new_grid[r, c] = 4 # Restore to background color 4
        
        # Special case: cursor at rows 61, 62 moves with x coordinate
        cursor_coords = np.argwhere((grid == 3))
        # We only care about cursors in rows 61, 62.
        # Find current cursor column
        cursor_cols = []
        for r in [61, 62]:
            row_vals = grid[r, :]
            col = np.where(row_vals == 3)[0]
            if col.size > 0:
                cursor_cols.append(col[0])
        
        if len(cursor_cols) >= 1:
            curr_cursor_c = cursor_cols[0]
            new_cursor_c = curr_cursor_c + dc
            # Update cursor positions
            for r in [61, 62]:
                # Clear row's existing cursor
                new_grid[r, np.where(grid[r, :] == 3)[0]] = 11
                # Place new cursor
                if 0 <= new_cursor_c < 64:
                    new_grid[r, new_cursor_c] = 3
    
    return new_grid

def is_level_complete(grid):
    # Level complete usually means the player block reached a target or collected something.
    # In this specific game, we don't have a win state provided, but typically it's when 
    # the entity reaches a certain coordinate or color.
    # For now, return False as no win condition was observed.
    return False

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state.
    The win condition is that all cells of the same color (excluding background 0)
    must be connected (4-connectivity) and form a single contiguous block.
    """
    grid = np.array(grid)
    colors = np.unique(grid)
    colors = colors[colors != 0]
    
    if len(colors) == 0:
        return False
        
    for color in colors:
        # Find all cells of this color
        cells = np.argwhere(grid == color)
        if len(cells) == 0:
            return False
        
        # Check connectivity
        start_node = cells[0]
        visited = set()
        stack = [start_node]
        
        while stack:
            curr = stack.pop()
            curr_tuple = tuple(curr)
            if curr_tuple not in visited:
                visited.add(curr_tuple)
                # Find neighbors of the same color
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = curr[0] + dx, curr[1] + dy
                    if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                        if grid[nx, ny] == color:
                            stack.append([nx, ny])
        
        if len(visited) != len(cells):
            return False
            
    return True
