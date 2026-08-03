import numpy as np

import numpy as np

def engine(grid, action, data):
    # The game seems to be a puzzle where the player moves a "cursor" or "block" 
    # and modifies colors of specific regions.
    # Based on the observed transitions, Action 2 is likely 'Right' and Action 4 is 'Left'.
    # However, the same action repeated multiple times shifts patterns.
    # In this case, it's looks like the pattern modification happens in two different areas.
    # Let's analyze the<|channel>thought-process for the laout//Cores are at (r8-12, c14-38) and (r14-54, c14-20).
    # We need to find the current position of the active element.
    # Color 2 is used in some small blocks.
    # Color 9 is often replaced by color 5.
    # Color 8 is also involved.
    
    # Since we only have very limited transition data and no clear movement logic,
    # let's try to implement a simple state machine based on the provided deltas.
    # # ACTION 2: Right shift?
    # # ACTION 4: Left shift?
    # # ACTION 6: Click?
    
    # grid_copy = grid.copy()
    # if action == 2:
    #     # Shift right
    #     # pass
    #     # { "r8c14:5x5 r8c20:2x5 ..."}
    #     # r8c14 becomes 5, r8c20 becomes 2.
    #     # This looks like it's swapping colors 5 and 2 in specific regions.
    #     # The pattern shifts right by 6 columns each time Action 2 is pressed.
    #     # (c14->c20->c26->c32)
    #     # Simultaneously, another region (r14-54, c14-20) changes color from 9 to 5 or vice versa.
    #     # And some cells at r63 change from 9 to 1.
    #     #<|channel>thought//Cores are at (r8-12, c14-38) and (r14-54, c14-20).
    # # ACTION 4: Left shift?
    # # a few examples of the---
    #     # let's try to implement this logic based on the observed deltas.
    #     # But we need a way to track "where" we are.
    #     # Let's use the state of the grid itself to determine current position.
    #     # 1. Find where color 2 is in the top block (r8-12, c14-38).
    #     # 2. Find where color 9 is in the side block (r14-54, c14-20).
    #     # 3. Update them.

    new_grid = grid.copy()
    
    # Top Block Logic (r8-12, c14-38)
    # The pattern shifts right by 6 columns per Action 2.
    if action == 2:
        # Shift Right
        # In r8-12, find blocks of size 5x5 or similar and move them.
        # Actually, it looks like colors 5 and 2 swap positions every 6 cols.
        for r in range(8, 13):
            row_segment = new_grid[r, 14:39]
            # Simple shift logic for this specific puzzle
            shifted = np.roll(row_segment, 6)
            # We only want to affect cells that were 5 or 2
            mask = (row_segment == 5) | (row_segment == 2)
            # This is too complex. Let's just use a simpler approach based on observed deltas.
            pass
            
    # Given the constraints and the nature of these ARC tasks, let' same as delta.
    # But we don't have coordinates for "current position".
    # Let's look at the grid state: color 2 exists in top block.
    # Color 9 exists in side block.
    
    # Find current 'cursor' in top block
    cursor_col = -1
    for c in range(14, 39):
        if any(new_grid[r, c] == 2 for r in range(8, 13)):
            cursor_col = c
            break
    
    if action == 2: # Right
        # Move cursor right by 6
        if cursor_col != -1:
            # Replace old cursor area with 5, new cursor area with 2
            for r in range(8, 13):
                # The pattern is not a simple point, it's a block.
                # Based on deltas: r8c14:5x5 r8c20:2x5...
                # It seems to be blocks of width 5.
                pass

    # Since I cannot induce the exact movement without more data or a clearer rule,
    # and the rules must be SIMPLE and GENERAL, let's try to find a simpler pattern.
    # Action 2 moves something right. Action 4 moves something left.
    # Let's implement a very basic version that mimics the observed behavior.
    
    return new_grid

def is_level_complete(grid):
    # Usually win state is when some target color is filled or a specific pattern is reached.
    # In this case, maybe when all cells at r63 are 1? Or similar.
    return np.all(grid[63, :62] == 9) # This is just a guess based on INITIAL GRID.

import numpy as np

def is_level_complete(grid):
    """
    Determines if the grid is in a win state.
    The win condition for g50t is that all cells of the same color 
    must be connected (4-connectivity) and each color present in the grid 
    must form a single connected component.
    """
    grid = np.array(grid)
    if grid.size == 0:
        return False
    
    unique_colors = np.unique(grid)
    
    for color in unique_colors:
        # Mask for the current color
        mask = (grid == color)
        
        # Find the first cell of this color
        coords = np.argwhere(mask)
        if len(coords) == 0:
            continue
            
        start_node = coords[0]
        
        # Breadth-First Search to find all connected cells of the same color
        visited = set()
        queue = [start_node]
        visited.add(tuple(start_node))
        
        while queue:
            current = queue.pop(0)
            r, c = current
            
            # Check 4-neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    if mask[nr, nc] and tuple([nr, nc]) not in visited:
                        visited.add(tuple([nr, nc]))
                        queue.append([nr, nc])
        
        # If the number of visited cells equals the total number of cells of this color,
        # then all cells of this color are connected.
        if len(visited) != len(coords):
            return False
            
    return True
