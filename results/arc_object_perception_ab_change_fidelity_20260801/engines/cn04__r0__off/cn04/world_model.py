import numpy as np

import numpy as np

def engine(grid, action, data):
    """
    Predicts the next grid state based on the same rules induced from observed transitions.
    ACTION 4 is a rightward shift of specific patterns in the same row/column range.
    """
    # The pattern seems to be that ACTION 4 shifts certain blocks of colors (specifically color 0)
    # and fills them with color 10.
    # We identify the "active" regions where these changes occur.
    # Based on thes deltas, it's la-rightward movement of 'voids' or 'blocks'.
    
    new_grid = grid.copy()
    
    if action == 4:
        # Identify all cells that are not color 10 (the background)
        # Find columns indices for each cell that is NOT color 10
        # For each row, we find the first column index that is not 10.
        # # This a simplified approach: we actually see the delta shifting a set of coordinates.
        # 
        # The<|channel>thought process here is a a simple horizontal shift of non-background elements.
        # For each row, extract the values that are not 10.
        # For own rows, 
        # Shift those values one position to the right.
        # Fill gaps with 10.
        
        # To avoid complex logic, let same as observed:
        # It looks like only specific rows/cols are affected.
        # Rows 14-31 are heavily modified. Row 0 is slightly modified.
        # 
        # Let's implement a general "shift right" for any element that isn't 10.
        # For each row, identify the sequence of colors that aren't 10.
        # Create a new row filled with 10.
        # Place the shifted sequence back into the row.
        
        for r in range(grid.shape[0]):
            row = grid[r]
            # Find indices where value != 10
            indices = np.where(row != 10)[0]
            if len(indices) == 0:
                continue
            
            # Get the actual values at these indices
            values = row[indices]
            
            # New positions will be current index + 1
            new_indices = indices + 1
            
            # Check if we shift out of bounds
            valid_mask = new_indices < grid.shape[1]
            final_indices = new_indices[valid_mask]
            final_values = values[valid_mask]
            
            # We must clear old positions first to avoid overlapping/smearing
            # But wait, the observed deltas show color 10 filling in the same pattern.
            # This is exactly what happens when you move an object and leave background behind.
            
            temp_row = np.full(grid.shape[1], 10)
            # Put original non-background elements shifted by 1
            for idx, val in zip(final_indices, final_values):
                temp_row[idx] = val
            
            # Now, some cells might have been overwritten or left as 10.
            # The observed data shows a very specific "block" movement.
            # Let's refine: only shift things that are NOT 10.
            new_grid[r] = temp_row
            
    return new_grid

def is_level_complete(grid):
    """
    Returns True if the level is complete (win state).
    Based on typical ARC games, this usually means reaching a target configuration.
    Since no win state was provided, we return False unless a known condition is met.
    """
    # In many of these levels, completion is when all 'voids' (color 0) are gone or aligned.
    # For now, without a target grid, we assume it's not complete.
    return False

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state.
    The win condition for cn04 is that all cells of the same color
    must be connected (4-connectivity) and each color must form a 
    single contiguous region.
    """
    grid = np.array(grid)
    unique_colors = np.unique(grid)
    
    for color in unique_colors:
        # Mask for the current color
        mask = (grid == color)
        # Find the first cell of this color
        coords = np.argwhere(mask)
        if len(coords) == 0:
            continue
            
        # Breadth-First Search to find all connected cells of the same color
        start_node = tuple(coords[0])
        visited = set()
        queue = [start_node]
        
        while queue:
            curr = queue.pop(0)
            if curr not in visited:
                visited.add(curr)
                # Check 4-neighbors
                r, c = curr
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = curr[0] + dr, curr[1] + dc
                    if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                        if grid[nr, nc] == color:
                            queue.append((nr, nc))
        
        # If the number of visited cells equals the total number of cells of that color,
        # the color forms a single contiguous region.
        if len(visited) != len(coords):
            return False
            
    return True
