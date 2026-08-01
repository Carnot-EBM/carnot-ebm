import numpy as np

import numpy as np

def engine(grid, action, data):
    # The game state is a 64x64 grid.
    # Colors: 0 (black), 4 (yellow), 8 (blue), 10 (grey), 14 (pink)
    # Action 4 is observed to move an "object" or "hole" rightward.
    # Based on the transitions, ACTION4 shifts a specific pattern of cells (the 'hole' and its surroundings)
    # to the right by 1 pixel/unit.
    
    if action == 4:
        # Identify the region that moves.
        # In the initial grid, there is a black area (color 0) starting at r14c11.
        # Looking at the<|channel>s, own-cell values are replaced by 10 (grey) and new cells to the right are replaced by 0 (black).
        # This movement is essentially shifting a window of color 0s and other colors within a bounding box.
        
        # Find all coordinates of color 0.
        zeros = np.argwhere(grid == 0)
        if zeros.size == 0:
            return grid
        
        # Get the bounding box of the zero-region.
        y_min, x_min = zeros[:, 0].min(), zeros[:, 1].min()
        y_max, x_max = zeros[:, 0].max(), zeros[:, 1].max()
        
        # Create a copy of the grid to modify.
        new_grid = grid.copy()
        
        # Shift the entire pattern of non-grey cells within the bounding box to the right.
        # We need to be able to handle the internal structure of the same object.
        # The observed deltas show that not only the 0s move, but also some 8s and 14s if they overlap.
        # The same region moves as a single unit.
        
        # Define the width of the moving area.
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        
        # Extract the window.
        window = grid[y_min:y_max+1, x_min:x_max+1]
        
        # Fill the leftmost column of the original position with grey (color 10).
        new_grid[y_min:y_max+1, x_min] = 10
        
        # Paste the window shifted by one to the right.
        # Ensure we don't go out of bounds.
        if x_max + 2 <= 64:
            new_grid[y_min:y_max+1, x_min+1:x_max+2] = window
            
    return new_grid

def is_level_complete(grid):
    # A level is complete when the object reaches a target or matches a win state.
    # In this specific game 'cn04', no win state was provided in the observed transitions.
    # However, usually it involves aligning objects or filling holes.
    # Based on typical ARC games, we check if the same pattern has reached a certain boundary.
    # The black region (color 0) moving rightward suggests it might be trying to reach something.
    # Let's assume for now that completion is based on the existence of color 4 at the top.
    # # Actually, without a WIN STATE grid, we must induce a general rule.
    # Looking at the initial grid, there is a yellow line (color 4) at r0c16-r0c47.
    # The hole moves under it.
    # Completion likely occurs when the hole aligns with the end of the yellow line or similar.
    # Since no win state is given, return False unless a clear condition is found.
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
            r, c = curr
            # Check 4-neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = curr[0] + dr, curr[1] + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    if grid[nr, nc] == color and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        
        # If the number of visited cells equals the total number of cells of that color,
        # the color is connected.
        if len(visited) != len(coords):
            return False
            
    return True
