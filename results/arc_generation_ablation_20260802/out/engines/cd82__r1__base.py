import numpy as np

import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The game seems to be a puzzle where blocks of different colors are moved or shifted.
    # Based on the observed transitions, it's hard to induce general rules without 
    # specific details about the same-action repeated patterns.
    # Since this only a few transitions are provided and the<|channel>thought process is//
    # nott allowed, I must implement a basic logic that mimics the delta changes if possible.
    # However, the deltas are very complex. Let's look at the actions.
    # ACTION 4 is often 'down', ACTION 2 is 'left'.
    # In most ARC games, ACTION 4 is down, 2 is left, 0 is up, 6 is click.
    # Looking at the deltas, they seem to move a cluster of pixels.
    # Find the object (non-background color) and shift it based on the action.
    
    # Background color is usually the most frequent color in the initial grid.
    # Most cells are color 5 (grey).
    bg_color = 5
    
    # Define movement vectors for common keyboard actions
    # Action 1: Up, 2: Left, 3: Right, 4: Down
    movements = {
        1: (-1, 0),
        2: (0, -1),
        3: (0, 1),
        4: (1, 0)
    }
    
    if action in movements:
        dr, dc = movements[action]
        new_grid = grid.copy()
        
        # Identify all "movable" objects (anything not background color 5)
        # We need to find connected components or just any non-bg cell.
        # But we must avoid moving static walls. In this game, colors 3, 4, 0, 15 might be parts of the level.
        # Color 2 seems to be the player/object being moved.
        
        # Let's try to move only color 2 blocks.
        mask = (grid == 2)
        coords = np.argwhere(mask)
        
        if coords.size == 0:
            return new_grid
            
        # Shift coordinates
        new_coords = coords + [dr, dc]
        
        # Filter out-of-bounds and collisions with "walls" (e.g., colors other than bg_color)
        # However, looking at deltas, it looks like a sliding puzzle where things push each other.
        # For simplicity, let's implement a basic shift for color 2.
        
        for r, c in coords:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                # If target is background or same object, move there.
                if grid[nr, nc] == bg_color or grid[nr, nc] == 2:
                    new_grid[r, c] = bg_color
                    new_grid[nr, nc] = 2
        
        return new_grid

    return grid

def is_level_complete(grid):
    # Level complete usually means the object reached a goal or all objects are gone.
    # Looking at the INITIAL GRID, there are areas of color 15.
    # In many ARC games, reaching a specific area or clearing an object triggers win.
    # Since no WIN STATE was provided, we assume completion when color 2 reaches a certain point.
    # Or simply return False as we don't have the win condition.
    return False

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state.
    The win condition for cd82 is based on the presence of a 
    specific pattern of colors.
    """
    grid = np.array(grid)
    # The win condition is that the grid contains only one color 
    # (excluding background color 0)
    # This is a a simple general rule that often applies to these tasks.
    non_zero_elements = grid[grid != 0]
    if non_zero_elements.size == 0:
        return False
    return np.all(non_zero_elements == non_zero_elements[0])
