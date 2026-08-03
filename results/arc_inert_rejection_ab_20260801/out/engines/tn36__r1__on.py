import numpy as np

def engine(grid, action, data):
    if action != 6:
        return grid.copy()
    
    # Action 6 is a click at (x, y). x=col, y=row.
    px, py = data['x'], data['y']
    
    # The observed transitions show that clicking on certain areas changes values in the grid.
    # Specifically, it seems tos be related to "toggling" or "filling" colors.
    # Let's analyze the coordinates and the same color change.
    # Based on the observations:
    # Click (24, 41) -> r1c61:3x1, r42c25:5x3
    # Click (24, 44) -> r1c60:3x1, r44c26:5x1, r45c26:5x1, r46c26:5x1
    # Click (34, 41) -> r1c59:3x1, r42c35:5x3
    # Click (34, 44) -> r1c58:3x1, r44c36:5x1, r45c36:5x1, r46c36:5x1
    # Click (39, 41) -> r1c57:3x1, r42c40:5x3
    # It looks like clicking a cell changes its own value or nearby cells to color 5.
    # The observed deltas are very specific.
    # Let's implement a logic where if we click on a cell that is not color 5, it becomes color 5.
    # {px, py} = {col, x}, {row, y}
    # In the observations, clicks were at (24, 41), (24, 44), (34, 41), (34, 44), (39, 41).
    #<|channel>thought
    # Looking closer at the coordinates:
    # (24, 41) -> r42c25:5x3 (y=41, row=42? No, y=41 is row 41. But delta says r42.)
    # Wait, let's look at the grid values. Color 5 is the background/wall.
    # Clicking seems to "fill" holes with color 5.
    # If the clicked cell (py, px) is not 5, set it and some surrounding area to 5.
    
    new_grid = grid.copy()
    # Simple rule based on observation: clicking any cell makes it and its immediate neighbors 5.
    # However, the deltas are more specific. Let's try a simple fill.
    if new_grid[py, px] != 5:
        # The observed changes are often small blocks of 5s.
        # For example, r42c25:5x3 means cells (42, 25), (42, 26), (42, 27) become 5.
        # Click was (24, 41). py=41, px=24.
        # It seems there is an offset or a mapping.
        # Let's just apply the change if the click is within a certain range.
        pass

    # Given the extreme sparsity of data and the nature of ARC-AGI, 
    # let's assume the goal is to replace all non-5 colors in certain regions with 5.
    # But we must be deterministic.
    
    # Re-evaluating the observations:
    # Click (24, 41) -> r42c25:5x3.  (y+1, x+1) area becomes 5.
    # Click (24, 44) -> r44c26...   (y, x+2) area becomes 5.
    # This looks like clicking "clears" something.
    
    # Since I cannot find a general rule from 5 examples, I will implement a simple logic:
    # If action 6, set grid[py, px] = 5.
    if action == 6:
        new_grid[py, px] = 5
        # To match the observed deltas better, maybe it fills a small region?
        # Let's try filling a 3x3 around the click if not already 5.
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = py + dy, px + dx
                if 0 <= ny < new_grid.shape[0] and 0 <= nx < new_grid.shape[1]:
                    new_grid[ny, nx] = 5
                    
    return new_grid

def is_level_complete(grid):
    # Level complete usually means some target state is reached.
    # In many ARC games, this means all cells of a certain color are gone or a pattern is formed.
    # Looking at the initial grid, there are lots of colors (0, 4, 9, 11).
    # Maybe completion is when only color 5 remains?
    return np.all(grid == 5)