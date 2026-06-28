import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Action 1: Move right
        # Find all non-wall cells (not 5) and move them one step right
        # Walls are 5, empty is 0, other colors are movable
        for r in range(H):
            for c in range(W - 1):
                if new_grid[r, c] != 5:
                    new_grid[r, c + 1] = new_grid[r, c]
                    new_grid[r, c] = 0
        return new_grid
    
    elif action == 2:
        # Action 2: Move left
        for r in range(H):
            for c in range(1, W):
                if new_grid[r, c] != 5:
                    new_grid[r, c - 1] = new_grid[r, c]
                    new_grid[r, c] = 0
        return new_grid
    
    elif action == 3:
        # Action 3: Move down
        for c in range(W):
            for r in range(H - 1):
                if new_grid[r, c] != 5:
                    new_grid[r + 1, c] = new_grid[r, c]
                    new_grid[r, c] = 0
        return new_grid
    
    elif action == 4:
        # Action 4: Move up
        for c in range(W):
            for r in range(H - 1, 0, -1):
                if new_grid[r, c] != 5:
                    new_grid[r - 1, c] = new_grid[r, c]
                    new_grid[r, c] = 0
        return new_grid
    
    elif action == 5:
        # Action 5: Toggle walls (5 <-> 0)
        new_grid = grid.copy()
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 0
                elif new_grid[r, c] == 0:
                    new_grid[r, c] = 5
        return new_grid
    
    elif action == 6:
        # Action 6: Click at pixel coordinates
        if data and 'x' in data and 'y' in data:
            px, py = data['x'], data['y']
            # Convert pixel to logical
            r, c = py // 1, px // 1
            if 0 <= r < H and 0 <= c < W:
                # Toggle the cell
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 0
                elif new_grid[r, c] == 0:
                    new_grid[r, c] = 5
        return new_grid
    
    elif action == 7:
        # Action 7: No-op or special action
        return grid
    
    return grid

def is_level_complete(grid):
    # Check if the level is complete
    # Based on the observed transitions, the level is complete when
    # certain conditions are met. Since we don't have explicit win state,
    # we check if the grid has been transformed significantly.
    # For now, return False as we don't have a clear win condition from the data
    return False