import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 3:
        # Action 3: Click at (data['x'], data['y'])
        # This action toggles the color of the clicked cell and its neighbors
        # The pattern suggests a toggle operation on a 3x3 area centered at the click
        # Based on the observed transitions, this action toggles cells in a specific pattern
        # We'll implement a toggle operation that matches the observed behavior
        
        # Create a copy of the grid
        new_grid = grid.copy()
        
        # Toggle the clicked cell and its neighbors
        # The pattern from the data suggests a 3x3 toggle centered at the click
        # But we need to be careful about the exact pattern
        
        # Based on the observed transitions, action 3 seems to toggle a 3x3 area
        # Let's implement a simple toggle operation
        x, y = data['x'], data['y']
        
        # Toggle the clicked cell and its neighbors
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    new_grid[ny, nx] = 1 - new_grid[ny, nx]
        
        return new_grid
    
    elif action == 2:
        # Action 2: Directional movement (up, down, left, right)
        # Based on the observed transitions, this action moves objects in a specific direction
        # The pattern suggests gravity-like behavior
        
        # Create a copy of the grid
        new_grid = grid.copy()
        
        # Determine the direction based on the action
        # Action 2 seems to move objects in a specific direction
        # Based on the observed transitions, it appears to be moving objects down
        
        # Move all non-background objects down
        for col in range(W):
            # Find all non-background cells in this column
            cells = []
            for row in range(H):
                if grid[row, col] != 0:
                    cells.append((row, grid[row, col]))
            
            # Move them down
            if cells:
                # Find the lowest position
                max_row = max(r for r, c in cells)
                # Shift all cells down
                for i, (row, color) in enumerate(cells):
                    new_row = row + (H - 1 - max_row)
                    new_grid[new_row, col] = color
        
        return new_grid
    
    else:
        # Default: return the grid unchanged
        return grid

def is_level_complete(grid):
    # Check if the grid is in a win state
    # Based on the observed win state, we need to check specific conditions
    
    # The win state has a specific pattern:
    # - A large block of color 9 on the left
    # - A block of color 10 in the middle
    # - A block of color 9 on the right
    # - A block of color 11 at the bottom
    
    # Check the pattern
    H, W = grid.shape
    
    # Check the left block (color 9)
    left_block = grid[:, :36]
    if not np.all(left_block == 9):
        return False
    
    # Check the middle block (color 10)
    middle_block = grid[:, 36:38]
    if not np.all(middle_block == 10):
        return False
    
    # Check the right block (color 9)
    right_block = grid[:, 38:62]
    if not np.all(right_block == 9):
        return False
    
    # Check the bottom block (color 11)
    bottom_block = grid[63, :]
    if not np.all(bottom_block == 11):
        return False
    
    return True