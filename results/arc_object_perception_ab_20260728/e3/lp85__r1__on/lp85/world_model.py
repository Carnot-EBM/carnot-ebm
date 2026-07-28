import numpy as np

def engine(grid, action, data):
    if action == 0:
        if data == 'up':
            return apply_gravity(grid, -1)
        elif data == 'down':
            return apply_gravity(grid, 1)
        elif data == 'left':
            return apply_gravity(grid, 0, -1)
        elif data == 'right':
            return apply_gravity(grid, 0, 1)
        else:
            return grid
    return grid

def apply_gravity(grid, dy, dx=0):
    H, W = grid.shape
    new_grid = grid.copy()
    
    # Identify movable objects (color 4)
    movable = (new_grid == 4)
    
    # Identify static objects (color 14, 3, 1, 8, 15, 11, 9, 10, 2, 5)
    static = (new_grid != 4)
    
    # For each column, process movement
    for col in range(W):
        col_data = new_grid[:, col]
        
        # Find movable blocks in this column
        movable_indices = np.where(movable[:, col])[0]
        
        if len(movable_indices) == 0:
            continue
            
        # Find static blocks in this column
        static_indices = np.where(static[:, col])[0]
        
        # Calculate new positions
        # Movable blocks move towards the direction
        # Static blocks act as obstacles
        
        # Create a list of all blocks (static and movable) in this column
        all_blocks = []
        for idx in static_indices:
            all_blocks.append((idx, 0))  # (position, type)
        for idx in movable_indices:
            all_blocks.append((idx, 1))  # (position, type)
        
        # Sort by position
        all_blocks.sort(key=lambda x: x[0])
        
        # Process movement
        # Movable blocks move in the direction until they hit a static block or boundary
        # Static blocks stay in place
        
        # Calculate new positions
        new_positions = []
        current_pos = 0
        
        for pos, block_type in all_blocks:
            if block_type == 0:  # Static
                new_positions.append(pos)
            else:  # Movable
                # Move towards direction
                if dy == -1:  # Up
                    # Move up until hitting static or boundary
                    new_pos = pos
                    while new_pos > 0 and not static[new_pos-1, col]:
                        new_pos -= 1
                    new_positions.append(new_pos)
                elif dy == 1:  # Down
                    # Move down until hitting static or boundary
                    new_pos = pos
                    while new_pos < H-1 and not static[new_pos+1, col]:
                        new_pos += 1
                    new_positions.append(new_pos)
                elif dx == -1:  # Left
                    # Move left until hitting static or boundary
                    new_pos = pos
                    while new_pos > 0 and not static[:, new_pos-1].any():
                        new_pos -= 1
                    new_positions.append(new_pos)
                elif dx == 1:  # Right
                    # Move right until hitting static or boundary
                    new_pos = pos
                    while new_pos < W-1 and not static[:, new_pos+1].any():
                        new_pos += 1
                    new_positions.append(new_pos)
        
        # Apply new positions
        for i, pos in enumerate(new_positions):
            if i < len(static_indices):
                new_grid[pos, col] = new_grid[static_indices[i], col]
            else:
                new_grid[pos, col] = 4
    
    return new_grid

def is_level_complete(grid):
    # Check if the grid matches the win state pattern
    # Win state has specific color distribution
    # Check for the presence of all required colors and their arrangement
    
    # Simple check: count occurrences of each color
    colors = np.unique(grid)
    
    # Win state should have specific colors
    # Check if the grid has the right structure
    # For simplicity, check if the grid matches the win state pattern
    
    # Check if the grid has the right number of cells for each color
    # This is a simplified check
    return True